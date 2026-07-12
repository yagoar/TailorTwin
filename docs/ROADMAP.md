# ROADMAP — planned improvements

Working document for continuing development across sessions. Written to
be executable by any assistant session (including smaller models):
every workstream states its goal linkage, exact files, steps, acceptance
criteria, and — importantly — whether it is safe to attempt without a
human or a full ML environment in the loop.

## 0. Read this first

### Project goals (the only two that matter)

1. **Simpler capture.** Taking a scan must be *easier* than taping the
   ~25 measurements by hand. Every input we ask of the user (tape
   girths, height, waist height) is friction — they are optional
   accuracy aids, never requirements. A feature that adds capture steps
   must buy accuracy that a tape measure can't.
2. **A CLO3D-ready avatar.** An accurate 3D body (the exported
   `*_fit_body.obj`) whose girths match the real body, importable into
   CLO3D (and similar) for virtual garment fitting.

Everything else in the pipeline — tape anchors, ring deform, waist
height, clean-fit — exists only to serve those two. When prioritising,
ask: *does this reduce user effort, or improve accuracy per unit of
user effort?*

### Session bootstrap

- Read `GUARDRAILS.md` **before** touching `measure/definitions/`,
  `measure/landmarks.py` vertex IDs, or anything citing Aldrich/dpm.
  The short version: no measurement definitions, vertex IDs, or
  drafting formulas from model memory — sources only.
- Light tests (no ML stack needed):
  `PYTHONPATH=src python -m pytest tests/ -q` — needs only
  `numpy pytest flask`. `test_yaiza_snapshot` and anything importing
  `torch`/`smplx`/`open3d` require the full env + the gitignored
  `data/body_models/smplx/SMPLX_FEMALE.npz`.
- The GUI validation tests need a (possibly empty, gitignored)
  placeholder at `data/body_models/smplx/SMPLX_FEMALE.npz` to enable
  the "female" gender.
- Full pipeline runs (fit, measure, ring-deform) only work on the
  user's machine. **Any change to fit/measure numerics must be
  verified there** — see "Definition of done" below.

### Definition of done for any accuracy-affecting change

1. Light test suite green.
2. `tests/test_yaiza_snapshot.py` green on the user's machine (or the
   diff explained and the snapshot deliberately regenerated).
3. Synthetic harness green once Workstream A exists.
4. A real scan re-run: history drift report clean
   (`data/results/history.sqlite`) and, for tape-anchored runs, the
   `*_tape_audit.json` shows no unexplained unanchored drift.

### Recently completed (branch `claude/measurement-accuracy-eval-x8t5i7`)

- Waist-height anchor (`--waist-height`, GUI field) — floor-relative,
  frame-robust waist Y; replaced and removed the HSV waist-string
  detection (which had a coordinate-frame bug vs clean-fit).
- Provenance & observability: `*_manifest.json` per run
  (`src/tailor_twin/manifest.py`), per-person measurement history + run
  drift report (`src/tailor_twin/history.py`,
  `data/results/history.sqlite`), tape-anchor audit
  (`audit_girth_drift` in `src/tailor_twin/fit/ring_deform.py`,
  `*_tape_audit.json`).

---

## Priority overview

| # | Workstream | Serves goal | Needs ML env | Model capability | Status |
|---|-----------|-------------|--------------|------------------|--------|
| A | Synthetic validation harness | both (safety net) | yes (user machine) | small OK, verify on user machine | todo |
| B | Self-rotation capture (no helper) | 1 | yes + research | **capable model + human review** | todo |
| C | CLO3D avatar quality | 2 | partly | small OK (docs/export), human verifies in CLO3D | todo |
| D | Anthropometric sanity layer | 1 | no | small OK **once sources provided** | blocked on sources |
| E | Uncertainty + block-critical tier | both | partly | small OK for tiering; medium for uncertainty | todo |
| F | In-process pipeline refactor | maintainability | yes (verification) | medium; gate on A | todo |
| G | Small fixes | — | no | small OK | todo |

Recommended order: **A → C → E(tier) → D → E(uncertainty) → F → B**.
A is first because it makes every later change checkable without the
user re-taping their body.

---

## Workstream A — Synthetic-body validation harness

**Why:** landmark rules and recipes are currently validated against one
person's tape numbers (plus a few cross-checks). SMPL-X can *generate*
arbitrary bodies, so extractor changes can be regression-tested across
the shape space without any capture.

**Not circular because** we are not checking absolute truth — we are
checking (a) that nothing *changes unintentionally* across code edits,
(b) that every landmark rule *succeeds* (no exceptions) on a wide range
of bodies, and (c) smoothness/invariance properties that must hold
regardless of truth.

**Files:** new `scripts/validate_synthetic.py` + new
`tests/test_synthetic_snapshot.py` (skipped when model file absent,
same pattern as `tests/test_yaiza_snapshot.py`).

**Steps:**
1. Sample N=30 bodies: fixed RNG seed, betas ~ N(0, 1) on the first 10
   components (rest zero), female model, canonical A-pose 30°
   (`tailor_twin.fit.refine_to_tape._build_a_pose`), no displacement.
2. For each body run `measure.seamly_extractor.extract_catalog` with a
   `build_landmark_set` built the same way `measure/cli.py` does.
3. Assert zero unexpected `skipped` codes (compare against a committed
   allowlist — some codes legitimately skip).
4. Write `{body_index: {code: value}}` to
   `tests/data/synthetic_snapshot.json` (committed); the test compares
   within ±0.05 cm and prints per-code diffs on failure.
5. Invariance checks: mirrored betas on symmetric components produce
   equal left/right code pairs; small beta perturbation (±0.05) moves
   every code < 1.5 cm (catches landmark rules that "jump" between
   vertices).
6. Bonus: run `ring_deform_cli` logic in-process on one synthetic body
   with 3 targets; assert targets hit within 0.3 cm and
   `audit_girth_drift` flags nothing unanchored beyond 1 cm.

**Acceptance:** harness runs on the user's machine in < 10 min; test is
auto-skipped without the model file; snapshot committed.

**Small-model guidance:** safe to write; do NOT hand-edit snapshot
values — they must come from an actual harness run on the user's
machine.

---

## Workstream B — Self-rotation capture (drop the helper, maybe the fusion)

**Why (goal 1):** the current SOP needs a second person orbiting for
30–45 s × 2 loops while the subject stands perfectly still. Subject
sway and odometry drift are the dominant error sources, and both are
artifacts of the *capture style*, not the sensor. If per-frame body
pose is solved, the subject can rotate in place with the phone static
(shelf/tripod) — no helper, no stillness requirement.

**Architecture idea:** skip TSDF fusion; fit one shared `betas` (and
optionally displacement) against per-keyframe observations —
segmented depth points and/or silhouettes — with a per-keyframe
`global_orient`/`transl` (and later `body_pose`). Sway becomes
per-frame pose variation that averages out instead of baking into
geometry.

**Phased plan:**
1. **B1 spike (existing captures, no new protocol):** fit shared betas
   directly to K≈20 segmented depth frames from an existing capture
   (reuse `fit/fit.py` losses; the frame loader and segmentation
   already exist in `io/stray_loader.py` + `preprocess/`). Compare the
   resulting catalog vs the TSDF-path catalog and vs tape. If within
   ~1 cm, fusion is demonstrably optional.
2. **B2:** allow per-frame pose (subject rotates; static phone). Needs
   2D keypoints per frame for initialisation — a new dependency
   decision (e.g. a pose-estimation model); discuss with the user
   before adding.
3. **B3:** new capture SOP + preflight support + GUI switch.

**Guardrails:** GUARDRAILS §12.4 applies in full — ground in SMPLify-X
conventions, validate visually, distance-heatmap check mandatory.

**Small-model guidance: do NOT attempt unsupervised.** This is
research-grade work with silent failure modes; it needs a capable
model session plus the user running captures. A smaller session CAN
prepare the ground: extract `fit_scan`'s loss assembly into functions
reusable for multi-frame targets, behind tests that pin current
behavior (requires Workstream A first).

---

## Workstream C — CLO3D avatar quality

**Why (goal 2):** `*_fit_body.obj` (A-pose, tape-anchored SMPL-X+D) is
already exported, but the CLO3D workflow is undocumented and untested
end-to-end.

**Steps:**
1. **Document the import workflow** (new `docs/clo3d_avatar.md`): which
   CLO3D import dialog/settings to use, and crucially the **unit
   setting** — the OBJ is in metres; CLO3D's OBJ import asks for a
   unit, picking the wrong one gives a 100× avatar. *The user must
   verify the exact dialog names in CLO3D — do not write them from
   model memory; leave TODO markers for anything unverified.*
2. **Verification checklist** in the same doc: after import, measure
   height + bust/waist/hip with CLO3D's own tape tool and compare to
   the run's CSV; acceptance is agreement within 0.5 cm.
3. **UVs:** the export currently has no texture coordinates. SPEC §6
   already lists `smplx_uv_2023.zip` as a download; extend
   `measure/exports.py::write_obj` to optionally write `vt`/`f v/vt`
   lines from that UV map (new `--obj-uv` flag, default off until
   verified in CLO3D).
4. **Pose options:** CLO3D avatars are often preferred with arms lower
   than 30°. `--apose-deg` already exists end-to-end; surface it in the
   GUI as an "Avatar pose" field (small forms + template change,
   mirror the waist-height field pattern) and document that
   measurements are pose-normalized so this is cosmetic.
5. Note in the doc: head/hands are the clean SMPL-X template by design
   (`fit/clean_fit.py`) — fine for garment fitting, not a face scan.

**Small-model guidance:** items 1–2 are documentation with explicit
TODOs for the user to verify inside CLO3D — safe. Item 3 touches an
export path — safe with a unit test comparing written OBJ line counts;
verify visually before defaulting on. Item 4 mirrors an existing
pattern — safe.

---

## Workstream D — Anthropometric sanity layer (blocked on sources)

**Why (goal 1):** height + weight + one or two girths predict the rest
of the body surprisingly well via published regressions. Used as a
*sanity check*, this catches capture failures for free: "extracted
thigh is 4 cm off the regression given hip and height — check the
scan". Never used to *replace* measurements.

**Blocker:** GUARDRAILS §12 forbids regression coefficients from model
memory. The user must place a citable source (paper/table scan) in
`references/anthropometry/` first. Candidate sources to look for:
published allometric equations from large anthropometric surveys
(e.g. ANSUR II / CAESAR-derived regression tables).

**Once sources exist:** new `src/tailor_twin/sanity.py` with
`check_plausibility(values, height_cm, weight_kg) -> list[warning]`,
each coefficient commented with its source page; wire after the
history step in `scan.py`; unit tests with the source's own worked
examples. Add an optional "Weight (kg)" GUI field (mirrors the
waist-height field pattern — it feeds only this check).

**Small-model guidance:** safe and well-scoped *after* the user
provides the source; do not start before.

---

## Workstream E — Per-measurement uncertainty + block-critical tier

### E1 — Block-critical tier (small, do early)

The Aldrich/dpm blocks consume ~25 of the 167 extracted codes. Tag
them so outputs lead with what matters:

1. Add `BLOCK_CRITICAL: frozenset[str]` to
   `measure/seamly_catalog.py`. **Derive membership from
   `measure/definitions/merged.yaml` + SPEC §9.1/§9.2 (the
   Aldrich #1–20 and dpm #1–32 tables), citing each code's source
   entry — not from memory.**
2. `measure/exports.py::write_csv`: add a `tier` column
   (`block`/`extended`); keep column order otherwise identical, and
   regenerate the yaiza snapshot on the user's machine (CSV is
   byte-compared by `tests/test_yaiza_snapshot.py` — coordinate with
   the user before merging).
3. GUI viewer: sort/badge block-critical codes first (viewer_data
   already ships polylines per code).

### E2 — Uncertainty estimates (medium)

For each planar-slice code, re-extract with the slice Y jittered
±2 mm (5 samples) and report the std as `± cm` alongside the value;
geodesic codes: jitter endpoint landmarks by one vertex-neighbourhood.
Implement as an opt-in `--uncertainty` flag on `measure/cli.py` (it
multiplies extraction time). Surface in CSV as an extra column and in
the GUI table. Validate on the synthetic harness (A): uncertainty must
be ≪ the code's value and stable across bodies.

**Small-model guidance:** E1 safe (metadata only, but respect the
snapshot-test coordination). E2 medium — needs the harness to verify.

---

## Workstream F — In-process pipeline (kill the subprocess chains)

**Why:** `scan.py` shells out to `ring_deform_cli`, `measure.cli`, and
`extract_bent_arm` with data flattened into CLI flags. The waist-Y
frame bug happened exactly at such a boundary. Structured in-process
calls make that class of bug unrepresentable and let stages share the
loaded body model (each subprocess currently re-loads SMPL-X).

**Steps (mechanical, behavior-preserving):**
1. In each CLI, split `main(argv)` into `parse(argv) -> options` and a
   pure `run(options) -> int` that takes real types (Path, dict,
   float) — CLIs stay as thin wrappers, so all existing invocations
   keep working.
2. `scan.py` calls the `run(options)` functions directly instead of
   `subprocess.run([sys.executable, "-m", ...])`.
3. Keep prints identical where feasible (the GUI streams stdout).

**Gate:** do NOT merge without Workstream A green plus one real scan
compared before/after (manifest + history make this diffable). Memory
note: in-process means torch stays resident across stages — if the
user's machine struggles, keep bent-arm as a subprocess.

**Small-model guidance:** the extraction is mechanical and safe to
draft, but verification requires the user's machine — deliver as a PR
the user tests, never self-merge on green light-tests alone.

---

## Workstream G — Small fixes (any session)

- [ ] Pre-existing ruff findings: unused imports in
      `fit/refine_to_tape.py` (`build_landmark_set`, `RECIPES`);
      `E702` semicolons in `gui/viewer_data.py`; a few more — run
      `ruff check src/` for the list.
- [ ] `fit/ring_deform_cli.py`: the audit baseline duplicates the
      pass-1 extraction — reuse it (pure speed, no behavior change).
- [ ] GUI: show the history drift report and tape audit in the run log
      panel more prominently (they're plain stdout lines today).
- [ ] `history.py`: tiny CLI (`python -m tailor_twin.history <person>`)
      to list a person's runs and per-code trend.
- [ ] Bent-arm: document in SPEC §11 the *option* of a second short
      bent-arm capture (SPEC §9.1 option 1) as a validation reference
      for the virtual re-pose — protocol text only until someone
      captures one.

---

## Explicitly out of scope (decided against for now)

- **Turntables / extra hardware** — violates the external constraints.
- **Multi-user features, cloud** — non-goals per SPEC §2.
- **Replacing tape anchors entirely** — they stay optional; the aim is
  needing *fewer* of them, not banning them.
- **Trimming the 167-code catalog** — tiering (E1) instead of
  deletion; removal is a user decision.
