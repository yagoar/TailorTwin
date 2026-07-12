# End-to-end test run — branch validation checklist

Step-by-step validation of everything added on the
`claude/measurement-accuracy-eval-*` branches: waist-height anchor,
provenance layers (history / manifest / tape audit), dead-code cleanup,
the synthetic harness, and the experimental fusion-free fit. Run on
your machine (ML env + SMPL-X model file + a real capture). Total
hands-on time ≈ 1–2 h, mostly waiting on pipeline runs.

Work through the phases in order — each one validates what the next
depends on. Record results in the table at the bottom.

---

## Phase 0 — environment sanity (~2 min)

```bash
cd tailor-twin && source .venv/bin/activate
git pull
pip install -e .        # picks up any new module layout
python -m pytest tests/ -q
```

**Pass:** everything green.
`test_modules_importable` finally runs here (it needs torch);
`test_synthetic_snapshot_regression` still skips (snapshot not
generated yet — Phase 2 fixes that).

---

## Phase 1 — numeric no-change gate (~2 min)

The dead-code cleanup and refactors must not have moved a single
number. The yaiza snapshot proves it:

```bash
python -m pytest tests/test_yaiza_snapshot.py tests/test_cross_figure_robustness.py -q
```

**Pass:** byte-identical CSV, drift budgets green.
**If it fails:** stop here — nothing later is trustworthy. Diff output
names the drifted codes; bring the diff back to a session.

---

## Phase 2 — seed the synthetic harness (~10 min)

```bash
python scripts/validate_synthetic.py --write
git add tests/data/synthetic_snapshot.json
git commit -m "test: seed synthetic-body snapshot"

python scripts/validate_synthetic.py            # must pass against itself
python scripts/validate_synthetic.py --perturb  # smoothness check
```

**Pass:** compare run OK; `--perturb` reports no code jumping > 1.5 cm.
**Record:** skip counts per body (printed per line). A code skipped on
*some* bodies but not others = a landmark rule fragile to shape — note
which code + the body's `betas_active` from the snapshot JSON, file for
a session.

---

## Phase 3 — calibrated scan with the waist-height anchor (~10 min)

Tape-measure once, standing straight: **height** (floor → crown) and
**waist height** (floor → the string tied at your natural waist,
vertical, at the side). Then run your usual capture:

```bash
tailor-twin scan data/captures/<CAPTURE> \
  --out-prefix data/results/yaiza_testrun/yaiza_testrun \
  --use-displacement --height <H> --waist-height <WH> \
  --tape-anchor G04=<bust> --tape-anchor G07=<waist> --tape-anchor G09=<hip>
```

(or the GUI: fill Height, Waist height, and the girths — same thing.)

**Check, in order:**
1. Log line `waist height <WH> cm above floor → Y override …` appears
   in the measure step.
2. `…_tape_audit.json` written; log says
   `tape audit: no unanchored code moved > 1.0 cm` (or lists what did —
   judge each: a neighbour like G05/G08 moving ≤ ~1.5 cm is the falloff
   band; anything big or distant is a bug).
3. `…_manifest.json` exists and records the exact flags.
4. `history: first recorded run …` (or drift vs your previous run).
5. **Viewer:** open the GUI 3D viewer → the G07 waist line must sit ON
   your tied string height, not the SMPL-X anatomical waist. This is
   THE visual check for the frame-bug fix.
6. CSV waist/bust/hip equal your tape anchors within 0.1 cm; A01 = your
   height.

**Then re-baseline** (waist-anchored codes legitimately moved off the
old buggy waist Y):

```bash
python -m tailor_twin.measure.cli data/results/yaiza_testrun/yaiza_testrun_smplx_fit.npz \
  --num-betas 300 --save-csv data/results/yaiza_measurements.baseline.csv
python -m pytest tests/test_yaiza_snapshot.py -q   # green again
git add data/results/yaiza_measurements.baseline.csv && git commit -m "test: re-baseline after waist-height anchor"
```

(Also regenerate the carmen baseline the same way if you want
`test_cross_figure_robustness` meaningful again.)

---

## Phase 4 — fusion-free A/B (the interesting one, ~15 min)

Same capture, same person name, new prefix, **Fusion-free fit** ticked
in the GUI (or `--no-fusion`):

```bash
tailor-twin scan data/captures/<CAPTURE> \
  --out-prefix data/results/yaiza_nofusion/yaiza_nofusion \
  --no-fusion --use-displacement --height <H> --waist-height <WH>
```

Deliberately **without** girth anchors — you want to see what the raw
fit does, not what the anchors force.

**Check, in order:**
1. **Frame check (do this FIRST):** open
   `yaiza_nofusion_scan_cloud.obj` together with Phase 3's
   `…_scan.obj` in MeshLab. The cloud must lie ON the mesh surface. If
   it's offset/rotated, STOP — projection-convention bug, report it.
2. Fit completes; chamfer printed (point-to-point values aren't
   directly comparable to mesh-based runs — just note it).
3. `history:` drift report prints the per-code deltas vs Phase 3
   automatically. Also compare both to TAPE:

```bash
python -m tailor_twin.history "Yaiza"        # runs + last-two drift
```

**Judge:** for bust/waist/hip/thigh vs your tape numbers — is the
fusion-free run closer, equal, or worse than the TSDF run? Eyeball the
fit body in the viewer for artifacts (collapsed armpits, inflated
belly = outlier points survived).

**Record the verdict** — it decides ROADMAP B1 (promote `--no-fusion`
to default, keep experimental, or fix).

---

## Phase 5 — CLO3D avatar (~15 min)

Import `yaiza_testrun_fit_body.obj` per `docs/clo3d_avatar.md`
(units = **metres** — that's the trap) and run its tape-vs-CSV
checklist. Fill in the `TODO(user)` markers in that doc while you're
in the dialogs.

**Pass:** A01/G04/G07/G09 agree with the CSV within 0.5 cm in CLO3D's
own tape tool.

---

## Phase 6 — wrap up (~5 min)

```bash
python -m pytest tests/ -q      # everything incl. new snapshots green
git push
```

Update `docs/ROADMAP.md`: B1 status with the Phase 4 verdict, tick the
CLO3D checklist item, and note any codes flagged in Phases 2–4 as new
workstream entries.

---

## Results

| Phase | Result | Notes |
|---|---|---|
| 0 env + tests | | |
| 1 numeric gate | | |
| 2 synthetic seed + perturb | | |
| 3 waist-height scan (viewer check!) | | |
| 3 tape audit clean | | |
| 4 cloud/mesh overlay | | |
| 4 A/B verdict vs tape | | |
| 5 CLO3D checklist | | |

## Troubleshooting

- **A flag error the moment a stage starts** — shouldn't happen
  (`tests/test_cli_wiring.py` guards the argparse chain), but if it
  does, the manifest shows the exact argv that failed.
- **Waist line NOT on the string in the viewer** — check the measure
  step log for the `waist height … → Y override` line; if present but
  wrong, report the printed Y + your mesh's min/max Y.
- **`--no-fusion` fit stretches toward the floor** — stray floor points
  survived matting; check the cloud in MeshLab for a ground patch and
  report (outlier removal parameters may need tuning).
- **History/audit/manifest steps print `skipped (…)`** — they're
  best-effort by design; the scan is still valid. Report the message.
- **Anything numeric drifts unexpectedly** — `_manifest.json` of both
  runs + `python -m tailor_twin.history` output pins down what changed
  between them; bring those.
