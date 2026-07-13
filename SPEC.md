# TailorTwin — Spec

A personal project to capture the user's body with an iPhone Pro / iPad Pro LiDAR sensor and extract the measurements needed to draft sewing blocks (bodice, skirt, sleeve, trouser) without needing another person to take tape measurements. Supports the Aldrich (5th ed.) and dresspatternmaking.com measurement systems.

This document is the authoritative project spec. The repository ships a Typer console script `tailor-twin` (`pip install -e .`) and a localhost Flask GUI at port 8060.

## 1. Goals

**North star (in priority order):**

1. **Simpler capture** — taking a scan must be easier than taping the
   ~25 block measurements by hand. Tape inputs (height, waist height,
   girth anchors) are optional accuracy aids, never requirements.
2. **A CLO3D-ready avatar** — an accurate 3D body whose girths match
   the real body, exported for virtual garment fitting.

Feature test: does it reduce user effort, or improve accuracy per unit
of user effort? See `docs/ROADMAP.md` for the prioritized plan.

- Capture body geometry with iPhone/iPad LiDAR + Stray Scanner
- Reconstruct a clean 3D mesh of the body
- Fit SMPL-X+D (parametric body model with per-vertex displacements) to the scan
- Place constructional lines (CF, CB, side seam, bust line, waist line, etc.) on the mesh
- Extract 160+ measurements covering both the Aldrich system and the dresspatternmaking.com bodice list (Seamly catalog codes)
- Output a populated SeamlyMe `.smis` file and Aldrich- / dpm-filtered CSVs
- Stretch goal: auto-generate Aldrich bodice/skirt/sleeve blocks as SVG

## 2. Non-Goals

- Multi-user system, accounts, auth, public web
- Real-time on-device reconstruction
- Tailor-grade accuracy on small circumferences (wrist, ankle below ±0.5 cm)
- Replacement for toile fitting iterations

## 3. Target Accuracy

After calibration against tape measurements:

| Measurement category | Target |
|---|---|
| Major planar circumferences (bust, waist, hips) | ±0.5 to 1 cm |
| Contoured paths (Upper Bust, Front Shoulder to Waist) | ±1 to 1.5 cm |
| Lengths between hard landmarks (nape to waist, sleeve length) | ±0.5 to 1 cm |
| Multi-point armhole and bust depths (dpm) | ±1 to 2 cm |
| Small circumferences (wrist, ankle) | ±0.5 to 1 cm absolute |

**Status (validated).** The bare SMPL-X+D fit lands ~1-3 cm vs tape and
repeats to ~1 cm across independent clean captures. The `--height` +
`--tape-anchor` calibration pass then pins the scale and any tape-measured
girths **exactly** (verified height/bust/waist/hip/calf within ~0.1 cm),
while preserving the scan's real cross-section shape. So the targets above
are met *with* a handful of tape anchors — the realistic, validated
operating mode — not from the bare scan alone. This matches the literature
floor: parametric fits sit ~2-3 cm on real bodies, and tape anchoring is
the documented way to reach drafting tolerance.

## 4. Hardware Prerequisites

- iPhone 12 Pro / Pro Max or newer Pro, OR iPad Pro 2020+ (must have LiDAR)
- Mac (Apple Silicon sufficient)
- ~1.2m clear radius around the subject (≈2.5m total diameter), plain wall, even diffuse lighting
- Compression wear, bralette without molded cups
- Tailor's tape measure
- Chalk or skin-safe marker
- SMPL-X account (registered)

## 5. Source Materials (Required for AI Grounding)

These must be in the repo under `references/` before Phase 6. The PDFs are full books — relevant page ranges are extracted on-demand by the `Read` tool's `pages` parameter and cited by page number in `merged.yaml` `source` fields.

- `references/aldrich/` — Aldrich *Metric Pattern Cutting for Women's Wear*, **6th edition** photo scans (`IMG_*.jpg`) of the relevant pages plus `aldrich_p178_179_notes.md`. Authoritative source for the Aldrich measurement definitions (figure-measurement instructions), the standard size chart, and Phase-9 drafting formulas. Historical `aldrich_full.pdf` (5th edition) was removed; citations like `references/aldrich_full.pdf p.178 …` inside `merged.yaml` remain as provenance for entries originally derived from the 5th-ed text — the page numbers still apply to the 6th-ed chapter that supersedes them via the notes file.
- `references/dpm_bodice.pdf` — dresspatternmaking.com bodice block instructions (2025 edition). Canonical source for the dpm bodice measurement worksheet and Upper Bust contoured-path definition.
- `references/dpm_pants.pdf` — dresspatternmaking.com pants block instructions (2025 edition). Canonical source for trouser-related measurements (used in later phases if pants block is in scope).
- `references/dpm_videos/<topic>/` — per-topic folders containing `transcript.txt`, `transcript.srt`, and periodic `frame_*.jpg` extracted via whisper.cpp + ffmpeg (the `scripts/ingest_video.sh` helper that produced them was removed during the TailorTwin rename; the artefacts remain). Citable as `references/dpm_videos/<topic>/transcript.srt:<timestamp>` or `frame_NNNNNN.jpg`. Audio and source videos are gitignored (regenerable via yt-dlp).

  Folders hold two kinds of content from the same dpm series, kept under a single root for simplicity:
  - **Measurement-taking** videos (e.g. `bodice_measurements/`) — primary Phase 6 source for measurement definitions.
  - **Drafting tutorial** videos (e.g. `bodice_front/`, `bodice_back/`, `sleeve/`, `pants_1/`…`pants_5/`) — primary Phase 9 source for block drafting formulas, *and* secondary Phase 6 source because the drafting tutorials sometimes **revise** measurement definitions from the earlier measurement-taking videos.

  Conflict rule: when a drafting-video revision conflicts with a measurement-taking video, the drafting video wins (newer authoritative version). The conflict must be flagged in the relevant `merged.yaml` entry per the schema in Section 9: list both citations under `sources` (drafting first), set `superseded_by`, and explain in `notes`.
- `references/smplx_vertex_landmarks.md` — built incrementally during Phase 5. For every anatomical landmark (apex, suprasternal notch, C7, etc.), records the verified SMPL-X vertex ID with a screenshot. Never extended without visual verification in Blender.
- `references/construction_rules.md` — built during Phase 5. Documents each constructional line rule (CF, CB, SS, BL, WL, HL, neck base, etc.) with rationale.
- `references/calibration_log.md` — built during Phase 7. Records each calibration adjustment with date, before/after values, and rationale.

## 6. SMPL-X Downloads

From smpl-x.is.tue.mpg.de into `data/body_models/smplx/`:

- `models_smplx_v1_1.zip` (NPZ+PKL, 830 MB) — main model
- `smplx_parts_segm.pkl` — body part segmentation
- `smplx_uv_2023.zip` — latest UV map
- `V02_05.zip` — VPoser v2.0 pose prior
- `smplx_blender_addon-1.0.3-20260511.zip` — Blender visualization

## 7. Architecture

```
iPhone Pro (LiDAR)              Mac
+------------------+            +-----------------------------+
| Stray Scanner    |  AirDrop   | Python pipeline             |
| (capture app)    | ---------> |                             |
|                  |            |  RVM matte -> TSDF fuse     |
|                  |            |   (intrinsics rescaled,     |
|                  |            |    floor cropped)           |
|                  |            |  -> SMPL-X+D fit            |
|                  |            |  -> clean-fit (symmetrise,  |
|                  |            |     head/hand, A-pose)      |
|                  |            |  -> tape-anchor calibration |
|                  |            |  -> measurement extraction  |
|                  |            |  -> CSV + OBJ + SMIS        |
+------------------+            +-----------------------------+
                                          |
                                          v
                            HTML viewer (GUI) / Blender / MeshLab
                            Optional: SVG block generator
```

Local only. No cloud. Replace Stray Scanner with a custom iOS app in optional Phase 10.

Capture intrinsics note: Stray reports `fx/fy/cx/cy` in the RGB resolution
(e.g. 1920×1440) while depth is 256×192 — `scan.py` auto-detects the RGB
size from `rgb.mp4` and rescales, else the projection is off by the ~7.5×
ratio and the fuse shatters. Segmentation defaults to RVM person matting;
`depth_threshold` keeps the floor/walls and is debug-only.

## 8. Repository Layout

```
tailor-twin/
  README.md
  SPEC.md                     # this file
  GUARDRAILS.md               # AI-generation rules
  pyproject.toml              # [project.scripts] tailor-twin = …
  .gitignore
  references/                 # source materials (see Section 5)
  data/
    body_models/smplx/        # gitignored
    vposer/                   # gitignored
    captures/                 # gitignored
    results/                  # gitignored
  src/tailor_twin/
    cli.py                    # Typer console script entry
    scan.py                   # end-to-end pipeline (capture → measurements)
    preflight.py              # capture sanity-check
    history.py                # per-person measurement history (SQLite) + CLI
    manifest.py               # per-run provenance JSON
    io/stray_loader.py
    preprocess/
      depth_filter.py
      segment.py
    reconstruct/
      tsdf.py                 # TSDF fusion (default chamfer target)
      frames_cloud.py         # EXPERIMENTAL --no-fusion multi-frame cloud
      cleanup.py
    fit/
      fit.py                  # SMPL-X+D mesh/cloud fitter (SMPLify-X based)
      align.py                # initial rigid alignment
      losses.py               # chamfer + priors
      vposer.py               # VPoser wrapper
      clean_fit.py            # symmetrise, head/hand mask, canonical A-pose
      ring_deform.py          # tape-anchor ring calibration + audit
      ring_deform_cli.py
      refine_to_tape.py       # betas-solve alternative calibrator
      refine_to_tape_cli.py
    measure/
      definitions/
        merged.yaml           # Aldrich + dpm definitions (provenance record)
        aldrich_size_chart_p13.yaml
      landmarks.py            # SMPL-X vertex landmarks + dynamic searches
      primitives.py           # recipe primitives (girth, geodesic, plumb, …)
      mesh_ops.py             # slice + loop selection helpers
      regions.py              # body-region vertex masks
      seamly_catalog.py       # all 245 Seamly codes
      seamly_extractor.py     # Seamly-catalog runner
      synthetic.py            # synthetic-body validation harness
      bent_arm.py             # elbow re-pose
      extract_bent_arm.py     # CLI wrapper
      exports.py              # CSV / SMIS / OBJ writers
      viewer.py               # Dash review tool + plotly helpers
      cli.py                  # `python -m tailor_twin.measure.cli`
    gui/
      app.py                  # Flask factory + routes
      cli wiring via __init__:serve()
      runner.py               # subprocess state machine
      forms.py                # validate, build_cmd
      config.py
      viewer_data.py          # scan listing + polyline compute
      templates/              # index.html, viewer.html
      static/                 # styles.css, scan.js, viewer.js, favicon.svg
  scripts/                    # dev utilities only
    dump_recipe_table.py      # regenerates docs/catalog_coverage.md
    validate_synthetic.py     # synthetic-body harness CLI (ROADMAP A)
    landmark_editor.py        # photo-based landmark Y editor
    pick_vertex.py            # interactive Plotly vertex picker
  tests/                      # unit + snapshot regression (see tests/)
  docs/
    ROADMAP.md                # prioritized improvement plan
    recipes.md
    catalog_coverage.md       # auto-generated
```

## 9. Measurement Spec

Three primitive measurement types. Every measurement is one primitive with explicit parameters.

1. **planar_slice**: plane origin + normal + region mask → convex hull perimeter of the slice (mimics a tape measure)
2. **geodesic_path**: ordered vertex IDs → sum of geodesic distances along mesh surface (via `potpourri3d` heat method)
3. **contoured_path**: closed geodesic path through waypoints (used for Upper Bust, Neck base)

Plus a fourth class for completeness:

4. **planar_segment**: a planar slice clipped to a region (e.g. Aldrich Back Width = horizontal slice at C7-15cm, clipped to back armscye-to-armscye)

Every entry in `merged.yaml` has these fields:

```yaml
- name: <canonical_name>
  aliases: [<alt_names>]
  type: planar_slice | geodesic_path | contoured_path | planar_segment
  parameters:
    # type-specific
  sources:                         # required, list, primary first
    - <book page, worksheet item, or transcript:timestamp>
  source_classification: body | standard | derived
  notes: <optional>                # required when sources has > 1 entry
  superseded_by: <optional>        # cite the source that overrides an earlier one
```

When a dpm drafting video revises an earlier measurement-taking definition (see Section 5), `sources` must list both, the drafting-video citation goes first (precedence), and `notes` must summarize what was revised. Example:

```yaml
sources:
  - references/dpm_videos/bodice_front/transcript.srt:00:12:34
  - references/dpm_videos/bodice_measurements/transcript.srt:00:05:10
notes: "Bodice-front drafting video revises the bust-depth landmark from <X> to <Y>; the older measurement-taking video shows the deprecated landmark."
superseded_by: references/dpm_videos/bodice_front/transcript.srt:00:12:34
```

`source_classification`:
- `body`: measured directly on the body
- `standard`: comes from a size chart, not the body (e.g. Aldrich Dart, Armscye depth, Waist to hip)
- `derived`: computed from other measurements

### 9.1 Aldrich Measurements (Source: Aldrich p.178-179)

The 20 Aldrich measurements as defined on pages 178-179, with corrections from earlier drafts:

| # | Name | Type | Notes |
|---|---|---|---|
| 1 | Bust | planar_slice | Fullest point through apex, horizontal |
| 2 | Waist | planar_slice | At natural waist, narrowest between ribcage and iliac crest |
| 2a | Low waist | planar_slice | 6cm below natural waist |
| 3 | Hips | planar_slice | ~21cm below waist, widest below waist |
| 4 | Back width | planar_segment | 15cm below C7, armscye-to-armscye on back |
| 5 | Chest | planar_segment | 7cm below front neck point, armscye-to-armscye on front |
| 6 | Shoulder | geodesic_path | Neck-shoulder point to shoulder-bone (acromion) |
| 7 | Neck size | contoured_path | Around base of neck touching front collar bone |
| 8 | Dart | (standard) | From size chart |
| 9 | Top arm | planar_slice | Biceps with arm bent — needs posed measurement |
| 10 | Wrist | planar_slice | With slight ease |
| 11 | Ankle | planar_slice | Around ankle bone (lateral malleolus level) |
| 12 | High ankle | planar_slice | Just above ankle (document exact offset) |
| 13 | Nape to waist | geodesic_path | C7 down to string at waist |
| 14 | Front shoulder to waist | geodesic_path | Front shoulder, over bust apex, to waist (contoured) |
| 15 | Armscye depth | (standard) | From size chart |
| 16 | Skirt length | (user-specified) | Hem position, not a body measurement |
| 17 | Waist to hip | (standard) | From size chart |
| 18 | Waist to floor | geodesic_path | At centre back |
| 19 | Body rise | (special) | Seated measurement; approximate from standing scan |
| 20 | Sleeve length | geodesic_path | Shoulder bone, over elbow, to wrist bone (arm bent) — needs posed measurement |

**Important Aldrich-specific behaviours:**

- **#5 Chest is a width segment, NOT a circumference.** Earlier drafts of this spec described it as a circumference; that was wrong.
- **#4 Back width is also a width segment, not a circumference.**
- **#9 Top arm and #20 Sleeve length require a bent-arm pose.** Three implementation options:
  1. Capture a second scan with arm bent (Aldrich pose)
  2. Compute on T-pose mesh with mathematical adjustment for elbow bend
  3. Use SMPL-X posed model to virtually bend the arm before measuring
  Recommendation: option 3 (cleanest, no second capture, well-defined)
- **#19 Body rise is seated.** Approximation from a standing scan = waist line height minus crotch point height, taken at the side. Expect ~1-2 cm discrepancy versus a tape measurement; calibrate.
- **#8, #15, #17 are "standard" measurements**, indexed by bust size from Aldrich's size chart (page 13). The system can still compute body-derived values for comparison but the canonical value comes from the chart.

### 9.2 dresspatternmaking.com Measurements (Source: dpm worksheet image)

The 32 dpm measurements (numbered to match the worksheet):

| # | Name | Type | Notes |
|---|---|---|---|
| 1 | Upper Bust | contoured_path | Underarm at back/sides, above bust at front |
| 2 | Upper Bust Front Arc | (derived/half) | Upper Bust ÷ 2 |
| 3 | Upper Bust Back Arc | (derived/half) | Upper Bust ÷ 2 |
| 4 | Bust | planar_slice | Same as Aldrich #1 |
| 5 | Bust Front Arc | planar_segment | Front half of bust circumference, split at side seam |
| 6 | Bust Back Arc | planar_segment | Back half of bust circumference, split at side seam |
| 7 | Waist | planar_slice | Same as Aldrich #2 (worksheet uses ÷4 for quarter) |
| 8 | Shoulder Length | geodesic_path | Same as Aldrich #6 |
| 9 | Side Length | geodesic_path | Underarm to waist along side seam |
| 10 | Full Length Front | geodesic_path | Shoulder-neck point, over bust, to waist (contoured) |
| 11 | Centre Front Length | geodesic_path | Front neck point to waist, straight down CF |
| 12 | Across Shoulder Front | planar_segment | Width across front shoulders |
| 13 | Shoulder Slope Front (SSF) | geodesic_path | Shoulder-neck to shoulder-bone, projected to a vertical reference for "slope" calculation |
| 14 | Bust Depth (A) on SS | geodesic_path | Along shoulder slope from shoulder-neck point to apex |
| 15 | Bust Depth (B) on CF | (derived) | Horizontal distance from CF line to apex at bust line |
| 16 | Bust Depth (C) on FLF | (derived) | Distance along Full Length Front from start to apex |
| 17 | Bust Depth at Side Seam | geodesic_path | From SS at bust-line level to apex |
| 18 | Bust Span | planar_segment | Distance between left and right apex points, ÷2 on worksheet |
| 19 | Front Armhole Depth (A) FLF | geodesic_path | Armhole curve at FLF reference |
| 20 | Front Armhole Depth (B) CF | (derived) | Armhole curve at CF reference |
| 21 | Front Armhole Depth (C) SS | (derived) | Armhole curve at SS reference |
| 22 | Across Chest | planar_segment | Front width at upper chest level |
| 23 | Full Length Back | geodesic_path | Shoulder-neck point down back to waist |
| 24 | Centre Back Length | geodesic_path | Same as Aldrich #13 (C7 to waist) |
| 25 | Across Shoulder Back | planar_segment | Width across back shoulders |
| 26 | Shoulder Slope Back (SSB) | geodesic_path | Back-side equivalent of SSF |
| 27 | Back Armhole Depth (A) SSB | geodesic_path | Armhole curve at SSB |
| 28 | Back Armhole Depth (B) CF | (derived) | At CF reference |
| 29 | Back Armhole Depth (C) FLB | (derived) | At FLB reference |
| 30 | Back Neck | contoured_path | Back half of Aldrich #7 Neck Size |
| 31 | Across Back | planar_segment | Back width similar to Aldrich #4 but possibly at different height — verify against dpm reference |
| 32 | Dart Placement Back | (derived) | Drafting calculation, not a body measurement |

**Open questions for Phase 6** (resolve with dpm source, not AI):

- Does dpm #31 "Across Back" use the same 15cm-below-C7 height as Aldrich #4? Or a different reference?
- For the three-point armhole depths (#19-21, #27-29), what defines the armhole curve waypoints precisely? Likely requires interpretation; expose as tunable parameters.
- Bust Depth (B) on CF (#15) and (C) on FLF (#16): are these distances along the body surface or straight-line projections? Verify against dpm source.

## 10. Phased Plan

See Section 12 for AI guardrails per phase. Timelines assume part-time work (8-12 hr/week) with AI coding assistance.

| Phase | What | Duration | AI Risk |
|---|---|---|---|
| 0 | Repo setup, dependencies, SMPL-X load | 2 hr | Free |
| 1 | Stray Scanner capture + loader | 0.5 day | Free (verify against actual format) |
| 2 | Segmentation + TSDF fusion | 1-2 days | Free for Open3D; verify segmentation API |
| 3 | Landmarks + initial pose | 0.5-1 day | Free |
| 4 | SMPL-X+D fitting | 3-5 days | Risky — ground in SMPLify-X repo |
| 5 | Constructional lines on mesh | 2-3 days | Risky — rules are human decisions |
| 6 | Measurement extraction (all ~45) | 2-3 days | Risky — definitions must cite sources |
| 7 | Calibration against tape measurements | 3-5 days | Human-bottlenecked |
| 8 | SQLite storage + HTML worksheet | 0.5 day | Free |
| 9 | (Optional) SVG block generation | 3-5 days | Risky — formulas must cite Aldrich pages |
| 10 | (Optional) Custom iOS capture app | 1-2 months | Boilerplate free, ARKit specifics verify |

Total to working scan-to-worksheet pipeline: **~6 weeks part-time**. +1-2 weeks for SVG blocks. Phase 10 is a separate ~1-2 month project.

## 11. Capture Protocol (SOP)

Repeat identically every scan:

1. Clear ~1.2m radius around the subject (≈2.5m total diameter). Plain wall. Even diffuse lighting, no window backlight.
2. Compression wear. Hair tied back. Bralette without molded cups.
3. Stand on a marked spot on the floor (taped X).
4. A-pose: arms ~40° from body, palms forward, feet shoulder-width.
5. Tie a string around natural waist (per Aldrich #2 instructions) and tape-measure the vertical floor → string distance; enter it as the waist-height anchor (`--waist-height` / GUI **Waist height**).
6. Breathing: mid-exhale, relaxed.
7. Helper holds phone at chest height, ~1.2-1.5m away.
8. Press record. Helper walks slowly around you, ~25-30s for full loop.
9. Optional second pass at head height.
10. **For Aldrich top-arm and sleeve-length**: also capture a frame with the arm in Aldrich's pose (hand on hip, arm bent). Either as a separate short scan or as a second segment of the same capture.
11. Stop. Transfer to Mac.
12. Log: timestamp, time of day, clothing notes, anything unusual.

If alone: phone on tripod, rotate slowly in place on a turntable or with deliberate small foot steps.

## 12. AI Guardrails

Reproduced as `GUARDRAILS.md` in the repo root.

### 12.1 Three categories

**Free.** AI generates without grounding. Normal code review.
- File I/O, glue code, CLI wrappers, SQLite schemas, Jinja templates
- Standard Open3D / PyTorch / trimesh operations
- SVG plumbing, test scaffolding

**Risky.** AI may generate, but must be grounded in a verifiable source and verified.
- Library APIs that change (RVM, SAM2, smplx, potpourri3d): fetch current README into the session before generating
- SMPL-X fitting code: ground in SMPLify-X repo
- Constructional line *rules*: code is fine; rule decisions are human

**Forbidden.** AI must not generate from memory.
- Aldrich measurement definitions: transcribed from `references/aldrich_p178_179.pdf`
- dpm measurement definitions: from `references/dpm_worksheet.png`
- SMPL-X vertex IDs claimed as anatomical landmarks: verified in Blender, committed to `references/smplx_vertex_landmarks.md` with screenshot
- Aldrich drafting formulas (Phase 9): transcribed from book pages
- Physical measurement values: only the user with a tape measure produces these

### 12.2 Prompting rules

When asking AI for code that touches Risky or Forbidden territory:

- Include relevant source material as context (paste excerpts, fetch documentation)
- Instruct explicitly: "If you cannot find this in the provided source material, say so. Do not infer or fill in plausibly."
- Require source attribution in comments: "Each definition/formula must have a comment naming the source page or section."

If AI produces content without source attribution where requested, treat as suspect and verify.

### 12.3 SMPL-X vertex landmark guardrail

SMPL-X has 10475 vertices. AI often invents plausible vertex IDs that are wrong. Wrong IDs cause all downstream measurements to drift.

**Rule: never commit a vertex ID to YAML or code without visually verifying it in Blender.**

Workflow:
1. AI suggests a vertex ID
2. Open T-posed SMPL-X mesh in Blender, jump to that vertex (use the SMPL-X Blender addon)
3. If correct: commit to `references/smplx_vertex_landmarks.md` with screenshot
4. If wrong: search the region, find a better vertex, commit verified ID

Authoritative sources: smplx package's `vertex_ids.py`, MeshCapade docs, published SMPL-X measurement papers.

### 12.4 SMPL-X fitting guardrail

SMPL-X fitting has many published implementations with conflicting conventions (axis order, joint indexing, pose parameterization, regularizer weights). AI blends these and produces code that runs but fits incorrectly.

Rules:
- Treat SMPLify-X (https://github.com/vchoutas/smplify-x) as ground truth
- Before generating, have AI fetch SMPLify-X's `fit_single_frame.py` and follow its conventions
- Always validate visually: SMPL-X mesh + scan in the same coordinate system. Misaligned axes are the most common silent failure.
- Distance heatmap check (Phase 4 Stage C) is mandatory before trusting any fit for measurements
- If fit looks reasonable but measurements are systematically off, suspect fitting code before measurement code

### 12.5 Measurement definition guardrail

Every entry in `merged.yaml` must have a `source` field. No exceptions.

If an entry lacks a source:
1. Find the source and add it, OR
2. If no source exists, treat as user-invented: add a note explaining the rationale, OR
3. Delete it

AI-suggested entries pass only after manual source verification.

### 12.6 Drafting formula guardrail (Phase 9)

Aldrich's formulas are precise (e.g. front armhole depth, neck width, dart positions, ease allowances). AI from memory produces formulas that look right and aren't. Wrong formulas → blocks that don't fit.

Rules:
- Scan or photograph the relevant Aldrich pages for each block
- Paste page text/images into the AI session
- Each formula in code must have a comment with page reference
- Cross-check: compute a few outputs by hand from the book; confirm code matches

### 12.7 Red flags during development

Stop and verify when:
- AI confidently provides specific SMPL-X vertex indices without you having loaded a reference document
- AI cites a paper or repo with specific line numbers or function names you haven't shown it
- AI's formula uses a constant or coefficient absent from your source material
- Generated code uses import paths or class names not in the library's docs
- "This should work" without a way to test it

### 12.8 Where AI is genuinely useful

- Fast typist for code you've designed
- Rubber duck for debugging
- Translator between domains ("Aldrich formula text → NumPy")
- Reviewer of YAML definitions against worksheet images
- Doc reader (fetch current library docs, summarize the API)
- Visualization helper (Open3D / Matplotlib inspection plots)

### 12.9 Where AI is not the right tool

- Domain expert on pattern drafting
- Memory for SMPL-X vertex semantics
- Independent source of measurement formulas
- Judge of whether a fit is "good enough"

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Bust apex not localized precisely | SMPL-X+D + apex search within bust region; verify visually on first calibration scan |
| Side seam definition drifts between scans | Pick one rule (straight down from underarm through narrowest waist), document, never change without logging |
| Armhole depths sensitive to construction curve | Make armhole rule explicit and tunable |
| Stray Scanner format changes or app pulled | Save raw captures permanently; Phase 10 builds own app |
| SMPL-X fit poor on user's body type | Higher beta count (100-300), more displacement iterations, FEMALE/MALE rather than NEUTRAL |
| Bust circumference varies with bra | Document bra type per scan |
| Lone capture introduces motion | Turntable; reject blurry captures via sharpness check |
| Calibration tape values taken alone are wrong | Get help once for the first calibration pass |
| AI hallucinates measurement definitions | Section 12.5 — every definition has a source field |
| AI invents SMPL-X vertex IDs | Section 12.3 — verify in Blender before committing |
| AI writes plausible-wrong fitting code | Section 12.4 — ground in SMPLify-X |
| Aldrich seated body-rise differs from standing-scan derivation | Calibrate; document expected offset |
| Bent-arm Aldrich measurements differ from T-pose mesh | Use SMPL-X posed model to virtually bend arm before measuring |

## 14. Bootstrap (historical — phase 0)

Original bootstrap order, kept here for posterity:

1. `pyproject.toml` set up; `pip install -e .` provides the `tailor-twin` CLI
2. Source materials placed in `references/` (Section 5)
3. SMPL-X model file in `data/body_models/smplx/SMPLX_FEMALE.npz` (single-user, female-target — see Section 13)
4. Phase-0 sanity check: `import smplx; m = smplx.create("data/body_models", model_type="smplx", gender="female"); assert m.faces.shape == (20908, 3)`
   - Note: `smplx.create()` joins `model_path` with `model_type` when given a directory, so pass the **parent** of the `smplx/` subfolder.

## 15. Dependencies

```toml
[project]
name = "tailor-twin"
# Pinned to 3.12 because Open3D 0.19 (latest as of 2026-05) ships wheels only
# up to 3.12. Revisit when Open3D supports 3.13+.
requires-python = ">=3.12,<3.13"
dependencies = [
  "numpy>=1.26",
  "scipy>=1.11",
  "torch>=2.2",
  "smplx>=0.1.28",
  "open3d>=0.18",
  "trimesh>=4.0",
  "potpourri3d>=0.0.8",
  "opencv-python>=4.9",
  "pillow>=10.0",
  "scikit-image>=0.22",
  "networkx>=3.2",
  "pyyaml>=6.0",
  "rich>=13.0",
  "typer>=0.12",
  "jinja2>=3.1",
  "svgwrite>=1.4",
]
```

Plus Blender (with SMPL-X addon) and MeshLab for visual inspection.

## 16. Open Questions

Resolve at relevant phase using source books, not AI:

- ~~Aldrich #4 Back Width vs dpm #31 Across Back: same height (15cm below C7) or different?~~ **Resolved:** different. Aldrich #4 is taken at a fixed 15 cm below C7 (book p.178 item 4); dpm #31 is taken at half of the C7-to-upper-bust vertical distance (dpm bodice_measurements transcript L267-269), which is figure-dependent. The two are not aliases and merged.yaml has them as separate entries.
- dpm three-point armhole depth waypoints: exact definition from dpm source
- dpm bust depth B (CF) and C (FLF): along-surface or straight-line?
- Body rise: expected offset between Aldrich seated and standing-scan derivation
- Optimal vertex IDs for: apex (left/right), suprasternal notch, C7, front neck point, shoulder-neck point, shoulder bone (acromion), underarm/armpit point
- Canonical *measurement* pose for `measure/normalize.py` (re-pose target after un-posing from the A-pose fit):
  - Default torso-and-leg pose: pure T-pose vs "modified T" with arms lowered ~10-15° to match Aldrich p.178-179 figure. Exact joint angles to fix during Phase 6.
  - Aldrich bent-arm pose for #9 Top arm and #20 Sleeve length: shoulder + elbow joint angles to fix from Aldrich source.
  - Whether armhole-depth measurements (#19-21, #27-29) need a third intermediate pose or work in the default. Decide once the construct/armhole.py rules are written.

These get resolved during Phase 5-7, not before starting.
