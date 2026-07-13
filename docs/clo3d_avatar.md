# CLO3D avatar workflow (Workstream C)

Goal 2 of the project: an accurate 3D avatar with the real body
measurements, usable in CLO3D for virtual garment fitting. The pipeline
already produces the mesh — this doc is the import + verification
procedure, to be completed on a machine with CLO3D installed.

Items marked **TODO(user)** must be verified inside CLO3D and filled in
here — do not write dialog names or settings from memory.

## What the pipeline exports

Every run writes `<prefix>_fit_body.obj`:

- The fitted **SMPL-X+D body** (not the raw scan) — watertight, 10 475
  vertices, in the canonical **A-pose** (default 30° arms; set with
  `--apose-deg`, purely cosmetic since measurements are pose-normalized
  before extraction).
- **Units: metres**, +Y up, +Z facing the viewer. A ~1.6 m person spans
  Y ≈ [-1.2, 0.4] roughly; the body is NOT floor-anchored at Y=0.
- Tape-anchored runs export the **calibrated** mesh — the same body the
  CSV/SMIS numbers were measured on, so girths in CLO3D should agree
  with the CSV by construction.
- Head and hands are the clean SMPL-X template (`clean_fit.py`) — fine
  for garment fitting, not a face likeness.
- No texture coordinates yet (UV export is a planned extension — see
  ROADMAP C item 3).

## Import procedure

1. In CLO3D, import the OBJ **as an avatar**, not as garment geometry.
   **TODO(user):** record the exact menu path used (e.g. File → Import →
   OBJ, and the "Load as Avatar" option name in your CLO3D version).
2. Set the import **unit to metres**. This is the step that goes wrong:
   OBJ files carry no units, and a wrong choice gives a 100× avatar.
   **TODO(user):** record the exact unit dropdown value and the scale %
   shown, if any.
3. If the avatar floats or sinks relative to CLO3D's floor, use the
   import's translation/ground option. **TODO(user):** note the setting.

## Verification checklist (do once per pipeline change)

Compare CLO3D's own tape tool against the run's CSV — this closes the
loop between our extractor and the garment software:

| Check | Our value (CSV code) | CLO3D tape | Pass if |
|---|---|---|---|
| Total height | `A01` | floor → crown | ±0.5 cm |
| Bust circumference | `G04` | around fullest bust | ±0.5 cm |
| Waist circumference | `G07` | at the pinned waist line | ±0.5 cm |
| Hip circumference | `G09` | around seat | ±0.5 cm |
| Back neck → waist | `H41` | CB nape → waist | ±1.0 cm |

**TODO(user):** run this once and record the numbers + CLO3D version
here. If a girth disagrees while A01 matches, suspect the measuring
*height* CLO3D's tape was placed at (our slice heights are defined by
landmarks, not proportions).

## Known gaps / planned

- **UVs** (ROADMAP C item 3): needs `smplx_uv_2023.zip` from the SMPL-X
  site (already listed in SPEC §6 downloads); then `exports.write_obj`
  can emit `vt` rows behind an `--obj-uv` flag.
- **Avatar pose GUI field** (ROADMAP C item 4): `--apose-deg` exists
  end-to-end; surface it on the Calibration card if reposing in CLO3D
  is inconvenient.
- **A-pose arm clearance**: if 30° arms collide with sleeves during
  drape, re-run with a larger `--apose-deg` — measurements are
  unaffected.
