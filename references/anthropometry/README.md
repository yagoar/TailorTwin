# Anthropometry sources — for Workstream D (plausibility sanity layer)

Status: **sources identified, data not yet downloaded.** This folder is
the provenance record + download instructions. The actual data files are
gitignored-large and were not fetchable from the session that wrote this
(the sandbox egress proxy denied the government/academic hosts); download
them on a machine with open network access, then a future session can
derive the regression coefficients here.

## What Workstream D needs

A *sanity check*, not a measurement source: given the user's height,
weight, and a couple of measured girths, predict the rest and flag any
extracted value that is implausibly far from the prediction — i.e. catch
a bad capture ("extracted thigh is 4 cm off what your hip + height
imply"). It must never overwrite a real measurement. Per `GUARDRAILS.md`
§12, every regression coefficient must trace to a citable source placed
here — none from model memory.

For that we need a dataset with, at minimum, **stature, weight, and the
girths the blocks use (bust/chest, waist, hip)** measured on the same
people, so girth-from-girth/height regressions can be fit.

## Candidate sources (researched 2026-07, ranked)

### Option 1 (RECOMMENDED, no download) — derive the prior from SMPL-X itself

The cleanest source is already in the repo. The SMPL-X female shape space
is a PCA over **CAESAR** (a large civilian 3-D scan survey), so sampling
betas and running our own extractor yields girth/height/volume tuples for
thousands of synthetic *civilian* bodies — self-consistent with the fit,
no external download, no military-population bias, no license question.

- **How:** extend the Workstream A synthetic harness — sample N≈2000
  bodies, extract the block-critical codes + a weight proxy (mesh volume ×
  ~1.01 g/cm³), fit the plausibility regressions on that table, commit the
  table + fitted coefficients.
- **What it validates:** internal consistency (does this body's waist
  match its hip/height *for a plausible human shape*). That is exactly
  what a capture-failure detector needs — it flags bodies the fit could
  not have produced from a real person.
- **What it does NOT give:** ground-truth population means. It inherits
  CAESAR's demographics. Fine for sanity; not a substitute for the user's
  own tape numbers.
- **Provenance to cite:** SMPL-X / CAESAR (Pavlakos et al., SMPL-X 2019;
  CAESAR survey). Coefficients derive from a committed, regenerable
  script — GUARDRAILS-clean because there are no memorized constants.

This is the recommended primary path: it removes the download blocker
entirely and is *more* appropriate for a civilian sewing user than a
military dataset.

### Option 2 — ANSUR II (only free dataset with the full girth set)

2012 Anthropometric Survey of U.S. Army Personnel. The only free,
citable dataset that measures **chest circumference, waist circumference
(natural + omphalion), and buttock/hip circumference plus stature and
weight on the same subjects**, so it supports true girth-to-girth
regressions.

- **Report:** Gordon, C.C., et al. (2014). *2012 Anthropometric Survey of
  U.S. Army Personnel: Methods and Summary Statistics.* Technical Report
  NATICK/TR-15/007. DTIC accession **ADA611869**.
  - PDF: `https://apps.dtic.mil/sti/tr/pdf/ADA611869.pdf`
- **Public data files:** `ANSUR II FEMALE Public.csv` (n≈1,986) and
  `ANSUR II MALE Public.csv` (n≈4,082), released for public use 2017.
  Mirrors (verify checksums against the report's counts):
  - OPEN Design Lab: `https://www.openlab.psu.edu/ansur2/`
  - Defense Centers for Public Health: `https://ph.health.mil/topics/workplacehealth/ergo/`
  - Community GitHub mirrors exist (e.g. `senihberkay/US-Army-ANSUR-II`) —
    only trust after diffing against an official mirror.
- **Units:** all dimensions in **mm** except `weightkg`.
- **Availability:** raw data approved for public release, distribution
  unlimited (per the 2017 Natick release; **confirm the exact
  distribution statement from the CSV header / report cover when
  downloaded** — marked TODO, do not assert a licence we haven't read).
- **CAVEAT (important):** military population — younger, fitter, lower and
  differently-distributed body fat than a general sewing customer. Use it
  for the *structure* of the regressions (which predictors matter, slopes)
  and treat absolute intercepts with suspicion. The female sample also has
  fewer high-BMI subjects than the civilian distribution.
- **Bust vs chest caveat:** ANSUR measures *chest circumference* at a
  defined bony landmark, not a bra bust girth over the fullest point. For
  a bodice block the relevant target is bust; document the offset rather
  than equating them.

### Option 3 — NHANES (best civilian marginals, incomplete girths)

CDC/NCHS, **public domain** (U.S. federal work, no copyright) — the
cleanest licence and a true civilian sample.

- Body Measures (BMX) files, per 2-year cycle, e.g.
  `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.htm`
- Summary report: *Anthropometric Reference Data for Children and Adults,
  United States* (Vital & Health Statistics, Series 3) — means, SDs,
  percentiles by age and sex.
- **LIMITATION:** recent NHANES cycles publish stature, weight, BMI, and
  **waist circumference** but **not bust/chest or hip circumference**, so
  NHANES alone cannot fit a bust/hip regression. (NHANES III, 1988–94,
  measured more girths — an option if older data is acceptable.)
- **Use:** validate the height / weight / waist *marginals* of Option 1 or
  2 against a real civilian distribution, and sanity-check that our
  synthetic or military-derived numbers aren't off in the population sense.

## Recommendation

Do **Option 1** (SMPL-X-derived, part of Workstream A) as the primary
plausibility prior — it is download-free, civilian-grounded, and
GUARDRAILS-clean. Optionally cross-check its marginals against **NHANES**
(Option 3) for realism. Reserve **ANSUR II** (Option 2) for when a true
measured girth-to-girth relationship is wanted and its military bias is
acceptable and documented.

In all cases the ultimate calibration is the **user's own tape numbers**,
now accumulating in `data/results/history.sqlite` — the plausibility
layer only has to catch gross capture failures, so a modest prior is
enough.

## Download blocker (this session)

The session that wrote this could not fetch the files: the sandbox egress
proxy answered 403 to CONNECT for `apps.dtic.mil`, `calmcode.io`,
`openlab.psu.edu`, `ph.health.mil`, and `wwwn.cdc.gov` (organization
network policy — not retried, per proxy README). Download on an
open-network machine, drop the CSV(s) + report PDF here, and record the
retrieval date + SHA-256 in this file before deriving coefficients.
