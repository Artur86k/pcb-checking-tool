# CLAUDE.md — Phone-Photo AOI (Golden-Reference Board Inspection)

## Goal

Given handheld photos of a populated PCB (Samsung S23 Ultra), detect placement defects by
comparing each component against the design intent:

- **Wrong / missing / extra parts** vs. the pick-and-place (PnP) file.
- **Placement offset** (XY shift, tombstone, billboard) beyond per-package tolerance.
- **Rotation errors**, including 180° polarity flips.
- **Wrong-family swaps** via body color (ΔE in Lab).
- **Coilcraft dot color** vs. the expected value from the BOM part number.
- **DNP bidirectional check**: every DNP position must be a bare pad; every populated
  position in the PnP must have a part.

The photo is just an unusual capture device. The whole approach is: **register the image to
the PnP coordinate frame so every refdes becomes an addressable ROI, then inspect known
locations — never search blindly.**

## Non-negotiable design principles

1. **Registration caps everything.** No measurement is trusted until the fiducial
   reprojection residual is validated. Log it every run; fail loud above threshold.
2. **Golden-reference differencing beats rules.** Whenever a known-good board is available,
   cache per-refdes reference crops/metrics and diff against them. Rules-only is the fallback,
   not the default.
3. **Color decisions are gated on color calibration.** The Coilcraft dot is often the *only*
   value discriminator on otherwise-identical parts. Never emit a color/dot verdict from
   uncorrected pixels — flag low confidence instead of guessing.
4. **Inspect per-refdes ROIs, not the whole frame.** Global object detection is not used for
   the primary path.

## Tech stack

- Python 3.11+
- OpenCV (`cv2`) — calibration, undistort, homography, phase correlation, FFT
- NumPy — FFT / cross-power spectrum math
- scikit-image — supplementary (regionprops, color, feature)
- Tesseract (`pytesseract`) — optional IC top-marking OCR
- Pillow — EXIF / IO
- pandas — PnP + BOM parsing and the output report
- pydantic — typed config and per-refdes records
- Optional: a small scikit-learn / lightweight CNN classifier ONLY if rules+golden don't
  cleanly separate presence/color cases. Do not reach for ML first.

Keep it dependency-light. No heavy detection frameworks unless a stage provably needs one.

## Repository layout

```
aoi/
  config.py            # pydantic settings: thresholds, paths, package-class tolerances
  io/
    pnp.py             # parse pick-and-place -> DataFrame[refdes,x,y,rot,side,footprint]
    bom.py             # parse BOM -> refdes -> {part_number, dnp, coilcraft_dot?}
    capture.py         # EXIF, load, orientation normalize
  calib/
    geometric.py       # calibrateCamera, undistort (checkerboard-derived intrinsics)
    color.py           # ColorChecker -> CCM (3x3 / root-polynomial), apply_ccm()
  register/
    fiducials.py       # detect fiducials/mounting holes, sub-pixel centroids
    homography.py      # solve board-mm <-> image-px, reprojection residual
    roi.py             # project PnP row -> pixel ROI using footprint dims + rotation
  golden/
    build.py           # run a known-good board, cache ref crops + metrics
    store.py           # per-refdes reference: crop, mean Lab, dot Lab, FFT template
  inspect/
    presence.py        # component-vs-bare-pad (FFT HF-energy ratio + edge/color)
    offset.py          # phase correlation -> sub-pixel XY shift -> mm
    rotation.py        # 2D-FFT / log-polar coarse angle + polarity feature disambig
    color.py           # body mean Lab, deltaE vs golden/reference
    coilcraft.py       # segment dot, corrected Lab, nearest-match to expected
    ocr.py             # optional Tesseract top-marking check
  decide/
    rules.py           # aggregate per-refdes flags -> verdict
    report.py          # annotated overlay PNG + CSV keyed by refdes
  cli.py               # `aoi inspect <board.jpg>`; `aoi build-golden <good.jpg>`
tests/
  fixtures/            # sample crops, synthetic ROIs, tiny PnP/BOM
```

## Data contracts

**PnP row** (normalize headers on ingest; vendors differ):
`refdes: str, x_mm: float, y_mm: float, rot_deg: float, side: {top,bottom}, footprint: str`

**BOM row:**
`refdes: str, part_number: str, dnp: bool, coilcraft_dot: Optional[str]`
Join PnP↔BOM on `refdes`. A refdes in PnP but `dnp=True` in BOM is a **DNP position** and
must be bare. A refdes populated in PnP with no BOM part is a data error — surface it.

**Per-refdes result** (one row in the report):
```
refdes, status{OK,ABSENT,PRESENT_ON_DNP,OFFSET,ROTATION,COLOR,DOT,OCR,LOW_CONF},
offset_mm, rot_error_deg, body_deltaE, dot_expected, dot_measured,
presence_conf, reg_residual_px, notes
```

## Pipeline stages & acceptance criteria

### 1. Calibration (once per camera mode + lighting)
- Geometric: checkerboard in the **same 200MP/focus mode** used for boards → `calibrateCamera`
  → undistort every frame. Edge radial distortion corrupts XY more than anything else.
- Color: ColorChecker (or known-patch card) in-frame → CCM per lighting setup → apply before
  ANY color metric. Phone AWB shifts dot colors otherwise.
- Lighting guidance in `docs/capture.md`: diffuse / near-coaxial to kill ENIG/HASL glare,
  shoot perpendicular. Specular highlights on joints are the top presence-detection breaker.

### 2. Registration
- Detect fiducials (blob → intensity-weighted centroid or circle fit, sub-pixel). Fall back to
  mounting-hole centers or two diagonal outline corners if no fiducials.
- Solve homography (≥4 pts; affine if only 3 and board is flat/square).
- **Gate:** reprojection residual < `MAX_RESIDUAL_PX` (default 2.0 px). Above → abort with a
  clear message, don't emit measurements.

### 3. Golden build (if a known-good board exists)
- Run stages 1–2, cache per-refdes: reference crop, mean Lab, dot Lab, FFT template.

### 4. Per-component inspection (loop over projected ROIs)
- **Presence / DNP:** classify component vs. bare pad. Primary signal = FFT high-frequency
  energy ratio (component body vs. laminate/silkscreen) + edge density + color. Bidirectional:
  live position empty → `ABSENT`; DNP position stuffed → `PRESENT_ON_DNP`.
- **Offset:** phase correlation (cross-power spectrum peak) live↔golden ROI → sub-pixel shift
  → mm via homography scale. Threshold per package class.
- **Rotation:** coarse angle from 2D-FFT magnitude / log-polar vs. golden (resolves mod 90/180)
  + polarity feature (pin-1 dot, cathode band, tantalum/MLCC mark) to disambiguate the
  0/90/180/270 quadrant. FFT alone cannot catch a 180° flip — always combine.
- **Color:** body mean Lab, ΔE vs golden/reference. Catches wrong-family swaps.
- **Coilcraft dot:** segment dot within the inductor ROI, corrected Lab, nearest-match to
  expected color from BOM part number. Gate on step-1 color calib; emit `LOW_CONF` not a
  forced verdict when match distance is high.
- **OCR (optional):** Tesseract on large laser-marked ICs → compare top-marking to BOM PN.

### 5. Decision & report
- Aggregate flags → per-refdes verdict.
- Annotated overlay: green/amber/red boxes on projected ROIs.
- CSV keyed by refdes with all measured values.
- Tune thresholds per package class: 0201 needs tighter mm offset but has fewer pixels →
  use phase-correlation confidence itself as a gate on the small parts.

## Thresholds (defaults — put in config, tune per board)

```
MAX_RESIDUAL_PX        = 2.0
OFFSET_MM_BY_CLASS     = {"0201":0.10,"0402":0.15,"0603":0.20,"SOT":0.25,"QFN":0.20}
ROT_ERR_DEG            = 8.0
BODY_DELTA_E           = 8.0     # CIEDE2000
DOT_DELTA_E            = 12.0    # above -> LOW_CONF, not fail
PRESENCE_CONF_MIN      = 0.6
```

## Known failure modes (check these first when results look wrong)

1. **Bad registration** — validate residual before trusting anything. Most "defects" are
   really misprojected ROIs.
2. **Glare → false ABSENT** — specular highlight reads as bare pad. Multi-angle capture or a
   polarizer. Consider a glare mask (very high-value pixel exclusion) in presence.py.
3. **0201/0402 near resolution limit** — if a single frame isn't sharp across the whole board,
   use focus-stacking capture (ULTRA_HIGH_RESOLUTION_SENSOR constraints on the S23 Ultra) and
   fuse before inspection.
4. **Uncorrected color** — any dot/color verdict without CCM applied is invalid by definition.
5. **PnP/BOM header drift** — normalize on ingest; assert every populated refdes joins.

## Conventions

- Keep each `inspect/*` module a pure function: `(roi, golden_ref, cfg) -> flag/metric`. No
  hidden global state; makes them unit-testable against `tests/fixtures/`.
- All geometry in board-mm internally; convert to px only at the homography boundary.
- Log `reg_residual_px` and per-stage timings every run.
- Prefer explicit `LOW_CONF` over silent wrong verdicts on color/dot/small parts.
- No network calls in the inspection path.

## CLI

```
aoi build-golden good_board.jpg --pnp board.pnp --bom board.csv --out golden/
aoi inspect dut_001.jpg --pnp board.pnp --bom board.csv --golden golden/ --report out/
```

## First tasks for Claude Code

1. Scaffold the repo layout above with typed stubs and a passing `pytest` on empty fixtures.
2. Implement `io/pnp.py` + `io/bom.py` with header normalization and the PnP↔BOM join +
   DNP flagging, with tests.
3. Implement `register/` (fiducial detection → homography → residual gate → ROI projection),
   validated on one real S23 capture with a visual overlay of projected ROIs.
4. Then `inspect/presence.py` (FFT HF-energy ratio) + the DNP bidirectional check.
5. Then `offset.py` (phase correlation) and `rotation.py`. Color/dot/OCR last.

Build and validate stage-by-stage against real captures; don't wire the full pipeline before
registration is proven on hardware.
