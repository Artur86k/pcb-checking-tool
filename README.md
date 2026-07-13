# PCB Photo Overlay / AOI Tool

Inspect a populated PCB from handheld phone photos by registering them to the
board's design data — no fixture, no scanner.

The Tkinter app (`python -m overlay_tool`) shows two plots (top / bottom side).
A DXF board outline is rasterized and never moves; each opened photo is
segmented, aligned (rotation / mirror / scale / perspective, ECC-refined on
silhouettes) and warped into the outline's mm frame. A pick-and-place file
projects every component's body rectangle onto the photo, so each refdes
becomes an addressable ROI.

## Features

- **Registration**: board segmentation → minAreaRect coarse align (4 rotations
  × mirror) → coarse-to-fine ECC homography refinement; silhouette IoU is
  reported and warned about when low.
- **Pick & place overlay**: Altium text export parsing, footprint → body-size
  rule chain (explicit `WxH` in the name, 4-digit chip codes imperial/metric,
  package table, photo-measured entries). Sizes verified against real board
  photos; approximate guesses are drawn dashed.
- **RAW support**: Samsung compressed DNGs load via the embedded full-size JPEG
  preview when LibRaw can't unpack the raw stream.
- **UI**: wheel zoom at cursor, left-drag pan, hover shows refdes / footprint /
  size / rotation, photo opacity slider, overlay PNG export.

## Distortion self-calibration

A single homography can't absorb phone-lens radial distortion or board
flex — component frames drift off their parts toward the frame edges (up
to ~0.9 mm measured on an S23 Ultra). The presence check therefore
self-calibrates each photo on the components themselves: every part is
located by template-matching against its footprint group's median
template, a degree-3 polynomial offset field is robust-fitted to the
residuals, and the board raster is remapped accordingly
(`overlay_tool/distortion.py`). Median frame-to-part offset drops from
~0.22 mm to ~0.05 mm; the displayed overlay switches to the corrected
raster after a presence check. No checkerboard or calibration shots
needed — any populated board is its own target.

## Presence check

The **Check presence** button compares the board against the BOM (open it
with **Open BOM…**; the "Не устанавливается" row defines DNP positions).
Verdict frames: **green** = board matches the BOM (part mounted where
expected, or DNP/not-in-BOM position empty); **red** = mismatch (expected
part missing, or a part mounted on a DNP/unknown position). Hover a frame
for the verdict and score. Without a BOM every position is treated as
expected-populated. Two detection backends:

- **CNN** (preferred): a small TorchScript classifier over body-aligned
  64×64 ROIs, used when `golden/presence_cnn.pt` exists and `torch` is
  installed. Train it on your own board set with
  `tools/build_presence_dataset.py` (crops + BOM labels from every photo),
  then `tools/audit_labels.py` (k-fold audit that finds sites where the
  photographed unit credibly disagrees with the BOM — genuinely missing
  parts must not be learned as "populated"), then
  `tools/train_presence_cnn.py`.
  Unlike color rules it survives green interposer-mounted BGAs and
  translucent passives (~98% held-out accuracy on a 17-photo set).
- **Color heuristic** (fallback): fraction of solder-mask-colored pixels
  inside the body frame.

## Usage

```
pip install opencv-python numpy matplotlib ezdxf rawpy pillow pandas openpyxl
python -m overlay_tool [board_outline.dxf]
```

Open the outline DXF, the top/bottom photos and the pick & place file from the
toolbar. File choices are remembered between sessions
(`~/.pcb_overlay_tool.json`).

Photos should be shot as perpendicular as possible on a dark, matte
background; the whole board must be in frame.

## Roadmap

Per-refdes defect detection (missing / wrong / misplaced components) using the
projected ROIs: FFT high-frequency energy, edge density and color metrics
against a known-good board — see `CLAUDE.md` for the full pipeline plan.
