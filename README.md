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

## Presence check

The **Check presence** button flags not-mounted components (red frames,
hover for the score). Two backends:

- **CNN** (preferred): a small TorchScript classifier over body-aligned
  64×64 ROIs, used when `golden/presence_cnn.pt` exists and `torch` is
  installed. Train it on your own board set with
  `tools/build_presence_dataset.py` (crops + BOM labels from every photo
  of a known-good board) followed by `tools/train_presence_cnn.py`.
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
