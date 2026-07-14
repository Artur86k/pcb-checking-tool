# Training the presence CNN

Step-by-step instruction for creating the classifier the **Check presence**
button uses (`golden/presence_cnn.pt`). Background and rationale:
[how-it-works.md](how-it-works.md#training-flow).

## 0. Prerequisites

```
pip install opencv-python numpy matplotlib ezdxf rawpy pillow pandas openpyxl torch
```

CPU torch is enough — training takes minutes.

Project data expected next to `overlay_tool/` (paths are constants at the
top of each `tools/` script — edit them for your board):

```
board_outline/<your board>.DXF     board outline
pick_place/<your board>.txt        Altium pick&place export
bom/<your board>.xlsx              BOM; a "Не устанавливается" row lists DNP refdes
pcb_photo_samples/                 photos of assembled board(s), whole board in frame
```

## 1. Collect photos

- Any number of boards of the *same design*; every photo of every unit is a
  training sample for all ~1000+ positions at once. 15–20 photos total is
  already enough for ~99% held-out accuracy.
- Both sides, whole board in frame, roughly perpendicular, any uniform
  backdrop (dark fabric and colored plastic both work). The sharpest camera
  mode pays off directly on 0201/0402 parts.
- Name files so that the date prefix (`YYYYMMDD_...`) distinguishes physical
  units — the audit uses it as the unit id.
- Photos of *other* board designs in the folder are fine: a content gate
  rejects them automatically.

## 2. Build the dataset

Edit the constants at the top of `tools/build_presence_dataset.py`:
`KNOWN` — one photo per side you have verified visually (used as reference
for automatic side detection of all others), and the outline / P&P / BOM
paths. Then:

```
python tools/build_presence_dataset.py
```

For every photo it prints the detected side, registration IoU and the
distortion-field fit; skipped photos are listed with the reason
(`different design or bad registration`). Output:
`golden/presence_dataset.npz` — body-aligned 64×64 crops of every
component position from every photo, labeled populated/DNP from the BOM.

## 3. Audit the labels  (do not skip)

```
python tools/audit_labels.py
```

Photographed units may genuinely lack BOM-required parts (or have solder
on DNP pads). Trained on raw BOM labels, the CNN memorizes the wrong
answer and the defect passes as OK forever. The audit cross-validates over
refdes — every site is predicted by a model that never saw it — and writes
`golden/label_suspects.json` with per-unit conflicts, e.g.:

```
20260712:XW1: BOM populated but looks bare (mean p(present) 0.0)
```

Review this list: it doubles as an assembly report for the photographed
units. Sites listed here are excluded from training and evaluation.

## 4. Train

```
python tools/train_presence_cnn.py
```

~19k samples / 25 epochs ≈ minutes on CPU. Reads the dataset and the
suspects file, prints held-out metrics every 5 epochs and writes
`golden/presence_cnn.pt` (TorchScript).

What to expect in the log:

- `test-photo` — accuracy on photos fully held out of training (set the
  `TEST_PHOTOS` constant to one good photo per side): capture robustness.
- `test-site` — accuracy on refdes never seen in training: generalization
  to new positions. Both should land near 0.99 / 0.98.
- Per-refdes error list — a handful of scattered singletons is normal;
  the same refdes failing on *many* photos means a label or footprint
  problem (check it in the photos, rerun the audit).

## 5. Use it

Restart the app — `Check presence` picks the model up automatically from
`golden/presence_cnn.pt` and reports "Presence (CNN)" in the status bar
(falls back to the color heuristic when the file or torch is missing).

## Retraining after new photos

Drop the new photos into `pcb_photo_samples/` and repeat steps 2–4 (the
whole cycle is ~30 min, mostly hands-off). Retrain whenever you add a new
unit, a new background/lighting setup, or a new camera mode — each new
condition the model has seen makes it more robust.

## Training design notes (why the scripts do what they do)

- **Oversampling by class × footprint rarity** — hundreds of chip passives
  otherwise dominate the loss and the few odd packages (SOT343, X2SON,
  interposer BGAs) stay underfitted → false "missing" on exactly the
  distinctive parts.
- **Blur / scale / flip / color augmentation** — capture sharpness varies
  between photos; without blur augmentation small parts on softer photos
  read as bare.
- **Flip-averaged inference** (in `presence_cnn.py`) stabilizes borderline
  small parts.
- **Threshold 0.5** favors catching missing parts (bare called "present"
  is the dangerous error); lowering it trades fewer false alarms for more
  missed defects.
