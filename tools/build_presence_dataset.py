"""Build the presence-CNN dataset: 64x64 body-aligned crops from all photos.

Side of each photo is auto-verified: register under both hypotheses and
compare the warped result against a reference warp of a known photo.
"""
import sys, glob, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from overlay_tool.outline import load_outline
from overlay_tool.pnp import load_pnp
from overlay_tool.bom import load_bom
from overlay_tool.register import load_photo, register_photo
from overlay_tool.presence import extract_roi

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWN = {"top": BASE + "/pcb_photo_samples/20260712_174803.jpg",
         "bottom": BASE + "/pcb_photo_samples/20260712_180418.dng"}
SIZE = 64
PPMM = 20.0
MARGIN = 0.4

o = load_outline(BASE + "/board_outline/SNT_rev3_1_board_outline.DXF")
parts = load_pnp(BASE + "/pick_place/Pick Place for SNT_rev3_1.txt")
bom = load_bom(BASE + "/bom/P71_TXRX_rev3_1.xlsx")
parts = [p for p in parts
         if o.xmin - 1 <= p.x_mm <= o.xmax + 1
         and o.ymin - 1 <= p.y_mm <= o.ymax + 1
         and p.refdes in bom]

def small_warp(path, side):
    img = load_photo(path)
    r = register_photo(img, o, side=side)
    return r, cv2.resize(r.warped_rgb, (128, 400))

refs = {}
for side, path in KNOWN.items():
    r, thumb = small_warp(path, side)
    refs[side] = thumb.astype(np.float32)

def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

photos = []
for path in sorted(glob.glob(BASE + "/pcb_photo_samples/*")):
    if not path.lower().endswith((".jpg", ".dng")):
        continue
    best = None
    for side in ("top", "bottom"):
        try:
            r, thumb = small_warp(path, side)
        except Exception as ex:
            continue
        c = corr(thumb.astype(np.float32), refs[side])
        if best is None or c > best[2]:
            best = (side, r, c)
    name = os.path.basename(path)
    if best is None or best[2] < 0.5 or best[1].iou < 0.95:
        print(f"skip {name}: corr={0 if best is None else round(best[2],2)}")
        continue
    photos.append((path, best[0], best[1]))
    print(f"{name}: {best[0]} corr={best[2]:.2f} iou={best[1].iou:.3f}")

X, y, ref_ids, photo_ids, sides = [], [], [], [], []
for path, side, r in photos:
    full = load_photo(path, max_dim=100000)
    f = full.shape[1] / cv2.resize(full, (1, 1)).shape[1]  # placeholder
    small_w = load_photo(path).shape[1]
    f = full.shape[1] / small_w
    for p in parts:
        if p.side != side:
            continue
        roi = extract_roi(full, p, o, r.homography, f, ppmm=PPMM,
                          margin_mm=MARGIN)
        if p.side == "bottom":
            roi = roi[:, ::-1]
        X.append(cv2.resize(roi, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
        y.append(0 if bom[p.refdes].dnp else 1)
        ref_ids.append(p.refdes)
        photo_ids.append(os.path.basename(path))
        sides.append(side)
    print(os.path.basename(path), "done", len(X))

np.savez_compressed(BASE + "/golden/presence_dataset.npz",
                    X=np.stack(X), y=np.array(y),
                    refdes=np.array(ref_ids), photo=np.array(photo_ids),
                    side=np.array(sides))
print("saved", len(X), "samples,",
      int(np.sum(np.array(y) == 0)), "bare /", int(np.sum(np.array(y) == 1)),
      "populated")
