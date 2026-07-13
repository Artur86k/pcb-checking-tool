"""Audit BOM labels against the photos, immune to per-site memorization.

The BOM says which positions are populated, but the photographed unit may
have genuinely missing parts (or stuffed DNP positions). Training directly
on BOM labels teaches the CNN to call such sites whatever the BOM says —
hiding real defects. This tool runs a k-fold cross-validation over refdes:
every site is predicted by a model that never saw that site, so a
confident, photo-consistent disagreement with the BOM marks a label
suspect. Suspects are written to golden/label_suspects.json, which
train_presence_cnn.py excludes from training.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from overlay_tool.pnp import load_pnp

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDS = 5
EPOCHS = 12
SEED = 0
BARE_CONSENSUS = 0.35     # mean p(present) below -> populated label suspect
PRESENT_CONSENSUS = 0.65  # mean p(present) above -> DNP label suspect

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

d = np.load(BASE + "/golden/presence_dataset.npz")
X, y = d["X"], d["y"].astype(np.int64)
refdes, photo = d["refdes"], d["photo"]

fp_by_ref = {p.refdes: p.footprint
             for p in load_pnp(BASE + "/pick_place/Pick Place for SNT_rev3_1.txt")}
fp = np.array([fp_by_ref.get(r, "?") for r in refdes])


class Crops(Dataset):
    def __init__(self, X, y, augment):
        self.X, self.y, self.augment = X, y, augment
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        img = self.X[i].astype(np.float32) / 255.0
        if self.augment:
            if rng.random() < 0.5:
                img = img[:, ::-1]
            if rng.random() < 0.5:
                img = img[::-1]
            s = int(rng.integers(52, 65))
            oy, ox = rng.integers(0, 64 - s + 1, 2)
            img = cv2.resize(np.ascontiguousarray(img[oy:oy + s, ox:ox + s]),
                             (64, 64), interpolation=cv2.INTER_LINEAR)
            if rng.random() < 0.5:
                img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.4, 1.8)))
            img = img * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08)
            img += rng.uniform(-0.05, 0.05, size=3)
            img = np.clip(img, 0, 1)
        t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return t.float(), self.y[i]


def block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                         nn.ReLU(), nn.MaxPool2d(2))


def make_net():
    return nn.Sequential(block(3, 16), block(16, 32), block(32, 64),
                         block(64, 128), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, 2))


uniq = rng.permutation(np.unique(refdes))
folds = np.array_split(uniq, FOLDS)
oof = np.full(len(y), np.nan)  # out-of-fold p(present)

for k, held in enumerate(folds):
    te = np.isin(refdes, held)
    tr = ~te
    net = make_net()
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    cls_counts = {c: max(int((y[tr] == c).sum()), 1) for c in (0, 1)}
    fp_counts = {f: int((fp[tr] == f).sum()) for f in np.unique(fp[tr])}
    w = np.array([1.0 / np.sqrt(cls_counts[c]) / np.sqrt(fp_counts[f])
                  for c, f in zip(y[tr], fp[tr])])
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                    num_samples=int(tr.sum()), replacement=True)
    dl = DataLoader(Crops(X[tr], y[tr], True), batch_size=128, sampler=sampler)
    for ep in range(EPOCHS):
        net.train()
        for xb, yb in dl:
            opt.zero_grad()
            crit(net(xb), yb).backward()
            opt.step()
        sched.step()
    net.eval()
    te_idx = np.where(te)[0]
    with torch.no_grad():
        for i0 in range(0, len(te_idx), 256):
            idx = te_idx[i0:i0 + 256]
            t = torch.from_numpy(
                X[idx].astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
            p = np.mean([torch.softmax(net(v), 1)[:, 1].numpy()
                         for v in (t, t.flip(3), t.flip(2), t.flip(2).flip(3))],
                        axis=0)
            oof[idx] = p
    print(f"fold {k + 1}/{FOLDS} done ({int(te.sum())} samples)")

suspects = {}
for ref in np.unique(refdes):
    m = refdes == ref
    mp = float(np.nanmean(oof[m]))
    label = int(y[m][0])
    if label == 1 and mp < BARE_CONSENSUS:
        suspects[ref] = {"bom": "populated", "looks": "bare", "mean_p": round(mp, 3)}
    elif label == 0 and mp > PRESENT_CONSENSUS:
        suspects[ref] = {"bom": "DNP", "looks": "present", "mean_p": round(mp, 3)}

print(f"\n{len(suspects)} label suspects (board disagrees with BOM):")
for ref, info in sorted(suspects.items(), key=lambda kv: kv[1]["mean_p"]):
    print(f"  {ref}: BOM {info['bom']} but looks {info['looks']} "
          f"(mean p(present) {info['mean_p']})")

out = BASE + "/golden/label_suspects.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(suspects, f, indent=2, ensure_ascii=False)
print("saved", out)
