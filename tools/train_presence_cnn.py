"""Train the presence CNN on the multi-photo dataset."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from overlay_tool.pnp import load_pnp

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PHOTOS = {"20260712_175150.jpg", "20260712_180923.dng"}
SEED = 0

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

d = np.load(BASE + "/golden/presence_dataset.npz")
X, y = d["X"], d["y"].astype(np.int64)
refdes, photo = d["refdes"], d["photo"]

# site holdout: 12% of refdes, all their samples
uniq = np.unique(refdes)
test_sites = set(rng.choice(uniq, size=int(0.12 * len(uniq)), replace=False))
in_test_photo = np.isin(photo, list(TEST_PHOTOS))
in_test_site = np.isin(refdes, list(test_sites))
train_m = ~in_test_photo & ~in_test_site
testP_m = in_test_photo & ~in_test_site
testS_m = in_test_site & ~in_test_photo
print(f"train {train_m.sum()}  test-photo {testP_m.sum()}  test-site {testS_m.sum()}")

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
            # scale + shift: random sub-crop, resized back
            s = int(rng.integers(52, 65))
            oy, ox = rng.integers(0, 64 - s + 1, 2)
            img = cv2.resize(np.ascontiguousarray(img[oy:oy + s, ox:ox + s]),
                             (64, 64), interpolation=cv2.INTER_LINEAR)
            # blur: capture sharpness varies a lot between photos; small
            # parts on soft photos were misread as bare without this
            if rng.random() < 0.5:
                img = cv2.GaussianBlur(img, (0, 0),
                                       float(rng.uniform(0.4, 1.8)))
            img = img * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08)
            img += rng.uniform(-0.05, 0.05, size=3)  # per-channel color cast
            img = np.clip(img, 0, 1)
        t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return t.float(), self.y[i]

def block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                         nn.ReLU(), nn.MaxPool2d(2))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(block(3, 16), block(16, 32), block(32, 64),
                               block(64, 128), nn.AdaptiveAvgPool2d(1),
                               nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, 2))
    def forward(self, x):
        return self.f(x)

net = Net()
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
EPOCHS = 25
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

# Oversample by class AND footprint rarity: hundreds of 0402s dominate the
# loss otherwise, and the few SOT343/BGA-6/X2SON sites (which look nothing
# like chips) end up underfitted — those were the false 'missing' verdicts.
fp_by_ref = {p.refdes: p.footprint
             for p in load_pnp(BASE + "/pick_place/Pick Place for SNT_rev3_1.txt")}
fp = np.array([fp_by_ref.get(r, "?") for r in refdes])
fp_counts = {f: int((fp[train_m] == f).sum()) for f in np.unique(fp[train_m])}
cls_counts = {c: int((y[train_m] == c).sum()) for c in (0, 1)}
sample_w = np.array([
    1.0 / np.sqrt(cls_counts[c]) / np.sqrt(fp_counts[f])
    for c, f in zip(y[train_m], fp[train_m])])
sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                num_samples=int(train_m.sum()),
                                replacement=True)
train_dl = DataLoader(Crops(X[train_m], y[train_m], True), batch_size=128,
                      sampler=sampler, num_workers=0)

def evaluate(mask, name):
    net.eval()
    dl = DataLoader(Crops(X[mask], y[mask], False), batch_size=256)
    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            preds.append(net(xb).argmax(1).numpy())
    p = np.concatenate(preds)
    t = y[mask]
    acc = (p == t).mean()
    rec_bare = (p[t == 0] == 0).mean() if (t == 0).any() else float("nan")
    rec_pop = (p[t == 1] == 1).mean() if (t == 1).any() else float("nan")
    print(f"  {name}: acc {acc:.3f}  bare-recall {rec_bare:.3f} "
          f"pop-recall {rec_pop:.3f} (n={mask.sum()})")
    return p

for epoch in range(EPOCHS):
    net.train()
    tot = n = 0
    for xb, yb in train_dl:
        opt.zero_grad()
        loss = crit(net(xb), yb)
        loss.backward()
        opt.step()
        tot += float(loss) * len(yb)
        n += len(yb)
    sched.step()
    print(f"epoch {epoch}: loss {tot/n:.4f}")
    if epoch % 5 == 4 or epoch == EPOCHS - 1:
        evaluate(testP_m, "test-photo")
        evaluate(testS_m, "test-site ")

# final report + DD21
pP = evaluate(testP_m, "final test-photo")
pS = evaluate(testS_m, "final test-site ")

net.eval()
dd21 = refdes == "DD21"
with torch.no_grad():
    logits = net(torch.stack([Crops(X, y, False)[i][0]
                              for i in np.where(dd21)[0]]))
    prob = torch.softmax(logits, 1)[:, 1].numpy()
print("DD21 (populated on interposer) p(present) per photo:",
      np.round(prob, 3), "photos:", photo[dd21])

# misclassified sites on test-photo set (consistent errors are interesting)
errP = np.where(testP_m)[0][pP != y[testP_m]]
from collections import Counter
print("test-photo errors by refdes:",
      Counter(zip(refdes[errP], y[errP])).most_common(20))

torch.jit.script(net).save(BASE + "/golden/presence_cnn.pt")
print("saved golden/presence_cnn.pt")
