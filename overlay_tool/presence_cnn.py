"""CNN-based mounted/not-mounted classification of per-refdes ROIs.

A small CNN (TorchScript, trained on body-aligned 64x64 crops of this board
family labeled populated/bare from the BOM) replaces the color heuristic:
it handles cases hue rules cannot, e.g. parts whose top is itself a green
PCB (interposer-mounted BGAs) or translucent small passives.

The model file is looked up in ./golden/presence_cnn.pt (or an explicit
path). torch is an optional dependency — `available()` says whether the
CNN path can be used; callers fall back to the color heuristic otherwise.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .outline import Outline
from .pnp import Part
from .presence import PresenceResult, extract_roi

MODEL_FILE = os.path.join("golden", "presence_cnn.pt")
INPUT_SIZE = 64
PPMM = 20.0
MARGIN_MM = 0.4
BATCH = 256

_model = None
_model_path = None


def _find_model(path: str | None = None) -> str | None:
    p = path or MODEL_FILE
    return p if os.path.isfile(p) else None


def available(path: str | None = None) -> bool:
    if _find_model(path) is None:
        return False
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _load(path: str | None = None):
    global _model, _model_path
    import torch
    p = _find_model(path)
    if _model is None or _model_path != p:
        _model = torch.jit.load(p, map_location="cpu")
        _model.eval()
        _model_path = p
    return _model


def _select(outline: Outline, parts: list[Part], side: str) -> list[Part]:
    return [p for p in parts
            if p.side == side
            and outline.xmin - 1 <= p.x_mm <= outline.xmax + 1
            and outline.ymin - 1 <= p.y_mm <= outline.ymax + 1]


def _run(sel: list[Part], rois: list[np.ndarray], side: str,
         model_path: str | None, thresh: float) -> dict[str, PresenceResult]:
    import torch
    model = _load(model_path)
    crops, means = [], []
    for roi in rois:
        if side == "bottom":  # match training-time appearance normalization
            roi = roi[:, ::-1]
        means.append(roi.reshape(-1, 3).mean(axis=0))
        crops.append(cv2.resize(roi, (INPUT_SIZE, INPUT_SIZE),
                                interpolation=cv2.INTER_AREA))
    results: dict[str, PresenceResult] = {}
    with torch.no_grad():
        for i0 in range(0, len(crops), BATCH):
            batch = np.stack(crops[i0:i0 + BATCH]).astype(np.float32) / 255.0
            t = torch.from_numpy(batch.transpose(0, 3, 1, 2))
            # flip-averaged prediction (presence is flip-invariant):
            # stabilizes borderline small parts
            prob = np.mean([
                torch.softmax(model(v), dim=1)[:, 1].numpy()
                for v in (t, t.flip(3), t.flip(2), t.flip(2).flip(3))],
                axis=0)
            for p, m, pp in zip(sel[i0:i0 + BATCH], means[i0:i0 + BATCH], prob):
                results[p.refdes] = PresenceResult(
                    part=p, bare_frac=float(1.0 - pp),
                    mean_rgb=tuple(float(v) for v in m),
                    present=bool(pp >= thresh))
    return results


def classify(full_rgb: np.ndarray, outline: Outline, parts: list[Part],
             side: str, photo_h: np.ndarray, photo_scale: float,
             model_path: str | None = None,
             thresh: float = 0.5) -> dict[str, PresenceResult]:
    """Presence verdict for every part of `side` from the full-res photo.

    `photo_h` / `photo_scale` as in presence.extract_roi. Returns the same
    result type as the color heuristic; `bare_frac` holds 1 - p(present).
    """
    sel = _select(outline, parts, side)
    rois = [extract_roi(full_rgb, p, outline, photo_h, photo_scale,
                        ppmm=PPMM, margin_mm=MARGIN_MM) for p in sel]
    return _run(sel, rois, side, model_path, thresh)


def classify_raster(raster: np.ndarray, ppmm: float, outline: Outline,
                    parts: list[Part], side: str,
                    model_path: str | None = None,
                    thresh: float = 0.5) -> dict[str, PresenceResult]:
    """Same as classify(), but from a (distortion-corrected) board raster."""
    from .distortion import extract_patch
    sel = _select(outline, parts, side)
    rois = [extract_patch(raster, outline, p, ppmm, margin_mm=MARGIN_MM)
            for p in sel]
    return _run(sel, rois, side, model_path, thresh)
