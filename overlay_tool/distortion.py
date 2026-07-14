"""Residual-distortion correction from the components themselves.

A single homography cannot absorb phone-lens radial distortion or board
flex; the leftover error shows up as component bodies sitting off their
projected frames (up to ~1 mm at the frame edges). The components are used
as a dense self-calibration target:

1. every populated part is located by template-matching against the median
   template of its footprint group from the same photo -> per-part offset
   vector (actual body vs projected frame position, in board mm);
2. a low-order 2D polynomial field is robust-fitted to those offsets
   (irls trimming kills bad matches and real placement outliers);
3. the field is applied as a remap on any raster warped from the photo, so
   frames and ROIs land on the parts.

Everything stays in board-mm; the field maps (x, y) -> (dx, dy) mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .outline import Outline
from .pnp import Part
from .presence import extract_roi

PPMM = 20.0          # measurement ROI resolution
SEARCH_MM = 0.6      # +- template search range
MIN_GROUP = 6        # members needed for a location template
MIN_SCORE = 0.4      # matchTemplate NCC below this -> ignore sample
POLY_DEG = 3
TRIM_SIGMA = 2.5
FIT_ROUNDS = 3


@dataclass
class OffsetField:
    """Polynomial offset field dx(x,y), dy(x,y) in board mm."""
    coef_x: np.ndarray
    coef_y: np.ndarray
    deg: int
    n_samples: int
    rmse_before: float
    rmse_after: float   # fit residual on inliers

    def __call__(self, x, y):
        A = _poly_terms(np.asarray(x, float), np.asarray(y, float), self.deg)
        return A @ self.coef_x, A @ self.coef_y


def _poly_terms(x, y, deg):
    # normalized to ~[-1,1] for conditioning; board coords are < 100 mm
    xn, yn = x / 100.0, y / 100.0
    cols = [np.ones_like(xn)]
    for d in range(1, deg + 1):
        for i in range(d + 1):
            cols.append(xn ** (d - i) * yn ** i)
    return np.stack(cols, axis=-1)


def _subpix(res, x, y, axis):
    if axis == 0:
        a, b, c = res[y, max(x - 1, 0)], res[y, x], res[y, min(x + 1, res.shape[1] - 1)]
    else:
        a, b, c = res[max(y - 1, 0), x], res[y, x], res[min(y + 1, res.shape[0] - 1), x]
    den = a - 2 * b + c
    return 0.0 if abs(den) < 1e-9 else float(np.clip((a - c) / (2 * den), -1, 1))


def measure_offsets(full_rgb: np.ndarray, outline: Outline, parts: list[Part],
                    side: str, photo_h: np.ndarray, photo_scale: float,
                    skip: set[str] | None = None) -> np.ndarray:
    """Per-part offsets. Returns (n,5): x_mm, y_mm, dx_mm, dy_mm, score.

    `skip`: refdes to exclude (known-DNP positions have no body to match).
    """
    sel = [p for p in parts if p.side == side
           and (skip is None or p.refdes not in skip)
           and outline.xmin - 1 <= p.x_mm <= outline.xmax + 1
           and outline.ymin - 1 <= p.y_mm <= outline.ymax + 1]
    gray = {}
    groups: dict[str, list[Part]] = {}
    for p in sel:
        roi = extract_roi(full_rgb, p, outline, photo_h, photo_scale,
                          ppmm=PPMM, margin_mm=SEARCH_MM)
        g = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray[p.refdes] = cv2.GaussianBlur(g, (0, 0), 1.0)
        groups.setdefault(p.footprint, []).append(p)

    m = int(round(SEARCH_MM * PPMM))
    out = []
    for members in groups.values():
        if len(members) < MIN_GROUP:
            continue
        tmpl = np.median(np.stack([gray[p.refdes][m:-m, m:-m]
                                   for p in members]), axis=0).astype(np.float32)
        if min(tmpl.shape) < 6:
            continue
        for p in members:
            res = cv2.matchTemplate(gray[p.refdes], tmpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x0, y0) = cv2.minMaxLoc(res)
            if score < MIN_SCORE:
                continue
            du = (x0 + _subpix(res, x0, y0, 0) - m) / PPMM
            dv = -(y0 + _subpix(res, x0, y0, 1) - m) / PPMM
            a = math.radians(p.rot_deg)
            c, s = math.cos(a), math.sin(a)
            out.append((p.x_mm, p.y_mm, du * c - dv * s, du * s + dv * c,
                        score))
    return np.array(out) if out else np.empty((0, 5))


def fit_field(samples: np.ndarray, deg: int = POLY_DEG) -> OffsetField | None:
    """Robust least-squares polynomial fit of the offset field."""
    if len(samples) < 4 * (deg + 1) * (deg + 2) // 2:
        return None
    x, y, dx, dy = (samples[:, i] for i in range(4))
    keep = np.ones(len(x), bool)
    coef_x = coef_y = None
    for _ in range(FIT_ROUNDS):
        A = _poly_terms(x[keep], y[keep], deg)
        coef_x, *_ = np.linalg.lstsq(A, dx[keep], rcond=None)
        coef_y, *_ = np.linalg.lstsq(A, dy[keep], rcond=None)
        Af = _poly_terms(x, y, deg)
        rx = dx - Af @ coef_x
        ry = dy - Af @ coef_y
        r = np.hypot(rx, ry)
        sigma = max(float(np.median(r[keep])) * 1.4826, 1e-3)
        keep = r < TRIM_SIGMA * sigma
    d0 = np.hypot(dx[keep], dy[keep])
    Af = _poly_terms(x[keep], y[keep], deg)
    r_in = np.hypot(dx[keep] - Af @ coef_x, dy[keep] - Af @ coef_y)
    return OffsetField(coef_x, coef_y, deg, int(keep.sum()),
                       float(np.sqrt((d0 ** 2).mean())),
                       float(np.sqrt((r_in ** 2).mean())))


def warp_board_raster(full_rgb: np.ndarray, outline: Outline,
                      photo_h: np.ndarray, photo_scale: float,
                      ppmm: float,
                      field: OffsetField | None = None) -> np.ndarray:
    """Warp the full-res photo into a board raster, optionally corrected.

    Returns an image of the whole board at `ppmm`, top-view frame, where
    the offset field (if given) has been compensated: content that sat
    displaced by D(x,y) is pulled back onto its nominal position.
    """
    z = ppmm / outline.scale
    T = np.array([[z, 0, 0], [0, z, 0], [0, 0, 1.0]])
    S = np.diag([1.0 / photo_scale, 1.0 / photo_scale, 1.0])
    H = T @ photo_h @ S
    w = int(round((outline.xmax - outline.xmin) * ppmm)) + 1
    h = int(round((outline.ymax - outline.ymin) * ppmm)) + 1
    raster = cv2.warpPerspective(full_rgb, H, (w, h), flags=cv2.INTER_LINEAR)
    if field is None:
        return raster
    # remap: corrected(u,v) = raster(u + dx*ppmm, v - dy*ppmm)
    u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    x_mm = u / ppmm + outline.xmin
    y_mm = outline.ymax - v / ppmm
    dx, dy = field(x_mm.ravel(), y_mm.ravel())
    map_x = (u + dx.reshape(h, w).astype(np.float32) * ppmm)
    map_y = (v - dy.reshape(h, w).astype(np.float32) * ppmm)
    return cv2.remap(raster, map_x, map_y, cv2.INTER_LINEAR)


def extract_patch(raster: np.ndarray, outline: Outline, part: Part,
                  ppmm: float, margin_mm: float = 0.0) -> np.ndarray:
    """Body-aligned patch of `part` from a board raster at `ppmm`."""
    a = math.radians(part.rot_deg)
    c, s = math.cos(a), math.sin(a)
    hl = part.length_mm / 2 + margin_mm
    hw = part.width_mm / 2 + margin_mm

    cx0, cy0 = part.center()

    def px(u, v):
        x = cx0 + u * c - v * s
        y = cy0 + u * s + v * c
        return ((x - outline.xmin) * ppmm, (outline.ymax - y) * ppmm)

    w_px = max(2, int(round(2 * hl * ppmm)))
    h_px = max(2, int(round(2 * hw * ppmm)))
    src = np.float32([px(-hl, hw), px(hl, hw), px(hl, -hw)])
    dst = np.float32([(0, 0), (w_px, 0), (w_px, h_px)])
    A = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(raster, A, (w_px, h_px), flags=cv2.INTER_LINEAR)
