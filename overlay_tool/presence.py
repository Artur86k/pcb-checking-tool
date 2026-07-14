"""Per-refdes presence metrics.

Pure functions: extract a body-aligned ROI for a part from the registered
photo, then compute scalar metrics that separate "component present" from
"bare pads" — FFT high-frequency energy ratio, edge density, color
saturation. Glare pixels are measured separately so a specular highlight
can be recognized instead of misread as a bare pad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .outline import Outline
from .pnp import Part

ROI_PPMM = 20.0      # ROI raster resolution, px per mm
ROI_MARGIN_MM = 0.2  # context kept around the body rectangle
GLARE_V = 250        # 8-bit value threshold for specular highlights
HF_CUTOFF = 0.15     # of Nyquist; energy above this radius counts as HF

# Mounted-check ("is there a part in the frame?") tuning
BARE_HUE_TOL = 14        # pixel counts as solder mask within this hue distance
BARE_SAT_MIN = 60        # ... and at least this saturation (8-bit)
BARE_VAL_MIN = 40        # ... and not near-black
BARE_FRAC_ABSENT = 0.45  # more solder-mask pixels than this -> not mounted


@dataclass
class RoiMetrics:
    hf_ratio: float       # high-frequency FFT energy / total AC energy
    edge_density: float   # mean Sobel gradient magnitude (0..1 scale)
    sat_mean: float       # mean HSV saturation (0..1)
    val_std: float        # brightness std dev (0..1)
    glare_frac: float     # fraction of near-saturated pixels
    px: tuple[int, int]   # ROI size, for diagnostics


def part_to_photo_homography(part: Part, outline: Outline,
                             photo_h: np.ndarray, photo_scale: float,
                             ppmm: float = ROI_PPMM,
                             margin_mm: float = ROI_MARGIN_MM,
                             ) -> tuple[np.ndarray, tuple[int, int]]:
    """Homography mapping full-res photo px -> body-aligned ROI px.

    `photo_h` is the registration homography (working-resolution photo px ->
    outline raster px); `photo_scale` = full-res width / working width.
    Returns (H, (roi_w, roi_h)).
    """
    hl = part.length_mm / 2 + margin_mm
    hw = part.width_mm / 2 + margin_mm
    w_px = int(round(2 * hl * ppmm))
    h_px = int(round(2 * hw * ppmm))

    # part-local (u,v) mm -> board mm -> outline raster px, for 3 anchors
    a = math.radians(part.rot_deg)
    c, s = math.cos(a), math.sin(a)

    cx0, cy0 = part.center()

    def raster(u, v):
        x = cx0 + u * c - v * s
        y = cy0 + u * s + v * c
        return ((x - outline.xmin) * outline.scale,
                (outline.ymax - y) * outline.scale)

    # ROI px corners <-> raster px (v up in mm, row down in ROI)
    src = np.float32([raster(-hl, hw), raster(hl, hw), raster(hl, -hw)])
    dst = np.float32([(0, 0), (w_px, 0), (w_px, h_px)])
    A = cv2.getAffineTransform(src, dst)          # raster -> ROI
    A3 = np.vstack([A, (0.0, 0.0, 1.0)])
    S = np.diag([1.0 / photo_scale, 1.0 / photo_scale, 1.0])
    return A3 @ photo_h @ S, (w_px, h_px)


def extract_roi(full_rgb: np.ndarray, part: Part, outline: Outline,
                photo_h: np.ndarray, photo_scale: float,
                ppmm: float = ROI_PPMM,
                margin_mm: float = ROI_MARGIN_MM) -> np.ndarray:
    H, size = part_to_photo_homography(part, outline, photo_h, photo_scale,
                                       ppmm, margin_mm)
    return cv2.warpPerspective(full_rgb, H, size, flags=cv2.INTER_AREA)


def roi_metrics(roi_rgb: np.ndarray) -> RoiMetrics:
    """Scalar presence metrics for one body-aligned ROI."""
    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    glare = hsv[:, :, 2] >= GLARE_V

    # FFT high-frequency energy ratio (Hann-windowed, DC removed)
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2((gray - gray.mean()) * win))
    power = np.abs(f) ** 2
    yy, xx = np.mgrid[0:h, 0:w]
    # radius normalized so 1.0 = Nyquist on each axis
    r = np.hypot((yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2)) / math.sqrt(2)
    total = float(power.sum()) or 1.0
    hf_ratio = float(power[r > HF_CUTOFF].sum() / total)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_density = float(np.hypot(gx, gy).mean() / 4.0)  # /4: Sobel gain

    return RoiMetrics(
        hf_ratio=hf_ratio,
        edge_density=edge_density,
        sat_mean=float(hsv[:, :, 1].mean() / 255.0),
        val_std=float(gray.std()),
        glare_frac=float(glare.mean()),
        px=(roi_rgb.shape[1], roi_rgb.shape[0]),
    )


# ---- mounted / not-mounted from the registered overlay photo ----------------
#
# An unpopulated position shows the board itself (green solder mask, some
# bare pad metal); a mounted part covers it with a black/grey/metal body.
# Classify each pixel inside the body rectangle as "solder mask" by hue
# distance to the board's own mask color (estimated from the photo, so
# white balance and lighting drop out), then threshold the mask fraction.


@dataclass
class PresenceResult:
    part: Part
    bare_frac: float                     # solder-mask pixel fraction in body
    mean_rgb: tuple[float, float, float]
    present: bool


def board_mask_hue(warped_rgb: np.ndarray,
                   board_mask: np.ndarray) -> tuple[float, float]:
    """Median (hue, sat) of the solder mask over the board silhouette.

    The mask dominates the board area, so the median over saturated,
    non-dark pixels lands on the mask color even with parts mounted.
    """
    hsv = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2HSV)
    sel = ((board_mask > 0) & (hsv[:, :, 1] >= BARE_SAT_MIN)
           & (hsv[:, :, 2] >= BARE_VAL_MIN))
    if not sel.any():
        return 60.0, 128.0  # generic green; better than crashing
    return (float(np.median(hsv[:, :, 0][sel])),
            float(np.median(hsv[:, :, 1][sel])))


def extract_frame_patch(warped_rgb: np.ndarray, outline: Outline, part: Part,
                        margin_mm: float = 0.0) -> np.ndarray:
    """Body-aligned patch of a part cut from the registered overlay photo.

    The photo is already warped into the outline raster (outline.scale
    px/mm, top-view frame for both sides), so a plain affine crop suffices.
    """
    a = math.radians(part.rot_deg)
    c, s = math.cos(a), math.sin(a)
    hl = part.length_mm / 2 + margin_mm
    hw = part.width_mm / 2 + margin_mm

    cx0, cy0 = part.center()

    def raster(u: float, v: float) -> tuple[float, float]:
        x = cx0 + u * c - v * s
        y = cy0 + u * s + v * c
        return ((x - outline.xmin) * outline.scale,
                (outline.ymax - y) * outline.scale)

    w_px = max(2, int(round(2 * hl * outline.scale)))
    h_px = max(2, int(round(2 * hw * outline.scale)))
    src = np.float32([raster(-hl, hw), raster(hl, hw), raster(hl, -hw)])
    dst = np.float32([(0, 0), (w_px, 0), (w_px, h_px)])
    A = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(warped_rgb, A, (w_px, h_px), flags=cv2.INTER_AREA)


def check_presence(warped_rgb: np.ndarray, outline: Outline,
                   parts: list[Part], side: str,
                   bare_thresh: float = BARE_FRAC_ABSENT,
                   ) -> dict[str, PresenceResult]:
    """Mounted-check every part of `side` against the registered photo."""
    hue0, _sat0 = board_mask_hue(warped_rgb, outline.mask)
    results: dict[str, PresenceResult] = {}
    for part in parts:
        if part.side != side:
            continue
        patch = extract_frame_patch(warped_rgb, outline, part)
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV).astype(np.float32)
        dh = np.abs(hsv[:, :, 0] - hue0)
        dh = np.minimum(dh, 180.0 - dh)  # OpenCV hue is circular over 180
        bare = ((dh <= BARE_HUE_TOL) & (hsv[:, :, 1] >= BARE_SAT_MIN)
                & (hsv[:, :, 2] >= BARE_VAL_MIN))
        frac = float(bare.mean())
        results[part.refdes] = PresenceResult(
            part=part,
            bare_frac=frac,
            mean_rgb=tuple(float(m) for m in patch.reshape(-1, 3).mean(axis=0)),
            present=frac < bare_thresh,
        )
    return results
