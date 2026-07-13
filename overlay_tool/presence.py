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

    def raster(u, v):
        x = part.x_mm + u * c - v * s
        y = part.y_mm + u * s + v * c
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
