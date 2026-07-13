"""Board outline: DXF parsing, mm-frame rasterization.

All geometry stays in board-mm; the raster is only a working frame for
registration and display (scale px/mm, y flipped so row 0 = ymax).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import ezdxf
import numpy as np

ARC_STEP_DEG = 4.0  # arc sampling resolution


@dataclass
class Outline:
    segments: list[np.ndarray]  # each (N,2) float64, mm
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    scale: float                # px per mm of the raster
    mask: np.ndarray            # uint8 silhouette, 255 inside board

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """matplotlib imshow extent (left, right, bottom, top) of the raster."""
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    def mm_to_px(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        return np.column_stack([
            (pts[:, 0] - self.xmin) * self.scale,
            (self.ymax - pts[:, 1]) * self.scale,
        ])


def _sample_arc(cx: float, cy: float, r: float,
                a0: float, a1: float) -> np.ndarray:
    """Sample a CCW arc from a0 to a1 (degrees)."""
    if a1 <= a0:
        a1 += 360.0
    n = max(2, int(math.ceil((a1 - a0) / ARC_STEP_DEG)) + 1)
    ang = np.radians(np.linspace(a0, a1, n))
    return np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)])


def load_outline(dxf_path: str, scale: float = 10.0) -> Outline:
    doc = ezdxf.readfile(dxf_path)
    segments: list[np.ndarray] = []
    for e in doc.modelspace():
        t = e.dxftype()
        if t == "LINE":
            s, p = e.dxf.start, e.dxf.end
            segments.append(np.array([[s.x, s.y], [p.x, p.y]]))
        elif t == "ARC":
            c = e.dxf.center
            segments.append(_sample_arc(c.x, c.y, e.dxf.radius,
                                        e.dxf.start_angle, e.dxf.end_angle))
        elif t == "CIRCLE":
            c = e.dxf.center
            segments.append(_sample_arc(c.x, c.y, e.dxf.radius, 0.0, 360.0))
        elif t == "LWPOLYLINE":
            pts = np.array([(p[0], p[1]) for p in e.get_points()])
            if e.closed:
                pts = np.vstack([pts, pts[:1]])
            segments.append(pts)
    if not segments:
        raise ValueError(f"No outline geometry found in {dxf_path}")

    allpts = np.vstack(segments)
    xmin, ymin = allpts.min(axis=0)
    xmax, ymax = allpts.max(axis=0)

    w = int(round((xmax - xmin) * scale)) + 1
    h = int(round((ymax - ymin) * scale)) + 1

    # Draw segments 1px, flood-fill background from the padded border,
    # silhouette = everything not reached. Robust to unordered segments.
    pad = 4
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
    for seg in segments:
        px = np.column_stack([
            (seg[:, 0] - xmin) * scale + pad,
            (ymax - seg[:, 1]) * scale + pad,
        ])
        cv2.polylines(canvas, [np.round(px).astype(np.int32)], False, 255, 1)

    flood = canvas.copy()
    ffmask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 255)
    mask = np.where(flood == 0, 255, canvas)[pad:-pad, pad:-pad]

    return Outline(segments, float(xmin), float(ymin), float(xmax), float(ymax),
                   scale, mask)
