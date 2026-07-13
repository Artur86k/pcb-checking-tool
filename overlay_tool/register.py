"""Photo -> board-outline registration.

Pipeline: segment the board against the (dark) background, coarse-align the
board quad to the outline quad trying all 4 rotations x mirror (mirror covers
bottom-side photos), then refine with an ECC homography on the silhouettes.
The outline never moves — only the photo is warped.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .outline import Outline

MAX_PHOTO_DIM = 2200  # working resolution for segmentation/warp


@dataclass
class RegistrationResult:
    warped_rgb: np.ndarray   # photo warped into the outline raster frame
    iou: float               # silhouette IoU after refinement
    homography: np.ndarray   # photo px -> outline raster px


def load_photo(path: str, max_dim: int = MAX_PHOTO_DIM) -> np.ndarray:
    """Load JPG/PNG via OpenCV or DNG via rawpy, downscaled, as RGB."""
    rgb = None
    if path.lower().endswith((".dng", ".raw", ".arw", ".cr2", ".nef")):
        try:
            import rawpy
            try:
                with rawpy.imread(path) as raw:
                    rgb = raw.postprocess(use_camera_wb=True, output_bps=8,
                                          no_auto_bright=False)
            except rawpy.LibRawError:
                # Samsung's compressed DNGs (JPEG-XL raw stream) can't be
                # unpacked by LibRaw — use the embedded full-size JPEG
                # preview instead, which is plenty for overlay work. The
                # failed unpack poisons the handle, so reopen the file.
                with rawpy.imread(path) as raw:
                    thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    bgr = cv2.imdecode(
                        np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_COLOR)
                    if bgr is not None:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                else:
                    rgb = thumb.data
        except Exception:
            rgb = None  # not a readable raw at all; try OpenCV below
    if rgb is None:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = max_dim / max(h, w)
    if s < 1.0:
        rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return rgb


def segment_board(rgb: np.ndarray) -> np.ndarray:
    """Binary mask of the board (largest bright object on dark background)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    sat = hsv[:, :, 1]
    # Board (green solder mask / gold) is both brighter and more saturated
    # than the dark fabric; take the union of the two Otsu splits.
    _, mv = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, ms = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.bitwise_or(mv, ms)

    # Open FIRST: fabric texture passes Otsu as speckle, and closing would
    # merge it into the board. Holes inside the board are handled by the
    # external-contour fill below instead of a close.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No board found in photo (segmentation empty)")
    big = max(contours, key=cv2.contourArea)
    if cv2.contourArea(big) < 0.05 * m.size:
        raise ValueError("Board too small in frame — segmentation failed?")
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [big], -1, 255, cv2.FILLED)

    # Strip thin appendages (attached wires) without eroding the board edge:
    # heavy erosion kills the wire, keep the surviving main blob, dilate it
    # back, and intersect with the original silhouette.
    ksz = max(9, (max(m.shape) // 80) | 1)
    kbig = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    core = cv2.erode(filled, kbig)
    cc, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cc:
        blob = np.zeros_like(m)
        cv2.drawContours(blob, [max(cc, key=cv2.contourArea)], -1, 255, cv2.FILLED)
        blob = cv2.dilate(blob, kbig)
        clipped = cv2.bitwise_and(filled, blob)
        cc2, _ = cv2.findContours(clipped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cc2:
            filled = np.zeros_like(m)
            cv2.drawContours(filled, [max(cc2, key=cv2.contourArea)], -1, 255,
                             cv2.FILLED)
    return filled


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(cv2.bitwise_and(a, b))
    union = np.count_nonzero(cv2.bitwise_or(a, b))
    return inter / union if union else 0.0


def _coarse_align(photo_mask: np.ndarray, outline_mask: np.ndarray,
                  mirror: bool) -> list[tuple[np.ndarray, float]]:
    """Candidate photo->outline homographies from minAreaRect corner matching.

    A bottom-side photo is a mirrored view of the top-view outline frame, so
    the mirror is dictated by which side the photo shows (`mirror=True`) —
    never guessed from fit quality, because a near-symmetric outline makes
    the wrong choice score almost as well. The 4 rotations are returned
    sorted by silhouette IoU, best first; ECC refinement decides among them.
    """
    ph, pw = outline_mask.shape

    def rect_corners(mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        box = cv2.boxPoints(cv2.minAreaRect(max(cnts, key=cv2.contourArea)))
        return box.astype(np.float32)

    src = rect_corners(photo_mask)
    dst = rect_corners(outline_mask)
    order = src[::-1] if mirror else src

    cands: list[tuple[np.ndarray, float]] = []
    for shift in range(4):                  # 4 rotations
        H = cv2.getPerspectiveTransform(np.roll(order, shift, axis=0), dst)
        warped = cv2.warpPerspective(photo_mask, H, (pw, ph))
        cands.append((H, _iou(warped, outline_mask)))
    cands.sort(key=lambda c: c[1], reverse=True)
    return cands


def register_photo(rgb: np.ndarray, outline: Outline,
                   side: str = "top") -> RegistrationResult:
    """Warp `rgb` into the outline's top-view raster frame.

    `side` must say which board side the photo shows: a bottom-side photo is
    mirrored into the top-view frame (flip back at display time for a
    bottom view).
    """
    photo_mask = segment_board(rgb)
    out_mask = outline.mask
    oh, ow = out_mask.shape

    candidates = _coarse_align(photo_mask, out_mask, mirror=(side == "bottom"))

    # ECC refinement on blurred silhouettes, coarse-to-fine blur to avoid
    # local minima. Refine the top coarse candidates — a near-symmetric
    # outline can rank a 180-degree flip first at the coarse stage, but the
    # correct orientation wins after refinement. ECC's warp maps template
    # (outline) -> input (photo), so seed with H^-1 and invert back after.
    blurred = {}
    for sigma in (12.0, 3.0):
        blurred[sigma] = (
            cv2.GaussianBlur(out_mask, (0, 0), sigma).astype(np.float32) / 255.0,
            cv2.GaussianBlur(photo_mask, (0, 0), sigma).astype(np.float32) / 255.0,
        )

    H, best_iou = candidates[0]
    for H_cand, coarse_iou in candidates[:3]:
        warp = np.linalg.inv(H_cand).astype(np.float32)
        warp /= warp[2, 2]
        cand_best_h, cand_best_iou = H_cand, coarse_iou
        for sigma in (12.0, 3.0):
            template, inp = blurred[sigma]
            try:
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
                cv2.findTransformECC(template, inp, warp, cv2.MOTION_HOMOGRAPHY,
                                     criteria)
            except cv2.error:
                continue
            H_ref = np.linalg.inv(warp.astype(np.float64))
            refined = cv2.warpPerspective(photo_mask, H_ref, (ow, oh))
            score = _iou(refined, out_mask)
            if score >= cand_best_iou:
                cand_best_h, cand_best_iou = H_ref, score
        if cand_best_iou > best_iou:
            H, best_iou = cand_best_h, cand_best_iou
        if best_iou > 0.97:  # good enough — skip remaining candidates
            break

    warped_rgb = cv2.warpPerspective(rgb, H, (ow, oh), flags=cv2.INTER_LINEAR)
    final_iou = _iou(cv2.warpPerspective(photo_mask, H, (ow, oh)), out_mask)
    return RegistrationResult(warped_rgb, final_iou, H)
