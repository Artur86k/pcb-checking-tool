"""Altium pick-and-place parsing and footprint -> body size estimation.

Sizes come from the footprint name via a rule chain:
  1. explicit "WxH" dimensions in the name (QFN-16_3X3MM, DFN-16_4x1.6, ...)
  2. 4-digit chip codes (0402C, 0805R, L_0603CS_red, 2016L, ...)
  3. a table of standard packages (SOT23, SC70, SOD-323, ...)
  4. fallback default, marked approximate (drawn dashed in the app)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

FRAME_MARGIN_MM = 0.1  # drawn frames extend this far beyond the body


@dataclass
class Part:
    refdes: str
    comment: str
    side: str        # "top" | "bottom"
    footprint: str
    x_mm: float
    y_mm: float
    rot_deg: float
    length_mm: float          # body size along X at rot=0
    width_mm: float           # body size along Y at rot=0
    size_exact: bool          # False -> size is a guess

    def corners(self) -> np.ndarray:
        """Body rectangle corners (4,2) in board-mm, top-view frame.

        This Altium export gives bottom-side rotation CCW in the top-view
        frame, same as top parts — no mirror negation. Verified on L31, the
        only non-orthogonal part: body measured at +14 deg CCW (top view)
        across 7 photos vs rot=200 -> +20 nominal; -rot would predict -20.
        """
        a = math.radians(self.rot_deg)
        c, s = math.cos(a), math.sin(a)
        hl = self.length_mm / 2 + FRAME_MARGIN_MM
        hw = self.width_mm / 2 + FRAME_MARGIN_MM
        return np.array([
            (self.x_mm + dx * c - dy * s, self.y_mm + dx * s + dy * c)
            for dx, dy in ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw))
        ])


# Imperial chip codes -> (length, width) mm
CHIP_IMPERIAL = {
    "0201": (0.6, 0.3), "0402": (1.0, 0.5), "0603": (1.6, 0.8),
    "0805": (2.0, 1.25), "1206": (3.2, 1.6), "1008": (2.5, 2.0),
    "1111": (2.8, 2.8), "1210": (3.2, 2.5), "1212": (3.2, 3.2),
    "1806": (4.5, 1.6), "2010": (5.0, 2.5), "2512": (6.3, 3.2),
    "0908": (2.3, 2.0), "0804": (2.0, 1.0),
}

# Known packages (body only, no leads) -> (length, width) mm, exact-ish
PACKAGES = {
    "SOT23": (2.9, 1.6), "SOT23-5": (2.9, 1.6), "SOT23-6": (2.9, 1.6),
    # SC70-5 body is along Y at 0 deg in this library (photo-verified)
    "SC70": (2.0, 1.25), "SC70-5": (1.25, 2.0), "SC70-6": (2.0, 1.25),
    "SOT343": (1.25, 2.0), "SOT416": (1.6, 0.8), "SOT523": (1.6, 0.8),
    "SOT563": (1.6, 1.2), "SOT883": (1.0, 0.6),
    "SOD-323": (1.7, 1.3), "SOD-323HE": (1.3, 1.7),  # HE body along Y at 0 deg
    "MSOP-8": (3.0, 3.0), "MSOP-8_PAD": (3.0, 3.0),
    "QSOP-16": (4.9, 3.9), "QFN-48": (7.0, 7.0),
    "TP0.5": (0.5, 0.5),
    "LED_0603": (0.8, 1.6), "LED_0606": (1.6, 1.6),  # LED body along Y at 0 deg
    # Tantalum footprints in this library have the body along Y at 0 deg
    "CE_TAN_A": (1.6, 3.2), "CE_TAN_B": (2.8, 3.5), "CE_TAN_D": (4.3, 7.3),
    "R4MATR_0804(4x0402)": (2.0, 1.0),
    "R8MATR": (3.8, 1.6),
    "ABM10AIG": (2.5, 2.0), "ABS05": (1.6, 1.0),
    "XFL4020": (4.0, 4.0), "IHLP2020BZ": (5.2, 5.5), "IHLP2525BD": (6.5, 6.9),
    "POWERPAK_1212-8": (3.3, 3.3), "U.FL-R-SMT": (2.6, 2.6),
    "H.FL": (3.0, 3.0),
    "L_MIDI": (5.5, 5.5),
    # Large ICs sized from the part datasheets (body, not courtyard)
    "BGA-484": (23.0, 23.0),      # XC7A200T-FBG484
    "BGA-184": (12.0, 12.0),      # ADSP-BF707BBCZ
    "TFBGA-216": (13.0, 13.0),    # STM32F469NIH6
    "UFBGA-169": (7.0, 7.0),      # STM32F469AEH6
    "LFCSP-72": (10.0, 10.0),     # AD9783BCPZ
    "LCC_18": (9.7, 10.1),        # u-blox MAX-M8Q module
    "SON-22__DQP-22": (5.0, 6.0), # TPS53318DQP
    # Small regulators / switches
    "SON-6": (2.0, 2.0),          # TPS62590/61160/60150 DRV
    "SON-9": (2.5, 2.5),          # TPS61236RWL (RWL0009 body 2.5x2.5)
    "SON-14": (3.0, 4.0),         # TPS63020DSJ
    "DQN_X2SON": (1.0, 1.0),      # LP5907SNX
    "MIC94064YMT": (2.0, 1.2),    # 6-pin Thin MLF
    "XQFN16": (1.8, 2.6),         # 74AVCH4T245GU, body along Y at 0 deg
    "12PIN_UQFN_1": (2.0, 1.7),   # TS3USBA225RUT
    "SOT416(SC-75)": (1.6, 0.8),
    "ESD0P4RFL": (1.0, 0.6),      # measured from board photo (SOD-882 size)
    "SM-22B": (4.1, 3.8),         # ETC4-1-2 / MABA transformers
    "EXC34CG": (1.25, 2.0),       # Panasonic 0805-size CM filter, body along Y
    # Connectors / misc, sized from the registered board photos
    "DF40C-80DS-0.4V(51)": (17.4, 3.7),        # Hirose mezzanine
    "FH19C-12S-0.5SH(05)": (4.5, 5.5),         # FPC, body along Y at rot0
    "FH33-45S-0.5SH_reversed": (25.0, 3.5),    # display FPC
    "813-22-010-30-000101": (10.2, 4.1),       # Mill-Max 2x5 spring pins
    "TL4105": (2.5, 2.5),                      # side tact switch body
    "SPM0687LR5H-1": (3.76, 4.72),             # Knowles mic, body along Y at 0 deg
    "CASE_466-03": (5.5, 6.2),                 # MRF1518 / MW6S004 PLD-1.5
    "NFBGA-60": (9.0, 6.0),                    # LMX9830 (module-size body)
    # DSBGA-6 bodies are along Y at 0 deg in this library (photo-verified);
    # exact entries here override the WxH parsed from the name.
    "BGA-6_1.2x0.8mm": (0.8, 1.2),             # SN74AUP2G17YFP / LMH2110TM
    "BGA-6_1.4x0.9mm": (0.9, 1.4),             # SN74AUC1G08YZP
    "BGA-8_1.6x0.9mm": (0.9, 1.6),
}

# Prefix-matched exact sizes (checked before chip-code rules; the Coilcraft
# wirewound bodies are fatter than the plain chip code suggests).
PREFIX_PACKAGES = {
    # Coilcraft CS wirewounds: body along Y at 0 deg in this library
    # (photo-verified on 0603CS, 0805CS and all seven 0402CS)
    "L_0603CS": (1.15, 1.8),
    "L_0805CS": (1.5, 2.3),
    "L_0402CS": (0.7, 1.12),
    "L_1111SQ": (2.8, 2.8),   # Coilcraft 1111SQ air core
    "L_0908SQ": (2.3, 2.0),   # Coilcraft 0908SQ air core
}

_DIMS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)(?:\s*[xX]\s*\d+(?:\.\d+)?)?\s*(?:mm)?",
)
_CHIP_RE = re.compile(r"^(?:L_)?(\d{4})(C|R|L|CS|C_HIGHCAP|CS_[a-z]+|L_.*)?$",
                      re.IGNORECASE)

DEFAULT_SIZE = (2.0, 2.0)


def footprint_size(fp: str) -> tuple[float, float, bool]:
    """Return (length_mm, width_mm, exact) for a footprint name."""
    if fp in PACKAGES:
        l, w = PACKAGES[fp]
        return l, w, True

    for key, (l, w) in PREFIX_PACKAGES.items():
        if fp.upper().startswith(key.upper()):
            return l, w, True

    # Explicit dimensions anywhere in the name, e.g. QFN-16_3X3MM,
    # DFN-16_4x1.6, BGA-6_1.2x0.8mm, 1806L_(4.5X1.6), LGA24_3.5x3x1
    m = _DIMS_RE.search(fp)
    if m:
        l, w = float(m.group(1)), float(m.group(2))
        if 0.2 <= l <= 60 and 0.2 <= w <= 60:
            return l, w, True

    # Chip-style 4-digit codes. Imperial codes come from the table; other
    # codes (e.g. 2016L, 2520) are metric: 2016 -> 2.0 x 1.6 mm.
    m = _CHIP_RE.match(fp)
    if m:
        code, suffix = m.group(1), (m.group(2) or "").upper()
        # Chip C/R/L footprints in this library all have the body along Y
        # at 0 deg (photo-confirmed: 0201L/0402L/0603L/0805L/2016L too).
        # Names with explicit dims like 1806L_(4.5X1.6) never reach this
        # branch and stay along X.
        swap = suffix in ("C", "R", "C_HIGHCAP") or suffix.startswith("L")
        if code in CHIP_IMPERIAL:
            l, w = CHIP_IMPERIAL[code]
        else:
            l, w = int(code[:2]) / 10.0, int(code[2:]) / 10.0
        if 0.2 <= w <= l <= 60:
            return (w, l, True) if swap else (l, w, True)

    # Fuzzy: known package as a prefix (SOT23-5X handled above; catches
    # variants like "SON-6", "SOT523" already exact-matched).
    for key, (l, w) in PACKAGES.items():
        if fp.upper().startswith(key.upper()):
            return l, w, False

    return DEFAULT_SIZE[0], DEFAULT_SIZE[1], False


_ROW_RE = re.compile(
    r'^(\S+)\s+(".*?"|\S+)\s+(TopLayer|BottomLayer)\s+(\S+)\s+'
    r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def load_pnp(path: str) -> list[Part]:
    """Parse an Altium 'Pick Place for ...' text export."""
    with open(path, encoding="cp1251", errors="replace") as f:
        raw_lines = f.read().splitlines()

    # Comments may contain embedded newlines; re-join lines until the quote
    # count is even so every logical row is one string.
    lines: list[str] = []
    buf = ""
    for ln in raw_lines:
        buf = (buf + " " + ln) if buf else ln
        if buf.count('"') % 2 == 0:
            lines.append(buf)
            buf = ""
    if buf:
        lines.append(buf)

    parts: list[Part] = []
    for ln in lines:
        m = _ROW_RE.match(ln)
        if not m or m.group(1) == "Designator":
            continue
        refdes, comment, layer, fp = m.group(1), m.group(2).strip('"'), m.group(3), m.group(4)
        x, y, rot = float(m.group(5)), float(m.group(6)), float(m.group(7))
        l, w, exact = footprint_size(fp)
        parts.append(Part(refdes, comment,
                          "top" if layer == "TopLayer" else "bottom",
                          fp, x, y, rot, l, w, exact))
    if not parts:
        raise ValueError(f"No components parsed from {path}")
    return parts
