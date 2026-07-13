"""BOM parsing: refdes -> part number + DNP flag.

The BOM is an Excel sheet with columns (name, description, refdes list,
count, package). Section headers have an empty refdes column. The row named
"Не устанавливается" holds the comma-separated list of DNP positions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

DNP_MARKER = "не устанавливается"


@dataclass
class BomItem:
    refdes: str
    part_number: str
    package: str
    dnp: bool


def load_bom(path: str) -> dict[str, BomItem]:
    df = pd.read_excel(path, header=None, dtype=str)
    items: dict[str, BomItem] = {}
    for _, row in df.iterrows():
        name = (row[0] or "").strip() if pd.notna(row[0]) else ""
        refs = row[2] if pd.notna(row[2]) else ""
        if not name or not refs:
            continue
        dnp = name.lower().startswith(DNP_MARKER)
        package = (row[4] or "").strip() if pd.notna(row[4]) else ""
        for ref in re.split(r"[,\s]+", refs.strip()):
            if ref:
                items[ref] = BomItem(ref, "" if dnp else name, package, dnp)
    if not items:
        raise ValueError(f"No BOM rows parsed from {path}")
    return items
