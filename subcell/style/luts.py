"""Single-hue lookup tables for rendering fluorescence channels.

A multi-channel micrograph is composited by mapping each channel through a ramp
from black to that channel's hue; the ramps are additive, so co-localization shows
as the sum of two hues rather than as whichever channel was drawn last.

The tables are the tab-separated ``.lut`` files under ``subcell/style/colormaps``
(the format ImageJ/Fiji reads, so the same channel colours can be applied to the
images outside Python): blue, cyan, green, light_blue, orange, pink, red, white and
yellow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LUT_DIR = Path(__file__).parent / "colormaps"


def load_lut(name: str) -> np.ndarray:
    """The table ``colormaps/<name>.lut`` as a 256 x 3 array of 0-255 values."""
    path = LUT_DIR / f"{name}.lut"
    if not path.exists():
        names = sorted(p.stem for p in LUT_DIR.glob("*.lut"))
        raise KeyError(f"no LUT named {name!r}; have {names}")
    return pd.read_csv(path, sep="\t", index_col=0).to_numpy()
