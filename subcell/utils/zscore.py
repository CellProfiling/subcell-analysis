"""Normalisation of per-cell measurements: log-transform when heavy-tailed, z-score within groups.

Used for the 22 morphology and intensity measurements the tissue clustering never
sees (``subcell.tissue.clustering.external_panel``, ``figures.ed2jk_cluster_annotation``,
``subcell.tissue.prevalence_support``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HEAVY_TAIL_MAX = 100   # columns whose maximum exceeds this (areas, raw 16-bit intensities)
CLIP = 8.0


def log_heavy_tailed(X) -> np.ndarray:
    """float64 copy of ``X`` with non-finite values set to 0 and heavy-tailed columns log(1 + x)-transformed."""
    E = np.nan_to_num(np.asarray(X, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    heavy = np.nanmax(E, 0) > HEAVY_TAIL_MAX
    E[:, heavy] = np.log1p(np.clip(E[:, heavy], 0, None))
    return E


def zscore_within(E: np.ndarray, groups, dtype=np.float64) -> np.ndarray:
    """Z-score each column of ``E`` within every level of ``groups``, clipped to +-8.

    ``groups`` has one entry per row (a column of a metadata table, say); use a
    constant for a z-score over the whole matrix. ``dtype`` is the type of the result.
    """
    out = np.empty(E.shape, dtype)
    for _, idx in pd.Series(np.arange(len(E))).groupby(np.asarray(groups)).indices.items():
        b = E[idx]
        out[idx] = ((b - b.mean(0)) / (b.std(0) + 1e-9)).astype(dtype)
    return np.clip(out, -CLIP, CLIP)
