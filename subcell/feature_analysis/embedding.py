"""Fitting and caching the 2-D UMAP embeddings of HPA FOV features.

Used by ``figures.ed3_harmonization`` (Extended Data Fig. 3a-d). UMAP is for
display only. Two consequences are enforced here rather than left to the caller:

  * Coordinates are cached under a key that includes the parameters *and* the
    number of rows. Coordinates are valid only for the exact row set they were
    fitted on, and a cache keyed on parameters alone silently pairs new labels
    with old coordinates -- which produces a plausible figure that is wrong.
  * Every panel of a comparison uses one parameter set, so panels differ only in
    the features they were fitted to.

Proximity in the result is not evidence of similarity. Coordinates are cached in
``config.HPA_UMAP_DIR / "umap_coords"``.
"""

from __future__ import annotations

import os
import pathlib
import time

import numpy as np
import torch
from umap import UMAP

from .. import config
from ..runlog import log

# One parameter set for every HPA panel. n_neighbors is high because the question
# is global organisation by plate or cell line, not local detail; the cosine
# metric matches the retrieval metric so the picture and the numbers describe the
# same geometry.
UMAP_PARAMS = dict(n_neighbors=50, min_dist=0.2, metric="cosine", random_state=42)


def fit_umap(feats, method, harmonized, params=None):
    """2-D coordinates, cached per (embedding, parameters, FOV count).

    The FOV count is part of the filename on purpose: coordinates are only
    valid for the exact row set they were fitted on, and a cache keyed by
    parameters alone would silently pair new labels with old coordinates.
    """
    p = dict(params or UMAP_PARAMS)
    tag = f"nn{p['n_neighbors']}_md{p['min_dist']}_n{len(feats)}"
    out = f"{config.HPA_UMAP_DIR}/umap_coords/{method}_{harmonized}__{tag}.npy"
    if os.path.exists(out):
        log(f"  cached UMAP: {out}")
        return np.load(out)

    # Cosine metric on L2-normalised rows: embedding magnitude varies by an
    # order of magnitude between models, and an un-normalised Euclidean UMAP
    # would show that rather than the neighbourhood structure being compared.
    x = torch.nn.functional.normalize(torch.from_numpy(feats).float(), dim=1).numpy()
    log(f"  fitting UMAP on {x.shape[0]:,} x {x.shape[1]}")
    t0 = time.perf_counter()
    coords = UMAP(n_components=2, verbose=False, **p).fit_transform(x)
    config.ensure(pathlib.Path(out).parent)
    np.save(out, coords.astype(np.float32))
    log(f"  UMAP done in {time.perf_counter() - t0:.0f}s")
    return coords
