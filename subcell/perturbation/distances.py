"""Sliced Wasserstein distance between two embedding distributions (Fig. 4c, Extended Data Fig. 5)."""

import numpy as np
from scipy.stats import wasserstein_distance


def compute_sliced_wasserstein(X1, X2, n_projections=1000, seed=42):
    """Sliced Wasserstein distance between two sets of embeddings.

    Both sets are projected onto ``n_projections`` random unit directions (standard
    normal, l2-normalised, drawn from ``numpy.random.default_rng(seed)``); the
    result is the mean one-dimensional Wasserstein distance over the directions.
    0 means identical projected distributions.

    Parameters
    ----------
    X1, X2 : arrays of shape (n_samples, n_features)

    References
    ----------
    Rabin et al., "Wasserstein Barycenter and Its Application to Texture Mixing"
    (SSVM 2011).
    """
    rng = np.random.default_rng(seed)
    d = X1.shape[1]

    distances = []
    for _ in range(n_projections):
        theta = rng.normal(0, 1, d)
        theta = theta / np.linalg.norm(theta)
        distances.append(wasserstein_distance(X1 @ theta, X2 @ theta))
    return float(np.mean(distances))
