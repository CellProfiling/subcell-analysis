"""Ward tree over a cluster-by-unit prevalence matrix, with bootstrap support.

Used by ``subcell.tissue.prevalence_support`` and ``figures.ed2jk_cluster_annotation``
(Extended Data Fig. 2k, Supplementary Note 2). Rows are clusters, columns are
prevalence units (donors or cores). Rows are z-scored so that grouping follows
profile shape rather than cluster size, then clustered with Ward linkage. Support
for a branch is the percentage of bootstrap replicates, resampling the columns with
replacement, in which the same leaf set appears as a clade of the recomputed tree.
"""

from __future__ import annotations

import numpy as np
import scipy.cluster.hierarchy as sch


def zscore_rows(M: np.ndarray) -> np.ndarray:
    return (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)


def clades(linkage: np.ndarray, labels) -> list[frozenset]:
    """Every proper leaf set (two or more leaves, not the root) of a linkage."""
    n = len(labels)
    members = {i: frozenset([labels[i]]) for i in range(n)}
    out = []
    for i, (a, b, *_) in enumerate(linkage):
        s = members[int(a)] | members[int(b)]
        members[n + i] = s
        if 1 < len(s) < n:
            out.append(s)
    return out


def ward_tree(X: np.ndarray, labels):
    """Ward linkage of the row-z-scored matrix ``X`` (rows named by ``labels``)."""
    return sch.linkage(zscore_rows(X), method="ward")


def _replicate_clades(X, labels, cols):
    return set(clades(ward_tree(X[:, cols], labels), labels))


def bootstrap_support(X, labels, linkage, n_boot: int, rng) -> dict:
    """Percentage of column-bootstrap replicates containing each clade of ``linkage``."""
    hits = {c: 0 for c in clades(linkage, labels)}
    for _ in range(n_boot):
        found = _replicate_clades(X, labels, rng.integers(0, X.shape[1], X.shape[1]))
        for c in hits:
            hits[c] += c in found
    return {c: 100 * v / n_boot for c, v in hits.items()}


def jackknife_support(X, labels, linkage) -> dict:
    """Percentage of leave-one-column-out trees containing each clade of ``linkage``."""
    hits = {c: 0 for c in clades(linkage, labels)}
    n = X.shape[1]
    for drop in range(n):
        found = _replicate_clades(X, labels, [i for i in range(n) if i != drop])
        for c in hits:
            hits[c] += c in found
    return {c: 100 * v / n for c, v in hits.items()}


def ward_bootstrap(P, n_boot: int, rng):
    """Tree of the prevalence table ``P`` (a DataFrame) with bootstrap support.

    Returns ``(leaf_order, linkage, labels, support)``; ``support`` maps each clade
    (frozenset of row labels) to its bootstrap percentage.
    """
    labels = list(P.index)
    X = P.to_numpy(float)
    linkage = ward_tree(X, labels)
    order = [labels[i] for i in sch.dendrogram(linkage, no_plot=True)["leaves"]]
    return order, linkage, labels, bootstrap_support(X, labels, linkage, n_boot, rng)
