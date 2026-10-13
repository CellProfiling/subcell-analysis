"""Hierarchical Leiden clustering: each level subclusters the clusters of the level before.

The clustering of the multiscale cell map (Fig. 5,
``notebooks/fig5_multiscale_cell_map.ipynb``), also used by
``subcell.multimodal.cluster_transfer`` (Fig. 6). The first level clusters over many
neighbours, for global structure; every later level re-clusters each cluster of the
previous level separately, at a higher resolution over fewer neighbours, for
increasingly local structure. A cluster with fewer members than the level's
neighbour count is carried over unsplit.

Level ``res`` is written to ``adata.obs["leiden_<res>"]``, relabelled to the
consecutive integers ``"0"``, ``"1"``, ... in order of first appearance.
"""

from __future__ import annotations

import pandas as pd
import scanpy as sc

RESOLUTIONS = (0, 0.15, 0.2, 0.25, 0.30, 0.33, 0.36)
NEIGHBORS = (125, 100, 90, 55, 40, 25, 10)


def _leiden(adata, n_neighbors, res, random_state):
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=None, metric="euclidean",
                    random_state=random_state)
    sc.tl.leiden(adata, flavor="igraph", resolution=res, n_iterations=2)
    return adata.obs["leiden"].astype(str)


def subcluster(adata, parent_col, res, n_neighbors, parent_label):
    """Labels ``"<parent>.<sub>"`` for the members of one parent cluster, indexed like ``adata.obs``."""
    sub = adata[adata.obs[parent_col] == parent_label].copy()
    if sub.n_obs < n_neighbors:
        return pd.Series(f"{parent_label}.0", index=sub.obs_names)
    return f"{parent_label}." + _leiden(sub, n_neighbors, res, random_state=111)


def hierarchical_leiden(adata, resolutions=RESOLUTIONS, neighbors=NEIGHBORS):
    """Add one ``leiden_<res>`` column per level to ``adata.obs``, in place.

    The first level also leaves its neighbour graph and ``obs["leiden"]`` on ``adata``.
    """
    adata.obs[f"leiden_{resolutions[0]}"] = _leiden(adata, neighbors[0], resolutions[0],
                                                    random_state=42).values
    for parent_res, res, n_neighbors in zip(resolutions, resolutions[1:], neighbors[1:]):
        parent_col = f"leiden_{parent_res}"
        labels = pd.concat([
            subcluster(adata, parent_col, res, n_neighbors, parent)
            for parent in adata.obs[parent_col].unique()
        ]).reindex(adata.obs_names)
        relabel = {old: str(new) for new, old in enumerate(labels.unique())}
        adata.obs[f"leiden_{res}"] = pd.Categorical(
            labels.map(relabel), categories=[str(i) for i in range(len(relabel))])
    return adata
