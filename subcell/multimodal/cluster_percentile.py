"""How well the image and sequence clusters hold together in each embedding space (Extended Data Fig. 7a).

Quantifies Fig. 6a,b; ``figures.ed7a_cluster_percentile`` plots it. In each target
space (image, sequence, MuSIC co-embedding) every protein pair is ranked by the
cosine similarity of the mean-centred embeddings, as a percentile of all pairs
(100 = most similar). A cluster's score in that space is the mean percentile of
its within-cluster pairs: 50 is no better than chance, 100 is a cluster of mutual
nearest neighbours.

Scored at every level with more than one cluster, for the clusters of
``subcell.multimodal.cluster_transfer`` (``<source>_leiden_<res>``, source image or
sequence) in all three spaces, and for a random baseline per source: its labels
permuted across proteins, which keeps the cluster sizes.

    python -m subcell.multimodal.cluster_percentile [--method-name NAME] [--run-dir DIR] [--out-dir DIR]

Inputs   ``config.CLUSTER_TRANSFER_DIR``/<method name>/clusters.csv and the run's
         embeddings (see ``subcell.multimodal.cluster_transfer``)
Outputs  beside clusters.csv: ``cluster_percentile_detailed.csv`` (one row per
         cluster, source, target and level) and ``cluster_percentile_summary.csv``
         (mean, s.d. and median over the clusters of each).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from subcell import config
from subcell.multimodal.cluster_transfer import (CLUSTERED, METHOD_NAME, input_paths,
                                                 read_embedding)
from subcell.runlog import log
from subcell.utils.hierarchical_leiden import RESOLUTIONS

TARGETS = ("image", "sequence", "multimodal")
# random baseline seed per source; the level adds int(res * 1000)
RANDOM_SEEDS = {"image": 42, "sequence": 43}


def percentile_matrix(X) -> np.ndarray:
    """Percentile rank of every pair's cosine similarity among all pairs, as a symmetric matrix.

    Rows of *X* are mean-centred first. Ties are broken by position (ordinal ranks).
    """
    X = X - X.mean(axis=0, keepdims=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = len(X)
    iu = np.triu_indices(n, k=1)
    sims = (X @ X.T)[iu]
    ranks = np.empty(len(sims), dtype=np.int64)
    ranks[np.argsort(sims)] = np.arange(1, len(sims) + 1)
    pct = np.zeros((n, n), dtype=np.float32)
    pct[iu] = ranks * (100.0 / len(sims))
    return pct + pct.T


def cluster_scores(pct, labels) -> list[tuple[str, int, int, float]]:
    """(cluster, members, within-cluster pairs, mean pair percentile) per cluster of 2 or more."""
    out = []
    for cid in np.unique(labels):
        idx = np.flatnonzero(labels == cid)
        if len(idx) < 2:
            continue
        pairs = pct[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)]
        out.append((str(cid), len(idx), len(pairs), float(pairs.mean())))
    return out


def random_labels(labels, source, res):
    rng = np.random.RandomState(RANDOM_SEEDS[source] + int(res * 1000))
    return labels[rng.permutation(len(labels))]


def main(method_name=METHOD_NAME, run_dir=None, out_dir=None):
    out_dir = config.ensure(out_dir or config.CLUSTER_TRANSFER_DIR / method_name)
    clusters = pd.read_csv(out_dir / "clusters.csv", index_col=0, dtype=str)
    paths = input_paths(method_name, run_dir)
    levels = [r for r in RESOLUTIONS
              if all(clusters[f"{s}_leiden_{r}"].nunique() > 1 for s in CLUSTERED)]

    rows = []
    for target in TARGETS:
        X = read_embedding(paths[target]).loc[clusters.index].to_numpy(np.float64)
        log(f"percentile ranks in {target} ({X.shape[0]:,} proteins x {X.shape[1]:,} dims)")
        pct = percentile_matrix(X)
        for res in levels:
            for source in CLUSTERED:
                labels = clusters[f"{source}_leiden_{res}"].to_numpy()
                for name, lab in [(source, labels),
                                  (f"random_{source}_sized", random_labels(labels, source, res))]:
                    for cid, size, n_pairs, mean in cluster_scores(pct, lab):
                        rows.append(dict(cluster_source=name, target_space=target,
                                         resolution=res, cluster_id=cid, n_proteins=size,
                                         n_pairs=n_pairs, mean_percentile=mean))
        del pct

    detailed = pd.DataFrame(rows)
    summary = (detailed.groupby(["cluster_source", "target_space", "resolution"])["mean_percentile"]
               .agg(n_clusters="size", mean="mean", std="std", median="median").reset_index())
    detailed.to_csv(out_dir / "cluster_percentile_detailed.csv", index=False)
    summary.to_csv(out_dir / "cluster_percentile_summary.csv", index=False)
    log(f"wrote {out_dir / 'cluster_percentile_summary.csv'}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method-name", default=METHOD_NAME)
    ap.add_argument("--run-dir", default=None,
                    help="read the run's embeddings from this folder instead")
    ap.add_argument("--out-dir", default=None,
                    help="folder holding clusters.csv, also written to")
    a = ap.parse_args()
    main(a.method_name, a.run_dir, a.out_dir)
