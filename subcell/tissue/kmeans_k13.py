"""k-means with k = 13 on the Harmony-corrected tissue features (Extended Data Fig. 2i-k).

The reported partition: k-means (k-means++ initialisation, squared Euclidean
distance, all 210,482 cells, no PCA) on the features corrected for the batch key
``variant`` -- ``core`` for the paper -- followed by the transfer of the original
cluster ids (``reference_labels.transfer_labels``) so that the clusters keep the
names the annotation tables use. Core correction removes the staining, exposure and
illumination offset that is constant within a core; the corrected values are not
comparable between cores, so clusters are interpreted through the withheld
measurements and per-donor prevalence (``figures.ed2jk_cluster_annotation``).

Any other key of ``harmonization.KEYS`` (``patient``, ``slide``, ...) can be run
for comparison; it writes to its own ``clustering_<key>_k13`` directory.

    python -m subcell.tissue.kmeans_k13 [core] [--matrix-only]

Inputs   ``config.TISSUE_META``, ``config.TISSUE_HARM``/harmonized_<variant>.npy,
         ``config.TISSUE_INTENSITY``, the original labelling
         (``reference_labels.ORIGINAL_LABELS``)
Outputs  ``config.TISSUE_CLUSTERS`` for ``core``: cell_clusters.csv, label_transfer.csv,
         all_labels.npz, transfer_*.csv, cluster profile and composition tables
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from subcell import config
from subcell.runlog import log
from subcell.tissue import clustering as ct
from subcell.tissue import reference_labels

K = 13
SEED = 0


def output_dir(variant):
    """Directory of the partition for one batch key (``config.TISSUE_CLUSTERS`` for core)."""
    tag = variant.replace("+", "_")
    # "v1labels": the clusters carry the ids of the original labelling (on-disk name)
    return config.TISSUE_DIR / f"clustering_{tag}_k{K}" / f"kmeans_k{K}_{tag}_v1labels"


def main(variant="core", matrix_only=False):
    outdir = config.ensure(output_dir(variant))
    if matrix_only:
        reference_labels.transfer_matrix(outdir)
        return
    feats = config.TISSUE_HARM / f"harmonized_{variant}.npy"
    assert feats.exists(), f"no such harmonized matrix: {feats}"

    cells = ct.load_cells(pd.read_csv(config.TISSUE_META))
    meta = cells.meta
    log(f"{len(meta):,} cells · {meta.core.nunique()} cores · "
        f"{meta.patient.nunique()} donors · {meta.slide.nunique()} acquisition blocks")

    X = np.load(feats).astype(np.float32)
    assert len(X) == len(meta), f"{len(X)} rows vs {len(meta)} cells"
    Xg = torch.as_tensor(X, device=ct.DEV)
    del X

    raw, _ = ct.kmeans_gpu(Xg, K, SEED)
    raw = ct.relabel(raw)
    log(f"k-means k={K}: sizes {np.bincount(raw).tolist()}")

    lab, match, _ = reference_labels.transfer_labels(meta, raw)
    meta["cluster"], meta["cluster_raw"] = lab, raw
    match.to_csv(outdir / "label_transfer.csv", index=False)

    # Score on `raw`, not `lab`: the transfer leaves gaps in the ids, and
    # silhouette / Davies-Bouldin / Calinski-Harabasz divide by empty clusters
    # and return nan. It is the same partition either way.
    sx, sr = ct.spatial_coherence(raw, cells.nbr, cells.core_codes, cells.n_core)
    sil = ct.silhouette_gpu(Xg, raw)
    chv, dbv = ct.ch_db_gpu(Xg, raw)
    log(f"quality of the k={K} partition: silhouette {sil:+.4f}  DB {dbv:.3f}  "
        f"CH {chv:.1f}  extCH {ct.ch_np(cells.ext, raw):.1f}  "
        f"spatial {sx:+.4f} ({sr:.2f}x)  "
        f"coreMax {ct.core_confound(raw, cells.core_codes, cells.n_core)[0]:.3f}")
    del Xg
    torch.cuda.empty_cache()

    meta.to_csv(outdir / "cell_clusters.csv", index=False)
    np.savez_compressed(outdir / "all_labels.npz", **{f"kmeans_k{K}": lab.astype(np.int16)})
    ct.finalize(meta, lab, cells, outdir)
    reference_labels.transfer_matrix(outdir)
    log(f"done -> {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("variant", nargs="?", default="core",
                    help="batch key whose corrected features are clustered (default: core)")
    ap.add_argument("--matrix-only", action="store_true",
                    help="only rewrite the transfer matrices of an existing run")
    a = ap.parse_args()
    main(a.variant, a.matrix_only)
