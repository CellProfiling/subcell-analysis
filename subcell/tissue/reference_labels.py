"""Reference labelling of the tissue cells and the transfer of its cluster ids.

The manual quality assessment (Extended Data Fig. 2j) was done on the clusters of a
k-means labelling of all 210,482 cells, the reference (k = 13) labelling
(``config.TISSUE_REF_LABELS``). Its ids are inherited from the original 126,264-cell
k = 19 labelling (``ORIGINAL_LABELS``, an input of this repository's analysis, not
regenerated here), so the 13 clusters carry the gapped ids 0, 1, 2, 3, 4, 5, 7, 10,
11, 13, 15, 16, 18 that the annotation tables and ``cluster_id_map.csv`` refer to.
On disk the original labelling is called "v1" and the reference-labelling directory
``clustering_v2``. This module

* ``transfer_labels``  renames a new partition to its best-matching original ids
  (Hungarian assignment on the contingency table of the cells both share);
* ``transfer_matrix``  writes the full original-by-new contingency tables;
* ``sweep`` / ``view`` regenerate the reference labelling: ``sweep`` clusters the
  core-corrected features with every method over k = 10-30 (resolutions 0.25-2.0
  for Leiden); ``view kmeans_k13`` relabels the k-means k = 13 partition with the
  original ids and writes ``clustering_v2/kmeans_k13_v1labels``.

The labelling is k-means k = 13 on the Harmony-corrected features. Core correction
and a slide+core correction produce bit-identical matrices (core is
nested in slide), so ``harmonized_core.npy`` is used for both.

    python -m subcell.tissue.reference_labels sweep
    python -m subcell.tissue.reference_labels view kmeans_k13

Inputs   ``config.TISSUE_META``, ``config.TISSUE_HARM``/harmonized_core.npy,
         ``config.TISSUE_INTENSITY``, ``ORIGINAL_LABELS``
Outputs  the directory two levels above ``config.TISSUE_REF_LABELS`` (``clustering_v2``)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from subcell import config
from subcell.runlog import log
from subcell.tissue import clustering

REF_DIR = config.TISSUE_REF_LABELS.parent.parent
ORIGINAL_LABELS = config.TISSUE_ORIGINAL_LABELS
FEATURES = config.TISSUE_HARM / "harmonized_core.npy"
RESOLUTIONS = [0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
               0.90, 1.00, 1.15, 1.30, 1.50, 1.75, 2.00]


def transfer_labels(meta, new_lab):
    """Rename new clusters to their best-matching original identity.

    The cells of the original labelling are a subset of the new cells, so the match
    is made on the shared cells: maximise the number of them that keep their name.
    Hungarian assignment on the contingency table, not greedy matching, which can
    strand a large cluster with a poor partner. New clusters beyond the original 19
    get fresh ids from 19 up and are reported as new.

    Returns the renamed labels, the per-cluster match table and the ARI between
    the two labellings on the shared cells. The table's column names are on-disk
    names: ``n_v1`` and ``v2_raw``/``n_v2*`` refer to the original and the new
    labelling.
    """
    orig = pd.read_csv(ORIGINAL_LABELS, usecols=["id", "cluster"])
    pos = pd.Index(meta["id"]).get_indexer(orig["id"])
    assert (pos >= 0).all(), "cells of the original labelling missing from the new cells"
    old, new = orig["cluster"].to_numpy(), new_lab[pos]
    k1, k2 = int(old.max()) + 1, int(new_lab.max()) + 1

    C = np.zeros((k1, k2), np.int64)
    np.add.at(C, (old, new), 1)
    r, c = linear_sum_assignment(-C)              # rows = original id, cols = new raw id
    remap = np.full(k2, -1, np.int64)
    remap[c] = r
    nxt = k1
    for j in range(k2):                           # new clusters with no original partner
        if remap[j] < 0:
            remap[j] = nxt
            nxt += 1
    matched = remap[new_lab]

    rows = []
    for j in range(k2):
        n_tot = int((new_lab == j).sum())
        n_shared = int(C[:, j].sum())
        oid = remap[j]
        ov = int(C[oid, j]) if oid < k1 else 0
        rows.append(dict(assigned=int(oid), v2_raw=j, is_new=oid >= k1,
                         n_v1=int(C[oid].sum()) if oid < k1 else 0,
                         n_v2=n_tot, n_v2_old_cells=n_shared,
                         n_v2_new_cells=n_tot - n_shared, overlap=ov,
                         kept=round(100 * ov / max(int(C[oid].sum()), 1), 1) if oid < k1 else 0.0,
                         purity=round(100 * ov / max(n_shared, 1), 1)))
    table = pd.DataFrame(rows).sort_values("assigned")
    ari = adjusted_rand_score(old, new)
    log(f"label transfer on {len(orig):,} shared cells   original k={k1}, new k={k2}   "
        f"ARI = {ari:.4f}   shared cells keeping their original name: "
        f"{C[r, c].sum() / len(orig) * 100:.1f}%")
    log("kept   = % of that original cluster that landed in the matched cluster\n"
        "purity = % of the new cluster's originally labelled cells that came from that "
        "original cluster\n"
        + table.to_string(index=False))
    return matched, table, ari


def transfer_matrix(outdir):
    """Original-label x new-cluster contingency tables, in both directions.

    ``label_transfer.csv`` reports only the matched pair; these show how each original
    cluster actually split and what each new cluster is a mixture of. Cells absent
    from the original labelling get their own row so the columns account for every
    cell. Reads ``cell_clusters.csv`` in ``outdir`` and writes ``transfer_*.csv``
    beside it; the file names and the ``v1_c`` / ``v2_c`` row and column prefixes
    (original label, new cluster) are on-disk names.
    """
    outdir = config.ensure(outdir)
    meta = pd.read_csv(outdir / "cell_clusters.csv", usecols=["id", "cluster"])
    orig = pd.read_csv(ORIGINAL_LABELS, usecols=["id", "cluster"]).rename(columns={"cluster": "v1"})
    df = meta.merge(orig, on="id", how="left")
    df["v1"] = df["v1"].fillna(-1).astype(int)          # -1 = not in the original labelling

    table = pd.crosstab(df["v1"], df["cluster"])
    table.index = [("new" if i < 0 else f"v1_c{i}") for i in table.index]
    table.columns = [f"v2_c{j}" for j in table.columns]
    table.index.name, table.columns.name = "v1_label", "v2_cluster"
    old = table.drop(index="new", errors="ignore")       # only cells that had an original label

    from_orig = 100 * old.div(old.sum(1), axis=0)         # where each original cluster went
    into_new = 100 * table.div(table.sum(0), axis=1)     # what each new cluster is made of
    into_orig = 100 * old.div(old.sum(0), axis=1)        # ... counting only originally labelled cells

    table.to_csv(outdir / "transfer_counts.csv")
    from_orig.round(2).to_csv(outdir / "transfer_from_v1_pct.csv")
    into_new.round(2).to_csv(outdir / "transfer_into_v2_pct.csv")
    into_orig.round(2).to_csv(outdir / "transfer_into_v2_oldcells_pct.csv")

    def ent(M):
        p = (M / 100).to_numpy()
        n = (p > 0).sum(1).clip(1)
        return pd.Series(-(p * np.log(p + 1e-12)).sum(1) / np.log(np.maximum(n, 2)), index=M.index)

    per_new = pd.DataFrame({
        "n_total": table.sum(0), "n_old_cells": old.sum(0),
        "n_new_cells": table.sum(0) - old.sum(0),
        "top_v1": into_orig.idxmax(0), "top_v1_pct": into_orig.max(0).round(1),
        "n_v1_ge1pct": (into_orig >= 1).sum(0), "n_v1_ge5pct": (into_orig >= 5).sum(0),
        "n_v1_ge10pct": (into_orig >= 10).sum(0), "mixing": ent(into_orig.T).round(3)})
    per_orig = pd.DataFrame({
        "n": old.sum(1), "top_v2": from_orig.idxmax(1), "top_v2_pct": from_orig.max(1).round(1),
        "n_v2_ge1pct": (from_orig >= 1).sum(1), "n_v2_ge5pct": (from_orig >= 5).sum(1),
        "n_v2_ge10pct": (from_orig >= 10).sum(1), "splitting": ent(from_orig).round(3)})
    per_new.to_csv(outdir / "transfer_per_v2_cluster.csv")
    per_orig.to_csv(outdir / "transfer_per_v1_cluster.csv")
    log("per new cluster, how many original labels feed it (originally labelled cells only):\n"
        + per_new.to_string()
        + "\nper original cluster, how many new clusters it split across:\n" + per_orig.to_string())
    return table, per_orig, per_new


def sweep():
    """All four clustering families over k = 10-30 on the core-corrected features."""
    cells = clustering.load_cells(pd.read_csv(config.TISSUE_META))
    clustering.sweep(FEATURES, cells, REF_DIR, RESOLUTIONS)


def view(name):
    """Relabel one swept labelling (e.g. ``kmeans_k13``) with the original ids.

    Writes ``<name>_v1labels/`` under the sweep directory: ``cell_clusters.csv``,
    ``label_transfer.csv``, the transfer matrices and the cluster profile and
    composition tables.
    """
    cells = clustering.load_cells(pd.read_csv(config.TISSUE_META))
    meta = cells.meta
    raw = clustering.relabel(np.load(REF_DIR / "all_labels.npz")[name].astype(np.int64))
    lab, match, _ = transfer_labels(meta, raw)
    meta["cluster"], meta["cluster_raw"] = lab, raw

    outdir = config.ensure(REF_DIR / f"{name}_v1labels")
    match.to_csv(outdir / "label_transfer.csv", index=False)
    meta.to_csv(outdir / "cell_clusters.csv", index=False)
    clustering.finalize(meta, lab, cells, outdir)
    transfer_matrix(outdir)
    log(f"done -> {outdir}")


def main(command="view", name="kmeans_k13"):
    """``sweep`` (cluster over k = 10-30) or ``view <name>`` (relabel one labelling)."""
    if command == "sweep":
        sweep()
    else:
        view(name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["sweep", "view"])
    ap.add_argument("name", nargs="?", default="kmeans_k13",
                    help="labelling to relabel for 'view' (a key of all_labels.npz)")
    a = ap.parse_args()
    main(a.command, a.name)
