"""Support for the cluster-prevalence tree and the choice of normalisation (Extended Data Fig. 2k).

1. Bootstrap and jackknife support for the Ward tree of the prevalence table
(Supplementary Methods, "Cluster prevalence and hierarchy"). Per-unit prevalence
profiles of the retained clusters are z-scored within cluster and clustered with
Ward linkage; the units (donors, or cores with ``--unit core``) are resampled with
replacement ``B`` times, the profiles recomputed and reclustered, and the support of
a branch is the percentage of replicates in which its exact leaf set reappears as a
clade. A leave-one-unit-out jackknife shows whether a single unit holds a branch up.
Branches under 70% are unresolved. The tree itself is drawn by
``figures.ed2jk_cluster_annotation`` with the same implementation
(``subcell.utils.dendrogram``).

2. Where each measurement's variance lives (total = between tissue x cancer
group + between cores within a group + within core), and how much the per-cluster
means of the withheld measurements change with the normalisation: z-scored within
acquisition block (the one drawn in Extended Data Fig. 2j), within core, or over the
whole dataset.

The retained clusters are those with at least ``naming.TISSUE_MIN_CORRECT`` (60%) of
their cells scored correct in the manual quality assessment (``config.TISSUE_ANNOT_K13``,
column ``correct``), as in Extended Data Fig. 2j,k.

    python -m subcell.tissue.prevalence_support [--unit patient]

Inputs   ``config.TISSUE_CLUSTERS``/cell_clusters.csv, ``config.TISSUE_ANNOT_K13``,
         ``config.TISSUE_INTENSITY``
Outputs  tables on stdout
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from subcell import config, naming
from subcell.runlog import log
from subcell.tissue import clustering
from subcell.utils import dendrogram, zscore

B = 2000
SEED = 0
UNIT_COLUMN = {"patient": "patient", "core": "core_id"}   # prevalence unit -> metadata column
# (tissue, cancer, label) groups of prevalence units, in figure order
GROUPS = [("Pancreas", "No", "Normal"), ("Pancreas", "Yes", "Cancer"),
          ("Liver", "No", "Normal"), ("Liver", "Yes", "Cancer")]


def load_annotation():
    """The manual annotation of the k = 13 clusters, indexed by cluster id."""
    a = pd.read_csv(config.TISSUE_ANNOT_K13, sep="\t", comment="#", dtype=str,
                    keep_default_na=False).fillna("")
    a["cluster"] = a["cluster"].astype(int)
    return a.set_index("cluster")


def correct_fraction(a, k):
    """Share of cluster *k*'s cells scored correct: "yes" is 1, "60%" is 0.6, blank 0."""
    v = str(a.loc[k, "correct"]).strip().lower()
    if v == "yes":
        return 1.0
    if v.endswith("%"):
        return float(v[:-1]) / 100.0
    return 0.0


def retained_clusters(a):
    """Clusters with at least ``naming.TISSUE_MIN_CORRECT`` of their cells correct."""
    return [k for k in a.index if correct_fraction(a, k) >= naming.TISSUE_MIN_CORRECT]


def load():
    """Cells of the reported partition and the retained cluster ids."""
    meta = pd.read_csv(config.TISSUE_CLUSTERS / "cell_clusters.csv",
                       usecols=["id", "cluster", "core_id", "patient", "slide", "tissue",
                                "cancer"])
    return meta, sorted(retained_clusters(load_annotation()))


def unit_order(meta, unit):
    """Prevalence units grouped by tissue x cancer; no donor spans two groups."""
    col = UNIT_COLUMN[unit]
    t = meta.groupby(col)[["tissue", "cancer"]].first()
    key = ((lambda s: (s[0], int(s.split("_R")[1]))) if unit == "core"
           else (lambda s: (len(s), s)))
    out, spans = [], []
    for tis, can, lbl in GROUPS:
        got = sorted(t.index[(t.tissue == tis) & (t.cancer == can)], key=key)
        if got:
            spans.append((len(out), len(out) + len(got), tis, lbl))
            out += got
    assert len(out) == t.shape[0], "a unit fell outside the tissue x cancer grid"
    return out, spans


def prevalence(meta, unit, units):
    ct = pd.crosstab(meta["cluster"], meta[UNIT_COLUMN[unit]])
    return (100 * ct / ct.sum(0))[units]


def tree_support(meta, keep, rng, unit="patient"):
    """Bootstrap and jackknife support, and co-membership at 2- and 3-group cuts."""
    units, _ = unit_order(meta, unit)
    P = prevalence(meta, unit, units).loc[keep]
    lab = list(P.index)
    X = P.to_numpy(float)
    L0 = dendrogram.ward_tree(X, lab)
    boot = dendrogram.bootstrap_support(X, lab, L0, B, rng)
    jack = dendrogram.jackknife_support(X, lab, L0)

    df = pd.DataFrame([{"branch": " ".join(f"c{x}" for x in sorted(c)), "size": len(c),
                        "bootstrap_pct": boot[c], "jackknife_pct": jack[c]}
                       for c in boot]).sort_values("bootstrap_pct", ascending=False)
    log(f"bootstrap (B={B}) and jackknife support over {X.shape[1]} {unit} units:\n"
        + df.round(1).to_string(index=False)
        + "\n>= 70% is the usual bar for a supported branch; < 50% means the branch "
          "is not reproducible under resampling of the units.")

    for ngrp in (2, 3):
        co = np.zeros((len(lab), len(lab)))
        for _ in range(B):
            cols = rng.integers(0, X.shape[1], X.shape[1])
            g = sch.fcluster(dendrogram.ward_tree(X[:, cols], lab), ngrp, criterion="maxclust")
            co += (g[:, None] == g[None, :])
        co = pd.DataFrame(100 * co / B, index=[f"c{x}" for x in lab],
                          columns=[f"c{x}" for x in lab])
        log(f"pairwise co-membership at a {ngrp}-group cut (% of bootstraps):\n"
            + co.round(0).astype(int).to_string())
    return P


def compare_normalisations(meta, keep):
    """Variance decomposition of each measurement and per-cluster means under three z-scores."""
    im = clustering.aligned_intensity(meta["id"])
    E = zscore.log_heavy_tailed(im[[c for c, _ in clustering.MEASURED]].to_numpy())

    grp = (meta["tissue"] + "_" + meta["cancer"]).to_numpy()
    core = meta["core_id"].to_numpy()
    rows = []
    for j, (_, lbl) in enumerate(clustering.MEASURED):
        x = E[:, j]
        tot = x.var()
        gm = pd.Series(x).groupby(grp).transform("mean").to_numpy()
        cm = pd.Series(x).groupby(core).transform("mean").to_numpy()
        rows.append({"metric": lbl.replace("\n", " "),
                     "between_condition_%": 100 * gm.var() / tot,
                     "between_core_within_condition_%": 100 * (cm - gm).var() / tot,
                     "within_core_%": 100 * (x - cm).var() / tot})
    v = pd.DataFrame(rows)
    log("where each metric's variance lives:\n" + v.round(1).to_string(index=False)
        + f"\nmean between-core-within-condition share: "
          f"{v['between_core_within_condition_%'].mean():.1f}%  vs between-condition "
          f"{v['between_condition_%'].mean():.1f}%")

    names = [lbl.replace("\n", " ") for _, lbl in clustering.MEASURED]
    cl = meta["cluster"].to_numpy()
    tables = {}
    for tag, Z in {"acquisition block": zscore.zscore_within(E, meta["slide"].to_numpy()),
                   "core": zscore.zscore_within(E, core),
                   "whole dataset": zscore.zscore_within(E, np.zeros(len(E)))}.items():
        tables[tag] = pd.DataFrame(Z, columns=names).groupby(cl).mean().loc[keep]
        log(f"per-cluster means, z-scored within {tag}:\n" + tables[tag].round(2).to_string())

    ref = tables["acquisition block"]
    for tag in ("core", "whole dataset"):
        d = (tables[tag] - ref).abs()
        r = np.corrcoef(tables[tag].to_numpy().ravel(), ref.to_numpy().ravel())[0, 1]
        log(f"{tag} vs acquisition block: Pearson r {r:.4f}   mean |diff| "
            f"{d.to_numpy().mean():.3f}   max {d.to_numpy().max():.3f}")
    return tables


def main(unit="patient"):
    rng = np.random.default_rng(SEED)
    meta, keep = load()
    log(f"{len(meta):,} cells · {meta.core_id.nunique()} cores · "
        f"{meta.patient.nunique()} donors · retained clusters {keep}")
    tree_support(meta, keep, rng, unit)
    compare_normalisations(meta, keep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--unit", default="patient", choices=sorted(UNIT_COLUMN),
                    help="prevalence unit: donor (default) or core")
    a = ap.parse_args()
    main(a.unit)
