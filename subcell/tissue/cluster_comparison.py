"""Which (batch correction, clustering family) gives the best cell groupings?

Supplementary Note 2, Supplementary Methods "Clustering" (the comparison behind the
choice of k-means on core-corrected features for Extended Data Fig. 2i-k). Every
harmonized feature matrix is clustered with the four families over a coarse k grid,
every labelling is scored on one common panel, and the combinations are ranked.

    batch correction : none · cohort (2) · acquisition block (9) · donor (19) · core (29)
    families         : k-means · Ward · Gaussian mixture · Leiden

SCORING. Silhouette, Davies-Bouldin and Calinski-Harabasz are computed in the space
being clustered, and every correction changes that space: a correction that
collapses the embedding scores a high silhouette while destroying the biology. They
are recorded but excluded from the cross-correction ranking, which uses only
quantities that do not depend on the embedding:

    ext_ch          Calinski-Harabasz on 22 morphology and intensity measurements
                    the clustering never sees, z-scored within core
    spatial_excess  neighbours in tissue space sharing the label, beyond the
                    per-core composition baseline
    stability       ARI between the full labelling and clusterings of two 80%
                    resamples
    core_maxfrac    cell-weighted largest share of any cluster coming from one core
                    (lower is better; high means the cluster is a batch)

Metrics drift with k, so labellings are percentile-ranked within k-bins before
averaging.

    python -m subcell.tissue.cluster_comparison

Inputs   ``config.TISSUE_HARM``/harmonized_<key>.npy, ``config.TISSUE_META``,
         ``config.TISSUE_INTENSITY``; the uncorrected features come from
         ``harmonization.FEATURE_CACHE``
Outputs  ``config.TISSUE_SWEEP``: combo_scores.csv (checkpointed after each
         correction), combo_ranking.csv, winner.csv
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from subcell import config
from subcell.runlog import log
from subcell.tissue import clustering as ct
from subcell.tissue.harmonization import FEATURE_CACHE

OUT_DIR = config.TISSUE_SWEEP
HARM = config.TISSUE_HARM

# correction name -> feature matrix. "acq" is the acquisition block (the filename
# prefix H1..H7, G1, G2), stored as harmonized_slide.npy.
VARIANTS = {
    "none": FEATURE_CACHE / "features.npy",
    "cohort": HARM / "harmonized_cohort.npy",
    "acq": HARM / "harmonized_slide.npy",
    "patient": HARM / "harmonized_patient.npy",
    "core": HARM / "harmonized_core.npy",
}
NLEVEL = {"none": 1, "cohort": 2, "acq": 9, "patient": 19, "core": 29}
DISPLAY = {"none": "no correction", "cohort": "cohort G/H (2)",
           "acq": "acquisition block (9)", "patient": "donor (19)",
           "core": "core (29)"}
RESOLUTIONS = [0.40, 0.60, 0.80, 1.00, 1.30, 1.75]

# embedding-invariant metrics only (see the module docstring)
XVAR = {"ext_ch": 1, "stability": 1, "spatial_excess": 1, "core_maxfrac": -1}
GEOM = ["silhouette", "db", "ch"]


def sweep_one(name, path, cells):
    """Score every (family, k) labelling on one harmonized matrix."""
    X = np.load(path).astype(np.float32)
    assert len(X) == len(cells.meta), f"{name}: {len(X)} rows vs {len(cells.meta)} cells"
    Xg = torch.as_tensor(X, device=ct.DEV)
    del X
    log(f"=== {name}: {tuple(Xg.shape)} ===")
    rows, _ = ct.sweep_methods(Xg, cells, ct.KS_COARSE, ct.KS_COARSE, RESOLUTIONS, extra={"harmonization": name})
    del Xg
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def rank(df):
    """Rank (correction, method, k) on embedding-invariant metrics only."""
    ok = df[df.min_frac > 0.001].copy()
    for c, s in XVAR.items():
        ok[f"r_{c}"] = (s * ok[c]).rank(pct=True)
    ok["xvar_score"] = ok[[f"r_{c}" for c in XVAR]].mean(1)
    ok["kbin"] = pd.cut(ok.k, [9, 13, 17, 21, 25, 31])
    ok["within_kbin"] = ok.groupby("kbin", observed=True)["xvar_score"].rank(pct=True)
    ok["combo"] = ok.harmonization + " x " + ok.family
    ok["batches"] = ok.harmonization.map(NLEVEL)
    return ok.sort_values("xvar_score", ascending=False)


def report(ok):
    show = ["harmonization", "method", "k", "ext_ch", "stability", "spatial_excess",
            "core_maxfrac", "silhouette", "db", "min_frac", "xvar_score"]
    log("combination ranking on embedding-invariant metrics only (silhouette/DB/CH "
        "are shown but not ranked: they are not comparable across corrections)")

    combo = ok.groupby("combo")["within_kbin"].agg(["mean", "size"]).sort_values(
        "mean", ascending=False)
    log("mean percentile within k-bin, by (correction x method):\n" + combo.round(4).to_string())

    m = ok.groupby("harmonization").agg(
        batches=("batches", "first"), score=("within_kbin", "mean"),
        ext_ch=("ext_ch", "mean"), stability=("stability", "mean"),
        spatial=("spatial_excess", "mean"), core_maxfrac=("core_maxfrac", "mean"))
    m.index = [DISPLAY.get(i, i) for i in m.index]
    log("marginal, batch correction (averaged over methods and k):\n"
        + m.sort_values("score", ascending=False).round(4).to_string())
    log("marginal, method (averaged over corrections and k):\n"
        + ok.groupby("family")["within_kbin"].mean().sort_values(ascending=False)
        .round(4).to_string())
    log("top 20 labellings by xvar_score:\n" + ok.head(20)[show].round(4).to_string(index=False))
    log("best labelling per correction:\n" + ok.groupby("harmonization").head(1)[show]
        .round(4).to_string(index=False))
    log("geometric metrics are not comparable across corrections (mean per correction):\n"
        + ok.groupby("harmonization")[GEOM].mean().round(4).to_string())

    best = ok.iloc[0]
    log(f"WINNER  harmonization={best.harmonization}  method={best.method}  "
        f"k={int(best.k)}   xvar_score {best.xvar_score:.4f}")
    return best


def main():
    config.ensure(OUT_DIR)
    cells = ct.load_cells(pd.read_csv(config.TISSUE_META))
    out = OUT_DIR / "combo_scores.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    for name, path in VARIANTS.items():
        if len(done) and name in set(done.harmonization):
            log(f"=== {name}: already scored, skipping")
            continue
        if not path.exists():
            log(f"=== {name}: {path} missing, skipping")
            continue
        t0 = time.time()
        df = sweep_one(name, path, cells)
        done = pd.concat([done, df], ignore_index=True)
        done.to_csv(out, index=False)   # checkpoint after every correction
        log(f"=== {name} done in {(time.time() - t0) / 60:.1f} min "
            f"({len(df)} labellings) -> {out}")

    ok = rank(done)
    ok.to_csv(OUT_DIR / "combo_ranking.csv", index=False)
    best = report(ok)
    pd.DataFrame([best]).to_csv(OUT_DIR / "winner.csv", index=False)
    log(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    main()
