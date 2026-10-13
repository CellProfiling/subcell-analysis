"""Drug-induced perturbation signatures in CM4AI (Fig. 4b, Extended Data Fig. 5; tables of Fig. 4c).

For each antibody, treated (paclitaxel or vorinostat) and untreated cells are
compared in two readouts:

* image intensity: log2(cell_q90 + 1) of the single-cell crops, Mann-Whitney U
  per protein, Benjamini-Hochberg across proteins (log2 fold change, adjusted P);
* SubCell embedding: sliced Wasserstein distance (1,000 projections, seed 42)
  between the two sets of embeddings, Mann-Whitney U per embedding dimension
  combined by Simes' method, then Benjamini-Hochberg across proteins.

Only antibodies with at least 20 cells in both conditions are kept. Embeddings
are standardised and sphered (ZCA-cor) with respect to the negative-control
cells before the comparison. This module draws Fig. 4b (UMAP of the sphered
embeddings) and the intensity and embedding volcano plots of Extended Data Fig. 5,
and writes the per-protein tables behind Fig. 4c. The joint intensity-versus-
sliced-Wasserstein scatter of Fig. 4c is not drawn here.

    python -m subcell.perturbation.cm4ai_signatures [--out-dir DIR] [--cache-dir DIR]

Inputs: ``config.features("MAE-CellS-ProtS-Pool", "b2ai_features")`` and
``config.CM4AI_METADATA``. Cached tables (``features/`` and
``intensity/``) and the UMAP coordinates go to ``config.CM4AI_DIR /
"perturbation_analysis"``; figures to ``config.figure_dir("fig4")`` and
``config.figure_dir("ed5")``.
"""

from __future__ import annotations

import argparse

import colorcet as cc
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from subcell import config, style
from subcell.io.features import scale
from subcell.perturbation.distances import compute_sliced_wasserstein
from subcell.perturbation.spherize import Spherize

MODEL = "MAE-CellS-ProtS-Pool"
SHIFT = "sliced_wasserstein"          # column holding the embedding shift
INTENSITY_COLS = ("cell_q75", "cell_q90", "cell_q95")
MIN_CELLS = 20                        # per condition and antibody
LOGFC_THRESH = 2
LOGP_THRESH = 2                       # -log10 of the adjusted P-value
# drug -> (file prefix of the cached embedding table, of the cached intensity table)
DRUGS = {"Paclitaxel": ("paclitaxel", "pacl"), "Vorinostat": ("vorinostat", "vorino")}

# Antibodies with known imaging artifacts, excluded from the signature analysis
# ("Proteins with known imaging artifacts were excluded" in the Methods).
REMOVED_ANTIBODIES = [
    "HPA000593",
    "HPA004689",
    "HPA009985",
    "HPA063716",
    "HPA064413",
    "HPA065946",
    "HPA067552",
    "HPA067561",
    "HPA069692",
    "HPA059206",
]


def plot_umap(umap_df, name, out_dir):
    """Fig. 4b: UMAP of the cell embeddings coloured by treatment."""
    fig = style.figure(style.WIDTH_1COL, height=62)
    ax = style.axes(fig, x=10, y=4, w=52, h=52)
    sns.scatterplot(
        x="UMAP1", y="UMAP2", hue="Treatment", palette=cc.glasbey_dark,
        s=1, alpha=0.5, data=umap_df, ax=ax, rasterized=True,
    )
    ax.set_box_aspect(1.0)
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles:
        handle.set_alpha(1)
    ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc=2,
              borderaxespad=0.0, markerscale=3, frameon=False)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    style.save(fig, name, out_dir)


def simes_pvalue(p_values):
    """Combine per-dimension p-values using Simes' method.

    Valid under positive regression dependency on subsets (PRDS), which is
    plausible for correlated deep-learning embedding dimensions. Fisher's method
    would require independence.
    """
    p_clean = p_values[~np.isnan(p_values)]
    k = len(p_clean)
    if k == 0:
        return np.nan
    p_sorted = np.sort(p_clean)
    ranks = np.arange(1, k + 1)
    return float(np.minimum(1.0, np.min(k * p_sorted / ranks)))


def _adjust(res_df):
    """Benjamini-Hochberg adjusted P and its -log10, for plotting."""
    res_df = res_df.dropna(subset=["p_value"])
    _, res_df["adj_p_value"], _, _ = multipletests(res_df["p_value"], method="fdr_bh")
    res_df["-log10_p"] = -np.log10(res_df["adj_p_value"] + 1e-300)
    return res_df


def analyze_perturbation_intensity(df, feature_col, drug):
    """Per-antibody log2 fold change of an intensity column, treated vs untreated."""
    results = []
    log_col = f"log2_{feature_col}"
    df[log_col] = np.log2(df[feature_col] + 1)

    for protein in df["protein"].unique():
        prot_df = df[df["protein"] == protein].reset_index(drop=True)

        gene_name = prot_df["gene_name"][0]
        assert np.all(prot_df["gene_name"] == gene_name), "not all genes are same for antibody"

        ctrl_vals = prot_df[prot_df["treatment"] == "Untreated"][log_col].values
        treat_vals = prot_df[prot_df["treatment"] == drug][log_col].values
        if len(ctrl_vals) < MIN_CELLS or len(treat_vals) < MIN_CELLS:
            continue

        p_val = stats.mannwhitneyu(
            treat_vals, ctrl_vals, alternative="two-sided", nan_policy="omit"
        ).pvalue
        results.append({
            "protein": protein,
            "gene_name": gene_name,
            "log2FC": np.mean(treat_vals) - np.mean(ctrl_vals),
            "p_value": p_val,
        })

    res_df = _adjust(pd.DataFrame(results))
    res_df = res_df.sort_values("log2FC", ascending=False).reset_index(drop=True)
    return res_df.sort_values(by="adj_p_value")


def analyze_perturbation_features(df, features, drug):
    """Per-antibody sliced Wasserstein shift of the embeddings, treated vs untreated.

    ``features`` is the (n_cells, n_dims) array aligned with the rows of ``df``.
    """
    results = []
    for protein in tqdm(df["protein"].unique()):
        is_protein = (df["protein"] == protein).to_numpy()
        prot_df = df[is_protein].reset_index(drop=True)
        prot_feats = features[is_protein]

        gene_name = prot_df["gene_name"][0]
        assert np.all(prot_df["gene_name"] == gene_name), "not all genes are same for antibody"

        untreated_feats = prot_feats[(prot_df["treatment"] == "Untreated").to_numpy()]
        treated_feats = prot_feats[(prot_df["treatment"] == drug).to_numpy()]
        if len(untreated_feats) < MIN_CELLS or len(treated_feats) < MIN_CELLS:
            continue

        shift_magnitude = compute_sliced_wasserstein(untreated_feats, treated_feats)

        # Mann-Whitney U per dimension (no normality assumption), combined via Simes' test.
        mwu_result = stats.mannwhitneyu(
            treated_feats, untreated_feats, axis=0,
            alternative="two-sided", nan_policy="omit",
        )
        results.append({
            "protein": protein,
            "gene_name": gene_name,
            SHIFT: shift_magnitude,
            "p_value": simes_pvalue(mwu_result.pvalue),
        })

    res_df = _adjust(pd.DataFrame(results))
    return res_df.sort_values(SHIFT, ascending=False).reset_index(drop=True)


def plot_volcano(res_df, name, out_dir, x_col, x_label, x_thresh=2, y_thresh=2,
                 plot_genes=(), protein_col="gene_name"):
    """Volcano plot of per-antibody effect size (``x_col``) against -log10 adjusted P.

    Points beyond both thresholds are coloured (up/down for the signed log2FC);
    antibodies in ``plot_genes`` -- significant in both readouts -- get a
    different marker. The ten largest effects are labelled.
    """
    res_df = res_df[~(res_df["protein"] == "NEGATIVE")].reset_index(drop=True)
    signed = x_col == "log2FC"

    res_df["Significance"] = "Not Significant"
    # strict inequalities, as in the significance rule of ``_run`` that builds the tables
    significant_y = res_df["-log10_p"] > y_thresh
    if signed:
        res_df.loc[significant_y & (res_df[x_col] > x_thresh), "Significance"] = "Significant (Up Reg.)"
        res_df.loc[significant_y & (res_df[x_col] < -x_thresh), "Significance"] = "Significant (Down Reg.)"
    else:
        res_df.loc[significant_y & (res_df[x_col] > x_thresh), "Significance"] = "Significant"

    res_df["Common Genes"] = res_df["gene_name"].isin(plot_genes)

    palette = {
        "Not Significant": "lightgrey",
        "Significant (Up Reg.)": "green",
        "Significant (Down Reg.)": "red",
        "Significant": "green",
    }
    sizes = {
        "Not Significant": 5,
        "Significant (Up Reg.)": 20,
        "Significant (Down Reg.)": 20,
        "Significant": 15,
    }
    fig = style.figure(style.WIDTH_1COL, height=62)
    ax = style.axes(fig, x=12, y=4, w=72, h=48)
    sns.scatterplot(
        data=res_df, x=x_col, y="-log10_p", hue="Significance", palette=palette,
        style="Common Genes", edgecolor=None, size="Significance", sizes=sizes,
        legend=False, alpha=0.75, ax=ax, rasterized=True,
    )
    x_max = res_df[x_col].max() + 0.1
    ax.set_xlim(-x_max if signed else 0, x_max)
    ax.set_box_aspect(0.75)
    ax.axvline(x=x_thresh, color="black", linestyle="--", alpha=0.5)
    if signed:
        ax.axvline(x=-x_thresh, color="black", linestyle="--", alpha=0.5)
    ax.axhline(y=y_thresh, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$-\log_{10}$(adjusted P)")

    label_df = res_df.loc[res_df[x_col].abs().nlargest(10).index]
    x_offset = label_df[x_col].max() * 0.015
    y_offset = label_df["-log10_p"].max() * 0.015
    for _, row in label_df.iterrows():
        ax.text(row[x_col] + x_offset, row["-log10_p"] + y_offset,
                row[protein_col].replace("_", ","), fontsize=style.FS_TINY, color="black")

    style.save(fig, name, out_dir)


def _cached_table(path, compute):
    """Read the table at *path*, computing and writing it first if absent."""
    if path.exists():
        return pd.read_csv(path)
    table = compute()
    table.to_csv(path, index=False)
    return table


def load_sphered_embeddings():
    """Per-cell metadata and the standardised, negative-control-sphered embeddings."""
    metadata_df, feature_data = torch.load(
        config.features(MODEL, "b2ai_features"), map_location="cpu", weights_only=False
    )
    keep = ~metadata_df["protein"].isin(REMOVED_ANTIBODIES)
    metadata_df = metadata_df[keep].reset_index(drop=True)
    feature_data = feature_data[keep.to_numpy()]

    feature_data = scale(feature_data)[0].numpy()
    sph = Spherize(epsilon=1e-8, center=True, method="ZCA-cor")
    sph = sph.fit(feature_data[(metadata_df["protein"] == "NEGATIVE").to_numpy()])
    return metadata_df, sph.transform(feature_data)


def main(cache_dir=None, out_dir=None):
    """Compute (or read cached) signatures and render Fig. 4b and Extended Data Fig. 5.

    With ``out_dir`` both figures are written there; by default Fig. 4b goes to
    ``figure_dir("fig4")`` and the volcano plots to ``figure_dir("ed5")``.
    """
    style.use_agg()
    with style.nature():
        _run(cache_dir, out_dir)


def _run(cache_dir, out_dir):
    from umap import UMAP

    cache = config.ensure(cache_dir or config.CM4AI_DIR / "perturbation_analysis")
    feat_dir = config.ensure(cache / "features")
    intensity_dir = config.ensure(cache / "intensity")
    fig4_dir = out_dir or config.figure_dir("fig4")
    ed5_dir = out_dir or config.figure_dir("ed5")

    df = pd.read_csv(
        config.CM4AI_METADATA,
        usecols=["protein", "gene_name", "treatment", *INTENSITY_COLS],
    )
    df = df[~df["protein"].isin(REMOVED_ANTIBODIES)].reset_index(drop=True)

    metadata_df, feature_data = load_sphered_embeddings()

    def fit_umap():
        coords = UMAP(
            n_components=2, metric="cosine", n_neighbors=20, min_dist=0.2,
            random_state=42, verbose=1, n_jobs=-1,
        ).fit_transform(feature_data)
        umap_df = pd.DataFrame(coords, columns=["UMAP1", "UMAP2"], index=metadata_df.index)
        umap_df["id"] = metadata_df["id"]
        umap_df["Protein"] = metadata_df["protein"]
        umap_df["Treatment"] = metadata_df["treatment"]
        return umap_df

    umap_df = _cached_table(feat_dir / "umap_feat.csv", fit_umap)
    plot_umap(umap_df, "fig4b_umap", fig4_dir)

    for drug, (feat_prefix, int_prefix) in DRUGS.items():
        feat_res = _cached_table(
            feat_dir / f"{feat_prefix}_{SHIFT}.csv",
            lambda: analyze_perturbation_features(metadata_df, feature_data, drug),
        )
        # An antibody counts as shifted when it moves more than the negative-control wells.
        feat_shift_thresh = feat_res[feat_res["protein"] == "NEGATIVE"][SHIFT].values[0]
        feat_sig_genes = feat_res[
            (feat_res[SHIFT] > feat_shift_thresh) & (feat_res["-log10_p"] > LOGP_THRESH)
        ]["gene_name"]

        panel = "a" if drug == "Paclitaxel" else "b"
        for int_col in INTENSITY_COLS:
            int_res = _cached_table(
                intensity_dir / f"{int_prefix}_{int_col}.csv",
                lambda: analyze_perturbation_intensity(df, int_col, drug),
            )
            int_sig_genes = int_res[
                (int_res["log2FC"].abs() > LOGFC_THRESH) & (int_res["-log10_p"] > LOGP_THRESH)
            ]["gene_name"]
            common_genes = set(feat_sig_genes) & set(int_sig_genes)

            stem = f"ed5{panel}_{drug.lower()}_{int_col}"
            plot_volcano(
                int_res, f"{stem}_intensity", ed5_dir, x_col="log2FC",
                x_label=f"log2 fold change of {int_col}",
                x_thresh=LOGFC_THRESH, y_thresh=LOGP_THRESH, plot_genes=common_genes,
            )
            plot_volcano(
                feat_res, f"{stem}_embedding", ed5_dir, x_col=SHIFT,
                x_label="Sliced Wasserstein distance",
                x_thresh=feat_shift_thresh, y_thresh=LOGP_THRESH, plot_genes=common_genes,
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", default=None,
                    help="cached tables (default: CM4AI_DIR/perturbation_analysis)")
    ap.add_argument("--out-dir", default=None,
                    help="write all figures here instead of figure_dir('fig4') / ('ed5')")
    a = ap.parse_args()
    main(a.cache_dir, a.out_dir)
