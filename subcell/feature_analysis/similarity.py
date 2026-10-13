"""Cell-line and localization similarity of model features, and Mantel statistics.

Feeds Fig. 3a-c and Extended Data Fig. 8a,b,e. Following Doron et al., the
single-cell features of each model are averaged within every cell line and
within every single-localization category and reduced to ten principal
components. Cosine similarities between the averaged profiles are compared with
the ground truth by the Mantel statistic: bulk mRNA similarity between cell lines
(HPA v24, ``config.RNA_CELLINE``) and the expert localization hierarchy
(``config.LOCATION_GROUPS``, Supplementary Table 2). Every (model, Harmony regime)
in ``config.FUNCTION_PREDICTION_MODELS`` is scored.

Inputs: ``config.features(model, harmonized=...)``.
Outputs, in ``config.MORPHOLOGY_DIR``: ``profiles/<model>_<harmonized>.npz``
(averaged profiles), ``rna_protein_cc_results.csv`` (Mantel table; only missing
rows are computed) and ``mantel_summary.csv``. ``figures.fig3ab_similarity`` and
``figures.fig3c_mantel`` draw the panels from these.

Run::

    python -m subcell.feature_analysis.similarity
"""

import functools
import os

import mantel
import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn import decomposition
from sklearn.cross_decomposition import CCA
from tqdm import tqdm

from subcell import config
from subcell.io.features import scale
from subcell.naming import HARM_METHODS, HARM_ORDER, HARM_SHORT, display
from subcell.runlog import log

# Self-supervised encoders only in the summary. bestfitting and ViT-Supervised
# exist for one harmonization target (none), so they cannot be compared across
# harmonizations.
MODEL_ORDER = list(HARM_METHODS)

# Every (model, harmonization) embedding scored, the same grid as function prediction.
GRID = config.FUNCTION_PREDICTION_MODELS
RESULT_COLUMNS = ["Method", "Harmonized", "PCA Dim", "RNAseq Mantel Statistic",
                  "Top-1 Accuracy", "Localization Mantel Statistic"]
MANTEL_PERMUTATIONS = 1000000
STATISTICS = ["RNAseq Mantel Statistic", "Localization Mantel Statistic"]


def get_cell_avg_features(df, features):
    features = features.cpu().numpy()

    cell_lines = df["atlas_name"].unique().tolist()
    cell_avg_features = []
    for cell_line in tqdm(cell_lines, desc="Cell lines"):
        cell_line_idxs = df["atlas_name"] == cell_line
        mean_cell_line_features = features[cell_line_idxs].mean(axis=0)
        cell_avg_features.append(mean_cell_line_features)
    cell_avg_features = np.array(cell_avg_features)
    return cell_lines, cell_avg_features


@functools.lru_cache(maxsize=1)
def rna_celline():
    return pd.read_csv(config.RNA_CELLINE, sep="\t")


def get_tpms(cell_lines):
    """Bulk mRNA TPMs per cell line, and the cell lines without any."""
    rna_seq_df = rna_celline()

    tpms = []
    removed_cell_lines = []
    for cell_line in cell_lines:
        cell_line_tpm = rna_seq_df[rna_seq_df["Cell line"] == cell_line]["TPM"].values
        if len(cell_line_tpm) == 0:
            removed_cell_lines.append(cell_line)
            continue
        tpms.append(cell_line_tpm)
    return np.array(tpms), removed_cell_lines


def pca_profiles(features, pca_dims):
    """Standardised profiles, reduced to *pca_dims* principal components."""
    scaled, _, _ = scale(features)
    if pca_dims is not None:
        scaled = decomposition.PCA(n_components=pca_dims, svd_solver="full").fit_transform(scaled)
    return scaled


def similarity(x):
    return 1 - squareform(pdist(x, metric="cosine"))


def cell_line_matrices(cell_lines, reduced_tpms, reduced_features):
    """mRNA and feature cosine similarity between cell lines.

    Both matrices and the labels are in the leaf order of the mRNA dendrogram, so
    the feature matrix is shown against an ordering it did not choose.
    """
    rna = similarity(reduced_tpms)
    feat = similarity(reduced_features)
    dn = dendrogram(linkage(rna), labels=list(cell_lines), no_plot=True)
    order = dn["leaves"]
    return list(dn["ivl"]), rna[np.ix_(order, order)], feat[np.ix_(order, order)]


def cell_line_statistics(cell_lines, cell_features, tpms, pca_dims):
    """Mantel statistic of cell-line features against mRNA, and CCA top-1 accuracy."""
    scaled_tpms = pca_profiles(tpms, pca_dims)
    scaled_averaged_features = pca_profiles(cell_features, pca_dims)
    _, rna_matrix, feature_matrix = cell_line_matrices(
        cell_lines, scaled_tpms, scaled_averaged_features
    )

    np.fill_diagonal(feature_matrix, 0)
    np.fill_diagonal(rna_matrix, 0)
    mantel_test = mantel.test(
        squareform(feature_matrix), squareform(rna_matrix), perms=MANTEL_PERMUTATIONS
    )
    log(f"RNAseq hierarchy mantel test: {mantel_test}, dim: {scaled_averaged_features.shape[1]}")

    ca = CCA(n_components=2)
    ca.fit(scaled_tpms, scaled_averaged_features)
    X_c, Y_c = ca.transform(scaled_tpms, scaled_averaged_features)
    cc_distance_matrix = cdist(X_c, Y_c, metric="euclidean")

    k = 1
    top_1_accuracy = np.mean(
        (
            np.tile(np.arange(cc_distance_matrix.shape[0])[:, np.newaxis], k)
            == np.argsort(cc_distance_matrix, axis=0)[0:k, :].T
        ).sum(axis=1)
    )
    return mantel_test[0], top_1_accuracy


def get_localization_avg_features(df, features):
    loc_not_na_idxs = ~df["locations"].isna()
    df = df[loc_not_na_idxs].reset_index(drop=True)
    features = features[loc_not_na_idxs]

    single_loc_idxs = ~df["locations"].str.contains(",")
    single_loc_df = df[single_loc_idxs].reset_index(drop=True)
    single_loc_features = features[single_loc_idxs]

    unique_locs = sorted(single_loc_df["locations"].unique().tolist())
    loc_avg_features = []
    for loc in tqdm(unique_locs, desc="Locations"):
        loc_idxs = single_loc_df["locations"] == loc
        mean_loc_features = single_loc_features[loc_idxs].mean(axis=0)
        loc_avg_features.append(mean_loc_features)
    loc_avg_features = np.array(loc_avg_features)
    return unique_locs, loc_avg_features


def location_matrices(protein_localizations, features, pca_dims):
    """Feature similarity between localization profiles, and the expert hierarchy.

    The hierarchy scores 1 for the same minor group and 0.5 for the same major
    group. Both matrices and the labels are in the hierarchy's leaf order; the
    major and minor groups are {name: [first, last]} index spans in that order.
    """
    loc_group_mapping = pd.read_csv(config.LOCATION_GROUPS, sep="\t")
    loc_group_mapping = loc_group_mapping[
        (loc_group_mapping["Original annotation"].isin(protein_localizations))
        & (~loc_group_mapping["Grouping 3"].isna())
    ].reset_index(drop=True)

    labels = loc_group_mapping["Original annotation"].tolist()
    features = np.array([features[protein_localizations.index(loc)] for loc in labels])
    feature_similarity = similarity(pca_profiles(features, pca_dims))

    truth = np.zeros((len(labels), len(labels)))
    high_level_labels = loc_group_mapping["Grouping 3"].tolist()
    for i in range(len(labels)):
        for j in range(len(labels)):
            if high_level_labels[i] == high_level_labels[j]:
                truth[i, j] = 0.5
    low_level_labels = loc_group_mapping["Grouping 2"].tolist()
    for i in range(len(labels)):
        for j in range(len(labels)):
            if low_level_labels[i] == low_level_labels[j]:
                truth[i, j] = 1

    leaves = dendrogram(linkage(truth), labels=labels, no_plot=True)["leaves"]

    def spans(names):
        groups = pd.DataFrame([names[i] for i in leaves]).groupby(0).groups
        return dict(sorted(((k, [v.min(), v.max()]) for k, v in groups.items()),
                           key=lambda item: item[1][0]))

    return dict(
        labels=[labels[i] for i in leaves],
        similarity=feature_similarity[np.ix_(leaves, leaves)],
        truth=truth[np.ix_(leaves, leaves)],
        major=spans(high_level_labels),
        minor=spans(low_level_labels),
    )


def location_statistic(protein_localizations, features, pca_dims):
    """Mantel statistic of localization features against the expert hierarchy."""
    m = location_matrices(protein_localizations, features, pca_dims)
    feature_matrix, truth = m["similarity"], m["truth"]
    np.fill_diagonal(feature_matrix, 0)
    np.fill_diagonal(truth, 0)

    mantel_test = mantel.test(
        squareform(feature_matrix), squareform(truth), perms=MANTEL_PERMUTATIONS
    )
    log(f"protein hierarchy mantel test: {mantel_test}, dim: {pca_dims}")
    return mantel_test[0]


def profiles(method, harmonized):
    """Mean feature profiles of one embedding: per cell line with bulk mRNA (and
    its TPMs), and per single localization.

    Cached under ``MORPHOLOGY_DIR/profiles``, since computing them reads the whole
    cell table.
    """
    path = config.MORPHOLOGY_DIR / "profiles" / f"{method}_{harmonized}.npz"
    if not path.exists():
        df, feature_data = torch.load(
            config.features(method, harmonized=harmonized),
            map_location="cpu", weights_only=False
        )
        cell_lines, cell_features = get_cell_avg_features(df, feature_data)
        locs, loc_features = get_localization_avg_features(df, feature_data)
        config.ensure(path.parent)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, cell_lines=np.array(cell_lines), cell_features=cell_features,
                 locs=np.array(locs), loc_features=np.asarray(loc_features))
        os.replace(tmp, path)
    with np.load(path) as z:
        cell_lines, cell_features = z["cell_lines"].tolist(), z["cell_features"]
        locs, loc_features = z["locs"].tolist(), z["loc_features"]
    tpms, removed_cell_lines = get_tpms(cell_lines)
    keep = [i for i, cell in enumerate(cell_lines) if cell not in removed_cell_lines]
    return [cell_lines[i] for i in keep], cell_features[keep], tpms, locs, loc_features


def mantel_row(method, harmonized, pca_dims):
    """Both Mantel statistics of one (model, harmonization) embedding."""
    log(f"Processing {method} and {harmonized}")
    cell_lines, cell_features, tpms, locs, loc_features = profiles(method, harmonized)

    rnaseq_mantel, top_1_accuracy = cell_line_statistics(
        cell_lines, cell_features, tpms, pca_dims
    )
    localization_mantel = location_statistic(locs, loc_features, pca_dims)
    return dict(zip(RESULT_COLUMNS, [method, harmonized, pca_dims, rnaseq_mantel,
                                     top_1_accuracy, localization_mantel]))


def mantel_table(save_folder, pca_dims=10):
    """The Mantel statistics of every GRID embedding at *pca_dims*.

    Cached in ``rna_protein_cc_results.csv``: only the rows it lacks are computed,
    and the file is rewritten after each, so adding a model costs that model only.
    """
    path = f"{save_folder}/rna_protein_cc_results.csv"
    df = (pd.read_csv(path) if os.path.isfile(path)
          else pd.DataFrame(columns=RESULT_COLUMNS))
    have = df[df["PCA Dim"] == pca_dims]
    done = set(zip(have["Method"], have["Harmonized"]))
    for method, harmonizations in GRID.items():
        for harmonized in harmonizations:
            if (method, harmonized) in done:
                continue
            row = mantel_row(method, harmonized, pca_dims)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(path + ".tmp", index=False)
            os.replace(path + ".tmp", path)
    return df[df["PCA Dim"] == pca_dims].reset_index(drop=True)


def main(save_folder=None, pca_dims: int = 10):
    """Score every embedding and write ``mantel_summary.csv``.

    The Mantel statistics are cached (see ``mantel_table``), so re-rendering the
    figures does not repeat the feature loads.

    *pca_dims* is the dimension the aggregated profiles are reduced to before the
    Mantel test. It is recorded in the output table, because the statistic depends
    on it and a figure without it is not reproducible.
    """
    save_folder = str(config.ensure(save_folder or config.MORPHOLOGY_DIR))
    df = mantel_table(save_folder, pca_dims)

    # Only the harmonization targets shown in the figures. The dropped two
    # (plate+microscope, plate+microscope+cell line) match plate to three
    # decimals on both statistics, so they add rows without information.
    df = df[df["Harmonized"].isin(HARM_ORDER)
            & df["Method"].isin(MODEL_ORDER)].reset_index(drop=True)

    df["Model"] = pd.Categorical(
        df["Method"].map(display),
        categories=[display(m) for m in MODEL_ORDER if m in set(df["Method"])],
        ordered=True,
    )
    df["Harmonized on"] = pd.Categorical(
        df["Harmonized"].map(lambda h: HARM_SHORT[h].replace("\n", " ")),
        categories=[HARM_SHORT[h].replace("\n", " ") for h in HARM_ORDER],
        ordered=True,
    )

    # change relative to each model's own uncorrected embedding
    for stat in STATISTICS:
        base = df[df["Harmonized"] == "none"].set_index("Method")[stat]
        df[f"{stat} (delta vs uncorrected)"] = df[stat] - df["Method"].map(base)

    df.to_csv(f"{save_folder}/mantel_summary.csv", index=False)
    log(f"wrote {save_folder}/mantel_summary.csv")
    for stat in STATISTICS:
        print(f"\n=== {stat} ===")
        print(
            df.pivot_table(index="Model", columns="Harmonized on", values=stat,
                           observed=True).to_string(float_format="%.4f")
        )
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--save-folder", default=None)
    ap.add_argument("--pca-dims", type=int, default=10)
    a = ap.parse_args()
    main(a.save_folder, a.pca_dims)
