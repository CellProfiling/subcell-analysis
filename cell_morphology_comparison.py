import os
import re

import mantel
import matplotlib

matplotlib.use("Agg")

import colorcet as cc
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import cm
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn import decomposition
from sklearn.cross_decomposition import CCA
from tqdm import tqdm

cmap = cm.nipy_spectral


def scale(features, features_mean=None, features_std=None):
    if features_mean is None:
        features_mean = features.mean(axis=0)
    if features_std is None:
        features_std = features.std(axis=0)
    transformed_features = (features - features_mean) / (features_std + 0.00001)
    return transformed_features, features_mean, features_std


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


def get_tpms(cell_lines):
    rna_seq_df = pd.read_csv("annotations/rna_celline.tsv", sep="\t")

    tpms = []
    removed_cell_lines = []
    for cell_line in cell_lines:
        cell_line_tpm = rna_seq_df[rna_seq_df["Cell line"] == cell_line]["TPM"].values
        if len(cell_line_tpm) == 0:
            removed_cell_lines.append(cell_line)
            continue
        tpms.append(cell_line_tpm)
    tpms = np.array(tpms)
    return tpms, [
        cell_lines[cell_lines.index(cell_line)] for cell_line in removed_cell_lines
    ]


def plot_cell_line_comparison(
    cell_lines, cell_features, tpms, pca_dims, method, save_folder
):
    scaled_averaged_features, _, _ = scale(cell_features)
    scaled_tpms, _, _ = scale(tpms)

    if pca_dims is not None:
        scaled_tpms = decomposition.PCA(
            n_components=pca_dims, svd_solver="full"
        ).fit_transform(scaled_tpms)

        scaled_averaged_features = decomposition.PCA(
            n_components=pca_dims, svd_solver="full"
        ).fit_transform(scaled_averaged_features)

    reduced_rna_matrix = 1 - squareform(pdist(scaled_tpms, metric="cosine"))
    reduced_averaged_features = 1 - squareform(
        pdist(scaled_averaged_features, metric="cosine")
    )

    ground_truth_Z = linkage(reduced_rna_matrix)
    dn = dendrogram(ground_truth_Z, labels=cell_lines)

    columns = dn["ivl"]
    plt.figure(figsize=(5, 5))
    plt.imshow(reduced_rna_matrix, cmap="Blues")
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    plt.savefig(f"{save_folder}rna_matrix.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.imshow(reduced_averaged_features, cmap="Blues")
    # plt.title(f"{method} similarity matrix")
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    plt.savefig(f"{save_folder}{method}_matrix.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    for i in range(len(reduced_averaged_features)):
        reduced_averaged_features[i, i] = 0
        reduced_rna_matrix[i, i] = 0

    mantel_test = mantel.test(
        squareform(reduced_averaged_features),
        squareform(reduced_rna_matrix),
        perms=1000000,
    )
    rnaseq_mantel_test = mantel_test[0]
    print(
        f"RNAseq hierarchy mantel test: {mantel_test}, dim: {scaled_averaged_features.shape[1]}"
    )

    ca = CCA(n_components=2)
    ca.fit(scaled_tpms, scaled_averaged_features)
    X_c, Y_c = ca.transform(scaled_tpms, scaled_averaged_features)

    cc_distance_matrix = cdist(X_c, Y_c, metric="euclidean")
    cc_df = pd.DataFrame(np.diag(cc_distance_matrix), index=cell_lines, columns=["cc"])
    cc_df = cc_df.sort_values("cc", ascending=True)

    avg_dist = np.mean(np.diag(cc_distance_matrix))
    print(f"Average distance: {avg_dist}")

    k = 1
    top_1_accuracy = np.mean(
        (
            np.tile(np.arange(cc_distance_matrix.shape[0])[:, np.newaxis], k)
            == np.argsort(cc_distance_matrix, axis=0)[0:k, :].T
        ).sum(axis=1)
    )

    return rnaseq_mantel_test, top_1_accuracy


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


def plot_location_comparison(
    protein_localizations, features, pca_dims, method, savedir
):
    loc_group_mapping = pd.read_csv("annotations/location_group_mapping.tsv", sep="\t")
    loc_group_mapping = loc_group_mapping[
        (loc_group_mapping["Original annotation"].isin(protein_localizations))
        & (~loc_group_mapping["Grouping 3"].isna())
    ].reset_index(drop=True)

    labels = loc_group_mapping["Original annotation"].tolist()
    features = np.array([features[protein_localizations.index(loc)] for loc in labels])

    scaled_features, features_mean, features_std = scale(features)

    if pca_dims is not None:
        scaled_features = decomposition.PCA(
            n_components=pca_dims, svd_solver="full"
        ).fit_transform(scaled_features)

    protein_distance_matrix = 1 - squareform(pdist(scaled_features, metric="cosine"))

    ground_truth_distance_matrix = np.zeros((len(labels), len(labels)))

    high_level_labels = loc_group_mapping["Grouping 3"].tolist()
    for i in range(len(labels)):
        for j in range(len(labels)):
            if high_level_labels[i] == high_level_labels[j]:
                ground_truth_distance_matrix[i, j] = 0.5

    low_level_labels = loc_group_mapping["Grouping 2"].tolist()
    for i in range(len(labels)):
        for j in range(len(labels)):
            if low_level_labels[i] == low_level_labels[j]:
                ground_truth_distance_matrix[i, j] = 1
    ground_truth_Z = linkage(ground_truth_distance_matrix)

    plt.figure()
    dn = dendrogram(ground_truth_Z, labels=labels)
    # plt.title("Protein ground truth hierarchy")
    plt.savefig(f"{savedir}HPA_Protein_ground_truth_hierarchy.pdf", bbox_inches="tight")
    plt.close()
    ground_truth_distance_matrix = ground_truth_distance_matrix[dn["leaves"], :][
        :, dn["leaves"]
    ]
    protein_distance_matrix = protein_distance_matrix[dn["leaves"], :][:, dn["leaves"]]

    high_level_groups = (
        pd.DataFrame([high_level_labels[i] for i in dn["leaves"]]).groupby(0).groups
    )
    high_level_groups = {
        k: [high_level_groups[k].min(), high_level_groups[k].max()]
        for k in high_level_groups.keys()
    }
    high_level_groups = dict(
        sorted(high_level_groups.items(), key=lambda item: item[1][0])
    )

    low_level_groups = (
        pd.DataFrame([low_level_labels[i] for i in dn["leaves"]]).groupby(0).groups
    )
    low_level_groups = {
        k: [low_level_groups[k].min(), low_level_groups[k].max()]
        for k in low_level_groups.keys()
    }

    columns = dn["ivl"]
    plt.figure(figsize=(5, 5), dpi=300)
    plt.imshow(ground_truth_distance_matrix, cmap="Blues")
    # plt.title("Location ground truth matrix")
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    print("")
    plt.savefig(f"{savedir}HPA_Protein_ground_truth.pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    plt.imshow(protein_distance_matrix, cmap="Blues")
    # plt.title("Protein Localization similarity matrix")
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    for i, r in enumerate(high_level_groups.keys()):
        min_index, max_index = high_level_groups[r]
        rect = patches.Rectangle(
            (min_index - 0.5, min_index - 0.5),
            max_index - min_index + 1,
            max_index - min_index + 1,
            linewidth=4,
            edgecolor="#32ff32",
            facecolor="none",
        )
        ax.add_patch(rect)
        text = ax.text(
            max_index + 1 if i < (len(high_level_groups) / 2 - 1) else min_index - 1,
            min_index + (max_index - min_index) / 2,
            r,
            ha="left" if i < (len(high_level_groups) / 2 - 1) else "right",
            va="center",
            color="#32ff32",
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor="black", edgecolor="none", alpha=0.1),
        )
        ax.add_artist(text)
    for r in low_level_groups.keys():
        min_index, max_index = low_level_groups[r]
        rect = patches.Rectangle(
            (min_index - 0.5, min_index - 0.5),
            max_index - min_index + 1,
            max_index - min_index + 1,
            linewidth=2,
            edgecolor="#ff0000",
            facecolor="none",
        )

        ax.add_patch(rect)

    plt.savefig(f"{savedir}{method}_HPA_location_similarity.pdf", bbox_inches="tight")
    plt.close()

    for i in range(len(protein_distance_matrix)):
        protein_distance_matrix[i, i] = 0
        ground_truth_distance_matrix[i, i] = 0

    mantel_test = mantel.test(
        squareform(protein_distance_matrix),
        squareform(ground_truth_distance_matrix),
        perms=1000000,
    )
    print(
        f"protein hierarchy mantel test: {mantel_test}, dim: {scaled_features.shape[1]}"
    )
    protein_hierarchy_mantel_test = mantel_test[0]
    return protein_hierarchy_mantel_test


if __name__ == "__main__":

    exp_folder = "/proj/hpa_subcell/allhpa_ablations_results/"

    exp_name_dir_dict = {
        "ViT-ProtS-Pool": "rybg_448_nc8_vitb16_pool_fp32_supcon",
        "MAE-CellS-ProtS-Pool": "rybg_448_nc8_contrast_mae_pool_vitb16_fp32_mr0.25_objmr0.0_supcon",
        "DINO4Cells-HPA": "/proj/hpa_subcell/allhpa_ablations_results/comp_exps/dino/hpa_features/",
        "ViT-Supervised": "rybg_448_nc8_vitb16_fp32_supervised",
        "bestfitting": "/proj/hpa_subcell/allhpa_ablations_results/comp_exps/bestfitting/hpa_features/",
    }

    crop_size = 640
    resize = -1

    pca_dim = 10

    save_folder = (
        "/proj/hpa_subcell/allhpa_ablations_results/subcell_results/cell_morphology/"
    )
    os.makedirs(save_folder, exist_ok=True)

    feature_path = f"ssl_model_ContrastiveLoss/analysis/best_model_ap/crop_{crop_size}_resize_{resize}/data/hpa_features/all_features.pth"

    all_methods = []
    all_pca_dims = []
    all_cell_mantel_tests = []
    all_top_1_accuracies = []
    all_prot_mantel_tests = []

    if os.path.isfile(f"{save_folder}rna_protein_cc_results.csv"):
        for i, (method, result_path) in enumerate(exp_name_dir_dict.items()):
            method_feature_path = (
                feature_path.replace("ssl_model_ContrastiveLoss/", "")
                if "CellS" not in method
                else feature_path
            )

            if method in ["bestfitting", "DINO4Cells-HPA"]:
                df, feature_data = torch.load(
                    f"{result_path}all_features.pth", map_location="cpu"
                )
            else:
                df, feature_data = (
                    torch.load(
                        f"{exp_folder}{result_path}/{method_feature_path}",
                        map_location="cpu",
                    )
                    if ".pth" not in result_path
                    else torch.load(result_path, map_location="cpu")
                )

            cell_lines, cell_features = get_cell_avg_features(df, feature_data)
            tpms, removed_cell_lines = get_tpms(cell_lines)

            cell_features = np.delete(
                cell_features,
                [cell_lines.index(cell) for cell in removed_cell_lines],
                axis=0,
            )
            cell_lines = np.delete(
                cell_lines, [cell_lines.index(cell) for cell in removed_cell_lines]
            )

            rnaseq_mantel_test, top_1_accuracy = plot_cell_line_comparison(
                cell_lines=cell_lines,
                cell_features=cell_features,
                tpms=tpms,
                pca_dims=pca_dim,
                method=method,
                save_folder=save_folder,
            )
            all_methods.append(method)
            all_pca_dims.append(pca_dim)
            all_cell_mantel_tests.append(rnaseq_mantel_test)
            all_top_1_accuracies.append(top_1_accuracy)

            locs, loc_avg_features = get_localization_avg_features(df, feature_data)
            prot_mantel_test = plot_location_comparison(
                protein_localizations=locs,
                features=loc_avg_features,
                pca_dims=pca_dim,
                method=method,
                savedir=save_folder,
            )
            all_prot_mantel_tests.append(prot_mantel_test)

        df = pd.DataFrame(
            {
                "Method": all_methods,
                "PCA Dim": all_pca_dims,
                "RNAseq Mantel Statistic": all_cell_mantel_tests,
                "Top-1 Accuracy": all_top_1_accuracies,
                "Localization Mantel Statistic": all_prot_mantel_tests,
            }
        )
        df.to_csv(f"{save_folder}rna_protein_cc_results.csv", index=False)
    else:
        df = pd.read_csv(f"{save_folder}rna_protein_cc_results.csv")

    df1 = df.rename(
        columns={
            "RNAseq Mantel Statistic": "Mantel Statistic",
        }
    )

    sns.set(font_scale=2)
    with sns.axes_style("ticks"):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax = sns.barplot(
            x="Method",
            y="Mantel Statistic",
            data=df1,
            hue="Method",
            palette=sns.color_palette("tab10", n_colors=len(df1)),
            # cc.glasbey_light,
            ax=ax,
        )
        for elem in ax.containers:
            ax.bar_label(
                elem,
                fmt="%.3f",
                label_type="edge",
                fontsize=26,
                padding=-30,
                # color="white",
            )
        ax.set_box_aspect(0.5)
        plt.xticks(rotation=45, ha="right")
        # plt.title("RNAseq")
        plt.savefig(f"{save_folder}rna_cc_results.pdf", bbox_inches="tight", dpi=300)
        plt.close()

        df2 = df.rename(
            columns={
                "Localization Mantel Statistic": "Mantel Statistic",
            }
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax = sns.barplot(
            x="Method",
            y="Mantel Statistic",
            data=df2,
            hue="Method",
            palette=sns.color_palette("tab10", n_colors=len(df2)),
            # cc.glasbey_light,
            ax=ax,
        )
        for elem in ax.containers:
            ax.bar_label(
                elem,
                fmt="%.3f",
                label_type="edge",
                fontsize=26,
                padding=-30,
                # color="white",
            )
        ax.set_box_aspect(0.5)
        plt.xticks(rotation=45, ha="right")
        # plt.title("Protein Localization")
        plt.savefig(
            f"{save_folder}protein_cc_results.pdf", bbox_inches="tight", dpi=300
        )
        plt.close()
