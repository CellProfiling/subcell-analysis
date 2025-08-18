import argparse
import os

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
import yaml
from harmony import harmonize
from sklearn import decomposition
from sklearn.metrics import mutual_info_score
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from utils.feature_analysis import scale
import matplotlib.ticker as mtick

EXP_NAME_DIR_DF = pd.read_csv("annotations/features_path.csv")
EXP_NAME_DIR_DF = EXP_NAME_DIR_DF[EXP_NAME_DIR_DF["Cell_Line"] == "all"].reset_index(
    drop=True
)

HARMONIZES = [
    "none",
    "microscope",
    "atlas_name",
    "if_plate_id",
    "if_plate_id_microscope",
    "if_plate_id_microscope_atlas_name",
]
BENCHMARK_MODELS = ["bestfitting"]


def create_bins(x):
    unique_xs = sorted(np.unique(x))
    bins = [unique_xs[0] - 0.5]
    for ind in range(len(unique_xs) - 1):
        bins.append(unique_xs[ind] + (unique_xs[ind + 1] - unique_xs[ind]) / 2)
    bins = bins + [unique_xs[-1] + 0.5]
    return bins


def mutual_info(x, y, code_bin, factor_bin):
    c_xy = np.histogram2d(x, y, (code_bin, factor_bin))[0]
    mi = 0
    N = np.sum(c_xy)
    for i in range(c_xy.shape[0]):
        for j in range(c_xy.shape[1]):
            p_i = np.sum(c_xy[i, :]) / N
            p_j = np.sum(c_xy[:, j]) / N
            p_ij = c_xy[i, j] / N
            if p_ij == 0:
                mi += 0
            else:
                mi += (p_ij) * np.log(p_ij / (p_i * p_j))
    return mi


def conditional_mutual_info(x, y, z, bins):
    c_xyz = np.histogramdd((x, y, z), bins)[0]
    N = np.sum(c_xyz)
    mi = 0
    p_z = c_xyz.sum(axis=(0, 1)) / N
    p_iz = c_xyz.sum(axis=(1)) / N
    p_jz = c_xyz.sum(axis=(0)) / N
    p_ijzs = c_xyz / N
    N = np.sum(c_xyz)
    for i in range(c_xyz.shape[0]):
        for j in range(c_xyz.shape[1]):
            for z in range(c_xyz.shape[2]):
                if p_ijzs[i, j, z] == 0:
                    mi += 0
                else:
                    mi += (p_ijzs[i, j, z]) * np.log(
                        (p_z[z] * p_ijzs[i, j, z]) / (p_iz[i, z] * p_jz[j, z])
                    )
    return mi


def MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


if __name__ == "__main__":
    exp_folder = "/scratch/groups/emmalu/subcell_ankit/features"

    save_folder = "/scratch/groups/emmalu/subcell_ankit/mutual_info"
    os.makedirs(save_folder, exist_ok=True)

    plots_folder = f"{save_folder}/plots"
    os.makedirs(plots_folder, exist_ok=True)

    code_bin = 50
    factor_bin = 50

    methods = []
    harmonizes = []
    mi_matrices = []
    for i, row in EXP_NAME_DIR_DF.iterrows():
        method = row["Method"]
        harmonized = row["Harmonized"]

        if not os.path.exists(f"{save_folder}/{method}_{harmonized}_mi_df.csv"):
            method_feat_path = (
                f"{exp_folder}/{row['Method']}/hpa_features/all_features.pth"
            )
            method_feat_path = (
                method_feat_path.replace(
                    "all_features", f"harmonized_features_{row['Harmonized']}"
                )
                if row["Harmonized"] != "none"
                else method_feat_path
            )
            print(f"Processing {row['Method']} and {row['Harmonized']}")

            df, feature_data = torch.load(
                method_feat_path, map_location="cpu", weights_only=False
            )

            df.loc[:, "Location"] = df["locations"].fillna("Negative")
            single_loc_idx = df["Location"].apply(lambda x: False if "," in x else True)

            df = df[single_loc_idx].reset_index(drop=True)
            feature_data = feature_data[single_loc_idx]

            feature_data, _, _ = scale(feature_data)

            lb = LabelEncoder()
            cell_line_mat = lb.fit_transform(df["atlas_name"].values)
            cell_line_unique = lb.classes_

            df["well_ids"] = (
                df["if_plate_id"].astype(str) + "_" + df["position"].astype(str)
            )
            lb = LabelEncoder()
            well_id_mat = lb.fit_transform(df["well_ids"].values)
            well_id_unique = lb.classes_

            lb = LabelEncoder()
            loc_mat = lb.fit_transform(df["Location"].values)
            loc_unique = lb.classes_

            factors = np.stack([loc_mat, cell_line_mat, well_id_mat], axis=1)

            whole_HPA_location_MI = []
            whole_HPA_cell_MI = []
            whole_HPA_well_MI = []

            for i in tqdm(range(feature_data.shape[1])):
                location_mi = mutual_info(
                    feature_data[:, i], factors[:, 0], code_bin, factor_bin
                )
                whole_HPA_location_MI.append(location_mi)
                cell_mi = mutual_info(
                    feature_data[:, i], factors[:, 1], code_bin, factor_bin
                )
                whole_HPA_cell_MI.append(cell_mi)
                well_mi = mutual_info(
                    feature_data[:, i], factors[:, 2], code_bin, factor_bin
                )
                whole_HPA_well_MI.append(well_mi)
                print(
                    f"Location MI: {location_mi}, Cell MI: {cell_mi}, Well MI: {well_mi}"
                )

            mi_df = pd.DataFrame(
                {
                    "location_mi": whole_HPA_location_MI,
                    "cell_mi": whole_HPA_cell_MI,
                    "well_mi": whole_HPA_well_MI,
                }
            )
            mi_df.to_csv(f"{save_folder}/{method}_{harmonized}_mi_df.csv", index=False)
        else:
            mi_df = pd.read_csv(f"{save_folder}/{method}_{harmonized}_mi_df.csv")

        methods.append(method)
        harmonizes.append(harmonized)
        mi_matrices.append(mi_df.values.T)

    dominant_fov_percentages = []
    for matrix_ind, (mi_matrix, mi_label) in enumerate(zip(mi_matrices, methods)):
        whole_HPA_protein_MI = mi_matrix[0, :]
        whole_HPA_cell_MI = mi_matrix[1, :]
        whole_HPA_well_MI = mi_matrix[2, :]
        high_protein = np.where(
            (whole_HPA_protein_MI > whole_HPA_cell_MI)
            & (whole_HPA_protein_MI > whole_HPA_well_MI)
        )[0]
        high_protein = high_protein[np.argsort(whole_HPA_protein_MI[high_protein])][
            ::-1
        ]
        high_cell = np.where(
            (whole_HPA_cell_MI > whole_HPA_protein_MI)
            & (whole_HPA_cell_MI > whole_HPA_well_MI)
        )[0]
        high_cell = high_cell[np.argsort(whole_HPA_cell_MI[high_cell])][::-1]
        high_well = np.where(
            (whole_HPA_well_MI > whole_HPA_protein_MI)
            & (whole_HPA_well_MI > whole_HPA_cell_MI)
        )[0]
        high_well = high_well[np.argsort(whole_HPA_well_MI[high_well])][::-1]
        new_indices = np.concatenate((high_protein, high_cell, high_well))
        dominant_fov_percentage = [
            len(high_protein) / len(new_indices),
            len(high_cell) / len(new_indices),
            len(high_well) / len(new_indices),
        ]
        dominant_fov_percentages.append(dominant_fov_percentage)

    dominant_fov_percentages = np.array(dominant_fov_percentages) * 100

    methods2plot = ["MAE-CellS-ProtS-Pool", "ViT-ProtS-Pool", "DINO4Cells-HPA"]

    for method in methods2plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=300)
        method_idxs = np.where((np.array(methods) == method))[0]

        plot_fov_percentage = dominant_fov_percentages[
            (np.where((np.array(methods) == method))[0])
        ]
        plot_harmonizes = [harmonizes[i] for i in method_idxs]
        reordered_idx = [plot_harmonizes.index(harmonize) for harmonize in HARMONIZES]
        plot_fov_percentage = plot_fov_percentage[reordered_idx]

        plt.bar(
            range(len(plot_fov_percentage)),
            bottom=0,
            height=plot_fov_percentage[:, 0],
            label="Protein",
        )
        plt.bar(
            range(len(plot_fov_percentage)),
            bottom=plot_fov_percentage[:, :1].sum(axis=1),
            height=plot_fov_percentage[:, 1],
            label="Cell line",
        )
        plt.bar(
            range(len(plot_fov_percentage)),
            bottom=plot_fov_percentage[:, :2].sum(axis=1),
            height=plot_fov_percentage[:, 2],
            label="Wells",
        )
        plt.legend(bbox_to_anchor=(1.75, 0.5), loc="center right", frameon=False)
        plt.ylabel("% encoding features")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.xticks(range(len(HARMONIZES)), HARMONIZES, rotation=90)
        plt.savefig(f"{plots_folder}/{method}.png", bbox_inches="tight", dpi=300)
        plt.close()
