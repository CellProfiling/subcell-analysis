import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from gprofiler import GProfiler
from scipy.stats import median_abs_deviation
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


from utils.plot_enriched import plot_enrich

UNIQUE_CATS = list(
    pd.read_csv("annotations/location_group_mapping.csv")[
        "Original annotation"
    ].unique()
)
UNIQUE_CATS.append("Negative")
cats_ordered = [
    "Nuclear membrane",
    "Nucleoli",
    "Nucleoli fibrillar center",
    "Nucleoli rim",
    "Nucleoplasm",
    "Nuclear bodies",
    "Nuclear speckles",
    "Actin filaments",
    "Focal adhesion sites",
    "Centrosome",
    "Centriolar satellite",
    "Cytosol",
    "Cytoplasmic bodies",
    "Intermediate filaments",
    "Microtubules",
    "Mitochondria",
    "Endoplasmic reticulum",
    "Golgi apparatus",
    "Plasma membrane",
    "Cell Junctions",
    "Vesicles",
    "Lipid droplets",
    "Peroxisomes",
    "Negative",
]


def add_basename(df):
    df["basename"] = (
        df["if_plate_id"].astype(str)
        + "_"
        + df["position"].astype(str)
        + "_"
        + df["sample"].astype(str)
    )
    return df


def add_loc_cols(df):
    df["locations"] = df["locations"].fillna("Negative")
    locations_list = df["locations"].str.split(",").tolist()
    labels_onehot = np.array(
        [[1 if cat in x else 0 for cat in UNIQUE_CATS] for x in locations_list]
    )
    df[UNIQUE_CATS] = labels_onehot
    return df


def plot_umap(df, feat_cols, save_folder):
    umap_ = umap.UMAP(
        n_components=2, metric="cosine", min_dist=0.2, n_neighbors=20, n_jobs=-1
    )
    umap_res = umap_.fit_transform(df[feat_cols])
    df["UMAP1"] = umap_res[:, 0]
    df["UMAP2"] = umap_res[:, 1]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue="locations",
        data=df,
        palette=cc.glasbey_light,
        s=6,
        alpha=0.5,
        ax=ax,
    )
    ax.set_box_aspect(1)
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles:
        handle.set_alpha(1)
    lgd = ax.legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(1.01, 1),
        loc=2,
        borderaxespad=0.0,
        markerscale=3,
    )
    plt.savefig(f"{save_folder}/umap_locations.pdf", dpi=300, bbox_inches="tight")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue="atlas_name",
        data=df,
        palette=cc.glasbey_light,
        s=6,
        alpha=0.5,
        ax=ax,
    )
    ax.set_box_aspect(1)
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles:
        handle.set_alpha(1)
    lgd = ax.legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(1.01, 1),
        loc=2,
        borderaxespad=0.0,
        markerscale=3,
    )
    plt.savefig(f"{save_folder}/umap_atlas.pdf", dpi=300, bbox_inches="tight")


def plot_channel_clustermap(df, channel_names, save_folder):
    protein_localizations = df["locations"].unique()
    loc_group_mapping = pd.read_csv("annotations/location_group_mapping.tsv", sep="\t")
    loc_group_mapping = loc_group_mapping[
        (loc_group_mapping["Original annotation"].isin(protein_localizations))
        & (~loc_group_mapping["Grouping 3"].isna())
    ].reset_index(drop=True)
    df = df[df["locations"].isin(loc_group_mapping["Original annotation"])]
    df["Location Group"] = df["locations"].map(
        dict(
            zip(
                loc_group_mapping["Original annotation"],
                loc_group_mapping["Grouping 3"],
            )
        )
    )
    df.index = df["locations"]

    for channel_name in channel_names:
        channel_name_cols = [col for col in df.columns if f"{channel_name}-" in col]
        channel_df = df[channel_name_cols]
        channel_df = channel_df.rename(
            columns={col: col.split("-")[1] for col in channel_df.columns}
        )
        lut = dict(zip(df["Location Group"].unique(), cc.glasbey_light))
        row_colors = df["Location Group"].map(lut)
        sns.clustermap(
            channel_df,
            cmap="Blues",
            row_colors=row_colors,
            # center=0,
            metric="cosine",
        )
        plt.savefig(f"{save_folder}/clustermap_{channel_name}.pdf", dpi=300)


if __name__ == "__main__":
    exp_folder = "/proj/hpa_subcell/allhpa_ablations_results/"

    exp_name_dir_dict = {
        "ViT-ProtS-Pool": "rybg_448_nc8_vitb16_pool_fp32_supcon",
        "MAE-CellS-ProtS-Pool": "rybg_448_nc8_contrast_mae_pool_vitb16_fp32_mr0.25_objmr0.0_supcon",
    }

    crop_size = 640
    resize = -1

    save_folder = (
        "/proj/hpa_subcell/allhpa_ablations_results/subcell_results/attention_analysis/"
    )
    os.makedirs(save_folder, exist_ok=True)

    n_channels = 4
    channel_names = ["MT", "ER", "Nuclei", "Protein"]
    n_transformer_attn_channels = 12
    n_pool_attn_channels = 2
    channel_cols = channel_names
    attn_cols = (
        channel_names
        + [f"AH {i+1}" for i in range(n_transformer_attn_channels)]
        + [f"PAH {i+1}" for i in range(n_pool_attn_channels)]
    )

    feature_path = f"ssl_model_ContrastiveLoss/analysis/best_model_ap/crop_{crop_size}_resize_{resize}/data/hpa_features"

    gene_info_df = pd.read_csv("annotations/subcellular_location.tsv", sep="\t")

    for i, (method, result_path) in enumerate(exp_name_dir_dict.items()):
        method_feature_path = (
            feature_path.replace("ssl_model_ContrastiveLoss/", "")
            if "CellS" not in method
            else feature_path
        )

        df, corr_mat = torch.load(
            f"{exp_folder}/{result_path}/{method_feature_path}/all_corr.pth",
            map_location="cpu",
        )

        exp_save_folder = f"{save_folder}/{method}"
        os.makedirs(exp_save_folder, exist_ok=True)

        nan_corr_idxs = np.isnan(corr_mat.numpy()).any(axis=(1, 2))
        corr_mat = corr_mat[~nan_corr_idxs]
        df = df[~nan_corr_idxs].reset_index(drop=True)

        nan_loc_idx = df["locations"].isna()
        df = df[~nan_loc_idx].reset_index(drop=True)
        corr_mat = corr_mat[~nan_loc_idx]

        reliable_genes = gene_info_df[
            gene_info_df["Reliability"].isin(["Enhanced", "Supported"])
        ]["Gene"].tolist()

        keep_genes = gene_info_df[
            gene_info_df["Single-cell variation intensity"].isna()
            & gene_info_df["Single-cell variation spatial"].isna()
            & gene_info_df["Cell cycle dependency"].isna()
            & gene_info_df["Reliability"].isin(["Enhanced", "Supported"])
        ]["Gene"].tolist()

        keep_idxs = df["ensembl_ids"].apply(
            lambda x: any([gene in x for gene in keep_genes])
        )
        print(f"Keep genes: {keep_idxs.sum()}")
        df = df[keep_idxs].reset_index(drop=True)
        corr_mat = corr_mat[keep_idxs]

        attn_cols_flatten = [
            f"{c}-{attn}" for c in channel_cols for attn in attn_cols[4:]
        ]
        df = add_loc_cols(df)
        corr = corr_mat[:, :n_channels, n_channels:].reshape(len(df), -1)

        loc_attn_df = df.copy()
        loc_attn_df[attn_cols_flatten] = corr
        loc_attn_df = loc_attn_df.groupby("locations")[attn_cols_flatten].mean()
        loc_attn_df["locations"] = loc_attn_df.index
        plot_channel_clustermap(loc_attn_df, channel_names, exp_save_folder)

        df = add_basename(df)
        df_fov_attn = df.copy()
        df_fov_attn[attn_cols_flatten] = corr
        df_fov_attn = df_fov_attn.groupby("basename")[attn_cols_flatten].mean()
        df_fov_attn["locations"] = df.groupby("basename")["locations"].first()
        df_fov_attn["atlas_name"] = df.groupby("basename")["atlas_name"].first()
        df_fov_attn["locations"] = df_fov_attn["locations"].apply(
            lambda x: x if "," not in x else "Multilocalizing"
        )
        plot_umap(df_fov_attn, attn_cols_flatten, exp_save_folder)
