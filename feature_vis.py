import argparse
import os

import colorcet as cc
import cuml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from harmony import harmonize
from sklearn import decomposition
from tqdm import tqdm

plt.switch_backend("agg")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def add_basename(df):
    df["basename"] = (
        df["if_plate_id"].astype(str)
        + "_"
        + df["position"].astype(str)
        + "_"
        + df["sample"].astype(str)
    )


def get_fov_features(df, pca_in_features):
    fov_features = (
        pd.DataFrame(
            pd.concat(
                (
                    pd.DataFrame(
                        pca_in_features,
                        columns=[f"f_{i}" for i in range(len(pca_in_features[0]))],
                    ),
                    df["basename"],
                ),
                axis=1,
            )
        )
        .groupby("basename")
        .mean()
        .values
    )
    return fov_features


def get_umap_embeddings(features_to_fit, in_features):
    reducer = cuml.UMAP(
        init="spectral",
        metric="euclidean",
        min_dist=0.1,
        n_neighbors=15,
        n_components=2,
        spread=4,
        n_epochs=2600,
        transform_queue_size=20,
        # random_state=42,
        verbose=True,
        n_jobs=-1,
    )
    reducer = reducer.fit(features_to_fit)

    in_features_split = np.array_split(in_features, 10)
    embed_features = []
    for in_features_split_i in in_features_split:
        embed_features.append(reducer.transform(in_features_split_i))
    embed_features = np.concatenate(embed_features, axis=0)

    # embed_features = reducer.transform(in_features[:50000])
    return embed_features


def plot_cell_line_umap(df, filename):
    df = df.rename(columns={"atlas_name": "Cell Line"})

    fig, ax = plt.subplots(1, figsize=(16, 12))
    sns.scatterplot(
        x="UMAP 1",
        y="UMAP 2",
        hue="Cell Line",
        palette=sns.color_palette(cc.glasbey_dark, len(df["Cell Line"].unique())),
        legend="brief",
        s=1,
        alpha=0.5,
        data=df,
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles:
        handle.set_alpha(1)
    lgd = ax.legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
        markerscale=5,
    )
    plt.savefig(filename, dpi=1000, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()


def plot_category_umap(df, filename):
    fig, ax = plt.subplots(1, figsize=(16, 12))
    df = df.sort_values(by="Category")
    sns.scatterplot(
        x="UMAP 1",
        y="UMAP 2",
        hue="Category",
        palette=sns.color_palette(cc.glasbey_dark, len(df["Category"].unique())),
        legend="brief",
        s=1 if len(df) > 20000 else 5,
        alpha=0.5 if len(df) > 20000 else 0.8,
        data=df,
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles:
        handle.set_alpha(1)
    lgd = ax.legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
        markerscale=5,
    )
    plt.savefig(filename, dpi=1000, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()


def get_per_gene_features(df, feature_data):
    cells_per_gene = df.groupby(["gene_names"]).groups
    keys_list = [k.split(",") for k in cells_per_gene.keys()]
    all_keys = [item for sublist in keys_list for item in sublist]
    all_genes = sorted(list(set(all_keys)))

    new_dict = {k: [] for k in all_genes}
    for k, v in cells_per_gene.items():
        k_split = k.split(",")
        if len(k_split) > 1:
            for k_s in k_split:
                new_dict[k_s].extend(v.tolist())
        else:
            new_dict[k].extend(v.tolist())

    locations = []
    features = []
    genes = []
    for gene, cell_idxs in tqdm(new_dict.items()):
        loc = df.loc[cell_idxs, "Category"].unique()
        loc = loc[0] if len(loc) == 1 else "Multilocalizing"
        locations.append(loc)
        genes.append(gene)
        features.append(feature_data[cell_idxs].mean(axis=0))

    df = pd.DataFrame({"gene": genes, "Category": locations})
    features = torch.stack(features)
    return df, features


if __name__ == "__main__":
    features_folder = "/proj/hpa_subcell/hpa_data/inference/dino_640"

    save_folder = f"{features_folder}/dim_red"
    os.makedirs(save_folder, exist_ok=True)
    plots_folder = f"{save_folder}/plots"
    os.makedirs(plots_folder, exist_ok=True)

    df, feature_data = torch.load(
        f"{features_folder}/all_features.pth", map_location="cpu"
    )
    add_basename(df)

    df.loc[df["locations"].isna(), "locations"] = "Negative"

    locations_list = df["locations"].str.split(",").tolist()
    df["Category"] = [
        x[0] if len(x) == 1 else "Multilocalizing" for x in locations_list
    ]

    if os.path.isfile(f"{save_folder}/umap_df.csv") and os.path.isfile(
        f"{save_folder}/harmonized_umap_df.csv"
    ):
        umap_df = pd.read_csv(f"{save_folder}/umap_df.csv")
        if "[" in umap_df["Category"][0]:
            umap_df["Category"] = umap_df["Category"].apply(
                lambda x: x.replace("[", "").replace("]", "").replace("'", "")
            )
        harmonized_umap_df = pd.read_csv(f"{save_folder}/harmonized_umap_df.csv")
        if "[" in harmonized_umap_df["Category"][0]:
            harmonized_umap_df["Category"] = harmonized_umap_df["Category"].apply(
                lambda x: x.replace("[", "").replace("]", "").replace("'", "")
            )
    else:
        pca = decomposition.PCA(n_components=256, random_state=42)
        pca_in_features = pca.fit_transform(feature_data)
        print(f"Explained variance: {pca.explained_variance_ratio_.sum()}")

        fov_features = get_fov_features(df, pca_in_features)
        umap_features = get_umap_embeddings(fov_features, pca_in_features)

        umap_df = df.copy()
        umap_df[["UMAP 1", "UMAP 2"]] = umap_features
        umap_df.to_csv(f"{save_folder}/umap_df.csv", index=False)

        vars_use = ["atlas_name"]
        harmonized_feat = harmonize(
            pca_in_features,
            df,
            batch_key=vars_use,
            use_gpu=True,
            verbose=True,
            random_state=42,
        )
        harmonized_fov_features = get_fov_features(df, harmonized_feat)
        harmonized_umap_features = get_umap_embeddings(
            harmonized_fov_features, harmonized_feat
        )
        harmonized_umap_df = df.copy()
        harmonized_umap_df[["UMAP 1", "UMAP 2"]] = harmonized_umap_features
        harmonized_umap_df.to_csv(f"{save_folder}/harmonized_umap_df.csv", index=False)

    plot_cell_line_umap(umap_df, f"{plots_folder}/umap_cell_lines.png")
    plot_category_umap(umap_df, f"{plots_folder}/umap_w_multiloc_categories.png")
    umap_df = umap_df[umap_df["Category"] != "Multilocalizing"]
    plot_category_umap(umap_df, f"{plots_folder}/umap_wo_multiloc_categories.png")

    plot_cell_line_umap(
        harmonized_umap_df, f"{plots_folder}/harmonized_umap_cell_lines.png"
    )
    plot_category_umap(
        harmonized_umap_df, f"{plots_folder}/harmonized_umap_w_multiloc_categories.png"
    )
    harmonized_umap_df = harmonized_umap_df[
        harmonized_umap_df["Category"] != "Multilocalizing"
    ]
    plot_category_umap(
        harmonized_umap_df, f"{plots_folder}/harmonized_umap_wo_multiloc_categories.png"
    )

    if not os.path.isfile(f"{save_folder}/gene_umap_df.csv"):
        gene_umap_df, avg_features = get_per_gene_features(df, feature_data)
        umap_features = get_umap_embeddings(avg_features, avg_features)
        gene_umap_df[["UMAP 1", "UMAP 2"]] = umap_features
        gene_umap_df.to_csv(f"{save_folder}/gene_umap_df.csv", index=False)
    else:
        gene_umap_df = pd.read_csv(f"{save_folder}/gene_umap_df.csv")

    plot_category_umap(gene_umap_df, f"{plots_folder}/gene_umap_w_multiloc_categories")
    gene_umap_df = gene_umap_df[gene_umap_df["Category"] != "Multilocalizing"]
    plot_category_umap(gene_umap_df, f"{plots_folder}/gene_umap_wo_multiloc_categories")
