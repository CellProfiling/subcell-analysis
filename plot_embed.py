import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import wandb
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
import colorcet as cc

EXP_NAME_DIR_DF = pd.read_csv("annotations/features_path.csv")
EXP_NAME_DIR_DF = EXP_NAME_DIR_DF[EXP_NAME_DIR_DF["Cell_Line"] == "all"].reset_index(
    drop=True
)

if __name__ == "__main__":
    exp_folder = "/scratch/groups/emmalu/subcell_ankit/features"

    save_folder = "/scratch/groups/emmalu/subcell_ankit/feature_plot"
    os.makedirs(save_folder, exist_ok=True)

    microscope_df = pd.read_csv(
        "/scratch/groups/emmalu/subcell_ankit/features/hpa_microscope_meta.csv"
    )
    microscope_df["plate_position_sample"] = microscope_df.apply(
        lambda x: f"{x['if_plate_id']}_{x['position']}_{x['sample']}", axis=1
    )

    for i, row in EXP_NAME_DIR_DF.iterrows():
        method = row["Method"]
        harmonized = row["Harmonized"]

        if not os.path.exists(f"{save_folder}/{method}_{harmonized}_umap.csv"):
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
            df.loc[:, "Location"] = df["Location"].apply(
                lambda x: "Multi-localized" if "," in x else x
            )
            df["plate_position_sample"] = df.apply(
                lambda x: f"{x['if_plate_id']}_{x['position']}_{x['sample']}", axis=1
            )
            df.loc[:, [f"feat_{i}" for i in range(feature_data.shape[1])]] = (
                feature_data
            )
            sample_df = (
                df.groupby("plate_position_sample")[
                    [f"feat_{i}" for i in range(feature_data.shape[1])]
                ]
                .mean()
                .reset_index()
            )

            sample_df = pd.merge(
                sample_df,
                df.groupby("plate_position_sample")["Location"].first().reset_index(),
                on="plate_position_sample",
                how="left",
            )
            sample_df = pd.merge(
                sample_df,
                df.groupby("plate_position_sample")["atlas_name"].first().reset_index(),
                on="plate_position_sample",
                how="left",
            )
            sample_df = pd.merge(
                sample_df,
                microscope_df[["plate_position_sample", "microscope"]],
                on="plate_position_sample",
                how="left",
            )
            sample_df = sample_df[~sample_df["microscope"].isna()].reset_index()

            umap_feats = UMAP(
                n_components=2,
                metric="cosine",
                n_neighbors=20,
                min_dist=0.1,
                random_state=42,
                verbose=1,
                n_jobs=-1,
            ).fit_transform(
                sample_df[[f"feat_{i}" for i in range(feature_data.shape[1])]].values
            )

            umap_df = pd.DataFrame(
                umap_feats, columns=["UMAP1", "UMAP2"], index=sample_df.index.tolist()
            )
            umap_df["Location"] = sample_df["Location"]
            umap_df["microscope"] = sample_df["microscope"]
            umap_df["atlas_name"] = sample_df["atlas_name"]

            umap_df.to_csv(f"{save_folder}/{method}_{harmonized}_umap.csv", index=True)
        else:
            umap_df = pd.read_csv(
                f"{save_folder}/{method}_{harmonized}_umap.csv", index_col=0
            )

        for category in ["Location", "microscope", "atlas_name"]:
            fig, ax = plt.subplots(figsize=(10, 8))
            if category == "Location":
                sns.scatterplot(
                    x="UMAP1",
                    y="UMAP2",
                    c="grey",
                    s=1,
                    alpha=0.3,
                    data=umap_df[umap_df["Location"] == "Multi-localized"],
                    ax=ax,
                    label="Multi-localized",
                )
                umap_df = umap_df[umap_df["Location"] != "Multi-localized"]
            sns.scatterplot(
                x="UMAP1",
                y="UMAP2",
                hue=category,
                palette=cc.glasbey_dark,
                s=3,
                alpha=0.6,
                data=umap_df,
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
                markerscale=3,
            )
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.savefig(
                f"{save_folder}/{method}_{harmonized}_{category}.png",
                dpi=500,
                bbox_inches="tight",
            )
            plt.close()
