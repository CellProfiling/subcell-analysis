import argparse
import glob
import os
import random
import shutil
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from harmony import harmonize
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from umap import UMAP

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.dataset import AllenCellDataset, collate_fn
from utils.load_model import get_dino_model, get_subcell_model
from utils.preprocess import (
    preprocess_input_bestfitting,
    preprocess_input_dino,
    preprocess_input_subcell,
)
from utils.train_mlp import eval_model, get_metrics, plot_umap, train_mlp


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def split_and_save_indices(
    train_idx,
    dino_df,
    train_indices_path,
    val_indices_path,
    test_size=0.2,
    random_state=42,
):

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=test_size,
        random_state=random_state,
        stratify=dino_df.iloc[train_idx]["cell_stage"].values,
    )
    torch.save(train_idx, train_indices_path)
    torch.save(val_idx, val_indices_path)


def plot_harmonized_umap(df, features, save_folder):
    Z = harmonize(features.numpy(), df, batch_key="cell_line", use_gpu=True)

    umap_feat = UMAP().fit_transform(Z)
    umap_df = pd.DataFrame(
        {"x": umap_feat[:, 0], "y": umap_feat[:, 1], "Cell Stage": df["cell_stage"]}
    )

    fig, ax = plt.subplots(1, figsize=(16, 12))
    sns.scatterplot(
        x="x",
        y="y",
        hue="Cell Stage",
        s=1,
        alpha=0.5,
        palette="flare",
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
        markerscale=5,
    )
    plt.savefig(
        f"{save_folder}/harmonized_umap.png",
        dpi=1000,
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    exp_name_dir_dict = {
        "DINO4Cells-WTC": "/scratch/groups/emmalu/subcell_ankit/allencell/dino/DINO_features_and_df.pth",
        "DINO-ImageNet": "/scratch/groups/emmalu/subcell_ankit/allencell/pretrained/pretrained_features_and_df.pth",
        "MAE-DNA-Struct-PerCh": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_bg_mae_channel_comb_nuc_prot_only_norm_perchannel/all_features.pth",
        "MAE-DNA-Struct-PerIm": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_bg_mae_channel_comb_nuc_prot_only_norm_perimage/all_features.pth",
        "MAE-DNA-Struct-Plasma-Concat-PerCh": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_bg_mae_channel_comb_concat_norm_perchannel/all_features.pth",
        "MAE-DNA-Struct-Plasma-Concat-PerIm": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_bg_mae_channel_comb_concat_norm_perimage/all_features.pth",
        "MAE-DNA-Struct-Plasma-PerCh": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_rbg_mae_channel_comb_plasma_nuc_prot_norm_perchannel/all_features.pth",
        "MAE-DNA-Struct-Plasma-PerIm": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_rbg_mae_channel_comb_plasma_nuc_prot_norm_perimage/all_features.pth",
        "ViT-DNA-Struct-PerCh": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_bg_vit_channel_comb_nuc_prot_only_norm_perchannel/all_features.pth",
        "ViT-DNA-Struct-PerIm": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_bg_vit_channel_comb_nuc_prot_only_norm_perimage/all_features.pth",
        "ViT-DNA-Struct-Plasma-Concat-PerCh": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_bg_vit_channel_comb_concat_norm_perchannel/all_features.pth",
        "ViT-DNA-Struct-Plasma-Concat-PerIm": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_bg_vit_channel_comb_concat_norm_perimage/all_features.pth",
        "ViT-DNA-Struct-Plasma-PerCh": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_rbg_vit_channel_comb_plasma_nuc_prot_norm_perchannel/all_features.pth",
        "ViT-DNA-Struct-Plasma-PerIm": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_rbg_vit_channel_comb_plasma_nuc_prot_norm_perimage/all_features.pth",
    }

    dino_feats, dino_df = torch.load(
        "/scratch/groups/emmalu/subcell_ankit/allencell/dino/DINO_features_and_df.pth",
        weights_only=False,
    )

    train_idx = torch.load(
        "/scratch/groups/emmalu/subcell_ankit/allencell/train_indices.pth",
        weights_only=False,
    )

    split_and_save_indices(
        train_idx,
        dino_df,
        "/scratch/groups/emmalu/subcell_ankit/allencell/train_indices1.pth",
        "/scratch/groups/emmalu/subcell_ankit/allencell/val_indices1.pth",
    )

    train_idx = torch.load(
        "/scratch/groups/emmalu/subcell_ankit/allencell/train_indices1.pth",
        weights_only=False,
    )

    val_idx = torch.load(
        "/scratch/groups/emmalu/subcell_ankit/allencell/val_indices1.pth",
        weights_only=False,
    )

    test_idx = torch.load(
        "/scratch/groups/emmalu/subcell_ankit/allencell/test_indices.pth",
        weights_only=False,
    )

    train_cell_ids = dino_df["CellId"][train_idx]
    val_cell_ids = dino_df["CellId"][val_idx]
    test_cell_ids = dino_df["CellId"][test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_metrics = []

    for method, result_path in exp_name_dir_dict.items():
        if method in ["DINO4Cells-WTC", "DINO-ImageNet"]:
            features, df = torch.load(result_path, weights_only=False)
            df["cell_line"] = df["Protein"]
        elif method == "Engineered":
            features, _, df = torch.load(result_path, weights_only=False)
            features = torch.from_numpy(features)
            df["CellId"] = df["ID"].apply(lambda x: int(x.replace(".ome", "")))
            df["cell_line"] = df["Protein"]
        else:
            df, features = torch.load(result_path, weights_only=False)
            df["cell_line"] = df["structure_name"]

        cell_stage_df = pd.get_dummies(df["cell_stage"])
        y = torch.tensor(cell_stage_df.values)
        unique_cats = cell_stage_df.columns

        train_cell_idx = df["CellId"].isin(train_cell_ids)
        train_x = features[train_cell_idx]
        train_y = y[train_cell_idx]

        val_cell_idx = df["CellId"].isin(val_cell_ids)
        val_x = features[val_cell_idx]
        val_y = y[val_cell_idx]

        test_cell_idx = df["CellId"].isin(test_cell_ids)
        test_x = features[test_cell_idx]
        test_y = y[test_cell_idx]

        save_folder = f"{os.path.dirname(result_path)}/classification"
        os.makedirs(save_folder, exist_ok=True)

        for i in range(10):
            set_random_seed(i)

            cls_save_folder = f"{save_folder}/seed_{i}"
            os.makedirs(cls_save_folder, exist_ok=True)

            if not os.path.isfile(f"{cls_save_folder}/test_preds.csv"):
                model = train_mlp(
                    train_x, train_y, val_x, val_y, device, unique_cats, cls_save_folder
                )
                val_results = eval_model(
                    val_x, val_y, unique_cats, model, seed=i, device=device
                )
                val_results.to_csv(f"{cls_save_folder}/val_preds.csv", index=False)

                test_results = eval_model(
                    test_x, test_y, unique_cats, model, seed=i, device=device
                )
                test_results.to_csv(f"{cls_save_folder}/test_preds.csv", index=False)
            else:
                val_results = pd.read_csv(f"{cls_save_folder}/val_preds.csv")
                test_results = pd.read_csv(f"{cls_save_folder}/test_preds.csv")

            df_true = test_results[[col + "_true" for col in unique_cats]]
            df_true = df_true.rename(
                columns={col: col.replace("_true", "") for col in df_true.columns}
            )
            df_pred = test_results[[col + "_pred" for col in unique_cats]]
            df_pred = df_pred.rename(
                columns={col: col.replace("_pred", "") for col in df_pred.columns}
            )
            cls_rep = pd.DataFrame(
                classification_report(
                    df_true.values.argmax(1),
                    df_pred.values.argmax(1),
                    output_dict=True,
                    target_names=unique_cats,
                    digits=4,
                )
            )
            cls_rep = cls_rep["macro avg"].T
            cls_rep["Method"] = method
            cls_rep["Seed"] = i

            all_metrics.append(cls_rep)

            get_metrics(
                cls_save_folder, val_results, tag="val", unique_cats=unique_cats
            )
            get_metrics(
                cls_save_folder, test_results, tag="test", unique_cats=unique_cats
            )

            plot_umap(cls_save_folder, val_results, tag="val", unique_cats=unique_cats)
            plot_umap(
                cls_save_folder, test_results, tag="test", unique_cats=unique_cats
            )

    all_metrics = pd.concat(all_metrics, axis=1).T.reset_index(drop=True)
    mean_df = all_metrics.groupby("Method").mean().drop(columns=["Seed", "support"])
    std_df = all_metrics.groupby("Method").std().drop(columns=["Seed", "support"])
    mean_std_df = pd.DataFrame(index=mean_df.index, columns=mean_df.columns)
    for col in mean_df.columns:
        for idx in mean_df.index:
            mean_val = mean_df.loc[idx, col]
            std_val = std_df.loc[idx, col]
            mean_std_df.loc[idx, col] = f"{mean_val:.3f} ± {std_val:.4f}"
    mean_std_df.to_csv(
        "/scratch/groups/emmalu/subcell_ankit/allencell/mean_std_metrics.csv"
    )

    sns.barplot(
        x="Method", y="f1-score", hue="Method", data=all_metrics, palette="tab20"
    )
    plt.xticks(rotation=90)
    plt.savefig(
        "/scratch/groups/emmalu/subcell_ankit/allencell/f1-score.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
