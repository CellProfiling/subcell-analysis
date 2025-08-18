import argparse
import copy
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
from sklearn.model_selection import GroupKFold, train_test_split
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
from utils.train_mlp_reg import eval_model_reg, plot_pred, plot_umap_reg, train_mlp_reg
import matplotlib

matplotlib.use("Agg")  

def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


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


def generate_data_splits(df):
    data_splits_folder = "/scratch/groups/emmalu/subcell_ankit/fucci/data_splits"
    os.makedirs(data_splits_folder, exist_ok=True)

    split_df = copy.deepcopy(df)
    group_kfold = GroupKFold(n_splits=10)
    for i, (train_index, test_index) in enumerate(
        group_kfold.split(features, df["GMM_cc"], df["ab_id"])
    ):
        train_abs = df["ab_id"].iloc[train_index].unique()
        test_abs = df["ab_id"].iloc[test_index].unique()
        assert (
            set(train_abs).intersection(set(test_abs)) == set()
        ), "Train and test sets should not overlap"

        split_df.loc[test_index, "test_split"] = int(i)

        print(np.unique(df["GMM_cc"][train_index], return_counts=True))
        print(np.unique(df["GMM_cc"][test_index], return_counts=True))

    split_df.to_csv(f"{data_splits_folder}/fucci_split_10_fold.csv", index=False)
    return split_df


def eval_classification(df, features, split_df, device, method, result_path):
    for split_idx in range(10):
        split_save_folder = f"{os.path.dirname(result_path)}/split_{split_idx}"
        os.makedirs(split_save_folder, exist_ok=True)

        val_abs = split_df[split_df["test_split"] == split_idx]["ab_id"].unique()
        train_abs = split_df[split_df["test_split"] != split_idx]["ab_id"].unique()

        cell_stage_df = pd.get_dummies(df["GMM_cc_label"])
        y = torch.tensor(cell_stage_df.values)
        unique_cats = cell_stage_df.columns

        train_cell_idx = df["ab_id"].isin(train_abs)
        train_x = features[train_cell_idx]
        train_y = y[train_cell_idx]

        val_cell_idx = df["ab_id"].isin(val_abs)
        val_x = features[val_cell_idx]
        val_y = y[val_cell_idx]

        save_folder = f"{split_save_folder}/classification"
        os.makedirs(save_folder, exist_ok=True)

        for i in range(10):
            set_random_seed(i)

            cls_save_folder = f"{save_folder}/seed_{i}"
            os.makedirs(cls_save_folder, exist_ok=True)

            print(
                f"Evaluating classification {method} on split {split_idx}, seed {i}",
                flush=True,
            )

            if not os.path.isfile(f"{cls_save_folder}/val_preds.csv"):
                model = train_mlp(
                    train_x,
                    train_y,
                    val_x,
                    val_y,
                    device,
                    unique_cats,
                    cls_save_folder,
                )
                val_results = eval_model(
                    val_x, val_y, unique_cats, model, seed=i, device=device
                )
                val_results.to_csv(f"{cls_save_folder}/val_preds.csv", index=False)
            else:
                print("Result already exist....")
                val_results = pd.read_csv(f"{cls_save_folder}/val_preds.csv")

            df_true = val_results[[col + "_true" for col in unique_cats]]
            df_true = df_true.rename(
                columns={col: col.replace("_true", "") for col in df_true.columns}
            )
            df_pred = val_results[[col + "_pred" for col in unique_cats]]
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

            get_metrics(
                cls_save_folder, val_results, tag="val", unique_cats=unique_cats
            )

            plot_umap(cls_save_folder, val_results, tag="val", unique_cats=unique_cats)


def eval_regression(df, features, split_df, device, method, result_path):
    for split_idx in range(10):
        split_save_folder = f"{os.path.dirname(result_path)}/split_{split_idx}"
        os.makedirs(split_save_folder, exist_ok=True)

        val_abs = split_df[split_df["test_split"] == split_idx]["ab_id"].unique()
        train_abs = split_df[split_df["test_split"] != split_idx]["ab_id"].unique()

        assert set(train_abs).intersection(val_abs) == set()

        y = torch.tensor(df["pseudotime"].values)

        train_cell_idx = df["ab_id"].isin(train_abs)
        train_x = features[train_cell_idx]
        train_y = y[train_cell_idx]

        val_cell_idx = df["ab_id"].isin(val_abs)
        val_x = features[val_cell_idx]
        val_y = y[val_cell_idx]

        save_folder = f"{split_save_folder}/regression"
        os.makedirs(save_folder, exist_ok=True)

        for i in range(10):
            set_random_seed(i)

            cls_save_folder = f"{save_folder}/seed_{i}"
            os.makedirs(cls_save_folder, exist_ok=True)

            print(
                f"Evaluating regression {method} on split {split_idx}, seed {i}",
                flush=True,
            )

            if not os.path.isfile(f"{cls_save_folder}/val_preds.csv"):
                model = train_mlp_reg(
                    train_x, train_y, val_x, val_y, device, cls_save_folder
                )
                val_results = eval_model_reg(val_x, val_y, model, seed=i, device=device)
                val_meta_df = df[val_cell_idx].copy().reset_index(drop=True)

                val_results[val_meta_df.columns] = val_meta_df
                val_results.to_csv(f"{cls_save_folder}/val_preds.csv", index=False)
            else:
                print("Result already exist....")
                val_results = pd.read_csv(f"{cls_save_folder}/val_preds.csv")

            plot_umap_reg(cls_save_folder, val_results)
            plot_pred(cls_save_folder, val_results)


if __name__ == "__main__":
    exp_name_dir_dict = {
        "MAE-Nuc-Zero-Orig": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_mae_channel_comb_nuc_zero_resize_False/all_features.pth",
        "MAE-Nuc-Const-Orig": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_mae_channel_comb_nuc_const_resize_False/all_features.pth",
        "MAE-Nuc-CDT-GMMN-Orig": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_mae_channel_comb_nuc_cdt_gmmn_resize_False/all_features.pth",
        "ViT-Nuc-Zero-Orig": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_vit_channel_comb_nuc_zero_resize_False/all_features.pth",
        "ViT-Nuc-Const-Orig": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_vit_channel_comb_nuc_const_resize_False/all_features.pth",
        "ViT-Nuc-CDT-GMMN-Orig": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_vit_channel_comb_nuc_cdt_gmmn_resize_False/all_features.pth",
        "MAE-Nuc-Zero-Resize": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_mae_channel_comb_nuc_zero_resize_True/all_features.pth",
        "MAE-Nuc-Const-Resize": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_mae_channel_comb_nuc_const_resize_True/all_features.pth",
        "MAE-Nuc-CDT-GMMN-Resize": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_mae_channel_comb_nuc_cdt_gmmn_resize_True/all_features.pth",
        "ViT-Nuc-Zero-Resize": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_vit_channel_comb_nuc_zero_resize_True/all_features.pth",
        "ViT-Nuc-Const-Resize": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_vit_channel_comb_nuc_const_resize_True/all_features.pth",
        "ViT-Nuc-CDT-GMMN-Resize": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_vit_channel_comb_nuc_cdt_gmmn_resize_True/all_features.pth",
    }

    save_folder = "/scratch/groups/emmalu/subcell_ankit/fucci"

    if not os.path.isfile(f"{save_folder}/data_splits/fucci_split_10_fold.csv"):
        df, features = torch.load(
            list(exp_name_dir_dict.values())[0], weights_only=False
        )
        split_df = generate_data_splits(df)
    else:
        split_df = pd.read_csv(f"{save_folder}/data_splits/fucci_split_10_fold.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for method, result_path in exp_name_dir_dict.items():
        df, features = torch.load(result_path, weights_only=False)
        print(f"Evaluating {method} on FUCCI dataset", flush=True)
        eval_classification(df, features, split_df, device, method, result_path)
        eval_regression(df, features, split_df, device, method, result_path)
