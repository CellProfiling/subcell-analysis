import argparse
import copy
import glob
import itertools
import os
import random
import shutil

import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from skimage.measure import regionprops
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils.eval_mlp import compute_metrics

allencell_dict = {
    "M0": "interphase",
    "M1M2": "prophase",
    "M3": "early prometaphase",
    "M4M5": "prometaphase/metaphase",
    "M6M7_complete": "anaphase/telophase paired",
    "M6M7_single": "anaphase/telophase unpaired",
}


def polar_boxplot_circular(df, save_path, n_bins=360, cmap="RdYlGn"):
    df["pseudotime_true_deg"] = df["pseudotime_true"].astype(float) * 2 * np.pi
    df["pseudotime_pred_deg"] = df["pseudotime_pred"].astype(float) * 2 * np.pi

    labels = np.linspace(0, 2 * np.pi, n_bins - 1).round(3)
    df["pseudotime_true_cut"] = pd.cut(
        df["pseudotime_true_deg"],
        bins=np.linspace(0, 2 * np.pi, n_bins),
        labels=labels,
        include_lowest=True,
    ).astype(float)

    df_avg = (
        df.groupby("pseudotime_true_cut", observed=False)[
            [
                "pseudotime_true_cut",
                "pseudotime_pred",
                "pseudotime_pred_deg",
                "var",
                "aleatoric",
            ]
        ]
        .mean()
        .reset_index(drop=True)
    )

    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, n_bins))
    width = (1 / n_bins) * 2 * np.pi
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    for i, row in df_avg.iterrows():
        angle = row["pseudotime_true_cut"]
        color_true = colors[i]
        pred_color_idx = np.argmin(np.abs(labels - row["pseudotime_pred_deg"]))
        color_pred = colors[pred_color_idx]
        ax.bar(
            x=angle,
            height=0.2,
            bottom=0.5,
            width=width,
            color=color_true,
            edgecolor="none",
        )
        ax.bar(
            x=angle,
            height=0.2,
            bottom=0.8,
            width=width,
            color=color_pred,
            edgecolor="none",
        )
    ax.plot(
        df_avg["pseudotime_true_cut"],
        1.2 + df_avg["var"],
        color="black",
        linewidth=2,
    )
    ax.set_ylim(0, np.min([1.7, 1 + df_avg["var"].max() + 0.2]))
    ax.set_yticks([1.2, 1.3, 1.4, 1.5, 1.6, 1.7], minor=False)
    ax.set_yticklabels([f"{i-1.2:.2f}" for i in ax.get_yticks()], fontsize=10)
    ax.xaxis.set_visible(False)
    plt.savefig(f"{save_path}_polar_boxplot.pdf", bbox_inches="tight", dpi=500)
    plt.close()


def plot_allencell(exp_name_dir_dict, save_folder):
    all_cls_res = []
    for method, result_path in exp_name_dir_dict.items():
        method_folder = f"{result_path}/classification"
        unique_cats = list(allencell_dict.keys())
        method_conf_mat = []
        for i in range(10):
            cls_save_folder = f"{method_folder}/seed_{i}"
            test_results = pd.read_csv(f"{cls_save_folder}/test_preds.csv")
            df_true = test_results[[col + "_true" for col in unique_cats]]
            df_true = df_true.rename(
                columns={col: col.replace("_true", "") for col in df_true.columns}
            )
            df_pred = test_results[[col + "_pred" for col in unique_cats]]
            df_pred = df_pred.rename(
                columns={col: col.replace("_pred", "") for col in df_pred.columns}
            )
            conf_mat = confusion_matrix(
                df_true.values.argmax(1), df_pred.values.argmax(1), normalize="true"
            )
            method_conf_mat.append(conf_mat)

            accuracy = accuracy_score(
                df_true.values.argmax(1), df_pred.values.argmax(1)
            )
            macro_f1 = f1_score(
                df_true.values.argmax(1),
                df_pred.values.argmax(1),
                average="macro",
            )
            micro_f1 = f1_score(
                df_true.values.argmax(1),
                df_pred.values.argmax(1),
                average="micro",
            )
            cls_rep = pd.DataFrame(
                {
                    "accuracy": [accuracy],
                    "macro F1": [macro_f1],
                    "micro F1": [micro_f1],
                }
            )
            cls_rep["Method"] = method
            cls_rep["Seed"] = i
            all_cls_res.append(cls_rep)
        method_conf_mat = np.stack(method_conf_mat)
        mean_conf_mat = method_conf_mat.mean(axis=0)
        sns.heatmap(
            mean_conf_mat,
            annot=True,
            fmt=".2f",
            square=True,
            cmap="Blues",
            xticklabels=[allencell_dict[cat] for cat in unique_cats],
            yticklabels=[allencell_dict[cat] for cat in unique_cats],
        )
        plt.savefig(
            f"{save_folder}/{method}_confusion_matrix.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    all_cls_res = pd.concat(all_cls_res, ignore_index=True)
    mean_df = all_cls_res.groupby("Method").mean().drop(columns=["Seed"])
    std_df = all_cls_res.groupby("Method").std().drop(columns=["Seed"])
    mean_std_df = pd.DataFrame(index=mean_df.index, columns=mean_df.columns)
    for col in mean_df.columns:
        for idx in mean_df.index:
            mean_val = mean_df.loc[idx, col]
            std_val = std_df.loc[idx, col]
            mean_std_df.loc[idx, col] = f"{mean_val:.3f} ± {std_val:.4f}"
    mean_std_df.to_csv(f"{save_folder}/mean_std_metrics.csv")

    for metric in ["accuracy", "macro F1", "micro F1"]:
        fix, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Method", y=metric, hue="Method", data=all_cls_res, ax=ax)
        for elem in ax.containers:
            ax.bar_label(elem, fmt="%.3f", label_type="edge", fontsize=26, padding=-30)
        ax.set_box_aspect(0.5)
        plt.xticks(rotation=45, ha="right")
        plt.savefig(f"{save_folder}/{metric}.pdf", dpi=300, bbox_inches="tight")
        plt.close()


def plot_fucci(exp_name_dir_dict, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    all_cls_res = []
    unique_cats = ["G1", "G1S", "G2"]
    for method, result_path in exp_name_dir_dict.items():
        method_reg_df = []
        method_cls_df = []
        for i in range(10):
            for j in range(10):
                split_reg_df = pd.read_csv(
                    f"{result_path}/split_{i}/regression/seed_{j}/val_preds.csv"
                )
                split_reg_df = split_reg_df.drop(
                    columns=[i for i in split_reg_df.columns if "feat" in i]
                )
                method_reg_df.append(split_reg_df)

                split_cls_df = pd.read_csv(
                    f"{result_path}/split_{i}/classification/seed_{j}/val_preds.csv"
                )
                split_cls_df = split_cls_df.drop(
                    columns=[i for i in split_cls_df.columns if "feat" in i]
                )
                method_cls_df.append(split_cls_df)

        method_reg_df = pd.concat(method_reg_df, ignore_index=True)
        method_cls_df = pd.concat(method_cls_df, ignore_index=True)

        df_true_label = method_cls_df[[f"{x}_true" for x in unique_cats]].values.argmax(
            axis=1
        )
        df_pred_label = method_cls_df[[f"{x}_pred" for x in unique_cats]].values.argmax(
            axis=1
        )

        accuracy = accuracy_score(df_true_label, df_pred_label)
        macro_f1 = f1_score(df_true_label, df_pred_label, average="macro")
        micro_f1 = f1_score(df_true_label, df_pred_label, average="micro")
        cls_rep = pd.DataFrame(
            {
                "accuracy": [accuracy],
                "macro F1": [macro_f1],
                "micro F1": [micro_f1],
            }
        )
        cls_rep["Method"] = method
        all_cls_res.append(cls_rep)

        conf_mat = confusion_matrix(df_true_label, df_pred_label, normalize="true")
        sns.heatmap(
            conf_mat,
            annot=True,
            fmt=".2f",
            square=True,
            cmap="Blues",
            xticklabels=unique_cats,
            yticklabels=unique_cats,
        )
        plt.savefig(
            f"{save_folder}/{method}_confusion_matrix.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        polar_boxplot_circular(
            method_reg_df,
            save_path=f"{save_folder}/{method}",
            n_bins=180,
            cmap="RdYlGn",
        )

    all_cls_res = pd.concat(all_cls_res, ignore_index=True)
    all_cls_res["Method"] = pd.Categorical(
        all_cls_res["Method"],
        categories=exp_name_dir_dict.keys(),
        ordered=True,
    )
    all_cls_res.to_csv(f"{save_folder}/metrics.csv")

    sns.set_style("ticks")
    for metric in ["accuracy", "macro F1", "micro F1"]:
        fix, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Method", y=metric, hue="Method", data=all_cls_res, ax=ax)
        for elem in ax.containers:
            ax.bar_label(elem, fmt="%.3f", label_type="edge", fontsize=26, padding=-30)
        ax.set_box_aspect(0.5)
        plt.xticks(rotation=45, ha="right")
        plt.savefig(f"{save_folder}/{metric}.pdf", dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    allencell_save_folder = "/scratch/groups/emmalu/subcell_ankit/allencell/results_all"
    os.makedirs(allencell_save_folder, exist_ok=True)
    exp_name_dir_dict_allencell = {
        "DINO4Cells-WTC": "/scratch/groups/emmalu/subcell_ankit/allencell/dino",
        "DINO-ImageNet": "/scratch/groups/emmalu/subcell_ankit/allencell/pretrained",
        "MAE-DNA-Struct-PerCh": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_bg_mae_channel_comb_nuc_prot_only_norm_perchannel",
        "MAE-DNA-Struct-PerIm": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_bg_mae_channel_comb_nuc_prot_only_norm_perimage",
        "MAE-DNA-Struct-Plasma-Concat-PerCh": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_bg_mae_channel_comb_concat_norm_perchannel",
        "MAE-DNA-Struct-Plasma-Concat-PerIm": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_bg_mae_channel_comb_concat_norm_perimage",
        "MAE-DNA-Struct-Plasma-PerCh": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_rbg_mae_channel_comb_plasma_nuc_prot_norm_perchannel",
        "MAE-DNA-Struct-Plasma-PerIm": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_rbg_mae_channel_comb_plasma_nuc_prot_norm_perimage",
        "ViT-DNA-Struct-PerCh": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_bg_vit_channel_comb_nuc_prot_only_norm_perchannel",
        "ViT-DNA-Struct-PerIm": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_bg_vit_channel_comb_nuc_prot_only_norm_perimage",
        "ViT-DNA-Struct-Plasma-Concat-PerCh": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_bg_vit_channel_comb_concat_norm_perchannel",
        "ViT-DNA-Struct-Plasma-Concat-PerIm": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_bg_vit_channel_comb_concat_norm_perimage",
        "ViT-DNA-Struct-Plasma-PerCh": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_rbg_vit_channel_comb_plasma_nuc_prot_norm_perchannel",
        "ViT-DNA-Struct-Plasma-PerIm": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_rbg_vit_channel_comb_plasma_nuc_prot_norm_perimage",
    }
    plot_allencell(exp_name_dir_dict_allencell, allencell_save_folder)

    allencell_save_folder = (
        "/scratch/groups/emmalu/subcell_ankit/allencell/results_best"
    )
    os.makedirs(allencell_save_folder, exist_ok=True)
    exp_name_dir_dict_allencell = {
        "DINO4Cells-WTC": "/scratch/groups/emmalu/subcell_ankit/allencell/dino",
        "DINO-ImageNet": "/scratch/groups/emmalu/subcell_ankit/allencell/pretrained",
        "MAE-ProtS-CellS-Pool": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_bg_mae_channel_comb_concat_norm_perchannel",
        "ViT-ProtS-Pool": "/scratch/groups/emmalu/subcell_ankit/allencell/subcell_bg_vit_channel_comb_concat_norm_perchannel",
    }
    plot_allencell(exp_name_dir_dict_allencell, allencell_save_folder)

    fucci_save_folder = "/scratch/groups/emmalu/subcell_ankit/fucci/results_all"
    os.makedirs(fucci_save_folder, exist_ok=True)

    exp_name_dir_dict_fucci = {
        "MAE-Nuc-Zero-Resize": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_mae_channel_comb_nuc_zero_resize_True",
        "ViT-Nuc-Zero-Resize": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_vit_channel_comb_nuc_zero_resize_True",
        "MAE-Nuc-Const-Resize": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_mae_channel_comb_nuc_const_resize_True",
        "ViT-Nuc-Const-Resize": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_vit_channel_comb_nuc_const_resize_True",
        "MAE-Nuc-CDT-GMMN-Resize": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_mae_channel_comb_nuc_cdt_gmmn_resize_True",
        "ViT-Nuc-CDT-GMMN-Resize": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_vit_channel_comb_nuc_cdt_gmmn_resize_True",
    }
    plot_fucci(exp_name_dir_dict_fucci, f"{fucci_save_folder}")

    fucci_save_folder = "/scratch/groups/emmalu/subcell_ankit/fucci/results"
    os.makedirs(fucci_save_folder, exist_ok=True)

    exp_name_dir_dict_fucci = {
        "MAE-Nuc-Zero": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_mae_channel_comb_nuc_zero_resize_True",
        "MAE-Nuc-Const": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_mae_channel_comb_nuc_const_resize_True",
        "MAE-Nuc-CDT-GMMN": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_mae_channel_comb_nuc_cdt_gmmn_resize_True",
    }
    plot_fucci(exp_name_dir_dict_fucci, f"{fucci_save_folder}/mae")

    exp_name_dir_dict_fucci = {
        "ViT-Nuc-Zero": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_vit_channel_comb_nuc_zero_resize_True",
        "ViT-Nuc-Const": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_vit_channel_comb_nuc_const_resize_True",
        "ViT-Nuc-CDT-GMMN": "/scratch/groups/emmalu/subcell_ankit/fucci/subcell_rbg_vit_channel_comb_nuc_cdt_gmmn_resize_True",
    }
    plot_fucci(exp_name_dir_dict_fucci, f"{fucci_save_folder}/vit")
