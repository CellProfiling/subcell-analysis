import argparse
import glob
import os
import shutil
import random

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
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    coverage_error,
    label_ranking_average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from tqdm import tqdm

plt.switch_backend("agg")

from utils.train_mlp import train_mlp, eval_model

UNIQUE_CATS = np.array(
    [
        cat
        for cat in pd.read_csv("annotations/location_group_mapping.csv")[
            "Original annotation"
        ]
        .unique()
        .tolist()
        if cat
        not in ["Cleavage furrow", "Midbody ring", "Rods & Rings", "Microtubule ends"]
    ]
    + ["Negative"]
)


CHALLENGE_CATS = [
    "Actin filaments",
    "Aggresome",
    "Centrosome",
    "Cytosol",
    "Endoplasmic reticulum",
    "Golgi apparatus",
    "Intermediate filaments",
    "Microtubules",
    "Mitochondria",
    "Mitotic spindle",
    "Nuclear bodies",
    "Nuclear membrane",
    "Nuclear speckles",
    "Nucleoli",
    "Nucleoli fibrillar center",
    "Nucleoplasm",
    "Plasma membrane",
    "Vesicles",
    "Negative",
]


def filter_classes(df, feature_data, unique_cats=UNIQUE_CATS):
    locations_list = df["locations"].str.split(",").tolist()
    labels_onehot = np.array(
        [[1 if cat in x else 0 for cat in unique_cats] for x in locations_list]
    )

    keep_idx = np.where(labels_onehot.sum(axis=1) > 0)[0]
    df = df.iloc[keep_idx].reset_index(drop=True)
    df[unique_cats] = labels_onehot[keep_idx]
    feature_data = feature_data[keep_idx]

    return df, feature_data


def preprocess_hidden_dataset(df, features):
    keep_idx = df[~df["annotated_label"].isna()].index
    df = df.iloc[keep_idx].reset_index(drop=True)
    features = features[keep_idx]

    keep_idx = df[~df["annotated_label"].isin(["Discarded", "Unsure"])].index
    df = df.iloc[keep_idx].reset_index(drop=True)
    features = features[keep_idx]
    df = df.rename(columns={"annotated_label": "locations"})
    df["locations"] = df["locations"].str.replace(", ", ",")
    df.loc[
        df["locations"].isin(["Negative", "Neg/Unspec", "Unspecific"]),
        "locations",
    ] = "Negative"

    return df, features


def get_atlas_name_classes(df):
    cell_lines = df["atlas_name"].unique()
    labels_onehot = pd.get_dummies(df["atlas_name"]).values
    return labels_onehot, cell_lines


def get_train_val_test_idx(df, feature_data, unique_cats=UNIQUE_CATS):
    train_antibodies = pd.read_csv(
        "annotations/splits/train_antibodies.txt", header=None
    )[0].to_list()
    val_antibodies = pd.read_csv(
        "annotations/splits/valid_antibodies.txt", header=None
    )[0].to_list()
    test_antibodies = pd.read_csv(
        "annotations/splits/test_antibodies.txt", header=None
    )[0].to_list()
    train_idxs = df[df["antibody"].isin(train_antibodies)].index.to_list()
    val_idxs = df[df["antibody"].isin(val_antibodies)].index.to_list()
    test_idxs = df[df["antibody"].isin(test_antibodies)].index.to_list()

    train_x = feature_data[train_idxs]
    train_y = torch.from_numpy(df[unique_cats].iloc[train_idxs].values)

    val_x = feature_data[val_idxs]
    val_y = torch.from_numpy(df[unique_cats].iloc[val_idxs].values)

    test_x = feature_data[test_idxs]
    test_y = torch.from_numpy(df[unique_cats].iloc[test_idxs].values)
    return train_x, train_y, val_x, val_y, test_x, test_y


def get_multilabel_df(df_true, df_pred):
    cols = df_true.columns

    avg_precisions = []
    aucs = []
    all_categories = []
    all_counts = []
    for cat in cols:
        if len(np.unique(df_true[cat])) != 2:
            continue
        avg_precision = average_precision_score(df_true[cat], df_pred[cat])
        avg_precisions.append(avg_precision)
        all_categories.append(cat)
        all_counts.append(df_true[cat].sum())
        auc = roc_auc_score(df_true[cat], df_pred[cat])
        aucs.append(auc)

    avg_precisions.append(average_precision_score(df_true.values, df_pred.values))
    aucs.append(roc_auc_score(df_true.values, df_pred.values))
    all_categories.append("Overall")
    all_counts.append(len(df_true))
    df_multilabel = (
        pd.DataFrame(
            {
                "Category": all_categories,
                "Average Precision": avg_precisions,
                "AUC": aucs,
                "Count": all_counts,
            }
        )
        .sort_values(by="Count", ascending=False)
        .reset_index(drop=True)
    )
    return df_multilabel


def plot_multilabel_metrics(
    df, metric="Average Precision", label="valid", save_folder="./"
):
    n_cats = len(df)
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(1, figsize=(16, 10))
    sns.barplot(
        x="Category",
        y=metric,
        hue="Category",
        palette=sns.color_palette(cc.glasbey_dark, n_cats),
        data=df,
        ax=ax,
        orient="v",
    )
    plt.ylim(0, 1)
    plt.xticks(rotation=90)
    plt.savefig(
        f"{save_folder}/{label}_{metric}.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close()


def get_metrics(save_folder, df_test, tag="test", unique_cats=UNIQUE_CATS):
    df_true = df_test[[col + "_true" for col in unique_cats]]
    df_true = df_true.rename(
        columns={col: col.replace("_true", "") for col in df_true.columns}
    )
    df_pred = df_test[[col + "_pred" for col in unique_cats]]
    df_pred = df_pred.rename(
        columns={col: col.replace("_pred", "") for col in df_pred.columns}
    )

    non_zero_cats = [col for col in unique_cats if df_true[col].sum() > 0]
    df_true = df_true[non_zero_cats]
    df_pred = df_pred[non_zero_cats]

    label_ranking_ap = label_ranking_average_precision_score(
        df_true.values, df_pred.values
    )
    coverage = coverage_error(df_true.values, df_pred.values)
    micro_avg_precision = average_precision_score(
        df_true.values, df_pred.values, average="micro"
    )

    df_multilabel = get_multilabel_df(df_true, df_pred)
    df_multilabel["Coverage Error"] = coverage
    df_multilabel["Label Ranking Average Precision"] = label_ranking_ap
    df_multilabel["Micro Average Precision"] = micro_avg_precision
    df_multilabel.to_csv(f"{save_folder}/{tag}_metrics.csv", index=False)

    plot_multilabel_metrics(
        df_multilabel,
        metric="Average Precision",
        label=tag,
        save_folder=save_folder,
    )
    plot_multilabel_metrics(
        df_multilabel, metric="AUC", label=tag, save_folder=save_folder
    )


def str2bool(v):
    return v.lower() in ("True", "true", "1")


def get_challenge_data(features_folder, df, unique_cats):
    challenge_df, challenge_feature_data = torch.load(
        f"{features_folder}/challenge_features/all_features.pth", map_location="cpu"
    )
    intersection = list(
        set(df["antibody"].unique()).intersection(
            set(challenge_df["antibody"].unique())
        )
    )
    keep_idx = challenge_df[
        ~challenge_df["antibody"].isin(intersection)
    ].index.to_numpy()
    challenge_df = challenge_df.loc[keep_idx].reset_index(drop=True)
    challenge_feature_data = challenge_feature_data[keep_idx]
    challenge_df, challenge_feature_data = preprocess_hidden_dataset(
        challenge_df, challenge_feature_data
    )
    challenge_df, challenge_feature_data = filter_classes(
        challenge_df, challenge_feature_data
    )
    challenge_y = torch.from_numpy(challenge_df[unique_cats].values)
    return challenge_feature_data, challenge_y


def get_bridge2ai_data(features_folder, unique_cats):
    bridge2ai_df, bridge2ai_feature_data = torch.load(
        f"{features_folder}/bridge2ai_features/all_features.pth", map_location="cpu"
    )
    bridge2ai_x = bridge2ai_feature_data
    bridge2ai_y = bridge2ai_df[unique_cats].values
    non_zero_idx = np.where(bridge2ai_y.sum(axis=1) > 0)[0]
    bridge2ai_x = bridge2ai_x[non_zero_idx]
    bridge2ai_y = torch.from_numpy(bridge2ai_y[non_zero_idx])
    return bridge2ai_x, bridge2ai_y


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-f", "--features_folder", type=str)
    argparser.add_argument("-hf", "--harmonize_features", type=str2bool, default=False)
    argparser.add_argument(
        "-cc", "--classification_cats", type=str, default="locations"  # "atlas_name"
    )
    argparser.add_argument("-uc", "--unique_cats", type=str, default="all_unique_cats")

    args = argparser.parse_args()

    features_folder = args.features_folder
    harmonize_features = args.harmonize_features
    classification_cats = args.classification_cats
    unique_cats_name = (
        args.unique_cats if classification_cats == "locations" else "atlas_name"
    )

    print(f"Parameters: {args}")

    save_folder = (
        f"{features_folder}/classification"
        if not harmonize_features
        else f"{features_folder}/classification_aligned"
    )
    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder, exist_ok=True)

    df, feature_data = torch.load(
        f"{features_folder}/all_features.pth", map_location="cpu"
    )
    if harmonize_features:
        if not os.path.isfile(f"{features_folder}/all_features_harmonize.pth"):
            vars_use = ["atlas_name"]
            feature_data = harmonize(
                feature_data.numpy(),
                df,
                batch_key=vars_use,
                use_gpu=True,
                verbose=True,
                random_state=42,
            )
            feature_data = torch.from_numpy(feature_data)
            torch.save(
                (df, feature_data), f"{features_folder}/all_features_harmonized.pth"
            )
        else:
            df, feature_data = torch.load(
                f"{features_folder}/all_features_harmonize.pth"
            )

    if classification_cats == "locations":
        df.loc[df["locations"].isna(), "locations"] = "Negative"
        unique_cats = (
            UNIQUE_CATS if unique_cats_name == "all_unique_cats" else CHALLENGE_CATS
        )
        df, feature_data = filter_classes(df, feature_data, unique_cats=unique_cats)

        challenge_x, challenge_y = get_challenge_data(features_folder, df, unique_cats)
        bridge2ai_x, bridge2ai_y = get_bridge2ai_data(features_folder, unique_cats)

    elif classification_cats == "atlas_name":
        unique_cats = df["atlas_name"].unique()
        df[unique_cats] = pd.get_dummies(df["atlas_name"])

    print(
        f"Found {len(df)} samples with {len(unique_cats)} unique categories: {unique_cats}"
    )

    train_x, train_y, val_x, val_y, test_x, test_y = get_train_val_test_idx(
        df, feature_data, unique_cats
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    for i in range(10):
        np.random.seed(i)
        random.seed(i)
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)

        cls_save_folder = f"{save_folder}/multiclass_{unique_cats_name}_seed_{i}"
        os.makedirs(cls_save_folder, exist_ok=True)

        if not os.path.isfile(f"{save_folder}/test_preds.csv"):
            model = train_mlp(
                train_x,
                train_y,
                val_x,
                val_y,
                test_x,
                test_y,
                device,
                unique_cats,
                cls_save_folder,
            )
            val_results = eval_model(
                val_x, val_y, unique_cats, model, seed=i, device=device
            )
            val_results.to_csv(f"{cls_save_folder}/val_preds.csv", index=False)

            test_results = eval_model(
                test_x, test_y, unique_cats, model, seed=i, device=device
            )
            test_results.to_csv(f"{cls_save_folder}/test_preds.csv", index=False)

            get_metrics(
                cls_save_folder, val_results, tag="val", unique_cats=unique_cats
            )
            get_metrics(
                cls_save_folder, test_results, tag="test", unique_cats=unique_cats
            )
            if classification_cats == "locations":
                hidden_test_results = eval_model(
                    challenge_x, challenge_y, unique_cats, model, seed=i, device=device
                )
                hidden_test_results.to_csv(
                    f"{cls_save_folder}/hidden_test_preds.csv", index=False
                )

                bridge2ai_results = eval_model(
                    bridge2ai_x, bridge2ai_y, unique_cats, model, seed=i, device=device
                )
                bridge2ai_results.to_csv(
                    f"{cls_save_folder}/bridge2ai_preds.csv", index=False
                )

                get_metrics(
                    cls_save_folder,
                    hidden_test_results,
                    tag="hidden_test",
                    unique_cats=unique_cats,
                )
                get_metrics(
                    cls_save_folder,
                    bridge2ai_results,
                    tag="bridge2ai",
                    unique_cats=unique_cats,
                )

        else:
            val_results = pd.read_csv(f"{cls_save_folder}/val_preds.csv")
            test_results = pd.read_csv(f"{cls_save_folder}/test_preds.csv")
            hidden_test_results = pd.read_csv(
                f"{cls_save_folder}/hidden_test_preds.csv"
            )
            bridge2ai_results = pd.read_csv(f"{cls_save_folder}/bridge2ai_preds.csv")

            get_metrics(
                cls_save_folder, val_results, tag="val", unique_cats=unique_cats
            )
            get_metrics(
                cls_save_folder, test_results, tag="test", unique_cats=unique_cats
            )
            get_metrics(
                cls_save_folder,
                hidden_test_results,
                tag="hidden_test",
                unique_cats=unique_cats,
            )
            get_metrics(
                cls_save_folder,
                bridge2ai_results,
                tag="bridge2ai",
                unique_cats=unique_cats,
            )
