"""Cell-cycle probes on FUCCI U2OS features (Fig. 2k,l).

The dataset has no nuclear stain, so each SubCell encoder is run with three
substitutes for the DNA channel (``config.FUCCI_VARIANTS``). On every variant's
cell features, an MLP classifies the FUCCI stage (G1, G1/S, G2) and a deep
evidential regressor predicts pseudotime, in ten-fold cross-validation with whole
antibodies held out together and ten seeds per fold.
``figures.fig2_cell_cycle`` summarises the classification results.

    python -m subcell.classification.cellcycle_fucci

Inputs: ``config.FUCCI_DIR/<variant>/all_features.pth``, written by
``subcell.inference.fucci_crops`` as ``(metadata, features)``; the metadata has
``ab_id``, ``GMM_cc``, ``GMM_cc_label`` and ``pseudotime``.
Outputs, under ``config.FUCCI_DIR``: ``data_splits/fucci_split_10_fold.csv`` (the
fold of each cell, written once and reused) and, per variant,
``split_<k>/classification/seed_<i>/`` and ``split_<k>/regression/seed_<i>/``.
"""

import argparse
import os

import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from subcell import config
from subcell.classification.evidential_regression import eval_model_reg, train_mlp_reg
from subcell.classification.probe import probe_seeds
from subcell.utils.seed import set_random_seed

N_SPLITS = 10
SEEDS = range(10)
SPLIT_CSV = config.FUCCI_DIR / "data_splits" / "fucci_split_10_fold.csv"


def feature_file(variant_dir):
    return config.FUCCI_DIR / variant_dir / "all_features.pth"


def generate_data_splits(df, features):
    """Assign every cell to a test fold (``test_split``) of a ten-fold
    cross-validation that keeps all cells of an antibody together
    (``GroupKFold`` on the antibody; folds are not balanced by FUCCI stage)."""
    split_df = df.copy()
    group_kfold = GroupKFold(n_splits=N_SPLITS)
    for i, (train_index, test_index) in enumerate(
        group_kfold.split(features, df["GMM_cc"], df["ab_id"])
    ):
        train_abs = df["ab_id"].iloc[train_index].unique()
        test_abs = df["ab_id"].iloc[test_index].unique()
        assert set(train_abs).isdisjoint(test_abs), "Train and test sets should not overlap"
        split_df.loc[test_index, "test_split"] = int(i)

    os.makedirs(SPLIT_CSV.parent, exist_ok=True)
    split_df.to_csv(SPLIT_CSV, index=False)
    return split_df


def fold_sets(df, features, split_df, split_idx, y):
    """Train and validation ``(features, y)`` of one fold, plus its folder."""
    val_abs = split_df[split_df["test_split"] == split_idx]["ab_id"].unique()
    train_abs = split_df[split_df["test_split"] != split_idx]["ab_id"].unique()
    assert set(train_abs).isdisjoint(val_abs)
    train_idx = df["ab_id"].isin(train_abs)
    val_idx = df["ab_id"].isin(val_abs)
    return (features[train_idx], y[train_idx]), (features[val_idx], y[val_idx]), val_idx


def eval_classification(df, features, split_df, device, method, result_dir):
    cell_stage_df = pd.get_dummies(df["GMM_cc_label"])
    y = torch.tensor(cell_stage_df.values)
    for split_idx in range(N_SPLITS):
        print(f"Evaluating classification {method} on split {split_idx}", flush=True)
        train, val, _ = fold_sets(df, features, split_df, split_idx, y)
        probe_seeds(
            f"{result_dir}/split_{split_idx}/classification", device, train,
            {"val": val}, cell_stage_df.columns, SEEDS,
        )


def eval_regression(df, features, split_df, device, method, result_dir):
    y = torch.tensor(df["pseudotime"].values)
    for split_idx in range(N_SPLITS):
        train, val, val_idx = fold_sets(df, features, split_df, split_idx, y)
        save_folder = f"{result_dir}/split_{split_idx}/regression"
        for i in SEEDS:
            folder = f"{save_folder}/seed_{i}"
            os.makedirs(folder, exist_ok=True)
            set_random_seed(i, deterministic=True)
            print(f"Evaluating regression {method} on split {split_idx}, seed {i}", flush=True)

            if not os.path.isfile(f"{folder}/val_preds.csv"):
                model = train_mlp_reg(*train, *val, device, folder)
                val_results = eval_model_reg(*val, model, seed=i, device=device)
                val_meta_df = df[val_idx].copy().reset_index(drop=True)
                val_results[val_meta_df.columns] = val_meta_df
                val_results.to_csv(f"{folder}/val_preds.csv", index=False)


def main(variants=None):
    """Run the classification and regression probes of ``variants`` (names from
    ``config.FUCCI_VARIANTS``; default all)."""
    variants = {k: config.FUCCI_VARIANTS[k] for k in (variants or config.FUCCI_VARIANTS)}
    if SPLIT_CSV.exists():
        split_df = pd.read_csv(SPLIT_CSV)
    else:
        df, features = torch.load(feature_file(next(iter(variants.values()))), weights_only=False)
        split_df = generate_data_splits(df, features)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for method, variant_dir in variants.items():
        df, features = torch.load(feature_file(variant_dir), weights_only=False)
        print(f"Evaluating {method} on FUCCI dataset", flush=True)
        result_dir = config.FUCCI_DIR / variant_dir
        eval_classification(df, features, split_df, device, method, result_dir)
        eval_regression(df, features, split_df, device, method, result_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", nargs="+", choices=list(config.FUCCI_VARIANTS),
                    help="model variants to run (default: all)")
    main(ap.parse_args().variants)
