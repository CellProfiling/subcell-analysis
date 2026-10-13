"""Drug prediction from single-cell CM4AI features (Fig. 4a).

Cells of MDA-MB-468 treated with paclitaxel, vorinostat or left untreated are
classified from the features of each model by nested ten-fold cross-validation,
with folds grouped by target gene (``config.CM4AI_SPLIT``). Hyperparameters are
chosen on the next fold as inner validation set; see
``subcell.classification.nested_cv.cross_val_mlp_folds``.

Two stages::

    python -m subcell.perturbation.cm4ai_drug_prediction crops
    python -m subcell.perturbation.cm4ai_drug_prediction predict [--methods ...] [--folds ...]

``crops`` cuts the masked single-cell crops that the models are run on (written
to ``$SUBCELL_ROOT/bridge2ai_crops``) from the colour-coded FOV images and
segmentations under ``config.ROOTS["CM4AI_DATA"]``. ``predict`` reads
``config.features(<method>, "b2ai_features")`` for every method in
``config.CM4AI_VARIANTS`` and writes per-fold results and ``all_results.csv`` to
``config.CM4AI_DIR / "drug_prediction"``.
"""

from __future__ import annotations

import argparse
import random

import cv2
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import regionprops
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

from subcell import config
from subcell.classification.nested_cv import cross_val_mlp_folds
from subcell.io.preprocess import safe_crop
from subcell.perturbation import CM4AI_METADATA

CROPS_DIR = config.CM4AI_CROPS
N_SPLITS = 10

# The FOV JPEGs are colour-coded renderings of the four stains. Each channel is
# read from its own colour component rather than from luminance, which would
# scale it by the colour's luma weight (e.g. DAPI, rendered blue, by 0.072).
HPA_COLORS = ["red", "yellow", "blue", "green"]
FOV_BGR_INDEX = {"red": 2, "yellow": 2, "blue": 0, "green": 1}


def process_single_fov(fov_id, cell_labels, fov_folder, mask_folder, save_folder, crop_size=1024):
    """Write the masked ``crop_size`` crop of every listed cell of one FOV."""
    fov = np.stack(
        [
            cv2.imread(f"{fov_folder}/{fov_id}{color}.jpg")[..., FOV_BGR_INDEX[color]]
            for color in HPA_COLORS
        ],
        axis=-1,
    )
    cell_mask = cv2.imread(f"{mask_folder}/{fov_id}cellmask.png", -1)
    mask_region_props = {r.label: r for r in regionprops(cell_mask)}

    for cell_label in cell_labels:
        bbox = mask_region_props[cell_label].bbox
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        crop_bbox = (
            int(center[0] - crop_size // 2),
            int(center[1] - crop_size // 2),
            int(center[0] + crop_size // 2),
            int(center[1] + crop_size // 2),
        )

        cell_bin_mask, _ = safe_crop(cell_mask == cell_label, crop_bbox)
        cell_image, _ = safe_crop(fov, crop_bbox)
        assert cell_image.shape[:2] == (crop_size, crop_size), "Size mismatch"
        cv2.imwrite(
            f"{save_folder}/{fov_id}crop_masked_{cell_label}_cell_image.png",
            cell_image * cell_bin_mask[..., None],
        )


def generate_cell_crops(n_jobs=-1, crop_size=1024):
    """Cut every cell listed in the metadata CSV out of its FOV, in parallel over FOVs."""
    data = config.ROOTS["CM4AI_DATA"]
    fov_folder = data / "images" / "all_z_filtered"
    mask_folder = data / "images" / "all_z_filtered_segmentations"
    save_folder = config.ensure(CROPS_DIR)

    df = pd.read_csv(CM4AI_METADATA, usecols=["id", "cell"])
    df["fov_id"] = df["id"].str.replace(r"cell\d+_crop_$", "", regex=True)

    Parallel(n_jobs=n_jobs)(
        delayed(process_single_fov)(
            fov_id, cells["cell"].astype(int).tolist(),
            fov_folder, mask_folder, save_folder, crop_size,
        )
        for fov_id, cells in tqdm(df.groupby("fov_id"))
    )


def get_cross_val_splits(df, n_splits=N_SPLITS):
    """Antibody -> fold table with antibodies sharing a gene in the same fold.

    Antibodies are grouped into connected components of the antibody-gene graph
    (a ``gene_name`` may list several genes joined by ``_``), the groups are
    shuffled with seed 42 and dealt into ``n_splits`` folds. Read from
    ``config.CM4AI_SPLIT``; written there first if absent.
    """
    split_file = config.CM4AI_SPLIT
    if split_file.exists():
        return pd.read_csv(split_file)

    pairs = df[["protein", "gene_name"]].drop_duplicates()
    pairs = pairs.assign(gene=pairs["gene_name"].str.split("_")).explode("gene")
    proteins, protein_idx = np.unique(pairs["protein"], return_inverse=True)
    _, gene_idx = np.unique(pairs["gene"], return_inverse=True)
    incidence = csr_matrix((np.ones(len(pairs)), (protein_idx, gene_idx)))
    n_groups, group_of = connected_components(incidence @ incidence.T, directed=False)
    groups = [proteins[group_of == g].tolist() for g in range(n_groups)]

    random.Random(42).shuffle(groups)

    split_df = pd.DataFrame(
        [
            (protein, fold_idx)
            for fold_idx, split in enumerate(np.array_split(np.arange(n_groups), n_splits))
            for g in split
            for protein in groups[g]
        ],
        columns=["protein", "fold_idx"],
    )
    split_df.to_csv(split_file, index=False)
    return split_df


def fold_prediction(features, treatments, fold_ids, fold_idx, num_splits, parameters, device):
    """Train and score one outer fold: test = ``fold_idx``, validation = the next fold."""
    test_idx = fold_ids == fold_idx
    val_idx = fold_ids == (fold_idx + 1) % num_splits
    train_idx = ~(test_idx | val_idx)

    label_encoder = LabelBinarizer().fit(treatments[train_idx])
    tensors = []
    for idx in [train_idx, val_idx, test_idx]:
        tensors.append(torch.from_numpy(features[idx]).float().to(device))
        tensors.append(
            torch.from_numpy(label_encoder.transform(treatments[idx])).float().to(device)
        )

    fold_results_df = cross_val_mlp_folds([tensors], parameters)[0]
    del tensors
    torch.cuda.empty_cache()
    return fold_results_df


def drug_prediction(all_method_data, save_folder, num_splits, n_jobs=1, folds=None):
    """Run the outer folds of every method, resuming from per-fold result files.

    ``all_method_data`` maps method -> frame with ``fv_*`` feature columns,
    ``treatment`` and ``fold_idx``. When every method has all its folds, the
    best configuration per fold (by macro AP) is written to ``all_results.csv``.
    """
    save_folder = config.ensure(f"{save_folder}/drug_prediction")

    parameters = {
        "n_layers": [1, 2, 3],
        "hidden_dim": [256, 512, 1024],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    }
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]

    method_file = lambda method: save_folder / f"{method}_results.csv"
    fold_file = lambda method, fold_idx: save_folder / f"{method}_fold{fold_idx}_results.csv"

    pending = [method for method in all_method_data if not method_file(method).is_file()]
    inputs = {
        method: (
            data[[c for c in data.columns if c.startswith("fv_")]].values,
            data["treatment"].values,
            data["fold_idx"].values,
        )
        for method, data in all_method_data.items()
        if method in pending
    }
    units = [
        (method, fold_idx)
        for method in pending
        for fold_idx in (range(num_splits) if folds is None else folds)
        if not fold_file(method, fold_idx).is_file()
    ]
    fold_results = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(fold_prediction)(
            *inputs[method], fold_idx, num_splits, parameters, devices[i % len(devices)]
        )
        for i, (method, fold_idx) in enumerate(units)
    )

    for (method, fold_idx), fold_results_df in zip(units, fold_results):
        print(f"{method} fold {fold_idx} done", flush=True)
        fold_results_df["Fold"] = fold_idx
        fold_results_df["Method"] = method
        fold_results_df.to_csv(fold_file(method, fold_idx), index=False)

    for method in pending:
        if all(fold_file(method, f).is_file() for f in range(num_splits)):
            pd.concat(
                [pd.read_csv(fold_file(method, f)) for f in range(num_splits)],
                ignore_index=True,
            ).to_csv(method_file(method), index=False)
    if not all(method_file(method).is_file() for method in all_method_data):
        return

    best_per_fold = []
    for method in all_method_data:
        method_result_df = pd.read_csv(method_file(method))
        best = method_result_df.groupby(["Method", "Fold"])["macro_ap"].idxmax()
        best_per_fold.append(method_result_df.loc[best])
    pd.concat(best_per_fold, ignore_index=True).to_csv(save_folder / "all_results.csv", index=False)


def load_method_data(method, n_splits=N_SPLITS):
    """Single-cell features of *method* with ``fv_*`` columns and a ``fold_idx`` column."""
    df, features = torch.load(
        config.features(method, "b2ai_features"), map_location="cpu", weights_only=False
    )
    keep = df["protein"] != "NEGATIVE"
    df = df[keep].reset_index(drop=True)
    features = features[keep]
    df.loc[:, [f"fv_{i}" for i in range(features.shape[1])]] = features

    split_df = get_cross_val_splits(df, n_splits)
    return df.assign(fold_idx=df["protein"].map(dict(zip(split_df["protein"], split_df["fold_idx"]))))


def main(methods=None, n_jobs=1, folds=None):
    methods = list(config.CM4AI_VARIANTS) if methods is None else methods
    all_method_data = {method: load_method_data(method) for method in methods}
    drug_prediction(
        all_method_data, config.ensure(config.CM4AI_DIR), N_SPLITS, n_jobs=n_jobs, folds=folds
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    crops = sub.add_parser("crops", help="cut masked single-cell crops from the FOV images")
    crops.add_argument("--n-jobs", type=int, default=-1)
    predict = sub.add_parser("predict", help="nested cross-validated drug prediction")
    predict.add_argument("--n-jobs", type=int, default=1)
    predict.add_argument("--methods", nargs="+", default=None, help="default: config.CM4AI_VARIANTS")
    predict.add_argument("--folds", type=int, nargs="+", default=None, help="default: all ten")
    a = ap.parse_args()
    if a.command == "crops":
        generate_cell_crops(n_jobs=a.n_jobs)
    else:
        main(a.methods, a.n_jobs, a.folds)
