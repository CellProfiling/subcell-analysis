"""Protein localization probes on HPA features (Fig. 2a-c; Supplementary Table 9).

Per model, an MLP (``subcell.classification.mlp``) is trained on ten seeds on single-cell
features of the HPA v23 train antibodies to predict the multi-label subcellular
localization (31 HPA categories, or the 19 Kaggle challenge categories), selected on
the validation antibodies and scored on the test antibodies at cell and FOV level and,
when its features exist, on the hidden Kaggle single-cell test set.

    python -m subcell.classification.localization --model MAE-CellS-ProtS-Pool
    python -m subcell.classification.localization --model MAE-CellS-ProtS-Pool -uc challenge_cats
    python -m subcell.classification.localization --model MAE-CellS-ProtS-Pool -e   # Kaggle test only

Inputs, under ``config.FEATURE_DIR/<model>``: ``hpa_features/all_features.pth``
(``(metadata, features)`` of the HPA cells), ``kaggle_features/all_features.pth``
(``scripts/export_kaggle_features.py``), the antibody splits ``config.HPA_SPLITS`` and the
category groupings ``config.LOCATION_GROUPS`` (Supplementary Table 2).
Outputs: ``hpa_features/classification/multiclass_<categories>_seed_<i>/`` with the
model, ``{val,test,test_fov,hidden_test}_preds.csv`` and ``*_metrics.csv`` (and plots).
With ``-hf``, features are first Harmony-corrected for cell line, cached as
``hpa_features/all_features_harmonized.pth``, and written to ``classification_aligned``.
"""

import argparse
import functools
import os
import shutil

import numpy as np
import pandas as pd
import torch

from subcell import config
from subcell.classification.mlp import TwoLayerProbe, eval_model, get_metrics, train_mlp
from subcell.runlog import log
from subcell.utils.seed import set_random_seed

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
CATEGORY_SETS = ("all_unique_cats", "challenge_cats")
# HPA annotations too rare to evaluate
EXCLUDED_CATS = ["Cleavage furrow", "Midbody ring", "Rods & Rings", "Microtubule ends"]


@functools.lru_cache(maxsize=None)
def hpa_categories():
    """The 31 HPA localization categories, and ``Negative``."""
    cats = pd.read_csv(config.LOCATION_GROUPS, sep="\t")["Original annotation"].unique().tolist()
    return np.array([c for c in cats if c not in EXCLUDED_CATS] + ["Negative"])


@functools.lru_cache(maxsize=None)
def challenge_groups():
    """HPA annotation -> Kaggle challenge category, where they differ."""
    table = pd.read_csv(config.LOCATION_GROUPS, sep="\t", keep_default_na=False)
    return {o: g for o, g in table[["Original annotation", "Grouping challenge"]].values if g and g != o}


def to_challenge_groups(locations):
    groups = challenge_groups()
    return locations.apply(
        lambda s: ",".join(dict.fromkeys(groups.get(t, t) for t in s.split(",")))
    )


def filter_classes(df, feature_data, unique_cats):
    """One-hot encode ``locations`` over ``unique_cats`` and drop the cells with none of them."""
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
    """Keep the annotated Kaggle cells and name their labels like the HPA ones."""
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


def get_train_val_test_idx(df, feature_data, unique_cats):
    """Features and labels of the train, validation and test antibodies, and the
    FOV of each test cell."""
    idxs = {
        split: df[df["antibody"].isin(
            pd.read_csv(config.HPA_SPLITS / f"{split}_antibodies.txt", header=None)[0]
        )].index.to_list()
        for split in ("train", "valid", "test")
    }

    def subset(split):
        idx = idxs[split]
        return feature_data[idx], torch.from_numpy(df[unique_cats].iloc[idx].values)

    test_fovs = (
        df["if_plate_id"].astype(str) + "_" + df["position"] + "_" + df["sample"].astype(str)
    ).iloc[idxs["test"]].values
    return *subset("train"), *subset("valid"), *subset("test"), test_fovs


def get_fov_results(results, fovs):
    """Cell predictions averaged within each FOV."""
    true_cols = [c for c in results.columns if c.endswith("_true")]
    assert (results.groupby(fovs)[true_cols].nunique() == 1).all().all()
    return (
        results.groupby(fovs)
        .agg({c: "mean" if c.endswith("_pred") else "first" for c in results.columns})
        .reset_index(drop=True)
    )


def get_challenge_data(model_dir, unique_cats):
    """Features and labels of the hidden Kaggle test cells of antibodies not in the
    train or validation split; ``(None, None)`` if the features are not there."""
    path = model_dir / "kaggle_features" / "all_features.pth"
    if not path.is_file():
        return None, None
    challenge_df, challenge_feature_data = torch.load(path, map_location="cpu", weights_only=False)
    seen = set()
    for split in ("train", "valid"):
        seen |= set(pd.read_csv(config.HPA_SPLITS / f"{split}_antibodies.txt", header=None)[0])
    keep_idx = challenge_df[~challenge_df["antibody"].isin(seen)].index.to_numpy()
    challenge_df = challenge_df.loc[keep_idx].reset_index(drop=True)
    challenge_feature_data = challenge_feature_data[keep_idx]
    challenge_df, challenge_feature_data = preprocess_hidden_dataset(
        challenge_df, challenge_feature_data
    )
    if list(unique_cats) == CHALLENGE_CATS:
        challenge_df["locations"] = to_challenge_groups(challenge_df["locations"])
    challenge_df, challenge_feature_data = filter_classes(
        challenge_df, challenge_feature_data, unique_cats=unique_cats
    )
    return challenge_feature_data, torch.from_numpy(challenge_df[unique_cats].values)


def score(x, y, unique_cats, model, seed, device, folder, tag):
    """Predictions of ``model`` on ``(x, y)`` into ``{tag}_preds.csv``, and their metrics."""
    results = eval_model(x, y, unique_cats, model, seed=seed, device=device)
    results = results.filter(regex="^(?!feat_)")
    results.to_csv(f"{folder}/{tag}_preds.csv", index=False)
    get_metrics(folder, results, tag=tag, unique_cats=unique_cats, single_label=False)
    return results


def load_probe(in_features, unique_cats, folder, device):
    """The trained probe saved in ``folder`` by ``train_mlp``."""
    probe = TwoLayerProbe(in_features, len(unique_cats), relu_feature=True)
    probe.load_state_dict(torch.load(f"{folder}/mlp_best_map_final.pth", weights_only=True))
    return probe.to(device)


def load_features(hpa_dir, harmonize_features):
    df, feature_data = torch.load(hpa_dir / "all_features.pth", map_location="cpu", weights_only=False)
    if not harmonize_features:
        return df, feature_data
    cache = hpa_dir / "all_features_harmonized.pth"
    if cache.is_file():
        return torch.load(cache, weights_only=False)
    from harmony import harmonize

    feature_data = harmonize(
        feature_data.numpy(), df, batch_key=["atlas_name"], use_gpu=True, verbose=True, random_state=42
    )
    feature_data = torch.from_numpy(feature_data)
    torch.save((df, feature_data), cache)
    return df, feature_data


def main(model, categories="all_unique_cats", seeds=range(10), harmonize_features=False, eval_only=False):
    """Train (or, with ``eval_only``, just score on the Kaggle test set) the
    localization probes of ``model`` on ``categories`` (one of ``CATEGORY_SETS``)."""
    model_dir = config.FEATURE_DIR / model
    hpa_dir = model_dir / "hpa_features"
    save_folder = hpa_dir / ("classification_aligned" if harmonize_features else "classification")
    os.makedirs(save_folder, exist_ok=True)
    unique_cats = hpa_categories() if categories == "all_unique_cats" else CHALLENGE_CATS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if eval_only:
        challenge_x, challenge_y = get_challenge_data(model_dir, unique_cats)
        if challenge_x is None:
            raise FileNotFoundError(
                f"{model_dir}/kaggle_features/all_features.pth; run scripts/export_kaggle_features.py"
            )
        for i in seeds:
            folder = f"{save_folder}/multiclass_{categories}_seed_{i}"
            probe = load_probe(challenge_x.shape[1], unique_cats, folder, device)
            score(challenge_x, challenge_y, unique_cats, probe, i, device, folder, "hidden_test")
        return

    df, feature_data = load_features(hpa_dir, harmonize_features)
    df.loc[df["locations"].isna(), "locations"] = "Negative"
    if categories != "all_unique_cats":
        df["locations"] = to_challenge_groups(df["locations"])
    df, feature_data = filter_classes(df, feature_data, unique_cats=unique_cats)
    challenge_x, challenge_y = get_challenge_data(model_dir, unique_cats)

    log(f"{len(df)} samples, {len(unique_cats)} categories")

    train_x, train_y, val_x, val_y, test_x, test_y, test_fovs = get_train_val_test_idx(
        df, feature_data, unique_cats
    )
    del feature_data

    for i in seeds:
        set_random_seed(i)
        folder = f"{save_folder}/multiclass_{categories}_seed_{i}"
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)

        probe = train_mlp(
            train_x, train_y, val_x, val_y, device, unique_cats, folder, relu_feature=True
        )
        score(val_x, val_y, unique_cats, probe, i, device, folder, "val")
        test_results = score(test_x, test_y, unique_cats, probe, i, device, folder, "test")
        test_fov_results = get_fov_results(test_results, test_fovs)
        test_fov_results.to_csv(f"{folder}/test_fov_preds.csv", index=False)
        get_metrics(folder, test_fov_results, tag="test_fov", unique_cats=unique_cats, single_label=False)
        if challenge_x is not None:
            score(challenge_x, challenge_y, unique_cats, probe, i, device, folder, "hidden_test")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True,
                    help="model directory under config.FEATURE_DIR, e.g. MAE-CellS-ProtS-Pool")
    ap.add_argument("-hf", "--harmonize_features", action="store_true",
                    help="Harmony-correct the features for cell line first")
    ap.add_argument("-uc", "--unique_cats", choices=CATEGORY_SETS, default="all_unique_cats",
                    help="31 HPA categories (default) or the 19 Kaggle challenge categories")
    ap.add_argument("-s", "--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("-e", "--eval_only", action="store_true",
                    help="score the trained probes on the hidden Kaggle test set only")
    a = ap.parse_args()
    main(a.model, a.unique_cats, a.seeds, a.harmonize_features, a.eval_only)
