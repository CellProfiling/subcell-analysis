"""Per-gene mean features, raw and after Harmony batch correction.

Feeds Fig. 3d (function prediction) and Extended Data Fig. 3 (harmonization).
For each model's cell-level feature tensor
(``config.features(model)``) it writes, beside the tensor, ``mean_feat_<cell line>[_harmonized_<keys>].csv``
(one row per gene) for U2OS and for all cell lines, and caches each corrected tensor as
``harmonized_features_<keys>.pth``. The index of every table written is recorded in
``config.MEAN_FEATURES_CSV`` with paths relative to ``${SUBCELL_ROOT}``
(readers apply ``config.expand``); rows of models not passed via ``--models`` are
kept.

Batch keys are the imaging plate, the microscope and the antibody atlas name.
``microscope_table`` here derives the microscope from the image width (M1 if
>= 2048 px, else M2); ``subcell.io.metadata.microscope_table`` reads
``HPA_microscope.csv`` instead, and the two name the same partition with M1 and M2
swapped.

Run (GPU needed for Harmony)::

    python -m subcell.feature_analysis.harmonization
"""

import os

import numpy as np
import pandas as pd
import torch
from harmony import harmonize

from subcell import config
from subcell.io.features import get_per_gene_mean_features

HARMONIZE_CATS = [
    ["if_plate_id"],
    ["microscope"],
    ["atlas_name"],
    ["if_plate_id", "microscope"],
    ["if_plate_id", "microscope", "atlas_name"],
]
CELL_LINES = ["U2OS", "all"]
MODELS = ["ViT-ProtS-Pool", "MAE-CellS-ProtS-Pool", "DINO4Cells-HPA", "ViT-Supervised", "bestfitting"]
UNCORRECTED_ONLY = ["bestfitting", "ViT-Supervised"]


def microscope_table():
    meta_df = (
        pd.read_csv(config.IF_IMAGES)
        .groupby(["if_plate_id", "position", "sample", "image_width"])
        .first()
        .reset_index()
    )
    meta_df["microscope"] = meta_df["image_width"].apply(
        lambda x: "M1" if x >= 2048 else "M2"
    )
    return meta_df


def mean_feature_csv(method_save_folder, cell_line, harmonize_cat=None):
    suffix = f"_harmonized_{'_'.join(harmonize_cat)}" if harmonize_cat else ""
    return f"{method_save_folder}/mean_feat_{cell_line}{suffix}.csv"


def write_mean_features(df, features, path, cell_line):
    if os.path.exists(path):
        print(f"{path} already exists.")
        return
    feat_dict = get_per_gene_mean_features(
        df, features, cell_lines=cell_line if cell_line != "all" else None
    )
    # via rename, so a reader never sees a half-written table under this name
    pd.DataFrame(feat_dict).T.to_csv(f"{path}.tmp")
    os.replace(f"{path}.tmp", path)


def harmonize_method(dir_path, harmonize_cats, meta_df, cell_lines=CELL_LINES):
    """Per-gene mean features, raw and after each Harmony correction.

    Returns {key: csv path}, keyed as in ``config.MEAN_FEATURES_CSV``.
    """
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(dir_path)))
    method_save_folder = os.path.dirname(dir_path)
    paths = {}

    # memory-mapped: the ~9 GB tensor is paged in from disk rather than held
    df, feature_data = torch.load(
        dir_path, map_location="cpu", weights_only=False, mmap=True
    )

    for cell_line in cell_lines:
        path = mean_feature_csv(method_save_folder, cell_line)
        write_mean_features(df, feature_data, path, cell_line)
        paths[f"{exp_name}_{cell_line}"] = path

    df1 = pd.merge(
        df,
        meta_df[["if_plate_id", "position", "sample", "microscope"]],
        on=["if_plate_id", "position", "sample"],
        how="left",
    )
    df1["microscope"] = df1["microscope"].fillna("M1")

    for harmonize_cat in harmonize_cats:
        harm_path = (
            f"{method_save_folder}/harmonized_features_{'_'.join(harmonize_cat)}.pth"
        )
        csv_paths = {
            cell_line: mean_feature_csv(method_save_folder, cell_line, harmonize_cat)
            for cell_line in cell_lines
        }
        if all(os.path.exists(p) for p in csv_paths.values()):
            print(f"Mean features for {exp_name} harmonized on {harmonize_cat} already exist.")
        elif os.path.exists(harm_path):
            print(
                f"Harmonized features for {exp_name} with categories {harmonize_cat} already exist."
            )
            df, harmonized_features = torch.load(
                harm_path, map_location="cpu", weights_only=False
            )
        else:
            print(f"Harmonizing {exp_name} with categories {harmonize_cat}")
            harmonized_features = harmonize(
                X=feature_data.numpy(),
                batch_mat=df1,
                batch_key=harmonize_cat,
                use_gpu=True,
                verbose=True,
                random_state=42,
            )
            harmonized_features = torch.from_numpy(harmonized_features.astype(np.float32, copy=False))
            torch.save((df, harmonized_features), harm_path)

        for cell_line, path in csv_paths.items():
            if not os.path.exists(path):
                write_mean_features(df, harmonized_features, path, cell_line)
            paths[f"{exp_name}_{cell_line}_harmonized_{'_'.join(harmonize_cat)}"] = path

    return paths


def main(models=MODELS):
    meta_df = microscope_table()
    paths = {}
    for model in models:
        paths.update(
            harmonize_method(
                str(config.features(model)),
                [] if model in UNCORRECTED_ONLY else HARMONIZE_CATS,
                meta_df,
            )
        )

    root = config.ROOT
    new = pd.DataFrame.from_dict(
        {k: "${SUBCELL_ROOT}/" + os.path.relpath(p, root).replace(os.sep, "/")
         for k, p in paths.items()},
        orient="index", columns=["path"],
    )
    new.index.name = "experiment_cell_line"
    # rows of models not processed in this call are kept; processed ones are replaced
    if config.MEAN_FEATURES_CSV.exists():
        old = pd.read_csv(config.MEAN_FEATURES_CSV, index_col="experiment_cell_line")
        merged = old.reindex(old.index.union(new.index, sort=False))
        merged.loc[new.index, "path"] = new["path"]
        new = merged
    config.ensure(config.MEAN_FEATURES_CSV.parent)
    new.to_csv(config.MEAN_FEATURES_CSV, index=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", nargs="+", default=MODELS)
    main(p.parse_args().models)
