"""Attention-head profiles of HPA fields of view, and their subspace overlap.

Feeds Fig. 3f,g, Extended Data Fig. 8f,g and the subspace-overlap numbers in the
Methods. For every cell, ``subcell.inference.attention`` stores the correlation
between each attention head's map and each input channel
(``config.ATTN_CORR_DIR/<run>/all_relations.pth``). Here the cells of a field of
view are summarised by their median correlation profile (``fov_profiles``), the
profiles are harmonized over microscope and embedded with UMAP
(``attention_umap``), and the overlap between the cell-line and localization
subspaces of the harmonized profiles is tested
(``subcell.feature_analysis.subspace_overlap``; results in ``<run>/plots``).

Run::

    python -m subcell.feature_analysis.attention_correlation
"""

import functools
import os

import numpy as np
import pandas as pd
import torch
from harmony import harmonize
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from umap import UMAP

from subcell import config
from subcell.io import metadata
from subcell.feature_analysis.subspace_overlap import run_subspace_entanglement

RUNS = ["dino", "subcell_rybg_mae", "subcell_rybg_vit", "cell_dino_hpa_vitl16"]
CHANNELS = ["MT", "ER", "Nuc", "Prot"]
# head names of the SubCell files, which predate the ``heads`` entry
HEADS = [f"AH-{i+1}" for i in range(12)] + ["PAH-1", "PAH-2"]


@functools.lru_cache(maxsize=1)
def unique_categories():
    """Localization categories of the expert table, in its order."""
    m = pd.read_csv(config.LOCATION_GROUPS, sep="\t")
    return np.array(m["Original annotation"].unique().tolist())


def umap_coordinates(df, feats):
    umap = UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=25,
        min_dist=0.1,
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )
    umap_feats = umap.fit_transform(feats)

    umap_df = pd.DataFrame(
        umap_feats, columns=["UMAP1", "UMAP2"], index=df.index.tolist()
    )
    umap_df["id"] = df["id"]
    umap_df["locations"] = df["locations"]
    umap_df["Location"] = df["Location"]
    umap_df["microscope"] = df["microscope"]
    umap_df["atlas_name"] = df["atlas_name"]
    return umap_df


def fov_profiles(exp_folder, measure="pearson"):
    """Per-FOV median of each cell's [channel x head] attention relation.

    Returns the FOV table (first cell of each FOV, plus its microscope) and the
    profiles, [FOVs, channels * heads] in channel-major order.
    """
    relations = torch.load(
        f"{exp_folder}/all_relations.pth", map_location="cpu", weights_only=False
    )
    df, values = relations["metadata"], relations[measure]
    df["plate_position_sample"] = df["id"].apply(lambda x: x.rsplit("_", 1)[0])

    nonnan_idx = ~df["locations"].isna()
    df = df[nonnan_idx].reset_index(drop=True)
    values = values[nonnan_idx.values]
    df["Location"] = df["locations"].apply(
        lambda x: "Multilocalizing" if "," in x else x
    )

    values = values.reshape(values.shape[0], -1)
    groups = df.groupby("plate_position_sample").groups
    profiles = np.array(
        [np.median(values[idxs], axis=0) for idxs in tqdm(groups.values(), total=len(groups))]
    )
    df = df.groupby("plate_position_sample").first().reset_index()

    # FOVs missing from the annotation table are taken to be M2
    microscope = metadata.microscope_table().set_index("plate_position_sample")["microscope"]
    df["microscope"] = df["plate_position_sample"].map(microscope).fillna("M2")
    return df, profiles


def harmonize_microscope(df, profiles):
    """FOV profiles with the microscope effect removed (Harmony)."""
    return harmonize(
        X=profiles, batch_mat=df, batch_key=["microscope"], use_gpu=True, random_state=42
    )


def attention_umap(exp_folder, df, profiles, measure="pearson", harmonized=None):
    """UMAP of the microscope-harmonized FOV profiles, cached beside the input.

    The cache is recomputed when it is older than ``all_relations.pth``, so a new
    extraction never pairs with coordinates from an old one. *harmonized* is the
    output of ``harmonize_microscope`` if the caller already has it.
    """
    path = f"{exp_folder}/feats/{'corr' if measure == 'pearson' else measure}_umap.csv"
    source = f"{exp_folder}/all_relations.pth"
    if os.path.exists(path) and os.path.getmtime(path) > os.path.getmtime(source):
        return pd.read_csv(path)
    if harmonized is None:
        harmonized = harmonize_microscope(df, profiles)
    umap_df = umap_coordinates(df, harmonized)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    umap_df.to_csv(path, index=False)
    return umap_df


def main(runs=RUNS):
    for run in runs:
        exp_folder = f"{config.ATTN_CORR_DIR}/{run}"
        plots_save_folder = config.ensure(f"{exp_folder}/plots")

        df, mean_corr = fov_profiles(exp_folder, "pearson")
        mean_corr_harm = harmonize_microscope(df, mean_corr)
        attention_umap(exp_folder, df, mean_corr, "pearson", harmonized=mean_corr_harm)

        categories = unique_categories()
        loc_labels = (
            df["locations"]
            .str.get_dummies(sep=",")
            .reindex(columns=categories, fill_value=0)
            .values
        )
        cell_labels = LabelEncoder().fit_transform(df["atlas_name"])

        run_subspace_entanglement(
            Z=mean_corr_harm,
            y_cell=cell_labels,
            y_loc=loc_labels,
            output_dir=str(plots_save_folder),
            loc_names=categories,
            max_rank=10,
            n_perm=1000,
            n_bootstrap=500,
            seed=0,
            n_jobs=-1,
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", default=RUNS,
                    help="subdirectories of config.ATTN_CORR_DIR")
    main(ap.parse_args().runs)
