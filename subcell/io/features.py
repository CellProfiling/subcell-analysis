"""Cell-level features, and the aggregations every analysis starts from.

The models emit one embedding per segmented cell. Which aggregation is correct
depends on the question, and getting it wrong is not visible in the output:

  per gene   for anything compared against a protein-level annotation
  per FOV    for anything compared against a plate, a microscope or a cell line.
             A field of view is a (plate, position, sample) triple carrying one
             antibody, hence one protein and one localization label.

``scale`` is the shared z-score; it adds a small epsilon to the standard
deviation rather than dropping zero-variance dimensions, so a feature matrix keeps
its column count and a cached index stays valid.
"""

import os
import pathlib

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .. import config
from ..runlog import log
from . import metadata

# ------------------------------------------- cell features -> per-FOV means


def feature_path(method, harmonized="none", dataset="hpa_features"):
    """Path to a model's cell-level feature tensor. See :func:`config.features`."""
    return str(config.features(method, dataset, harmonized))


def fov_features(method, harmonized):
    """Per-FOV mean embedding plus its metadata, cached to disk.

    The cell table is up to 1.2M x 2048, so the mean is accumulated with a
    scatter-add rather than a pandas groupby (which would upcast to float64 and
    need ~14GB).
    """
    cache = f"{config.HPA_UMAP_DIR}/fov_features/{method}_{harmonized}.npz"
    meta_cache = cache.replace(".npz", "_meta.parquet")
    if os.path.exists(cache) and os.path.exists(meta_cache):
        with np.load(cache) as z:
            feats = z["feats"]
        log(f"  cached FOV features: {feats.shape}")
        return feats, pd.read_parquet(meta_cache)

    src = feature_path(method, harmonized)
    log(f"  loading {src}")
    df, cell_feats = torch.load(src, map_location="cpu", weights_only=False)
    df = df.reset_index(drop=True)
    assert len(df) == cell_feats.shape[0], "metadata/feature row mismatch"
    log(f"  {len(df):,} cells x {cell_feats.shape[1]} dims")

    df["plate_position_sample"] = (df["if_plate_id"].astype(str) + "_"
                                   + df["position"].astype(str) + "_"
                                   + df["sample"].astype(str))
    codes, fovs = pd.factorize(df["plate_position_sample"], sort=True)

    feats = metadata.group_mean(cell_feats, codes, len(fovs))
    counts = np.bincount(codes, minlength=len(fovs)).astype(np.float64)
    del cell_feats
    log(f"  averaged into {len(fovs):,} FOVs")

    # constant within a FOV, so first() is exact -- asserted, not assumed
    cols = ["atlas_name", "if_plate_id", "antibody", "ensembl_ids", "gene_names",
            "locations"]
    grp = df.groupby("plate_position_sample")
    for col in ["ensembl_ids", "locations"]:
        assert int((grp[col].nunique(dropna=False).reindex(fovs) > 1).sum()) == 0, \
            f"a FOV carries more than one {col}"
    meta = grp[cols].first().reindex(fovs).reset_index(drop=True)
    meta["plate_position_sample"] = fovs
    meta["n_cells"] = counts.astype(int)

    # microscope lives in a separate annotation table; FOVs missing from it
    # cannot be coloured by microscope, so drop them (2 of ~85k)
    micro = metadata.microscope_table()
    meta = pd.merge(meta, micro, on="plate_position_sample", how="left")
    keep = (~meta["microscope"].isna()).values
    if int((~keep).sum()):
        log(f"  dropping {int((~keep).sum())} FOVs with no microscope record")
    feats, meta = feats[keep], meta[keep].reset_index(drop=True)

    config.ensure(pathlib.Path(cache).parent)
    np.savez(cache, feats=feats)
    meta.to_parquet(meta_cache)
    return feats, meta


def scale(features, features_mean=None, features_std=None):
    """Z-score columns; returns ``(scaled, mean, std)``. Pass a fit's mean and std
    to apply it to other data."""
    if features_mean is None:
        features_mean = features.mean(axis=0)
    if features_std is None:
        features_std = features.std(axis=0)
    transformed_features = (features - features_mean) / (features_std + 1e-5)
    return transformed_features, features_mean, features_std


def create_cell_mapping(df, key="ensembl_ids"):
    """Row indices of the cells of each gene; a cell labelled with several
    comma-separated ids counts for each of them."""
    cells_per_ensembl = df.groupby([key]).groups
    keys_list = [k.split(",") for k in cells_per_ensembl.keys()]
    all_keys = [item for sublist in keys_list for item in sublist]
    all_ensembls = sorted(list(set(all_keys)))

    ensembl_dict = {k: [] for k in all_ensembls}
    for k, v in cells_per_ensembl.items():
        k_split = k.split(",")
        if len(k_split) > 1:
            for k_s in k_split:
                ensembl_dict[k_s].extend(v.tolist())
        else:
            ensembl_dict[k].extend(v.tolist())
    return ensembl_dict


def generate_uniprot_mapping(ensembl_dict):
    """Regroup ``{ensembl id: rows}`` by reviewed UniProt entry (``config.ENSG2UNIPROT``)."""
    all_ensembls = list(ensembl_dict.keys())

    map_df = pd.read_csv(config.ENSG2UNIPROT, sep="\t")
    map_df = map_df[map_df["Reviewed"] == "reviewed"]
    map_df = map_df[map_df["From"].isin(all_ensembls)]
    map_dict = map_df.groupby("Entry").agg(list)["From"].to_dict()

    uniprot_dict = {k: [ensembl_dict[x] for x in v] for k, v in map_dict.items()}
    uniprot_dict = {
        k: [item for sublist in v for item in sublist] for k, v in uniprot_dict.items()
    }
    return uniprot_dict


def get_per_gene_mean_features(
    df, features, cell_lines=None, remove_unreliable=True, return_uniprot=True
):
    """Mean cell embedding per gene (or UniProt entry), as ``{id: tensor}``.

    ``cell_lines`` restricts to those ``atlas_name`` values; ``remove_unreliable``
    keeps genes whose HPA reliability is Enhanced, Supported or Approved
    (``config.SUBCELL_LOCATION``); ``return_uniprot`` keys by UniProt entry rather
    than gene name.
    """
    if cell_lines is not None:
        if type(cell_lines) == str:
            cell_lines = [cell_lines]
        keep_idxs = df["atlas_name"].isin(cell_lines)
        df = df[keep_idxs].reset_index(drop=True)
        features = features[keep_idxs]

    if remove_unreliable:
        reliability_df = pd.read_csv(config.SUBCELL_LOCATION, sep="\t")
        reliable_genes = set(
            reliability_df[
                reliability_df["Reliability"].isin(
                    ["Enhanced", "Supported", "Approved"]
                )
            ]["Gene"]
        )
        keep_idxs = df["ensembl_ids"].apply(
            lambda x: True if set(x.split(",")).issubset(reliable_genes) else False
        )
        df = df[keep_idxs].reset_index(drop=True)
        features = features[keep_idxs]

    if return_uniprot:
        ensembl_dict = create_cell_mapping(df, key="ensembl_ids")
        gene_dict = generate_uniprot_mapping(ensembl_dict)
    else:
        gene_dict = create_cell_mapping(df, key="gene_names")

    all_genes = list(gene_dict.keys())
    gene_feat_dict = {k: [] for k in all_genes}
    for k, v in tqdm(gene_dict.items()):
        gene_data = features[v]
        gene_feat_dict[k] = torch.mean(gene_data, dim=0)

    return gene_feat_dict
