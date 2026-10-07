"""
Metadata loading and merging for the cpg0000 JUMP Pilot compound benchmark.

Data flow:
    experiment-metadata.tsv      : plate -> cell_type, timepoint, perturbation
    compound_platemap.txt        : well  -> broad_sample (NaN = DMSO)
    compound_metadata.tsv        : broad_sample -> pert_iname, control_type
    usable_moa_metadata.csv      : compound -> moas (list, as string literal)

All 15 compound plates use the same platemap (JUMP-Target-1_compound_platemap),
so the well->compound mapping is identical across plates. The plate-level
attributes (cell_type, timepoint) are the only thing that varies by plate.
"""

import os
from ast import literal_eval

import pandas as pd
import yaml


def load_plate_config(project_root: str = None) -> dict:
    """Load configs/plates.yaml."""
    project_root = project_root or os.environ["PROJECT_ROOT"]
    with open(os.path.join(project_root, "configs", "plates.yaml")) as f:
        return yaml.safe_load(f)


def load_plate_info(project_root: str = None) -> pd.DataFrame:
    """
    Returns DataFrame with one row per plate:
        [plate, cell_type, timepoint, perturbation]
    """
    cfg = load_plate_config(project_root)
    rows = [{"plate": p, **attrs} for p, attrs in cfg["plates"].items()]
    return pd.DataFrame(rows)


def load_well_compound_map(metadata_root: str = None) -> pd.DataFrame:
    """
    Load the compound platemap: well -> broad_sample -> compound info.

    Returns DataFrame with one row per well (384 rows):
        [well, broad_sample, compound, control_type]

    DMSO wells have broad_sample = NaN, compound = "DMSO", control_type = "negcon".
    """
    metadata_root = metadata_root or os.environ["METADATA_ROOT"]

    platemap = pd.read_csv(
        os.path.join(metadata_root, "JUMP-Target-1_compound_platemap.txt"),
        sep="\t",
    )
    # well_position, broad_sample, solvent

    cmpd = pd.read_csv(
        os.path.join(metadata_root, "JUMP-Target-1_compound_metadata.tsv"),
        sep="\t",
    )
    # broad_sample, InChIKey, pert_iname, pubchem_cid, gene, pert_type, control_type, smiles

    df = platemap.merge(
        cmpd[["broad_sample", "pert_iname", "control_type"]],
        on="broad_sample",
        how="left",
    )
    df = df.rename(columns={"well_position": "well", "pert_iname": "compound"})

    # Wells with no broad_sample are DMSO negative controls
    is_dmso = df["broad_sample"].isna()
    df.loc[is_dmso, "compound"] = "DMSO"
    df.loc[is_dmso, "control_type"] = "negcon"

    return df[["well", "broad_sample", "compound", "control_type"]]


def load_moa_map(moa_path: str = None) -> pd.DataFrame:
    """
    Load compound -> MoA mapping.

    Returns DataFrame:
        [compound, moas]
    where moas is a list[str] (already parsed from string literals).

    Only ~192 compounds have "usable" MoAs (MoA class with >=2 members).
    Compounds not in this table will have NaN moas after merge.
    """
    moa_path = moa_path or os.environ["MOA_METADATA"]
    df = pd.read_csv(moa_path)
    df["moas"] = df["moas"].apply(literal_eval)
    return df


def build_full_metadata(
    project_root: str = None,
    metadata_root: str = None,
    moa_path: str = None,
) -> pd.DataFrame:
    """
    Build the complete metadata table: one row per (plate, well).

    Returns DataFrame with columns:
        plate        : str, e.g. "BR00117010"
        well         : str, e.g. "A01"
        cell_type    : str, "A549" or "U2OS"
        timepoint    : int, 24 or 48
        perturbation : str, "compound"
        broad_sample : str or NaN
        compound     : str (pert_iname, or "DMSO" for controls)
        c : str or NaN ("negcon", "poscon_*", or NaN for treatments)
        moas         : list[str] or NaN

    Shape: (n_plates * 384, 9) = (15 * 384, 9) = (5760, 9)
    """
    plates = load_plate_info(project_root)
    wells = load_well_compound_map(metadata_root)
    moas = load_moa_map(moa_path)

    # Cross-join plates with wells (every plate has the same 384-well layout)
    df = plates.merge(wells, how="cross")

    # Merge MoA (left join — many compounds will have NaN)
    df = df.merge(moas, on="compound", how="left")

    return df
