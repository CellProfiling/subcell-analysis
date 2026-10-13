"""The annotation tables, loaded once and filtered consistently.

``methods_table`` reads ``methods_info.csv`` (one row per Method, Cell_Line,
Harmonized), drops duplicate rows and selects ``Cell_Line == "all"`` so every
analysis evaluates the same set of models. The other helpers load the FOV ->
microscope table, the tissue per-cell metadata, multi-label annotation sets and
per-group means.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from .. import config, naming


@functools.lru_cache(maxsize=1)
def _methods_csv() -> pd.DataFrame:
    df = pd.read_csv(config.METHODS_CSV)
    paths = [c for c in df.columns if c.endswith("_Embed")]
    df[paths] = df[paths].map(lambda v: config.expand(v) if isinstance(v, str) else v)
    return df


def read_methods_csv() -> pd.DataFrame:
    """``methods_info.csv`` with its embedding paths resolved against the roots."""
    return _methods_csv().copy()


def methods_table(cell_line: str = "all", drop_supervised: bool = False,
                  harmonized: str | None = None) -> pd.DataFrame:
    """The (Method, Cell_Line, Harmonized) rows to evaluate or plot.

    ``cell_line="all"`` selects the models fitted across every HPA cell line; the
    per-cell-line variants are a separate experiment and are never mixed in.
    *drop_supervised* removes ViT-Supervised and bestfitting, which were trained
    on the HPA localization labels and so cannot be scored on label recovery
    without circularity. *harmonized* pins a single Harmony batch key.
    """
    df = read_methods_csv()
    df = df.drop_duplicates(["Method", "Cell_Line", "Harmonized"])
    df = df[df["Cell_Line"] == cell_line]
    if drop_supervised:
        df = df[~df["Method"].isin(naming.SUPERVISED)]
    if harmonized is not None:
        df = df[df["Harmonized"] == harmonized]
    return df.reset_index(drop=True)


def function_prediction_methods() -> pd.DataFrame:
    """The (Method, Cell_Line, Harmonized) rows of the GO / Reactome function
    prediction (Fig. 3d, Extended Data Fig. 8c).

    ``Method_Name`` is the results directory; ``Image_Embed`` the per-gene mean
    feature table written by ``subcell.feature_analysis.harmonization``.
    """
    rows = []
    for method, harmonizations in config.FUNCTION_PREDICTION_MODELS.items():
        for cell_line in config.FUNCTION_PREDICTION_CELL_LINES:
            for harmonized in harmonizations:
                suffix = "" if harmonized == "none" else f"_harmonized_{harmonized}"
                rows.append({
                    "Method": method,
                    "Method_Name": f"{method}_{cell_line}{suffix}",
                    "Cell_Line": cell_line,
                    "Harmonized": harmonized,
                    "Image_Embed": str(config.FEATURE_DIR / method / "hpa_features"
                                       / f"mean_feat_{cell_line}{suffix}.csv"),
                })
    df = pd.DataFrame(rows)
    return df.sort_values(["Method", "Cell_Line", "Harmonized"]).reset_index(drop=True)


@functools.lru_cache(maxsize=2)
def microscope_table() -> pd.DataFrame:
    """FOV -> imaging instrument, keyed on ``plate_position_sample``.

    A separate HPA annotation table, so a handful of FOVs present in the feature
    tensor are missing from it and cannot be coloured by microscope. Callers drop
    those rather than filling them, since an imputed instrument would appear in a
    batch-effect figure as if it were measured.
    """
    df = pd.read_csv(config.MICROSCOPE_CSV)[["plate_position_sample", "microscope"]]
    if not df["plate_position_sample"].is_unique:
        raise ValueError(f"{config.MICROSCOPE_CSV} has duplicate FOV keys")
    return df


def tissue_metadata() -> pd.DataFrame:
    """Per-cell tissue metadata as written by the harmonisation run.

    Columns: ``id, slide, core, fov, core_id, cohort, patient, tissue, cancer``.
    One row per cell, in the row order of the harmonised feature matrices and the
    cached UMAP coordinates, so it may be indexed against either without a join.
    """
    return pd.read_csv(config.TISSUE_META)


def label_sets(meta: pd.DataFrame, column: str) -> list[set]:
    """A comma-separated annotation column as one label set per row.

    Both label columns used for retrieval are genuinely multi-label:

      * ``ensembl_ids`` -- 3.3% of FOVs use an antibody targeting several genes at
        once (HPA051757 -> ANKRD36 / ANKRD36B / ANKRD36C), and 172 genes appear in
        more than one such group.
      * ``locations`` -- roughly half of all FOVs are annotated to more than one
        compartment.

    Treating either raw string as an atomic label would score two FOVs that share
    a gene, or share a compartment, as a negative pair. An unannotated FOV gets
    the empty set: it can never be positive for anyone and is disjoint from
    everything, so it stays in the pool as a valid negative rather than being
    dropped, which would change the negative pool between models.
    """
    raw = meta[column].astype(str)
    empty = meta[column].isna() | raw.isin(["nan", "", "Negative"])
    return [set() if bad else {t.strip() for t in v.split(",") if t.strip()}
            for v, bad in zip(raw, empty)]


def group_mean(features, group_codes, n_groups: int) -> np.ndarray:
    """Per-group mean of a feature matrix, without going through pandas.

    The cell-level table is up to 1.2M x 2048; a pandas groupby upcasts it to
    float64 (~20 GB) before reducing. A scatter-add accumulates in float64 but
    keeps the input in float32, which fits.
    """
    feats = np.asarray(features, dtype=np.float32)
    sums = np.zeros((n_groups, feats.shape[1]), dtype=np.float64)
    np.add.at(sums, group_codes, feats)
    counts = np.bincount(group_codes, minlength=n_groups).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"{int((counts == 0).sum())} of {n_groups} groups are empty")
    return (sums / counts[:, None]).astype(np.float32)
