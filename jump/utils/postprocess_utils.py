"""
Postprocessing pipeline utilities for well-level embeddings.

All normalization/feature-selection is done via pycytominer. This is a direct
port of the pipeline from sub-cell-embed/ankit_eval/scripts/jump_utils.py.

Pipeline for one (model, cell_type) combination:
    raw well embeddings
      -> [optional] feature_select (variance + correlation threshold)
      -> normalize step 1 (standardize/mad_robustize per-plate, OR spherize on DMSO)
      -> normalize step 2 (same options, or None)
      -> processed embeddings ready for mAP eval

The full postprocessing grid has 18 normalization orderings:
    sphere_then_stand : 4 spherize * 2 standardize = 8
    stand_then_sphere : 2 standardize * 4 spherize = 8
    just_stand        : 2 standardize * None       = 2
"""

import pandas as pd
from pycytominer import feature_select, normalize


SPHERIZE_METHODS = ["ZCA", "ZCA-cor", "PCA", "PCA-cor"]
STANDARDIZE_METHODS = ["mad_robustize", "standardize"]


def get_norm_grid() -> list[tuple]:
    """
    Build the full 18-combination normalization grid.

    Returns list of (step1, step2) tuples where each step is a method name or None.
    """
    sphere_then_stand = [(s, t) for s in SPHERIZE_METHODS for t in STANDARDIZE_METHODS]
    stand_then_sphere = [(t, s) for t in STANDARDIZE_METHODS for s in SPHERIZE_METHODS]
    just_stand = [(t, None) for t in STANDARDIZE_METHODS]
    return sphere_then_stand + stand_then_sphere + just_stand


def feature_select_step(
    df: pd.DataFrame,
    feature_cols: list[str],
    metadata_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop low-variance and highly-correlated features via pycytominer.feature_select.

    Args:
        df: DataFrame with metadata + feature columns.
        feature_cols: List of feature column names.
        metadata_cols: List of metadata column names (preserved through).

    Returns:
        (filtered_df, surviving_feature_cols)
    """
    out = feature_select(
        profiles=df,
        features=feature_cols,
        samples="all",
        image_features=False,
        corr_threshold=0.9,
        corr_method="pearson",
        freq_cut=0.05,
        unique_cut=0.01,
        operation=["variance_threshold", "correlation_threshold"],
    )
    surviving = [c for c in out.columns if c not in metadata_cols]
    return out, surviving


def normalize_step(
    df: pd.DataFrame,
    feature_cols: list[str],
    metadata_cols: list[str],
    method: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply one normalization step via pycytominer.normalize.

    Two families of methods:
      - "mad_robustize" / "standardize": applied per-plate (groupby plate, normalize each)
      - "ZCA" / "ZCA-cor" / "PCA" / "PCA-cor": spherize transform fit on DMSO wells,
        applied globally (batch correction)
      - None: no-op

    Args:
        df: DataFrame with metadata + feature columns. Must have "plate" and
            "compound" columns in metadata_cols (compound needed for DMSO filter).
        feature_cols: List of feature column names.
        metadata_cols: List of metadata column names.
        method: One of the values in SPHERIZE_METHODS, STANDARDIZE_METHODS, or None.

    Returns:
        (normalized_df, feature_cols)
        feature_cols may change after sphering (pycytominer can rename/reduce).
    """
    if method is None:
        return df.copy(), feature_cols

    if method in STANDARDIZE_METHODS:
        # Per-plate normalization
        pieces = []
        for _, plate_df in df.groupby("plate"):
            norm = normalize(
                profiles=plate_df,
                features=feature_cols,
                meta_features=metadata_cols,
                method=method,
                mad_robustize_epsilon=1e-18,
            )
            pieces.append(norm)
        out = pd.concat(pieces, ignore_index=True)
        return out, feature_cols

    if method in SPHERIZE_METHODS:
        # Fit spherize transform on DMSO wells, apply to all
        out = normalize(
            profiles=df,
            features=feature_cols,
            meta_features=metadata_cols,
            samples="compound == 'DMSO'",
            method="spherize",
            spherize_method=method,
            spherize_epsilon=1e-6,
        )
        surviving = [c for c in out.columns if c not in metadata_cols]
        return out, surviving

    raise ValueError(f"Unknown normalization method: {method}")


def apply_pipeline(
    df: pd.DataFrame,
    feature_cols: list[str],
    metadata_cols: list[str],
    do_feature_select: bool,
    norm_step1: str | None,
    norm_step2: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply the full postprocessing pipeline: [feature_select] -> norm1 -> norm2.

    Returns:
        (processed_df, final_feature_cols)
    """
    if do_feature_select:
        df, feature_cols = feature_select_step(df, feature_cols, metadata_cols)

    df, feature_cols = normalize_step(df, feature_cols, metadata_cols, norm_step1)
    df, feature_cols = normalize_step(df, feature_cols, metadata_cols, norm_step2)

    return df, feature_cols
