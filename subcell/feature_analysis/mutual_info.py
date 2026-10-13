"""Mutual information between each feature dimension and the HPA annotations.

Feeds Extended Data Fig. 3f. For every model and Harmony regime, each dimension of
the standardized single-localization FOV features is binned (50 x 50) against
three factors, localization, cell line and well (plate and position), and its
mutual information with each is recorded. A dimension is attributed to whichever
factor it shares the most information with.

Inputs: ``config.features(model, harmonized=...)`` for every (Method, Harmonized)
row of ``methods_info.csv`` plus Cell-DINO-HPA.
Outputs, in ``config.MUTUAL_INFO_DIR``: ``<method>_<harmonized>_mi_df.csv`` (one
row per dimension; existing tables are reused) and ``dominant_factor_summary.csv``
(percent and count of dimensions dominated by each factor).

Run::

    python -m subcell.feature_analysis.mutual_info
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from subcell import config
from subcell.io.features import scale
from subcell.io.metadata import methods_table
from subcell.naming import HARM_SHORT, display
from subcell.runlog import log

HARMONIZES = ["none", "microscope", "atlas_name", "if_plate_id"]
CODE_BINS = 50
FACTOR_BINS = 50


def mutual_info(x, y, code_bin, factor_bin):
    """Mutual information (nats) between two variables from a 2-D histogram."""
    c_xy = np.histogram2d(x, y, (code_bin, factor_bin))[0]
    mi = 0
    N = np.sum(c_xy)
    for i in range(c_xy.shape[0]):
        for j in range(c_xy.shape[1]):
            p_i = np.sum(c_xy[i, :]) / N
            p_j = np.sum(c_xy[:, j]) / N
            p_ij = c_xy[i, j] / N
            if p_ij != 0:
                mi += p_ij * np.log(p_ij / (p_i * p_j))
    return mi


def models_and_regimes():
    """(Method, Harmonized) rows: the methods table plus Cell-DINO-HPA."""
    table = methods_table()
    cell_dino = pd.DataFrame(
        {"Method": "Cell-DINO-HPA", "Harmonized": table["Harmonized"].unique()}
    )
    return pd.concat([table, cell_dino], ignore_index=True)[["Method", "Harmonized"]]


def dimension_mi(method, harmonized):
    """Per-dimension MI with localization, cell line and well for one feature matrix."""
    df, feature_data = torch.load(
        config.features(method, harmonized=harmonized),
        map_location="cpu", weights_only=False,
    )
    df.loc[:, "Location"] = df["locations"].fillna("Negative")
    single_loc = df["Location"].apply(lambda x: "," not in x)
    df = df[single_loc].reset_index(drop=True)
    feature_data = feature_data[single_loc]
    feature_data, _, _ = scale(feature_data)

    df["well_ids"] = df["if_plate_id"].astype(str) + "_" + df["position"].astype(str)
    factors = {
        col: LabelEncoder().fit_transform(df[src].values)
        for col, src in [("location_mi", "Location"), ("cell_mi", "atlas_name"),
                         ("well_mi", "well_ids")]
    }
    return pd.DataFrame({
        col: [mutual_info(feature_data[:, d], codes, CODE_BINS, FACTOR_BINS)
              for d in tqdm(range(feature_data.shape[1]), desc=col)]
        for col, codes in factors.items()
    })


def dominant_counts(mi_df):
    """Dimensions where localization, cell line or well has the highest MI."""
    mi = mi_df.values.T
    loc, cell, well = mi
    return [int(((loc > cell) & (loc > well)).sum()),
            int(((cell > loc) & (cell > well)).sum()),
            int(((well > loc) & (well > cell)).sum())]


def main():
    save_folder = config.ensure(config.MUTUAL_INFO_DIR)
    rows = []
    for method, harmonized in models_and_regimes().itertuples(index=False):
        path = save_folder / f"{method}_{harmonized}_mi_df.csv"
        if path.exists():
            mi_df = pd.read_csv(path)
        else:
            log(f"Processing {method} and {harmonized}")
            mi_df = dimension_mi(method, harmonized)
            mi_df.to_csv(path, index=False)
        counts = dominant_counts(mi_df)
        n = sum(counts)
        rows.append({
            "method": method,
            "display": display(method),
            "harmonized": harmonized,
            # newlines are for figure tick labels, not for a csv cell
            "harmonized_label": HARM_SHORT.get(harmonized, harmonized).replace("\n", " "),
            "pct_localization": 100 * counts[0] / n,
            "pct_cell_line": 100 * counts[1] / n,
            "pct_well": 100 * counts[2] / n,
            "n_localization": counts[0],
            "n_cell_line": counts[1],
            "n_well": counts[2],
            "n_dimensions": n,
        })

    summary = pd.DataFrame(rows)
    # ordered as the figures read: method, then increasing harmonization
    summary["_h"] = summary["harmonized"].apply(
        lambda h: HARMONIZES.index(h) if h in HARMONIZES else len(HARMONIZES)
    )
    summary = summary.sort_values(["method", "_h"]).drop(columns="_h")
    summary.to_csv(save_folder / "dominant_factor_summary.csv", index=False)
    log(f"wrote {save_folder / 'dominant_factor_summary.csv'}")
    return summary


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    main()
