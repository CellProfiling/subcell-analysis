"""Fig. 3c and Extended Data Fig. 8e -- Mantel statistics of cell-line and localization structure.

Bar plots, one file per statistic, each axis at a fixed size in pixels at the
figure resolution, with fixed colours, fonts and title placement. Cell Lines is the
upper plot, with tick marks but no tick labels; Protein Localization is the lower
one, with the model names and "Method" under it.

Two versions are written, as for Fig. 3d (``figures.fig3d_function_prediction``):
Fig. 3c without SubCell-P (``fig3c_*`` in ``figure_dir("fig3")``) and Extended Data
Fig. 8e with SubCell-P last (``ed8e_*`` in ``figure_dir("ed8")``).

Each bar is one model at the harmonization that gives it the highest statistic,
over all six Harmony targets ("reported the best results for each model", Methods).
The statistics come from ``subcell.feature_analysis.similarity.mantel_table``,
which computes any missing row.

    python -m figures.fig3c_mantel
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

from subcell import config, style
from subcell.style import register_fonts
from subcell.naming import ED8_MODELS, FIG3_MODELS, display
from subcell.style.palettes import FIG3_COLOR
from subcell.style.rc import FIG3_RC

# (statistic, title, axes width x height, title centre and baseline relative to
# the axes' top centre), in pixels at style.DPI.
PANELS = [
    ("RNAseq Mantel Statistic", "Cell Lines", (740, 371.5), (-38, 86)),
    ("Localization Mantel Statistic", "Protein Localization", (728, 366), (-39, 68)),
]
TEXT = "#262626"
BAR_RC = FIG3_RC | {"axes.linewidth": 0.6, "xtick.major.size": 0.7, "ytick.major.size": 0.7}


def best(df, stat, models):
    """Each model at its best harmonization for *stat*, in plotting order."""
    df = df[df["Method"].isin(models)]
    return df.loc[df.groupby("Method")[stat].idxmax()].set_index("Method").loc[models]


def draw(df, models, stat, title, size, title_at, labelled, stem):
    w, h = size
    rows = best(df, stat, models)
    with plt.rc_context(BAR_RC):
        fig = plt.figure(figsize=(3 * w / style.DPI, 3 * h / style.DPI), dpi=style.DPI)
        ax = fig.add_axes([1 / 3, 1 / 3, 1 / 3, 1 / 3])
        x = list(range(len(models)))
        bars = ax.bar(x, rows[stat], width=0.8, color=[FIG3_COLOR[m] for m in models],
                      edgecolor="black", linewidth=0.6)
        ax.bar_label(bars, fmt="%.3f", fontsize=5, color=TEXT, padding=0.6)
        ax.set_xlim(-0.5, len(models) - 0.5)
        ax.set_ylim(0, 1.05 * rows[stat].max())    # matplotlib's default margin
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_ylabel("Mantel Statistic", fontsize=5, color=TEXT, labelpad=1.5)
        ax.set_xticks(x)
        if labelled:
            ax.set_xticklabels([display(m) for m in models], rotation=45, ha="right",
                               rotation_mode="anchor")
            ax.set_xlabel("Method", fontsize=5, color=TEXT, labelpad=1.5)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="y", pad=0.5)
        dx, dy = title_at
        ax.set_title(title, fontsize=8, color="black", x=0.5 + dx / w, pad=dy * 72 / style.DPI)
        ax.spines[["top", "right"]].set_visible(False)
        for ext in ("pdf", "png"):
            fig.savefig(f"{stem}.{ext}", dpi=style.DPI, bbox_inches="tight", pad_inches=0.01,
                        bbox_extra_artists=[ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts])
        plt.close(fig)
    return pd.DataFrame({"Statistic": stat, "Model": [display(m) for m in models],
                         "Harmonized": rows["Harmonized"].values, "Value": rows[stat].values})


def main(out_dir=None, pca_dims: int = 10):
    style.use_agg()
    register_fonts()
    # imported here: the module pulls in torch and the mantel package
    from subcell.feature_analysis import similarity

    df = similarity.mantel_table(str(config.MORPHOLOGY_DIR), pca_dims)
    for prefix, (models, _) in (("fig3c", FIG3_MODELS), ("ed8e", ED8_MODELS)):
        out = config.ensure(out_dir or config.figure_dir("fig3" if prefix == "fig3c" else "ed8"))
        values = []
        for i, (stat, title, size, title_at) in enumerate(PANELS):
            tag = title.lower().replace(" ", "_")
            values.append(draw(df, models, stat, title, size, title_at,
                               labelled=i == len(PANELS) - 1,
                               stem=f"{out}/{prefix}_{tag}"))
        pd.concat(values, ignore_index=True).to_csv(f"{out}/{prefix}_mantel.csv", index=False)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--pca-dims", type=int, default=10)
    a = ap.parse_args()
    main(a.out_dir, a.pca_dims)
