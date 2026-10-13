"""Fig. 3d and Extended Data Fig. 8c -- macro AP of probes predicting GO / Reactome membership.

One file per database (six panels), a legend and a table of the tests. Fig. 3d
compares the models of ``naming.FIG3_MODELS`` with SubCell as the reference;
Extended Data Fig. 8c adds SubCell-P, last, as the reference the others are tested
against (``naming.ED8_MODELS``). Each violin is the 30 fold values of one model (ten
folds x three fold assignments), at the harmonization that gives the model its best
mean macro AP. The dashed line and grey band are the mean and s.d. of the
random-feature baseline, scored like the models: the best of the 81 probe
configurations per fold, for each of the four random feature widths, pooled.

Paired two-sided Wilcoxon tests compare every model with the reference. A bracket is
drawn when P < 0.05: black when the reference scores higher, in the other model's
colour when that model does.

Inputs: the per-fold results of ``subcell.classification.function_prediction``
(``config.FUNCTION_PREDICTION_REPEATS``). Outputs: ``fig3d_<i>_<database>``,
``fig3d_legend`` and ``fig3d_wilcoxon.csv`` in ``config.figure_dir("fig3")``, and
``ed8c_*`` of the with-SubCell-P set in ``config.figure_dir("ed8")``. The paper
uses all cell lines (the protein and group counts in the Methods are those of the
all-cell-lines inputs); the U2OS variant is written to a ``U2OS`` subfolder.

    python -m figures.fig3d_function_prediction                  # all, then U2OS
    python -m figures.fig3d_function_prediction --cell-line all
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import wilcoxon

from subcell import config
from subcell.runlog import log
from subcell.io.metadata import function_prediction_methods
from subcell.naming import ED8_MODELS, FIG3_MODELS, display
from subcell.style import register_fonts, use_agg
from subcell.style.palettes import FIG3_COLOR
from subcell.style.rc import FIG3_RC
from subcell.utils.metrics import stars


# Panel order and titles; "Level 1" is the coarser hierarchy level.
DATABASES = [
    ("go_bp_level_2", "GO Biological Process\nLevel 1"),
    ("go_bp_level_3", "GO Biological Process\nLevel 2"),
    ("go_cc_level_2", "GO Cellular Component\nLevel 1"),
    ("go_cc_level_3", "GO Cellular Component\nLevel 2"),
    ("reactome_level_0", "Reactome\nLevel 1"),
    ("reactome_level_1", "Reactome\nLevel 2"),
]

# Geometry of the row of six panels, in pixels at DPI: width of the legend row,
# left edge and size of each axes. The panels are written as separate files.
DPI = 600
W_PX = 4873
AXES_LEFT_PX = [271, 1049, 1864, 2638, 3424, 4209]
AXES_WIDTH_PX, AXES_HEIGHT_PX = 660, 391
EDGE = "#3D3D3D"
BAND = "#E5E5E5"
BASELINE = "#808080"


def load(cell_line):
    """Per-fold Macro AP of every model at its best harmonization, and of the
    random baseline: {db: (DataFrame[Method, fold, ap], (mean, sd))}."""
    methods = function_prediction_methods()
    methods = methods[methods["Cell_Line"] == cell_line]
    out = {}
    for db, _ in DATABASES:
        rows, random = [], []
        for repeat, root in enumerate(config.FUNCTION_PREDICTION_REPEATS):
            for _, m in methods.iterrows():
                base = f"{root}/{m.Method_Name}/{db}"
                best = pd.read_csv(f"{base}/Image_Embed_results.csv").groupby("Fold").macro_ap.max()
                rows += [(m.Method, m.Harmonized, f"{f}_{repeat}", v) for f, v in best.items()]
                randoms = [f for f in os.listdir(base) if f.startswith("Random_") and f.endswith("_results.csv")]
                for f in randoms:
                    random += pd.read_csv(f"{base}/{f}").groupby("Fold").macro_ap.max().tolist()
        df = pd.DataFrame(rows, columns=["Method", "Harmonized", "fold", "ap"])
        best_harm = df.groupby(["Method", "Harmonized"]).ap.mean().groupby("Method").idxmax()
        keep = df.set_index(["Method", "Harmonized"]).index.isin(best_harm.tolist())
        out[db] = (df[keep].drop(columns="Harmonized"), (np.mean(random), np.std(random, ddof=1)))
    return out


def darker(colour, f=0.72):
    """A darker shade of a model colour, so a thin bracket in it stays legible."""
    r, g, b = matplotlib.colors.to_rgb(colour)
    return (r * f, g * f, b * f)


def compare(df, models, reference):
    """Two-sided paired Wilcoxon of each model against the reference."""
    wide = df.pivot(index="fold", columns="Method", values="ap")
    res = []
    for m in models:
        if m == reference:
            continue
        d = wide[m] - wide[reference]
        res.append({"Method": m, "mean_diff": d.mean(), "p": wilcoxon(wide[m], wide[reference]).pvalue})
    return pd.DataFrame(res)


def draw_axis(ax, df, random, models, reference, title, ylabel):
    """One database: violins, random band, brackets against the reference."""
    r_mean, r_sd = random
    df = df[df["Method"].isin(models)]
    ax.axhspan(r_mean - r_sd, r_mean + r_sd, color=BAND, ec="#D8D8D8", lw=0.4, zorder=0)
    ax.axhline(r_mean, color=BASELINE, ls="--", lw=0.6, dashes=(3, 2), zorder=1)
    sns.violinplot(
        data=df, x="Method", y="ap", hue="Method", order=models, hue_order=models,
        palette=FIG3_COLOR, saturation=1, linewidth=0.6, linecolor=EDGE,
        inner="point", inner_kws={"s": 0.25, "color": EDGE, "alpha": 0.6},
        legend=False, ax=ax, zorder=2,
    )
    test = compare(df, models, reference)

    # brackets from the reference, nearest first, stacked upwards above the
    # violin outlines
    lo = min(ax.dataLim.y0, r_mean - r_sd)
    span = ax.dataLim.y1 - lo
    y = ax.dataLim.y1 + 0.03 * span
    step, drop = 0.075 * span, 0.02 * span
    ref_x = models.index(reference)
    near_first = test.assign(dist=[abs(models.index(m) - ref_x) for m in test.Method]).sort_values("dist")
    for _, t in near_first.iterrows():
        s = stars(t.p)
        if not s:
            continue
        colour = "black" if t.mean_diff < 0 else darker(FIG3_COLOR[t.Method])
        x = models.index(t.Method)
        ax.plot([ref_x, ref_x, x, x], [y - drop, y, y, y - drop], color=colour, lw=0.6,
                solid_capstyle="butt", zorder=3)
        ax.text((ref_x + x) / 2, y, s, ha="center", va="bottom", fontsize=5,
                color=colour, zorder=3)
        y += step
    ax.set_ylim(lo - 0.02 * span, y + 0.05 * span)
    ax.set_xlabel(title, fontsize=8, labelpad=2.5, linespacing=1.15)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.set_ylabel(ylabel, fontsize=9, labelpad=2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([])
    ax.tick_params(axis="both", pad=1)
    sns.despine(ax=ax)
    return test


def draw_plots(data, models, reference, out, prefix):
    """One file per database, ``<prefix>_<i>_<database>``; the canvas is trimmed to
    what is drawn around the axes."""
    w = AXES_WIDTH_PX / DPI
    h = AXES_HEIGHT_PX / DPI
    stats = []
    with plt.rc_context(FIG3_RC):
        for i, (db, title) in enumerate(DATABASES):
            fig = plt.figure(figsize=(w * 2, h * 2), dpi=DPI)
            ax = fig.add_axes([0.25, 0.25, 0.5, 0.5])
            test = draw_axis(ax, *data[db], models, reference, title, "Macro AP" if i == 0 else "")
            test.insert(0, "Database", title.replace("\n", " "))
            stats.append(test)
            stem = f"{out}/{prefix}_{i + 1}_{db}"
            for ext in ("pdf", "png"):
                fig.savefig(f"{stem}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.01,
                            bbox_extra_artists=[ax.xaxis.label, ax.yaxis.label, *ax.texts])
            plt.close(fig)
    stats = pd.concat(stats, ignore_index=True)
    stats["Method"] = stats["Method"].map(display)
    stats.to_csv(f"{out}/{prefix}_wilcoxon.csv", index=False)


def draw_legend(models, reference, out, prefix):
    """The legend row, and under it what the brackets mean."""
    others = [m for m in models if m != reference]
    with plt.rc_context(FIG3_RC):
        fig = plt.figure(figsize=(W_PX / DPI, 190 / DPI), dpi=DPI)
        handles = [Patch(fc=FIG3_COLOR[m], ec=EDGE, lw=0.5, label=display(m)) for m in models]
        handles.append(Line2D([], [], color=BASELINE, ls="--", lw=1.0, dashes=(3, 2),
                              label="Random Baseline"))
        top = fig.legend(handles=handles, loc="upper left", mode="expand",
                         bbox_to_anchor=(AXES_LEFT_PX[0] / W_PX, 1.0,
                                         (AXES_LEFT_PX[-1] + AXES_WIDTH_PX - AXES_LEFT_PX[0]) / W_PX, 0.0),
                         ncol=len(handles), frameon=False, fontsize=8, handlelength=2.4,
                         handleheight=0.9, handletextpad=0.8, borderaxespad=0.0)
        fig.add_artist(top)
        ref_line = Line2D([], [], color="black", lw=0.8)
        other_lines = tuple(Line2D([], [], color=darker(FIG3_COLOR[m]), lw=0.8) for m in others)
        blank = Line2D([], [], lw=0)
        fig.legend(
            [ref_line, other_lines, blank],
            [f"{display(reference)} higher",
             "other model higher (bracket in its colour)",
             f"two-sided paired Wilcoxon vs {display(reference)}: "
             "* P < 0.05, ** P < 0.01, *** P < 0.001"],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
            loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
            fontsize=6.5, handlelength=2.4, handletextpad=0.6, columnspacing=2.5,
            borderaxespad=0.0,
        )
        for ext in ("pdf", "png"):
            fig.savefig(f"{out}/{prefix}_legend.{ext}", dpi=DPI)
        plt.close(fig)


def main(cell_lines=("all", "U2OS"), out_dir=None):
    use_agg()
    register_fonts()

    def figure_dir(name):
        return config.ensure(os.path.join(out_dir, name)) if out_dir else config.figure_dir(name)

    for cell_line in cell_lines:
        data = load(cell_line)
        for (models, reference), fig_dir, prefix in ((FIG3_MODELS, "fig3", "fig3d"),
                                                      (ED8_MODELS, "ed8", "ed8c")):
            out = figure_dir(fig_dir)
            if cell_line != "all":
                out = config.ensure(f"{out}/{cell_line}")
            draw_plots(data, models, reference, out, prefix)
            draw_legend(models, reference, out, prefix)
            log(f"wrote {out}/{prefix}_*")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell-line", nargs="+", default=["all", "U2OS"], choices=["all", "U2OS"])
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    main(a.cell_line, a.out_dir)
