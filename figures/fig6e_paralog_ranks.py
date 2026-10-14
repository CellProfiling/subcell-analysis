"""Fig. 6e -- sequence-similar paralog pairs placed by their multimodal similarity.

Protein pairs at their percentile ranks in the sequence (x) and multimodal (y)
embeddings (``subcell.multimodal.paralog_ranks``). Pairs whose proteins share all
their U2OS locations are navy, those sharing none green.

* ``fig6e_overview``: all pairs, the zoomed region outlined.
* ``fig6e_paralog_ranks``: the zoom, the top 2% by sequence, with the paralog pairs
  that share their locations in orange and those that do not in red, each labelled
  at the right.

    python -m figures.fig6e_paralog_ranks [--method-name NAME] [--in-dir DIR] [--out-dir DIR]

Inputs   ``config.PARALOG_RANKS_DIR``/<method name>/{overview,background,paralogs}.csv
Outputs  ``config.figure_dir("fig6")``/fig6e_overview, fig6e_paralog_ranks
"""

from __future__ import annotations

import argparse

import matplotlib.patches as mpatches
import pandas as pd

from subcell import config, style
from subcell.multimodal.embeddings import METHOD_NAME
from subcell.multimodal.paralog_ranks import MIN_MULTIMODAL_PCT, MIN_SEQUENCE_PCT
from subcell.runlog import log
from subcell.style import layout as L

BACKGROUND = {"all": ("lightgray", 0.3, 0.15), "none": ("#9bdf8e", 7.5, 0.12),
              "same": ("navy", 7.5, 0.12)}   # kind -> (colour, marker area, alpha)
OVERVIEW = {"all": ("gray", 0.6, 0.3), "same": ("navy", 6.0, 0.12),
            "none": ("#9bdf8e", 6.0, 0.12)}  # drawn in this order
SAME, DIFFERENT = "#FF8400", "#be2929"         # paralogs sharing their locations or not
BOX = "#d64322"                               # outline of the zoomed region
Y_MIN = MIN_MULTIMODAL_PCT
LABEL_X = 100.5                                # data x of the label column
LABEL_SPACING = 2.5                            # least y gap between labels, in percentile


def load(method_name=METHOD_NAME, in_dir=None):
    path = in_dir or config.PARALOG_RANKS_DIR / method_name
    return [pd.read_csv(f"{path}/{f}.csv") for f in ("overview", "background", "paralogs")]


def render_overview(ov):
    fig = L.figure("single", height=72)
    ax = L.axes(fig, x=12, y=3, w=60, h=60)
    for kind, (color, size, alpha) in OVERVIEW.items():
        o = ov[ov["kind"] == kind]
        ax.scatter(o["sequence_pct"], o["multimodal_pct"], c=color, s=size, alpha=alpha,
                   linewidths=0, rasterized=True, zorder=1)
    ax.plot([0, 100], [0, 100], "k--", lw=style.LW_LINE, alpha=0.8, zorder=4)
    ax.add_patch(mpatches.Rectangle((MIN_SEQUENCE_PCT, Y_MIN), 100 - MIN_SEQUENCE_PCT,
                                    100 - Y_MIN, fill=False, edgecolor=BOX,
                                    lw=style.LW_LINE * 1.5, zorder=15))
    ax.set_xlim(0, 101.5)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Percentile rank, sequence embedding")
    ax.set_ylabel("Percentile rank, multimodal embedding")
    ax.grid(alpha=0.3, lw=style.LW_MIN, zorder=0)
    return fig


def label_heights(y):
    """Label y positions: the points' own, pushed apart to ``LABEL_SPACING``, spread evenly if they overrun."""
    order = sorted(range(len(y)), key=lambda i: y[i])
    out = [0.0] * len(y)
    prev = None
    for i in order:
        out[i] = max(Y_MIN, y[i]) if prev is None else max(y[i], prev + LABEL_SPACING)
        prev = out[i]
    if prev is not None and prev > 99 and len(y) > 1:
        step = (99 - Y_MIN) / (len(y) - 1)
        for rank, i in enumerate(order):
            out[i] = Y_MIN + rank * step
    return out


def render(bg, pairs):
    fig = L.figure("single", height=112)
    ax = L.axes(fig, x=12, y=4, w=44, h=96)
    for kind, (color, size, alpha) in BACKGROUND.items():
        b = bg[bg["kind"] == kind]
        ax.scatter(b["sequence_pct"], b["multimodal_pct"], c=color, s=size, alpha=alpha,
                   linewidths=0, rasterized=True, zorder=1)
    colors = [SAME if s else DIFFERENT for s in pairs["same_locations"]]
    ax.scatter(pairs["sequence_pct"], pairs["multimodal_pct"], c=colors, s=40, marker="X",
               edgecolors="black", linewidths=0.6, alpha=0.85, zorder=7)
    for (_, p), color, y in zip(pairs.iterrows(), colors,
                                label_heights(list(pairs["multimodal_pct"]))):
        ax.plot([p["sequence_pct"], LABEL_X], [p["multimodal_pct"], y], color="black",
                lw=style.LW_AXIS, alpha=0.5, clip_on=False, zorder=3)
        ax.text(LABEL_X, y, p["pair"], fontsize=style.FS_TINY, fontweight="bold", ha="left",
                va="center", clip_on=False, zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color,
                          linewidth=style.LW_AXIS))
    ax.set_xlim(MIN_SEQUENCE_PCT, 100.1)
    ax.set_ylim(Y_MIN, 100)
    ax.set_xticks(range(MIN_SEQUENCE_PCT, 101))
    ax.set_xlabel("Percentile rank, sequence embedding")
    ax.set_ylabel("Percentile rank, multimodal embedding")
    ax.grid(alpha=0.3, lw=style.LW_MIN, zorder=0)
    return fig


def main(method_name=METHOD_NAME, in_dir=None, out_dir=None):
    style.use_agg()
    ov, bg, pairs = load(method_name, in_dir)
    log(f"{len(bg):,} background pairs, {len(pairs)} paralog pairs from {method_name}")
    out_dir = out_dir or config.figure_dir("fig6")
    formats = ("pdf", "png", "svg")
    with style.nature():
        style.save(render_overview(ov), "fig6e_overview", out_dir, formats=formats)
        style.save(render(bg, pairs), "fig6e_paralog_ranks", out_dir, formats=formats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method-name", default=METHOD_NAME)
    ap.add_argument("--in-dir", default=None, help="folder holding overview.csv, background.csv and paralogs.csv")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    main(a.method_name, a.in_dir, a.out_dir)
