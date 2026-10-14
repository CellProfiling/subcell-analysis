"""Extended Data Fig. 7a -- within-cluster percentile rank of the image and sequence clusters in each space.

Quantifies Fig. 6a,b (``figures.fig6ab_cluster_transfer``). Each line is one (cluster source ->
scored space) combination: the mean, over clusters, of the mean percentile rank of
a cluster's within-cluster pairs (``subcell.multimodal.cluster_percentile``), with
a band of +- s.d. across clusters. 50 (dotted) is chance; the random baselines are
each source's clusters with labels permuted, scored in the source's own space.

    python -m figures.ed7a_cluster_percentile [--method-name NAME] [--in-dir DIR] [--out-dir DIR]

Inputs   ``config.CLUSTER_TRANSFER_DIR``/<method name>/cluster_percentile_summary.csv
Outputs  ``config.figure_dir("ed7")``/ed7a_cluster_percentile
"""

from __future__ import annotations

import argparse

import pandas as pd

from subcell import config, style
from subcell.multimodal.embeddings import METHOD_NAME
from subcell.runlog import log
from subcell.style import layout as L
from subcell.style import palettes as pal

IMAGE, SEQUENCE, TRANSFER, CROSS, RANDOM = "#5191bc", "#1d3a42", "#d4a017", "#c47a20", "#999999"

# (cluster source, scored space, line colour, marker colour, line style, marker, label)
LINES = [
    ("image", "image", IMAGE, IMAGE, "-", "o", "SubCell → SubCell"),
    ("sequence", "sequence", SEQUENCE, SEQUENCE, "-", "s", "ESM-2 → ESM-2"),
    ("image", "multimodal", TRANSFER, IMAGE, "-", "o", "SubCell → co-embedding"),
    ("sequence", "multimodal", TRANSFER, SEQUENCE, "-", "s", "ESM-2 → co-embedding"),
    ("image", "sequence", CROSS, IMAGE, "-", "D", "SubCell → ESM-2"),
    ("sequence", "image", CROSS, SEQUENCE, "-", "D", "ESM-2 → SubCell"),
    ("random_image_sized", "image", RANDOM, IMAGE, "--", "o", "Random (SubCell sizes)"),
    ("random_sequence_sized", "sequence", RANDOM, SEQUENCE, "--", "s", "Random (ESM-2 sizes)"),
]


def load(method_name=METHOD_NAME, in_dir=None) -> pd.DataFrame:
    path = in_dir or config.CLUSTER_TRANSFER_DIR / method_name
    return pd.read_csv(f"{path}/cluster_percentile_summary.csv")


def render(summary):
    fig = L.figure("single", height=62)
    ax = L.axes(fig, x=11, y=3, w=46, h=48)
    handles = []
    for source, target, line, face, ls, marker, label in LINES:
        sub = summary[(summary["cluster_source"] == source)
                      & (summary["target_space"] == target)].sort_values("resolution")
        x, y, sd = sub["resolution"], sub["mean"], sub["std"]
        ax.fill_between(x, y - sd, y + sd, color=line, alpha=0.1, linewidth=0)
        (h,) = ax.plot(x, y, color=line, ls=ls, lw=style.LW_LINE, marker=marker, ms=2.6,
                       mfc=face, mec=line, mew=style.LW_AXIS, label=label)
        handles.append(h)
    ax.axhline(50, color=pal.MUTED, ls=":", lw=style.LW_AXIS, zorder=0)
    ax.set_ylim(0, 102)
    ax.set_xticks(sorted(summary["resolution"].unique()))
    ax.set_xlabel("Clustering resolution")
    ax.set_ylabel("Mean within-cluster percentile rank")
    L.despine(ax)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
              fontsize=style.FS_TINY, handlelength=2.2)
    return fig


def main(method_name=METHOD_NAME, in_dir=None, out_dir=None):
    style.use_agg()
    summary = load(method_name, in_dir)
    log(f"{len(summary)} (source, space, level) rows from {method_name}")
    with style.nature():
        style.save(render(summary), "ed7a_cluster_percentile", out_dir or config.figure_dir("ed7"),
                   formats=("pdf", "png", "svg"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method-name", default=METHOD_NAME)
    ap.add_argument("--in-dir", default=None,
                    help="folder holding cluster_percentile_summary.csv")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    main(a.method_name, a.in_dir, a.out_dir)
