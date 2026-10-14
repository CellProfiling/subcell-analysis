"""Extended Data Fig. 7c -- the remaining GO:BP processes, continuing the ranking of Fig. 6f.

Cliff's delta of within- vs across-process pair similarity in the sequence, image
and multimodal embeddings for the processes after Fig. 6f's top 25, by multimodal
delta, 27 per panel; error bars span the 2.5-97.5 percentiles of the bootstrap
draws. Every panel shares one y-axis range.

    python -m figures.ed7c_process_enrichment [--method-name NAME] [--in-dir DIR] [--out-dir DIR]

Inputs   ``config.COMPARTMENT_ENRICHMENT_DIR``/<method name>/gobp_summary.csv
Outputs  ``config.figure_dir("ed7")``/ed7c_process_enrichment_<first>-<last>, one per panel
"""

from __future__ import annotations

import argparse
import textwrap

import numpy as np

from subcell import config, style
from subcell.multimodal.embeddings import METHOD_NAME
from subcell.runlog import log
from subcell.style import layout as L
from figures.fig6f_process_enrichment import EMBEDDINGS, N_TOP, load, ranked

PER_PANEL = 27
WRAP = 34           # characters per line of a process name


def render(df, processes, ylim):
    fig = L.figure(L.ED_WIDTH, height=62)
    ax = L.axes(fig, x=12, y=3, w=L.ED_WIDTH - 14, h=24)
    x = np.arange(len(processes))
    for emb, (label, color) in EMBEDDINGS.items():
        e = df[df["embedding"] == emb].set_index("compartment").loc[processes]
        m = e["cliffs_delta_mean"].to_numpy()
        ax.errorbar(x, m, yerr=[m - e["cliffs_delta_ci_lower"], e["cliffs_delta_ci_upper"] - m],
                    fmt="none", ecolor="black", elinewidth=style.LW_AXIS, capsize=1.2,
                    capthick=style.LW_AXIS, alpha=0.8, zorder=3)
        ax.scatter(x, m, s=5, color=color, linewidths=0, label=label, zorder=4)
    ax.axhline(0, color="black", ls="--", lw=style.LW_AXIS, alpha=0.5, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(p, WRAP) for p in processes], rotation=90, ha="center",
                       va="top", multialignment="center", fontsize=style.FS_TINY,
                       linespacing=1.0)
    ax.set_xlim(-0.6, PER_PANEL - 0.4)   # equal spacing on every panel, the last included
    ax.set_ylim(*ylim)
    ax.set_ylabel("Cliff's delta")
    ax.legend(loc="upper right", frameon=False, fontsize=style.FS_SMALL, handletextpad=0.2,
              ncol=3)
    L.despine(ax)
    return fig


def main(method_name=METHOD_NAME, in_dir=None, out_dir=None):
    style.use_agg()
    df = load(method_name, in_dir)
    rest = ranked(df)[N_TOP:]
    shown = df[df["compartment"].isin(rest)]
    lo, hi = shown["cliffs_delta_ci_lower"].min(), shown["cliffs_delta_ci_upper"].max()
    ylim = (min(0.0, lo) - 0.05, hi + 0.1)   # headroom for the legend
    log(f"{len(rest)} processes after the top {N_TOP}, {PER_PANEL} per panel")
    out_dir = out_dir or config.figure_dir("ed7")
    with style.nature():
        for start in range(0, len(rest), PER_PANEL):
            panel = rest[start:start + PER_PANEL]
            first, last = N_TOP + start + 1, N_TOP + start + len(panel)
            style.save(render(df, panel, ylim), f"ed7c_process_enrichment_{first}-{last}",
                       out_dir, formats=("pdf", "png", "svg"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method-name", default=METHOD_NAME)
    ap.add_argument("--in-dir", default=None, help="folder holding gobp_summary.csv")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    main(a.method_name, a.in_dir, a.out_dir)
