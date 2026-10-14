"""Fig. 6f -- the GO:BP processes whose proteins the multimodal embedding draws closest together.

Cliff's delta of within- vs across-process pair similarity
(``subcell.multimodal.compartment_enrichment``, label set ``gobp``) in the sequence,
image and multimodal embeddings, for the 25 processes with the highest multimodal
delta, in that order; error bars span the 2.5-97.5 percentiles of the bootstrap
draws.

    python -m figures.fig6f_process_enrichment [--method-name NAME] [--in-dir DIR] [--out-dir DIR]

Inputs   ``config.COMPARTMENT_ENRICHMENT_DIR``/<method name>/gobp_summary.csv
Outputs  ``config.figure_dir("fig6")``/fig6f_process_enrichment
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from subcell import config, style
from subcell.multimodal.embeddings import METHOD_NAME
from subcell.runlog import log
from subcell.style import layout as L

# embedding -> (label, colour); sequence is ESM-2, image SubCell, multimodal their
# MuSIC co-embedding
EMBEDDINGS = {"sequence": ("Sequence", "#1d3a42"), "image": ("Image", "#5191bc"),
              "multimodal": ("Multimodal", "#F4AC45")}
N_TOP = 25


def load(method_name=METHOD_NAME, in_dir=None) -> pd.DataFrame:
    path = in_dir or config.COMPARTMENT_ENRICHMENT_DIR / method_name
    return pd.read_csv(f"{path}/gobp_summary.csv")


def ranked(df) -> list[str]:
    """Processes by multimodal delta, highest first."""
    mm = df[df["embedding"] == "multimodal"].sort_values("cliffs_delta_mean", ascending=False)
    return mm["compartment"].tolist()


def render(df, processes):
    fig = L.figure("double", height=108)
    ax = L.axes(fig, x=40, y=4, w=139, h=40)
    x = np.arange(len(processes))
    for emb, (label, color) in EMBEDDINGS.items():
        e = df[df["embedding"] == emb].set_index("compartment").loc[processes]
        m = e["cliffs_delta_mean"].to_numpy()
        ax.errorbar(x, m, yerr=[m - e["cliffs_delta_ci_lower"], e["cliffs_delta_ci_upper"] - m],
                    fmt="none", ecolor="black", elinewidth=style.LW_AXIS, capsize=1.5,
                    capthick=style.LW_AXIS, alpha=0.8, zorder=3)
        ax.scatter(x, m, s=5, color=color, linewidths=0, label=label, zorder=4)
    ax.axhline(0, color="black", ls="--", lw=style.LW_AXIS, alpha=0.5, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(processes, rotation=45, ha="right", rotation_mode="anchor",
                       fontsize=style.FS_TINY)
    ax.set_xlim(-0.6, len(processes) - 0.4)
    ax.set_ylabel("Cliff's delta")
    ax.legend(loc="upper right", frameon=False, fontsize=style.FS_SMALL, handletextpad=0.2)
    L.despine(ax)
    return fig


def main(method_name=METHOD_NAME, in_dir=None, out_dir=None):
    style.use_agg()
    df = load(method_name, in_dir)
    order = ranked(df)
    log(f"{len(order)} processes from {method_name}; plotting the top {N_TOP}")
    with style.nature():
        style.save(render(df, order[:N_TOP]), "fig6f_process_enrichment",
                   out_dir or config.figure_dir("fig6"), formats=("pdf", "png", "svg"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method-name", default=METHOD_NAME)
    ap.add_argument("--in-dir", default=None, help="folder holding gobp_summary.csv")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    main(a.method_name, a.in_dir, a.out_dir)
