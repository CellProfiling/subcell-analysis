"""Fig. 6c -- similarity enrichment within compartments, per embedding.

Cliff's delta of within- vs across-compartment pair similarity
(``subcell.multimodal.compartment_enrichment``) in the ESM-2 sequence embedding,
the SubCell image embedding and their MuSIC co-embedding.

* ``fig6c_radar``: the 20 Grouping 1 locations of U2OS proteins, grouped by
  compartment (nuclear, cytoplasmic organelles, vesicular, other).
* ``fig6c_suborganellar``: GO:CC compartments within mitochondria, ER, vesicles
  and the Golgi, each organelle's ordered by the co-embedding's delta; error bars
  span the 2.5-97.5 percentiles of the bootstrap draws.

    python -m figures.fig6c_compartment_enrichment [--method-name NAME] [--in-dir DIR] [--out-dir DIR]

Inputs   ``config.COMPARTMENT_ENRICHMENT_DIR``/<method name>/{coarse,suborganellar}_summary.csv
Outputs  ``config.figure_dir("fig6")``/fig6c_radar, fig6c_suborganellar
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from subcell import config, style
from subcell.multimodal.compartment_labels import SUBORGANELLAR_TERMS
from subcell.multimodal.embeddings import METHOD_NAME
from subcell.runlog import log
from subcell.style import layout as L
from subcell.style import palettes as pal

# embedding -> (label, colour); sequence is ESM-2, image SubCell, multimodal their
# MuSIC co-embedding
EMBEDDINGS = {"sequence": ("Sequence", "#1d3a42"), "image": ("Image", "#5191bc"),
              "multimodal": ("Multimodal", "#F4AC45")}

# Radar order, counter-clockwise from the right
LOCATION_ORDER = [
    # nuclear
    "Nucleoplasm", "Nucleoli", "Nuclear speckles", "Nuclear bodies",
    "Nucleoli fibrillar center", "Nuclear membrane",
    # cytoplasmic organelles
    "Cytosol", "Golgi apparatus", "Endoplasmic reticulum", "Mitochondria",
    # vesicular
    "Vesicles", "Lysosomes", "Endosomes",
    # other
    "Plasma membrane", "Actin filaments", "Centrosome", "Microtubules",
    "Intermediate filaments", "Lipid droplets", "Peroxisomes",
]

# GO:CC term -> bar label within its organelle's panel
SHORT_LABELS = {
    "mitochondrial outer membrane": "outer membrane",
    "mitochondrial inner membrane": "inner membrane",
    "mitochondrial intermembrane space": "intermembrane space",
    "mitochondrial matrix": "matrix",
    "endoplasmic reticulum membrane": "membrane", "endoplasmic reticulum lumen": "lumen",
    "vesicle membrane": "membrane", "vesicle lumen": "lumen",
    "Golgi membrane": "membrane", "cis-Golgi network": "cis network",
    "trans-Golgi network": "trans network",
}


def load(label_set, method_name=METHOD_NAME, in_dir=None) -> pd.DataFrame:
    path = in_dir or config.COMPARTMENT_ENRICHMENT_DIR / method_name
    return pd.read_csv(f"{path}/{label_set}_summary.csv")


def radar(df):
    fig = L.figure("single", height=80)
    ax = fig.add_axes([0.24, 0.2, 0.52, 0.52 * 89 / 80], projection="polar")
    angles = np.linspace(0, 2 * np.pi, len(LOCATION_ORDER), endpoint=False)
    closed = np.append(angles, angles[0])
    for emb, (label, color) in EMBEDDINGS.items():
        v = df[df["embedding"] == emb].set_index("compartment").loc[LOCATION_ORDER,
                                                                    "cliffs_delta_mean"]
        v = np.append(v.to_numpy(), v.iloc[0])
        ax.plot(closed, v, "o-", color=color, lw=style.LW_LINE, ms=2.2, mec="white",
                mew=0.3, label=label)
        ax.fill(closed, v, color=color, alpha=0.15, lw=0)
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    for a, name in zip(angles, LOCATION_ORDER):
        c = np.cos(a)
        ax.text(a, 1.07, name, fontsize=style.FS_TINY, va="center",
                ha="left" if c > 0.1 else "right" if c < -0.1 else "center")
    ax.set_ylim(0, 1)
    ax.set_rlabel_position(360 / len(LOCATION_ORDER) / 2)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=style.FS_TINY, colors=pal.INK2)
    ax.grid(color=pal.GRID, lw=style.LW_MIN)
    ax.spines["polar"].set_linewidth(style.LW_AXIS)
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=style.FS_SMALL,
               bbox_to_anchor=(0.5, 0.0))
    L.text(fig, 2, 2, "Cliff's delta, within vs across compartment", fontsize=style.FS_SMALL,
           color=pal.INK2)
    return fig


def suborganellar(df):
    mm = df[df["embedding"] == "multimodal"].set_index("compartment")["cliffs_delta_mean"]
    organelles = {o: sorted(terms, key=lambda t: -mm[t]) for o, terms in SUBORGANELLAR_TERMS.items()}
    n_bars = sum(len(t) for t in organelles.values())
    left, gap, width = 13.0, 10.0, L.resolve_width("double")
    unit = (width - left - 2.0 - gap * (len(organelles) - 1)) / n_bars
    fig = L.figure("double", height=59)
    x0, bar = left, 0.26
    for k, (organelle, terms) in enumerate(organelles.items()):
        ax = L.axes(fig, x0, 10, unit * len(terms), 28)
        x = np.arange(len(terms))
        for i, (emb, (label, color)) in enumerate(EMBEDDINGS.items()):
            e = df[df["embedding"] == emb].set_index("compartment").loc[terms]
            m = e["cliffs_delta_mean"].to_numpy()
            err = [m - e["cliffs_delta_ci_lower"], e["cliffs_delta_ci_upper"] - m]
            ax.bar(x + (i - 1) * bar, m, bar, yerr=err, color=color, label=label,
                   error_kw=dict(lw=style.LW_AXIS, capsize=1.2, capthick=style.LW_AXIS))
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[t] for t in terms], rotation=40, ha="right")
        ax.set_xlim(-0.5, len(terms) - 0.5)
        ax.set_ylim(bottom=0)
        ax.margins(y=0.1)
        ax.set_title(organelle, fontsize=style.FS_BASE, pad=3)
        if k == 0:
            ax.set_ylabel("Cliff's delta")
        L.despine(ax)
        x0 += unit * len(terms) + gap
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=style.FS_SMALL, bbox_to_anchor=(0.5, 1.0))
    return fig


def main(method_name=METHOD_NAME, in_dir=None, out_dir=None):
    style.use_agg()
    coarse, sub = load("coarse", method_name, in_dir), load("suborganellar", method_name, in_dir)
    log(f"{coarse['compartment'].nunique()} locations, {sub['compartment'].nunique()} "
        f"sub-organellar compartments from {method_name}")
    out_dir = out_dir or config.figure_dir("fig6")
    formats = ("pdf", "png", "svg")
    with style.nature():
        style.save(radar(coarse), "fig6c_radar", out_dir, formats=formats)
        style.save(suborganellar(sub), "fig6c_suborganellar", out_dir, formats=formats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method-name", default=METHOD_NAME)
    ap.add_argument("--in-dir", default=None,
                    help="folder holding the *_summary.csv files")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    main(a.method_name, a.in_dir, a.out_dir)
