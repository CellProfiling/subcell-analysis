"""Fig. 6a,b -- hierarchical Leiden clusters of the image and sequence embeddings on the co-embedding.

One row per clustered space, one panel per level of the hierarchy. Each panel is
the UMAP of the MuSIC co-embedding, coloured by the clusters found in the SubCell
image embedding or in the ESM-2 sequence embedding alone. Clusters that stay
compact on the co-embedding are structure it keeps from that modality.

Colours follow the published panel: ``tab20`` sampled evenly over the clusters of
each level, so above 20 clusters colours repeat and single clusters cannot be told
apart by colour. The panel shows how coherent clusters stay, not which is which.

* ``fig6a_cluster_transfer``: SubCell image clusters on the co-embedding (Fig. 6a).
* ``fig6b_cluster_transfer``: ESM-2 sequence clusters on the co-embedding (Fig. 6b).
* ``fig6ab_cluster_transfer_native``: each space's clusters on its own UMAP, for reference.
* ``fig6ab_cluster_transfer_cross``: image clusters on the sequence UMAP and the reverse.

    python -m figures.fig6ab_cluster_transfer [--method-name NAME] [--in-dir DIR] [--out-dir DIR]

Inputs   ``config.CLUSTER_TRANSFER_DIR``/<method name>/clusters.csv
         (``subcell.multimodal.cluster_transfer``)
Outputs  ``config.figure_dir("fig6")``/fig6a_cluster_transfer, fig6b_cluster_transfer,
         fig6ab_cluster_transfer_native, fig6ab_cluster_transfer_cross
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from subcell import config, style
from subcell.multimodal.embeddings import METHOD_NAME
from subcell.runlog import log
from subcell.style import layout as L
from subcell.style import palettes as pal
from subcell.utils.hierarchical_leiden import RESOLUTIONS

SPACE_LABELS = {"image": "SubCell", "sequence": "ESM-2", "multimodal": "MuSIC\nco-embedding"}

# (clustered space, UMAP space) per row; one row per panel for Fig. 6a,b
PANEL_ROWS = {"fig6a": [("image", "multimodal")], "fig6b": [("sequence", "multimodal")]}
NATIVE_ROWS = [("image", "image"), ("sequence", "sequence")]
CROSS_ROWS = [("image", "sequence"), ("sequence", "image")]

LEFT, GAP, TOP = 25.0, 1.5, 7.0
# Dot diameter as a fraction of panel width, about 0.55 pt on these panels: smaller
# than the published panel's (scanpy size=10, a 3.2 pt dot on 2.9 in panels, 1.5%).
DOT_FRAC = 0.0092
# Display rotation of each space's UMAP, in degrees counter-clockwise. UMAP axes are
# arbitrary and a rotation keeps every distance, so this changes the drawing only,
# not clusters.csv.
ROTATE = {"image": 115, "sequence": 35, "multimodal": 115}


def load(method_name=METHOD_NAME, in_dir=None) -> pd.DataFrame:
    path = (in_dir or config.CLUSTER_TRANSFER_DIR / method_name)
    return pd.read_csv(f"{path}/clusters.csv", index_col=0, dtype={
        f"{s}_leiden_{r}": str for s in SPACE_LABELS for r in RESOLUTIONS})


def rotated(emb, degrees):
    t = np.deg2rad(degrees)
    return emb @ np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])


def tab20(n: int) -> np.ndarray:
    """``tab20`` sampled at *n* even steps, as scanpy colours a categorical by a palette name."""
    return plt.get_cmap("tab20")(np.linspace(0, 1, n))


def render(df, rows):
    w = (L.resolve_width("double") - LEFT - 2.0 - GAP * (len(RESOLUTIONS) - 1)) / len(RESOLUTIONS)
    fig = L.figure("double", height=TOP + len(rows) * (w + 4.0) + 1.0)
    order = L.shuffled(len(df))
    for i, (clustered, space) in enumerate(rows):
        emb = rotated(df[[f"{space}_umap_1", f"{space}_umap_2"]].to_numpy(),
                      ROTATE.get(space, 0))
        d = DOT_FRAC * w * L.MM * 72.0
        y = TOP + i * (w + 4.0)
        L.text(fig, 1.5, y + w / 2,
               f"{SPACE_LABELS[clustered]} clusters\non {SPACE_LABELS[space]}",
               va="center", fontsize=style.FS_SMALL, linespacing=1.3)
        for j, res in enumerate(RESOLUTIONS):
            labels = df[f"{clustered}_leiden_{res}"].astype(int).to_numpy()
            n = labels.max() + 1
            ax = L.axes(fig, LEFT + j * (w + GAP), y, w, w)
            ax.scatter(emb[order, 0], emb[order, 1], s=d ** 2, c=tab20(n)[labels[order]],
                       alpha=0.95, linewidths=0, rasterized=True)
            L.limits(ax, emb)
            L.bare(ax)
            ax.text(1.0, 0.0, f"{n:,}", transform=ax.transAxes, ha="right", va="top",
                    fontsize=style.FS_TINY, color=pal.INK2)
            if i == 0:
                ax.set_title(f"Resolution {res}", fontsize=style.FS_SMALL, pad=2)
    L.text(fig, LEFT, 1.5,
           f"Hierarchical Leiden clusters (number of clusters at lower right), {len(df):,} proteins",
           fontsize=style.FS_SMALL, color=pal.INK2)
    return fig


def main(method_name=METHOD_NAME, in_dir=None, out_dir=None):
    style.use_agg()
    df = load(method_name, in_dir)
    log(f"{len(df):,} proteins from {method_name}")
    out_dir = out_dir or config.figure_dir("fig6")
    formats = ("pdf", "png", "svg")
    with style.nature():
        for panel, rows in PANEL_ROWS.items():
            style.save(render(df, rows), f"{panel}_cluster_transfer", out_dir, formats=formats)
        style.save(render(df, NATIVE_ROWS), "fig6ab_cluster_transfer_native", out_dir, formats=formats)
        style.save(render(df, CROSS_ROWS), "fig6ab_cluster_transfer_cross", out_dir, formats=formats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method-name", default=METHOD_NAME)
    ap.add_argument("--in-dir", default=None,
                    help="folder holding clusters.csv (default: under config.CLUSTER_TRANSFER_DIR)")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    main(a.method_name, a.in_dir, a.out_dir)
