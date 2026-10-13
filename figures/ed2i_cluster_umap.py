"""Extended Data Fig. 2i (right) -- the k = 13 partition on the tissue embedding.

All thirteen clusters on the UMAP of the core-corrected features (the matrix the
clustering was run on), each direct-labelled at its median position, with the
cluster sizes in a key beside it. Cluster ids are the canonical c1..c13 of
``config.TISSUE_CLUSTER_IDS`` (written by ``figures.ed2jk_cluster_annotation``), so
they match panels j and k. Clusters with under ``naming.TISSUE_MIN_CORRECT`` of their
cells scored as correct are excluded from the prevalence tree and marked as such.
Distances between clusters in a UMAP are not evidence of similarity.

    python -m figures.ed2i_cluster_umap [--embedding core] [--out-dir DIR]

Inputs   ``config.TISSUE_UMAPS``, ``config.TISSUE_CLUSTERS``/cell_clusters.csv,
         ``config.TISSUE_CLUSTER_IDS``
Outputs  ``config.figure_dir("ed2")``/ed2i_cluster_umap
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from subcell import config, naming, style
from subcell.runlog import log
from subcell.style import layout as L
from subcell.style import palettes as pal


def load(embedding: str = "core"):
    """Cluster labels, the matching embedding rows and the canonical id map."""
    umaps = dict(np.load(config.TISSUE_UMAPS))
    if embedding not in umaps:
        raise KeyError(f"no '{embedding}' embedding in {config.TISSUE_UMAPS}; "
                       f"have {sorted(umaps)}")
    cells = pd.read_csv(config.TISSUE_CLUSTERS / "cell_clusters.csv",
                        usecols=["row", "cluster"])
    idmap = naming.load_cluster_ids()
    # `row` indexes the full feature matrix, so the embedding is subset by it
    # rather than assumed to be row-aligned with the cluster table.
    emb = umaps[embedding][cells["row"].to_numpy()]
    return emb, cells["cluster"].to_numpy(), idmap


def render(emb, labels, idmap):
    order = list(idmap["source_id"])                 # canonical c1..c13 order
    name = dict(zip(idmap["source_id"], idmap["cluster"]))
    kept = set(idmap.loc[idmap["in_clustering"], "source_id"])
    counts = pd.Series(labels).value_counts()
    cmap = dict(zip(order, pal.categorical(len(order), labelled=True)))

    big_w, y_big = 88.0, 10.0
    fig = L.figure("double", height=y_big + big_w + 3.0)

    codes = pd.Categorical(labels, categories=order).codes
    colors = np.array([cmap[k] for k in order])
    o = L.shuffled(len(emb))

    ax = L.axes(fig, 10.0, y_big, big_w, big_w)
    d = L.marker_size(emb, big_w)
    ax.scatter(emb[o, 0], emb[o, 1], s=d ** 2, c=colors[codes][o], alpha=0.95,
               linewidths=0, rasterized=True)
    cents = L.centroids(emb, codes, len(order))
    for (cx, cy), k in zip(cents, order):
        if np.isfinite(cx):
            L.outlined_text(ax, cx, cy, name[k], cmap[k],
                            fontsize=style.FS_SMALL, stroke=1.4)
    L.limits(ax, emb)
    L.bare(ax)
    L.axis_key(ax, emb)
    L.panel_label(fig, "i", 1.5, 1.5)
    L.text(fig, 10.0, 3.0,
           f"k-means, k = {naming.TISSUE_K} on core-corrected SubCell features, "
           f"{len(emb):,} cells", fontsize=style.FS_SMALL, color=pal.INK2)

    # key beside the embedding
    x_key = 10.0 + big_w + 4.0
    y = y_big + 2.0
    for k in order:
        excluded = k not in kept
        L.swatch(fig, x_key, y, 2.0, 1.6, cmap[k])
        L.text(fig, x_key + 3.5, y + 0.8,
               f"{name[k]}   n = {counts.get(k, 0):,}" + ("   (excluded)" if excluded else ""),
               fontsize=style.FS_TINY, va="center",
               color=pal.MUTED if excluded else pal.INK)
        y += 4.0
    L.text(fig, x_key, y + 2.0,
           f"Excluded: under {naming.TISSUE_MIN_CORRECT:.0%} of cells\nscored as correct, so the\n"
           "cluster is defined by an\nimaging fault rather than\nby cells.",
           fontsize=style.FS_TINY, color=pal.MUTED, linespacing=1.5)
    return fig


def main(out_dir=None, embedding="core"):
    style.use_agg()
    emb, labels, idmap = load(embedding)
    log(f"{len(emb):,} cells · embedding '{embedding}' · {len(idmap)} clusters")
    with style.nature():
        style.save(render(emb, labels, idmap), "ed2i_cluster_umap",
                   out_dir or config.figure_dir("ed2"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--embedding", default="core",
                    help="which harmonized embedding to draw on (default: core, "
                         "the matrix the clustering was run on)")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    main(a.out_dir, a.embedding)
