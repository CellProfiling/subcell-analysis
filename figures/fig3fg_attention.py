"""Fig. 3f,g and Extended Data Fig. 8f,g -- what the attention heads look at.

f  Pearson correlation between each attention head's map and the protein
   channel, averaged over the cells of each single localization, clustered on both
   axes (average linkage, cosine distance), with the major location groups as a
   colour strip. Only localizations with at least ``MIN_FOVS`` (10) FOVs are
   shown; this drops Microtubule ends (6 FOVs). Each model's panel is scaled over
   its own range: a shared one, stretched to SubCell-P's negative values, would
   wash out SubCell.
g  UMAP of the FOV attention profiles (each cell's correlation of every head with
   all four input channels, median per FOV, harmonized on microscope), coloured by
   cell line and by protein localization, labelled at category medians.

SubCell is drawn for Fig. 3 (``fig3f_SubCell``, ``fig3g_SubCell`` in
``figure_dir("fig3")``), SubCell-P for Extended Data Fig. 8 (``ed8f_SubCell-P``,
``ed8g_SubCell-P`` in ``figure_dir("ed8")``) and Cell-DINO-HPA, which has 16
last-layer heads and no pooling heads, beside SubCell (``fig3f_Cell-DINO-HPA``,
``fig3g_Cell-DINO-HPA``). The inputs are
``config.ATTN_CORR_DIR/<run>/all_relations.pth`` from ``subcell.inference.attention``;
the FOV profiles and the UMAP come from ``subcell.feature_analysis.attention_correlation``.
Type is Arial (or its metric twin) at 5-7 pt, strokes are 0.25-1 pt, sizes are in
millimetres, and every panel is written as PDF and JPEG.

    python -m figures.fig3fg_attention
"""

from __future__ import annotations

import argparse

import colorcet as cc
import numpy as np
import pandas as pd
import torch
from adjustText import adjust_text
from matplotlib.ticker import MultipleLocator
from scipy.cluster.hierarchy import dendrogram, linkage

from subcell import config, style
from subcell.runlog import log
from subcell.style import layout as L
from subcell.naming import display

RUNS = {"MAE-CellS-ProtS-Pool": "subcell_rybg_mae", "ViT-ProtS-Pool": "subcell_rybg_vit",
        "Cell-DINO-HPA": "cell_dino_hpa_vitl16"}
MEASURE = "pearson"
FORMATS = ("pdf", "jpg")
ROW = 1.8                     # mm per heatmap row: 5 pt labels do not touch
COL = 2.6                     # mm per heatmap column
DENDRO = "#6E6E6E"
MULTI = "Multilocalizing"
MIN_FOVS = 10
GROUP_COLOURS = {"Endomembrane system": "#05ACC6", "Mitochondria": "#97FF00",
                 "Cytosol": "#B500FF", "Cytoskeleton": "#D60000",
                 "Plasma membrane": "#018700", "Nucleoli": "#FF8EC8", "Nucleus": "#FFA52F"}
# the categories labelled on panel g, under their display spellings
CELL_LINE_LABELS = {"REH": "REH", "HEL": "HEL", "Hep-G2": "Hep-G2", "PC-3": "PC-3",
                    "HEK293": "HEK-293", "Rh30": "RH-30", "U-251MG": "U-251MG",
                    "U2OS": "U2OS", "A-431": "A-431"}
LOCATION_LABELS = {"Plasma membrane": "Plasma Membrane", "Vesicles": "Vesicles",
                   "Golgi apparatus": "Golgi Apparatus", "Cytosol": "Cytosol",
                   "Mitochondria": "Mitochondria", MULTI: "Multi-localizing",
                   "Nucleoplasm": "Nucleoplasm", "Nucleoli": "Nucleoli"}


def location_groups():
    """Single-localization category -> major group, in the expert table's order."""
    m = pd.read_csv(config.LOCATION_GROUPS, sep="\t")
    m = m[~m["Grouping 3"].isna()]
    return dict(zip(m["Original annotation"], m["Grouping 3"]))


def location_matrix(folder):
    """Mean Pearson correlation of each head with the protein channel, over the
    cells of each single localization with at least ``MIN_FOVS`` FOVs."""
    from subcell.feature_analysis.attention_correlation import CHANNELS, HEADS

    relations = torch.load(f"{folder}/all_relations.pth", map_location="cpu", weights_only=False)
    meta = relations["metadata"]
    keep = meta["locations"].isin(location_groups()).values
    table = pd.DataFrame(relations[MEASURE][:, CHANNELS.index("Prot")].numpy()[keep],
                         columns=relations.get("heads", HEADS))
    table["Location"] = meta["locations"].values[keep]
    fovs = meta.loc[keep, "plate_position_sample"].groupby(table["Location"].values).nunique()
    table = table[table["Location"].isin(fovs.index[fovs >= MIN_FOVS])]
    return table.groupby("Location").mean(), table["Location"].value_counts(), fovs


def draw_heatmap(matrix, method, out, prefix):
    groups = location_groups()
    rows = linkage(matrix.values, method="average", metric="cosine")
    cols = linkage(matrix.values.T, method="average", metric="cosine")
    r = dendrogram(rows, no_plot=True)["leaves"]
    c = dendrogram(cols, no_plot=True)["leaves"]
    data = matrix.values[np.ix_(r, c)]
    names, heads = matrix.index[r].tolist(), matrix.columns[c].tolist()

    h, w = len(names) * ROW, len(heads) * COL
    left, top, strip = 12.0, 13.0, 1.8
    hx = left + strip + 0.4
    fig = L.figure(hx + w + 24.0, height=top + h + 15.0)

    ax = L.axes(fig, hx, top, w, h)
    # scaled over this model's own range, from 0 unless the correlations go negative
    vmin, vmax = min(0.0, data.min()), data.max()
    im = ax.imshow(data, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels([x.replace("-", " ") for x in heads], rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=style.FS_MIN)
    ax.yaxis.tick_right()
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=style.FS_MIN)
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Locations", fontsize=style.FS_SMALL, labelpad=3)
    ax.tick_params(length=1.2, pad=1.0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    sx = L.axes(fig, left, top, strip, h)
    sx.imshow(np.array([[[int(GROUP_COLOURS[groups[n]][i:i + 2], 16) / 255 for i in (1, 3, 5)]]
                        for n in names]), aspect="auto", interpolation="nearest")
    sx.set_xticks([0])
    sx.set_xticklabels(["Location Group"], rotation=45, ha="right", rotation_mode="anchor",
                       fontsize=style.FS_MIN)
    sx.set_yticks([])
    sx.tick_params(length=0, pad=1.0)
    for spine in sx.spines.values():
        spine.set_visible(False)

    for box, Z, orientation in ((L.axes(fig, 1.0, top, left - 1.0, h), rows, "left"),
                                (L.axes(fig, hx, 3.0, w, top - 3.5), cols, "top")):
        dendrogram(Z, ax=box, orientation=orientation, no_labels=True,
                   link_color_func=lambda k: DENDRO)
        for coll in box.collections:
            coll.set_linewidth(0.4)
        box.set_axis_off()

    cb = L.colorbar(fig, im, 3.5, 3.0, 1.6, 9.0, label="Correlation")
    cb.ax.yaxis.set_label_position("left")
    cb.ax.yaxis.set_major_locator(MultipleLocator(0.2 if vmax - vmin <= 0.9 else 0.3))
    return style.save(fig, f"{prefix}f_{display(method)}", out, formats=FORMATS)


def colours(categories):
    """glasbey_dark in order of first appearance."""
    return dict(zip(dict.fromkeys(categories), cc.glasbey_dark))


def draw_umaps(umap_df, label, out, prefix):
    emb = umap_df[["UMAP1", "UMAP2"]].values
    side, gap, top = 57.0, 4.0, 5.0
    span = emb.max(0) - emb.min(0)
    high = side * span[1] / span[0]            # equal aspect, so the panel fits the data
    fig = L.figure(2 * side + gap + 2.0, height=high + top + 1.0)
    size = L.marker_size(emb, side)
    panels = (("atlas_name", "Colored by Cell Line", CELL_LINE_LABELS),
              ("Location", "Colored by Protein Localization", LOCATION_LABELS))
    for k, (column, title, names) in enumerate(panels):
        ax = L.axes(fig, 1.0 + k * (side + gap), top, side, high)
        cats = umap_df[column].values
        multi = cats == MULTI
        if column == "Location":
            ax.scatter(emb[multi, 0], emb[multi, 1], s=size, c="#A0A0A0", linewidths=0,
                       alpha=0.5, rasterized=True)
        palette = colours(cats[~multi] if column == "Location" else cats)
        draw = L.shuffled(len(emb))
        draw = draw[~multi[draw]] if column == "Location" else draw
        ax.scatter(emb[draw, 0], emb[draw, 1], s=size, linewidths=0, alpha=0.5,
                   c=[palette[x] for x in cats[draw]], rasterized=True)
        L.limits(ax, emb)
        texts = [L.outlined_text(ax, np.median(emb[cats == cat, 0]), np.median(emb[cats == cat, 1]),
                                 text, "black", fontsize=style.FS_SMALL)
                 for cat, text in names.items() if (cats == cat).any()]
        # category medians can coincide; move the labels apart, never the points
        adjust_text(texts, ax=ax, only_move={"text": "xy"}, expand=(1.1, 1.3))
        ax.set_axis_off()
        ax.set_title(title, fontsize=style.FS_BASE)
        if k == 0:
            L.axis_key(ax, emb, labels=("UMAP1", "UMAP2"))
    return style.save(fig, f"{prefix}g_{label}", out, formats=FORMATS)


def main(out_dir=None):
    style.use_agg()
    style.register_fonts()
    from subcell.feature_analysis.attention_correlation import attention_umap, fov_profiles

    with style.nature():
        for method, run in RUNS.items():
            prefix = "ed8" if method == "ViT-ProtS-Pool" else "fig3"
            out = config.ensure(out_dir or config.figure_dir(prefix))
            folder = f"{config.ATTN_CORR_DIR}/{run}"
            matrix, cells, fovs = location_matrix(folder)
            matrix.assign(cells=cells, FOVs=fovs).round(4).to_csv(f"{out}/{prefix}f_{display(method)}.csv")
            draw_heatmap(matrix, method, out, prefix)
            fov, profiles = fov_profiles(folder, MEASURE)
            draw_umaps(attention_umap(folder, fov, profiles, MEASURE), display(method), out, prefix)
            log(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    main(a.out_dir)
