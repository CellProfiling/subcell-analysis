"""Fig. 3a,b and Extended Data Fig. 8a,b -- cell-line and localization similarity.

a  Cosine similarity between cell lines by bulk mRNA (left) and by the model's
   cell-line-aggregated features (right). Both matrices are in the leaf order of
   the mRNA dendrogram, so the feature matrix is judged against an ordering it
   did not choose.
b  Cosine similarity between the model's localization-aggregated features, in the
   order of the expert hierarchy, with the seven major location groups boxed in
   green and the minor ones in magenta.

SubCell is drawn for Fig. 3 and SubCell-P for Extended Data Fig. 8; ``--models``
draws others (written beside Fig. 3). Each panel uses the harmonization that gives
the model its best Mantel statistic for that comparison, so they are the matrices
behind the bars of Fig. 3c (``figures.fig3c_mantel``). The statistics and profiles
come from ``subcell.feature_analysis.similarity`` (``config.MORPHOLOGY_DIR``).
Type is Arial (or its metric twin) at 5-7 pt, strokes are 0.25-1 pt, sizes are in
millimetres, with one matrix row every 1.8 mm, the pitch at which 5 pt labels do
not touch. Each panel is written as PDF and JPEG: ``fig3a_SubCell``,
``fig3b_SubCell`` in ``figure_dir("fig3")``; ``ed8a_SubCell-P``, ``ed8b_SubCell-P``
in ``figure_dir("ed8")``.

    python -m figures.fig3ab_similarity
"""

from __future__ import annotations

import argparse

import matplotlib.patches as patches
import pandas as pd

from subcell import config, style
from subcell.runlog import log
from subcell.feature_analysis import similarity
from subcell.naming import ED8_MODELS, FIG3_MODELS, display
from subcell.style import layout as L

MODELS = [FIG3_MODELS[1], ED8_MODELS[1]]       # SubCell, SubCell-P
SUBCELL_P = ED8_MODELS[1]
PCA_DIMS = 10
CMAP, VMIN, VMAX = "Blues", -0.8, 1.0
MAJOR, MINOR = "#32FF32", "#CF03C3"
SHORT = {
    "BJ [Human fibroblast]": "BJ",
    "Nucleoli fibrillar center": "Nucleoli fibr. center",
    "Endoplasmic reticulum": "ER",
    "Intermediate filaments": "Interm. filaments",
    "Focal adhesion sites": "Focal adh. sites",
}
CELL = 1.8                                   # mm per matrix row
FORMATS = ("pdf", "jpg")


def best_harmonization(table, method, stat):
    rows = table[table["Method"] == method]
    return rows.loc[rows[stat].idxmax(), "Harmonized"]


def heatmap(ax, matrix, labels, *, ylabels=True):
    im = ax.imshow(matrix, cmap=CMAP, vmin=VMIN, vmax=VMAX, interpolation="nearest")
    names = [SHORT.get(n, n) for n in labels]
    ax.set_xticks(range(len(names)))
    # vertical: at a 1.8 mm pitch, 45-degree 5 pt labels overlap their neighbours
    ax.set_xticklabels(names, rotation=90, fontsize=style.FS_MIN)
    ax.set_yticks(range(len(names)) if ylabels else [])
    if ylabels:
        ax.set_yticklabels(names, fontsize=style.FS_MIN)
    ax.tick_params(length=1.2, pad=1.0)
    for spine in ax.spines.values():
        spine.set_visible(True)
    return im


def draw_cell_lines(method, harmonized, out, prefix):
    cell_lines, features, tpms, _, _ = similarity.profiles(method, harmonized)
    names, rna, feat = similarity.cell_line_matrices(
        cell_lines, similarity.pca_profiles(tpms, PCA_DIMS),
        similarity.pca_profiles(features, PCA_DIMS))
    side = len(names) * CELL
    left, top, gap, bottom = 13.0, 5.0, 2.0, 13.5
    cb_x = left + 2 * side + gap + 2.0
    fig = L.figure(cb_x + 2.5 + 10.0, height=top + side + bottom)
    ax = L.axes(fig, left, top, side, side)
    heatmap(ax, rna, names)
    ax.set_title("mRNA", fontsize=style.FS_BASE)
    ax = L.axes(fig, left + side + gap, top, side, side)
    im = heatmap(ax, feat, names, ylabels=False)
    ax.set_title(display(method), fontsize=style.FS_BASE)
    cb_h = 0.78 * side                        # bottom-aligned
    L.colorbar(fig, im, cb_x, top + side - cb_h, 2.5, cb_h, label="Cosine Similarity")
    return style.save(fig, f"{prefix}a_{display(method)}", out, formats=FORMATS)


def draw_localizations(method, harmonized, out, prefix):
    _, _, _, locs, features = similarity.profiles(method, harmonized)
    m = similarity.location_matrices(locs, features, PCA_DIMS)
    side = len(m["labels"]) * CELL
    left, top, bottom = 18.5, 5.0, 20.0
    fig = L.figure(left + side + 1.5, height=top + side + bottom)
    ax = L.axes(fig, left, top, side, side)
    heatmap(ax, m["similarity"], m["labels"])
    ax.set_title(display(method), fontsize=style.FS_BASE)
    major = m["major"]
    for i, (name, (lo, hi)) in enumerate(major.items()):
        ax.add_patch(patches.Rectangle((lo - 0.5, lo - 0.5), hi - lo + 1, hi - lo + 1,
                                       linewidth=1.0, edgecolor=MAJOR, facecolor="none"))
        right = i < len(major) / 2 - 1
        ax.text(hi + 1 if right else lo - 1, lo + (hi - lo) / 2, name,
                ha="left" if right else "right", va="center", color=MAJOR,
                fontsize=style.FS_SMALL, fontweight="bold",
                bbox=dict(facecolor="black", edgecolor="none", alpha=0.1, pad=0.8))
    for lo, hi in m["minor"].values():
        ax.add_patch(patches.Rectangle((lo - 0.5, lo - 0.5), hi - lo + 1, hi - lo + 1,
                                       linewidth=0.5, edgecolor=MINOR, facecolor="none"))
    return style.save(fig, f"{prefix}b_{display(method)}", out, formats=FORMATS)


def main(out_dir=None, models=MODELS):
    style.use_agg()
    style.register_fonts()

    table = similarity.mantel_table(str(config.MORPHOLOGY_DIR), PCA_DIMS)
    used = {"fig3": [], "ed8": []}
    with style.nature():
        for method in models:
            prefix = "ed8" if method == SUBCELL_P else "fig3"
            out = config.ensure(out_dir or config.figure_dir(prefix))
            for panel, stat, draw in (
                    ("a", "RNAseq Mantel Statistic", draw_cell_lines),
                    ("b", "Localization Mantel Statistic", draw_localizations)):
                harmonized = best_harmonization(table, method, stat)
                draw(method, harmonized, out, prefix)
                value = table.loc[(table["Method"] == method)
                                  & (table["Harmonized"] == harmonized), stat].iloc[0]
                used[prefix].append({"Panel": panel, "Model": display(method),
                                     "Statistic": stat, "Harmonized": harmonized,
                                     "Mantel": value})
    for prefix, rows in used.items():
        if rows:
            out = config.ensure(out_dir or config.figure_dir(prefix))
            pd.DataFrame(rows).to_csv(f"{out}/{prefix}ab_harmonization.csv", index=False)
            log(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--models", nargs="+", default=MODELS,
                    help="internal model names; SubCell-P goes to ED Fig. 8, others to Fig. 3")
    a = ap.parse_args()
    main(a.out_dir, a.models)
