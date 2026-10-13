"""Extended Data Fig. 3 -- factors of variation in the HPA feature spaces, and harmonization.

Panels
------
a-d  UMAP of FOV-aggregated features: four self-supervised encoders (rows) under
     four Harmony regimes (columns), coloured by protein localization (a), cell
     line (b), microscope (c) and plate ID ordered by acquisition (d).
e    Cross-plate retrieval mAP under each regime, with localization as the
     positive label (signal) and cell line as the positive label (nuisance).
f    Share of feature dimensions most informative for each annotated factor,
     by a mutual-information attribution.

One page, and one figure per panel group
----------------------------------------
``ed3`` is the whole figure on one Extended Data page: 180 x 170 mm, 8 pt bold
lower-case panel keys, all other type 5-7 pt, strokes 0.25-1 pt, written as an RGB
JPEG at the highest resolution under 10 MB. The a-d keys sit in a column on the
right, since under each block they would leave the embeddings ~10 mm. Each panel
group is also written as its own figure (``ed3a_umap_<covariate>``,
``ed3e_retrieval_map``, ``ed3f_mutual_info``), with per-level FOV counts in the
legends, at a size where the embeddings can be read in detail. All are written to
``figure_dir("ed3")``.

Inputs: FOV embeddings and their UMAPs (``subcell.feature_analysis.embedding``),
``retrieval_map.csv`` (``subcell.feature_analysis.retrieval``) and the mutual-information
tables (``subcell.feature_analysis.mutual_info``).

Coordinates are identical between a-d; only the colouring changes, so the four
blocks are directly comparable.

Usage
-----
    python -m figures.ed3_harmonization                     # everything
    python -m figures.ed3_harmonization --panels combined   # the one-page figure
    python -m figures.ed3_harmonization --panels a e
"""

from __future__ import annotations

import argparse

import colorcet as cc
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from subcell import config, naming, style
from subcell.feature_analysis import embedding
from subcell.io import features
from subcell.runlog import log
from subcell.style import layout as L

PRINCIPAL_MICROSCOPES = ("M1", "M2")

COVARIATE_LABEL = {"Location": "Localization", "atlas_name": "Cell line",
                   "microscope": "Microscope", "if_plate_id": "Plate ID"}
TITLE_NOUN = {"Location": "localization", "atlas_name": "cell line",
              "microscope": "microscope", "if_plate_id": "plate ID"}
TASK_LABEL = {"location": "Localization retrieval mAP",
              "cell_line": "Cell-line retrieval mAP"}
FACTOR_LABEL = {"location_mi": "Localization", "cell_mi": "Cell line",
                "well_mi": "Well"}

# matplotlib tab10 for models and factors.
TAB10 = plt.get_cmap("tab10").colors
METHOD_COLOR = {"ViT-ProtS-Pool": TAB10[0], "MAE-CellS-ProtS-Pool": TAB10[1],
                "DINO4Cells-HPA": TAB10[2], "Cell-DINO-HPA": TAB10[3]}
FACTOR_COLOR = {"location_mi": TAB10[0], "cell_mi": TAB10[1], "well_mi": TAB10[2]}
OTHER_GREY = "#d9d9d9"
FRAME_GREY = "#999999"
POINT_S, ALPHA = 0.05, 0.5

LEGEND_KW = dict(frameon=False, fontsize=style.FS_TINY, handletextpad=0.2,
                 columnspacing=0.6, labelspacing=0.2, borderpad=0.0,
                 borderaxespad=0.0, title_fontsize=5.5)
HEADING_FS = 6.5


# ------------------------------------------------------------------ labelling

def covariate_labels(meta: pd.DataFrame, column: str) -> pd.Series:
    """Colour level per FOV, with anything uncolourable as ``"other"``.

    Localization keeps only unambiguous single-compartment FOVs: a
    multi-localized well has no one compartment to colour it by, and colouring it
    as its first annotation would invent a label. Microscope pools everything
    outside the two principal HPA instruments.
    """
    if column == "Location":
        loc = meta["locations"].fillna("Negative")
        single = (~loc.str.contains(",", regex=False)) & (loc != "Negative")
        return loc.where(single, "other")
    if column == "microscope":
        m = meta["microscope"].astype(str)
        return m.where(m.isin(PRINCIPAL_MICROSCOPES), "other")
    return meta[column].astype(str)


def level_palette(labels: pd.Series) -> dict:
    """Colour per level, ordered by abundance so it is stable across panels.

    Up to 20 levels take tab20's ten dark shades and then its ten light ones, so
    adjacent levels differ in hue; beyond that, glasbey_dark.
    """
    levels = [v for v in labels.value_counts().index if v != "other"]
    if len(levels) <= 20:
        tab20 = plt.get_cmap("tab20").colors
        colours = [tab20[i] for i in range(0, 20, 2)] + [tab20[i] for i in range(1, 20, 2)]
    else:
        colours = cc.glasbey_dark
    return dict(zip(levels, colours[:len(levels)]))


def plate_ids(meta: pd.DataFrame) -> np.ndarray:
    """Plate ID as a number, for the continuous ramp in panel d.

    1,586 plates make a legend useless, and HPA plate ids increase roughly with
    acquisition date, so the ramp itself is the result: smooth colour across the
    manifold is temporal drift, uniform colour noise is a corrected embedding.
    """
    v = pd.to_numeric(meta["if_plate_id"], errors="coerce")
    return v.fillna(v.median()).to_numpy()


def dot_handles(cmap: dict, labels: dict | None = None, other: bool = False):
    """Round legend markers, one per level, plus a grey "other"."""
    names = dict(cmap, **({"other": OTHER_GREY} if other else {}))
    return [mlines.Line2D([], [], marker="o", linestyle="", markersize=2.6,
                          markerfacecolor=c, markeredgewidth=0,
                          label=(labels or {}).get(lv, lv))
            for lv, c in names.items()]


def legend_height(handles, ncol: int, title: str) -> float:
    """Height in mm of a legend as drawn, measured on a scratch figure."""
    f = plt.figure(figsize=(7.2, 7.2))
    leg = f.legend(handles=handles, ncol=ncol, title=title, **LEGEND_KW)
    f.canvas.draw()
    h = leg.get_window_extent().height / f.dpi * 25.4
    plt.close(f)
    return h


def place_legend(fig, handles, x: float, y: float, ncol: int, title: str):
    """A legend whose top edge is centred on (x, y) mm."""
    fw, fh = L.size_mm(fig)
    fig.legend(handles=handles, loc="upper center", ncol=ncol, title=title,
               bbox_to_anchor=(x / fw, 1.0 - y / fh), **LEGEND_KW)


# --------------------------------------------------------------------- inputs

def load_panel_data(harmonizations=None):
    """Cached FOV embeddings and one shared metadata table.

    The metadata is taken from the uncorrected run and asserted identical in row
    count against every other, because Harmony does not reorder rows and a
    mismatch here would silently colour one model's embedding with another's
    labels.
    """
    harms = harmonizations or naming.HARM_ORDER
    coords, meta = {}, None
    for method in naming.HARM_METHODS:
        for harm in harms:
            feats, m = features.fov_features(method, harm)
            coords[f"{method}_{harm}"] = embedding.fit_umap(feats, method, harm)
            if meta is None:
                meta = m
            elif len(m) != len(meta):
                raise ValueError(
                    f"{method}_{harm} has {len(m):,} FOVs, expected {len(meta):,}")
    return coords, meta


# ---------------------------------------------------------------- panels a-d

COLORBAR_H = 9.0


def key_spec(meta, covariate: str, counts: bool):
    """(handles, ncol) for a categorical key, or None for the plate ramp."""
    if covariate in naming.GRADIENT_COVARIATES:
        return None
    labels = covariate_labels(meta, covariate)
    cmap = level_palette(labels)
    n = labels.value_counts()
    names = {lv: f"{lv} ({n[lv]:,})" for lv in n.index} if counts else None
    handles = dot_handles(cmap, names, other=(labels == "other").any())
    return handles, 1 if len(handles) <= 8 else 4


def key_height(meta, covariate: str, counts: bool = False) -> float:
    """Height in mm of the key under one embedding block."""
    spec = key_spec(meta, covariate, counts)
    if spec is None:
        return COLORBAR_H
    return legend_height(*spec, COVARIATE_LABEL[covariate])


def plate_colorbar(fig, handle, x: float, y: float, w: float):
    """The plate-ID ramp as a horizontal colourbar with its top edge at (x, y) mm."""
    cb = L.colorbar(fig, handle, x, y, w, 1.2, orientation="horizontal")
    cb.set_label("Plate ID (acquisition order)", fontsize=style.FS_TINY, labelpad=1.5)
    cb.ax.tick_params(labelsize=style.FS_TINY, length=1.5, pad=1)
    cb.solids.set_alpha(1)


def umap_block(fig, coords, meta, covariate: str, x: float, y: float, panel: float,
               gap: float, *, counts: bool = False, draw_key: bool = True):
    """The model x regime embeddings coloured by *covariate*, and their key.

    *x, y* is the top-left of the first panel in mm; column headers are drawn
    above it and row labels to its left, so the caller leaves room for both. The
    key is centred under the panels, 1 mm below them; with ``draw_key=False`` the
    caller places it, and the returned scatter handle feeds the plate colourbar.
    """
    harms, methods = naming.HARM_ORDER, naming.HARM_METHODS
    gradient = covariate in naming.GRADIENT_COVARIATES
    if gradient:
        values = plate_ids(meta)
    else:
        labels = covariate_labels(meta, covariate)
        cmap = level_palette(labels)
        rest = ~labels.isin(list(cmap)).to_numpy()
        masks = {lv: (labels == lv).to_numpy() for lv in cmap}

    handle = None
    for r, method in enumerate(methods):
        for c, harm in enumerate(harms):
            key = f"{method}_{harm}"
            if key not in coords:
                continue
            emb = coords[key]
            ax = L.axes(fig, x + c * (panel + gap), y + r * (panel + gap), panel, panel)
            if gradient:
                handle = ax.scatter(emb[:, 0], emb[:, 1], c=values, cmap="viridis",
                                    s=POINT_S, alpha=ALPHA, linewidths=0,
                                    rasterized=True)
            else:
                if rest.any():
                    ax.scatter(emb[rest, 0], emb[rest, 1], s=POINT_S, c=OTHER_GREY,
                               linewidths=0, alpha=ALPHA * 0.6, rasterized=True)
                for lv, colour in cmap.items():
                    m = masks[lv]
                    ax.scatter(emb[m, 0], emb[m, 1], s=POINT_S, c=[colour],
                               linewidths=0, alpha=ALPHA, rasterized=True)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.3)
                sp.set_color(FRAME_GREY)
            if r == 0:
                ax.set_title(naming.HARM_SHORT[harm], fontsize=style.FS_TINY, pad=1.5)
            if c == 0:
                ax.set_ylabel(naming.display(method), fontsize=style.FS_TINY,
                              labelpad=2)

    if not draw_key:
        return handle
    w = len(harms) * (panel + gap) - gap
    y_key = y + len(methods) * (panel + gap) - gap + 1.0
    if gradient:
        plate_colorbar(fig, handle, x + 0.22 * w, y_key, 0.56 * w)
    else:
        handles, ncol = key_spec(meta, covariate, counts)
        place_legend(fig, handles, x + w / 2, y_key, ncol, COVARIATE_LABEL[covariate])
    return handle


def block_heading(fig, letter: str, covariate: str, x: float, y: float, w: float,
                  label_w: float):
    """Panel letter at the block's left and its title centred over the panels."""
    L.panel_label(fig, letter, x - label_w, y)
    L.text(fig, x + w / 2, y + 0.3,
           f"Effect of harmonization on {TITLE_NOUN[covariate]}",
           fontsize=HEADING_FS, ha="center")


def umap_grid(coords, meta, covariate: str, letter: str):
    """One embedding block as its own 183 mm figure."""
    left, top, gap = 22.0, 11.0, 1.2
    n_col, n_row = len(naming.HARM_ORDER), len(naming.HARM_METHODS)
    panel = (L.WIDTH_2COL - left - 2.0 - (n_col - 1) * gap) / n_col
    grid_h = n_row * (panel + gap) - gap
    fig = L.figure("double", height=top + grid_h + 1.0
                   + key_height(meta, covariate, counts=True) + 2.0)
    umap_block(fig, coords, meta, covariate, left, top, panel, gap, counts=True)
    block_heading(fig, letter, covariate, left, 2.0,
                  n_col * (panel + gap) - gap, left - 1.5)
    return fig


# ------------------------------------------------------------------ panel e


def metric_axes(ax, xs):
    """Shared styling for the small metric panels."""
    ax.set_xticks(xs)
    ax.set_xticklabels([naming.HARM_SHORT[h] for h in naming.HARM_ORDER],
                       fontsize=style.FS_TINY, rotation=45, ha="right")
    ax.set_xlim(-0.6, len(xs) - 0.4)
    ax.grid(axis="y", alpha=0.22, linewidth=0.35)
    ax.tick_params(labelsize=style.FS_TINY, length=1.8, pad=1.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def metric_heading(fig, letter: str, text: str, x0: float, x1: float, y: float):
    """Panel letter and a centred title over a row of metric axes."""
    L.panel_label(fig, letter, x0 - 8.0, y)
    L.text(fig, (x0 + x1) / 2, y + 0.3, text, fontsize=HEADING_FS, ha="center")


def retrieval_axes(fig, x: float, y: float, w: float, h: float, gap: float,
                   legend_y: float):
    """Retrieval mAP against harmonisation strength, one w x h axes per task.

    A slope plot rather than grouped bars: the question is the change within a
    model across regimes, which connected lines show directly while bars force the
    eye across gaps. The dotted line is the random baseline, the analytic
    expectation of AP for each pool composition.
    """
    df = pd.read_csv(config.HPA_UMAP_DIR / "retrieval" / "retrieval_map.csv")
    xs = np.arange(len(naming.HARM_ORDER))
    for j, task in enumerate(TASK_LABEL):
        ax = L.axes(fig, x + j * (w + gap), y, w, h)
        sub = df[df["task"] == task]
        for m in naming.HARM_METHODS:
            g = sub[sub["method"] == m].set_index("harmonized")
            vals = [g.loc[k, "mAP"] if k in g.index else np.nan
                    for k in naming.HARM_ORDER]
            ax.plot(xs, vals, "-o", color=METHOD_COLOR[m], markersize=2.6,
                    linewidth=1.0, label=naming.display(m), markeredgecolor="white",
                    markeredgewidth=0.4)
        chance = float(sub["chance_mAP"].mean())
        ax.axhline(chance, color="black", linestyle=":", linewidth=0.9, zorder=0)
        ax.set_ylabel(TASK_LABEL[task], fontsize=style.FS_TINY, labelpad=2)
        ax.set_ylim(bottom=min(0.0, chance * 0.5))
        ax.yaxis.set_major_locator(mtick.MaxNLocator(5))
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
        metric_axes(ax, xs)
    handles, _ = ax.get_legend_handles_labels()
    place_legend(fig, handles, x + w + gap / 2, legend_y, len(handles), "Model")


def retrieval_panel(letter: str = "e"):
    """Panel e as its own figure."""
    grid = L.Grid(n_col=2, n_row=1, width=L.WIDTH_1COL, left=12.0, right=2.0,
                  top=7.0, bottom=22.0, wspace=13.0, aspect=1.0)
    fig = L.figure(L.WIDTH_1COL, height=grid.height)
    retrieval_axes(fig, grid.left, grid.top, grid.panel_w, grid.panel_h,
                   grid.wspace, legend_y=grid.top + grid.panel_h + 13.0)
    metric_heading(fig, letter, "Effect of harmonization on retrieval",
                   grid.left, grid.left + 2 * grid.panel_w + grid.wspace, 1.0)
    return fig


# ------------------------------------------------------------------ panel f


def mutual_info_shares(method: str, harmonized: str):
    """Share of feature dimensions most informative for each annotated factor.

    Each dimension is attributed to whichever of localization, cell line and well
    it shares the most information with, then the counts become percentages. The
    MI values are read from the cached tables, so nothing is recomputed here.
    """
    f = config.MUTUAL_INFO_DIR / f"{method}_{harmonized}_mi_df.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    cols = ["location_mi", "cell_mi", "well_mi"]
    winner = d[cols].to_numpy().argmax(axis=1)
    counts = np.bincount(winner, minlength=3).astype(float)
    return dict(zip(cols, 100 * counts / max(counts.sum(), 1)))


def mutual_info_axes(fig, x: float, y: float, w: float, h: float, gap: float,
                     legend_y: float):
    """Stacked bars: which factor each feature dimension is most informative for."""
    xs = np.arange(len(naming.HARM_ORDER))
    for j, method in enumerate(naming.HARM_METHODS):
        ax = L.axes(fig, x + j * (w + gap), y, w, h)
        shares = [mutual_info_shares(method, k) for k in naming.HARM_ORDER]
        if any(v is None for v in shares):
            log(f"  no MI table for {method}; its column is left empty")
            ax.axis("off")
            continue
        bottom = np.zeros(len(xs))
        for factor, colour in FACTOR_COLOR.items():
            vals = np.array([v[factor] for v in shares])
            ax.bar(xs, vals, bottom=bottom, width=0.72, color=colour,
                   linewidth=0, label=FACTOR_LABEL[factor])
            bottom += vals
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylabel("Feature dimensions" if j == 0 else "", fontsize=5.5,
                      labelpad=2)
        ax.set_title(naming.display(method), fontsize=5.5, pad=1.5)
        metric_axes(ax, xs)
    handles, _ = ax.get_legend_handles_labels()
    n = len(naming.HARM_METHODS)
    place_legend(fig, handles, x + (n * (w + gap) - gap) / 2, legend_y, 3,
                 "Dominant factor per dimension")


def mutual_info_panel(letter: str = "f"):
    """Panel f as its own figure."""
    grid = L.Grid(n_col=len(naming.HARM_METHODS), n_row=1, width=L.WIDTH_1_5COL,
                  left=12.0, right=2.0, top=7.0, bottom=22.0, wspace=11.0,
                  aspect=1.0)
    fig = L.figure(L.WIDTH_1_5COL, height=grid.height)
    n = len(naming.HARM_METHODS)
    mutual_info_axes(fig, grid.left, grid.top, grid.panel_w, grid.panel_h,
                     grid.wspace, legend_y=grid.top + grid.panel_h + 13.0)
    metric_heading(fig, letter,
                   "Effect of harmonization on feature composition", grid.left,
                   grid.left + n * (grid.panel_w + grid.wspace) - grid.wspace, 1.0)
    return fig


# ------------------------------------------------------------- one-page figure

PANELS = "abcdef"
COVARIATE_OF = dict(zip("abcd", naming.HPA_COVARIATES))


def combined_figure(coords, meta):
    """Every panel on one Extended Data page, 180 x 170 mm.

    a-d are 2 x 2 embedding blocks. Their keys sit in a column on the right, which
    is what keeps the embeddings ~14 mm at 5 pt type on a 170 mm page. e and f
    share the bottom row with their legends below.
    """
    n_col, n_row = len(naming.HARM_ORDER), len(naming.HARM_METHODS)
    gap, label_w, gutter, head, block_gap = 0.6, 4.5, 3.0, 5.5, 2.0
    metric_h, metric_head, metric_foot = 22.0, 6.0, 13.5
    y_metric = L.ED_MAX_HEIGHT - metric_foot - metric_h - metric_head
    grid_h = (y_metric - 0.5 - 2 * head - block_gap - 2.5) / 2
    panel = (grid_h - (n_row - 1) * gap) / n_row
    grid_w = n_col * (panel + gap) - gap
    xs = (label_w + 0.5, label_w + 0.5 + grid_w + gutter + label_w)
    fig = L.figure(L.ED_WIDTH, height=L.ED_MAX_HEIGHT)

    handles, y = {}, 0.5
    for pair in ("ab", "cd"):
        for bx, p in zip(xs, pair):
            cov = COVARIATE_OF[p]
            block_heading(fig, p, cov, bx, y, grid_w, label_w)
            handles[cov] = umap_block(fig, coords, meta, cov, bx, y + head, panel,
                                      gap, draw_key=False)
        y += head + grid_h + block_gap

    # Keys in panel order, spread down the height of the blocks.
    kx = xs[1] + grid_w + 3.0
    keys = []
    for cov in naming.HPA_COVARIATES:
        if cov in naming.GRADIENT_COVARIATES:
            keys.append((cov, None, COLORBAR_H))
        else:
            h = key_spec(meta, cov, counts=False)[0]
            keys.append((cov, h, legend_height(h, 2, COVARIATE_LABEL[cov])))
    top, bottom = 0.5 + head, y - block_gap
    spare = (bottom - top - sum(k[2] for k in keys)) / (len(keys) - 1)
    ky = top
    for cov, h, kh in keys:
        if h is None:
            plate_colorbar(fig, handles[cov], kx + 2.0, ky, L.ED_WIDTH - kx - 4.0)
        else:
            place_legend(fig, h, (kx + L.ED_WIDTH) / 2, ky, 2, COVARIATE_LABEL[cov])
        ky += kh + spare

    top = y_metric + metric_head
    w = (L.ED_WIDTH - 53.0) / 6
    e_x, e_gap, f_gap = 9.0, 10.0, 7.0
    legend_y = top + metric_h + 8.5
    retrieval_axes(fig, e_x, top, w, metric_h, e_gap, legend_y)
    metric_heading(fig, "e", "Effect of harmonization on retrieval", e_x,
                   e_x + 2 * w + e_gap, y_metric)
    f_x = e_x + 2 * (w + e_gap) + 2.0
    mutual_info_axes(fig, f_x, top, w, metric_h, f_gap, legend_y)
    metric_heading(fig, "f", "Effect of harmonization on feature composition", f_x,
                   f_x + 4 * w + 3 * f_gap, y_metric)
    return fig


# --------------------------------------------------------------------- driver

TARGETS = list(PANELS) + ["combined"]


def main(targets=TARGETS, out_dir=None):
    style.use_agg()
    out = out_dir or config.figure_dir("ed3")
    need_umaps = [p for p in targets if p in COVARIATE_OF]

    with style.nature():
        if need_umaps or "combined" in targets:
            coords, meta = load_panel_data()
        for p in need_umaps:
            cov = COVARIATE_OF[p]
            log(f"panel {p}: UMAP grid coloured by {cov}")
            style.save(umap_grid(coords, meta, cov, p), f"ed3{p}_umap_{cov}", out,
                       formats=("jpg",))
        if "e" in targets:
            log("panel e: retrieval mAP")
            style.save(retrieval_panel(), "ed3e_retrieval_map", out,
                       formats=("jpg",))
        if "f" in targets:
            log("panel f: mutual-information shares")
            style.save(mutual_info_panel(), "ed3f_mutual_info", out,
                       formats=("jpg",))
        if "combined" in targets:
            log("all panels on one page")
            style.save(combined_figure(coords, meta), "ed3", out, formats=("jpg",))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panels", nargs="+", default=TARGETS, choices=TARGETS,
                    help="panels to render; 'combined' is the one-page figure")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    main(a.panels, a.out_dir)
