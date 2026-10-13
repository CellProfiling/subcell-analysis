"""Extended Data Fig. 2i (left) -- tissue UMAPs before and after batch correction.

A grid of UMAPs of the concatenated MT and VIM SubCell embeddings: one row per
Harmony batch key, by default no correction and the reported core-level correction
(``--variants`` adds cohort, acquisition block or donor), and one column per covariate,
by default the four of the legend (tissue, cancer, donor, core; ``--fields`` selects
others, e.g. the acquisition block). Reading along a row shows what that correction
left behind; reading down a column shows which correction removes it. The UMAPs are
computed by ``subcell.tissue.harmonization`` (stage ``umap``) and are for visualization
only.

Marker size is computed from the embedding (``style.layout.marker_size``) and drawn at
high alpha in one shuffled pass, so no category sits permanently on top. With 9, 19
and 29 categories colour cannot carry identity: every category is direct-labelled at
its median position and colour is only a grouping cue.

    python -m figures.ed2i_tissue_umaps [--variants none cohort slide patient core]
        [--fields tissue cancer patient core_id] [--out-dir DIR]

Inputs   ``config.TISSUE_UMAPS``, ``config.TISSUE_META``
Outputs  ``config.figure_dir("ed2")``/ed2i_tissue_umaps
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from subcell import config, naming, style
from subcell.io import metadata
from subcell.runlog import log
from subcell.style import layout as L
from subcell.style import palettes as pal

# donor + block is left out: it is the composite of two keys that already have a
# column each. Keys keep their on-disk names (patient = donor, slide = acquisition block).
VARIANTS = ["none", "cohort", "slide", "patient", "core"]
PAPER_VARIANTS = ["none", "core"]   # the two rows of Extended Data Fig. 2i
# the four covariates of its legend; the acquisition block is available via --fields
FIG_FIELDS = [f for f in naming.TISSUE_FIELDS if f[0] != "slide"]


def field_palette(values: pd.Series, column: str) -> dict:
    """Colour per level for one covariate (one column of the grid)."""
    cats = sorted(values.unique())
    if column == "tissue":
        return {c: pal.TISSUE_PAL[c] for c in cats}
    if column == "cancer":
        return {c: pal.CANCER_PAL[c] for c in cats}
    # 9, 19 or 29 levels: no palette separates these, so the figure labels them
    return dict(zip(cats, pal.categorical(len(cats), labelled=True)))


def short_label(value, column: str) -> str:
    """The category name as drawn on the panel (core ids ``G_R1`` drop the cohort letter)."""
    return str(value).split("_")[-1] if column == "core_id" else str(value)


def separate_if_overlapping(cents: np.ndarray, emb: np.ndarray,
                            min_frac: float = 0.30) -> np.ndarray:
    """Pull two labels apart when their medians nearly coincide (two-level rows only)."""
    if len(cents) != 2:
        return cents
    span = float(np.max(emb[:, 1]) - np.min(emb[:, 1]))
    if np.linalg.norm(cents[0] - cents[1]) >= min_frac * span:
        return cents
    mid = cents.mean(axis=0)
    out = cents.copy()
    out[:, 0] = mid[0]
    out[:, 1] = [mid[1] + 0.17 * span, mid[1] - 0.17 * span]
    return out


def render(umaps, meta, variants=None, fields=None):
    variants = [v for v in (variants or PAPER_VARIANTS) if v in umaps]
    fields = fields or FIG_FIELDS

    covariates = []
    for col, label in fields:
        cats = sorted(meta[col].unique())
        cmap = field_palette(meta[col], col)
        codes = pd.Categorical(meta[col], categories=cats).codes
        covariates.append((col, label, cats, np.array([cmap[c] for c in cats]), codes,
                           cmap))

    # One row per batch key, one column per covariate.
    grid = L.Grid(n_col=len(covariates), n_row=len(variants), width=L.WIDTH_2COL,
                  left=17.0, right=1.5, top=12.0, bottom=4.0,
                  wspace=1.0, hspace=1.0, aspect=1.0)
    fig = L.figure("double", height=grid.height)
    order = L.shuffled(len(meta))

    for i, v in enumerate(variants):
        emb = umaps[v]
        if len(emb) != len(meta):
            raise ValueError(f"embedding '{v}' has {len(emb):,} rows, metadata "
                             f"{len(meta):,}")
        diam = L.marker_size(emb, grid.panel_w)
        log(f"  {v:8s} marker {diam:.3f} pt = {diam * style.DPI / 72:.2f} px")
        for j, (col, label, cats, colors, codes, cmap) in enumerate(covariates):
            ax = grid.ax(fig, i, j)
            ax.scatter(emb[order, 0], emb[order, 1], s=diam ** 2,
                       c=colors[codes][order], alpha=0.95, linewidths=0,
                       rasterized=True)
            L.limits(ax, emb)
            L.bare(ax)

            cents = separate_if_overlapping(
                L.centroids(emb, codes, len(cats)), emb)
            fs = style.FS_SMALL if len(cats) <= 2 else style.FS_TINY
            for (cx, cy), cat in zip(cents, cats):
                if np.isfinite(cx):
                    L.outlined_text(ax, cx, cy, short_label(cat, col), cmap[cat],
                                    fontsize=fs, stroke=1.0)

            if i == 0:
                ax.set_title(f"{label}\n{len(cats)} categories",
                             fontsize=style.FS_SMALL, pad=2.5, linespacing=1.3)
            if j == 0:
                n = naming.TISSUE_HARM_NLEVELS[v]
                ax.text(-0.10, 0.5, naming.TISSUE_HARM_TITLE[v] + (f"\n{n} batches" if n else ""),
                        color=pal.INK2, fontsize=style.FS_SMALL, rotation=90,
                        ha="center", va="center", transform=ax.transAxes,
                        linespacing=1.3)
                if i == len(variants) - 1:
                    L.axis_key(ax, emb)

    L.panel_label(fig, "i", 1.5, 1.5)
    L.text(fig, 17.0, 2.5,
           f"SubCell tissue embedding, {len(meta):,} cells from {meta['core_id'].nunique()} "
           f"TMA cores of {meta['patient'].nunique()} donors   ·   covariate across, Harmony batch key "
           "down",
           fontsize=style.FS_SMALL, color=pal.INK2)
    return fig


def main(out_dir=None, variants=None, fields=None):
    """``fields``: metadata columns to colour by (default: the four of the paper legend)."""
    style.use_agg()
    log(f"loading {config.TISSUE_UMAPS}")
    umaps = dict(np.load(config.TISSUE_UMAPS))
    meta = metadata.tissue_metadata()
    labels = dict(naming.TISSUE_FIELDS)
    with style.nature():
        fig = render(umaps, meta, variants, [(f, labels[f]) for f in fields] if fields else None)
        style.save(fig, "ed2i_tissue_umaps", out_dir or config.figure_dir("ed2"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", nargs="+", default=None, choices=VARIANTS,
                    help=f"batch keys to draw as rows (default: {' '.join(PAPER_VARIANTS)})")
    ap.add_argument("--fields", nargs="+", default=None, choices=[f for f, _ in naming.TISSUE_FIELDS],
                    help="metadata columns to colour by, one column each "
                         f"(default: {' '.join(f for f, _ in FIG_FIELDS)})")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    main(a.out_dir, a.variants, a.fields)
