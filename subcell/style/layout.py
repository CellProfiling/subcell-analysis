"""Figure geometry in millimetres, and the panel furniture every figure repeats.

Nature specifies figure size in millimetres at the published size, so that is the
unit here; matplotlib's inches appear only inside :func:`figure`. A figure is
built by placing axes at explicit millimetre positions rather than by a
constrained layout, because the panel sizes are what the guidelines constrain --
a solver that satisfies a tight bounding box will happily hand back a 210 mm-wide
figure.

Widths
------
======================  =======  ==================================================
single column            89 mm   the default; most panels are better read small
1.5 column              120 mm   a wide heatmap or a 4-panel row
double column           183 mm   grids and maps only
maximum height          247 mm   one page less the caption; hard limit
Extended Data page      180 mm   wide by at most 170 mm tall, legend set below
======================  =======  ==================================================

Origin for :func:`axes` is the **top-left** of the figure, x rightwards and y
downwards, because figures are described top-to-bottom in captions and reading a
layout written that way does not require inverting every coordinate in your head.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import palettes as pal
from .rc import FS_PANEL, FS_SMALL, FS_TINY, LW_AXIS

MM = 1.0 / 25.4                 # millimetres -> inches

WIDTH_1COL = 89.0
WIDTH_1_5COL = 120.0
WIDTH_2COL = 183.0
MAX_HEIGHT = 247.0
ED_WIDTH = 180.0
ED_MAX_HEIGHT = 170.0

WIDTHS = {"single": WIDTH_1COL, "1.5": WIDTH_1_5COL, "double": WIDTH_2COL}


def resolve_width(width) -> float:
    """A width name or a number of millimetres, as millimetres."""
    if isinstance(width, str):
        if width not in WIDTHS:
            raise ValueError(f"width must be a number of mm or one of {list(WIDTHS)}")
        return WIDTHS[width]
    return float(width)


# ------------------------------------------------------------------- figures

def figure(width="single", height=None, *, aspect=None, strict=True):
    """A figure sized in millimetres.

    Exactly one of *height* (mm) or *aspect* (height/width) is required.

    With ``strict``, exceeding 183 mm wide or 247 mm tall raises rather than
    warns: an oversized figure is not a style preference, it is a figure
    production will reject or reduce, and a reduction silently takes the type
    below the 5 pt floor. Pass ``strict=False`` for a working draft.
    """
    import matplotlib.pyplot as plt

    w = resolve_width(width)
    if (height is None) == (aspect is None):
        raise ValueError("pass exactly one of height (mm) or aspect (h/w)")
    h = float(height) if height is not None else w * float(aspect)

    problems = []
    if w > WIDTH_2COL + 1e-6:
        problems.append(f"width {w:.1f} mm exceeds the 183 mm double-column maximum")
    if h > MAX_HEIGHT + 1e-6:
        problems.append(f"height {h:.1f} mm exceeds the {MAX_HEIGHT:g} mm page maximum")
    if problems and strict:
        raise ValueError("; ".join(problems) + " (pass strict=False for a draft)")
    for p in problems:
        print(f"  WARNING {p}")

    return plt.figure(figsize=(w * MM, h * MM))


def size_mm(fig) -> tuple[float, float]:
    w, h = fig.get_size_inches()
    return w * 25.4, h * 25.4


def axes(fig, x, y, w, h, **kw):
    """Axes at a millimetre position measured from the figure's top-left corner.

    ``x, y`` is the top-left of the axes box; ``w, h`` its size.
    """
    fw, fh = size_mm(fig)
    return fig.add_axes([x / fw, 1.0 - (y + h) / fh, w / fw, h / fh], **kw)


def text(fig, x, y, s, *, va="top", ha="left", **kw):
    """Figure text at a millimetre position from the top-left corner."""
    fw, fh = size_mm(fig)
    return fig.text(x / fw, 1.0 - y / fh, s, va=va, ha=ha, **kw)


# --------------------------------------------------------------- panel grids


@dataclass(frozen=True)
class Grid:
    """A regular grid of panels, solved so the whole thing fits a given width.

    ``Grid(n_col, n_row, width, ...)`` computes the panel size from the width and
    the gaps rather than the other way round, which is the direction that keeps a
    figure inside 183 mm. ``pos(i, j)`` gives the millimetre box for a cell.
    """

    n_col: int
    n_row: int
    width: float
    left: float = 10.0
    right: float = 2.0
    top: float = 6.0
    bottom: float = 8.0
    wspace: float = 2.0
    hspace: float = 2.0
    aspect: float = 1.0        # panel height / panel width

    @property
    def panel_w(self) -> float:
        inner = self.width - self.left - self.right - self.wspace * (self.n_col - 1)
        if inner <= 0:
            raise ValueError(
                f"{self.n_col} columns and {self.wspace} mm gaps leave no room in "
                f"{self.width} mm; reduce the margins or the column count")
        return inner / self.n_col

    @property
    def panel_h(self) -> float:
        return self.panel_w * self.aspect

    @property
    def height(self) -> float:
        return (self.top + self.bottom + self.panel_h * self.n_row
                + self.hspace * (self.n_row - 1))

    def pos(self, row: int, col: int) -> tuple[float, float, float, float]:
        return (self.left + col * (self.panel_w + self.wspace),
                self.top + row * (self.panel_h + self.hspace),
                self.panel_w, self.panel_h)

    def ax(self, fig, row: int, col: int, **kw):
        return axes(fig, *self.pos(row, col), **kw)


# ------------------------------------------------------------- panel furniture


def panel_label(fig, letter: str, x: float, y: float, **kw):
    """A panel key: lower-case, bold, at a millimetre position.

    Lower-case bold is Nature's convention. Set at 8 pt, one step above body type,
    so the key is findable without competing with the data.
    """
    opts = dict(fontsize=FS_PANEL, fontweight="bold", color=pal.INK,
                ha="left", va="top")
    opts.update(kw)
    return text(fig, x, y, letter, **opts)


def swatch(fig, x, y, w, h, color, **kw):
    """A filled rectangle at a millimetre position, added to the figure.

    Legend keys in these figures are hand-placed rather than produced by
    ``ax.legend``: the entries have to line up with panels positioned to the
    millimetre, and a legend that sizes itself moves them.
    """
    from matplotlib.patches import Rectangle

    fw, fh = size_mm(fig)
    r = Rectangle((x / fw, 1.0 - (y + h) / fh), w / fw, h / fh,
                  facecolor=color, edgecolor="none", transform=fig.transFigure,
                  figure=fig, **kw)
    fig.patches.append(r)
    return r


def bare(ax):
    """Strip an axes to its content: no ticks, no spines.

    For embeddings and micrographs, where the axes units are arbitrary and a
    frame plus four tick labels is pure ink. Embeddings get :func:`axis_key`
    instead.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def despine(ax, keep=("left", "bottom")):
    """Leave only the named spines."""
    for name, s in ax.spines.items():
        s.set_visible(name in keep)
    return ax


def axis_key(ax, emb, labels=("UMAP 1", "UMAP 2"), frac=0.16):
    """An L-shaped key in the corner instead of axes on every panel of a grid.

    A 30-panel embedding grid with full axes spends most of its ink on 120
    meaningless tick labels. One key on one panel says the same thing: the axes
    are arbitrary but the two directions are these.
    """
    x0, x1 = float(np.min(emb[:, 0])), float(np.max(emb[:, 0]))
    y0, y1 = float(np.min(emb[:, 1])), float(np.max(emb[:, 1]))
    span = x1 - x0
    pad, ln = 0.03 * span, frac * span
    bx, by = x0 + pad, y0 + pad
    line = dict(color=pal.INK2, lw=0.6, solid_capstyle="butt", clip_on=False,
                zorder=6)
    ax.plot([bx, bx + ln], [by, by], **line)
    ax.plot([bx, bx], [by, by + ln], **line)
    ax.text(bx + ln / 2, by - 0.035 * (y1 - y0), labels[0], color=pal.INK2,
            fontsize=FS_TINY, ha="center", va="top")
    ax.text(bx - 0.028 * span, by + ln / 2, labels[1], color=pal.INK2,
            fontsize=FS_TINY, ha="right", va="center", rotation=90)


def outlined_text(ax, x, y, s, color, *, fontsize=FS_TINY, stroke=1.2, **kw):
    """A category label drawn over a dense scatter, kept legible by a halo.

    Direct labelling is what carries identity in the high-cardinality figures, so
    the label has to survive being drawn on top of 200,000 points of arbitrary
    colour. The halo is white and thin enough not to erase the data under it.
    """
    import matplotlib.patheffects as pe

    opts = dict(ha="center", va="center", fontweight="bold", zorder=7)
    opts.update(kw)
    t = ax.text(x, y, s, color=color, fontsize=fontsize, **opts)
    t.set_path_effects([pe.withStroke(linewidth=stroke, foreground="white")])
    return t


def centroids(emb, codes, n) -> np.ndarray:
    """Median position per category, for direct labelling.

    Median rather than mean: a category with a detached satellite lobe puts its
    mean in the gap between the two, which is where no point of that category is.
    """
    out = np.full((n, 2), np.nan)
    for c in range(n):
        m = codes == c
        if m.any():
            out[c] = np.median(emb[m, 0]), np.median(emb[m, 1])
    return out


# ----------------------------------------------------------- scatter density


def marker_size(emb, panel_mm: float, dpi: int = 600, *, min_px: float = 2.2,
                fudge: float = 1.25, n_sub: int = 40000, seed: int = 0) -> float:
    """Marker diameter in points, from the embedding's own point spacing.

    Computed, not chosen. The median nearest-neighbour spacing of a 210,000-point
    embedding is ~0.07% of its width, so in a 60 mm panel the largest
    non-overlapping marker is a fraction of a point -- well under one pixel even
    at 600 dpi. Overlap is therefore limited by output resolution rather than by
    marker size, and the correct response is the smallest marker the raster can
    still render as a solid dot (~2 px), drawn at high alpha. That gives a crisp
    density field where one cell is one pixel, instead of a translucent blob
    whose colour washes out wherever points stack.

    Returns a *diameter* in points; matplotlib's ``s`` is an area, so pass
    ``s=marker_size(...) ** 2``.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    k = min(n_sub, len(emb))
    s = rng.choice(len(emb), k, replace=False)
    d, _ = cKDTree(emb[s]).query(emb[s], k=2)
    # Nearest-neighbour distance scales as 1/sqrt(n) in 2-D, so a subsample's
    # spacing is corrected back to the full set rather than used directly.
    nn = float(np.median(d[:, 1])) * np.sqrt(k / len(emb))
    width = float(np.max(emb[:, 0]) - np.min(emb[:, 0]))
    diam = nn / width * (panel_mm * MM) * 72.0 * fudge
    return max(diam, min_px * 72.0 / dpi)


def shuffled(n: int, seed: int = 0) -> np.ndarray:
    """A draw order that occludes no category systematically.

    One ``scatter`` call per category paints the last category permanently on top,
    which at 29 categories decides what the reader sees. A single shuffled pass
    makes occlusion unbiased.
    """
    return np.random.default_rng(seed).permutation(n)


def limits(ax, emb, pad=0.02):
    """Equal-aspect limits with a small margin. Never distort an embedding."""
    x0, x1 = float(np.min(emb[:, 0])), float(np.max(emb[:, 0]))
    y0, y1 = float(np.min(emb[:, 1])), float(np.max(emb[:, 1]))
    ax.set_xlim(x0 - pad * (x1 - x0), x1 + pad * (x1 - x0))
    ax.set_ylim(y0 - pad * (y1 - y0), y1 + pad * (y1 - y0))
    ax.set_aspect("equal")
    return ax


def colorbar(fig, mappable, x, y, w, h, label=None, *, orientation="vertical",
             **kw):
    """A colourbar as an explicitly placed axes, in millimetres.

    ``fig.colorbar`` steals space from a neighbour, which moves a panel that was
    positioned to the millimetre.
    """
    cax = axes(fig, x, y, w, h)
    cb = fig.colorbar(mappable, cax=cax, orientation=orientation, **kw)
    cb.outline.set_linewidth(LW_AXIS)
    cb.ax.tick_params(labelsize=FS_SMALL, width=LW_AXIS, length=1.6, pad=1.2)
    if label:
        cb.set_label(label, fontsize=FS_SMALL, labelpad=2)
    return cb
