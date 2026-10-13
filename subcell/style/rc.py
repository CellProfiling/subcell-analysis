"""Nature figure typography and rcParams.

The specification this encodes (Nature "Formatting guide" / "Final figure
submission", and the Nature Portfolio artwork guidelines):

  type        sans-serif, 5-7 pt *at final published size*. 7 pt for axis labels
              and titles, 6 pt for tick labels and legends, 5 pt floor. Nothing
              in a figure may fall below 5 pt.
  panel keys  lower-case bold letters, set slightly larger than body type (8 pt).
  fonts       Arial or Helvetica. Embedded as Type 42 (TrueType) so the text
              stays editable and selectable in the PDF -- Type 3 is rejected.
  rules       0.25 pt minimum stroke; below that a line drops out on press. 0.5 pt
              is used for axes so it survives any down-scaling of the figure.
  colour      RGB for the online version; no reliance on colour alone to carry
              identity (see ``palettes``).
  rasters     300 dpi minimum for colour/greyscale images, 600-1200 dpi for line
              art. Point clouds are rasterised inside an otherwise-vector PDF.

``register_fonts`` adds the fonts found in ``config.FONT_DIR`` to matplotlib's
registry. Liberation Sans has Arial's glyph widths and Nimbus Sans has
Helvetica's, so either substitutes for the specified face without reflowing text;
without any of them matplotlib falls back to DejaVu Sans, whose glyphs are wider.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl

from .. import config

# ------------------------------------------------------------------ type size
#
# Named so every figure uses the same size for the same role.

FS_MIN = 5.0        # floor from the guidelines; style.check reports anything smaller
FS_TINY = 5.0       # dense small-multiple titles, in-plot category labels
FS_SMALL = 6.0      # tick labels, legends
FS_BASE = 7.0       # axis labels, panel titles, body text
FS_PANEL = 8.0      # bold panel keys (a, b, c)

LW_MIN = 0.25       # thinnest stroke that survives printing
LW_AXIS = 0.5       # axes, ticks, rules
LW_LINE = 0.75      # data lines

# Arial first (the specified face), then the metric-compatible substitutes, then
# DejaVu so a plot still renders where none is installed.
FONT_STACK = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]

_FONTS_REGISTERED = False


def register_fonts(font_dir=None, quiet: bool = True) -> list[str]:
    """Add the Arial/Helvetica-metric fonts in ``config.FONT_DIR`` to matplotlib.

    Idempotent, and safe when the directory is absent (returns an empty list and
    warns once, since the figure will then render in DejaVu at slightly different
    widths).
    """
    global _FONTS_REGISTERED
    import matplotlib.font_manager as fm

    if _FONTS_REGISTERED:
        return sorted({f.name for f in fm.fontManager.ttflist})

    d = Path(font_dir or config.FONT_DIR)
    added = []
    if d.is_dir():
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in (".ttf", ".otf"):
                fm.fontManager.addfont(str(f))
                added.append(f.name)
    _FONTS_REGISTERED = True

    available = {f.name for f in fm.fontManager.ttflist}
    if not quiet:
        print(f"registered {len(added)} font files from {d}")
    if not (available & {"Arial", "Liberation Sans", "Helvetica", "Nimbus Sans"}):
        warnings.warn(
            f"no Arial- or Helvetica-metric font found (looked in {d}); figures "
            "will render in DejaVu Sans, whose glyphs are wider, so text laid out "
            "to fit a panel may overflow it. Set SUBCELL_FONT_DIR.",
            stacklevel=2,
        )
    return added


# Nature's requirements as rcParams. Applied through a context manager rather
# than mutating the global state at import, so importing this module never
# changes how unrelated code plots.
RC = {
    # --- type
    "font.family": "sans-serif",
    "font.sans-serif": FONT_STACK,
    "font.size": FS_BASE,
    "axes.labelsize": FS_BASE,
    "axes.titlesize": FS_BASE,
    "figure.titlesize": FS_BASE,
    "legend.fontsize": FS_SMALL,
    "legend.title_fontsize": FS_SMALL,
    "xtick.labelsize": FS_SMALL,
    "ytick.labelsize": FS_SMALL,
    "axes.titleweight": "normal",   # bold is reserved for panel keys
    "axes.labelweight": "normal",
    # --- strokes
    "axes.linewidth": LW_AXIS,
    "lines.linewidth": LW_LINE,
    "grid.linewidth": LW_MIN,
    "patch.linewidth": LW_AXIS,
    "hatch.linewidth": LW_AXIS,
    "xtick.major.width": LW_AXIS,
    "ytick.major.width": LW_AXIS,
    "xtick.minor.width": LW_MIN,
    "ytick.minor.width": LW_MIN,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.minor.size": 1.2,
    "ytick.minor.size": 1.2,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
    "axes.labelpad": 2.0,
    "axes.titlepad": 3.0,
    # --- frame: two spines, no box, no grid. A box round every panel is ink
    #     that carries no information and Nature's own figures do not use one.
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": False,
    # --- legend
    "legend.frameon": False,
    "legend.handlelength": 1.2,
    "legend.handletextpad": 0.4,
    "legend.labelspacing": 0.3,
    "legend.columnspacing": 0.8,
    "legend.borderaxespad": 0.2,
    "legend.borderpad": 0.0,
    "legend.markerscale": 1.0,
    # --- markers and errorbars
    "lines.markersize": 2.0,
    "lines.markeredgewidth": 0.0,
    "errorbar.capsize": 1.0,
    "scatter.edgecolors": "none",
    "boxplot.flierprops.markersize": 1.5,
    # --- output. Type 42 keeps text editable; Type 3 is rejected by production.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",         # SVG text stays text, not paths
    "pdf.compression": 6,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    # Panels are placed at explicit sizes, so a tight bbox that silently shrinks
    # or grows the figure would defeat the width check in save.py.
    "savefig.bbox": None,
    "savefig.pad_inches": 0.0,
    # --- text colour: near-black rather than pure black, which prints harshly
    "text.color": "#111111",
    "axes.edgecolor": "#111111",
    "axes.labelcolor": "#111111",
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    # A fixed salt for SVG element ids, so the same figure gives the same bytes.
    "svg.hashsalt": "subcell",
}


class nature:
    """rcParams context manager. ``with nature(): ...`` around every figure.

    Accepts overrides for the rare panel that needs them, so a local deviation is
    visible at the call site instead of hidden in a module-level rcParams update::

        with nature(**{"xtick.labelsize": 5}):
            ...
    """

    def __init__(self, **overrides):
        register_fonts()
        self._ctx = mpl.rc_context({**RC, **overrides})

    def __enter__(self):
        return self._ctx.__enter__()

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def use_agg() -> None:
    """Select the non-interactive backend.

    Figures are written to disk without a display; this skips backend probing.
    """
    mpl.use("Agg")


# rcParams of the Fig. 3 / Extended Data Fig. 8 plots.
FIG3_RC = {
    "font.family": "sans-serif", "font.sans-serif": FONT_STACK,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.labelsize": 5, "ytick.labelsize": 5,
}
