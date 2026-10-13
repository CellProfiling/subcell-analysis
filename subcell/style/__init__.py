"""The Nature figure specification, as code.

Every figure module starts the same way::

    from subcell import style
    from subcell.style import layout as L

    style.use_agg()
    with style.nature():
        fig = L.figure("double", height=120)
        ax = L.axes(fig, x=12, y=8, w=60, h=45)
        ...
        style.save(fig, "ed2i_cluster_umap")

``nature()`` sets the rcParams, ``layout`` places panels in millimetres, and
``save`` checks the result against the guidelines before writing it. What each
rule is and why it is that number is documented in the submodule that owns it:

``rc``        type sizes, stroke widths, font embedding, output settings
``layout``    figure widths and heights, panel placement, panel furniture
``palettes``  colour, plus the OKLab/CVD measurement that decides a palette
``save``      the conformance check and the writer
"""

from . import layout, palettes
from .layout import (MAX_HEIGHT, MM, WIDTH_1COL, WIDTH_1_5COL, WIDTH_2COL,
                     Grid, axes, bare, despine, figure, panel_label,
                     swatch, text)
from .palettes import GHOST, GRID, INK, INK2, MUTED, SURFACE, glasbey, method_color
from .rc import (FS_BASE, FS_MIN, FS_PANEL, FS_SMALL, FS_TINY, LW_AXIS,
                 LW_LINE, LW_MIN, RC, nature, register_fonts, use_agg)
from .save import DPI, check, save

__all__ = [
    # entry points
    "nature", "use_agg", "register_fonts", "figure", "save", "check",
    # geometry
    "layout", "axes", "Grid", "bare", "despine", "panel_label", "swatch",
    "text",
    "MM", "WIDTH_1COL", "WIDTH_1_5COL", "WIDTH_2COL", "MAX_HEIGHT",
    # type
    "RC", "FS_MIN", "FS_TINY", "FS_SMALL", "FS_BASE", "FS_PANEL",
    "LW_MIN", "LW_AXIS", "LW_LINE",
    # colour
    "palettes", "glasbey", "method_color",
    "INK", "INK2", "MUTED", "GHOST", "GRID", "SURFACE",
    # output
    "DPI",
]
