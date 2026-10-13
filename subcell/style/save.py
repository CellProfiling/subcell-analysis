"""Writing a figure, and checking it against the guidelines before it goes out.

:func:`check` walks the finished figure and measures what the guidelines actually
constrain -- physical size, type size, stroke width, font embedding, raster
resolution. It runs on every save, so a violation is caught at the point it is
introduced rather than in a production query weeks later.

The checks are on the *rendered* figure, not on the rcParams. That matters: an
rcParams-only check passes a figure whose axis labels were set to 4 pt at the call
site, which is exactly the mistake that gets made.
"""

from __future__ import annotations

from pathlib import Path

from .. import config
from . import layout
from .rc import FS_BASE, FS_MIN, FS_PANEL, LW_MIN

LW_MAX = 1.0        # strokes are set between 0.25 and 1 pt

# Vector PDF for submission, PNG for viewing.
FORMATS = ("pdf", "png")

# Nature asks for 300 dpi minimum on colour rasters and 600-1200 on line art.
# Point clouds here are rasterised inside a vector PDF, so the figure dpi sets
# their resolution while all text stays vector at any dpi.
DPI = 600
DPI_MIN = 300
# Extended Data files "must not exceed 10 Mb"; a JPEG is written at the highest
# resolution that stays under it.
JPEG_MAX_BYTES = 10_000_000
JPEG_DPI_MAX = 1200


def check(fig, *, dpi: int = DPI, strict: bool = False) -> list[str]:
    """Guideline violations in a finished figure, as a list of messages.

    Empty means the figure conforms. With *strict*, raises instead of returning.
    """
    problems = []
    w_mm, h_mm = layout.size_mm(fig)

    if w_mm > layout.WIDTH_2COL + 0.5:
        problems.append(
            f"width {w_mm:.1f} mm exceeds the 183 mm double-column maximum")
    elif (w_mm > layout.WIDTH_1_5COL + 0.5 and w_mm < layout.WIDTH_2COL - 0.5
          and abs(w_mm - layout.ED_WIDTH) > 0.5):
        # Not a violation, but a figure between the standard widths gets scaled
        # to one of them, and scaling is what pushes type below 5 pt.
        problems.append(
            f"width {w_mm:.1f} mm is between the standard widths (89 / 120 / 183 mm); "
            "it will be scaled to fit a column, taking all type down with it")
    if h_mm > layout.MAX_HEIGHT + 0.5:
        problems.append(f"height {h_mm:.1f} mm exceeds the {layout.MAX_HEIGHT:g} mm "
                        "page maximum")

    if dpi < DPI_MIN:
        problems.append(f"{dpi} dpi is below the 300 dpi raster minimum")

    # --- type. Every Text in the figure, including tick labels, which are only
    #     created during a draw, so the figure is drawn first.
    fig.canvas.draw()
    small, large = {}, {}
    for t in fig.findobj(match=lambda o: hasattr(o, "get_fontsize")):
        try:
            if not str(t.get_text()).strip():
                continue
        except Exception:
            continue
        size = float(t.get_fontsize())
        if size < FS_MIN - 1e-6:
            small.setdefault(round(size, 2), []).append(str(t.get_text())[:24])
        # 8 pt is reserved for the bold panel keys; everything else tops out at 7
        elif size > FS_BASE + 1e-6 and not (
                abs(size - FS_PANEL) < 1e-6 and t.get_fontweight() == "bold"):
            large.setdefault(round(size, 2), []).append(str(t.get_text())[:24])
    for size, examples in sorted(small.items()):
        problems.append(
            f"{len(examples)} text object(s) at {size:g} pt, below the {FS_MIN:g} pt "
            f"minimum (e.g. {', '.join(examples[:3])!r})")
    for size, examples in sorted(large.items()):
        problems.append(
            f"{len(examples)} text object(s) at {size:g} pt, above the {FS_BASE:g} pt "
            f"maximum for anything but a panel key (e.g. {', '.join(examples[:3])!r})")

    # --- strokes. Hairlines drop out on press; 0 is matplotlib's "as thin as
    #     possible", which is a printer-dependent hairline, not a width.
    thin = thick = 0
    for ax in fig.get_axes():
        widths = [float(line.get_linewidth()) for line in ax.get_lines()]
        widths += [float(s.get_linewidth()) for s in ax.spines.values() if s.get_visible()]
        thin += sum(0.0 < lw < LW_MIN - 1e-6 for lw in widths)
        thick += sum(lw > LW_MAX + 1e-6 for lw in widths)
    if thin:
        problems.append(f"{thin} stroke(s) thinner than the {LW_MIN:g} pt minimum")
    if thick:
        problems.append(f"{thick} stroke(s) thicker than the {LW_MAX:g} pt maximum")

    if strict and problems:
        raise ValueError("figure does not conform:\n  " + "\n  ".join(problems))
    return problems


def save(fig, name: str, out_dir=None, *, formats=FORMATS, dpi: int = DPI,
         subdir: str | None = None, close: bool = True, strict: bool = False,
         quiet: bool = False) -> list[Path]:
    """Write *fig* in every format, after checking it.

    Returns the paths written. Violations are printed, and raise under *strict*;
    the default is to print, because a draft render of a figure whose data is
    still changing should not be blocked by a 4.8 pt tick label.
    """
    out = Path(out_dir) if out_dir is not None else config.figure_dir(subdir)
    config.ensure(out)

    problems = check(fig, dpi=dpi, strict=strict)
    w_mm, h_mm = layout.size_mm(fig)
    if not quiet:
        print(f"  {name}: {w_mm:.1f} x {h_mm:.1f} mm at {dpi} dpi", flush=True)
    for p in problems:
        print(f"  WARNING {p}", flush=True)

    written = []
    for ext in formats:
        path = out / f"{name}.{ext}"
        if ext in ("jpg", "jpeg"):
            used = write_jpeg(fig, path)
            note = f" ({used} dpi, {path.stat().st_size / 1e6:.1f} MB)"
        else:
            # facecolor is passed explicitly: rcParams' savefig.facecolor is
            # honoured inconsistently across backends for figures built with add_axes.
            fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
            note = ""
        written.append(path)
        if not quiet:
            print(f"    -> {path}{note}", flush=True)

    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return written


def write_jpeg(fig, path, *, max_bytes: int = JPEG_MAX_BYTES,
               dpi_max: int = JPEG_DPI_MAX, quality: int = 95) -> int:
    """The highest-resolution RGB JPEG of *fig* that fits in *max_bytes*.

    Rendered once at *dpi_max* and resampled down, so the search costs one render
    rather than one per candidate. Chroma is not subsampled, which keeps thin
    coloured lines and small coloured text sharp. Returns the dpi written.
    """
    import io

    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi_max, facecolor=fig.get_facecolor())
    full = Image.open(buf).convert("RGB")

    def encode(d):
        im = full if d == dpi_max else full.resize(
            (round(full.width * d / dpi_max), round(full.height * d / dpi_max)),
            Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, "JPEG", quality=quality, subsampling=0, optimize=True, dpi=(d, d))
        return out.getvalue()

    best_dpi, best = dpi_max, encode(dpi_max)
    if len(best) > max_bytes:
        lo, hi, best = DPI_MIN, dpi_max, None
        while hi - lo > 5:
            mid = (lo + hi) // 2
            data = encode(mid)
            if len(data) <= max_bytes:
                lo, best = mid, data
            else:
                hi = mid
        if best is None:
            best = encode(lo)
            if len(best) > max_bytes:
                raise ValueError(f"{path}: over {max_bytes / 1e6:g} MB even at {lo} dpi")
        best_dpi = lo
    Path(path).write_bytes(best)
    return best_dpi
