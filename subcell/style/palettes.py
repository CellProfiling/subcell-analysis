"""Colour, and the arithmetic that decides whether a palette is usable.

Colour never carries identity alone. No palette separates many categories
reliably: the best available 29-slot palette reaches a worst-pair OKLab ΔE of
10.2 in normal vision, below the ~15 discriminability floor, and 0.7 under
simulated protanopia (two of the 29 are effectively the same colour). Even
Okabe-Ito, built for colour-vision deficiency, clears the floor under all three
deficiencies only at two categories. High-cardinality figures therefore
direct-label their categories and provide single-category small multiples where
per-category reading is required. :func:`check` measures this; ``python -m
subcell.style.palettes`` prints the measurements for the palettes used.

The ΔE floor of 15 is a working threshold for "two colours are still distinct
when each covers a few pixels", the regime of a 210,000-point UMAP, paired with a
severity-1.0 dichromacy simulation (the strict end of the range). The numbers
rank palettes and show that colour alone is insufficient; they are not an
absolute verdict on a given figure.

Also here: neutrals, per-model colours, and the ordinal scales of the tissue
annotation heatmap.
"""

from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------- neutrals
SURFACE = "#ffffff"     # figure and axes background
INK = "#111111"         # primary text and rules; not pure black, which prints hard
INK2 = "#3f3f3f"        # secondary text: row labels, axis keys
MUTED = "#7a7a7a"       # de-emphasised text: excluded categories, footnotes
GHOST = "#e6e6e6"       # context layer behind a highlighted subset
GRID = "#d5d5d5"        # heatmap cell borders


# ---------------------------------------------------------- model identity
#
# Bound to the model, not to its position in a list, so reordering rows or
# legends never repaints a series between two figures of the same set.

METHOD_COLOR = {
    "MAE-CellS-ProtS-Pool": "#0072b2",   # SubCell        (Okabe-Ito blue)
    "ViT-ProtS-Pool": "#e69f00",         # SubCell-P      (orange)
    "DINO4Cells-HPA": "#009e73",         # DINO4Cells-HPA (bluish green)
    "Cell-DINO-HPA": "#000000",          # Cell-DINO-HPA  (Okabe-Ito black)
    "DINO4Cells-CP": "#56b4e9",          # DINO4Cells-CP   (sky blue)
    "ViT-Supervised": "#cc79a7",         # reddish purple
    "bestfitting": "#d55e00",            # vermillion
    "CellProfiler": "#7a7a7a",           # the classical baseline, deliberately grey
    "DeepProfiler": "#b3a2c7",
}


# Model colours of Fig. 3 and Extended Data Fig. 8.
FIG3_COLOR = {
    "MAE-CellS-ProtS-Pool": "#5090BC",
    "ViT-ProtS-Pool": "#C03D3E",
    "DINO4Cells-HPA": "#DC8569",
    "Cell-DINO-HPA": "#CCB974",
    "ViT-Supervised": "#508F74",
    "bestfitting": "#7D72B2",
}


def method_color(method: str, fallback: str = MUTED) -> str:
    """Colour of a model by its internal name."""
    return METHOD_COLOR.get(method, fallback)


# ------------------------------------------------------- categorical palettes

# Okabe-Ito: eight colours constructed to degrade as gracefully as possible under
# protanopia, deuteranopia and tritanopia. It is the default below nine categories
# because it maximises the minimum separation, not because it passes: measured by
# :func:`check`, the first two colours clear the floor under every deficiency
# (min ΔE 29.2) and every larger prefix does not (n>=3: min ΔE 8.6, falling to 7.6
# by n=4 and flat thereafter). Above two categories, colour is a redundant cue
# here and never the only one.
OKABE_ITO = ["#0072b2", "#e69f00", "#009e73", "#cc79a7",
             "#56b4e9", "#d55e00", "#f0e442", "#000000"]

# Two-level pairs, checked with :func:`check`; used where a figure splits on a
# binary covariate and the two colours must never be confused.
TISSUE_PAL = {"Pancreas": "#1f4e9c", "Liver": "#d94801"}
CANCER_PAL = {"No": "#2a9d8f", "Yes": "#b2182b"}
CONDITION_PAL = {"Normal": "#e8e8e8", "Cancer": "#555555"}

# Manual quality assessment of the tissue clusters (Extended Data Fig. 2j). The
# fields are graded, not boolean: "yes", a percentage, or blank.
QUALITY_SCALE = {"yes": "#4f9d69", "pct": "#c5e2d0", "": "#ffffff"}


def glasbey(n: int, variant: str = "glasbey_bw_minc_20") -> list[str]:
    """*n* maximally separated colours in a fixed order.

    Glasbey construction (Glasbey et al. 2007) as implemented in colorcet:
    colours are chosen by maximising perceptual distance from all previously
    chosen ones, rather than picked by hand. Measured on 29 slots, worst-pair
    normal-vision ΔE:

        glasbey_bw_minc_20   10.2   <- default, best of the family
        glasbey                10.1
        glasbey_dark            9.8
        glasbey_bw              8.8
        glasbey_hv              7.6

    Still below the ΔE 15 floor -- at 29 categories nothing clears it -- so a
    figure using this must direct-label. The order is fixed and never cycled: a
    palette that wraps assigns one colour to two categories.
    """
    import colorcet as cc

    pal = getattr(cc, variant)
    hexes = [c if isinstance(c, str) else to_hex(c) for c in pal]
    if n > len(hexes):
        raise ValueError(f"colorcet.{variant} has {len(hexes)} slots, asked for {n}")
    return hexes[:n]


def categorical(n: int, labelled: bool = False) -> list[str]:
    """The right categorical palette for *n* categories.

    Up to eight, Okabe-Ito, the best-separated small palette. Above eight,
    glasbey, whose worst pair is indistinguishable under protanopia; *labelled*
    must then be True, to acknowledge at the call site that the figure
    direct-labels its categories rather than relying on the key.
    """
    if n <= len(OKABE_ITO):
        return OKABE_ITO[:n]
    if not labelled:
        raise ValueError(
            f"{n} categories cannot be separated by colour alone (best available "
            f"worst-pair ΔE ≈ {check(glasbey(n))['normal']:.1f}, floor 15). Pass "
            "labelled=True once the figure direct-labels each category."
        )
    return glasbey(n)


# ------------------------------------------------------ measurement: OKLab
#
# OKLab (Ottosson 2020) is used rather than CIELAB because its hue lines are
# straighter and its lightness is closer to perceptual, so a Euclidean distance
# in it is a better stand-in for "these two look different". Distances are
# reported x100 so the numbers land on a familiar ΔE-like scale.

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])

# Machado, Oliveira & Fernandes (2009) dichromacy matrices at severity 1.0,
# applied in linear RGB. Chosen over the Brettel construction because they are a
# single matrix per deficiency and reproduce it to within a JND.
_CVD = {
    "protan": np.array([[0.152286, 1.052583, -0.204868],
                        [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968],
                        [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779],
                        [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]]),
}

DELTA_E_FLOOR = 15.0    # working threshold for "distinct at a few pixels"


def to_hex(rgb) -> str:
    r, g, b = (int(round(255 * float(v))) for v in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _srgb(colors) -> np.ndarray:
    """(n, 3) sRGB in 0..1 from anything matplotlib understands."""
    from matplotlib.colors import to_rgb

    seq = colors.values() if isinstance(colors, dict) else colors
    return np.array([to_rgb(c) for c in seq], dtype=float)


def _linearize(srgb: np.ndarray) -> np.ndarray:
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def oklab(colors) -> np.ndarray:
    """(n, 3) OKLab coordinates."""
    lin = _linearize(_srgb(colors))
    lms = np.cbrt(lin @ _M1.T)
    return lms @ _M2.T


def simulate(colors, deficiency: str) -> np.ndarray:
    """sRGB as seen with *deficiency* in ``{protan, deutan, tritan}``."""
    lin = _linearize(_srgb(colors)) @ _CVD[deficiency].T
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def worst_pair(colors) -> tuple[float, int, int]:
    """Smallest pairwise OKLab distance (x100) and the two indices attaining it."""
    lab = oklab(colors) * 100.0
    d = np.linalg.norm(lab[:, None, :] - lab[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return float(d[i, j]), int(i), int(j)


def check(colors, floor: float = DELTA_E_FLOOR) -> dict:
    """Worst-pair separation in normal vision and under each dichromacy.

    Returns the four worst-pair ΔE values, the colliding pair in each case, the
    lightness range and the mean chroma. A palette *passes* when every value
    clears *floor*; ``result["passes"]`` says so. Nothing in this module raises on
    a failure -- a large categorical palette cannot pass, and the response is to
    direct-label, not to pick different colours.
    """
    seq = list(colors.values()) if isinstance(colors, dict) else list(colors)
    keys = list(colors) if isinstance(colors, dict) else list(range(len(seq)))

    out = {"n": len(seq)}
    for name, cols in [("normal", seq)] + [(d, simulate(seq, d)) for d in _CVD]:
        de, i, j = worst_pair(cols)
        out[name] = de
        out[f"{name}_pair"] = (keys[i], keys[j])

    lab = oklab(seq)
    out["lightness_range"] = (float(lab[:, 0].min()), float(lab[:, 0].max()))
    out["mean_chroma"] = float(np.hypot(lab[:, 1], lab[:, 2]).mean() * 100)
    out["floor"] = floor
    out["passes"] = all(out[k] >= floor for k in ("normal", *_CVD))
    return out


def report(colors, name: str = "palette", floor: float = DELTA_E_FLOOR) -> str:
    """:func:`check` as a block of text for a run log or a methods note."""
    r = check(colors, floor)
    lines = [f"{name}: {r['n']} colours, floor ΔE {floor:g} "
             f"-> {'PASS' if r['passes'] else 'FAIL'}"]
    for k in ("normal", "protan", "deutan", "tritan"):
        a, b = r[f"{k}_pair"]
        lines.append(f"  {k:<8} worst ΔE {r[k]:5.1f}   ({a} vs {b})")
    lo, hi = r["lightness_range"]
    lines.append(f"  lightness {lo:.2f}-{hi:.2f}   mean chroma {r['mean_chroma']:.1f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    print(report(OKABE_ITO, "Okabe-Ito (8)"))
    print()
    for v in ("glasbey_bw_minc_20", "glasbey", "glasbey_dark", "glasbey_bw",
              "glasbey_hv"):
        print(report(glasbey(29, v), f"{v} (29)"))
    print()
    print(report(TISSUE_PAL, "tissue"))
    print(report(CANCER_PAL, "cancer"))
