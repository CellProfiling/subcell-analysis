"""Cell-cycle phase of each cell from its 63x confocal images only.

Standard FUCCI + DNA-content gating per cell line, on the 63x cells that pass the object QC (whole nucleus; finite,
positive nuclear measurements):
  GMNN-high  log10 nuclear GMNN (z-sum) above its Otsu threshold (GMNN accumulates from S phase onwards)
  DNA        nuclear DAPI z-sum relative to the 2N peak (the DNA mode of the GMNN-low cells)
  4N peak    the DNA mode of the GMNN-high cells above 1.3 x 2N (the confocal z-sum compresses it below 2)
  G1         GMNN-low, DNA below 2N + 1/4 of the 2N-4N interval
  S          GMNN-high, DNA below 2N + 3/4 of the interval
  G2         GMNN-high, above it
  invalid    GMNN-low with more DNA than G1 (excluded)
hTERT-RPE1 (serum-starved) has no GMNN-high 4N population and is not assigned. Check on the other microscope: the
integrated 40x DAPI of the same cells (nuclei without a saturated pixel), relative to its own 2N peak, per phase."""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from skimage.filters import threshold_otsu

from subcell import config
from subcell.microscope_robustness import data

COLS = ["nuc_gmnn_mean", "nuc_cdt1_mean", "nuc_gmnnsum_mean", "nuc_cdt1sum_mean", "nuc_dapisum_int", "nuc_area_um2"]


def g1_peak(dna, lgmnn):
    """DNA mode of the GMNN-low cells (Otsu on log GMNN)"""
    lg = np.log(dna[lgmnn < threshold_otsu(lgmnn)])
    grid = np.linspace(*np.percentile(lg, [1, 99]), 400)
    return float(np.exp(grid[np.argmax(gaussian_kde(lg)(grid))]))


def mode(x):
    lg = np.log(x[np.isfinite(x) & (x > 0)])
    g = np.linspace(*np.percentile(lg, [1, 99]), 512)
    return float(np.exp(g[gaussian_kde(lg)(g).argmax()]))


def measurements_63x():
    """the QC-passed 63x nuclei of every JNH044 well that have a crop (a cell of the benchmark's crop set)"""
    crops = pd.read_parquet(config.MICROSCOPE_CROPS / "crops_index.parquet", columns=["id", "mag", "wtag", "unit", "label"])
    c63 = crops[crops.mag == "63x"].rename(columns={"id": "crop_id"})
    wells = sorted(p.parent.name for p in config.MICROSCOPE_MEASURE.glob(f"{data.EXPERIMENT}_*/63x.parquet"))
    m = pd.concat([pd.read_parquet(config.MICROSCOPE_MEASURE / w / "63x.parquet") for w in wells], ignore_index=True)
    m = m.merge(c63[["crop_id", "wtag", "unit", "label"]], on=["wtag", "unit", "label"], how="left")
    v = m[COLS].astype(float)
    m = m[(np.isfinite(v).all(1) & (v > 0).all(1) & m.qc_ok.astype(bool)).values]
    return m.dropna(subset=["crop_id"]).reset_index(drop=True)              # nuclei without a crop have no cell to label


def dna_40x(ids):
    """40x integrated nuclear DAPI (maximum projection) by crop id; nuclei with a saturated pixel -> NaN"""
    ix = pd.read_parquet(config.MICROSCOPE_CROPS / "crops_index.parquet", columns=["id", "mag", "wtag", "unit", "label"])
    ix = ix[(ix.mag == "40x") & ix.id.isin(ids)]
    out = []
    for w, g in ix.groupby("wtag"):
        m = pd.read_parquet(config.MICROSCOPE_MEASURE / w / "40x.parquet", columns=["unit", "label", "nuc_dapi_int", "nuc_dapi_satfrac"])
        out.append(g.merge(m, on=["unit", "label"], how="left"))
    d = pd.concat(out).set_index("id")
    return d.nuc_dapi_int.where(d.nuc_dapi_satfrac == 0)


def assign():
    """-> (phases: crop_id, cell_line, phase, dna_rel, gmnn_high; summary: counts per line and the 40x check)"""
    m = measurements_63x()
    d40 = dna_40x(set(m.crop_id.dropna()))
    out, rows = [], []
    for line, g in m.groupby("cell_line"):
        lg = np.log10(g.nuc_gmnnsum_mean.astype(float).values)
        dna = g.nuc_dapisum_int.astype(float).values
        hi = lg >= threshold_otsu(lg)
        rel = dna / g1_peak(dna, lg)
        p4 = mode(rel[hi & (rel > 1.3)]) if (hi & (rel > 1.3)).sum() > 50 else np.nan
        lo_b, hi_b = 1 + 0.25 * (p4 - 1), 1 + 0.75 * (p4 - 1)
        ph = np.where(~hi & (rel < lo_b), "G1", np.where(hi & (rel <= hi_b), "S", np.where(hi & (rel > hi_b), "G2", "invalid")))
        r = dict(cell_line=line, cells=len(g), peak_4n=round(p4, 3), **{f"n_{p}": int((ph == p).sum()) for p in ("G1", "S", "G2", "invalid")})
        x = d40.reindex(g.crop_id).values
        okx = np.isfinite(x) & (x > 0)
        u = mode(x[okx & (ph == "G1")]) if (okx & (ph == "G1")).sum() > 30 else np.nan
        for p in ("G1", "S", "G2"):
            k = okx & (ph == p)
            r[f"dna40_median_{p}"] = round(float(np.median(x[k] / u)), 3) if k.sum() > 10 else np.nan
            r[f"n40_{p}"] = int(k.sum())
        rows.append(r)
        out.append(pd.DataFrame({"crop_id": g.crop_id.values, "cell_line": line, "phase": ph, "dna_rel": rel, "gmnn_high": hi}))
    return pd.concat(out, ignore_index=True), pd.DataFrame(rows)


def run():
    ph, summary = assign()
    od = config.ensure(config.MICROSCOPE_DIR)
    ph.to_parquet(od / "phases_63x.parquet", index=False)
    summary.to_csv(od / "phases_63x_summary.csv", index=False)
    return ph, summary


def load():
    """crop id -> phase (G1 / S / G2 / invalid)"""
    return pd.read_parquet(config.MICROSCOPE_DIR / "phases_63x.parquet").set_index("crop_id").phase
