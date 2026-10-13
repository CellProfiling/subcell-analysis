"""Inputs of the cross-microscope benchmark: cells, features, centring, measurements, the Pixels reference."""

import glob
import json

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch.utils.data import DataLoader

from subcell import config, naming
from subcell.io.microscope_crops import MicroscopeCropDataset

EXPERIMENT = "JNH044"                               # the acquisition imaged on both microscopes
MAGS = ("40x", "63x")
LINES = ["HeLa", "KOLF2.1J", "U2OS", "hTERT-RPE1"]
CYCLING = ["HeLa", "KOLF2.1J", "U2OS"]              # hTERT-RPE1 is serum-starved and does not cycle
PHASES = ["G1", "S", "G2"]

# feature runs (<model>_norm_<normalisation>, the default normalisation of each model) in figure order, with the
# internal model name of subcell.naming, which gives the display names
RUNS = [("subcell_rybg_mae_norm_perimage", "MAE-CellS-ProtS-Pool"),
        ("subcell_rybg_vit_norm_perimage", "ViT-ProtS-Pool"),
        ("cell_dino_hpa_vitl16_norm_selfnorm", "Cell-DINO-HPA"),
        ("dino_norm_dino", "DINO4Cells-HPA"),
        ("subcell_vit_supervised_norm_perimage", "ViT-Supervised"),
        ("bestfitting_norm_bestfitting", "bestfitting")]
MODELS = [r for r, _ in RUNS]
SUBCELL = MODELS[0]
PIXELS = "pixels32"
ORDER = MODELS + [PIXELS]
NAME = {r: naming.display(i) for r, i in RUNS} | {PIXELS: "Pixels"}


def cells():
    """the 21,002 JNH044 cells imaged at 40x and 63x, sorted by id, with
      complete  the cell is complete in both images (63x: inside the field, not touching its border; 40x: not cut by
                its source tile, not touching the mosaic border): 10,726 cells, the cells of every test
      comp      connected components of the 63x-field / 40x-tile graph of the complete cells (two cells sharing a 63x
                field or a 40x tile are in one component; 157): the statistical unit"""
    c = pd.read_parquet(config.MICROSCOPE_CELLS).sort_values("id").reset_index(drop=True)
    c["complete"] = (c.complete4063.astype(bool) & ~c.cut_by_source_40x.astype(bool)
                     & ~c.cell_touches_border_40x.astype(bool))
    c["hc"] = c.hc4063.astype(bool) & c.complete

    def comps(sub):
        f = pd.factorize(sub.field)[0]
        t = pd.factorize(sub.tile40)[0] + f.max() + 1
        n = t.max() + 1
        _, lab = connected_components(coo_matrix((np.ones(len(sub)), (f, t)), shape=(n, n)), directed=False)
        return lab[f]

    c["comp"] = None
    k = c.complete.values
    c.loc[k, "comp"] = [f"c{v}" for v in comps(c[k])]
    return c


def complete(c=None):
    """the complete cells (the cells of every test), in id order"""
    c = cells() if c is None else c
    return c[c.complete.values].reset_index(drop=True)


def feature_dir(run):
    return config.MICROSCOPE_DIR / "features" / run if run == PIXELS else config.MICROSCOPE_FEATURES / run


def features(run, ids):
    """{mag: float32 (N, D)} with rows in the order of ids; the files' rows are sorted by id and agree between the
    two magnifications"""
    X, ids0 = {}, None
    for m in MAGS:
        meta, x = torch.load(feature_dir(run) / f"{m}_features_pooled.pth", weights_only=False)
        mid = meta["id"].astype(str).values
        assert (np.sort(mid) == mid).all(), f"{run} {m}: rows not sorted by id"
        ids0 = mid if ids0 is None else ids0
        assert (mid == ids0).all(), f"{run} {m}: ids differ between magnifications"
        x = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
        X[m] = np.ascontiguousarray(x.astype(np.float32))
    pos = pd.Index(ids0).get_indexer(ids)
    assert (pos >= 0).all(), f"{run}: lacks cells of the benchmark"
    X = {m: x[pos] for m, x in X.items()}
    for m, x in X.items():
        assert np.isfinite(x).all(), f"{run} {m}: non-finite features"
    return X


def unit(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.where(n > 0, n, 1)).astype(np.float32)


def centred(X):
    """each microscope's mean embedding over all matched cells removed (the only correction: it uses the microscope
    label and removes a constant offset that would otherwise dominate every cosine similarity)"""
    return {m: x - x.mean(0) for m, x in X.items()}


def measurements(c, mag, cols):
    """per-object measurements of each cell's own objects at one magnification, in the order of c"""
    m = pd.concat([pd.read_parquet(p, columns=["wtag", "unit", "label"] + cols)
                   for p in sorted(glob.glob(str(config.MICROSCOPE_MEASURE / f"{EXPERIMENT}_*" / f"{mag}.parquet")))])
    k = c[["well", f"unit_{mag}", f"label_{mag}"]].set_axis(["wtag", "unit", "label"], axis=1).astype({"label": int})
    j = k.merge(m.astype({"label": int}), on=["wtag", "unit", "label"], how="left", validate="many_to_one")
    return j[cols].astype(float).reset_index(drop=True)


def gmnn_cdt1(c):
    """log ratio of the nuclear GMNN and CDT1 intensities (sums over z) at 63x"""
    m = measurements(c, "63x", ["nuc_gmnnsum_int", "nuc_cdt1sum_int"])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(m.nuc_gmnnsum_int.values / m.nuc_cdt1sum_int.values)


SOURCE_PX = 2048                                    # the 63x field and the 40x tile the crops are cut from: 2,048 px squares


def full_window(c):
    """the cells whose whole window lies inside both images (no zero padding at either magnification)"""
    ix = pd.read_parquet(config.MICROSCOPE_CROPS / "crops_index.parquet", columns=["id", "mag", "source_y0", "source_x0", "crop_px"])
    ok = np.ones(len(c), bool)
    for m in MAGS:
        w = ix[ix.mag == m].set_index("id").loc[c.id]
        ok &= ((w.source_y0 >= 0) & (w.source_x0 >= 0) & (w.source_y0 + w.crop_px <= SOURCE_PX)
               & (w.source_x0 + w.crop_px <= SOURCE_PX)).values
    return ok


# ------------------------------------------------------------------------------------- the Pixels reference
def _pixels32(images):
    """[MT, DNA, CDT1, GMNN] each min-max scaled over the 448-px model input (SubCell's per-channel input), area-
    downsampled to 32 x 32 px and flattened (4,096 values)"""
    small = []
    for im in images:                                   # one image per protein, channels [red, blue, protein]
        lo, hi = im.min((0, 1)), im.max((0, 1))
        small.append(cv2.resize((im - lo) / (hi - lo + 1e-8), (32, 32), interpolation=cv2.INTER_AREA))
    cdt1, gmnn = small
    return np.concatenate([cdt1[..., :2].transpose(2, 0, 1).ravel(), cdt1[..., 2].ravel(), gmnn[..., 2].ravel()]).astype(np.float32)


def build_pixels(workers=8):
    """the Pixels reference of every matched cell at both magnifications, from the models' exact inputs (unmasked,
    the 40x crop aligned onto the 63x window, 448 px), in the model feature format"""
    c = cells()
    cp = dict(proteins=["cdt1", "gmnn"], channels=["red", "blue", "protein"], resize_to=448,
              meta_cols=["id", "wtag", "mag", "cell_line"], align={"40x": str(config.MICROSCOPE_ALIGN)})
    out = config.ensure(feature_dir(PIXELS))
    for mag in MAGS:
        ds = MicroscopeCropDataset(str(config.MICROSCOPE_CROPS), str(config.MICROSCOPE_CROPS / "crops_index.parquet"), mag, cp,
                                  _pixels32, set(c.id))
        assert len(ds) == len(c), (mag, len(ds), len(c))
        dl = DataLoader(ds, batch_size=32, num_workers=workers, collate_fn=lambda b: b)
        X, metas = [], []
        for batch in dl:
            for x, meta in batch:
                X.append(x)
                metas.append(meta)
        meta = pd.concat(metas, ignore_index=True)
        assert (meta.id.values == c.id.values).all()
        torch.save((meta, torch.from_numpy(np.stack(X))), out / f"{mag}_features_pooled.pth")
    json.dump(dict(model=PIXELS, kind="reference", description="[MT, DNA, CDT1, GMNN] 32 x 32 px, per-channel min-max, unmasked",
                   dim_per_protein=None, n_cells=len(c)), open(out / "features_layout.json", "w"), indent=1)
    return out
