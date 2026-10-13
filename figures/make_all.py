"""Render every figure, or list what each one needs.

    python -m figures.make_all --list        # inputs of each figure and whether they exist
    python -m figures.make_all               # render every figure whose inputs exist
    python -m figures.make_all fig3c ed3     # selected figures

Each entry names the cached results its figure reads. A figure whose inputs are
absent is reported and skipped: the analysis that writes them has to be run first
(see README).
"""

from __future__ import annotations

import argparse
import importlib
import traceback

from subcell import config
from subcell.perturbation import CM4AI_METADATA
from subcell.runlog import log

_ATTENTION_RUNS = ("subcell_rybg_mae", "subcell_rybg_vit")
_CLUSTER_TRANSFER_RUN = "MAE-CellS-ProtS-Pool_U2OS_harmonized_if_plate_id_microscope"
_MICROSCOPE_TABLES = ("identity.csv", "transfer.csv", "transfer_direction_delta.csv",
                      "geometry.csv", "panel_a_cells.csv", "gallery.csv")

# name -> (module, paths that must exist). ed2jk runs before ed2i_clusters because
# it writes the cluster-number crosswalk the latter reads.
FIGURES = {
    "fig2": ("figures.fig2_cell_cycle",
             [config.ALLENCELL_DIR / v / "classification" / "seed_0" / "test_preds.csv"
              for v in config.ALLENCELL_VARIANTS.values()]
             + [config.FUCCI_DIR / v / "split_0" / "classification" / "seed_0" / "val_preds.csv"
                for v in config.FUCCI_VARIANTS.values()]),
    "fig3ab": ("figures.fig3ab_similarity",
               [config.MORPHOLOGY_DIR, config.RNA_CELLINE]),
    "fig3c": ("figures.fig3c_mantel",
              [config.MORPHOLOGY_DIR]),
    "fig3d": ("figures.fig3d_function_prediction",
              config.FUNCTION_PREDICTION_REPEATS),
    "fig3fg": ("figures.fig3fg_attention",
               [config.ATTN_CORR_DIR / run / "all_relations.pth" for run in _ATTENTION_RUNS]),
    "fig4": ("figures.fig4_perturbation",
             [config.features("MAE-CellS-ProtS-Pool", "b2ai_features"), CM4AI_METADATA]),
    "ed1": ("figures.ed1_microscopes",
            [config.MICROSCOPE_DIR / f for f in _MICROSCOPE_TABLES] + [config.MICROSCOPE_CROPS]),
    "ed2jk": ("figures.ed2jk_cluster_annotation",
              [config.TISSUE_ANNOT_K13, config.TISSUE_REF_LABELS, config.TISSUE_INTENSITY,
               config.TISSUE_CLUSTERS / "cell_clusters.csv"]),
    "ed2i_umaps": ("figures.ed2i_tissue_umaps",
                   [config.TISSUE_UMAPS, config.TISSUE_META]),
    "ed2i_clusters": ("figures.ed2i_cluster_umap",
                      [config.TISSUE_UMAPS, config.TISSUE_CLUSTERS / "cell_clusters.csv",
                       config.TISSUE_CLUSTER_IDS]),
    "ed3": ("figures.ed3_harmonization",
            [config.HPA_UMAP_DIR / "retrieval" / "retrieval_map.csv", config.MUTUAL_INFO_DIR]),
    "ed7b": ("figures.ed7b_interaction_enrichment",
             [config.MULTIMODAL_ENRICHMENT_DIR]),
    "fig6_clusters": ("figures.fig6_cluster_transfer",
                      [config.CLUSTER_TRANSFER_DIR / _CLUSTER_TRANSFER_RUN / "clusters.csv"]),
    "fig6_percentile": ("figures.fig6_cluster_percentile",
                        [config.CLUSTER_TRANSFER_DIR / _CLUSTER_TRANSFER_RUN
                         / "cluster_percentile_summary.csv"]),
}


def missing(paths):
    return [p for p in paths if not p.exists()]


def report():
    print(f"{'figure':<20} {'status':<9} inputs")
    print("-" * 78)
    for name, (mod, inputs) in FIGURES.items():
        gone = missing(inputs)
        status = "ready" if not gone else f"missing {len(gone)}"
        print(f"{name:<20} {status:<9} {mod}")
        for p in gone:
            print(f"{'':<30} absent: {p}")


def main(names=None):
    todo = names or list(FIGURES)
    done, skipped, failed = [], [], []
    for name in todo:
        if name not in FIGURES:
            raise SystemExit(f"unknown figure {name!r}; have {list(FIGURES)}")
        mod_name, inputs = FIGURES[name]
        gone = missing(inputs)
        if gone:
            log(f"SKIP {name}: {len(gone)} input(s) absent, first {gone[0]}")
            skipped.append(name)
            continue
        log(f"=== {name} ({mod_name})")
        try:
            mod = importlib.import_module(mod_name)
            mod.main()
            done.append(name)
        except Exception:
            traceback.print_exc()
            failed.append(name)
    log("")
    log(f"rendered {len(done)}  skipped {len(skipped)}  failed {len(failed)}")
    if skipped:
        log(f"  skipped: {', '.join(skipped)}")
    if failed:
        log(f"  failed:  {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("figures", nargs="*", help="figure names (default: all)")
    ap.add_argument("--list", action="store_true",
                    help="show each figure's inputs and whether they exist")
    a = ap.parse_args()
    if a.list:
        report()
    else:
        raise SystemExit(main(a.figures or None))
