"""Fig. 2g,i,k -- cell-cycle stage classification on Allen Cell and FUCCI U2OS.

Panels drawn
------------
g  Allen Cell: bar plots of macro F1 (and accuracy, micro F1) per model, over ten
   seeds, plus the mean +- s.d. table behind Supplementary Table 10.
i  Allen Cell: row-normalised confusion matrix of every model, averaged over ten
   seeds (the figure shows the two best).
k  FUCCI U2OS: the same three bar plots for the three substitutes of the nuclear
   channel the dataset does not have (empty channel, constant-intensity nuclear
   mask, mean of the CDT1 and GMNN channels), over ten-fold cross-validation
   grouped by antibody (whole antibodies held out together), with ten classifiers
   per fold pooled.

Not drawn here: the yeast panels (h, j) and the pseudotime regression panel (l);
their probe results are written by ``subcell.classification.yeast`` and
``subcell.classification.cellcycle_fucci``.

Inputs are the cached probe results under ``config.ALLENCELL_DIR`` and
``config.FUCCI_DIR`` (``subcell.classification.cellcycle_allencell``,
``cellcycle_fucci``). Which model variants are plotted is
``config.ALLENCELL_VARIANTS`` / ``config.FUCCI_VARIANTS``; ``--allencell-sweep``
plots every channel combination and normalization of Supplementary Table 10
(``config.ALLENCELL_SWEEP``) into a separate folder.
Outputs: ``fig2g_*``, ``fig2i_confusion_*`` and ``fig2k_*`` in ``config.figure_dir("fig2")``.

    python -m figures.fig2_cell_cycle                     # every panel
    python -m figures.fig2_cell_cycle --datasets fucci
    python -m figures.fig2_cell_cycle --allencell-sweep   # Supplementary Table 10 variants
"""

from __future__ import annotations

import argparse

from subcell import config, style
from subcell.classification import cellcycle_results
from subcell.runlog import log

DATASETS = ("allencell", "fucci")


def main(datasets=DATASETS, out_dir=None, allencell_sweep=False, seeds=range(10)):
    style.use_agg()
    out = config.ensure(out_dir) if out_dir else config.figure_dir("fig2")

    with style.nature():
        if "allencell" in datasets:
            variants = (config.ALLENCELL_SWEEP if allencell_sweep
                        else config.ALLENCELL_VARIANTS)
            log(f"Allen Cell: {len(variants)} variants -> panels g, i")
            target = config.ensure(out / "supplementary_table10") if allencell_sweep else out
            cellcycle_results.plot_allencell(variants, config.ALLENCELL_DIR, target, seeds)
        if "fucci" in datasets:
            log(f"FUCCI U2OS: {len(config.FUCCI_VARIANTS)} variants -> panel k")
            cellcycle_results.plot_fucci(config.FUCCI_VARIANTS, config.FUCCI_DIR, out, seeds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--allencell-sweep", action="store_true",
                    help="the full channel/normalization sweep of Supplementary "
                         "Table 10 instead of the four reported variants")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    main(a.datasets, a.out_dir, a.allencell_sweep)
