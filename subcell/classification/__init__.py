"""Probes trained on frozen features: the benchmarks of Fig. 2 and Fig. 3d.

``probe``                  the two-layer MLP probe, its training and per-class metrics
``evidential_regression``  deep evidential regression of FUCCI pseudotime (Fig. 2l)
``nested_cv``              grid-searched MLP probes in nested cross-validation
``localization``           HPA and Kaggle localization (Fig. 2a-c; Supplementary Table 9)
``yeast``                  yeast localization and cell cycle (Fig. 2f,h,j; Supplementary Table 1)
``cellcycle_allencell``    Allen Cell cell-cycle stage (Fig. 2g,i; Supplementary Table 10)
``cellcycle_fucci``        FUCCI U2OS stage and pseudotime (Fig. 2k,l)
``cellcycle_results``      summaries of the cell-cycle probes for ``figures.fig2_cell_cycle``
``function_prediction``    GO / Reactome membership from per-gene embeddings (Fig. 3d)
"""
