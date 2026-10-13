"""Running an encoder over a dataset to produce cell-level features.

One module per imaging dataset, because each has its own channel set, pixel size
and segmentation:

``hpa``             HPA, HPA Kaggle test and CM4AI crops, all compared encoders (Figs 2-4)
``allencell``       Allen Cell hiPSC projections (Fig. 2g,i); the crops come from ``allencell_crops``
``opencell``        OpenCell crops (Fig. 2e)
``yeast``           yeast localization and cell-cycle datasets (Fig. 2f,h,j)
``fucci``           FUCCI U2OS crops (Fig. 2k,l); the crops come from ``fucci_crops``
``microscopes``     widefield / confocal crops of Extended Data Fig. 1
``attention``       attention maps scored against the input channels (Fig. 3f,g)

The loop shared by ``hpa``, ``allencell``, ``opencell``, ``yeast`` and ``fucci`` is in ``subcell.utils.inference``. These are the
GPU stages; everything downstream reads their cached output.
"""
