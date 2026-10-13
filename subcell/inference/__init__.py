"""Encoder inputs and cell-level features, one module per imaging dataset.

Feature extraction (GPU), each run from a file in ``configs/``:

``hpa``              HPA, HPA Kaggle test and CM4AI crops, all compared encoders (Figs 2-4)
``attention``        attention maps of the HPA cells scored against the input channels (Fig. 3f,g)
``allencell``        Allen Cell hiPSC projections (Fig. 2g,i)
``fucci``            FUCCI U2OS crops (Fig. 2k,l)
``opencell``         OpenCell crops (Fig. 2e)
``yeast``            yeast localization and cell-cycle datasets (Fig. 2f,h,j)
``microscopes``      widefield / confocal crops (Extended Data Fig. 1)

Input preparation:

``hpa_metadata``     the ``crops_metadata.csv`` of the HPA crops
``kaggle_metadata``  the metadata of the Kaggle test crops
``allencell_crops``  Allen Cell projections at the HPA pixel size
``fucci_crops``      FUCCI U2OS single-cell crops
``kaggle_features``  files the Kaggle features beside the HPA ones (``config.FEATURE_DIR``)

The batch loop shared by the feature extractors is ``subcell.utils.inference``.
"""

