"""Cross-microscope benchmark (Supplementary Note 1, Extended Data Fig. 1).

The same JNH044 cells (HeLa, KOLF2.1J, U2OS, hTERT-RPE1; DNA, microtubules, GMNN, CDT1) imaged on a widefield
microscope at 40x and a confocal microscope at 63x, matched one-to-one; 10,726 cells complete in both images. Six
four-channel models and a 32 x 32-pixel reference (Pixels) embed both images of every cell; embeddings are centred
per microscope and compared by cosine similarity. Ground truth for the cell-cycle phase comes from the 63x images only.

Modules (computation only; the plots are drawn by figures/ed1_microscopes.py):
  data       cells, features, centring, measurements, the Pixels reference
  stats      stratified two-level component bootstrap, component sign-flip test, paired bootstrap P, Holm
  phases     G1 / S / G2 from 63x DNA content and GMNN, checked against the 40x DNA content of the same cells
  retrieval  normalised mAP@k (copairs' AP truncated at k) for cell, cell-line and phase identity
  transfer   linear readouts trained on one microscope and tested on the same or the other one
  geometry   linear CKA and the Mantel correlation between the two microscopes' embeddings
  examples   the example cells of panel a and the queries of the gallery

Inputs: the model features of ``subcell.inference.microscopes`` (config.MICROSCOPE_FEATURES), the crops and the
per-object measurements of the image pipeline (config.MICROSCOPE_CROPS, MICROSCOPE_MEASURE, MICROSCOPE_CELLS).

Run (outputs in config.MICROSCOPE_DIR):
  python -m subcell.microscope_robustness [pixels phases identity transfer geometry examples]   (default: all, in order)
  python -m subcell.microscope_robustness check_sklearn       (the GPU readouts against scikit-learn on one fold)
"""
