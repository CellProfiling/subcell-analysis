"""What the HPA feature spaces encode: Fig. 3a-c,f,g and Extended Data Figs. 3 and 8.

``harmonization``          per-gene mean features and Harmony over microscope, cell line, plate
``embedding``              cached UMAP coordinates of FOV features (Extended Data Fig. 3a-d)
``retrieval``              cross-plate retrieval mAP (Extended Data Fig. 3e)
``mutual_info``            mutual information of each dimension with the annotations (Extended Data Fig. 3f)
``similarity``             cell-line and localization similarity, Mantel statistics (Fig. 3a-c)
``attention_correlation``  attention-head profiles of the HPA fields of view (Fig. 3f,g)
``subspace_overlap``       overlap of the cell-line and localization subspaces (Methods)
"""
