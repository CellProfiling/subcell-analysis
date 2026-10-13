"""Tissue microarray analysis (Supplementary Note 2, Extended Data Fig. 2i-k).

``harmonization``       Harmony over the candidate batch keys and the UMAPs
``clustering``          k-means, Ward, GMM and Leiden with the cluster-quality scores
``cluster_comparison``  every (batch key, clustering family, k) combination
``kmeans_k13``          the reported partition: k-means with k = 13 on core-corrected features
``reference_labels``    the reference labelling and the label transfer onto it
``prevalence_support``  bootstrap and jackknife support of the prevalence tree
"""
