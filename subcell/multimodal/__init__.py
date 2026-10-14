"""Image and sequence co-embedding (Fig. 6, Extended Data Fig. 7).

``music``                   MuSIC co-embedding of the per-gene image embeddings
``mirage``                  MIRAGE co-embedding
``interaction_enrichment``  similarity enrichment of interacting protein pairs (Fig. 6d, Extended Data Fig. 7b)
``embeddings``              the image, sequence and co-embedding of one MuSIC run, and their similarities
``cluster_transfer``        image and sequence clusters on the co-embedding (Fig. 6a,b)
``cluster_percentile``      how those clusters hold together in each space (Extended Data Fig. 7a)
``compartment_labels``      compartment and process labels of the proteins, and how they were made
``compartment_enrichment``  similarity enrichment within compartments and processes (Fig. 6c,f, Extended Data Fig. 7c)
``paralog_ranks``           paralog pairs by sequence and multimodal similarity (Fig. 6e)

Slurm jobs for the analyses of Fig. 6a-c,e,f are in ``subcell/multimodal/slurm/``.
"""
