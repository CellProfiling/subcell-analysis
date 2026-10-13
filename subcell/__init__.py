"""Analysis and figure code for the SubCell manuscript.

Layout
------
``subcell.config``                 every filesystem path, overridable by environment variable
``subcell.naming``                 display names and orderings
``subcell.runlog``                 timestamped progress log
``subcell.utils``                  seeding, losses and metrics shared across packages
``subcell.io``                     feature, metadata and image-dataset loading
``subcell.models``                 encoders and probe heads
``subcell.style``                  the Nature figure specification (geometry, type, colour, saving)
``subcell.inference``              GPU feature and attention extraction per dataset
``subcell.classification``         probes, Fig. 2 benchmarks, Fig. 3d function prediction
``subcell.feature_analysis``       Fig. 3a-c,f,g and Extended Data Figs. 3 and 8
``subcell.perturbation``           CM4AI drug perturbation (Fig. 4a-c, Extended Data Fig. 5)
``subcell.multimodal``             MuSIC, MIRAGE and interaction enrichment (Fig. 6d, Extended Data Fig. 7b)
``subcell.microscope_robustness``  widefield vs confocal (Extended Data Fig. 1, Supplementary Note 1)
``subcell.tissue``                 tissue microarray cohort (Extended Data Fig. 2, Supplementary Note 2)
``figures/`` (repository root)     one module per manuscript figure; renders cached analysis output
"""
