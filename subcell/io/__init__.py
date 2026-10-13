"""Loading features, annotation tables and images.

``features``          cell-level feature tensors, per-gene / per-FOV aggregation, scaling
``metadata``          annotation tables, filtered identically everywhere they are used
``datasets``          torch Datasets for each imaging dataset
``preprocess``        cropping and per-model input normalization shared by the Datasets
``display``           rendering multi-channel crops and attention maps as RGB panels
``complexes``         GO / Reactome / CORUM / hu.MAP / BioPlex / STRING group membership
``microscope_crops``  crops for the widefield vs confocal comparison
"""
