"""Helpers shared by more than one analysis.

``seed``        seeding every random number generator a run touches
``losses``      sigmoid focal loss for the multilabel probes
``metrics``     multilabel classification metrics and significance stars
``inference``   model loading and the batch loop shared by the feature extractors
``dendrogram``  Ward clustering with bootstrap and jackknife clade support
"""
