"""Encoders, probe heads and checkpoint loading.

``vit_pool``                             the SubCell encoders (ViT with attention pooling)
``vision_transformer``                   the DINO ViT of the DINO4Cells baselines
``cell_dino``                            Cell-DINO wrapper for the dinov2 backbone
``inception_v3``, ``cls_inception_v3``   the bestfitting baseline
``mlp``                                  probe heads used on frozen features
``load``                                 config block -> ready encoder, per method
``mirage``                               MIRAGE image/sequence co-embedding (Extended Data Fig. 7b)
"""
