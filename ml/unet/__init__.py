"""Dysplasia U-Net experiment: tile-level segmentation of LGD/HGD regions.

Deliberately independent from the embedding-based main pipeline
(ml/meta_arch.py, configs/ml.yaml): own hydra config (configs/unet.yaml), own
entrypoint (python -m ml.unet.train), own output dir. Only shared, read-only
reuse: configs/base.yaml and ml/data.DataModule.
"""
