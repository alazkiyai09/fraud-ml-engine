"""Lazy autoencoder loader."""


def load_autoencoder():
    from src.models.anomaly.legacy.models.autoencoder import AutoencoderDetector

    return AutoencoderDetector
