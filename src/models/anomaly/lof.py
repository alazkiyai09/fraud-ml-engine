"""Lazy LOF loader."""


def load_lof():
    from src.models.anomaly.legacy.models.lof import LOFDetector

    return LOFDetector
