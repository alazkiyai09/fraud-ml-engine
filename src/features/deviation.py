"""Compatibility surface for deviation feature generation."""

from __future__ import annotations

from src.features.transformers.deviation_features import DeviationFeatures

__all__ = ["DeviationFeatures", "build_deviation_features"]


def build_deviation_features(frame, **kwargs):
    """Fit + transform helper used by scripts and notebooks."""
    transformer = DeviationFeatures(**kwargs)
    transformer.fit(frame)
    return transformer.transform(frame)
