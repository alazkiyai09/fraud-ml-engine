"""Compatibility surface for velocity feature generation."""

from __future__ import annotations

from src.features.transformers.velocity_features import VelocityFeatures

__all__ = ["VelocityFeatures", "build_velocity_features"]


def build_velocity_features(frame, **kwargs):
    """Fit + transform helper used by scripts and notebooks."""
    transformer = VelocityFeatures(**kwargs)
    transformer.fit(frame)
    return transformer.transform(frame)
