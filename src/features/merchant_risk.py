"""Compatibility surface for merchant risk feature generation."""

from __future__ import annotations

from src.features.transformers.merchant_features import MerchantRiskFeatures

__all__ = ["MerchantRiskFeatures", "build_merchant_risk_features"]


def build_merchant_risk_features(frame, labels, **kwargs):
    """Fit + transform helper for merchant-level Bayesian fraud risk features."""
    transformer = MerchantRiskFeatures(**kwargs)
    transformer.fit(frame, labels)
    return transformer.transform(frame)
