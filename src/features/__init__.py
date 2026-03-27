"""
Fraud Feature Engineering Pipeline

A sklearn-compatible feature engineering pipeline for fraud detection.
Includes velocity features, deviation features, and merchant risk features.
"""

__version__ = "0.1.0"

from src.features.transformers.velocity_features import VelocityFeatures
from src.features.transformers.deviation_features import DeviationFeatures
from src.features.transformers.merchant_features import MerchantRiskFeatures
from src.features.feature_selection.shap_selector import SHAPSelector

__all__ = [
    "VelocityFeatures",
    "DeviationFeatures",
    "MerchantRiskFeatures",
    "SHAPSelector",
]
