"""Fraud Model Explainability Toolkit."""

__version__ = "0.1.0"
__author__ = "Fraud Detection Team"

from .explainers import BaseExplainer, SHAPExplainer, LIMEExplainer, PDPExplainer
from .api import ExplainerFactory, create_explainer
from .reports import ReportGenerator
from .models import ModelLoader, load_model
from .utils import (
    validate_consistency,
    validate_explanation_quality,
    format_risk_factors,
    format_importance_scores
)

__all__ = [
    # Explainers
    "BaseExplainer",
    "SHAPExplainer",
    "LIMEExplainer",
    "PDPExplainer",
    # Factory
    "ExplainerFactory",
    "create_explainer",
    # Reports
    "ReportGenerator",
    # Models
    "ModelLoader",
    "load_model",
    # Utils
    "validate_consistency",
    "validate_explanation_quality",
    "format_risk_factors",
    "format_importance_scores",
]
