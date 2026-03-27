"""Explainability wrappers."""

__all__ = ["load_shap_explainer", "load_lime_explainer"]


def load_shap_explainer():
    from src.explainability.legacy.explainers.shap_explainer import SHAPExplainer

    return SHAPExplainer


def load_lime_explainer():
    from src.explainability.legacy.explainers.lime_explainer import LIMEExplainer

    return LIMEExplainer
