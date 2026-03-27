"""Public benchmark metadata for classical models."""

from __future__ import annotations


def describe_classical_stack() -> dict:
    return {
        "models": [
            "logistic_regression",
            "random_forest",
            "xgboost",
            "focal_loss_mlp",
            "lightgbm_placeholder",
        ],
        "imbalance_strategies": ["smote", "adasyn", "undersampling", "class_weight"],
        "source": "src/models/classical/legacy",
    }
