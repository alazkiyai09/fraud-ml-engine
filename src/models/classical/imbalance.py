"""Imbalance-handling helpers."""


def supported_strategies() -> list[str]:
    return ["smote", "adasyn", "undersampling", "class_weight", "focal_loss"]
