"""Lazy sklearn baseline loaders."""


def load_sklearn_baselines():
    from src.models.classical.legacy.models.baseline import (
        LogisticRegressionBaseline,
        RandomForestBaseline,
    )

    return {
        "logistic_regression": LogisticRegressionBaseline,
        "random_forest": RandomForestBaseline,
    }
