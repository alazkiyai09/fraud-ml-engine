"""Lazy isolation forest loader."""


def load_isolation_forest():
    from src.models.anomaly.legacy.models.isolation_forest import IsolationForestDetector

    return IsolationForestDetector
