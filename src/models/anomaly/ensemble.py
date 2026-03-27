"""Anomaly model metadata."""


def describe_anomaly_stack() -> dict:
    return {
        "source": "src/models/anomaly/legacy",
        "models": ["isolation_forest", "lof", "one_class_svm", "autoencoder"],
    }
