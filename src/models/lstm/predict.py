"""Lazy predictor loader."""


def load_predictor():
    from src.models.lstm.legacy.inference import FraudPredictor

    return FraudPredictor
