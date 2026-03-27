"""Lazy XGBoost wrapper loader."""


def load_xgboost_wrapper():
    from src.models.classical.legacy.models.xgboost_wrapper import XGBoostWrapper

    return XGBoostWrapper
