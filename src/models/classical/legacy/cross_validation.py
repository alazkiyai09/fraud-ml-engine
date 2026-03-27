"""
Stratified K-Fold cross-validation for imbalanced classification.

Maintains class distribution across folds for reliable evaluation
of imbalanced classification techniques.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Any, Dict, List
from src.models.classical.legacy.config import config
from src.models.classical.legacy.metrics.metrics import compute_all_metrics


def stratified_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Any,
    technique: str,
    apply_resampling: callable = None,
) -> Dict[str, List[float]]:
    """
    Perform stratified K-fold cross-validation with specified technique.

    Args:
        X: Feature matrix
        y: Target labels
        estimator: Classifier instance with fit/predict/predict_proba methods
        technique: Name of the technique for logging
        apply_resampling: Optional function to apply resampling (X, y) -> (X, y)

    Returns:
        Dictionary with lists of metric scores across folds
    """
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    # Store metrics for each fold
    fold_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auprc": [],
        "auroc": [],
        f"recall_at_fpr_{int(config.FPR_THRESHOLD * 100)}pct": [],
    }

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Apply resampling if technique requires it
        if apply_resampling is not None:
            X_train_resampled, y_train_resampled = apply_resampling(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        # Clone and fit estimator
        from sklearn.base import clone
        estimator_clone = clone(estimator)
        estimator_clone.fit(X_train_resampled, y_train_resampled)

        # Make predictions
        y_pred = estimator_clone.predict(X_test)
        y_proba = estimator_clone.predict_proba(X_test)

        # Compute metrics
        metrics = compute_all_metrics(
            y_test,
            y_pred,
            y_proba[:, 1],  # Probability of positive class
            fpr_threshold=config.FPR_THRESHOLD,
        )

        # Store metrics
        for metric_name, value in metrics.items():
            fold_metrics[metric_name].append(value)

    return fold_metrics
