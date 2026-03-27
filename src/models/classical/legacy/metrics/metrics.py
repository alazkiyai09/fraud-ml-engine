"""
Metric calculations for imbalanced classification evaluation.

Includes standard metrics (accuracy, precision, recall, F1) as well as
metrics specifically designed for imbalanced data (AUPRC, AUROC, Recall@FPR).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    roc_curve,
)
from typing import Dict


def calculate_auprc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calculate Area Under the Precision-Recall Curve.

    AUPRC is more informative than AUROC for imbalanced datasets.

    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for the positive class

    Returns:
        AUPRC score as a float
    """
    return average_precision_score(y_true, y_proba)


def calculate_auroc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calculate Area Under the ROC Curve.

    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for the positive class

    Returns:
        AUROC score as a float
    """
    return roc_auc_score(y_true, y_proba)


def calculate_recall_at_fpr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fpr_threshold: float = 0.01
) -> float:
    """
    Calculate Recall at a specific False Positive Rate threshold.

    This metric is crucial for fraud detection where we want to minimize
    false positives while maximizing fraud detection.

    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for the positive class
        fpr_threshold: Maximum acceptable FPR (default: 0.01 = 1%)

    Returns:
        Recall score at the specified FPR threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    # Find the index where FPR is just below the threshold
    idx = np.where(fpr <= fpr_threshold)[0]

    if len(idx) == 0:
        # If no threshold meets the FPR requirement, return 0
        return 0.0

    # Return the highest TPR (recall) at or below the FPR threshold
    return float(tpr[idx[-1]])


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    fpr_threshold: float = 0.01
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for imbalanced classification.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
        y_proba: Predicted probabilities for the positive class
        fpr_threshold: FPR threshold for Recall@FPR calculation

    Returns:
        Dictionary containing all metric scores
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auprc": calculate_auprc(y_true, y_proba),
        "auroc": calculate_auroc(y_true, y_proba),
        f"recall_at_fpr_{int(fpr_threshold * 100)}pct": calculate_recall_at_fpr(
            y_true, y_proba, fpr_threshold
        ),
    }

    return metrics
