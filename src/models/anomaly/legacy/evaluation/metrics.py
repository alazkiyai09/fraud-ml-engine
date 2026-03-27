"""Evaluation metrics for anomaly detection."""

from typing import Dict, Tuple, List
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
from pathlib import Path


def compute_detection_metrics(
    y_true: np.ndarray,  # type: ignore[valid-type]
    anomaly_scores: np.ndarray,  # type: ignore[valid-type]
    threshold: float
) -> Dict[str, float]:
    """
    Compute detection metrics at a given threshold.

    Args:
        y_true: True binary labels (0=normal, 1=anomaly).
        anomaly_scores: Anomaly scores (higher = more anomalous).
        threshold: Decision threshold.

    Returns:
        Dictionary containing:
            - detection_rate (True Positive Rate / Recall)
            - false_positive_rate
            - precision
            - recall
            - f1
            - true_positives
            - false_positives
            - true_negatives
            - false_negatives
    """
    y_pred = (anomaly_scores >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Calculate metrics
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = detection_rate
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }


def optimize_threshold(
    y_val: np.ndarray,  # type: ignore[valid-type]
    scores_val: np.ndarray,  # type: ignore[valid-type]
    target_fpr: float = 0.01
) -> float:
    """
    Find threshold that achieves target FPR on validation set.

    Args:
        y_val: Validation labels.
        scores_val: Validation anomaly scores.
        target_fpr: Target false positive rate (default: 0.01 = 1%).

    Returns:
        Optimal threshold value.
    """
    # Use only class 0 samples for threshold calculation
    normal_scores = scores_val[y_val == 0]

    if len(normal_scores) == 0:
        raise ValueError("No normal samples in validation set.")

    # Find threshold that achieves target FPR
    threshold = np.quantile(normal_scores, 1 - target_fpr)

    return threshold


def optimize_threshold_f1(
    y_val: np.ndarray,  # type: ignore[valid-type]
    scores_val: np.ndarray  # type: ignore[valid-type]
) -> Tuple[float, float]:
    """
    Find threshold that maximizes F1 score on validation set.

    Args:
        y_val: Validation labels.
        scores_val: Validation anomaly scores.

    Returns:
        Tuple of (optimal_threshold, best_f1_score).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val, scores_val)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_idx]

    return best_threshold, best_f1


def roc_curve_data(
    y_true: np.ndarray,  # type: ignore[valid-type]
    scores: np.ndarray  # type: ignore[valid-type]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve data.

    Args:
        y_true: True binary labels.
        scores: Anomaly scores.

    Returns:
        Tuple of (fpr, tpr, auc_score).
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    auc_score = auc(fpr, tpr)

    return fpr, tpr, auc_score


def pr_curve_data(
    y_true: np.ndarray,  # type: ignore[valid-type]
    scores: np.ndarray  # type: ignore[valid-type]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Precision-Recall curve data.

    Args:
        y_true: True binary labels.
        scores: Anomaly scores.

    Returns:
        Tuple of (precision, recall, auc_score).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores, pos_label=1)
    auc_score = auc(recall, precision)

    return precision, recall, auc_score


def plot_roc_curve(
    y_true: np.ndarray,  # type: ignore[valid-type]
    scores_dict: Dict[str, np.ndarray],  # type: ignore[valid-type]
    save_path: str = None
) -> None:
    """
    Plot ROC curves for multiple models.

    Args:
        y_true: True binary labels.
        scores_dict: Dictionary mapping model names to anomaly scores.
        save_path: Path to save plot.
    """
    plt.figure(figsize=(10, 8))

    for model_name, scores in scores_dict.items():
        fpr, tpr, auc_score = roc_curve_data(y_true, scores)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Detection Rate)')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")

    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,  # type: ignore[valid-type]
    scores_dict: Dict[str, np.ndarray],  # type: ignore[valid-type]
    save_path: str = None
) -> None:
    """
    Plot Precision-Recall curves for multiple models.

    Args:
        y_true: True binary labels.
        scores_dict: Dictionary mapping model names to anomaly scores.
        save_path: Path to save plot.
    """
    plt.figure(figsize=(10, 8))

    for model_name, scores in scores_dict.items():
        precision, recall, auc_score = pr_curve_data(y_true, scores)
        plt.plot(recall, precision, label=f'{model_name} (AUC = {auc_score:.3f})')

    plt.xlabel('Recall (Detection Rate)')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to {save_path}")

    plt.close()


def compute_all_metrics(
    y_true: np.ndarray,  # type: ignore[valid-type]
    scores: np.ndarray,  # type: ignore[valid-type]
    threshold: float
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for a model.

    Args:
        y_true: True binary labels.
        scores: Anomaly scores.
        threshold: Decision threshold.

    Returns:
        Dictionary with all metrics.
    """
    # Detection metrics at threshold
    detection_metrics = compute_detection_metrics(y_true, scores, threshold)

    # ROC AUC
    _, _, auc_roc = roc_curve_data(y_true, scores)

    # PR AUC
    _, _, auc_pr = pr_curve_data(y_true, scores)

    return {
        **detection_metrics,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr
    }
