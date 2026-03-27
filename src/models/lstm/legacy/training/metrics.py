"""
Evaluation metrics for fraud detection.

Focuses on precision-recall metrics for imbalanced classification.
"""

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities
        threshold: Decision threshold for binary predictions

    Returns:
        Dictionary of metric names and values
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Primary metric: Precision-Recall AUC (better for imbalanced data)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auc_pr = auc(recall, precision)

    # Additional metrics
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    accuracy = accuracy_score(y_true, y_pred)
    precision_val = precision_score(y_true, y_pred, zero_division=0)
    recall_val = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return {
        'auc_pr': float(auc_pr),
        'auc_roc': float(auc_roc),
        'accuracy': float(accuracy),
        'precision': float(precision_val),
        'recall': float(recall_val),
        'f1': float(f1),
        'true_negatives': int(cm[0, 0]),
        'false_positives': int(cm[0, 1]),
        'false_negatives': int(cm[1, 0]),
        'true_positives': int(cm[1, 1])
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics for display.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Formatted string
    """
    lines = [
        f"AUC-PR: {metrics['auc_pr']:.4f}",
        f"AUC-ROC: {metrics['auc_roc']:.4f}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"F1: {metrics['f1']:.4f}",
        "",
        "Confusion Matrix:",
        f"  TN={metrics['true_negatives']}, FP={metrics['false_positives']}",
        f"  FN={metrics['false_negatives']}, TP={metrics['true_positives']}"
    ]
    return "\n".join(lines)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = "f1"
) -> float:
    """
    Find optimal threshold based on specified metric.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ("f1", "precision", "recall", "accuracy")

    Returns:
        Optimal threshold value
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0.5
    best_score = -1

    metric_fn = {
        'f1': lambda y_t, y_p: f1_score(y_t, y_p, zero_division=0),
        'precision': lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0),
        'recall': lambda y_t, y_p: recall_score(y_t, y_p, zero_division=0),
        'accuracy': accuracy_score
    }[metric]

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        score = metric_fn(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold


class MetricTracker:
    """
    Track metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.train_metrics = []
        self.val_metrics = []
        self.best_metric = -np.inf
        self.best_epoch = 0
        self.patience_counter = 0

    def update(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        monitor: str = "auc_pr"
    ):
        """
        Update tracked metrics.

        Args:
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            monitor: Metric to monitor for best model
        """
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)

        # Check if this is the best model
        current_metric = val_metrics[monitor]
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.best_epoch = len(self.val_metrics) - 1
            self.patience_counter = 0
            return True  # New best model
        else:
            self.patience_counter += 1
            return False  # Not best model

    def get_history(self) -> Dict[str, List]:
        """
        Get metric history.

        Returns:
            Dictionary with training history
        """
        history = {
            'train': {},
            'val': {}
        }

        if self.train_metrics:
            for key in self.train_metrics[0].keys():
                history['train'][key] = [m[key] for m in self.train_metrics]

        if self.val_metrics:
            for key in self.val_metrics[0].keys():
                history['val'][key] = [m[key] for m in self.val_metrics]

        return history

    def should_stop_early(self, patience: int) -> bool:
        """
        Check if training should stop early.

        Args:
            patience: Number of epochs to wait for improvement

        Returns:
            True if should stop, False otherwise
        """
        return self.patience_counter >= patience
