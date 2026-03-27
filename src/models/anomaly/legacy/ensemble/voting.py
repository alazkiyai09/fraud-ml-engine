"""Voting ensemble for anomaly detection."""

from typing import List
import numpy as np


def voting_ensemble(
    scores_list: List[np.ndarray],  # type: ignore[valid-type]
    method: str = "average",
    weights: List[float] = None
) -> np.ndarray:  # type: ignore[valid-type]
    """
    Combine anomaly scores from multiple models using voting.

    Args:
        scores_list: List of anomaly score arrays, each of shape (n_samples,).
        method: Combination method ('average' or 'majority').
                - 'average': Weighted average of scores
                - 'majority': Binary voting (requires thresholding first)
        weights: Optional weights for each model (for 'average' method).
                 If None, equal weights are used.

    Returns:
        Combined anomaly scores of shape (n_samples,).

    Raises:
        ValueError: If method is invalid or dimensions don't match.
    """
    n_models = len(scores_list)

    if n_models == 0:
        raise ValueError("scores_list must not be empty.")

    # Check all arrays have same shape
    shape = scores_list[0].shape
    for i, scores in enumerate(scores_list):
        if scores.shape != shape:
            raise ValueError(f"All score arrays must have same shape. "
                           f"Array 0 has shape {shape}, array {i} has shape {scores.shape}.")

    if method == "average":
        if weights is None:
            # Equal weights
            combined = np.mean(scores_list, axis=0)
        else:
            # Weighted average
            if len(weights) != n_models:
                raise ValueError(f"Number of weights ({len(weights)}) must match "
                               f"number of models ({n_models}).")
            weights = np.array(weights) / np.sum(weights)  # Normalize
            combined = np.average(scores_list, axis=0, weights=weights)

    elif method == "majority":
        # Binary voting: each model votes anomaly if score > its threshold
        # This requires binary predictions, not raw scores
        # For simplicity, we'll normalize scores to [0,1] and use 0.5 as threshold
        binary_votes = []
        for scores in scores_list:
            # Normalize scores to [0,1]
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            binary_votes.append((scores_norm >= 0.5).astype(int))

        combined = np.mean(binary_votes, axis=0)

    else:
        raise ValueError(f"Invalid method: {method}. Must be 'average' or 'majority'.")

    return combined


def voting_ensemble_binary(
    predictions_list: List[np.ndarray],  # type: ignore[valid-type]
    weights: List[float] = None
) -> np.ndarray:  # type: ignore[valid-type]
    """
    Combine binary predictions from multiple models using weighted majority voting.

    Args:
        predictions_list: List of binary prediction arrays (0=normal, 1=anomaly).
        weights: Optional weights for each model.

    Returns:
        Combined binary predictions.
    """
    n_models = len(predictions_list)

    if n_models == 0:
        raise ValueError("predictions_list must not be empty.")

    # Check all arrays have same shape
    shape = predictions_list[0].shape
    for i, preds in enumerate(predictions_list):
        if preds.shape != shape:
            raise ValueError(f"All prediction arrays must have same shape.")

    if weights is None:
        # Equal weights, majority voting
        combined = (np.sum(predictions_list, axis=0) >= n_models / 2).astype(int)
    else:
        # Weighted voting
        if len(weights) != n_models:
            raise ValueError(f"Number of weights must match number of models.")
        weights = np.array(weights)
        combined = (np.average(predictions_list, axis=0, weights=weights) >= 0.5).astype(int)

    return combined
