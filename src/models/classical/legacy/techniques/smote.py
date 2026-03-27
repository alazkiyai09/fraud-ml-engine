"""
SMOTE (Synthetic Minority Over-sampling Technique).

Generates synthetic samples for the minority class by interpolating
between nearest neighbors.
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from typing import Tuple
from src.models.classical.legacy.config import config


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: float = None,
    random_state: int = None,
    k_neighbors: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling to the minority class.

    Args:
        X: Feature matrix
        y: Target labels
        sampling_strategy: Target ratio of minority to majority samples
                          (None means 1:1 balanced)
        random_state: Random seed for reproducibility
        k_neighbors: Number of nearest neighbors for SMOTE

    Returns:
        Resampled feature matrix and labels
    """
    if random_state is None:
        random_state = config.RANDOM_STATE

    if sampling_strategy is None:
        sampling_strategy = config.SAMPLING_STRATEGY

    sampler = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors,
    )

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    return X_resampled, y_resampled
