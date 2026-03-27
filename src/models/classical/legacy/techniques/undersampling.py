"""
Random undersampling technique for imbalanced classification.

Reduces the majority class to balance the class distribution.
"""

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from typing import Tuple
from src.models.classical.legacy.config import config


def apply_random_undersampling(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: float = None,
    random_state: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random undersampling to the majority class.

    Args:
        X: Feature matrix
        y: Target labels
        sampling_strategy: Target ratio of minority to majority samples
                          (None means 1:1 balanced)
        random_state: Random seed for reproducibility

    Returns:
        Resampled feature matrix and labels
    """
    if random_state is None:
        random_state = config.RANDOM_STATE

    if sampling_strategy is None:
        sampling_strategy = config.SAMPLING_STRATEGY

    sampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    return X_resampled, y_resampled
