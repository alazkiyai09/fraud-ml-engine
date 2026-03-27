"""
Central configuration for Imbalanced Classification Benchmark.
All random states and hyperparameters are defined here for reproducibility.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Any


@dataclass
class Config:
    """Configuration object for the benchmark project."""

    # Reproducibility
    RANDOM_STATE: int = 42

    # Cross-validation
    N_FOLDS: int = 5
    TEST_SIZE: float = 0.2

    # Focal Loss parameters
    FOCAL_ALPHA: float = 0.25
    FOCAL_GAMMA: float = 2.0

    # Class weights for scikit-learn models
    CLASS_WEIGHT: str = 'balanced'

    # Resampling parameters
    SAMPLING_STRATEGY: float = 0.1  # Target minority ratio after resampling

    # Model hyperparameters
    LOGISTIC_REGRESSION_PARAMS: dict[str, Any] = None
    RANDOM_FOREST_PARAMS: dict[str, Any] = None
    XGBOOST_PARAMS: dict[str, Any] = None

    # Metrics
    FPR_THRESHOLD: float = 0.01  # For Recall@FPR metric

    # Visualization
    FIGURE_DPI: int = 300
    FIGURE_SIZE: tuple[int, int] = (10, 6)

    # Paths
    PROJECT_ROOT: Path = None
    DATA_DIR: Path = None
    RESULTS_DIR: Path = None

    def __post_init__(self):
        """Set up paths and default parameters."""
        if self.PROJECT_ROOT is None:
            self.PROJECT_ROOT = Path(__file__).parent.parent
        if self.DATA_DIR is None:
            self.DATA_DIR = self.PROJECT_ROOT / "data"
        if self.RESULTS_DIR is None:
            self.RESULTS_DIR = self.PROJECT_ROOT / "results"
            self.RESULTS_DIR.mkdir(exist_ok=True)

        # Default model parameters with random_state
        if self.LOGISTIC_REGRESSION_PARAMS is None:
            self.LOGISTIC_REGRESSION_PARAMS = {
                'random_state': self.RANDOM_STATE,
                'max_iter': 1000,
                'solver': 'lbfgs',
            }

        if self.RANDOM_FOREST_PARAMS is None:
            self.RANDOM_FOREST_PARAMS = {
                'random_state': self.RANDOM_STATE,
                'n_estimators': 100,
                'max_depth': 10,
            }

        if self.XGBOOST_PARAMS is None:
            self.XGBOOST_PARAMS = {
                'random_state': self.RANDOM_STATE,
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'eval_metric': 'logloss',
            }


# Global configuration instance
config = Config()
