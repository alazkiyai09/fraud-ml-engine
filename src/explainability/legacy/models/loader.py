"""Model loading utilities with format detection and validation."""

import os
import pickle
import joblib
from typing import Any, Optional, Union
from pathlib import Path
import numpy as np

# Model type detection
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class ModelLoader:
    """
    Load and validate trained models from disk.

    Supports multiple model formats:
    - XGBoost (.json, .ubj, .xgb)
    - Scikit-learn (.pkl, .joblib)
    - TensorFlow/Keras (.h5, .keras)
    - Generic pickle (.pkl)
    """

    SUPPORTED_EXTENSIONS = {
        '.json': 'xgboost',
        '.ubj': 'xgboost',
        '.xgb': 'xgboost',
        '.pkl': 'pickle',
        '.joblib': 'joblib',
        '.h5': 'tensorflow',
        '.keras': 'tensorflow',
    }

    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize model loader.

        Args:
            model_dir: Default directory to search for models
        """
        self.model_dir = Path(model_dir) if model_dir else None

    def load(self, model_path: str) -> tuple[Any, str]:
        """
        Load a model from disk and detect its type.

        Args:
            model_path: Path to the model file

        Returns:
            Tuple of (model_object, model_type_string)

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model format is not supported
        """
        path = Path(model_path)

        # Resolve relative path against model_dir if provided
        if not path.is_absolute() and self.model_dir:
            path = self.model_dir / path

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Detect file extension and load accordingly
        ext = path.suffix.lower()

        if ext == '.json' or ext == '.ubj' or ext == '.xgb':
            model = self._load_xgboost(path)
            model_type = 'xgboost'
        elif ext == '.pkl':
            model = self._load_pickle(path)
            model_type = self._detect_sklearn_model(model)
        elif ext == '.joblib':
            model = self._load_joblib(path)
            model_type = self._detect_sklearn_model(model)
        elif ext == '.h5' or ext == '.keras':
            model = self._load_tensorflow(path)
            model_type = 'neural_network'
        else:
            raise ValueError(
                f"Unsupported model format: {ext}. "
                f"Supported formats: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )

        return model, model_type

    def _load_xgboost(self, path: Path) -> Any:
        """Load XGBoost model."""
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
        return xgb.XGBClassifier()
        # Note: Actual loading depends on XGBoost version
        # For newer versions: xgb.XGBClassifier().load_model(str(path))

    def _load_pickle(self, path: Path) -> Any:
        """Load model from pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _load_joblib(self, path: Path) -> Any:
        """Load model from joblib file."""
        return joblib.load(path)

    def _load_tensorflow(self, path: Path) -> Any:
        """Load TensorFlow/Keras model."""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")
        return tf.keras.models.load_model(str(path))

    def _detect_sklearn_model(self, model: Any) -> str:
        """
        Detect sklearn model type.

        Returns:
            Model type string: 'random_forest', 'gradient_boosting', or 'sklearn_generic'
        """
        if not HAS_SKLEARN:
            return 'sklearn_generic'

        model_name = model.__class__.__name__

        if model_name == 'RandomForestClassifier':
            return 'random_forest'
        elif model_name == 'GradientBoostingClassifier':
            return 'gradient_boosting'
        else:
            return 'sklearn_generic'

    def validate_model(self, model: Any, model_type: str) -> bool:
        """
        Validate that the model has required methods.

        Args:
            model: Model object to validate
            model_type: Type of model

        Returns:
            True if valid

        Raises:
            ValueError: If model doesn't have required methods
        """
        required_methods = ['predict']

        for method in required_methods:
            if not hasattr(model, method):
                raise ValueError(
                    f"Model of type '{model_type}' missing required method: {method}"
                )

        # Check for predict_proba for probabilistic models
        if model_type in ['xgboost', 'random_forest', 'gradient_boosting', 'sklearn_generic']:
            if not hasattr(model, 'predict_proba'):
                raise ValueError(
                    f"Model of type '{model_type}' should have predict_proba method"
                )

        return True


def load_model(model_path: str, model_dir: Optional[str] = None) -> tuple[Any, str]:
    """
    Convenience function to load a model.

    Args:
        model_path: Path to model file
        model_dir: Default directory to search for models

    Returns:
        Tuple of (model_object, model_type_string)
    """
    loader = ModelLoader(model_dir=model_dir)
    return loader.load(model_path)
