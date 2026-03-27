"""Model loading service with lazy initialization."""

import logging
from pathlib import Path

import joblib
import numpy as np

from src.api.legacy_app.core.config import settings


logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Exception raised when model fails to load."""

    pass


class ModelLoader:
    """
    Service for loading and managing model artifacts.

    Provides lazy loading of model and pipeline with health check capabilities.
    """

    def __init__(
        self,
        model_path: str | None = None,
        pipeline_path: str | None = None,
    ):
        """
        Initialize model loader.

        Args:
            model_path: Path to trained model file
            pipeline_path: Path to fitted pipeline file
        """
        self.model_path = model_path or settings.model_path
        self.pipeline_path = pipeline_path or settings.pipeline_path

        self._model: Any = None
        self._pipeline: Any = None
        self._model_loaded = False

    def load_model(self) -> None:
        """
        Load model and pipeline artifacts from disk.

        Raises:
            ModelLoadError: If model or pipeline files are not found or fail to load
        """
        if self._model_loaded:
            logger.debug("Model already loaded, skipping")
            return

        try:
            # Check if files exist
            model_file = Path(self.model_path)
            pipeline_file = Path(self.pipeline_path)

            if not model_file.exists():
                raise ModelLoadError(f"Model file not found: {self.model_path}")

            if not pipeline_file.exists():
                raise ModelLoadError(
                    f"Pipeline file not found: {self.pipeline_path}"
                )

            logger.info(f"Loading model from {self.model_path}")
            self._model = joblib.load(self.model_path)

            logger.info(f"Loading pipeline from {self.pipeline_path}")
            self._pipeline = joblib.load(self.pipeline_path)

            self._model_loaded = True
            logger.info("Model and pipeline loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelLoadError(f"Model loading failed: {str(e)}") from e

    def get_model(self) -> Any:
        """
        Get loaded model, loading if necessary.

        Returns:
            Loaded model object

        Raises:
            ModelLoadError: If model fails to load
        """
        if not self._model_loaded:
            self.load_model()

        return self._model

    def get_pipeline(self) -> Any:
        """
        Get loaded pipeline, loading if necessary.

        Returns:
            Loaded pipeline object

        Raises:
            ModelLoadError: If pipeline fails to load
        """
        if not self._model_loaded:
            self.load_model()

        return self._pipeline

    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        return self._model_loaded

    def unload_model(self) -> None:
        """Unload model and pipeline from memory."""
        self._model = None
        self._pipeline = None
        self._model_loaded = False
        logger.info("Model unloaded from memory")

    def reload_model(self) -> None:
        """
        Reload model and pipeline from disk.

        Useful for hot-reloading updated models.
        """
        self.unload_model()
        self.load_model()
        logger.info("Model reloaded successfully")

    def get_model_info(self) -> dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Dictionary with model information
        """
        if not self._model_loaded:
            raise ModelLoadError("Model not loaded. Call load_model() first.")

        model = self._model
        pipeline = self._pipeline

        # Extract model info
        info = {
            "model_type": type(model).__name__,
            "model_path": str(self.model_path),
            "pipeline_path": str(self.pipeline_path),
        }

        # Try to get feature names from pipeline
        try:
            if hasattr(pipeline, "get_feature_names_out"):
                info["features"] = pipeline.get_feature_names_out().tolist()
            elif hasattr(pipeline, "feature_names_in_"):
                info["features"] = pipeline.feature_names_in_.tolist()
            else:
                info["features"] = []
        except Exception as e:
            logger.warning(f"Could not extract feature names: {e}")
            info["features"] = []

        # Try to get model parameters
        try:
            if hasattr(model, "get_params"):
                info["params"] = model.get_params()
        except Exception as e:
            logger.warning(f"Could not extract model parameters: {e}")

        return info
