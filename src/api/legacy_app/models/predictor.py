"""Fraud prediction model wrapper."""

import logging
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.api.legacy_app.models.schemas import TransactionRequest
from src.api.legacy_app.services.model_loader import ModelLoader, ModelLoadError
from src.api.legacy_app.utils.helpers import classify_risk_tier, compute_latency_ms, get_top_risk_factors


logger = logging.getLogger(__name__)


class FraudPredictor:
    """
    Wrapper for fraud detection model and feature pipeline.

    Combines the FraudFeaturePipeline for feature extraction with
    an XGBoost model for fraud probability prediction.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_path: str,
        model_version: str = "1.0.0",
    ) -> None:
        """
        Initialize fraud predictor.

        Args:
            model_path: Path to trained model file
            pipeline_path: Path to fitted pipeline file
            model_version: Model version string
        """
        self.loader = ModelLoader(model_path, pipeline_path)
        self.model_version = model_version
        self._model: Any = None
        self._pipeline: Any = None

    def load_model(self) -> None:
        """
        Load model and pipeline artifacts from disk.

        Raises:
            ModelLoadError: If model or pipeline fails to load
        """
        self.loader.load_model()
        self._model = self.loader.get_model()
        self._pipeline = self.loader.get_pipeline()
        logger.info("FraudPredictor model loaded successfully")

    def _ensure_loaded(self) -> None:
        """Ensure model and pipeline are loaded."""
        if self._model is None or self._pipeline is None:
            self.load_model()

    def predict_single(self, transaction: TransactionRequest) -> dict[str, Any]:
        """
        Predict fraud probability for a single transaction.

        Args:
            transaction: Transaction request with all required fields

        Returns:
            Dictionary with:
                - fraud_probability: float (0-1)
                - risk_tier: str (LOW, MEDIUM, HIGH, CRITICAL)
                - risk_factors: list[str]
                - latency_ms: float

        Raises:
            ModelLoadError: If model fails to load
            ValueError: If prediction fails
        """
        start_time_ns = time.time_ns()
        self._ensure_loaded()

        try:
            # Convert transaction to DataFrame
            transaction_df = pd.DataFrame([{
                "transaction_id": transaction.transaction_id,
                "user_id": transaction.user_id,
                "merchant_id": transaction.merchant_id,
                "amount": transaction.amount,
                "timestamp": transaction.timestamp,
            }])

            # Extract features using pipeline
            features = self._pipeline.transform(transaction_df)

            # Get prediction probability
            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba(features)[:, 1][0]
            else:
                # Fallback for models without predict_proba
                proba = float(self._model.predict(features)[0])

            # Classify risk tier
            risk_tier = classify_risk_tier(proba)

            # Get top risk factors (use feature values as proxy)
            feature_names = self._pipeline.get_feature_names_out()
            risk_factors = get_top_risk_factors(
                feature_names, features[0], top_n=3
            )

            # Compute latency
            end_time_ns = time.time_ns()
            latency_ms = compute_latency_ms(start_time_ns, end_time_ns)

            return {
                "fraud_probability": float(proba),
                "risk_tier": risk_tier,
                "risk_factors": risk_factors,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {str(e)}") from e

    def predict_batch(
        self, transactions: list[TransactionRequest]
    ) -> list[dict[str, Any]]:
        """
        Predict fraud probabilities for multiple transactions.

        Args:
            transactions: List of transaction requests

        Returns:
            List of prediction dictionaries with same structure as predict_single

        Raises:
            ModelLoadError: If model fails to load
            ValueError: If prediction fails
        """
        start_time_ns = time.time_ns()
        self._ensure_loaded()

        try:
            # Convert transactions to DataFrame
            transactions_df = pd.DataFrame([
                {
                    "transaction_id": t.transaction_id,
                    "user_id": t.user_id,
                    "merchant_id": t.merchant_id,
                    "amount": t.amount,
                    "timestamp": t.timestamp,
                }
                for t in transactions
            ])

            # Extract features using pipeline
            features = self._pipeline.transform(transactions_df)

            # Get prediction probabilities
            if hasattr(self._model, "predict_proba"):
                probabilities = self._model.predict_proba(features)[:, 1]
            else:
                probabilities = self._model.predict(features).astype(float)

            # Get feature names
            feature_names = self._pipeline.get_feature_names_out()

            # Build prediction results
            predictions = []
            for i, (transaction, proba) in enumerate(zip(transactions, probabilities)):
                # Classify risk tier
                risk_tier = classify_risk_tier(float(proba))

                # Get top risk factors
                risk_factors = get_top_risk_factors(
                    feature_names, features[i], top_n=3
                )

                predictions.append({
                    "fraud_probability": float(proba),
                    "risk_tier": risk_tier,
                    "risk_factors": risk_factors,
                    "latency_ms": 0.0,  # Will be updated with total time
                })

            # Distribute total latency across predictions
            end_time_ns = time.time_ns()
            total_latency_ms = compute_latency_ms(start_time_ns, end_time_ns)
            avg_latency_ms = total_latency_ms / len(transactions)

            for pred in predictions:
                pred["latency_ms"] = avg_latency_ms

            return predictions

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise ValueError(f"Batch prediction failed: {str(e)}") from e

    def get_model_info(self) -> dict[str, Any]:
        """
        Return model metadata.

        Returns:
            Dictionary with:
                - model_version: str
                - model_type: str
                - features: list[str]
                - metrics: dict[str, float]
                - last_updated: datetime

        Raises:
            ModelLoadError: If model is not loaded
        """
        self._ensure_loaded()

        # Get info from loader
        info = self.loader.get_model_info()

        # Add additional metadata
        info["model_version"] = self.model_version
        info["last_updated"] = datetime.utcnow().isoformat()

        # Add placeholder metrics (in production, load from training metadata)
        info["metrics"] = {
            "precision": 0.85,
            "recall": 0.78,
            "f1_score": 0.81,
            "auc_roc": 0.92,
        }

        return info

    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        return self.loader.is_model_loaded()
