"""
Inference interface for fraud detection models.

Provides easy-to-use interface for making predictions with trained models.
"""

from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class FraudPredictor:
    """
    High-level interface for fraud prediction.

    Supports both LSTM (sequential) and baseline (single-transaction) models.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "lstm",
        device: Optional[str] = None
    ):
        """
        Load trained model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            model_type: "lstm" or "baseline"
            device: Device for inference ("cuda", "cpu", or None for auto)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.model_type = model_type

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']

        # Reconstruct model
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"✓ Model loaded from {checkpoint_path}")
        print(f"  Type: {model_type}")
        print(f"  Device: {self.device}")

    def _build_model(self) -> nn.Module:
        """Reconstruct model architecture from config."""
        if self.model_type == "lstm":
            from src.models.lstm.legacy.models.lstm_attention import LSTMAttentionClassifier

            return LSTMAttentionClassifier(
                input_dim=self.config['model']['input_dim'],
                hidden_dim=self.config['model']['hidden_dim'],
                num_layers=self.config['model']['num_layers'],
                num_heads=self.config['model']['num_heads'],
                dropout=self.config['model']['dropout'],
                bidirectional=self.config['model']['bidirectional']
            )
        else:
            from src.models.lstm.legacy.models.baseline import BaselineMLP

            return BaselineMLP(
                input_dim=self.config['model']['input_dim'],
                hidden_dims=self.config['baseline']['hidden_dims'],
                dropout=self.config['baseline']['dropout']
            )

    def predict(
        self,
        features: Union[np.ndarray, torch.Tensor, List],
        lengths: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
        threshold: float = 0.5,
        return_proba: bool = True
    ) -> Dict:
        """
        Make fraud predictions.

        Args:
            features: Transaction features
                - For LSTM: (num_sequences, max_seq_len, num_features)
                - For baseline: (num_samples, num_features) or (num_samples, 1, num_features)
            lengths: Actual sequence lengths (required for LSTM)
                Shape: (num_samples,)
            threshold: Decision threshold for binary prediction
            return_proba: Whether to return probabilities

        Returns:
            Dictionary with:
                - predictions: Binary predictions (0 or 1)
                - probabilities: Fraud probabilities (if return_proba=True)
                - attention: Attention weights (LSTM only, if available)
        """
        # Convert to tensor
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)

        if self.model_type == "lstm":
            if lengths is None:
                raise ValueError("lengths required for LSTM model")

            if not isinstance(lengths, torch.Tensor):
                lengths = torch.tensor(lengths, dtype=torch.long)

            # Add batch dimension if needed
            if features.dim() == 2:
                features = features.unsqueeze(0)
            if lengths.dim() == 0:
                lengths = lengths.unsqueeze(0)

            # Move to device
            features = features.to(self.device)
            lengths = lengths.to(self.device)

            # Predict
            with torch.no_grad():
                probabilities, attention = self.model(features, lengths, return_attention=True)

        else:  # baseline
            # Add batch dimension if needed
            if features.dim() == 1:
                features = features.unsqueeze(0)
            elif features.dim() == 3:
                # Take last transaction if sequence input
                features = features[:, -1, :]

            # Move to device
            features = features.to(self.device)

            # Predict
            with torch.no_grad():
                probabilities = self.model(features)
                attention = None

        # Convert to numpy
        probabilities = probabilities.squeeze(-1).cpu().numpy()
        predictions = (probabilities >= threshold).astype(int)

        result = {
            'predictions': predictions
        }

        if return_proba:
            result['probabilities'] = probabilities

        if attention is not None:
            result['attention'] = attention.cpu().numpy()

        return result

    def predict_user(
        self,
        user_history: pd.DataFrame,
        feature_columns: List[str],
        time_column: str = "transaction_time",
        max_sequence_length: Optional[int] = None
    ) -> Dict:
        """
        Predict fraud for a user's transaction history.

        Args:
            user_history: DataFrame with user's transactions
            feature_columns: List of feature column names
            time_column: Timestamp column name
            max_sequence_length: Max transactions to consider (uses config if None)

        Returns:
            Prediction dictionary
        """
        if max_sequence_length is None:
            max_sequence_length = self.config['sequence']['max_sequence_length']

        # Sort by time
        user_history = user_history.sort_values(time_column)

        # Extract features
        features = user_history[feature_columns].values

        # Take last N transactions
        if len(features) > max_sequence_length:
            features = features[-max_sequence_length:]

        # Create sequence (1, seq_len, num_features)
        num_features = len(feature_columns)
        sequence = np.zeros((1, max_sequence_length, num_features))
        actual_length = len(features)
        sequence[0, -actual_length:, :] = features

        # Predict
        result = self.predict(
            features=sequence,
            lengths=[actual_length]
        )

        # Add context
        result['num_transactions'] = actual_length

        return result

    def explain_prediction(
        self,
        features: Union[np.ndarray, torch.Tensor],
        lengths: Union[np.ndarray, torch.Tensor, List],
        top_k: int = 3
    ) -> Dict:
        """
        Explain prediction using attention weights (LSTM only).

        Args:
            features: Transaction sequences (num_sequences, max_seq_len, num_features)
            lengths: Actual sequence lengths
            top_k: Number of top transactions to highlight

        Returns:
            Dictionary with explanation
        """
        if self.model_type != "lstm":
            return {"error": "Attention explanation only available for LSTM model"}

        result = self.predict(features, lengths, return_attention=True)

        if 'attention' not in result:
            return {"error": "Could not extract attention weights"}

        # Average attention across heads
        attention = result['attention'].squeeze(0)  # (num_heads, seq_len)
        avg_attention = attention.mean(axis=0)  # (seq_len,)

        # Get top-k attended transactions
        actual_length = lengths[0] if isinstance(lengths, (list, np.ndarray)) else lengths.item()
        valid_attention = avg_attention[-actual_length:]

        top_indices = np.argsort(valid_attention)[-top_k:][::-1]

        explanation = {
            'top_attention_indices': top_indices.tolist(),
            'top_attention_weights': valid_attention[top_indices].tolist(),
            'attention_weights': valid_attention.tolist(),
            'prediction': result['predictions'][0],
            'probability': result['probabilities'][0]
        }

        return explanation


class ONNXPredictor:
    """
    Predictor using ONNX Runtime for production deployment.

    Faster inference and no PyTorch dependency.
    """

    def __init__(
        self,
        onnx_path: str,
        model_type: str = "lstm"
    ):
        """
        Load ONNX model.

        Args:
            onnx_path: Path to ONNX model file
            model_type: "lstm" or "baseline"
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime not installed. Install with: pip install onnxruntime")

        self.onnx_path = Path(onnx_path)
        self.model_type = model_type

        # Create ONNX Runtime session
        self.session = ort.InferenceSession(str(self.onnx_path))

        print(f"✓ ONNX model loaded from {onnx_path}")

    def predict(
        self,
        features: np.ndarray,
        lengths: Optional[np.ndarray] = None,
        threshold: float = 0.5
    ) -> Dict:
        """
        Make predictions with ONNX model.

        Args:
            features: Transaction features
            lengths: Sequence lengths (required for LSTM)
            threshold: Decision threshold

        Returns:
            Prediction dictionary
        """
        if self.model_type == "lstm":
            if lengths is None:
                raise ValueError("lengths required for LSTM model")

            # Ensure correct shapes
            if features.ndim == 2:
                features = np.expand_dims(features, axis=0)

            # Run inference
            output = self.session.run(
                None,
                {
                    'padded_sequences': features.astype(np.float32),
                    'lengths': lengths.astype(np.int64)
                }
            )
        else:
            # Baseline model
            if features.ndim == 1:
                features = np.expand_dims(features, axis=0)

            output = self.session.run(
                None,
                {'input': features.astype(np.float32)}
            )

        probabilities = output[0].squeeze()
        predictions = (probabilities >= threshold).astype(int)

        return {
            'predictions': predictions if isinstance(predictions, np.ndarray) else np.array([predictions]),
            'probabilities': probabilities if isinstance(probabilities, np.ndarray) else np.array([probabilities])
        }


def load_predictor(
    checkpoint_path: str,
    use_onnx: bool = False,
    model_type: str = "lstm"
):
    """
    Factory function to load predictor.

    Args:
        checkpoint_path: Path to model file
        use_onnx: Use ONNX Runtime instead of PyTorch
        model_type: "lstm" or "baseline"

    Returns:
        Predictor instance
    """
    if use_onnx:
        return ONNXPredictor(checkpoint_path, model_type)
    else:
        return FraudPredictor(checkpoint_path, model_type)
