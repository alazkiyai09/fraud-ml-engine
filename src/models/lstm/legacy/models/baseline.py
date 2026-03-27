"""
Baseline single-transaction fraud detection model.

Simple MLP that processes each transaction independently.
Used for comparison with sequential LSTM model.
"""

from typing import List

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """
    Multi-layer perceptron for single-transaction fraud detection.

    Uses only the current transaction features (no historical context).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch norm
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(nn.ReLU())

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
               or (batch_size, seq_len, input_dim) - only uses last transaction

        Returns:
            Output tensor of shape (batch_size, 1) with fraud probabilities
        """
        # If sequence input, only use the last transaction
        if x.dim() == 3:
            # Shape: (batch_size, seq_len, input_dim) -> (batch_size, input_dim)
            x = x[:, -1, :]

        return self.network(x)


class BaselineMLPWithAttention(nn.Module):
    """
    Baseline model that uses simple attention over sequence.

    Still uses only current transaction features for final prediction,
    but can attend to historical transactions for context.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.3
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim

        # Projection for attention
        self.attention_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

        # Build classifier layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with attention over sequence.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            return_attention: If True, return attention weights

        Returns:
            Output tensor of shape (batch_size, 1)
            If return_attention=True, also returns attention weights
        """
        batch_size, seq_len, _ = x.shape

        # Compute attention scores
        # Shape: (batch_size, seq_len, 1)
        attention_scores = self.attention_proj(x)

        # Apply softmax over sequence length
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Apply attention to input
        # Shape: (batch_size, input_dim)
        attended_input = torch.sum(x * attention_weights, dim=1)

        # Classify
        output = self.classifier(attended_input)

        if return_attention:
            return output, attention_weights.squeeze(-1)
        return output


def create_baseline_model(
    input_dim: int,
    config: dict
) -> BaselineMLP:
    """
    Factory function to create baseline model from config.

    Args:
        input_dim: Number of input features
        config: Configuration dictionary

    Returns:
        BaselineMLP model
    """
    return BaselineMLP(
        input_dim=input_dim,
        hidden_dims=config['baseline']['hidden_dims'],
        dropout=config['baseline']['dropout']
    )
