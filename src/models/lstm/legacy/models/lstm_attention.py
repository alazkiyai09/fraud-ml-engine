"""
LSTM model with attention mechanism for sequential fraud detection.

Processes variable-length transaction sequences and extracts attention weights.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMAttentionClassifier(nn.Module):
    """
    LSTM with multi-head attention for fraud detection on transaction sequences.

    Architecture:
        1. LSTM processes sequential transactions
        2. Attention mechanism weights each transaction's importance
        3. Final prediction based on attended representation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Args:
            input_dim: Number of input features per transaction
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm_output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.lstm_output_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_output_dim, self.lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.lstm_output_dim // 2, 1),
            nn.Sigmoid()
        )

        # Store attention weights for visualization
        self.last_attention_weights = None

    def forward(
        self,
        padded_sequences: torch.Tensor,
        lengths: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through LSTM with attention.

        Args:
            padded_sequences: (batch_size, max_seq_len, input_dim)
            lengths: (batch_size,) actual sequence lengths
            return_attention: If True, return attention weights

        Returns:
            predictions: (batch_size, 1) fraud probabilities
            attention_weights: (batch_size, num_heads, max_seq_len, max_seq_len) if requested
        """
        batch_size, max_seq_len, _ = padded_sequences.shape

        # Sort by length (descending) for pack_padded_sequence
        sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_sequences = padded_sequences[sorted_indices]

        # Pack sequences for efficiency
        packed_input = pack_padded_sequence(
            sorted_sequences,
            sorted_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True
        )

        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # Unpack
        lstm_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=max_seq_len
        )  # Shape: (batch_size, max_seq_len, lstm_output_dim)

        # Create attention mask
        # Mask should be False for valid positions, True for padding
        attention_mask = torch.arange(max_seq_len, device=lengths.device)[None, :] >= sorted_lengths[:, None]

        # Apply multi-head attention
        # Query, Key, Value all come from LSTM output
        attended_output, attention_weights = self.attention(
            lstm_output,
            lstm_output,
            lstm_output,
            key_padding_mask=attention_mask,
            need_weights=True
        )  # attended_output: (batch_size, max_seq_len, lstm_output_dim)
           # attention_weights: (batch_size, max_heads, max_seq_len)

        # Store attention weights for visualization
        self.last_attention_weights = attention_weights

        # Layer normalization and residual connection
        attended_output = self.layer_norm(attended_output + lstm_output)

        # Extract final representation (use last actual transaction, not padding)
        # Get the last valid index for each sequence
        last_indices = (sorted_lengths - 1).clamp(min=0)

        # Gather the output at the last valid position
        batch_indices = torch.arange(batch_size, device=lstm_output.device)
        final_output = attended_output[batch_indices, last_indices]

        # Unsort to match original order
        _, unsorted_indices = torch.sort(sorted_indices)
        final_output = final_output[unsorted_indices]

        # Classification
        predictions = self.classifier(final_output)

        if return_attention:
            # Unsort attention weights
            attention_weights = attention_weights[unsorted_indices]
            return predictions, attention_weights

        return predictions, None

    def extract_attention_weights(
        self,
        padded_sequences: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract attention weights for visualization.

        Args:
            padded_sequences: (batch_size, max_seq_len, input_dim)
            lengths: (batch_size,) actual sequence lengths

        Returns:
            attention_weights: (batch_size, num_heads, max_seq_len)
        """
        with torch.no_grad():
            self.eval()
            _ = self.forward(padded_sequences, lengths, return_attention=True)
            return self.last_attention_weights

    def predict(
        self,
        padded_sequences: torch.Tensor,
        lengths: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with binary threshold.

        Args:
            padded_sequences: (batch_size, max_seq_len, input_dim)
            lengths: (batch_size,) actual sequence lengths
            threshold: Decision threshold

        Returns:
            predictions: (batch_size,) binary predictions (0 or 1)
            probabilities: (batch_size,) fraud probabilities
        """
        self.eval()
        with torch.no_grad():
            probs, _ = self.forward(padded_sequences, lengths)
            probs = probs.squeeze(-1)
            preds = (probs >= threshold).long()
        return preds, probs

    def get_embeddings(
        self,
        padded_sequences: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract final embeddings before classification layer.

        Useful for analysis or downstream tasks.

        Args:
            padded_sequences: (batch_size, max_seq_len, input_dim)
            lengths: (batch_size,) actual sequence lengths

        Returns:
            embeddings: (batch_size, lstm_output_dim // 2)
        """
        self.eval()
        with torch.no_grad():
            batch_size, max_seq_len, _ = padded_sequences.shape

            # Sort and pack
            sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
            sorted_sequences = padded_sequences[sorted_indices]

            packed_input = pack_padded_sequence(
                sorted_sequences,
                sorted_lengths.cpu(),
                batch_first=True,
                enforce_sorted=True
            )

            # LSTM
            packed_output, _ = self.lstm(packed_input)
            lstm_output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=max_seq_len
            )

            # Attention mask
            attention_mask = torch.arange(max_seq_len, device=lengths.device)[None, :] >= sorted_lengths[:, None]

            # Attention
            attended_output, _ = self.attention(
                lstm_output,
                lstm_output,
                lstm_output,
                key_padding_mask=attention_mask
            )

            # Layer norm
            attended_output = self.layer_norm(attended_output + lstm_output)

            # Extract last valid position
            last_indices = (sorted_lengths - 1).clamp(min=0)
            batch_indices = torch.arange(batch_size, device=lstm_output.device)
            final_output = attended_output[batch_indices, last_indices]

            # Unsort
            _, unsorted_indices = torch.sort(sorted_indices)
            final_output = final_output[unsorted_indices]

            # First layer of classifier (before ReLU)
            embeddings = self.classifier[0](final_output)

        return embeddings


def create_lstm_model(
    input_dim: int,
    config: dict
) -> LSTMAttentionClassifier:
    """
    Factory function to create LSTM model from config.

    Args:
        input_dim: Number of input features
        config: Configuration dictionary

    Returns:
        LSTMAttentionClassifier model
    """
    return LSTMAttentionClassifier(
        input_dim=input_dim,
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        bidirectional=config['model']['bidirectional']
    )
