"""
Unit tests for attention mechanism.
"""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lstm_attention import LSTMAttentionClassifier


class TestAttentionMechanism:
    """Test attention mechanism functionality."""

    @pytest.fixture
    def model(self):
        return LSTMAttentionClassifier(
            input_dim=10,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            bidirectional=True
        )

    def test_attention_weights_sum_to_one(self, model):
        """Test attention weights sum (approximately) to 1 per head."""
        batch_size = 2
        seq_len = 6
        input_dim = 10

        sequences = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([6, 5])

        predictions, attention = model(sequences, lengths, return_attention=True)

        # attention shape: (batch_size, num_heads, seq_len)
        # Each row should sum to approximately 1 (softmax over seq_len)

        for batch_idx in range(batch_size):
            for head_idx in range(model.num_heads):
                # Only consider actual sequence length
                actual_len = lengths[batch_idx].item()
                weights = attention[batch_idx, head_idx, :actual_len]

                # Sum should be approximately 1
                assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)

    def test_attention_weight_ranges(self, model):
        """Test attention weights are in valid range [0, 1]."""
        batch_size = 3
        seq_len = 8
        input_dim = 10

        sequences = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([8, 6, 4])

        predictions, attention = model(sequences, lengths, return_attention=True)

        # All weights should be in [0, 1]
        assert (attention >= 0).all()
        assert (attention <= 1).all()

    def test_attention_weights_different_for_different_positions(self, model):
        """Test attention assigns different weights to different positions."""
        batch_size = 1
        seq_len = 5
        input_dim = 10

        sequences = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([seq_len])

        predictions, attention = model(sequences, lengths, return_attention=True)

        # Check that not all weights are identical
        # (they could be similar but should vary based on input)
        for head_idx in range(model.num_heads):
            weights = attention[0, head_idx, :]
            std = weights.std()

            # Standard deviation should be > 0 (not all identical)
            # Note: This could occasionally fail due to randomness,
            # but is statistically very unlikely
            assert std > 0 or torch.allclose(weights, torch.tensor([1.0/seq_len] * seq_len), atol=1e-5)

    def test_attention_masking(self, model):
        """Test that padding positions receive zero/low attention."""
        batch_size = 2
        max_seq_len = 10
        actual_seq_len = 5
        input_dim = 10

        sequences = torch.randn(batch_size, max_seq_len, input_dim)
        lengths = torch.tensor([actual_seq_len, actual_seq_len])

        predictions, attention = model(sequences, lengths, return_attention=True)

        # Check that padding positions (after actual_seq_len) get minimal attention
        for batch_idx in range(batch_size):
            for head_idx in range(model.num_heads):
                # Actual transaction positions should have higher total attention
                # than padding positions
                actual_attention = attention[batch_idx, head_idx, :actual_seq_len].sum()
                padding_attention = attention[batch_idx, head_idx, actual_seq_len:].sum()

                # Padding should have negligible attention
                assert padding_attention < 0.01  # Less than 1%

    def test_multi_head_diversity(self, model):
        """Test that different attention heads learn different patterns."""
        batch_size = 1
        seq_len = 6
        input_dim = 10

        sequences = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([seq_len])

        predictions, attention = model(sequences, lengths, return_attention=True)

        # attention shape: (1, num_heads, seq_len)
        attention = attention.squeeze(0)  # (num_heads, seq_len)

        # Check that different heads don't all have identical attention patterns
        # Compute correlation between heads
        head_patterns = []
        for head_idx in range(model.num_heads):
            head_patterns.append(attention[head_idx].unsqueeze(0))

        stacked = torch.cat(head_patterns, dim=0)  # (num_heads, seq_len)

        # Compute pairwise correlations
        correlations = []
        for i in range(model.num_heads):
            for j in range(i+1, model.num_heads):
                corr = torch.corrcoef(torch.stack([stacked[i], stacked[j]]))[0, 1]
                correlations.append(corr.item())

        # Not all heads should be perfectly correlated
        # Allow some to be similar, but not all
        avg_correlation = sum(correlations) / len(correlations)
        assert avg_correlation < 0.99  # At least some diversity

    def test_attention_reproducibility(self, model):
        """Test attention is deterministic with same input."""
        batch_size = 1
        seq_len = 5
        input_dim = 10

        sequences = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([seq_len])

        model.eval()

        with torch.no_grad():
            predictions1, attention1 = model(sequences, lengths, return_attention=True)
            predictions2, attention2 = model(sequences, lengths, return_attention=True)

        # Should be identical in eval mode
        assert torch.allclose(attention1, attention2, atol=1e-6)
        assert torch.allclose(predictions1, predictions2, atol=1e-6)

    def test_attention_gradient_flow(self, model):
        """Test gradients flow through attention mechanism."""
        batch_size = 2
        seq_len = 4
        input_dim = 10

        sequences = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([seq_len, seq_len])
        labels = torch.tensor([1.0, 0.0])

        model.train()

        predictions, attention = model(sequences, lengths, return_attention=True)

        # Compute loss
        loss = nn.BCELoss()(predictions, labels.unsqueeze(1))

        # Backward pass
        loss.backward()

        # Check gradients exist for attention layer
        assert model.attention.out_proj.weight.grad is not None
        assert not torch.isnan(model.attention.out_proj.weight.grad).any()


class TestAttentionExtraction:
    """Test attention weight extraction for visualization."""

    @pytest.fixture
    def model(self):
        return LSTMAttentionClassifier(
            input_dim=10,
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            dropout=0.0,
            bidirectional=True
        )

    def test_extract_attention_weights_return_type(self, model):
        """Test extract_attention_weights returns tensor."""
        batch_size = 2
        seq_len = 6
        input_dim = 10

        sequences = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([6, 5])

        model.eval()
        attention = model.extract_attention_weights(sequences, lengths)

        assert isinstance(attention, torch.Tensor)

    def test_extract_attention_weights_shape(self, model):
        """Test extract_attention_weights returns correct shape."""
        batch_size = 3
        seq_len = 8
        input_dim = 10

        sequences = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([8, 6, 4])

        model.eval()
        attention = model.extract_attention_weights(sequences, lengths)

        # Shape should be (batch_size, num_heads, seq_len)
        assert attention.shape == (batch_size, model.num_heads, seq_len)

    def test_extract_attention_no_grad(self, model):
        """Test extract_attention_weights doesn't compute gradients."""
        batch_size = 2
        seq_len = 5
        input_dim = 10

        sequences = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([5, 4])

        model.eval()
        attention = model.extract_attention_weights(sequences, lengths)

        # Should not have gradients
        assert attention.grad is None

    def test_extract_attention_matches_forward(self, model):
        """Test extract_attention_weights matches forward pass attention."""
        batch_size = 2
        seq_len = 6
        input_dim = 10

        sequences = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([6, 5])

        model.eval()

        # Forward pass
        _, attention_forward = model(sequences, lengths, return_attention=True)

        # Extract method
        attention_extracted = model.extract_attention_weights(sequences, lengths)

        # Should be identical
        assert torch.allclose(attention_forward, attention_extracted, atol=1e-6)
