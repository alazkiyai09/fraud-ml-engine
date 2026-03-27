"""
Unit tests for model architectures.
"""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lstm_attention import LSTMAttentionClassifier
from src.models.baseline import BaselineMLP


class TestLSTMAttentionClassifier:
    """Test LSTM + Attention model."""

    @pytest.fixture
    def model_config(self):
        return {
            'input_dim': 10,
            'hidden_dim': 32,
            'num_layers': 2,
            'num_heads': 2,
            'dropout': 0.1,
            'bidirectional': True
        }

    @pytest.fixture
    def model(self, model_config):
        return LSTMAttentionClassifier(**model_config)

    def test_model_initialization(self, model, model_config):
        """Test model initializes correctly."""
        assert model.input_dim == model_config['input_dim']
        assert model.hidden_dim == model_config['hidden_dim']
        assert model.num_layers == model_config['num_layers']
        assert model.num_heads == model_config['num_heads']

        # Check LSTM dimensions
        lstm_output_dim = model_config['hidden_dim'] * 2  # bidirectional
        assert model.lstm_output_dim == lstm_output_dim

    def test_forward_pass(self, model):
        """Test forward pass with batch input."""
        batch_size = 4
        seq_len = 8
        input_dim = 10

        sequences = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([8, 7, 6, 5])

        predictions, attention = model(sequences, lengths)

        # Check output shape
        assert predictions.shape == (batch_size, 1)
        assert predictions.min() >= 0 and predictions.max() <= 1  # Probabilities

    def test_forward_pass_with_attention(self, model):
        """Test forward pass returns attention weights when requested."""
        batch_size = 4
        seq_len = 8
        input_dim = 10

        sequences = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([8, 7, 6, 5])

        predictions, attention = model(sequences, lengths, return_attention=True)

        # Check attention shape
        # (batch_size, num_heads, seq_len)
        assert attention is not None
        assert attention.shape[0] == batch_size
        assert attention.shape[1] == model.num_heads
        assert attention.shape[2] == seq_len

    def test_variable_length_sequences(self, model):
        """Test model handles variable-length sequences correctly."""
        batch_size = 3
        max_seq_len = 10
        input_dim = 10

        sequences = torch.randn(batch_size, max_seq_len, input_dim)
        lengths = torch.tensor([10, 3, 7])

        predictions, _ = model(sequences, lengths)

        assert predictions.shape == (batch_size, 1)
        assert torch.isfinite(predictions).all()

    def test_extract_attention_weights(self, model):
        """Test attention weight extraction."""
        batch_size = 2
        seq_len = 6
        input_dim = 10

        sequences = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([6, 4])

        attention = model.extract_attention_weights(sequences, lengths)

        assert attention.shape == (batch_size, model.num_heads, seq_len)

    def test_predict_method(self, model):
        """Test predict method with threshold."""
        batch_size = 4
        seq_len = 8
        input_dim = 10

        sequences = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([8, 7, 6, 5])

        preds, probs = model.predict(sequences, lengths, threshold=0.5)

        assert preds.shape == (batch_size,)
        assert probs.shape == (batch_size,)
        assert torch.isin(preds, torch.tensor([0, 1])).all()

    def test_get_embeddings(self, model):
        """Test embedding extraction."""
        batch_size = 2
        seq_len = 5
        input_dim = 10

        sequences = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([5, 3])

        embeddings = model.get_embeddings(sequences, lengths)

        # Shape should be (batch_size, lstm_output_dim // 2)
        expected_dim = model.lstm_output_dim // 2
        assert embeddings.shape == (batch_size, expected_dim)


class TestBaselineMLP:
    """Test baseline MLP model."""

    @pytest.fixture
    def model_config(self):
        return {
            'input_dim': 10,
            'hidden_dims': [32, 16],
            'dropout': 0.2
        }

    @pytest.fixture
    def model(self, model_config):
        return BaselineMLP(**model_config)

    def test_model_initialization(self, model, model_config):
        """Test model initializes correctly."""
        assert model.input_dim == model_config['input_dim']
        assert len(model.hidden_dims) == len(model_config['hidden_dims'])

    def test_forward_pass_single_transaction(self, model):
        """Test forward pass with single transaction input."""
        batch_size = 4
        input_dim = 10

        x = torch.randn(batch_size, input_dim)
        output = model(x)

        assert output.shape == (batch_size, 1)
        assert output.min() >= 0 and output.max() <= 1  # Probabilities

    def test_forward_pass_sequence_input(self, model):
        """Test forward pass with sequence input (should use last transaction)."""
        batch_size = 4
        seq_len = 8
        input_dim = 10

        x = torch.randn(batch_size, seq_len, input_dim)
        output = model(x)

        assert output.shape == (batch_size, 1)

    def test_backward_pass(self, model):
        """Test backward pass computes gradients."""
        batch_size = 4
        input_dim = 10

        x = torch.randn(batch_size, input_dim)
        y = torch.tensor([0, 1, 0, 1], dtype=torch.float32).view(-1, 1)

        output = model(x)
        loss = nn.BCELoss()(output, y)

        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()


class TestModelComparison:
    """Integration tests comparing models."""

    @pytest.fixture
    def sample_data(self):
        return {
            'batch_size': 8,
            'seq_len': 10,
            'input_dim': 15
        }

    def test_lstm_vs_baseline_shapes(self, sample_data):
        """Test both models produce compatible output shapes."""
        lstm = LSTMAttentionClassifier(
            input_dim=sample_data['input_dim'],
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            bidirectional=True
        )

        baseline = BaselineMLP(
            input_dim=sample_data['input_dim'],
            hidden_dims=[32, 16],
            dropout=0.1
        )

        sequences = torch.randn(
            sample_data['batch_size'],
            sample_data['seq_len'],
            sample_data['input_dim']
        )
        lengths = torch.tensor([10, 9, 8, 7, 6, 5, 4, 3])

        # LSTM output
        lstm_output, _ = lstm(sequences, lengths)

        # Baseline output (uses last transaction)
        baseline_output = baseline(sequences)

        # Both should have same batch output shape
        assert lstm_output.shape == baseline_output.shape == (sample_data['batch_size'], 1)

    def test_model_parameter_counts(self, sample_data):
        """Test and compare parameter counts."""
        lstm = LSTMAttentionClassifier(
            input_dim=sample_data['input_dim'],
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            bidirectional=True
        )

        baseline = BaselineMLP(
            input_dim=sample_data['input_dim'],
            hidden_dims=[32, 16],
            dropout=0.1
        )

        lstm_params = sum(p.numel() for p in lstm.parameters())
        baseline_params = sum(p.numel() for p in baseline.parameters())

        # LSTM should have more parameters
        assert lstm_params > baseline_params

        print(f"\nLSTM parameters: {lstm_params:,}")
        print(f"Baseline parameters: {baseline_params:,}")
        print(f"Ratio: {lstm_params / baseline_params:.2f}x")
