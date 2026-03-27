# LSTM Fraud Detection

Sequential fraud detection using LSTM with attention mechanism. This project demonstrates how transaction history improves fraud detection compared to single-transaction models.

## Overview

**Hypothesis**: Sequential transaction patterns provide valuable context for fraud detection beyond individual transaction features.

**Key Features**:
- Variable-length sequence modeling with LSTM
- Multi-head attention for interpretable predictions
- Temporal train/val/test split (no data leakage)
- Efficient `pack_padded_sequence` for variable-length inputs
- Comparison with single-transaction baseline
- Attention weight visualization
- ONNX export for production deployment

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LSTM + Attention Fraud Detection                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: Transaction Sequences                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ [Tx_1, Tx_2, ..., Tx_N]  (N = variable sequence length)            │    │
│  │  each Tx: (num_features,)                                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                          Bi-LSTM                                     │    │
│  │  hidden_dim: 128, num_layers: 2, bidirectional                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Multi-Head Attention                             │    │
│  │  num_heads: 4  →  Learns transaction importance                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Layer Norm + Residual                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Classification Head                            │    │
│  │  FC(256 → 64) → ReLU → Dropout → FC(64 → 1) → Sigmoid               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    ↓                                        │
│  Output: Fraud Probability (0-1)                                           │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Key Features:                                                               │
│  • Variable-length sequences with pack_padded_sequence                       │
│  • Multi-head attention for interpretable predictions                       │
│  • Temporal train/val/test split (no leakage)                               │
│  • Class weighting for imbalanced data                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
lstm_fraud_detection/
├── data/
│   ├── processed/              # Preprocessed sequence data
│   └── raw/                    # Raw transaction data
├── src/
│   ├── data/
│   │   ├── preprocessing.py    # Sequence creation, temporal split
│   │   └── dataset.py          # PyTorch Dataset with packed sequences
│   ├── models/
│   │   ├── lstm_attention.py   # LSTM + Attention model
│   │   └── baseline.py         # Single-transaction MLP baseline
│   ├── training/
│   │   ├── trainer.py          # Training loop with early stopping
│   │   └── metrics.py          # Evaluation metrics (AUC-PR, etc.)
│   ├── utils/
│   │   ├── visualization.py    # Attention weight plotting
│   │   └── export.py           # ONNX export
│   └── inference.py            # Inference interface
├── tests/
│   ├── test_models.py          # Model architecture tests
│   └── test_attention.py       # Attention mechanism tests
├── configs/
│   └── config.yaml             # Hyperparameters
├── scripts/
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Requirements

Your transaction data should contain:
- **User identifier**: Unique ID per customer
- **Timestamp**: Transaction datetime for temporal ordering
- **Features**: Transaction features (amount, merchant, location, etc.)
- **Label**: Binary fraud indicator (0 or 1)

Example columns:
```
user_id, transaction_time, amount, merchant_category, distance_from_home, is_fraud
```

## Usage

### 1. Prepare Configuration

Edit `configs/config.yaml` to set hyperparameters:

```yaml
sequence:
  max_sequence_length: 10  # Number of historical transactions
  min_sequence_length: 3   # Minimum transactions per sequence

model:
  hidden_dim: 128
  num_layers: 2
  num_heads: 4
  dropout: 0.3
  bidirectional: true

training:
  batch_size: 64
  num_epochs: 50
  learning_rate: 0.001
  early_stopping_patience: 10
```

### 2. Train Models

```bash
# Train LSTM model only
python scripts/train.py \
    --config configs/config.yaml \
    --model lstm \
    --data data/transactions.csv \
    --features amount merchant_category distance_from_home \
    --user-col user_id \
    --time-col transaction_time \
    --label-col is_fraud \
    --output-dir results

# Train baseline model only
python scripts/train.py --model baseline ...

# Train both models for comparison
python scripts/train.py --model both ...
```

### 3. Evaluate Model

```bash
python scripts/evaluate.py \
    --checkpoint results/checkpoints/lstm/best_model.pt \
    --data data/test_transactions.csv \
    --features amount merchant_category distance_from_home \
    --visualize-samples 10 \
    --threshold 0.5 \
    --output-dir evaluation_results
```

### 4. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_attention.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Results

The model is evaluated using:

**Primary Metric**: AUC-PR (Area Under Precision-Recall Curve)
- More informative than AUC-ROC for imbalanced data
- Focuses on performance on the minority (fraud) class

**Secondary Metrics**:
- AUC-ROC
- Precision, Recall, F1-Score
- Confusion Matrix

### Expected Performance

Based on similar fraud detection tasks:

| Model | AUC-PR | AUC-ROC | F1-Score |
|-------|--------|---------|----------|
| Baseline (Single Tx) | ~0.15-0.25 | ~0.70-0.80 | ~0.30-0.40 |
| LSTM + Attention | ~0.25-0.40 | ~0.80-0.90 | ~0.45-0.60 |

*Note: Actual results depend on your data quality and fraud patterns.*

## Key Implementation Details

### Variable-Length Sequences

```python
# Padded input (batch_size, max_seq_len, num_features)
padded_sequences = torch.zeros(batch_size, max_seq_len, num_features)

# Actual lengths (for efficiency)
lengths = torch.tensor([10, 7, 6, ...])

# pack_padded_sequence removes padding for efficient LSTM processing
packed = pack_padded_sequence(padded_sequences, lengths, batch_first=True)
lstm_output, _ = self.lstm(packed)
```

### Temporal Split

```python
# Splits by sequence order (NO SHUFFLING)
# Ensures no future data leaks into training
train_data = sequences[:train_end]
val_data = sequences[train_end:val_end]
test_data = sequences[val_end:]
```

### Attention Extraction

```python
# Forward pass with attention weights
predictions, attention = model(sequences, lengths, return_attention=True)

# attention shape: (batch_size, num_heads, seq_len)
# Shows which transactions were most important
```

## ONNX Export

Models are exported to ONNX for production deployment:

```bash
# Automatically exported during training if enabled in config
onnx:
  enabled: true
  opset_version: 17

# Models saved to: results/onnx/
# - lstm_fraud_detector.onnx
# - baseline_fraud_detector.onnx
```

Use with ONNX Runtime:

```python
import onnxruntime as ort

session = ort.InferenceSession("results/onnx/lstm_fraud_detector.onnx")

predictions = session.run(
    None,
    {
        'padded_sequences': your_sequences,
        'lengths': your_lengths
    }
)
```

## Visualization

Attention weights show which transactions influenced the prediction:

```python
# Visualize attention for specific predictions
from src.utils.visualization import plot_attention_weights

plot_attention_weights(
    attention_weights=attention[0].cpu().numpy(),
    save_path="attention_plot.png"
)
```

This helps explain:
- Which transactions flagged fraud
- Temporal patterns in user behavior
- Model decision-making process

## Technical Notes

### Class Imbalance

Fraud detection typically has 0.1-2% positive cases. We address this by:

1. **Class Weighting**: Higher weight for fraud cases
2. **AUC-PR Metric**: Focuses on positive class performance
3. **Threshold Tuning**: Optimize for business requirements

### Memory Efficiency

- `pack_padded_sequence` reduces computation by ~40-60%
- Gradient clipping prevents exploding gradients
- DataLoader with `pin_memory=True` for GPU transfer

### Reproducibility

All random seeds are set:
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
```

## References

1. [LSTM for Sequence Classification](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
2. [Multi-Head Attention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
3. [Pack Padded Sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html)
4. [ONNX Export](https://pytorch.org/docs/stable/onnx.html)

## License

MIT License

## Author

Built for AI Engineer portfolio and PhD applications.
Demonstrates deep learning expertise in fraud detection.
