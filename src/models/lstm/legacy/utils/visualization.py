"""
Visualization utilities for attention weights and model predictions.
"""

from typing import List, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch


def plot_attention_weights(
    attention_weights: np.ndarray,
    transaction_features: Optional[pd.DataFrame] = None,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Attention Weights"
):
    """
    Visualize attention weights for a sequence.

    Args:
        attention_weights: Attention weights array
            - For multi-head: (num_heads, seq_len) or (seq_len, seq_len)
            - For simple attention: (seq_len,)
        transaction_features: DataFrame with transaction details
        feature_names: Names of features to display
        save_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))

    # Handle different attention weight formats
    if attention_weights.ndim == 1:
        # Simple attention: (seq_len,)
        weights = attention_weights
        axes.bar(range(len(weights)), weights)
        axes.set_xlabel('Transaction Position')
        axes.set_ylabel('Attention Weight')
        axes.set_title(title)
    elif attention_weights.ndim == 2:
        # Multi-head or self-attention
        if attention_weights.shape[0] < attention_weights.shape[1]:
            # (num_heads, seq_len) - averaged attention per position
            weights = attention_weights
            im = axes.imshow(weights, cmap='viridis', aspect='auto')
            axes.set_xlabel('Sequence Position')
            axes.set_ylabel('Attention Head')
            axes.set_title(title)
            plt.colorbar(im, ax=axes)
        else:
            # (seq_len, seq_len) - self-attention matrix
            weights = attention_weights
            im = axes.imshow(weights, cmap='viridis', aspect='auto')
            axes.set_xlabel('Key Position')
            axes.set_ylabel('Query Position')
            axes.set_title(title)
            plt.colorbar(im, ax=axes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def visualize_predictions(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    num_samples: int = 5,
    save_path: Optional[str] = None
):
    """
    Visualize model predictions with attention weights.

    Args:
        model: Trained LSTM model
        dataset: Dataset to visualize
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_path: Directory to save visualizations
    """
    model.eval()

    # Get random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for i, idx in enumerate(indices):
        # Get sample
        sequences, labels, lengths = dataset[idx]

        # Add batch dimension
        sequences = sequences.unsqueeze(0).to(device)
        lengths = torch.tensor([lengths]).to(device)

        # Get predictions and attention
        with torch.no_grad():
            predictions, attention_weights = model(sequences, lengths, return_attention=True)

        # Convert to numpy
        attention_weights = attention_weights.cpu().numpy()

        # Average over heads: (1, num_heads, seq_len) -> (seq_len,)
        if attention_weights.ndim == 3:
            attention_weights = attention_weights[0].mean(axis=0)
        elif attention_weights.ndim == 2:
            attention_weights = attention_weights[0]

        # Truncate to actual sequence length
        actual_length = lengths[0].item()
        attention_weights = attention_weights[:actual_length]

        # Create plot
        fig, axes = plt.subplots(1, 1, figsize=(10, 4))

        positions = range(1, actual_length + 1)
        axes.bar(positions, attention_weights)
        axes.set_xlabel('Transaction Position (oldest → newest)')
        axes.set_ylabel('Attention Weight')
        axes.set_title(
            f"Sample {i+1}: "
            f"True={labels.item():.0f}, "
            f"Pred={predictions.item():.3f}"
        )

        plt.tight_layout()

        if save_path:
            import os
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"attention_sample_{i+1}.png"),
                       dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None
):
    """
    Plot training metrics over epochs.

    Args:
        history: Training history from Trainer
        save_path: Path to save figure
    """
    metrics_to_plot = ['auc_pr', 'auc_roc', 'loss']

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5*len(metrics_to_plot), 4))

    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_to_plot):
        if metric in history['train']:
            ax.plot(history['train'][metric], label='Train', marker='o')
        if metric in history['val']:
            ax.plot(history['val'][metric], label='Val', marker='s')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_model_comparison(
    lstm_metrics: Dict,
    baseline_metrics: Dict,
    save_path: Optional[str] = None
):
    """
    Compare LSTM and baseline model performance.

    Args:
        lstm_metrics: LSTM model metrics
        baseline_metrics: Baseline model metrics
        save_path: Path to save figure
    """
    metrics = ['auc_pr', 'auc_roc', 'precision', 'recall', 'f1']
    labels = ['AUC-PR', 'AUC-ROC', 'Precision', 'Recall', 'F1']

    lstm_values = [lstm_metrics[m] for m in metrics]
    baseline_values = [baseline_metrics[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, lstm_values, width, label='LSTM + Attention', alpha=0.8)
    bars2 = ax.bar(x + width/2, baseline_values, width, label='Baseline (Single Tx)', alpha=0.8)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def create_architecture_diagram(save_path: Optional[str] = None):
    """
    Create a text-based architecture diagram for the README.

    Args:
        save_path: Path to save the diagram
    """
    diagram = """
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
"""

    if save_path:
        with open(save_path, 'w') as f:
            f.write(diagram)
    else:
        print(diagram)

    return diagram
