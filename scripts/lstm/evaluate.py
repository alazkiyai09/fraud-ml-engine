#!/usr/bin/env python
"""
Evaluation script for trained fraud detection models.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/lstm/best_model.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import prepare_data
from src.data.dataset import FraudSequenceDataset, collate_fn
from src.models.lstm_attention import LSTMAttentionClassifier
from src.models.baseline import BaselineMLP
from src.training.trainer import Trainer
from src.utils.visualization import visualize_predictions
import yaml


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct model from config
    config = checkpoint['config']

    # Determine model type from checkpoint structure
    # This is a simple heuristic - you might want to store model_type in checkpoint
    if 'attention' in str(checkpoint['model_state_dict'].keys()):
        model = LSTMAttentionClassifier(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            dropout=config['model']['dropout'],
            bidirectional=config['model']['bidirectional']
        )
    else:
        model = BaselineMLP(
            input_dim=config['model']['input_dim'],
            hidden_dims=config['baseline']['hidden_dims'],
            dropout=config['baseline']['dropout']
        )

    model.load_state_dict(checkpoint['model_state_dict'])

    return model, config


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained fraud detection model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to test data CSV file")
    parser.add_argument("--features", type=str, nargs="+", required=True,
                       help="List of feature column names")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (overrides checkpoint config)")
    parser.add_argument("--user-col", type=str, default="user_id",
                       help="User identifier column name")
    parser.add_argument("--time-col", type=str, default="transaction_time",
                       help="Timestamp column name")
    parser.add_argument("--label-col", type=str, default="is_fraud",
                       help="Fraud label column name")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--visualize-samples", type=int, default=10,
                       help="Number of samples to visualize with attention")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Decision threshold for predictions")

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}")
    model, checkpoint_config = load_checkpoint(args.checkpoint, device)

    # Use provided config or checkpoint config
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint_config

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare test data
    print(f"\nLoading data from {args.data}")
    df = pd.read_csv(args.data)

    # Update config with actual input dimension
    config['model']['input_dim'] = len(args.features)

    # Prepare data
    print("\nPreparing test data...")
    _, _, test_data, _, _ = prepare_data(
        df=df,
        feature_columns=args.features,
        config=config
    )

    # Create test dataset and dataloader
    test_dataset = FraudSequenceDataset(test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()

    # Run evaluation
    print("\nRunning evaluation...")
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:  # LSTM model
                sequences, labels, lengths = batch
                sequences = sequences.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)

                probs, _ = model(sequences, lengths)
            else:  # Baseline model
                sequences, labels = batch
                sequences = sequences.to(device)
                labels = labels.to(device)

                probs = model(sequences)

            probs = probs.squeeze(-1)
            preds = (probs >= args.threshold).long()

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    from sklearn.metrics import (
        precision_recall_curve, auc, roc_auc_score,
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report
    )

    # Calculate metrics
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    auc_pr = auc(recall, precision)
    auc_roc = roc_auc_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision_val = precision_score(all_labels, all_predictions, zero_division=0)
    recall_val = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    cm = confusion_matrix(all_labels, all_predictions)

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nDecision Threshold: {args.threshold}")
    print(f"\nPrimary Metrics:")
    print(f"  AUC-PR (Precision-Recall): {auc_pr:.4f}")
    print(f"  AUC-ROC:                  {auc_roc:.4f}")
    print(f"\nThreshold-Dependent Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision_val:.4f}")
    print(f"  Recall:    {recall_val:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives:  {cm[1, 1]}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions,
                              target_names=['Legitimate', 'Fraud'],
                              digits=4))

    # Save results
    results_path = output_dir / "evaluation_results.txt"
    with open(results_path, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Data: {args.data}\n")
        f.write(f"Decision Threshold: {args.threshold}\n\n")
        f.write(f"AUC-PR: {auc_pr:.4f}\n")
        f.write(f"AUC-ROC: {auc_roc:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision_val:.4f}\n")
        f.write(f"Recall: {recall_val:.4f}\n")
        f.write(f"F1: {f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_predictions,
                                    target_names=['Legitimate', 'Fraud'],
                                    digits=4))

    print(f"\n✓ Results saved to {results_path}")

    # Visualize attention weights (only for LSTM)
    if isinstance(model, LSTMAttentionClassifier) and args.visualize_samples > 0:
        print(f"\nGenerating {args.visualize_samples} attention visualizations...")
        viz_dir = output_dir / "attention_visualizations"
        visualize_predictions(
            model=model,
            dataset=test_dataset,
            device=device,
            num_samples=args.visualize_samples,
            save_path=str(viz_dir)
        )
        print(f"✓ Visualizations saved to {viz_dir}")


if __name__ == "__main__":
    main()
