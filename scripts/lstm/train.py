#!/usr/bin/env python
"""
Main training script for LSTM fraud detection.

Usage:
    python scripts/train.py --config configs/config.yaml
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import prepare_data
from src.data.dataset import FraudSequenceDataset, collate_fn
from src.models.lstm_attention import create_lstm_model
from src.models.baseline import create_baseline_model
from src.training.trainer import Trainer
from src.utils.visualization import (
    plot_training_history,
    plot_model_comparison,
    create_architecture_diagram
)
from src.utils.export import export_model
import yaml


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train LSTM fraud detection model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "baseline", "both"],
                       help="Model to train")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to training data CSV file")
    parser.add_argument("--features", type=str, nargs="+", required=True,
                       help="List of feature column names")
    parser.add_argument("--user-col", type=str, default="user_id",
                       help="User identifier column name")
    parser.add_argument("--time-col", type=str, default="transaction_time",
                       help="Timestamp column name")
    parser.add_argument("--label-col", type=str, default="is_fraud",
                       help="Fraud label column name")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set seed
    set_seed(config['seed'])

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)

    # Update config with actual input dimension
    config['model']['input_dim'] = len(args.features)

    # Prepare data
    print("\n" + "="*60)
    print("PREPARING DATA")
    print("="*60)
    train_data, val_data, test_data, scaler, class_weights = prepare_data(
        df=df,
        feature_columns=args.features,
        config=config
    )

    # Create datasets
    train_dataset = FraudSequenceDataset(train_data)
    val_dataset = FraudSequenceDataset(val_data)
    test_dataset = FraudSequenceDataset(test_data)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    results = {}

    # Train LSTM model
    if args.model in ["lstm", "both"]:
        print("\n" + "="*60)
        print("TRAINING LSTM MODEL")
        print("="*60)

        # Create model
        lstm_model = create_lstm_model(
            input_dim=len(args.features),
            config=config
        )

        # Create trainer
        lstm_trainer = Trainer(
            model=lstm_model,
            config=config,
            checkpoint_dir=str(output_dir / "checkpoints" / "lstm")
        )

        # Train
        lstm_history = lstm_trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            class_weights=class_weights
        )

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics, _ = lstm_trainer.validate(test_loader)
        print("\nTest Set Metrics:")
        print(f"AUC-PR: {test_metrics['auc_pr']:.4f}")
        print(f"AUC-ROC: {test_metrics['auc_roc']:.4f}")
        print(f"F1: {test_metrics['f1']:.4f}")

        results['lstm'] = test_metrics

        # Save training plot
        plot_training_history(lstm_history, save_path=str(output_dir / "lstm_training_history.png"))

        # Export to ONNX
        if config['onnx']['enabled']:
            print("\nExporting LSTM to ONNX...")
            export_model(
                model=lstm_model,
                model_type="lstm",
                config=config,
                output_dir=str(output_dir / "onnx"),
                validate=True
            )

    # Train baseline model
    if args.model in ["baseline", "both"]:
        print("\n" + "="*60)
        print("TRAINING BASELINE MODEL")
        print("="*60)

        # Create model
        baseline_model = create_baseline_model(
            input_dim=len(args.features),
            config=config
        )

        # Create trainer
        baseline_trainer = Trainer(
            model=baseline_model,
            config=config,
            checkpoint_dir=str(output_dir / "checkpoints" / "baseline")
        )

        # Train
        baseline_history = baseline_trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            class_weights=class_weights
        )

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics, _ = baseline_trainer.validate(test_loader)
        print("\nTest Set Metrics:")
        print(f"AUC-PR: {test_metrics['auc_pr']:.4f}")
        print(f"AUC-ROC: {test_metrics['auc_roc']:.4f}")
        print(f"F1: {test_metrics['f1']:.4f}")

        results['baseline'] = test_metrics

        # Save training plot
        plot_training_history(baseline_history, save_path=str(output_dir / "baseline_training_history.png"))

        # Export to ONNX
        if config['onnx']['enabled']:
            print("\nExporting baseline to ONNX...")
            export_model(
                model=baseline_model,
                model_type="baseline",
                config=config,
                output_dir=str(output_dir / "onnx"),
                validate=True
            )

    # Create comparison plot if both models trained
    if args.model == "both" and len(results) == 2:
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(f"\nLSTM AUC-PR:     {results['lstm']['auc_pr']:.4f}")
        print(f"Baseline AUC-PR: {results['baseline']['auc_pr']:.4f}")
        print(f"Improvement:     {results['lstm']['auc_pr'] - results['baseline']['auc_pr']:.4f}")

        plot_model_comparison(
            results['lstm'],
            results['baseline'],
            save_path=str(output_dir / "model_comparison.png")
        )

    # Save results
    results_path = output_dir / "test_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types for JSON serialization
        serializable_results = {}
        for model_name, metrics in results.items():
            serializable_results[model_name] = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in metrics.items()
            }
        json.dump(serializable_results, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}")
    print(f"✓ Test metrics saved to {results_path}")


if __name__ == "__main__":
    main()
