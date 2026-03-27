"""
Training loop for fraud detection models.

Handles training, validation, checkpointing, and early stopping.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics, MetricTracker


class Trainer:
    """
    Trainer for fraud detection models.

    Supports both LSTM and baseline models with class weights.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        checkpoint_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            model: PyTorch model to train
            config: Configuration dictionary
            checkpoint_dir: Directory to save checkpoints
            device: Device to train on ('cuda', 'cpu', or None for auto)
        """
        self.model = model
        self.config = config

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Loss function
        self.criterion = nn.BCELoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Gradient clipping
        self.grad_clip_value = config['training'].get('gradient_clip_value', 1.0)

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metric tracker
        self.tracker = MetricTracker()

        # Class weights (moved to device)
        self.class_weights = None

    def set_class_weights(self, weights: np.ndarray):
        """
        Set class weights for loss calculation.

        Args:
            weights: Array of shape (2,) with weights for [class_0, class_1]
        """
        self.class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted loss.

        Args:
            predictions: (batch_size, 1) predicted probabilities
            targets: (batch_size,) true labels

        Returns:
            Loss tensor
        """
        # Reshape targets
        targets = targets.view(-1, 1)

        # Base loss
        loss = self.criterion(predictions, targets)

        # Apply class weights if available
        if self.class_weights is not None:
            # Weight each sample based on its class
            weights = targets * self.class_weights[1] + (1 - targets) * self.class_weights[0]
            loss = (loss * weights).mean()

        return loss

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Tuple[Dict[str, float], float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Tuple of (metrics_dict, average_loss)
        """
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            # Move to device
            if len(batch) == 3:  # LSTM model
                sequences, labels, lengths = batch
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                # Forward pass
                predictions, _ = self.model(sequences, lengths)
            else:  # Baseline model
                sequences, labels = batch
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                predictions = self.model(sequences)

            # Compute loss
            loss = self.compute_loss(predictions, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip_value > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            all_predictions.extend(predictions.squeeze(-1).detach().cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

        # Compute metrics
        avg_loss = total_loss / len(train_loader)
        metrics = compute_metrics(
            np.array(all_targets),
            np.array(all_predictions)
        )
        metrics['loss'] = avg_loss

        return metrics, avg_loss

    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[Dict[str, float], float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (metrics_dict, average_loss)
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                if len(batch) == 3:  # LSTM model
                    sequences, labels, lengths = batch
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device)
                    lengths = lengths.to(self.device)

                    # Forward pass
                    predictions, _ = self.model(sequences, lengths)
                else:  # Baseline model
                    sequences, labels = batch
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    predictions = self.model(sequences)

                # Compute loss
                loss = self.compute_loss(predictions, labels)

                # Track metrics
                total_loss += loss.item()
                all_predictions.extend(predictions.squeeze(-1).cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        metrics = compute_metrics(
            np.array(all_targets),
            np.array(all_predictions)
        )
        metrics['loss'] = avg_loss

        return metrics, avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        class_weights: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            class_weights: Optional class weights for imbalanced data

        Returns:
            Training history dictionary
        """
        # Set class weights
        if class_weights is not None:
            self.set_class_weights(class_weights)

        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics, train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics, val_loss = self.validate(val_loader)

            # Update tracker
            is_best = self.tracker.update(train_metrics, val_metrics)

            # Update learning rate
            self.scheduler.step(val_metrics[self.config['checkpoint']['monitor']])

            # Print metrics
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val AUC-PR: {val_metrics['auc_pr']:.4f}, Val AUC-ROC: {val_metrics['auc_roc']:.4f}")

            # Save checkpoint
            if is_best:
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"✓ New best model saved (AUC-PR: {val_metrics['auc_pr']:.4f})")

            # Early stopping
            if self.tracker.should_stop_early(self.config['training']['early_stopping_patience']):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        print(f"\nTraining complete. Best epoch: {self.tracker.best_epoch + 1}")
        print(f"Best Val AUC-PR: {self.tracker.best_metric:.4f}")

        return self.tracker.get_history()

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint
