"""
Experiment orchestration for imbalanced classification benchmark.

Coordinates the evaluation of all 6 techniques across stratified K-fold CV.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.models.classical.legacy.config import config
from src.models.classical.legacy.cross_validation import stratified_cross_validation
from src.models.classical.legacy.models.baseline import LogisticRegressionBaseline, RandomForestBaseline
from src.models.classical.legacy.models.xgboost_wrapper import XGBoostWrapper
from src.models.classical.legacy.models.focal_loss import FocalLossClassifier
from src.models.classical.legacy.techniques.undersampling import apply_random_undersampling
from src.models.classical.legacy.techniques.smote import apply_smote
from src.models.classical.legacy.techniques.adasyn import apply_adasyn


@dataclass
class ExperimentResult:
    """Result object for a single technique evaluation."""

    technique: str
    metrics: Dict[str, float] = field(default_factory=dict)  # Mean metrics across folds
    fold_results: Dict[str, List[float]] = field(default_factory=dict)  # All fold results
    std_metrics: Dict[str, float] = field(default_factory=dict)  # Std across folds

    def __str__(self) -> str:
        """String representation of results."""
        lines = [f"\n=== {self.technique} ==="]
        lines.append("Mean Metrics (5-fold CV):")
        for metric, value in self.metrics.items():
            std = self.std_metrics.get(metric, 0.0)
            lines.append(f"  {metric}: {value:.4f} (+/- {std:.4f})")
        return "\n".join(lines)


class ExperimentRunner:
    """
    Orchestrates the benchmark experiments.

    Evaluates 6 techniques:
    1. Baseline (no imbalance handling)
    2. Random Undersampling
    3. SMOTE
    4. ADASYN
    5. Class Weighting
    6. Focal Loss
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize experiment runner.

        Args:
            X: Feature matrix
            y: Target labels
        """
        self.X = X
        self.y = y
        self.results: List[ExperimentResult] = []

    def run_all_experiments(self) -> List[ExperimentResult]:
        """
        Run experiments for all 6 techniques.

        Returns:
            List of experiment results
        """
        print("Starting imbalanced classification benchmark...")
        print(f"Dataset: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"Fraud rate: {self.y.mean():.4f} ({self.y.mean()*100:.2f}%)")
        print(f"Cross-validation: {config.N_FOLDS}-fold stratified\n")

        # Technique 1: Baseline (Logistic Regression)
        print("[1/6] Running Baseline (Logistic Regression)...")
        self._run_baseline()

        # Technique 2: Random Undersampling
        print("[2/6] Running Random Undersampling...")
        self._run_random_undersampling()

        # Technique 3: SMOTE
        print("[3/6] Running SMOTE...")
        self._run_smote()

        # Technique 4: ADASYN
        print("[4/6] Running ADASYN...")
        self._run_adasyn()

        # Technique 5: Class Weighting
        print("[5/6] Running Class Weighting...")
        self._run_class_weight()

        # Technique 6: Focal Loss
        print("[6/6] Running Focal Loss...")
        self._run_focal_loss()

        print("\n✓ All experiments completed!")
        return self.results

    def _run_baseline(self):
        """Run baseline experiment (no imbalance handling)."""
        estimator = LogisticRegressionBaseline(class_weight=None)
        fold_results = stratified_cross_validation(
            self.X, self.y, estimator, "baseline"
        )
        self.results.append(self._aggregate_results("baseline", fold_results))

    def _run_random_undersampling(self):
        """Run random undersampling experiment."""
        estimator = LogisticRegressionBaseline(class_weight=None)
        fold_results = stratified_cross_validation(
            self.X,
            self.y,
            estimator,
            "random_undersampling",
            apply_resampling=apply_random_undersampling,
        )
        self.results.append(
            self._aggregate_results("random_undersampling", fold_results)
        )

    def _run_smote(self):
        """Run SMOTE experiment."""
        estimator = LogisticRegressionBaseline(class_weight=None)
        fold_results = stratified_cross_validation(
            self.X,
            self.y,
            estimator,
            "smote",
            apply_resampling=apply_smote,
        )
        self.results.append(self._aggregate_results("smote", fold_results))

    def _run_adasyn(self):
        """Run ADASYN experiment."""
        estimator = LogisticRegressionBaseline(class_weight=None)
        fold_results = stratified_cross_validation(
            self.X,
            self.y,
            estimator,
            "adasyn",
            apply_resampling=apply_adasyn,
        )
        self.results.append(self._aggregate_results("adasyn", fold_results))

    def _run_class_weight(self):
        """Run class weighting experiment."""
        estimator = LogisticRegressionBaseline(
            class_weight=config.CLASS_WEIGHT
        )
        fold_results = stratified_cross_validation(
            self.X, self.y, estimator, "class_weight"
        )
        self.results.append(self._aggregate_results("class_weight", fold_results))

    def _run_focal_loss(self):
        """Run Focal Loss experiment."""
        estimator = FocalLossClassifier(
            input_dim=self.X.shape[1],
            alpha=config.FOCAL_ALPHA,
            gamma=config.FOCAL_GAMMA,
            epochs=100,
        )
        fold_results = stratified_cross_validation(
            self.X, self.y, estimator, "focal_loss"
        )
        self.results.append(self._aggregate_results("focal_loss", fold_results))

    def _aggregate_results(
        self, technique: str, fold_results: Dict[str, List[float]]
    ) -> ExperimentResult:
        """
        Aggregate fold results into mean and std metrics.

        Args:
            technique: Name of the technique
            fold_results: Dictionary of metric lists across folds

        Returns:
            ExperimentResult with aggregated metrics
        """
        mean_metrics = {}
        std_metrics = {}

        for metric_name, values in fold_results.items():
            mean_metrics[metric_name] = float(np.mean(values))
            std_metrics[metric_name] = float(np.std(values))

        return ExperimentResult(
            technique=technique,
            metrics=mean_metrics,
            fold_results=fold_results,
            std_metrics=std_metrics,
        )

    def format_results_table(self) -> pd.DataFrame:
        """
        Format results as a pandas DataFrame.

        Returns:
            DataFrame with techniques as rows and metrics as columns
        """
        data = []
        for result in self.results:
            row = {"Technique": result.technique}
            for metric, value in result.metrics.items():
                std = result.std_metrics[metric]
                row[metric] = f"{value:.4f} ± {std:.4f}"
            data.append(row)

        df = pd.DataFrame(data)
        df = df.set_index("Technique")

        # Reorder columns for better presentation
        metric_order = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auprc",
            "auroc",
            f"recall_at_fpr_{int(config.FPR_THRESHOLD * 100)}pct",
        ]
        df = df[[col for col in metric_order if col in df.columns]]

        return df

    def save_results(self, filepath: Optional[Path] = None):
        """
        Save results to CSV.

        Args:
            filepath: Path to save results (default: results/results.csv)
        """
        if filepath is None:
            filepath = config.RESULTS_DIR / "results.csv"

        df = self.format_results_table()
        df.to_csv(filepath)
        print(f"\nResults saved to {filepath}")

    def print_results(self):
        """Print all experiment results."""
        print("\n" + "=" * 70)
        print("EXPERIMENT RESULTS")
        print("=" * 70)

        df = self.format_results_table()
        print(df.to_string())

        print("\n" + "=" * 70)

        # Print individual technique details
        for result in self.results:
            print(result)


def run_all_experiments(X: np.ndarray, y: np.ndarray) -> List[ExperimentResult]:
    """
    Convenience function to run all experiments.

    Args:
        X: Feature matrix
        y: Target labels

    Returns:
        List of experiment results
    """
    runner = ExperimentRunner(X, y)
    return runner.run_all_experiments()
