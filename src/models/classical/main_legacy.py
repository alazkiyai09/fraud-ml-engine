"""
Main entry point for the Imbalanced Classification Benchmark.

Runs all 6 techniques and generates publication-quality visualizations.
"""

import argparse
from pathlib import Path
import numpy as np
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.classical.legacy.config import config
from src.models.classical.legacy.data_loader import load_or_generate_data
from src.models.classical.legacy.experiment import ExperimentRunner
from src.models.classical.legacy.visualization import create_all_visualizations


def main(
    data_path: Path = None,
    synthetic: bool = True,
    n_samples: int = 100000,
    fraud_rate: float = 0.0017,
):
    """
    Run the complete imbalanced classification benchmark.

    Args:
        data_path: Path to fraud detection dataset (optional)
        synthetic: Whether to use synthetic data if no data file provided
        n_samples: Number of samples for synthetic data
        fraud_rate: Fraud rate for synthetic data
    """
    print("=" * 70)
    print("IMBALANCED CLASSIFICATION BENCHMARK")
    print("=" * 70)
    print("\nTechniques to evaluate:")
    print("  1. Baseline (Logistic Regression)")
    print("  2. Random Undersampling")
    print("  3. SMOTE")
    print("  4. ADASYN")
    print("  5. Class Weighting")
    print("  6. Focal Loss")
    print("\n" + "=" * 70)

    # Load data
    if synthetic and data_path is None:
        X, y = load_or_generate_data(
            filepath=data_path,
            generate_if_missing=True,
        )
    else:
        if data_path is None or not data_path.exists():
            print(f"\nError: Data file not found at {data_path}")
            print("Please provide a valid data file or use --synthetic flag")
            return

        X, y = load_or_generate_data(filepath=data_path, generate_if_missing=False)

    print("\n" + "-" * 70)

    # Run experiments
    runner = ExperimentRunner(X, y)
    results = runner.run_all_experiments()

    # Print results
    runner.print_results()

    # Save results
    runner.save_results()

    # Create visualizations
    create_all_visualizations(results)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print(f"  - results.csv: Tabulated results")
    print(f"  - metrics_comparison.png: Metrics bar chart")
    print(f"  - recall_at_fpr.png: Recall@FPR comparison")
    print(f"  - metrics_heatmap.png: Metrics heatmap")
    print(f"  - ranking.png: F1 and AUPRC ranking")
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Imbalanced Classification Benchmark for Fraud Detection"
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to fraud detection dataset (CSV format)",
    )

    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=True,
        help="Use synthetic data if no data file provided (default: True)",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=100000,
        help="Number of samples for synthetic data (default: 100000)",
    )

    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.0017,
        help="Fraud rate for synthetic data (default: 0.0017 = 0.17%%)",
    )

    args = parser.parse_args()

    main(
        data_path=args.data,
        synthetic=args.synthetic,
        n_samples=args.n_samples,
        fraud_rate=args.fraud_rate,
    )
