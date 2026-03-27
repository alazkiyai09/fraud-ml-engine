"""Main training script for anomaly detection benchmark."""

import argparse
import yaml
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd

# Import models
from src.models.anomaly.legacy.models import (
    IsolationForestDetector,
    OneClassSVMDetector,
    LOFDetector,
    AutoencoderDetector
)

# Import ensemble methods
from src.models.anomaly.legacy.ensemble import voting_ensemble, stacking_ensemble

# Import evaluation metrics
from src.models.anomaly.legacy.evaluation import (
    optimize_threshold,
    compute_all_metrics,
    plot_roc_curve,
    plot_precision_recall_curve
)

# Import preprocessing
from src.models.anomaly.legacy.preprocessing import load_and_split_data, save_splits

# Import failure analysis
from src.models.anomaly.legacy.evaluation.failure_analysis import (
    analyze_failures,
    export_failure_cases,
    compare_model_failures
)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def tune_contamination_param(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    contamination_range: List[float],
    target_fpr: float = 0.01
) -> float:
    """
    Find optimal contamination parameter for a model.

    Args:
        model: Unfitted anomaly detector.
        X_val: Validation features.
        y_val: Validation labels.
        contamination_range: List of contamination values to try.
        target_fpr: Target false positive rate.

    Returns:
        Best contamination value.
    """
    best_contamination = contamination_range[0]
    best_distance = float('inf')

    for contamination in contamination_range:
        # Clone model and fit with current contamination
        from copy import deepcopy
        model_copy = deepcopy(model)
        model_copy.contamination = contamination

        try:
            model_copy.fit(X_val[y_val == 0])  # Train on class 0 only
            scores = model_copy.predict_anomaly_score(X_val)

            # Calculate achieved FPR
            threshold = np.quantile(scores[y_val == 0], 1 - target_fpr)
            predictions = (scores >= threshold).astype(int)
            fpr = np.mean(predictions[y_val == 0])

            # Find contamination closest to target FPR
            distance = abs(fpr - target_fpr)
            if distance < best_distance:
                best_distance = distance
                best_contamination = contamination

        except Exception as e:
            print(f"  Error with contamination={contamination}: {e}")
            continue

    return best_contamination


def run_single_model(
    model_name: str,
    model,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict
) -> Dict:
    """
    Train and evaluate a single model.

    Args:
        model_name: Name of the model.
        model: Model instance.
        X_train: Training features (class 0 only).
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Test features.
        y_test: Test labels.
        config: Configuration dict.

    Returns:
        Results dictionary.
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    results = {
        'model': model_name,
        'training_time': 0,
        'prediction_time': 0
    }

    # Train on class 0 only
    start_time = time.time()
    model.fit(X_train)
    results['training_time'] = time.time() - start_time
    print(f"Training time: {results['training_time']:.2f}s")

    # Generate anomaly scores
    start_time = time.time()
    scores_train = model.predict_anomaly_score(X_train)
    scores_val = model.predict_anomaly_score(X_val)
    scores_test = model.predict_anomaly_score(X_test)
    results['prediction_time'] = time.time() - start_time
    print(f"Prediction time: {results['prediction_time']:.2f}s")

    # Optimize threshold on validation set
    target_fpr = config['evaluation']['target_fpr']
    threshold = optimize_threshold(y_val, scores_val, target_fpr=target_fpr)
    print(f"Optimal threshold: {threshold:.4f}")

    # Compute metrics on test set
    metrics = compute_all_metrics(y_test, scores_test, threshold)
    results.update(metrics)

    # Print key metrics
    print(f"\nTest Set Results (target FPR: {target_fpr*100}%):")
    print(f"  Detection Rate (Recall): {metrics['detection_rate']*100:.2f}%")
    print(f"  False Positive Rate: {metrics['false_positive_rate']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR: {metrics['auc_pr']:.4f}")

    # Store predictions and scores
    results['predictions'] = (scores_test >= threshold).astype(int)
    results['scores'] = scores_test

    return results


def run_ensemble(
    model_results: List[Dict],
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict
) -> List[Dict]:
    """
    Train and evaluate ensemble methods.

    Args:
        model_results: List of results from individual models.
        X_test: Test features.
        y_test: Test labels.
        config: Configuration dict.

    Returns:
        List of ensemble results.
    """
    print(f"\n{'='*60}")
    print("Training Ensemble Methods")
    print(f"{'='*60}")

    ensemble_results = []
    target_fpr = config['evaluation']['target_fpr']

    # Voting ensemble (average)
    print("\nVoting Ensemble (Average):")
    scores_list = [result['scores'] for result in model_results]
    voting_scores = voting_ensemble(scores_list, method='average')

    # Optimize threshold
    threshold = np.quantile(voting_scores, 1 - target_fpr)
    metrics = compute_all_metrics(y_test, voting_scores, threshold)

    print(f"  Detection Rate: {metrics['detection_rate']*100:.2f}%")
    print(f"  FPR: {metrics['false_positive_rate']*100:.2f}%")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")

    ensemble_results.append({
        'model': 'Voting (Average)',
        **metrics,
        'predictions': (voting_scores >= threshold).astype(int),
        'scores': voting_scores
    })

    return ensemble_results


def save_results(
    all_results: List[Dict],
    results_dir: str,
    y_test: np.ndarray,
    feature_names: List[str],
    config: Dict = None
) -> None:
    """
    Save results to disk.

    Args:
        all_results: List of result dictionaries.
        results_dir: Directory to save results.
        y_test: Test labels.
        feature_names: Feature names.
        config: Configuration dictionary (for data paths).
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save metrics summary
    metrics_df = pd.DataFrame([
        {k: v for k, v in result.items()
         if k not in ['predictions', 'scores']}
        for result in all_results
    ])
    metrics_df.to_csv(results_path / f"metrics_{timestamp}.csv", index=False)
    print(f"\nMetrics saved to {results_path / f'metrics_{timestamp}.csv'}")

    # Save predictions and scores
    for result in all_results:
        model_name = result['model'].replace(' ', '_').replace('(', '').replace(')', '')
        np.save(results_path / f"scores_{model_name}_{timestamp}.npy", result['scores'])
        np.save(results_path / f"predictions_{model_name}_{timestamp}.npy", result['predictions'])

    # Plot ROC curves
    scores_dict = {result['model']: result['scores'] for result in all_results}
    plot_roc_curve(
        y_test,
        scores_dict,
        save_path=str(results_path / f"roc_curve_{timestamp}.png")
    )

    # Plot PR curves
    plot_precision_recall_curve(
        y_test,
        scores_dict,
        save_path=str(results_path / f"pr_curve_{timestamp}.png")
    )

    # Save failure analysis (skip if config not provided)
    if config is not None and 'data' in config:
        predictions_dict = {
            result['model']: result['predictions']
            for result in all_results
        }
        try:
            x_test_path = Path(config['data']['processed_path']).parent / "raw" / "X_test.npy"
            comparison = compare_model_failures(
                np.load(x_test_path),
                y_test,
                predictions_dict,
                scores_dict,
                feature_names
            )
            comparison.to_csv(results_path / f"failure_comparison_{timestamp}.csv", index=False)
        except (FileNotFoundError, KeyError) as e:
            print(f"Skipping failure analysis: {e}")

    print(f"All results saved to {results_dir}")


def main(args):
    """Main training pipeline."""
    # Load configuration
    config = load_config(args.config)
    print("Configuration loaded successfully.")

    # Load and prepare data
    if args.use_preprocessed:
        print("\nLoading preprocessed data...")
        X_train, X_val, X_test, y_val, y_test, feature_names = load_and_split_data(
            args.data,
            scaler_type="standard"
        )
    else:
        print("\nLoading and preprocessing data...")
        X_train, X_val, X_test, y_val, y_test, feature_names = load_and_split_data(
            args.data,
            scaler_type="standard"
        )

        # Save preprocessed splits
        save_splits(
            X_train, X_val, X_test, y_val, y_test,
            config['data']['processed_path']
        )

    print(f"\nData shape: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"Feature count: {len(feature_names)}")

    # Initialize models
    models = {
        'Isolation Forest': IsolationForestDetector(
            contamination=config['models']['isolation_forest']['contamination'],
            n_estimators=config['models']['isolation_forest']['n_estimators'],
            random_state=config['data']['random_state']
        ),
        'One-Class SVM': OneClassSVMDetector(
            nu=config['models']['one_class_svm']['nu'],
            kernel=config['models']['one_class_svm']['kernel'],
            gamma=config['models']['one_class_svm']['gamma']
        ),
        'LOF': LOFDetector(
            contamination=config['models']['lof']['contamination'],
            n_neighbors=config['models']['lof']['n_neighbors'],
            algorithm=config['models']['lof']['algorithm'],
            metric=config['models']['lof']['metric']
        ),
        'Autoencoder': AutoencoderDetector(
            input_dim=X_train.shape[1],
            hidden_dims=config['models']['autoencoder']['architecture']['hidden_dims'],
            latent_dim=config['models']['autoencoder']['architecture']['latent_dim'],
            device=config['models']['autoencoder']['training']['device']
        )
    }

    # Run benchmark for each model
    all_results = []
    for model_name, model in models.items():
        result = run_single_model(
            model_name,
            model,
            X_train,
            X_val,
            y_val,
            X_test,
            y_test,
            config
        )
        all_results.append(result)

    # Run ensemble methods
    ensemble_results = run_ensemble(
        all_results[:4],  # Individual models only
        X_test,
        y_test,
        config
    )
    all_results.extend(ensemble_results)

    # Save results
    save_results(
        all_results,
        args.results_dir,
        y_test,
        feature_names,
        config
    )

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    summary_df = pd.DataFrame([
        {
            'Model': r['model'],
            'Detection Rate': f"{r['detection_rate']*100:.2f}%",
            'FPR': f"{r['false_positive_rate']*100:.2f}%",
            'F1': f"{r['f1']:.4f}",
            'AUC-ROC': f"{r['auc_roc']:.4f}",
            'AUC-PR': f"{r['auc_pr']:.4f}"
        }
        for r in all_results
    ])
    print(summary_df.to_string(index=False))

    print(f"\n{'='*60}")
    print("Benchmark completed successfully!")
    print(f"Results saved to: {args.results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly Detection Benchmark")
    parser.add_argument(
        '--data',
        type=str,
        default='data/raw/creditcard.csv',
        help='Path to data file (CSV format)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--use-preprocessed',
        action='store_true',
        help='Use preprocessed data if available'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='data/results',
        help='Directory to save results'
    )

    args = parser.parse_args()
    main(args)
