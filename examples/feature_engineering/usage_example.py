"""
Example usage of the Fraud Feature Engineering Pipeline.

This script demonstrates:
1. Creating synthetic transaction data
2. Training the feature engineering pipeline
3. Extracting and analyzing features
4. Serializing and loading the pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
import sys
sys.path.append("..")

from src.pipeline import FraudFeaturePipeline
from src.transformers.velocity_features import VelocityFeatures
from src.transformers.deviation_features import DeviationFeatures
from src.transformers.merchant_features import MerchantRiskFeatures
from src.feature_selection.shap_selector import SHAPSelector


def create_synthetic_data(n_transactions=1000, fraud_rate=0.05, seed=42):
    """
    Create synthetic transaction data for demonstration.

    Parameters
    ----------
    n_transactions : int
        Number of transactions to generate
    fraud_rate : float
        Proportion of fraudulent transactions
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Synthetic transaction data with fraud labels
    """
    np.random.seed(seed)

    # Generate users and merchants
    n_users = 50
    n_merchants = 20

    data = {
        "user_id": np.random.randint(1, n_users + 1, n_transactions),
        "merchant_id": np.random.randint(1, n_merchants + 1, n_transactions),
        "amount": np.random.uniform(10, 1000, n_transactions),
        "timestamp": pd.date_range("2024-01-01", periods=n_transactions, freq="1min"),
    }

    df = pd.DataFrame(data)

    # Add hour of day for deviation features
    df["hour_of_day"] = df["timestamp"].dt.hour

    # Generate fraud labels (with some patterns)
    fraud = np.zeros(n_transactions, dtype=int)

    # Higher fraud for certain merchants
    high_fraud_merchants = [1, 5, 10]
    for merchant in high_fraud_merchants:
        mask = df["merchant_id"] == merchant
        fraud[mask] = np.random.choice([0, 1], size=sum(mask), p=[0.7, 0.3])

    # Higher fraud for high amounts
    high_amount_mask = df["amount"] > 500
    fraud[high_amount_mask] = np.random.choice(
        [0, 1], size=sum(high_amount_mask), p=[0.8, 0.2]
    )

    # Fill remaining with base fraud rate
    remaining = (fraud == 0) & (~high_amount_mask | ~df["merchant_id"].isin(high_fraud_merchants))
    fraud[remaining] = np.random.choice([0, 1], size=sum(remaining), p=[1 - fraud_rate, fraud_rate])

    df["fraud"] = fraud

    return df


def example_basic_usage():
    """Demonstrate basic usage of individual transformers."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage of Individual Transformers")
    print("=" * 60)

    # Create synthetic data
    df = create_synthetic_data(n_transactions=500)
    X = df.drop(columns=["fraud"])
    y = df["fraud"]

    print(f"\nDataset shape: {X.shape}")
    print(f"Fraud rate: {y.mean():.2%}")

    # 1. Velocity Features
    print("\n1. Velocity Features")
    print("-" * 40)
    velocity = VelocityFeatures(
        user_col="user_id",
        datetime_col="timestamp",
        amount_col="amount",
        time_windows=[(1, "h"), (24, "h")],
        features=["count", "sum", "mean"],
    )
    velocity.fit(X)
    X_velocity = velocity.transform(X)
    print(f"Velocity features shape: {X_velocity.shape}")
    print(f"Sample velocity features:\n{X_velocity.head()}")

    # 2. Deviation Features
    print("\n2. Deviation Features")
    print("-" * 40)
    deviation = DeviationFeatures(
        user_col="user_id",
        features=["amount", "hour_of_day"],
        window_size=30,
    )
    deviation.fit(X)
    X_deviation = deviation.transform(X)
    print(f"Deviation features shape: {X_deviation.shape}")
    print(f"Sample deviation features:\n{X_deviation.head()}")

    # 3. Merchant Risk Features
    print("\n3. Merchant Risk Features")
    print("-" * 40)
    merchant = MerchantRiskFeatures(
        merchant_col="merchant_id",
        alpha=1.0,
        beta=1.0,
    )
    merchant.fit(X, y)
    X_merchant = merchant.transform(X)
    print(f"Merchant features shape: {X_merchant.shape}")
    print(f"Sample merchant features:\n{X_merchant.head()}")
    print(f"Global fraud rate: {merchant.global_fraud_rate_:.2%}")


def example_pipeline():
    """Demonstrate complete pipeline usage."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Complete Pipeline")
    print("=" * 60)

    # Create synthetic data
    df = create_synthetic_data(n_transactions=1000)
    X = df.drop(columns=["fraud"])
    y = df["fraud"]

    # Create and fit pipeline
    pipeline = FraudFeaturePipeline(
        user_col="user_id",
        merchant_col="merchant_id",
        datetime_col="timestamp",
        amount_col="amount",
        time_windows=[(1, "h"), (24, "h"), (7, "d")],
        velocity_features=["count", "sum", "mean"],
        deviation_features=["amount", "hour_of_day"],
        merchant_alpha=1.0,
        merchant_beta=1.0,
        use_shap_selection=False,
        scale_features=False,
    )

    # Fit and transform
    X_features = pipeline.fit_transform(X, y)

    print(f"\nOriginal data shape: {X.shape}")
    print(f"Feature matrix shape: {X_features.shape}")
    print(f"Number of features: {X_features.shape[1]}")
    print(f"\nFeature names:\n{pipeline.get_feature_names_out()}")

    print(f"\nSample features:\n{X_features.head()}")


def example_with_shap_selection():
    """Demonstrate pipeline with SHAP feature selection."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Pipeline with SHAP Feature Selection")
    print("=" * 60)

    # Create synthetic data
    df = create_synthetic_data(n_transactions=1000)
    X = df.drop(columns=["fraud"])
    y = df["fraud"]

    # Create pipeline with SHAP selection
    pipeline = FraudFeaturePipeline(
        user_col="user_id",
        merchant_col="merchant_id",
        datetime_col="timestamp",
        amount_col="amount",
        time_windows=[(1, "h"), (24, "h")],
        use_shap_selection=True,
        n_features=15,
    )

    # Fit and transform
    X_features = pipeline.fit_transform(X, y)

    print(f"\nFeature matrix shape after SHAP selection: {X_features.shape}")
    print(f"Number of selected features: {X_features.shape[1]}")
    print(f"\nSelected features:\n{pipeline.get_feature_names_out()}")

    # Get feature importance
    shap_selector = pipeline.pipeline_.named_steps["shap_selection"]
    importance_df = shap_selector.get_feature_importance()
    print(f"\nTop 10 most important features:\n{importance_df.head(10)}")


def example_serialization():
    """Demonstrate pipeline serialization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Pipeline Serialization")
    print("=" * 60)

    # Create and fit pipeline
    df = create_synthetic_data(n_transactions=500)
    X = df.drop(columns=["fraud"])
    y = df["fraud"]

    pipeline = FraudFeaturePipeline(
        user_col="user_id",
        merchant_col="merchant_id",
        datetime_col="timestamp",
        amount_col="amount",
    )

    pipeline.fit(X, y)

    # Save pipeline
    filepath = "fraud_pipeline.pkl"
    pipeline.save(filepath)
    print(f"\nPipeline saved to {filepath}")

    # Load pipeline
    loaded_pipeline = FraudFeaturePipeline.load(filepath)
    print(f"Pipeline loaded from {filepath}")

    # Transform with loaded pipeline
    X_transformed = loaded_pipeline.transform(X)
    print(f"Transformed data shape: {X_transformed.shape}")

    # Clean up
    import os
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"\nCleaned up {filepath}")


def example_inference_with_new_data():
    """Demonstrate inference on new data."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Inference with New Data")
    print("=" * 60)

    # Train pipeline
    train_df = create_synthetic_data(n_transactions=1000, seed=42)
    X_train = train_df.drop(columns=["fraud"])
    y_train = train_df["fraud"]

    pipeline = FraudFeaturePipeline(
        user_col="user_id",
        merchant_col="merchant_id",
        datetime_col="timestamp",
        amount_col="amount",
    )

    pipeline.fit(X_train, y_train)

    # Create new data with unseen users/merchants
    print("\nCreating test data with unseen users and merchants...")
    new_data = pd.DataFrame({
        "user_id": [999, 999, 1000],  # Unseen users
        "merchant_id": [999, 1, 1000],  # Mix of unseen and seen merchants
        "timestamp": [
            pd.Timestamp("2024-02-01 10:00"),
            pd.Timestamp("2024-02-01 10:05"),
            pd.Timestamp("2024-02-01 10:10"),
        ],
        "amount": [150.0, 200.0, 300.0],
    })

    print("\nNew data:")
    print(new_data)

    # Transform new data
    X_new = pipeline.transform(new_data)

    print("\nExtracted features:")
    print(X_new)

    print("\nNote: Unseen users/merchants handled gracefully!")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("FRAUD FEATURE ENGINEERING PIPELINE - EXAMPLES")
    print("=" * 60)

    example_basic_usage()
    example_pipeline()
    example_with_shap_selection()
    example_serialization()
    example_inference_with_new_data()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
