# Fraud Feature Engineering Pipeline

A sklearn-compatible feature engineering pipeline specifically designed for fraud detection. Combines domain expertise from financial fraud detection with modern ML practices.

## Features

- **Velocity Features**: Transaction frequency and amounts over time windows (e.g., transactions per hour)
- **Deviation Features**: Z-scores and ratios comparing current behavior to user historical patterns
- **Merchant Risk Features**: Bayesian-smoothed merchant fraud rates to handle low-volume merchants
- **SHAP-Based Feature Selection**: Explainable feature selection using SHAP values
- **Joblib Serialization**: Save/load trained pipelines for production deployment

## Installation

```bash
git clone <repository-url>
cd fraud_feature_engineering
pip install -r requirements.txt
```

### Requirements

- Python >= 3.9
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- shap >= 0.42.0
- joblib >= 1.3.0

## Quick Start

```python
import pandas as pd
from src.pipeline import FraudFeaturePipeline

# Prepare your data
# Required columns: user_id, merchant_id, timestamp, amount
X = df[["user_id", "merchant_id", "timestamp", "amount"]]
y = df["fraud"]  # Binary fraud labels

# Create and fit pipeline
pipeline = FraudFeaturePipeline(
    user_col="user_id",
    merchant_col="merchant_id",
    datetime_col="timestamp",
    amount_col="amount",
    time_windows=[(1, "h"), (24, "h"), (7, "d")],
)

# Extract features
X_features = pipeline.fit_transform(X, y)

# Save for production
pipeline.save("fraud_pipeline.pkl")

# Load for inference
loaded_pipeline = FraudFeaturePipeline.load("fraud_pipeline.pkl")
X_new_features = loaded_pipeline.transform(X_new)
```

## Transformers

### 1. VelocityFeatures

Computes transaction velocity over rolling time windows.

**Parameters:**
- `user_col`: User identifier column
- `datetime_col`: Timestamp column
- `amount_col`: Transaction amount column
- `time_windows`: List of (size, unit) tuples, e.g., `[(1, 'h'), (24, 'h'), (7, 'd')]`
- `features`: Which features to compute: `['count', 'sum', 'mean', 'time_since_last']`

**Generated Features:**
- `velocity_count_{window}`: Number of transactions in window
- `velocity_sum_{window}`: Total amount in window
- `velocity_mean_{window}`: Average amount in window
- `velocity_time_since_last_s`: Seconds since user's last transaction

**Example:**
```python
from src.transformers.velocity_features import VelocityFeatures

velocity = VelocityFeatures(
    user_col="user_id",
    datetime_col="timestamp",
    amount_col="amount",
    time_windows=[(1, "h"), (24, "h")],
)

X_velocity = velocity.fit_transform(X)
```

### 2. DeviationFeatures

Compares current behavior to user's historical patterns.

**Parameters:**
- `user_col`: User identifier column
- `features`: Columns to compute deviations for (e.g., `['amount', 'hour_of_day']`)
- `window_size`: Number of historical transactions to use (default: 30)

**Generated Features:**
- `deviation_{feature}_zscore`: Z-score from user's historical mean
- `deviation_{feature}_ratio`: Ratio to user's historical mean

**Example:**
```python
from src.transformers.deviation_features import DeviationFeatures

deviation = DeviationFeatures(
    user_col="user_id",
    features=["amount", "hour_of_day"],
    window_size=30,
)

X_deviation = deviation.fit_transform(X)
```

### 3. MerchantRiskFeatures

Computes merchant-level fraud rates with Bayesian smoothing.

**Parameters:**
- `merchant_col`: Merchant identifier column
- `alpha`: Beta distribution prior parameter (pseudo-fraud counts)
- `beta`: Beta distribution prior parameter (pseudo-legitimate counts)
- `global_rate_weight`: Weight for global fraud rate (0-1)

**Generated Features:**
- `merchant_fraud_rate`: Smoothed fraud rate
- `merchant_fraud_count`: Number of fraud transactions at merchant
- `merchant_total_count`: Total transactions at merchant

**Example:**
```python
from src.transformers.merchant_features import MerchantRiskFeatures

merchant = MerchantRiskFeatures(
    merchant_col="merchant_id",
    alpha=1.0,
    beta=1.0,
)

X_merchant = merchant.fit_transform(X, y)  # y required for fraud labels
```

### 4. SHAPSelector

Feature selection using SHAP values from tree-based models.

**Parameters:**
- `estimator`: Model to use for SHAP (default: RandomForestClassifier)
- `n_features`: Number of top features to select
- `threshold`: Minimum SHAP value threshold (optional)

**Example:**
```python
from src.feature_selection.shap_selector import SHAPSelector

selector = SHAPSelector(n_features=20)
X_selected = selector.fit_transform(X, y)

# Get feature importance
importance_df = selector.get_feature_importance()
print(importance_df.head(10))
```

## Feature Importance Analysis

### Understanding Feature Importances

The pipeline integrates SHAP (SHapley Additive exPlanations) for model-agnostic feature importance. This helps:

1. **Interpretability**: Understand which features drive fraud predictions
2. **Feature Selection**: Select the most predictive features
3. **Compliance**: Explain model decisions to auditors/regulators

### Running Feature Importance Analysis

```python
from src.pipeline import FraudFeaturePipeline

# Train pipeline with SHAP selection
pipeline = FraudFeaturePipeline(
    user_col="user_id",
    merchant_col="merchant_id",
    datetime_col="timestamp",
    amount_col="amount",
    use_shap_selection=True,
    n_features=20,
)

pipeline.fit(X, y)

# Get feature importance
shap_selector = pipeline.pipeline_.named_steps["shap_selection"]
importance_df = shap_selector.get_feature_importance()

# Display top features
print(importance_df.head(15))
```

### Example Output

```
               feature  importance
0    velocity_count_1h     0.245
1    deviation_amount_zscore   0.198
2    merchant_fraud_rate    0.167
3    velocity_sum_24h      0.134
4    velocity_mean_1h      0.112
5    deviation_amount_ratio    0.098
6    merchant_fraud_count   0.087
7    velocity_count_24h    0.076
8    velocity_time_since_last_s  0.065
9    merchant_total_count  0.054
```

### Visualizing SHAP Values

```python
import shap

# Get SHAP values
shap_values = shap_selector.shap_values_
feature_names = shap_selector.feature_names_in_

# Create summary plot
shap.summary_plot(shap_values, X, feature_names=feature_names)

# Create waterfall plot for single prediction
explainer = shap.TreeExplainer(shap_selector.estimator_)
shap.waterfall_plot(explainer(X.iloc[0]))
```

## Production Considerations

### Handling Unseen Categories

All transformers handle unseen users/merchants gracefully:

- **VelocityFeatures**: Uses 0 for new users (no history)
- **DeviationFeatures**: Falls back to global statistics
- **MerchantRiskFeatures**: Uses global fraud rate for unseen merchants

### Preventing Data Leakage

- Velocity features use **strict time-based windows** (no future data)
- Deviation features are computed **per-user** using only historical data
- No features use target information (except merchant fraud rate, which requires `y` in `fit()`)

### Serialization

```python
# Save pipeline
pipeline.save("models/fraud_pipeline_v1.pkl")

# Load pipeline
pipeline = FraudFeaturePipeline.load("models/fraud_pipeline_v1.pkl")

# Make predictions
X_features = pipeline.transform(X_new)
```

## Testing

Run the test suite:

```bash
# Run pytest
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run unittest
python -m unittest discover tests/ -p "test_*_unittest.py"
```

### Test Coverage

- `test_velocity_features.py`: Tests for velocity feature computation
- `test_deviation_features.py`: Tests for deviation feature computation
- `test_merchant_features.py`: Tests for merchant risk features and Bayesian smoothing
- `test_pipeline.py`: Tests for complete pipeline

## Examples

See `examples/usage_example.py` for comprehensive examples:

```bash
cd examples
python usage_example.py
```

Examples include:
1. Basic usage of individual transformers
2. Complete pipeline with all features
3. SHAP-based feature selection
4. Pipeline serialization
5. Inference with new data (unseen users/merchants)

## Architecture

```
fraud_feature_engineering/
├── src/
│   ├── transformers/
│   │   ├── base.py                  # Shared utilities
│   │   ├── velocity_features.py     # Velocity computation
│   │   ├── deviation_features.py    # User deviation
│   │   └── merchant_features.py     # Merchant risk
│   ├── feature_selection/
│   │   └── shap_selector.py         # SHAP selection
│   └── pipeline.py                  # Complete pipeline
├── tests/                           # pytest and unittest
├── examples/
│   └── usage_example.py             # Demonstration
├── requirements.txt
└── README.md
```

## Design Philosophy

1. **sklearn-compatible**: All transformers inherit from `BaseEstimator` and `TransformerMixin`
2. **No data leakage**: Time-based features prevent look-ahead bias
3. **Production-ready**: Handles unseen categories, serializable with joblib
4. **Explainable**: SHAP integration for feature importance
5. **Domain-aware**: Based on real-world fraud detection patterns

## Performance Tips

1. **Start simple**: Begin with velocity features, add deviation/merchant as needed
2. **Tune time windows**: Different fraud patterns have different time scales
3. **Feature selection**: Use SHAP to reduce dimensionality before modeling
4. **Batch processing**: Pipeline is efficient for large datasets
5. **Monitor feature drift**: Re-compute statistics periodically

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please ensure:
- All tests pass (`pytest tests/`)
- New features include tests
- Code follows PEP 8 style guidelines

## Citation

If you use this pipeline in your work, please cite:

```bibtex
@software{fraud_feature_engineering,
  title={Fraud Feature Engineering Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/fraud-feature-engineering}
}
```

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.
