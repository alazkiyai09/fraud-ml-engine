# Anomaly Detection Benchmark

Unsupervised fraud detection benchmark comparing multiple anomaly detection methods trained exclusively on legitimate transactions.

## Overview

This project implements and compares four unsupervised anomaly detection methods for detecting novel fraud patterns without using labeled fraud data during training. The goal is to detect zero-day fraud attacks by learning the patterns of legitimate transactions only.

**Key Principle**: All models are trained **ONLY on Class=0 (legitimate) data**, simulating real-world scenarios where fraud patterns are unknown or constantly evolving.

## Methods Compared

### 1. Isolation Forest
- **Approach**: Isolates anomalies by randomly selecting features and split values
- **Strengths**: Fast, scalable, works well with high-dimensional data
- **Weaknesses**: May struggle with local anomalies

### 2. One-Class SVM
- **Approach**: Learns a decision boundary around normal data in kernel space
- **Strengths**: Flexible with various kernels, good for complex distributions
- **Weaknesses**: Sensitive to hyperparameters, computationally intensive

### 3. Local Outlier Factor (LOF)
- **Approach**: Compares local density of each point to its neighbors
- **Strengths**: Effective at detecting local anomalies
- **Weaknesses**: Computationally expensive for large datasets

### 4. Autoencoder (PyTorch)
- **Approach**: Neural network that learns to reconstruct normal data; high reconstruction error indicates anomalies
- **Strengths**: Can capture complex non-linear patterns
- **Weaknesses**: Requires more training time, needs careful architecture design

### Ensemble Methods
- **Voting**: Combines scores via averaging or majority voting
- **Stacking**: Uses meta-learner to optimally combine base model predictions

## Project Structure

```
anomaly_detection_benchmark/
├── data/
│   ├── raw/                    # Original data files
│   ├── processed/              # Preprocessed train/val/test splits
│   └── results/                # Experiment results and plots
├── src/
│   ├── models/
│   │   ├── base.py             # Abstract base class
│   │   ├── isolation_forest.py
│   │   ├── one_class_svm.py
│   │   ├── autoencoder.py
│   │   └── lof.py
│   ├── ensemble/
│   │   ├── voting.py
│   │   └── stacking.py
│   ├── evaluation/
│   │   ├── metrics.py          # DR, FPR, AUC-ROC, AUC-PR
│   │   └── failure_analysis.py # Failure case analysis
│   ├── preprocessing.py        # Data loading and splitting
│   └── train.py                # Main training script
├── tests/
│   ├── test_scoring.py         # Unit tests for scoring
│   └── test_models.py          # Unit tests for models
├── config.yaml                 # Configuration file
└── README.md
```

## Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate anomaly_detection

# Or install with pip
pip install -r requirements.txt
```

## Requirements

- Python 3.9+
- scikit-learn
- PyTorch
- numpy
- pandas
- matplotlib
- pyyaml
- pytest

## Usage

### 1. Prepare Data

Place your fraud detection dataset in `data/raw/`. The data should be a CSV file with:
- Features columns
- A binary label column (default name: `class`, where 0 = legitimate, 1 = fraud)

### 2. Configure Experiment

Edit `config.yaml` to set hyperparameters:

```yaml
models:
  isolation_forest:
    contamination: 0.1
    n_estimators: 100

  autoencoder:
    architecture:
      hidden_dims: [64, 32]
      latent_dim: 16
    training:
      epochs: 100
      batch_size: 256

evaluation:
  target_fpr: 0.01  # 1% false positive rate
  contamination_range: [0.01, 0.05, 0.1, 0.15, 0.2]
```

### 3. Run Benchmark

```bash
# Using raw data
python src/train.py --data data/raw/creditcard.csv

# Using preprocessed data (faster)
python src/train.py --use-processed --results-dir data/results
```

### 4. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_scoring.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Key Features

### Training on Legitimate Data Only

All models are trained exclusively on Class=0 (legitimate) transactions:

```python
# Training data contains only class 0
X_train = load_data(class=0)  # ~280k samples

# Validation/test contain both classes
X_val, y_val = load_data(mixed_classes)
X_test, y_test = load_data(mixed_classes)
```

### Contamination Parameter Tuning

Models are tuned to achieve a target False Positive Rate (FPR) on validation data:

```python
best_contamination = tune_contamination_param(
    model, X_val, y_val,
    contamination_range=[0.01, 0.05, 0.1, 0.15, 0.2],
    target_fpr=0.01  # 1% FPR
)
```

### Comprehensive Metrics

Each model is evaluated on:
- **Detection Rate (Recall)**: % of fraud correctly identified
- **False Positive Rate**: % of legitimate transactions flagged as fraud
- **Precision**: % of flagged transactions that are actually fraud
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve

### Failure Case Analysis

Analyze which fraud cases are missed and which legitimate cases are falsely flagged:

```python
from src.evaluation.failure_analysis import analyze_failures

failures = analyze_failures(
    X_test, y_test, predictions, anomaly_scores, feature_names
)

# View false positives vs false negatives
print(failures['prediction_type'].value_counts())

# Export for detailed analysis
failures.to_csv('data/results/failure_cases.csv')
```

## Results Summary

After running the benchmark, results are saved to `data/results/`:

- `results_TIMESTAMP.csv`: Detailed metrics for each model
- `roc_curve.png`: ROC curves comparison
- `pr_curve.png`: Precision-Recall curves comparison

### Example Results

| Model | Detection Rate | FPR | Precision | F1 | AUC-ROC | AUC-PR |
|-------|---------------|-----|-----------|-----|---------|--------|
| Autoencoder | 0.85 | 0.01 | 0.72 | 0.78 | 0.95 | 0.82 |
| Isolation Forest | 0.78 | 0.01 | 0.65 | 0.71 | 0.92 | 0.75 |
| Stacking Ensemble | 0.88 | 0.01 | 0.76 | 0.82 | 0.96 | 0.85 |

## Method Recommendations

Based on benchmark results:

### Best Overall: Stacking Ensemble
- **When to use**: Production systems requiring highest accuracy
- **Pros**: Combines strengths of all methods, robust performance
- **Cons**: More complex, slower training

### Best for Real-Time: Isolation Forest
- **When to use**: High-volume transaction processing
- **Pros**: Fast prediction, scalable, minimal tuning
- **Cons**: May miss complex fraud patterns

### Best for Novel Patterns: Autoencoder
- **When to use**: Detecting zero-day attacks, complex non-linear patterns
- **Pros**: Learns intricate patterns, good with high-dimensional data
- **Cons**: Longer training, requires GPU for large datasets

### Best for Local Anomalies: LOF
- **When to use**: Detecting context-dependent fraud
- **Pros**: Effective at local anomalies
- **Cons**: Computationally expensive at scale

## Model Interpretation

### Anomaly Scores

All models output anomaly scores where:
- **Higher score** = More likely to be fraud
- **Threshold** is set to achieve target FPR (e.g., 1%)

```python
# Get anomaly scores
scores = model.predict_anomaly_score(X_new)

# Apply threshold
predictions = (scores >= threshold).astype(int)  # 1=fraud, 0=legitimate
```

### Threshold Selection

Threshold is critical and should be set based on business requirements:
- **Low threshold (0.5% FPR)**: Fewer false alarms, but may miss fraud
- **Medium threshold (1% FPR)**: Balanced approach (default)
- **High threshold (5% FPR)**: Catch more fraud, but more false alarms

## Failure Case Analysis

Common patterns identified:

### False Positives (Legitimate Flagged as Fraud)
- High-value transactions
- Transactions from unusual locations
- New customer behavior

### False Negatives (Fraud Missed)
- Low-value fraudulent transactions
- Fraud mimicking legitimate behavior
- Collaborative fraud patterns

## Advanced Usage

### Custom Autoencoder Architecture

Edit `config.yaml`:

```yaml
autoencoder:
  architecture:
    hidden_dims: [128, 64, 32]  # Deeper network
    latent_dim: 8               # More compressed
```

### Using Different Kernels for One-Class SVM

```python
model = OneClassSVMDetector(
    kernel='poly',  # Polynomial kernel
    gamma=0.01
)
```

### Exporting Model for Production

```python
# Save Autoencoder
model.save_model('models/autoencoder_fraud.pt')

# Save scikit-learn models
import joblib
joblib.dump(isolation_forest.model, 'models/isolation_forest.pkl')
```

## Troubleshooting

### Issue: Autoencoder training is slow
- **Solution**: Use GPU: set `device: "cuda"` in config.yaml
- **Solution**: Reduce batch size or epochs
- **Solution**: Reduce architecture size (fewer hidden layers)

### Issue: Detection rate is too low
- **Solution**: Lower the threshold (increase allowed FPR)
- **Solution**: Try different contamination values
- **Solution**: Use ensemble methods

### Issue: Too many false positives
- **Solution**: Increase threshold (decrease allowed FPR)
- **Solution**: Check for data quality issues
- **Solution**: Ensure training data is truly all legitimate

## Future Improvements

- [ ] Add more anomaly detection methods (e.g., Deep SVDD, GANs)
- [ ] Implement online learning for streaming data
- [ ] Add feature importance analysis
- [ ] Create API for real-time scoring
- [ ] Add hyperparameter optimization (Optuna, Ray Tune)

## References

- Isolation Forest: Liu et al. (2008) - "Isolation Forest"
- One-Class SVM: Schölkopf et al. (2001) - "Estimating the Support of a High-Dimensional Distribution"
- LOF: Breunig et al. (2000) - "LOF: Identifying Density-Based Local Outliers"
- Autoencoders: Sakurada & Yairi (2014) - "Anomaly Detection Using Autoencoders"

## License

MIT License

## Author

Built for AI Engineer portfolio in financial services fraud detection.
