# Fraud Model Explainability

Explainability tools for fraud detection models with regulatory compliance support.

## Overview

This project provides tools to make fraud detection models interpretable for analysts, auditors, and regulators. It supports multiple explanation methods (SHAP, LIME, Partial Dependence Plots) and generates professional HTML reports for model governance.

## Features

- **SHAP Explanations**: Fast, accurate explanations for tree-based models (XGBoost, Random Forest)
- **LIME Explanations**: Model-agnostic local explanations with consistency guarantees
- **Partial Dependence Plots**: Global feature insights and non-linearity detection
- **HTML Report Generation**: Professional reports for non-technical users
- **Streamlit UI**: Interactive interface for fraud analysts
- **Multi-Model Support**: XGBoost, Random Forest, Gradient Boosting, Neural Networks
- **Validation Tools**: Consistency checks, quality validation, speed benchmarks

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd fraud_model_explainability

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Python API

```python
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from src.api import create_explainer
from src.utils import format_risk_factors
from src.reports import ReportGenerator

# Prepare data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
feature_names = [f'feature_{i}' for i in range(20)]

# Train model
model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create explainer
explainer = create_explainer(
    model=model,
    model_type='xgboost',
    explainer_type='shap',
    training_data=X[:500],
    feature_names=feature_names
)

# Explain a transaction
X_sample = X[0]
local_explanation = explainer.explain_local(X_sample, feature_names)

# Format for display
risk_factors = format_risk_factors(local_explanation, top_n=5)
for factor in risk_factors:
    print(f"{factor['description']}: {factor['direction']} risk (score: {factor['importance']:.4f})")

# Generate HTML report
generator = ReportGenerator()
html_report = generator.generate_html_report(
    transaction_id='TXN-001',
    prediction=model.predict_proba(X_sample.reshape(1, -1))[0][1],
    predicted_class='Fraud',
    risk_factors=risk_factors,
    global_importance=None,
    model_metadata={
        'name': 'XGBoost Fraud Model',
        'version': '1.0.0',
        'type': 'xgboost'
    }
)

# Save report
generator.save_report(html_report, 'reports/fraud_report_TXN-001.html')
```

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Then:
1. Upload your model file
2. Upload training data (CSV)
3. Configure explainer settings
4. Explain individual transactions
5. Generate and download HTML reports

## Architecture

```
fraud_model_explainability/
├── src/
│   ├── explainers/          # Explanation methods (SHAP, LIME, PDP)
│   ├── reports/             # HTML report generation
│   ├── models/              # Model loading utilities
│   ├── utils/               # Validation and formatting
│   └── api/                 # Explainer factory
├── app/                     # Streamlit UI
├── tests/                   # Unit tests
└── data/models/             # Sample models for testing
```

## Supported Model Types

| Model Type | Recommended Explainer | Notes |
|------------|---------------------|-------|
| XGBoost | SHAP (TreeExplainer) | Fastest, most accurate |
| Random Forest | SHAP (TreeExplainer) | Fast, accurate |
| Gradient Boosting | SHAP (TreeExplainer) | Fast, accurate |
| Neural Networks | SHAP (Deep/Kernel) or LIME | Requires background data |
| Other sklearn | LIME | Model-agnostic |

## Regulatory Compliance

### SR 11-7 (Federal Reserve)

This system supports SR 11-7 requirements for model risk management:

- **Model Documentation**: HTML reports document model predictions and factors
- **Validation**: Consistency checks ensure reproducible explanations
- **Performance Monitoring**: Speed benchmarks track explanation generation time
- **Governance**: Model metadata and version tracking for audit trails

### EU AI Act

Compliance features for high-risk AI systems:

- **Explainability**: Both local and global explanations available
- **Transparency**: Clear, human-readable risk factor descriptions
- **Documentation**: Professional reports for regulatory submissions
- **Record Keeping**: All explanations can be exported and archived

### Key Compliance Features

1. **Consistency**: Same input produces same explanation (validated)
2. **Traceability**: Model metadata, timestamps, and transaction IDs
3. **Performance**: Explanations generated in <2 seconds (configurable)
4. **Quality**: Validation checks for null/infinite values
5. **Stability**: Tests for robustness under input perturbations

## Explanation Methods

### SHAP (SHapley Additive exPlanations)

**Best for**: Tree-based models, global feature importance

**Advantages**:
- Theoretically grounded (Shapley values)
- Fast for tree models (TreeExplainer)
- Both local and global explanations
- Additive feature attribution

**Use when**:
- Using XGBoost, Random Forest, or similar
- Need global importance rankings
- Speed is critical

### LIME (Local Interpretable Model-agnostic Explanations)

**Best for**: Any model type, local explanations

**Advantages**:
- Model-agnostic
- Local fidelity guaranteed
- Easy to understand
- Consistent with fixed random seed

**Use when**:
- Using neural networks or other black-box models
- Need local explanations only
- Model-agnostic approach preferred

### Partial Dependence Plots (PDP)

**Best for**: Global feature behavior, non-linearity detection

**Advantages**:
- Shows marginal effects
- Identifies non-linear relationships
- 2-way interaction plots
- Clear visualizations

**Use when**:
- Understanding global feature behavior
- Detecting non-linearities
- Presenting to non-technical stakeholders

## API Reference

### ExplainerFactory

```python
from src.api import ExplainerFactory

factory = ExplainerFactory(default_explainer='shap')

# Create single explainer
explainer = factory.create(
    model=model,
    model_type='xgboost',
    explainer_type='shap',
    training_data=X_train,
    feature_names=feature_names
)

# Create multiple explainers
explainers = factory.create_multiple(
    model=model,
    model_type='xgboost',
    explainer_types=['shap', 'lime'],
    training_data=X_train,
    feature_names=feature_names
)
```

### Validation Functions

```python
from src.utils import validate_consistency, benchmark_explanation_speed

# Check consistency (same input = same explanation)
result = validate_consistency(
    explainer=explainer,
    X=X_sample,
    feature_names=feature_names,
    n_runs=5,
    tolerance=1e-5
)
print(f"Consistent: {result['is_consistent']}")

# Benchmark speed
result = benchmark_explanation_speed(
    explainer=explainer,
    X=X_sample,
    feature_names=feature_names,
    n_runs=50,
    target_seconds=2.0
)
print(f"Mean time: {result['mean_time']:.4f}s")
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_shap.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Categories

1. **test_shap.py**: SHAP explainer unit tests
2. **test_lime.py**: LIME explainer unit tests
3. **test_reports.py**: Report generation tests
4. **test_consistency.py**: Validation and consistency tests

## Performance Guidelines

| Operation | Target Time | Notes |
|-----------|-------------|-------|
| Local explanation | <2 seconds | Per transaction |
| Global explanation | <30 seconds | For 1000 samples |
| Report generation | <1 second | HTML generation |
| Consistency check | <10 seconds | 5 runs |

Optimization tips:
- Use SHAP TreeExplainer for tree models (fastest)
- Sample large datasets for global explanations
- Use background data for SHAP (100-500 samples)
- Enable caching for repeated explanations

## Best Practices

1. **Always validate consistency** before deploying to production
2. **Use appropriate explainers** for your model type
3. **Document feature descriptions** for non-technical users
4. **Keep training data** for LIME and SHAP background
5. **Set random seeds** for reproducibility
6. **Test explanation speed** before real-time deployment
7. **Archive reports** for audit and compliance

## Troubleshooting

### SHAP errors

**Problem**: "training_data is required"
**Solution**: Provide background data (100-500 random samples)

**Problem**: Slow explanations
**Solution**: Use TreeExplainer for tree models, reduce background data size

### LIME errors

**Problem**: Inconsistent explanations
**Solution**: Ensure `random_state` is set when creating explainer

**Problem**: Feature mismatch
**Solution**: Ensure training features match prediction features

### Report generation

**Problem**: Reports not styled correctly
**Solution**: Check that CSS is included in HTML template

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this project in your work, please cite:

```bibtex
@software{fraud_model_explainability,
  title={Fraud Model Explainability Toolkit},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/fraud-model-explainability}
}
```

## Acknowledgments

- SHAP: Lundberg & Lee (2017)
- LIME: Ribeiro et al. (2016)
- scikit-learn: Pedregosa et al. (2011)

## Contact

For questions or support, please open an issue on GitHub or contact the Model Risk Management team.

---

**Disclaimer**: This tool is provided for informational purposes and should be used in conjunction with other fraud detection methods and human expertise. Always validate explanations before using them for critical decisions.
