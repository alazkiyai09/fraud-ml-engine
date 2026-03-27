# 💳 Credit Card Fraud Detection - EDA Dashboard

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/Dash-2.14%2B-orange)](https://dash.plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A production-grade interactive Exploratory Data Analysis (EDA) dashboard for credit card fraud detection. Built with Plotly Dash, this dashboard provides comprehensive visualizations and insights into fraud patterns in financial transaction data.

## 🌟 Features

- **Interactive Visualizations**: 5 dynamic Plotly charts for comprehensive data exploration
- **Real-time Filtering**: Filter transactions by amount range and toggle log scale
- **Export Functionality**: Export individual charts as standalone HTML files
- **Summary Statistics**: Key metrics displayed in an easy-to-read card layout
- **Responsive Design**: Clean, professional interface optimized for data analysis
- **Type Hints & Docstrings**: Fully documented code for easy maintenance
- **Unit Tests**: 80%+ test coverage with pytest

## 📊 Visualizations

| Visualization | Description |
|--------------|-------------|
| **Class Distribution** | Bar chart showing fraud vs legitimate transaction counts with log scale |
| **Amount Histogram** | Overlapping histograms comparing amount distributions |
| **Correlation Heatmap** | Feature correlation matrix with color intensity |
| **Time Patterns** | Hourly transaction analysis with fraud rate trends |
| **PCA Scatter Plot** | 2D PCA visualization showing class separation |

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
cd /path/to/your/projects
git clone <repository-url>
cd 30Days_Project
```

2. **Create virtual environment** (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dataset Setup

1. **Download the Kaggle Credit Card Fraud Detection dataset**
   - Visit: https://www.kaggle.com/mlg-ulb/creditcardfraud
   - Download `creditcard.csv`

2. **Place the dataset in the data directory**
```bash
mv /path/to/creditcard.csv data/
```

### Running the Dashboard

Start the dashboard server:

```bash
python -m fraud_detection_dashboard.app
```

Or use the module directly:

```bash
cd fraud_detection_dashboard
python app.py
```

The dashboard will be available at: **http://127.0.0.1:8050**

## 📖 Usage Guide

### Interactive Filters

1. **Amount Range Slider**
   - Adjust the min/max range to filter transactions
   - Charts update automatically based on selection
   - Display shows current selected range

2. **Log Scale Toggle**
   - Check "Enable" to apply logarithmic scale to Amount Histogram
   - Useful for visualizing skewed distributions

3. **Export Button**
   - Click "Export Dashboard" to save all charts as HTML files
   - Files are saved to `outputs/` directory with timestamps
   - Exported files can be opened in any web browser

### Summary Statistics Card

The top card displays key metrics:
- **Total Transactions**: Overall dataset size
- **Fraud Count**: Number of fraudulent transactions (red border)
- **Legitimate Count**: Number of legitimate transactions (teal border)
- **Average Amount**: Mean transaction amount across all classes

### Chart Interactions

- **Hover**: Hover over any chart element to see detailed values
- **Zoom**: Click and drag on charts to zoom into regions
- **Pan**: Click and drag the pan button to move around
- **Reset**: Double-click to reset zoom/pan
- **Legend**: Click legend items to show/hide specific traces

## 🏗️ Project Structure

```
30Days_Project/
├── fraud_detection_dashboard/
│   ├── __init__.py           # Package initialization
│   ├── app.py                # Main application entry point
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── visualizations.py     # Plotly figure generation
│   ├── layout.py             # Dash layout components
│   ├── callbacks.py          # Interactive callbacks
│   └── utils.py              # Helper functions and constants
│
├── tests/
│   ├── test_data_loader.py   # Data module tests
│   ├── test_visualizations.py
│   ├── test_layout.py
│   ├── test_callbacks.py
│   └── test_utils.py
│
├── data/
│   └── creditcard.csv        # Kaggle dataset (not included)
│
├── outputs/                  # Exported HTML files
├── requirements.txt          # Python dependencies
├── pytest.ini               # Test configuration
├── setup.py                 # Package setup
└── README.md                # This file
```

## 🧪 Running Tests

Execute the full test suite:

```bash
# Run all tests with coverage
pytest

# Run with detailed output
pytest -v

# Run specific test file
pytest tests/test_data_loader.py

# Run with coverage HTML report
pytest --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html  # On Windows: start htmlcov/index.html
```

## 📝 Module Documentation

### Data Module (`data_loader.py`)

```python
from fraud_detection_dashboard import load_fraud_data, preprocess_data

# Load dataset
df = load_fraud_data('data/creditcard.csv')

# Preprocess with time normalization
df_processed = preprocess_data(df, normalize_time=True)
```

### Visualizations Module (`visualizations.py`)

```python
from fraud_detection_dashboard.visualizations import (
    plot_class_distribution,
    plot_amount_histogram,
    plot_correlation_heatmap,
    plot_time_patterns,
    plot_pca_scatter
)

# Create individual figures
fig1 = plot_class_distribution(df)
fig2 = plot_amount_histogram(df, log_scale=True)
fig3 = plot_correlation_heatmap(df)
fig4 = plot_time_patterns(df)
fig5 = plot_pca_scatter(df, sample_size=5000)
```

### Utilities Module (`utils.py`)

```python
from fraud_detection_dashboard.utils import (
    calculate_summary_statistics,
    export_to_html,
    format_currency
)

# Calculate statistics
stats = calculate_summary_statistics(df)

# Export figure to HTML
export_to_html(fig, 'outputs/analysis.html')

# Format values
amount_str = format_currency(1234.56)  # "$1,234.56"
```

## 🎨 Color Scheme

- **Fraud Transactions**: `#FF6B6B` (Red/Coral)
- **Legitimate Transactions**: `#4ECDC4` (Teal)
- **Background**: `#F5F6FA` (Light Gray)
- **Cards**: `#ECF0F1` (Gray)
- **Header**: `#2C3E50` (Dark Blue)

## 📸 Screenshots

<!-- TODO: Add screenshots after running the dashboard -->

### Dashboard Overview
*Full dashboard layout with summary statistics card*

### Class Distribution
*Bar chart showing fraud vs legitimate transactions*

### Amount Histogram
*Overlapping histograms with log scale option*

### Correlation Heatmap
*Feature correlation matrix visualization*

### Time Patterns
*Hourly transaction analysis with fraud rates*

### PCA Scatter Plot
*2D PCA showing class separation patterns*

## 🔧 Configuration

### Customization Options

**Change Server Port** (`app.py`):
```python
main(port=8080)  # Default: 8050
```

**Adjust Amount Range** (`layout.py`):
```python
dcc.RangeSlider(
    id='amount-range-slider',
    min=0,
    max=1000,  # Adjust max value
    step=10,
)
```

**Modify Color Scheme** (`utils.py`):
```python
FRAUD_COLOR = "#FF6B6B"  # Change fraud color
LEGIT_COLOR = "#4ECDC4"  # Change legitimate color
```

## 🐛 Troubleshooting

### Common Issues

**1. Dataset Not Found**
```
FileNotFoundError: Dataset file not found: data/creditcard.csv
```
**Solution**: Download the Kaggle dataset and place it in the `data/` directory.

**2. Port Already in Use**
```
Address already in use
```
**Solution**: Either stop the process using port 8050 or run on a different port:
```bash
python -m fraud_detection_dashboard.app --port 8051
```

**3. Import Errors**
```
ModuleNotFoundError: No module named 'dash'
```
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

**4. Memory Issues with Large Datasets**
**Solution**: The PCA scatter plot automatically samples data for datasets >10,000 rows. Adjust `sample_size` parameter if needed.

## 📚 Dataset Information

**Source**: Kaggle Credit Card Fraud Detection
**URL**: https://www.kaggle.com/mlg-ulb/creditcardfraud

**Dataset Details**:
- **Transactions**: 284,807
- **Features**: 31 (Time, V1-V28, Amount, Class)
- **Fraud Cases**: 492 (0.173%)
- **Time Period**: 2 days
- **Feature Transformation**: V1-V28 are PCA-transformed features

**Features**:
- `Time`: Seconds elapsed between each transaction
- `V1-V28`: PCA-transformed features (confidential)
- `Amount`: Transaction amount
- `Class`: 1 = Fraud, 0 = Legitimate

## 🤝 Contributing

This is a portfolio project. Suggestions for improvements:
1. Additional visualization types (t-SNE, UMAP)
2. Machine learning model integration
3. Real-time data streaming support
4. Additional filters (time range, feature values)
5. Export to PDF/PNG functionality

## 📄 License

MIT License - feel free to use this project for learning and portfolio purposes.

## 👤 Author

**Name**: [Your Name]
**Background**: 3+ years in fraud detection with SAS Fraud Management
**Portfolio**: AI/ML for Financial Services

## 🙏 Acknowledgments

- Kaggle for the Credit Card Fraud Detection dataset
- Plotly Dash team for the excellent framework
- Financial services community for fraud detection research

## 📞 Contact

For questions or collaboration:
- **GitHub**: [Your GitHub Profile]
- **LinkedIn**: [Your LinkedIn Profile]
- **Email**: [Your Email]

---

⭐ If you find this project helpful, please consider giving it a star!

**Built with ❤️ for the financial services and AI/ML community**
