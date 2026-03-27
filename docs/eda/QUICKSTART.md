# 🚀 Quick Start - Fraud Detection EDA Dashboard

## 5-Minute Setup

### Step 1: Download Dataset
```bash
# Visit: https://www.kaggle.com/mlg-ulb/creditcardfraud
# Download creditcard.csv and place in data/ directory
mv ~/Downloads/creditcard.csv data/
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Dashboard
```bash
python run_dashboard.py
```

Visit: **http://127.0.0.1:8050**

---

## Command Reference

| Command | Description |
|---------|-------------|
| `python run_dashboard.py` | Start dashboard |
| `python -m pytest` | Run tests |
| `pytest --cov` | Run with coverage |
| `black .` | Format code |
| `flake8 .` | Check style |

---

## File Locations

| Purpose | Path |
|---------|------|
| Dataset | `data/creditcard.csv` |
| Main App | `fraud_detection_dashboard/app.py` |
| Visuals | `fraud_detection_dashboard/visualizations.py` |
| Tests | `tests/` |
| Exports | `outputs/` |

---

## Key Functions

```python
# Load data
from fraud_detection_dashboard import load_fraud_data
df = load_fraud_data('data/creditcard.csv')

# Create visualizations
from fraud_detection_dashboard.visualizations import plot_class_distribution
fig = plot_class_distribution(df)

# Calculate stats
from fraud_detection_dashboard.utils import calculate_summary_statistics
stats = calculate_summary_statistics(df)
```

---

## Troubleshooting

**Dataset not found?**
→ Download from Kaggle link above

**Port in use?**
→ `python run_dashboard.py --port 8051`

**Tests failing?**
→ Ensure all dependencies installed: `pip install -r requirements.txt`

---

## Next Steps

- Read [README.md](README.md) for full documentation
- Check [DEVELOPMENT.md](DEVELOPMENT.md) for extending the dashboard
- Review inline code documentation with docstrings
