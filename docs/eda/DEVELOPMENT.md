# Development Guide - Fraud Detection EDA Dashboard

This guide is for developers who want to extend or maintain this dashboard.

## Development Setup

### 1. Install Development Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -r requirements.txt

# Install in editable mode for development
pip install -e .
```

### 2. Install Pre-commit Hooks (Optional)

```bash
# Install black for code formatting
pip install black pre-commit

# Format code
black fraud_detection_dashboard/ tests/

# Run linter
flake8 fraud_detection_dashboard/ tests/

# Run type checker
mypy fraud_detection_dashboard/
```

## Project Architecture

### Module Responsibilities

```
fraud_detection_dashboard/
│
├── app.py                 # [ENTRY POINT] App initialization, server setup
│   ├── create_app()       # Main factory function
│   └── main()             # CLI entry point
│
├── data_loader.py         # [DATA LAYER] Data loading, validation, preprocessing
│   ├── load_fraud_data()  # CSV loading with validation
│   ├── validate_data()    # Schema validation
│   └── preprocess_data()  # Data cleaning, feature engineering
│
├── visualizations.py      # [PRESENTATION LAYER] Plotly figure generation
│   ├── plot_class_distribution()
│   ├── plot_amount_histogram()
│   ├── plot_correlation_heatmap()
│   ├── plot_time_patterns()
│   └── plot_pca_scatter()
│
├── layout.py              # [UI LAYER] Dash HTML/DCC components
│   ├── create_header()
│   ├── create_summary_card()
│   ├── create_filters()
│   ├── create_charts_grid()
│   └── create_dashboard_layout()
│
├── callbacks.py           # [INTERACTIVITY LAYER] Dash callback functions
│   ├── register_callbacks()
│   └── get_filtered_stats()
│
└── utils.py               # [HELPER LAYER] Utilities, constants, formatting
    ├── FRAUD_COLOR, LEGIT_COLOR
    ├── calculate_summary_statistics()
    ├── export_to_html()
    └── format_*() helpers
```

### Data Flow

```
CSV File
   ↓
data_loader.load_fraud_data()
   ↓
data_loader.preprocess_data()
   ↓
[DataFrame stored in memory]
   ↓
┌─────────────────────────────────────┐
│  Dash Callbacks (user interaction)  │
│         (filter data)               │
└─────────────────────────────────────┘
   ↓
visualizations.plot_*()
   ↓
[Plotly Figures]
   ↓
Display in Dashboard or Export to HTML
```

## Adding New Features

### Adding a New Visualization

1. **Create function in `visualizations.py`**:

```python
def plot_new_visualization(df: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Create a new visualization.

    Parameters
    ----------
    df : pd.DataFrame
        The fraud detection dataset.
    **kwargs : dict
        Additional parameters.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    # Your implementation here
    fig = go.Figure()
    return fig
```

2. **Add to layout in `layout.py`**:

```python
# In create_charts_grid()
html.Div(
    style={'backgroundColor': 'white', 'padding': '15px'},
    children=[dcc.Graph(id='new-visualization-chart')]
)
```

3. **Update callback in `callbacks.py`**:

```python
# Add to output list
Output('new-visualization-chart', 'figure')

# Add to update_charts function
fig_new = plot_new_visualization(df_filtered)

# Add to return tuple
return (..., fig_new)
```

4. **Write tests in `tests/test_visualizations.py`**:

```python
class TestPlotNewVisualization:
    def test_basic_creation(self, sample_data):
        fig = plot_new_visualization(sample_data)
        assert isinstance(fig, go.Figure)
```

### Adding a New Filter

1. **Add filter component in `layout.py`**:

```python
dcc.Dropdown(
    id='new-filter-dropdown',
    options=[...],
    value='default'
)
```

2. **Update callback in `callbacks.py`**:

```python
Input('new-filter-dropdown', 'value')

def update_charts(amount_range, log_scale_value, new_filter_value):
    # Use new_filter_value in logic
    pass
```

### Adding API Endpoint

If you need to add API endpoints for external access:

```python
# In app.py
from dash import Input, Output, html

@app.server.route('/api/stats')
def get_stats():
    # Return JSON statistics
    pass
```

## Testing Strategy

### Test Coverage Goals

- Target: **80%+** code coverage
- Use `pytest --cov` to check coverage

### Test Categories

1. **Unit Tests**: Test individual functions
2. **Integration Tests**: Test module interactions
3. **Callback Tests**: Test Dash callback logic

### Writing Tests

```python
# Import required modules
import pytest
import pandas as pd

# Use fixtures for common test data
@pytest.fixture
def sample_data():
    return pd.DataFrame({...})

# Test class
class TestMyFunction:
    def test_basic_case(self, sample_data):
        result = my_function(sample_data)
        assert result is not None

    def test_edge_case(self):
        # Test empty, null, extreme values
        pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fraud_detection_dashboard --cov-report=html

# Run specific test
pytest tests/test_visualizations.py::TestPlotClassDistribution

# Run with verbose output
pytest -v
```

## Performance Optimization

### Known Performance Considerations

1. **PCA Scatter Plot**: Automatically samples datasets >10,000 rows
   - Adjust `sample_size` parameter as needed

2. **Correlation Heatmap**: Can be slow with many features
   - Consider feature selection for very wide datasets

3. **Dashboard Loading**: Large datasets may delay initial load
   - Implement data loading progress indicator if needed

### Optimization Techniques

```python
# 1. Sampling for large datasets
df_sample = df.sample(n=10000, random_state=42)

# 2. Caching expensive computations
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_operation(df_hash):
    pass

# 3. Lazy loading for charts
# Only compute chart when user views it
```

## Debugging

### Common Issues and Solutions

**Issue**: Callback not firing
- **Solution**: Check component IDs match between layout and callback

**Issue**: Figure not updating
- **Solution**: Ensure all callback outputs are returned in correct order

**Issue**: Memory errors with large datasets
- **Solution**: Use sampling or chunking, increase system memory

### Debug Mode

```bash
# Run with debug mode enabled
python -m fraud_detection_dashboard.app --debug

# Enable Dash dev tools
app = dash.Dash(__name__, dev_tools_hot_reload=True)
```

## Code Style

### Formatting

We use `black` for code formatting:

```bash
# Format all code
black fraud_detection_dashboard/ tests/

# Check formatting without modifying
black --check fraud_detection_dashboard/ tests/
```

### Type Hints

All functions should include type hints:

```python
def my_function(param1: str, param2: Optional[int] = None) -> dict[str, Any]:
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    One-line summary.

    Extended description with details.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.

    Returns
    -------
    return_type
        Description of return value.

    Examples
    --------
    >>> function_name('value')
    result
    """
```

## Deployment

### Production Deployment

1. **Set debug to False**:
```python
main(debug=False)
```

2. **Use production WSGI server**:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8050 fraud_detection_dashboard.app:create_app()
```

3. **Environment variables**:
```bash
export DATA_PATH=/path/to/data.csv
export DASH_HOST=0.0.0.0
export DASH_PORT=8050
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8050
CMD ["python", "-m", "fraud_detection_dashboard.app"]
```

## Resources

- [Dash Documentation](https://dash.plotly.com/)
- [Plotly Python Reference](https://plotly.com/python/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## Changelog

### Version 1.0.0 (Current)

- Initial release
- 5 interactive visualizations
- Real-time filtering
- Export functionality
- 80%+ test coverage
- Comprehensive documentation
