"""
Unit tests for visualizations module.

Tests cover all plot generation functions including validation,
correct data transformations, and proper figure structures.
"""

import numpy as np
import pandas as pd
import pytest
from plotly.graph_objects import Figure

from fraud_detection_dashboard.utils import FRAUD_COLOR, LEGIT_COLOR
from fraud_detection_dashboard.visualizations import (
    plot_amount_histogram,
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_pca_scatter,
    plot_time_patterns,
)


class TestPlotClassDistribution:
    """Tests for plot_class_distribution function."""

    def test_plot_class_distribution_basic(self, sample_data):
        """Test basic class distribution plot creation."""
        fig = plot_class_distribution(sample_data)

        assert isinstance(fig, Figure)
        assert len(fig.data) == 1  # One bar trace
        assert fig.data[0].type == 'bar'

    def test_plot_class_distribution_colors(self, sample_data):
        """Test that correct colors are used."""
        fig = plot_class_distribution(sample_data)

        # Check marker colors
        assert fig.data[0].marker.color == [LEGIT_COLOR, FRAUD_COLOR]

    def test_plot_class_distribution_labels(self, sample_data):
        """Test x-axis labels are correct."""
        fig = plot_class_distribution(sample_data)

        # Check x-axis labels
        assert list(fig.data[0].x) == ['Legitimate', 'Fraud']

    def test_plot_class_distribution_missing_column(self):
        """Test error when Class column is missing."""
        df = pd.DataFrame({'Amount': [100, 200, 300]})

        with pytest.raises(KeyError, match="must contain 'Class' column"):
            plot_class_distribution(df)

    def test_plot_class_distribution_all_fraud(self):
        """Test with all fraudulent transactions."""
        df = pd.DataFrame({'Class': [1, 1, 1], 'Amount': [100, 200, 300]})
        fig = plot_class_distribution(df)

        assert fig.data[0].y[1] == 3  # 3 fraud transactions
        assert fig.data[0].y[0] == 0  # 0 legitimate

    def test_plot_class_distribution_all_legitimate(self):
        """Test with all legitimate transactions."""
        df = pd.DataFrame({'Class': [0, 0, 0], 'Amount': [100, 200, 300]})
        fig = plot_class_distribution(df)

        assert fig.data[0].y[0] == 3  # 3 legitimate
        assert fig.data[0].y[1] == 0  # 0 fraud

    def test_plot_class_distribution_title(self, sample_data):
        """Test figure has appropriate title."""
        fig = plot_class_distribution(sample_data)

        assert 'Class Distribution' in fig.layout.title.text


class TestPlotAmountHistogram:
    """Tests for plot_amount_histogram function."""

    def test_plot_amount_histogram_basic(self, sample_data):
        """Test basic amount histogram creation."""
        fig = plot_amount_histogram(sample_data)

        assert isinstance(fig, Figure)
        assert len(fig.data) == 2  # Two histogram traces
        assert all(trace.type == 'histogram' for trace in fig.data)

    def test_plot_amount_histogram_colors(self, sample_data):
        """Test correct colors are used."""
        fig = plot_amount_histogram(sample_data)

        assert fig.data[0].marker.color == LEGIT_COLOR
        assert fig.data[1].marker.color == FRAUD_COLOR

    def test_plot_amount_histogram_log_scale(self, sample_data):
        """Test log scale option."""
        fig = plot_amount_histogram(sample_data, log_scale=True)

        assert fig.layout.xaxis.type == 'log'

    def test_plot_amount_histogram_linear_scale(self, sample_data):
        """Test linear scale (default)."""
        fig = plot_amount_histogram(sample_data, log_scale=False)

        assert fig.layout.xaxis.type is None  # Linear is default

    def test_plot_amount_histogram_missing_columns(self):
        """Test error when required columns are missing."""
        df = pd.DataFrame({'V1': [1, 2, 3]})

        with pytest.raises(KeyError, match="Missing required columns"):
            plot_amount_histogram(df)

    def test_plot_amount_histogram_overlay_mode(self, sample_data):
        """Test histograms use overlay mode."""
        fig = plot_amount_histogram(sample_data)

        assert fig.layout.barmode == 'overlay'


class TestPlotCorrelationHeatmap:
    """Tests for plot_correlation_heatmap function."""

    def test_plot_correlation_heatmap_basic(self, full_sample_data):
        """Test basic correlation heatmap creation."""
        fig = plot_correlation_heatmap(full_sample_data)

        assert isinstance(fig, Figure)
        assert len(fig.data) == 1  # One heatmap trace
        assert fig.data[0].type == 'heatmap'

    def test_plot_correlation_heatmap_colors(self, full_sample_data):
        """Test heatmap uses diverging colorscale."""
        fig = plot_correlation_heatmap(full_sample_data)

        assert fig.data[0].colorscale == 'RdBu'
        assert fig.data[0].zmid == 0  # Centered at 0

    def test_plot_correlation_heatmap_range(self, full_sample_data):
        """Test correlation values are in valid range."""
        fig = plot_correlation_heatmap(full_sample_data)

        assert fig.data[0].zmin == -1
        assert fig.data[0].zmax == 1

    def test_plot_correlation_heatmap_no_v_columns(self):
        """Test heatmap with no V-columns."""
        df = pd.DataFrame({
            'Amount': [100, 200, 300],
            'Time': [0, 3600, 7200],
            'Class': [0, 1, 0],
        })

        fig = plot_correlation_heatmap(df)

        # Should still work with Amount, Time, and Class
        assert isinstance(fig, Figure)

    def test_plot_correlation_heatmap_dimensions(self, full_sample_data):
        """Test heatmap dimensions match feature count."""
        fig = plot_correlation_heatmap(full_sample_data)

        z_array = fig.data[0].z
        assert z_array.shape[0] == z_array.shape[1]  # Square matrix


class TestPlotTimePatterns:
    """Tests for plot_time_patterns function."""

    def test_plot_time_patterns_with_hour(self, sample_data_with_hour):
        """Test time patterns with Hour column."""
        fig = plot_time_patterns(sample_data_with_hour)

        assert isinstance(fig, Figure)
        assert 'Temporal' in fig.layout.title.text

    def test_plot_time_patterns_with_time_column(self, sample_data):
        """Test time patterns with Time column (no Hour)."""
        if 'Hour' in sample_data.columns:
            sample_data = sample_data.drop(columns=['Hour'])

        fig = plot_time_patterns(sample_data)

        assert isinstance(fig, Figure)

    def test_plot_time_patterns_missing_class(self):
        """Test error when Class column is missing."""
        df = pd.DataFrame({'Time': [0, 3600, 7200]})

        with pytest.raises(KeyError, match="must contain 'Class' column"):
            plot_time_patterns(df)

    def test_plot_time_patterns_no_time_column(self):
        """Test error when neither Time nor Hour exists."""
        df = pd.DataFrame({'Class': [0, 1, 0], 'Amount': [100, 200, 300]})

        with pytest.raises(KeyError, match="must contain either 'Hour' or 'Time'"):
            plot_time_patterns(df)

    def test_plot_time_patterns_subplots(self, sample_data_with_hour):
        """Test figure has correct number of subplots."""
        fig = plot_time_patterns(sample_data_with_hour)

        # Should have 2 rows of subplots
        assert fig.layout.grid.rows == 2


class TestPlotPCAScatter:
    """Tests for plot_pca_scatter function."""

    def test_plot_pca_scatter_basic(self, full_sample_data):
        """Test basic PCA scatter plot creation."""
        fig = plot_pca_scatter(full_sample_data)

        assert isinstance(fig, Figure)
        assert len(fig.data) == 2  # Two scatter traces (legit and fraud)
        assert all(trace.type == 'scatter' for trace in fig.data)

    def test_plot_pca_scatter_colors(self, full_sample_data):
        """Test correct colors are used."""
        fig = plot_pca_scatter(full_sample_data)

        assert fig.data[0].marker.color == LEGIT_COLOR
        assert fig.data[1].marker.color == FRAUD_COLOR

    def test_plot_pca_scatter_sample_size(self, full_sample_data):
        """Test sampling functionality."""
        original_size = len(full_sample_data)
        sample_size = min(5, original_size)

        fig = plot_pca_scatter(full_sample_data, sample_size=sample_size)

        # Figure should be created successfully
        assert isinstance(fig, Figure)

    def test_plot_pca_scatter_invalid_n_components(self, full_sample_data):
        """Test error with invalid n_components."""
        with pytest.raises(ValueError, match="must be at least 2"):
            plot_pca_scatter(full_sample_data, n_components=1)

    def test_plot_pca_scatter_insufficient_features(self):
        """Test error with insufficient V-features."""
        df = pd.DataFrame({
            'Class': [0, 1, 0],
            'V1': [1.0, 2.0, 3.0],
        })

        with pytest.raises(ValueError, match="Need at least 2 V-features"):
            plot_pca_scatter(df, n_components=2)

    def test_plot_pca_scatter_missing_class(self):
        """Test error when Class column is missing."""
        df = pd.DataFrame({
            'V1': [1.0, 2.0, 3.0],
            'V2': [0.5, 1.5, 2.5],
        })

        with pytest.raises(KeyError, match="must contain 'Class' column"):
            plot_pca_scatter(df)

    def test_plot_pca_scatter_handles_nan(self):
        """Test that NaN values are handled correctly."""
        df = pd.DataFrame({
            'Class': [0, 1, 0, 1],
            'V1': [1.0, np.nan, 3.0, 4.0],
            'V2': [0.5, 1.5, np.nan, 2.5],
            'V3': [1.0, 2.0, 3.0, 4.0],
        })

        fig = plot_pca_scatter(df)

        # Should successfully create figure despite NaN values
        assert isinstance(fig, Figure)

    def test_plot_pca_scatter_explained_variance(self, full_sample_data):
        """Test that explained variance is included in title."""
        fig = plot_pca_scatter(full_sample_data)

        assert 'Explained Variance' in fig.layout.title.text
        assert 'PC1=' in fig.layout.title.text
        assert 'PC2=' in fig.layout.title.text


# ===== Fixtures =====

@pytest.fixture
def sample_data():
    """Create minimal sample dataframe for testing."""
    return pd.DataFrame({
        'Time': [0, 3600, 7200],
        'V1': [1.0, -1.0, 0.5],
        'Amount': [100.0, 200.0, 150.0],
        'Class': [0, 1, 0],
    })


@pytest.fixture
def sample_data_with_hour():
    """Create sample dataframe with Hour column."""
    return pd.DataFrame({
        'Time': [0, 3600, 7200],
        'Hour': [0, 1, 2],
        'V1': [1.0, -1.0, 0.5],
        'Amount': [100.0, 200.0, 150.0],
        'Class': [0, 1, 0],
    })


@pytest.fixture
def full_sample_data():
    """Create sample dataframe with multiple V-features."""
    data = {
        'Time': [0, 3600, 7200, 10800, 14400],
        'Amount': [100.0, 200.0, 150.0, 50.0, 300.0],
        'Class': [0, 1, 0, 0, 1],
    }
    # Add V1-V5 features
    for i in range(1, 6):
        data[f'V{i}'] = np.random.randn(5)

    return pd.DataFrame(data)
