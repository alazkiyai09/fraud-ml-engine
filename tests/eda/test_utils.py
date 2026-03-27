"""
Unit tests for utils module.

Tests cover utility functions including summary statistics,
export functionality, and formatting helpers.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from fraud_detection_dashboard.utils import (
    FRAUD_COLOR,
    LEGIT_COLOR,
    calculate_summary_statistics,
    export_to_html,
    format_currency,
    format_number,
)


class TestColorConstants:
    """Tests for color constant definitions."""

    def test_fraud_color_value(self):
        """Test fraud color is correctly defined."""
        assert FRAUD_COLOR == "#FF6B6B"

    def test_legit_color_value(self):
        """Test legitimate color is correctly defined."""
        assert LEGIT_COLOR == "#4ECDC4"


class TestCalculateSummaryStatistics:
    """Tests for calculate_summary_statistics function."""

    def test_calculate_basic_statistics(self, sample_data):
        """Test basic summary statistics calculation."""
        stats = calculate_summary_statistics(sample_data)

        assert isinstance(stats, dict)
        assert stats['total_transactions'] == 3
        assert stats['fraud_count'] == 1
        assert stats['legit_count'] == 2

    def test_fraud_percentage_calculation(self, sample_data):
        """Test fraud percentage is calculated correctly."""
        stats = calculate_summary_statistics(sample_data)

        expected_percentage = round((1 / 3) * 100, 2)
        assert stats['fraud_percentage'] == expected_percentage

    def test_amount_statistics(self, sample_data):
        """Test amount-related statistics."""
        stats = calculate_summary_statistics(sample_data)

        # Total amount: 100 + 200 + 150 = 450
        assert stats['total_amount'] == 450.0

        # Average amount: 450 / 3 = 150
        assert stats['avg_amount'] == 150.0

        # Average fraud amount (Class=1): 200
        assert stats['avg_fraud_amount'] == 200.0

        # Average legit amount (Class=0): (100 + 150) / 2 = 125
        assert stats['avg_legit_amount'] == 125.0

    def test_feature_count(self, sample_data):
        """Test feature count excludes Class column."""
        stats = calculate_summary_statistics(sample_data)

        # 4 columns total, minus Class = 3 features
        assert stats['feature_count'] == 3

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pd.DataFrame({'Class': [], 'Amount': []})
        stats = calculate_summary_statistics(df)

        assert stats['total_transactions'] == 0
        assert stats['fraud_count'] == 0
        assert stats['legit_count'] == 0
        assert stats['fraud_percentage'] == 0
        assert stats['avg_fraud_amount'] == 0.0
        assert stats['avg_legit_amount'] == 0.0

    def test_all_fraud_transactions(self):
        """Test with all fraudulent transactions."""
        df = pd.DataFrame({
            'Class': [1, 1, 1],
            'Amount': [100, 200, 300],
        })
        stats = calculate_summary_statistics(df)

        assert stats['fraud_count'] == 3
        assert stats['legit_count'] == 0
        assert stats['fraud_percentage'] == 100.0
        assert stats['avg_legit_amount'] == 0.0

    def test_all_legitimate_transactions(self):
        """Test with all legitimate transactions."""
        df = pd.DataFrame({
            'Class': [0, 0, 0],
            'Amount': [100, 200, 300],
        })
        stats = calculate_summary_statistics(df)

        assert stats['fraud_count'] == 0
        assert stats['legit_count'] == 3
        assert stats['fraud_percentage'] == 0.0

    def test_missing_class_column_raises_error(self):
        """Test that missing Class column raises KeyError."""
        df = pd.DataFrame({'Amount': [100, 200, 300]})

        with pytest.raises(KeyError, match="Missing required columns"):
            calculate_summary_statistics(df)

    def test_missing_amount_column_raises_error(self):
        """Test that missing Amount column raises KeyError."""
        df = pd.DataFrame({'Class': [0, 1, 0]})

        with pytest.raises(KeyError, match="Missing required columns"):
            calculate_summary_statistics(df)

    def test_return_types(self, sample_data):
        """Test that return types match expectations."""
        stats = calculate_summary_statistics(sample_data)

        assert isinstance(stats['total_transactions'], int)
        assert isinstance(stats['fraud_count'], int)
        assert isinstance(stats['legit_count'], int)
        assert isinstance(stats['fraud_percentage'], float)
        assert isinstance(stats['total_amount'], float)
        assert isinstance(stats['avg_amount'], float)


class TestExportToHtml:
    """Tests for export_to_html function."""

    def test_export_to_html_success(self, tmp_path):
        """Test successful HTML export."""
        fig = go.Figure(data=go.Bar(x=[1, 2, 3], y=[4, 5, 6]))
        output_path = tmp_path / "test_output.html"

        export_to_html(fig, str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_export_creates_parent_directories(self, tmp_path):
        """Test that export creates parent directories."""
        fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
        output_path = tmp_path / "subdir" / "nested" / "output.html"

        export_to_html(fig, str(output_path))

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_export_with_pathlib_path(self, tmp_path):
        """Test export with pathlib.Path object."""
        fig = go.Figure(data=go.Bar(x=[1, 2], y=[3, 4]))
        output_path = tmp_path / "output.html"

        export_to_html(fig, output_path)

        assert output_path.exists()

    def test_export_empty_filepath_raises_error(self):
        """Test that empty filepath raises ValueError."""
        fig = go.Figure()

        with pytest.raises(ValueError, match="Filepath cannot be empty"):
            export_to_html(fig, "")

    def test_export_invalid_path_raises_error(self):
        """Test export to invalid path raises IOError."""
        fig = go.Figure()

        # Try to write to a location that can't be created
        # This is a bit tricky to test cross-platform, so we'll use
        # a mock scenario
        with pytest.raises(IOError, match="Failed to export"):
            export_to_html(fig, "/nonexistent/root/path/file.html")

    def test_export_html_contains_plotly(self, tmp_path):
        """Test exported HTML contains Plotly content."""
        fig = go.Figure(data=go.Bar(x=[1, 2, 3], y=[4, 5, 6]))
        output_path = tmp_path / "test.html"

        export_to_html(fig, str(output_path))

        content = output_path.read_text()
        assert "plotly" in content.lower()
        assert "<html" in content


class TestFormatCurrency:
    """Tests for format_currency function."""

    @pytest.mark.parametrize("value,expected", [
        (0, "$0.00"),
        (100, "$100.00"),
        (1234.56, "$1,234.56"),
        (1000000, "$1,000,000.00"),
        (-500, "-$500.00"),
    ])
    def test_format_currency_values(self, value, expected):
        """Test currency formatting with various values."""
        result = format_currency(value)
        assert result == expected

    def test_format_currency_with_decimals(self):
        """Test currency formatting preserves decimals."""
        result = format_currency(1234.5678)
        assert result == "$1,234.57"  # Rounded to 2 decimals


class TestFormatNumber:
    """Tests for format_number function."""

    @pytest.mark.parametrize("value,expected", [
        (0, "0"),
        (100, "100"),
        (1234, "1,234"),
        (1234567, "1,234,567"),
        (1234.56, "1,234.56"),
        (1000000.99, "1,000,000.99"),
    ])
    def test_format_number_values(self, value, expected):
        """Test number formatting with various values."""
        result = format_number(value)
        assert result == expected

    def test_format_number_integer(self):
        """Test integer formatting."""
        result = format_number(123456)
        assert result == "123,456"

    def test_format_number_float(self):
        """Test float formatting."""
        result = format_number(1234.5678)
        assert result == "1,234.57"  # Rounded to 2 decimals


# ===== Fixtures =====

@pytest.fixture
def sample_data():
    """Create sample dataframe for testing."""
    return pd.DataFrame({
        'Class': [0, 1, 0],
        'Amount': [100.0, 200.0, 150.0],
        'V1': [1.0, -1.0, 0.5],
    })
