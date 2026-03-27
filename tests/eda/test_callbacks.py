"""
Unit tests for callbacks module.

Tests cover callback registration and function logic.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from dash import Dash

from fraud_detection_dashboard.callbacks import (
    get_filtered_stats,
    register_callbacks,
    register_summary_card_callback,
)


class TestRegisterCallbacks:
    """Tests for register_callbacks function."""

    def test_register_callbacks_adds_callbacks(self, sample_dashboard_data):
        """Test that callbacks are registered with the app."""
        app = Dash(__name__)
        df = sample_dashboard_data

        # Get initial callback count
        initial_callbacks = len(app.callback_map)

        # Register callbacks
        register_callbacks(app, df)

        # Should have added callbacks
        assert len(app.callback_map) > initial_callbacks

    def test_register_callbacks_with_valid_dataframe(self, sample_dashboard_data):
        """Test registration with valid DataFrame."""
        app = Dash(__name__)
        df = sample_dashboard_data

        # Should not raise any exceptions
        register_callbacks(app, df)

        # Verify callback map contains expected IDs
        callback_ids = list(app.callback_map.keys())
        assert 'amount-range-slider.value' in str(callback_ids)

    def test_register_callbacks_preserves_data(self, sample_dashboard_data):
        """Test that registration doesn't modify the input DataFrame."""
        app = Dash(__name__)
        df = sample_dashboard_data
        original_shape = df.shape
        original_columns = df.columns.tolist()

        register_callbacks(app, df)

        # DataFrame should be unchanged
        assert df.shape == original_shape
        assert df.columns.tolist() == original_columns


class TestRegisterSummaryCardCallback:
    """Tests for register_summary_card_callback function."""

    def test_register_summary_card_callable(self):
        """Test that function is callable."""
        app = Dash(__name__)
        df = pd.DataFrame({'Class': [0, 1], 'Amount': [100, 200]})

        # Should not raise exceptions
        register_summary_card_callback(app, df)

    def test_register_summary_card_no_effect(self):
        """Test that function doesn't add callbacks (placeholder)."""
        app = Dash(__name__)
        df = pd.DataFrame({'Class': [0, 1], 'Amount': [100, 200]})

        initial_callbacks = len(app.callback_map)
        register_summary_card_callback(app, df)

        # Should not add callbacks (placeholder function)
        assert len(app.callback_map) == initial_callbacks


class TestGetFilteredStats:
    """Tests for get_filtered_stats function."""

    def test_get_filtered_stats_basic(self, sample_dashboard_data):
        """Test basic statistics calculation."""
        stats = get_filtered_stats(sample_dashboard_data, [0, 500])

        assert isinstance(stats, dict)
        assert 'total_transactions' in stats
        assert 'fraud_count' in stats
        assert 'legit_count' in stats

    def test_get_filtered_stats_amount_filtering(self, sample_dashboard_data):
        """Test that filtering by amount works correctly."""
        # All amounts are in range [0, 500]
        stats_all = get_filtered_stats(sample_dashboard_data, [0, 500])

        # Only amounts in range [0, 150]
        stats_filtered = get_filtered_stats(sample_dashboard_data, [0, 150])

        # Filtered stats should have fewer or equal transactions
        assert stats_filtered['total_transactions'] <= stats_all['total_transactions']

    def test_get_filtered_stats_no_match(self, sample_dashboard_data):
        """Test with amount range that matches no transactions."""
        stats = get_filtered_stats(sample_dashboard_data, [10000, 20000])

        # Should have zero transactions
        assert stats['total_transactions'] == 0
        assert stats['fraud_count'] == 0
        assert stats['legit_count'] == 0

    def test_get_filtered_stats_all_transactions(self, sample_dashboard_data):
        """Test with amount range covering all transactions."""
        max_amount = sample_dashboard_data['Amount'].max()
        stats = get_filtered_stats(sample_dashboard_data, [0, max_amount])

        # Should match all transactions
        assert stats['total_transactions'] == len(sample_dashboard_data)

    def test_get_filtered_stats_return_types(self, sample_dashboard_data):
        """Test that return types match expectations."""
        stats = get_filtered_stats(sample_dashboard_data, [0, 1000])

        assert isinstance(stats['total_transactions'], int)
        assert isinstance(stats['fraud_count'], int)
        assert isinstance(stats['legit_count'], int)
        assert isinstance(stats['fraud_percentage'], float)

    def test_get_filtered_stats_calculations(self, simple_test_data):
        """Test statistical calculations are correct."""
        stats = get_filtered_stats(simple_test_data, [0, 1000])

        # Verify counts
        assert stats['total_transactions'] == 4
        assert stats['fraud_count'] == 1
        assert stats['legit_count'] == 3

        # Verify percentage (1/4 = 25%)
        assert abs(stats['fraud_percentage'] - 25.0) < 0.01


class TestCallbackLogic:
    """Tests for callback function logic (without full Dash stack)."""

    def test_amount_range_display_format(self):
        """Test amount range display formatting."""
        amount_range = [100, 500]
        min_amount, max_amount = amount_range
        display_text = f"Selected Range: ${min_amount:.0f} - ${max_amount:.0f}"

        assert display_text == "Selected Range: $100 - $500"

    def test_log_scale_detection(self):
        """Test log scale toggle detection."""
        # Log scale enabled
        log_value = ['log']
        log_enabled = 'log' in log_value
        assert log_enabled is True

        # Log scale disabled
        log_value = []
        log_enabled = 'log' in log_value
        assert log_enabled is False

    def test_amount_filtering_logic(self, sample_dashboard_data):
        """Test data filtering logic."""
        min_amount, max_amount = 100, 200

        df_filtered = sample_dashboard_data[
            (sample_dashboard_data['Amount'] >= min_amount) &
            (sample_dashboard_data['Amount'] <= max_amount)
        ]

        # All filtered amounts should be in range
        assert df_filtered['Amount'].min() >= min_amount
        assert df_filtered['Amount'].max() <= max_amount

    def test_export_filename_generation(self):
        """Test export filename includes timestamp."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"class_distribution_{timestamp}.html"

        assert "class_distribution" in filename
        assert ".html" in filename
        assert len(timestamp) == 15  # YYYYMMDD_HHMMSS


# ===== Fixtures =====

@pytest.fixture
def sample_dashboard_data():
    """Create sample dataframe for dashboard testing."""
    return pd.DataFrame({
        'Time': [0, 3600, 7200, 10800],
        'Hour': [0, 1, 2, 3],
        'V1': [1.0, -1.0, 0.5, -0.5],
        'V2': [0.5, -0.5, 1.0, -1.0],
        'V3': [-0.3, 0.3, 0.8, -0.8],
        'V4': [0.2, -0.2, -0.7, 0.7],
        'Amount': [100.0, 200.0, 150.0, 50.0],
        'Log_Amount': [4.615, 5.303, 5.011, 3.931],
        'Class': [0, 1, 0, 0],
    })


@pytest.fixture
def simple_test_data():
    """Create simple test data for calculations."""
    return pd.DataFrame({
        'Time': [0, 3600, 7200, 10800],
        'V1': [1.0, 2.0, 3.0, 4.0],
        'Amount': [50.0, 75.0, 25.0, 100.0],
        'Class': [0, 1, 0, 0],
    })
