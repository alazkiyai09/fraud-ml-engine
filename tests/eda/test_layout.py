"""
Unit tests for layout module.

Tests cover layout component creation and structure validation.
"""

import pytest
from dash import html

from fraud_detection_dashboard.layout import (
    create_charts_grid,
    create_dashboard_layout,
    create_filters,
    create_header,
    create_summary_card,
)


class TestCreateHeader:
    """Tests for create_header function."""

    def test_create_header_returns_div(self):
        """Test that header returns a Div component."""
        header = create_header()
        assert isinstance(header, html.Div)

    def test_create_header_has_title(self):
        """Test that header contains title."""
        header = create_header()
        # Header should contain H1 element
        assert any(isinstance(child, html.H1) for child in header.children)

    def test_create_header_styling(self):
        """Test header has proper styling attributes."""
        header = create_header()
        assert 'style' in header.keys()
        assert 'backgroundColor' in header['style']


class TestCreateSummaryCard:
    """Tests for create_summary_card function."""

    def test_create_summary_card_returns_div(self, sample_stats):
        """Test that summary card returns a Div component."""
        card = create_summary_card(sample_stats)
        assert isinstance(card, html.Div)

    def test_create_summary_card_contains_stats(self, sample_stats):
        """Test that summary card displays statistics."""
        card = create_summary_card(sample_stats)
        # Should contain the total transactions number
        card_html = str(card)
        assert str(sample_stats['total_transactions']) in card_html

    def test_create_summary_card_fraud_display(self, sample_stats):
        """Test that fraud count is displayed correctly."""
        card = create_summary_card(sample_stats)
        card_html = str(card)
        # Should show fraud count
        assert str(sample_stats['fraud_count']) in card_html

    def test_create_summary_card_empty_stats(self):
        """Test handling of empty statistics dictionary."""
        empty_stats = {
            'total_transactions': 0,
            'fraud_count': 0,
            'legit_count': 0,
            'fraud_percentage': 0,
            'total_amount': 0,
            'avg_amount': 0,
            'avg_fraud_amount': 0,
            'avg_legit_amount': 0,
        }
        card = create_summary_card(empty_stats)
        assert isinstance(card, html.Div)


class TestCreateFilters:
    """Tests for create_filters function."""

    def test_create_filters_returns_div(self):
        """Test that filters returns a Div component."""
        filters = create_filters()
        assert isinstance(filters, html.Div)

    def test_create_filters_has_slider(self):
        """Test that filters contains range slider component."""
        filters = create_filters()
        filters_str = str(filters)
        # Should contain amount range slider
        assert 'amount-range-slider' in filters_str

    def test_create_filters_has_toggle(self):
        """Test that filters contains log scale toggle."""
        filters = create_filters()
        filters_str = str(filters)
        assert 'log-scale-toggle' in filters_str

    def test_create_filters_has_export_button(self):
        """Test that filters contains export button."""
        filters = create_filters()
        filters_str = str(filters)
        assert 'export-button' in filters_str


class TestCreateChartsGrid:
    """Tests for create_charts_grid function."""

    def test_create_charts_grid_returns_div(self):
        """Test that charts grid returns a Div component."""
        grid = create_charts_grid()
        assert isinstance(grid, html.Div)

    def test_create_charts_grid_has_all_charts(self):
        """Test that grid contains all chart placeholders."""
        grid = create_charts_grid()
        grid_str = str(grid)

        # Should contain all chart IDs
        chart_ids = [
            'class-dist-chart',
            'amount-hist-chart',
            'correlation-heatmap',
            'time-patterns-chart',
            'pca-scatter-chart',
        ]

        for chart_id in chart_ids:
            assert chart_id in grid_str


class TestCreateDashboardLayout:
    """Tests for create_dashboard_layout function."""

    def test_create_dashboard_layout_returns_div(self, sample_stats):
        """Test that dashboard layout returns a Div component."""
        layout = create_dashboard_layout(sample_stats)
        assert isinstance(layout, html.Div)

    def test_create_dashboard_layout_contains_header(self, sample_stats):
        """Test that layout contains header section."""
        layout = create_dashboard_layout(sample_stats)
        layout_str = str(layout)
        # Should contain header-related content
        assert 'Credit Card Fraud Detection' in layout_str

    def test_create_dashboard_layout_contains_summary(self, sample_stats):
        """Test that layout contains summary section."""
        layout = create_dashboard_layout(sample_stats)
        layout_str = str(layout)
        assert 'Summary Statistics' in layout_str

    def test_create_dashboard_layout_contains_filters(self, sample_stats):
        """Test that layout contains filters section."""
        layout = create_dashboard_layout(sample_stats)
        layout_str = str(layout)
        assert 'Interactive Filters' in layout_str

    def test_create_dashboard_layout_contains_charts(self, sample_stats):
        """Test that layout contains charts section."""
        layout = create_dashboard_layout(sample_stats)
        layout_str = str(layout)
        # Should contain chart IDs
        assert 'class-dist-chart' in layout_str

    def test_create_dashboard_layout_styling(self, sample_stats):
        """Test that layout has proper styling."""
        layout = create_dashboard_layout(sample_stats)
        assert 'style' in layout.keys()
        assert 'backgroundColor' in layout['style']


# ===== Fixtures =====

@pytest.fixture
def sample_stats():
    """Create sample statistics dictionary."""
    return {
        'total_transactions': 284807,
        'fraud_count': 492,
        'legit_count': 284315,
        'fraud_percentage': 0.17,
        'total_amount': 59602345.57,
        'avg_amount': 88.35,
        'avg_fraud_amount': 122.21,
        'avg_legit_amount': 88.29,
    }
