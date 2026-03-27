"""Unit tests for report generation."""

import pytest
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from reports.generator import ReportGenerator


@pytest.fixture
def sample_risk_factors():
    """Sample risk factors for testing."""
    return [
        {
            'feature': 'transaction_amount',
            'description': 'Transaction Amount',
            'importance': 0.8456,
            'abs_importance': 0.8456,
            'direction': 'increases',
            'impact_level': 'Very High'
        },
        {
            'feature': 'is_international',
            'description': 'Is International Transaction',
            'importance': 0.5234,
            'abs_importance': 0.5234,
            'direction': 'increases',
            'impact_level': 'High'
        },
        {
            'feature': 'hour_of_day',
            'description': 'Hour of Day',
            'importance': -0.3421,
            'abs_importance': 0.3421,
            'direction': 'decreases',
            'impact_level': 'Medium'
        }
    ]


@pytest.fixture
def sample_global_importance():
    """Sample global importance data."""
    return [
        {
            'feature': 'transaction_amount',
            'description': 'Transaction Amount',
            'abs_importance': 0.75
        },
        {
            'feature': 'is_international',
            'description': 'Is International Transaction',
            'abs_importance': 0.52
        },
        {
            'feature': 'hour_of_day',
            'description': 'Hour of Day',
            'abs_importance': 0.34
        }
    ]


@pytest.fixture
def sample_model_metadata():
    """Sample model metadata."""
    return {
        'name': 'XGBoost Fraud Detection Model',
        'version': '2.1.0',
        'type': 'xgboost',
        'training_date': '2024-01-15',
        'last_validated': '2024-01-20'
    }


class TestReportGenerator:
    """Test suite for ReportGenerator."""

    def test_initialization(self):
        """Test ReportGenerator initialization."""
        generator = ReportGenerator()
        assert generator.template_dir is not None

    def test_initialization_custom_template_dir(self, tmp_path):
        """Test initialization with custom template directory."""
        generator = ReportGenerator(template_dir=str(tmp_path))
        assert generator.template_dir == tmp_path

    def test_generate_html_report_basic(self, sample_risk_factors, sample_model_metadata):
        """Test basic HTML report generation."""
        generator = ReportGenerator()

        html = generator.generate_html_report(
            transaction_id='TXN-12345',
            prediction=0.85,
            predicted_class='Fraud',
            risk_factors=sample_risk_factors,
            global_importance=None,
            model_metadata=sample_model_metadata
        )

        assert isinstance(html, str)
        assert len(html) > 0
        assert '<!DOCTYPE html>' in html
        assert 'TXN-12345' in html
        assert '85.00%' in html
        assert 'Fraud' in html

    def test_generate_html_report_with_global_importance(
        self, sample_risk_factors, sample_global_importance, sample_model_metadata
    ):
        """Test report generation with global importance."""
        generator = ReportGenerator()

        html = generator.generate_html_report(
            transaction_id='TXN-67890',
            prediction=0.72,
            predicted_class='Fraud',
            risk_factors=sample_risk_factors,
            global_importance=sample_global_importance,
            model_metadata=sample_model_metadata
        )

        assert 'Global Feature Importance' in html
        assert 'Transaction Amount' in html

    def test_generate_html_report_contains_risk_factors(
        self, sample_risk_factors, sample_model_metadata
    ):
        """Test that report contains all risk factors."""
        generator = ReportGenerator()

        html = generator.generate_html_report(
            transaction_id='TXN-11111',
            prediction=0.65,
            predicted_class='Fraud',
            risk_factors=sample_risk_factors,
            model_metadata=sample_model_metadata
        )

        # Check each risk factor is present
        for factor in sample_risk_factors:
            assert factor['description'] in html
            assert str(factor['importance']) in html
            assert factor['direction'] in html

    def test_generate_html_report_contains_model_info(
        self, sample_risk_factors, sample_model_metadata
    ):
        """Test that report contains model information."""
        generator = ReportGenerator()

        html = generator.generate_html_report(
            transaction_id='TXN-22222',
            prediction=0.45,
            predicted_class='Legitimate',
            risk_factors=sample_risk_factors,
            model_metadata=sample_model_metadata
        )

        assert 'Model Information' in html
        assert sample_model_metadata['name'] in html
        assert sample_model_metadata['version'] in html
        assert sample_model_metadata['type'] in html

    def test_generate_html_report_contains_compliance_info(
        self, sample_risk_factors, sample_model_metadata
    ):
        """Test that report contains regulatory compliance information."""
        generator = ReportGenerator()

        html = generator.generate_html_report(
            transaction_id='TXN-33333',
            prediction=0.55,
            predicted_class='Fraud',
            risk_factors=sample_risk_factors,
            model_metadata=sample_model_metadata
        )

        assert 'Regulatory Compliance' in html
        assert 'SR 11-7' in html
        assert 'EU AI Act' in html

    def test_generate_html_report_different_risk_levels(
        self, sample_risk_factors, sample_model_metadata
    ):
        """Test report generation with different risk levels."""
        generator = ReportGenerator()

        # Test Critical risk level
        html_critical = generator.generate_html_report(
            transaction_id='TXN-CRITICAL',
            prediction=0.95,
            predicted_class='Fraud',
            risk_factors=sample_risk_factors,
            model_metadata=sample_model_metadata
        )
        assert 'Critical' in html_critical

        # Test Low risk level
        html_low = generator.generate_html_report(
            transaction_id='TXN-LOW',
            prediction=0.15,
            predicted_class='Legitimate',
            risk_factors=sample_risk_factors,
            model_metadata=sample_model_metadata
        )
        assert 'Very Low' in html_low

    def test_save_report(self, sample_risk_factors, sample_model_metadata, tmp_path):
        """Test saving report to file."""
        generator = ReportGenerator()

        html = generator.generate_html_report(
            transaction_id='TXN-SAVE',
            prediction=0.75,
            predicted_class='Fraud',
            risk_factors=sample_risk_factors,
            model_metadata=sample_model_metadata
        )

        output_path = tmp_path / 'test_report.html'
        generator.save_report(html, str(output_path))

        assert output_path.exists()

        # Verify content
        with open(output_path, 'r') as f:
            saved_html = f.read()
        assert saved_html == html

    def test_save_report_creates_directories(
        self, sample_risk_factors, sample_model_metadata, tmp_path
    ):
        """Test that save_report creates parent directories."""
        generator = ReportGenerator()

        html = generator.generate_html_report(
            transaction_id='TXN-DIR',
            prediction=0.65,
            predicted_class='Fraud',
            risk_factors=sample_risk_factors,
            model_metadata=sample_model_metadata
        )

        # Create path with non-existent subdirectories
        output_path = tmp_path / 'reports' / '2024' / '01' / 'report.html'
        generator.save_report(html, str(output_path))

        assert output_path.exists()

    def test_get_risk_level(self, sample_risk_factors, sample_model_metadata):
        """Test risk level categorization."""
        generator = ReportGenerator()

        # Test various probabilities
        test_cases = [
            (0.85, 'Critical'),
            (0.65, 'High'),
            (0.45, 'Medium'),
            (0.25, 'Low'),
            (0.10, 'Very Low')
        ]

        for prob, expected_level in test_cases:
            html = generator.generate_html_report(
                transaction_id=f'TXN-{int(prob*100)}',
                prediction=prob,
                predicted_class='Fraud' if prob >= 0.5 else 'Legitimate',
                risk_factors=sample_risk_factors,
                model_metadata=sample_model_metadata
            )
            assert expected_level in html

    def test_report_contains_css_styles(self, sample_risk_factors, sample_model_metadata):
        """Test that report includes CSS styling."""
        generator = ReportGenerator()

        html = generator.generate_html_report(
            transaction_id='TXN-CSS',
            prediction=0.70,
            predicted_class='Fraud',
            risk_factors=sample_risk_factors,
            model_metadata=sample_model_metadata
        )

        assert '<style>' in html
        assert '</style>' in html
        assert 'container' in html
        assert 'risk-factor-card' in html


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
