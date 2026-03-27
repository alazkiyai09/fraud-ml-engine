"""
Unit tests for data_loader module.

Tests cover data loading, validation, preprocessing, and information retrieval.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection_dashboard.data_loader import (
    EXPECTED_COLUMNS,
    get_data_info,
    load_fraud_data,
    preprocess_data,
    validate_data,
)


class TestLoadFraudData:
    """Tests for load_fraud_data function."""

    def test_load_fraud_data_success(self, sample_dataframe, tmp_path):
        """Test successful data loading."""
        # Create temporary CSV file
        csv_path = tmp_path / "creditcard.csv"
        sample_dataframe.to_csv(csv_path, index=False)

        # Load data
        df = load_fraud_data(csv_path, validate=False)

        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_dataframe)
        assert list(df.columns) == list(sample_dataframe.columns)

    def test_load_fraud_data_file_not_found(self):
        """Test loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_fraud_data("nonexistent_file.csv")

    def test_load_fraud_data_with_validation(self, valid_dataframe, tmp_path):
        """Test loading with validation enabled."""
        csv_path = tmp_path / "creditcard.csv"
        valid_dataframe.to_csv(csv_path, index=False)

        df = load_fraud_data(csv_path, validate=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_fraud_data_invalid_csv(self, tmp_path):
        """Test loading invalid CSV raises ValueError."""
        csv_path = tmp_path / "invalid.csv"
        csv_path.write_text("This is not valid CSV data")

        with pytest.raises(ValueError, match="Failed to read CSV file"):
            load_fraud_data(csv_path, validate=False)

    def test_load_fraud_data_with_pathlib(self, sample_dataframe, tmp_path):
        """Test loading with pathlib.Path object."""
        csv_path = tmp_path / "creditcard.csv"
        sample_dataframe.to_csv(csv_path, index=False)

        df = load_fraud_data(csv_path, validate=False)

        assert len(df) == len(sample_dataframe)


class TestValidateData:
    """Tests for validate_data function."""

    def test_validate_data_valid_schema(self, valid_dataframe):
        """Test validation with correct schema."""
        result = validate_data(valid_dataframe)
        assert result is True

    def test_validate_data_missing_columns(self):
        """Test validation fails with missing columns."""
        df = pd.DataFrame({'Time': [0], 'Amount': [100]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_data(df)

    def test_validate_data_invalid_class_dtype(self):
        """Test validation fails with non-numeric Class column."""
        df = pd.DataFrame({
            col: [0] for col in EXPECTED_COLUMNS
        })
        df['Class'] = ['fraud', 'legit']

        with pytest.raises(ValueError, match="'Class' column must be numeric"):
            validate_data(df)

    def test_validate_data_invalid_class_values(self, valid_dataframe):
        """Test validation fails with invalid Class values."""
        df = valid_dataframe.copy()
        df.loc[0, 'Class'] = 2

        with pytest.raises(ValueError, match="contains invalid values"):
            validate_data(df)

    def test_validate_data_missing_class_values(self):
        """Test validation fails with missing Class values."""
        df = pd.DataFrame({
            col: [0, 1, np.nan] for col in EXPECTED_COLUMNS
        })
        df['Class'] = [0, 1, np.nan]

        with pytest.raises(ValueError, match="contains missing values"):
            validate_data(df)

    def test_validate_data_missing_amount_values(self):
        """Test validation fails with missing Amount values."""
        df = pd.DataFrame({
            col: [0, 1, 2] for col in EXPECTED_COLUMNS
        })
        df.loc[2, 'Amount'] = np.nan

        with pytest.raises(ValueError, match="contains missing values"):
            validate_data(df)


class TestPreprocessData:
    """Tests for preprocess_data function."""

    def test_preprocess_data_basic(self, sample_dataframe):
        """Test basic preprocessing."""
        result = preprocess_data(sample_dataframe, normalize_time=False)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)
        assert 'Log_Amount' in result.columns
        assert 'Hour' not in result.columns

    def test_preprocess_data_with_time_normalization(self, sample_dataframe):
        """Test preprocessing with time normalization."""
        # Ensure Time column exists
        if 'Time' not in sample_dataframe.columns:
            sample_dataframe['Time'] = [0, 3600, 7200]

        result = preprocess_data(sample_dataframe, normalize_time=True)

        assert 'Hour' in result.columns
        assert result['Hour'].min() >= 0
        assert result['Hour'].max() <= 23

    def test_preprocess_data_creates_log_amount(self, sample_dataframe):
        """Test that Log_Amount column is created correctly."""
        result = preprocess_data(sample_dataframe)

        assert 'Log_Amount' in result.columns
        # Verify log transformation: log(1 + x)
        expected_log = np.log1p(sample_dataframe['Amount'].iloc[0])
        assert abs(result['Log_Amount'].iloc[0] - expected_log) < 1e-6

    def test_preprocess_data_handles_missing_values(self, tmp_path):
        """Test that rows with missing values are removed."""
        df = pd.DataFrame({
            'Time': [0, 3600, 7200],
            'V1': [1.0, np.nan, 3.0],
            'Amount': [100, 200, 300],
            'Class': [0, 1, 0],
        })
        # Add remaining expected columns
        for col in EXPECTED_COLUMNS - {'Time', 'V1', 'Amount', 'Class'}:
            df[col] = 0.0

        result = preprocess_data(df, normalize_time=False)

        # Should remove the row with NaN
        assert len(result) == 2
        assert not result['V1'].isnull().any()

    def test_preprocess_data_class_dtype_conversion(self, sample_dataframe):
        """Test that Class column is converted to integer."""
        sample_dataframe['Class'] = sample_dataframe['Class'].astype(float)
        result = preprocess_data(sample_dataframe)

        assert result['Class'].dtype == np.int64

    def test_preprocess_data_copy_not_modified(self, sample_dataframe):
        """Test that original DataFrame is not modified."""
        original_columns = set(sample_dataframe.columns)
        preprocess_data(sample_dataframe, normalize_time=True)

        assert set(sample_dataframe.columns) == original_columns


class TestGetDataInfo:
    """Tests for get_data_info function."""

    def test_get_data_info_basic(self, sample_dataframe):
        """Test basic data info retrieval."""
        info = get_data_info(sample_dataframe)

        assert isinstance(info, dict)
        assert 'shape' in info
        assert 'memory_usage' in info
        assert 'column_types' in info
        assert 'has_missing' in info
        assert 'missing_count' in info

    def test_get_data_info_shape(self, sample_dataframe):
        """Test shape calculation."""
        info = get_data_info(sample_dataframe)

        assert info['shape'] == sample_dataframe.shape

    def test_get_data_info_missing_values(self):
        """Test missing value detection."""
        df = pd.DataFrame({
            'Time': [0, 1],
            'Amount': [100, np.nan],
            'Class': [0, 1],
        })
        # Add remaining columns
        for col in EXPECTED_COLUMNS - {'Time', 'Amount', 'Class'}:
            df[col] = 0.0

        info = get_data_info(df)

        assert info['has_missing'] is True
        assert info['missing_count']['Amount'] == 1

    def test_get_data_info_no_missing_values(self, sample_dataframe):
        """Test with no missing values."""
        info = get_data_info(sample_dataframe)

        assert info['has_missing'] is False


# ===== Fixtures =====

@pytest.fixture
def sample_dataframe():
    """Create a minimal sample dataframe for testing."""
    return pd.DataFrame({
        'Time': [0, 3600, 7200],
        'V1': [1.0, -1.0, 0.5],
        'V2': [0.5, -0.5, 1.0],
        'Amount': [100.0, 200.0, 150.0],
        'Class': [0, 1, 0],
    })


@pytest.fixture
def valid_dataframe():
    """Create a valid dataframe with all expected columns."""
    # Create minimal valid dataframe with all expected columns
    data = {col: [0.0, 1.0, 2.0] for col in EXPECTED_COLUMNS - {'Class'}}
    data['Class'] = [0, 1, 0]
    return pd.DataFrame(data)
