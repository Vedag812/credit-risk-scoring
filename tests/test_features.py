"""
test_features.py — Unit tests for feature engineering
======================================================

Tests verify that feature engineering functions produce correct
output columns, types, and handle edge cases gracefully.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import (
    create_financial_ratios,
    create_risk_buckets,
    create_interaction_features,
    encode_categoricals,
)


@pytest.fixture
def sample_data():
    """Create a small sample DataFrame for testing."""
    return pd.DataFrame({
        "person_age": [25, 35, 55],
        "person_income": [50000, 80000, 120000],
        "person_home_ownership": ["RENT", "MORTGAGE", "OWN"],
        "person_emp_length": [2, 8, 20],
        "loan_intent": ["EDUCATION", "PERSONAL", "MEDICAL"],
        "loan_grade": ["B", "A", "C"],
        "loan_amnt": [10000, 20000, 15000],
        "loan_int_rate": [11.5, 7.5, 14.0],
        "loan_percent_income": [0.20, 0.25, 0.125],
        "cb_person_default_on_file": ["N", "N", "Y"],
        "cb_person_cred_hist_length": [4, 12, 25],
    })


def test_financial_ratios_creates_columns(sample_data):
    """Test that financial ratios creates expected columns."""
    result = create_financial_ratios(sample_data)
    expected_cols = ["loan_to_income", "income_per_emp_year",
                     "credit_hist_to_age", "interest_burden"]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_financial_ratios_no_nan(sample_data):
    """Test that financial ratios don't produce NaN values."""
    result = create_financial_ratios(sample_data)
    for col in ["loan_to_income", "income_per_emp_year",
                "credit_hist_to_age", "interest_burden"]:
        assert not result[col].isna().any(), f"NaN found in {col}"


def test_loan_to_income_positive(sample_data):
    """Loan-to-income ratio should always be positive."""
    result = create_financial_ratios(sample_data)
    assert (result["loan_to_income"] >= 0).all()


def test_risk_buckets_creates_columns(sample_data):
    """Test that risk bucketing creates expected columns."""
    result = create_risk_buckets(sample_data)
    for col in ["age_bucket", "emp_stability", "income_tier"]:
        assert col in result.columns, f"Missing column: {col}"


def test_risk_buckets_no_nulls(sample_data):
    """All rows should be assigned to a bucket."""
    result = create_risk_buckets(sample_data)
    for col in ["age_bucket", "emp_stability", "income_tier"]:
        assert not result[col].isna().any(), f"NaN in {col}"


def test_interaction_features(sample_data):
    """Test interaction feature creation."""
    pairs = [["loan_percent_income", "person_emp_length"]]
    result = create_interaction_features(sample_data, pairs)
    expected_col = "loan_percent_income_x_person_emp_length"
    assert expected_col in result.columns


def test_encode_categoricals(sample_data):
    """Test that one-hot encoding removes original categorical columns."""
    result = encode_categoricals(sample_data)
    remaining_cats = result.select_dtypes(include=["object", "category"]).columns
    assert len(remaining_cats) == 0, f"Categorical columns remaining: {list(remaining_cats)}"


def test_encode_categoricals_all_numeric(sample_data):
    """After encoding, all columns should be numeric."""
    result = encode_categoricals(sample_data)
    for col in result.columns:
        assert pd.api.types.is_numeric_dtype(result[col]), f"{col} is not numeric"


def test_original_data_not_modified(sample_data):
    """Feature engineering should NOT modify the original DataFrame."""
    original_cols = list(sample_data.columns)
    _ = create_financial_ratios(sample_data)
    assert list(sample_data.columns) == original_cols, "Original data was modified!"
