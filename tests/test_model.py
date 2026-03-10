"""
test_model.py — Unit tests for model training and evaluation
=============================================================

Tests verify that model training produces valid metrics
and that the full pipeline runs without errors.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_config, load_and_clean
from src.model import calculate_gini, calculate_ks_statistic
from src.scorecard import probability_to_score, get_risk_category


@pytest.fixture
def config():
    return load_config()


class TestMetrics:
    """Test evaluation metric calculations."""

    def test_gini_perfect_model(self):
        """Perfect model should have Gini = 1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        gini = calculate_gini(y_true, y_prob)
        assert gini == 1.0

    def test_gini_random_model(self):
        """Random model should have Gini close to 0."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_prob = np.random.random(1000)
        gini = calculate_gini(y_true, y_prob)
        assert -0.15 < gini < 0.15

    def test_gini_range(self):
        """Gini should be between -1 and 1."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.3, 0.7, 0.4, 0.6, 0.5, 0.8])
        gini = calculate_gini(y_true, y_prob)
        assert -1 <= gini <= 1

    def test_ks_perfect_separation(self):
        """Perfectly separated data should have KS = 1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        ks = calculate_ks_statistic(y_true, y_prob)
        assert ks == 1.0

    def test_ks_range(self):
        """KS should be between 0 and 1."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.3, 0.7, 0.4, 0.6, 0.5, 0.8])
        ks = calculate_ks_statistic(y_true, y_prob)
        assert 0 <= ks <= 1


class TestScorecard:
    """Test scorecard conversion logic."""

    def test_low_pd_high_score(self, config):
        """Low probability of default should give high credit score."""
        score = probability_to_score(0.01, config)
        assert score > 700

    def test_high_pd_low_score(self, config):
        """High probability of default should give low credit score."""
        score = probability_to_score(0.80, config)
        assert score < 550

    def test_score_monotonicity(self, config):
        """Higher PD should always give lower score."""
        pds = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
        scores = [probability_to_score(p, config) for p in pds]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1], (
                f"Score not monotonic: PD={pds[i]} score={scores[i]} vs "
                f"PD={pds[i+1]} score={scores[i+1]}"
            )

    def test_risk_categories(self):
        """Risk categories should map correctly."""
        assert get_risk_category(800) == "Excellent"
        assert get_risk_category(720) == "Good"
        assert get_risk_category(670) == "Fair"
        assert get_risk_category(620) == "Poor"
        assert get_risk_category(500) == "Very Poor"


class TestDataLoader:
    """Test data loading and cleaning."""

    def test_load_produces_dataframe(self, config):
        """Loading should return a pandas DataFrame."""
        df = load_and_clean(config)
        assert isinstance(df, pd.DataFrame)

    def test_target_column_exists(self, config):
        """Dataset must have 'loan_status' target column."""
        df = load_and_clean(config)
        assert "loan_status" in df.columns

    def test_target_is_binary(self, config):
        """Target column should only contain 0 and 1."""
        df = load_and_clean(config)
        assert set(df["loan_status"].unique()).issubset({0, 1})

    def test_no_missing_values(self, config):
        """After cleaning, there should be no missing values."""
        df = load_and_clean(config)
        assert df.isnull().sum().sum() == 0
