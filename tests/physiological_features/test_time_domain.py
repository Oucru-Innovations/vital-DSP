import pytest
import numpy as np
from vitalDSP.physiological_features.time_domain import TimeDomainFeatures

@pytest.fixture
def sample_nn_intervals():
    # Provides a fixture for testing
    return [800, 810, 790, 805, 795, 820, 780]

def test_sdnn(sample_nn_intervals):
    tdf = TimeDomainFeatures(sample_nn_intervals)
    result = tdf.compute_sdnn()
    expected = np.std(sample_nn_intervals)
    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_rmssd(sample_nn_intervals):
    tdf = TimeDomainFeatures(sample_nn_intervals)
    result = tdf.compute_rmssd()
    diff_nn_intervals = np.diff(sample_nn_intervals)
    expected = np.sqrt(np.mean(diff_nn_intervals**2))
    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_nn50(sample_nn_intervals):
    tdf = TimeDomainFeatures(sample_nn_intervals)
    result = tdf.compute_nn50()
    diff_nn_intervals = np.abs(np.diff(sample_nn_intervals))
    expected = np.sum(diff_nn_intervals > 50)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_pnn50(sample_nn_intervals):
    tdf = TimeDomainFeatures(sample_nn_intervals)
    result = tdf.compute_pnn50()
    nn50 = tdf.compute_nn50()
    expected = 100.0 * nn50 / len(sample_nn_intervals)
    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_median_nn(sample_nn_intervals):
    tdf = TimeDomainFeatures(sample_nn_intervals)
    result = tdf.compute_median_nn()
    expected = np.median(sample_nn_intervals)
    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_iqr_nn(sample_nn_intervals):
    tdf = TimeDomainFeatures(sample_nn_intervals)
    result = tdf.compute_iqr_nn()
    expected = np.percentile(sample_nn_intervals, 75) - np.percentile(sample_nn_intervals, 25)
    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_mean_nn(sample_nn_intervals):
    tdf = TimeDomainFeatures(sample_nn_intervals)
    result = tdf.compute_mean_nn()
    expected = np.mean(sample_nn_intervals)
    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_std_nn(sample_nn_intervals):
    tdf = TimeDomainFeatures(sample_nn_intervals)
    result = tdf.compute_std_nn()  # This calls compute_sdnn()
    expected = tdf.compute_sdnn()
    assert np.isclose(result, expected), f"Expected {expected}, but got {result}"
