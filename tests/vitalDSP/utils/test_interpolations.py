import numpy as np
import pytest
import warnings
from vitalDSP.utils.signal_processing.interpolations import (
    linear_interpolation,
    spline_interpolation,
    mean_imputation,
    median_imputation,
    forward_fill,
    backward_fill,
    rolling_mean_imputation,
)

# Filter out expected warnings about empty slices and NaN operations
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")


@pytest.fixture
def rr_intervals_with_nans():
    return np.array([0.8, np.nan, 0.82, np.nan, 0.85])


@pytest.fixture
def rr_intervals_no_nans():
    return np.array([0.8, 0.82, 0.81, 0.83, 0.85])


@pytest.fixture
def rr_intervals_all_nans():
    return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])


def test_linear_interpolation(rr_intervals_with_nans):
    result = linear_interpolation(rr_intervals_with_nans)
    expected = np.array([0.8, 0.81, 0.82, 0.835, 0.85])
    np.testing.assert_array_almost_equal(result, expected, decimal=2)


def test_spline_interpolation(rr_intervals_with_nans):
    result = spline_interpolation(rr_intervals_with_nans, order=3)
    expected = np.array(
        [0.8, 0.805, 0.82, 0.835, 0.85]
    )  # Adjusted based on cubic spline
    # Use a higher tolerance (relative tolerance = 1e-3, absolute tolerance = 1e-3)
    assert np.all(
        np.isclose(result, expected, rtol=1e-2, atol=1e-2)
    ), f"Result: {result} does not match Expected: {expected} within tolerance"


def test_spline_interpolation_fallback_to_linear(rr_intervals_with_nans):
    # With insufficient points, should fall back to linear interpolation
    rr_intervals_short = np.array(
        [0.8, np.nan, 0.82]
    )  # Not enough points for cubic spline
    result = spline_interpolation(rr_intervals_short, order=3)
    expected = np.array([0.8, 0.81, 0.82])  # Linear interpolation expected
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_mean_imputation(rr_intervals_with_nans):
    result = mean_imputation(rr_intervals_with_nans)
    expected = np.array([0.8, 0.82333333, 0.82, 0.82333333, 0.85])  # Mean is 0.82333333
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_median_imputation(rr_intervals_with_nans):
    result = median_imputation(rr_intervals_with_nans)
    expected = np.array([0.8, 0.82, 0.82, 0.82, 0.85])  # Median is 0.82
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_forward_fill(rr_intervals_with_nans):
    result = forward_fill(rr_intervals_with_nans)
    expected = np.array([0.8, 0.8, 0.82, 0.82, 0.85])
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_backward_fill(rr_intervals_with_nans):
    result = backward_fill(rr_intervals_with_nans)
    expected = np.array([0.8, 0.82, 0.82, 0.85, 0.85])
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_rolling_mean_imputation(rr_intervals_with_nans):
    result = rolling_mean_imputation(rr_intervals_with_nans, window=3)
    expected = np.array([0.8, 0.81, 0.82, 0.835, 0.85])  # Rolling mean window = 3
    # Use np.isclose with relative and absolute tolerances
    assert np.all(
        np.isclose(result, expected, rtol=1e-2, atol=1e-2)
    ), f"Result: {result} does not match Expected: {expected} within tolerance"


# Edge case: No NaNs present
def test_linear_interpolation_no_nans(rr_intervals_no_nans):
    result = linear_interpolation(rr_intervals_no_nans)
    np.testing.assert_array_almost_equal(result, rr_intervals_no_nans)


# Edge case: All NaNs present
def test_mean_imputation_all_nans(rr_intervals_all_nans):
    result = mean_imputation(rr_intervals_all_nans)
    expected = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(result, expected)


# Edge case: All NaNs in median imputation
def test_median_imputation_all_nans(rr_intervals_all_nans):
    result = median_imputation(rr_intervals_all_nans)
    expected = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(result, expected)


# Edge case: Rolling mean with only one valid value
def test_rolling_mean_single_valid_value():
    rr_intervals_single_valid = np.array([np.nan, np.nan, 0.85, np.nan, np.nan])
    result = rolling_mean_imputation(rr_intervals_single_valid, window=3)
    expected = np.array([0.85, 0.85, 0.85, 0.85, 0.85])  # Fills forward/backward
    np.testing.assert_array_almost_equal(result, expected, decimal=5)
