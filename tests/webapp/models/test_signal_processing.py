import pytest
import numpy as np
from vitalDSP_webapp.models.signal_processing import filter_signal


# Test for empty input data
def test_filter_signal_empty():
    data = np.array([])  # Empty data array
    filtered_data = filter_signal(data)

    # Ensure the function returns an empty array when given empty input
    assert filtered_data.size == 0, "Filtering empty data should return an empty array"


# Test for basic input data
def test_filter_signal_basic():
    data = np.array([1, 2, 3, 4, 5])  # Simple test data
    filtered_data = filter_signal(data)

    # Since no real filtering is done in the placeholder, the output should match the input
    np.testing.assert_array_equal(
        filtered_data, data, "The filtered data should match the input data"
    )


# Test for large input data
def test_filter_signal_large():
    data = np.random.rand(10000)  # Large random data array
    filtered_data = filter_signal(data)

    # Check that the output is the same as the input for this placeholder function
    np.testing.assert_array_equal(
        filtered_data, data, "The filtered data should match the input for large arrays"
    )


# Test for handling NaN values in input
def test_filter_signal_with_nan():
    data = np.array([1, 2, np.nan, 4, 5])  # Data with NaN values
    filtered_data = filter_signal(data)

    # Ensure the function correctly handles NaN values (in this case, doesn't modify them)
    np.testing.assert_array_equal(
        filtered_data, data, "The filtered data should preserve NaN values"
    )
