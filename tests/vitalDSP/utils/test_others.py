import pytest
import numpy as np
from vitalDSP.utils.signal_processing.normalization import z_score_normalization, min_max_normalization


def test_z_score_normalization():
    # Test Z-Score normalization on a simple array
    signal = np.array([1, 2, 3, 4, 5])

    # Calculate expected output programmatically using the same formula
    mean = np.mean(signal)
    std = np.std(signal)
    expected_output = (signal - mean) / std

    normalized_signal = z_score_normalization(signal)

    # Ensure the output is correct using the programmatically calculated expected output
    np.testing.assert_almost_equal(normalized_signal, expected_output, decimal=6)

    # Test with a single value (edge case)
    signal_single = np.array([5])
    normalized_single = z_score_normalization(signal_single)
    expected_single = np.array([0.0])

    assert np.all(normalized_single == expected_single)

    # Test with a constant signal (edge case, std will be 0, expect zeros)
    signal_constant = np.array([3, 3, 3, 3, 3])
    normalized_constant = z_score_normalization(signal_constant)
    expected_constant = np.zeros_like(signal_constant)

    assert np.all(normalized_constant == expected_constant)


def test_min_max_normalization():
    # Test Min-Max normalization on a simple array with default range [0, 1]
    signal = np.array([1, 2, 3, 4, 5])
    normalized_signal = min_max_normalization(signal)
    expected_output = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    np.testing.assert_almost_equal(normalized_signal, expected_output, decimal=6)

    # Test Min-Max normalization with a custom range [-1, 1]
    normalized_signal_custom = min_max_normalization(signal, min_value=-1, max_value=1)
    expected_output_custom = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    np.testing.assert_almost_equal(
        normalized_signal_custom, expected_output_custom, decimal=6
    )

    # Test with a single value (edge case)
    signal_single = np.array([5])
    normalized_single = min_max_normalization(signal_single)
    expected_single = np.array(
        [0.0]
    )  # Min-Max normalization of a single value will return 0 in default range

    assert np.all(normalized_single == expected_single)

    # Test with a constant signal (edge case, min and max will be the same, expect zeros)
    signal_constant = np.array([3, 3, 3, 3, 3])
    normalized_constant = min_max_normalization(signal_constant)
    expected_constant = np.zeros_like(signal_constant)

    assert np.all(normalized_constant == expected_constant)
