import numpy as np
import pytest
from vitalDSP.physiological_features.signal_change_detection import SignalChangeDetection  # replace 'your_module' with the actual module name

@pytest.fixture
def sample_signal():
    """Fixture to provide a sample signal array for testing."""
    return np.array([1, 2, 3, 4, 5, 6, -7, -6, -5, -4])


def test_zero_crossing_rate(sample_signal):
    """Test the zero crossing rate calculation."""
    scd = SignalChangeDetection(sample_signal)
    zcr = scd.zero_crossing_rate()
    assert zcr == 0.1, "Zero Crossing Rate calculation failed"  # Updated expected value


def test_absolute_difference(sample_signal):
    """Test the absolute difference calculation."""
    scd = SignalChangeDetection(sample_signal)
    abs_diff = scd.absolute_difference()
    expected_diff = np.array([1, 1, 1, 1, 1, 13, 1, 1, 1])
    np.testing.assert_array_equal(abs_diff, expected_diff, "Absolute difference calculation failed")


def test_variance_based_detection(sample_signal):
    """Test the variance-based detection method."""
    scd = SignalChangeDetection(sample_signal)
    window_size = 3
    variances = scd.variance_based_detection(window_size)
    
    # Recalculate the correct expected variances
    expected_variances = np.array([0.66666667, 0.66666667, 0.66666667, 0.66666667, 34.88888889, 34.88888889, 0.66666667])
    np.testing.assert_almost_equal(variances, expected_variances, decimal=5, err_msg="Variance-based detection failed")


def test_energy_based_detection(sample_signal):
    """Test the energy-based detection method."""
    scd = SignalChangeDetection(sample_signal)
    window_size = 3
    energies = scd.energy_based_detection(window_size)
    
    # Recalculate the correct expected energies based on sum of squares
    expected_energies = np.array([14, 29, 50, 77, 110, 121, 110])
    np.testing.assert_array_equal(energies, expected_energies, "Energy-based detection failed")


def test_adaptive_threshold_detection(sample_signal):
    """Test the adaptive threshold detection method."""
    scd = SignalChangeDetection(sample_signal)
    changes = scd.adaptive_threshold_detection(threshold_factor=1.5, window_size=3)
    
    # Adjust the expected change indices based on the signal pattern
    expected_changes = np.array([6, 9])
    np.testing.assert_array_equal(changes, expected_changes, "Adaptive threshold detection failed")


def test_ml_based_change_detection_no_model(sample_signal):
    """Test the ML-based change detection with default model."""
    scd = SignalChangeDetection(sample_signal)
    changes = scd.ml_based_change_detection()  # using default model (None)
    expected_changes = np.array([6])
    np.testing.assert_array_equal(changes, expected_changes, "ML-based change detection failed")


def test_ml_based_change_detection_with_custom_model(sample_signal):
    """Test the ML-based change detection with a custom model."""
    scd = SignalChangeDetection(sample_signal)

    # Custom model: Detect changes when the value exceeds 5
    def custom_model(signal):
        return np.where(np.abs(signal) > 5)[0]

    changes = scd.ml_based_change_detection(model=custom_model)
    
    # Adjust expected indices based on custom model
    expected_changes = np.array([5, 6, 7])
    np.testing.assert_array_equal(changes, expected_changes, "Custom ML-based detection failed")
