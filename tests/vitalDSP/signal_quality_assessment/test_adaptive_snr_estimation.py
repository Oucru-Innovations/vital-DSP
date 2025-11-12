import numpy as np
import pytest
from vitalDSP.signal_quality_assessment.adaptive_snr_estimation import (
    sliding_window_snr,
    adaptive_threshold_snr,
    recursive_snr_estimation,
)


# Test for sliding_window_snr
def test_sliding_window_snr():
    # Test with a simple sine wave signal
    signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01)) + 0.1 * np.random.normal(
        size=1000
    )

    # Check default window size and step size
    snr_estimates = sliding_window_snr(signal)
    assert len(snr_estimates) > 0
    assert isinstance(snr_estimates, np.ndarray)

    # Check for smaller window size and step size
    snr_estimates = sliding_window_snr(signal, window_size=50, step_size=25)
    assert len(snr_estimates) > 0

    # Check for edge case where window_size is larger than the signal length
    short_signal = signal[:10]
    snr_estimates = sliding_window_snr(short_signal, window_size=50, step_size=25)
    assert len(snr_estimates) == 0  # No SNR estimates as window is larger than signal


def test_adaptive_threshold_snr():
    signal = np.sin(
        2 * np.pi * 0.2 * np.arange(0, 1000, 0.01)
    ) + 0.1 * np.random.normal(size=100000)
    snr_estimate = adaptive_threshold_snr(signal, threshold=0.3)
    assert np.isfinite(snr_estimate)  # SNR should be finite for this signal


def test_adaptive_threshold_snr_edge_case_no_noise():
    # Test with a signal that crosses the threshold
    signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01)) + 0.1 * np.random.normal(
        size=1000
    )

    # Test default threshold
    snr_estimate = adaptive_threshold_snr(signal)
    assert isinstance(snr_estimate, float)

    # Test with a lower threshold
    snr_estimate = adaptive_threshold_snr(signal, threshold=0.2)
    assert snr_estimate > 0

    # Test with a threshold that excludes the entire signal
    # snr_estimate = adaptive_threshold_snr(signal, threshold=10)
    # assert snr_estimate == float('-inf')  # No signal segment above threshold

    # # Test with a threshold that includes the entire signal
    # snr_estimate = adaptive_threshold_snr(signal, threshold=0)
    # assert snr_estimate > 0


def test_adaptive_threshold_snr_edge_case_all_noise():
    signal = np.random.normal(0, 0.1, 1000)  # Pure noise
    snr_estimate = adaptive_threshold_snr(signal, threshold=0.5)
    assert np.isinf(
        snr_estimate
    )  # Signal to noise ratio should return inf when there's only noise (signal is near zero)


def test_recursive_snr_estimation():
    # Test with a simple sine wave signal
    signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01)) + 0.1 * np.random.normal(
        size=1000
    )

    # Test with default alpha value
    snr_estimates = recursive_snr_estimation(signal)
    assert len(snr_estimates) == len(signal)
    assert isinstance(snr_estimates, np.ndarray)

    # Test with different alpha value (higher smoothing)
    snr_estimates = recursive_snr_estimation(signal, alpha=0.99)
    assert len(snr_estimates) == len(signal)

    # Test with different alpha value (less smoothing)
    snr_estimates = recursive_snr_estimation(signal, alpha=0.5)
    assert len(snr_estimates) == len(signal)

    # Edge case: constant signal
    constant_signal = np.ones(1000)
    snr_estimates = recursive_snr_estimation(constant_signal)
    assert np.all(np.isinf(snr_estimates))  # Inf SNR since noise is zero

    # Edge case: signal is all noise (random)
    random_signal = np.random.normal(size=1000)
    snr_estimates = recursive_snr_estimation(random_signal)
    assert np.any(snr_estimates)  # Should not be all zeros or infinities


def test_recursive_snr_estimation_edge_case_constant_signal():
    signal = np.ones(1000)  # Constant signal (no variation)
    snr_estimates = recursive_snr_estimation(signal, alpha=0.9)
    assert np.all(
        np.isinf(snr_estimates)
    )  # With a constant signal, noise power will be zero, resulting in infinite SNR


def test_sliding_window_snr_zero_noise_power():
    """Test sliding_window_snr when noise_power is zero.
    
    This test covers line 60 in adaptive_snr_estimation.py where
    snr = float("inf") when noise_power == 0.
    """
    # Create a signal with constant windows (zero variance = zero noise power)
    # Use a signal that has constant values within each window
    signal = np.ones(200)  # Constant signal (zero variance)
    
    # Use window_size and step_size that will create windows with zero variance
    snr_estimates = sliding_window_snr(signal, window_size=50, step_size=25)
    
    # All windows should have infinite SNR since noise_power (variance) is zero
    assert len(snr_estimates) > 0
    assert np.all(np.isinf(snr_estimates))
    assert np.all(snr_estimates == float("inf"))


def test_adaptive_threshold_snr_zero_noise_power():
    """Test adaptive_threshold_snr when noise_power is zero.
    
    This test covers line 101 in adaptive_snr_estimation.py where
    return float("inf") when noise_power == 0.
    """
    # Create a signal where noise segments have zero variance
    # Signal segments above threshold, noise segments below threshold but constant
    signal = np.concatenate([
        np.ones(100) * 2.0,  # Signal segments (above threshold)
        np.zeros(100)        # Noise segments (constant, zero variance)
    ])
    
    snr_estimate = adaptive_threshold_snr(signal, threshold=0.5)
    
    # Noise power should be zero (constant noise segments), so SNR should be infinite
    assert snr_estimate == float("inf")


def test_adaptive_threshold_snr_zero_signal_power():
    """Test adaptive_threshold_snr when signal_power is zero.
    
    This test covers line 104 in adaptive_snr_estimation.py where
    return -float("inf") when signal_power == 0.
    
    Note: This edge case is difficult to trigger naturally because:
    - signal_power = 0 requires all signal_segments to be zero
    - But zeros are only in signal_segments if threshold <= 0
    - And noise_segments requires threshold > 0 to be non-empty
    - This creates a logical contradiction
    
    We use mocking to simulate this edge case for coverage purposes.
    """
    from unittest.mock import patch
    
    # Create a signal with both signal and noise segments
    signal = np.concatenate([
        np.ones(100) * 0.1,              # Signal segments (above threshold)
        np.random.normal(0, 0.1, 100)   # Noise segments
    ])
    
    # Patch np.mean in the module where it's used
    # This simulates the case where signal_power = 0
    import vitalDSP.signal_quality_assessment.adaptive_snr_estimation as snr_module
    
    call_count = [0]
    original_mean = np.mean
    
    def mock_mean(arr, **kwargs):
        call_count[0] += 1
        # On the first call (signal_power), return 0
        # On the second call (noise_power), return normal value
        if call_count[0] == 1:
            return 0.0
        else:
            return original_mean(arr, **kwargs)
    
    with patch.object(snr_module.np, 'mean', side_effect=mock_mean):
        snr_estimate = adaptive_threshold_snr(signal, threshold=0.05)
        # Signal power should be zero (mocked), so SNR should be -inf
        assert snr_estimate == -float("inf")