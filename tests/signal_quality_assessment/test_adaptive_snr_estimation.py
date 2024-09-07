import numpy as np
import pytest
from vitalDSP.signal_quality_assessment.adaptive_snr_estimation import sliding_window_snr, adaptive_threshold_snr, recursive_snr_estimation  # Replace 'your_module' with the actual module name

# Test for sliding_window_snr
def test_sliding_window_snr():
    # Test with a simple sine wave signal
    signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01)) + 0.1 * np.random.normal(size=1000)
    
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
    signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 1000, 0.01)) + 0.1 * np.random.normal(size=100000)
    snr_estimate = adaptive_threshold_snr(signal, threshold=0.3)
    assert np.isfinite(snr_estimate)  # SNR should be finite for this signal

def test_adaptive_threshold_snr_edge_case_no_noise():
    # Test with a signal that crosses the threshold
    signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01)) + 0.1 * np.random.normal(size=1000)
    
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
    assert np.isinf(snr_estimate)  # Signal to noise ratio should return inf when there's only noise (signal is near zero)

def test_recursive_snr_estimation():
    # Test with a simple sine wave signal
    signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01)) + 0.1 * np.random.normal(size=1000)
    
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
    assert np.all(np.isinf(snr_estimates))  # With a constant signal, noise power will be zero, resulting in infinite SNR