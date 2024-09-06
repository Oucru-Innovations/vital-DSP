import numpy as np
import pytest
from vitalDSP.signal_quality_assessment.adaptive_snr_estimation import sliding_window_snr, adaptive_threshold_snr, recursive_snr_estimation  # Replace 'your_module' with the actual module name

# Test for sliding_window_snr
def test_sliding_window_snr():
    signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 1000, 0.01)) + 0.1 * np.random.normal(size=100000)
    snr_estimates = sliding_window_snr(signal, window_size=100, step_size=50)
    assert len(snr_estimates) > 0  # Check that SNR estimates are returned
    assert np.all(np.isfinite(snr_estimates))  # Check that all values are finite

def test_adaptive_threshold_snr():
    signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 1000, 0.01)) + 0.1 * np.random.normal(size=100000)
    snr_estimate = adaptive_threshold_snr(signal, threshold=0.3)
    assert np.isfinite(snr_estimate)  # SNR should be finite for this signal

def test_adaptive_threshold_snr_edge_case_no_noise():
    signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 1000, 0.01))  # Pure signal with no noise
    snr_estimate = adaptive_threshold_snr(signal, threshold=0.01)
    assert snr_estimate > 100  # SNR should be very high when there's no noise, but not exactly inf due to floating point limitations

def test_adaptive_threshold_snr_edge_case_all_noise():
    signal = np.random.normal(0, 0.1, 1000)  # Pure noise
    snr_estimate = adaptive_threshold_snr(signal, threshold=0.5)
    assert np.isinf(snr_estimate)  # Signal to noise ratio should return inf when there's only noise (signal is near zero)

def test_recursive_snr_estimation():
    signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 1000, 0.01)) + 0.1 * np.random.normal(size=100000)
    snr_estimates = recursive_snr_estimation(signal, alpha=0.9)
    assert len(snr_estimates) == len(signal)  # Ensure one SNR estimate per signal point
    assert np.all(np.isfinite(snr_estimates))  # Check that all estimates are finite

def test_recursive_snr_estimation_edge_case_constant_signal():
    signal = np.ones(1000)  # Constant signal (no variation)
    snr_estimates = recursive_snr_estimation(signal, alpha=0.9)
    assert np.all(np.isinf(snr_estimates))  # With a constant signal, noise power will be zero, resulting in infinite SNR