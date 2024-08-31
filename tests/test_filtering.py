import numpy as np
import pytest
from vitalDSP.filtering.signal_filtering import SignalFiltering, BandpassFilter
from vitalDSP.filtering.advanced_signal_filtering import AdvancedSignalFiltering

@pytest.fixture
def sample_signal():
    """Fixture for creating a sample signal."""
    return np.sin(np.linspace(0, 2 * np.pi, 100)) + np.random.normal(0, 0.1, 100)

# Test initialization
def test_bandpass_filter_initialization():
    bp_filter = BandpassFilter(band_type="butter", fs=100)
    assert bp_filter.band_type == "butter"
    assert bp_filter.fs == 100

# Test signal_bypass for Butterworth filter
def test_signal_bypass_butter():
    bp_filter = BandpassFilter(band_type="butter", fs=100)
    cutoff = 0.3
    order = 4
    a_pass = 3
    rp = 4
    rs = 40
    b, a = bp_filter.signal_bypass(cutoff=cutoff, order=order, a_pass=a_pass, rp=rp, rs=rs, btype='low')
    assert len(b) == order + 1
    assert len(a) == order + 1

# Test signal_bypass for Chebyshev filter
def test_signal_bypass_cheby1():
    bp_filter = BandpassFilter(band_type="cheby1", fs=100)
    cutoff = 0.3
    order = 4
    a_pass = 3
    rp = 4
    rs = 40
    b, a = bp_filter.signal_bypass(cutoff=cutoff, order=order, a_pass=a_pass, rp=rp, rs=rs, btype='low')
    assert len(b) == order + 1
    assert len(a) == order + 1

# Test lowpass filtering with valid signal
def test_signal_lowpass_filter():
    signal_data = np.array([1, 2, 3, 4, 5], dtype=float)
    bp_filter = BandpassFilter(band_type="butter", fs=100)
    cutoff = 0.3
    filtered_signal = bp_filter.signal_lowpass_filter(signal_data, cutoff=cutoff, order=4)
    assert len(filtered_signal) == len(signal_data)
    assert np.all(np.isfinite(filtered_signal))

# Test highpass filtering with valid signal
def test_signal_highpass_filter():
    signal_data = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 
                            2, 3, 4, 5, 4, 3, 2, 1,
                            2, 3, 4, 5, 4, 3, 2, 1,
                            2, 3, 4, 5, 4, 3, 2, 1,
                            2, 3, 4, 5, 4, 3, 2, 1], dtype=float)
    bp_filter = BandpassFilter(band_type="butter", fs=100)
    cutoff = 0.3
    filtered_signal = bp_filter.signal_highpass_filter(signal_data, cutoff=cutoff, order=4)
    assert len(filtered_signal) == len(signal_data)
    assert np.all(np.isfinite(filtered_signal))

# Test highpass filter with short signal to raise ValueError
def test_signal_highpass_filter_short_signal():
    signal_data = np.array([1, 2], dtype=float)
    bp_filter = BandpassFilter(band_type="butter", fs=100)
    cutoff = 0.3
    with pytest.raises(ValueError, match="The length of the input vector x must be greater than"):
        bp_filter.signal_highpass_filter(signal_data, cutoff=cutoff, order=4)

# Test lowpass filter with short signal to raise ValueError
def test_signal_lowpass_filter_short_signal():
    signal_data = np.array([1, 2], dtype=float)
    bp_filter = BandpassFilter(band_type="butter", fs=100)
    cutoff = 0.3
    with pytest.raises(ValueError, match="The length of the input vector x must be greater than"):
        bp_filter.signal_highpass_filter(signal_data, cutoff=cutoff, order=4)

def test_moving_average(sample_signal):
    filter = SignalFiltering(sample_signal)
    window_size = 5
    filtered_signal = filter.moving_average(window_size=window_size)
    assert len(filtered_signal) == (len(sample_signal)), "Filtered signal length mismatch"

def test_gaussian(sample_signal):
    filter = SignalFiltering(sample_signal)
    filtered_signal = filter.gaussian(sigma=1.0)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Gaussian filtering failed"

def test_butterworth(sample_signal):
    filter = SignalFiltering(sample_signal)
    filtered_signal = filter.butterworth(fs=100,order=2, cutoff=0.3)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    # assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Butterworth filtering failed"

def test_median(sample_signal):
    filter = SignalFiltering(sample_signal)
    kernel_size = 3
    filtered_signal = filter.median(kernel_size=kernel_size)
    assert len(filtered_signal) == (len(sample_signal)), "Filtered signal length mismatch"
    # assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Median filtering failed"

def test_savgol_filter(sample_signal):
    filtered_signal = SignalFiltering.savgol_filter(sample_signal, window_length=5, polyorder=2)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Savitzky-Golay filtering failed"
    with pytest.raises(ValueError, match="window_length must be a positive odd integer"):
        SignalFiltering.savgol_filter(sample_signal, window_length=4, polyorder=2)
    with pytest.raises(ValueError, match="window_length is too small for the polynomial order"):
        SignalFiltering.savgol_filter(sample_signal, window_length=3, polyorder=3)

def test_gaussian_filter1d_basic():
    signal = np.array([1, 2, 3, 4, 5])
    sigma = 1.0
    smoothed_signal = SignalFiltering.gaussian_filter1d(signal, sigma)
    assert len(smoothed_signal) == len(signal), "Filtered signal length mismatch"
    # expected_output = np.array([1.14285714, 2.14285714, 3.0, 4.0, 4.85714286])
    # np.testing.assert_almost_equal(smoothed_signal, expected_output, decimal=1)

def test_kalman_filter(sample_signal):
    filter = AdvancedSignalFiltering(sample_signal)
    filtered_signal = filter.kalman_filter(R=1e-3, Q=1e-4)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Kalman filtering failed"

def test_optimization_based_filter(sample_signal):
    sample_signal = np.array([1, 2, 3, 2, 1, 2, 3, 4, 3, 2])
    target = np.array([1, 1.5, 2.5, 3, 2.5, 2, 2.5, 3.5, 3, 2.5])
    filter = AdvancedSignalFiltering(sample_signal)
    filtered_signal = filter.optimization_based_filtering(target,loss_type='mse')
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Optimization-based filtering failed"

def test_gradient_descent_filter(sample_signal):
    sample_signal = np.array([1, 2, 3, 2, 1, 2, 3, 4, 3, 2])
    target = np.array([1, 1.5, 2.5, 3, 2.5, 2, 2.5, 3.5, 3, 2.5])
    filter = AdvancedSignalFiltering(sample_signal)
    filtered_signal = filter.gradient_descent_filter(target,learning_rate=0.01,iterations=20)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    # assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Gradient Descent filtering failed"

def test_ensemble_filtering(sample_signal):
    af = AdvancedSignalFiltering(sample_signal)
    filter = SignalFiltering(sample_signal)
    filters = [af.kalman_filter, af.kalman_filter]
    filtered_signal = af.ensemble_filtering(filters,method='mean')
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Ensemble filtering failed"

def test_convolution_based_filter(sample_signal):
    kernel = np.array([0.25, 0.5, 0.25])
    filter = AdvancedSignalFiltering(sample_signal)
    filtered_signal = filter.convolution_based_filter(custom_kernel=kernel)
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Convolution-based filtering failed"

def test_attention_based_filter(sample_signal):
    filter = AdvancedSignalFiltering(sample_signal)
    filtered_signal = filter.attention_based_filter(attention_weights=np.array([0.1, 0.2, 0.3, 0.4]))
    assert len(filtered_signal) == len(sample_signal), "Filtered signal length mismatch"
    assert np.all(np.abs(filtered_signal - sample_signal) < 1), "Attention-based filtering failed"

if __name__ == "__main__":
    pytest.main()
