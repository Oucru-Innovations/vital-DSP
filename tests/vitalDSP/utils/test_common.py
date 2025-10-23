import pytest
import numpy as np
import warnings
from vitalDSP.utils.config_utilities.common import (
    argrelextrema,
    find_peaks,
    filtfilt,
    pearsonr,
    coherence,
    grangercausalitytests,
    dtw_distance_windowed,
    deprecated,
)


# Test for argrelextrema
def test_argrelextrema_maxima():
    signal = np.array([1, 3, 2, 4, 3, 5, 4])
    maxima = argrelextrema(signal, comparator=np.greater, order=1)
    assert np.array_equal(maxima, np.array([1, 3, 5]))


def test_argrelextrema_minima():
    signal = np.array([1, 3, 2, 4, 3, 5, 4])
    minima = argrelextrema(signal, comparator=np.less, order=1)
    assert np.array_equal(minima, np.array([2, 4]))


def test_argrelextrema_order_too_small():
    signal = np.array([1, 3])
    with pytest.raises(ValueError):
        argrelextrema(signal, comparator=np.greater, order=1)


def test_argrelextrema_invalid_order():
    """Test argrelextrema with order < 1"""
    signal = np.array([1, 3, 2, 4, 3, 5, 4])
    with pytest.raises(ValueError, match="Order must be an int >= 1"):
        argrelextrema(signal, comparator=np.greater, order=0)


# Test for find_peaks
def test_find_peaks():
    signal = np.array([0, 1, 0, 2, 0, 3, 0])
    peaks = find_peaks(signal, height=1)
    assert np.array_equal(peaks, np.array([1, 3, 5]))


def test_find_peaks_with_distance():
    signal = np.array([0, 0, 0, 1, 2, 0, 2, 0, 0, 0])
    peaks = find_peaks(signal, distance=2)
    assert np.array_equal(peaks, np.array([4, 6]))


def test_find_peaks_with_threshold():
    """Test find_peaks with threshold parameter"""
    signal = np.array([0, 1, 0, 2, 0, 3, 0])
    peaks = find_peaks(signal, threshold=0.5)
    assert np.array_equal(peaks, np.array([1, 3, 5]))


def test_find_peaks_with_prominence():
    """Test find_peaks with prominence parameter"""
    signal = np.array([0, 1, 0, 2, 0, 3, 0])
    peaks = find_peaks(signal, prominence=1.5, distance=4)
    assert len(peaks) >= 0  # Checks that function runs without error


def test_find_peaks_with_width():
    """Test find_peaks with width parameter"""
    signal = np.array([0, 0, 1, 2, 3, 2, 1, 0, 0])
    peaks = find_peaks(signal, width=1)
    assert isinstance(peaks, np.ndarray)


def test_find_peaks_width_edge_cases():
    """Test find_peaks with width parameter where left_width or right_width is empty"""
    # Signal where peak is at the beginning (no left width)
    signal = np.array([5, 4, 3, 2, 1, 0])
    peaks = find_peaks(signal, width=1)
    assert isinstance(peaks, np.ndarray)


# Test for filtfilt
def test_filtfilt():
    b = np.array([0.0675, 0.1349, 0.0675])
    a = np.array([1.0000, -1.1430, 0.4128])
    signal = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    filtered_signal = filtfilt(b, a, signal)
    expected_output = np.convolve(signal, b, mode="same")
    expected_output = np.convolve(expected_output[::-1], b, mode="same")[::-1]
    assert np.allclose(filtered_signal, expected_output)


# Test for pearsonr
def test_pearsonr():
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 2, 3, 5])
    corr = pearsonr(x, y)
    assert (
        0.9819 <= corr <= 0.9829
    ), f"Expected in range [0.9819, 0.9829], but got {corr}"


def test_pearsonr_different_length():
    x = np.array([1, 2, 3])
    y = np.array([1, 2])
    with pytest.raises(ValueError):
        pearsonr(x, y)


# Test for coherence
def test_coherence():
    x = np.sin(2 * np.pi * np.linspace(0, 1, 500))
    y = np.sin(2 * np.pi * np.linspace(0, 1, 500) + np.pi / 4)
    freqs, coh = coherence(x, y, fs=500)
    assert freqs.shape[0] > 0
    assert coh.shape[0] > 0


# Test for grangercausalitytests
def test_grangercausalitytests():
    data = np.random.rand(100, 2)
    results = grangercausalitytests(data, max_lag=4, verbose=False)
    assert isinstance(results, dict)
    assert len(results) == 4  # Should have 4 lags


def test_grangercausalitytests_verbose():
    """Test grangercausalitytests with verbose=True"""
    data = np.random.rand(100, 2)
    results = grangercausalitytests(data, max_lag=2, verbose=True)
    assert isinstance(results, dict)
    assert len(results) == 2


# Test for dtw_distance_windowed
def test_dtw_distance_windowed():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 4, 5, 6])
    distance = dtw_distance_windowed(x, y, window=2)
    assert distance >= 0


def test_dtw_distance_windowed_exact():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    distance = dtw_distance_windowed(x, y)
    assert np.isclose(distance, 0)


# Test for deprecated decorator
def test_deprecated_decorator():
    """Test the deprecated decorator"""

    @deprecated("This function is deprecated")
    def old_function():
        return "result"

    # Test that the function still works
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = old_function()
        assert result == "result"
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "old_function is deprecated" in str(w[-1].message)
