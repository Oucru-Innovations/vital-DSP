import pytest
import numpy as np
from vitalDSP.utils.signal_processing.normalization import z_score_normalization, min_max_normalization
from vitalDSP.utils.signal_processing.attention_weights import AttentionWeights
from vitalDSP.utils.signal_processing.convolutional_kernels import ConvolutionKernels
from vitalDSP.utils.signal_processing.scaler import StandardScaler


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

    # Test with a single value (edge case — constant signal returns midpoint of range)
    signal_single = np.array([5])
    normalized_single = min_max_normalization(signal_single)
    # Returns midpoint of [0, 1] = 0.5 for a constant (single-value) signal
    expected_single = np.array([0.5])

    assert np.all(normalized_single == expected_single)

    # Test with a constant signal (edge case, min and max will be the same; returns midpoint)
    signal_constant = np.array([3, 3, 3, 3, 3])
    normalized_constant = min_max_normalization(signal_constant)
    # Returns midpoint of [0, 1] = 0.5 for a constant signal
    expected_constant = np.full_like(signal_constant, 0.5, dtype=float)

    assert np.all(normalized_constant == expected_constant)


# --- AttentionWeights tests (covers lines 108, 175, 202) ---

def test_attention_weights_linear_descending():
    """Test linear descending weights (line 108: not ascending branch)."""
    weights = AttentionWeights.linear(size=5, ascending=False)
    assert len(weights) == 5
    np.testing.assert_almost_equal(np.sum(weights), 1.0, decimal=6)
    # Descending: first weight > last weight
    assert weights[0] > weights[-1]


def test_attention_weights_exponential_descending():
    """Test exponential descending weights (line 175: ascending=False branch)."""
    weights = AttentionWeights.exponential(size=5, ascending=False, base=2.0)
    assert len(weights) == 5
    np.testing.assert_almost_equal(np.sum(weights), 1.0, decimal=6)
    # Descending exponential: first weight > last weight
    assert weights[0] > weights[-1]


def test_attention_weights_custom_weights():
    """Test custom_weights normalization (line 202)."""
    raw = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    weights = AttentionWeights.custom_weights(raw)
    assert len(weights) == 5
    np.testing.assert_almost_equal(np.sum(weights), 1.0, decimal=6)


def test_attention_weights_linear_ascending():
    """Test linear ascending weights (default ascending=True)."""
    weights = AttentionWeights.linear(size=5, ascending=True)
    assert len(weights) == 5
    np.testing.assert_almost_equal(np.sum(weights), 1.0, decimal=6)
    assert weights[-1] > weights[0]


def test_attention_weights_exponential_ascending():
    """Test exponential ascending weights (ascending=True)."""
    weights = AttentionWeights.exponential(size=5, ascending=True, base=2.0)
    np.testing.assert_almost_equal(np.sum(weights), 1.0, decimal=6)
    assert weights[-1] > weights[0]


def test_attention_weights_uniform():
    """Test uniform weights."""
    weights = AttentionWeights.uniform(size=5)
    np.testing.assert_almost_equal(np.sum(weights), 1.0, decimal=6)
    assert np.all(weights == weights[0])


def test_attention_weights_gaussian():
    """Test Gaussian weights."""
    weights = AttentionWeights.gaussian(size=5, sigma=1.0)
    np.testing.assert_almost_equal(np.sum(weights), 1.0, decimal=6)


# --- ConvolutionKernels tests (covers lines 77, 144) ---

def test_convolution_kernels_smoothing_even_size_raises():
    """Test that even kernel size raises ValueError (line 77)."""
    with pytest.raises(ValueError, match="Kernel size must be odd"):
        ConvolutionKernels.smoothing(size=4)


def test_convolution_kernels_custom_kernel():
    """Test custom_kernel returns the kernel as-is (line 144)."""
    kernel = np.array([0.2, 0.5, 0.3])
    result = ConvolutionKernels.custom_kernel(kernel)
    np.testing.assert_array_equal(result, kernel)


def test_convolution_kernels_smoothing_valid():
    """Test that valid (odd) kernel size returns correct array."""
    kernel = ConvolutionKernels.smoothing(size=3)
    assert len(kernel) == 3
    np.testing.assert_almost_equal(np.sum(kernel), 1.0, decimal=6)


def test_convolution_kernels_sharpening():
    """Test sharpening kernel."""
    kernel = ConvolutionKernels.sharpening()
    np.testing.assert_array_equal(kernel, np.array([-1, 2, -1]))


def test_convolution_kernels_edge_detection():
    """Test edge detection kernel."""
    kernel = ConvolutionKernels.edge_detection()
    np.testing.assert_array_equal(kernel, np.array([-1, 0, 1]))


# --- StandardScaler tests (covers lines 103, 106) ---

def test_standard_scaler_transform_not_fitted_raises():
    """Test that transform before fit raises ValueError (line 103)."""
    scaler = StandardScaler()
    with pytest.raises(ValueError, match="not been fitted yet"):
        scaler.transform(np.array([1, 2, 3]))


def test_standard_scaler_transform_constant_signal():
    """Test that constant signal returns zero-centered values (line 106: std_==0 branch)."""
    scaler = StandardScaler()
    signal = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    scaler.fit(signal)
    result = scaler.transform(signal)
    # std == 0 -> return (signal - mean).astype(float) = all zeros
    np.testing.assert_array_equal(result, np.zeros(5))


def test_standard_scaler_fit_transform():
    """Test fit_transform convenience method."""
    scaler = StandardScaler()
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = scaler.fit_transform(signal)
    assert len(result) == 5
    # Mean should be close to 0 after standardization
    np.testing.assert_almost_equal(np.mean(result), 0.0, decimal=6)


def test_standard_scaler_normal_transform():
    """Test normal transform with non-constant signal."""
    scaler = StandardScaler()
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scaler.fit(signal)
    result = scaler.transform(signal)
    np.testing.assert_almost_equal(np.mean(result), 0.0, decimal=6)
    np.testing.assert_almost_equal(np.std(result), 1.0, decimal=6)
