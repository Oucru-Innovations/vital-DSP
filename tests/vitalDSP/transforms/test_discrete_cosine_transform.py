import pytest
import numpy as np
from vitalDSP.transforms.discrete_cosine_transform import DiscreteCosineTransform


@pytest.fixture
def sample_signal():
    np.random.seed(42)
    return np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)


def test_init(sample_signal):
    """Test initialization of DiscreteCosineTransform"""
    dct = DiscreteCosineTransform(sample_signal)
    assert np.array_equal(dct.signal, sample_signal)


def test_compute_dct(sample_signal):
    """Test DCT computation"""
    dct = DiscreteCosineTransform(sample_signal)
    dct_coefficients = dct.compute_dct()

    assert isinstance(dct_coefficients, np.ndarray)
    assert len(dct_coefficients) == len(sample_signal)


def test_compute_idct(sample_signal):
    """Test IDCT computation"""
    dct = DiscreteCosineTransform(sample_signal)
    dct_coefficients = dct.compute_dct()
    reconstructed_signal = dct.compute_idct(dct_coefficients)

    assert isinstance(reconstructed_signal, np.ndarray)
    assert len(reconstructed_signal) == len(sample_signal)


def test_compress_signal(sample_signal):
    """Test signal compression with different thresholds"""
    dct = DiscreteCosineTransform(sample_signal)

    # Test with default threshold
    compressed_coefficients = dct.compress_signal(threshold=0.1)
    assert isinstance(compressed_coefficients, np.ndarray)
    assert len(compressed_coefficients) == len(sample_signal)

    # Verify that some coefficients are zeroed out
    assert np.sum(compressed_coefficients == 0) > 0

    # Test with different threshold values
    compressed_low = dct.compress_signal(threshold=0.05)
    compressed_high = dct.compress_signal(threshold=0.5)

    # Higher threshold should zero out more coefficients
    assert np.sum(compressed_high == 0) >= np.sum(compressed_low == 0)


def test_roundtrip(sample_signal):
    """Test that DCT -> IDCT approximately recovers the original signal"""
    dct = DiscreteCosineTransform(sample_signal)
    dct_coefficients = dct.compute_dct()
    reconstructed_signal = dct.compute_idct(dct_coefficients)

    # Account for windowing by comparing with windowed signal
    windowed_signal = sample_signal * np.hamming(len(sample_signal))

    # The reconstructed signal should be close to the windowed original
    assert np.allclose(reconstructed_signal, windowed_signal, atol=1e-10)


def test_dct_norm_parameter(sample_signal):
    """Test DCT with different normalization parameters"""
    dct = DiscreteCosineTransform(sample_signal)

    # Test with 'ortho' normalization
    dct_ortho = dct.compute_dct(norm='ortho')
    assert isinstance(dct_ortho, np.ndarray)

    # Test with None normalization
    dct_none = dct.compute_dct(norm=None)
    assert isinstance(dct_none, np.ndarray)


def test_idct_norm_parameter(sample_signal):
    """Test IDCT with different normalization parameters"""
    dct = DiscreteCosineTransform(sample_signal)
    dct_coefficients = dct.compute_dct()

    # Test with 'ortho' normalization
    idct_ortho = dct.compute_idct(dct_coefficients, norm='ortho')
    assert isinstance(idct_ortho, np.ndarray)

    # Test with None normalization
    idct_none = dct.compute_idct(dct_coefficients, norm=None)
    assert isinstance(idct_none, np.ndarray)
