import pytest
import numpy as np
import warnings
from vitalDSP.utils.signal_processing.mother_wavelets import Wavelet

# Filter out expected warnings about invalid mathematical operations
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in.*")


def test_haar():
    lo, hi = Wavelet.haar()
    expected_lo = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    np.testing.assert_array_almost_equal(lo, expected_lo)
    assert lo.size == 2
    assert hi.size == 2


def test_db():
    lo, hi = Wavelet.db(order=4)
    assert isinstance(lo, np.ndarray)
    assert isinstance(hi, np.ndarray)
    assert lo.size > 0
    assert hi.size == lo.size


def test_sym():
    lo, hi = Wavelet.sym(order=4)
    assert isinstance(lo, np.ndarray)
    assert isinstance(hi, np.ndarray)
    assert lo.size > 0
    assert hi.size == lo.size


def test_coif():
    lo, hi = Wavelet.coif(order=2)
    assert isinstance(lo, np.ndarray)
    assert isinstance(hi, np.ndarray)
    assert lo.size > 0
    assert hi.size == lo.size


def test_gauchy():
    result = Wavelet.gauchy(sigma=1.0, N=6)
    assert isinstance(result, np.ndarray)
    assert result.size == 6


def test_mexican_hat():
    result = Wavelet.mexican_hat(sigma=1.0, N=6)
    assert isinstance(result, np.ndarray)
    assert result.size == 6


def test_morlet():
    result = Wavelet.morlet(sigma=1.0, N=6, f=1.0)
    assert isinstance(result, np.ndarray)
    assert result.size == 6


def test_meyer():
    result = Wavelet.meyer(N=6)
    assert isinstance(result, np.ndarray)
    assert result.size == 6


def test_biorthogonal():
    result = Wavelet.biorthogonal(N=6, p=2, q=2)
    assert isinstance(result, np.ndarray)
    assert result.size == 6


def test_reverse_biorthogonal():
    result = Wavelet.reverse_biorthogonal(N=6, p=2, q=2)
    assert isinstance(result, np.ndarray)
    assert result.size == 6


def test_complex_gaussian():
    result = Wavelet.complex_gaussian(sigma=1.0, N=6)
    assert isinstance(result, np.ndarray)
    assert result.size == 6


def test_shannon():
    result = Wavelet.shannon(N=6)
    assert isinstance(result, np.ndarray)
    assert result.size == 6


def test_cmor():
    result = Wavelet.cmor(sigma=1.0, N=6, f=1.0)
    assert isinstance(result, np.ndarray)
    assert result.size == 6


def test_fbsp():
    result = Wavelet.fbsp(N=6, m=5, s=0.5)
    assert isinstance(result, np.ndarray)
    assert result.size == 6


def test_mhat():
    result = Wavelet.mhat(sigma=1.0, N=6)
    assert isinstance(result, np.ndarray)
    assert result.size == 6


def test_custom_wavelet():
    custom_wavelet = np.array([0.2, 0.5, 0.2])
    result = Wavelet.custom_wavelet(custom_wavelet)
    np.testing.assert_array_equal(result, custom_wavelet)


def test_wavelet_orders():
    """Test that wavelets work with different valid orders."""
    # Test Daubechies with different orders
    for order in [2, 4, 6, 8]:
        lo, hi = Wavelet.db(order=order)
        assert isinstance(lo, np.ndarray)
        assert lo.size > 0
        assert hi.size == lo.size

    # Test Symlets with different orders
    for order in [2, 4, 6, 8]:
        lo, hi = Wavelet.sym(order=order)
        assert isinstance(lo, np.ndarray)
        assert lo.size > 0
        assert hi.size == lo.size

    # Test Coiflets with different orders
    for order in [2, 4, 6]:
        lo, hi = Wavelet.coif(order=order)
        assert isinstance(lo, np.ndarray)
        assert lo.size > 0
        assert hi.size == lo.size


def test_wavelet_edge_cases():
    """Test wavelet behavior with edge case orders."""
    try:
        lo, hi = Wavelet.db(order=1)
        if lo.size > 0:
            assert isinstance(lo, np.ndarray)
    except (ValueError, RuntimeWarning):
        pass

    try:
        lo, hi = Wavelet.sym(order=1)
        if lo.size > 0:
            assert isinstance(lo, np.ndarray)
    except (ValueError, RuntimeWarning):
        pass

    try:
        lo, hi = Wavelet.coif(order=1)
        if lo.size > 0:
            assert isinstance(lo, np.ndarray)
    except (ValueError, RuntimeWarning):
        pass
