import pytest
import numpy as np
from vitalDSP.utils.mother_wavelets import Wavelet
def test_haar():
    result = Wavelet.haar()
    expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    np.testing.assert_array_almost_equal(result, expected)

def test_db():
    result = Wavelet.db(order=4)
    assert isinstance(result, np.ndarray)
    assert result.size > 0

def test_sym():
    result = Wavelet.sym(order=4)
    assert isinstance(result, np.ndarray)
    assert result.size > 0

def test_coif():
    result = Wavelet.coif(order=2)
    assert isinstance(result, np.ndarray)
    assert result.size > 0

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
