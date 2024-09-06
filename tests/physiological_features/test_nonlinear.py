import pytest
import numpy as np
from vitalDSP.physiological_features.nonlinear import NonlinearFeatures

@pytest.fixture
def sample_signal():
    return np.array([800, 810, 790, 805, 795, 805, 800, 805, 790, 800])

@pytest.fixture
def nonlinear_features(sample_signal):
    return NonlinearFeatures(signal=sample_signal, fs=1000)

def test_nonlinear_features_initialization(sample_signal):
    nf = NonlinearFeatures(signal=sample_signal, fs=1000)
    assert np.array_equal(nf.signal, sample_signal)
    assert nf.fs == 1000

def test_compute_sample_entropy(nonlinear_features):
    result = nonlinear_features.compute_sample_entropy(m=2, r=0.2)
    assert isinstance(result, float)
    assert not np.isnan(result)

def test_compute_sample_entropy_edge_case():
    nf = NonlinearFeatures(signal=[800], fs=1000)
    result = nf.compute_sample_entropy(m=2, r=0.2)
    assert result == 0

def test_compute_approximate_entropy(nonlinear_features):
    result = nonlinear_features.compute_approximate_entropy(m=2, r=0.2)
    assert isinstance(result, float)
    assert not np.isnan(result)

def test_compute_fractal_dimension(nonlinear_features):
    result = nonlinear_features.compute_fractal_dimension(kmax=5)
    assert isinstance(result, float)
    assert not np.isnan(result)

def test_compute_fractal_dimension_edge_case():
    nf = NonlinearFeatures(signal=[800], fs=1000)
    result = nf.compute_fractal_dimension(kmax=5)
    assert result == 0

def test_compute_lyapunov_exponent(nonlinear_features):
    result = nonlinear_features.compute_lyapunov_exponent()
    # assert isinstance(result, float)
    assert result >= 0, "Lyapunov exponent should be non-negative"

def test_compute_lyapunov_exponent_constant_signal():
    nf = NonlinearFeatures(signal=np.ones(100), fs=1000)
    result = nf.compute_lyapunov_exponent()
    assert result == 0, "Lyapunov exponent of a constant signal should be zero"

# Additional edge case for very short signals
def test_compute_lyapunov_exponent_short_signal():
    nf = NonlinearFeatures(signal=[800, 810], fs=1000)
    result = nf.compute_lyapunov_exponent()
    assert result == 0, "Lyapunov exponent for short signals should return zero"
