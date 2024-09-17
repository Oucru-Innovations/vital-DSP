import pytest
import numpy as np
from vitalDSP.physiological_features.nonlinear import NonlinearFeatures


@pytest.fixture
def sample_signal():
    return np.array([800, 810, 790, 805, 795, 805, 800, 805, 790, 800])


@pytest.fixture
def sample_signal_all_zeroes():
    return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


@pytest.fixture
def nonlinear_features(sample_signal):
    return NonlinearFeatures(signal=sample_signal, fs=1000)


@pytest.fixture
def nonlinear_features_all_zeroes(sample_signal_all_zeroes):
    return NonlinearFeatures(signal=sample_signal_all_zeroes, fs=1000)


def test_nonlinear_features_initialization(sample_signal):
    nf = NonlinearFeatures(signal=sample_signal, fs=1000)
    assert np.array_equal(nf.signal, sample_signal)
    assert nf.fs == 1000


def test_compute_sample_entropy(nonlinear_features):
    result = nonlinear_features.compute_sample_entropy(m=2, r=0.2)
    # assert isinstance(result, float)
    assert not np.isnan(result)


def test_compute_sample_entropy_normal(nonlinear_features):
    result = nonlinear_features.compute_sample_entropy(m=1, r=0.01)
    # assert isinstance(result, float)
    assert not np.isnan(result)


def test_compute_sample_entropy_edge_case():
    nf = NonlinearFeatures(signal=[800], fs=1000)
    result = nf.compute_sample_entropy(m=2, r=0.2)
    assert result == 0


def test_compute_sample_entropy_small_N(m=100000):
    nf = NonlinearFeatures(signal=[800], fs=1000)
    result = nf.compute_sample_entropy(m=2, r=0.2)
    assert result == 0


def test_compute_approximate_entropy(nonlinear_features):
    result = nonlinear_features.compute_approximate_entropy(m=2, r=0.2)
    assert isinstance(result, float)
    assert not np.isnan(result)


def test_compute_approximate_entropy(nonlinear_features_all_zeroes):
    result = nonlinear_features_all_zeroes.compute_approximate_entropy(m=2, r=0.2)
    assert result == 0


def test_compute_approximate_entropy_small_N(nonlinear_features_all_zeroes):
    result = nonlinear_features_all_zeroes.compute_approximate_entropy(m=200000, r=0.2)
    assert result == 0


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


def test_compute_recurrence_features_basic_case():
    # Basic test with a simple sinusoidal signal
    signal = np.sin(np.linspace(0, 4 * np.pi, 100))
    nonlinear_features = NonlinearFeatures(signal)

    # Run the recurrence feature computation with a threshold of 0.2
    features = nonlinear_features.compute_recurrence_features(
        threshold=0.2, sample_size=500
    )

    assert isinstance(features, dict), "Expected output to be a dictionary"
    assert "recurrence_rate" in features, "Expected 'recurrence_rate' in the result"
    assert "determinism" in features, "Expected 'determinism' in the result"
    assert "laminarity" in features, "Expected 'laminarity' in the result"

    # Check that recurrence features are reasonable values (0 <= x <= 1)
    assert (
        0 <= features["recurrence_rate"] <= 1
    ), "Recurrence rate should be between 0 and 1"
    assert 0 <= features["determinism"] <= 1, "Determinism should be between 0 and 1"
    assert 0 <= features["laminarity"] <= 1, "Laminarity should be between 0 and 1"


def test_compute_recurrence_features_constant_signal():
    # Test case with a constant signal
    signal = np.ones(100)
    nonlinear_features = NonlinearFeatures(signal)

    features = nonlinear_features.compute_recurrence_features(
        threshold=0.2, sample_size=500
    )

    assert (
        features["recurrence_rate"] == 0
    ), "Expected recurrence rate to be 0 for constant signal"
    assert (
        features["determinism"] == 0
    ), "Expected determinism to be 0 for constant signal"
    assert (
        features["laminarity"] == 0
    ), "Expected laminarity to be 0 for constant signal"


def test_compute_recurrence_features_small_signal():
    # Test case with a very small signal
    signal = np.array([1, 2])
    nonlinear_features = NonlinearFeatures(signal)

    features = nonlinear_features.compute_recurrence_features(
        threshold=0.2, sample_size=10
    )

    # Check if the recurrence_rate, determinism, and laminarity are NaN or 0 for small signals
    assert (
        np.isnan(features["recurrence_rate"]) or features["recurrence_rate"] == 0
    ), "Expected recurrence rate to be NaN or 0 for very small signal"
    assert (
        np.isnan(features["determinism"]) or features["determinism"] == 0
    ), "Expected determinism to be NaN or 0 for very small signal"
    assert (
        np.isnan(features["laminarity"]) or features["laminarity"] == 0
    ), "Expected laminarity to be NaN or 0 for very small signal"


def test_compute_approximate_entropy_basic_case():
    # Basic test case with a sinusoidal signal
    signal = np.sin(np.linspace(0, 4 * np.pi, 1000))
    nonlinear_features = NonlinearFeatures(signal)

    # Run the approximate entropy computation with default m and r values
    approx_entropy = nonlinear_features.compute_approximate_entropy(m=2, r=0.2)

    assert isinstance(approx_entropy, float), "Expected output to be a float"
    assert not np.isnan(approx_entropy), "Approximate entropy should not be NaN"
    assert (
        approx_entropy > 0
    ), "Approximate entropy should be positive for a periodic signal"


def test_compute_approximate_entropy_constant_signal():
    # Test case with a constant signal
    signal = np.ones(1000)
    nonlinear_features = NonlinearFeatures(signal)

    approx_entropy = nonlinear_features.compute_approximate_entropy(m=2, r=0.2)

    # For constant signals, approximate entropy should be zero
    assert (
        approx_entropy == 0
    ), "Expected approximate entropy to be 0 for constant signal"


def test_compute_approximate_entropy_random_noise():
    # Test case with random noise
    signal = np.random.randn(1000)
    nonlinear_features = NonlinearFeatures(signal)

    approx_entropy = nonlinear_features.compute_approximate_entropy(m=2, r=0.2)

    assert isinstance(approx_entropy, float), "Expected output to be a float"
    assert not np.isnan(
        approx_entropy
    ), "Approximate entropy should not be NaN for random noise"
    assert approx_entropy > 0, "Approximate entropy should be positive for random noise"


def test_compute_approximate_entropy_short_signal():
    # Test case with a very short signal
    signal = np.array([1, 2, 3])
    nonlinear_features = NonlinearFeatures(signal)

    approx_entropy = nonlinear_features.compute_approximate_entropy(m=2, r=0.2)

    # For very short signals, approximate entropy calculation may not be meaningful, should return 0
    assert (
        approx_entropy == 0
    ), "Expected approximate entropy to be 0 for very short signals"


def test_compute_approximate_entropy_phi_m_zero_case():
    # Test case where phi_m or phi_m1 is zero (could be artificial or rare real-world cases)
    signal = np.zeros(1000)
    nonlinear_features = NonlinearFeatures(signal)

    approx_entropy = nonlinear_features.compute_approximate_entropy(m=2, r=0.2)

    # If phi_m or phi_m1 is zero, the function should handle it gracefully and return 0
    assert (
        approx_entropy == 0
    ), "Expected approximate entropy to handle zero phi_m gracefully"


def test_compute_approximate_entropy_edge_case():
    # Test case with a very small difference between phi_m and phi_m1
    signal = np.linspace(0, 1, 1000)  # Slowly increasing signal
    nonlinear_features = NonlinearFeatures(signal)

    approx_entropy = nonlinear_features.compute_approximate_entropy(m=2, r=0.01)

    assert isinstance(approx_entropy, float), "Expected output to be a float"

    # Allow a small tolerance for floating-point errors
    tolerance = 1e-1
    assert (
        approx_entropy >= -tolerance
    ), f"Expected approximate entropy to be non-negative or close to 0 within tolerance, but got {approx_entropy}"


def test_compute_recurrence_features_random_signal():
    # Test case with random noise
    signal = np.random.rand(1000)
    nonlinear_features = NonlinearFeatures(signal)

    features = nonlinear_features.compute_recurrence_features(
        threshold=0.2, sample_size=1000
    )

    assert isinstance(features, dict), "Expected output to be a dictionary"
    assert "recurrence_rate" in features, "Expected 'recurrence_rate' in the result"
    assert "determinism" in features, "Expected 'determinism' in the result"
    assert "laminarity" in features, "Expected 'laminarity' in the result"

    # Check that recurrence features are reasonable values (0 <= x <= 1)
    assert (
        0 <= features["recurrence_rate"] <= 1
    ), "Recurrence rate should be between 0 and 1"
    assert 0 <= features["determinism"] <= 1, "Determinism should be between 0 and 1"
    assert 0 <= features["laminarity"] <= 1, "Laminarity should be between 0 and 1"


def test_compute_recurrence_features_large_sample_size():
    # Test case with a large sample size
    signal = np.sin(np.linspace(0, 4 * np.pi, 500))
    nonlinear_features = NonlinearFeatures(signal)

    features = nonlinear_features.compute_recurrence_features(
        threshold=0.3, sample_size=10000
    )

    assert isinstance(features, dict), "Expected output to be a dictionary"
    assert "recurrence_rate" in features, "Expected 'recurrence_rate' in the result"
    assert "determinism" in features, "Expected 'determinism' in the result"
    assert "laminarity" in features, "Expected 'laminarity' in the result"

    # Check that recurrence features are reasonable values (0 <= x <= 1)
    assert (
        0 <= features["recurrence_rate"] <= 1
    ), "Recurrence rate should be between 0 and 1"
    assert 0 <= features["determinism"] <= 1, "Determinism should be between 0 and 1"
    assert 0 <= features["laminarity"] <= 1, "Laminarity should be between 0 and 1"


def test_compute_recurrence_features_invalid_signal():
    # Test case with an empty or invalid signal
    signal = np.array([])
    nonlinear_features = NonlinearFeatures(signal)

    features = nonlinear_features.compute_recurrence_features(
        threshold=0.2, sample_size=500
    )

    assert (
        features["recurrence_rate"] == 0
    ), "Expected recurrence rate to be 0 for empty signal"
    assert features["determinism"] == 0, "Expected determinism to be 0 for empty signal"
    assert features["laminarity"] == 0, "Expected laminarity to be 0 for empty signal"
