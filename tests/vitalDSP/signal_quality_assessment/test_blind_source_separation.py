import pytest
import numpy as np
from vitalDSP.signal_quality_assessment.blind_source_separation import (
    center_signal,
    whiten_signal,
    ica_artifact_removal,
    pca_artifact_removal,
    jade_ica,
)


def test_center_signal():
    # Test with a basic signal
    signal = np.array([[1, 2, 3], [4, 5, 6]])
    centered_signal, mean_signal = center_signal(signal)

    expected_centered_signal = np.array([[-1, 0, 1], [-1, 0, 1]])
    expected_mean_signal = np.array([[2], [5]])

    np.testing.assert_array_almost_equal(centered_signal, expected_centered_signal)
    np.testing.assert_array_almost_equal(mean_signal, expected_mean_signal)

    # Test with a single row signal
    signal = np.array([[10, 20, 30]])
    centered_signal, mean_signal = center_signal(signal)

    expected_centered_signal = np.array([[-10, 0, 10]])
    expected_mean_signal = np.array([[20]])

    np.testing.assert_array_almost_equal(centered_signal, expected_centered_signal)
    np.testing.assert_array_almost_equal(mean_signal, expected_mean_signal)


def test_whiten_signal():
    # Test with a simple signal
    signal = np.array([[1, 2], [3, 4]])
    whitened_signal, whitening_matrix = whiten_signal(signal)

    # Check shapes of the outputs
    assert whitened_signal.shape == signal.shape
    assert whitening_matrix.shape == (signal.shape[1], signal.shape[1])

    # Verify the signal is whitened (covariance should approximate identity matrix)
    # cov = np.cov(whitened_signal, rowvar=False)
    # np.testing.assert_array_almost_equal(cov, np.identity(whitened_signal.shape[1]), decimal=4)


def test_ica_artifact_removal():
    # Test ICA with a simple signal
    signals = np.array([[1, 2, 3], [4, 5, 6]])
    separated_signals = ica_artifact_removal(signals)

    # Check the shape of separated signals
    assert separated_signals.shape == signals.shape

    # Ensure that separated signals are within a reasonable range
    # assert np.all(np.abs(separated_signals) < 10)


def test_pca_artifact_removal():
    # Test PCA with reduced components
    signals = np.array([[1, 2, 3], [4, 5, 6]])
    reduced_signals = pca_artifact_removal(signals, n_components=1)

    # Check the shape of the reduced signals
    assert reduced_signals.shape == (1, signals.shape[1])

    # Test PCA with all components
    reduced_signals = pca_artifact_removal(signals)
    assert reduced_signals.shape == signals.shape


# Test with a basic set of signals
def test_jade_ica_basic(monkeypatch):
    signals = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Mock whiten_signal to avoid using the actual implementation
    def mock_whiten_signal(signals):
        mean = np.mean(signals, axis=1, keepdims=True)
        std = np.std(signals, axis=1, keepdims=True)
        whitened_signals = (signals - mean) / std
        return whitened_signals, np.eye(
            signals.shape[0]
        )  # Return an identity matrix for the whitening matrix

    # Use monkeypatch to replace the whiten_signal function in the module
    monkeypatch.setattr(
        "src.vitalDSP.signal_quality_assessment.blind_source_separation.whiten_signal",
        mock_whiten_signal,
    )

    separated_signals = jade_ica(signals)

    assert (
        separated_signals.shape == signals.shape
    ), "Output shape should match input shape"
    assert not np.isnan(
        separated_signals
    ).any(), "No NaN values should be present in the output"


# Test with random signals
def test_jade_ica_random_signals(monkeypatch):
    np.random.seed(42)
    signals = np.random.randn(
        5, 5
    )  # 5 signals, 5 samples each to make the covariance matrix square

    # Mock whiten_signal
    def mock_whiten_signal(signals):
        mean = np.mean(signals, axis=1, keepdims=True)
        std = np.std(signals, axis=1, keepdims=True)
        whitened_signals = (signals - mean) / std
        return whitened_signals, np.eye(signals.shape[0])

    monkeypatch.setattr(
        "src.vitalDSP.signal_quality_assessment.blind_source_separation.whiten_signal",
        mock_whiten_signal,
    )

    separated_signals = jade_ica(signals)

    assert (
        separated_signals.shape == signals.shape
    ), "Output shape should match input shape"
    assert not np.isnan(
        separated_signals
    ).any(), "No NaN values should be present in the output"


# Test with a single signal (should raise an error)
def test_jade_ica_single_signal():
    signals = np.array([1, 2, 3, 4, 5])  # Single signal

    with pytest.raises(ValueError, match="at least two signals"):
        jade_ica(signals)


# Test with a large number of iterations
def test_jade_ica_large_iterations(monkeypatch):
    signals = np.random.randn(
        3, 3
    )  # 3 signals, 3 samples each to make the covariance matrix square

    # Mock whiten_signal
    def mock_whiten_signal(signals):
        mean = np.mean(signals, axis=1, keepdims=True)
        std = np.std(signals, axis=1, keepdims=True)
        whitened_signals = (signals - mean) / std
        return whitened_signals, np.eye(signals.shape[0])

    monkeypatch.setattr(
        "src.vitalDSP.signal_quality_assessment.blind_source_separation.whiten_signal",
        mock_whiten_signal,
    )

    separated_signals = jade_ica(signals, max_iter=5000)  # Large number of iterations

    assert (
        separated_signals.shape == signals.shape
    ), "Output shape should match input shape"
    assert not np.isnan(
        separated_signals
    ).any(), "No NaN values should be present in the output"


# Test tolerance handling (we won't actually check convergence here, but tolerance should be accepted)
def test_jade_ica_tolerance(monkeypatch):
    signals = np.random.randn(3, 3)  # 3 signals, 3 samples each

    # Mock whiten_signal
    def mock_whiten_signal(signals):
        mean = np.mean(signals, axis=1, keepdims=True)
        std = np.std(signals, axis=1, keepdims=True)
        whitened_signals = (signals - mean) / std
        return whitened_signals, np.eye(signals.shape[0])

    monkeypatch.setattr(
        "src.vitalDSP.signal_quality_assessment.blind_source_separation.whiten_signal",
        mock_whiten_signal,
    )

    separated_signals = jade_ica(signals, tol=1e-7)  # Very small tolerance

    assert (
        separated_signals.shape == signals.shape
    ), "Output shape should match input shape"
    assert not np.isnan(
        separated_signals
    ).any(), "No NaN values should be present in the output"


# Test with an empty signal (edge case)
def test_jade_ica_empty_signal():
    signals = np.array([])  # Empty signal

    with pytest.raises(ValueError):
        jade_ica(signals)


# Test with NaN values in input signals (should raise an error)
def test_jade_ica_with_nan_values():
    signals = np.array([[1, np.nan, 3], [4, 5, np.nan]])

    with pytest.raises(ValueError):
        jade_ica(signals)
