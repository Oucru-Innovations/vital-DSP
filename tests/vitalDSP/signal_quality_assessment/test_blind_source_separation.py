import pytest
import numpy as np
from vitalDSP.signal_quality_assessment.blind_source_separation import center_signal, whiten_signal, ica_artifact_removal, pca_artifact_removal, jade_ica

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

# def test_jade_ica():
#     # Test JADE ICA with a simple signal
#     signals = np.array([[1, 2, 3], [4, 5, 6]])
#     separated_signals = jade_ica(signals)

#     # Check the shape of separated signals
#     assert separated_signals.shape == signals.shape

#     # Ensure numerical stability
#     assert np.all(np.abs(separated_signals) < 10)
