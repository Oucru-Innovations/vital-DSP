import pytest
import numpy as np
from vitalDSP.advanced_computation.sparse_signal_processing import (
    SparseSignalProcessing,
)


# Sample test signals
@pytest.fixture
def test_signal():
    np.random.seed(42)
    return np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)


@pytest.fixture
def sparse_processing(test_signal):
    return SparseSignalProcessing(test_signal)


def test_sparse_representation_fft(sparse_processing, test_signal):
    # Testing sparse representation using FFT
    sparse_rep = sparse_processing.sparse_representation(np.fft.fft)
    assert isinstance(sparse_rep, np.ndarray)
    assert sparse_rep.shape == test_signal.shape
    assert np.allclose(
        sparse_rep, np.fft.fft(test_signal)
    )  # Ensure it matches FFT output


def test_sparse_representation_custom_basis(sparse_processing, test_signal):
    # Custom basis example (identity function)
    def custom_basis(signal):
        return signal  # Identity transform for testing

    sparse_rep = sparse_processing.sparse_representation(custom_basis)
    assert isinstance(sparse_rep, np.ndarray)
    assert np.array_equal(sparse_rep, test_signal)


def test_thresholding(sparse_processing, test_signal):
    # Testing thresholding with FFT representation
    sparse_rep = sparse_processing.sparse_representation(np.fft.fft)
    thresholded_sparse = sparse_processing.thresholding(sparse_rep, threshold=0.1)
    assert isinstance(thresholded_sparse, np.ndarray)
    assert sparse_rep.shape == thresholded_sparse.shape
    assert np.all(
        np.abs(thresholded_sparse[np.abs(sparse_rep) < 0.1]) == 0
    )  # Ensure thresholding worked


def test_thresholding_all_below_threshold(sparse_processing):
    # Test case where all values are below the threshold
    sparse_rep = np.array([0.01, 0.02, 0.05, 0.03, 0.001])
    thresholded_sparse = sparse_processing.thresholding(sparse_rep, threshold=0.1)
    assert np.all(thresholded_sparse == 0)  # All values should be zero


def test_thresholding_all_above_threshold(sparse_processing):
    # Test case where all values are above the threshold
    sparse_rep = np.array([1.0, 0.5, 0.8, 1.2])
    thresholded_sparse = sparse_processing.thresholding(sparse_rep, threshold=0.1)
    assert np.array_equal(
        thresholded_sparse, sparse_rep
    )  # No values should be set to zero


def test_reconstruction_fft(sparse_processing, test_signal):
    # Testing reconstruction using inverse FFT
    sparse_rep = sparse_processing.sparse_representation(np.fft.fft)
    thresholded_sparse = sparse_processing.thresholding(sparse_rep, threshold=0.1)
    reconstructed_signal = sparse_processing.reconstruction(
        thresholded_sparse, np.fft.ifft
    )
    assert isinstance(reconstructed_signal, np.ndarray)
    assert reconstructed_signal.shape == test_signal.shape
    assert np.allclose(
        reconstructed_signal, np.fft.ifft(thresholded_sparse)
    )  # Check reconstruction accuracy


def test_reconstruction_custom_basis(sparse_processing, test_signal):
    # Custom inverse basis example (identity function)
    def custom_inverse_basis(signal):
        return signal  # Identity inverse transform

    sparse_rep = sparse_processing.sparse_representation(custom_inverse_basis)
    reconstructed_signal = sparse_processing.reconstruction(
        sparse_rep, custom_inverse_basis
    )
    assert isinstance(reconstructed_signal, np.ndarray)
    assert np.array_equal(
        reconstructed_signal, test_signal
    )  # Ensure the reconstruction matches the original signal
