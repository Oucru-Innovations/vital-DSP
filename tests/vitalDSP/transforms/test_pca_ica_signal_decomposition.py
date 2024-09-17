import pytest
import numpy as np
from vitalDSP.transforms.pca_ica_signal_decomposition import (
    PCASignalDecomposition,
    ICASignalDecomposition,
)


def test_pca_init():
    # Test initialization of PCA class
    signals = np.random.rand(5, 100)
    pca = PCASignalDecomposition(signals, n_components=2)

    assert np.array_equal(pca.signals, signals)
    assert pca.n_components == 2


def test_pca_compute_pca():
    # Test PCA computation
    signals = np.random.rand(5, 100)
    pca = PCASignalDecomposition(signals, n_components=2)
    pca_result = pca.compute_pca()

    assert pca_result.shape == (5, 2)

    # Test when n_components is None (default)
    pca = PCASignalDecomposition(signals)
    pca_result = pca.compute_pca()

    assert pca_result.shape == (5, 100)


def test_pca_invalid_input():
    # Test PCA with invalid input
    signals = np.random.rand(5)

    with pytest.raises(ValueError):
        pca = PCASignalDecomposition(signals)
        pca.compute_pca()


def test_ica_init():
    # Test initialization of ICA class
    signals = np.random.rand(5, 100)
    ica = ICASignalDecomposition(signals, max_iter=500, tolerance=1e-5)

    assert np.array_equal(ica.signals, signals)
    assert ica.max_iter == 500
    assert ica.tolerance == 1e-5


def test_ica_compute_ica():
    # Test ICA computation
    signals = np.random.rand(5, 100)
    ica = ICASignalDecomposition(signals)
    ica_result = ica.compute_ica()

    # Check if the shape of the result matches the input
    assert ica_result.shape == (5, 100)


def test_ica_invalid_input():
    # Test ICA with invalid input
    signals = np.random.rand(5)

    with pytest.raises(ValueError):
        ica = ICASignalDecomposition(signals)
        ica.compute_ica()
