import pytest
import numpy as np
from vitalDSP.transforms.fourier_transform import FourierTransform


@pytest.fixture
def example_signal():
    # Generate an example signal: a sinusoidal wave with noise
    np.random.seed(42)
    return np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)


def test_init(example_signal):
    # Test the initialization of the class
    ft = FourierTransform(example_signal)
    assert np.array_equal(ft.signal, example_signal)


def test_compute_dft(example_signal):
    # Test the DFT computation
    ft = FourierTransform(example_signal)
    frequency_content = ft.compute_dft()

    # Assert that the DFT output has the correct length and type
    assert isinstance(frequency_content, np.ndarray)
    assert len(frequency_content) == len(example_signal)


def test_compute_idft(example_signal):
    # Test the IDFT computation
    ft = FourierTransform(example_signal)
    frequency_content = ft.compute_dft()
    reconstructed_signal = ft.compute_idft(frequency_content)

    # The IDFT should return a signal of the same length
    assert isinstance(reconstructed_signal, np.ndarray)
    assert len(reconstructed_signal) == len(example_signal)


def test_filter_frequencies_low_cutoff(example_signal):
    # Test the filtering with a low cutoff frequency
    ft = FourierTransform(example_signal)
    filtered_signal = ft.filter_frequencies(low_cutoff=0.5, fs=10.0)

    # Assert that the filtered signal is an ndarray and has the correct length
    assert isinstance(filtered_signal, np.ndarray)
    assert len(filtered_signal) == len(example_signal)


def test_filter_frequencies_high_cutoff(example_signal):
    # Test the filtering with a high cutoff frequency
    ft = FourierTransform(example_signal)
    filtered_signal = ft.filter_frequencies(high_cutoff=2.0, fs=10.0)

    # Assert that the filtered signal is an ndarray and has the correct length
    assert isinstance(filtered_signal, np.ndarray)
    assert len(filtered_signal) == len(example_signal)


def test_filter_frequencies_bandpass(example_signal):
    # Test the filtering with both low and high cutoff frequencies (bandpass)
    ft = FourierTransform(example_signal)
    filtered_signal = ft.filter_frequencies(low_cutoff=0.5, high_cutoff=2.0, fs=10.0)

    # Assert that the filtered signal is an ndarray and has the correct length
    assert isinstance(filtered_signal, np.ndarray)
    assert len(filtered_signal) == len(example_signal)
