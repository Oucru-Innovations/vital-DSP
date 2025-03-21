import numpy as np
import pytest
from vitalDSP.preprocess.preprocess_operations import (
    preprocess_signal,
    estimate_baseline,
    respiratory_filtering
)
from vitalDSP.preprocess.noise_reduction import (
    wavelet_denoising,
    savgol_denoising,
    median_denoising,
    gaussian_denoising,
    moving_average_denoising,
)


@pytest.fixture
def noisy_signal():
    return np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.2, 100)


def test_wavelet_denoising(noisy_signal):
    denoised_signal = wavelet_denoising(noisy_signal, wavelet_name="haar", level=2)
    assert len(denoised_signal) == len(noisy_signal)
    # assert np.var(denoised_signal) < np.var(noisy_signal)


def test_savgol_denoising(noisy_signal):
    denoised_signal = savgol_denoising(noisy_signal, window_length=5, polyorder=2)
    assert len(denoised_signal) == len(noisy_signal)
    assert np.var(denoised_signal) < np.var(noisy_signal)


def test_median_denoising(noisy_signal):
    denoised_signal = median_denoising(noisy_signal, kernel_size=3)
    assert len(denoised_signal) == len(noisy_signal)
    # assert np.var(denoised_signal) < np.var(noisy_signal)


def test_gaussian_denoising(noisy_signal):
    denoised_signal = gaussian_denoising(noisy_signal, sigma=1.5)
    assert len(denoised_signal) == len(noisy_signal)
    assert np.var(denoised_signal) < np.var(noisy_signal)


def test_moving_average_denoising(noisy_signal):
    denoised_signal = moving_average_denoising(noisy_signal, window_size=5)
    assert len(denoised_signal) == len(noisy_signal)
    # assert np.var(denoised_signal) < np.var(noisy_signal)


@pytest.fixture
def test_signal():
    return np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.2, 100)


def test_preprocess_bandpass_wavelet(test_signal):
    sampling_rate = 1000
    preprocessed_signal = preprocess_signal(
        test_signal,
        sampling_rate,
        filter_type="bandpass",
        noise_reduction_method="wavelet",
    )
    assert len(preprocessed_signal) == len(test_signal)
    # assert np.var(preprocessed_signal) < np.var(test_signal)


def test_preprocess_butterworth_median(test_signal):
    sampling_rate = 1000
    preprocessed_signal = preprocess_signal(
        test_signal,
        sampling_rate,
        filter_type="butterworth",
        noise_reduction_method="median",
    )
    assert len(preprocessed_signal) == len(test_signal)
    assert np.var(preprocessed_signal) < np.var(test_signal)


def test_preprocess_chebyshev_gaussian(test_signal):
    sampling_rate = 1000
    preprocessed_signal = preprocess_signal(
        test_signal,
        sampling_rate,
        filter_type="chebyshev",
        noise_reduction_method="gaussian",
    )
    assert len(preprocessed_signal) == len(test_signal)
    assert np.var(preprocessed_signal) < np.var(test_signal)


def test_preprocess_invalid_filter(test_signal):
    sampling_rate = 1000
    with pytest.raises(
        ValueError,
        match="Unsupported filter type. Choose from 'bandpass', 'butterworth', 'chebyshev', or 'elliptic'.",
    ):
        preprocess_signal(
            test_signal,
            sampling_rate,
            filter_type="invalid",
            noise_reduction_method="gaussian",
        )


def test_preprocess_invalid_noise_reduction(test_signal):
    sampling_rate = 1000
    with pytest.raises(
        ValueError,
        match="Unsupported noise reduction method. Choose from 'wavelet', 'savgol', 'median', 'gaussian', or 'moving_average'.",
    ):
        preprocess_signal(
            test_signal,
            sampling_rate,
            filter_type="bandpass",
            noise_reduction_method="invalid",
        )


def test_preprocess_elliptic_savgol(test_signal):
    sampling_rate = 1000
    preprocessed_signal = preprocess_signal(
        test_signal,
        sampling_rate,
        filter_type="elliptic",
        noise_reduction_method="savgol",
    )
    assert len(preprocessed_signal) == len(test_signal)
    assert np.var(preprocessed_signal) < np.var(test_signal)


def test_preprocess_ignore_ignore(test_signal):
    sampling_rate = 1000
    preprocessed_signal = preprocess_signal(
        test_signal,
        sampling_rate,
        filter_type="ignore",
        noise_reduction_method="ignore",
    )
    assert len(preprocessed_signal) == len(test_signal)
    assert np.var(preprocessed_signal) == np.var(test_signal)


def test_preprocess_bandpass_moving_average(test_signal):
    sampling_rate = 1000
    preprocessed_signal = preprocess_signal(
        test_signal,
        sampling_rate,
        filter_type="bandpass",
        noise_reduction_method="moving_average",
    )
    assert len(preprocessed_signal) == len(test_signal)
    
    signal_baseline = estimate_baseline(preprocessed_signal,
                                        sampling_rate,
                                        method="low_pass")
    assert len(signal_baseline) == len(test_signal)
    signal_baseline = estimate_baseline(preprocessed_signal,
                                        sampling_rate,
                                        method="polynomial_fit")
    assert len(signal_baseline) == len(test_signal)
    signal_baseline = estimate_baseline(preprocessed_signal,
                                        sampling_rate,
                                        method="median_filter")
    assert len(signal_baseline) == len(test_signal)
    with pytest.raises(
        ValueError,
        match="Unsupported baseline estimation method: invalid",
    ):
        estimate_baseline(preprocessed_signal,
                        sampling_rate,
                        method="invalid")
    # assert np.var(preprocessed_signal) < np.var(test_signal)
