import numpy as np
import pytest
import warnings
from scipy.signal import medfilt
from vitalDSP.filtering.artifact_removal import ArtifactRemoval

# Mark all tests in this module as unit tests
pytestmark = [pytest.mark.unit, pytest.mark.fast]

# Filter out expected complex warnings from wavelet operations
# Note: np.ComplexWarning was deprecated and removed in newer NumPy versions
try:
    warnings.filterwarnings("ignore", category=np.ComplexWarning, message="Casting complex values to real discards the imaginary part")
except AttributeError:
    # ComplexWarning no longer exists in newer NumPy versions
    pass


@pytest.fixture
def test_signal():
    return np.random.rand(100)


def test_mean_subtraction(test_signal):
    ar = ArtifactRemoval(test_signal)
    clean_signal = ar.mean_subtraction()
    expected_signal = test_signal - np.mean(test_signal)
    assert np.allclose(clean_signal, expected_signal), "Mean subtraction failed."


def test_baseline_correction(test_signal):
    ar = ArtifactRemoval(test_signal)
    cutoff = 0.5
    fs = 1000
    clean_signal = ar.baseline_correction(cutoff=cutoff, fs=fs)
    assert len(clean_signal) == len(
        test_signal
    ), "Baseline correction output size mismatch."
    assert not np.allclose(
        clean_signal, test_signal
    ), "Baseline correction did not modify the signal."


def test_median_filter_removal(test_signal):
    # Initialize the ArtifactRemoval object with the test signal
    ar = ArtifactRemoval(test_signal)
    # Apply median filter removal with kernel size of 3
    clean_signal = ar.median_filter_removal(kernel_size=3)
    # Apply SciPy's medfilt without padding to match the length of clean_signal
    # Use 'constant' mode in medfilt to avoid padding for consistency with our implementation
    expected_signal = medfilt(test_signal, kernel_size=3)
    # The custom implementation may apply padding to the signal, so slice the signals accordingly
    pad_len = 3 // 2
    expected_signal_no_padding = expected_signal[pad_len:-pad_len]
    clean_signal_no_padding = clean_signal[pad_len:-pad_len]
    # Assert that the clean_signal matches the expected_signal (both without padding)
    assert np.allclose(
        clean_signal_no_padding, expected_signal_no_padding
    ), "Median filter removal failed."


def test_wavelet_denoising(test_signal):
    ar = ArtifactRemoval(test_signal)
    clean_signal = ar.wavelet_denoising(wavelet_type="db", level=2, order=4)
    assert len(clean_signal) == len(
        test_signal
    ), "Wavelet denoising output size mismatch."
    assert not np.allclose(
        clean_signal, test_signal
    ), "Wavelet denoising did not remove noise."

    clean_signal = ar.wavelet_denoising(
        wavelet_type="db", level=2, order=4, smooth="gaussian"
    )
    assert len(clean_signal) == len(
        test_signal
    ), "Wavelet denoising output size mismatch."

    clean_signal = ar.wavelet_denoising(
        wavelet_type="db", level=2, order=4, smooth="median"
    )
    assert len(clean_signal) == len(
        test_signal
    ), "Wavelet denoising output size mismatch."

    clean_signal = ar.wavelet_denoising(
        wavelet_type="db", level=2, order=4, smooth="moving_average"
    )
    assert len(clean_signal) == len(
        test_signal
    ), "Wavelet denoising output size mismatch."

    clean_signal = ar.wavelet_denoising(wavelet_type="haar", level=2, order=4)
    assert len(clean_signal) == len(
        test_signal
    ), "Wavelet denoising output size mismatch."

    clean_signal = ar.wavelet_denoising(wavelet_type="sym", level=2, order=4)
    assert len(clean_signal) == len(
        test_signal
    ), "Wavelet denoising output size mismatch."

    clean_signal = ar.wavelet_denoising(wavelet_type="coif", level=2, order=4)
    assert len(clean_signal) == len(
        test_signal
    ), "Wavelet denoising output size mismatch."

    clean_signal = ar.wavelet_denoising(
        wavelet_type="custom",
        level=2,
        order=4,
        custom_wavelet=np.array([1, 2, 3, 2, 1]),
    )
    assert len(clean_signal) == len(
        test_signal
    ), "Wavelet denoising output size mismatch."

    clean_signal = ar.wavelet_denoising(
        wavelet_type="custom",
        smoothing="gaussian",
        level=2,
        order=4,
        custom_wavelet=np.array([1, 2, 3, 2, 1]),
    )
    assert len(clean_signal) == len(
        test_signal
    ), "Wavelet denoising output size mismatch."

    clean_signal = ar.wavelet_denoising(
        wavelet_type="custom",
        smoothing="median",
        level=2,
        order=4,
        custom_wavelet=np.array([1, 2, 3, 2, 1]),
    )
    assert len(clean_signal) == len(
        test_signal
    ), "Wavelet denoising output size mismatch."

    clean_signal = ar.wavelet_denoising(
        wavelet_type="custom",
        smoothing="moving_average",
        level=2,
        order=4,
        custom_wavelet=np.array([1, 2, 3, 2, 1]),
    )
    assert len(clean_signal) == len(
        test_signal
    ), "Wavelet denoising output size mismatch."

    with pytest.raises(ValueError):
        ar.wavelet_denoising(
            wavelet_type="custom", level=2, order=4, custom_wavelet=None
        )

    with pytest.raises(ValueError):
        ar.wavelet_denoising(wavelet_type="invalid", level=2, order=4)

    with pytest.raises(ValueError):
        ar.wavelet_denoising(
            wavelet_type="custom", smoothing="invalid", level=2, order=4
        )


def test_adaptive_filtering(test_signal):
    reference_signal = np.ones_like(test_signal)
    ar = ArtifactRemoval(test_signal)
    clean_signal = ar.adaptive_filtering(
        reference_signal, learning_rate=0.01, num_iterations=100
    )
    assert len(clean_signal) == len(
        test_signal
    ), "Adaptive filtering output size mismatch."
    assert np.all(
        np.abs(clean_signal - reference_signal) < np.abs(test_signal - reference_signal)
    ), "Adaptive filtering did not work properly."


def test_notch_filter(test_signal):
    ar = ArtifactRemoval(test_signal)
    clean_signal = ar.notch_filter(freq=50, fs=1000, Q=30)
    assert len(clean_signal) == len(test_signal), "Notch filter output size mismatch."
    assert not np.allclose(
        clean_signal, test_signal
    ), "Notch filter did not remove interference."


def test_pca_artifact_removal(test_signal):
    ar = ArtifactRemoval(test_signal)
    clean_signal = ar.pca_artifact_removal(num_components=1, window_size=4, overlap=2)
    assert len(clean_signal) == len(
        test_signal
    ), "PCA artifact removal output size mismatch."
    assert not np.allclose(
        clean_signal, test_signal
    ), "PCA artifact removal did not work properly."

    # with pytest.raises(ValueError):
    #     ar.pca_artifact_removal(num_components=1, window_size=100, overlap=2)


def test_ica_artifact_removal(test_signal):
    ar = ArtifactRemoval(test_signal)
    window_size = 3
    step_size = 1
    clean_signal = ar.ica_artifact_removal(
        num_components=1, window_size=window_size, step_size=step_size, batch_size=200
    )
    assert len(clean_signal) == (
        len(test_signal) // step_size - window_size + step_size
    ), "ICA artifact removal output size mismatch."
    assert not np.allclose(
        clean_signal[: min(len(clean_signal), len(test_signal))],
        test_signal[: min(len(clean_signal), len(test_signal))],
    ), "ICA artifact removal did not work properly."

    clean_signal = ar.ica_artifact_removal(num_components=1, batch_size=200)
    assert (
        len(clean_signal.shape) == 1
    ), "ICA artifact removal doesnot flatten the signal."
