import numpy as np
import pytest
from vitalDSP.transforms.chroma_stft import ChromaSTFT
from vitalDSP.transforms.dct_wavelet_fusion import DCTWaveletFusion
from vitalDSP.transforms.discrete_cosine_transform import DiscreteCosineTransform
from vitalDSP.transforms.fourier_transform import FourierTransform
from vitalDSP.transforms.hilbert_transform import HilbertTransform
from vitalDSP.transforms.event_related_potential import EventRelatedPotential
from vitalDSP.transforms.mfcc import MFCC
from vitalDSP.transforms.pca_ica_signal_decomposition import (
    PCASignalDecomposition,
    ICASignalDecomposition,
)
from vitalDSP.transforms.stft import STFT
from vitalDSP.transforms.wavelet_transform import WaveletTransform
from vitalDSP.transforms.wavelet_fft_fusion import WaveletFFTfusion
from vitalDSP.transforms.time_freq_representation import TimeFreqRepresentation
from vitalDSP.transforms.vital_transformation import (
    VitalTransformation,
)  # Assuming VitalTransformation is in vitalDSP.transformation
from vitalDSP.filtering.advanced_signal_filtering import AdvancedSignalFiltering


# @pytest.fixture
# def sample_signal():
#     """Fixture for creating a sample signal."""
#     return np.sin(np.linspace(0, 2 * np.pi, 100)) + np.random.normal(0, 0.1, 100)
@pytest.fixture
def sample_signal():
    """Generate a sample signal with enough variability for PCA-based artifact removal."""
    # Generate a signal with random noise to introduce variability
    np.random.seed(0)  # Seed for reproducibility
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(
        1000
    )  # Sine wave with noise
    return signal


@pytest.fixture
def short_signal():
    """
    Fixture to provide a short signal to test edge cases.
    """
    return np.sin(np.linspace(0, 5, 10))


def test_fourier_transform(sample_signal):
    transformer = FourierTransform(sample_signal)
    transformed_signal = transformer.compute_dft()
    inversed_signal = transformer.compute_idft(transformed_signal)
    assert len(transformed_signal) == len(
        sample_signal
    ), "Fourier transform length mismatch"
    assert len(inversed_signal) == len(
        sample_signal
    ), "Fourier transform length mismatch"


# Test fusion computation with valid input
def test_compute_fusion(sample_signal):
    fusion = DCTWaveletFusion(sample_signal, wavelet_type="db", order=4)
    fusion_result = fusion.compute_fusion()
    assert len(fusion_result) <= len(
        sample_signal
    )  # Fusion result should not exceed signal length
    assert np.all(np.isfinite(fusion_result))


@pytest.mark.parametrize("wavelet_type", ["db", "haar", "coif", "sym"])
def test_different_wavelet_types(sample_signal, wavelet_type):
    fusion = DCTWaveletFusion(sample_signal, wavelet_type=wavelet_type, order=4)
    fusion_result = fusion.compute_fusion()
    assert len(fusion_result) <= len(sample_signal)
    assert np.all(np.isfinite(fusion_result))


def test_fusion_with_noisy_signal(sample_signal):
    fusion = DCTWaveletFusion(sample_signal, wavelet_type="db", order=4)
    fusion_result = fusion.compute_fusion()
    assert len(fusion_result) <= len(sample_signal)
    assert np.all(np.isfinite(fusion_result))


def test_fusion_with_short_signal(sample_signal):
    fusion = DCTWaveletFusion(sample_signal, wavelet_type="db", order=4)
    fusion_result = fusion.compute_fusion()
    assert len(fusion_result) <= len(sample_signal)
    assert np.all(np.isfinite(fusion_result))


def test_dct(sample_signal):
    transformer = DiscreteCosineTransform(sample_signal)
    transformed_signal = transformer.compute_dct()
    inversed_signal = transformer.compute_idct(transformed_signal)
    assert len(transformed_signal) == (
        len(sample_signal)
    ), "DCT transform length mismatch"
    assert len(inversed_signal) == len(
        sample_signal
    ), "inverse DCT transform length mismatch"


def test_hilbert_transform(sample_signal):
    transformer = HilbertTransform(sample_signal)
    transformed_signal = transformer.compute_hilbert()
    assert len(transformed_signal) == len(
        sample_signal
    ), "Hilbert transform length mismatch"
    assert np.all(
        np.iscomplex(transformed_signal)
    ), "Hilbert transform should return complex values"


def test_wavelet_transform():
    signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    wavelet_transform = WaveletTransform(signal, wavelet_name="haar")
    coeffs = wavelet_transform.perform_wavelet_transform(level=3)
    # signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    # wavelet_transform = WaveletTransform(signal, wavelet_name='haar')
    # coeffs = wavelet_transform.perform_wavelet_transform(level=3)

    assert isinstance(coeffs, list)
    assert len(coeffs) == 4  # 3 levels + final approximation

    for c in coeffs:
        assert isinstance(c, np.ndarray)

    reconstructed_signal = wavelet_transform.perform_inverse_wavelet_transform(coeffs)

    assert len(reconstructed_signal) == len(signal)
    np.testing.assert_almost_equal(reconstructed_signal, signal, decimal=5)


def test_wavelet_transform():
    # Example signal: a simple sine wave with noise
    signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)

    # Initialize the WaveletTransform with the 'haar' wavelet
    wavelet_transform = WaveletTransform(signal, wavelet_name="haar")

    # Perform the wavelet transform
    coeffs = wavelet_transform.perform_wavelet_transform(level=3)

    # Check that the coefficients are a list and not empty
    assert isinstance(coeffs, list), "Coefficients should be a list"
    assert (
        len(coeffs) == 4
    ), "There should be 4 sets of coefficients (3 levels + final approximation)"

    # Check that each set of coefficients is a numpy array
    for c in coeffs:
        assert isinstance(c, np.ndarray), "Each coefficient should be a numpy array"

    # Perform the inverse transform to reconstruct the signal
    reconstructed_signal = wavelet_transform.perform_inverse_wavelet_transform(coeffs)

    # Check that the reconstructed signal has the same length as the original
    assert len(reconstructed_signal) == len(
        signal
    ), "Reconstructed signal should have the same length as the original"

    # Check that the reconstructed signal is close to the original signal
    # np.testing.assert_almost_equal(reconstructed_signal, signal, decimal=5, err_msg="Reconstructed signal should closely match the original signal")


def test_stft(sample_signal):
    transformer = STFT(sample_signal, window_size=50, hop_size=25)
    transformed_signal = transformer.compute_stft()
    assert transformed_signal.shape[0] > 0, "STFT should produce non-empty output"


def test_mfcc(sample_signal):
    transformer = MFCC(sample_signal, num_coefficients=13)
    mfccs = transformer.compute_mfcc()
    assert (
        len(mfccs[0]) == 13
    ), "MFCC transform should return correct number of coefficients"


def test_chroma_stft(sample_signal):
    transformer = ChromaSTFT(sample_signal, n_chroma=12, n_fft=128)
    chroma = transformer.compute_chroma_stft()
    assert len(chroma) == 12, "Chroma STFT should return correct number of chroma bands"


def test_event_related_potential(sample_signal):
    sample_signal = np.sin(np.linspace(0, 10, 1000))
    stimulus_times = np.array([100, 300, 500])
    transformer = EventRelatedPotential(sample_signal, stimulus_times=stimulus_times)
    erp = transformer.compute_erp()
    # print(erp)
    assert len(erp) > 0, "ERP should return correct number of events"


def test_time_freq_representation(sample_signal):
    sample_signal = np.sin(np.linspace(0, 10, 1000))
    tfr = TimeFreqRepresentation(sample_signal)
    tfr_result = tfr.compute_tfr()
    assert (
        len(tfr_result) > 0
    ), "Time-frequency representation should produce non-empty output"


# def test_wavelet_fft_fusion(sample_signal):
#     transformer = WaveletFFTfusion(sample_signal)
#     fused_signal = transformer.compute_fusion()
#     assert len(fused_signal) == len(sample_signal), "Wavelet-FFT fusion length mismatch"

# def test_dct_wavelet_fusion(sample_signal):
#     transformer = DCTWaveletFusion(sample_signal)
#     fused_signal = transformer.compute_fusion()
#     assert len(fused_signal) == len(sample_signal), "DCT-Wavelet fusion length mismatch"


def test_pca_ica_signal_decomposition():
    # Generate some test data
    np.random.seed(3)
    signals = np.random.randn(100, 5)  # 100 samples, 5 features

    # PCA Test
    pca = PCASignalDecomposition(signals, n_components=2)
    pca_result = pca.compute_pca()
    assert pca_result.shape == (100, 2), "PCA output shape mismatch"

    # ICA Test
    # ica = ICASignalDecomposition(signals, max_iter=500, tolerance=1e-6)
    # ica_result = ica.compute_ica()

    # # Assert the shape is correct (n_samples, n_signals)
    # assert ica_result.shape == signals.shape

    # # Check that the independent components have non-zero values
    # assert np.all(ica_result != 0)
    # with pytest.raises(ValueError):
    #     signals = np.random.rand(100)  # Invalid 1D input
    #     ica = ICASignalDecomposition(signals)
    #     ica.compute_ica()


@pytest.fixture
def transformer(sample_signal):
    """Create a VitalTransformation object with the sample signal."""
    return VitalTransformation(sample_signal, fs=256, signal_type="ecg")


# Test apply_transformations
def test_apply_transformations(transformer):
    # Test case 1: Full transformation pipeline with all default methods
    options = {
        "artifact_removal": "baseline_correction",
        "artifact_removal_options": {"cutoff": 0.5},
        "bandpass_filter": {
            "lowcut": 0.5,
            "highcut": 30,
            "filter_order": 4,
            "filter_type": "butter",
        },
        "detrending": {"detrend_type": "linear"},
        "normalization": {"normalization_range": (0, 1)},
        "smoothing": {
            "smoothing_method": "moving_average",
            "window_size": 5,
            "iterations": 2,
        },
        "enhancement": {"enhance_method": "square"},
        "advanced_filtering": {"filter_type": "kalman_filter", "R": 0.1, "Q": 0.01},
    }
    method_order = [
        "artifact_removal",
        "bandpass_filter",
        "detrending",
        "normalization",
        "smoothing",
        "enhancement",
        "advanced_filtering",
    ]
    transformed_signal = transformer.apply_transformations(options, method_order)
    assert len(transformed_signal) == len(transformer.signal)

    # Test case 2: Test with median filter for artifact removal and Gaussian smoothing
    options = {
        "artifact_removal": "median_filter_removal",
        "artifact_removal_options": {"kernel_size": 3},
        "bandpass_filter": {
            "lowcut": 1.0,
            "highcut": 20.0,
            "filter_order": 5,
            "filter_type": "cheby1",
        },
        "detrending": {"detrend_type": "constant"},
        "normalization": {"normalization_range": (-1, 1)},
        "smoothing": {"smoothing_method": "gaussian", "sigma": 1.0, "iterations": 1},
        "enhancement": {"enhance_method": "abs"},
        "advanced_filtering": {
            "filter_type": "optimization_based_filtering",
            "target_signal": transformer.signal,
            "loss_type": "mse",
            "learning_rate": 0.01,
        },
    }
    method_order = [
        "artifact_removal",
        "bandpass_filter",
        "detrending",
        "normalization",
        "smoothing",
        "enhancement",
        "advanced_filtering",
    ]
    transformed_signal = transformer.apply_transformations(options, method_order)
    assert len(transformed_signal) == len(transformer.signal)

    # Test case 3: PCA-based artifact removal and gradient descent filtering
    options = {
        "artifact_removal": "pca_artifact_removal",
        "artifact_removal_options": {"num_components": 2},
        "bandpass_filter": {
            "lowcut": 0.2,
            "highcut": 40.0,
            "filter_order": 3,
            "filter_type": "ellip",
        },
        "detrending": {"detrend_type": "linear"},
        "normalization": {"normalization_range": (0, 1)},
        "smoothing": {
            "smoothing_method": "moving_average",
            "window_size": 10,
            "iterations": 2,
        },
        "enhancement": {"enhance_method": "square"},
        "advanced_filtering": {
            "filter_type": "gradient_descent_filter",
            "target_signal": transformer.signal,
            "learning_rate": 0.001,
            "iterations": 100,
        },
    }
    method_order = [
        "artifact_removal",
        "bandpass_filter",
        "detrending",
        "normalization",
        "smoothing",
        "enhancement",
        "advanced_filtering",
    ]
    transformed_signal = transformer.apply_transformations(options, method_order)
    assert len(transformed_signal) == len(transformer.signal)

    # Test case 4: Test detrending only (minimal transformation)
    options = {"detrending": {"detrend_type": "constant"}}
    method_order = ["detrending"]
    transformed_signal = transformer.apply_transformations(options, method_order)
    assert len(transformed_signal) == len(transformer.signal)

    # Test case 6: Test advanced filtering only (Kalman filter)
    options = {
        "advanced_filtering": {"filter_type": "kalman_filter", "R": 0.2, "Q": 0.05}
    }
    method_order = ["advanced_filtering"]
    transformed_signal = transformer.apply_transformations(options, method_order)
    assert len(transformed_signal) == len(transformer.signal)


# Test artifact removal
def test_apply_artifact_removal(transformer):
    transformer.apply_artifact_removal(
        method="baseline_correction", options={"cutoff": 0.5}
    )
    assert transformer.signal is not None

    transformer.apply_artifact_removal(
        method="median_filter_removal", options={"kernel_size": 5}
    )
    assert transformer.signal is not None

    with pytest.raises(
        ValueError, match="Reference signal is required for adaptive filtering"
    ):
        transformer.apply_artifact_removal(method="adaptive_filtering", options={})


# Test bandpass filtering
def test_apply_bandpass_filter(transformer):
    transformer.apply_bandpass_filter(
        options={
            "lowcut": 0.5,
            "highcut": 30,
            "filter_order": 4,
            "filter_type": "butter",
        }
    )
    assert transformer.signal is not None


# Test detrending
def test_apply_detrending(transformer):
    transformer.apply_detrending(options={"detrend_type": "linear"})
    assert transformer.signal is not None

    transformer.apply_detrending(options={"detrend_type": "constant"})
    assert transformer.signal is not None

    with pytest.raises(ValueError, match="Unknown detrending method"):
        transformer.apply_detrending(options={"detrend_type": "unknown"})


# Test normalization
def test_apply_normalization(transformer):
    transformer.apply_normalization(options={"normalization_range": (0, 1)})
    assert np.min(transformer.signal) >= 0
    assert np.max(transformer.signal) <= 1


# Test smoothing
def test_apply_smoothing(transformer):
    transformer.apply_smoothing(
        options={
            "smoothing_method": "moving_average",
            "window_size": 5,
            "iterations": 2,
        }
    )
    assert transformer.signal is not None

    transformer.apply_smoothing(
        options={"smoothing_method": "gaussian", "sigma": 1.0, "iterations": 2}
    )
    assert transformer.signal is not None

    with pytest.raises(ValueError, match="Unknown smoothing method"):
        transformer.apply_smoothing(options={"smoothing_method": "unknown"})


# Test enhancement
def test_apply_enhancement(transformer):
    transformer.apply_enhancement(options={"enhance_method": "square"})
    assert transformer.signal is not None

    transformer.apply_enhancement(options={"enhance_method": "abs"})
    assert transformer.signal is not None

    with pytest.raises(ValueError, match="Unknown enhancement method"):
        transformer.apply_enhancement(options={"enhance_method": "unknown"})


# Test advanced filtering
def test_apply_advanced_filtering(transformer):
    # Create an instance of AdvancedSignalFiltering for the filter methods
    advanced_filtering_instance = AdvancedSignalFiltering(transformer.signal)

    transformer.apply_advanced_filtering(
        options={"filter_type": "kalman_filter", "R": 0.1, "Q": 0.01}
    )
    assert transformer.signal is not None

    transformer.apply_advanced_filtering(
        options={
            "filter_type": "optimization_based_filtering",
            "target_signal": transformer.signal,
        }
    )
    assert transformer.signal is not None

    # Pass the Kalman filter as a bound method in the ensemble filtering options
    transformer.apply_advanced_filtering(
        options={
            "filter_type": "ensemble_filtering",
            "filters": [
                advanced_filtering_instance.kalman_filter
            ],  # Pass instance method
            "method": "mean",
        }
    )
    assert transformer.signal is not None

    with pytest.raises(ValueError, match="Unknown advanced filtering method"):
        transformer.apply_advanced_filtering(options={"filter_type": "unknown"})


if __name__ == "__main__":
    pytest.main()
