import pytest
import numpy as np
from scipy import signal
from vitalDSP.filtering.artifact_removal import ArtifactRemoval
from vitalDSP.filtering.advanced_signal_filtering import AdvancedSignalFiltering
from vitalDSP.filtering.signal_filtering import BandpassFilter, SignalFiltering
from vitalDSP.transforms.vital_transformation import VitalTransformation


@pytest.fixture
def sample_signal():
    # Fixture to generate a sample ECG/PPG signal for testing
    return np.random.randn(1000)


@pytest.fixture
def transformer(sample_signal):
    # Fixture to create an instance of VitalTransformation
    return VitalTransformation(sample_signal, fs=256, signal_type="ecg")


def test_apply_artifact_removal(transformer, mocker):
    # Mocking ArtifactRemoval
    mocker.patch(
        "vitalDSP.filtering.artifact_removal.ArtifactRemoval.baseline_correction",
        return_value=transformer.signal,
    )

    # Test baseline correction
    transformer.apply_artifact_removal(
        method="baseline_correction", options={"cutoff": 0.5}
    )
    assert transformer.signal is not None

    # Test mean subtraction
    mocker.patch(
        "vitalDSP.filtering.artifact_removal.ArtifactRemoval.mean_subtraction",
        return_value=transformer.signal,
    )
    transformer.apply_artifact_removal(method="mean_subtraction")
    assert transformer.signal is not None

    # Test median filter removal
    mocker.patch(
        "vitalDSP.filtering.artifact_removal.ArtifactRemoval.median_filter_removal",
        return_value=transformer.signal,
    )
    transformer.apply_artifact_removal(
        method="median_filter_removal", options={"kernel_size": 3}
    )
    assert transformer.signal is not None

    # Test wavelet denoising
    mocker.patch(
        "vitalDSP.filtering.artifact_removal.ArtifactRemoval.wavelet_denoising",
        return_value=transformer.signal,
    )
    transformer.apply_artifact_removal(
        method="wavelet_denoising",
        options={"wavelet_type": "db", "level": 1, "order": 4},
    )
    assert transformer.signal is not None

    # Test adaptive filtering with reference signal
    reference_signal = np.random.randn(1000)
    mocker.patch(
        "vitalDSP.filtering.artifact_removal.ArtifactRemoval.adaptive_filtering",
        return_value=transformer.signal,
    )
    transformer.apply_artifact_removal(
        method="adaptive_filtering", options={"reference_signal": reference_signal}
    )
    assert transformer.signal is not None

    # Test PCA artifact removal
    mocker.patch(
        "vitalDSP.filtering.artifact_removal.ArtifactRemoval.pca_artifact_removal",
        return_value=transformer.signal,
    )
    transformer.apply_artifact_removal(
        method="pca_artifact_removal", options={"num_components": 1}
    )
    assert transformer.signal is not None

    # Test ICA artifact removal
    mocker.patch(
        "vitalDSP.filtering.artifact_removal.ArtifactRemoval.ica_artifact_removal",
        return_value=transformer.signal,
    )
    transformer.apply_artifact_removal(
        method="ica_artifact_removal",
        options={"num_components": 1, "max_iterations": 1000},
    )
    assert transformer.signal is not None

    # Test invalid method
    with pytest.raises(ValueError):
        transformer.apply_artifact_removal(method="unknown_method")


def test_apply_bandpass_filter(transformer, mocker):
    # Mocking BandpassFilter
    mocker.patch(
        "vitalDSP.filtering.signal_filtering.BandpassFilter.signal_highpass_filter",
        return_value=transformer.signal,
    )
    mocker.patch(
        "vitalDSP.filtering.signal_filtering.BandpassFilter.signal_lowpass_filter",
        return_value=transformer.signal,
    )

    # Test bandpass filter application
    transformer.apply_bandpass_filter(
        options={
            "lowcut": 0.5,
            "highcut": 30,
            "filter_order": 4,
            "filter_type": "butter",
        }
    )
    assert transformer.signal is not None


def test_apply_detrending(transformer):
    # Test linear detrending
    transformer.apply_detrending(options={"detrend_type": "linear"})
    assert transformer.signal is not None

    # Test constant detrending
    transformer.apply_detrending(options={"detrend_type": "constant"})
    assert transformer.signal is not None

    # Test invalid detrending method
    with pytest.raises(ValueError):
        transformer.apply_detrending(options={"detrend_type": "unknown"})


def test_apply_normalization(transformer):
    # Test normalization to range (0, 1)
    transformer.apply_normalization(options={"normalization_range": (0, 1)})
    assert np.min(transformer.signal) >= 0 and np.max(transformer.signal) <= 1


def test_apply_smoothing(transformer, mocker):
    # Mocking SignalFiltering
    mocker.patch(
        "vitalDSP.filtering.signal_filtering.SignalFiltering.moving_average",
        return_value=transformer.signal,
    )
    mocker.patch(
        "vitalDSP.filtering.signal_filtering.SignalFiltering.gaussian",
        return_value=transformer.signal,
    )

    # Test moving average smoothing
    transformer.apply_smoothing(
        options={
            "smoothing_method": "moving_average",
            "window_size": 5,
            "iterations": 2,
        }
    )
    assert transformer.signal is not None

    # Test Gaussian smoothing
    transformer.apply_smoothing(
        options={"smoothing_method": "gaussian", "sigma": 1.0, "iterations": 2}
    )
    assert transformer.signal is not None

    # Test invalid smoothing method
    with pytest.raises(ValueError):
        transformer.apply_smoothing(options={"smoothing_method": "unknown"})


def test_apply_enhancement(transformer):
    # Test squaring enhancement
    transformer.apply_enhancement(options={"enhance_method": "square"})
    # assert np.array_equal(transformer.signal, np.square(transformer.signal))
    assert len(transformer.signal) > 0

    # Test absolute enhancement
    transformer.apply_enhancement(options={"enhance_method": "abs"})
    # assert np.array_equal(transformer.signal, np.abs(transformer.signal))
    assert len(transformer.signal) > 0

    # Test invalid enhancement method
    with pytest.raises(ValueError):
        transformer.apply_enhancement(options={"enhance_method": "unknown"})


def test_apply_advanced_filtering(transformer, mocker):
    # Mocking AdvancedSignalFiltering
    mocker.patch(
        "vitalDSP.filtering.advanced_signal_filtering.AdvancedSignalFiltering.kalman_filter",
        return_value=transformer.signal,
    )

    transformer.apply_advanced_filtering()
    assert transformer.signal is not None

    # Test Kalman filter application
    transformer.apply_advanced_filtering(
        options={"filter_type": "kalman_filter", "R": 0.1, "Q": 0.01}
    )
    assert transformer.signal is not None

    # Test convolution_based_filter filter application
    transformer.apply_advanced_filtering(
        options={
            "filter_type": "convolution_based_filter",
            "kernel_type": "smoothing",
            "kernel_size": 5,
        }
    )
    assert transformer.signal is not None

    # Test convolution_based_filter filter application
    transformer.apply_advanced_filtering(
        options={
            "filter_type": "attention_based_filter",
            "attention_type": "uniform",
            "size": 5,
        }
    )
    assert transformer.signal is not None

    # Test convolution_based_filter filter application
    transformer.apply_advanced_filtering(
        options={
            "filter_type": "adaptive_filtering",
            "desired_signal": transformer.signal,
            "mu": 0.01,
            "filter_order": 5,
        }
    )
    assert transformer.signal is not None

    # Test invalid advanced filtering method
    with pytest.raises(ValueError):
        transformer.apply_advanced_filtering(options={"filter_type": "unknown_filter"})


def test_apply_transformations(transformer, mocker):
    # Mocking the methods used in the transformation sequence
    mocker.patch.object(transformer, "apply_artifact_removal")
    mocker.patch.object(transformer, "apply_bandpass_filter")
    mocker.patch.object(transformer, "apply_detrending")
    mocker.patch.object(transformer, "apply_normalization")
    mocker.patch.object(transformer, "apply_smoothing")
    mocker.patch.object(transformer, "apply_enhancement")
    mocker.patch.object(transformer, "apply_advanced_filtering")

    # Test the complete transformation process
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

    transformed_signal = transformer.apply_transformations(
        options=options, method_order=method_order
    )

    # Assert the signal has been transformed and mock methods were called
    assert transformed_signal is not None
    transformer.apply_artifact_removal.assert_called_once()
    transformer.apply_bandpass_filter.assert_called_once()
    transformer.apply_detrending.assert_called_once()
    transformer.apply_normalization.assert_called_once()
    transformer.apply_smoothing.assert_called_once()
    transformer.apply_enhancement.assert_called_once()
    transformer.apply_advanced_filtering.assert_called_once()
