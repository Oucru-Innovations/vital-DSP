import pytest
import numpy as np
from vitalDSP.visualization.artefact_removal_visualization import ArtifactRemovalVisualization
from vitalDSP.filtering.artifact_removal import ArtifactRemoval

@pytest.fixture
def signal():
    return np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)

@pytest.fixture
def reference_signal():
    return np.sin(np.linspace(0, 10, 100)) * 0.5

def test_set_reference_signal(signal, reference_signal):
    # Test setting the reference signal in ArtifactRemoval
    ar_visualization = ArtifactRemovalVisualization(signal, reference_signal)
    ar_visualization._set_reference_signal()
    
    # Check if reference signal is correctly set
    assert np.all(ar_visualization.artifact_removal.signal == signal)
    assert np.all(ar_visualization.reference_signal == reference_signal)

def test_visualize_artifact_removal_mean_subtraction(signal, reference_signal):
    ar_visualization = ArtifactRemovalVisualization(signal, reference_signal)
    ar_visualization.visualize_artifact_removal(method="mean_subtraction")
    
    # Perform mean subtraction manually and compare
    expected_output = signal - np.mean(signal)
    cleaned_signal = ar_visualization.artifact_removal.mean_subtraction()
    assert np.allclose(cleaned_signal, expected_output)

def test_visualize_artifact_removal_baseline_correction(signal, reference_signal):
    ar_visualization = ArtifactRemovalVisualization(signal, reference_signal)
    ar_visualization.visualize_artifact_removal(method="baseline_correction")

    # Apply baseline correction
    cleaned_signal = ar_visualization.artifact_removal.baseline_correction(cutoff=0.5, fs=1000)
    assert cleaned_signal is not None
    assert len(cleaned_signal) == len(signal)

def test_visualize_artifact_removal_median_filter_removal(signal, reference_signal):
    ar_visualization = ArtifactRemovalVisualization(signal, reference_signal)
    ar_visualization.visualize_artifact_removal(method="median_filter_removal")

    # Apply median filter removal
    cleaned_signal = ar_visualization.artifact_removal.median_filter_removal(kernel_size=3)
    assert cleaned_signal is not None
    assert len(cleaned_signal) == len(signal)

def test_visualize_artifact_removal_wavelet_denoising(signal, reference_signal):
    ar_visualization = ArtifactRemovalVisualization(signal, reference_signal)
    ar_visualization.visualize_artifact_removal(method="wavelet_denoising")

    # Apply wavelet denoising
    cleaned_signal = ar_visualization.artifact_removal.wavelet_denoising(level=1)
    assert cleaned_signal is not None
    assert len(cleaned_signal) == len(signal)

def test_visualize_artifact_removal_adaptive_filtering(signal, reference_signal):
    ar_visualization = ArtifactRemovalVisualization(signal, reference_signal)
    ar_visualization.visualize_artifact_removal(method="adaptive_filtering")

    # Apply adaptive filtering
    cleaned_signal = ar_visualization.artifact_removal.adaptive_filtering(reference_signal)
    assert cleaned_signal is not None
    assert len(cleaned_signal) == len(signal)

def test_visualize_artifact_removal_notch_filter(signal, reference_signal):
    ar_visualization = ArtifactRemovalVisualization(signal, reference_signal)
    ar_visualization.visualize_artifact_removal(method="notch_filter")

    # Apply notch filter
    cleaned_signal = ar_visualization.artifact_removal.notch_filter(freq=50, fs=1000)
    assert cleaned_signal is not None
    assert len(cleaned_signal) == len(signal)

# def test_visualize_artifact_removal_pca_artifact_removal(signal, reference_signal):
#     # Adjusted signal to have sufficient dimensionality for PCA
#     test_signal = np.array([np.linspace(0, 1, 100), np.linspace(1, 2, 100)]).T
#     ar_visualization = ArtifactRemovalVisualization(test_signal, reference_signal)
#     ar_visualization.visualize_artifact_removal(method="pca_artifact_removal")

#     # Apply PCA artifact removal
#     cleaned_signal = ar_visualization.artifact_removal.pca_artifact_removal(num_components=1)
#     assert cleaned_signal is not None
#     assert len(cleaned_signal) == len(test_signal)

# def test_visualize_artifact_removal_ica_artifact_removal(signal, reference_signal):
#     # Adjusted signal to have sufficient dimensionality for ICA
#     test_signal = np.array([np.linspace(0, 1, 100), np.linspace(1, 2, 100)]).T
#     ar_visualization = ArtifactRemovalVisualization(test_signal, reference_signal)
#     ar_visualization.visualize_artifact_removal(method="ica_artifact_removal")

#     # Apply ICA artifact removal
#     cleaned_signal = ar_visualization.artifact_removal.ica_artifact_removal(num_components=1)
#     assert cleaned_signal is not None
#     assert len(cleaned_signal) == len(test_signal)

def test_visualize_all_removals(signal, reference_signal):
    ar_visualization = ArtifactRemovalVisualization(signal, reference_signal)
    ar_visualization.visualize_all_removals()

    # Check that all methods are applied correctly
    assert ar_visualization.artifact_removal.mean_subtraction() is not None
    assert ar_visualization.artifact_removal.baseline_correction() is not None
    assert ar_visualization.artifact_removal.median_filter_removal() is not None
    assert ar_visualization.artifact_removal.wavelet_denoising() is not None
    assert ar_visualization.artifact_removal.adaptive_filtering(reference_signal) is not None
    assert ar_visualization.artifact_removal.notch_filter() is not None
    # assert ar_visualization.artifact_removal.pca_artifact_removal() is not None
    # assert ar_visualization.artifact_removal.ica_artifact_removal() is not None

def test_invalid_method(signal, reference_signal):
    ar_visualization = ArtifactRemovalVisualization(signal, reference_signal)
    with pytest.raises(ValueError):
        ar_visualization.visualize_artifact_removal(method="invalid_method")
