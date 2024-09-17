import pytest
import numpy as np
from scipy.signal import coherence
from unittest.mock import patch
from vitalDSP.preprocess.preprocess_operations import preprocess_signal
from vitalDSP.respiratory_analysis.respiratory_analysis import PreprocessConfig
from vitalDSP.physiological_features.coherence_analysis import CoherenceAnalysis


# Sample signals for testing
@pytest.fixture
def sample_signals():
    signal1 = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)
    signal2 = np.sin(np.linspace(0, 10, 1000) + 0.1) + np.random.normal(0, 0.1, 1000)
    return signal1, signal2


# Sample PreprocessConfig for testing
@pytest.fixture
def sample_preprocess_config():
    return PreprocessConfig()


# Test the initialization of CoherenceAnalysis
def test_init(sample_signals):
    signal1, signal2 = sample_signals
    coherence_analysis = CoherenceAnalysis(signal1, signal2, fs=500)
    assert coherence_analysis.signal1 is signal1
    assert coherence_analysis.signal2 is signal2
    assert coherence_analysis.fs == 500


# Mock the preprocess_signal function for testing
@patch("vitalDSP.physiological_features.coherence_analysis.preprocess_signal")
def test_preprocess_signals(mock_preprocess, sample_signals, sample_preprocess_config):
    signal1, signal2 = sample_signals
    mock_preprocess.side_effect = (
        lambda signal, *args, **kwargs: signal
    )  # Return the signal as-is

    coherence_analysis = CoherenceAnalysis(signal1, signal2, fs=500)

    preprocessed_signal1, preprocessed_signal2 = coherence_analysis.preprocess_signals(
        preprocess_config1=sample_preprocess_config,
        preprocess_config2=sample_preprocess_config,
    )

    # Assert that preprocess_signal was called twice
    assert mock_preprocess.call_count == 2
    assert np.array_equal(preprocessed_signal1, signal1)
    assert np.array_equal(preprocessed_signal2, signal2)


def test_align_signals(sample_signals):
    signal1, signal2 = sample_signals
    coherence_analysis = CoherenceAnalysis(signal1, signal2, fs=500)

    aligned_signal1, aligned_signal2 = coherence_analysis.align_signals(
        signal1, signal2
    )
    assert len(aligned_signal1) == len(aligned_signal2)


# Mock the preprocess_signal function and compute_coherence
@patch("vitalDSP.physiological_features.coherence_analysis.preprocess_signal")
def test_compute_coherence(mock_preprocess, sample_signals, sample_preprocess_config):
    signal1, signal2 = sample_signals
    mock_preprocess.side_effect = (
        lambda signal, *args, **kwargs: signal
    )  # Return the signal as-is

    coherence_analysis = CoherenceAnalysis(signal1, signal2, fs=500)

    f, Cxy = coherence_analysis.compute_coherence(
        preprocess_config1=sample_preprocess_config,
        preprocess_config2=sample_preprocess_config,
    )

    # Assert that preprocess_signal was called twice
    assert mock_preprocess.call_count == 2
    assert len(f) > 0
    assert len(Cxy) > 0


# Suppress the graphical output of matplotlib in the test
@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.figure")
def test_plot_coherence(mock_figure, mock_show, sample_signals):
    signal1, signal2 = sample_signals
    coherence_analysis = CoherenceAnalysis(signal1, signal2, fs=500)
    f = np.linspace(0, 100, 100)
    Cxy = np.random.rand(100)

    coherence_analysis.plot_coherence(f, Cxy)

    # Ensure that the plot was created and show was called once
    mock_figure.assert_called_once()
    mock_show.assert_called_once()
