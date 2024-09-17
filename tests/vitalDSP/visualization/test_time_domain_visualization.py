import pytest
from unittest.mock import patch
import numpy as np
import plotly.graph_objs as go
from vitalDSP.visualization.time_domain_visualization import (
    TrendAnalysisVisualizer,
    SignalSegmentationVisualizer,
    SignalPowerAnalysisVisualizer,
    SignalChangeDetectionVisualizer,
    PeakDetectionVisualizer,
    EnvelopeDetectionVisualizer,
    CrossSignalAnalysisVisualizer,
)


@pytest.fixture
def mock_show():
    with patch("plotly.graph_objs.Figure.show") as mocked_show:
        yield mocked_show


@pytest.fixture
def signal():
    np.random.seed(42)
    return np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)


@pytest.fixture
def trend_analysis(signal):
    return type("TrendAnalysis", (object,), {"signal": signal})()


@pytest.fixture
def trend_visualizer(trend_analysis):
    return TrendAnalysisVisualizer(trend_analysis)


@pytest.fixture
def signal_segmentation():
    return type("SignalSegmentation", (object,), {})()


@pytest.fixture
def segmentation_visualizer(signal_segmentation):
    return SignalSegmentationVisualizer(signal_segmentation)


@pytest.fixture
def signal_power_analysis():
    return type("SignalPowerAnalysis", (object,), {})()


@pytest.fixture
def power_visualizer(signal_power_analysis):
    return SignalPowerAnalysisVisualizer(signal_power_analysis)


@pytest.fixture
def signal_change_detection():
    return type("SignalChangeDetection", (object,), {})()


@pytest.fixture
def change_detection_visualizer(signal_change_detection):
    return SignalChangeDetectionVisualizer(signal_change_detection)


@pytest.fixture
def peak_detection():
    return type("PeakDetection", (object,), {})()


@pytest.fixture
def peak_visualizer(peak_detection):
    return PeakDetectionVisualizer(peak_detection)


@pytest.fixture
def envelope_detection():
    return type("EnvelopeDetection", (object,), {})()


@pytest.fixture
def envelope_visualizer(envelope_detection):
    return EnvelopeDetectionVisualizer(envelope_detection)


@pytest.fixture
def cross_signal_analysis():
    return type("CrossSignalAnalysis", (object,), {})()


@pytest.fixture
def cross_signal_visualizer(cross_signal_analysis):
    return CrossSignalAnalysisVisualizer(cross_signal_analysis)


def test_plot_moving_average(trend_visualizer, signal, mock_show):
    moving_avg = np.convolve(signal, np.ones(5) / 5, mode="valid")
    trend_visualizer.plot_moving_average(moving_avg, window_size=5)
    assert mock_show.called


def test_plot_weighted_moving_average(trend_visualizer, signal, mock_show):
    weighted_avg = np.convolve(signal, np.linspace(0.1, 1, len(signal)), mode="valid")
    trend_visualizer.plot_weighted_moving_average(weighted_avg)
    assert mock_show.called


def test_plot_exponential_smoothing(trend_visualizer, signal, mock_show):
    exp_smoothed = np.convolve(signal, np.ones(5) / 5, mode="valid")  # Dummy data
    trend_visualizer.plot_exponential_smoothing(exp_smoothed, alpha=0.5)
    assert mock_show.called


def test_plot_linear_trend(trend_visualizer, signal, mock_show):
    linear_trend = np.linspace(0, 1, len(signal))
    trend_visualizer.plot_linear_trend(linear_trend)
    assert mock_show.called


def test_plot_polynomial_trend(trend_visualizer, signal, mock_show):
    polynomial_trend = np.polyval(
        [0.1, -0.5, 0.2], np.arange(len(signal))
    )  # Dummy data
    trend_visualizer.plot_polynomial_trend(polynomial_trend, degree=2)
    assert mock_show.called


### Tests for SignalSegmentationVisualizer ###
def test_plot_segments(segmentation_visualizer, mock_show):
    segments = [np.random.rand(50), np.random.rand(50)]
    segmentation_visualizer.plot_segments(segments)
    assert mock_show.called


### Tests for SignalPowerAnalysisVisualizer ###
def test_plot_signal_with_power(power_visualizer, signal, mock_show):
    power = signal**2  # Dummy power data
    power_visualizer.plot_signal_with_power(signal, power)
    assert mock_show.called


def test_plot_psd(power_visualizer, mock_show):
    freqs = np.linspace(0, 50, 100)
    psd = np.random.rand(100)
    power_visualizer.plot_psd(freqs, psd)
    assert mock_show.called


### Tests for SignalChangeDetectionVisualizer ###
def test_plot_changes(change_detection_visualizer, signal, mock_show):
    change_points = [10, 50, 90]
    change_detection_visualizer.plot_changes(signal, change_points)
    assert mock_show.called


### Tests for PeakDetectionVisualizer ###
def test_plot_peaks(peak_visualizer, signal, mock_show):
    peaks = [20, 40, 80]
    peak_visualizer.plot_peaks(signal, peaks)
    assert mock_show.called


### Tests for EnvelopeDetectionVisualizer ###
def test_plot_envelope(envelope_visualizer, signal, mock_show):
    envelope = np.abs(signal)  # Dummy envelope data
    envelope_visualizer.plot_envelope(signal, envelope)
    assert mock_show.called


### Tests for CrossSignalAnalysisVisualizer ###
def test_plot_cross_signal(cross_signal_visualizer, signal, mock_show):
    signal2 = signal * 0.8  # Dummy second signal
    cross_signal_visualizer.plot_cross_signal(signal, signal2)
    assert mock_show.called


def test_plot_coherence(cross_signal_visualizer, mock_show):
    freqs = np.linspace(0, 50, 100)
    coherence = np.random.rand(100)
    cross_signal_visualizer.plot_coherence(freqs, coherence)
    assert mock_show.called


def test_plot_cross_correlation(cross_signal_visualizer, mock_show):
    lags = np.arange(-50, 50)
    cross_corr = np.random.rand(100)
    cross_signal_visualizer.plot_cross_correlation(lags, cross_corr)
    assert mock_show.called
