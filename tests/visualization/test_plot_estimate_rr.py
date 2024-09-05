import pytest
import numpy as np
from vitalDSP.respiratory_analysis.estimate_rr import (
    fft_based_rr,
    peak_detection_rr,
    time_domain_rr,
    frequency_domain_rr,
)
from vitalDSP.visualization.plot_estimate_rr import plot_rr_estimations  # Replace with actual module name

# Replace the actual signal processing functions with simple stubs for testing
def fft_based_rr_stub(signal, sampling_rate, preprocess=None, **kwargs):
    return 12.5

def peak_detection_rr_stub(signal, sampling_rate, preprocess=None, **kwargs):
    return 13.0

def time_domain_rr_stub(signal, sampling_rate, preprocess=None, **kwargs):
    return 12.8

def frequency_domain_rr_stub(signal, sampling_rate, preprocess=None, **kwargs):
    return 13.2

def find_peaks_stub(signal):
    return np.array([10, 50, 100]),  # Example peak indices

# Now we replace the original methods in the testing environment
@pytest.fixture
def setup_stubs(monkeypatch):
    # Replace real implementations with our stubs
    monkeypatch.setattr("vitalDSP.respiratory_analysis.estimate_rr.fft_based_rr", fft_based_rr_stub)
    monkeypatch.setattr("vitalDSP.respiratory_analysis.estimate_rr.peak_detection_rr", peak_detection_rr_stub)
    monkeypatch.setattr("vitalDSP.respiratory_analysis.estimate_rr.time_domain_rr", time_domain_rr_stub)
    monkeypatch.setattr("vitalDSP.respiratory_analysis.estimate_rr.frequency_domain_rr", frequency_domain_rr_stub)
    monkeypatch.setattr("vitalDSP.utils.common.find_peaks", find_peaks_stub)

def test_plot_rr_estimations(setup_stubs):
    signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01)) + 0.1 * np.random.normal(size=1000)
    sampling_rate = 100

    # Simulate the function without relying on Plotly rendering
    fig = plot_rr_estimations(signal, sampling_rate, preprocess="bandpass", lowcut=0.1, highcut=0.5)

    # Ensure that 6 traces have been added (2 for the signal, 1 for FFT, 1 for peaks, 1 for autocorrelation, 1 for PSD)
    assert len(fig.data) == 6

    # Test that the correct subplot titles were set based on the RR stubs
    # assert fig.layout.annotations[0].text == "FFT-Based RR Estimation: 12.50 BPM"
    # assert fig.layout.annotations[1].text == "Peak Detection RR Estimation: 13.00 BPM"
    # assert fig.layout.annotations[2].text == "Time-Domain RR Estimation: 12.80 BPM"
    # assert fig.layout.annotations[3].text == "Frequency-Domain RR Estimation: 13.20 BPM"

def test_plot_rr_without_preprocessing(setup_stubs):
    signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01)) + 0.1 * np.random.normal(size=1000)
    sampling_rate = 100

    # Simulate the function without preprocessing
    fig = plot_rr_estimations(signal, sampling_rate)

    # Ensure that 6 traces have been added (2 for the signal, 1 for FFT, 1 for peaks, 1 for autocorrelation, 1 for PSD)
    assert len(fig.data) == 6

    # Test that the correct subplot titles were set based on the RR stubs
    # assert fig.layout.annotations[0].text == "FFT-Based RR Estimation: 12.50 BPM"
    # assert fig.layout.annotations[1].text == "Peak Detection RR Estimation: 13.00 BPM"
    # assert fig.layout.annotations[2].text == "Time-Domain RR Estimation: 12.80 BPM"
    # assert fig.layout.annotations[3].text == "Frequency-Domain RR Estimation: 13.20 BPM"

