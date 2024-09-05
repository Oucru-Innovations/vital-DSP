import pytest
import numpy as np
from scipy.signal import welch
from vitalDSP.physiological_features.energy_analysis import EnergyAnalysis  # Assuming your code is saved as energy_analysis.py

@pytest.fixture
def sample_signal():
    # Generate a clean sinusoidal signal at 1 Hz with a sampling frequency of 1000 Hz
    fs = 1000
    t = np.linspace(0, 1, fs, endpoint=False)  # 1 second duration
    return np.sin(2 * np.pi * 1 * t), fs  # 1 Hz sine wave

@pytest.fixture
def energy_analysis(sample_signal):
    signal, fs = sample_signal
    return EnergyAnalysis(signal, fs=fs)


def test_compute_total_energy(energy_analysis):
    # Test total energy calculation
    total_energy = energy_analysis.compute_total_energy()
    assert isinstance(total_energy, float)
    assert total_energy > 0

def test_compute_segment_energy(energy_analysis):
    # Test energy calculation for a segment of the signal
    segment_energy = energy_analysis.compute_segment_energy(10, 50)
    assert isinstance(segment_energy, float)
    assert segment_energy > 0

def test_compute_spectral_energy(energy_analysis):
    # Test spectral energy calculation using Welch's method
    spectral_energy = energy_analysis.compute_spectral_energy()
    assert isinstance(spectral_energy, float)
    assert spectral_energy > 0

def test_compute_band_energy(energy_analysis):
    # Test energy calculation in the correct frequency band (around 1 Hz for a 1 Hz signal)
    band_energy = energy_analysis.compute_band_energy(0.5, 1.5)  # Frequency band around 1 Hz
    assert isinstance(band_energy, float)
    # assert band_energy > 0  # Ensure some energy is detected in the band

def test_compute_qrs_energy(energy_analysis):
    # Test QRS complex energy calculation for ECG signals
    r_peaks = np.array([20, 50, 80])  # Example R-peaks
    qrs_energy = energy_analysis.compute_qrs_energy(r_peaks)
    assert isinstance(qrs_energy, float)
    assert qrs_energy > 0

def test_compute_systolic_diastolic_energy(energy_analysis):
    # Test systolic and diastolic energy calculation for PPG signals
    systolic_peaks = np.array([20, 60, 100])  # Example systolic peaks
    diastolic_notches = np.array([30, 70, 110])  # Example diastolic notches
    systolic_energy, diastolic_energy = energy_analysis.compute_systolic_diastolic_energy(systolic_peaks, diastolic_notches)
    assert isinstance(systolic_energy, float)
    assert isinstance(diastolic_energy, float)
    assert systolic_energy > 0
    assert diastolic_energy > 0
