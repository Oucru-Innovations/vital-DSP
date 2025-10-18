"""
Physiological Features Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations
- SciPy integration for advanced signal processing
- Comprehensive signal analysis

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.physiological_features.energy_analysis import EnergyAnalysis
    >>> signal = np.random.randn(1000)
    >>> processor = EnergyAnalysis(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""


import numpy as np
from scipy.signal import welch


class EnergyAnalysis:
    """
    A class for computing energy-related features from physiological signals (ECG, PPG, EEG).

    Attributes:
        signal (np.array): The physiological signal (ECG, PPG, EEG).
        fs (int): The sampling frequency of the signal in Hz. Default is 1000 Hz.
    """

    def __init__(self, signal, fs=1000):
        """
        Initializes the EnergyAnalysis object.

        Args:
            signal (np.array): The physiological signal.
            fs (int): The sampling frequency of the signal in Hz. Default is 1000 Hz.
        """
        self.signal = np.array(signal)
        self.fs = fs  # Sampling frequency

    def compute_total_energy(self):
        """
        Computes the total energy of the signal using the sum of squared signal values.

        Returns:
            float: The total energy of the signal.

        Example:
            >>> ecg_signal = [...]  # Sample ECG signal
            >>> ea = EnergyAnalysis(ecg_signal)
            >>> total_energy = ea.compute_total_energy()
            >>> print(f"Total Energy: {total_energy}")
        """
        return np.sum(self.signal**2)

    def compute_segment_energy(self, start_idx, end_idx):
        """
        Computes the energy of a specific segment of the signal.

        Args:
            start_idx (int): The starting index of the segment.
            end_idx (int): The ending index of the segment.

        Returns:
            float: The energy of the segment.

        Example:
            >>> ppg_signal = [...]  # Sample PPG signal
            >>> ea = EnergyAnalysis(ppg_signal)
            >>> segment_energy = ea.compute_segment_energy(100, 200)
            >>> print(f"Segment Energy: {segment_energy}")
        """
        segment = self.signal[start_idx:end_idx]
        return np.sum(segment**2)

    def compute_spectral_energy(self):
        """
        Computes the total spectral energy of the signal using Welch's method for power spectral density (PSD).

        Returns:
            float: The total spectral energy of the signal.

        Example:
            >>> ecg_signal = [...]  # Sample ECG signal
            >>> ea = EnergyAnalysis(ecg_signal)
            >>> spectral_energy = ea.compute_spectral_energy()
            >>> print(f"Spectral Energy: {spectral_energy}")
        """
        f, psd = welch(self.signal, fs=self.fs, nperseg=len(self.signal) // 4)
        spectral_energy = np.sum(psd)
        return spectral_energy

    def compute_band_energy(self, low_freq, high_freq):
        """
        Computes the energy in a specific frequency band for EEG signals.

        Args:
            low_freq (float): The lower bound of the frequency band (in Hz).
            high_freq (float): The upper bound of the frequency band (in Hz).

        Returns:
            float: The energy in the specified frequency band.

        Example:
            >>> eeg_signal = [...]  # Sample EEG signal
            >>> ea = EnergyAnalysis(eeg_signal)
            >>> band_energy = ea.compute_band_energy(8, 12)  # Alpha band (8-12 Hz)
            >>> print(f"Band Energy (8-12 Hz): {band_energy}")
        """
        f, psd = welch(self.signal, fs=self.fs, nperseg=len(self.signal) // 4)
        band_psd = psd[(f >= low_freq) & (f <= high_freq)]
        band_energy = np.sum(band_psd)
        return band_energy

    def compute_qrs_energy(self, r_peaks):
        """
        Computes the energy of the QRS complex in an ECG signal based on detected R-peaks.

        Args:
            r_peaks (np.array): Indices of R-peaks detected in the ECG signal.

        Returns:
            float: The total energy of the QRS complex.

        Example:
            >>> r_peaks = [50, 150, 250]  # Detected R-peaks
            >>> qrs_energy = ea.compute_qrs_energy(r_peaks)
            >>> print(f"QRS Energy: {qrs_energy}")
        """
        qrs_energy = 0.0
        for r_peak in r_peaks:
            qrs_segment = self.signal[
                max(0, r_peak - int(self.fs * 0.02)) : min(
                    len(self.signal), r_peak + int(self.fs * 0.02)
                )
            ]  # 20ms before and after R peak
            qrs_energy += np.sum(qrs_segment**2)
        return qrs_energy

    def compute_systolic_diastolic_energy(self, systolic_peaks, diastolic_notches):
        """
        Computes the energy of systolic and diastolic phases in a PPG signal.

        Args:
            systolic_peaks (np.array): Indices of systolic peaks.
            diastolic_notches (np.array): Indices of diastolic notches.

        Returns:
            tuple: Systolic and diastolic energy values.

        Example:
            >>> systolic_peaks = [100, 300, 500]
            >>> diastolic_notches = [150, 350, 550]
            >>> systolic_energy, diastolic_energy = ea.compute_systolic_diastolic_energy(systolic_peaks, diastolic_notches)
            >>> print(f"Systolic Energy: {systolic_energy}, Diastolic Energy: {diastolic_energy}")
        """
        systolic_energy = 0.0
        diastolic_energy = 0.0

        for systolic, diastolic in zip(systolic_peaks, diastolic_notches):
            systolic_segment = self.signal[
                max(0, systolic - int(self.fs * 0.05)) : min(
                    len(self.signal), systolic + int(self.fs * 0.05)
                )
            ]  # 50ms around systolic
            diastolic_segment = self.signal[
                max(0, diastolic - int(self.fs * 0.05)) : min(
                    len(self.signal), diastolic + int(self.fs * 0.05)
                )
            ]  # 50ms around diastolic
            systolic_energy += np.sum(systolic_segment**2)
            diastolic_energy += np.sum(diastolic_segment**2)

        return systolic_energy, diastolic_energy
