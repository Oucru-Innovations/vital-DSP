"""
Feature Engineering Module for Physiological Signal Processing

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
- Feature extraction capabilities

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.feature_engineering.ppg_light_features import PpgLightFeatures
    >>> signal = np.random.randn(1000)
    >>> processor = PpgLightFeatures(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np
from scipy.signal import find_peaks


class PPGLightFeatureExtractor:
    """
    A class to extract physiological features from PPG signals based on raw data
    from infrared (IR) and red light sources. This includes SpO2, Perfusion Index (PI),
    Respiratory Rate (RR), and Photoplethysmogram Ratio (PPR).

    Parameters:
    -----------
    ir_signal : np.array
        The infrared light PPG signal.
    red_signal : np.array
        The red light PPG signal (optional for features like PI and RR).
    sampling_freq : int
        The sampling frequency of the signals in Hz.

    Example usage:
    --------------
    ppg_extractor = PPGLightFeatureExtractor(ir_signal, red_signal, sampling_freq)
    spo2, times_spo2 = ppg_extractor.calculate_spo2()
    pi, times_pi = ppg_extractor.calculate_perfusion_index()
    rr, times_rr = ppg_extractor.calculate_respiratory_rate()
    ppr, times_ppr = ppg_extractor.calculate_ppr()
    """

    def __init__(self, ir_signal, red_signal=None, sampling_freq=100):
        self.ir_signal = ir_signal
        self.red_signal = red_signal
        self.sampling_freq = sampling_freq

    def calculate_spo2(self, window_seconds=1):
        """
        Calculate SpO2 based on infrared (IR) and red light PPG signals.

        Parameters:
        -----------
        window_seconds : int, optional
            The window length in seconds to calculate SpO2 (default is 1 second).

        Returns:
        --------
        spo2_values : np.array
            Calculated SpO2 values for each window of the signal.
        timestamps : np.array
            Time (in seconds) for each SpO2 value.
        """
        if self.red_signal is None:
            raise ValueError("Red signal is required to compute SpO2.")

        window_size = int(window_seconds * self.sampling_freq)
        spo2_values = []
        timestamps = []

        for start in range(0, len(self.ir_signal), window_size):
            end = start + window_size
            if end > len(self.ir_signal):
                break
            ir_segment = self.ir_signal[start:end]
            red_segment = self.red_signal[start:end]

            # AC and DC components
            ir_ac = np.max(ir_segment) - np.min(ir_segment)
            red_ac = np.max(red_segment) - np.min(red_segment)
            ir_dc = np.mean(ir_segment)
            red_dc = np.mean(red_segment)

            # SpO2 calculation
            ratio = (red_ac / red_dc) / (ir_ac / ir_dc)
            spo2 = np.clip(110 - 25 * ratio, 0, 100)  # Clip between 0 and 100
            # spo2 = 110 - 25 * ratio  # Empirical constants (A=110, B=25)
            spo2_values.append(spo2)
            timestamps.append(start / self.sampling_freq)
        return np.array(spo2_values), np.array(timestamps)

    def calculate_perfusion_index(self, window_seconds=1):
        """
        Calculate the Perfusion Index (PI) from the infrared (IR) PPG signal.

        Parameters:
        -----------
        window_seconds : int, optional
            The window length in seconds to calculate PI (default is 1 second).

        Returns:
        --------
        pi_values : np.array
            Calculated perfusion index values for each window.
        timestamps : np.array
            Time (in seconds) for each PI value.
        """
        window_size = int(window_seconds * self.sampling_freq)
        pi_values = []
        timestamps = []

        for start in range(0, len(self.ir_signal), window_size):
            end = start + window_size
            if end > len(self.ir_signal):
                break
            segment = self.ir_signal[start:end]

            # AC and DC components
            ac_component = np.max(segment) - np.min(segment)
            dc_component = np.mean(segment)

            if dc_component > 0:
                pi = ac_component / dc_component
            else:
                pi = 0  # Avoid division by negative or zero

            pi_values.append(pi)
            timestamps.append(start / self.sampling_freq)
        return np.array(pi_values), np.array(timestamps)

    def calculate_respiratory_rate(self, window_seconds=60):
        """
        Calculate the Respiratory Rate (RR) from a PPG signal by detecting low-frequency oscillations.

        Parameters:
        -----------
        window_seconds : int, optional
            The window length in seconds to calculate RR (default is 60 seconds).

        Returns:
        --------
        rr_values : np.array
            Calculated respiratory rate (in breaths per minute) for each window.
        timestamps : np.array
            Time (in seconds) for each RR value.
        """
        window_size = int(window_seconds * self.sampling_freq)
        rr_values = []
        timestamps = []

        for start in range(0, len(self.ir_signal), window_size):
            end = start + window_size
            if end > len(self.ir_signal):
                break
            segment = self.ir_signal[start:end]

            # Extract low-frequency component (respiratory component)
            low_freq_signal = segment - np.mean(segment)
            peaks, _ = find_peaks(
                low_freq_signal, distance=self.sampling_freq // 2
            )  # Detect respiratory peaks

            if len(peaks) > 0:
                breaths_per_minute = len(peaks) * (60 / window_seconds)
            else:
                breaths_per_minute = 0  # Assign 0 if no peaks are detected
            rr_values.append(breaths_per_minute)
            timestamps.append(start / self.sampling_freq)
        # Ensure that rr_values is always non-empty, even if no peaks are found
        if len(rr_values) == 0:
            rr_values.append(0)
            timestamps.append(0)
        return np.array(rr_values), np.array(timestamps)

    def calculate_ppr(self, window_seconds=1):
        """
        Calculate the Photoplethysmogram Ratio (PPR) between infrared (IR) and red light PPG signals.

        Parameters:
        -----------
        window_seconds : int, optional
            The window length in seconds to calculate PPR (default is 1 second).

        Returns:
        --------
        ppr_values : np.array
            Calculated PPR values for each window.
        timestamps : np.array
            Time (in seconds) for each PPR value.
        """
        if self.red_signal is None:
            raise ValueError("Red signal is required to compute PPR.")

        window_size = int(window_seconds * self.sampling_freq)
        ppr_values = []
        timestamps = []

        for start in range(0, len(self.ir_signal), window_size):
            end = start + window_size
            if end > len(self.ir_signal):
                break
            ir_segment = self.ir_signal[start:end]
            red_segment = self.red_signal[start:end]

            ir_ac = np.max(ir_segment) - np.min(ir_segment)
            red_ac = np.max(red_segment) - np.min(red_segment)
            ir_dc = np.mean(ir_segment)
            red_dc = np.mean(red_segment)

            if ir_dc > 0 and red_dc > 0:
                ppr = np.clip(
                    (red_ac / red_dc) / (ir_ac / ir_dc), 0, None
                )  # Clip to non-negative values
            else:
                ppr = 0  # Set to zero if DC components are invalid
            ppr_values.append(ppr)
            timestamps.append(start / self.sampling_freq)

        return np.array(ppr_values), np.array(timestamps)
