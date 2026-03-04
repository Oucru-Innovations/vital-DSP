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
    >>> from vitalDSP.feature_engineering.ppg_autonomic_features import PpgAutonomicFeatures
    >>> signal = np.random.randn(1000)
    >>> processor = PpgAutonomicFeatures(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np

# from scipy.signal import find_peaks
from vitalDSP.utils.signal_processing.peak_detection import PeakDetection


class PPGAutonomicFeatures:
    """
    A class to compute respiratory and autonomic features from PPG signals.

    Features included:
    - Respiratory Rate Variability (RRV)
    - Respiratory Sinus Arrhythmia (RSA)
    - Autonomic Nervous System Balance (Fractal Dimension, DFA)

    Example usage:
    ```
    import numpy as np
    from PPGRespiratoryAutonomicFeatures import PPGRespiratoryAutonomicFeatures

    # Simulated PPG signal data
    ppg_signal = np.random.rand(1000)  # Replace with actual PPG signal
    fs = 100  # Sampling frequency in Hz

    features = PPGRespiratoryAutonomicFeatures(ppg_signal, fs)

    rrv = features.compute_rrv()
    rsa = features.compute_rsa()
    fractal = features.compute_fractal_dimension()
    dfa_value = features.compute_dfa()

    print(f"RRV: {rrv}, RSA: {rsa}, Fractal Dimension: {fractal}, DFA: {dfa_value}")
    ```
    """

    def __init__(self, ppg_signal, sampling_frequency):
        """
        Initializes the class with PPG signal and sampling frequency.

        Args:
            ppg_signal (np.array): Array of PPG signal values.
            sampling_frequency (int): Sampling frequency in Hz.
        """
        if not isinstance(ppg_signal, np.ndarray):
            raise TypeError("Input signal must be a numpy array")
        if len(ppg_signal) < 2:
            raise ValueError("PPG signal is too short to compute features")
        if np.isnan(ppg_signal).any() or np.isinf(ppg_signal).any():
            raise ValueError("PPG signal contains invalid values")

        self.ppg_signal = ppg_signal
        self.fs = sampling_frequency

    def compute_rrv(self):
        """
        Computes Respiratory Rate Variability (RRV) from the PPG signal.

        Returns:
            float: Respiratory rate variability value.
        """
        # peaks, _ = find_peaks(self.ppg_signal, distance=self.fs/2)
        peak_detector = PeakDetection(self.ppg_signal, method="ppg_first_derivative")
        peaks = peak_detector.detect_peaks()  # Indices of peaks in the PPG signal
        if len(peaks) < 2:
            raise ValueError("No peaks detected in PPG signal")

        rr_intervals = np.diff(peaks) / self.fs
        rrv = np.std(rr_intervals)
        return rrv

    def compute_rsa(self):
        """
        Computes Respiratory Sinus Arrhythmia (RSA) from the PPG signal
        by bandpass filtering pulse-to-pulse intervals in the respiratory
        frequency band (0.15-0.4 Hz).

        Returns:
            float: RSA amplitude (std of respiratory-band pulse interval variability)
                   in seconds.
        """
        peak_detector = PeakDetection(self.ppg_signal, method="ppg_first_derivative")
        peaks = peak_detector.detect_peaks()
        if len(peaks) < 4:
            raise ValueError("Not enough peaks detected in PPG signal for RSA")

        intervals = np.diff(peaks) / self.fs
        mean_interval = np.mean(intervals)
        centered = intervals - mean_interval

        n = len(centered)
        fft_vals = np.fft.rfft(centered)
        freqs = np.fft.rfftfreq(n, d=mean_interval)

        resp_mask = (freqs >= 0.15) & (freqs <= 0.4)
        fft_filtered = np.zeros_like(fft_vals)
        fft_filtered[resp_mask] = fft_vals[resp_mask]

        respiratory_component = np.fft.irfft(fft_filtered, n=n)
        return float(np.std(respiratory_component))

    def compute_fractal_dimension(self, k_max=10):
        """
        Computes the fractal dimension of the PPG signal using the Higuchi method.

        Args:
            k_max (int): The maximum number of intervals to calculate (default is 10).

        Returns:
            float: Fractal dimension of the signal.
        """
        N = len(self.ppg_signal)
        if N < 10:
            raise ValueError("PPG signal is too short to compute fractal dimension")

        Lk = np.zeros(k_max)
        for k in range(1, k_max + 1):
            Lmk = []
            for m in range(k):
                Lm = (
                    np.sum(np.abs(np.diff(self.ppg_signal[m:N:k])))
                    * (N - 1)
                    / (int((N - m) / k) * k)
                )
                Lmk.append(Lm)
            Lk[k - 1] = np.mean(Lmk)

        # Handle cases where log(Lk) might produce negative values or zero
        if np.any(Lk <= 0):
            raise ValueError(
                "Logarithmic values for fractal dimension cannot be computed due to non-positive values in Lk"
            )

        fractal_dim = np.polyfit(np.log(np.arange(1, k_max + 1)), np.log(Lk), 1)[0]
        return (
            float(fractal_dim) if fractal_dim > 0 else 0.001
        )  # Ensure valid float value

    def compute_dfa(self, min_scale=4, max_scale=None, num_scales=20):
        """
        Computes the Detrended Fluctuation Analysis (DFA) of the PPG signal
        using proper multi-scale analysis.

        DFA measures the fractal scaling properties by computing the fluctuation
        function F(n) at multiple window sizes n, then fitting log(F) vs log(n).

        Args:
            min_scale (int): Minimum window size (default 4).
            max_scale (int): Maximum window size (default N//4).
            num_scales (int): Number of scales to evaluate (default 20).

        Returns:
            float: DFA scaling exponent (alpha).
        """
        N = len(self.ppg_signal)
        if N < 16:
            raise ValueError("PPG signal is too short to compute DFA")

        if max_scale is None:
            max_scale = N // 4

        integrated = np.cumsum(self.ppg_signal - np.mean(self.ppg_signal))

        scales = np.unique(
            np.floor(np.logspace(np.log10(min_scale), np.log10(max_scale), num=num_scales))
        ).astype(int)

        fluctuations = []
        valid_scales = []

        for scale in scales:
            n_segments = N // scale
            if n_segments < 2:
                continue

            truncated = integrated[: n_segments * scale]
            segments = truncated.reshape((n_segments, scale))
            x = np.arange(scale)

            rms_values = np.zeros(n_segments)
            for j in range(n_segments):
                coeffs = np.polyfit(x, segments[j], 1)
                trend = np.polyval(coeffs, x)
                rms_values[j] = np.sqrt(np.mean((segments[j] - trend) ** 2))

            mean_rms = np.mean(rms_values)
            if mean_rms > 0:
                fluctuations.append(mean_rms)
                valid_scales.append(scale)

        if len(valid_scales) < 2:
            raise ValueError("Not enough valid scales to compute DFA")

        log_scales = np.log(np.array(valid_scales, dtype=float))
        log_fluct = np.log(np.array(fluctuations, dtype=float))
        dfa_alpha = np.polyfit(log_scales, log_fluct, 1)[0]
        return float(dfa_alpha) if dfa_alpha > 0 else 0.001


# import pandas as pd
# import os
# if __name__ == "__main__":
#     ppg_signal = np.random.rand(1000)
#     fname = '20190109T151032.026+0700_1050000_1080000.csv'
#     PATH = 'D:\Workspace\Data\\24EIa\output\sample'

#     ppg_signal = pd.read_csv(os.path.join(PATH, fname))['PLETH'].values
#     fs = 100
#     features = PPGAutonomicFeatures(ppg_signal, fs)
#     rrv = features.compute_rrv()
#     rsa = features.compute_rsa()
#     fractal = features.compute_fractal_dimension()
#     dfa_value = features.compute_dfa()
#     print(f"RRV: {rrv}, RSA: {rsa}, Fractal Dimension: {fractal}, DFA: {dfa_value}")
