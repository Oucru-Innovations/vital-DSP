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
- Feature extraction capabilities

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.feature_engineering.ecg_ppg_synchronization_features import ECGPPGSynchronization
    >>> signal = np.random.randn(1000)
    >>> processor = EcgPpgSynchronyzationFeatures(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np
from vitalDSP.utils.signal_processing.peak_detection import PeakDetection


class ECGPPGSynchronization:
    """
    A class for analyzing synchronization between ECG and PPG signals to compute combined features
    like Pulse Transit Time (PTT), Pulse Arrival Time (PAT), Electromechanical Delay (EMD),
    Respiratory Sinus Arrhythmia (RSA), and Heart Rate-Pulse Rate synchronization.

    Attributes:
        ecg_signal (np.array): The ECG signal.
        ppg_signal (np.array): The PPG signal.
        ecg_fs (int): The sampling frequency of the ECG signal in Hz.
        ppg_fs (int): The sampling frequency of the PPG signal in Hz.
    """

    def __init__(self, ecg_signal, ppg_signal, ecg_fs, ppg_fs):
        """
        Initializes the ECGPPGSynchronization object.

        Args:
            ecg_signal (np.array): The ECG signal.
            ppg_signal (np.array): The PPG signal.
            ecg_fs (int): The sampling frequency of the ECG signal in Hz.
            ppg_fs (int): The sampling frequency of the PPG signal in Hz.

        Raises:
            ValueError: If the input signals are not valid or if sampling frequencies are not positive.
        """
        if not isinstance(ecg_signal, np.ndarray) or not isinstance(
            ppg_signal, np.ndarray
        ):
            raise TypeError("Both ECG and PPG signals must be numpy arrays.")
        if len(ecg_signal) == 0 or len(ppg_signal) == 0:
            raise ValueError("Both ECG and PPG signals must have non-zero length.")
        if (
            np.isnan(ecg_signal).any()
            or np.isnan(ppg_signal).any()
            or np.isinf(ecg_signal).any()
            or np.isinf(ppg_signal).any()
        ):
            raise ValueError("ECG or PPG signal contains invalid values.")
        if ecg_fs <= 0 or ppg_fs <= 0:
            raise ValueError("Sampling frequencies must be positive integers.")

        self.ecg_signal = ecg_signal
        self.ppg_signal = ppg_signal
        self.ecg_fs = ecg_fs
        self.ppg_fs = ppg_fs

    def detect_r_peaks(self):
        """
        Detects R-peaks in the ECG signal using a simple threshold-based method.

        Returns:
            np.array: Indices of detected R-peaks in the ECG signal.

        Raises:
            ValueError: If no R-peaks are detected in the signal.

        Example:
            >>> ecg_signal = np.sin(np.linspace(0, 10 * np.pi, 1000))  # Simulated ECG signal
            >>> sync_analyzer = ECGPPGSynchronization(ecg_signal, np.array([]), 1000, 100)
            >>> r_peaks = sync_analyzer.detect_r_peaks()
            >>> print(f"R-peaks: {r_peaks}")
        """
        try:
            if len(self.ecg_signal) < self.ecg_fs * 2:
                raise ValueError("ECG signal is too short to detect peaks.")
            detector = PeakDetection(self.ecg_signal, method="ecg_r_peak")
            r_peaks = detector.detect_peaks()
            if len(r_peaks) == 0:
                raise ValueError("No R-peaks detected in the ECG signal.")
            return r_peaks
        except ValueError as e:
            raise ValueError(str(e))  # Return a ValueError
        except Exception as e:
            raise RuntimeError(f"Error detecting R-peaks: {str(e)}")

    def detect_systolic_peaks(self):
        """
        Detects systolic peaks in the PPG signal.

        Returns:
            np.array: Indices of detected systolic peaks in the PPG signal.

        Raises:
            ValueError: If no systolic peaks are detected in the signal.

        Example:
            >>> ppg_signal = np.sin(np.linspace(0, 10 * np.pi, 1000))  # Simulated PPG signal
            >>> sync_analyzer = ECGPPGSynchronization(np.array([]), ppg_signal, 1000, 100)
            >>> systolic_peaks = sync_analyzer.detect_systolic_peaks()
            >>> print(f"Systolic peaks: {systolic_peaks}")
        """
        try:
            if len(self.ppg_signal) < self.ppg_fs * 2:
                raise ValueError("PPG signal is too short to detect peaks.")
            detector = PeakDetection(self.ppg_signal, method="ppg_first_derivative")
            systolic_peaks = detector.detect_peaks()
            if len(systolic_peaks) == 0:
                raise ValueError("No systolic peaks detected in the PPG signal.")
            return systolic_peaks
        except ValueError as e:
            raise ValueError(str(e))  # Return a ValueError
        except Exception as e:
            raise RuntimeError(f"Error detecting systolic peaks: {str(e)}")

    def _detect_ppg_feet(self, systolic_peaks=None):
        """
        Detects PPG pulse feet (onset points) as the minimum value before each
        systolic peak.

        Returns:
            np.array: Indices of PPG foot points.
        """
        if systolic_peaks is None:
            systolic_peaks = self.detect_systolic_peaks()
        feet = []
        for i, peak in enumerate(systolic_peaks):
            if i == 0:
                search_start = max(0, peak - int(0.5 * self.ppg_fs))
            else:
                search_start = systolic_peaks[i - 1]
            if search_start >= peak:
                continue
            segment = self.ppg_signal[search_start:peak]
            foot_idx = search_start + np.argmin(segment)
            feet.append(foot_idx)
        return np.array(feet)

    def compute_ptt(self, r_peaks=None, systolic_peaks=None):
        """
        Computes Pulse Transit Time (PTT) between ECG R-peaks and PPG foot
        (pulse onset). PTT approximates arterial pulse wave transit time.

        PTT = PPG_foot_time - ECG_R_peak_time

        Args:
            r_peaks (np.array): Indices of R-peaks in the ECG signal.
            systolic_peaks (np.array): Indices of systolic peaks in the PPG signal.

        Returns:
            float: The mean PTT value in milliseconds.
        """
        try:
            if r_peaks is None:
                r_peaks = self.detect_r_peaks()
            if systolic_peaks is None:
                systolic_peaks = self.detect_systolic_peaks()
            ppg_feet = self._detect_ppg_feet(systolic_peaks)
            if len(ppg_feet) == 0:
                raise ValueError("No PPG foot points detected.")
            ptt_values = []
            for r_peak in r_peaks:
                r_peak_time = r_peak / self.ecg_fs
                valid_feet = ppg_feet[ppg_feet / self.ppg_fs > r_peak_time]
                if len(valid_feet) > 0:
                    nearest_foot_time = valid_feet[0] / self.ppg_fs
                    ptt_ms = (nearest_foot_time - r_peak_time) * 1000
                    if ptt_ms > 0:
                        ptt_values.append(ptt_ms)
            if len(ptt_values) == 0:
                raise ValueError("No valid PTT values could be calculated.")
            return np.mean(ptt_values)
        except Exception as e:
            raise RuntimeError(f"Error computing Pulse Transit Time: {str(e)}")

    def compute_pat(self, r_peaks=None, systolic_peaks=None):
        """
        Computes Pulse Arrival Time (PAT) between ECG R-peaks and PPG
        systolic peaks. PAT = PEP + PTT, measured from R-peak to
        the PPG systolic peak.

        Args:
            r_peaks (np.array): Indices of R-peaks in the ECG signal.
            systolic_peaks (np.array): Indices of systolic peaks in the PPG signal.

        Returns:
            float: The mean PAT value in milliseconds.
        """
        try:
            if r_peaks is None:
                r_peaks = self.detect_r_peaks()
            if systolic_peaks is None:
                systolic_peaks = self.detect_systolic_peaks()
            pat_values = []
            for r_peak in r_peaks:
                r_peak_time = r_peak / self.ecg_fs
                valid_peaks_time = systolic_peaks[
                    systolic_peaks / self.ppg_fs > r_peak_time
                ]
                if len(valid_peaks_time) > 0:
                    nearest_peak_time = valid_peaks_time[0] / self.ppg_fs
                    pat_ms = (nearest_peak_time - r_peak_time) * 1000
                    if pat_ms > 0:
                        pat_values.append(pat_ms)
            if len(pat_values) == 0:
                raise ValueError("No valid PAT values could be calculated.")
            return np.mean(pat_values)
        except Exception as e:
            raise RuntimeError(f"Error computing Pulse Arrival Time: {str(e)}")

    def compute_ppg_rise_time(self, r_peaks=None, systolic_peaks=None):
        """
        Compute PPG systolic rise time (time from PPG foot to systolic peak).

        Note
        ----
        This quantity is sometimes referred to as "Electromechanical Delay (EMD)"
        in the literature, but that terminology is physiologically inaccurate.
        PAT - PTT equals the systolic rise time of the PPG pulse, not the true
        EMD (which requires Q-onset or pre-ejection period estimation).

        Args:
            r_peaks (np.array): Indices of R-peaks in the ECG signal.
            systolic_peaks (np.array): Indices of systolic peaks in the PPG signal.

        Returns:
            float: The mean PPG systolic rise time in milliseconds.
        """
        try:
            if r_peaks is None:
                r_peaks = self.detect_r_peaks()
            if systolic_peaks is None:
                systolic_peaks = self.detect_systolic_peaks()
            ppg_feet = self._detect_ppg_feet(systolic_peaks)
            if len(ppg_feet) == 0:
                raise ValueError("No PPG foot points detected.")

            ptt_ms = self.compute_ptt(r_peaks, systolic_peaks)
            pat_ms = self.compute_pat(r_peaks, systolic_peaks)

            rise_time_ms = pat_ms - ptt_ms
            if rise_time_ms < 0:
                rise_time_ms = 0.0
            return rise_time_ms
        except Exception as e:
            raise RuntimeError(f"Error computing PPG systolic rise time: {str(e)}")

    def compute_emd(self, *args, **kwargs):
        """Deprecated: Use compute_ppg_rise_time() instead.

        This method computed PAT - PTT, which is the PPG systolic rise time,
        not true Electromechanical Delay (EMD). Renamed for accuracy.
        """
        import warnings
        warnings.warn(
            "compute_emd() is deprecated and will be removed in a future version. "
            "Use compute_ppg_rise_time() instead. Note: this computes PPG rise time "
            "(PAT - PTT), not true Electromechanical Delay.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.compute_ppg_rise_time(*args, **kwargs)

    def compute_hr_pr_sync(self, r_peaks=None, systolic_peaks=None):
        """
        Computes the synchronization between heart rate (HR) from ECG and
        pulse rate (PR) from PPG as a ratio of mean rates.

        Args:
            r_peaks (np.array): Indices of R-peaks in the ECG signal.
            systolic_peaks (np.array): Indices of systolic peaks in the PPG signal.

        Returns:
            float: The HR-PR synchronization ratio (close to 1.0 if well-synchronized).
        """
        try:
            if r_peaks is None:
                r_peaks = self.detect_r_peaks()
            if systolic_peaks is None:
                systolic_peaks = self.detect_systolic_peaks()
            if len(r_peaks) < 2 or len(systolic_peaks) < 2:
                raise ValueError(
                    "Not enough peaks to calculate HR or PR synchronization."
                )

            rr_intervals_sec = np.diff(r_peaks) / self.ecg_fs
            hr_bpm = 60.0 / rr_intervals_sec
            if len(hr_bpm) == 0:
                raise ValueError("No valid HR intervals could be calculated.")

            pp_intervals_sec = np.diff(systolic_peaks) / self.ppg_fs
            pr_bpm = 60.0 / pp_intervals_sec
            if len(pr_bpm) == 0:
                raise ValueError("No valid PR intervals could be calculated.")

            hr_pr_sync_ratio = np.mean(hr_bpm) / np.mean(pr_bpm)
            return hr_pr_sync_ratio
        except Exception as e:
            raise RuntimeError(f"Error computing HR-PR synchronization: {str(e)}")

    def compute_rsa(self, r_peaks=None, systolic_peaks=None):
        """
        Computes Respiratory Sinus Arrhythmia (RSA) by bandpass filtering
        RR intervals in the respiratory frequency band (0.15-0.4 Hz).

        RSA reflects the variation in heart rate that occurs with breathing,
        measured as the peak-to-trough amplitude of the respiratory-band HRV.

        Args:
            r_peaks (np.array): Indices of R-peaks in the ECG signal.
            systolic_peaks (np.array): Indices of systolic peaks in the PPG signal
                (used if r_peaks not available).

        Returns:
            float: RSA amplitude (standard deviation of respiratory-band HRV) in seconds.
        """
        try:
            if r_peaks is None:
                r_peaks = self.detect_r_peaks()

            if len(r_peaks) < 4:
                raise ValueError("Not enough R-peaks to compute RSA.")

            rr_intervals = np.diff(r_peaks) / self.ecg_fs

            rr_mean = np.mean(rr_intervals)
            rr_centered = rr_intervals - rr_mean

            n = len(rr_centered)
            fft_vals = np.fft.rfft(rr_centered)
            freqs = np.fft.rfftfreq(n, d=rr_mean)

            # Bandpass: keep only the respiratory frequency band (0.15 - 0.4 Hz)
            resp_mask = (freqs >= 0.15) & (freqs <= 0.4)
            fft_filtered = np.zeros_like(fft_vals)
            fft_filtered[resp_mask] = fft_vals[resp_mask]

            rr_respiratory = np.fft.irfft(fft_filtered, n=n)

            rsa_amplitude = np.std(rr_respiratory)
            return float(rsa_amplitude)
        except Exception as e:
            raise RuntimeError(
                f"Error computing Respiratory Sinus Arrhythmia: {str(e)}"
            )
