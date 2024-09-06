import numpy as np
from vitalDSP.utils.peak_detection import PeakDetection


class CrossCorrelationFeatures:
    """
    A class for computing cross-correlation features between physiological signals (ECG, PPG, EEG).

    Attributes:
        signal1 (np.array): The first physiological signal (e.g., ECG).
        signal2 (np.array): The second physiological signal (e.g., PPG).
        fs (int): The sampling frequency of the signals in Hz. Default is 1000 Hz.
    """

    def __init__(self, signal1, signal2, fs=1000):
        """
        Initializes the CrossCorrelationFeatures object.

        Args:
            signal1 (np.array): The first physiological signal (e.g., ECG).
            signal2 (np.array): The second physiological signal (e.g., PPG).
            fs (int): The sampling frequency of the signals in Hz. Default is 1000 Hz.
        """
        self.signal1 = np.array(signal1)
        self.signal2 = np.array(signal2)
        self.fs = fs  # Sampling frequency

    def compute_cross_correlation(self, mode="full"):
        """
        Computes the cross-correlation between two physiological signals (e.g., ECG and PPG).

        Args:
            mode (str): Specifies the mode for cross-correlation. Can be 'full', 'valid', or 'same'.
                        Default is 'full'.

        Returns:
            tuple: The cross-correlation values and the corresponding lag.

        Example:
            >>> ecg_signal = [...]  # Sample ECG signal
            >>> ppg_signal = [...]  # Sample PPG signal
            >>> ccf = CrossCorrelationFeatures(ecg_signal, ppg_signal)
            >>> cross_corr, lag = ccf.compute_cross_correlation()
            >>> print(f"Cross-Correlation: {cross_corr}, Lag: {lag}")
        """
        cross_corr = np.correlate(
            self.signal1 - np.mean(self.signal1),
            self.signal2 - np.mean(self.signal2),
            mode=mode,
        )
        lag = int(
            np.argmax(np.abs(cross_corr)) - (len(self.signal1) - 1)
        )  # Ensure lag is an integer
        return cross_corr, lag

    def compute_normalized_cross_correlation(self):
        """
        Computes the normalized cross-correlation between two physiological signals to account for differences
        in amplitude.

        Returns:
            tuple: The normalized cross-correlation values and the corresponding lag.

        Example:
            >>> ecg_signal = [...]  # Sample ECG signal
            >>> ppg_signal = [...]  # Sample PPG signal
            >>> ccf = CrossCorrelationFeatures(ecg_signal, ppg_signal)
            >>> norm_corr, lag = ccf.compute_normalized_cross_correlation()
            >>> print(f"Normalized Cross-Correlation: {norm_corr}, Lag: {lag}")
        """
        cross_corr, lag = self.compute_cross_correlation()
        norm_corr = cross_corr / (
            np.std(self.signal1) * np.std(self.signal2) * len(self.signal1)
        )
        return norm_corr, lag

    def compute_pulse_transit_time(self, r_peaks):
        """
        Computes Pulse Transit Time (PTT) as the time difference between the ECG signal (R-peaks)
        and the corresponding PPG signal (foot of the waveform). PTT is used to estimate blood pressure
        and vascular health.

        Args:
            r_peaks (np.array): Indices of R-peaks detected in the ECG signal.

        Returns:
            float: The average Pulse Transit Time (PTT) in milliseconds.

        Example:
            >>> r_peaks = [50, 150, 250]  # Detected R-peaks
            >>> ptt = ccf.compute_pulse_transit_time(r_peaks)
            >>> print(f"Pulse Transit Time: {ptt} ms")
        """
        # Detect foot of the PPG waveform
        foot_detector = PeakDetection(self.signal2, method="ppg_first_derivative")
        foot_points = foot_detector.detect_peaks()

        ptt_values = []
        for r_peak in r_peaks:
            # Find the closest foot point in the PPG signal after the R-peak
            foot_after_r = [foot for foot in foot_points if foot > r_peak]
            if foot_after_r:
                ptt = (
                    (foot_after_r[0] - r_peak) * 1000 / self.fs
                )  # Convert to milliseconds
                ptt_values.append(ptt)

        return np.mean(ptt_values) if ptt_values else 0.0

    def compute_lag(self, r_peaks):
        """
        Computes the time lag between the ECG signal (R-peaks) and the corresponding PPG signal
        (foot of the waveform) based on cross-correlation.

        Args:
            r_peaks (np.array): Indices of R-peaks detected in the ECG signal.

        Returns:
            float: The average time lag between the signals in milliseconds.

        Example:
            >>> r_peaks = [50, 150, 250]  # Detected R-peaks
            >>> lag = ccf.compute_lag(r_peaks)
            >>> print(f"Average Lag: {lag} ms")
        """
        foot_detector = PeakDetection(self.signal2, method="ppg_first_derivative")
        foot_points = foot_detector.detect_peaks()

        lag_values = []
        for r_peak in r_peaks:
            foot_after_r = [foot for foot in foot_points if foot > r_peak]
            if foot_after_r:
                lag = (
                    (foot_after_r[0] - r_peak) * 1000 / self.fs
                )  # Convert to milliseconds
                lag_values.append(lag)

        return np.mean(lag_values) if lag_values else 0.0
