import numpy as np
from vitalDSP.physiological_features.peak_detection import PeakDetection

class WaveformMorphology:
    """
    A class for computing morphological features from physiological waveforms (ECG, PPG, EEG).

    Attributes:
        waveform (np.array): The waveform signal (ECG, PPG, EEG).
        fs (int): The sampling frequency of the signal in Hz. Default is 1000 Hz.
        signal_type (str): The type of signal ('ECG', 'PPG', 'EEG').
    """

    def __init__(self, waveform, fs=256, signal_type="ECG"):
        """
        Initializes the WaveformMorphology object.

        Args:
            waveform (np.array): The waveform signal.
            fs (int): The sampling frequency of the signal in Hz. Default is 1000 Hz.
            signal_type (str): The type of signal (ECG, PPG, EEG).
        """
        self.waveform = np.array(waveform)
        self.fs = fs  # Sampling frequency
        self.signal_type = signal_type  # 'ECG', 'PPG', 'EEG'

    def compute_amplitude(self):
        """
        Computes the amplitude (max-min) of the waveform, representing the peak-to-peak range.

        Returns:
            float: The amplitude of the waveform.

        Example:
            >>> waveform = [0.5, 0.8, 1.2, 0.9, 0.7]
            >>> wm = WaveformMorphology(waveform, signal_type="ECG")
            >>> amplitude = wm.compute_amplitude()
            >>> print(f"Amplitude: {amplitude}")
        """
        return np.max(self.waveform) - np.min(self.waveform)

    def compute_volume(self, peaks):
        """
        Computes the area under the waveform for a given peak, typically used to analyze
        the volume of QRS complexes (ECG) or systolic/diastolic volumes (PPG).

        Args:
            peaks (np.array): Indices of detected peaks.

        Returns:
            float: The computed area (AUC) for the given peaks.

        Example:
            >>> peaks = [1, 5, 9]
            >>> volume = wm.compute_volume(peaks)
            >>> print(f"Volume: {volume}")
        """
        auc = 0.0
        for peak in peaks:
            if peak > 0 and peak < len(self.waveform) - 1:
                auc += np.trapz(
                    self.waveform[peak - 1 : peak + 2]
                )  # AUC for one peak region
        return auc

    def compute_qrs_duration(self):
        """
        Computes the duration of the QRS complex in an ECG waveform.

        Returns:
            float: The QRS duration in milliseconds.

        Example:
            >>> ecg_signal = [...]  # Sample ECG signal
            >>> wm = WaveformMorphology(ecg_signal, signal_type="ECG")
            >>> qrs_duration = wm.compute_qrs_duration()
            >>> print(f"QRS Duration: {qrs_duration} ms")
        """
        if self.signal_type != "ECG":
            raise ValueError("QRS duration can only be computed for ECG signals.")

        # Detect R-peaks in the ECG signal
        peak_detector = PeakDetection(self.waveform, method="ecg_r_peak")
        r_peaks = peak_detector.detect_peaks()

        # Assume that Q-wave starts before the R peak and S-wave ends after the R peak
        qrs_duration = 0.0
        for r_peak in r_peaks:
            q_start = r_peak - int(self.fs * 0.02)  # 20 ms before R peak
            s_end = r_peak + int(self.fs * 0.02)  # 20 ms after R peak
            qrs_duration += (s_end - q_start) * 1000 / self.fs

        return qrs_duration / len(r_peaks) if len(r_peaks) > 0 else 0.0

    def compute_ppg_dicrotic_notch(self):
        """
        Detects the dicrotic notch in a PPG waveform and computes its timing.

        Returns:
            float: The timing of the dicrotic notch in milliseconds.

        Example:
            >>> ppg_signal = [...]  # Sample PPG signal
            >>> wm = WaveformMorphology(ppg_signal, signal_type="PPG")
            >>> notch_timing = wm.compute_ppg_dicrotic_notch()
            >>> print(f"Dicrotic Notch Timing: {notch_timing} ms")
        """
        if self.signal_type != "PPG":
            raise ValueError("Dicrotic notch can only be computed for PPG signals.")

        # Detect peaks and dicrotic notches in the PPG signal
        peak_detector = PeakDetection(self.waveform, method="ppg_first_derivative")
        systolic_peaks = peak_detector.detect_peaks()

        dicrotic_notch_detector = PeakDetection(
            self.waveform, method="ppg_second_derivative"
        )
        dicrotic_notches = dicrotic_notch_detector.detect_peaks()

        # Compute the time difference between systolic peak and dicrotic notch
        notch_timing = 0.0
        for peak, notch in zip(systolic_peaks, dicrotic_notches):
            if notch > peak:
                notch_timing += (
                    (notch - peak) * 1000 / self.fs
                )  # Convert to milliseconds

        return notch_timing / len(systolic_peaks) if len(systolic_peaks) > 0 else 0.0

    def compute_wave_ratio(self, peaks, notch_points):
        """
        Computes the ratio of systolic to diastolic volumes in a PPG waveform.

        Args:
            peaks (np.array): Indices of systolic peaks.
            notch_points (np.array): Indices of dicrotic notches.

        Returns:
            float: The ratio of systolic to diastolic areas.

        Example:
            >>> systolic_peaks = [5, 15, 25]
            >>> dicrotic_notches = [10, 20, 30]
            >>> wave_ratio = wm.compute_wave_ratio(systolic_peaks, dicrotic_notches)
            >>> print(f"Systolic/Diastolic Ratio: {wave_ratio}")
        """
        if self.signal_type != "PPG":
            raise ValueError("Wave ratio can only be computed for PPG signals.")

        systolic_volume = self.compute_volume(peaks)
        diastolic_volume = self.compute_volume(notch_points)

        if diastolic_volume == 0:
            return np.inf  # Avoid division by zero

        return systolic_volume / diastolic_volume

    def compute_eeg_wavelet_features(self):
        """
        Computes EEG wavelet features by applying a wavelet transform and extracting relevant
        frequency bands.

        Returns:
            dict: A dictionary of extracted EEG wavelet features (e.g., delta, theta, alpha).

        Example:
            >>> eeg_signal = [...]  # Sample EEG signal
            >>> wm = WaveformMorphology(eeg_signal, signal_type="EEG")
            >>> wavelet_features = wm.compute_eeg_wavelet_features()
            >>> print(wavelet_features)
        """
        if self.signal_type != "EEG":
            raise ValueError("Wavelet features can only be computed for EEG signals.")

        wavelet_coeffs = np.abs(np.convolve(self.waveform, np.ones(10), "same"))

        # Extract frequency bands based on wavelet decomposition
        delta = wavelet_coeffs[: len(wavelet_coeffs) // 5]
        theta = wavelet_coeffs[len(wavelet_coeffs) // 5 : len(wavelet_coeffs) // 4]
        alpha = wavelet_coeffs[len(wavelet_coeffs) // 4 : len(wavelet_coeffs) // 3]

        return {
            "delta_power": np.sum(delta),
            "theta_power": np.sum(theta),
            "alpha_power": np.sum(alpha),
        }

    def compute_slope(self):
        """
        Computes the slope of the waveform by calculating its first derivative.

        Returns:
            float: The average slope of the waveform.

        Example:
            >>> signal = [0.5, 0.7, 1.0, 0.8, 0.6]
            >>> wm = WaveformMorphology(signal)
            >>> slope = wm.compute_slope()
            >>> print(f"Slope: {slope}")
        """
        slope = np.gradient(self.waveform)
        return np.mean(slope)

    def compute_curvature(self):
        """
        Computes the curvature of the waveform by analyzing its second derivative.

        Returns:
            float: The average curvature of the waveform.

        Example:
            >>> signal = [0.5, 0.7, 1.0, 0.8, 0.6]
            >>> wm = WaveformMorphology(signal)
            >>> curvature = wm.compute_curvature()
            >>> print(f"Curvature: {curvature}")
        """
        first_derivative = np.gradient(self.waveform)
        second_derivative = np.gradient(first_derivative)
        curvature = np.abs(second_derivative) / (1 + first_derivative**2) ** (3 / 2)
        return np.mean(curvature)
