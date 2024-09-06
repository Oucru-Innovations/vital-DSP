import numpy as np
from vitalDSP.utils.peak_detection import PeakDetection
from scipy.stats import skew


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

    def detect_troughs(self, peaks):
        """
        Detects the troughs (valleys) in the PPG waveform between systolic peaks.

        Parameters
        ----------
        peaks : np.array
            Indices of detected systolic peaks in the PPG waveform.

        Returns
        -------
        troughs : np.array
            Indices of the detected troughs between the systolic peaks.

        Example
        -------
        >>> waveform = np.sin(np.linspace(0, 2*np.pi, 100))  # Simulated PPG signal
        >>> wm = WaveformMorphology(waveform, signal_type="PPG")
        >>> peaks = PeakDetection(waveform).detect_peaks()
        >>> troughs = wm.detect_troughs(peaks)
        >>> print(f"Troughs: {troughs}")
        """
        troughs = []
        for i in range(len(peaks) - 1):
            segment = self.waveform[peaks[i] : peaks[i + 1]]
            trough = np.argmin(segment) + peaks[i]
            troughs.append(trough)
        return np.array(troughs)

    def detect_notches(self):
        """
        Detects the dicrotic notches in a PPG waveform using second derivative.

        Returns
        -------
        notches : np.array
            Indices of detected dicrotic notches in the PPG waveform.

        Example
        -------
        >>> waveform = np.sin(np.linspace(0, 2*np.pi, 100))  # Simulated PPG signal
        >>> wm = WaveformMorphology(waveform, signal_type="PPG")
        >>> notches = wm.detect_notches()
        >>> print(f"Dicrotic Notches: {notches}")
        """
        if self.signal_type != "PPG":
            raise ValueError("Notches can only be detected for PPG signals.")

        notch_detector = PeakDetection(self.waveform, method="ppg_second_derivative")
        notches = notch_detector.detect_peaks()
        return notches

    def detect_q_session(self, r_peaks):
        """
        Detects the Q sessions (start and end) in the ECG waveform based on R peaks.

        Parameters
        ----------
        r_peaks : np.array
            Indices of detected R peaks in the ECG waveform.

        Returns
        -------
        q_sessions : list of tuples
            Each tuple contains the start and end index of a Q session.

        Example
        -------
        >>> waveform = np.sin(np.linspace(0, 2*np.pi, 100))  # Simulated ECG signal
        >>> wm = WaveformMorphology(waveform, signal_type="ECG")
        >>> r_peaks = PeakDetection(waveform).detect_peaks()
        >>> q_sessions = wm.detect_q_session(r_peaks)
        >>> print(f"Q Sessions: {q_sessions}")
        """
        if self.signal_type != "ECG":
            raise ValueError("Q sessions can only be detected for ECG signals.")

        q_sessions = []
        for r_peak in r_peaks:
            q_start = max(
                0, r_peak - int(self.fs * 0.04)
            )  # Example: 40 ms before R peak
            q_end = r_peak
            q_sessions.append((q_start, q_end))
        return q_sessions

    def detect_r_session(self, r_peaks):
        """
        Detects the R sessions (start and end) in the ECG waveform.

        Parameters
        ----------
        r_peaks : np.array
            Indices of detected R peaks in the ECG waveform.

        Returns
        -------
        r_sessions : list of tuples
            Each tuple contains the start and end index of an R session.

        Example
        -------
        >>> waveform = np.sin(np.linspace(0, 2*np.pi, 100))  # Simulated ECG signal
        >>> wm = WaveformMorphology(waveform, signal_type="ECG")
        >>> r_peaks = PeakDetection(waveform).detect_peaks()
        >>> r_sessions = wm.detect_r_session(r_peaks)
        >>> print(f"R Sessions: {r_sessions}")
        """
        r_sessions = [
            (r_peak - int(self.fs * 0.02), r_peak + int(self.fs * 0.02))
            for r_peak in r_peaks
        ]
        return r_sessions

    def detect_s_session(self, r_peaks):
        """
        Detects the S sessions (start and end) in the ECG waveform based on R peaks.

        Parameters
        ----------
        r_peaks : np.array
            Indices of detected R peaks in the ECG waveform.

        Returns
        -------
        s_sessions : list of tuples
            Each tuple contains the start and end index of an S session.

        Example
        -------
        >>> waveform = np.sin(np.linspace(0, 2*np.pi, 100))  # Simulated ECG signal
        >>> wm = WaveformMorphology(waveform, signal_type="ECG")
        >>> r_peaks = PeakDetection(waveform).detect_peaks()
        >>> s_sessions = wm.detect_s_session(r_peaks)
        >>> print(f"S Sessions: {s_sessions}")
        """
        if self.signal_type != "ECG":
            raise ValueError("S sessions can only be detected for ECG signals.")

        s_sessions = []
        for r_peak in r_peaks:
            s_start = r_peak
            s_end = min(
                len(self.waveform), r_peak + int(self.fs * 0.04)
            )  # Example: 40 ms after R peak
            s_sessions.append((s_start, s_end))
        return s_sessions

    def detect_qrs_session(self, r_peaks):
        """
        Detects the QRS complex sessions (start and end) in the ECG waveform.

        Parameters
        ----------
        r_peaks : np.array
            Indices of detected R peaks in the ECG waveform.

        Returns
        -------
        qrs_sessions : list of tuples
            Each tuple contains the start and end index of a QRS session.

        Example
        -------
        >>> waveform = np.sin(np.linspace(0, 2*np.pi, 100))  # Simulated ECG signal
        >>> wm = WaveformMorphology(waveform, signal_type="ECG")
        >>> r_peaks = PeakDetection(waveform).detect_peaks()
        >>> qrs_sessions = wm.detect_qrs_session(r_peaks)
        >>> print(f"QRS Sessions: {qrs_sessions}")
        """
        q_sessions = self.detect_q_session(r_peaks)
        s_sessions = self.detect_s_session(r_peaks)

        qrs_sessions = [(q[0], s[1]) for q, s in zip(q_sessions, s_sessions)]
        return qrs_sessions

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

    def compute_volume(self, peaks1, peaks2, mode="peak"):
        """
        Compute the area under the curve between two sets of peaks.

        Parameters
        ----------
        peaks1 : numpy.ndarray
            The first set of peaks (e.g., systolic peaks).
        peaks2 : numpy.ndarray
            The second set of peaks (e.g., diastolic peaks).
        mode : str, optional
            The type of area computation method ("peak" or "trough"). Default is "peak".

        Returns
        -------
        volume : float
            The mean area between the two sets of peaks.

        Examples
        --------
        >>> signal = np.random.randn(1000)
        >>> peaks1 = np.array([100, 200, 300])
        >>> peaks2 = np.array([150, 250, 350])
        >>> extractor = PhysiologicalFeatureExtractor(signal)
        >>> volume = extractor.compute_volume(peaks1, peaks2)
        >>> print(volume)
        """
        if mode == "peak":
            areas = [
                np.trapz(self.waveform[p1:p2])
                for p1, p2 in zip(peaks1, peaks2)
                if p1 < p2
            ]
        elif mode == "trough":
            areas = [
                np.trapz(self.waveform[min(p, t) : max(p, t)])
                for p, t in zip(peaks1, peaks2)
            ]
        else:
            raise ValueError("Volume mode must be 'peak' or 'trough'.")
        return np.mean(areas) if areas else 0.0

    def compute_volume_sequence(self, peaks):
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

    def compute_skewness(self):
        """
        Compute the skewness of the signal.

        Returns
        -------
        skewness : float
            The skewness of the signal.

        Examples
        --------
        >>> signal = np.random.randn(1000)
        >>> extractor = PhysiologicalFeatureExtractor(signal)
        >>> skewness = extractor.compute_skewness()
        >>> print(skewness)
        """
        return skew(self.waveform)

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

    def compute_duration(self, peaks1, peaks2, mode):
        """
        Compute the mean duration between two sets of peaks.

        Parameters
        ----------
        peaks1 : numpy.ndarray
            The first set of peaks (e.g., systolic peaks).
        peaks2 : numpy.ndarray
            The second set of peaks (e.g., diastolic peaks).
        mode : str, optional
            The type of duration computation method ("peak" or "trough"). Default is "peak".

        Returns
        -------
        duration : float
            The mean duration between the two sets of peaks in seconds.

        Examples
        --------
        >>> peaks1 = np.array([100, 200, 300])
        >>> peaks2 = np.array([150, 250, 350])
        >>> extractor = PhysiologicalFeatureExtractor(np.random.randn(1000))
        >>> duration = extractor.compute_duration(peaks1, peaks2)
        >>> print(duration)
        """
        if mode == "peak":
            durations = [
                (p2 - p1) / self.fs for p1, p2 in zip(peaks1, peaks2) if p1 < p2
            ]
        elif mode == "trough":
            durations = [(t - p) / self.fs for p, t in zip(peaks1, peaks2)]
        else:
            raise ValueError("Duration mode must be 'peak' or 'trough'.")
        return np.mean(durations) if durations else 0.0

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

    def compute_slope(self, peaks1, peaks2, mode="peak"):
        """
        Compute the mean slope between two sets of peaks.

        Parameters
        ----------
        peaks1 : numpy.ndarray
            The first set of peaks.
        peaks2 : numpy.ndarray
            The second set of peaks.
        mode : str, optional
        The type of duration computation method ("peak" or "trough"). Default is "peak".

        Returns
        -------
        slope : float
            The mean slope between the two sets of peaks.

        Examples
        --------
        >>> peaks1 = np.array([100, 200, 300])
        >>> peaks2 = np.array([150, 250, 350])
        >>> extractor = PhysiologicalFeatureExtractor(np.random.randn(1000))
        >>> slope = extractor.compute_slope(peaks1, peaks2)
        >>> print(slope)
        """
        if mode == "peak":
            slopes = [
                (self.waveform[p2] - self.waveform[p1]) / (p2 - p1)
                for p1, p2 in zip(peaks1, peaks2)
                if p1 < p2
            ]
        elif mode == "trough":
            slopes = [
                (self.waveform[p] - self.waveform[t]) / (p - t)
                for p, t in zip(peaks1, peaks2)
            ]
        else:
            raise ValueError("Slope mode must be 'peak' or 'trough'.")
        return np.mean(slopes) if slopes else 0.0

    def compute_slope_sequence(self):
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
