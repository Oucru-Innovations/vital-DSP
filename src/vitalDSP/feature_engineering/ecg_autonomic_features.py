import numpy as np
from vitalDSP.utils.peak_detection import PeakDetection
from vitalDSP.physiological_features.waveform import WaveformMorphology


class ECGExtractor:
    """
    A class to extract ECG features including:
    - P-wave analysis (duration, amplitude)
    - PR Interval (P-wave to QRS onset)
    - QRS Complex (width, amplitude)
    - ST Segment (elevation, depression)
    - QT Interval (QRS onset to T-wave end)
    - Detection of Arrhythmias (AFib, VTach, Bradycardia)

    Example usage:
    ```
    ecg_signal = np.random.rand(1000)  # Replace with actual ECG signal
    fs = 250  # Sampling frequency in Hz

    extractor = ECGExtractor(ecg_signal, fs)

    p_wave_duration = extractor.compute_p_wave_duration()
    pr_interval = extractor.compute_pr_interval()
    qrs_width = extractor.compute_qrs_width()
    qt_interval = extractor.compute_qt_interval()
    st_segment = extractor.compute_st_segment()
    arrhythmias = extractor.detect_arrhythmias()

    print(f"P-wave Duration: {p_wave_duration}, PR Interval: {pr_interval}, QRS Width: {qrs_width}")
    ```
    """

    def __init__(self, ecg_signal, sampling_frequency):
        if not isinstance(ecg_signal, np.ndarray):
            raise TypeError("Input signal must be a numpy array")
        if len(ecg_signal) < 2:
            raise ValueError("ECG signal is too short to compute features")
        if np.isnan(ecg_signal).any() or np.isinf(ecg_signal).any():
            raise ValueError("ECG signal contains invalid values")

        self.ecg_signal = ecg_signal
        self.fs = sampling_frequency

        # Initialize the WaveformMorphology for Q, R, S, T detection
        self.morphology = WaveformMorphology(
            ecg_signal, fs=sampling_frequency, signal_type="ECG"
        )

    def detect_r_peaks(self):
        """
        Detects R-peaks from the ECG signal using WaveformMorphology.

        Returns:
            np.array: Array of indices where R-peaks are detected.
        """
        detector = PeakDetection(self.ecg_signal, method="ecg_r_peak")
        r_peaks = detector.detect_peaks()
        if len(r_peaks) == 0:
            raise ValueError("No R-peaks detected in ECG signal")
        return r_peaks

    def compute_p_wave_duration(self, r_peaks=None):
        """
        Computes the P-wave duration based on R-peak detection.

        Returns:
            float: Duration of P-wave in seconds.
        """
        if r_peaks is None:
            r_peaks = self.detect_r_peaks()  # Detect R-peaks first
        p_waves = self.morphology.detect_q_session(r_peaks)
        if not p_waves:
            return 0.0
        durations = [(end - start) / self.fs for start, end in p_waves]
        return np.mean(durations)

    def compute_pr_interval(self, r_peaks=None):
        """
        Computes the PR interval (P-wave to QRS onset).

        Returns:
            float: PR interval in seconds.
        """
        if r_peaks is None:
            r_peaks = self.detect_r_peaks()  # Detect R-peaks first
        pr_intervals = self.morphology.detect_q_session(r_peaks)
        if not pr_intervals:
            return 0.0
        durations = [(end - start) / self.fs for start, end in pr_intervals]
        return np.mean(durations)

    def compute_qrs_duration(self, r_peaks=None):
        """
        Computes the QRS duration using WaveformMorphology.

        Returns:
            float: The mean duration of QRS complexes in seconds.
        """
        if r_peaks is None:
            r_peaks = self.detect_r_peaks()  # Detect R-peaks first
        qrs_durations = self.morphology.detect_qrs_session(r_peaks)
        if not qrs_durations:
            return 0.0
        durations = [(end - start) / self.fs for start, end in qrs_durations]
        return np.mean(durations)

    def compute_s_wave(self, r_peaks=None):
        """
        Detects the S-wave based on the R-peaks using WaveformMorphology.

        Returns:
            np.array: Indices of detected S-wave points.
        """
        if r_peaks is None:
            r_peaks = self.detect_r_peaks()
        return self.morphology.detect_s_session(r_peaks)

    def compute_qt_interval(self):
        """
        Computes the QT interval (from QRS onset to T-wave end).

        Returns:
            float: QT interval in seconds.
        """
        qrs_duration = (
            self.morphology.compute_qrs_duration()
        )  # No arguments passed here
        return qrs_duration

    def compute_st_segment(self):
        """
        Computes the ST segment elevation or depression.

        Returns:
            float: Mean ST segment deviation from baseline.
        """
        return self.morphology.compute_qrs_duration()

    def detect_arrhythmias(self, r_peaks=None):
        """
        Detects basic arrhythmias such as:
        - Atrial Fibrillation (AFib)
        - Ventricular Tachycardia (VTach)
        - Bradycardia (slow heart rate)

        Returns:
            dict: Dictionary containing the detected arrhythmias.
        """
        if r_peaks is None:
            r_peaks = self.detect_r_peaks()
        rr_intervals = np.diff(r_peaks) / self.fs
        mean_rr = np.mean(rr_intervals)

        arrhythmias = {"AFib": False, "VTach": False, "Bradycardia": False}

        # Detect Atrial Fibrillation (AFib): Irregular RR intervals
        if np.std(rr_intervals) > 0.2 * mean_rr:
            arrhythmias["AFib"] = True

        # Detect Ventricular Tachycardia (VTach): Sustained high heart rate (> 100 bpm)
        if np.mean(rr_intervals) < 0.6:  # Corresponds to HR > 100 bpm
            arrhythmias["VTach"] = True

        # Detect Bradycardia: Slow heart rate (< 60 bpm)
        if np.mean(rr_intervals) > 1.0:  # Corresponds to HR < 60 bpm
            arrhythmias["Bradycardia"] = True

        return arrhythmias
