"""
ECG Autonomic Features Module for Physiological Signal Processing

This module provides comprehensive ECG feature extraction capabilities focusing
on autonomic nervous system analysis. It implements advanced algorithms for
detecting ECG waveform components, computing intervals, and identifying
arrhythmias for cardiovascular health assessment.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- P-wave analysis (duration, amplitude)
- PR Interval computation (P-wave to QRS onset)
- QRS Complex analysis (width, amplitude)
- ST Segment analysis (elevation, depression)
- QT Interval computation (QRS onset to T-wave end)
- Arrhythmia detection (AFib, VTach, Bradycardia)
- Waveform morphology analysis
- Comprehensive ECG feature extraction

Examples:
--------
Basic ECG feature extraction:
    >>> import numpy as np
    >>> from vitalDSP.feature_engineering.ecg_autonomic_features import ECGExtractor
    >>> ecg_signal = np.random.rand(1000)  # Replace with actual ECG signal
    >>> fs = 250  # Sampling frequency in Hz
    >>> extractor = ECGExtractor(ecg_signal, fs)
    >>> p_wave_duration = extractor.compute_p_wave_duration()
    >>> pr_interval = extractor.compute_pr_interval()
    >>> qrs_width = extractor.compute_qrs_width()
    >>> print(f"P-wave Duration: {p_wave_duration}, PR Interval: {pr_interval}")

Advanced ECG analysis:
    >>> qt_interval = extractor.compute_qt_interval()
    >>> st_segment = extractor.compute_st_segment()
    >>> arrhythmias = extractor.detect_arrhythmias()
    >>> print(f"QT Interval: {qt_interval}, ST Segment: {st_segment}")
    >>> print(f"Arrhythmias detected: {arrhythmias}")

Comprehensive feature extraction:
    >>> all_features = extractor.extract_all_features()
    >>> print(f"Extracted {len(all_features)} ECG features")
"""

import numpy as np
from vitalDSP.utils.signal_processing.peak_detection import PeakDetection
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

    def _find_p_onset(self, p_peak, search_start, derivative):
        """Find P-wave onset by looking for the last derivative zero-crossing
        before the P-peak, or fall back to a percentage-based boundary."""
        onset = search_start
        if p_peak > search_start and p_peak < len(derivative):
            seg_deriv = derivative[search_start:p_peak]
            if len(seg_deriv) > 0:
                zero_crossings = np.where(np.diff(np.sign(seg_deriv)))[0]
                if len(zero_crossings) > 0:
                    onset = search_start + zero_crossings[-1]
                else:
                    # Fallback: use the point of minimum absolute derivative
                    onset = search_start + np.argmin(np.abs(seg_deriv))
        return onset

    def _pair_p_peaks_q_valleys(self, p_peaks, q_valleys):
        """Pair each P-peak with the nearest Q-valley that follows it."""
        pairs = []
        q_idx = 0
        for p in p_peaks:
            while q_idx < len(q_valleys) and q_valleys[q_idx] <= p:
                q_idx += 1
            if q_idx < len(q_valleys):
                pairs.append((p, q_valleys[q_idx]))
        return pairs

    def compute_p_wave_duration(self, r_peaks=None):
        """
        Computes the P-wave duration by finding the onset and offset
        around each detected P-peak.

        Returns:
            float: Mean duration of P-waves in seconds.
        """
        if r_peaks is None:
            r_peaks = self.detect_r_peaks()
        q_valleys = self.morphology.detect_q_valley(r_peaks=r_peaks)
        p_peaks = self.morphology.detect_p_peak(
            r_peaks=r_peaks, q_valleys=q_valleys
        )
        if len(p_peaks) == 0:
            return 0.0

        pairs = self._pair_p_peaks_q_valleys(p_peaks, q_valleys)
        if len(pairs) == 0:
            return 0.0

        durations = []
        derivative = np.diff(self.ecg_signal)
        for p_peak, q_valley in pairs:
            search_start = max(0, p_peak - int(self.fs * 0.12))
            onset = self._find_p_onset(p_peak, search_start, derivative)

            offset = q_valley
            if q_valley > p_peak and p_peak < len(derivative):
                seg_deriv = derivative[p_peak:q_valley]
                if len(seg_deriv) > 0:
                    zero_crossings = np.where(np.diff(np.sign(seg_deriv)))[0]
                    if len(zero_crossings) > 0:
                        offset = p_peak + zero_crossings[0]
                    else:
                        offset = p_peak + np.argmin(np.abs(seg_deriv))

            if offset > onset:
                durations.append((offset - onset) / self.fs)

        return np.mean(durations) if durations else 0.0

    def compute_pr_interval(self, r_peaks=None):
        """
        Computes the PR interval from P-wave onset to QRS onset (Q-valley).

        Returns:
            float: Mean PR interval in seconds.
        """
        if r_peaks is None:
            r_peaks = self.detect_r_peaks()
        q_valleys = self.morphology.detect_q_valley(r_peaks=r_peaks)
        p_peaks = self.morphology.detect_p_peak(
            r_peaks=r_peaks, q_valleys=q_valleys
        )
        if len(p_peaks) == 0 or len(q_valleys) == 0:
            return 0.0

        pairs = self._pair_p_peaks_q_valleys(p_peaks, q_valleys)
        if len(pairs) == 0:
            return 0.0

        derivative = np.diff(self.ecg_signal)
        intervals = []
        for p_peak, q_valley in pairs:
            search_start = max(0, p_peak - int(self.fs * 0.12))
            p_onset = self._find_p_onset(p_peak, search_start, derivative)

            if q_valley > p_onset:
                intervals.append((q_valley - p_onset) / self.fs)

        return np.mean(intervals) if intervals else 0.0

    def compute_qrs_duration(self, r_peaks=None):
        """
        Computes the QRS duration using WaveformMorphology.

        Returns:
            float: The mean duration of QRS complexes in seconds.
        """
        if r_peaks is None:
            r_peaks = self.detect_r_peaks()  # Detect R-peaks first
        qrs_durations = self.morphology.detect_qrs_session(r_peaks)
        # Convert to numpy array if it's a list (e.g., from mocked return value)
        if isinstance(qrs_durations, list):
            qrs_durations = np.array(qrs_durations)
        # Check if empty using len() for compatibility with both list and array
        if len(qrs_durations) == 0:
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
        q_valleys = self.morphology.detect_q_valley()
        t_peaks = self.morphology.detect_t_peak()
        if len(q_valleys) == 0 or len(t_peaks) == 0:
            return 0.0
        if q_valleys[0] > t_peaks[0]:
            t_peaks = t_peaks[1:]
        if len(t_peaks) == 0:
            return 0.0
        sessions = []
        for i in range(min(len(q_valleys), len(t_peaks))):
            q_start = q_valleys[i]
            t_end = t_peaks[i]
            if q_start < t_end:
                sessions.append((q_start, t_end))
        if len(sessions) == 0:
            return 0.0
        sessions = np.array(sessions)

        return self.morphology.compute_duration(sessions=sessions, mode="Custom")

    def compute_st_interval(self):
        """
        Computes the ST segment duration (from S-wave to T-wave peak).

        Returns:
            float: Mean ST segment duration in seconds.
        """
        s_valleys = self.morphology.detect_s_valley()
        t_peaks = self.morphology.detect_t_peak()
        if len(s_valleys) == 0 or len(t_peaks) == 0:
            return 0.0
        if s_valleys[0] > t_peaks[0]:
            t_peaks = t_peaks[1:]
        if len(t_peaks) == 0:
            return 0.0
        sessions = []
        for i in range(min(len(s_valleys), len(t_peaks))):
            s_start = s_valleys[i]
            t_end = t_peaks[i]
            if s_start < t_end:
                sessions.append((s_start, t_end))
        if len(sessions) == 0:
            return 0.0
        sessions = np.array(sessions)

        return self.morphology.compute_duration(sessions=sessions, mode="Custom")

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
