import numpy as np
from scipy.stats import linregress
from vitalDSP.utils.peak_detection import PeakDetection
from vitalDSP.preprocess.preprocess_operations import (
    PreprocessConfig,
    preprocess_signal,
)
from vitalDSP.physiological_features.waveform import WaveformMorphology


class PhysiologicalFeatureExtractor:
    """
    A class to extract various physiological features from ECG and PPG signals, such as durations,
    areas, amplitude variability, slope ratios, and dicrotic notch locations.

    Features for PPG:
    - Systolic/diastolic duration, area, amplitude variability
    - Signal skewness, slope, peak trends, and dicrotic notch locations

    Features for ECG:
    - QRS duration, area, T-wave area
    - Amplitude variability, QRS-T ratios, and QRS slope
    - Signal skewness, peak trends

    Methods
    -------
    preprocess_signal(preprocess_config)
        Preprocess the signal by applying bandpass filtering and noise reduction.
    extract_features(signal_type='ECG', preprocess_config=None)
        Extract all features (morphology, volume, amplitude variability, dicrotic notch) for ECG or PPG signals.
    """

    def __init__(self, signal, fs=1000):
        """
        Initialize the PhysiologicalFeatureExtractor class.

        Parameters
        ----------
        signal : numpy.ndarray
            The input physiological signal (ECG or PPG).
        fs : int, optional
            Sampling frequency in Hz. Default is 1000.
        """
        self.signal = np.asarray(signal, dtype=np.float64)
        self.fs = fs

    def detect_troughs(self, peaks):
        """
        Detect troughs (valleys) in the signal based on the given peaks.

        Parameters
        ----------
        peaks : numpy.ndarray
            The indices of detected peaks.

        Returns
        -------
        troughs : numpy.ndarray
            The indices of detected troughs.

        Examples
        --------
        >>> peaks = np.array([100, 200, 300])
        >>> troughs = extractor.detect_troughs(peaks)
        >>> print(troughs)
        """
        troughs = []
        for i in range(len(peaks) - 1):
            # Find the local minimum between two consecutive peaks
            segment = self.signal[peaks[i] : peaks[i + 1]]
            trough = np.argmin(segment) + peaks[i]
            troughs.append(trough)
        return np.array(troughs)

    def compute_peak_trend(self, peaks):
        """
        Compute the trend slope of peak amplitudes over time.

        Parameters
        ----------
        peaks : numpy.ndarray
            The set of peaks.

        Returns
        -------
        trend_slope : float
            The slope of the peak amplitude trend over time.

        Examples
        --------
        >>> signal = np.random.randn(1000)
        >>> peaks = np.array([100, 200, 300])
        >>> extractor = PhysiologicalFeatureExtractor(signal)
        >>> trend_slope = extractor.compute_peak_trend(peaks)
        >>> print(trend_slope)
        """
        amplitudes = self.signal[peaks]
        if len(peaks) > 1:
            slope, _, _, _, _ = linregress(peaks, amplitudes)
            return slope
        return 0.0

    def compute_amplitude_variability(self, peaks):
        """
        Compute the variability of the amplitudes at the given peak locations.

        Parameters
        ----------
        peaks : numpy.ndarray
            The set of peaks.

        Returns
        -------
        variability : float
            The amplitude variability (standard deviation of the peak amplitudes).

        Examples
        --------
        >>> signal = np.random.randn(1000)
        >>> peaks = np.array([100, 200, 300])
        >>> extractor = PhysiologicalFeatureExtractor(signal)
        >>> variability = extractor.compute_amplitude_variability(peaks)
        >>> print(variability)
        """
        amplitudes = self.signal[peaks]
        return np.std(amplitudes) if len(amplitudes) > 1 else 0.0

    def get_preprocess_signal(self, preprocess_config):
        """
        Preprocess the signal by applying bandpass filtering and noise reduction.

        Parameters
        ----------
        preprocess_config : PreprocessConfig
            Configuration for both signal filtering and artifact removal.

        Returns
        -------
        clean_signal : numpy.ndarray
            The preprocessed signal, cleaned of noise and artifacts.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)
        >>> preprocess_config = PreprocessConfig()
        >>> extractor = PhysiologicalFeatureExtractor(signal, fs=1000)
        >>> preprocessed_signal = extractor.preprocess_signal(preprocess_config)
        >>> print(preprocessed_signal)
        """
        return preprocess_signal(
            signal=self.signal,
            sampling_rate=self.fs,
            filter_type=preprocess_config.filter_type,
            lowcut=preprocess_config.lowcut,
            highcut=preprocess_config.highcut,
            order=preprocess_config.order,
            noise_reduction_method=preprocess_config.noise_reduction_method,
            wavelet_name=preprocess_config.wavelet_name,
            level=preprocess_config.level,
            window_length=preprocess_config.window_length,
            polyorder=preprocess_config.polyorder,
            kernel_size=preprocess_config.kernel_size,
            sigma=preprocess_config.sigma,
            respiratory_mode=preprocess_config.respiratory_mode,
        )

    def extract_features(self, signal_type="ECG", preprocess_config=None):
        """
        Extract all physiological features from the signal for either ECG or PPG.

        Parameters
        ----------
        signal_type : str, optional
            The type of signal ("ECG" or "PPG"). Default is "ECG".
        preprocess_config : PreprocessConfig, optional
            The configuration object for signal preprocessing. If None, default settings are used.

        Returns
        -------
        features : dict
            A dictionary containing the extracted features, such as durations, areas, amplitude variability,
            slopes, skewness, peak trends, and dicrotic notch locations.

        Examples
        --------
        >>> signal = np.random.randn(1000)
        >>> preprocess_config = PreprocessConfig()
        >>> extractor = PhysiologicalFeatureExtractor(signal, fs=1000)
        >>> features = extractor.extract_features(signal_type="ECG", preprocess_config=preprocess_config)
        >>> print(features)
        """
        if preprocess_config is None:
            preprocess_config = PreprocessConfig()

        # Preprocess the signal
        clean_signal = self.get_preprocess_signal(preprocess_config)

        # Baseline correction
        clean_signal = clean_signal - np.min(clean_signal)

        # Initialize the morphology class
        morphology = WaveformMorphology(
            clean_signal, fs=self.fs, signal_type=signal_type
        )

        # Initialize features as an empty dictionary before the try block
        features = {}

        try:
            if signal_type == "PPG":
                features = {
                    "systolic_duration": np.nan,
                    "diastolic_duration": np.nan,
                    "systolic_area": np.nan,
                    "diastolic_area": np.nan,
                    "systolic_slope": np.nan,
                    "diastolic_slope": np.nan,
                    "signal_skewness": np.nan,
                    "peak_trend_slope": np.nan,
                    "systolic_amplitude_variability": np.nan,
                    "diastolic_amplitude_variability": np.nan,
                }
                # Detect peaks and troughs in the PPG signal
                peaks = PeakDetection(
                    clean_signal, method="ppg_first_derivative"
                ).detect_peaks()
                troughs = morphology.detect_troughs(peaks)

                # Ensure peaks and troughs are numpy arrays
                peaks = np.array(peaks, dtype=int)
                troughs = np.array(troughs, dtype=int)

                # Adjust lengths and alignments
                # Since troughs are between peaks, len(troughs) = len(peaks) - 1
                # For computations, we need to align the indices correctly
                if peaks[0] < troughs[0]:
                    # The first peak comes before the first trough, remove the first peak
                    peaks = peaks[1:]
                if len(troughs) > len(peaks):
                    troughs = troughs[: len(peaks)]
                else:
                    peaks = peaks[: len(troughs)]

                # Compute systolic area (from trough to peak)
                systolic_areas = [
                    np.trapz(clean_signal[troughs[i] : peaks[i]])
                    for i in range(len(peaks))
                    if troughs[i] < peaks[i]
                ]
                systolic_area = np.mean(systolic_areas) if systolic_areas else 0.0

                # Compute diastolic area (from peak to next trough)
                diastolic_areas = []
                for i in range(len(peaks) - 1):
                    start = peaks[i]
                    end = troughs[i + 1]
                    if start < end:
                        area = np.trapz(clean_signal[start:end])
                        diastolic_areas.append(area)
                diastolic_area = np.mean(diastolic_areas) if diastolic_areas else 0.0

                # Compute systolic duration (time from trough to peak)
                systolic_durations = [
                    (peaks[i] - troughs[i]) / self.fs
                    for i in range(len(peaks))
                    if peaks[i] > troughs[i]
                ]
                systolic_duration = (
                    np.mean(systolic_durations) if systolic_durations else 0.0
                )

                # Compute diastolic duration (time from peak to next trough)
                diastolic_durations = [
                    (troughs[i + 1] - peaks[i]) / self.fs
                    for i in range(len(peaks) - 1)
                    if troughs[i + 1] > peaks[i]
                ]
                diastolic_duration = (
                    np.mean(diastolic_durations) if diastolic_durations else 0.0
                )

                # Compute systolic slope (rate of increase from trough to peak)
                systolic_slopes = [
                    (clean_signal[peaks[i]] - clean_signal[troughs[i]])
                    / (peaks[i] - troughs[i])
                    for i in range(len(peaks))
                    if peaks[i] > troughs[i]
                ]
                systolic_slope = np.mean(systolic_slopes) if systolic_slopes else 0.0

                # Compute diastolic slope (rate of decrease from peak to next trough)
                diastolic_slopes = [
                    (clean_signal[troughs[i + 1]] - clean_signal[peaks[i]])
                    / (troughs[i + 1] - peaks[i])
                    for i in range(len(peaks) - 1)
                    if troughs[i + 1] > peaks[i]
                ]
                diastolic_slope = np.mean(diastolic_slopes) if diastolic_slopes else 0.0

                # Compute other features
                # Dicrotic notch detection can be challenging; if not reliable, it can be omitted or handled separately
                # For simplicity, we'll omit dicrotic notch features here

                systolic_variability = self.compute_amplitude_variability(peaks)
                diastolic_variability = self.compute_amplitude_variability(troughs)
                signal_skewness = morphology.compute_skewness()
                peak_trend_slope = self.compute_peak_trend(peaks)

                features = {
                    "systolic_duration": systolic_duration,
                    "diastolic_duration": diastolic_duration,
                    "systolic_area": systolic_area,
                    "diastolic_area": diastolic_area,
                    "systolic_slope": systolic_slope,
                    "diastolic_slope": diastolic_slope,
                    "signal_skewness": signal_skewness,
                    "peak_trend_slope": peak_trend_slope,
                    "systolic_amplitude_variability": systolic_variability,
                    "diastolic_amplitude_variability": diastolic_variability,
                }

            elif signal_type == "ECG":
                features = {
                    "qrs_duration": np.nan,
                    "qrs_area": np.nan,
                    "qrs_amplitude": np.nan,
                    "qrs_slope": np.nan,
                    "t_wave_area": np.nan,
                    "heart_rate": np.nan,
                    "r_peak_amplitude_variability": np.nan,
                    "signal_skewness": np.nan,
                    "peak_trend_slope": np.nan,
                }
                # Detect R-peaks in the ECG signal
                peak_detector = PeakDetection(clean_signal, method="ecg_r_peak")
                r_peaks = peak_detector.detect_peaks()
                r_peaks = np.array(r_peaks, dtype=int)

                # Detect Q and S points around R peaks
                q_points = []
                s_points = []
                for r_peak in r_peaks:
                    # Q point detection (40 ms before R peak)
                    q_start = max(0, r_peak - int(self.fs * 0.04))
                    q_end = r_peak
                    q_segment = clean_signal[q_start:q_end]
                    if len(q_segment) > 0:
                        q_point = np.argmin(q_segment) + q_start
                        q_points.append(q_point)
                    else:
                        q_points.append(q_start)

                    # S point detection (40 ms after R peak)
                    s_start = r_peak
                    s_end = min(len(clean_signal), r_peak + int(self.fs * 0.04))
                    s_segment = clean_signal[s_start:s_end]
                    if len(s_segment) > 0:
                        s_point = np.argmin(s_segment) + s_start
                        s_points.append(s_point)
                    else:
                        s_points.append(s_end)

                # Compute QRS durations
                # qrs_durations = [
                #     (s_points[i] - q_points[i]) / self.fs
                #     for i in range(len(r_peaks))
                #     if s_points[i] > q_points[i]
                # ]
                qrs_durations = morphology.compute_qrs_duration(r_peaks)

                qrs_duration = np.mean(qrs_durations) if qrs_durations else 0.0

                # Compute QRS areas
                qrs_areas = [
                    np.trapz(clean_signal[q_points[i] : s_points[i]])
                    for i in range(len(r_peaks))
                    if s_points[i] > q_points[i]
                ]
                qrs_area = np.mean(qrs_areas) if qrs_areas else 0.0

                # Compute QRS amplitude
                qrs_amplitudes = clean_signal[r_peaks]
                qrs_amplitude = (
                    np.mean(qrs_amplitudes) if len(qrs_amplitudes) > 0 else 0.0
                )

                # Compute QRS slopes
                qrs_slopes = [
                    (clean_signal[r_peaks[i]] - clean_signal[q_points[i]])
                    / (r_peaks[i] - q_points[i])
                    for i in range(len(r_peaks))
                    if r_peaks[i] > q_points[i] and (r_peaks[i] - q_points[i]) != 0
                ]
                qrs_slope = np.mean(qrs_slopes) if qrs_slopes else 0.0

                # Compute RR intervals and heart rate
                rr_intervals = np.diff(r_peaks) / self.fs  # in seconds
                average_rr_interval = (
                    np.mean(rr_intervals) if len(rr_intervals) > 0 else 0.0
                )
                heart_rate = (
                    60 / average_rr_interval if average_rr_interval > 0 else 0.0
                )

                # Compute amplitude variability
                amplitude_variability = (
                    np.std(qrs_amplitudes) / np.mean(qrs_amplitudes)
                    if np.mean(qrs_amplitudes) != 0
                    else 0.0
                )

                # Compute T-wave areas
                t_wave_areas = []
                for i in range(len(r_peaks)):
                    s_point = s_points[i]
                    t_start = s_point
                    t_end = min(
                        len(clean_signal), s_point + int(self.fs * 0.3)
                    )  # 300 ms after S point
                    t_segment = clean_signal[t_start:t_end]
                    if len(t_segment) > 0:
                        # Compute area under the T-wave segment
                        area = np.trapz(t_segment)
                        t_wave_areas.append(area)
                t_wave_area = np.mean(t_wave_areas) if t_wave_areas else 0.0

                # Compute signal skewness
                signal_skewness = morphology.compute_skewness()

                # Compute peak trend
                peak_trend_slope = self.compute_peak_trend(r_peaks)

                features = {
                    "qrs_duration": qrs_duration,
                    "qrs_area": qrs_area,
                    "qrs_amplitude": qrs_amplitude,
                    "qrs_slope": qrs_slope,
                    "t_wave_area": t_wave_area,
                    "heart_rate": heart_rate,
                    "r_peak_amplitude_variability": amplitude_variability,
                    "signal_skewness": signal_skewness,
                    "peak_trend_slope": peak_trend_slope,
                }

            else:
                raise ValueError(f"Unsupported signal type: {signal_type}")

        except Exception as e:
            print(f"Error during feature extraction: {e}")
            features = {
                key: np.nan for key in features
            }  # Set all features to np.nan in case of error

        return features
