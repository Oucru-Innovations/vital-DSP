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
- Configurable parameters and settings

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.feature_engineering.morphology_features import MorphologyFeatures
    >>> signal = np.random.randn(1000)
    >>> processor = MorphologyFeatures(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np
from scipy.stats import linregress

# from vitalDSP.utils.signal_processing.peak_detection import PeakDetection
from vitalDSP.preprocess.preprocess_operations import (
    PreprocessConfig,
    preprocess_signal,
)
from vitalDSP.physiological_features.waveform import WaveformMorphology
import warnings
import logging as logger


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

    Examples
    --------
    >>> import numpy as np
    >>> from vitalDSP.feature_engineering.morphology_features import PhysiologicalFeatureExtractor
    >>> from vitalDSP.preprocess.preprocess_operations import PreprocessConfig
    >>>
    >>> # Example 1: ECG feature extraction
    >>> ecg_signal = np.random.randn(1000)  # Simulated ECG signal
    >>> extractor = PhysiologicalFeatureExtractor(ecg_signal, fs=256)
    >>> ecg_features = extractor.extract_features(signal_type="ECG")
    >>> print(f"ECG features extracted: {len(ecg_features)}")
    >>> print(f"QRS duration: {ecg_features.get('qrs_duration', 'N/A')}")
    >>>
    >>> # Example 2: PPG feature extraction with preprocessing
    >>> ppg_signal = np.random.randn(2000)  # Simulated PPG signal
    >>> extractor_ppg = PhysiologicalFeatureExtractor(ppg_signal, fs=128)
    >>> config = PreprocessConfig(
    ...     filter_type="bandpass",
    ...     lowcut=0.5,
    ...     highcut=8.0,
    ...     noise_reduction_method="wavelet"
    ... )
    >>> ppg_features = extractor_ppg.extract_features(signal_type="PPG", preprocess_config=config)
    >>> print(f"PPG features extracted: {len(ppg_features)}")
    >>> print(f"Systolic duration: {ppg_features.get('systolic_duration', 'N/A')}")
    >>>
    >>> # Example 3: EEG feature extraction
    >>> eeg_signal = np.random.randn(1500)  # Simulated EEG signal
    >>> extractor_eeg = PhysiologicalFeatureExtractor(eeg_signal, fs=512)
    >>> eeg_features = extractor_eeg.extract_features(signal_type="EEG")
    >>> print(f"EEG features extracted: {len(eeg_features)}")
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
        warnings.warn(
            "Deprecated. Please use vitalDSP.physiological_features.waveform.WaveformMorphology instead.",
            DeprecationWarning,
        )
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

    def extract_features(
        self, signal_type="ECG", preprocess_config=None, peak_config=None, options=None
    ):
        """
        Extract all physiological features from the signal for either ECG or PPG.

        Parameters
        ----------
        signal_type : str, optional
            The type of signal. Options: 'ECG', 'PPG', 'EEG'. Default is "ECG".
        preprocess_config : PreprocessConfig, optional
            The configuration object for signal preprocessing. If None, default settings are used.
        peak_config : dict, optional
            Configuration for peak detection parameters. If None, default settings are used.

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

        # Initialize features as an empty dictionary before the try block
        features = {}

        # Preprocess the signal
        try:
            clean_signal = self.get_preprocess_signal(preprocess_config)
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            return {key: np.nan for key in features}  # Set all features to NaN

        # Baseline correction
        clean_signal = clean_signal - np.min(clean_signal)

        # Initialize the morphology class
        try:
            morphology = WaveformMorphology(
                clean_signal,
                fs=self.fs,
                signal_type=signal_type,
                peak_config=peak_config,
                options=options,
            )
        except Exception as e:
            logger.error(f"Error initializing morphology: {e}")
            return {key: np.nan for key in features}  # Set all features to NaN

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
                    "heart_rate": np.nan,
                    "systolic_amplitude_variability": np.nan,
                    "diastolic_amplitude_variability": np.nan,
                }

                features = {
                    "systolic_duration": morphology.get_duration(
                        session_type="systolic"
                    ),
                    "diastolic_duration": morphology.get_duration(
                        session_type="diastolic"
                    ),
                    "systolic_area": morphology.get_area(
                        interval_type="Sys-to-Notch", signal_type="PPG"
                    ),
                    "diastolic_area": morphology.get_area(
                        interval_type="Notch-to-Dia", signal_type="PPG"
                    ),
                    "systolic_slope": morphology.get_slope(
                        slope_type="systolic",
                        window=(
                            peak_config["window_size"]
                            if "window_size" in peak_config
                            else 5
                        ),
                        slope_unit=(
                            peak_config["slope_unit"]
                            if "slope_unit" in peak_config
                            else "radians"
                        ),
                    ),
                    "diastolic_slope": morphology.get_slope(
                        slope_type="diastolic",
                        window=(
                            peak_config["window_size"]
                            if "window_size" in peak_config
                            else 5
                        ),
                        slope_unit=(
                            peak_config["slope_unit"]
                            if "slope_unit" in peak_config
                            else "radians"
                        ),
                    ),
                    "signal_skewness": morphology.get_signal_skewness(),
                    "peak_trend_slope": morphology.get_peak_trend_slope(),
                    "heart_rate": morphology.get_heart_rate(),
                    "systolic_amplitude_variability": morphology.get_amplitude_variability(
                        interval_type="Sys-to-Baseline", signal_type="PPG"
                    ),
                    "diastolic_amplitude_variability": morphology.get_amplitude_variability(
                        interval_type="Dia-to-Baseline", signal_type="PPG"
                    ),
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

                features = {
                    "qrs_duration": morphology.get_duration(session_type="qrs"),
                    "qrs_area": morphology.get_area(
                        interval_type="QRS", signal_type="ECG"
                    ),
                    "qrs_amplitude": morphology.get_qrs_amplitude(),
                    "qrs_slope": morphology.get_slope(slope_type="qrs", window=5),
                    "t_wave_area": morphology.get_area(
                        interval_type="T-to-S", signal_type="ECG"
                    ),
                    "heart_rate": morphology.get_heart_rate(),
                    "r_peak_amplitude_variability": morphology.get_amplitude_variability(
                        interval_type="R-to-Baseline", signal_type="ECG"
                    ),
                    "signal_skewness": morphology.get_signal_skewness(
                        signal_type="ECG"
                    ),
                    "peak_trend_slope": morphology.get_peak_trend_slope(),
                }

            else:
                raise ValueError(f"Unsupported signal type: {signal_type}")

        except Exception as e:
            # print(f"Error during feature extraction: {e}")
            # features = {
            #     key: np.nan for key in features
            # }  # Set all features to np.nan in case of error
            logger.error(f"Error during feature extraction: {e}")
            features = {
                key: np.nan for key in features
            }  # Set all features to NaN in case of error

        return features
