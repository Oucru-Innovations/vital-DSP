"""
Physiological Features Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations
- Comprehensive signal analysis

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.physiological_features.hrv_analysis import HrvAnalysis
    >>> signal = np.random.randn(1000)
    >>> processor = HrvAnalysis(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np
from vitalDSP.physiological_features.time_domain import TimeDomainFeatures
from vitalDSP.physiological_features.frequency_domain import FrequencyDomainFeatures
from vitalDSP.physiological_features.nonlinear import NonlinearFeatures
import logging as logger


class HRVFeatures:
    """
    A class to compute HRV features from physiological signals such as ECG and PPG.

    It combines time-domain, frequency-domain, and nonlinear features into one comprehensive feature extraction module.

    Attributes
    ----------
    nn_intervals : np.array
        The NN intervals (in milliseconds) between heartbeats.
    signal : np.array
        The physiological signal (ECG, PPG, etc.).
    fs : int
        The sampling frequency in Hz. Default is 1000 Hz.

    Methods
    -------
    compute_all_features()
        Computes all HRV features and returns them in a dictionary format.

    Examples
    --------
    >>> import numpy as np
    >>> from vitalDSP.physiological_features.hrv_analysis import HRVFeatures
    >>>
    >>> # Example 1: Using ECG signal
    >>> ecg_signal = np.random.randn(1000)  # Simulated ECG signal
    >>> hrv_ecg = HRVFeatures(ecg_signal, fs=256, signal_type="ECG")
    >>> hrv_features = hrv_ecg.compute_all_features()
    >>> print(f"SDNN: {hrv_features.get('sdnn', 'N/A')}")
    >>>
    >>> # Example 2: Using PPG signal
    >>> ppg_signal = np.random.randn(2000)  # Simulated PPG signal
    >>> hrv_ppg = HRVFeatures(ppg_signal, fs=128, signal_type="PPG")
    >>> hrv_features = hrv_ppg.compute_all_features()
    >>> print(f"RMSSD: {hrv_features.get('rmssd', 'N/A')}")
    >>>
    >>> # Example 3: Using pre-computed NN intervals
    >>> nn_intervals = np.array([800, 850, 820, 900, 780])  # RR intervals in ms
    >>> hrv_precomputed = HRVFeatures(None, nn_intervals=nn_intervals, fs=100)
    >>> hrv_features = hrv_precomputed.compute_all_features()
    >>> print(f"pNN50: {hrv_features.get('pnn50', 'N/A')}")
    """

    def __init__(
        self, signals, nn_intervals=None, fs=100, signal_type="PPG", options=None
    ):
        """
        Initializes the HRVFeatures object.

        Parameters
        ----------
        nn_intervals : np.array
            The NN intervals (in milliseconds) between heartbeats.
        signals : np.array
            The physiological signal (e.g., ECG, PPG).
        signal_type : str, optional (default="PPG")
            The type of signal. Options: 'ECG', 'PPG', 'EEG'.
        fs : int, optional
            The sampling frequency in Hz. Default is 100 Hz.
        """
        if nn_intervals is None:
            # Lazy import to avoid circular import
            from vitalDSP.transforms.beats_transformation import RRTransformation

            rr_transformation = RRTransformation(
                signals, fs=fs, signal_type=signal_type, options=options
            )
            rr_intervals = rr_transformation.process_rr_intervals()

            # Ensure valid RR intervals are returned
            if rr_intervals is None or len(rr_intervals) == 0:
                raise ValueError(
                    "RR interval transformation failed to extract valid RR intervals."
                )

            self.nn_intervals = np.array(rr_intervals)
        else:
            self.nn_intervals = np.array(nn_intervals)

        # Check for invalid NN intervals
        if len(self.nn_intervals) == 0:
            raise ValueError("NN intervals cannot be empty.")
        if np.all(self.nn_intervals == 0):
            raise ValueError("NN intervals cannot contain all zeros.")

        self.signal = np.array(signals) if signals is not None else None
        self.fs = fs

    def compute_all_features(self, include_complex_methods=None, **kwargs):
        """
        Computes all nonlinear features of the signal, with an option to skip
        time-consuming methods.

        Args:
            include_complex_methods (bool, optional): Whether to compute the time-consuming
                methods: compute_sample_entropy, compute_approximate_entropy, and
                compute_recurrence_features. If None (default), automatically enables for
                signals with ≥50 NN intervals.
            **kwargs: Additional parameters for specific feature computations.

        Returns:
            dict: A dictionary containing all the computed features.

        Example usage
        -------
        >>> nn_intervals = [800, 810, 790, 805, 795]  # NN intervals in ms
        >>> ecg_signal = np.random.randn(1000)  # Example ECG signal
        >>> fs = 1000  # Sampling frequency in Hz
        >>> hrv = HRVFeatures(nn_intervals, ecg_signal, fs)
        >>> all_features = hrv.compute_all_features()
        >>> print(all_features)
        """
        features = {}

        # Auto-determine whether to include complex methods
        if include_complex_methods is None:
            # Enable complex methods if we have sufficient data (≥50 intervals)
            include_complex_methods = len(self.nn_intervals) >= 50

        # Time-domain features
        time_features = TimeDomainFeatures(self.nn_intervals)
        for feature, method in [
            ("sdnn", time_features.compute_sdnn),
            ("rmssd", time_features.compute_rmssd),
            ("nn50", time_features.compute_nn50),
            ("pnn50", time_features.compute_pnn50),
            ("mean_nn", time_features.compute_mean_nn),
            ("median_nn", time_features.compute_median_nn),
            ("iqr_nn", time_features.compute_iqr_nn),
            ("std_nn", time_features.compute_std_nn),
            ("pnn_20", time_features.compute_pnn20),
            ("cvnn", time_features.compute_cvnn),
            ("hrv_triangular_index", time_features.compute_hrv_triangular_index),
            ("tinn", time_features.compute_tinn),
            ("sdsd", time_features.compute_sdsd),
        ]:
            try:
                features[feature] = method()
            except Exception as e:
                features[feature] = np.nan
                logger.error(f"Error computing {feature}: {e}")

        # Interpolate NN intervals to uniform 4 Hz sampling for frequency analysis
        nn_times = np.cumsum(self.nn_intervals) / 1000.0
        nn_times = nn_times - nn_times[0]
        if len(nn_times) > 1:
            uniform_times = np.arange(0, nn_times[-1], 1.0 / 4.0)
            nn_interp = np.interp(uniform_times, nn_times, self.nn_intervals)
        else:
            nn_interp = self.nn_intervals

        # Frequency-domain features
        freq_features = FrequencyDomainFeatures(nn_interp, fs=4)
        for feature, method in [
            # ("psd", freq_features.compute_psd),
            ("lf_power", freq_features.compute_lf),
            ("hf_power", freq_features.compute_hf),
            ("lf_hf_ratio", freq_features.compute_lf_hf_ratio),
            ("ulf_power", freq_features.compute_ulf),
            ("vlf_power", freq_features.compute_vlf),
            ("total_power", freq_features.compute_total_power),
            ("lfnu_power", freq_features.compute_lfnu),
            ("hfnu_power", freq_features.compute_hfnu),
        ]:
            try:
                features[feature] = method()
            except Exception as e:
                features[feature] = np.nan
                logger.error(f"Error computing {feature}: {e}")

        # Nonlinear features
        if self.signal is not None:
            nonlinear_features = NonlinearFeatures(self.nn_intervals, self.fs)
            for feature, method in [
                # ("sample_entropy", nonlinear_features.compute_sample_entropy),
                # ("approx_entropy", nonlinear_features.compute_approximate_entropy),
                ("fractal_dimension", nonlinear_features.compute_fractal_dimension),
                ("lyapunov_exponent", nonlinear_features.compute_lyapunov_exponent),
                ("dfa", nonlinear_features.compute_dfa),
                (
                    "poincare",
                    lambda: nonlinear_features.compute_poincare_features(),
                ),
                # ("recurrence", nonlinear_features.compute_recurrence_features),
            ]:
                try:
                    if feature == "poincare":
                        poincare_result = method()
                        features["poincare_sd1"] = poincare_result["sd1"]
                        features["poincare_sd2"] = poincare_result["sd2"]
                    else:
                        features[feature] = method()
                except Exception as e:
                    features[feature] = np.nan
                    logger.error(f"Error computing {feature}: {e}")

            try:
                if include_complex_methods:
                    features["sample_entropy"] = nonlinear_features.compute_sample_entropy(
                        **kwargs
                    )
                    features["approximate_entropy"] = (
                        nonlinear_features.compute_approximate_entropy(**kwargs)
                    )
                    recurrence_dict = nonlinear_features.compute_recurrence_features(
                        **kwargs
                    )
                    features["recurrence_rate"] = recurrence_dict["recurrence_rate"]
                    features["determinism"] = recurrence_dict["determinism"]
                    features["laminarity"] = recurrence_dict["laminarity"]
                else:
                    features["sample_entropy"] = None
                    features["approximate_entropy"] = None
                    features["recurrence_rate"] = None
                    features["determinism"] = None
                    features["laminarity"] = None
            except Exception as e:
                logger.error(f"Error computing complex features: {e}")
                features["sample_entropy"] = None
                features["approximate_entropy"] = None
                features["recurrence_rate"] = None
                features["determinism"] = None
                features["laminarity"] = None

        return features
