import numpy as np
from vitalDSP.physiological_features.time_domain import TimeDomainFeatures
from vitalDSP.physiological_features.frequency_domain import FrequencyDomainFeatures
from vitalDSP.physiological_features.nonlinear import NonlinearFeatures
from vitalDSP.transforms.beats_transformation import RRTransformation
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
        signal : np.array
            The physiological signal (e.g., ECG, PPG).
        fs : int, optional
            The sampling frequency in Hz. Default is 1000 Hz.
        """
        if nn_intervals is None:
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

        self.signal = np.array(signals)
        self.fs = fs

    def compute_all_features(self, include_complex_methods=False, **kwargs):
        """
        Computes all nonlinear features of the signal, with an option to skip
        time-consuming methods.

        Args:
            include_complex_methods (bool): Whether to compute the time-consuming
                methods: compute_sample_entropy, compute_approximate_entropy, and
                compute_recurrence_features.
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

        # Frequency-domain features
        freq_features = FrequencyDomainFeatures(self.nn_intervals, fs=4)
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
        nonlinear_features = NonlinearFeatures(self.signal, self.fs)
        for feature, method in [
            # ("sample_entropy", nonlinear_features.compute_sample_entropy),
            # ("approx_entropy", nonlinear_features.compute_approximate_entropy),
            ("fractal_dimension", nonlinear_features.compute_fractal_dimension),
            ("lyapunov_exponent", nonlinear_features.compute_lyapunov_exponent),
            ("dfa", nonlinear_features.compute_dfa),
            (
                "poincare",
                lambda: nonlinear_features.compute_poincare_features(self.nn_intervals),
            ),
            # ("recurrence", nonlinear_features.compute_recurrence_features),
        ]:
            try:
                if feature == "poincare":
                    poincare_sd1, poincare_sd2 = method()
                    features["poincare_sd1"] = poincare_sd1
                    features["poincare_sd2"] = poincare_sd2
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
