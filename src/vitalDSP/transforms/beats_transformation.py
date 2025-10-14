import numpy as np

# from scipy.signal import find_peaks
# from vitalDSP.utils.common import find_peaks
from vitalDSP.physiological_features.waveform import WaveformMorphology
from vitalDSP.transforms.vital_transformation import VitalTransformation
from vitalDSP.preprocess.preprocess_operations import (
    PreprocessConfig,
    preprocess_signal,
)
from vitalDSP.utils.signal_processing.interpolations import (
    linear_interpolation,
    spline_interpolation,
    mean_imputation,
    median_imputation,
    forward_fill,
    backward_fill,
    rolling_mean_imputation,
)


class RRTransformation:
    """
    A class to perform RR interval computation and enhancement from ECG/PPG signals,
    incorporating peak detection, invalid interval removal, imputation, and advanced filtering techniques.

    This class leverages the `VitalTransformation` for initial signal processing and then applies RR interval computation
    and enhancement steps.

    Parameters
    ----------
    signal : np.array
        The input ECG or PPG signal.
    fs : int
        Sampling frequency of the signal.
    signal_type : str, optional
        The type of the signal ('ecg' or 'ppg'). Default is 'ecg'.
    options : dict, optional
        Options to customize the signal processing and RR interval computations.

    Methods
    -------
    compute_rr_intervals(preprocess_config=None, peak_config=None)
        Compute the RR intervals after applying transformations and peak detection.
    remove_invalid_rr_intervals(rr_intervals, min_rr=0.3, max_rr=2.0)
        Remove invalid RR intervals based on physiological constraints and trends.
    impute_rr_intervals(rr_intervals, method='adaptive', **kwargs)
        Impute invalid RR intervals using various methods.
    enhance_rr_intervals(rr_intervals, options=None)
        Apply enhancements to the RR intervals, such as peak enhancement.
    """

    def __init__(self, signal, fs, signal_type="ECG", options=None):
        """
        Initialize the RRTransformation class.

        Parameters
        ----------
        signal : np.array
            The input ECG or PPG signal.
        fs : int
            Sampling frequency of the signal.
        signal_type : str, optional
            The type of the signal ('ecg' or 'ppg'). Default is 'ecg'.
        options : dict, optional
            Options to customize the signal processing and RR interval computations.
        """
        self.signal = signal
        self.fs = fs
        self.signal_type = signal_type
        self.options = options

        # Apply the initial signal processing using VitalTransformation
        self.transformer = VitalTransformation(signal, fs=fs, signal_type=signal_type)
        self.transformed_signal = self.transformer.apply_transformations(self.options)

    def compute_rr_intervals(self, preprocess_config=None, peak_config=None):
        """
        Compute the RR intervals after applying transformations and peak detection.

        Parameters
        ----------
        preprocess_config : PreprocessConfig, optional
            Configuration for signal preprocessing. If None, default settings are used.
        peak_config : dict, optional
            Configuration for peak detection parameters. If None, default settings are used.

        Returns
        -------
        rr_intervals : np.array
            The RR intervals computed from the detected peaks.
        """
        if preprocess_config is None:
            preprocess_config = PreprocessConfig()

        # Preprocess the signal
        clean_signal = preprocess_signal(
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

        waveform = WaveformMorphology(
            waveform=clean_signal,
            fs=self.fs,
            signal_type=self.signal_type,
            peak_config=peak_config,
        )
        if waveform.signal_type == "ECG":
            peaks = waveform.r_peaks
        else:
            peaks = waveform.systolic_peaks
        if len(peaks) == 0:
            raise ValueError(
                "No peaks detected in the signal. RR interval computation failed."
            )

        # Compute RR intervals from the detected peaks
        rr_intervals = np.diff(peaks) / self.fs  # Time between peaks (in seconds)

        if len(rr_intervals) == 0:
            raise ValueError(
                "RR interval transformation failed to extract valid RR intervals."
            )

        return rr_intervals

    def remove_invalid_rr_intervals(
        self,
        rr_intervals,
        min_rr=0.3,
        max_rr=2.0,
        std_dev_factor=2.0,
        sudden_change_threshold=0.2,
    ):
        """
        Removes invalid RR intervals based on physiological limits, standard deviation filtering,
        and sudden change detection. Gradual trends are reconsidered as valid.

        Parameters
        ----------
        rr_intervals : np.array
            The array of RR intervals (in seconds).
        min_rr : float, optional
            The minimum allowable RR interval (default is 0.3 seconds).
        max_rr : float, optional
            The maximum allowable RR interval (default is 2.0 seconds).
        std_dev_factor : float, optional
            The factor used for standard deviation filtering (default is 2.0).
        sudden_change_threshold : float, optional
            The relative threshold for detecting sudden changes in RR intervals. Expressed as a percentage of the previous interval.

        Returns
        -------
        np.array
            The array of RR intervals with invalid intervals removed (or marked as NaN).

        Example
        -------
        >>> valid_rr_intervals = remove_invalid_rr_intervals(rr_intervals)
        """
        rr_intervals_filtered = np.copy(rr_intervals)

        # Step 1: Remove values that fall outside physiological limits
        rr_intervals_filtered = np.where(
            (rr_intervals_filtered >= min_rr) & (rr_intervals_filtered <= max_rr),
            rr_intervals_filtered,
            np.nan,
        )

        # Step 2: Compute mean and standard deviation of valid RR intervals
        valid_rr_intervals = rr_intervals_filtered[~np.isnan(rr_intervals_filtered)]
        mean_rr = np.mean(valid_rr_intervals)
        std_rr = np.std(valid_rr_intervals)

        # Step 3: Mark intervals outside the range of mean ± std_dev_factor * std_rr as invalid
        rr_intervals_filtered = np.where(
            (rr_intervals_filtered >= mean_rr - std_dev_factor * std_rr)
            & (rr_intervals_filtered <= mean_rr + std_dev_factor * std_rr),
            rr_intervals_filtered,
            np.nan,
        )

        # Step 4: Detect sudden changes
        for i in range(1, len(rr_intervals_filtered)):
            if not np.isnan(rr_intervals_filtered[i]) and not np.isnan(
                rr_intervals_filtered[i - 1]
            ):
                change = (
                    np.abs(rr_intervals_filtered[i] - rr_intervals_filtered[i - 1])
                    / rr_intervals_filtered[i - 1]
                )
                # If a sudden change is detected and it isn't part of a trend, mark it invalid
                if change > sudden_change_threshold:
                    rr_intervals_filtered[i] = np.nan

        # Step 5: Reconsider trends using a running average
        rr_intervals_filtered = self._reconsider_trends(rr_intervals_filtered)

        return rr_intervals_filtered

    def _reconsider_trends(self, rr_intervals, window_size=5):
        """
        Reconsiders RR intervals that show a gradual trend (increase or decrease) and avoids marking them as invalid.

        Parameters
        ----------
        rr_intervals : np.array
            The array of RR intervals with invalid intervals marked as NaN.
        window_size : int, optional
            The size of the window used to detect trends. Defaults to 5.

        Returns
        -------
        np.array
            The array of RR intervals with trends reconsidered.
        """
        rr_intervals_adaptive = np.copy(rr_intervals)
        running_mean = np.full(len(rr_intervals), np.nan)

        # Calculate a running mean of valid RR intervals
        for i in range(window_size, len(rr_intervals)):
            if not np.isnan(rr_intervals[i - window_size : i]).any():
                running_mean[i] = np.mean(rr_intervals[i - window_size : i])

        # If an RR interval is invalid, but close to the running mean, it could be a part of a trend
        for i in range(window_size, len(rr_intervals)):
            if np.isnan(rr_intervals[i]) and not np.isnan(running_mean[i]):
                if np.abs(rr_intervals[i] - running_mean[i]) < 0.1 * running_mean[i]:
                    # Reconsider this interval as valid
                    rr_intervals_adaptive[i] = running_mean[i]

        return rr_intervals_adaptive

    def impute_rr_intervals(self, intervals, method="adaptive", **kwargs):
        """
        Impute invalid RR intervals (NaN values) using various methods, with adaptive selection based on data patterns.

        Parameters
        ----------
        intervals : np.array
            The array of RR intervals with NaN values for invalid intervals.
        method : str, optional
            The imputation method to use. If 'adaptive', the function selects the best method based on data.
            Other options are 'linear', 'spline', 'mean', 'median', 'forward_fill', 'backward_fill', 'rolling_mean'.
            Defaults to 'adaptive'.
        **kwargs
            Additional keyword arguments for specific imputation methods, such as `order` for spline or `window` for rolling mean.

        Returns
        -------
        np.array
            The array of RR intervals with invalid intervals imputed.

        Examples
        --------
        >>> intervals = np.array([0.8, np.nan, 0.82, np.nan, np.nan, 0.85, 0.83])
        >>> imputed_intervals = impute_intervals(intervals, method='spline', order=3)
        >>> print(imputed_intervals)
        """
        # Handle adaptive imputation selection
        if method == "adaptive":
            nan_ratio = np.isnan(intervals).sum() / len(intervals)
            if nan_ratio < 0.05:
                return linear_interpolation(intervals)
            elif nan_ratio < 0.2:
                order = kwargs.get("order", 3)
                return spline_interpolation(intervals, order)
            else:
                window = kwargs.get("window", 5)
                return rolling_mean_imputation(intervals, window)

        # Manually specified method
        elif method == "linear":
            return linear_interpolation(intervals)
        elif method == "spline":
            order = kwargs.get("order", 3)
            return spline_interpolation(intervals, order)
        elif method == "mean":
            return mean_imputation(intervals)
        elif method == "median":
            return median_imputation(intervals)
        elif method == "forward_fill":
            return forward_fill(intervals)
        elif method == "backward_fill":
            return backward_fill(intervals)
        elif method == "rolling_mean":
            window = kwargs.get("window", 5)
            return rolling_mean_imputation(intervals, window)
        else:
            raise ValueError(f"Unsupported imputation method: {method}")

    def process_rr_intervals(self, remove_invalid=True, impute_invalid=True):
        """
        Full RR interval computation process:
        1. Compute RR intervals from the transformed signal.
        2. Remove invalid intervals.
        3. Impute missing values.
        4. Apply enhancement (if needed).

        Parameters
        ----------
        remove_invalid : bool, optional
            Whether to remove invalid RR intervals. Default is True.
        impute_invalid : bool, optional
            Whether to impute missing RR intervals. Default is True.
        Returns
        -------
        final_rr_intervals : np.array
            The fully processed RR intervals.
        """
        # Step 1: Compute RR intervals
        rr_intervals = self.compute_rr_intervals()

        if remove_invalid:
            # Step 2: Remove invalid RR intervals
            rr_intervals = self.remove_invalid_rr_intervals(rr_intervals)

        if impute_invalid:
            # Step 3: Impute missing RR intervals
            rr_intervals = self.impute_rr_intervals(rr_intervals)

        return rr_intervals
