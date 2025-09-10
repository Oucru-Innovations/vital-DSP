import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample
from vitalDSP.utils.common import deprecated


class BeatToBeatAnalysis:
    """
    A class for advanced beat-to-beat and heart rate variability (HRV) analysis in ECG and PPG signals.

    Attributes:
        signal (np.array): The ECG or PPG signal.
        r_peaks (np.array): The indices of R-peaks (ECG) or systolic peaks (PPG).
        fs (int): The sampling frequency of the signal in Hz. Default is 1000 Hz.
        signal_type (str): The type of signal ('ECG' or 'PPG').
    """

    def __init__(self, signal, r_peaks, fs=1000, signal_type="ECG"):
        """
        Initializes the BeatToBeatAnalysis object.

        Args:
            signal (np.array): The ECG or PPG signal.
            r_peaks (np.array): The indices of R-peaks (ECG) or systolic peaks (PPG).
            fs (int): The sampling frequency of the signal in Hz. Default is 1000 Hz.
            signal_type (str): The type of signal ('ECG' or 'PPG'). Default is 'ECG'.
        """
        self.signal = np.array(signal)
        self.r_peaks = np.array(r_peaks)
        self.fs = fs  # Sampling frequency
        self.signal_type = signal_type  # 'ECG' or 'PPG'

    @deprecated("Use vitalDSP.transforms.beat_transformation.RRTransformation instead.")
    def compute_rr_intervals(self, correction_method=None, threshold=150):
        """
        Computes the R-R intervals (for ECG) or P-P intervals (for PPG) from the detected peaks.
        Optionally applies a correction method to fix false beat detections.

        Args:
            correction_method (str): Method to correct false detections ('interpolation', 'resampling', or 'adaptive_threshold').
                                    Default is None (no correction).
            threshold (int): Threshold for adaptive correction methods in milliseconds. Default is 150 ms.

        Returns:
            np.array: The corrected R-R or P-P intervals in milliseconds.

        Example:
            >>> r_peaks = [100, 200, 310, 420]  # Detected R-peaks or PPG systolic peaks
            >>> btb = BeatToBeatAnalysis(ecg_signal, r_peaks)
            >>> rr_intervals = btb.compute_rr_intervals(correction_method='interpolation')
            >>> print(f"R-R Intervals: {rr_intervals}")
        """
        rr_intervals = np.diff(self.r_peaks) * 1000 / self.fs  # Convert to milliseconds

        # Apply correction if specified
        if correction_method == "interpolation":
            rr_intervals = self._correct_by_interpolation(rr_intervals)
        elif correction_method == "resampling":
            rr_intervals = self._correct_by_resampling(rr_intervals)
        elif correction_method == "adaptive_threshold":
            rr_intervals = self._correct_by_adaptive_threshold(rr_intervals, threshold)

        return rr_intervals

    def _correct_by_interpolation(self, rr_intervals):
        """
        Corrects false beat detections by applying linear interpolation on irregular intervals.

        Args:
            rr_intervals (np.array): The R-R or P-P intervals.

        Returns:
            np.array: The corrected R-R or P-P intervals after interpolation.
        """
        # Identify irregular intervals (e.g., very short or long intervals)
        outliers = np.abs(rr_intervals - np.median(rr_intervals)) > 1.5 * np.std(
            rr_intervals
        )

        # Interpolate over outliers
        if np.any(outliers):
            x = np.arange(len(rr_intervals))
            valid_points = x[~outliers]
            valid_intervals = rr_intervals[~outliers]
            f_interp = interp1d(
                valid_points, valid_intervals, kind="linear", fill_value="extrapolate"
            )
            rr_intervals[outliers] = f_interp(x[outliers])

        return rr_intervals

    def _correct_by_resampling(self, rr_intervals, new_rate=100):
        """
        Corrects false detections by resampling the R-R or P-P intervals at a regular rate.

        Args:
            rr_intervals (np.array): The R-R or P-P intervals.
            new_rate (int): The new sampling rate for resampling the intervals.

        Returns:
            np.array: The corrected R-R or P-P intervals after resampling.
        """
        num_points = len(rr_intervals)
        
        # Avoid division by zero and ensure we have a valid target length
        if self.fs <= 0:
            # If fs is invalid, return original intervals
            return rr_intervals
            
        target_length = num_points * new_rate // self.fs
        if target_length <= 0:
            # If target length is invalid, return original intervals
            return rr_intervals
            
        try:
            resampled_intervals = resample(rr_intervals, target_length)
            return resampled_intervals
        except (ZeroDivisionError, ValueError):
            # If resampling fails, return original intervals
            return rr_intervals

    def _correct_by_adaptive_threshold(self, rr_intervals, threshold=150):
        """
        Corrects false detections by removing intervals that are significantly shorter or longer than the mean.

        Args:
            rr_intervals (np.array): The R-R or P-P intervals.
            threshold (int): The threshold for detecting false beats in milliseconds. Default is 150 ms.

        Returns:
            np.array: The corrected R-R or P-P intervals after applying the adaptive threshold.
        """
        mean_rr = np.mean(rr_intervals)
        # Remove intervals that are much shorter or longer than the mean
        valid_rr_intervals = rr_intervals[
            (rr_intervals > mean_rr - threshold) & (rr_intervals < mean_rr + threshold)
        ]
        return valid_rr_intervals

    @deprecated(
        "Use vitalDSP.physiological_features.time_domain.compute_mean_nn() instead."
    )
    def compute_mean_rr(self):
        """
        Computes the mean of the R-R intervals.

        Returns:
            float: The mean R-R interval in milliseconds.
        """
        rr_intervals = self.compute_rr_intervals()
        return np.mean(rr_intervals)

    @deprecated(
        "Use vitalDSP.physiological_features.time_domain.compute_std_nn() instead."
    )
    def compute_sdnn(self):
        """
        Computes the standard deviation of the R-R intervals (SDNN), a common HRV metric.

        Returns:
            float: The standard deviation of R-R intervals (SDNN).
        """
        rr_intervals = self.compute_rr_intervals()
        return np.std(rr_intervals)

    @deprecated(
        "Use vitalDSP.physiological_features.time_domain.compute_rmssd() instead."
    )
    def compute_rmssd(self):
        """
        Computes the Root Mean Square of Successive Differences (RMSSD), a measure of short-term
        heart rate variability (HRV).

        Returns:
            float: The RMSSD value in milliseconds.
        """
        rr_intervals = self.compute_rr_intervals()
        successive_diffs = np.diff(rr_intervals)
        return np.sqrt(np.mean(successive_diffs**2))

    @deprecated(
        "Use vitalDSP.physiological_features.time_domain.compute_pnn50() instead."
    )
    def compute_pnn50(self):
        """
        Computes the percentage of successive R-R intervals that differ by more than 50 milliseconds (pNN50),
        a commonly used HRV metric.

        Returns:
            float: The percentage of R-R intervals differing by more than 50 ms (pNN50).
        """
        rr_intervals = self.compute_rr_intervals()
        diff_rr_intervals = np.abs(np.diff(rr_intervals))
        nn50 = np.sum(diff_rr_intervals > 50)
        pnn50 = (nn50 / len(rr_intervals)) * 100
        return pnn50

    def compute_hr(self):
        """
        Computes the heart rate (HR) from the R-R or P-P intervals.

        Returns:
            np.array: The heart rate (in beats per minute) for each R-R interval.
        """
        rr_intervals = self.compute_rr_intervals()
        hr = (
            60 * 1000 / rr_intervals
        )  # Convert R-R intervals to heart rate in beats per minute (bpm)
        return hr

    def detect_arrhythmias(self, threshold=150):
        """
        Detects arrhythmias by identifying irregular R-R intervals based on a threshold for variability.

        Args:
            threshold (int): The threshold for detecting abnormal R-R intervals in milliseconds.
                             Default is 150 ms.

        Returns:
            list: Indices of arrhythmic beats.
        """
        rr_intervals = self.compute_rr_intervals()
        mean_rr = np.mean(rr_intervals)
        arrhythmic_beats = np.where(np.abs(rr_intervals - mean_rr) > threshold)[0]
        return arrhythmic_beats
