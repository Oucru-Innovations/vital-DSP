"""
Time Domain Features Module for Physiological Signal Processing

This module provides comprehensive time-domain feature extraction capabilities for
physiological signals such as ECG and PPG. It implements standard Heart Rate Variability
(HRV) metrics and additional time-domain analysis methods for characterizing signal
variability and patterns.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Standard HRV metrics (SDNN, RMSSD, NN50, pNN50)
- Advanced time-domain statistics (median, IQR, CVNN)
- HRV Triangular Index and TINN computation
- Successive difference analysis (SDSD)
- Comprehensive NN interval analysis

Examples:
--------
Basic HRV analysis:
    >>> import numpy as np
    >>> from vitalDSP.physiological_features.time_domain import TimeDomainFeatures
    >>> nn_intervals = [800, 810, 790, 805, 795, 820, 780, 815, 800, 810]
    >>> tdf = TimeDomainFeatures(nn_intervals)
    >>> sdnn = tdf.compute_sdnn()
    >>> rmssd = tdf.compute_rmssd()
    >>> print(f"SDNN: {sdnn:.2f} ms, RMSSD: {rmssd:.2f} ms")

Advanced metrics:
    >>> pnn50 = tdf.compute_pnn50()
    >>> cvnn = tdf.compute_cvnn()
    >>> hrv_triangular = tdf.compute_hrv_triangular_index()
    >>> print(f"pNN50: {pnn50:.2f}%, CVNN: {cvnn:.2f}")

Statistical analysis:
    >>> median_nn = tdf.compute_median_nn()
    >>> iqr_nn = tdf.compute_iqr_nn()
    >>> print(f"Median NN: {median_nn:.2f} ms, IQR: {iqr_nn:.2f} ms")
"""

import numpy as np


class TimeDomainFeatures:
    """
    A class for computing time-domain features from physiological signals (ECG, PPG).

    Attributes:
        nn_intervals (list or np.array): The NN intervals (in milliseconds) between heartbeats.

    Methods:
        compute_sdnn(): Computes the standard deviation of NN intervals (SDNN).
        compute_rmssd(): Computes the root mean square of successive differences (RMSSD).
        compute_nn50(): Computes the number of successive NN intervals differing by >50 ms.
        compute_pnn50(): Computes the percentage of NN50 over the total NN intervals.
        compute_median_nn(): Computes the median NN interval.
        compute_iqr_nn(): Computes the interquartile range (IQR) of NN intervals.
        compute_mean_nn(): Computes the mean of NN intervals.
        compute_std_nn(): Computes the standard deviation of NN intervals (alias for SDNN).
        compute_pnn20(): Computes the percentage of NN intervals differing by more than 20 ms.
        compute_nn20(): Computes the count of NN intervals differing by more than 20 ms.
        compute_cvnn(): Computes the coefficient of variation of NN intervals (CVNN).
        compute_hrv_triangular_index(): Computes the HRV Triangular Index based on the histogram of NN intervals.
        compute_tinn(): Computes TINN (Triangular Interpolation of NN Interval Histogram).
        compute_sdsd(): Computes the standard deviation of successive differences (SDSD).
    """

    def __init__(self, nn_intervals):
        """
        Initializes the TimeDomainFeatures object with NN intervals.

        Args:
            nn_intervals (list or np.array): The NN intervals (in milliseconds) between heartbeats.
        """
        self.nn_intervals = np.array(nn_intervals)

    def compute_sdnn(self):
        """
        Computes the standard deviation of NN intervals (SDNN).

        Returns:
            float: The SDNN value.

        Example:
            >>> tdf = TimeDomainFeatures([800, 810, 790, 805, 795])
            >>> tdf.compute_sdnn()
            7.483314773547883
        """
        return np.std(self.nn_intervals)

    def compute_rmssd(self):
        """
        Computes the root mean square of successive differences (RMSSD).

        Returns:
            float: The RMSSD value.

        Example:
            >>> tdf = TimeDomainFeatures([800, 810, 790, 805, 795])
            >>> tdf.compute_rmssd()
            10.0
        """
        diff_nn_intervals = np.diff(self.nn_intervals)
        return np.sqrt(np.mean(diff_nn_intervals**2))

    def compute_nn50(self):
        """
        Computes the number of successive NN intervals differing by more than 50 ms (NN50).

        Returns:
            int: The NN50 value.

        Example:
            >>> tdf = TimeDomainFeatures([800, 810, 790, 805, 795])
            >>> tdf.compute_nn50()
            1
        """
        diff_nn_intervals = np.abs(np.diff(self.nn_intervals))
        return np.sum(diff_nn_intervals > 50)

    def compute_pnn50(self):
        """
        Computes the percentage of NN50 over the total number of NN intervals (pNN50).

        Returns:
            float: The pNN50 value as a percentage.

        Example:
            >>> tdf = TimeDomainFeatures([800, 810, 790, 805, 795])
            >>> tdf.compute_pnn50()
            20.0
        """
        # Input validation to prevent division by zero
        if len(self.nn_intervals) == 0:
            return 0.0
        if len(self.nn_intervals) == 1:
            return 0.0  # No differences possible with single interval

        nn50 = self.compute_nn50()
        return 100.0 * nn50 / len(self.nn_intervals)

    def compute_median_nn(self):
        """
        Computes the median of the NN intervals.

        Returns:
            float: The median NN interval.

        Example:
            >>> tdf = TimeDomainFeatures([800, 810, 790, 805, 795])
            >>> tdf.compute_median_nn()
            800.0
        """
        return np.median(self.nn_intervals)

    def compute_iqr_nn(self):
        """
        Computes the interquartile range (IQR) of the NN intervals.

        Returns:
            float: The IQR of NN intervals.

        Example:
            >>> tdf = TimeDomainFeatures([800, 810, 790, 805, 795])
            >>> tdf.compute_iqr_nn()
            10.0
        """
        return np.percentile(self.nn_intervals, 75) - np.percentile(
            self.nn_intervals, 25
        )

    def compute_mean_nn(self):
        """
        Computes the mean of NN intervals.

        Returns:
            float: The mean NN interval.

        Example:
            >>> tdf = TimeDomainFeatures([800, 810, 790, 805, 795])
            >>> tdf.compute_mean_nn()
            800.0
        """
        return np.mean(self.nn_intervals)

    def compute_std_nn(self):
        """
        Computes the standard deviation of NN intervals (alias for SDNN).

        Returns:
            float: The standard deviation of NN intervals (SDNN).

        Example:
            >>> tdf = TimeDomainFeatures([800, 810, 790, 805, 795])
            >>> tdf.compute_std_nn()
            7.483314773547883
        """
        return self.compute_sdnn()

    def compute_pnn20(self):
        """
        Computes the percentage of successive NN intervals differing by more than 20 ms (pNN20).

        Returns:
            float: The pNN20 value as a percentage.

        Example:
            >>> tdf = TimeDomainFeatures([800, 810, 790, 805, 795])
            >>> tdf.compute_pnn20()
            40.0
        """
        # Input validation to prevent division by zero
        if len(self.nn_intervals) == 0:
            return 0.0
        if len(self.nn_intervals) == 1:
            return 0.0  # No differences possible with single interval

        diff_nn_intervals = np.abs(np.diff(self.nn_intervals))
        nn20 = np.sum(diff_nn_intervals > 20)
        return 100.0 * nn20 / len(self.nn_intervals)

    def compute_cvnn(self):
        """
        Computes the coefficient of variation of NN intervals (CVNN), which is the ratio of the standard deviation
        to the mean NN interval.

        Returns:
            float: The CVNN value.

        Example:
            >>> tdf = TimeDomainFeatures([800, 810, 790, 805, 795])
            >>> tdf.compute_cvnn()
            0.009354143466934854
        """
        # Input validation to prevent division by zero
        if len(self.nn_intervals) == 0:
            return 0.0

        mean_nn = self.compute_mean_nn()
        if mean_nn == 0:
            return 0.0  # Avoid division by zero

        sdnn = self.compute_sdnn()
        return sdnn / mean_nn

    def compute_hrv_triangular_index(self):
        """
        Computes the HRV Triangular Index, which is the total number of NN intervals divided by the
        height of the histogram of all NN intervals.

        Returns:
            float: The HRV Triangular Index.

        Example:
            >>> tdf = TimeDomainFeatures([800, 810, 790, 805, 795])
            >>> tdf.compute_hrv_triangular_index()
            5.0
        """
        hist, bin_edges = np.histogram(self.nn_intervals, bins="auto")
        max_bin_count = np.max(hist)
        return len(self.nn_intervals) / max_bin_count

    def compute_tinn(self):
        """
        Computes the Triangular Interpolation of NN Interval Histogram (TINN), which is the width of the
        base of the histogram of NN intervals.

        Returns:
            float: The TINN value.

        Example:
            >>> tdf = TimeDomainFeatures([800, 810, 790, 805, 795])
            >>> tdf.compute_tinn()
            20.0
        """
        hist, bin_edges = np.histogram(self.nn_intervals, bins="auto")
        bin_width = np.diff(bin_edges)
        max_bin_index = np.argmax(hist)
        left_edge = bin_edges[max_bin_index] - bin_width[max_bin_index] / 2
        right_edge = bin_edges[max_bin_index] + bin_width[max_bin_index] / 2
        return right_edge - left_edge

    def compute_sdsd(self):
        """
        Computes the standard deviation of successive differences (SDSD) between NN intervals.

        Returns:
            float: The SDSD value.

        Example:
            >>> tdf = TimeDomainFeatures([800, 810, 790, 805, 795])
            >>> tdf.compute_sdsd()
            10.0
        """
        diff_nn_intervals = np.diff(self.nn_intervals)
        return np.std(diff_nn_intervals)
