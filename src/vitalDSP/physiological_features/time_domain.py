"""
time_domain.py

This module provides a TimeDomainFeatures class for extracting time-domain features
from physiological signals such as ECG and PPG. The class includes methods for
standard HRV metrics like SDNN, RMSSD, NN50, pNN50, and additional time-domain
metrics such as median NN intervals and interquartile range (IQR).

Class:
    - TimeDomainFeatures: A class to compute various time-domain features for physiological signals.

Example usage:
    >>> nn_intervals = [800, 810, 790, 805, 795]
    >>> tdf = TimeDomainFeatures(nn_intervals)
    >>> sdnn = tdf.compute_sdnn()
    >>> rmssd = tdf.compute_rmssd()
    >>> print(f"SDNN: {sdnn}, RMSSD: {rmssd}")
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
