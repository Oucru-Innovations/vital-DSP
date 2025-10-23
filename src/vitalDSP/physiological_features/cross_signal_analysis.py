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
- Configurable parameters and settings
- Comprehensive signal analysis

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.physiological_features.cross_signal_analysis import CrossSignalAnalysis
    >>> signal = np.random.randn(1000)
    >>> processor = CrossSignalAnalysis(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np
from vitalDSP.utils.config_utilities.common import (
    pearsonr,
    coherence,
    grangercausalitytests,
)


class CrossSignalAnalysis:
    """
    A comprehensive class to analyze interactions between two physiological signals, such as ECG and respiration.

    The class provides several methods to compute various metrics that describe the relationships between the two signals,
    such as correlation, coherence, phase synchronization, mutual information, and Granger causality.

    Methods
    -------
    compute_correlation : function
        Computes the Pearson correlation coefficient between two signals.
    compute_cross_correlation : function
        Computes the cross-correlation between two signals.
    compute_coherence : function
        Computes the coherence between two signals.
    compute_phase_synchronization : function
        Computes the phase synchronization index between two signals.
    compute_mutual_information : function
        Computes the mutual information between two signals.
    compute_granger_causality : function
        Computes Granger causality between two signals.
    """

    def __init__(self, signal1, signal2, fs=1.0):
        """
        Initialize the CrossSignalAnalysis class with two signals.

        Parameters
        ----------
        signal1 : numpy.ndarray
            The first input signal, such as ECG.
        signal2 : numpy.ndarray
            The second input signal, such as respiration.
        fs : float, optional
            The sampling frequency of the signals. Default is 1.0.

        Notes
        -----
        Both signals should be of the same length and sampled at the same frequency.
        """
        self.signal1 = signal1
        self.signal2 = signal2
        self.fs = fs

    def compute_correlation(self):
        """
        Compute the Pearson correlation coefficient between the two signals.

        The Pearson correlation coefficient is a measure of the linear relationship between two signals.

        Returns
        -------
        correlation_coefficient : float
            The Pearson correlation coefficient between the two signals. A value close to 1 indicates a strong positive correlation,
            while a value close to -1 indicates a strong negative correlation.

        Examples
        --------
        >>> analysis = CrossSignalAnalysis(signal1, signal2)
        >>> corr = analysis.compute_correlation()
        >>> print(corr)
        """
        return pearsonr(self.signal1, self.signal2)

    def compute_cross_correlation(self, max_lag=None):
        """
        Compute the cross-correlation between the two signals.

        Cross-correlation measures the similarity between two signals as a function of the time-lag applied to one of them.
        It is useful for identifying time delays between the signals.

        Parameters
        ----------
        max_lag : int or None, optional
            Maximum lag to compute cross-correlation. If None, the full range is used. Default is None.

        Returns
        -------
        lags : numpy.ndarray
            Array of lags at which the cross-correlation was computed.
        cross_corr : numpy.ndarray
            The cross-correlation between the two signals.

        Examples
        --------
        >>> analysis = CrossSignalAnalysis(signal1, signal2)
        >>> lags, cross_corr = analysis.compute_cross_correlation(max_lag=50)
        >>> print(lags, cross_corr)
        """
        if max_lag is None:
            max_lag = len(self.signal1) - 1
        cross_corr = np.correlate(
            self.signal1 - np.mean(self.signal1),
            self.signal2 - np.mean(self.signal2),
            mode="full",
        )
        lags = np.arange(-max_lag, max_lag + 1)
        return (
            lags,
            cross_corr[
                len(cross_corr) // 2 - max_lag : len(cross_corr) // 2 + max_lag + 1
            ],
        )

    def compute_coherence(self, nperseg=256):
        """
        Compute the coherence between the two signals.

        Coherence is a measure of the degree of synchronization between two signals in the frequency domain.
        It is often used to determine how much one signal corresponds to another at different frequencies.

        Parameters
        ----------
        nperseg : int, optional
            Length of each segment for coherence computation. Default is 256.

        Returns
        -------
        freqs : numpy.ndarray
            Frequency array.
        coherence : numpy.ndarray
            Coherence between the two signals. Values range from 0 (no coherence) to 1 (complete coherence).

        Examples
        --------
        >>> analysis = CrossSignalAnalysis(signal1, signal2)
        >>> freqs, coh = analysis.compute_coherence(nperseg=256)
        >>> print(freqs, coh)
        """
        freqs, coh = coherence(self.signal1, self.signal2, fs=self.fs, nperseg=nperseg)
        return freqs, coh

    def compute_phase_synchronization(self):
        """
        Compute the phase synchronization index between the two signals.

        Phase synchronization measures the degree to which the phases of the two signals are synchronized over time,
        regardless of their amplitude. It is useful for analyzing coupling between oscillatory signals.

        Returns
        -------
        psi : float
            The phase synchronization index (PSI) between the two signals. Values close to 1 indicate strong synchronization.

        Examples
        --------
        >>> analysis = CrossSignalAnalysis(signal1, signal2)
        >>> psi = analysis.compute_phase_synchronization()
        >>> print(psi)
        """
        phase_diff = np.angle(
            np.exp(
                1j
                * (
                    np.unwrap(np.angle(self.signal1))
                    - np.unwrap(np.angle(self.signal2))
                )
            )
        )
        psi = np.abs(np.mean(np.exp(1j * phase_diff)))
        return psi

    def compute_mutual_information(self, bins=10):
        """
        Compute the mutual information between the two signals.

        Mutual information is a measure of the amount of information obtained about one signal through the other.
        It captures both linear and non-linear dependencies between the signals.

        Parameters
        ----------
        bins : int, optional
            Number of bins for histogram estimation. Default is 10.

        Returns
        -------
        mutual_info : float
            The mutual information between the two signals. Higher values indicate greater dependency between the signals.

        Examples
        --------
        >>> analysis = CrossSignalAnalysis(signal1, signal2)
        >>> mi = analysis.compute_mutual_information(bins=10)
        >>> print(mi)
        """
        hist_2d, _, _ = np.histogram2d(self.signal1, self.signal2, bins=bins)
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        px_py = px[:, None] * py[None, :]
        non_zero = pxy > 0
        mutual_info = np.sum(pxy[non_zero] * np.log(pxy[non_zero] / px_py[non_zero]))
        return mutual_info

    def compute_granger_causality(self, max_lag=10):
        """
        Compute the Granger causality between the two signals.

        Granger causality is a statistical hypothesis test to determine whether one time series can predict another.
        It does not imply true causality but can indicate predictive relationships between signals.

        Parameters
        ----------
        max_lag : int, optional
            Maximum lag to consider for causality. Default is 10.

        Returns
        -------
        gc_result : dict
            Granger causality results with keys 'signal1->signal2' and 'signal2->signal1'.
            Each key contains the test statistic and p-value indicating whether one signal Granger-causes the other.

        Examples
        --------
        >>> analysis = CrossSignalAnalysis(signal1, signal2)
        >>> gc_result = analysis.compute_granger_causality(max_lag=10)
        >>> print(gc_result)
        """
        data = np.vstack([self.signal1, self.signal2]).T

        # Signal 1 -> Signal 2
        gc_test_12 = grangercausalitytests(data, max_lag=max_lag, verbose=False)

        # Reversing data to check Signal 2 -> Signal 1
        data_reversed = np.vstack([self.signal2, self.signal1]).T
        gc_test_21 = grangercausalitytests(
            data_reversed, max_lag=max_lag, verbose=False
        )

        # Returning the F-test statistic for the max_lag
        gc_result = {
            "signal1->signal2": gc_test_12[max_lag]["ssr_ftest"],
            "signal2->signal1": gc_test_21[max_lag]["ssr_ftest"],
        }

        return gc_result
