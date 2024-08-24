import numpy as np
from utils.common import pearsonr, coherence, grangercausalitytests


class CrossSignalAnalysis:
    """
    A comprehensive class to analyze interactions between two physiological signals, such as ECG and respiration.

    Methods
    -------
    compute_correlation : function
        Computes the correlation between two signals.
    compute_cross_correlation : function
        Computes the cross-correlation between two signals.
    compute_coherence : function
        Computes the coherence between two signals.
    compute_phase_synchronization : function
        Computes phase synchronization between two signals.
    compute_mutual_information : function
        Computes mutual information between two signals.
    compute_granger_causality : function
        Computes Granger causality between two signals.
    """

    def __init__(self, signal1, signal2, fs=1.0):
        """
        Initialize the CrossSignalAnalysis class with two signals.

        Parameters
        ----------
        signal1 : numpy.ndarray
            The first input signal (e.g., ECG).
        signal2 : numpy.ndarray
            The second input signal (e.g., respiration).
        fs : float, optional
            The sampling frequency of the signals. Default is 1.0.
        """
        self.signal1 = signal1
        self.signal2 = signal2
        self.fs = fs

    def compute_correlation(self):
        """
        Compute the Pearson correlation coefficient between the two signals.

        Returns
        -------
        correlation_coefficient : float
            The Pearson correlation coefficient between the two signals.

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

        Parameters
        ----------
        nperseg : int, optional
            Length of each segment for coherence computation. Default is 256.

        Returns
        -------
        freqs : numpy.ndarray
            Frequency array.
        coherence : numpy.ndarray
            Coherence between the two signals.

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

        Returns
        -------
        psi : float
            The phase synchronization index between the two signals.

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

        Parameters
        ----------
        bins : int, optional
            Number of bins for histogram estimation. Default is 10.

        Returns
        -------
        mutual_info : float
            The mutual information between the two signals.

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

        Parameters
        ----------
        max_lag : int, optional
            Maximum lag to consider for causality. Default is 10.

        Returns
        -------
        gc_result : dict
            Granger causality results with keys 'signal1->signal2' and 'signal2->signal1'.

        Examples
        --------
        >>> analysis = CrossSignalAnalysis(signal1, signal2)
        >>> gc_result = analysis.compute_granger_causality(max_lag=10)
        >>> print(gc_result)
        """
        data = np.vstack([self.signal1, self.signal2]).T
        gc_test = grangercausalitytests(data, max_lag, verbose=False)
        gc_result = {
            "signal1->signal2": gc_test[max_lag]["ssr_ftest"],
            "signal2->signal1": gc_test[max_lag]["ssr_ftest"],
        }
        return gc_result
