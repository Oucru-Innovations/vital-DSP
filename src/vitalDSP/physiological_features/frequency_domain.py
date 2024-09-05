import numpy as np
from scipy.signal import welch


class FrequencyDomainFeatures:
    """
    A class for computing frequency-domain features from physiological signals (ECG, PPG).

    Attributes:
        nn_intervals (list or np.array): The NN intervals (in milliseconds) between heartbeats.
        fs (int): The sampling frequency (Hz). Defaults to 4 Hz, typically used for HRV analysis.

    Methods:
        compute_psd(): Computes the power spectral density (PSD) for HRV and returns LF, HF, and LF/HF ratio.
        compute_lf(): Computes the Low-Frequency (LF) power from the PSD.
        compute_hf(): Computes the High-Frequency (HF) power from the PSD.
        compute_lf_hf_ratio(): Computes the ratio of LF to HF power (LF/HF ratio).
    """

    def __init__(self, nn_intervals, fs=4):
        """
        Initializes the FrequencyDomainFeatures object with NN intervals and a sampling frequency.

        Args:
            nn_intervals (list or np.array): The NN intervals (in milliseconds) between heartbeats.
            fs (int): The sampling frequency (Hz) for HRV analysis. Default is 4 Hz.
        """
        if len(nn_intervals) == 0:
            raise ValueError("nn_intervals cannot be empty")
        if np.all(np.array(nn_intervals) == 0):
            raise ValueError("nn_intervals cannot contain all zeros")

        self.nn_intervals = np.array(nn_intervals)
        self.fs = fs

    def compute_psd(self):
        """
        Computes the Power Spectral Density (PSD) using Welch's method to estimate the
        power in Low-Frequency (LF) and High-Frequency (HF) bands.

        Returns:
            tuple: Low-Frequency (LF) power and High-Frequency (HF) power.

        Example:
            >>> nn_intervals = [800, 810, 790, 805, 795]
            >>> fdf = FrequencyDomainFeatures(nn_intervals)
            >>> lf, hf = fdf.compute_psd()
            >>> print(f"LF: {lf}, HF: {hf}")
        """
        """Computes the Power Spectral Density (PSD) using Welch's method."""
        # Detrend the NN intervals and compute the PSD
        f, psd = welch(
            self.nn_intervals - np.mean(self.nn_intervals),
            fs=self.fs,
            nperseg=len(self.nn_intervals),
        )

        # Define frequency bands for HRV analysis relative to the sampling frequency
        lf_band = (0.04, 0.15)  # Low Frequency
        hf_band = (0.15, 0.40)  # High Frequency

        # Integrate the PSD over LF and HF bands
        lf_power = np.trapz(
            psd[(f >= lf_band[0]) & (f <= lf_band[1])],
            f[(f >= lf_band[0]) & (f <= lf_band[1])],
        )
        hf_power = np.trapz(
            psd[(f >= hf_band[0]) & (f <= hf_band[1])],
            f[(f >= hf_band[0]) & (f <= hf_band[1])],
        )

        return lf_power, hf_power

    def compute_lf(self):
        """
        Computes the Low-Frequency (LF) power from the PSD.

        Returns:
            float: The LF power.

        Example:
            >>> nn_intervals = [800, 810, 790, 805, 795]
            >>> fdf = FrequencyDomainFeatures(nn_intervals)
            >>> lf = fdf.compute_lf()
            >>> print(f"LF: {lf}")
        """
        lf, _ = self.compute_psd()
        return lf

    def compute_hf(self):
        """
        Computes the High-Frequency (HF) power from the PSD.

        Returns:
            float: The HF power.

        Example:
            >>> nn_intervals = [800, 810, 790, 805, 795]
            >>> fdf = FrequencyDomainFeatures(nn_intervals)
            >>> hf = fdf.compute_hf()
            >>> print(f"HF: {hf}")
        """
        _, hf = self.compute_psd()
        return hf

    def compute_lf_hf_ratio(self):
        """
        Computes the ratio of LF to HF power (LF/HF ratio).

        Returns:
            float: The LF/HF ratio.

        Example:
            >>> nn_intervals = [800, 810, 790, 805, 795]
            >>> fdf = FrequencyDomainFeatures(nn_intervals)
            >>> lf_hf_ratio = fdf.compute_lf_hf_ratio()
            >>> print(f"LF/HF Ratio: {lf_hf_ratio}")
        """
        lf, hf = self.compute_psd()
        if hf == 0:
            return np.inf  # Avoid division by zero
        return lf / hf
