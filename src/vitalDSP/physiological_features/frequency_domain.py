"""
Frequency Domain Features Module for Physiological Signal Processing

This module provides comprehensive frequency-domain feature extraction capabilities
for physiological signals including ECG, PPG, and other vital signs. It implements
Heart Rate Variability (HRV) analysis in the frequency domain with power spectral
density computation and autonomic nervous system assessment.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Power Spectral Density (PSD) computation
- Low-Frequency (LF) and High-Frequency (HF) power analysis
- LF/HF ratio for autonomic balance assessment
- Ultra-Low-Frequency (ULF) and Very-Low-Frequency (VLF) analysis
- Normalized frequency domain metrics (LFnu, HFnu)
- Total power computation across frequency bands
- Comprehensive HRV frequency domain analysis

Examples:
--------
Basic frequency domain analysis:
    >>> import numpy as np
    >>> from vitalDSP.physiological_features.frequency_domain import FrequencyDomainFeatures
    >>> nn_intervals = [800, 810, 790, 805, 795, 820, 780, 815, 800, 810]
    >>> fdf = FrequencyDomainFeatures(nn_intervals, fs=4)
    >>> psd_result = fdf.compute_psd()
    >>> print(f"LF: {psd_result['lf']:.2f}, HF: {psd_result['hf']:.2f}")

Autonomic balance assessment:
    >>> lf_hf_ratio = fdf.compute_lf_hf_ratio()
    >>> lfnu = fdf.compute_lfnu()
    >>> hfnu = fdf.compute_hfnu()
    >>> print(f"LF/HF ratio: {lf_hf_ratio:.2f}, LFnu: {lfnu:.2f}, HFnu: {hfnu:.2f}")

Comprehensive frequency analysis:
    >>> total_power = fdf.compute_total_power()
    >>> ulf = fdf.compute_ulf()
    >>> vlf = fdf.compute_vlf()
    >>> print(f"Total power: {total_power:.2f}, ULF: {ulf:.2f}, VLF: {vlf:.2f}")
"""

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
        compute_ulf(): Computes ULF power (0.0033 to 0.04 Hz).
        compute_vlf(): Computes VLF power (0.0033 to 0.04 Hz).
        compute_total_power(): Computes the total power of ULF, VLF, LF, and HF bands.
        compute_lfnu(): Computes normalized LF power (LFnu).
        compute_hfnu(): Computes normalized HF power (HFnu).
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
        power in ULF, VLF, LF, and HF bands.

        Returns:
            dict: Dictionary containing frequency domain HRV metrics:
                - 'ulf_power': Ultra-Low Frequency power (ms²)
                - 'vlf_power': Very-Low Frequency power (ms²)
                - 'lf_power': Low Frequency power (ms²)
                - 'hf_power': High Frequency power (ms²)
                - 'lf_hf_ratio': LF/HF ratio
                - 'total_power': Total spectral power (ms²)
                - 'frequencies': Frequency array
                - 'psd': Power spectral density array

        Example:
            >>> nn_intervals = [800, 810, 790, 805, 795]
            >>> fdf = FrequencyDomainFeatures(nn_intervals)
            >>> psd_result = fdf.compute_psd()
            >>> print(f"LF: {psd_result['lf_power']}, HF: {psd_result['hf_power']}")
        """
        f, psd = welch(
            self.nn_intervals - np.mean(self.nn_intervals),
            fs=self.fs,
            nperseg=len(self.nn_intervals),
        )

        # Standard HRV frequency bands (Task Force of ESC/NASPE, 1996)
        ulf_band = (0.0, 0.003)  # Ultra Low Frequency
        vlf_band = (0.003, 0.04)  # Very Low Frequency
        lf_band = (0.04, 0.15)  # Low Frequency
        hf_band = (0.15, 0.40)  # High Frequency

        # Integrate the PSD over ULF, VLF, LF, and HF bands
        ulf_power = np.trapz(
            psd[(f >= ulf_band[0]) & (f <= ulf_band[1])],
            f[(f >= ulf_band[0]) & (f <= ulf_band[1])],
        )
        vlf_power = np.trapz(
            psd[(f >= vlf_band[0]) & (f <= vlf_band[1])],
            f[(f >= vlf_band[0]) & (f <= vlf_band[1])],
        )
        lf_power = np.trapz(
            psd[(f >= lf_band[0]) & (f <= lf_band[1])],
            f[(f >= lf_band[0]) & (f <= lf_band[1])],
        )
        hf_power = np.trapz(
            psd[(f >= hf_band[0]) & (f <= hf_band[1])],
            f[(f >= hf_band[0]) & (f <= hf_band[1])],
        )

        # Calculate LF/HF ratio
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0

        # Calculate total power
        total_power = ulf_power + vlf_power + lf_power + hf_power

        # Return as dictionary for API compatibility
        return {
            "ulf_power": ulf_power,
            "vlf_power": vlf_power,
            "lf_power": lf_power,
            "hf_power": hf_power,
            "lf_hf_ratio": lf_hf_ratio,
            "total_power": total_power,
            "frequencies": f,
            "psd": psd,
        }

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
        psd_result = self.compute_psd()
        return psd_result["lf_power"]

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
        psd_result = self.compute_psd()
        return psd_result["hf_power"]

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
        psd_result = self.compute_psd()
        return psd_result["lf_hf_ratio"]

    def compute_ulf(self):
        """
        Computes the Ultra-Low Frequency (ULF) power from the PSD.

        Returns:
            float: The ULF power.

        Example:
            >>> nn_intervals = [800, 810, 790, 805, 795]
            >>> fdf = FrequencyDomainFeatures(nn_intervals)
            >>> ulf = fdf.compute_ulf()
            >>> print(f"ULF: {ulf}")
        """
        psd_result = self.compute_psd()
        return psd_result["ulf_power"]

    def compute_vlf(self):
        """
        Computes the Very Low Frequency (VLF) power from the PSD.

        Returns:
            float: The VLF power.

        Example:
            >>> nn_intervals = [800, 810, 790, 805, 795]
            >>> fdf = FrequencyDomainFeatures(nn_intervals)
            >>> vlf = fdf.compute_vlf()
            >>> print(f"VLF: {vlf}")
        """
        psd_result = self.compute_psd()
        return psd_result["vlf_power"]

    def compute_total_power(self):
        """
        Computes the total power, which is the sum of power in ULF, VLF, LF, and HF bands.

        Returns:
            float: The total power.

        Example:
            >>> nn_intervals = [800, 810, 790, 805, 795]
            >>> fdf = FrequencyDomainFeatures(nn_intervals)
            >>> total_power = fdf.compute_total_power()
            >>> print(f"Total Power: {total_power}")
        """
        psd_result = self.compute_psd()
        return psd_result["total_power"]

    def compute_lfnu(self):
        """
        Computes the normalized Low-Frequency (LFnu) power as a percentage of total power (LF + HF).

        Returns:
            float: The LFnu value.

        Example:
            >>> nn_intervals = [800, 810, 790, 805, 795]
            >>> fdf = FrequencyDomainFeatures(nn_intervals)
            >>> lfnu = fdf.compute_lfnu()
            >>> print(f"LFnu: {lfnu}")
        """
        psd_result = self.compute_psd()
        lf = psd_result["lf_power"]
        hf = psd_result["hf_power"]
        total_lf_hf = lf + hf
        if total_lf_hf == 0:
            return 0
        return 100.0 * lf / total_lf_hf

    def compute_hfnu(self):
        """
        Computes the normalized High-Frequency (HFnu) power as a percentage of total power (LF + HF).

        Returns:
            float: The HFnu value.

        Example:
            >>> nn_intervals = [800, 810, 790, 805, 795]
            >>> fdf = FrequencyDomainFeatures(nn_intervals)
            >>> hfnu = fdf.compute_hfnu()
            >>> print(f"HFnu: {hfnu}")
        """
        psd_result = self.compute_psd()
        lf = psd_result["lf_power"]
        hf = psd_result["hf_power"]
        total_lf_hf = lf + hf
        if total_lf_hf == 0:
            return 0
        return 100.0 * hf / total_lf_hf
