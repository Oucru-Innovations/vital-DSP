"""
Health Report Feature Extraction Module

This module extracts comprehensive health features from physiological signals
for use with vitalDSP's HealthReportGenerator.

Features extracted:
- HRV features (time domain and frequency domain)
- Respiratory features
- Time domain features
- Frequency domain features
- Statistical features
- Morphological features (if applicable)
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
import scipy.stats
import scipy.signal

logger = logging.getLogger(__name__)


class HealthFeatureExtractor:
    """Extract health features from physiological signals."""

    def __init__(self, sampling_frequency: float, signal_type: str = "ECG"):
        """
        Initialize feature extractor.

        Args:
            sampling_frequency: Sampling frequency in Hz
            signal_type: Type of signal ("ECG", "PPG", "RESP", or "Unknown")
        """
        self.fs = sampling_frequency
        self.signal_type = signal_type.upper()
        logger.info(
            f"Initialized HealthFeatureExtractor for {signal_type} at {sampling_frequency} Hz"
        )

    def extract_all_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract all available health features from signal.

        Args:
            signal_data: 1D array of signal values

        Returns:
            Dictionary of feature names and values for HealthReportGenerator
        """
        logger.info(
            f"Extracting health features from {len(signal_data)} samples ({len(signal_data)/self.fs:.1f}s)"
        )

        features = {}

        try:
            # Time domain features (always applicable)
            features.update(self._extract_time_domain_features(signal_data))

            # Statistical features (always applicable)
            features.update(self._extract_statistical_features(signal_data))

            # Frequency domain features (always applicable)
            features.update(self._extract_frequency_features(signal_data))

            # Signal-specific features
            if self.signal_type in ["ECG", "PPG"]:
                # Extract peaks for cardiac signals
                peaks = self._detect_peaks(signal_data)

                if len(peaks) >= 5:  # Need at least 5 peaks for HRV
                    features.update(self._extract_hrv_features(peaks))
                    features.update(self._extract_heart_rate_features(peaks))
                else:
                    logger.warning(
                        f"Only {len(peaks)} peaks detected, skipping HRV features (need ≥5)"
                    )

            if self.signal_type in ["RESP", "ECG", "PPG"]:
                # Try to extract respiratory features
                try:
                    features.update(self._extract_respiratory_features(signal_data))
                except Exception as e:
                    logger.warning(f"Could not extract respiratory features: {e}")

            logger.info(f"Successfully extracted {len(features)} health features")
            return features

        except Exception as e:
            logger.error(f"Error extracting health features: {e}", exc_info=True)
            # Return basic features if extraction fails
            return {
                "signal_mean": float(np.mean(signal_data)),
                "signal_std": float(np.std(signal_data)),
                "signal_length_seconds": len(signal_data) / self.fs,
            }

    def _extract_time_domain_features(
        self, signal_data: np.ndarray
    ) -> Dict[str, float]:
        """Extract time domain features from signal."""
        features = {}

        try:
            features["signal_mean"] = float(np.mean(signal_data))
            features["signal_std"] = float(np.std(signal_data))
            features["signal_min"] = float(np.min(signal_data))
            features["signal_max"] = float(np.max(signal_data))
            features["signal_range"] = float(np.ptp(signal_data))
            features["signal_rms"] = float(np.sqrt(np.mean(signal_data**2)))

            # Percentiles
            features["signal_p25"] = float(np.percentile(signal_data, 25))
            features["signal_p50"] = float(np.percentile(signal_data, 50))  # median
            features["signal_p75"] = float(np.percentile(signal_data, 75))
            features["signal_iqr"] = features["signal_p75"] - features["signal_p25"]

            # Peak-to-peak amplitude
            features["signal_ptp"] = float(np.ptp(signal_data))

            logger.debug(f"Extracted {len(features)} time domain features")
            return features

        except Exception as e:
            logger.error(f"Error in time domain feature extraction: {e}")
            return {}

    def _extract_statistical_features(
        self, signal_data: np.ndarray
    ) -> Dict[str, float]:
        """Extract statistical features from signal."""
        features = {}

        try:
            features["skewness"] = float(scipy.stats.skew(signal_data))
            features["kurtosis"] = float(scipy.stats.kurtosis(signal_data))

            # Coefficient of variation
            mean_val = np.mean(signal_data)
            if mean_val != 0:
                features["coefficient_of_variation"] = float(
                    np.std(signal_data) / mean_val
                )

            # Sample entropy (simplified version)
            try:
                features["sample_entropy"] = float(
                    self._calculate_sample_entropy(signal_data)
                )
            except Exception as e:
                logger.error(f"Error calculating sample entropy: {e}")
                pass

            logger.debug(f"Extracted {len(features)} statistical features")
            return features

        except Exception as e:
            logger.error(f"Error in statistical feature extraction: {e}")
            return {}

    def _extract_frequency_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features from signal."""
        features = {}

        try:
            # Compute power spectral density
            freqs, psd = scipy.signal.welch(
                signal_data, fs=self.fs, nperseg=min(256, len(signal_data) // 4)
            )

            # Dominant frequency
            dominant_idx = np.argmax(psd)
            features["dominant_frequency"] = float(freqs[dominant_idx])
            features["dominant_power"] = float(psd[dominant_idx])

            # Spectral centroid
            features["spectral_centroid"] = float(np.sum(freqs * psd) / np.sum(psd))

            # Total power
            features["total_power"] = float(np.sum(psd))

            # Power in frequency bands (for cardiac signals)
            if self.signal_type in ["ECG", "PPG"]:
                # VLF: 0.003-0.04 Hz, LF: 0.04-0.15 Hz, HF: 0.15-0.4 Hz
                vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
                lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                hf_mask = (freqs >= 0.15) & (freqs < 0.4)

                features["power_vlf"] = (
                    float(np.sum(psd[vlf_mask])) if np.any(vlf_mask) else 0.0
                )
                features["power_lf"] = (
                    float(np.sum(psd[lf_mask])) if np.any(lf_mask) else 0.0
                )
                features["power_hf"] = (
                    float(np.sum(psd[hf_mask])) if np.any(hf_mask) else 0.0
                )

                if features["power_hf"] > 0:
                    features["lf_hf_ratio"] = (
                        features["power_lf"] / features["power_hf"]
                    )

            logger.debug(f"Extracted {len(features)} frequency domain features")
            return features

        except Exception as e:
            logger.error(f"Error in frequency domain feature extraction: {e}")
            return {}

    def _detect_peaks(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Detect peaks in cardiac signals (R-peaks for ECG, systolic peaks for PPG).

        Args:
            signal_data: 1D array of signal values

        Returns:
            Array of peak indices
        """
        try:
            # Normalize signal
            signal_norm = (signal_data - np.mean(signal_data)) / np.std(signal_data)

            # Calculate minimum distance between peaks (based on max heart rate of 200 BPM)
            min_distance = int(self.fs * 60 / 200)  # samples

            # Find peaks with adaptive threshold
            peaks, properties = scipy.signal.find_peaks(
                signal_norm,
                distance=min_distance,
                prominence=0.5,  # Require peaks to be prominent
                height=0.5,  # Require peaks to be above threshold
            )

            logger.debug(f"Detected {len(peaks)} peaks in signal")
            return peaks

        except Exception as e:
            logger.error(f"Error in peak detection: {e}")
            return np.array([])

    def _extract_hrv_features(self, peaks: np.ndarray) -> Dict[str, float]:
        """
        Extract HRV (Heart Rate Variability) features from peaks.

        Args:
            peaks: Array of peak indices

        Returns:
            Dictionary of HRV features
        """
        features = {}

        try:
            if len(peaks) < 5:
                logger.warning("Insufficient peaks for HRV analysis")
                return features

            # Calculate RR intervals (in milliseconds)
            rr_intervals = np.diff(peaks) / self.fs * 1000  # ms

            # Remove outliers (RR intervals outside 300-2000 ms)
            valid_mask = (rr_intervals >= 300) & (rr_intervals <= 2000)
            rr_intervals = rr_intervals[valid_mask]

            if len(rr_intervals) < 4:
                logger.warning("Insufficient valid RR intervals for HRV")
                return features

            # Time domain HRV features
            features["hrv_mean_rr"] = float(np.mean(rr_intervals))
            features["hrv_sdnn"] = float(np.std(rr_intervals, ddof=1))

            # RMSSD - Root mean square of successive differences
            rr_diff = np.diff(rr_intervals)
            features["hrv_rmssd"] = float(np.sqrt(np.mean(rr_diff**2)))

            # NN50 and pNN50
            nn50_count = np.sum(np.abs(rr_diff) > 50)
            features["hrv_nn50"] = float(nn50_count)
            features["hrv_pnn50"] = (
                float(100 * nn50_count / len(rr_diff)) if len(rr_diff) > 0 else 0.0
            )

            # SDSD - Standard deviation of successive differences
            features["hrv_sdsd"] = (
                float(np.std(rr_diff, ddof=1)) if len(rr_diff) > 1 else 0.0
            )

            # Coefficient of variation
            if features["hrv_mean_rr"] > 0:
                features["hrv_cv"] = float(
                    features["hrv_sdnn"] / features["hrv_mean_rr"]
                )

            logger.debug(f"Extracted {len(features)} HRV features")
            return features

        except Exception as e:
            logger.error(f"Error in HRV feature extraction: {e}")
            return {}

    def _extract_heart_rate_features(self, peaks: np.ndarray) -> Dict[str, float]:
        """
        Extract heart rate features from peaks.

        Args:
            peaks: Array of peak indices

        Returns:
            Dictionary of heart rate features
        """
        features = {}

        try:
            if len(peaks) < 2:
                return features

            # Calculate RR intervals
            rr_intervals = np.diff(peaks) / self.fs * 1000  # ms

            # Remove outliers
            valid_mask = (rr_intervals >= 300) & (rr_intervals <= 2000)
            rr_intervals = rr_intervals[valid_mask]

            if len(rr_intervals) == 0:
                return features

            # Convert to heart rate (BPM)
            heart_rates = 60000 / rr_intervals

            features["mean_hr"] = float(np.mean(heart_rates))
            features["std_hr"] = float(np.std(heart_rates))
            features["min_hr"] = float(np.min(heart_rates))
            features["max_hr"] = float(np.max(heart_rates))
            features["range_hr"] = features["max_hr"] - features["min_hr"]

            logger.debug(f"Extracted {len(features)} heart rate features")
            return features

        except Exception as e:
            logger.error(f"Error in heart rate feature extraction: {e}")
            return {}

    def _extract_respiratory_features(
        self, signal_data: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract respiratory features from signal.

        Args:
            signal_data: 1D array of signal values

        Returns:
            Dictionary of respiratory features
        """
        features = {}

        try:
            # For ECG/PPG: Extract respiratory component using baseline wander
            if self.signal_type in ["ECG", "PPG"]:
                # Low-pass filter to extract baseline (respiratory component)
                sos = scipy.signal.butter(4, 0.5, btype="low", fs=self.fs, output="sos")
                respiratory_signal = scipy.signal.sosfilt(sos, signal_data)
            else:
                respiratory_signal = signal_data

            # Detect respiratory peaks (inhalations)
            min_distance = int(self.fs * 60 / 40)  # Min 40 breaths/min
            peaks, _ = scipy.signal.find_peaks(
                respiratory_signal,
                distance=min_distance,
                prominence=np.std(respiratory_signal) * 0.3,
            )

            if len(peaks) >= 2:
                # Calculate respiratory rate
                breath_intervals = np.diff(peaks) / self.fs  # seconds
                respiratory_rate = 60 / np.mean(breath_intervals)  # breaths per minute

                features["respiratory_rate"] = float(respiratory_rate)
                features["respiratory_rate_std"] = float(np.std(60 / breath_intervals))

                # Breathing regularity (coefficient of variation)
                features["breathing_regularity"] = float(
                    np.std(breath_intervals) / np.mean(breath_intervals)
                )

                logger.debug(
                    f"Extracted respiratory rate: {respiratory_rate:.1f} breaths/min"
                )

            return features

        except Exception as e:
            logger.error(f"Error in respiratory feature extraction: {e}")
            return {}

    def _calculate_sample_entropy(
        self, signal_data: np.ndarray, m: int = 2, r: float = None
    ) -> float:
        """
        Calculate sample entropy of signal (simplified version).

        Args:
            signal_data: 1D array of signal values
            m: Embedding dimension (default 2)
            r: Tolerance (default 0.2 * std)

        Returns:
            Sample entropy value
        """
        if r is None:
            r = 0.2 * np.std(signal_data)

        N = len(signal_data)

        # Count matches for m and m+1
        def count_matches(m_val):
            count = 0
            for i in range(N - m_val):
                template = signal_data[i : i + m_val]
                for j in range(i + 1, N - m_val):
                    if np.max(np.abs(template - signal_data[j : j + m_val])) <= r:
                        count += 1
            return count

        A = count_matches(m)
        B = count_matches(m + 1)

        if A == 0 or B == 0:
            return 0.0

        return -np.log(B / A)


def extract_health_features_from_data(
    signal_data: np.ndarray, sampling_frequency: float, signal_type: str = "ECG"
) -> Dict[str, float]:
    """
    Convenience function to extract health features from signal data.

    Args:
        signal_data: 1D array of signal values
        sampling_frequency: Sampling frequency in Hz
        signal_type: Type of signal ("ECG", "PPG", "RESP", or "Unknown")

    Returns:
        Dictionary of feature names and values for HealthReportGenerator
    """
    extractor = HealthFeatureExtractor(sampling_frequency, signal_type)
    return extractor.extract_all_features(signal_data)
