"""
vitalDSP Machine Learning Feature Extraction Module

Comprehensive feature extraction pipeline for machine learning applications
with physiological signals (ECG, PPG, EEG, respiratory signals).

This module provides:
- Automated feature extraction from multiple domains
- Feature engineering and selection
- Dimensionality reduction
- Pipeline integration with scikit-learn
- Feature importance analysis

Author: vitalDSP Team
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Comprehensive feature extraction for physiological signals.

    Extracts features from multiple domains:
    - Time domain (statistical features)
    - Frequency domain (spectral features)
    - Time-frequency domain (wavelet, STFT)
    - Nonlinear dynamics (entropy, fractal dimension)
    - Morphological features (peaks, waveform shape)

    Compatible with scikit-learn pipelines.

    Parameters
    ----------
    signal_type : str, default='ecg'
        Type of physiological signal ('ecg', 'ppg', 'eeg', 'resp')
    sampling_rate : float
        Sampling rate of the signal in Hz
    domains : list of str, default=['time', 'frequency', 'nonlinear']
        Feature domains to extract from
    normalize : bool, default=True
        Whether to normalize features
    feature_selection : str, optional
        Feature selection method ('kbest', 'mutual_info', 'pca', None)
    n_features : int, optional
        Number of features to select (if feature_selection is not None)

    Attributes
    ----------
    feature_names_ : list
        Names of extracted features
    n_features_in_ : int
        Number of input samples
    feature_importances_ : dict
        Feature importance scores (if available)

    Examples
    --------
    >>> from vitalDSP.ml_models import FeatureExtractor
    >>> extractor = FeatureExtractor(signal_type='ecg', sampling_rate=250)
    >>> features = extractor.fit_transform(signals)
    >>> print(f"Extracted {features.shape[1]} features")

    >>> # Use in scikit-learn pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> pipeline = Pipeline([
    ...     ('features', FeatureExtractor(signal_type='ecg', sampling_rate=250)),
    ...     ('classifier', RandomForestClassifier())
    ... ])
    >>> pipeline.fit(X_train, y_train)
    """

    def __init__(
        self,
        signal_type: str = "ecg",
        sampling_rate: float = 250.0,
        domains: Optional[List[str]] = None,
        normalize: bool = True,
        feature_selection: Optional[str] = None,
        n_features: Optional[int] = None,
        **kwargs,
    ):
        self.signal_type = signal_type.lower()
        self.sampling_rate = sampling_rate
        self.domains = domains or ["time", "frequency", "nonlinear"]
        self.normalize = normalize
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.kwargs = kwargs

        # Will be set during fit
        self.feature_names_ = []
        self.scaler_ = None
        self.selector_ = None
        self.feature_importances_ = {}

    def fit(
        self, X: Union[np.ndarray, List[np.ndarray]], y: Optional[np.ndarray] = None
    ):
        """
        Fit the feature extractor.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_timesteps)
            Training signals
        y : array-like of shape (n_samples,), optional
            Target values for supervised feature selection

        Returns
        -------
        self : object
            Fitted transformer
        """
        # Extract features from training data
        X_features = self._extract_features(X)

        # Initialize scaler
        if self.normalize:
            self.scaler_ = RobustScaler()
            X_features = self.scaler_.fit_transform(X_features)

        # Initialize feature selector
        if self.feature_selection and self.n_features:
            if self.feature_selection == "kbest":
                self.selector_ = SelectKBest(f_classif, k=self.n_features)
            elif self.feature_selection == "mutual_info":
                self.selector_ = SelectKBest(mutual_info_classif, k=self.n_features)
            elif self.feature_selection == "pca":
                self.selector_ = PCA(n_components=self.n_features)

            if self.selector_ and y is not None:
                self.selector_.fit(X_features, y)

        return self

    def transform(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Transform signals to feature matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_timesteps)
            Input signals

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Extracted features
        """
        # Extract features
        X_features = self._extract_features(X)

        # Normalize
        if self.normalize and self.scaler_:
            X_features = self.scaler_.transform(X_features)

        # Feature selection
        if self.selector_:
            X_features = self.selector_.transform(X_features)

        return X_features

    def _extract_features(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Extract all features from signals."""
        # Convert to list of signals if necessary
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                signals = [X]
            elif X.ndim == 2:
                signals = [X[i] for i in range(len(X))]
            else:
                raise ValueError(f"Invalid input shape: {X.shape}")
        else:
            signals = X

        # Extract features for each signal
        all_features = []
        feature_names = []

        for signal in signals:
            features = {}

            # Time domain features
            if "time" in self.domains:
                time_features = self._extract_time_domain(signal)
                features.update(time_features)

            # Frequency domain features
            if "frequency" in self.domains:
                freq_features = self._extract_frequency_domain(signal)
                features.update(freq_features)

            # Nonlinear features
            if "nonlinear" in self.domains:
                nonlinear_features = self._extract_nonlinear_features(signal)
                features.update(nonlinear_features)

            # Morphological features
            if "morphology" in self.domains:
                morph_features = self._extract_morphological_features(signal)
                features.update(morph_features)

            # Time-frequency features
            if "time_frequency" in self.domains:
                tf_features = self._extract_time_frequency_features(signal)
                features.update(tf_features)

            all_features.append(list(features.values()))

            # Store feature names (only once)
            if not feature_names:
                feature_names = list(features.keys())

        self.feature_names_ = feature_names
        return np.array(all_features)

    def _extract_time_domain(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract time domain statistical features."""
        features = {}

        try:
            # Basic statistics
            features["mean"] = np.mean(signal)
            features["std"] = np.std(signal)
            features["var"] = np.var(signal)
            features["min"] = np.min(signal)
            features["max"] = np.max(signal)
            features["range"] = features["max"] - features["min"]
            features["median"] = np.median(signal)

            # Percentiles
            features["q25"] = np.percentile(signal, 25)
            features["q75"] = np.percentile(signal, 75)
            features["iqr"] = features["q75"] - features["q25"]

            # Higher order moments
            features["skewness"] = self._skewness(signal)
            features["kurtosis"] = self._kurtosis(signal)

            # Signal energy and power
            features["energy"] = np.sum(signal**2)
            features["power"] = np.mean(signal**2)
            features["rms"] = np.sqrt(features["power"])

            # Zero crossings
            features["zero_crossings"] = np.sum(np.diff(np.sign(signal)) != 0)

            # Mean absolute deviation
            features["mad"] = np.mean(np.abs(signal - features["mean"]))

            # Coefficient of variation
            if features["mean"] != 0:
                features["cv"] = features["std"] / abs(features["mean"])
            else:
                features["cv"] = 0

            # Peak-to-peak amplitude
            features["ptp"] = np.ptp(signal)

            # Signal changes
            diff_signal = np.diff(signal)
            features["mean_diff"] = np.mean(np.abs(diff_signal))
            features["std_diff"] = np.std(diff_signal)

        except Exception as e:
            warnings.warn(f"Error extracting time domain features: {e}")
            # Return zeros for failed features
            for key in [
                "mean",
                "std",
                "var",
                "min",
                "max",
                "range",
                "median",
                "q25",
                "q75",
                "iqr",
                "skewness",
                "kurtosis",
                "energy",
                "power",
                "rms",
                "zero_crossings",
                "mad",
                "cv",
                "ptp",
                "mean_diff",
                "std_diff",
            ]:
                features.setdefault(key, 0.0)

        return features

    def _extract_frequency_domain(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features using FFT."""
        features = {}

        try:
            # Compute FFT
            fft_vals = np.fft.rfft(signal)
            fft_freq = np.fft.rfftfreq(len(signal), 1.0 / self.sampling_rate)
            psd = np.abs(fft_vals) ** 2

            # Total power
            features["total_power"] = np.sum(psd)

            # Frequency bands (ECG/PPG specific)
            if self.signal_type in ["ecg", "ppg"]:
                # HRV frequency bands
                vlf_band = (fft_freq >= 0.003) & (fft_freq < 0.04)
                lf_band = (fft_freq >= 0.04) & (fft_freq < 0.15)
                hf_band = (fft_freq >= 0.15) & (fft_freq < 0.4)

                features["vlf_power"] = np.sum(psd[vlf_band])
                features["lf_power"] = np.sum(psd[lf_band])
                features["hf_power"] = np.sum(psd[hf_band])

                # LF/HF ratio
                if features["hf_power"] > 0:
                    features["lf_hf_ratio"] = (
                        features["lf_power"] / features["hf_power"]
                    )
                else:
                    features["lf_hf_ratio"] = 0

                # Normalized powers
                total_power = (
                    features["vlf_power"] + features["lf_power"] + features["hf_power"]
                )
                if total_power > 0:
                    features["lf_norm"] = features["lf_power"] / total_power
                    features["hf_norm"] = features["hf_power"] / total_power
                else:
                    features["lf_norm"] = 0
                    features["hf_norm"] = 0

            # Spectral statistics
            features["spectral_mean"] = (
                np.sum(fft_freq * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
            )
            features["spectral_std"] = (
                np.sqrt(
                    np.sum((fft_freq - features["spectral_mean"]) ** 2 * psd)
                    / np.sum(psd)
                )
                if np.sum(psd) > 0
                else 0
            )

            # Spectral entropy
            psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
            psd_norm = psd_norm[psd_norm > 0]
            features["spectral_entropy"] = -np.sum(psd_norm * np.log2(psd_norm))

            # Peak frequency
            features["peak_frequency"] = fft_freq[np.argmax(psd)]

            # Bandwidth
            cumsum_psd = np.cumsum(psd)
            total = cumsum_psd[-1]
            if total > 0:
                f_low = fft_freq[np.argmax(cumsum_psd >= 0.05 * total)]
                f_high = fft_freq[np.argmax(cumsum_psd >= 0.95 * total)]
                features["bandwidth"] = f_high - f_low
            else:
                features["bandwidth"] = 0

        except Exception as e:
            warnings.warn(f"Error extracting frequency domain features: {e}")
            # Return zeros for failed features
            keys = [
                "total_power",
                "spectral_mean",
                "spectral_std",
                "spectral_entropy",
                "peak_frequency",
                "bandwidth",
            ]
            if self.signal_type in ["ecg", "ppg"]:
                keys.extend(
                    [
                        "vlf_power",
                        "lf_power",
                        "hf_power",
                        "lf_hf_ratio",
                        "lf_norm",
                        "hf_norm",
                    ]
                )
            for key in keys:
                features.setdefault(key, 0.0)

        return features

    def _extract_nonlinear_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract nonlinear dynamics features."""
        features = {}

        try:
            # Sample Entropy (simplified)
            features["sample_entropy"] = self._sample_entropy(
                signal, m=2, r=0.2 * np.std(signal)
            )

            # Approximate Entropy
            features["approximate_entropy"] = self._approximate_entropy(
                signal, m=2, r=0.2 * np.std(signal)
            )

            # Detrended Fluctuation Analysis (simplified)
            features["dfa_alpha"] = self._dfa(signal)

            # Hurst exponent
            features["hurst_exponent"] = self._hurst_exponent(signal)

            # Lyapunov exponent (simplified)
            features["lyapunov"] = self._lyapunov_exponent(signal)

            # Correlation dimension
            features["correlation_dim"] = self._correlation_dimension(signal)

        except Exception as e:
            warnings.warn(f"Error extracting nonlinear features: {e}")
            for key in [
                "sample_entropy",
                "approximate_entropy",
                "dfa_alpha",
                "hurst_exponent",
                "lyapunov",
                "correlation_dim",
            ]:
                features.setdefault(key, 0.0)

        return features

    def _extract_morphological_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract morphological/shape features."""
        features = {}

        try:
            # Find peaks
            from scipy.signal import find_peaks

            peaks, properties = find_peaks(
                signal, distance=int(0.5 * self.sampling_rate)
            )

            features["n_peaks"] = len(peaks)

            if len(peaks) > 0:
                # Peak statistics
                peak_amplitudes = signal[peaks]
                features["mean_peak_amplitude"] = np.mean(peak_amplitudes)
                features["std_peak_amplitude"] = np.std(peak_amplitudes)

                # Inter-peak intervals
                if len(peaks) > 1:
                    ipi = np.diff(peaks) / self.sampling_rate
                    features["mean_ipi"] = np.mean(ipi)
                    features["std_ipi"] = np.std(ipi)
                    features["rmssd_ipi"] = np.sqrt(np.mean(np.diff(ipi) ** 2))
                else:
                    features["mean_ipi"] = 0
                    features["std_ipi"] = 0
                    features["rmssd_ipi"] = 0
            else:
                features["mean_peak_amplitude"] = 0
                features["std_peak_amplitude"] = 0
                features["mean_ipi"] = 0
                features["std_ipi"] = 0
                features["rmssd_ipi"] = 0

            # Signal slope features
            slopes = np.diff(signal)
            features["mean_slope"] = np.mean(np.abs(slopes))
            features["max_slope"] = np.max(np.abs(slopes))

        except Exception as e:
            warnings.warn(f"Error extracting morphological features: {e}")
            for key in [
                "n_peaks",
                "mean_peak_amplitude",
                "std_peak_amplitude",
                "mean_ipi",
                "std_ipi",
                "rmssd_ipi",
                "mean_slope",
                "max_slope",
            ]:
                features.setdefault(key, 0.0)

        return features

    def _extract_time_frequency_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract time-frequency domain features using STFT."""
        features = {}

        try:
            from scipy import signal as sp_signal

            # Short-Time Fourier Transform
            nperseg = min(256, len(signal) // 4)
            f, t, Zxx = sp_signal.stft(signal, self.sampling_rate, nperseg=nperseg)

            # Spectrogram statistics
            spectrogram = np.abs(Zxx)
            features["stft_mean"] = np.mean(spectrogram)
            features["stft_std"] = np.std(spectrogram)
            features["stft_max"] = np.max(spectrogram)

            # Spectral flux (change over time)
            spec_flux = np.sqrt(np.sum(np.diff(spectrogram, axis=1) ** 2, axis=0))
            features["spectral_flux_mean"] = np.mean(spec_flux)
            features["spectral_flux_std"] = np.std(spec_flux)

        except Exception as e:
            warnings.warn(f"Error extracting time-frequency features: {e}")
            for key in [
                "stft_mean",
                "stft_std",
                "stft_max",
                "spectral_flux_mean",
                "spectral_flux_std",
            ]:
                features.setdefault(key, 0.0)

        return features

    # Helper functions for nonlinear features

    @staticmethod
    def _skewness(x):
        """Compute skewness."""
        n = len(x)
        if n < 3:
            return 0
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0
        return np.sum((x - mean) ** 3) / (n * std**3)

    @staticmethod
    def _kurtosis(x):
        """Compute kurtosis."""
        n = len(x)
        if n < 4:
            return 0
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0
        return np.sum((x - mean) ** 4) / (n * std**4) - 3

    @staticmethod
    def _sample_entropy(signal, m, r):
        """Simplified Sample Entropy calculation."""

        def _maxdist(xmi, xmj):
            return max([abs(ua - va) for ua, va in zip(xmi, xmj)])

        n = len(signal)
        if n < m + 1:
            return 0

        B = 0
        A = 0

        # Create templates
        xm = np.array([signal[i : i + m] for i in range(n - m)])
        xm1 = np.array([signal[i : i + m + 1] for i in range(n - m)])

        for i in range(len(xm)):
            template_m = xm[i]
            template_m1 = xm1[i]

            # Count matches for m
            for j in range(len(xm)):
                if i != j:
                    if _maxdist(template_m, xm[j]) <= r:
                        B += 1

            # Count matches for m+1
            for j in range(len(xm1)):
                if i != j:
                    if _maxdist(template_m1, xm1[j]) <= r:
                        A += 1

        B = B / (n - m)
        A = A / (n - m)

        if A == 0 or B == 0:
            return 0

        return -np.log(A / B)

    @staticmethod
    def _approximate_entropy(signal, m, r):
        """Simplified Approximate Entropy calculation."""

        def _maxdist(xmi, xmj):
            return max([abs(ua - va) for ua, va in zip(xmi, xmj)])

        n = len(signal)
        if n < m + 1:
            return 0

        def _phi(m):
            xm = np.array([signal[i : i + m] for i in range(n - m + 1)])
            C = []
            for i in range(len(xm)):
                template = xm[i]
                count = sum(
                    [1 for j in range(len(xm)) if _maxdist(template, xm[j]) <= r]
                )
                C.append(count / (n - m + 1))
            return np.sum(np.log(C)) / (n - m + 1)

        return abs(_phi(m) - _phi(m + 1))

    @staticmethod
    def _dfa(signal, min_box=4, max_box=None):
        """Simplified Detrended Fluctuation Analysis."""
        n = len(signal)
        if max_box is None:
            max_box = n // 4

        # Integrate signal
        y = np.cumsum(signal - np.mean(signal))

        scales = []
        fluct = []

        for box_size in range(min_box, max_box + 1, 4):
            if box_size >= n:
                break

            n_boxes = n // box_size
            boxes = y[: n_boxes * box_size].reshape(n_boxes, box_size)

            # Detrend each box
            trends = np.array([np.polyfit(range(box_size), box, 1) for box in boxes])
            trend_lines = np.array(
                [np.polyval(trend, range(box_size)) for trend in trends]
            )

            # Calculate fluctuation
            F = np.sqrt(np.mean((boxes - trend_lines) ** 2))

            scales.append(box_size)
            fluct.append(F)

        if len(scales) < 2:
            return 0

        # Fit line in log-log plot
        coeffs = np.polyfit(np.log(scales), np.log(fluct), 1)
        return coeffs[0]

    @staticmethod
    def _hurst_exponent(signal):
        """Calculate Hurst exponent using R/S analysis."""
        n = len(signal)
        if n < 20:
            return 0.5

        lags = range(2, min(n // 2, 100))
        tau = []

        for lag in lags:
            # Calculate standard deviation
            std = np.std(signal)
            if std == 0:
                continue

            # Calculate range
            mean = np.mean(signal)
            cumsum = np.cumsum(signal - mean)
            R = np.max(cumsum[:lag]) - np.min(cumsum[:lag])

            # R/S statistic
            RS = R / std if std > 0 else 0
            tau.append(RS)

        if len(tau) < 2:
            return 0.5

        # Fit line
        coeffs = np.polyfit(np.log(list(lags)[: len(tau)]), np.log(tau), 1)
        return coeffs[0]

    @staticmethod
    def _lyapunov_exponent(signal, delay=1):
        """Simplified Lyapunov exponent estimation."""
        n = len(signal)
        if n < 10:
            return 0

        # Create delayed embedding
        embedded = np.array([signal[i : i + 2] for i in range(0, n - delay - 1, delay)])

        if len(embedded) < 2:
            return 0

        # Calculate average divergence
        divergences = []
        for i in range(len(embedded) - 1):
            d0 = np.linalg.norm(embedded[i] - embedded[i + 1])
            if d0 > 0:
                divergences.append(np.log(d0))

        if len(divergences) == 0:
            return 0

        return np.mean(divergences)

    @staticmethod
    def _correlation_dimension(signal, max_dist=None):
        """Simplified correlation dimension estimation."""
        n = len(signal)
        if n < 10:
            return 0

        # Create embedding
        embedded = signal.reshape(-1, 1)

        # Calculate pairwise distances
        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(embedded))

        if max_dist is None:
            max_dist = np.std(signal)

        # Count pairs within distance
        if max_dist > 0:
            corr_sum = np.sum(distances < max_dist) - n  # Exclude diagonal
            corr_dim = corr_sum / (n * (n - 1))
            if corr_dim > 0:
                return -np.log(corr_dim) / np.log(max_dist)

        return 0

    def get_feature_names(self) -> List[str]:
        """
        Get names of extracted features.

        Returns
        -------
        feature_names : list of str
            Names of all extracted features
        """
        return self.feature_names_

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importance scores (if available).

        Returns
        -------
        importances : dict
            Dictionary mapping feature names to importance scores
        """
        return self.feature_importances_


class FeatureEngineering:
    """
    Advanced feature engineering for physiological signals.

    Provides methods for:
    - Feature construction
    - Feature interaction
    - Polynomial features
    - Domain-specific transformations

    Parameters
    ----------
    interaction_terms : bool, default=False
        Whether to create interaction terms
    polynomial_degree : int, default=1
        Degree of polynomial features (1 = no polynomial features)

    Examples
    --------
    >>> from vitalDSP.ml_models import FeatureEngineering
    >>> engineer = FeatureEngineering(interaction_terms=True)
    >>> X_engineered = engineer.fit_transform(X_features)
    """

    def __init__(
        self, interaction_terms: bool = False, polynomial_degree: int = 1, **kwargs
    ):
        self.interaction_terms = interaction_terms
        self.polynomial_degree = polynomial_degree
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the feature engineer."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features."""
        X_transformed = X.copy()

        # Add polynomial features
        if self.polynomial_degree > 1:
            from sklearn.preprocessing import PolynomialFeatures

            poly = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)
            X_transformed = poly.fit_transform(X_transformed)

        # Add interaction terms
        if self.interaction_terms and X.shape[1] > 1:
            # Create pairwise interactions
            n_features = X.shape[1]
            interactions = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    interactions.append(X[:, i] * X[:, j])

            if interactions:
                interactions = np.column_stack(interactions)
                X_transformed = np.hstack([X_transformed, interactions])

        return X_transformed

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit and transform features."""
        return self.fit(X, y).transform(X)


# Convenience function
def extract_features(
    signals: Union[np.ndarray, List[np.ndarray]],
    signal_type: str = "ecg",
    sampling_rate: float = 250.0,
    domains: Optional[List[str]] = None,
    return_dataframe: bool = False,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Quick feature extraction from physiological signals.

    Parameters
    ----------
    signals : array-like
        Input signals
    signal_type : str, default='ecg'
        Type of signal
    sampling_rate : float, default=250.0
        Sampling rate in Hz
    domains : list of str, optional
        Feature domains to extract
    return_dataframe : bool, default=False
        If True, return pandas DataFrame with feature names

    Returns
    -------
    features : ndarray or DataFrame
        Extracted features

    Examples
    --------
    >>> features = extract_features(ecg_signals, signal_type='ecg', sampling_rate=250)
    >>> print(f"Shape: {features.shape}")
    """
    extractor = FeatureExtractor(
        signal_type=signal_type, sampling_rate=sampling_rate, domains=domains
    )

    features = extractor.fit_transform(signals)

    if return_dataframe:
        return pd.DataFrame(features, columns=extractor.get_feature_names())

    return features
