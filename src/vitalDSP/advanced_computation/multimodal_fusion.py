"""
Advanced Computation Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.advanced_computation.multimodal_fusion import MultimodalFusion
    >>> signal = np.random.randn(1000)
    >>> processor = MultimodalFusion(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""


import numpy as np


class MultimodalFusion:
    """
    Multimodal Fusion class for combining multiple signals to improve health assessments.

    The class supports various fusion strategies, including weighted sum, concatenation, PCA-based fusion, maximum, and minimum.

    Example Usage
    -------------
    signal1 = np.sin(np.linspace(0, 10, 100))
    signal2 = np.cos(np.linspace(0, 10, 100))
    fusion = MultimodalFusion([signal1, signal2])

    # Weighted Sum Fusion
    fused_signal_weighted = fusion.fuse(strategy="weighted_sum", weights=[0.6, 0.4])
    print("Fused Signal (Weighted Sum):", fused_signal_weighted)

    # Concatenation Fusion
    fused_signal_concat = fusion.fuse(strategy="concatenation")
    print("Fused Signal (Concatenation):", fused_signal_concat)

    # PCA-based Fusion
    fused_signal_pca = fusion.fuse(strategy="pca", n_components=1)
    print("Fused Signal (PCA):", fused_signal_pca)

    # Maximum Fusion
    fused_signal_max = fusion.fuse(strategy="maximum")
    print("Fused Signal (Maximum):", fused_signal_max)

    # Minimum Fusion
    fused_signal_min = fusion.fuse(strategy="minimum")
    print("Fused Signal (Minimum):", fused_signal_min)
    """

    def __init__(self, signals):
        """
        Initialize the MultimodalFusion class with the signals to be fused.

        Parameters
        ----------
        signals : list of numpy.ndarray
            A list of signals to be fused. Each signal should be a 1D numpy array of the same length.

        Raises
        ------
        ValueError
            If the input signals are not a list or if they are not numpy arrays of the same length.
        """
        if not isinstance(signals, list) or not all(
            isinstance(s, np.ndarray) for s in signals
        ):
            raise ValueError("Signals must be a list of numpy arrays.")

        signal_lengths = [len(signal) for signal in signals]
        if len(set(signal_lengths)) != 1:
            raise ValueError("All signals must have the same length.")

        self.signals = signals

    def fuse(self, strategy="weighted_sum", **kwargs):
        """
        Fuse multiple signals using the specified strategy.

        Parameters
        ----------
        strategy : str, optional
            The fusion strategy to use. Options include "weighted_sum", "concatenation", "pca", "maximum", "minimum" (default is "weighted_sum").
        kwargs : dict, optional
            Additional arguments depending on the fusion strategy.
            - weights (list of float): Weights for the weighted sum strategy (required if strategy="weighted_sum").
            - n_components (int): Number of principal components to keep for PCA (optional, default=1).

        Returns
        -------
        numpy.ndarray
            The fused signal based on the chosen strategy.

        Raises
        ------
        ValueError
            If an unknown fusion strategy is specified or if required arguments are missing.

        Examples
        --------
        >>> signal1 = np.sin(np.linspace(0, 10, 100))
        >>> signal2 = np.cos(np.linspace(0, 10, 100))
        >>> fusion = MultimodalFusion([signal1, signal2])
        >>> fused_signal = fusion.fuse(strategy="weighted_sum", weights=[0.6, 0.4])
        >>> print(fused_signal)
        """
        if strategy == "weighted_sum":
            return self._weighted_sum_fusion(kwargs.get("weights"))
        elif strategy == "concatenation":
            return self._concatenation_fusion()
        elif strategy == "pca":
            return self._pca_fusion(kwargs.get("n_components", 1))
        elif strategy == "maximum":
            return self._maximum_fusion()
        elif strategy == "minimum":
            return self._minimum_fusion()
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

    def _weighted_sum_fusion(self, weights):
        """
        Fuse signals using a weighted sum.

        Parameters
        ----------
        weights : list of float
            Weights for each signal. The length of weights must match the number of signals.

        Returns
        -------
        numpy.ndarray
            The fused signal as a weighted sum of the input signals.

        Raises
        ------
        ValueError
            If weights are not provided or if the number of weights does not match the number of signals.

        Examples
        --------
        >>> signal1 = np.array([1, 2, 3])
        >>> signal2 = np.array([4, 5, 6])
        >>> fusion = MultimodalFusion([signal1, signal2])
        >>> fused_signal = fusion._weighted_sum_fusion(weights=[0.7, 0.3])
        >>> print(fused_signal)
        """
        if weights is None or len(weights) != len(self.signals):
            raise ValueError(
                "Weights must be provided and must match the number of signals."
            )

        fused_signal = np.zeros_like(self.signals[0])
        for signal, weight in zip(self.signals, weights):
            fused_signal += weight * signal
        return fused_signal

    def _concatenation_fusion(self):
        """
        Fuse signals by concatenation.

        Returns
        -------
        numpy.ndarray
            The concatenated signal.

        Examples
        --------
        >>> signal1 = np.array([1, 2, 3])
        >>> signal2 = np.array([4, 5, 6])
        >>> fusion = MultimodalFusion([signal1, signal2])
        >>> fused_signal = fusion._concatenation_fusion()
        >>> print(fused_signal)
        """
        return np.concatenate(self.signals, axis=0)

    def _pca_fusion(self, n_components=1):
        """
        Fuse signals using Principal Component Analysis (PCA).

        Parameters
        ----------
        n_components : int, optional
            Number of principal components to keep (default is 1).

        Returns
        -------
        numpy.ndarray
            The fused signal based on PCA.

        Examples
        --------
        >>> signal1 = np.array([1, 2, 3, 4])
        >>> signal2 = np.array([4, 5, 6, 7])
        >>> fusion = MultimodalFusion([signal1, signal2])
        >>> fused_signal = fusion._pca_fusion(n_components=1)
        >>> print(fused_signal)
        """
        signals_matrix = np.vstack(self.signals).T
        mean_centered = signals_matrix - np.mean(signals_matrix, axis=0)
        covariance_matrix = np.cov(mean_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort the eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top n_components
        selected_eigenvectors = sorted_eigenvectors[:, :n_components]

        # Project the signals onto the selected eigenvectors
        pca_fused = np.dot(mean_centered, selected_eigenvectors)

        return pca_fused[:, 0] if n_components == 1 else pca_fused

    def _maximum_fusion(self):
        """
        Fuse signals by taking the maximum value at each point across all signals.

        Returns
        -------
        numpy.ndarray
            The fused signal using maximum selection.

        Examples
        --------
        >>> signal1 = np.array([1, 3, 5])
        >>> signal2 = np.array([2, 4, 6])
        >>> fusion = MultimodalFusion([signal1, signal2])
        >>> fused_signal = fusion._maximum_fusion()
        >>> print(fused_signal)
        """
        return np.max(np.vstack(self.signals), axis=0)

    def _minimum_fusion(self):
        """
        Fuse signals by taking the minimum value at each point across all signals.

        Returns
        -------
        numpy.ndarray
            The fused signal using minimum selection.

        Examples
        --------
        >>> signal1 = np.array([1, 3, 5])
        >>> signal2 = np.array([2, 4, 6])
        >>> fusion = MultimodalFusion([signal1, signal2])
        >>> fused_signal = fusion._minimum_fusion()
        >>> print(fused_signal)
        """
        return np.min(np.vstack(self.signals), axis=0)
