"""
Utility Functions Module for Physiological Signal Processing

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
    >>> from vitalDSP.signal_processing.scaler import Scaler
    >>> signal = np.random.randn(1000)
    >>> processor = Scaler(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np


class StandardScaler:
    """
    A custom implementation of the Standard Scaler, which standardizes signals by removing the mean and scaling to unit variance.

    This class can be used to normalize signals before applying transformations like the Discrete Wavelet Transform (DWT).

    Attributes
    ----------
    mean_ : float or None
        The mean of the signal after fitting.
    std_ : float or None
        The standard deviation of the signal after fitting.

    Methods
    -------
    fit(signal)
        Calculate mean and standard deviation for scaling.
    transform(signal)
        Scale the signal using the mean and standard deviation.
    fit_transform(signal)
        Fit to the data and then transform it.
    """

    def __init__(self):
        """
        Initialize the StandardScaler with mean and standard deviation set to None.
        """
        self.mean_ = None
        self.std_ = None

    def fit(self, signal):
        """
        Fit the scaler to the signal by calculating the mean and standard deviation.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be standardized.

        Returns
        -------
        None
        """
        self.mean_ = np.mean(signal)
        self.std_ = np.std(signal)

    def transform(self, signal):
        """
        Transform the signal using the fitted scaler by removing the mean and scaling to unit variance.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be transformed.

        Returns
        -------
        scaled_signal : numpy.ndarray
            The standardized signal with zero mean and unit variance.

        Raises
        ------
        ValueError
            If the scaler has not been fitted yet.

        Examples
        --------
        >>> scaler = StandardScaler()
        >>> scaler.fit(np.array([1, 2, 3, 4, 5]))
        >>> scaled_signal = scaler.transform(np.array([1, 2, 3, 4, 5]))
        >>> print(scaled_signal)
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("The scaler has not been fitted yet. Call 'fit' first.")
        return (signal - self.mean_) / self.std_

    def fit_transform(self, signal):
        """
        Fit to the signal, then transform it.

        This is a convenience method that combines `fit` and `transform` into a single step.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be fitted and transformed.

        Returns
        -------
        scaled_signal : numpy.ndarray
            The standardized signal with zero mean and unit variance.

        Examples
        --------
        >>> scaler = StandardScaler()
        >>> scaled_signal = scaler.fit_transform(np.array([1, 2, 3, 4, 5]))
        >>> print(scaled_signal)
        """
        self.fit(signal)
        return self.transform(signal)
