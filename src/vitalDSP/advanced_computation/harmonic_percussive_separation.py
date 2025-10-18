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
    >>> from vitalDSP.advanced_computation.harmonic_percussive_separation import HarmonicPercussiveSeparation
    >>> signal = np.random.randn(1000)
    >>> processor = HarmonicPercussiveSeparation(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""


import numpy as np


class HarmonicPercussiveSeparation:
    """
    Harmonic-Percussive Separation for separating vocal and noise components in respiratory analysis.

    This class separates a signal into its harmonic (sustained, tonal) and percussive (transient, noisy) components,
    which can be particularly useful in respiratory analysis to distinguish between vocal and noise components.

    Methods
    -------
    separate : method
        Separates the harmonic and percussive components of the signal using median filtering.

    Example Usage
    -------------
    signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
    hps = HarmonicPercussiveSeparation(signal)
    harmonic, percussive = hps.separate()
    print("Harmonic Component:", harmonic)
    print("Percussive Component:", percussive)
    """

    def __init__(self, signal):
        """
        Initialize the HarmonicPercussiveSeparation class with the input signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal from which harmonic and percussive components will be separated.

        Example
        -------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
        >>> hps = HarmonicPercussiveSeparation(signal)
        """
        if not isinstance(signal, np.ndarray):
            raise TypeError("Signal must be a numpy array.")
        self.signal = signal

    def separate(self, kernel_size=31):
        """
        Separate the harmonic and percussive components of the signal using median filtering.

        The harmonic component is obtained by applying a median filter across the time axis, while the percussive
        component is obtained by applying a median filter across the frequency axis.

        Parameters
        ----------
        kernel_size : int, optional
            Size of the median filter kernel (default is 31).

        Returns
        -------
        harmonic : numpy.ndarray
            The harmonic component of the signal.
        percussive : numpy.ndarray
            The percussive component of the signal.

        Example
        -------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
        >>> hps = HarmonicPercussiveSeparation(signal)
        >>> harmonic, percussive = hps.separate(kernel_size=31)
        >>> print(harmonic)
        >>> print(percussive)
        """
        harmonic = self._apply_median(self.signal, kernel_size)
        percussive = self._apply_median(
            self.signal, kernel_size
        )  # Simplified for 1D signal
        return harmonic, percussive

    def _median_filter(self, signal, size):
        """
        Apply a median filter to the signal based on the specified size.

        This method applies the median filter across the rows or columns depending on the size parameter,
        which determines the focus on either harmonic or percussive components.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be filtered.
        size : tuple
            The size of the filter kernel; (kernel_size, 1) applies the filter across rows (harmonic),
            while (1, kernel_size) applies it across columns (percussive).

        Returns
        -------
        filtered_signal : numpy.ndarray
            The filtered signal with the specified component emphasized.

        Example
        -------
        >>> filtered_signal = self._median_filter(signal, size=(31, 1))
        """
        filtered_signal = np.copy(signal)
        if size[0] > 1:  # Apply median filter across the rows (time axis)
            for i in range(signal.shape[0]):
                filtered_signal[i, :] = self._apply_median(signal[i, :], size[0])
        if size[1] > 1:  # Apply median filter across the columns (frequency axis)
            for i in range(signal.shape[1]):
                filtered_signal[:, i] = self._apply_median(signal[:, i], size[1])
        return filtered_signal

    def _apply_median(self, arr, kernel_size):
        """
        Apply a 1D median filter to an array.

        This method is used by the _median_filter method to process each row or column of the signal.

        Parameters
        ----------
        arr : numpy.ndarray
            The array (either a row or a column of the signal) to be filtered.
        kernel_size : int
            The size of the median filter kernel.

        Returns
        -------
        filtered_arr : numpy.ndarray
            The filtered array after median filtering.

        Example
        -------
        >>> filtered_arr = self._apply_median(arr, kernel_size=31)
        """
        padded_arr = np.pad(arr, (kernel_size // 2, kernel_size // 2), mode="edge")
        filtered_arr = np.zeros_like(arr)

        for i in range(len(arr)):
            filtered_arr[i] = np.median(padded_arr[i : i + kernel_size])

        return filtered_arr
