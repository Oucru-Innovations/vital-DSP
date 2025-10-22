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
    >>> from vitalDSP.advanced_computation.sparse_signal_processing import SparseSignalProcessing
    >>> signal = np.random.randn(1000)
    >>> processor = SparseSignalProcessing(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np


class SparseSignalProcessing:
    """
    Sparse Signal Processing for efficient representation and processing of signals.

    This class provides methods for transforming a signal into a sparse domain,
    applying thresholding to remove noise, and reconstructing the signal from its sparse representation.

    Methods
    -------
    sparse_representation : method
        Represents the signal using a specified sparse basis (e.g., wavelets, DCT, FFT).
    thresholding : method
        Applies a threshold to the sparse representation to denoise the signal.
    reconstruction : method
        Reconstructs the signal from its sparse representation.

    Example Usage
    -------------
    signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
    sparse_processing = SparseSignalProcessing(signal)
    sparse_rep = sparse_processing.sparse_representation(np.fft.fft)
    thresholded_sparse = sparse_processing.thresholding(sparse_rep, threshold=0.1)
    reconstructed_signal = sparse_processing.reconstruction(thresholded_sparse, np.fft.ifft)
    print("Reconstructed Signal:", reconstructed_signal)
    """

    def __init__(self, signal):
        """
        Initialize the SparseSignalProcessing class with the signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be processed.
        """
        self.signal = signal

    def sparse_representation(self, basis):
        """
        Represent the signal using a sparse basis.

        This method transforms the signal into a sparse domain, such as using wavelets, DCT (Discrete Cosine Transform),
        or FFT (Fast Fourier Transform), allowing for efficient processing and manipulation.

        Parameters
        ----------
        basis : callable
            A function or method to transform the signal to the sparse domain (e.g., `np.fft.fft` for FFT, or
            `wavelet_transform.perform_wavelet_transform` for wavelet transform).

        Returns
        -------
        numpy.ndarray
            The sparse representation of the signal.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
        >>> sparse_processing = SparseSignalProcessing(signal)
        >>> sparse_rep = sparse_processing.sparse_representation(np.fft.fft)
        >>> print(sparse_rep)
        """
        sparse_rep = basis(self.signal)
        return sparse_rep

    def thresholding(self, sparse_rep, threshold):
        """
        Apply a threshold to the sparse representation to denoise the signal.

        This method sets values in the sparse representation that are below the threshold to zero, effectively removing
        small coefficients that may correspond to noise.

        Parameters
        ----------
        sparse_rep : numpy.ndarray
            The sparse representation of the signal.
        threshold : float
            The threshold value. Coefficients with absolute values below this threshold will be set to zero.

        Returns
        -------
        numpy.ndarray
            The thresholded sparse representation.

        Examples
        --------
        >>> thresholded_sparse = sparse_processing.thresholding(sparse_rep, threshold=0.1)
        >>> print(thresholded_sparse)
        """
        sparse_rep[np.abs(sparse_rep) < threshold] = 0
        return sparse_rep

    def reconstruction(self, sparse_rep, inverse_basis):
        """
        Reconstruct the signal from its sparse representation.

        This method applies the inverse transform to the sparse representation to reconstruct the signal in its original domain.

        Parameters
        ----------
        sparse_rep : numpy.ndarray
            The sparse representation of the signal.
        inverse_basis : callable
            A function or method to inverse-transform the signal from the sparse domain
            (e.g., `np.fft.ifft` for inverse FFT, or `wavelet_transform.perform_inverse_wavelet_transform` for inverse wavelet transform).

        Returns
        -------
        numpy.ndarray
            The reconstructed signal.

        Examples
        --------
        >>> reconstructed_signal = sparse_processing.reconstruction(thresholded_sparse, np.fft.ifft)
        >>> print(reconstructed_signal)
        """
        reconstructed_signal = inverse_basis(sparse_rep)
        return reconstructed_signal
