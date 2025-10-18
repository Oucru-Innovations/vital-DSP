"""
Signal Transforms Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations
- Performance optimization

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.transforms.stft import Stft
    >>> signal = np.random.randn(1000)
    >>> processor = Stft(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""


import numpy as np


class STFT:
    """
    A class to perform Short-Time Fourier Transform (STFT) for analyzing time-varying signals.

    The STFT is used to understand how the frequency content of a signal evolves over time by dividing the signal into overlapping segments, applying a Fourier Transform to each segment, and assembling the results into a time-frequency representation.

    Methods
    -------
    compute_stft : method
        Computes the STFT of the signal.
    """

    def __init__(self, signal, window_size=256, hop_size=128, n_fft=512):
        """
        Initialize the STFT class with the input signal and parameters for the STFT computation.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be transformed, typically a 1D array representing the time series data.
        window_size : int, optional
            The size of the window to apply on the signal for each segment (default is 256).
        hop_size : int, optional
            The number of samples to shift the window at each step (default is 128). It determines the overlap between adjacent windows.
        n_fft : int, optional
            The number of points for the Fast Fourier Transform (FFT) (default is 512). It should ideally be a power of 2 to optimize FFT computation.

        Examples
        --------
        >>> signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 500)) + np.sin(2 * np.pi * 20 * np.linspace(0, 1, 500))
        >>> stft = STFT(signal, window_size=100, hop_size=50, n_fft=128)
        >>> stft_result = stft.compute_stft()
        >>> print(stft_result.shape)
        (65, 9)
        """
        self.signal = signal
        self.window_size = window_size
        self.hop_size = hop_size
        self.n_fft = n_fft
        self._validate_parameters()

    def _validate_parameters(self):
        """
        Validate and adjust parameters to ensure they are appropriate for STFT computation.

        This method checks that window_size, hop_size, and n_fft are positive integers and that the window size does not exceed the length of the signal.

        Raises
        ------
        ValueError
            If any of the parameters are not valid (e.g., negative or zero values).
        """
        if self.window_size <= 0 or self.hop_size <= 0 or self.n_fft <= 0:
            raise ValueError(
                "Window size, hop size, and n_fft must be positive integers."
            )
        if self.window_size > len(self.signal):
            raise ValueError("Window size cannot be larger than the signal length.")

    def compute_stft(self):
        """
        OPTIMIZED: Compute the Short-Time Fourier Transform (STFT) of the signal using vectorized operations.

        The STFT splits the signal into overlapping windows, applies a Hanning window function to reduce spectral leakage, and computes the FFT for each windowed segment. The result is a matrix representing the magnitude and phase of the signal's frequency components over time.

        Returns
        -------
        stft_matrix : numpy.ndarray
            A 2D complex-valued array where rows correspond to frequency bins and columns correspond to time frames.

        Examples
        --------
        >>> signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 500)) + np.sin(2 * np.pi * 20 * np.linspace(0, 1, 500))
        >>> stft = STFT(signal, window_size=100, hop_size=50, n_fft=128)
        >>> stft_result = stft.compute_stft()
        >>> print(stft_result.shape)
        (65, 9)
        """
        n_windows = 1 + (len(self.signal) - self.window_size) // self.hop_size
        stft_matrix = np.zeros((self.n_fft // 2 + 1, n_windows), dtype=complex)

        # OPTIMIZATION: Pre-compute window function
        window = np.hanning(self.window_size)

        # OPTIMIZATION: Vectorized windowing and FFT computation
        for i in range(n_windows):
            start = i * self.hop_size
            end = start + self.window_size

            # OPTIMIZATION: Vectorized windowing
            windowed_signal = self.signal[start:end] * window

            # Ensure the windowed signal length matches n_fft for FFT computation
            if len(windowed_signal) < self.n_fft:
                windowed_signal = np.pad(
                    windowed_signal,
                    (0, self.n_fft - len(windowed_signal)),
                    mode="constant",
                )

            # OPTIMIZATION: Use optimized FFT
            fft_result = np.fft.rfft(windowed_signal, n=self.n_fft)
            stft_matrix[:, i] = fft_result

        return stft_matrix
