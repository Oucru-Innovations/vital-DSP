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

Examples:
---------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.transforms.wavelet_fft_fusion import WaveletFftFusion
    >>> signal = np.random.randn(1000)
    >>> processor = WaveletFftFusion(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np
from vitalDSP.transforms.wavelet_transform import WaveletTransform


class WaveletFFTfusion:
    """
    A class to perform the fusion of Wavelet Transform and FFT on a signal.

    This fusion technique combines the time-frequency localization capability of wavelet transforms
    with the frequency domain analysis provided by the Fast Fourier Transform (FFT). This method is
    particularly useful for signals that require both time-domain and frequency-domain analysis.

    Methods
    -------
    compute_fusion : method
        Computes the fusion of wavelet transform and FFT for the signal.
    """

    def __init__(self, signal, wavelet_type="db", order=4, **kwargs):
        """
        Initialize the WaveletFFTfusion class with the signal, wavelet type, and wavelet order.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be analyzed.
        wavelet_type : str, optional
            The type of wavelet to use (default is 'db').
        order : int, optional
            The order of the wavelet (default is 4).
        kwargs : dict, optional
            Additional parameters for specific wavelet types (if required).

        Raises
        ------
        ValueError
            If the provided wavelet type is not supported by the WaveletTransform class.
        """
        self.signal = signal
        self.wavelet_type = wavelet_type
        self.order = order
        self.kwargs = kwargs

    def compute_fusion(self):
        """
        Compute the fusion of Wavelet Transform and FFT for the signal.

        This method first applies the discrete wavelet transform (DWT) to the signal to obtain
        wavelet coefficients. Then, it computes the FFT of the original signal. The fusion is
        performed by multiplying corresponding wavelet and FFT coefficients.

        Returns
        -------
        numpy.ndarray
            The fusion of wavelet coefficients and FFT coefficients.

        Example
        -------
        >>> import numpy as np
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> fusion = WaveletFFTfusion(signal, wavelet_type='db', order=4)
        >>> fusion_result = fusion.compute_fusion()
        >>> print(fusion_result)
        """
        # Perform the Wavelet Transform
        wavelet_transform = WaveletTransform(
            self.signal, wavelet_name=self.wavelet_type
        )
        wavelet_coeffs = wavelet_transform.perform_wavelet_transform(level=self.order)

        # Perform the FFT
        fft_coeffs = np.fft.fft(self.signal)

        # wavelet_coeffs is a list of per-level arrays of potentially different lengths.
        # fft_coeffs is a 1D array (length = signal length).
        # Scale each level array by the magnitude of the corresponding FFT coefficient,
        # then pad all level arrays to the same length so np.array() produces a 2-D result.
        n_levels = len(wavelet_coeffs)
        fft_scalars = np.abs(fft_coeffs[:n_levels])

        scaled = [w * scalar for w, scalar in zip(wavelet_coeffs, fft_scalars)]

        # Pad each level to the length of the longest level so the result is rectangular.
        max_len = max(len(a) for a in scaled)
        fusion_result = np.array([np.pad(a, (0, max_len - len(a))) for a in scaled])

        return fusion_result
