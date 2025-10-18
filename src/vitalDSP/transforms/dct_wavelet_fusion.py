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
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.transforms.dct_wavelet_fusion import DctWaveletFusion
    >>> signal = np.random.randn(1000)
    >>> processor = DctWaveletFusion(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""


from vitalDSP.transforms.discrete_cosine_transform import DiscreteCosineTransform
from vitalDSP.transforms.wavelet_transform import WaveletTransform
import numpy as np


class DCTWaveletFusion:
    """
    A class to perform fusion of Discrete Cosine Transform (DCT) and Wavelet Transform on a signal.

    This class allows for the combination of DCT, which is effective for frequency-domain analysis, and Wavelet Transform, which excels at capturing both frequency and location information. The fusion of these two transforms can be particularly useful in signal processing tasks such as denoising, feature extraction, and data compression.

    Methods
    -------
    compute_fusion : method
        Computes the fusion of DCT and Wavelet Transform for the given signal.
    """

    def __init__(self, signal, wavelet_type="db", order=4, **kwargs):
        """
        Initialize the DCTWaveletFusion class with the signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be transformed.
        wavelet_type : str, optional
            The type of wavelet to use for the Wavelet Transform (default is 'db').
        order : int, optional
            The order of the wavelet used in the Wavelet Transform (default is 4).
        kwargs : dict, optional
            Additional parameters for the specific wavelet or other options.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> fusion = DCTWaveletFusion(signal, wavelet_type='db', order=4)
        >>> fusion_result = fusion.compute_fusion()
        >>> print(fusion_result)
        """
        self.signal = signal
        self.wavelet_type = wavelet_type
        self.order = order
        self.kwargs = kwargs

    def _match_lengths(self, dct_coeffs, wavelet_coeffs):
        """
        Helper method to match the lengths of DCT and Wavelet coefficients.

        Parameters
        ----------
        dct_coeffs : numpy.ndarray
            DCT coefficients of the signal.
        wavelet_coeffs : numpy.ndarray
            Wavelet coefficients of the signal.

        Returns
        -------
        dct_coeffs, wavelet_coeffs : numpy.ndarray
            The adjusted DCT and Wavelet coefficients with matching lengths.
        """
        min_length = min(len(dct_coeffs), len(wavelet_coeffs))
        return dct_coeffs[:min_length], wavelet_coeffs[:min_length]

    def compute_fusion(self):
        """
        Compute the fusion of Discrete Cosine Transform (DCT) and Wavelet Transform for the signal.

        The fusion process involves computing the DCT of the signal, followed by a Wavelet Transform. The resulting coefficients from both transforms are then combined multiplicatively to achieve a fusion that incorporates features from both the frequency and time-frequency domains.

        Returns
        -------
        numpy.ndarray
            The fused signal, combining DCT and Wavelet Transform coefficients.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> fusion = DCTWaveletFusion(signal)
        >>> fusion_result = fusion.compute_fusion()
        >>> print(fusion_result)
        """
        # Compute DCT coefficients
        dct = DiscreteCosineTransform(self.signal)
        dct_coeffs = dct.compute_dct()

        # Compute Wavelet Transform coefficients and use only approximation coefficients
        wavelet_transform = WaveletTransform(
            self.signal, wavelet_name=self.wavelet_type
        )
        wavelet_coeffs = wavelet_transform.perform_wavelet_transform(level=self.order)

        # Extract the approximation coefficients (usually the first element of the wavelet transform result)
        approx_coeffs = wavelet_coeffs[0]

        # Match lengths of DCT and approximation wavelet coefficients
        dct_coeffs, approx_coeffs = self._match_lengths(dct_coeffs, approx_coeffs)

        # Perform the fusion by multiplying corresponding DCT and Wavelet coefficients
        fusion_result = np.multiply(dct_coeffs, approx_coeffs)

        return fusion_result
