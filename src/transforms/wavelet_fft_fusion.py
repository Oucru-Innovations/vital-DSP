import numpy as np
from transforms.wavelet_transform import WaveletTransform

class WaveletFFTfusion:
    """
    A class to perform fusion of Wavelet Transform and FFT.

    Methods:
    - compute_fusion: Computes the fusion of wavelet and FFT for the signal.
    """

    def __init__(self, signal, wavelet_type='db', order=4, **kwargs):
        """
        Initialize the WaveletFFTfusion class with the signal.

        Parameters:
        signal (numpy.ndarray): The input signal.
        wavelet_type (str): The type of wavelet to use.
        order (int): The order of the wavelet.
        kwargs: Additional parameters for the specific wavelet.
        """
        self.signal = signal
        self.wavelet_type = wavelet_type
        self.order = order
        self.kwargs = kwargs

    def compute_fusion(self):
        """
        Compute the fusion of Wavelet Transform and FFT for the signal.

        Returns:
        numpy.ndarray: The fusion of wavelet and FFT.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> fusion = WaveletFFTfusion(signal)
        >>> fusion_result = fusion.compute_fusion()
        >>> print(fusion_result)
        """
        wavelet_transform = WaveletTransform(self.signal,wavelet_name=self.wavelet_type)
        wavelet_coeffs = wavelet_transform.perform_wavelet_transform(level=self.order)
        fft_coeffs = np.fft.fft(self.signal)
        fusion_result = [w * f for w, f in zip(wavelet_coeffs, fft_coeffs)]
        return fusion_result
