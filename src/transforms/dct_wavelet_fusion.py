from transforms.discrete_cosine_transform import DiscreteCosineTransform
from transforms.wavelet_transform import WaveletTransform


class DCTWaveletFusion:
    """
    A class to perform fusion of DCT and Wavelet Transform.

    Methods:
    - compute_fusion: Computes the fusion of DCT and wavelet for the signal.
    """

    def __init__(self, signal, wavelet_type="db", order=4, **kwargs):
        """
        Initialize the DCTWaveletFusion class with the signal.

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
        Compute the fusion of DCT and Wavelet Transform for the signal.

        Returns:
        numpy.ndarray: The fusion of DCT and wavelet.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> fusion = DCTWaveletFusion(signal)
        >>> fusion_result = fusion.compute_fusion()
        >>> print(fusion_result)
        """
        dct = DiscreteCosineTransform(self.signal)
        dct_coeffs = dct.compute_dct()
        wavelet_transform = WaveletTransform(
            self.signal, wavelet_name=self.wavelet_type
        )
        wavelet_coeffs = wavelet_transform.perform_wavelet_transform(
            level=self.order, **self.kwargs
        )
        # wavelet_transform = WaveletTransform(self.signal)
        # wavelet_coeffs = wavelet_transform.compute_wavelet_transform(wavelet_type=self.wavelet_type, order=self.order, **self.kwargs)
        fusion_result = [d * w for d, w in zip(dct_coeffs, wavelet_coeffs)]
        return fusion_result
