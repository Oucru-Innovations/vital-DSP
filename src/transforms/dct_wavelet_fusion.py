from transforms.discrete_cosine_transform import DiscreteCosineTransform
from transforms.wavelet_transform import WaveletTransform
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
        # Compute the Discrete Cosine Transform (DCT) coefficients
        dct = DiscreteCosineTransform(self.signal)
        dct_coeffs = dct.compute_dct()

        # Compute the Wavelet Transform coefficients
        wavelet_transform = WaveletTransform(self.signal, wavelet_name=self.wavelet_type)
        wavelet_coeffs = wavelet_transform.perform_wavelet_transform(level=self.order)

        # Perform the fusion by multiplying corresponding DCT and Wavelet coefficients
        fusion_result = np.multiply(dct_coeffs, wavelet_coeffs)

        return fusion_result
