import numpy as np
from transforms.wavelet_transform import WaveletTransform

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

        # Ensure wavelet_coeffs and fft_coeffs are compatible for multiplication
        if len(fft_coeffs) > len(wavelet_coeffs):
            fft_coeffs = fft_coeffs[: len(wavelet_coeffs)]
        elif len(fft_coeffs) < len(wavelet_coeffs):
            wavelet_coeffs = wavelet_coeffs[: len(fft_coeffs)]

        # Fusion by multiplying wavelet and FFT coefficients
        fusion_result = np.array([w * f for w, f in zip(wavelet_coeffs, fft_coeffs)])

        return fusion_result
