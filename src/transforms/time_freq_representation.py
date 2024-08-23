from transforms.stft import STFT
from transforms.wavelet_transform import WaveletTransform


class TimeFreqRepresentation:
    """
    A class to generate Time-Frequency Representations for machine learning.

    Methods:
    - compute_tfr: Computes the time-frequency representation of the signal.
    """

    def __init__(self, signal, method="stft", **kwargs):
        """
        Initialize the TimeFreqRepresentation class with the signal.

        Parameters:
        signal (numpy.ndarray): The input signal.
        method (str): The method to use for time-frequency representation ('stft', 'wavelet').
        kwargs: Additional parameters for the specific method.
        """
        self.signal = signal
        self.method = method
        self.kwargs = kwargs

    def compute_tfr(self):
        """
        Compute the Time-Frequency Representation of the signal.

        Returns:
        numpy.ndarray: The time-frequency representation of the signal.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> tfr = TimeFreqRepresentation(signal)
        >>> tfr_result = tfr.compute_tfr()
        >>> print(tfr_result)
        """
        if self.method == "stft":
            stft = STFT(self.signal, **self.kwargs)
            return stft.compute_stft()
        elif self.method == "wavelet":
            wavelet_transform = WaveletTransform(self.signal, **self.kwargs)
            return wavelet_transform.compute_wavelet_transform()
        else:
            raise ValueError("Unsupported method. Use 'stft' or 'wavelet'.")
