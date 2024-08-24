from transforms.stft import STFT
from transforms.wavelet_transform import WaveletTransform

class TimeFreqRepresentation:
    """
    A class to generate Time-Frequency Representations (TFR) of signals for machine learning and signal analysis.

    This class supports multiple methods for computing the TFR, including Short-Time Fourier Transform (STFT) and Wavelet Transform. These representations are useful in analyzing the frequency content of signals as it changes over time, which is particularly valuable in tasks like speech processing, biomedical signal analysis, and other time-series data applications.

    Methods
    -------
    compute_tfr : method
        Computes the time-frequency representation of the signal using the specified method.
    """

    def __init__(self, signal, method="stft", **kwargs):
        """
        Initialize the TimeFreqRepresentation class with the signal and method.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be transformed into a time-frequency representation.
        method : str, optional
            The method to use for time-frequency representation ('stft' or 'wavelet'). Default is 'stft'.
        kwargs : dict, optional
            Additional parameters specific to the chosen method, such as window size for STFT or wavelet type for Wavelet Transform.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> tfr = TimeFreqRepresentation(signal, method='stft', window='hann', nperseg=256)
        >>> tfr_result = tfr.compute_tfr()
        >>> print(tfr_result)
        """
        self.signal = signal
        self.method = method
        self.kwargs = kwargs

    def compute_tfr(self):
        """
        Compute the Time-Frequency Representation (TFR) of the signal.

        Depending on the chosen method, this function computes either the Short-Time Fourier Transform (STFT) or the Wavelet Transform to represent the signal in both time and frequency domains.

        Returns
        -------
        numpy.ndarray
            The time-frequency representation of the signal. The output format depends on the method:
            - For STFT: A 2D array where rows correspond to time segments and columns to frequency bins.
            - For Wavelet Transform: A 2D array where rows correspond to time segments and columns to scales or frequencies.

        Raises
        ------
        ValueError
            If an unsupported method is specified.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> tfr = TimeFreqRepresentation(signal, method='wavelet', wavelet_name='db4', level=4)
        >>> tfr_result = tfr.compute_tfr()
        >>> print(tfr_result)
        """
        if self.method == "stft":
            stft = STFT(self.signal, **self.kwargs)
            return stft.compute_stft()
        elif self.method == "wavelet":
            wavelet_transform = WaveletTransform(self.signal, **self.kwargs)
            return wavelet_transform.perform_wavelet_transform()
        else:
            raise ValueError("Unsupported method. Use 'stft' or 'wavelet'.")
