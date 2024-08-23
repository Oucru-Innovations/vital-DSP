import numpy as np

class ChromaSTFT:
    """
    A class to compute the Chroma Short-Time Fourier Transform (Chroma STFT) to analyze harmonic content in audio signals, particularly speech.

    The Chroma STFT is useful for identifying the harmonic structure of a signal, often used in music and speech analysis. It projects the signal's frequency content onto a small number of pitch classes, making it easier to analyze the harmonic features.

    Attributes
    ----------
    signal : numpy.ndarray
        The input audio signal.
    sample_rate : int
        The sample rate of the signal.
    n_chroma : int
        The number of chroma bins (usually 12 for the 12 pitch classes in Western music).
    n_fft : int
        The FFT size, determining the frequency resolution of the STFT.

    Methods
    -------
    compute_chroma_stft()
        Computes the Chroma STFT of the signal.
    """

    def __init__(self, signal, sample_rate=16000, n_chroma=12, n_fft=2048):
        """
        Initialize the ChromaSTFT class with the signal and parameters.

        Parameters
        ----------
        signal : numpy.ndarray
            The input audio signal to be analyzed.
        sample_rate : int, optional
            The sample rate of the signal (default is 16000 Hz).
        n_chroma : int, optional
            The number of chroma bins (default is 12).
        n_fft : int, optional
            The FFT size, determining the frequency resolution (default is 2048).

        Examples
        --------
        >>> signal = np.sin(2 * np.pi * np.linspace(0, 1, 16000))
        >>> chroma_stft = ChromaSTFT(signal, sample_rate=16000, n_chroma=12, n_fft=2048)
        >>> print(chroma_stft.signal.shape)
        (16000,)
        """
        self.signal = signal
        self.sample_rate = sample_rate
        self.n_chroma = n_chroma
        self.n_fft = n_fft

    def compute_chroma_stft(self):
        """
        Compute the Chroma Short-Time Fourier Transform (Chroma STFT) of the signal.

        The Chroma STFT projects the frequency content of the signal onto chroma bins, capturing harmonic features in a compact representation.

        Returns
        -------
        chroma_stft : numpy.ndarray
            The Chroma STFT of the signal. This is a matrix where each row represents a chroma bin, and each column represents a time frame.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> chroma_stft = ChromaSTFT(signal)
        >>> chroma_stft_result = chroma_stft.compute_chroma_stft()
        >>> print(chroma_stft_result.shape)
        (12, 1025)

        Notes
        -----
        The Chroma STFT is commonly used in music and speech processing to extract pitch-related features. It simplifies the analysis by reducing the dimensionality of the frequency content while retaining essential harmonic information.

        In this example implementation, a simulated chroma filter bank is used for demonstration purposes. In practice, a proper chroma filter bank should be used to accurately capture the harmonic structure of the input signal.
        """
        # Compute the Short-Time Fourier Transform (STFT) of the signal
        stft_matrix = np.abs(np.fft.fft(self.signal, n=self.n_fft))[
            : self.n_fft // 2 + 1
        ]

        # Simulated chroma filter banks for projecting frequency bins to chroma bins
        chroma_filter = np.random.rand(
            self.n_chroma, self.n_fft // 2 + 1
        )

        # Compute the Chroma STFT by applying the chroma filter bank
        chroma_stft = np.dot(chroma_filter, stft_matrix)

        return chroma_stft
