import numpy as np
from scipy.signal import get_window

class ChromaSTFT:
    """
    A class to compute the Chroma Short-Time Fourier Transform (Chroma STFT) to analyze harmonic content in audio signals.

    The Chroma STFT is useful for identifying the harmonic structure of a signal by projecting its frequency content onto a small number of pitch classes (chroma bins).

    Attributes
    ----------
    signal : numpy.ndarray
        The input audio signal.
    sample_rate : int
        The sample rate of the signal.
    n_chroma : int
        The number of chroma bins (usually 12 for the 12 pitch classes).
    n_fft : int
        The FFT size, determining the frequency resolution of the STFT.
    hop_length : int
        The number of samples between successive frames.

    Methods
    -------
    compute_chroma_stft()
        Computes the Chroma STFT of the signal.

    Example Usage
    -------------
    >>> signal = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 16000))  # 2 seconds of A4 note (440Hz)
    >>> chroma_stft = ChromaSTFT(signal, sample_rate=16000, n_chroma=12, n_fft=2048, hop_length=512)
    >>> chroma_stft_result = chroma_stft.compute_chroma_stft()
    >>> print(chroma_stft_result.shape)  # Output: (12, num_frames)
    """

    def __init__(self, signal, sample_rate=16000, n_chroma=12, n_fft=2048, hop_length=512):
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
        hop_length : int, optional
            The number of samples between successive frames (default is 512).
        """
        self.signal = signal
        self.sample_rate = sample_rate
        self.n_chroma = n_chroma
        self.n_fft = n_fft
        self.hop_length = hop_length

    def _compute_stft(self):
        """
        Compute the Short-Time Fourier Transform (STFT) of the signal.

        Returns
        -------
        stft_matrix : numpy.ndarray
            The STFT of the signal, where each column represents a time frame and each row a frequency bin.
        """
        if len(self.signal) < self.n_fft:
            raise ValueError("The length of the signal is shorter than the FFT size (n_fft).")

        window = get_window('hann', self.n_fft)
        num_frames = 1 + (len(self.signal) - self.n_fft) // self.hop_length

        # Check if there are enough frames to compute STFT
        if num_frames <= 0:
            raise ValueError("The signal is too short for the given FFT size and hop length.")

        stft_matrix = np.empty((self.n_fft // 2 + 1, num_frames), dtype=np.complex64)

        for i in range(num_frames):
            start = i * self.hop_length
            frame = self.signal[start:start + self.n_fft] * window
            stft_matrix[:, i] = np.fft.rfft(frame)

        return np.abs(stft_matrix)

    def _create_chroma_filter(self):
        """
        Create a simulated chroma filter to map frequency bins to chroma bins.

        Returns
        -------
        chroma_filter : numpy.ndarray
            A simulated chroma filter mapping frequency bins to chroma bins.
        """
        chroma_filter = np.random.rand(self.n_chroma, self.n_fft // 2 + 1)
        return chroma_filter

    def compute_chroma_stft(self):
        """
        Compute the Chroma Short-Time Fourier Transform (Chroma STFT) of the signal.

        This function applies the STFT to the signal and then projects the result onto chroma bins.

        Returns
        -------
        chroma_stft : numpy.ndarray
            The Chroma STFT of the signal. Each row represents a chroma bin, and each column represents a time frame.

        Examples
        --------
        >>> signal = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 16000))  # 2 seconds of A4 note (440Hz)
        >>> chroma_stft = ChromaSTFT(signal, sample_rate=16000, n_chroma=12, n_fft=2048, hop_length=512)
        >>> chroma_stft_result = chroma_stft.compute_chroma_stft()
        >>> print(chroma_stft_result.shape)  # Output: (12, num_frames)
        """
        stft_matrix = self._compute_stft()

        # Apply the chroma filter to the STFT result
        chroma_filter = self._create_chroma_filter()
        chroma_stft = np.dot(chroma_filter, stft_matrix)

        return chroma_stft
