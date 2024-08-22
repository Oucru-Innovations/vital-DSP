import numpy as np

class ChromaSTFT:
    """
    A class to compute Chroma Short-Time Fourier Transform (Chroma STFT) to analyze harmonic content in speech.

    Methods:
    - compute_chroma_stft: Computes the Chroma STFT of the signal.
    """

    def __init__(self, signal, sample_rate=16000, n_chroma=12, n_fft=2048):
        """
        Initialize the ChromaSTFT class with the signal.

        Parameters:
        signal (numpy.ndarray): The input audio signal.
        sample_rate (int): The sample rate of the signal.
        n_chroma (int): The number of chroma bins.
        n_fft (int): The FFT size.
        """
        self.signal = signal
        self.sample_rate = sample_rate
        self.n_chroma = n_chroma
        self.n_fft = n_fft

    def compute_chroma_stft(self):
        """
        Compute the Chroma Short-Time Fourier Transform (Chroma STFT) of the signal.

        Returns:
        numpy.ndarray: The Chroma STFT of the signal.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> chroma_stft = ChromaSTFT(signal)
        >>> chroma_stft_result = chroma_stft.compute_chroma_stft()
        >>> print(chroma_stft_result)
        """
        stft_matrix = np.abs(np.fft.fft(self.signal, n=self.n_fft))[:self.n_fft//2+1]
        chroma_filter = np.random.rand(self.n_chroma, self.n_fft//2+1)  # Simulated chroma filter banks
        chroma_stft = np.dot(chroma_filter, stft_matrix)
        return chroma_stft
