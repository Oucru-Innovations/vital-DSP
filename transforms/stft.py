import numpy as np

class STFT:
    """
    A class to perform Short-Time Fourier Transform (STFT) to analyze time-varying signals.

    Methods:
    - compute_stft: Computes the STFT of the signal.
    """

    def __init__(self, signal, window_size=256, hop_size=128):
        """
        Initialize the STFT class with the signal.

        Parameters:
        signal (numpy.ndarray): The input signal to be analyzed.
        window_size (int): The size of the window for STFT.
        hop_size (int): The hop size (step size) for STFT.
        """
        self.signal = signal
        self.window_size = window_size
        self.hop_size = hop_size

    def compute_stft(self):
        """
        Compute the Short-Time Fourier Transform (STFT) of the signal.

        Returns:
        numpy.ndarray: The STFT of the signal, with time along one axis and frequency along the other.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> stft = STFT(signal)
        >>> stft_result = stft.compute_stft()
        >>> print(stft_result)
        """
        n_windows = (len(self.signal) - self.window_size) // self.hop_size + 1
        stft_matrix = np.zeros((self.window_size, n_windows), dtype=complex)

        for i in range(n_windows):
            windowed_signal = self.signal[i * self.hop_size:i * self.hop_size + self.window_size] * np.hanning(self.window_size)
            stft_matrix[:, i] = np.fft.fft(windowed_signal)
        
        return stft_matrix
