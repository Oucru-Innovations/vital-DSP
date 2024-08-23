import numpy as np

class STFT:
    """
    A class to perform Short-Time Fourier Transform (STFT) to analyze time-varying signals.

    Methods:
    - compute_stft: Computes the STFT of the signal.
    """

    def __init__(self, signal, window_size=256, hop_size=128, n_fft=512):
        """
        Initialize the STFT class.

        Parameters:
        signal (numpy.ndarray): The input signal.
        window_size (int): The size of the window to apply.
        hop_size (int): The hop size between windows.
        n_fft (int): The number of FFT points.
        """
        self.signal = signal
        self.window_size = window_size
        self.hop_size = hop_size
        self.n_fft = n_fft
        self._validate_parameters()

    def _validate_parameters(self):
        """
        Validate and adjust parameters to prevent negative dimensions.
        """
        if self.window_size <= 0 or self.hop_size <= 0 or self.n_fft <= 0:
            raise ValueError("Window size, hop size, and n_fft must be positive integers.")
        if self.window_size > len(self.signal):
            raise ValueError("Window size cannot be larger than the signal length.")

    def compute_stft(self):
        """
        Compute the Short-Time Fourier Transform (STFT) of the signal.

        Returns:
        numpy.ndarray: The STFT of the signal.
        """
        n_windows = 1 + (len(self.signal) - self.window_size) // self.hop_size
        stft_matrix = np.zeros((self.n_fft // 2 + 1, n_windows), dtype=complex)

        for i in range(n_windows):
            start = i * self.hop_size
            end = start + self.window_size
            windowed_signal = self.signal[start:end] * np.hanning(self.window_size)
            
            # Ensure the windowed signal length matches n_fft for FFT computation
            if len(windowed_signal) < self.n_fft:
                windowed_signal = np.pad(windowed_signal, (0, self.n_fft - len(windowed_signal)), mode='constant')
            
            fft_result = np.fft.rfft(windowed_signal, n=self.n_fft)
            stft_matrix[:, i] = fft_result
        return stft_matrix