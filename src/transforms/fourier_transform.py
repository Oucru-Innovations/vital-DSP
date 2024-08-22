import numpy as np

class FourierTransform:
    """
    A class to perform Fourier Transform to analyze frequency content in signals like ECG/EEG.

    Methods:
    - compute_dft: Computes the Discrete Fourier Transform (DFT) of the signal.
    - compute_idft: Computes the Inverse Discrete Fourier Transform (IDFT) to reconstruct the signal.
    """

    def __init__(self, signal):
        """
        Initialize the FourierTransform class with the signal.

        Parameters:
        signal (numpy.ndarray): The input signal to be transformed.
        """
        self.signal = signal

    def compute_dft(self):
        """
        Compute the Discrete Fourier Transform (DFT) of the signal.

        Returns:
        numpy.ndarray: The frequency domain representation of the signal.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> ft = FourierTransform(signal)
        >>> frequency_content = ft.compute_dft()
        >>> print(frequency_content)
        """
        N = len(self.signal)
        dft = np.zeros(N, dtype=complex)
        for k in range(N):
            for n in range(N):
                angle = -2j * np.pi * k * n / N
                dft[k] += self.signal[n] * np.exp(angle)
        return dft

    def compute_idft(self, frequency_content):
        """
        Compute the Inverse Discrete Fourier Transform (IDFT) to reconstruct the signal.

        Parameters:
        frequency_content (numpy.ndarray): The frequency domain representation of the signal.

        Returns:
        numpy.ndarray: The time-domain signal reconstructed from its frequency components.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> ft = FourierTransform(signal)
        >>> frequency_content = ft.compute_dft()
        >>> reconstructed_signal = ft.compute_idft(frequency_content)
        >>> print(reconstructed_signal)
        """
        N = len(frequency_content)
        idft = np.zeros(N, dtype=complex)
        for n in range(N):
            for k in range(N):
                angle = 2j * np.pi * k * n / N
                idft[n] += frequency_content[k] * np.exp(angle)
        return idft / N
