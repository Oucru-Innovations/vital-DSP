import numpy as np


class DiscreteCosineTransform:
    """
    A class to perform Discrete Cosine Transform (DCT) used in compression and feature extraction.

    Methods:
    - compute_dct: Computes the DCT of the signal.
    - compute_idct: Computes the Inverse DCT to reconstruct the signal.
    """

    def __init__(self, signal):
        """
        Initialize the DiscreteCosineTransform class with the signal.

        Parameters:
        signal (numpy.ndarray): The input signal to be transformed.
        """
        self.signal = signal

    def compute_dct(self):
        """
        Compute the Discrete Cosine Transform (DCT) of the signal.

        Returns:
        numpy.ndarray: The DCT coefficients of the signal.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> dct = DiscreteCosineTransform(signal)
        >>> dct_coefficients = dct.compute_dct()
        >>> print(dct_coefficients)
        """
        N = len(self.signal)
        result = np.zeros(N, dtype=np.float64)
        for k in range(N):
            sum_value = 0
            for n in range(N):
                sum_value += self.signal[n] * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
            result[k] = sum_value * np.sqrt(2 / N)
        result[0] /= np.sqrt(2)
        return result

    def compute_idct(self, dct_coefficients):
        """
        Compute the Inverse Discrete Cosine Transform (IDCT) to reconstruct the signal.

        Parameters:
        dct_coefficients (numpy.ndarray): The DCT coefficients of the signal.

        Returns:
        numpy.ndarray: The time-domain signal reconstructed from its DCT coefficients.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> dct = DiscreteCosineTransform(signal)
        >>> dct_coefficients = dct.compute_dct()
        >>> reconstructed_signal = dct.compute_idct(dct_coefficients)
        >>> print(reconstructed_signal)
        """
        N = len(dct_coefficients)
        signal_reconstructed = np.zeros(N, dtype=np.float64)
        for n in range(N):
            sum_value = dct_coefficients[0] / np.sqrt(2)
            for k in range(1, N):
                sum_value += dct_coefficients[k] * np.cos(
                    np.pi * k * (2 * n + 1) / (2 * N)
                )
            signal_reconstructed[n] = sum_value * np.sqrt(2 / N)
        return signal_reconstructed
