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
        return np.real(np.fft.fft(self.signal * 2))[:len(self.signal)//2]

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
        full_spectrum = np.concatenate((dct_coefficients, dct_coefficients[::-1]))
        return np.real(np.fft.ifft(full_spectrum)) / 2
