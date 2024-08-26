import numpy as np

class DiscreteCosineTransform:
    """
    A class to perform Discrete Cosine Transform (DCT) and its inverse (IDCT).

    The DCT is widely used in signal processing, particularly for data compression (such as JPEG image compression) and feature extraction. It transforms a signal from the time domain into the frequency domain, where most of the signal information is compacted into a few coefficients. The inverse DCT (IDCT) reconstructs the signal from its DCT coefficients.

    Methods
    -------
    compute_dct : method
        Computes the DCT of the signal.
    compute_idct : method
        Computes the Inverse DCT to reconstruct the signal.
    """

    def __init__(self, signal):
        """
        Initialize the DiscreteCosineTransform class with the signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be transformed.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> dct = DiscreteCosineTransform(signal)
        >>> print(dct.signal)
        """
        self.signal = signal

    def compute_dct(self):
        """
        Compute the Discrete Cosine Transform (DCT) of the signal.

        The DCT converts the input signal from the time domain to the frequency domain, emphasizing the low-frequency components of the signal. It is particularly effective for compressing signals because it tends to concentrate most of the signal's energy into a few low-frequency components.

        Returns
        -------
        numpy.ndarray
            The DCT coefficients of the signal.

        Examples
        --------
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

        The IDCT converts the DCT coefficients back into the time domain, reconstructing the original signal. This process is essential in applications like image compression, where the signal is compressed using DCT and then decompressed back to its original form using IDCT.

        Parameters
        ----------
        dct_coefficients : numpy.ndarray
            The DCT coefficients of the signal.

        Returns
        -------
        numpy.ndarray
            The time-domain signal reconstructed from its DCT coefficients.

        Examples
        --------
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
