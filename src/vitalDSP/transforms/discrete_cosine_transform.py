import numpy as np
from scipy.fftpack import dct, idct

class DiscreteCosineTransform:
    """
    A class to perform Discrete Cosine Transform (DCT) and its inverse (IDCT).

    The DCT is widely used in signal processing, particularly for data compression (such as JPEG image compression)
    and feature extraction. It transforms a signal from the time domain into the frequency domain, where most of the
    signal information is compacted into a few coefficients. The inverse DCT (IDCT) reconstructs the signal from its
    DCT coefficients.

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

    def compute_dct(self, norm='ortho'):
        """
        Compute the Discrete Cosine Transform (DCT) of the signal.

        The DCT converts the input signal from the time domain to the frequency domain, emphasizing the low-frequency
        components of the signal. It is particularly effective for compressing signals because it tends to concentrate
        most of the signal's energy into a few low-frequency components.

        Parameters
        ----------
        norm : str, optional
            Normalization type. Default is 'ortho' for orthogonal DCT, which is usually recommended.

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
        # Apply a window function to reduce spectral leakage
        windowed_signal = self.signal * np.hamming(len(self.signal))
        dct_coefficients = dct(windowed_signal, norm=norm)
        return dct_coefficients

    def compute_idct(self, dct_coefficients, norm='ortho'):
        """
        Compute the Inverse Discrete Cosine Transform (IDCT) to reconstruct the signal.

        The IDCT converts the DCT coefficients back into the time domain, reconstructing the original signal.
        This process is essential in applications like image compression, where the signal is compressed using DCT
        and then decompressed back to its original form using IDCT.

        Parameters
        ----------
        dct_coefficients : numpy.ndarray
            The DCT coefficients of the signal.
        norm : str, optional
            Normalization type. Default is 'ortho' for orthogonal DCT.

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
        reconstructed_signal = idct(dct_coefficients, norm=norm)
        return reconstructed_signal

    def compress_signal(self, threshold=0.1):
        """
        Compress the signal by zeroing out small DCT coefficients.

        This method removes high-frequency components that are below the threshold, which often represent noise,
        improving the signal's reconstruction quality and reducing its size.

        Parameters
        ----------
        threshold : float, optional
            The threshold below which DCT coefficients are set to zero. Default is 0.1.

        Returns
        -------
        numpy.ndarray
            The compressed DCT coefficients.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> dct = DiscreteCosineTransform(signal)
        >>> compressed_coefficients = dct.compress_signal(threshold=0.05)
        >>> print(compressed_coefficients)
        """
        dct_coefficients = self.compute_dct()
        max_coeff = np.max(np.abs(dct_coefficients))
        compressed_coefficients = np.where(np.abs(dct_coefficients) > threshold * max_coeff, dct_coefficients, 0)
        return compressed_coefficients
