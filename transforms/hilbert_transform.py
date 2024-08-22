import numpy as np

class HilbertTransform:
    """
    A class to perform the Hilbert Transform to generate analytic signals, useful for QRS detection in ECG.

    Methods:
    - compute_hilbert: Computes the Hilbert Transform of the signal.
    """

    def __init__(self, signal):
        """
        Initialize the HilbertTransform class with the signal.

        Parameters:
        signal (numpy.ndarray): The input signal to be transformed.
        """
        self.signal = signal

    def compute_hilbert(self):
        """
        Compute the Hilbert Transform to obtain the analytic signal.

        Returns:
        numpy.ndarray: The analytic signal with real and imaginary components.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> ht = HilbertTransform(signal)
        >>> analytic_signal = ht.compute_hilbert()
        >>> print(analytic_signal)
        """
        N = len(self.signal)
        H = np.zeros(N)
        if N % 2 == 0:
            H[0] = 1
            H[N//2] = 1
            H[1:N//2] = 2
        else:
            H[0] = 1
            H[1:(N+1)//2] = 2

        hilbert_signal = np.fft.ifft(np.fft.fft(self.signal) * H)
        return hilbert_signal
