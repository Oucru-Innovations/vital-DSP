import numpy as np

class HilbertTransform:
    """
    A class to perform the Hilbert Transform, which is used to generate analytic signals.

    The Hilbert Transform is a key tool in signal processing, particularly for generating the analytic signal from a real-valued signal. The analytic signal is complex, with the original signal as the real part and the Hilbert transform as the imaginary part. This is particularly useful in applications like QRS detection in ECG signals, where phase and amplitude information are crucial.

    Methods
    -------
    compute_hilbert : method
        Computes the Hilbert Transform of the signal to obtain the analytic signal.
    """

    def __init__(self, signal):
        """
        Initialize the HilbertTransform class with the input signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The real-valued input signal to be transformed.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100))
        >>> ht = HilbertTransform(signal)
        >>> print(ht.signal)
        """
        self.signal = signal

    def compute_hilbert(self):
        """
        Compute the Hilbert Transform to obtain the analytic signal.

        The Hilbert Transform is applied in the frequency domain by first taking the Fourier transform of the input signal, modifying the Fourier coefficients to zero out the negative frequencies, and then applying the inverse Fourier transform. This process effectively shifts the signal in such a way that the imaginary part represents the phase information, while the real part remains the original signal.

        Returns
        -------
        numpy.ndarray
            The analytic signal with both real and imaginary components, where the real part is the original signal and the imaginary part is the Hilbert transform.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> ht = HilbertTransform(signal)
        >>> analytic_signal = ht.compute_hilbert()
        >>> print(analytic_signal)

        Notes
        -----
        The analytic signal is often used in applications where the instantaneous amplitude and phase of the signal are required, such as in the detection of QRS complexes in ECG signals, modulation, and demodulation in communications, and envelope detection in various signal processing tasks.
        """
        N = len(self.signal)
        H = np.zeros(N)

        # Construct the Hilbert Transform multiplier in the frequency domain
        if N % 2 == 0:
            H[0] = 1
            H[N // 2] = 1
            H[1 : N // 2] = 2
        else:
            H[0] = 1
            H[1 : (N + 1) // 2] = 2

        # Apply the Hilbert Transform using the FFT
        hilbert_signal = np.fft.ifft(np.fft.fft(self.signal) * H)
        return hilbert_signal
