import numpy as np


class FourierTransform:
    """
    A class to perform Fourier Transform for analyzing the frequency content in signals such as ECG/EEG.

    The Fourier Transform is a mathematical technique that transforms a time-domain signal into its constituent
    frequencies, providing insights into the signal's frequency content. This class allows for both the computation
    of the Discrete Fourier Transform (DFT) and the Inverse Discrete Fourier Transform (IDFT), making it possible
    to analyze and reconstruct signals.

    Methods
    -------
    compute_dft : method
        Computes the Discrete Fourier Transform (DFT) of the signal.
    compute_idft : method
        Computes the Inverse Discrete Fourier Transform (IDFT) to reconstruct the signal.
    """

    def __init__(self, signal):
        """
        Initialize the FourierTransform class with the input signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be transformed. The signal should be a 1D array representing time-domain data,
            such as an ECG or EEG signal.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> ft = FourierTransform(signal)
        >>> print(ft.signal)
        """
        self.signal = signal

    def compute_dft(self):
        """
        Compute the Discrete Fourier Transform (DFT) of the input signal.

        The DFT converts the time-domain signal into the frequency domain, allowing for the analysis of its frequency
        components. This is particularly useful in identifying periodicities, filtering, and spectral analysis of
        biomedical signals like ECG and EEG.

        Returns
        -------
        numpy.ndarray
            The frequency domain representation of the signal, where each element corresponds to a specific
            frequency component.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> ft = FourierTransform(signal)
        >>> frequency_content = ft.compute_dft()
        >>> print(frequency_content)
        """
        # Apply a window function to reduce spectral leakage
        windowed_signal = self.signal * np.hamming(len(self.signal))
        dft = np.fft.fft(windowed_signal)
        return dft

    def compute_idft(self, frequency_content):
        """
        Compute the Inverse Discrete Fourier Transform (IDFT) to reconstruct the time-domain signal.

        The IDFT converts the frequency-domain data back into the time domain, reconstructing the original signal
        from its frequency components. This is useful for understanding how different frequency components contribute
        to the overall signal and for signal reconstruction after processing in the frequency domain.

        Parameters
        ----------
        frequency_content : numpy.ndarray
            The frequency domain representation of the signal, as obtained from the DFT.

        Returns
        -------
        numpy.ndarray
            The time-domain signal reconstructed from its frequency components.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> ft = FourierTransform(signal)
        >>> frequency_content = ft.compute_dft()
        >>> reconstructed_signal = ft.compute_idft(frequency_content)
        >>> print(reconstructed_signal)
        """
        idft = np.fft.ifft(frequency_content)
        return (
            idft.real
        )  # Return the real part, as the imaginary part should be negligible

    def filter_frequencies(self, low_cutoff=None, high_cutoff=None, fs=1.0):
        """
        Apply a bandpass filter in the frequency domain.

        Parameters
        ----------
        low_cutoff : float, optional
            The low cutoff frequency. Frequencies below this will be zeroed out.
        high_cutoff : float, optional
            The high cutoff frequency. Frequencies above this will be zeroed out.
        fs : float, optional
            The sampling frequency of the signal. Default is 1.0.

        Returns
        -------
        numpy.ndarray
            The filtered signal in the time domain.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> ft = FourierTransform(signal)
        >>> filtered_signal = ft.filter_frequencies(low_cutoff=0.5, high_cutoff=2.0, fs=10)
        >>> print(filtered_signal)
        """
        dft = self.compute_dft()
        freqs = np.fft.fftfreq(len(self.signal), d=1 / fs)

        if low_cutoff is not None:
            dft[np.abs(freqs) < low_cutoff] = 0
        if high_cutoff is not None:
            dft[np.abs(freqs) > high_cutoff] = 0

        filtered_signal = self.compute_idft(dft)
        return filtered_signal
