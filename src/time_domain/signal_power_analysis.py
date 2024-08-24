import numpy as np

class SignalPowerAnalysis:
    """
    A comprehensive class for measuring the power of physiological signals.

    This class provides methods to compute various power-related metrics, including
    RMSE, mean square value, total power, peak power, SNR, PSD, band power, and
    signal energy. These metrics are essential for analyzing the characteristics of
    physiological signals in applications such as ECG, EEG, and PPG analysis.

    Methods
    -------
    compute_rmse :
        Computes the Root Mean Square Error (RMSE) of the signal.
    compute_mean_square :
        Computes the mean square value of the signal.
    compute_total_power :
        Computes the total power of the signal.
    compute_peak_power :
        Computes the peak power of the signal.
    compute_snr :
        Computes the Signal-to-Noise Ratio (SNR) of the signal.
    compute_psd :
        Computes the Power Spectral Density (PSD) of the signal.
    compute_band_power :
        Computes the power of the signal within a specific frequency band.
    compute_energy :
        Computes the total energy of the signal.
    """

    def __init__(self, signal):
        """
        Initialize the SignalPowerAnalysis class with the input signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input physiological signal to be analyzed.
        """
        self.signal = signal

    def compute_rmse(self):
        """
        Compute the Root Mean Square Error (RMSE) of the signal.

        RMSE is a measure of the magnitude of the signal and is commonly used to
        assess the variability or "loudness" of a signal.

        Returns
        -------
        float
            The RMSE of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> spa = SignalPowerAnalysis(signal)
        >>> rmse = spa.compute_rmse()
        >>> print(rmse)
        3.3166247903554
        """
        rmse = np.sqrt(np.mean(self.signal**2))
        return rmse

    def compute_mean_square(self):
        """
        Compute the mean square value of the signal.

        The mean square value is a measure of the average power of the signal over time.

        Returns
        -------
        float
            The mean square value of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> spa = SignalPowerAnalysis(signal)
        >>> mean_square = spa.compute_mean_square()
        >>> print(mean_square)
        11.0
        """
        mean_square = np.mean(self.signal**2)
        return mean_square

    def compute_total_power(self):
        """
        Compute the total power of the signal.

        Total power represents the sum of the squared amplitudes normalized by the
        length of the signal, giving an indication of the overall power content.

        Returns
        -------
        float
            The total power of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> spa = SignalPowerAnalysis(signal)
        >>> total_power = spa.compute_total_power()
        >>> print(total_power)
        11.0
        """
        total_power = np.sum(self.signal**2) / len(self.signal)
        return total_power

    def compute_peak_power(self):
        """
        Compute the peak power of the signal.

        Peak power refers to the maximum instantaneous power, which is the highest value
        of the squared signal.

        Returns
        -------
        float
            The peak power of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> spa = SignalPowerAnalysis(signal)
        >>> peak_power = spa.compute_peak_power()
        >>> print(peak_power)
        25.0
        """
        peak_power = np.max(self.signal**2)
        return peak_power

    def compute_snr(self, noise_signal):
        """
        Compute the Signal-to-Noise Ratio (SNR) of the signal.

        SNR is a measure of how much the signal power exceeds the noise power, expressed
        in decibels (dB). A higher SNR indicates a cleaner signal with less noise.

        Parameters
        ----------
        noise_signal : numpy.ndarray
            The noise component of the signal.

        Returns
        -------
        float
            The SNR of the signal in decibels (dB).

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> noise_signal = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        >>> spa = SignalPowerAnalysis(signal)
        >>> snr = spa.compute_snr(noise_signal)
        >>> print(snr)
        20.0
        """
        signal_power = np.mean(self.signal**2)
        noise_power = np.mean(noise_signal**2)
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def compute_psd(self, fs=1.0, nperseg=256):
        """
        Compute the Power Spectral Density (PSD) of the signal.

        PSD represents the power distribution of a signal as a function of frequency,
        providing insights into the frequency content of the signal.

        Parameters
        ----------
        fs : float, optional
            The sampling frequency of the signal. Default is 1.0.
        nperseg : int, optional
            Length of each segment for PSD computation. Default is 256.

        Returns
        -------
        freqs : numpy.ndarray
            Array of frequency bins.
        psd : numpy.ndarray
            Power Spectral Density values.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> spa = SignalPowerAnalysis(signal)
        >>> freqs, psd = spa.compute_psd(fs=10.0, nperseg=5)
        >>> print(freqs)
        >>> print(psd)
        """
        freqs = np.fft.rfftfreq(len(self.signal), d=1 / fs)
        psd = np.abs(np.fft.rfft(self.signal)) ** 2 / len(self.signal)
        return freqs, psd

    def compute_band_power(self, band, fs=1.0):
        """
        Compute the power of the signal within a specific frequency band.

        Band power is useful in applications like EEG analysis, where power in specific
        frequency bands (e.g., alpha, beta, delta) is of interest.

        Parameters
        ----------
        band : tuple
            Frequency band as a tuple (low_freq, high_freq).
        fs : float, optional
            The sampling frequency of the signal. Default is 1.0.

        Returns
        -------
        float
            The power within the specified band.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> spa = SignalPowerAnalysis(signal)
        >>> band_power = spa.compute_band_power((1, 3), fs=10.0)
        >>> print(band_power)
        """
        freqs, psd = self.compute_psd(fs)
        band_power = np.trapz(
            psd[(freqs >= band[0]) & (freqs <= band[1])],
            freqs[(freqs >= band[0]) & (freqs <= band[1])],
        )
        return band_power

    def compute_energy(self):
        """
        Compute the total energy of the signal.

        The energy of a signal is the sum of the squared amplitudes and is related to
        the power by the length of the signal.

        Returns
        -------
        float
            The total energy of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> spa = SignalPowerAnalysis(signal)
        >>> energy = spa.compute_energy()
        >>> print(energy)
        55.0
        """
        energy = np.sum(self.signal**2)
        return energy
