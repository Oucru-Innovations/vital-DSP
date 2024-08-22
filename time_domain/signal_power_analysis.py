import numpy as np

class SignalPowerAnalysis:
    """
    A comprehensive class for measuring the power of physiological signals.

    Methods:
    - compute_rmse: Computes the Root Mean Square Error (RMSE) of the signal.
    - compute_mean_square: Computes the mean square value of the signal.
    - compute_total_power: Computes the total power of the signal.
    - compute_peak_power: Computes the peak power of the signal.
    - compute_snr: Computes the Signal-to-Noise Ratio (SNR) of the signal.
    - compute_psd: Computes the Power Spectral Density (PSD) of the signal.
    """

    def __init__(self, signal):
        """
        Initialize the SignalPowerAnalysis class with the signal.

        Parameters:
        signal (numpy.ndarray): The input physiological signal.
        """
        self.signal = signal

    def compute_rmse(self):
        """
        Compute the Root Mean Square Error (RMSE) of the signal.

        Returns:
        float: The RMSE of the signal.
        """
        rmse = np.sqrt(np.mean(self.signal ** 2))
        return rmse

    def compute_mean_square(self):
        """
        Compute the mean square value of the signal.

        Returns:
        float: The mean square value of the signal.
        """
        mean_square = np.mean(self.signal ** 2)
        return mean_square

    def compute_total_power(self):
        """
        Compute the total power of the signal.

        Returns:
        float: The total power of the signal.
        """
        total_power = np.sum(self.signal ** 2) / len(self.signal)
        return total_power

    def compute_peak_power(self):
        """
        Compute the peak power of the signal.

        Returns:
        float: The peak power of the signal.
        """
        peak_power = np.max(self.signal ** 2)
        return peak_power

    def compute_snr(self, noise_signal):
        """
        Compute the Signal-to-Noise Ratio (SNR) of the signal.

        Parameters:
        noise_signal (numpy.ndarray): The noise component of the signal.

        Returns:
        float: The SNR of the signal in decibels (dB).
        """
        signal_power = np.mean(self.signal ** 2)
        noise_power = np.mean(noise_signal ** 2)
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def compute_psd(self, fs=1.0, nperseg=256):
        """
        Compute the Power Spectral Density (PSD) of the signal.

        Parameters:
        fs (float): The sampling frequency of the signal.
        nperseg (int): Length of each segment for PSD computation.

        Returns:
        numpy.ndarray: Frequency array.
        numpy.ndarray: Power Spectral Density values.
        """
        freqs = np.fft.rfftfreq(len(self.signal), d=1/fs)
        psd = np.abs(np.fft.rfft(self.signal)) ** 2 / len(self.signal)
        return freqs, psd

    def compute_band_power(self, band, fs=1.0):
        """
        Compute the power of the signal within a specific frequency band.

        Parameters:
        band (tuple): Frequency band as a tuple (low_freq, high_freq).
        fs (float): Sampling frequency of the signal.

        Returns:
        float: The power within the specified band.
        """
        freqs, psd = self.compute_psd(fs)
        band_power = np.trapz(psd[(freqs >= band[0]) & (freqs <= band[1])], freqs[(freqs >= band[0]) & (freqs <= band[1])])
        return band_power

    def compute_energy(self):
        """
        Compute the total energy of the signal.

        Returns:
        float: The total energy of the signal.
        """
        energy = np.sum(self.signal ** 2)
        return energy
