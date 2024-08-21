import numpy as np

class SignalQuality:
    """
    A class to assess the quality of signals using various metrics.
    
    Methods:
    - snr: Computes the Signal-to-Noise Ratio.
    - psnr: Computes the Peak Signal-to-Noise Ratio.
    - mse: Computes the Mean Square Error between the original and processed signals.
    """

    def __init__(self, original_signal, processed_signal=None):
        """
        Initialize the SignalQuality class with the original and processed signals.

        Parameters:
        original_signal (numpy.ndarray): The original signal.
        processed_signal (numpy.ndarray, optional): The processed or noisy signal for comparison.
        """
        if not isinstance(original_signal, np.ndarray):
            original_signal = np.array(original_signal)
        self.original_signal = original_signal

        if processed_signal is not None:
            if not isinstance(processed_signal, np.ndarray):
                processed_signal = np.array(processed_signal)
            self.processed_signal = processed_signal
        else:
            self.processed_signal = None

    def snr(self):
        """
        Compute the Signal-to-Noise Ratio (SNR) of the signal.

        SNR is a measure of signal quality comparing the level of the signal to the level of noise.

        Returns:
        float: The SNR value in decibels (dB).

        Example:
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        >>> noisy_signal = signal + noise
        >>> sq = SignalQuality(signal, noisy_signal)
        >>> print(sq.snr())
        """
        if self.processed_signal is None:
            raise ValueError("Processed signal is required to compute SNR.")

        signal_power = np.mean(self.original_signal ** 2)
        noise_power = np.mean((self.original_signal - self.processed_signal) ** 2)
        snr_value = 10 * np.log10(signal_power / noise_power)
        return snr_value

    def psnr(self):
        """
        Compute the Peak Signal-to-Noise Ratio (PSNR) of the signal.

        PSNR compares the maximum possible signal power to the noise power, which can be useful in image and signal processing.

        Returns:
        float: The PSNR value in decibels (dB).

        Example:
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        >>> noisy_signal = signal + noise
        >>> sq = SignalQuality(signal, noisy_signal)
        >>> print(sq.psnr())
        """
        if self.processed_signal is None:
            raise ValueError("Processed signal is required to compute PSNR.")

        mse_value = np.mean((self.original_signal - self.processed_signal) ** 2)
        max_signal = np.max(self.original_signal)
        psnr_value = 10 * np.log10(max_signal ** 2 / mse_value)
        return psnr_value

    def mse(self):
        """
        Compute the Mean Square Error (MSE) between the original and processed signals.

        MSE measures the average squared difference between the original and processed signals.

        Returns:
        float: The MSE value.

        Example:
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        >>> noisy_signal = signal + noise
        >>> sq = SignalQuality(signal, noisy_signal)
        >>> print(sq.mse())
        """
        if self.processed_signal is None:
            raise ValueError("Processed signal is required to compute MSE.")

        mse_value = np.mean((self.original_signal - self.processed_signal) ** 2)
        return mse_value

    def snr_of_noise(self, noise_signal):
        """
        Compute the Signal-to-Noise Ratio (SNR) given a noise signal.

        Parameters:
        noise_signal (numpy.ndarray): The noise signal.

        Returns:
        float: The SNR value in decibels (dB).

        Example:
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        >>> sq = SignalQuality(signal)
        >>> print(sq.snr_of_noise(noise))
        """
        if not isinstance(noise_signal, np.ndarray):
            noise_signal = np.array(noise_signal)

        signal_power = np.mean(self.original_signal ** 2)
        noise_power = np.mean(noise_signal ** 2)
        snr_value = 10 * np.log10(signal_power / noise_power)
        return snr_value
