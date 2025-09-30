import numpy as np


class SignalQuality:
    """
    A class to assess the quality of signals using various metrics.

    This class provides methods to compute commonly used signal quality metrics such as
    Signal-to-Noise Ratio (SNR), Peak Signal-to-Noise Ratio (PSNR), and Mean Square Error (MSE).
    It can be used to evaluate the impact of noise or other processing on the original signal.

    Methods
    -------
    snr : function
        Computes the Signal-to-Noise Ratio.
    psnr : function
        Computes the Peak Signal-to-Noise Ratio.
    mse : function
        Computes the Mean Square Error between the original and processed signals.
    snr_of_noise : function
        Computes the Signal-to-Noise Ratio given a noise signal.

    Examples
    --------
    >>> import numpy as np
    >>> from vitalDSP.signal_quality_assessment.signal_quality import SignalQuality
    >>>
    >>> # Example 1: Basic signal quality assessment
    >>> original_signal = np.sin(np.linspace(0, 10, 1000))
    >>> noise = np.random.normal(0, 0.1, 1000)
    >>> noisy_signal = original_signal + noise
    >>> sq = SignalQuality(original_signal, noisy_signal)
    >>> print(f"SNR: {sq.snr():.2f} dB")
    >>> print(f"PSNR: {sq.psnr():.2f} dB")
    >>> print(f"MSE: {sq.mse():.6f}")
    >>>
    >>> # Example 2: Quality assessment with different noise levels
    >>> high_noise = np.random.normal(0, 0.5, 1000)
    >>> very_noisy_signal = original_signal + high_noise
    >>> sq_high_noise = SignalQuality(original_signal, very_noisy_signal)
    >>> print(f"High noise SNR: {sq_high_noise.snr():.2f} dB")
    >>>
    >>> # Example 3: Using noise signal directly
    >>> sq_noise = SignalQuality(original_signal)
    >>> snr_from_noise = sq_noise.snr_of_noise(noise)
    >>> print(f"SNR from noise: {snr_from_noise:.2f} dB")
    """

    def __init__(self, original_signal, processed_signal=None):
        """
        Initialize the SignalQuality class with the original and processed signals.

        Parameters
        ----------
        original_signal : numpy.ndarray
            The original, clean signal.
        processed_signal : numpy.ndarray, optional
            The processed or noisy signal for comparison. If not provided, some methods will
            require an alternative signal to compare against.

        Notes
        -----
        The original_signal is required for all computations, while processed_signal is optional
        and only needed for some metrics like SNR, PSNR, and MSE.
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

        SNR is a measure of signal quality that compares the level of the desired signal
        to the level of background noise. A higher SNR indicates a cleaner signal with
        less noise.

        Returns
        -------
        snr_value : float
            The SNR value in decibels (dB).

        Raises
        ------
        ValueError
            If processed_signal is not provided during initialization.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        >>> noisy_signal = signal + noise
        >>> sq = SignalQuality(signal, noisy_signal)
        >>> print(sq.snr())
        14.154543666201898
        """
        if self.processed_signal is None:
            raise ValueError("Processed signal is required to compute SNR.")

        signal_power = np.mean(self.original_signal**2)
        noise_power = np.mean((self.original_signal - self.processed_signal) ** 2)
        snr_value = 10 * np.log10(signal_power / noise_power)
        return snr_value

    def psnr(self):
        """
        Compute the Peak Signal-to-Noise Ratio (PSNR) of the signal.

        PSNR compares the maximum possible signal power to the noise power. It is commonly
        used in image and signal processing to assess the quality of signal reconstruction
        or compression techniques.

        Returns
        -------
        psnr_value : float
            The PSNR value in decibels (dB).

        Raises
        ------
        ValueError
            If processed_signal is not provided during initialization.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        >>> noisy_signal = signal + noise
        >>> sq = SignalQuality(signal, noisy_signal)
        >>> print(sq.psnr())
        26.020599913279625
        """
        if self.processed_signal is None:
            raise ValueError("Processed signal is required to compute PSNR.")

        mse_value = np.mean((self.original_signal - self.processed_signal) ** 2)
        max_signal = np.max(self.original_signal)
        psnr_value = 10 * np.log10(max_signal**2 / mse_value)
        return psnr_value

    def mse(self):
        """
        Compute the Mean Square Error (MSE) between the original and processed signals.

        MSE measures the average squared difference between the original and processed signals.
        It is commonly used to quantify the error introduced by noise or signal processing.

        Returns
        -------
        mse_value : float
            The MSE value.

        Raises
        ------
        ValueError
            If processed_signal is not provided during initialization.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        >>> noisy_signal = signal + noise
        >>> sq = SignalQuality(signal, noisy_signal)
        >>> print(sq.mse())
        0.010000000000000002
        """
        if self.processed_signal is None:
            raise ValueError("Processed signal is required to compute MSE.")

        mse_value = np.mean((self.original_signal - self.processed_signal) ** 2)
        return mse_value

    def snr_of_noise(self, noise_signal):
        """
        Compute the Signal-to-Noise Ratio (SNR) given a noise signal.

        This method calculates the SNR by comparing the power of the original signal
        to the power of the provided noise signal, without needing a processed signal.

        Parameters
        ----------
        noise_signal : numpy.ndarray
            The noise signal.

        Returns
        -------
        snr_value : float
            The SNR value in decibels (dB).

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        >>> sq = SignalQuality(signal)
        >>> print(sq.snr_of_noise(noise))
        20.0
        """
        if not isinstance(noise_signal, np.ndarray):
            noise_signal = np.array(noise_signal)

        signal_power = np.mean(self.original_signal**2)
        noise_power = np.mean(noise_signal**2)
        snr_value = 10 * np.log10(signal_power / noise_power)
        return snr_value
