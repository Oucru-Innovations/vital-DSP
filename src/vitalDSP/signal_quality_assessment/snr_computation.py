import numpy as np


def snr_power_ratio(signal, noise):
    """
    Calculate the SNR using the power ratio method.

    Parameters
    ----------
    signal : numpy.ndarray
        The original clean signal.
    noise : numpy.ndarray
        The noise signal.

    Returns
    -------
    snr : float
        The SNR in dB.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> noise = 0.1 * np.random.normal(size=signal.shape)
    >>> snr = snr_power_ratio(signal, noise)
    >>> print(snr)
    """
    power_signal = np.mean(np.square(signal))
    power_noise = np.mean(np.square(noise))

    if power_noise == 0:
        raise ZeroDivisionError("Noise power is zero, cannot compute SNR.")

    snr = 10 * np.log10(power_signal / power_noise)
    return snr


def snr_peak_to_peak(signal, noise):
    """
    Calculate the SNR using the peak-to-peak amplitude method.

    Parameters
    ----------
    signal : numpy.ndarray
        The original clean signal.
    noise : numpy.ndarray
        The noise signal.

    Returns
    -------
    snr : float
        The SNR in dB.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> noise = 0.1 * np.random.normal(size=signal.shape)
    >>> snr = snr_peak_to_peak(signal, noise)
    >>> print(snr)
    """
    signal_ptp = np.ptp(signal)
    noise_ptp = np.ptp(noise)

    if noise_ptp == 0:
        raise ZeroDivisionError(
            "Noise peak-to-peak amplitude is zero, cannot compute SNR."
        )

    snr = 20 * np.log10(signal_ptp / noise_ptp)
    return snr


def snr_mean_square(signal, noise):
    """
    Calculate the SNR using the mean square of the signal and noise.

    Parameters
    ----------
    signal : numpy.ndarray
        The original clean signal.
    noise : numpy.ndarray
        The noise signal.

    Returns
    -------
    snr : float
        The SNR in dB.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> noise = 0.1 * np.random.normal(size=signal.shape)
    >>> snr = snr_mean_square(signal, noise)
    >>> print(snr)
    """
    mean_square_signal = np.mean(signal**2)
    mean_square_noise = np.mean(noise**2)

    if mean_square_noise == 0:
        raise ZeroDivisionError("Noise mean square is zero, cannot compute SNR.")

    snr = 10 * np.log10(mean_square_signal / mean_square_noise)
    return snr


def crest_factor(signal):
    """
    Calculate the crest factor of the signal, which is the ratio of the peak amplitude to the RMS value.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal.

    Returns
    -------
    crest_factor : float
        The crest factor of the signal.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> cf = crest_factor(signal)
    >>> print(cf)
    """
    peak_amplitude = np.max(np.abs(signal))
    rms_value = np.sqrt(np.mean(signal**2))

    if rms_value == 0:
        raise ZeroDivisionError("RMS value is zero, cannot compute crest factor.")

    crest_factor = peak_amplitude / rms_value
    return crest_factor


def harmonic_distortion(signal, fundamental_freq, sampling_rate, harmonics=5):
    """
    Calculate the total harmonic distortion (THD) of the signal.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal.
    fundamental_freq : float
        The fundamental frequency of the signal.
    sampling_rate : float
        The sampling rate of the signal.
    harmonics : int, optional (default=5)
        The number of harmonics to include in the THD calculation.

    Returns
    -------
    thd : float
        The total harmonic distortion as a percentage.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 50 * np.arange(0, 1, 1/1000))
    >>> thd = harmonic_distortion(signal, fundamental_freq=50, sampling_rate=1000)
    >>> print(thd)
    """
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)
    fft_spectrum = np.abs(np.fft.fft(signal))

    fundamental_amplitude = np.max(
        fft_spectrum[
            np.where(
                (freqs >= fundamental_freq - 0.1) & (freqs <= fundamental_freq + 0.1)
            )
        ]
    )

    if fundamental_amplitude == 0:
        return 0  # If fundamental amplitude is zero, THD cannot be computed.

    harmonic_amplitudes = []
    for i in range(2, harmonics + 1):
        harmonic_freq = i * fundamental_freq
        harmonic_amplitude = np.max(
            fft_spectrum[
                np.where(
                    (freqs >= harmonic_freq - 0.1) & (freqs <= harmonic_freq + 0.1)
                )
            ]
        )
        harmonic_amplitudes.append(harmonic_amplitude)

    thd = np.sqrt(np.sum(np.square(harmonic_amplitudes))) / fundamental_amplitude * 100
    return thd


def signal_to_noise_and_distortion_ratio(signal, noise):
    """
    Calculate the Signal-to-Noise and Distortion Ratio (SINAD).

    Parameters
    ----------
    signal : numpy.ndarray
        The original clean signal.
    noise : numpy.ndarray
        The noise and distortion components of the signal.

    Returns
    -------
    sinad : float
        The SINAD in dB.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> noise = 0.1 * np.random.normal(size=signal.shape)
    >>> sinad = signal_to_noise_and_distortion_ratio(signal, noise)
    >>> print(sinad)
    """
    rms_signal = np.sqrt(np.mean(signal**2))
    rms_noise = np.sqrt(np.mean(noise**2))

    if rms_noise == 0:
        raise ZeroDivisionError("Noise RMS is zero, cannot compute SINAD.")

    sinad = 20 * np.log10(rms_signal / rms_noise)
    return sinad


def signal_to_noise_and_interference_ratio(signal, interference):
    """
    Calculate the Signal-to-Noise and Interference Ratio (SNIR).

    Parameters
    ----------
    signal : numpy.ndarray
        The original clean signal.
    interference : numpy.ndarray
        The interference in the signal.

    Returns
    -------
    snir : float
        The SNIR in dB.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> interference = 0.1 * np.random.normal(size=signal.shape)
    >>> snir = signal_to_noise_and_interference_ratio(signal, interference)
    >>> print(snir)
    """
    rms_signal = np.sqrt(np.mean(signal**2))
    rms_interference = np.sqrt(np.mean(interference**2))

    if rms_interference == 0:
        raise ZeroDivisionError("Interference RMS is zero, cannot compute SNIR.")

    snir = 20 * np.log10(rms_signal / rms_interference)
    return snir
