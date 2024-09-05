import numpy as np
from scipy.signal import welch
from vitalDSP.preprocess.preprocess import preprocess_signal


def frequency_domain_rr(
    signal, sampling_rate, preprocess=None, nperseg=None, **preprocess_kwargs
):
    """
    Estimate respiratory rate using frequency-domain methods, particularly the Welch method.

    Parameters
    ----------
    signal : numpy.ndarray
        The input respiratory signal.
    sampling_rate : float
        The sampling rate of the signal in Hz.
    preprocess : str, optional
        The preprocessing method to apply before estimation (e.g., "bandpass", "wavelet").
    nperseg : int, optional
        Length of each segment for Welch method. If None, it defaults to 256.
    preprocess_kwargs : dict, optional
        Additional arguments for the preprocessing function.

    Returns
    -------
    rr : float
        Estimated respiratory rate in breaths per minute.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> rr = frequency_domain_rr(signal, sampling_rate=100, preprocess='bandpass', lowcut=0.1, highcut=0.5)
    >>> print(rr)
    """
    # Apply preprocessing if specified
    if preprocess:
        signal = preprocess_signal(
            signal, sampling_rate, filter_type=preprocess, **preprocess_kwargs
        )

    # Compute the power spectral density using the Welch method
    freqs, psd = welch(signal, fs=sampling_rate, nperseg=nperseg)

    # Find the peak frequency in the power spectral density
    peak_freq = freqs[np.argmax(psd)]

    # Convert frequency to breaths per minute (bpm)
    return peak_freq * 60
