import numpy as np
from vitalDSP.respiratory_analysis.preprocess.preprocess import preprocess_signal


def time_domain_rr(signal, sampling_rate, preprocess=None, **preprocess_kwargs):
    """
    Estimate respiratory rate using time-domain methods, particularly autocorrelation.

    Parameters
    ----------
    signal : numpy.ndarray
        The input respiratory signal.
    sampling_rate : float
        The sampling rate of the signal in Hz.
    preprocess : str, optional
        The preprocessing method to apply before estimation (e.g., "bandpass", "wavelet").
    preprocess_kwargs : dict, optional
        Additional arguments for the preprocessing function.

    Returns
    -------
    rr : float
        Estimated respiratory rate in breaths per minute.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> rr = time_domain_rr(signal, sampling_rate=100, preprocess='bandpass', lowcut=0.1, highcut=0.5)
    >>> print(rr)
    """
    # Apply preprocessing if specified
    if preprocess:
        signal = preprocess_signal(
            signal, sampling_rate, filter_type=preprocess, **preprocess_kwargs
        )

    # Normalize the signal
    signal = (signal - np.mean(signal)) / np.std(signal)

    # Compute the autocorrelation of the signal
    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]

    # Find the first peak in the autocorrelation
    peaks = np.diff(autocorr)
    rr_interval = np.argmax(peaks) / sampling_rate

    # Convert interval to breaths per minute
    rr = 60 / rr_interval

    return rr
