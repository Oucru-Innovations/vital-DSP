import numpy as np
from vitalDSP.respiratory_analysis.preprocess.preprocess import preprocess_signal

def fft_based_rr(signal, sampling_rate, preprocess=None, **preprocess_kwargs):
    """
    Estimate respiratory rate using the FFT (Fast Fourier Transform) method.

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
    >>> rr = fft_based_rr(signal, sampling_rate=100, preprocess='bandpass', lowcut=0.1, highcut=0.5)
    >>> print(rr)
    """
    if preprocess:
        signal = preprocess_signal(signal, sampling_rate, filter_type=preprocess, **preprocess_kwargs)

    n = len(signal)
    freq = np.fft.fftfreq(n, d=1/sampling_rate)
    fft_spectrum = np.fft.fft(signal)

    peak_freq = freq[np.argmax(np.abs(fft_spectrum))]

    rr = np.abs(peak_freq) * 60

    return rr
