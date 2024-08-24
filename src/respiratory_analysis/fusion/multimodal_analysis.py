import numpy as np
from respiratory_analysis.estimate_rr.peak_detection_rr import peak_detection_rr
from respiratory_analysis.estimate_rr.frequency_domain_rr import frequency_domain_rr

def multimodal_analysis(signals, sampling_rate, preprocess=None, **preprocess_kwargs):
    """
    Perform multimodal analysis by combining multiple signals for robust respiratory rate estimation.

    Parameters
    ----------
    signals : list of numpy.ndarray
        List of input signals (e.g., respiratory, ECG, PPG).
    sampling_rate : float
        The sampling rate of the signals in Hz.
    preprocess : str, optional
        The preprocessing method to apply to all signals (e.g., "bandpass", "wavelet").
    preprocess_kwargs : dict, optional
        Additional arguments for the preprocessing function.

    Returns
    -------
    rr_multimodal : float
        The combined respiratory rate estimate in breaths per minute.

    Examples
    --------
    >>> signals = [np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01)),
                   np.sin(2 * np.pi * 0.25 * np.arange(0, 10, 0.01))]
    >>> rr_multimodal = multimodal_analysis(signals, sampling_rate=100, preprocess='bandpass', lowcut=0.1, highcut=0.5)
    >>> print(rr_multimodal)
    """
    rr_estimates = []

    # Estimate RR from each signal using different methods
    for signal in signals:
        rr_peak = peak_detection_rr(signal, sampling_rate, preprocess=preprocess, **preprocess_kwargs)
        rr_freq = frequency_domain_rr(signal, sampling_rate, preprocess=preprocess, **preprocess_kwargs)
        rr_estimates.append(np.mean([rr_peak, rr_freq]))

    # Combine RR estimates from all signals
    rr_multimodal = np.mean(rr_estimates)

    return rr_multimodal
