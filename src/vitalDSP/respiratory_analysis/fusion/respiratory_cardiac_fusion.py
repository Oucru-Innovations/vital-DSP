import numpy as np
from vitalDSP.respiratory_analysis.estimate_rr.time_domain_rr import time_domain_rr
from vitalDSP.respiratory_analysis.estimate_rr.frequency_domain_rr import frequency_domain_rr

def respiratory_cardiac_fusion(resp_signal, cardiac_signal, sampling_rate, preprocess=None, **preprocess_kwargs):
    """
    Fuse respiratory and cardiac signals to perform a comprehensive analysis of respiratory rate.

    Parameters
    ----------
    resp_signal : numpy.ndarray
        The respiratory signal (e.g., from a respiratory belt or nasal cannula).
    cardiac_signal : numpy.ndarray
        The cardiac signal (e.g., ECG or PPG).
    sampling_rate : float
        The sampling rate of the signals in Hz.
    preprocess : str, optional
        The preprocessing method to apply to both signals (e.g., "bandpass", "wavelet").
    preprocess_kwargs : dict, optional
        Additional arguments for the preprocessing function.

    Returns
    -------
    rr_fusion : float
        The fused respiratory rate estimate in breaths per minute.

    Examples
    --------
    >>> resp_signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> cardiac_signal = np.sin(2 * np.pi * 0.3 * np.arange(0, 10, 0.01))
    >>> rr_fusion = respiratory_cardiac_fusion(resp_signal, cardiac_signal, sampling_rate=100, preprocess='bandpass', lowcut=0.1, highcut=0.5)
    >>> print(rr_fusion)
    """
    # Estimate RR from respiratory signal using time-domain analysis
    rr_resp = time_domain_rr(resp_signal, sampling_rate, preprocess=preprocess, **preprocess_kwargs)

    # Estimate RR from cardiac signal using frequency-domain analysis
    rr_cardiac = frequency_domain_rr(cardiac_signal, sampling_rate, preprocess=preprocess, **preprocess_kwargs)

    # Combine RR estimates from both signals
    rr_fusion = np.mean([rr_resp, rr_cardiac])

    return rr_fusion
