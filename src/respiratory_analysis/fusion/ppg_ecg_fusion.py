import numpy as np
from respiratory_analysis.estimate_rr.fft_based_rr import fft_based_rr
from respiratory_analysis.estimate_rr.time_domain_rr import time_domain_rr

def ppg_ecg_fusion(ppg_signal, ecg_signal, sampling_rate, preprocess=None, **preprocess_kwargs):
    """
    Fuse PPG and ECG signals to improve the estimation of respiratory rate.

    Parameters
    ----------
    ppg_signal : numpy.ndarray
        The PPG (Photoplethysmogram) signal.
    ecg_signal : numpy.ndarray
        The ECG (Electrocardiogram) signal.
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
    >>> ppg_signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> ecg_signal = np.sin(2 * np.pi * 0.3 * np.arange(0, 10, 0.01))
    >>> rr_fusion = ppg_ecg_fusion(ppg_signal, ecg_signal, sampling_rate=100, preprocess='bandpass', lowcut=0.1, highcut=0.5)
    >>> print(rr_fusion)
    """
    # Estimate RR from PPG using FFT
    rr_ppg = fft_based_rr(ppg_signal, sampling_rate, preprocess=preprocess, **preprocess_kwargs)

    # Estimate RR from ECG using time-domain analysis
    rr_ecg = time_domain_rr(ecg_signal, sampling_rate, preprocess=preprocess, **preprocess_kwargs)

    # Combine RR estimates from both signals
    rr_fusion = np.mean([rr_ppg, rr_ecg])

    return rr_fusion
