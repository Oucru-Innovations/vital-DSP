import numpy as np

def z_score_normalization(signal):
    """
    Normalize the signal using Z-score normalization.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal to be normalized.

    Returns
    -------
    normalized_signal : numpy.ndarray
        The Z-score normalized signal.

    Examples
    --------
    >>> signal = np.array([1, 2, 3, 4, 5])
    >>> normalized_signal = z_score_normalization(signal)
    >>> print(normalized_signal)
    [-1.26491106 -0.63245553  0.          0.63245553  1.26491106]
    """
    mean = np.mean(signal)
    std = np.std(signal)
    normalized_signal = (signal - mean) / std
    return normalized_signal

def min_max_normalization(signal, min_value=0, max_value=1):
    """
    Normalize the signal using Min-Max scaling.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal to be normalized.
    min_value : float, optional (default=0)
        The minimum value of the desired range.
    max_value : float, optional (default=1)
        The maximum value of the desired range.

    Returns
    -------
    normalized_signal : numpy.ndarray
        The Min-Max normalized signal.

    Examples
    --------
    >>> signal = np.array([1, 2, 3, 4, 5])
    >>> normalized_signal = min_max_normalization(signal, min_value=-1, max_value=1)
    >>> print(normalized_signal)
    [-1.  -0.5  0.   0.5  1. ]
    """
    min_signal = np.min(signal)
    max_signal = np.max(signal)
    normalized_signal = (signal - min_signal) / (max_signal - min_signal)
    normalized_signal = normalized_signal * (max_value - min_value) + min_value
    return normalized_signal
