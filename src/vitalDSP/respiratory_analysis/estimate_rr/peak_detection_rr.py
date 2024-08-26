from vitalDSP.utils.common import find_peaks
# from scipy.signal import find_peaks
from vitalDSP.respiratory_analysis.preprocess.preprocess import preprocess_signal

def peak_detection_rr(signal, sampling_rate, preprocess=None, min_peak_distance=0.5, height=None, threshold=None, prominence=None, width=None, **preprocess_kwargs):
    """
    Estimate respiratory rate using peak detection, with optional preprocessing.

    Parameters
    ----------
    signal : numpy.ndarray
        The input respiratory signal.
    sampling_rate : float
        The sampling rate of the signal in Hz.
    preprocess : str, optional
        The preprocessing method to apply before estimation (e.g., "bandpass", "wavelet").
    min_peak_distance : float, optional (default=0.5)
        Minimum distance between peaks in seconds.
    height : float or None, optional
        Minimum height required for a peak. Peaks below this value are ignored.
    threshold : float or None, optional
        Minimum difference between a peak and its neighboring points.
    prominence : float or None, optional
        Minimum prominence of peaks, which measures how much a peak stands out relative to its surroundings.
    width : int or None, optional
        Minimum width required for a peak, measured as the number of samples.
    preprocess_kwargs : dict, optional
        Additional arguments for the preprocessing function.

    Returns
    -------
    rr : float
        Estimated respiratory rate in breaths per minute.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> rr = peak_detection_rr(signal, sampling_rate=100, preprocess='bandpass', lowcut=0.1, highcut=0.5)
    >>> print(rr)
    """
    # Apply preprocessing if specified
    if preprocess:
        signal = preprocess_signal(signal, sampling_rate, filter_type=preprocess, **preprocess_kwargs)

    # Convert minimum peak distance from seconds to samples
    distance = int(min_peak_distance * sampling_rate)
    # Detect peaks using the custom find_peaks function
    peaks = find_peaks(signal, height=height, distance=distance, threshold=threshold, prominence=prominence, width=width)

    # Calculate the number of peaks per minute
    num_peaks = len(peaks)
    duration_minutes = len(signal) / sampling_rate / 60
    rr = num_peaks / duration_minutes

    return rr
