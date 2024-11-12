import numpy as np
import warnings
import functools
import logging as logger

def deprecated(reason):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {reason}",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def argrelextrema(signal, comparator=np.greater, order=1):
    """
    Find the relative extrema (maxima or minima) in a 1D signal.

    This function identifies local maxima or minima in a signal by comparing each point with its neighbors.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal in which to find the extrema.
    comparator : function, optional
        The comparison function to use (e.g., np.greater for maxima, np.less for minima).
        Defaults to np.greater.
    order : int, optional
        The number of points on each side to use for comparison. Must be a positive integer.
        Defaults to 1.

    Returns
    -------
    extrema : numpy.ndarray
        Indices of the relative extrema in the signal.

    Raises
    ------
    ValueError
        If the `order` is less than 1 or if the signal is too short to find extrema with the given order.

    Examples
    --------
    >>> signal = np.array([1, 3, 2, 4, 3, 5, 4])
    >>> maxima = argrelextrema(signal, comparator=np.greater, order=1)
    >>> print(maxima)
    [1, 3, 5]
    """
    if order < 1:
        raise ValueError("Order must be an int >= 1")
    if signal.size < order * 2 + 1:
        raise ValueError("Input signal is too small")

    extrema = (
        np.where(
            comparator(signal[order:-order], signal[: -2 * order])
            & comparator(signal[order:-order], signal[2 * order :])
        )[0]
        + order
    )
    return extrema


def find_peaks(
    signal, height=None, distance=None, threshold=None, prominence=None, width=None
):
    """
    Identify peaks in a 1D signal.

    This function finds local maxima in a signal that meet specific criteria such as minimum height, distance, and prominence.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal in which to find peaks.
    height : float or None, optional
        Minimum height required for a peak. Peaks below this value are ignored.
    distance : int or None, optional
        Minimum number of samples required between neighboring peaks.
    threshold : float or None, optional
        Minimum difference between a peak and its neighboring points.
    prominence : float or None, optional
        Minimum prominence of peaks, which measures how much a peak stands out relative to its surroundings.
    width : int or None, optional
        Minimum width required for a peak, measured as the number of samples.

    Returns
    -------
    peaks : numpy.ndarray
        Indices of the peaks in the signal that meet the specified criteria.

    Examples
    --------
    >>> signal = np.array([0, 1, 0, 2, 0, 3, 0])
    >>> peaks = find_peaks(signal, height=1)
    >>> print(peaks)
    [1, 3, 5]
    """
    peaks = []
    signal_len = len(signal)

    for i in range(1, signal_len - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:  # Peak condition
            if height is not None and signal[i] < height:
                continue
            if threshold is not None and (
                signal[i] - signal[i - 1] < threshold
                or signal[i] - signal[i + 1] < threshold
            ):
                continue
            if peaks and distance is not None and i - peaks[-1] < distance:
                continue
            if prominence is not None:
                left_base = np.min(signal[max(0, i - int(distance / 2)) : i])
                right_base = np.min(signal[i : min(signal_len, i + int(distance / 2))])
                if signal[i] - max(left_base, right_base) < prominence:
                    continue
            if width is not None:
                left_width = np.where(signal[:i] < signal[i] / 2)[0]
                right_width = np.where(signal[i:] < signal[i] / 2)[0]
                if not left_width.size or not right_width.size:
                    continue
                peak_width = (i - left_width[-1]) + right_width[0]
                if peak_width < width:
                    continue
            peaks.append(i)
    return np.array(peaks)


def filtfilt(b, a, signal):
    """
    Apply a forward-backward filter to a signal.

    This function applies a linear filter twice, once forward and once backward, to eliminate phase distortion.

    Parameters
    ----------
    b : numpy.ndarray
        Numerator coefficients of the filter (the feedforward part).
    a : numpy.ndarray
        Denominator coefficients of the filter (the feedback part).
    signal : numpy.ndarray
        The input signal to be filtered.

    Returns
    -------
    y : numpy.ndarray
        The filtered signal.

    Examples
    --------
    >>> b = np.array([0.0675, 0.1349, 0.0675])
    >>> a = np.array([1.0000, -1.1430, 0.4128])
    >>> signal = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    >>> filtered_signal = filtfilt(b, a, signal)
    >>> print(filtered_signal)
    """
    y = np.convolve(signal, b, mode="same")
    y = np.convolve(y[::-1], b, mode="same")
    return y[::-1]


def pearsonr(x, y):
    """
    Compute the Pearson correlation coefficient between two signals.

    The Pearson correlation coefficient is a measure of linear correlation between two signals, with a value between -1 and 1.

    Parameters
    ----------
    x : numpy.ndarray
        First input signal.
    y : numpy.ndarray
        Second input signal.

    Returns
    -------
    float
        Pearson correlation coefficient.

    Raises
    ------
    ValueError
        If the input arrays do not have the same length.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4])
    >>> y = np.array([1, 2, 3, 5])
    >>> corr = pearsonr(x, y)
    >>> print(corr)
    0.98
    """
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length.")

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    cov_xy = np.sum((x - mean_x) * (y - mean_y))
    std_x = np.sqrt(np.sum((x - mean_x) ** 2))
    std_y = np.sqrt(np.sum((y - mean_y) ** 2))

    correlation = cov_xy / (std_x * std_y)
    return correlation


def coherence(x, y, fs=1.0, nperseg=256):
    """
    Compute the coherence between two signals.

    Coherence measures the degree of correlation between two signals in the frequency domain.

    Parameters
    ----------
    x : numpy.ndarray
        First input signal.
    y : numpy.ndarray
        Second input signal.
    fs : float, optional
        Sampling frequency of the signals (default is 1.0).
    nperseg : int, optional
        Length of each segment for coherence computation (default is 256).

    Returns
    -------
    numpy.ndarray
        Frequency array.
    numpy.ndarray
        Coherence values.

    Examples
    --------
    >>> x = np.sin(2 * np.pi * np.linspace(0, 1, 500))
    >>> y = np.sin(2 * np.pi * np.linspace(0, 1, 500) + np.pi / 4)
    >>> freqs, coh = coherence(x, y, fs=500)
    >>> print(freqs, coh)
    """

    def periodogram(signal):
        freqs = np.fft.rfftfreq(len(signal), d=1 / fs)
        psd = np.abs(np.fft.rfft(signal)) ** 2 / len(signal)
        return freqs, psd

    def cross_spectrum(x, y):
        return np.fft.rfft(x) * np.conj(np.fft.rfft(y)) / len(x)

    freqs, psd_x = periodogram(x)
    _, psd_y = periodogram(y)
    csd_xy = cross_spectrum(x, y)
    coherence = np.abs(csd_xy) ** 2 / (psd_x * psd_y)

    return freqs, coherence


def grangercausalitytests(data, max_lag, verbose=False):
    """
    Perform Granger causality tests to determine if one time series can predict another.

    The Granger causality test evaluates whether the past values of one time series can provide statistically significant information about the future values of another time series.

    Parameters
    ----------
    data : numpy.ndarray
        The input data array with shape (n_samples, 2), where the first column is the dependent variable.
    max_lag : int
        Maximum lag to consider for causality.
    verbose : bool, optional
        If True, prints out detailed test statistics (default is False).

    Returns
    -------
    dict
        Granger causality test results, including F-test statistics for each lag.

    Examples
    --------
    >>> data = np.random.rand(100, 2)
    >>> results = grangercausalitytests(data, max_lag=4, verbose=True)
    >>> print(results)
    """

    def lag_matrix(signal, max_lag):
        n = len(signal)
        lagged_data = np.zeros((n - max_lag, max_lag + 1))
        for i in range(max_lag + 1):
            lagged_data[:, i] = signal[max_lag - i : n - i]
        return lagged_data

    n = len(data)
    results = {}

    for lag in range(1, max_lag + 1):
        x_lagged = lag_matrix(data[:, 0], lag)
        y_lagged = lag_matrix(data[:, 1], lag)

        beta_y = np.linalg.lstsq(y_lagged, data[lag:, 0], rcond=None)[0]
        beta_x = np.linalg.lstsq(x_lagged, data[lag:, 1], rcond=None)[0]

        ssr_full = np.sum((data[lag:, 0] - y_lagged @ beta_y) ** 2)
        ssr_reduced = np.sum((data[lag:, 0] - x_lagged @ beta_x) ** 2)

        df_full = n - 2 * lag - 1
        f_statistic = ((ssr_reduced - ssr_full) / lag) / (ssr_full / df_full)

        results[lag] = {"ssr_ftest": f_statistic}

        if verbose:
            logger.info(f"Lag: {lag}, F-statistic: {f_statistic}")

    return results


def dtw_distance_windowed(x, y, window=None):
    """
    Compute the Dynamic Time Warping (DTW) distance between two sequences using a sliding window.

    Parameters
    ----------
    x : numpy.ndarray
        The first time series.
    y : numpy.ndarray
        The second time series.
    window : int, optional
        The size of the window for the DTW computation. If None, it uses the full sequence.

    Returns
    -------
    float
        The DTW distance between the two sequences.
    """
    if window is None:
        window = len(x)

    # Initialize the cost matrix with infinity
    dtw_matrix = np.full((len(x), len(y)), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, len(x)):
        for j in range(max(1, i - window), min(len(y), i + window)):
            cost = (x[i] - y[j]) ** 2
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # Insertion
                dtw_matrix[i, j - 1],  # Deletion
                dtw_matrix[i - 1, j - 1],
            )  # Match

    return np.sqrt(dtw_matrix[len(x) - 1, len(y) - 1])
