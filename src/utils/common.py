import numpy as np

def argrelextrema(signal, comparator=np.greater, order=1):
    """
    Custom implementation of finding relative extrema (maxima or minima) in a signal.

    Parameters:
    - signal (numpy.ndarray): The input signal.
    - comparator (function): Comparison function (np.greater for maxima, np.less for minima).
    - order (int): How many points on each side to consider.

    Returns:
    - extrema (numpy.ndarray): Indices of the relative extrema.
    """
    if order < 1:
        raise ValueError('Order must be an int >= 1')
    if signal.size < order * 2 + 1:
        raise ValueError('Input signal is too small')

    extrema = np.where(comparator(signal[order:-order], signal[:-2*order]) &
                       comparator(signal[order:-order], signal[2*order:]))[0] + order
    return extrema

def find_peaks(signal, height=None, distance=None, threshold=None, prominence=None, width=None):
    """
    Custom utility to find peaks in a 1D signal.

    Parameters:
    - signal (numpy.ndarray): The input signal.
    - height (float or None): Required height of peaks.
    - distance (int or None): Required minimum horizontal distance (in number of samples) between neighboring peaks.
    - threshold (float or None): Required threshold between adjacent points to consider a peak.
    - prominence (float or None): Required prominence of peaks.
    - width (int or None): Required width of peaks.

    Returns:
    - peaks (numpy.ndarray): Indices of peaks in the signal.
    """
    peaks = []
    signal_len = len(signal)
    
    # Iterate over the signal to find peaks
    for i in range(1, signal_len - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:  # Peak condition
            if height is not None and signal[i] < height:
                continue
            if threshold is not None and (signal[i] - signal[i-1] < threshold or signal[i] - signal[i+1] < threshold):
                continue
            if peaks and distance is not None and i - peaks[-1] < distance:
                continue
            if prominence is not None:
                left_base = np.min(signal[max(0, i-int(distance/2)):i])
                right_base = np.min(signal[i:min(signal_len, i+int(distance/2))])
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
    Custom implementation of the forward-backward filter (filtfilt).

    Parameters:
    - b (numpy.ndarray): Numerator (b) of the filter.
    - a (numpy.ndarray): Denominator (a) of the filter.
    - signal (numpy.ndarray): The input signal to filter.

    Returns:
    - y (numpy.ndarray): The filtered signal.
    """
    y = np.convolve(signal, b, mode='same')
    y = np.convolve(y[::-1], b, mode='same')
    return y[::-1]


def pearsonr(x, y):
    """
    Compute the Pearson correlation coefficient between two signals.

    Parameters:
    - x (numpy.ndarray): First signal.
    - y (numpy.ndarray): Second signal.

    Returns:
    float: Pearson correlation coefficient.
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

import numpy as np

def coherence(x, y, fs=1.0, nperseg=256):
    """
    Compute the coherence between two signals.

    Parameters:
    - x (numpy.ndarray): First signal.
    - y (numpy.ndarray): Second signal.
    - fs (float): Sampling frequency of the signals.
    - nperseg (int): Length of each segment for coherence computation.

    Returns:
    numpy.ndarray: Frequency array.
    numpy.ndarray: Coherence values.
    """
    def periodogram(signal):
        freqs = np.fft.rfftfreq(len(signal), d=1/fs)
        psd = np.abs(np.fft.rfft(signal)) ** 2 / len(signal)
        return freqs, psd

    def cross_spectrum(x, y):
        return np.fft.rfft(x) * np.conj(np.fft.rfft(y)) / len(x)

    freqs, psd_x = periodogram(x)
    _, psd_y = periodogram(y)
    csd_xy = cross_spectrum(x, y)
    coherence = np.abs(csd_xy) ** 2 / (psd_x * psd_y)
    
    return freqs, coherence

import numpy as np

def grangercausalitytests(data, max_lag, verbose=False):
    """
    Perform Granger causality tests to determine if one time series can predict another.

    Parameters:
    - data (numpy.ndarray): The input data array with shape (n_samples, 2), where the first column is the dependent variable.
    - max_lag (int): Maximum lag to consider for causality.

    Returns:
    dict: Granger causality test results with F-test statistics.
    """
    def lag_matrix(signal, max_lag):
        n = len(signal)
        lagged_data = np.zeros((n - max_lag, max_lag + 1))
        for i in range(max_lag + 1):
            lagged_data[:, i] = signal[max_lag - i:n - i]
        return lagged_data

    n = len(data)
    results = {}
    
    for lag in range(1, max_lag + 1):
        x_lagged = lag_matrix(data[:, 0], lag)
        y_lagged = lag_matrix(data[:, 1], lag)
        
        # Perform linear regression on the lagged data
        beta_y = np.linalg.lstsq(y_lagged, data[lag:, 0], rcond=None)[0]
        beta_x = np.linalg.lstsq(x_lagged, data[lag:, 1], rcond=None)[0]

        ssr_full = np.sum((data[lag:, 0] - y_lagged @ beta_y) ** 2)
        ssr_reduced = np.sum((data[lag:, 0] - x_lagged @ beta_x) ** 2)

        df_full = n - 2 * lag - 1
        df_reduced = n - lag - 1
        f_statistic = ((ssr_reduced - ssr_full) / lag) / (ssr_full / df_full)

        results[lag] = {
            'ssr_ftest': f_statistic
        }
        
        if verbose:
            print(f"Lag: {lag}, F-statistic: {f_statistic}")

    return results
