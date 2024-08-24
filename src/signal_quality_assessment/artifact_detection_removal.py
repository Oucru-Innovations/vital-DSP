import numpy as np

def threshold_artifact_detection(signal, threshold=0.5):
    """
    Detect artifacts in the signal based on amplitude thresholding.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal.
    threshold : float, optional (default=0.5)
        The amplitude threshold for artifact detection.

    Returns
    -------
    artifact_indices : numpy.ndarray
        Indices of the detected artifacts.

    Examples
    --------
    >>> signal = np.array([0.1, 0.2, 1.0, 0.5, 0.4, 2.0])
    >>> artifacts = threshold_artifact_detection(signal, threshold=0.6)
    >>> print(artifacts)
    """
    artifact_indices = np.where(np.abs(signal) > threshold)[0]
    return artifact_indices

def z_score_artifact_detection(signal, z_threshold=3.0):
    """
    Detect artifacts based on Z-score analysis, which identifies points that deviate significantly from the mean.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal.
    z_threshold : float, optional (default=3.0)
        The Z-score threshold for detecting artifacts.

    Returns
    -------
    artifact_indices : numpy.ndarray
        Indices of the detected artifacts.

    Examples
    --------
    >>> signal = np.random.normal(size=100)
    >>> signal[50] = 10  # Inject an artifact
    >>> artifacts = z_score_artifact_detection(signal)
    >>> print(artifacts)
    """
    mean = np.mean(signal)
    std_dev = np.std(signal)
    z_scores = np.abs((signal - mean) / std_dev)
    artifact_indices = np.where(z_scores > z_threshold)[0]
    return artifact_indices

def kurtosis_artifact_detection(signal, kurt_threshold=3.0):
    """
    Detect artifacts based on kurtosis, which measures the "tailedness" of the signal distribution.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal.
    kurt_threshold : float, optional (default=3.0)
        The kurtosis threshold for artifact detection.

    Returns
    -------
    artifact_indices : numpy.ndarray
        Indices of the detected artifacts.

    Examples
    --------
    >>> signal = np.random.normal(size=100)
    >>> signal[50] = 10  # Inject an artifact
    >>> artifacts = kurtosis_artifact_detection(signal, kurt_threshold=5.0)
    >>> print(artifacts)
    """
    n = len(signal)
    mean = np.mean(signal)
    std_dev = np.std(signal)
    kurtosis = np.sum(((signal - mean) / std_dev) ** 4) / n
    if kurtosis > kurt_threshold:
        artifact_indices = np.array([np.argmax(np.abs(signal))])
    else:
        artifact_indices = np.array([])
    return artifact_indices

def moving_average_artifact_removal(signal, window_size=5):
    """
    Remove artifacts using a moving average filter.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal.
    window_size : int, optional (default=5)
        The size of the moving average window.

    Returns
    -------
    cleaned_signal : numpy.ndarray
        The signal with artifacts smoothed out.

    Examples
    --------
    >>> signal = np.array([1, 2, 3, 100, 5, 6, 7])
    >>> cleaned_signal = moving_average_artifact_removal(signal, window_size=3)
    >>> print(cleaned_signal)
    """
    padded_signal = np.pad(signal, (window_size // 2, window_size - 1 - window_size // 2), mode='edge')
    smoothed_signal = np.convolve(padded_signal, np.ones(window_size) / window_size, mode='valid')
    return smoothed_signal

def wavelet_artifact_removal(signal, wavelet_func, level=3):
    """
    Remove artifacts from the signal using wavelet decomposition.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal.
    wavelet_func : callable
        The wavelet function to use for decomposition.
    level : int, optional (default=3)
        The decomposition level.

    Returns
    -------
    cleaned_signal : numpy.ndarray
        The signal with artifacts removed.

    Examples
    --------
    >>> def haar_wavelet(x):
    ...     return np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # Simple Haar wavelet example
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    >>> signal[200:300] = 2  # Inject an artifact
    >>> cleaned_signal = wavelet_artifact_removal(signal, haar_wavelet, level=3)
    >>> print(cleaned_signal)
    """
    # Perform wavelet decomposition
    coeffs = []
    current_signal = signal
    for _ in range(level):
        low_pass, high_pass = wavelet_func(current_signal)
        coeffs.append(high_pass)
        current_signal = low_pass

    # Zero out the highest frequency components (artifacts)
    coeffs[-1] = np.zeros_like(coeffs[-1])

    # Perform wavelet reconstruction
    for i in range(level - 1, -1, -1):
        low_pass, high_pass = wavelet_func(current_signal)
        current_signal = low_pass + np.pad(high_pass, (0, len(current_signal) - len(high_pass)), 'constant')

    return current_signal

def median_filter_artifact_removal(signal, kernel_size=3):
    """
    Remove artifacts using a median filter.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal.
    kernel_size : int, optional (default=3)
        The size of the median filter kernel.

    Returns
    -------
    cleaned_signal : numpy.ndarray
        The signal with artifacts removed.

    Examples
    --------
    >>> signal = np.array([1, 2, 3, 100, 5, 6, 7])
    >>> cleaned_signal = median_filter_artifact_removal(signal, kernel_size=3)
    >>> print(cleaned_signal)
    """
    padded_signal = np.pad(signal, (kernel_size // 2, kernel_size - 1 - kernel_size // 2), mode='edge')
    filtered_signal = np.array([np.median(padded_signal[i:i + kernel_size]) for i in range(len(signal))])
    return filtered_signal

def adaptive_threshold_artifact_detection(signal, window_size=100, std_factor=2.0):
    """
    Detect artifacts adaptively by comparing each segment's standard deviation to a threshold.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal.
    window_size : int, optional (default=100)
        The size of the window for local analysis.
    std_factor : float, optional (default=2.0)
        Factor of the standard deviation above which points are considered artifacts.

    Returns
    -------
    artifact_indices : numpy.ndarray
        Indices of the detected artifacts.

    Examples
    --------
    >>> signal = np.random.normal(size=1000)
    >>> signal[500:510] = 10  # Inject an artifact
    >>> artifacts = adaptive_threshold_artifact_detection(signal, window_size=50, std_factor=2.0)
    >>> print(artifacts)
    """
    num_windows = len(signal) // window_size
    artifact_indices = []

    for i in range(num_windows):
        segment = signal[i * window_size:(i + 1) * window_size]
        segment_mean = np.mean(segment)
        segment_std = np.std(segment)
        segment_artifacts = np.where(np.abs(segment - segment_mean) > std_factor * segment_std)[0] + i * window_size
        artifact_indices.extend(segment_artifacts)

    return np.array(artifact_indices)

def iterative_artifact_removal(signal, max_iterations=5, threshold=0.5):
    """
    Iteratively remove artifacts by applying a threshold-based removal and refining the signal.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal.
    max_iterations : int, optional (default=5)
        The maximum number of iterations to perform.
    threshold : float, optional (default=0.5)
        The threshold for detecting artifacts.

    Returns
    -------
    cleaned_signal : numpy.ndarray
        The signal with artifacts removed after iterative refinement.

    Examples
    --------
    >>> signal = np.array([1, 2, 3, 100, 5, 6, 7])
    >>> cleaned_signal = iterative_artifact_removal(signal, max_iterations=3, threshold=0.6)
    >>> print(cleaned_signal)
    """
    cleaned_signal = signal.copy()

    for _ in range(max_iterations):
        artifact_indices = threshold_artifact_detection(cleaned_signal, threshold=threshold)
        if len(artifact_indices) == 0:
            break
        cleaned_signal[artifact_indices] = np.median(cleaned_signal)

    return cleaned_signal
