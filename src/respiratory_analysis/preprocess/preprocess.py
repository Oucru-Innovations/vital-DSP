from respiratory_analysis.preprocess.noise_reduction import (
    wavelet_denoising,
    savgol_denoising,
    median_denoising,
    gaussian_denoising,
    moving_average_denoising,
)
from filtering.signal_filtering import SignalFiltering

def preprocess_signal(
    signal,
    sampling_rate,
    filter_type="bandpass",
    lowcut=0.1,
    highcut=0.5,
    order=4,
    noise_reduction_method="wavelet",
    wavelet_name="haar",
    level=1,
    window_length=5,
    polyorder=2,
    kernel_size=3,
    sigma=1.0,
):
    """
    Preprocesses the signal by applying bandpass filtering and noise reduction.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal to be preprocessed.
    sampling_rate : float
        The sampling rate of the signal in Hz.
    filter_type : str, optional (default="bandpass")
        The type of filtering to apply ('bandpass', 'butterworth', 'chebyshev', 'elliptic').
    lowcut : float, optional (default=0.1)
        The lower cutoff frequency for bandpass filtering.
    highcut : float, optional (default=0.5)
        The upper cutoff frequency for bandpass filtering.
    order : int, optional (default=4)
        The order of the filter.
    noise_reduction_method : str, optional (default="wavelet")
        The noise reduction method to apply ('wavelet', 'savgol', 'median', 'gaussian', 'moving_average').
    wavelet_name : str, optional (default="db4")
        The type of wavelet to use for wavelet denoising.
    level : int, optional (default=1)
        The level of wavelet decomposition for wavelet denoising.
    window_length : int, optional (default=5)
        The window length for Savitzky-Golay filtering.
    polyorder : int, optional (default=2)
        The polynomial order for Savitzky-Golay filtering.
    kernel_size : int, optional (default=3)
        The kernel size for median filtering.
    sigma : float, optional (default=1.0)
        The standard deviation for Gaussian filtering.

    Returns
    -------
    preprocessed_signal : numpy.ndarray
        The preprocessed signal.

    Examples
    --------
    >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.2, 100)
    >>> sampling_rate = 1000
    >>> preprocessed_signal = preprocess_signal(signal, sampling_rate, filter_type='bandpass', noise_reduction_method='wavelet')
    >>> print(preprocessed_signal)
    """
    # Apply bandpass filtering or other filter types
    signal_filter = SignalFiltering(signal)
    if filter_type == "bandpass":
        filtered_signal = signal_filter.bandpass(lowcut, highcut, sampling_rate, order=order)
    elif filter_type == "butterworth":
        filtered_signal = signal_filter.butterworth(cutoff=highcut, fs=sampling_rate, order=order, btype="low")
    elif filter_type == "chebyshev":
        filtered_signal = signal_filter.chebyshev(cutoff=highcut, fs=sampling_rate, order=order, btype="low")
    elif filter_type == "elliptic":
        filtered_signal = signal_filter.elliptic(cutoff=highcut, fs=sampling_rate, order=order, btype="low")
    else:
        raise ValueError("Unsupported filter type. Choose from 'bandpass', 'butterworth', 'chebyshev', or 'elliptic'.")

    # Apply noise reduction
    if noise_reduction_method == "wavelet":
        preprocessed_signal = wavelet_denoising(filtered_signal, wavelet_name=wavelet_name, level=level)
    elif noise_reduction_method == "savgol":
        preprocessed_signal = savgol_denoising(filtered_signal, window_length=window_length, polyorder=polyorder)
    elif noise_reduction_method == "median":
        preprocessed_signal = median_denoising(filtered_signal, kernel_size=kernel_size)
    elif noise_reduction_method == "gaussian":
        preprocessed_signal = gaussian_denoising(filtered_signal, sigma=sigma)
    elif noise_reduction_method == "moving_average":
        preprocessed_signal = moving_average_denoising(filtered_signal, window_size=window_length)
    else:
        raise ValueError("Unsupported noise reduction method. Choose from 'wavelet', 'savgol', 'median', 'gaussian', or 'moving_average'.")

    return preprocessed_signal
