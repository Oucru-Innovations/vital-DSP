from vitalDSP.preprocess.noise_reduction import (
    wavelet_denoising,
    savgol_denoising,
    median_denoising,
    gaussian_denoising,
    moving_average_denoising,
)
from vitalDSP.filtering.signal_filtering import SignalFiltering
from scipy.signal import butter, filtfilt
import numpy as np


class PreprocessConfig:
    """
    Configuration class for signal preprocessing, which includes filtering and noise reduction parameters.

    Attributes
    ----------
    filter_type : str
        The type of filtering to apply ('bandpass', 'butterworth', 'chebyshev', 'elliptic').
    noise_reduction_method : str
        The noise reduction method to apply ('wavelet', 'savgol', 'median', 'gaussian', 'moving_average').
    lowcut : float
        The lower cutoff frequency for filtering.
    highcut : float
        The upper cutoff frequency for filtering.
    order : int
        The order of the filter.
    wavelet_name : str
        The name of the wavelet to use for wavelet-based noise reduction.
    level : int
        The level of wavelet decomposition.
    window_length : int
        The window length for Savitzky-Golay filtering.
    polyorder : int
        The polynomial order for Savitzky-Golay filtering.
    kernel_size : int
        The kernel size for median filtering.
    sigma : float
        The standard deviation for Gaussian filtering.
    respiratory_mode: bool
        Apply the preprocessing function specifically for respiratory signals (e.g., PPG or ECG-derived respiration).
    repreprocess: bool
        Re preprocessing function
    """

    def __init__(
        self,
        filter_type="bandpass",
        noise_reduction_method="wavelet",
        lowcut=0.1,
        highcut=0.5,
        order=4,
        wavelet_name="haar",
        level=1,
        window_length=5,
        polyorder=2,
        kernel_size=3,
        sigma=1.0,
        respiratory_mode=False,
        repreprocess=False,
    ):
        self.filter_type = filter_type
        self.noise_reduction_method = noise_reduction_method
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.wavelet_name = wavelet_name
        self.level = level
        self.window_length = window_length
        self.polyorder = polyorder
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.respiratory_mode = respiratory_mode
        self.repreprocess = repreprocess


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
    respiratory_mode=False,
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
    wavelet_name : str, optional (default="db")
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
    respiratory_mode : bool, optional (default=False)
        If True, apply preprocessing specifically for respiratory signals (e.g., PPG or ECG-derived respiration).


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
    # Ensure signal is in float64 format
    signal = np.asarray(signal, dtype=np.float64)
    # Optionally, clip large signal values to avoid overflow
    signal = np.clip(signal, -1e10, 1e10)
    # Apply bandpass filtering or other filter types
    signal_filter = SignalFiltering(signal)
    if filter_type == "bandpass":
        filtered_signal = signal_filter.bandpass(
            lowcut, highcut, sampling_rate, order=order
        )
    elif filter_type == "butterworth":
        filtered_signal = signal_filter.butterworth(
            cutoff=highcut, fs=sampling_rate, order=order, btype="low"
        )
    elif filter_type == "chebyshev":
        filtered_signal = signal_filter.chebyshev(
            cutoff=highcut, fs=sampling_rate, order=order, btype="low"
        )
    elif filter_type == "elliptic":
        filtered_signal = signal_filter.elliptic(
            cutoff=highcut, fs=sampling_rate, order=order, btype="low"
        )
    elif filter_type == "ignore":
        filtered_signal = signal.copy()
    else:
        raise ValueError(
            "Unsupported filter type. Choose from 'bandpass', 'butterworth', 'chebyshev', or 'elliptic'."
        )

    # Additional preprocessing for respiratory signals (if respiratory_mode is True)
    if respiratory_mode:
        filtered_signal = respiratory_filtering(
            signal, sampling_rate, lowcut, highcut, order
        )

    # Apply noise reduction
    if noise_reduction_method == "wavelet":
        preprocessed_signal = wavelet_denoising(
            filtered_signal, wavelet_name=wavelet_name, level=level
        )
    elif noise_reduction_method == "savgol":
        preprocessed_signal = savgol_denoising(
            filtered_signal, window_length=window_length, polyorder=polyorder
        )
    elif noise_reduction_method == "median":
        preprocessed_signal = median_denoising(filtered_signal, kernel_size=kernel_size)
    elif noise_reduction_method == "gaussian":
        preprocessed_signal = gaussian_denoising(filtered_signal, sigma=sigma)
    elif noise_reduction_method == "moving_average":
        preprocessed_signal = moving_average_denoising(
            filtered_signal, window_size=window_length
        )
    elif noise_reduction_method == "ignore":
        preprocessed_signal = filtered_signal
    else:
        raise ValueError(
            "Unsupported noise reduction method. Choose from 'wavelet', 'savgol', 'median', 'gaussian', or 'moving_average'."
        )

    return preprocessed_signal


def respiratory_filtering(signal, sampling_rate, lowcut=0.1, highcut=0.5, order=4):
    """
    Filters the signal specifically for respiratory-related frequency bands.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal to be filtered.
    sampling_rate : float
        The sampling rate of the signal in Hz.
    lowcut : float, optional (default=0.1)
        The lower cutoff frequency for respiratory filtering.
    highcut : float, optional (default=0.5)
        The upper cutoff frequency for respiratory filtering.
    order : int, optional (default=4)
        The order of the filter.

    Returns
    -------
    filtered_signal : numpy.ndarray
        The filtered respiratory signal.

    Examples
    --------
    >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.2, 100)
    >>> sampling_rate = 1000
    >>> filtered_signal = respiratory_filtering(signal, sampling_rate)
    >>> print(filtered_signal)
    """
    b, a = butter(
        order,
        [lowcut / (0.5 * sampling_rate), highcut / (0.5 * sampling_rate)],
        btype="band",
    )
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal
