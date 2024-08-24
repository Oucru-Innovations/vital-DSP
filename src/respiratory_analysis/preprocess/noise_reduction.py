import numpy as np
from transforms.wavelet_transform import WaveletTransform
from filtering.signal_filtering import SignalFiltering

def wavelet_denoising(signal, wavelet_name='haar', level=1):
    """
    Performs wavelet denoising on the input signal using the custom wavelet transform.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal to be denoised.
    wavelet_name : str, optional (default='haar')
        The type of wavelet to use.
    level : int, optional (default=1)
        The level of wavelet decomposition.

    Returns
    -------
    denoised_signal : numpy.ndarray
        The denoised signal.

    Examples
    --------
    >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    >>> denoised_signal = wavelet_denoising(signal, wavelet_name='haar', level=2)
    >>> print(denoised_signal)
    """
    wavelet_transform = WaveletTransform(signal, wavelet_name=wavelet_name)
    coeffs = wavelet_transform.perform_wavelet_transform(level=level)

    # Apply soft thresholding to the detail coefficients
    for i in range(len(coeffs) - 1):
        sigma = np.median(np.abs(coeffs[i])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(coeffs[i])))
        coeffs[i] = np.sign(coeffs[i]) * np.maximum(np.abs(coeffs[i]) - uthresh, 0)

    denoised_signal = wavelet_transform.perform_inverse_wavelet_transform(coeffs)
    return denoised_signal

def savgol_denoising(signal, window_length, polyorder):
    """
    Applies Savitzky-Golay filtering for noise reduction.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal to be denoised.
    window_length : int
        The length of the filter window (must be odd).
    polyorder : int
        The order of the polynomial to fit.

    Returns
    -------
    denoised_signal : numpy.ndarray
        The denoised signal.

    Examples
    --------
    >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    >>> denoised_signal = savgol_denoising(signal, window_length=5, polyorder=2)
    >>> print(denoised_signal)
    """
    return SignalFiltering.savgol_filter(signal, window_length, polyorder)

def median_denoising(signal, kernel_size=3):
    """
    Applies median filtering for noise reduction.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal to be denoised.
    kernel_size : int, optional (default=3)
        The size of the median filter kernel.

    Returns
    -------
    denoised_signal : numpy.ndarray
        The denoised signal.

    Examples
    --------
    >>> signal = np.array([1, 2, 3, 100, 5, 6, 7, 8, 9])
    >>> denoised_signal = median_denoising(signal, kernel_size=3)
    >>> print(denoised_signal)
    [1 2 3 5 5 6 7 8 9]
    """
    signal_filter = SignalFiltering(signal)
    return signal_filter.median(kernel_size)

def gaussian_denoising(signal, sigma=1.0):
    """
    Applies Gaussian filtering for noise reduction.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal to be denoised.
    sigma : float, optional (default=1.0)
        The standard deviation of the Gaussian kernel.

    Returns
    -------
    denoised_signal : numpy.ndarray
        The denoised signal.

    Examples
    --------
    >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    >>> denoised_signal = gaussian_denoising(signal, sigma=1.5)
    >>> print(denoised_signal)
    """
    signal_filter = SignalFiltering(signal)
    return signal_filter.gaussian(sigma)

def moving_average_denoising(signal, window_size):
    """
    Applies moving average filtering for noise reduction.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal to be denoised.
    window_size : int
        The size of the moving window.

    Returns
    -------
    denoised_signal : numpy.ndarray
        The denoised signal.

    Examples
    --------
    >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    >>> denoised_signal = moving_average_denoising(signal, window_size=5)
    >>> print(denoised_signal)
    """
    signal_filter = SignalFiltering(signal)
    return signal_filter.moving_average(window_size)
