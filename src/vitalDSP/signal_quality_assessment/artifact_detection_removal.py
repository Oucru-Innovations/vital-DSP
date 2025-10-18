"""
Signal Quality Assessment Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Multiple processing methods and functions
- NumPy integration for numerical computations
- Pattern and anomaly detection

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.signal_quality_assessment.artifact_detection_removal import ArtifactDetectionRemoval
    >>> signal = np.random.randn(1000)
    >>> processor = ArtifactDetectionRemoval(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""


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
    padded_signal = np.pad(
        signal, (window_size // 2, window_size - 1 - window_size // 2), mode="edge"
    )
    smoothed_signal = np.convolve(
        padded_signal, np.ones(window_size) / window_size, mode="valid"
    )
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
        current_signal = low_pass + np.pad(
            high_pass, (0, len(current_signal) - len(high_pass)), "constant"
        )

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
    padded_signal = np.pad(
        signal, (kernel_size // 2, kernel_size - 1 - kernel_size // 2), mode="edge"
    )
    filtered_signal = np.array(
        [np.median(padded_signal[i : i + kernel_size]) for i in range(len(signal))]
    )
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
        segment = signal[i * window_size : (i + 1) * window_size]
        segment_mean = np.mean(segment)
        segment_std = np.std(segment)
        segment_artifacts = (
            np.where(np.abs(segment - segment_mean) > std_factor * segment_std)[0]
            + i * window_size
        )
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
        artifact_indices = threshold_artifact_detection(
            cleaned_signal, threshold=threshold
        )

        if len(artifact_indices) == 0:
            break

        # Replace only artifact indices with the median of the non-artifact portion of the signal
        if len(artifact_indices) > 0:
            # Get valid (non-artifact) values
            valid_indices = np.delete(np.arange(len(cleaned_signal)), artifact_indices)
            if len(valid_indices) > 0:
                median_value = np.median(cleaned_signal[valid_indices])
                cleaned_signal[artifact_indices] = median_value
            else:
                # If no valid values, set artifacts to 0
                cleaned_signal[artifact_indices] = 0

    return cleaned_signal


class ArtifactDetectionRemoval:
    """
    A comprehensive artifact detection and removal class for physiological signals.
    
    This class provides various methods for detecting and removing artifacts from
    physiological signals including ECG, PPG, EEG, and other vital signs.
    
    Parameters
    ----------
    signal : numpy.ndarray
        The input signal to process.
    fs : float, optional
        Sampling frequency of the signal (default: 250 Hz).
    
    Examples
    --------
    >>> import numpy as np
    >>> signal = np.random.randn(1000)
    >>> adr = ArtifactDetectionRemoval(signal, fs=250)
    >>> artifacts = adr.detect_artifacts()
    >>> cleaned_signal = adr.remove_artifacts()
    """
    
    def __init__(self, signal, fs=250):
        """
        Initialize the ArtifactDetectionRemoval class.
        
        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to process.
        fs : float, optional
            Sampling frequency of the signal (default: 250 Hz).
        """
        self.signal = np.array(signal)
        self.fs = fs
        self.artifact_indices = None
    
    def detect_artifacts(self, method='threshold', **kwargs):
        """
        Detect artifacts in the signal using various methods.
        
        Parameters
        ----------
        method : str, optional
            Method to use for artifact detection. Options:
            - 'threshold': Amplitude thresholding (default)
            - 'z_score': Z-score based detection
            - 'kurtosis': Kurtosis based detection
            - 'adaptive': Adaptive threshold detection
        **kwargs
            Additional parameters for the specific detection method.
            
        Returns
        -------
        numpy.ndarray
            Indices of detected artifacts.
        """
        if method == 'threshold':
            threshold = kwargs.get('threshold', 0.5)
            self.artifact_indices = threshold_artifact_detection(self.signal, threshold)
        elif method == 'z_score':
            z_threshold = kwargs.get('z_threshold', 3.0)
            self.artifact_indices = z_score_artifact_detection(self.signal, z_threshold)
        elif method == 'kurtosis':
            kurt_threshold = kwargs.get('kurt_threshold', 3.0)
            self.artifact_indices = kurtosis_artifact_detection(self.signal, kurt_threshold)
        elif method == 'adaptive':
            window_size = kwargs.get('window_size', 100)
            std_factor = kwargs.get('std_factor', 2.0)
            self.artifact_indices = adaptive_threshold_artifact_detection(
                self.signal, window_size, std_factor
            )
        else:
            raise ValueError(f"Unknown detection method: {method}")
            
        return self.artifact_indices
    
    def remove_artifacts(self, method='moving_average', **kwargs):
        """
        Remove artifacts from the signal using various methods.
        
        Parameters
        ----------
        method : str, optional
            Method to use for artifact removal. Options:
            - 'moving_average': Moving average filter (default)
            - 'wavelet': Wavelet-based removal
            - 'median': Median filter
            - 'iterative': Iterative removal
        **kwargs
            Additional parameters for the specific removal method.
            
        Returns
        -------
        numpy.ndarray
            Signal with artifacts removed.
        """
        if method == 'moving_average':
            window_size = kwargs.get('window_size', 5)
            return moving_average_artifact_removal(self.signal, window_size)
        elif method == 'wavelet':
            wavelet_func = kwargs.get('wavelet_func', 'db4')
            level = kwargs.get('level', 3)
            return wavelet_artifact_removal(self.signal, wavelet_func, level)
        elif method == 'median':
            kernel_size = kwargs.get('kernel_size', 3)
            return median_filter_artifact_removal(self.signal, kernel_size)
        elif method == 'iterative':
            max_iterations = kwargs.get('max_iterations', 5)
            threshold = kwargs.get('threshold', 0.5)
            return iterative_artifact_removal(self.signal, max_iterations, threshold)
        else:
            raise ValueError(f"Unknown removal method: {method}")
    
    def process(self, detection_method='threshold', removal_method='moving_average', 
                detection_kwargs=None, removal_kwargs=None):
        """
        Complete processing pipeline: detect and remove artifacts.
        
        Parameters
        ----------
        detection_method : str, optional
            Method for artifact detection (default: 'threshold').
        removal_method : str, optional
            Method for artifact removal (default: 'moving_average').
        detection_kwargs : dict, optional
            Parameters for detection method.
        removal_kwargs : dict, optional
            Parameters for removal method.
            
        Returns
        -------
        numpy.ndarray
            Processed signal with artifacts removed.
        """
        if detection_kwargs is None:
            detection_kwargs = {}
        if removal_kwargs is None:
            removal_kwargs = {}
            
        # Detect artifacts
        self.detect_artifacts(detection_method, **detection_kwargs)
        
        # Remove artifacts
        cleaned_signal = self.remove_artifacts(removal_method, **removal_kwargs)
        
        return cleaned_signal