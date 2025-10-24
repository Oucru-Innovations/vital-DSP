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

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.signal_quality_assessment.blind_source_separation import BlindSourceSeparation
    >>> signal = np.random.randn(1000)
    >>> processor = BlindSourceSeparation(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np


def center_signal(signal):
    """
    Center the signal by subtracting the mean.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal matrix, where each row is a different signal.

    Returns
    -------
    centered_signal : numpy.ndarray
        The centered signal matrix.
    mean_signal : numpy.ndarray
        The mean of the original signal.

    Examples
    --------
    >>> signal = np.array([[1, 2, 3], [4, 5, 6]])
    >>> centered_signal, mean_signal = center_signal(signal)
    >>> print(centered_signal)
    """
    signal = np.atleast_2d(signal)

    # Check for NaN values
    if np.isnan(signal).any():
        raise ValueError(
            "Input signal contains NaN values. Please clean the data before processing."
        )

    mean_signal = np.mean(signal, axis=1, keepdims=True)
    centered_signal = signal - mean_signal
    return centered_signal, mean_signal


def whiten_signal(signal):
    """
    Whiten the signal (decorrelate and scale to unit variance).

    Parameters
    ----------
    signal : numpy.ndarray
        The input centered signal matrix.

    Returns
    -------
    whitened_signal : numpy.ndarray
        The whitened signal matrix.
    whitening_matrix : numpy.ndarray
        The matrix used to whiten the signal.

    Examples
    --------
    >>> signal = np.array([[1, 2, 3], [4, 5, 6]])
    >>> whitened_signal, whitening_matrix = whiten_signal(signal)
    >>> print(whitened_signal)
    """
    signal = np.atleast_2d(signal)

    # Check for NaN values
    if np.isnan(signal).any():
        raise ValueError(
            "Input signal contains NaN values. Please clean the data before processing."
        )

    cov = np.cov(signal, rowvar=False)

    # Add a small positive constant to avoid numerical instability
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.where(eigenvalues <= 0, 1e-10, eigenvalues)

    whitening_matrix = np.dot(eigenvectors, np.diag(1.0 / np.sqrt(eigenvalues)))
    whitened_signal = np.dot(whitening_matrix.T, signal.T).T

    return whitened_signal, whitening_matrix


def create_synthetic_components(signal, n_components=3, method='derivatives'):
    """
    Create synthetic signal components from a 1D signal for ICA processing.

    When only a single signal channel is available, this function generates
    additional synthetic components to enable blind source separation techniques
    like ICA and PCA. This is particularly useful for physiological signals
    where we want to separate artifacts from the underlying signal.

    Parameters
    ----------
    signal : numpy.ndarray
        The 1D input signal
    n_components : int, optional (default=3)
        Number of synthetic components to generate (including original)
        Minimum is 2, recommended 3-5
    method : str, optional (default='derivatives')
        Method for creating synthetic components:
        - 'derivatives': Uses signal derivatives and delayed versions
        - 'wavelet': Uses wavelet decomposition at different scales
        - 'frequency': Uses bandpass filtered versions

    Returns
    -------
    synthetic_signals : numpy.ndarray
        Matrix of shape (n_components, len(signal)) with synthetic components

    Notes
    -----
    The synthetic components are designed to capture different aspects of the signal:
    - Original signal
    - First derivative (captures rapid changes, useful for detecting spikes)
    - Delayed version (captures temporal patterns)
    - Second derivative (captures acceleration, useful for motion artifacts)
    - Smoothed version (captures baseline trends)

    Examples
    --------
    >>> import numpy as np
    >>> signal = np.sin(2*np.pi*1*np.linspace(0,1,1000)) + 0.1*np.random.randn(1000)
    >>> synthetic = create_synthetic_components(signal, n_components=3)
    >>> print(synthetic.shape)  # (3, 1000)
    """
    if signal.ndim > 1:
        raise ValueError("Input must be a 1D signal. Use signals directly if multi-channel.")

    if n_components < 2:
        raise ValueError("n_components must be at least 2 for ICA to work.")

    signal = np.asarray(signal)
    n_samples = len(signal)
    components = []

    if method == 'derivatives':
        # Component 1: Original signal
        components.append(signal)

        # Component 2: First derivative (captures rapid changes)
        if n_components >= 2:
            derivative1 = np.gradient(signal)
            components.append(derivative1)

        # Component 3: Delayed version (captures temporal patterns)
        if n_components >= 3:
            delay = max(1, n_samples // 20)  # 5% delay
            delayed = np.roll(signal, delay)
            delayed[:delay] = delayed[delay]  # Avoid wrap-around artifacts
            components.append(delayed)

        # Component 4: Second derivative (captures acceleration)
        if n_components >= 4:
            derivative2 = np.gradient(derivative1)
            components.append(derivative2)

        # Component 5: Smoothed version (captures baseline)
        if n_components >= 5:
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(signal, sigma=max(1, n_samples // 100))
            components.append(smoothed)

        # Additional components: More delays with different lags
        for i in range(5, n_components):
            delay = max(1, n_samples // (10 + i))
            delayed = np.roll(signal, delay)
            delayed[:delay] = delayed[delay]
            components.append(delayed)

    elif method == 'frequency':
        # Component 1: Original signal
        components.append(signal)

        # Create bandpass filtered versions at different frequency bands
        from scipy.signal import butter, filtfilt

        # Determine signal sampling frequency (assume 250 Hz if not specified)
        fs = 250  # This could be passed as parameter

        # Define frequency bands
        bands = [
            (0.5, 5),    # Very low frequency (baseline)
            (5, 15),     # Low frequency
            (15, 40),    # Mid frequency
            (40, 100),   # High frequency
        ]

        for low, high in bands[:n_components-1]:
            try:
                nyq = fs / 2
                low_norm = low / nyq
                high_norm = high / nyq
                b, a = butter(2, [low_norm, high_norm], btype='band')
                filtered = filtfilt(b, a, signal)
                components.append(filtered)
            except:
                # Fallback to derivative if filtering fails
                components.append(np.gradient(signal))

            if len(components) >= n_components:
                break

        # Fill remaining with derivatives if needed
        while len(components) < n_components:
            components.append(np.gradient(components[-1]))

    else:
        raise ValueError(f"Unknown method: {method}. Use 'derivatives' or 'frequency'.")

    # Convert to array and ensure correct shape
    synthetic_signals = np.array(components[:n_components])

    return synthetic_signals


def ica_artifact_removal(signals, max_iter=1000, tol=1e-5, auto_synthetic=True, n_components=3):
    """
    Perform ICA to separate sources and remove artifacts.

    This enhanced version automatically handles both multi-channel and single-channel
    (1D) signals. For 1D signals, it creates synthetic components to enable ICA.

    Parameters
    ----------
    signals : numpy.ndarray
        The mixed signals. Can be:
        - 1D array: Single signal (will auto-generate synthetic components)
        - 2D array: Multiple signals, where each row is a different signal
    max_iter : int, optional (default=1000)
        The maximum number of iterations for the ICA algorithm.
    tol : float, optional (default=1e-5)
        Tolerance for the convergence of the algorithm.
    auto_synthetic : bool, optional (default=True)
        If True and input is 1D, automatically create synthetic components
    n_components : int, optional (default=3)
        Number of synthetic components to create for 1D signals
        Only used when auto_synthetic=True

    Returns
    -------
    separated_signals : numpy.ndarray
        The matrix of separated signals.
        If input was 1D, returns the first (cleaned) component.

    Raises
    ------
    ValueError
        If signals contain NaN values or have insufficient dimensionality

    Examples
    --------
    >>> # Example 1: Multi-channel signals
    >>> signals = np.array([[1, 2, 3], [4, 5, 6]])
    >>> separated = ica_artifact_removal(signals)
    >>> print(separated.shape)  # (2, 3)
    >>>
    >>> # Example 2: Single channel (1D) signal - auto synthetic
    >>> signal_1d = np.sin(2*np.pi*np.linspace(0,10,1000)) + 0.1*np.random.randn(1000)
    >>> cleaned = ica_artifact_removal(signal_1d)
    >>> print(cleaned.shape)  # (1000,) - returns cleaned 1D signal
    >>>
    >>> # Example 3: Single channel with custom n_components
    >>> cleaned = ica_artifact_removal(signal_1d, n_components=5)
    >>> print(cleaned.shape)  # (1000,)
    """
    # Check for NaN values first
    if np.isnan(signals).any():
        raise ValueError(
            "Input signals contain NaN values. Please clean the data before processing."
        )

    # Handle 1D input
    is_1d_input = signals.ndim == 1
    if is_1d_input:
        if not auto_synthetic:
            raise ValueError(
                "ICA requires at least 2 signal components. "
                "For 1D signals, set auto_synthetic=True to generate synthetic components, "
                "or provide multi-channel signals."
            )

        # Create synthetic components
        signals = create_synthetic_components(signals, n_components=n_components, method='derivatives')

    signals = np.atleast_2d(signals)

    # Validate dimensionality
    if signals.shape[0] < 2:
        raise ValueError(
            f"ICA requires at least 2 signal components, got {signals.shape[0]}. "
            f"Use auto_synthetic=True for 1D signals."
        )

    # Center the signals
    centered_signals, _ = center_signal(signals)

    # Whiten the signals
    whitened_signals, _ = whiten_signal(centered_signals)

    num_components = whitened_signals.shape[0]
    weights = np.random.rand(num_components, num_components)

    for i in range(max_iter):
        # Update weights using FastICA
        weights_new = np.dot(
            np.tanh(np.dot(weights, whitened_signals)).dot(whitened_signals.T)
            / whitened_signals.shape[1],
            weights.T,
        )
        weights_new /= np.linalg.norm(weights_new, axis=1, keepdims=True)

        # Check convergence
        if np.max(np.abs(np.abs(np.diag(np.dot(weights_new, weights.T))) - 1)) < tol:
            break

        weights = weights_new

    separated_signals = np.dot(weights, whitened_signals)

    # If input was 1D, return the first (cleaned) component as 1D array
    if is_1d_input:
        # The first separated component typically contains the cleaned signal
        # while artifacts are separated into other components
        return separated_signals[0]

    return separated_signals


def pca_artifact_removal(signals, n_components=None):
    """
    Perform PCA to reduce dimensionality and remove artifacts.

    Parameters
    ----------
    signals : numpy.ndarray
        The mixed signals matrix, where each row is a different signal.
    n_components : int, optional (default=None)
        Number of components to keep. If None, all components are kept.

    Returns
    -------
    reduced_signals : numpy.ndarray
        The matrix of reduced signals.

    Examples
    --------
    >>> signals = np.array([[1, 2, 3], [4, 5, 6]])
    >>> reduced_signals = pca_artifact_removal(signals, n_components=1)
    >>> print(reduced_signals)
    """
    signals = np.atleast_2d(signals)

    # Check for NaN values
    if np.isnan(signals).any():
        raise ValueError(
            "Input signals contain NaN values. Please clean the data before processing."
        )

    # Center the signals
    centered_signals, mean_signal = center_signal(signals)

    # Compute covariance matrix
    cov_matrix = np.cov(centered_signals)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvectors by eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]

    reduced_signals = np.dot(eigenvectors.T, centered_signals)
    return reduced_signals


def jade_ica(signals, max_iter=1000, tol=1e-5):
    """
    Perform Joint Approximate Diagonalization of Eigenmatrices (JADE) for ICA.

    Parameters
    ----------
    signals : numpy.ndarray
        The mixed signals matrix, where each row is a different signal.
    max_iter : int, optional (default=1000)
        The maximum number of iterations for the ICA algorithm.
    tol : float, optional (default=1e-5)
        Tolerance for the convergence of the algorithm.

    Returns
    -------
    separated_signals : numpy.ndarray
        The matrix of separated signals.

    Examples
    --------
    >>> # Input must be square matrix
    >>> signals = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> separated_signals = jade_ica(signals)
    >>> print(separated_signals)
    """
    signals = np.atleast_2d(signals)

    # Check for NaN values
    if np.isnan(signals).any():
        raise ValueError(
            "Input signals contain NaN values. Please clean the data before processing."
        )

    # Ensure there are at least two signals
    if signals.shape[0] < 2:
        raise ValueError("Input must contain at least two signals for JADE.")

    def jade(signals):
        # Whiten the signals
        whitened_signals, _ = whiten_signal(signals)

        # Compute the covariance matrix (ensure it's square)
        cov_matrix = np.cov(whitened_signals, rowvar=False)

        if cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError("Covariance matrix must be square for JADE.")

        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Avoid numerical issues by ensuring all eigenvalues are positive
        eigenvalues = np.where(eigenvalues <= 0, 1e-10, eigenvalues)

        # Fix dimensionality of B matrix to match whitened signals
        B = np.dot(eigenvectors, np.diag(np.sqrt(eigenvalues)))

        U = np.dot(B.T, whitened_signals.T).T

        # Ensure U is square before inverting
        if U.shape[0] != U.shape[1]:
            raise ValueError("Matrix U must be square to perform inversion.")

        return np.dot(np.linalg.inv(U), whitened_signals)

    # Call the internal JADE algorithm
    separated_signals = jade(signals)

    return separated_signals
