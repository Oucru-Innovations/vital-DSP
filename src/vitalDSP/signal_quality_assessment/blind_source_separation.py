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
    cov = np.cov(signal)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    whitening_matrix = np.dot(eigenvectors, np.diag(1.0 / np.sqrt(eigenvalues)))
    whitened_signal = np.dot(whitening_matrix.T, signal)
    return whitened_signal, whitening_matrix


def ica_artifact_removal(signals, max_iter=1000, tol=1e-5):
    """
    Perform ICA to separate sources and remove artifacts.

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
    >>> signals = np.array([[1, 2, 3], [4, 5, 6]])
    >>> separated_signals = ica_artifact_removal(signals)
    >>> print(separated_signals)
    """
    # Center the signals
    centered_signals, _ = center_signal(signals)

    # Whiten the signals
    whitened_signals, _ = whiten_signal(centered_signals)

    # Initialize random weights
    num_components = whitened_signals.shape[0]
    weights = np.random.rand(num_components, num_components)

    for i in range(max_iter):
        # Update weights using the fixed-point algorithm (FastICA)
        weights_new = np.dot(
            np.tanh(np.dot(weights, whitened_signals)).dot(whitened_signals.T)
            / whitened_signals.shape[1],
            weights.T,
        )
        weights_new /= np.linalg.norm(weights_new, axis=1, keepdims=True)

        if np.max(np.abs(np.abs(np.diag(np.dot(weights_new, weights.T))) - 1)) < tol:
            break

        weights = weights_new

    separated_signals = np.dot(weights, whitened_signals)
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
    >>> signals = np.array([[1, 2, 3], [4, 5, 6]])
    >>> separated_signals = jade_ica(signals)
    >>> print(separated_signals)
    """

    def jade(signals):
        # JADE implementation
        # num_signals = signals.shape[0]
        # Whitening
        whitened_signals, whitening_matrix = whiten_signal(signals)
        # Covariance matrix
        cov_matrix = np.cov(whitened_signals)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        B = eigenvectors * np.sqrt(eigenvalues)
        U = np.dot(B.T, whitened_signals)
        return np.dot(np.linalg.inv(U), whitened_signals)

    separated_signals = jade(signals)
    return separated_signals
