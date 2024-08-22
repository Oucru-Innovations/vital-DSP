import numpy as np

class PCASignalDecomposition:
    """
    A class to perform PCA for signal decomposition.

    Methods:
    - compute_pca: Computes PCA on the signal.
    """

    def __init__(self, signals):
        """
        Initialize the PCASignalDecomposition class with multiple signals.

        Parameters:
        signals (numpy.ndarray): The input signals (each row is a signal).
        """
        self.signals = signals

    def compute_pca(self):
        """
        Compute Principal Component Analysis (PCA) on the signals.

        Returns:
        numpy.ndarray: The principal components of the signals.

        Example Usage:
        >>> signals = np.random.rand(5, 100)
        >>> pca = PCASignalDecomposition(signals)
        >>> pca_result = pca.compute_pca()
        >>> print(pca_result)
        """
        mean_signal = np.mean(self.signals, axis=0)
        centered_signals = self.signals - mean_signal
        covariance_matrix = np.cov(centered_signals)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        principal_components = np.dot(eigenvectors[:, sorted_indices].T, centered_signals)
        return principal_components

class ICASignalDecomposition:
    """
    A class to perform ICA for signal decomposition.

    Methods:
    - compute_ica: Computes ICA on the signal.
    """

    def __init__(self, signals, max_iter=1000, tolerance=1e-6):
        """
        Initialize the ICASignalDecomposition class with multiple signals.

        Parameters:
        signals (numpy.ndarray): The input signals (each row is a signal).
        max_iter (int): Maximum number of iterations for convergence.
        tolerance (float): Tolerance for convergence.
        """
        self.signals = signals
        self.max_iter = max_iter
        self.tolerance = tolerance

    def compute_ica(self):
        """
        Compute Independent Component Analysis (ICA) on the signals.

        Returns:
        numpy.ndarray: The independent components of the signals.

        Example Usage:
        >>> signals = np.random.rand(5, 100)
        >>> ica = ICASignalDecomposition(signals)
        >>> ica_result = ica.compute_ica()
        >>> print(ica_result)
        """
        # Center and whiten the data
        mean_signal = np.mean(self.signals, axis=0)
        centered_signals = self.signals - mean_signal
        cov = np.cov(centered_signals)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        whitening_matrix = np.dot(eig_vecs / np.sqrt(eig_vals), eig_vecs.T)
        whitened_signals = np.dot(whitening_matrix, centered_signals)

        # Initialize weights
        n_components, n_samples = whitened_signals.shape
        W = np.random.randn(n_components, n_components)

        # Perform ICA using the FastICA algorithm
        for i in range(self.max_iter):
            W_old = W.copy()
            W = np.dot(np.tanh(np.dot(W, whitened_signals)).dot(whitened_signals.T) / n_samples, W)
            W = W / np.linalg.norm(W, axis=1, keepdims=True)

            # Check for convergence
            if np.linalg.norm(W - W_old) < self.tolerance:
                break

        independent_components = np.dot(W, whitened_signals)
        return independent_components
