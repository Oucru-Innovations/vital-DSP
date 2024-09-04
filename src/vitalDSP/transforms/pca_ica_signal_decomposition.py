import numpy as np


class PCASignalDecomposition:
    """
    A class to perform Principal Component Analysis (PCA) for signal decomposition.

    PCA is a dimensionality reduction technique that transforms the original signals into a set of linearly uncorrelated components, known as principal components. These components capture the maximum variance in the data and are ordered by the amount of variance they explain.

    Methods
    -------
    compute_pca : method
        Computes PCA on the signals and returns the principal components.
    """

    def __init__(self, signals, n_components=None):
        """
        Initialize the PCASignalDecomposition class with multiple signals.

        Parameters
        ----------
        signals : numpy.ndarray
            The input signals, where each row represents a different signal (shape: n_samples, n_features).
        n_components : int, optional
            Number of principal components to retain (default is None, which retains all components).

        Examples
        --------
        >>> signals = np.random.rand(5, 100)
        >>> pca = PCASignalDecomposition(signals, n_components=2)
        >>> pca_result = pca.compute_pca()
        >>> print(pca_result)
        """
        self.signals = signals
        self.n_components = n_components

    def compute_pca(self):
        """
        Compute Principal Component Analysis (PCA) on the signals.

        This method performs the following steps:
        1. Centers the data by subtracting the mean signal from each signal.
        2. Computes the covariance matrix of the centered signals.
        3. Computes the eigenvalues and eigenvectors of the covariance matrix.
        4. Sorts the eigenvectors by their corresponding eigenvalues in descending order.
        5. Selects the top n_components eigenvectors to form the principal components.

        Returns
        -------
        numpy.ndarray
            The principal components of the signals, where each column represents a principal component.

        Examples
        --------
        >>> signals = np.random.rand(5, 100)
        >>> pca = PCASignalDecomposition(signals, n_components=2)
        >>> pca_result = pca.compute_pca()
        >>> print(pca_result)
        """
        # Check if the input is a 2D array
        if self.signals.ndim != 2:
            raise ValueError(
                "Input signals must be a 2D array with shape (n_samples, n_features)."
            )

        # Step 1: Center the data (subtract the mean)
        mean_signal = np.mean(self.signals, axis=0)
        centered_signals = self.signals - mean_signal

        # Step 2: Compute the covariance matrix
        covariance_matrix = np.cov(centered_signals, rowvar=False)

        # Step 3: Compute eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Step 4: Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Step 5: Select the top n_components eigenvectors (principal components)
        if self.n_components is None or self.n_components > self.signals.shape[1]:
            self.n_components = self.signals.shape[1]
        principal_components = np.dot(
            centered_signals, eigenvectors[:, : self.n_components]
        )

        return principal_components


class ICASignalDecomposition:
    """
    A class to perform Independent Component Analysis (ICA) for signal decomposition.

    ICA is a computational technique for separating a multivariate signal into additive, independent components. It is commonly used in fields like biomedical signal processing (e.g., EEG data) to isolate underlying sources.

    Methods
    -------
    compute_ica : method
        Computes ICA on the signals and returns the independent components.
    """

    def __init__(self, signals, max_iter=1000, tolerance=1e-6):
        """
        Initialize the ICASignalDecomposition class with multiple signals.

        Parameters
        ----------
        signals : numpy.ndarray
            The input signals, where each row represents a different signal (shape: n_samples, n_features).
        max_iter : int, optional
            Maximum number of iterations for the FastICA algorithm to converge (default is 1000).
        tolerance : float, optional
            Tolerance for the convergence of the algorithm (default is 1e-6).

        Examples
        --------
        >>> signals = np.random.rand(5, 100)
        >>> ica = ICASignalDecomposition(signals, max_iter=1000, tolerance=1e-6)
        >>> ica_result = ica.compute_ica()
        >>> print(ica_result)
        """
        self.signals = signals
        self.max_iter = max_iter
        self.tolerance = tolerance

    def compute_ica(self):
        """
        Compute Independent Component Analysis (ICA) on the signals.

        This method performs the following steps:
        1. Centers and whitens the data to prepare it for ICA.
        2. Initializes random weights and iteratively updates them using the FastICA algorithm.
        3. Checks for convergence based on the specified tolerance.
        4. Returns the independent components of the signals.

        Returns
        -------
        numpy.ndarray
            The independent components of the signals, where each column represents an independent component.

        Examples
        --------
        >>> signals = np.random.rand(5, 100)
        >>> ica = ICASignalDecomposition(signals)
        >>> ica_result = ica.compute_ica()
        >>> print(ica_result)
        """
        # Check if the input is a 2D array
        if self.signals.ndim != 2:
            raise ValueError(
                "Input signals must be a 2D array with shape (n_samples, n_features)."
            )

        # Step 1: Center and whiten the data
        mean_signal = np.mean(self.signals, axis=0)
        centered_signals = self.signals - mean_signal
        cov = np.cov(centered_signals, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        whitening_matrix = np.dot(
            eig_vecs / np.sqrt(eig_vals + 1e-6), eig_vecs.T
        )  # Whitening
        whitened_signals = np.dot(centered_signals, whitening_matrix)

        # Step 2: Initialize weights (for n_signals)
        n_signals = whitened_signals.shape[1]
        W = np.random.randn(n_signals, n_signals)

        # Step 3: Perform ICA using the FastICA algorithm
        for i in range(self.max_iter):
            W_old = W.copy()
            W = np.dot(
                np.tanh(np.dot(W, whitened_signals.T)).dot(whitened_signals)
                / whitened_signals.shape[0],
                W,
            )
            W = W / np.linalg.norm(W, axis=1, keepdims=True)

            # Check for convergence
            if np.linalg.norm(W - W_old) < self.tolerance:
                break

        # Step 4: Compute the independent components
        independent_components = np.dot(W, whitened_signals.T).T
        return independent_components
