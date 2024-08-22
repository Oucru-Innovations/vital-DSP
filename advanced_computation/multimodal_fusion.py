import numpy as np

class MultimodalFusion:
    """
    Multimodal Fusion for combining multiple signals to improve health assessments.

    Methods:
    - fuse: Fuses multiple signals using various strategies including weighted sum, concatenation, custom PCA-based fusion, and more.

    Example Usage:
    --------------
    signal1 = np.sin(np.linspace(0, 10, 100))
    signal2 = np.cos(np.linspace(0, 10, 100))
    fusion = MultimodalFusion([signal1, signal2])
    fused_signal_weighted = fusion.fuse(strategy="weighted_sum", weights=[0.6, 0.4])
    print("Fused Signal (Weighted Sum):", fused_signal_weighted)

    fused_signal_concat = fusion.fuse(strategy="concatenation")
    print("Fused Signal (Concatenation):", fused_signal_concat)

    fused_signal_pca = fusion.fuse(strategy="pca")
    print("Fused Signal (PCA):", fused_signal_pca)

    fused_signal_max = fusion.fuse(strategy="maximum")
    print("Fused Signal (Maximum):", fused_signal_max)
    """

    def __init__(self, signals):
        """
        Initialize the MultimodalFusion class with the signals to be fused.

        Parameters:
        signals (list of numpy.ndarray): A list of signals to be fused.
        """
        self.signals = signals

    def fuse(self, strategy="weighted_sum", **kwargs):
        """
        Fuse multiple signals using the specified strategy.

        Parameters:
        strategy (str): The fusion strategy to use. Options include "weighted_sum", "concatenation", "pca", "maximum", "minimum".
        kwargs: Additional arguments depending on the fusion strategy.
            - weights (list of float): Weights for the weighted sum strategy (required if strategy="weighted_sum").
            - n_components (int): Number of principal components to keep for PCA (optional, default=1).

        Returns:
        numpy.ndarray: The fused signal.
        """
        if strategy == "weighted_sum":
            return self._weighted_sum_fusion(kwargs.get('weights'))
        elif strategy == "concatenation":
            return self._concatenation_fusion()
        elif strategy == "pca":
            return self._pca_fusion(kwargs.get('n_components', 1))
        elif strategy == "maximum":
            return self._maximum_fusion()
        elif strategy == "minimum":
            return self._minimum_fusion()
        else:
            raise ValueError("Unknown fusion strategy: {}".format(strategy))

    def _weighted_sum_fusion(self, weights):
        """
        Fuse signals using a weighted sum.

        Parameters:
        weights (list of float): Weights for each signal.

        Returns:
        numpy.ndarray: The fused signal.
        """
        if weights is None or len(weights) != len(self.signals):
            raise ValueError("Weights must be provided and must match the number of signals.")

        fused_signal = np.zeros_like(self.signals[0])
        for signal, weight in zip(self.signals, weights):
            fused_signal += weight * signal
        return fused_signal

    def _concatenation_fusion(self):
        """
        Fuse signals by concatenation.

        Returns:
        numpy.ndarray: The concatenated signal.
        """
        return np.concatenate(self.signals, axis=0)

    def _pca_fusion(self, n_components=1):
        """
        Fuse signals using Principal Component Analysis (PCA).

        Parameters:
        n_components (int): Number of principal components to keep.

        Returns:
        numpy.ndarray: The fused signal based on PCA.
        """
        signals_matrix = np.vstack(self.signals).T
        mean_centered = signals_matrix - np.mean(signals_matrix, axis=0)
        covariance_matrix = np.cov(mean_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort the eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        sorted_eigenvalues = eigenvalues[sorted_indices]

        # Select the top n_components
        selected_eigenvectors = sorted_eigenvectors[:, :n_components]

        # Project the signals onto the selected eigenvectors
        pca_fused = np.dot(mean_centered, selected_eigenvectors)
        
        return pca_fused[:, 0] if n_components == 1 else pca_fused

    def _maximum_fusion(self):
        """
        Fuse signals by taking the maximum value at each point.

        Returns:
        numpy.ndarray: The fused signal using maximum selection.
        """
        return np.max(np.vstack(self.signals), axis=0)

    def _minimum_fusion(self):
        """
        Fuse signals by taking the minimum value at each point.

        Returns:
        numpy.ndarray: The fused signal using minimum selection.
        """
        return np.min(np.vstack(self.signals), axis=0)
