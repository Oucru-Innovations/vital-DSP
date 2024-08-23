import numpy as np


class SparseSignalProcessing:
    """
    Sparse Signal Processing for efficient representation and processing of signals.

    Methods:
    - sparse_representation: Represents the signal using a sparse basis.
    - thresholding: Applies a threshold to the sparse representation to denoise the signal.
    - reconstruction: Reconstructs the signal from its sparse representation.

    Example Usage:
    --------------
    signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
    sparse_processing = SparseSignalProcessing(signal)
    sparse_rep = sparse_processing.sparse_representation(np.fft.fft)
    thresholded_sparse = sparse_processing.thresholding(sparse_rep, threshold=0.1)
    reconstructed_signal = sparse_processing.reconstruction(thresholded_sparse, np.fft.ifft)
    print("Reconstructed Signal:", reconstructed_signal)
    """

    def __init__(self, signal):
        self.signal = signal

    def sparse_representation(self, basis):
        """
        Represent the signal using a sparse basis (e.g., wavelets, DCT).

        Parameters:
        basis (callable): A function or method to transform the signal to the sparse domain.

        Returns:
        numpy.ndarray: The sparse representation of the signal.
        """
        sparse_rep = basis(self.signal)
        return sparse_rep

    def thresholding(self, sparse_rep, threshold):
        """
        Apply a threshold to the sparse representation to denoise the signal.

        Parameters:
        sparse_rep (numpy.ndarray): The sparse representation of the signal.
        threshold (float): The threshold value.

        Returns:
        numpy.ndarray: The thresholded sparse representation.
        """
        sparse_rep[np.abs(sparse_rep) < threshold] = 0
        return sparse_rep

    def reconstruction(self, sparse_rep, inverse_basis):
        """
        Reconstruct the signal from its sparse representation.

        Parameters:
        sparse_rep (numpy.ndarray): The sparse representation of the signal.
        inverse_basis (callable): A function or method to inverse-transform the signal from the sparse domain.

        Returns:
        numpy.ndarray: The reconstructed signal.
        """
        reconstructed_signal = inverse_basis(sparse_rep)
        return reconstructed_signal
