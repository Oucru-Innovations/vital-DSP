import numpy as np
from utils.mother_wavelets import Wavelet


class ArtifactRemoval:
    """
    A class for removing various types of artifacts from signals.

    Methods:
    - mean_subtraction: Removes artifacts by subtracting the mean of the signal.
    - baseline_correction: Corrects baseline drift by applying a high-pass filter.
    - median_filter_removal: Removes spike artifacts using a median filter.
    - wavelet_denoising: Removes noise using wavelet-based denoising with various mother wavelets.
    - adaptive_filtering: Uses an adaptive filter to remove artifacts correlated with reference signals.
    - notch_filter: Removes powerline interference using a notch filter.
    - pca_artifact_removal: Uses Principal Component Analysis (PCA) to remove artifacts.
    - ica_artifact_removal: Uses Independent Component Analysis (ICA) to remove artifacts using NumPy.
    """

    def __init__(self, signal):
        """
        Initialize the ArtifactRemoval class with the signal.

        Parameters:
        signal (numpy.ndarray): The input signal from which artifacts need to be removed.
        """
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)
        self.signal = signal

    def mean_subtraction(self):
        """
        Remove artifacts by subtracting the mean of the signal.

        This method is effective for removing constant or slow-varying baseline artifacts.

        Returns:
        numpy.ndarray: The artifact-removed signal.
        """
        return self.signal - np.mean(self.signal)

    def baseline_correction(self, cutoff=0.5, fs=1000):
        """
        Correct baseline drift by applying a high-pass filter.

        This method is particularly effective for removing low-frequency baseline wander in ECG or PPG signals.

        Parameters:
        cutoff (float): The cutoff frequency for the high-pass filter.
        fs (float): The sampling frequency of the signal.

        Returns:
        numpy.ndarray: The baseline-corrected signal.
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b = [1, -1]
        # a = [1, -normal_cutoff]
        clean_signal = np.convolve(self.signal, b, mode="same") / (1 - normal_cutoff)
        return clean_signal

    def median_filter_removal(self, kernel_size=3):
        """
        Remove spike artifacts using a median filter.

        This method is effective for removing sharp spike artifacts from the signal.

        Parameters:
        kernel_size (int): The size of the median filter kernel.

        Returns:
        numpy.ndarray: The artifact-removed signal.
        """
        padded_signal = np.pad(
            self.signal, (kernel_size // 2, kernel_size // 2), mode="edge"
        )
        clean_signal = np.zeros_like(self.signal)
        for i in range(len(self.signal)):
            clean_signal[i] = np.median(padded_signal[i : i + kernel_size])
        return clean_signal

    def wavelet_denoising(self, wavelet_type="db", level=1, order=4):
        """
        Remove noise using wavelet-based denoising with various mother wavelets.

        This method decomposes the signal using wavelets and thresholds the wavelet coefficients to remove noise.

        Parameters:
        wavelet_type (str): The type of wavelet to use ('haar', 'db', 'sym', 'coif', 'custom').
        level (int): The level of decomposition.
        order (int): The order of the wavelet (used for 'db', 'sym', and 'coif').

        Returns:
        numpy.ndarray: The denoised signal.
        """
        wavelet = Wavelet()

        if wavelet_type == "haar":
            mother_wavelet = wavelet.haar()
        elif wavelet_type == "db":
            mother_wavelet = wavelet.db(order)
        elif wavelet_type == "sym":
            mother_wavelet = wavelet.sym(order)
        elif wavelet_type == "coif":
            mother_wavelet = wavelet.coif(order)
        elif wavelet_type == "custom":
            raise ValueError("Use 'custom_wavelet' method for custom wavelets.")
        else:
            raise ValueError(
                "Invalid wavelet_type. Must be 'haar', 'db', 'sym', 'coif', or 'custom'."
            )

        # Wavelet decomposition
        approx_coeffs = self.signal.copy()
        detail_coeffs = []

        for _ in range(level):
            # Convolution with the low-pass and high-pass filters (approximation and detail coefficients)
            approx = np.convolve(approx_coeffs, mother_wavelet, mode="full")[::2]
            detail = np.convolve(approx_coeffs, mother_wavelet[::-1], mode="full")[::2]
            approx_coeffs = approx
            detail_coeffs.append(detail)

        # Thresholding detail coefficients
        threshold = (
            np.sqrt(2 * np.log(len(self.signal)))
            * np.median(np.abs(detail_coeffs[-1]))
            / 0.6745
        )
        for i in range(len(detail_coeffs)):
            detail_coeffs[i] = np.sign(detail_coeffs[i]) * np.maximum(
                np.abs(detail_coeffs[i]) - threshold, 0
            )

        # Wavelet reconstruction
        for i in reversed(range(level)):
            approx_coeffs = np.convolve(
                np.repeat(approx_coeffs, 2), mother_wavelet, mode="full"
            )[: len(detail_coeffs[i])]
            approx_coeffs += np.convolve(
                np.repeat(detail_coeffs[i], 2), mother_wavelet[::-1], mode="full"
            )[: len(detail_coeffs[i])]

        return approx_coeffs[: len(self.signal)]

    def adaptive_filtering(
        self, reference_signal, learning_rate=0.01, num_iterations=100
    ):
        """
        Use an adaptive filter to remove artifacts correlated with a reference signal.

        This method is particularly effective for removing artifacts that are correlated with another signal (e.g., EOG artifacts in EEG).

        Parameters:
        reference_signal (numpy.ndarray): The reference signal correlated with the artifact.
        learning_rate (float): The learning rate for the adaptive filter.
        num_iterations (int): The number of iterations for adaptation.

        Returns:
        numpy.ndarray: The artifact-removed signal.
        """
        filtered_signal = self.signal.copy()
        for _ in range(num_iterations):
            error = filtered_signal - reference_signal
            filtered_signal -= learning_rate * error
        return filtered_signal

    def notch_filter(self, freq=50, fs=1000, Q=30):
        """
        Remove powerline interference using a notch filter.

        This method is effective for removing specific frequency artifacts like powerline interference (50/60 Hz).

        Parameters:
        freq (float): The frequency to be removed (e.g., 50 Hz for powerline interference).
        fs (float): The sampling frequency of the signal.
        Q (float): The quality factor of the notch filter.

        Returns:
        numpy.ndarray: The artifact-removed signal.
        """
        nyquist = 0.5 * fs
        w0 = freq / nyquist
        b = [1, -2 * np.cos(2 * np.pi * w0), 1]
        a = [1, -2 * np.cos(2 * np.pi * w0) / Q, 1 / (Q**2)]
        clean_signal = np.convolve(self.signal, b, mode="same") / np.convolve(
            self.signal, a, mode="same"
        )
        return clean_signal

    def pca_artifact_removal(self, num_components=1):
        """
        Use Principal Component Analysis (PCA) to remove artifacts.

        This method removes artifacts by reconstructing the signal with a reduced number of principal components.

        Parameters:
        num_components (int): The number of principal components to retain.

        Returns:
        numpy.ndarray: The artifact-removed signal.
        """
        signal_mean = np.mean(self.signal)
        centered_signal = self.signal - signal_mean
        covariance_matrix = np.cov(centered_signal)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        selected_components = eigenvectors[:, sorted_indices[:num_components]]
        reconstructed_signal = (
            np.dot(centered_signal, selected_components).dot(selected_components.T)
            + signal_mean
        )
        return reconstructed_signal

    def ica_artifact_removal(
        self, num_components=1, max_iterations=1000, tol=1e-5, seed=23
    ):
        """
        Use Independent Component Analysis (ICA) to remove artifacts using NumPy.

        This method separates the signal into independent components and allows for the removal of specific components identified as artifacts.

        Parameters:
        num_components (int): The number of independent components to retain.
        max_iterations (int): The maximum number of iterations for convergence.
        tol (float): The tolerance level for convergence.

        Returns:
        numpy.ndarray: The artifact-removed signal.

        Example:
        >>> signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        >>> ar = ArtifactRemoval(signal)
        >>> clean_signal = ar.ica_artifact_removal(num_components=1)
        >>> print(clean_signal)
        """
        # Center the signal
        signal_centered = self.signal - np.mean(self.signal, axis=0)

        # Whitening (PCA step)
        cov_matrix = np.cov(signal_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        whitening_matrix = eigenvectors.dot(np.diag(1.0 / np.sqrt(eigenvalues))).dot(
            eigenvectors.T
        )
        X_whitened = signal_centered.dot(whitening_matrix.T)

        # Initialize weights randomly
        np.random.seed(seed)
        W = np.random.rand(num_components, X_whitened.shape[1])

        for i in range(max_iterations):
            # Update the weights using the FastICA algorithm
            W_new = (
                np.dot(X_whitened.T, np.tanh(np.dot(X_whitened, W.T)))
                / X_whitened.shape[0]
                - np.mean(1 - np.tanh(np.dot(X_whitened, W.T)) ** 2, axis=0) * W
            )
            W_new /= np.linalg.norm(W_new, axis=1)[
                :, np.newaxis
            ]  # Normalize the weights

            # Check for convergence
            if np.max(np.abs(np.abs(np.diag(np.dot(W_new, W.T))) - 1)) < tol:
                W = W_new
                break
            W = W_new

        # Separate the independent components
        S = np.dot(W, X_whitened.T).T

        # Reconstruct the signal from the components
        reconstructed_signal = np.dot(S, np.linalg.pinv(W)).dot(
            whitening_matrix
        ) + np.mean(self.signal, axis=0)

        return reconstructed_signal
