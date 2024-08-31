import numpy as np
from scipy.signal import butter, filtfilt, convolve, medfilt
from scipy.signal.windows import gaussian
from vitalDSP.utils.mother_wavelets import Wavelet
from sklearn.decomposition import IncrementalPCA

class ArtifactRemoval:
    """
    A class for removing various types of artifacts from signals.

    Methods
    -------
    mean_subtraction : function
        Removes artifacts by subtracting the mean of the signal.
    baseline_correction : function
        Corrects baseline drift by applying a high-pass filter.
    median_filter_removal : function
        Removes spike artifacts using a median filter.
    wavelet_denoising : function
        Removes noise using wavelet-based denoising with various mother wavelets.
    adaptive_filtering : function
        Uses an adaptive filter to remove artifacts correlated with reference signals.
    notch_filter : function
        Removes powerline interference using a notch filter.
    pca_artifact_removal : function
        Uses Principal Component Analysis (PCA) to remove artifacts.
    ica_artifact_removal : function
        Uses Independent Component Analysis (ICA) to remove artifacts using NumPy.
    """

    def __init__(self, signal):
        """
        Initialize the ArtifactRemoval class with the signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal from which artifacts need to be removed.

        Notes
        -----
        - The signal should be a 1D array.
        - If the signal is not already a NumPy array, it will be converted.
        """
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)
        self.signal = signal

    def mean_subtraction(self):
        """
        Remove artifacts by subtracting the mean of the signal.

        This method is effective for removing constant or slow-varying baseline artifacts,
        which are common in many physiological signals like ECG or EEG.

        Returns
        -------
        clean_signal : numpy.ndarray
            The artifact-removed signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> ar = ArtifactRemoval(signal)
        >>> clean_signal = ar.mean_subtraction()
        >>> print(clean_signal)
        [0 1 2 3 4]
        """
        return self.signal - np.mean(self.signal)

    def baseline_correction(self, cutoff=0.5, fs=1000):
        """
        Correct baseline drift by applying a high-pass filter.

        This method is particularly effective for removing low-frequency baseline wander
        in signals such as ECG or PPG, where baseline drift can obscure important features.

        Parameters
        ----------
        cutoff : float
            The cutoff frequency for the high-pass filter.
        fs : float
            The sampling frequency of the signal.

        Returns
        -------
        clean_signal : numpy.ndarray
            The baseline-corrected signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> ar = ArtifactRemoval(signal)
        >>> clean_signal = ar.baseline_correction(cutoff=0.5, fs=1000)
        >>> print(clean_signal)
        [-0.4995 -0.4995 -0.4995 -0.4995 -0.4995]
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b = [1, -1]
        clean_signal = np.convolve(self.signal, b, mode="same") / (1 - normal_cutoff)
        return clean_signal

    def median_filter_removal(self, kernel_size=3):
        """
        Remove spike artifacts using a median filter.

        This method is particularly useful for removing sharp spikes or noise in the signal,
        such as motion artifacts in PPG or EOG signals.

        Parameters
        ----------
        kernel_size : int
            The size of the median filter kernel. A larger kernel size will smooth more but may
            remove important signal features.

        Returns
        -------
        clean_signal : numpy.ndarray
            The artifact-removed signal.

        Examples
        --------
        >>> signal = np.array([1, 100, 3, 4, 5])
        >>> ar = ArtifactRemoval(signal)
        >>> clean_signal = ar.median_filter_removal(kernel_size=3)
        >>> print(clean_signal)
        [1 3 4 4 5]
        """
        padded_signal = np.pad(
            self.signal, (kernel_size // 2, kernel_size // 2), mode="edge"
        )
        clean_signal = np.zeros_like(self.signal)
        for i in range(len(self.signal)):
            clean_signal[i] = np.median(padded_signal[i : i + kernel_size])
        return clean_signal

    def wavelet_denoising(self, wavelet_type="db", level=1, order=4,
                          custom_wavelet=None, smoothing='lowpass', **smoothing_params):
        """
        Remove noise using wavelet-based denoising with various mother wavelets.

        This method decomposes the signal into approximation and detail coefficients using
        wavelets, thresholds the detail coefficients, and reconstructs the signal. It is effective
        for denoising signals where noise is present at multiple scales.

        Parameters
        ----------
        wavelet_type : str, optional
            The type of wavelet to use ('haar', 'db', 'sym', 'coif', 'custom'). Default is 'db'.
        level : int, optional
            The level of decomposition. Higher levels capture more global features. Default is 1.
        order : int, optional
            The order of the wavelet (used for 'db', 'sym', and 'coif' wavelets). Default is 4.
        custom_wavelet : numpy.ndarray, optional
            A custom wavelet provided by the user if `wavelet_type` is 'custom'. Default is None.

        Returns
        -------
        clean_signal : numpy.ndarray
            The denoised signal with the same length as the original signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> ar = ArtifactRemoval(signal)
        >>> clean_signal = ar.wavelet_denoising(wavelet_type='db', level=2, order=4)
        >>> print(clean_signal)

        >>> # Example using a custom wavelet
        >>> custom_wavelet = np.array([0.2, 0.5, 0.2])
        >>> clean_signal = ar.wavelet_denoising(wavelet_type='custom', custom_wavelet=custom_wavelet)
        >>> print(clean_signal)
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
            if custom_wavelet is None:
                raise ValueError("A custom wavelet must be provided if wavelet_type is 'custom'.")
            mother_wavelet = custom_wavelet
        else:
            raise ValueError(
                "Invalid wavelet_type. Must be 'haar', 'db', 'sym', 'coif', or 'custom'."
            )

        # Wavelet decomposition
        approx_coeffs = self.signal.copy()
        detail_coeffs = []

        for _ in range(level):
            # Convolution with the low-pass and high-pass filters (approximation and detail coefficients)
            approx = np.convolve(approx_coeffs, mother_wavelet, mode="full")
            detail = np.convolve(approx_coeffs, mother_wavelet[::-1], mode="full")

            # Downsample
            approx_coeffs = approx[::2]
            detail_coeffs.append(detail[::2])

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
            # Upsample
            upsampled_approx = np.zeros(len(approx_coeffs) * 2)
            upsampled_approx[::2] = approx_coeffs
            upsampled_detail = np.zeros(len(detail_coeffs[i]) * 2)
            upsampled_detail[::2] = detail_coeffs[i]

            # Truncate to the same length before summing
            upsampled_approx = upsampled_approx[:len(upsampled_detail)]
            approx_coeffs = (
                np.convolve(upsampled_approx, mother_wavelet, mode="full")[:len(upsampled_approx)]
                + np.convolve(upsampled_detail, mother_wavelet[::-1], mode="full")[:len(upsampled_detail)]
            )

        # Adjust the length of the output signal to match the original signal length
        clean_signal = approx_coeffs[: len(self.signal)]

        # Apply smoothing if specified
        if smoothing:
            clean_signal = self._apply_smoothing(clean_signal, smoothing, **smoothing_params)

        return clean_signal

    @staticmethod
    def _apply_smoothing(signal, method, **kwargs):
        """
        Apply a smoothing method to the signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be smoothed.
        method : str
            The type of smoothing to apply ('lowpass', 'gaussian', 'median', 'moving_average').
        kwargs : dict
            Additional parameters for the smoothing method.

        Returns
        -------
        smoothed_signal : numpy.ndarray
            The smoothed signal.
        """
        if method == 'lowpass':
            cutoff = kwargs.get('cutoff', 0.2)
            fs = kwargs.get('fs', 1.0)
            order = kwargs.get('order', 5)
            return ArtifactRemoval._lowpass_filter(signal, cutoff, fs, order)
        elif method == 'gaussian':
            sigma = kwargs.get('sigma', 1.0)
            return ArtifactRemoval._gaussian_smoothing(signal, sigma)
        elif method == 'median':
            kernel_size = kwargs.get('kernel_size', 3)
            return ArtifactRemoval._median_smoothing(signal, kernel_size)
        elif method == 'moving_average':
            window_size = kwargs.get('window_size', 5)
            return ArtifactRemoval._moving_average_smoothing(signal, window_size)
        else:
            raise ValueError(f"Unsupported smoothing method: {method}")

    @staticmethod
    def _gaussian_smoothing(signal, sigma):
        size = int(6 * sigma + 1)
        gaussian_kernel = gaussian(size, sigma)
        smoothed_signal = convolve(signal, gaussian_kernel, mode='same') / np.sum(gaussian_kernel)
        return smoothed_signal

    @staticmethod
    def _median_smoothing(signal, kernel_size):
        return medfilt(signal, kernel_size)

    @staticmethod
    def _moving_average_smoothing(signal, window_size):
        kernel = np.ones(window_size) / window_size
        smoothed_signal = convolve(signal, kernel, mode='same')
        return smoothed_signal

    @staticmethod
    def _lowpass_filter(signal, cutoff, fs, order=5):
        """
        Apply a low-pass Butterworth filter to smooth the signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be smoothed.
        cutoff : float
            The cutoff frequency for the low-pass filter.
        fs : float
            The sampling frequency of the signal.
        order : int, optional
            The order of the Butterworth filter. Default is 5.

        Returns
        -------
        smoothed_signal : numpy.ndarray
            The smoothed signal.
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        smoothed_signal = filtfilt(b, a, signal)
        return smoothed_signal

    def adaptive_filtering(self, reference_signal, learning_rate=0.01, num_iterations=100):
        """
        Use an adaptive filter to remove artifacts correlated with a reference signal.

        Adaptive filtering is particularly useful for removing artifacts that are correlated
        with another signal, such as EOG artifacts in EEG recordings.

        Parameters
        ----------
        reference_signal : numpy.ndarray
            The reference signal correlated with the artifact.
        learning_rate : float
            The learning rate for the adaptive filter.
        num_iterations : int
            The number of iterations for adaptation.

        Returns
        -------
        clean_signal : numpy.ndarray
            The artifact-removed signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> reference_signal = np.array([1, 1, 1, 1, 1])
        >>> ar = ArtifactRemoval(signal)
        >>> clean_signal = ar.adaptive_filtering(reference_signal, learning_rate=0.01, num_iterations=100)
        >>> print(clean_signal)
        """
        filtered_signal = self.signal.copy()
        # Ensure the signal is cast to float64 for numerical stability
        filtered_signal = self.signal.astype(np.float64)
        reference_signal = reference_signal.astype(np.float64)

        for _ in range(num_iterations):
            error = filtered_signal - reference_signal
            filtered_signal -= learning_rate * error
        return filtered_signal

    def notch_filter(self, freq=50, fs=1000, Q=30):
        """
        Remove powerline interference using a notch filter.

        This method is effective for removing specific frequency artifacts like powerline
        interference (50/60 Hz) from physiological signals.

        Parameters
        ----------
        freq : float
            The frequency to be removed (e.g., 50 Hz for powerline interference).
        fs : float
            The sampling frequency of the signal.
        Q : float
            The quality factor of the notch filter, which controls the bandwidth of the filter.

        Returns
        -------
        clean_signal : numpy.ndarray
            The artifact-removed signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> ar = ArtifactRemoval(signal)
        >>> clean_signal = ar.notch_filter(freq=50, fs=1000, Q=30)
        >>> print(clean_signal)
        """
        nyquist = 0.5 * fs
        w0 = freq / nyquist
        b = [1, -2 * np.cos(2 * np.pi * w0), 1]
        a = [1, -2 * np.cos(2 * np.pi * w0) / Q, 1 / (Q**2)]
        clean_signal = np.convolve(self.signal, b, mode="same") / np.convolve(
            self.signal, a, mode="same"
        )
        return clean_signal

    def pca_artifact_removal(self, num_components=1, window_size=100, overlap=50):
        """
        Use Principal Component Analysis (PCA) to remove artifacts.

        This method removes artifacts by reconstructing the signal with a reduced number
        of principal components, which can be particularly useful for signals with multiple
        overlapping noise sources.

        Parameters
        ----------
        num_components : int
            The number of principal components to retain.
        window_size : int, optional
            The size of each window used to segment the signal (default is 100).
        overlap : int, optional
            The number of samples that each window should overlap (default is 50).

        Returns
        -------
        clean_signal : numpy.ndarray
            The artifact-removed signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> ar = ArtifactRemoval(signal)
        >>> clean_signal = ar.pca_artifact_removal(num_components=1, window_size=2, overlap=1)
        >>> print(clean_signal)
        """
        # Segment the signal into overlapping windows
        segments = []
        for start in range(0, len(self.signal) - window_size + 1, window_size - overlap):
            segment = self.signal[start:start + window_size]
            if len(segment) == window_size:
                segments.append(segment)

        segments = np.array(segments)

        # If there are no valid segments, return the original signal
        if segments.size == 0:
            raise ValueError("No valid segments available for PCA. Ensure the signal has sufficient length and variability.")

        # Center the segments by subtracting the mean
        signal_mean = np.mean(segments, axis=0)
        centered_signal = segments - signal_mean

        # Handle cases where the covariance matrix might be degenerate
        covariance_matrix = np.cov(centered_signal, rowvar=False)

        # Check if covariance matrix is 2D
        if covariance_matrix.ndim < 2 or covariance_matrix.shape[0] < num_components:
            raise ValueError("Covariance matrix is not 2D or has insufficient dimensionality. Ensure the input signal has sufficient variation.")

        # Perform eigenvalue decomposition with error handling
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Linear Algebra error during eigenvalue decomposition: {e}")

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        selected_components = eigenvectors[:, sorted_indices[:num_components]]

        # Reconstruct the signal using the selected components
        reconstructed_segments = np.dot(centered_signal, selected_components).dot(selected_components.T) + signal_mean

        # Reconstruct the full signal by averaging overlapping windows
        clean_signal = np.zeros(len(self.signal))
        count = np.zeros(len(self.signal))

        for i, start in enumerate(range(0, len(self.signal) - window_size + 1, window_size - overlap)):
            clean_signal[start:start + window_size] += reconstructed_segments[i]
            count[start:start + window_size] += 1

        # Avoid division by zero by checking count
        clean_signal = np.divide(clean_signal, count, out=np.zeros_like(clean_signal), where=count != 0)

        return clean_signal

    def ica_artifact_removal(self, num_components=1, max_iterations=1000, tol=1e-5, seed=23, window_size=None, step_size=None, batch_size=1000):
        """
        Use Independent Component Analysis (ICA) to remove artifacts using IncrementalPCA.

        This method separates the signal into independent components and allows for the
        removal of specific components identified as artifacts. ICA is particularly useful
        for separating mixed signals into their independent sources.

        Parameters
        ----------
        num_components : int
            The number of independent components to retain.
        max_iterations : int
            The maximum number of iterations for convergence.
        tol : float
            The tolerance level for convergence.
        seed : int
            The seed for random number generation to ensure reproducibility.
        window_size : int, optional
            The size of the sliding window to create a multi-dimensional signal. If None, no windowing is applied.
        step_size : int, optional
            The step size for the sliding window. Must be used with window_size. If None, no windowing is applied.
        batch_size : int, optional
            The batch size for IncrementalPCA to manage memory usage.

        Returns
        -------
        clean_signal : numpy.ndarray
            The artifact-removed signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        >>> ar = ArtifactRemoval(signal)
        >>> clean_signal = ar.ica_artifact_removal(num_components=1, window_size=4, step_size=2)
        >>> print(clean_signal)
        """
        # Apply windowing if window_size and step_size are provided
        if window_size and step_size:
            segments = [self.signal[i:i+window_size] for i in range(0, len(self.signal) - window_size + 1, step_size)]
            multi_dimensional_signal = np.array(segments).T
        else:
            multi_dimensional_signal = self.signal.reshape(-1, 1)

        # Validate the number of components based on the signal's dimensionality
        n_features = multi_dimensional_signal.shape[1]
        if num_components > n_features:
            raise ValueError(f"n_components={num_components} invalid for n_features={n_features}. Ensure the signal has sufficient dimensionality for PCA.")

        # Center the signal
        signal_centered = multi_dimensional_signal - np.mean(multi_dimensional_signal, axis=0)

        # Apply IncrementalPCA for whitening
        ipca = IncrementalPCA(n_components=num_components, batch_size=batch_size)
        X_whitened = ipca.fit_transform(signal_centered)

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
            W_new /= np.linalg.norm(W_new, axis=1)[:, np.newaxis]  # Normalize the weights

            # Check for convergence
            if np.max(np.abs(np.abs(np.diag(np.dot(W_new, W.T))) - 1)) < tol:
                W = W_new
                break
            W = W_new

        # Separate the independent components
        S = np.dot(W, X_whitened.T).T

        # Reconstruct the signal from the components
        reconstructed_signal = np.dot(S, np.linalg.pinv(W)).dot(ipca.components_).dot(ipca.components_.T) + np.mean(multi_dimensional_signal, axis=0)

        # Stack segments and calculate the mean across the segments if windowing is applied
        if window_size and step_size:
            stacked_segments = np.column_stack([reconstructed_signal[i] for i in range(len(reconstructed_signal))])
            final_signal = np.mean(stacked_segments, axis=1)
        else:
            final_signal = reconstructed_signal.flatten()

        return final_signal
