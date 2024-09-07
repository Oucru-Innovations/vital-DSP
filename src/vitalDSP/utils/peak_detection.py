import numpy as np
from vitalDSP.utils.common import find_peaks, filtfilt, argrelextrema
from vitalDSP.utils.scaler import StandardScaler
from vitalDSP.filtering.signal_filtering import SignalFiltering


class PeakDetection:
    """
    A class to detect peaks in various physiological signals like ECG, PPG, EEG, respiratory, and arterial blood pressure (ABP) signals.

    Methods
    -------
    detect_peaks : function
        Detects the peaks in the signal using the selected method.
    """

    def __init__(self, signal, method="threshold", **kwargs):
        """
        Initialize the PeakDetection class with the signal and the selected peak detection method.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be analyzed for peak detection.
        method : str, optional
            The method to use for peak detection. Default is "threshold".
            Available methods are:
            - 'threshold'
            - 'savgol'
            - 'gaussian'
            - 'rel_extrema'
            - 'scaler_threshold'
            - 'ecg_r_peak'
            - 'ecg_derivative'
            - 'ppg_first_derivative'
            - 'ppg_second_derivative'
            - 'eeg_wavelet'
            - 'eeg_bandpass'
            - 'resp_autocorrelation'
            - 'resp_zero_crossing'
            - 'abp_systolic'
            - 'abp_diastolic'
        kwargs : dict
            Additional parameters specific to the selected method.
        """
        self.signal = signal
        self.method = method
        self.kwargs = kwargs

    def detect_peaks(self):
        """
        Detect peaks in the signal based on the selected method.

        Returns
        -------
        peaks : numpy.ndarray
            The indices of the detected peaks.

        Raises
        ------
        ValueError
            If an invalid method is selected.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        >>> detector = PeakDetection(signal, method="threshold", threshold=4)
        >>> peaks = detector.detect_peaks()
        >>> print(peaks)
        """
        if self.method == "threshold":
            return self._threshold_based_detection()
        elif self.method == "savgol":
            return self._savgol_based_detection()
        elif self.method == "gaussian":
            return self._gaussian_based_detection()
        elif self.method == "rel_extrema":
            return self._relative_extrema_detection()
        elif self.method == "scaler_threshold":
            return self._scaler_threshold_based_detection()
        elif self.method == "ecg_r_peak":
            return self._ecg_r_peak_detection()
        elif self.method == "ecg_derivative":
            return self._ecg_derivative_detection()
        elif self.method == "ppg_first_derivative":
            return self._ppg_first_derivative_detection()
        elif self.method == "ppg_second_derivative":
            return self._ppg_second_derivative_detection()
        elif self.method == "eeg_wavelet":
            return self._eeg_wavelet_detection()
        elif self.method == "eeg_bandpass":
            return self._eeg_bandpass_detection()
        elif self.method == "resp_autocorrelation":
            return self._resp_autocorrelation_detection()
        elif self.method == "resp_zero_crossing":
            return self._resp_zero_crossing_detection()
        elif self.method == "abp_systolic":
            return self._abp_systolic_peak_detection()
        elif self.method == "abp_diastolic":
            return self._abp_diastolic_peak_detection()
        else:
            raise ValueError(
                "Invalid method selected. Choose from the provided methods."
            )

    def _threshold_based_detection(self):
        """
        Detect peaks using a simple thresholding approach.

        This method identifies peaks in the signal that exceed a specified threshold and are separated by a minimum distance.

        Returns
        -------
        peaks : numpy.ndarray
            Indices of peaks detected based on the threshold.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        >>> detector = PeakDetection(signal, method="threshold", threshold=4, distance=50)
        >>> peaks = detector._threshold_based_detection()
        >>> print(peaks)
        """
        threshold = self.kwargs.get("threshold", 0.5)
        distance = self.kwargs.get("distance", 50)
        return find_peaks(self.signal, height=threshold, distance=distance)

    def _savgol_based_detection(self):
        """
        Detect peaks using the Savitzky-Golay filter to smooth the signal.

        The Savitzky-Golay filter is applied to the signal to reduce noise and highlight peaks, which are then detected.

        Returns
        -------
        peaks : numpy.ndarray
            Indices of peaks detected after smoothing with the Savitzky-Golay filter.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        >>> detector = PeakDetection(signal, method="savgol", window_length=11, polyorder=2)
        >>> peaks = detector._savgol_based_detection()
        >>> print(peaks)
        """
        window_length = self.kwargs.get("window_length", 11)
        polyorder = self.kwargs.get("polyorder", 2)
        smoothed_signal = SignalFiltering.savgol_filter(
            self.signal, window_length=window_length, polyorder=polyorder
        )
        return find_peaks(smoothed_signal)

    def _gaussian_based_detection(self):
        """
        Detect peaks by smoothing the signal with a Gaussian filter.

        Gaussian filtering is applied to smooth the signal, making it easier to identify peaks.

        Returns
        -------
        peaks : numpy.ndarray
            Indices of peaks detected after Gaussian filtering.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        >>> detector = PeakDetection(signal, method="gaussian", sigma=1)
        >>> peaks = detector._gaussian_based_detection()
        >>> print(peaks)
        """
        sigma = self.kwargs.get("sigma", 1)
        smoothed_signal = SignalFiltering.gaussian_filter1d(self.signal, sigma=sigma)
        return find_peaks(smoothed_signal)

    def _relative_extrema_detection(self):
        """
        Detect relative extrema (peaks) in the signal using the specified order.

        This method identifies peaks by finding points that are higher than their neighboring values within a defined window.

        Returns
        -------
        peaks : numpy.ndarray
            Indices of peaks detected as relative extrema.

        Examples
        --------
        >>> signal = np.array([1, 3, 2, 4, 3, 5, 4])
        >>> detector = PeakDetection(signal, method="rel_extrema", order=1)
        >>> peaks = detector._relative_extrema_detection()
        >>> print(peaks)
        """
        order = self.kwargs.get("order", 1)
        peaks = argrelextrema(self.signal, np.greater, order=order)
        return peaks

    def _scaler_threshold_based_detection(self):
        """
        Detect peaks after scaling the signal and applying a threshold.

        The signal is standardized using a scaler, and peaks are detected based on the standardized values exceeding a threshold.

        Returns
        -------
        peaks : numpy.ndarray
            Indices of peaks detected after scaling and thresholding.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        >>> detector = PeakDetection(signal, method="scaler_threshold", threshold=1, distance=50)
        >>> peaks = detector._scaler_threshold_based_detection()
        >>> print(peaks)
        """
        scaler = StandardScaler()
        standardized_signal = scaler.fit_transform(self.signal)
        threshold = self.kwargs.get("threshold", 1)
        distance = self.kwargs.get("distance", 50)
        return find_peaks(standardized_signal, height=threshold, distance=distance)

    def _ecg_r_peak_detection(self):
        """
        Detect R-peaks in ECG signals using the derivative and squared signal method.

        This method enhances the R-peaks by taking the derivative of the ECG signal, squaring it, and applying a moving window integrator.

        Returns
        -------
        peaks : numpy.ndarray
            Indices of R-peaks detected in the ECG signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 1, 2, 1, 2])
        >>> detector = PeakDetection(signal, method="ecg_r_peak")
        >>> peaks = detector._ecg_r_peak_detection()
        >>> print(peaks)
        """
        diff_signal = np.diff(self.signal)
        squared_signal = diff_signal**2
        integrator = np.convolve(squared_signal, np.ones(150), "same")
        return find_peaks(integrator, distance=150, height=np.mean(integrator))

    def _ecg_derivative_detection(self):
        """
        Detect peaks in the ECG signal based on the absolute derivative.

        Peaks are identified by calculating the absolute value of the signal's derivative and detecting significant increases.

        Returns
        -------
        peaks : numpy.ndarray
            Indices of peaks detected in the ECG signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 1, 2, 1, 2])
        >>> detector = PeakDetection(signal, method="ecg_derivative")
        >>> peaks = detector._ecg_derivative_detection()
        >>> print(peaks)
        """
        diff_signal = np.diff(self.signal)
        return find_peaks(np.abs(diff_signal), height=np.mean(np.abs(diff_signal)))

    def _ppg_first_derivative_detection(self):
        """
        Detect peaks in PPG signals using the first derivative.

        This method identifies peaks by calculating the first derivative of the PPG signal and detecting points of significant positive change.

        Returns
        -------
        peaks : numpy.ndarray
            Indices of peaks detected in the first derivative of the PPG signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        >>> detector = PeakDetection(signal, method="ppg_first_derivative")
        >>> peaks = detector._ppg_first_derivative_detection()
        >>> print(peaks)
        """
        diff_signal = np.diff(self.signal)
        return find_peaks(diff_signal, height=np.mean(diff_signal))

    def _ppg_second_derivative_detection(self):
        """
        Detect peaks in PPG signals using the second derivative.

        This method enhances the detection of peaks by calculating the second derivative of the PPG signal, which highlights inflection points.

        Returns
        -------
        peaks : numpy.ndarray
            Indices of peaks detected in the second derivative of the PPG signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        >>> detector = PeakDetection(signal, method="ppg_second_derivative")
        >>> peaks = detector._ppg_second_derivative_detection()
        >>> print(peaks)
        """
        diff_signal = np.diff(self.signal, n=2)
        return find_peaks(diff_signal, height=np.mean(diff_signal))

    def _eeg_wavelet_detection(self):
        """
        Detect peaks in EEG signals using wavelet transformation.

        The signal is transformed using wavelets, and peaks are detected based on the transformed coefficients.

        Returns
        -------
        peaks : numpy.ndarray
            Indices of peaks detected in the wavelet-transformed EEG signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 1, 2, 1, 2])
        >>> detector = PeakDetection(signal, method="eeg_wavelet")
        >>> peaks = detector._eeg_wavelet_detection()
        >>> print(peaks)
        """
        wavelet_coeffs = np.abs(np.convolve(self.signal, np.ones(10), "same"))
        return find_peaks(wavelet_coeffs, height=np.mean(wavelet_coeffs))

    def _eeg_bandpass_detection(self):
        """
        Detect peaks in EEG signals after applying a bandpass filter.

        A bandpass filter is applied to the signal to isolate specific frequency ranges before peak detection.

        Returns
        -------
        peaks : numpy.ndarray
            Indices of peaks detected in the bandpass-filtered EEG signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 1, 2, 1, 2])
        >>> detector = PeakDetection(signal, method="eeg_bandpass", lowcut=0.5, highcut=50, fs=100)
        >>> peaks = detector._eeg_bandpass_detection()
        >>> print(peaks)
        """
        lowcut = self.kwargs.get("lowcut", 0.5)
        highcut = self.kwargs.get("highcut", 50)
        fs = self.kwargs.get("fs", 1.0)
        b, a = SignalFiltering.butter(5, [lowcut, highcut], btype="band", fs=fs)
        filtered_signal = filtfilt(b, a, self.signal)
        return find_peaks(filtered_signal, height=np.mean(filtered_signal))

    def _resp_autocorrelation_detection(self):
        """
        Detect peaks in respiratory signals using autocorrelation.

        Autocorrelation is used to find repeating patterns in the signal, with peaks representing periodic components.

        Returns
        -------
        peaks : numpy.ndarray
            Indices of peaks detected in the autocorrelated respiratory signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 1, 2, 1, 2])
        >>> detector = PeakDetection(signal, method="resp_autocorrelation")
        >>> peaks = detector._resp_autocorrelation_detection()
        >>> print(peaks)
        """
        autocorr = np.correlate(self.signal, self.signal, mode="full")
        return find_peaks(autocorr, distance=len(self.signal) // 2)

    def _resp_zero_crossing_detection(self):
        """
        Detect peaks in respiratory signals using zero-crossing method.

        Peaks are detected by finding zero crossings in the signal, which correspond to points where the signal changes direction.

        Returns
        -------
        peaks : numpy.ndarray
            Indices of peaks detected by finding zero crossings.

        Examples
        --------
        >>> signal = np.array([1, -1, 1, -1, 1, -1])
        >>> detector = PeakDetection(signal, method="resp_zero_crossing")
        >>> peaks = detector._resp_zero_crossing_detection()
        >>> print(peaks)
        """
        # Detect zero-crossings (sign changes)
        zero_crossings = np.where(np.diff(np.sign(self.signal)))[0]

        # Check if zero_crossings is non-empty and contains enough points
        if zero_crossings.size == 0:
            raise ValueError("No zero-crossings detected in the signal.")

        # Handle cases where there may be an odd number of zero-crossings
        if len(zero_crossings) % 2 != 0:
            zero_crossings = zero_crossings[
                :-1
            ]  # Ensure an even number for peak detection

        # Return every second zero-crossing as a peak
        peaks = zero_crossings[::2]

        return peaks

    def _abp_systolic_peak_detection(self):
        """
        Detect systolic peaks in ABP signals using Savitzky-Golay smoothing.

        Systolic peaks are identified in arterial blood pressure (ABP) signals by smoothing the signal and detecting peaks.

        Returns
        -------
        peaks : numpy.ndarray
            Indices of systolic peaks detected in the ABP signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 3, 2, 1])
        >>> detector = PeakDetection(signal, method="abp_systolic")
        >>> peaks = detector._abp_systolic_peak_detection()
        >>> print(peaks)
        """
        smoothed_signal = SignalFiltering.savgol_filter(
            self.signal, window_length=11, polyorder=2
        )
        return find_peaks(smoothed_signal, distance=60)

    def _abp_diastolic_peak_detection(self):
        """
        Detect diastolic peaks in ABP signals by inverting and smoothing the signal.

        Diastolic peaks are identified by inverting the arterial blood pressure (ABP) signal, smoothing it, and detecting peaks.

        Returns
        -------
        peaks : numpy.ndarray
            Indices of diastolic peaks detected in the ABP signal.

        Examples
        --------
        >>> signal = np.array([3, 2, 1, 2, 3, 4, 3])
        >>> detector = PeakDetection(signal, method="abp_diastolic")
        >>> peaks = detector._abp_diastolic_peak_detection()
        >>> print(peaks)
        """
        inverted_signal = -self.signal
        smoothed_signal = SignalFiltering.savgol_filter(
            inverted_signal, window_length=11, polyorder=2
        )
        return find_peaks(smoothed_signal, distance=60)
