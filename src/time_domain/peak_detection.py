import numpy as np
from utils.common import find_peaks, filtfilt, argrelextrema, StandardScaler
from filtering.signal_filtering import SignalFiltering


class PeakDetection:
    """
    A class to detect peaks in various physiological signals like ECG, PPG, EEG, respiratory, and arterial blood pressure (ABP) signals.

    Methods:
    - detect_peaks: Detects the peaks in the signal using the selected method.
    """

    def __init__(self, signal, method="threshold", **kwargs):
        """
        Initialize the PeakDetection class with the signal.

        Parameters:
        signal (numpy.ndarray): The input signal.
        method (str): The method to use for peak detection ('threshold', 'savgol', 'gaussian', 'rel_extrema', 'scaler_threshold', 'ecg_r_peak', 'ecg_derivative', 'ppg_first_derivative', 'ppg_second_derivative', 'eeg_wavelet', 'eeg_bandpass', 'resp_autocorrelation', 'resp_zero_crossing', 'abp_systolic', 'abp_diastolic').
        kwargs: Additional parameters for specific methods.
        """
        self.signal = signal
        self.method = method
        self.kwargs = kwargs

    def detect_peaks(self):
        """
        Detect peaks in the signal based on the selected method.

        Returns:
        numpy.ndarray: The indices of the detected peaks.
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
        threshold = self.kwargs.get("threshold", 0.5)
        distance = self.kwargs.get("distance", 50)
        return find_peaks(self.signal, height=threshold, distance=distance)

    def _savgol_based_detection(self):
        window_length = self.kwargs.get("window_length", 11)
        polyorder = self.kwargs.get("polyorder", 2)
        smoothed_signal = SignalFiltering.savgol_filter(
            self.signal, window_length=window_length, polyorder=polyorder
        )
        return find_peaks(smoothed_signal)

    def _gaussian_based_detection(self):
        sigma = self.kwargs.get("sigma", 1)
        smoothed_signal = SignalFiltering.gaussian_filter1d(self.signal, sigma=sigma)
        return find_peaks(smoothed_signal)

    def _relative_extrema_detection(self):
        order = self.kwargs.get("order", 1)
        peaks = argrelextrema(self.signal, np.greater, order=order)
        return peaks

    def _scaler_threshold_based_detection(self):
        scaler = StandardScaler()
        standardized_signal = scaler.fit_transform(self.signal)
        threshold = self.kwargs.get("threshold", 1)
        distance = self.kwargs.get("distance", 50)
        return find_peaks(standardized_signal, height=threshold, distance=distance)

    def _ecg_r_peak_detection(self):
        diff_signal = np.diff(self.signal)
        squared_signal = diff_signal**2
        integrator = np.convolve(squared_signal, np.ones(150), "same")
        return find_peaks(integrator, distance=150, height=np.mean(integrator))

    def _ecg_derivative_detection(self):
        diff_signal = np.diff(self.signal)
        return find_peaks(np.abs(diff_signal), height=np.mean(np.abs(diff_signal)))

    def _ppg_first_derivative_detection(self):
        diff_signal = np.diff(self.signal)
        return find_peaks(diff_signal, height=np.mean(diff_signal))

    def _ppg_second_derivative_detection(self):
        diff_signal = np.diff(self.signal, n=2)
        return find_peaks(diff_signal, height=np.mean(diff_signal))

    def _eeg_wavelet_detection(self):
        wavelet_coeffs = np.abs(np.convolve(self.signal, np.ones(10), "same"))
        return find_peaks(wavelet_coeffs, height=np.mean(wavelet_coeffs))

    def _eeg_bandpass_detection(self):
        lowcut = self.kwargs.get("lowcut", 0.5)
        highcut = self.kwargs.get("highcut", 50)
        fs = self.kwargs.get("fs", 1.0)
        b, a = SignalFiltering.butter(5, [lowcut, highcut], btype="band", fs=fs)
        filtered_signal = filtfilt(b, a, self.signal)
        return find_peaks(filtered_signal, height=np.mean(filtered_signal))

    def _resp_autocorrelation_detection(self):
        autocorr = np.correlate(self.signal, self.signal, mode="full")
        return find_peaks(autocorr, distance=len(self.signal) // 2)

    def _resp_zero_crossing_detection(self):
        zero_crossings = np.where(np.diff(np.sign(self.signal)))[0]
        peaks = zero_crossings[::2]
        return peaks

    def _abp_systolic_peak_detection(self):
        smoothed_signal = SignalFiltering.savgol_filter(
            self.signal, window_length=11, polyorder=2
        )
        return find_peaks(smoothed_signal, distance=60)

    def _abp_diastolic_peak_detection(self):
        inverted_signal = -self.signal
        smoothed_signal = SignalFiltering.savgol_filter(
            inverted_signal, window_length=11, polyorder=2
        )
        return find_peaks(smoothed_signal, distance=60)
