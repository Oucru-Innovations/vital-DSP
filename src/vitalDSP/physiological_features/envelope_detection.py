import numpy as np
from vitalDSP.transforms.wavelet_transform import WaveletTransform


class EnvelopeDetection:
    """
    A comprehensive class for detecting the envelope of physiological signals.

    This class offers several methods to compute the envelope of a given signal using different techniques,
    ranging from classical methods like the Hilbert transform to more advanced approaches like wavelet
    transform and machine learning-inspired methods.

    Methods
    -------
    hilbert_envelope : function
        Computes the envelope using the Hilbert transform.
    moving_average_envelope : function
        Computes the envelope using a moving average filter.
    absolute_value_envelope : function
        Computes the envelope by taking the absolute value and applying exponential smoothing.
    peak_envelope : function
        Computes the envelope by identifying and connecting peaks in the signal.
    wavelet_envelope : function
        Computes the envelope using wavelet decomposition.
    adaptive_filter_envelope : function
        Computes the envelope using an adaptive filtering technique.
    ml_based_envelope : function
        Computes the envelope using a machine learning-inspired approach.
    """

    def __init__(self, signal):
        """
        Initialize the EnvelopeDetection class with the signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input physiological signal. This could be any time-series data such as ECG, PPG, or EEG signals.
        """
        self.signal = signal

    def wavelet_envelope(self, wavelet_name="db", level=1):
        """
        Compute the envelope using wavelet transform.

        This method decomposes the signal using wavelet transform, typically capturing
        the low-frequency components which represent the envelope.

        Parameters
        ----------
        wavelet_name : str, optional (default="db")
            The type of wavelet to use, such as 'db' (Daubechies), 'haar', 'sym', etc.
        level : int, optional (default=1)
            The level of decomposition, which determines the resolution of the envelope.

        Returns
        -------
        envelope : numpy.ndarray
            The computed envelope of the signal.
        """
        wavelet_transform = WaveletTransform(self.signal, wavelet_name=wavelet_name)
        coeffs = wavelet_transform.perform_wavelet_transform(level=level)

        # The envelope is typically represented by the low-frequency approximation coefficients
        envelope = np.abs(coeffs[-1])

        # Calculate the repetition factor and resample the envelope
        repeat_factor = len(self.signal) // len(envelope)
        remainder = len(self.signal) % len(envelope)

        # Repeat the envelope to match the signal length
        resampled_envelope = np.repeat(envelope, repeat_factor)

        # Handle the case where the signal length isn't a multiple of the envelope length
        if remainder > 0:
            resampled_envelope = np.concatenate(
                [resampled_envelope, envelope[:remainder]]
            )

        return resampled_envelope

    def hilbert_envelope(self):
        """
        Compute the envelope using the Hilbert transform.

        The Hilbert transform is a mathematical operation that produces the analytic signal of a real-valued
        signal, allowing for the computation of the amplitude envelope.

        Returns
        -------
        envelope : numpy.ndarray
            The computed envelope of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> ed = EnvelopeDetection(signal)
        >>> envelope = ed.hilbert_envelope()
        >>> print(envelope)
        [1. 2. 3. 4. 5.]
        """
        analytic_signal = self.signal + 1j * np.imag(
            np.fft.ifft(np.fft.fft(self.signal) * 2)
        )
        envelope = np.abs(analytic_signal)
        return envelope

    def moving_average_envelope(self, window_size):
        """
        Compute the envelope using a moving average filter.

        The moving average envelope is computed by taking the absolute value of the signal and then applying
        a moving average filter, which smooths the signal and provides the envelope.

        Parameters
        ----------
        window_size : int
            The size of the moving average window. A larger window results in a smoother envelope.

        Returns
        -------
        envelope : numpy.ndarray
            The computed envelope of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> ed = EnvelopeDetection(signal)
        >>> envelope = ed.moving_average_envelope(3)
        >>> print(envelope)
        [2. 3. 4.]
        """
        cumsum = np.cumsum(np.insert(np.abs(self.signal), 0, 0))
        moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
        return moving_avg

    def absolute_value_envelope(self, smoothing_factor=0.01):
        """
        Compute the envelope by taking the absolute value and applying exponential smoothing.

        This method first computes the absolute value of the signal and then applies an exponential smoothing
        function to estimate the envelope.

        Parameters
        ----------
        smoothing_factor : float, optional
            The smoothing factor for exponential smoothing, with a typical range between 0 and 1.
            Smaller values result in more smoothing.

        Returns
        -------
        envelope : numpy.ndarray
            The smoothed envelope of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> ed = EnvelopeDetection(signal)
        >>> envelope = ed.absolute_value_envelope(0.1)
        >>> print(envelope)
        """
        absolute_signal = np.abs(self.signal)
        smoothed_signal = np.zeros_like(absolute_signal)
        smoothed_signal[0] = absolute_signal[0]
        for i in range(1, len(absolute_signal)):
            smoothed_signal[i] = (1 - smoothing_factor) * smoothed_signal[
                i - 1
            ] + smoothing_factor * absolute_signal[i]
        return smoothed_signal

    def peak_envelope(self, interpolation_method="linear"):
        """
        Compute the envelope by identifying and connecting peaks in the signal.

        This method finds local maxima (peaks) in the signal and connects them using interpolation
        to form the envelope.

        Parameters
        ----------
        interpolation_method : str, optional
            The method to use for interpolation, which can be 'linear', 'quadratic', or 'cubic'.

        Returns
        -------
        envelope : numpy.ndarray
            The computed envelope of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> ed = EnvelopeDetection(signal)
        >>> envelope = ed.peak_envelope('linear')
        >>> print(envelope)
        """
        peaks = (
            np.where(
                (self.signal[1:-1] > self.signal[:-2])
                & (self.signal[1:-1] > self.signal[2:])
            )[0]
            + 1
        )
        envelope = np.interp(
            np.arange(len(self.signal)),
            peaks,
            self.signal[peaks],
            left=self.signal[0],
            right=self.signal[-1],
        )
        return envelope

    def adaptive_filter_envelope(self, step_size=0.01, filter_order=10):
        """
        Compute the envelope using adaptive filtering.

        Adaptive filtering adjusts its parameters dynamically to track the signal characteristics,
        making it suitable for real-time envelope detection.

        Parameters
        ----------
        step_size : float, optional
            The step size for the adaptive filter, which controls the speed of adaptation.
        filter_order : int, optional
            The order of the adaptive filter, which determines the filter's complexity.

        Returns
        -------
        envelope : numpy.ndarray
            The computed envelope of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> ed = EnvelopeDetection(signal)
        >>> envelope = ed.adaptive_filter_envelope(0.01, 10)
        >>> print(envelope)
        """
        y = np.zeros_like(self.signal)
        w = np.zeros(filter_order)
        for i in range(filter_order, len(self.signal)):
            x = self.signal[i - filter_order : i][::-1]
            y[i] = np.dot(w, x)
            e = self.signal[i] - y[i]
            w += step_size * e * x
        return np.abs(y)

    def ml_based_envelope(self, model=None):
        """
        Compute the envelope using a machine learning-inspired method.

        This method allows the use of a custom machine learning model or function
        to predict the envelope of the signal.

        Parameters
        ----------
        model : callable or None, optional
            A custom model or function for predicting the envelope. If None, a simple moving average model is used.

        Returns
        -------
        envelope : numpy.ndarray
            The computed envelope of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> ed = EnvelopeDetection(signal)
        >>> envelope = ed.ml_based_envelope()
        >>> print(envelope)
        """
        if model is None:
            # Example simple model: predict next value as a weighted sum of previous values
            model = lambda x: np.convolve(x, np.ones(5) / 5, mode="same")

        envelope = model(np.abs(self.signal))
        return envelope
