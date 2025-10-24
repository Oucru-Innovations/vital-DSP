"""
Signal Filtering Module for Physiological Signal Processing

This module provides comprehensive signal filtering capabilities for physiological
signals including ECG, PPG, EEG, and other vital signs. It implements various
filtering techniques including bandpass, lowpass, highpass, and notch filters
with multiple filter types and adaptive parameter optimization.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Multiple filter types (Butterworth, Chebyshev, Elliptic, Bessel)
- Bandpass, lowpass, highpass, and notch filtering
- Adaptive parameter optimization
- Signal validation and error handling
- Real-time filtering capabilities
- Comprehensive filter design options

Examples:
--------
Basic bandpass filtering:
    >>> import numpy as np
    >>> from vitalDSP.filtering.signal_filtering import SignalFiltering, BandpassFilter
    >>> signal = np.random.randn(1000) + np.sin(np.linspace(0, 10, 1000))
    >>> filter_obj = SignalFiltering(signal, fs=250)
    >>> filtered = filter_obj.bandpass_filter(low=0.5, high=40)
    >>> print(f"Filtered signal shape: {filtered.shape}")

Advanced filtering with different types:
    >>> bp_filter = BandpassFilter(band_type="butter", fs=250)
    >>> butter_filtered = bp_filter.filter(signal, lowcut=0.5, highcut=40)
    >>> cheby_filter = BandpassFilter(band_type="cheby1", fs=250)
    >>> cheby_filtered = cheby_filter.filter(signal, lowcut=0.5, highcut=40)

Notch filtering for power line interference:
    >>> notch_filtered = filter_obj.notch_filter(freq=50, quality_factor=30)
    >>> print(f"Notch filtered signal shape: {notch_filtered.shape}")
"""

import numpy as np

# from scipy.signal import lfilter
from scipy import signal
import warnings
from vitalDSP.utils.data_processing.validation import SignalValidator
from vitalDSP.utils.config_utilities.adaptive_parameters import (
    optimize_filtering_parameters,
)

warnings.filterwarnings("ignore")


class BandpassFilter:
    def __init__(self, band_type="butter", fs=100):
        """
        Initializes the BandpassFilter with the specified filter type and sampling frequency.

        Parameters
        ----------
        band_type : str
            Type of the bandpass filter (e.g., 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel').
        fs : float
            Sampling frequency of the signal.

        Examples
        --------
        >>> filter = BandpassFilter(band_type="butter", fs=100)
        """
        self.band_type = band_type
        self.fs = fs

    def signal_bypass(self, cutoff, order, a_pass=3, rp=4, rs=40, btype="high"):
        """
        Generate the filter coefficients for the specified filter type and parameters.

        Parameters
        ----------
        cutoff : float
            Cutoff frequency of the filter.
        order : int
            Order of the filter.
        a_pass : float
            Passband ripple for Chebyshev and Elliptic filters.
        rp : float
            Passband ripple for Elliptic filters.
        rs : float
            Stopband attenuation for Elliptic filters.
        btype : str
            Type of filter ('high', 'low', 'bandpass').

        Returns
        -------
        b, a : tuple
            Numerator (b) and denominator (a) polynomials of the IIR filter.

        Examples
        --------
        >>> bp_filter = BandpassFilter("butter", fs=100)
        >>> b, a = bp_filter.signal_bypass(cutoff=0.3, order=4, a_pass=3, rp=4, rs=40, btype='low')
        >>> print(b, a)
        """
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        if self.band_type == "cheby1":
            b, a = signal.cheby1(
                order, a_pass, normal_cutoff, btype=btype, analog=False
            )
        elif self.band_type == "cheby2":
            b, a = signal.cheby2(
                order, a_pass, normal_cutoff, btype=btype, analog=False
            )
        elif self.band_type == "ellip":
            b, a = signal.ellip(order, rp, rs, normal_cutoff, btype=btype, analog=False)
        elif self.band_type == "bessel":
            b, a = signal.bessel(order, normal_cutoff, btype=btype, analog=False)
        else:
            b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
        return b, a

    def signal_lowpass_filter(self, data, cutoff, order=3, a_pass=3, rp=4, rs=40):
        """
        Apply a low-pass filter to the data.

        Parameters
        ----------
        data : numpy.ndarray
            The input signal.
        cutoff : float
            Cutoff frequency of the filter.
        order : int, optional
            Order of the filter. Default is 3.
        a_pass : float, optional
            Passband ripple for Chebyshev and Elliptic filters. Default is 3.
        rp : float, optional
            Passband ripple for Elliptic filters. Default is 4.
        rs : float, optional
            Stopband attenuation for Elliptic filters. Default is 40.

        Returns
        -------
        y : numpy.ndarray
            The filtered signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> bp_filter = BandpassFilter("butter", fs=100)
        >>> filtered_signal = bp_filter.signal_lowpass_filter(signal, cutoff=0.3, order=4)
        >>> print(filtered_signal)
        """
        b, a = self.signal_bypass(cutoff, order, a_pass, rp, rs, btype="low")
        y = signal.lfilter(b, a, data)
        return y

    def signal_highpass_filter(self, data, cutoff, order=5, a_pass=3, rp=4, rs=40):
        """
        Apply a high-pass filter to the data.

        Parameters
        ----------
        data : numpy.ndarray
            The input signal.
        cutoff : float
            Cutoff frequency of the filter.
        order : int, optional
            Order of the filter. Default is 5.
        a_pass : float, optional
            Passband ripple for Chebyshev and Elliptic filters. Default is 3.
        rp : float, optional
            Passband ripple for Elliptic filters. Default is 4.
        rs : float, optional
            Stopband attenuation for Elliptic filters. Default is 40.

        Returns
        -------
        y : numpy.ndarray
            The filtered signal.

        Raises
        ------
        ValueError
            If the length of the input signal is too short for the specified filter.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> bp_filter = BandpassFilter("butter", fs=100)
        >>> filtered_signal = bp_filter.signal_highpass_filter(signal, cutoff=0.3, order=4)
        >>> print(filtered_signal)
        """
        b, a = self.signal_bypass(cutoff, order, a_pass, rp, rs, btype="high")
        padlen = 3 * max(len(b), len(a))  # Minimum required pad length

        if len(data) <= padlen:
            raise ValueError(
                f"The length of the input vector x must be greater than {padlen}. "
                f"Consider reducing the filter order or increasing the signal length."
            )

        y = signal.filtfilt(b, a, data)
        return y


class SignalFiltering:
    """
    A class for applying various filtering techniques to signals.

    This class provides methods for common signal filtering tasks, such as applying moving averages,
    Gaussian filters, Butterworth filters, and median filters. These techniques are essential for
    preprocessing signals in various fields, including biomedical signal processing (e.g., ECG, EEG).

    Methods
    -------
    savgol_filter : static method
        Applies a Savitzky-Golay filter.
    moving_average : function
        Applies a moving average filter.
    gaussian : function
        Applies a Gaussian filter.
    butterworth : function
        Applies a Butterworth filter.
    chebyshev : function
        Applies a Chebyshev filter.
    elliptic : function
        Applies an Elliptic filter.
    bandpass : function
        Applies a bandpass filter using a selected filter type.
    median : function
        Applies a median filter.
    _apply_iir_filter : function
        Internal method to apply IIR filters.

    Examples
    --------
    >>> import numpy as np
    >>> from vitalDSP.filtering.signal_filtering import SignalFiltering
    >>>
    >>> # Example 1: Basic signal filtering
    >>> signal = np.random.randn(1000)  # Simulated signal
    >>> sf = SignalFiltering(signal)
    >>> filtered_signal = sf.bandpass(lowcut=0.5, highcut=30, fs=256, order=4)
    >>> print(f"Filtered signal shape: {filtered_signal.shape}")
    >>>
    >>> # Example 2: Moving average filtering
    >>> ma_filtered = sf.moving_average(window_size=5)
    >>> print(f"Moving average filtered: {ma_filtered.shape}")
    >>>
    >>> # Example 3: Gaussian filtering
    >>> gaussian_filtered = sf.gaussian(sigma=1.0)
    >>> print(f"Gaussian filtered: {gaussian_filtered.shape}")
    >>>
    >>> # Example 4: Savitzky-Golay filtering
    >>> sg_filtered = SignalFiltering.savgol_filter(signal, window_length=5, polyorder=2)
    >>> print(f"Savitzky-Golay filtered: {sg_filtered.shape}")
    """

    def __init__(self, signal):
        """
        Initialize the SignalFiltering class with the signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be filtered.

        Raises
        ------
        TypeError
            If the input signal is not a numpy array, it will be converted.
        ValueError
            If the signal is empty or invalid.
        """
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)

        # Validate signal - allow NaN values for transformation contexts
        SignalValidator.validate_signal(
            signal,
            min_length=2,
            allow_empty=False,
            allow_nan=True,
            signal_name="signal",
        )

        self.signal = signal

    @staticmethod
    def savgol_filter(signal, window_length, polyorder):
        """
        Apply a Savitzky-Golay filter to smooth the signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal.
        window_length : int
            The length of the filter window (must be odd).
        polyorder : int
            The order of the polynomial to fit.

        Returns
        -------
        smoothed_signal : numpy.ndarray
            The smoothed signal.

        Examples
        --------
        >>> import numpy as np
        >>> signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> filter = SignalFiltering(signal)
        >>> smoothed_signal = filter.savgol_filter(signal, 3, 2)
        >>> print(smoothed_signal)
        [1. 2. 3. 4. 5. 6. 7. 8. 9.]
        """
        if window_length % 2 == 0 or window_length < 1:
            raise ValueError("window_length must be a positive odd integer")
        if window_length < polyorder + 2:
            raise ValueError("window_length is too small for the polynomial order")

        half_window = (window_length - 1) // 2
        b = np.asmatrix(
            [
                [k**i for i in range(polyorder + 1)]
                for k in range(-half_window, half_window + 1)
            ]
        )
        m = np.linalg.pinv(b).A[0]
        firstvals = signal[0] - np.abs(signal[1 : half_window + 1][::-1] - signal[0])
        lastvals = signal[-1] + np.abs(signal[-half_window - 1 : -1][::-1] - signal[-1])
        signal = np.concatenate((firstvals, signal, lastvals))
        smoothed_signal = np.convolve(m, signal, mode="valid")
        return smoothed_signal

    def moving_average(self, window_size, iterations=1, method="edge"):
        """
        Applies a moving average filter to the signal with optional repeated scanning.

        This method smooths the signal by averaging neighboring data points within a defined window size.
        Optionally, the smoothing can be repeated multiple times for enhanced effect. This technique is
        commonly used to reduce random noise and reveal trends in signals like EEG, ECG, and PPG.

        Parameters
        ----------
        window_size : int
            The size of the moving window.
        iterations : int, optional
            The number of times to apply the moving average for additional smoothing. Default is 1.
        method : str, optional
            Padding method: 'edge' (default), 'reflect', or 'constant'. Different methods may yield better results
            for specific types of signals (e.g., vital signals like EEG, ECG, PPG).

        Returns
        -------
        filtered_signal : numpy.ndarray
            The filtered signal.

        Examples
        --------
        >>> import numpy as np
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> sf = SignalFiltering(signal)
        >>> filtered_signal = sf.moving_average(3, iterations=2, method="reflect")
        >>> print(filtered_signal)
        """
        filtered_signal = self.signal.copy()

        for _ in range(iterations):
            # Apply padding to the signal based on the chosen method
            padded_signal = np.pad(
                filtered_signal,
                (window_size // 2, window_size - 1 - window_size // 2),
                mode=method,
            )
            # Calculate the moving average
            filtered_signal = np.convolve(
                padded_signal, np.ones(window_size) / window_size, mode="valid"
            )

        return filtered_signal

    def gaussian(self, sigma=1.0, iterations=1):
        """
        Applies a Gaussian filter to the signal.

        Parameters
        ----------
        sigma : float
            The standard deviation of the Gaussian kernel.
        iterations : int, optional
            The number of times to apply the Gaussian filter for additional smoothing. Default is 1.

        Returns
        -------
        filtered_signal : numpy.ndarray
            The filtered signal.

        Examples
        --------
        >>> import numpy as np
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> sf = SignalFiltering(signal)
        >>> filtered_signal = sf.gaussian(1.5)
        >>> print(filtered_signal)
        """
        size = int(6 * sigma + 1) if int(6 * sigma + 1) % 2 != 0 else int(6 * sigma + 2)
        kernel = self.gaussian_kernel(size, sigma)
        filtered_signal = self.signal.copy()

        for _ in range(iterations):
            filtered_signal = np.convolve(filtered_signal, kernel, mode="same")

        return filtered_signal

    @staticmethod
    def gaussian_filter1d(signal, sigma):
        """
        Custom implementation of a 1D Gaussian filter.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal.
        sigma : float
            The standard deviation of the Gaussian kernel.

        Returns
        -------
        smoothed_signal : numpy.ndarray
            The smoothed signal.

        Examples
        --------
        >>> import numpy as np
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> smoothed_signal = SignalFiltering.gaussian_filter1d(signal, 1.0)
        >>> print(smoothed_signal)
        [1.14285714 2.14285714 3. 4. 5.]
        """
        radius = int(4 * sigma + 0.5)
        x = np.arange(-radius, radius + 1)
        gaussian_kernel = np.exp(-0.5 * (x / sigma) ** 2)
        gaussian_kernel /= gaussian_kernel.sum()

        # Pad the signal to handle boundary effects
        pad_width = radius
        padded_signal = np.pad(signal, pad_width, mode="edge")

        # Convolve with the Gaussian kernel
        smoothed_signal = np.convolve(padded_signal, gaussian_kernel, mode="valid")
        return smoothed_signal[: len(signal)]

    @staticmethod
    def gaussian_kernel(size, sigma):
        """
        Generate a Gaussian kernel.

        Parameters
        ----------
        size : int
            The size of the kernel (must be odd).
        sigma : float
            Standard deviation of the Gaussian distribution.

        Returns
        -------
        kernel : numpy.ndarray
            The Gaussian kernel.

        Examples
        --------
        >>> kernel = SignalFiltering.gaussian_kernel(5, sigma=1.0)
        >>> print(kernel)
        [0.05448868 0.24420134 0.40261995 0.24420134 0.05448868]
        """
        ax = np.arange(-(size // 2), (size // 2) + 1)
        kernel = np.exp(-0.5 * (ax / sigma) ** 2)
        kernel = kernel / np.sum(kernel)
        return kernel

    def butterworth(
        self, cutoff, fs, order=4, btype="low", iterations=1, adaptive=True
    ):
        """
        Apply a Butterworth filter to the signal.

        Parameters
        ----------
        cutoff : float
            Cutoff frequency of the filter.
        fs : float
            Sampling frequency of the signal.
        order : int, optional
            Order of the Butterworth filter. Default is 4.
        btype : str, optional
            Type of filter - 'low' or 'high'. Default is 'low'.
        iterations : int, optional
            The number of times to apply the Butterworth filter for additional filtering. Default is 1.
        adaptive : bool, optional
            Whether to use adaptive parameter adjustment. Default is True.

        Returns
        -------
        filtered_signal : numpy.ndarray
            The filtered signal.

        Examples
        --------
        >>> import numpy as np
        >>> fs = 1000  # Sampling frequency
        >>> cutoff = 0.5  # Cutoff frequency
        >>> signal = np.sin(2 * np.pi * 0.1 * np.arange(0, 10, 1/fs)) + np.random.randn(10 * fs) * 0.1
        >>> sf = SignalFiltering(signal)
        >>> filtered_signal = sf.butterworth(cutoff, fs, order=4)
        >>> print(filtered_signal)
        """
        # Input validation
        SignalValidator.validate_signal(
            self.signal, min_length=10, signal_name="input signal"
        )
        cutoff, fs = SignalValidator.validate_frequency_parameters(cutoff, fs)
        order = SignalValidator.validate_filter_order(order)

        if iterations < 1:
            raise ValueError("Iterations must be positive")

        # Adaptive parameter adjustment
        if adaptive:
            base_params = {
                "cutoff": cutoff,
                "fs": fs,
                "order": order,
                "iterations": iterations,
            }
            optimized_params = optimize_filtering_parameters(
                self.signal, fs, base_params
            )
            cutoff = optimized_params["cutoff"]
            fs = optimized_params["fs"]
            order = optimized_params["order"]
            iterations = optimized_params["iterations"]

        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = self.butter(order, normal_cutoff, btype=btype, fs=fs)
        filtered_signal = self.signal.copy()

        for _ in range(iterations):
            filtered_signal = self._apply_iir_filter(b, a, filtered_signal)

        return filtered_signal

    def butter(self, order, cutoff, btype="low", fs=1.0):
        """
        Custom implementation of the Butterworth filter design using bilinear transformation.

        Parameters
        ----------
        order : int
            The order of the filter.
        cutoff : float or list of float
            The critical frequency or frequencies.
        btype : str
            The type of filter ('low', 'high', 'band').
        fs : float
            The sampling frequency.

        Returns
        -------
        b, a : tuple
            Numerator (b) and denominator (a) polynomials of the IIR filter.

        Examples
        --------
        >>> import numpy as np
        >>> b, a = SignalFiltering().butter(4, 0.3, btype='low', fs=1.0)
        >>> print(b, a)
        """
        # Check if cutoff is already normalized (between 0 and 1)
        if np.all(np.array(cutoff) <= 1.0):
            # Cutoff is already normalized
            normalized_cutoff = np.array(cutoff)
        else:
            # Cutoff needs normalization
            nyquist = 0.5 * fs
            normalized_cutoff = np.array(cutoff) / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype=btype, analog=False)
        # if btype == "low":
        #     poles = np.exp(
        #         1j * np.pi * (np.arange(1, 2 * order + 1, 2) + order - 1) / (2 * order)
        #     )
        #     poles = poles[np.real(poles) < 0]
        #     z, p = np.zeros(0), poles
        # elif btype == "high":
        #     z, p = self.butter(order, cutoff, btype="low")
        #     z, p = -z, -p
        # elif btype == "band":
        #     low = np.min(normalized_cutoff)
        #     high = np.max(normalized_cutoff)
        #     z_low, p_low = self.butter(order, low, btype="high")
        #     z_high, p_high = self.butter(order, high, btype="low")
        #     z = np.concatenate([z_low, z_high])
        #     p = np.concatenate([p_low, p_high])
        # else:
        #     raise ValueError("Invalid btype. Must be 'low', 'high', or 'band'.")

        # b, a = np.poly(z), np.poly(p)
        # b = np.atleast_1d(b)
        # a = np.atleast_1d(a)
        # b /= np.abs(np.sum(a))
        return b, a

    def chebyshev(self, cutoff, fs, order=4, btype="low", ripple=0.05, iterations=1):
        """
        Custom implementation of the Chebyshev Type I filter.

        Parameters
        ----------
        cutoff : float
            Cutoff frequency of the filter.
        fs : float
            Sampling frequency of the signal.
        order : int, optional
            Order of the Chebyshev filter. Default is 4.
        btype : str, optional
            Type of filter - 'low' or 'high'. Default is 'low'.
        ripple : float, optional
            The maximum ripple allowed in the passband. Default is 0.05.
        iterations : int, optional
            The number of times to apply the Chebyshev filter for additional filtering. Default is 1.

        Returns
        -------
        filtered_signal : numpy.ndarray
            The filtered signal.
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        eps = np.sqrt(10 ** (ripple / 10) - 1)
        k = np.arange(1, 2 * order, 2)
        poles = np.exp(1j * np.pi * (k + order - 1) / (2 * order))
        poles = poles[np.real(poles) < 0]
        poles = np.concatenate([poles, -poles.conj()])
        poles /= eps ** (1 / order)
        if btype == "low":
            poles = poles * normal_cutoff
        elif btype == "high":
            poles = normal_cutoff / poles
        z = np.zeros(0)
        b, a = np.poly(z), np.poly(poles)
        b = np.atleast_1d(b)
        a = np.atleast_1d(a)
        b = b / np.abs(np.polyval(b, 1))
        a = a / np.abs(np.polyval(a, 1))

        filtered_signal = self.signal.copy()

        for _ in range(iterations):
            filtered_signal = self._apply_iir_filter(b, a, filtered_signal)

        return filtered_signal

    def elliptic(
        self,
        cutoff,
        fs,
        order=4,
        btype="low",
        ripple=0.05,
        stopband_attenuation=40,
        iterations=1,
    ):
        """
        Custom implementation of the Elliptic filter.

        Parameters
        ----------
        cutoff : float
            Cutoff frequency of the filter.
        fs : float
            Sampling frequency of the signal.
        order : int, optional
            Order of the Elliptic filter. Default is 4.
        btype : str, optional
            Type of filter - 'low' or 'high'. Default is 'low'.
        ripple : float, optional
            The maximum ripple allowed in the passband. Default is 0.05.
        stopband_attenuation : float, optional
            Minimum attenuation in the stopband. Default is 40 dB.
        iterations : int, optional
            The number of times to apply the Elliptic filter for additional filtering. Default is 1.

        Returns
        -------
        filtered_signal : numpy.ndarray
            The filtered signal.
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist

        eps = np.sqrt(10 ** (ripple / 10) - 1)
        k = eps / np.sqrt(1 + eps**2)

        poles = []
        for i in range(1, order + 1):
            beta = np.arcsin(k)
            theta = (2 * i - 1) * np.pi / (2 * order)
            poles.append(np.exp(1j * (theta + beta)))
            poles.append(np.exp(1j * (theta - beta)))

        poles = np.array(poles)
        poles = poles[np.real(poles) < 0]

        if btype == "low":
            poles = poles * normal_cutoff
        elif btype == "high":
            poles = normal_cutoff / poles

        z = np.zeros(0)
        b, a = np.poly(z), np.poly(poles)
        b = np.atleast_1d(b)
        a = np.atleast_1d(a)
        b = b / np.abs(np.polyval(b, 1))
        a = a / np.abs(np.polyval(a, 1))

        filtered_signal = self.signal.copy()

        for _ in range(iterations):
            filtered_signal = self._apply_iir_filter(b, a, filtered_signal)

        return filtered_signal

    def chebyshev2(
        self, cutoff, fs, order=4, btype="low", stopband_attenuation=40, iterations=1
    ):
        """
        Chebyshev Type II filter implementation.

        The Chebyshev Type II filter has a flat passband and equiripple stopband attenuation.
        Unlike Type I, it has zeros in the stopband which provides sharper roll-off
        characteristics. This filter is useful when you need better stopband attenuation
        with acceptable passband characteristics.

        Parameters
        ----------
        cutoff : float or list
            Cutoff frequency of the filter. For bandpass/bandstop, provide [low, high].
        fs : float
            Sampling frequency of the signal.
        order : int, optional
            Order of the Chebyshev Type II filter. Default is 4.
        btype : str, optional
            Type of filter - 'low', 'high', 'band', or 'bandstop'. Default is 'low'.
        stopband_attenuation : float, optional
            Minimum attenuation required in the stopband in dB. Default is 40 dB.
        iterations : int, optional
            The number of times to apply the filter. Default is 1.

        Returns
        -------
        filtered_signal : numpy.ndarray
            The filtered signal.

        Examples
        --------
        >>> import numpy as np
        >>> from vitalDSP.filtering.signal_filtering import SignalFiltering
        >>> signal = np.random.randn(1000)
        >>> sf = SignalFiltering(signal)
        >>> filtered = sf.chebyshev2(cutoff=50, fs=250, order=4, btype='low', stopband_attenuation=40)
        >>> print(f"Filtered signal shape: {filtered.shape}")
        """
        from scipy import signal as sp_signal

        nyquist = fs / 2

        # Normalize cutoff frequency
        if isinstance(cutoff, list):
            normal_cutoff = [c / nyquist for c in cutoff]
        else:
            normal_cutoff = cutoff / nyquist

        # Design Chebyshev Type II filter
        b, a = sp_signal.cheby2(order, stopband_attenuation, normal_cutoff, btype=btype, analog=False)

        # Apply filter iteratively
        filtered_signal = self.signal.copy()
        for _ in range(iterations):
            filtered_signal = sp_signal.filtfilt(b, a, filtered_signal)

        return filtered_signal

    def bessel(self, cutoff, fs, order=4, btype="low", iterations=1):
        """
        Bessel (Thomson) filter implementation.

        The Bessel filter has a maximally flat group delay, which means it preserves
        the waveform shape of filtered signals in the passband. This makes it ideal
        for applications where maintaining pulse shape is critical, such as ECG and
        PPG signal processing.

        Parameters
        ----------
        cutoff : float or list
            Cutoff frequency of the filter. For bandpass/bandstop, provide [low, high].
        fs : float
            Sampling frequency of the signal.
        order : int, optional
            Order of the Bessel filter. Default is 4.
        btype : str, optional
            Type of filter - 'low', 'high', 'band', or 'bandstop'. Default is 'low'.
        iterations : int, optional
            The number of times to apply the filter. Default is 1.

        Returns
        -------
        filtered_signal : numpy.ndarray
            The filtered signal.

        Notes
        -----
        The Bessel filter is particularly useful for:
        - ECG signal processing (preserves QRS complex shape)
        - PPG signal processing (preserves pulse waveform)
        - Applications requiring linear phase response
        - Situations where overshoot and ringing must be minimized

        Examples
        --------
        >>> import numpy as np
        >>> from vitalDSP.filtering.signal_filtering import SignalFiltering
        >>> # ECG-like signal with sharp peaks
        >>> t = np.linspace(0, 1, 1000)
        >>> ecg_signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)
        >>> sf = SignalFiltering(ecg_signal)
        >>> # Apply Bessel lowpass to remove high-frequency noise while preserving peak shape
        >>> filtered = sf.bessel(cutoff=50, fs=1000, order=4, btype='low')
        >>> print(f"Filtered signal shape: {filtered.shape}")
        """
        from scipy import signal as sp_signal

        nyquist = fs / 2

        # Normalize cutoff frequency
        if isinstance(cutoff, list):
            normal_cutoff = [c / nyquist for c in cutoff]
        else:
            normal_cutoff = cutoff / nyquist

        # Design Bessel filter
        # Note: Bessel filter is also known as Thomson filter
        b, a = sp_signal.bessel(order, normal_cutoff, btype=btype, analog=False, norm='phase')

        # Apply filter iteratively
        filtered_signal = self.signal.copy()
        for _ in range(iterations):
            filtered_signal = sp_signal.filtfilt(b, a, filtered_signal)

        return filtered_signal

    def bandpass(
        self, lowcut, highcut, fs, order=4, filter_type="butter", iterations=1
    ):
        """
        Apply a bandpass filter using the selected filter type.

        Parameters
        ----------
        lowcut : float
            The lower cutoff frequency.
        highcut : float
            The upper cutoff frequency.
        fs : float
            The sampling frequency.
        order : int, optional
            The order of the filter. Default is 4.
        filter_type : str, optional
            Type of filter to apply ('butter', 'cheby', 'elliptic'). Default is 'butter'.

        Returns
        -------
        filtered_signal : numpy.ndarray
            The filtered signal.

        Examples
        --------
        >>> import numpy as np
        >>> fs = 1000  # Sampling frequency
        >>> lowcut = 0.5  # Lower cutoff frequency
        >>> highcut = 50  # Upper cutoff frequency
        >>> signal = np.sin(2 * np.pi * 10 * np.arange(0, 10, 1/fs)) + np.random.randn(10 * fs) * 0.1
        >>> sf = SignalFiltering(signal)
        >>> filtered_signal = sf.bandpass(lowcut, highcut, fs, order=4, filter_type='butter')
        >>> print(filtered_signal)
        """
        # nyquist = 0.5 * fs
        # low = lowcut / nyquist
        # high = highcut / nyquist

        filtered_signal = self.signal.copy()

        for _ in range(iterations):
            if filter_type == "butter":
                b, a = self.butter(order, [lowcut, highcut], btype="band", fs=fs)
            elif filter_type == "cheby":
                b, a = self.chebyshev(order, [lowcut, highcut], btype="band", fs=fs)
            elif filter_type == "elliptic":
                b, a = self.elliptic(order, [lowcut, highcut], btype="band", fs=fs)
            else:
                raise ValueError(
                    "Unsupported filter type. Choose from 'butter', 'cheby', or 'elliptic'."
                )

            filtered_signal = self._apply_iir_filter(b, a, filtered_signal)

        return filtered_signal

    def median(self, kernel_size=3, iterations=1, method="edge"):
        """
        Apply a median filter to the signal with optional repeated filtering.

        Parameters
        ----------
        kernel_size : int, optional
            Size of the median filter kernel. Default is 3.
        iterations : int, optional
            The number of times to apply the median filter for additional filtering. Default is 1.
        method : str, optional
            Padding method: 'edge' (default), 'reflect', or 'constant'.

        Returns
        -------
        filtered_signal : numpy.ndarray
            The filtered signal.

        Examples
        --------
        >>> import numpy as np
        >>> signal = np.array([1, 2, 3, 100, 5, 6, 7, 8, 9, 10])  # Note the spike at value 100
        >>> sf = SignalFiltering(signal)
        >>> filtered_signal = sf.median(kernel_size=3)
        >>> print(filtered_signal)
        [1 2 3 5 5 6 7 8 9 9]
        """
        filtered_signal = self.signal.copy()

        for _ in range(iterations):
            padded_signal = np.pad(
                filtered_signal,
                (kernel_size // 2, kernel_size - 1 - kernel_size // 2),
                mode=method,
            )
            filtered_signal = np.array(
                [
                    np.median(padded_signal[i : i + kernel_size])
                    for i in range(len(self.signal))
                ]
            )

        return filtered_signal

    def _apply_iir_filter(self, b, a, filtered_signal=None):
        """
        Apply an IIR filter using the provided coefficients.

        This method is enhanced to prevent division by zero or issues with infinity by adding small
        epsilon values where necessary.

        Parameters
        ----------
        b : numpy.ndarray
            Numerator (b) polynomial coefficients of the IIR filter.
        a : numpy.ndarray
            Denominator (a) polynomial coefficients of the IIR filter.

        Returns
        -------
        filtered_signal : numpy.ndarray
            The filtered signal.
        """
        if filtered_signal is None:
            # filtered_signal = np.zeros_like(self.signal)
            filtered_signal = self.signal.copy()
        # epsilon = 1e-10  # Small constant to prevent division by zero

        # # Ensure the first coefficient of a is not zero to avoid instability
        # a[0] = max(a[0], epsilon)

        # for i in range(len(self.signal)):
        #     filtered_signal[i] = b[0] * self.signal[i]

        #     if i > 0 and len(b) > 1:
        #         filtered_signal[i] = np.clip(
        #             np.real(filtered_signal[i])
        #             + (
        #                 b[1] * self.signal[i - 1]
        #                 - a[1] * np.real(filtered_signal[i - 1])
        #             ),
        #             -1e10,
        #             1e10,
        #         )
        #         # filtered_signal[i] += (
        #         #     b[1] * self.signal[i - 1] - a[1] * filtered_signal[i - 1]
        #         # )

        #     if i > 1 and len(b) > 2:
        #         filtered_signal[i] = np.clip(
        #             np.real(filtered_signal[i])
        #             + (
        #                 b[2] * self.signal[i - 2]
        #                 - a[2] * np.real(filtered_signal[i - 2])
        #             ),
        #             -1e10,
        #             1e10,
        #         )
        #         # filtered_signal[i] += (
        #         #     b[2] * self.signal[i - 2] - a[2] * filtered_signal[i - 2]
        #         # )

        #     # Normalize by a[0] to ensure stability
        #     filtered_signal[i] = np.real(filtered_signal[i]) / a[0]

        #     # Apply epsilon to avoid infinities or NaNs
        #     filtered_signal[i] = np.nan_to_num(
        #         filtered_signal[i], nan=0.0, posinf=0.0, neginf=0.0
        #     )

        # Use filtfilt to avoid phase distortion
        filtered_signal = signal.filtfilt(b, a, filtered_signal)
        return filtered_signal
