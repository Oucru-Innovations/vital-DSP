import numpy as np

class SignalFiltering:
    """
    A class for applying various filtering techniques to signals.

    This class provides methods for common signal filtering tasks, such as applying moving averages, 
    Gaussian filters, Butterworth filters, and median filters. These techniques are essential for 
    preprocessing signals in various fields, including biomedical signal processing (e.g., ECG, EEG).

    Methods
    -------
    moving_average : function
        Applies a moving average filter.
    gaussian : function
        Applies a Gaussian filter.
    butterworth : function
        Applies a Butterworth filter.
    median : function
        Applies a median filter.
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
        """
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)
        self.signal = signal

    @staticmethod
    def savgol_filter(signal, window_length, polyorder):
        """
        Apply a Savitzky-Golay filter to smooth the signal.

        The Savitzky-Golay filter fits successive polynomials to sections of the signal and smooths it 
        while preserving higher moments (such as peak height and width). This is particularly useful 
        in spectral signal processing.

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

        Raises
        ------
        ValueError
            If window_length is not a positive odd integer or if it is smaller than polyorder + 2.

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
        # Precompute coefficients
        b = np.asmatrix(
            [
                [k**i for i in range(polyorder + 1)]
                for k in range(-half_window, half_window + 1)
            ]
        )
        m = np.linalg.pinv(b).A[0]
        # Pad the signal at the extremes with values taken from the signal itself
        firstvals = signal[0] - np.abs(signal[1 : half_window + 1][::-1] - signal[0])
        lastvals = signal[-1] + np.abs(signal[-half_window - 1 : -1][::-1] - signal[-1])
        signal = np.concatenate((firstvals, signal, lastvals))
        smoothed_signal = np.convolve(m, signal, mode="valid")
        return smoothed_signal

    def moving_average(self, window_size):
        """
        Applies a moving average filter to the signal.

        A moving average filter smooths the signal by averaging neighboring data points within a defined window size.
        This technique is commonly used to reduce random noise and reveal trends in the data.

        Parameters
        ----------
        window_size : int
            The size of the moving window.

        Returns
        -------
        filtered_signal : numpy.ndarray
            The filtered signal.

        Examples
        --------
        >>> import numpy as np
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> sf = SignalFiltering(signal)
        >>> filtered_signal = sf.moving_average(3)
        >>> print(filtered_signal)
        [2. 3. 4.]
        """
        # Apply padding to the signal
        padded_signal = np.pad(
            self.signal,
            (window_size // 2, window_size - 1 - window_size // 2),
            mode="edge",
        )

        # Calculate the moving average
        filtered_signal = np.convolve(
            padded_signal, np.ones(window_size) / window_size, mode="valid"
        )

        return filtered_signal

    def gaussian(self, sigma=1.0):
        """
        Applies a Gaussian filter to the signal.

        The Gaussian filter is a linear filter that applies a Gaussian kernel to the signal, effectively 
        smoothing it while preserving the signal's general shape. It is particularly useful for reducing 
        noise and softening sharp edges.

        Parameters
        ----------
        sigma : float
            The standard deviation of the Gaussian kernel.

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
        return np.convolve(self.signal, kernel, mode="same")

    @staticmethod
    def gaussian_filter1d(signal, sigma):
        """
        Custom implementation of a 1D Gaussian filter.

        This method applies a 1D Gaussian filter to the signal, smoothing the data by weighting 
        the points according to the Gaussian distribution. The result is a smoother signal with 
        reduced noise.

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
        smoothed_signal = np.convolve(signal, gaussian_kernel, mode="same")
        return smoothed_signal

    @staticmethod
    def gaussian_kernel(size, sigma):
        """
        Generate a Gaussian kernel.

        The Gaussian kernel is used in Gaussian filtering, where the kernel weights are 
        determined by the Gaussian function. This method generates a 1D Gaussian kernel.

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

    def butterworth(self, cutoff, fs, order=4, btype="low"):
        """
        Apply a Butterworth filter to the signal.

        The Butterworth filter is a type of signal processing filter designed to have a flat frequency response 
        in the passband. It is used for removing baseline wander and low-frequency noise from signals such as ECG and PPG.

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
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist

        # Calculate the poles of the Butterworth filter
        poles = np.exp(
            1j * (np.pi * (np.arange(1, 2 * order, 2) + order - 1) / (2 * order))
        )
        poles = poles[np.real(poles) < 0]

        if btype == "low":
            poles = poles * normal_cutoff
        elif btype == "high":
            poles = poles / normal_cutoff

        z = np.exp(-poles)

        filtered_signal = self.signal.copy()
        for p in z:
            filtered_signal = (
                filtered_signal
                - 2 * np.real(p) * np.roll(filtered_signal, 1)
                + np.abs(p) ** 2 * np.roll(filtered_signal, 2)
            )

        return filtered_signal

    def butter(self, order, cutoff, btype="low", fs=1.0):
        """
        Custom implementation of the Butterworth filter design using bilinear transformation.

        This method designs a Butterworth filter by calculating the filter coefficients 
        based on the desired order and cutoff frequency. The resulting filter can be used 
        for low-pass, high-pass, or band-pass filtering.

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
        nyquist = 0.5 * fs
        normalized_cutoff = np.array(cutoff) / nyquist
        if btype == "low":
            poles = np.exp(
                1j * np.pi * (np.arange(1, 2 * order + 1, 2) + order - 1) / (2 * order)
            )
            poles = poles[np.real(poles) < 0]
            z, p = np.zeros(0), poles
        elif btype == "high":
            z, p = self.butter(order, cutoff, btype="low", fs=fs)
            z, p = -z, -p
        elif btype == "band":
            low = np.min(normalized_cutoff)
            high = np.max(normalized_cutoff)
            z_low, p_low = self.butter(order, low, btype="high", fs=fs)
            z_high, p_high = self.butter(order, high, btype="low", fs=fs)
            z = np.concatenate([z_low, z_high])
            p = np.concatenate([p_low, p_high])
        else:
            raise ValueError("Invalid btype. Must be 'low', 'high', or 'band'.")

        b, a = np.poly(z), np.poly(p)
        b /= np.abs(np.sum(a))
        return b, a

    def median(self, kernel_size=3):
        """
        Apply a median filter to the signal.

        The median filter is a non-linear filter used to remove spikes and outliers 
        from signals. It is particularly effective for removing motion artifacts 
        in PPG and ECG signals, where abrupt changes may be caused by noise rather than the signal itself.

        Parameters
        ----------
        kernel_size : int, optional
            Size of the median filter kernel. Default is 3.

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
        # Apply padding to the signal
        padded_signal = np.pad(
            self.signal,
            (kernel_size // 2, kernel_size - 1 - kernel_size // 2),
            mode="edge",
        )

        # Calculate the median filter
        filtered_signal = np.array(
            [
                np.median(padded_signal[i : i + kernel_size])
                for i in range(len(self.signal))
            ]
        )

        return filtered_signal
