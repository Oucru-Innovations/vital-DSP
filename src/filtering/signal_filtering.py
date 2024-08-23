import numpy as np

class SignalFiltering:
    """
    A class for applying various filtering techniques to signals.

    Methods:
    - moving_average: Applies a moving average filter.
    - gaussian: Applies a Gaussian filter.
    - butterworth: Applies a Butterworth filter.
    - median: Applies a median filter.
    """

    def __init__(self, signal):
        """
        Initialize the SignalFiltering class with the signal.

        Parameters:
        signal (numpy.ndarray): The input signal to be filtered.
        """
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)
        self.signal = signal

    @staticmethod
    def savgol_filter(signal, window_length, polyorder):
        """
        Custom implementation of the Savitzky-Golay filter.

        Parameters:
        - signal (numpy.ndarray): The input signal.
        - window_length (int): The length of the filter window (must be odd).
        - polyorder (int): The order of the polynomial to fit.

        Returns:
        - smoothed_signal (numpy.ndarray): The smoothed signal.
        """
        if window_length % 2 == 0 or window_length < 1:
            raise ValueError("window_length must be a positive odd integer")
        if window_length < polyorder + 2:
            raise ValueError("window_length is too small for the polynomials order")

        half_window = (window_length - 1) // 2
        # Precompute coefficients
        b = np.asmatrix([[k**i for i in range(polyorder + 1)] for k in range(-half_window, half_window + 1)])
        m = np.linalg.pinv(b).A[0]
        # Pad the signal at the extremes with values taken from the signal itself
        firstvals = signal[0] - np.abs(signal[1:half_window+1][::-1] - signal[0])
        lastvals = signal[-1] + np.abs(signal[-half_window-1:-1][::-1] - signal[-1])
        signal = np.concatenate((firstvals, signal, lastvals))
        smoothed_signal = np.convolve(m, signal, mode='valid')
        return smoothed_signal
    
    def moving_average(self, window_size):
        """
        Apply a moving average filter to the signal.

        This filter smooths the signal by averaging values within a sliding window.

        Parameters:
        window_size (int): The size of the moving window.

        Returns:
        numpy.ndarray: The smoothed signal.

        Example:
        >>> signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> sf = SignalFiltering(signal)
        >>> filtered_signal = sf.moving_average(window_size=3)
        >>> print(filtered_signal)
        [2. 3. 4. 5. 6. 7. 8. 9.]
        """
        # Apply padding to the signal
        padded_signal = np.pad(self.signal, (window_size // 2, window_size - 1 - window_size // 2), mode='edge')
        
        # Calculate the moving average
        filtered_signal = np.convolve(padded_signal, np.ones(window_size) / window_size, mode='valid')
        
        return filtered_signal
    
    def gaussian(self, sigma=1.0):
        """
        Apply a Gaussian filter to the signal.

        This filter reduces noise by applying a Gaussian kernel, preserving important features while smoothing out noise.

        Parameters:
        sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
        numpy.ndarray: The smoothed signal.

        Example:
        >>> signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> sf = SignalFiltering(signal)
        >>> filtered_signal = sf.gaussian(sigma=1.0)
        >>> print(filtered_signal)
        [1.429 2.286 3.286 4.286 5.286 6.286 7.286 8.286 9.429]
        """
        size = int(6 * sigma + 1) if int(6 * sigma + 1) % 2 != 0 else int(6 * sigma + 2)
        kernel = self.gaussian_kernel(size, sigma)
        return np.convolve(self.signal, kernel, mode='same')
    
    @staticmethod
    def gaussian_filter1d(signal, sigma):
        """
        Custom implementation of a 1D Gaussian filter.

        Parameters:
        - signal (numpy.ndarray): The input signal.
        - sigma (float): The standard deviation of the Gaussian kernel.

        Returns:
        - smoothed_signal (numpy.ndarray): The smoothed signal.
        """
        radius = int(4 * sigma + 0.5)
        x = np.arange(-radius, radius + 1)
        gaussian_kernel = np.exp(-0.5 * (x / sigma) ** 2)
        gaussian_kernel /= gaussian_kernel.sum()
        smoothed_signal = np.convolve(signal, gaussian_kernel, mode='same')
        return smoothed_signal

    @staticmethod
    def gaussian_kernel(size, sigma):
        """
        Generate a Gaussian kernel.

        Parameters:
        size (int): The size of the kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian distribution.

        Returns:
        numpy.ndarray: The Gaussian kernel.

        Example:
        >>> kernel = SignalFiltering.gaussian_kernel(5, sigma=1.0)
        >>> print(kernel)
        """
        ax = np.arange(-(size // 2), (size // 2) + 1)
        kernel = np.exp(-0.5 * (ax / sigma) ** 2)
        kernel = kernel / np.sum(kernel)
        return kernel

    def butterworth(self, cutoff, fs, order=4, btype='low'):
        """
        Apply a Butterworth filter to the signal.

        This filter removes baseline wander and low-frequency noise from signals like ECG/PPG.

        Parameters:
        cutoff (float): Cutoff frequency of the filter.
        fs (float): Sampling frequency of the signal.
        order (int, optional): Order of the Butterworth filter. Default is 4.
        btype (str, optional): Type of filter - 'low' or 'high'. Default is 'low'.

        Returns:
        numpy.ndarray: The filtered signal.

        Example:
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
        poles = np.exp(1j * (np.pi * (np.arange(1, 2*order, 2) + order - 1) / (2*order)))
        poles = poles[np.real(poles) < 0]
        
        if btype == 'low':
            poles = poles * normal_cutoff
        elif btype == 'high':
            poles = poles / normal_cutoff

        z = np.exp(-poles)

        filtered_signal = self.signal.copy()
        for p in z:
            filtered_signal = filtered_signal - 2*np.real(p)*np.roll(filtered_signal, 1) + np.abs(p)**2*np.roll(filtered_signal, 2)

        return filtered_signal

    def butter(self, order, cutoff, btype='low', fs=1.0):
        """
        Custom implementation of the Butterworth filter design.
        Butterworth filter using bilinear transformation
        
        Parameters:
        - order (int): The order of the filter.
        - cutoff (float or list of float): The critical frequency or frequencies.
        - btype (str): The type of filter ('low', 'high', 'band').
        - fs (float): The sampling frequency.

        Returns:
        - b, a (tuple): Numerator (b) and denominator (a) polynomials of the IIR filter.
        """
        nyquist = 0.5 * fs
        normalized_cutoff = np.array(cutoff) / nyquist
        if btype == 'low':
            poles = np.exp(1j * np.pi * (np.arange(1, 2 * order + 1, 2) + order - 1) / (2 * order))
            poles = poles[np.real(poles) < 0]
            z, p = np.zeros(0), poles
        elif btype == 'high':
            z, p = self.butter(order, cutoff, btype='low', fs=fs)
            z, p = -z, -p
        elif btype == 'band':
            low = np.min(normalized_cutoff)
            high = np.max(normalized_cutoff)
            z_low, p_low = self.butter(order, low, btype='high', fs=fs)
            z_high, p_high = self.butter(order, high, btype='low', fs=fs)
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

        This non-linear filter is effective at removing spikes and outliers, such as motion artifacts in PPG signals.

        Parameters:
        kernel_size (int, optional): Size of the median filter kernel. Default is 3.

        Returns:
        numpy.ndarray: The filtered signal.

        Example:
        >>> signal = np.array([1, 2, 3, 100, 5, 6, 7, 8, 9, 10])  # Note the spike at value 100
        >>> sf = SignalFiltering(signal)
        >>> filtered_signal = sf.median(kernel_size=3)
        >>> print(filtered_signal)
        [1 2 3 5 5 6 7 8 9 9]
        """
        # Apply padding to the signal
        padded_signal = np.pad(self.signal, (kernel_size // 2, kernel_size - 1 - kernel_size // 2), mode='edge')
        
        # Calculate the median filter
        filtered_signal = np.array([
            np.median(padded_signal[i:i + kernel_size]) 
            for i in range(len(self.signal))
        ])
        
        return filtered_signal
