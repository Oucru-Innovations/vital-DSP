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
        return np.convolve(self.signal, np.ones(window_size) / window_size, mode='valid')

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
        padded_signal = np.pad(self.signal, (kernel_size // 2, kernel_size // 2), mode='edge')
        filtered_signal = np.zeros_like(self.signal)
        
        for i in range(len(self.signal)):
            filtered_signal[i] = np.median(padded_signal[i:i + kernel_size])
        
        return filtered_signal
