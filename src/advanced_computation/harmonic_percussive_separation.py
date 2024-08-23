import numpy as np


class HarmonicPercussiveSeparation:
    """
    Harmonic-Percussive Separation for separating vocal and noise components in respiratory analysis.

    Methods:
    - separate: Separates the harmonic and percussive components of the signal using median filtering.

    Example Usage:
    --------------
    signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
    hps = HarmonicPercussiveSeparation(signal)
    harmonic, percussive = hps.separate()
    print("Harmonic Component:", harmonic)
    print("Percussive Component:", percussive)
    """

    def __init__(self, signal):
        self.signal = signal

    def separate(self, kernel_size=31):
        """
        Separate the harmonic and percussive components of the signal using median filtering.

        Parameters:
        kernel_size (int): Size of the median filter kernel.

        Returns:
        tuple: Harmonic and percussive components of the signal.
        """
        harmonic = self._median_filter(self.signal, size=(kernel_size, 1))
        percussive = self._median_filter(self.signal, size=(1, kernel_size))
        return harmonic, percussive

    def _median_filter(self, signal, size):
        """
        Apply a median filter to the signal.

        Parameters:
        signal (numpy.ndarray): The input signal to be filtered.
        size (tuple): The size of the filter kernel.

        Returns:
        numpy.ndarray: The filtered signal.
        """
        filtered_signal = np.copy(signal)
        if size[0] > 1:  # Apply median filter across the rows
            for i in range(signal.shape[0]):
                filtered_signal[i, :] = self._apply_median(signal[i, :], size[0])
        if size[1] > 1:  # Apply median filter across the columns
            for i in range(signal.shape[1]):
                filtered_signal[:, i] = self._apply_median(signal[:, i], size[1])
        return filtered_signal

    def _apply_median(self, arr, kernel_size):
        """
        Apply a 1D median filter to an array.

        Parameters:
        arr (numpy.ndarray): The array to be filtered.
        kernel_size (int): The size of the median filter kernel.

        Returns:
        numpy.ndarray: The filtered array.
        """
        padded_arr = np.pad(arr, (kernel_size // 2, kernel_size // 2), mode="edge")
        filtered_arr = np.zeros_like(arr)

        for i in range(len(arr)):
            filtered_arr[i] = np.median(padded_arr[i : i + kernel_size])

        return filtered_arr
