import numpy as np


class EMD:
    """
    Empirical Mode Decomposition (EMD) for decomposing non-linear and non-stationary signals into Intrinsic Mode Functions (IMFs).

    EMD is particularly useful for analyzing signals that are non-linear and non-stationary, where traditional methods like Fourier Transform may not be effective.

    Methods
    -------
    emd : method
        Performs the EMD on the input signal and returns the IMFs.

    Example Usage
    -------------
    >>> signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
    >>> emd = EMD(signal)
    >>> imfs = emd.emd()
    >>> print("IMFs:", imfs)
    """

    def __init__(self, signal):
        """
        Initialize the EMD class with the signal to be decomposed.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal that will be decomposed into IMFs.

        Notes
        -----
        The signal should be a one-dimensional numpy array representing the time series data.
        """
        if not isinstance(signal, np.ndarray):
            raise ValueError("The input signal must be a numpy.ndarray.")
        if signal.ndim != 1:
            raise ValueError("The input signal must be one-dimensional.")
        self.signal = signal

    def emd(self, max_imfs=None, stop_criterion=0.05):
        """
        Perform Empirical Mode Decomposition (EMD) on the input signal.

        Parameters
        ----------
        max_imfs : int or None, optional
            Maximum number of IMFs to extract. If None, extract all possible IMFs (default is None).
        stop_criterion : float, optional
            The stopping criterion for the sifting process, which controls how close the IMF is to an ideal IMF (default is 0.05).

        Returns
        -------
        imfs : list of numpy.ndarray
            A list of IMFs (Intrinsic Mode Functions) extracted from the signal.

        Notes
        -----
        Each IMF represents a simple oscillatory mode embedded in the signal. The sum of all IMFs plus the final residual will reconstruct the original signal.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
        >>> emd = EMD(signal)
        >>> imfs = emd.emd()
        >>> print("IMFs:", imfs)
        """
        signal = self.signal
        imfs = []

        while True:
            h = signal
            sd = np.inf

            while sd > stop_criterion:
                peaks = self._find_peaks(h)
                valleys = self._find_peaks(-h)
                upper_env = self._interpolate(peaks, h[peaks])
                lower_env = self._interpolate(valleys, h[valleys])
                mean_env = (upper_env + lower_env) / 2
                h_new = h - mean_env
                sd = np.sum((h - h_new) ** 2) / np.sum(h**2)
                h = h_new

            imfs.append(h)
            signal = signal - h

            if max_imfs is not None and len(imfs) >= max_imfs:
                break
            if np.all(np.abs(signal) < stop_criterion):
                break

        return imfs

    def _find_peaks(self, signal):
        """
        Identify the peaks in the signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal in which peaks are to be found.

        Returns
        -------
        peaks : numpy.ndarray
            Indices of the peaks in the signal.

        Notes
        -----
        This method uses a simple comparison to identify peaks where a point is higher than its neighboring points.
        """
        peaks = (
            np.where((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:]))[0] + 1
        )
        return peaks

    def _interpolate(self, x, y):
        """
        Interpolate the given points to create an envelope.

        Parameters
        ----------
        x : numpy.ndarray
            The x-coordinates (indices) of the points to be interpolated.
        y : numpy.ndarray
            The y-coordinates (values) of the points to be interpolated.

        Returns
        -------
        interpolated : numpy.ndarray
            The interpolated envelope of the signal.

        Notes
        -----
        Linear interpolation is used to create the envelope from the peaks or valleys.
        """
        interpolated = np.interp(np.arange(len(self.signal)), x, y)
        return interpolated
