import numpy as np
import warnings


class EMD:
    """
    Empirical Mode Decomposition (EMD) for decomposing non-linear and non-stationary signals into Intrinsic Mode Functions (IMFs).

    EMD is particularly useful for analyzing signals that are non-linear and non-stationary, where traditional methods like Fourier Transform may not be effective.

    Methods
    -------
    emd : method
        Performs the EMD on the input signal and returns the IMFs.

    Examples
    --------
    >>> import numpy as np
    >>> from vitalDSP.advanced_computation.emd import EMD
    >>>
    >>> # Example 1: Basic EMD decomposition
    >>> signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
    >>> emd = EMD(signal)
    >>> imfs = emd.emd()
    >>> print(f"Number of IMFs: {len(imfs)}")
    >>>
    >>> # Example 2: EMD with limited number of IMFs
    >>> emd_limited = EMD(signal)
    >>> imfs_limited = emd_limited.emd(max_imfs=3)
    >>> print(f"Limited IMFs: {len(imfs_limited)}")
    >>>
    >>> # Example 3: EMD with custom stop criterion
    >>> emd_custom = EMD(signal)
    >>> imfs_custom = emd_custom.emd(stop_criterion=0.01)
    >>> print(f"Custom stop criterion IMFs: {len(imfs_custom)}")
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

    def emd(self, max_imfs=None, stop_criterion=0.05, max_sifting_iterations=20, max_decomposition_iterations=10):
        """
        Perform Empirical Mode Decomposition (EMD) on the input signal.

        Parameters
        ----------
        max_imfs : int or None, optional
            Maximum number of IMFs to extract. If None, extract all possible IMFs (default is None).
        stop_criterion : float, optional
            The stopping criterion for the sifting process, which controls how close the IMF is to an ideal IMF (default is 0.05).
        max_sifting_iterations : int, optional
            Maximum number of sifting iterations per IMF to prevent infinite loops (default is 20).
        max_decomposition_iterations : int, optional
            Maximum number of decomposition iterations to prevent excessive computation (default is 10).

        Returns
        -------
        imfs : list of numpy.ndarray
            A list of IMFs (Intrinsic Mode Functions) extracted from the signal.

        Notes
        -----
        Each IMF represents a simple oscillatory mode embedded in the signal. The sum of all IMFs plus the final residual will reconstruct the original signal.
        
        OPTIMIZATION: Added convergence limits to prevent infinite loops and improve reliability.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
        >>> emd = EMD(signal)
        >>> imfs = emd.emd()
        >>> print("IMFs:", imfs)
        """
        signal = self.signal
        imfs = []
        decomposition_iterations = 0

        # Handle edge case: max_imfs=0 should return empty list
        if max_imfs == 0:
            return imfs

        while True:
            decomposition_iterations += 1
            
            # Safety check: prevent excessive decomposition iterations
            if decomposition_iterations > max_decomposition_iterations:
                warnings.warn(f"EMD decomposition stopped after {max_decomposition_iterations} iterations. "
                           f"Signal may not be suitable for EMD decomposition.")
                break
            
            h = signal
            sd = np.inf
            sifting_iterations = 0

            while sd > stop_criterion:
                sifting_iterations += 1
                
                # Safety check: prevent infinite sifting loops
                if sifting_iterations > max_sifting_iterations:
                    warnings.warn(f"Sifting process stopped after {max_sifting_iterations} iterations "
                               f"for IMF {len(imfs) + 1}. Convergence may be poor.")
                    break
                
                peaks = self._find_peaks(h)
                valleys = self._find_peaks(-h)

                if len(peaks) < 2 or len(valleys) < 2:
                    # Not enough peaks/valleys to perform interpolation; stop decomposition
                    warnings.warn(f"Insufficient extrema found for IMF {len(imfs) + 1}. "
                               f"Stopping decomposition.")
                    break

                try:
                    upper_env = self._interpolate(peaks, h[peaks])
                    lower_env = self._interpolate(valleys, h[valleys])
                except Exception as e:
                    warnings.warn(f"Interpolation failed for IMF {len(imfs) + 1}: {e}. "
                               f"Stopping decomposition.")
                    break
                
                mean_env = (upper_env + lower_env) / 2
                h_new = h - mean_env
                
                # Prevent division by zero
                signal_power = np.sum(h**2)
                if signal_power == 0:
                    warnings.warn(f"Signal power is zero for IMF {len(imfs) + 1}. "
                               f"Stopping decomposition.")
                    break
                
                sd = np.sum((h - h_new) ** 2) / signal_power
                h = h_new

            imfs.append(h)
            signal = signal - h

            if max_imfs is not None and len(imfs) >= max_imfs:
                break
            if np.all(np.abs(signal) < stop_criterion):
                break
                
            # Additional safety check: if signal becomes too small, stop
            if np.max(np.abs(signal)) < stop_criterion * 10:
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
        if len(x) < 2:  # Not enough points to interpolate
            return np.zeros(len(self.signal))
        interpolated = np.interp(np.arange(len(self.signal)), x, y)
        return interpolated
