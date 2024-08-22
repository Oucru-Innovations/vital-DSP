import numpy as np

class EMD:
    """
    Empirical Mode Decomposition (EMD) for decomposing non-linear and non-stationary signals into IMFs.

    Methods:
    - emd: Performs the EMD on the input signal and returns the IMFs.

    Example Usage:
    --------------
    signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
    emd = EMD(signal)
    imfs = emd.emd()
    print("IMFs:", imfs)
    """

    def __init__(self, signal):
        self.signal = signal

    def emd(self, max_imfs=None, stop_criterion=0.05):
        """
        Perform Empirical Mode Decomposition (EMD) on the input signal.

        Parameters:
        max_imfs (int or None): Maximum number of IMFs to extract. If None, extract all possible IMFs.
        stop_criterion (float): The stopping criterion for the sifting process (e.g., 0.05).

        Returns:
        list: A list of IMFs extracted from the signal.
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
                sd = np.sum((h - h_new) ** 2) / np.sum(h ** 2)
                h = h_new

            imfs.append(h)
            signal = signal - h

            if max_imfs is not None and len(imfs) >= max_imfs:
                break
            if np.all(signal < stop_criterion):
                break

        return imfs

    def _find_peaks(self, signal):
        peaks = np.where((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:]))[0] + 1
        return peaks

    def _interpolate(self, x, y):
        interpolated = np.interp(np.arange(len(self.signal)), x, y)
        return interpolated
