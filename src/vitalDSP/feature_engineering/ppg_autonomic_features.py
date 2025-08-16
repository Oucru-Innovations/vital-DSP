import numpy as np

# from scipy.signal import find_peaks
from vitalDSP.utils.peak_detection import PeakDetection


class PPGAutonomicFeatures:
    """
    A class to compute respiratory and autonomic features from PPG signals.

    Features included:
    - Respiratory Rate Variability (RRV)
    - Respiratory Sinus Arrhythmia (RSA)
    - Autonomic Nervous System Balance (Fractal Dimension, DFA)

    Example usage:
    ```
    import numpy as np
    from PPGRespiratoryAutonomicFeatures import PPGRespiratoryAutonomicFeatures

    # Simulated PPG signal data
    ppg_signal = np.random.rand(1000)  # Replace with actual PPG signal
    fs = 100  # Sampling frequency in Hz

    features = PPGRespiratoryAutonomicFeatures(ppg_signal, fs)

    rrv = features.compute_rrv()
    rsa = features.compute_rsa()
    fractal = features.compute_fractal_dimension()
    dfa_value = features.compute_dfa()

    print(f"RRV: {rrv}, RSA: {rsa}, Fractal Dimension: {fractal}, DFA: {dfa_value}")
    ```
    """

    def __init__(self, ppg_signal, sampling_frequency):
        """
        Initializes the class with PPG signal and sampling frequency.

        Args:
            ppg_signal (np.array): Array of PPG signal values.
            sampling_frequency (int): Sampling frequency in Hz.
        """
        if not isinstance(ppg_signal, np.ndarray):
            raise TypeError("Input signal must be a numpy array")
        if len(ppg_signal) < 2:
            raise ValueError("PPG signal is too short to compute features")
        if np.isnan(ppg_signal).any() or np.isinf(ppg_signal).any():
            raise ValueError("PPG signal contains invalid values")

        self.ppg_signal = ppg_signal
        self.fs = sampling_frequency

    def compute_rrv(self):
        """
        Computes Respiratory Rate Variability (RRV) from the PPG signal.

        Returns:
            float: Respiratory rate variability value.
        """
        # peaks, _ = find_peaks(self.ppg_signal, distance=self.fs/2)
        peak_detector = PeakDetection(self.ppg_signal, method="ppg_first_derivative")
        peaks = peak_detector.detect_peaks()  # Indices of peaks in the PPG signal
        if len(peaks) < 2:
            raise ValueError("No peaks detected in PPG signal")

        rr_intervals = np.diff(peaks) / self.fs
        rrv = np.std(rr_intervals)
        return rrv

    def compute_rsa(self):
        """
        Computes Respiratory Sinus Arrhythmia (RSA) from the PPG signal.

        RSA is measured by the difference in heart rate during inhalation and exhalation.

        Returns:
            float: RSA value (average difference in peak intervals).
        """
        # peaks, _ = find_peaks(self.ppg_signal, distance=self.fs/2)
        peak_detector = PeakDetection(self.ppg_signal, method="ppg_first_derivative")
        peaks = peak_detector.detect_peaks()  # Indices of peaks in the PPG signal
        if len(peaks) < 2:
            raise ValueError("No peaks detected in PPG signal")

        intervals = np.diff(peaks) / self.fs
        inhalation_intervals = intervals[::2]
        exhalation_intervals = intervals[1::2]

        if len(inhalation_intervals) == 0 or len(exhalation_intervals) == 0:
            return 0.0

        rsa = np.abs(np.mean(inhalation_intervals) - np.mean(exhalation_intervals))
        return rsa

    def compute_fractal_dimension(self, k_max=10):
        """
        Computes the fractal dimension of the PPG signal using the Higuchi method.

        Args:
            k_max (int): The maximum number of intervals to calculate (default is 10).

        Returns:
            float: Fractal dimension of the signal.
        """
        N = len(self.ppg_signal)
        if N < 10:
            raise ValueError("PPG signal is too short to compute fractal dimension")

        Lk = np.zeros(k_max)
        for k in range(1, k_max + 1):
            Lmk = []
            for m in range(k):
                Lm = (
                    np.sum(np.abs(np.diff(self.ppg_signal[m:N:k])))
                    * (N - 1)
                    / (((N - m) / k) * k)
                )
                Lmk.append(Lm)
            Lk[k - 1] = np.mean(Lmk)

        # Handle cases where log(Lk) might produce negative values or zero
        if np.any(Lk <= 0):
            raise ValueError(
                "Logarithmic values for fractal dimension cannot be computed due to non-positive values in Lk"
            )

        fractal_dim = np.polyfit(np.log(np.arange(1, k_max + 1)), np.log(Lk), 1)[0]
        return (
            float(fractal_dim) if fractal_dim > 0 else 0.001
        )  # Ensure valid float value

    def compute_dfa(self, window_size=10):
        """
        Computes the Detrended Fluctuation Analysis (DFA) of the PPG signal.

        DFA is useful for measuring the complexity of time-series data.

        Args:
            window_size (int): The window size for detrending (default is 10).

        Returns:
            float: DFA value of the PPG signal.
        """
        N = len(self.ppg_signal)
        if N < window_size:
            raise ValueError("PPG signal is too short to compute DFA")

        integrated = np.cumsum(self.ppg_signal - np.mean(self.ppg_signal))
        F_n = np.zeros(N // window_size)

        for i in range(0, len(F_n)):
            start = i * window_size
            end = (i + 1) * window_size
            x_range = np.arange(start, end)
            poly_coeff = np.polyfit(x_range, integrated[start:end], 1)
            trend = np.polyval(poly_coeff, x_range)
            F_n[i] = np.sqrt(np.mean((integrated[start:end] - trend) ** 2))

        # Handle potential negative or zero values in F_n before applying log
        if np.any(F_n <= 0):
            raise ValueError(
                "Logarithmic values for DFA cannot be computed due to non-positive values in F_n"
            )

        dfa_value = np.polyfit(np.log(np.arange(1, len(F_n) + 1)), np.log(F_n), 1)[0]
        return float(dfa_value) if dfa_value > 0 else 0.001  # Ensure valid float value

# import pandas as pd
# import os
# if __name__ == "__main__":
#     ppg_signal = np.random.rand(1000)
#     fname = '20190109T151032.026+0700_1050000_1080000.csv'
#     PATH = 'D:\Workspace\Data\\24EIa\output\sample'
    
#     ppg_signal = pd.read_csv(os.path.join(PATH, fname))['PLETH'].values
#     fs = 100
#     features = PPGAutonomicFeatures(ppg_signal, fs)
#     rrv = features.compute_rrv()
#     rsa = features.compute_rsa()
#     fractal = features.compute_fractal_dimension()
#     dfa_value = features.compute_dfa()
#     print(f"RRV: {rrv}, RSA: {rsa}, Fractal Dimension: {fractal}, DFA: {dfa_value}")