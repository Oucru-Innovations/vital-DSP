import numpy as np
from ..utils.signal_processing.peak_detection import PeakDetection


class ECGPPGSynchronization:
    """
    A class for analyzing synchronization between ECG and PPG signals to compute combined features
    like Pulse Transit Time (PTT), Pulse Arrival Time (PAT), Electromechanical Delay (EMD),
    Respiratory Sinus Arrhythmia (RSA), and Heart Rate-Pulse Rate synchronization.

    Attributes:
        ecg_signal (np.array): The ECG signal.
        ppg_signal (np.array): The PPG signal.
        ecg_fs (int): The sampling frequency of the ECG signal in Hz.
        ppg_fs (int): The sampling frequency of the PPG signal in Hz.
    """

    def __init__(self, ecg_signal, ppg_signal, ecg_fs, ppg_fs):
        """
        Initializes the ECGPPGSynchronization object.

        Args:
            ecg_signal (np.array): The ECG signal.
            ppg_signal (np.array): The PPG signal.
            ecg_fs (int): The sampling frequency of the ECG signal in Hz.
            ppg_fs (int): The sampling frequency of the PPG signal in Hz.

        Raises:
            ValueError: If the input signals are not valid or if sampling frequencies are not positive.
        """
        if not isinstance(ecg_signal, np.ndarray) or not isinstance(
            ppg_signal, np.ndarray
        ):
            raise TypeError("Both ECG and PPG signals must be numpy arrays.")
        if len(ecg_signal) == 0 or len(ppg_signal) == 0:
            raise ValueError("Both ECG and PPG signals must have non-zero length.")
        if (
            np.isnan(ecg_signal).any()
            or np.isnan(ppg_signal).any()
            or np.isinf(ecg_signal).any()
            or np.isinf(ppg_signal).any()
        ):
            raise ValueError("ECG or PPG signal contains invalid values.")
        if ecg_fs <= 0 or ppg_fs <= 0:
            raise ValueError("Sampling frequencies must be positive integers.")

        self.ecg_signal = ecg_signal
        self.ppg_signal = ppg_signal
        self.ecg_fs = ecg_fs
        self.ppg_fs = ppg_fs

    def detect_r_peaks(self):
        """
        Detects R-peaks in the ECG signal using a simple threshold-based method.

        Returns:
            np.array: Indices of detected R-peaks in the ECG signal.

        Raises:
            ValueError: If no R-peaks are detected in the signal.

        Example:
            >>> ecg_signal = np.sin(np.linspace(0, 10 * np.pi, 1000))  # Simulated ECG signal
            >>> sync_analyzer = ECGPPGSynchronization(ecg_signal, np.array([]), 1000, 100)
            >>> r_peaks = sync_analyzer.detect_r_peaks()
            >>> print(f"R-peaks: {r_peaks}")
        """
        try:
            if len(self.ecg_signal) < self.ecg_fs * 2:
                raise ValueError("ECG signal is too short to detect peaks.")
            detector = PeakDetection(self.ecg_signal, method="ecg_r_peak")
            r_peaks = detector.detect_peaks()
            if len(r_peaks) == 0:
                raise ValueError("No R-peaks detected in the ECG signal.")
            return r_peaks
        except ValueError as e:
            raise ValueError(str(e))  # Return a ValueError
        except Exception as e:
            raise RuntimeError(f"Error detecting R-peaks: {str(e)}")

    def detect_systolic_peaks(self):
        """
        Detects systolic peaks in the PPG signal.

        Returns:
            np.array: Indices of detected systolic peaks in the PPG signal.

        Raises:
            ValueError: If no systolic peaks are detected in the signal.

        Example:
            >>> ppg_signal = np.sin(np.linspace(0, 10 * np.pi, 1000))  # Simulated PPG signal
            >>> sync_analyzer = ECGPPGSynchronization(np.array([]), ppg_signal, 1000, 100)
            >>> systolic_peaks = sync_analyzer.detect_systolic_peaks()
            >>> print(f"Systolic peaks: {systolic_peaks}")
        """
        try:
            if len(self.ppg_signal) < self.ppg_fs * 2:
                raise ValueError("PPG signal is too short to detect peaks.")
            detector = PeakDetection(self.ppg_signal, method="ppg_first_derivative")
            systolic_peaks = detector.detect_peaks()
            if len(systolic_peaks) == 0:
                raise ValueError("No systolic peaks detected in the PPG signal.")
            return systolic_peaks
        except ValueError as e:
            raise ValueError(str(e))  # Return a ValueError
        except Exception as e:
            raise RuntimeError(f"Error detecting systolic peaks: {str(e)}")

    def compute_ptt(self, r_peaks=None, systolic_peaks=None):
        """
        Computes Pulse Transit Time (PTT) between ECG R-peaks and PPG systolic peaks.

        Args:
            r_peaks (np.array): Indices of R-peaks in the ECG signal.
            systolic_peaks (np.array): Indices of systolic peaks in the PPG signal.

        Returns:
            float: The mean PTT value in milliseconds.

        Raises:
            ValueError: If no valid PTT values can be calculated.

        Example:
            >>> ptt = sync_analyzer.compute_ptt(r_peaks, systolic_peaks)
            >>> print(f"Pulse Transit Time: {ptt} ms")
        """
        try:
            if r_peaks is None:
                r_peaks = self.detect_r_peaks()
            if systolic_peaks is None:
                systolic_peaks = self.detect_systolic_peaks()
            ptt_values = []
            for r_peak in r_peaks:
                valid_peaks = systolic_peaks[systolic_peaks > r_peak]
                if len(valid_peaks) > 0:
                    nearest_systolic_peak = valid_peaks.min()
                    ptt = (
                        (nearest_systolic_peak - r_peak) / self.ppg_fs * 1000
                    )  # Convert to ms
                    ptt_values.append(ptt)
            if len(ptt_values) == 0:
                raise ValueError("No valid PTT values could be calculated.")
            return np.mean(ptt_values)
        except Exception as e:
            raise RuntimeError(f"Error computing Pulse Transit Time: {str(e)}")

    def compute_pat(self, r_peaks=None, systolic_peaks=None):
        """
        Computes Pulse Arrival Time (PAT) between ECG R-peaks and PPG systolic peaks.

        Args:
            r_peaks (np.array): Indices of R-peaks in the ECG signal.
            systolic_peaks (np.array): Indices of systolic peaks in the PPG signal.

        Returns:
            float: The mean PAT value in milliseconds.

        Raises:
            ValueError: If no valid PAT values can be calculated.

        Example:
            >>> pat = sync_analyzer.compute_pat(r_peaks, systolic_peaks)
            >>> print(f"Pulse Arrival Time: {pat} ms")
        """
        try:
            if r_peaks is None:
                r_peaks = self.detect_r_peaks()
            if systolic_peaks is None:
                systolic_peaks = self.detect_systolic_peaks()
            pat_values = []
            for r_peak in r_peaks:
                valid_peaks = systolic_peaks[systolic_peaks > r_peak]
                if len(valid_peaks) > 0:
                    nearest_systolic_peak = valid_peaks.min()
                    pat = (
                        (nearest_systolic_peak - r_peak) / self.ppg_fs * 1000
                    )  # Convert to ms
                    pat_values.append(pat)
            if len(pat_values) == 0:
                raise ValueError("No valid PAT values could be calculated.")
            return np.mean(pat_values)
        except Exception as e:
            raise RuntimeError(f"Error computing Pulse Arrival Time: {str(e)}")

    def compute_emd(self, r_peaks=None, systolic_peaks=None):
        """
        Computes Electromechanical Delay (EMD) between ECG R-peaks and PPG systolic peaks.

        Args:
            r_peaks (np.array): Indices of R-peaks in the ECG signal.
            systolic_peaks (np.array): Indices of systolic peaks in the PPG signal.

        Returns:
            float: The mean EMD value in milliseconds.

        Raises:
            ValueError: If no valid EMD values can be calculated.

        Example:
            >>> emd = sync_analyzer.compute_emd(r_peaks, systolic_peaks)
            >>> print(f"Electromechanical Delay: {emd} ms")
        """
        try:
            if r_peaks is None:
                r_peaks = self.detect_r_peaks()
            if systolic_peaks is None:
                systolic_peaks = self.detect_systolic_peaks()
            emd_values = []
            for r_peak in r_peaks:
                valid_peaks = systolic_peaks[systolic_peaks > r_peak]
                if len(valid_peaks) > 0:
                    nearest_systolic_peak = valid_peaks.min()
                    emd = (
                        (nearest_systolic_peak - r_peak) / self.ppg_fs * 1000
                    )  # Convert to ms
                    emd_values.append(emd)
            if len(emd_values) == 0:
                raise ValueError("No valid EMD values could be calculated.")
            return np.mean(emd_values)
        except Exception as e:
            raise RuntimeError(f"Error computing Electromechanical Delay: {str(e)}")

    def compute_hr_pr_sync(self, r_peaks=None, systolic_peaks=None):
        """
        Computes the synchronization between heart rate (HR) from ECG and pulse rate (PR) from PPG.

        Args:
            r_peaks (np.array): Indices of R-peaks in the ECG signal.
            systolic_peaks (np.array): Indices of systolic peaks in the PPG signal.

        Returns:
            float: The HR-PR synchronization ratio (should be close to 1 if well-synchronized).

        Raises:
            ValueError: If no valid HR or PR intervals can be computed.

        Example:
            >>> hr_pr_sync = sync_analyzer.compute_hr_pr_sync(r_peaks, systolic_peaks)
            >>> print(f"HR-PR Synchronization Ratio: {hr_pr_sync}")
        """
        try:
            if r_peaks is None:
                r_peaks = self.detect_r_peaks()
            if systolic_peaks is None:
                systolic_peaks = self.detect_systolic_peaks()
            if len(r_peaks) < 2 or len(systolic_peaks) < 2:
                raise ValueError(
                    "Not enough peaks to calculate HR or PR synchronization."
                )

            # Compute heart rate (HR) from ECG R-peaks (in beats per minute)
            hr_intervals = (
                np.diff(r_peaks) / self.ecg_fs * 60
            )  # Convert intervals to BPM
            if len(hr_intervals) == 0:
                raise ValueError("No valid HR intervals could be calculated.")

            # Compute pulse rate (PR) from PPG systolic peaks (in beats per minute)
            pr_intervals = (
                np.diff(systolic_peaks) / self.ppg_fs * 60
            )  # Convert intervals to BPM
            if len(pr_intervals) == 0:
                raise ValueError("No valid PR intervals could be calculated.")

            # Calculate the ratio of HR to PR
            hr_pr_sync_ratio = np.mean(hr_intervals) / np.mean(pr_intervals)
            return hr_pr_sync_ratio
        except Exception as e:
            raise RuntimeError(f"Error computing HR-PR synchronization: {str(e)}")

    def compute_rsa(self, r_peaks=None, systolic_peaks=None):
        """
        Computes Respiratory Sinus Arrhythmia (RSA) by analyzing the synchronized variations
        in ECG and PPG signals during respiration.

        Args:
            r_peaks (np.array): Indices of R-peaks in the ECG signal.
            systolic_peaks (np.array): Indices of systolic peaks in the PPG signal.

        Returns:
            float: The mean RSA value (Respiratory Sinus Arrhythmia).

        Raises:
            ValueError: If no valid RSA values can be computed.

        Example:
            >>> rsa = sync_analyzer.compute_rsa(r_peaks, systolic_peaks)
            >>> print(f"Respiratory Sinus Arrhythmia: {rsa}")
        """
        try:
            if r_peaks is None:
                r_peaks = self.detect_r_peaks()
            if systolic_peaks is None:
                systolic_peaks = self.detect_systolic_peaks()
            rsa_values = []
            for r_peak in r_peaks:
                valid_peaks = systolic_peaks[systolic_peaks > r_peak]
                if len(valid_peaks) > 0:
                    nearest_systolic_peak = valid_peaks.min()
                    rsa = (
                        nearest_systolic_peak - r_peak
                    ) / self.ppg_fs  # Convert to seconds
                    rsa_values.append(rsa)
            if len(rsa_values) == 0:
                raise ValueError("No valid RSA values could be computed.")
            return np.mean(rsa_values)
        except Exception as e:
            raise RuntimeError(
                f"Error computing Respiratory Sinus Arrhythmia: {str(e)}"
            )
