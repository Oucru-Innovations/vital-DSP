"""
Signal type detection utilities for vitalDSP webapp.

This module provides centralized functions for detecting signal types based on
signal characteristics. These functions can be used across different callbacks
to automatically identify whether a signal is ECG, PPG, respiratory, or other types.

Future enhancement: These functions could be contributed to vitalDSP core library
for broader use in signal processing applications.
"""

import numpy as np
import logging
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


class SignalTypeDetector:
    """
    Automated signal type detection using statistical and frequency-domain features.

    This class provides methods to automatically classify biomedical signals into
    categories such as ECG, PPG, respiratory, or general signals based on their
    characteristics.

    Attributes
    ----------
    signal_data : numpy.ndarray
        The input signal to be classified
    sampling_freq : float
        Sampling frequency in Hz

    Methods
    -------
    detect_type()
        Automatically detect the signal type
    detect_respiratory_type()
        Detect respiratory signal type (ECG-derived, PPG-derived, or direct respiratory)
    get_signal_features()
        Extract statistical and frequency features for classification

    Examples
    --------
    >>> import numpy as np
    >>> from vitalDSP_webapp.utils.signal_type_detection import SignalTypeDetector
    >>>
    >>> # Create sample PPG signal
    >>> t = np.linspace(0, 10, 1000)
    >>> ppg_signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(1000)
    >>>
    >>> detector = SignalTypeDetector(ppg_signal, sampling_freq=100)
    >>> signal_type = detector.detect_type()
    >>> print(f"Detected signal type: {signal_type}")
    """

    def __init__(self, signal_data, sampling_freq=1000):
        """
        Initialize the signal type detector.

        Parameters
        ----------
        signal_data : numpy.ndarray
            The input signal to classify
        sampling_freq : float, optional
            Sampling frequency in Hz (default: 1000)
        """
        self.signal_data = np.array(signal_data)
        self.sampling_freq = sampling_freq
        self._features = None

    def get_signal_features(self):
        """
        Extract statistical and frequency-domain features from the signal.

        Returns
        -------
        dict
            Dictionary containing extracted features:
            - mean: Signal mean
            - std: Standard deviation
            - range: Signal range (max - min)
            - dominant_freq: Dominant frequency (Hz)
            - peak_power: Power at dominant frequency
            - freq_range: Frequency range of significant power
            - snr: Estimate of signal-to-noise ratio
        """
        if self._features is not None:
            return self._features

        try:
            # Statistical features
            mean_val = np.mean(self.signal_data)
            std_val = np.std(self.signal_data)
            range_val = np.max(self.signal_data) - np.min(self.signal_data)

            # Frequency domain features
            fft_result = np.abs(np.fft.rfft(self.signal_data))
            freqs = np.fft.rfftfreq(len(self.signal_data), 1 / self.sampling_freq)

            # Find dominant frequency (excluding DC component)
            fft_no_dc = fft_result.copy()
            fft_no_dc[0] = 0  # Remove DC
            peak_idx = np.argmax(fft_no_dc)
            dominant_freq = freqs[peak_idx]
            peak_power = fft_no_dc[peak_idx]

            # Calculate frequency range where power > 10% of peak
            threshold = 0.1 * peak_power
            significant_freqs = freqs[fft_no_dc > threshold]
            freq_range = (np.min(significant_freqs), np.max(significant_freqs)) if len(significant_freqs) > 0 else (0, 0)

            # Estimate SNR (signal power vs high-frequency noise)
            signal_band = fft_result[freqs < dominant_freq * 3]
            noise_band = fft_result[freqs > dominant_freq * 3]
            snr = np.mean(signal_band) / (np.mean(noise_band) + 1e-10) if len(noise_band) > 0 else 0

            self._features = {
                'mean': mean_val,
                'std': std_val,
                'range': range_val,
                'dominant_freq': dominant_freq,
                'peak_power': peak_power,
                'freq_range': freq_range,
                'snr': snr
            }

            return self._features

        except Exception as e:
            logger.error(f"Error extracting signal features: {e}")
            return {}

    def detect_type(self):
        """
        Automatically detect the type of biomedical signal.

        Uses frequency and statistical characteristics to classify signals into:
        - 'ecg': Electrocardiogram (heart rate ~0.8-2.0 Hz, sharp peaks)
        - 'ppg': Photoplethysmogram (heart rate ~0.8-2.0 Hz, smooth waveform)
        - 'respiratory': Respiratory signal (breathing rate ~0.1-0.5 Hz)
        - 'unknown': Cannot determine signal type

        Returns
        -------
        str
            Detected signal type ('ecg', 'ppg', 'respiratory', or 'unknown')

        Notes
        -----
        The detection is based on:
        1. Dominant frequency range
        2. Signal morphology (smoothness vs sharpness)
        3. Statistical properties

        For more advanced classification, consider using machine learning models
        that could be contributed to vitalDSP core library in the future.
        """
        try:
            features = self.get_signal_features()

            if not features:
                return 'unknown'

            dominant_freq = features['dominant_freq']
            range_val = features['range']
            std_val = features['std']

            # Cardiac signals: 0.8-2.0 Hz (48-120 BPM)
            # Respiratory signals: 0.1-0.5 Hz (6-30 BPM)

            if 0.8 <= dominant_freq <= 2.5:
                # Cardiac frequency range
                # Distinguish ECG from PPG based on morphology

                # Calculate signal sharpness (high frequency content)
                fft_result = np.abs(np.fft.rfft(self.signal_data))
                freqs = np.fft.rfftfreq(len(self.signal_data), 1 / self.sampling_freq)

                # ECG has more high-frequency content (sharp QRS complexes)
                high_freq_power = np.sum(fft_result[freqs > 10])
                total_power = np.sum(fft_result)
                hf_ratio = high_freq_power / (total_power + 1e-10)

                if hf_ratio > 0.1:  # ECG has sharper features
                    return 'ecg'
                else:  # PPG has smoother waveform
                    return 'ppg'

            elif 0.1 <= dominant_freq <= 0.8:
                # Respiratory frequency range
                return 'respiratory'
            else:
                return 'unknown'

        except Exception as e:
            logger.error(f"Error detecting signal type: {e}")
            return 'unknown'

    def detect_respiratory_type(self):
        """
        Detect the type of respiratory signal.

        Classifies respiratory signals into:
        - 'respiratory': Direct respiratory measurement (e.g., chest impedance)
        - 'ppg': PPG-derived respiratory signal (respiratory sinus arrhythmia)
        - 'ecg': ECG-derived respiratory signal (EDR, baseline wander)

        Returns
        -------
        str
            Respiratory signal type ('respiratory', 'ppg', or 'ecg')

        Notes
        -----
        This function is particularly useful for respiratory rate estimation
        where the signal source affects the optimal analysis method.
        """
        try:
            features = self.get_signal_features()

            if not features:
                return 'ppg'  # Default fallback

            std_val = features['std']
            range_val = features['range']
            dominant_freq = features['dominant_freq']

            # Respiratory signals typically have dominant frequency around 0.2-0.5 Hz (12-30 BPM)
            # PPG respiratory component is usually lower amplitude and smoother
            # ECG respiratory component is more complex (baseline wander + RSA)

            if 0.1 < dominant_freq < 0.8 and range_val < 2 * std_val:
                # Low amplitude, smooth -> PPG-derived
                return 'ppg'
            elif 0.1 < dominant_freq < 1.0 and range_val > 3 * std_val:
                # High amplitude, variable -> Direct respiratory
                return 'respiratory'
            elif dominant_freq > 0.5:
                # Higher frequency content -> ECG-derived
                return 'ecg'
            else:
                # Default to PPG for most ambiguous cases
                return 'ppg'

        except Exception as e:
            logger.warning(f"Respiratory signal type detection failed: {e}")
            return 'ppg'  # Default fallback


def detect_signal_type(signal_data, sampling_freq=1000):
    """
    Convenience function to detect general signal type.

    Parameters
    ----------
    signal_data : numpy.ndarray
        Input signal
    sampling_freq : float, optional
        Sampling frequency in Hz (default: 1000)

    Returns
    -------
    str
        Detected signal type ('ecg', 'ppg', 'respiratory', or 'unknown')

    Examples
    --------
    >>> import numpy as np
    >>> # ECG-like signal with sharp peaks
    >>> ecg = np.concatenate([np.zeros(50), [0, 0.5, 1, 0.5, 0], np.zeros(45)]) * 100
    >>> signal_type = detect_signal_type(np.tile(ecg, 10), sampling_freq=100)
    >>> print(signal_type)  # Should print 'ecg'
    """
    detector = SignalTypeDetector(signal_data, sampling_freq)
    return detector.detect_type()


def detect_respiratory_signal_type(signal_data, sampling_freq=1000):
    """
    Convenience function to detect respiratory signal type.

    Parameters
    ----------
    signal_data : numpy.ndarray
        Input respiratory signal
    sampling_freq : float, optional
        Sampling frequency in Hz (default: 1000)

    Returns
    -------
    str
        Respiratory signal type ('respiratory', 'ppg', or 'ecg')

    Examples
    --------
    >>> import numpy as np
    >>> # Direct respiratory signal
    >>> t = np.linspace(0, 30, 3000)
    >>> resp = np.sin(2 * np.pi * 0.25 * t) + 0.1 * np.random.randn(3000)
    >>> signal_type = detect_respiratory_signal_type(resp, sampling_freq=100)
    >>> print(signal_type)  # Should print 'respiratory' or 'ppg'
    """
    detector = SignalTypeDetector(signal_data, sampling_freq)
    return detector.detect_respiratory_type()
