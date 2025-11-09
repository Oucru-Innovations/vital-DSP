"""
Signal Transforms Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations
- Signal transformation methods

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.transforms.hilbert_transform import HilbertTransform
    >>> signal = np.random.randn(1000)
    >>> processor = HilbertTransform(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np


class HilbertTransform:
    """
    A class to perform the Hilbert Transform, which is used to generate analytic signals.

    The Hilbert Transform is a key tool in signal processing, particularly for generating the analytic signal from a real-valued signal. The analytic signal is complex, with the original signal as the real part and the Hilbert transform as the imaginary part. This is particularly useful in applications like QRS detection in ECG signals, where phase and amplitude information are crucial.

    Methods
    -------
    compute_hilbert : method
        Computes the Hilbert Transform of the signal to obtain the analytic signal.
    """

    def __init__(self, signal):
        """
        Initialize the HilbertTransform class with the input signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The real-valued input signal to be transformed.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100))
        >>> ht = HilbertTransform(signal)
        >>> print(ht.signal)
        """
        self.signal = signal

    def compute_hilbert(self):
        """
        Compute the Hilbert Transform to obtain the analytic signal.

        The Hilbert Transform is applied in the frequency domain by first taking the Fourier transform of the input signal, modifying the Fourier coefficients to zero out the negative frequencies, and then applying the inverse Fourier transform. This process effectively shifts the signal in such a way that the imaginary part represents the phase information, while the real part remains the original signal.

        Returns
        -------
        numpy.ndarray
            The analytic signal with both real and imaginary components, where the real part is the original signal and the imaginary part is the Hilbert transform.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> ht = HilbertTransform(signal)
        >>> analytic_signal = ht.compute_hilbert()
        >>> print(analytic_signal)

        Notes
        -----
        The analytic signal is often used in applications where the instantaneous amplitude and phase of the signal are required, such as in the detection of QRS complexes in ECG signals, modulation, and demodulation in communications, and envelope detection in various signal processing tasks.
        """
        N = len(self.signal)
        H = np.zeros(N)

        # Construct the Hilbert Transform multiplier in the frequency domain
        if N % 2 == 0:
            H[0] = 1
            H[N // 2] = 1
            H[1 : N // 2] = 2
        else:
            H[0] = 1
            H[1 : (N + 1) // 2] = 2

        # Apply the Hilbert Transform using the FFT
        hilbert_signal = np.fft.ifft(np.fft.fft(self.signal) * H)
        return hilbert_signal

    def envelope(self):
        """
        Compute the envelope of the signal using the Hilbert Transform.

        The envelope is the magnitude of the analytic signal and represents the instantaneous amplitude of the signal.
        This is particularly useful in applications such as PPG signal analysis, where the envelope can be used to
        assess pulse amplitude variations.

        Returns
        -------
        numpy.ndarray
            The envelope of the input signal.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> ht = HilbertTransform(signal)
        >>> envelope = ht.envelope()
        >>> print(envelope)
        """
        analytic_signal = self.compute_hilbert()
        return np.abs(analytic_signal)

    def instantaneous_phase(self):
        """
        Compute the instantaneous phase of the signal using the Hilbert Transform.

        The instantaneous phase is the phase angle of the analytic signal and is useful in applications such as
        ECG analysis, where phase information can help in detecting the QRS complex or other waveform characteristics.

        Returns
        -------
        numpy.ndarray
            The instantaneous phase of the input signal.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> ht = HilbertTransform(signal)
        >>> phase = ht.instantaneous_phase()
        >>> print(phase)
        """
        analytic_signal = self.compute_hilbert()
        return np.angle(analytic_signal)
