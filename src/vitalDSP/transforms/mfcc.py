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

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.transforms.mfcc import Mfcc
    >>> signal = np.random.randn(1000)
    >>> processor = Mfcc(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""


import numpy as np


class MFCC:
    """
    A class to compute Mel-Frequency Cepstral Coefficients (MFCC) for audio signals.

    MFCCs are widely used in audio processing tasks, particularly in speech recognition. This class provides methods to preprocess an audio signal and compute its MFCCs, which represent the short-term power spectrum of a sound.

    Methods
    -------
    dct : method
        Computes the Discrete Cosine Transform (DCT) of the filter bank energies.
    compute_mfcc : method
        Computes the MFCCs of the input signal.
    """

    def __init__(self, signal, sample_rate=16000, num_filters=40, num_coefficients=13):
        """
        Initialize the MFCC class with the signal and relevant parameters.

        Parameters
        ----------
        signal : numpy.ndarray
            The input audio signal.
        sample_rate : int, optional
            The sample rate of the signal in Hertz (default is 16000 Hz).
        num_filters : int, optional
            The number of Mel filters to apply (default is 40).
        num_coefficients : int, optional
            The number of MFCC coefficients to extract (default is 13).

        Notes
        -----
        - The signal is expected to be a 1D numpy array representing the audio data.
        - The sample rate should match the rate at which the audio was originally recorded.
        """
        self.signal = signal
        self.sample_rate = sample_rate
        self.num_filters = num_filters
        self.num_coefficients = num_coefficients

    def dct(self, signal):
        """
        Compute the Discrete Cosine Transform (DCT) of the input signal.

        The DCT is applied to the filter bank energies to reduce the dimensionality and decorrelate the filter bank coefficients, producing the MFCCs.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal or filter banks matrix from which DCT is computed.

        Returns
        -------
        numpy.ndarray
            The DCT coefficients representing the MFCCs.

        Examples
        --------
        >>> signal = np.array([[1, 2, 3], [4, 5, 6]])
        >>> mfcc = MFCC(signal)
        >>> dct_result = mfcc.dct(signal)
        >>> print(dct_result)
        """
        n = signal.shape[1]
        result = np.zeros((signal.shape[0], self.num_coefficients))

        for k in range(1, self.num_coefficients + 1):
            result[:, k - 1] = np.sum(
                signal * np.cos(np.pi * (np.arange(n) + 0.5) * k / n), axis=1
            )

        return result * np.sqrt(2 / n)

    def compute_mfcc(self):
        """
        Compute the Mel-Frequency Cepstral Coefficients (MFCC) of the input signal.

        This method processes the input audio signal by applying pre-emphasis, framing, windowing, FFT, and filter banks, followed by the DCT to extract the MFCCs.

        Returns
        -------
        numpy.ndarray
            A 2D array where each row contains the MFCCs for a frame of the signal.

        Steps
        -----
        1. Pre-emphasis: Emphasizes higher frequencies in the signal.
        2. Framing: Divides the signal into overlapping frames.
        3. Windowing: Applies a Hamming window to each frame to reduce spectral leakage.
        4. FFT and Power Spectrum: Converts each frame to the frequency domain and computes the power spectrum.
        5. Mel Filter Banks: Applies a set of filters to the power spectrum to obtain Mel frequency bands.
        6. DCT: Computes the DCT of the log filter bank energies to obtain the MFCCs.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> mfcc = MFCC(signal)
        >>> mfcc_result = mfcc.compute_mfcc()
        >>> print(mfcc_result)
        """
        # Step 1: Pre-emphasis
        emphasized_signal = np.append(
            self.signal[0], self.signal[1:] - 0.97 * self.signal[:-1]
        )

        # Step 2: Framing
        frame_size = 0.025  # 25 ms
        frame_stride = 0.01  # 10 ms
        frame_length, frame_step = (
            frame_size * self.sample_rate,
            frame_stride * self.sample_rate,
        )
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = (
            int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1
        )

        # Padding the signal to ensure that all frames have equal number of samples
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized_signal, z)

        # Step 3: Windowing
        indices = (
            np.tile(np.arange(0, frame_length), (num_frames, 1))
            + np.tile(
                np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)
            ).T
        )
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        frames *= np.hamming(frame_length)

        # Step 4: FFT and Power Spectrum
        NFFT = 512  # FFT size
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        pow_frames = (1.0 / NFFT) * (mag_frames**2)

        # Step 5: Mel Filter Banks
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (self.sample_rate / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.num_filters + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin = np.floor((NFFT + 1) * hz_points / self.sample_rate)
        fbank = np.zeros((self.num_filters, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, self.num_filters + 1):
            f_m_minus = int(bin[m - 1])
            f_m = int(bin[m])
            f_m_plus = int(bin[m + 1])

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)  # Convert to dB

        # Step 6: Mel-frequency Cepstral Coefficients (MFCCs)
        mfcc = self.dct(filter_banks)
        return mfcc
