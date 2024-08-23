import numpy as np


class MFCC:
    """
    A class to compute Mel-Frequency Cepstral Coefficients (MFCC) for audio signals.

    Methods:
    - compute_mfcc: Computes the MFCC of the signal.
    """

    def __init__(self, signal, sample_rate=16000, num_filters=40, num_coefficients=13):
        """
        Initialize the MFCC class with the signal.

        Parameters:
        signal (numpy.ndarray): The input audio signal.
        sample_rate (int): The sample rate of the signal.
        num_filters (int): The number of Mel filters to apply.
        num_coefficients (int): The number of MFCC coefficients to extract.
        """
        self.signal = signal
        self.sample_rate = sample_rate
        self.num_filters = num_filters
        self.num_coefficients = num_coefficients

    def dct(self, signal):
        """
        Compute the Discrete Cosine Transform (DCT) of the input signal.

        Parameters:
        signal (numpy.ndarray): The input signal or filter banks matrix.
        num_coefficients (int): The number of coefficients to retain in the MFCC.

        Returns:
        numpy.ndarray: The DCT coefficients.
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
        Compute the Mel-Frequency Cepstral Coefficients (MFCC) of the signal.

        Returns:
        numpy.ndarray: The MFCC of the signal.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> mfcc = MFCC(signal)
        >>> mfcc_result = mfcc.compute_mfcc()
        >>> print(mfcc_result)
        """
        # Pre-emphasis
        emphasized_signal = np.append(
            self.signal[0], self.signal[1:] - 0.97 * self.signal[:-1]
        )

        # Frame the signal into overlapping frames
        frame_size = 0.025
        frame_stride = 0.01
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
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized_signal, z)
        indices = (
            np.tile(np.arange(0, frame_length), (num_frames, 1))
            + np.tile(
                np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)
            ).T
        )
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        frames *= np.hamming(frame_length)

        # Fourier-Transform and Power Spectrum
        NFFT = 512
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        pow_frames = (1.0 / NFFT) * (mag_frames**2)

        # Filter Banks
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
        filter_banks = 20 * np.log10(filter_banks)

        # Mel-frequency Cepstral Coefficients (MFCCs)
        mfcc = self.dct(filter_banks)
        return mfcc
