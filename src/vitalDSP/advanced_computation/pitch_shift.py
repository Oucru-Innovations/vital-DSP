import numpy as np


class PitchShift:
    """
    A class for pitch shifting and pitch detection, primarily used in the analysis of vocal pitch.
    This is particularly useful in diagnosing speech disorders and other vocal characteristics.

    Methods
    -------
    shift_pitch : method
        Shifts the pitch of the input signal by a specified number of semitones.
    detect_pitch : method
        Detects the pitch of the input signal using the autocorrelation method.

    Example Usage
    -------------
    signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 1000))  # A simple sine wave at 440 Hz (A4)
    pitch_shift = PitchShift(signal, sampling_rate=1000)
    shifted_signal = pitch_shift.shift_pitch(semitones=2)
    print("Shifted Signal:", shifted_signal)

    detected_pitch = pitch_shift.detect_pitch()
    print("Detected Pitch:", detected_pitch, "Hz")
    """

    def __init__(self, signal, sampling_rate=1000):
        """
        Initialize the PitchShift class with the input signal and its sampling rate.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal (e.g., a vocal recording) to be processed.
        sampling_rate : int, optional
            The sampling rate of the input signal in Hz. Default is 1000 Hz.
        """
        self.signal = signal
        self.sampling_rate = sampling_rate

    def shift_pitch(self, semitones=1):
        """
        Shift the pitch of the input signal by a given number of semitones.

        This method shifts the pitch of the signal without altering its duration, which is useful
        for modifying vocal pitch in a controlled manner.

        Parameters
        ----------
        semitones : int, optional
            The number of semitones to shift the pitch. A positive value raises the pitch,
            while a negative value lowers it. Default is 1 semitone.

        Returns
        -------
        shifted_signal : numpy.ndarray
            The pitch-shifted signal.

        Examples
        --------
        >>> signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 1000))  # 440 Hz signal
        >>> pitch_shift = PitchShift(signal, sampling_rate=1000)
        >>> shifted_signal = pitch_shift.shift_pitch(semitones=3)
        >>> print("Shifted Signal:", shifted_signal)
        """
        factor = 2 ** (semitones / 12.0)
        indices = np.round(np.arange(0, len(self.signal), factor))
        indices = indices[indices < len(self.signal)].astype(int)
        shifted_signal = self.signal[indices]
        return shifted_signal

    def detect_pitch(self):
        """
        Detect the pitch of the input signal using the autocorrelation method.

        The autocorrelation method is commonly used to estimate the fundamental frequency (pitch) of a signal.
        This method is particularly effective for monophonic signals like speech or musical notes.

        Returns
        -------
        pitch : float
            The detected pitch of the signal in Hertz (Hz).

        Examples
        --------
        >>> signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 1000))  # 440 Hz signal
        >>> pitch_shift = PitchShift(signal, sampling_rate=1000)
        >>> detected_pitch = pitch_shift.detect_pitch()
        >>> print("Detected Pitch:", detected_pitch, "Hz")
        """
        autocorr = np.correlate(self.signal, self.signal, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        # Handle silence (if the signal is flat)
        if np.all(autocorr == 0):
            return float("inf")

        d = np.diff(autocorr)

        # Find the first positive difference to ignore the zero-lag peak
        try:
            start = np.where(d > 0)[0][0]
        except IndexError:
            # In case no positive difference is found (e.g., for silence)
            return float("inf")

        # Find the first peak after the zero-lag peak
        peak = np.argmax(autocorr[start:]) + start
        pitch = self.sampling_rate / peak if peak > 0 else float("inf")

        return pitch
