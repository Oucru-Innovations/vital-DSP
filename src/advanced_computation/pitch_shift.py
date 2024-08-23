import numpy as np

class PitchShift:
    """
    Pitch Shift analysis for analyzing vocal pitch, useful in diagnosing speech disorders.

    Methods:
    - shift_pitch: Shifts the pitch of the input signal by a given semitone factor.
    - detect_pitch: Detects the pitch of the input signal using autocorrelation.

    Example Usage:
    --------------
    signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 1000))  # A simple sine wave at 440 Hz (A4)
    pitch_shift = PitchShift(signal)
    shifted_signal = pitch_shift.shift_pitch(semitones=2)
    print("Shifted Signal:", shifted_signal)

    detected_pitch = pitch_shift.detect_pitch()
    print("Detected Pitch:", detected_pitch)
    """

    def __init__(self, signal, sampling_rate=1000):
        self.signal = signal
        self.sampling_rate = sampling_rate

    def shift_pitch(self, semitones=1):
        """
        Shift the pitch of the input signal by a given semitone factor.

        Parameters:
        semitones (int): The number of semitones to shift the pitch.

        Returns:
        numpy.ndarray: The pitch-shifted signal.
        """
        factor = 2 ** (semitones / 12.0)
        indices = np.round(np.arange(0, len(self.signal), factor))
        indices = indices[indices < len(self.signal)].astype(int)
        return self.signal[indices]

    def detect_pitch(self):
        """
        Detect the pitch of the input signal using autocorrelation.

        Returns:
        float: The detected pitch in Hz.
        """
        autocorr = np.correlate(self.signal, self.signal, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        d = np.diff(autocorr)
        start = np.where(d > 0)[0][0]
        peak = np.argmax(autocorr[start:]) + start
        pitch = self.sampling_rate / peak
        return pitch
