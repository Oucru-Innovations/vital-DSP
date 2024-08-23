import numpy as np


class EventRelatedPotential:
    """
    A class to compute Event-Related Potentials (ERP) for detecting brain responses to stimuli in EEG.

    Methods:
    - compute_erp: Computes the ERP of the signal.
    """

    def __init__(
        self,
        signal,
        stimulus_times,
        pre_stimulus=0.1,
        post_stimulus=0.4,
        sample_rate=1000,
    ):
        """
        Initialize the EventRelatedPotential class with the signal.

        Parameters:
        signal (numpy.ndarray): The input EEG signal.
        stimulus_times (numpy.ndarray): Times of the stimulus events.
        pre_stimulus (float): Time before the stimulus to include in the ERP (in seconds).
        post_stimulus (float): Time after the stimulus to include in the ERP (in seconds).
        sample_rate (int): The sample rate of the signal.
        """
        self.signal = signal
        self.stimulus_times = stimulus_times
        self.pre_stimulus = int(pre_stimulus * sample_rate)
        self.post_stimulus = int(post_stimulus * sample_rate)
        self.sample_rate = sample_rate

    def compute_erp(self):
        """
        Compute the Event-Related Potentials (ERP) of the signal.

        Returns:
        numpy.ndarray: The average ERP of the signal.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> stimulus_times = np.array([100, 300, 500])
        >>> erp = EventRelatedPotential(signal, stimulus_times)
        >>> erp_result = erp.compute_erp()
        >>> print(erp_result)
        """
        erps = []
        for stimulus_time in self.stimulus_times:
            start = stimulus_time - self.pre_stimulus
            end = stimulus_time + self.post_stimulus
            erp_segment = self.signal[start:end]
            erps.append(erp_segment)
        return np.mean(erps, axis=0)
