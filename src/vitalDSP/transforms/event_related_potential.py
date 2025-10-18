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
    >>> from vitalDSP.transforms.event_related_potential import EventRelatedPotential
    >>> signal = np.random.randn(1000)
    >>> processor = EventRelatedPotential(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""


import numpy as np


class EventRelatedPotential:
    """
    A class to compute Event-Related Potentials (ERP) for detecting brain responses to stimuli in EEG.

    Event-Related Potentials are averaged EEG signals that are time-locked to specific sensory, cognitive, or motor events. This class provides a method to extract and average these potentials from continuous EEG data based on stimulus onset times.

    ERPs are useful in neuroscience research to study the brain's response to specific events and are commonly used in experiments involving visual, auditory, or other sensory stimuli.

    Methods
    -------
    compute_erp : method
        Computes the ERP of the signal based on stimulus times.
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
        Initialize the EventRelatedPotential class with the signal and parameters for ERP computation.

        Parameters
        ----------
        signal : numpy.ndarray
            The input EEG signal from which ERPs will be computed. This should be a 1D array representing the EEG data over time.
        stimulus_times : numpy.ndarray
            An array of times (in samples) when stimuli occurred. These times should correspond to indices in the signal array.
        pre_stimulus : float, optional
            The duration before the stimulus to include in the ERP segment (in seconds). Default is 0.1 seconds.
        post_stimulus : float, optional
            The duration after the stimulus to include in the ERP segment (in seconds). Default is 0.4 seconds.
        sample_rate : int, optional
            The sampling rate of the EEG signal (in Hz). Default is 1000 Hz.

        Attributes
        ----------
        pre_stimulus : int
            The number of samples corresponding to the pre-stimulus duration.
        post_stimulus : int
            The number of samples corresponding to the post-stimulus duration.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 1000))
        >>> stimulus_times = np.array([100, 300, 500])
        >>> erp = EventRelatedPotential(signal, stimulus_times, pre_stimulus=0.1, post_stimulus=0.4, sample_rate=1000)
        >>> erp_result = erp.compute_erp()
        >>> print(erp_result)
        """
        self.signal = signal
        self.stimulus_times = stimulus_times
        self.pre_stimulus = int(pre_stimulus * sample_rate)
        self.post_stimulus = int(post_stimulus * sample_rate)
        self.sample_rate = sample_rate

    def compute_erp(self):
        """
        Compute the Event-Related Potentials (ERP) of the signal.

        This method extracts segments of the EEG signal around each stimulus time, starting from `pre_stimulus` seconds before the stimulus and ending `post_stimulus` seconds after. These segments are then averaged to produce the ERP, which represents the brain's averaged response to the stimuli.

        The method handles edge cases where the start or end of the segment might fall outside the signal's bounds by ignoring such segments.

        Returns
        -------
        numpy.ndarray
            The average ERP of the signal, which is a 1D array representing the time-locked average response across all stimuli.

        Raises
        ------
        ValueError
            If no valid ERP segments are found, possibly due to stimulus times being out of the signal's bounds.

        Examples
        --------
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

            # Handle edge cases where start or end might go out of signal bounds
            if start < 0 or end > len(self.signal):
                continue

            erp_segment = self.signal[start:end]
            erps.append(erp_segment)

        # Return the average ERP, ensuring there are segments to average
        if erps:
            return np.mean(erps, axis=0)
        else:
            raise ValueError(
                "No valid ERP segments were found. Check stimulus times and signal length."
            )
