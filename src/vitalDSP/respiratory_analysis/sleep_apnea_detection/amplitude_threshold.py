from vitalDSP.preprocess.preprocess_operations import preprocess_signal


def detect_apnea_amplitude(
    signal,
    sampling_rate,
    threshold,
    min_duration=10,
    preprocess=None,
    **preprocess_kwargs
):
    """
    Detect sleep apnea events based on amplitude thresholding.

    Parameters
    ----------
    signal : numpy.ndarray
        The input respiratory signal.
    sampling_rate : float
        The sampling rate of the signal in Hz.
    threshold : float
        Amplitude threshold below which an apnea event is detected.
    min_duration : int, optional (default=10)
        Minimum duration (in seconds) for which the signal must be below the threshold to be considered an apnea event.
    preprocess : str, optional
        The preprocessing method to apply before detection (e.g., "bandpass", "wavelet").
    preprocess_kwargs : dict, optional
        Additional arguments for the preprocessing function.

    Returns
    -------
    apnea_events : list of tuple
        List of apnea events, each represented as a tuple (start_time, end_time) in seconds.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 60, 0.01)) * 0.2
    >>> apnea_events = detect_apnea_amplitude(signal, sampling_rate=100, threshold=0.1)
    >>> print(apnea_events)
    """
    if preprocess:
        signal = preprocess_signal(
            signal, sampling_rate, filter_type=preprocess, **preprocess_kwargs
        )

    # Identify segments where the signal amplitude is below the threshold
    below_threshold = signal < threshold
    apnea_events = []
    current_start = None

    for i, below in enumerate(below_threshold):
        if below and current_start is None:
            current_start = i
        elif not below and current_start is not None:
            duration = (i - current_start) / sampling_rate
            if duration >= min_duration:
                apnea_events.append((current_start / sampling_rate, i / sampling_rate))
            current_start = None

    # Check if the signal ends during an apnea event
    if current_start is not None:
        duration = (len(signal) - current_start) / sampling_rate
        if duration >= min_duration:
            apnea_events.append(
                (current_start / sampling_rate, len(signal) / sampling_rate)
            )

    return apnea_events
