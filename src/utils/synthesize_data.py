import numpy as np

def generate_sinusoidal(frequency, sampling_rate, duration, amplitude=1.0, phase=0.0):
    """
    Generate a sinusoidal wave.

    Parameters
    ----------
    frequency : float
        Frequency of the sinusoid in Hz.
    sampling_rate : float
        Sampling rate in Hz.
    duration : float
        Duration of the signal in seconds.
    amplitude : float, optional (default=1.0)
        Amplitude of the sinusoid.
    phase : float, optional (default=0.0)
        Phase shift of the sinusoid in radians.

    Returns
    -------
    signal : numpy.ndarray
        The generated sinusoidal signal.

    Examples
    --------
    >>> signal = generate_sinusoidal(frequency=1.0, sampling_rate=100.0, duration=5.0)
    >>> print(signal)
    """
    t = np.arange(0, duration, 1 / sampling_rate)
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return signal

def generate_square_wave(frequency, sampling_rate, duration, amplitude=1.0, duty_cycle=0.5):
    """
    Generate a square wave.

    Parameters
    ----------
    frequency : float
        Frequency of the square wave in Hz.
    sampling_rate : float
        Sampling rate in Hz.
    duration : float
        Duration of the signal in seconds.
    amplitude : float, optional (default=1.0)
        Amplitude of the square wave.
    duty_cycle : float, optional (default=0.5)
        Duty cycle of the square wave (fraction of one period in which the signal is high).

    Returns
    -------
    signal : numpy.ndarray
        The generated square wave signal.

    Examples
    --------
    >>> signal = generate_square_wave(frequency=1.0, sampling_rate=100.0, duration=5.0)
    >>> print(signal)
    """
    t = np.arange(0, duration, 1 / sampling_rate)
    signal = amplitude * np.where(np.mod(t * frequency, 1) < duty_cycle, 1.0, -1.0)
    return signal

def generate_noisy_signal(base_signal, noise_level=0.1):
    """
    Add Gaussian noise to a base signal.

    Parameters
    ----------
    base_signal : numpy.ndarray
        The input signal to which noise will be added.
    noise_level : float, optional (default=0.1)
        The standard deviation of the Gaussian noise.

    Returns
    -------
    noisy_signal : numpy.ndarray
        The signal with added Gaussian noise.

    Examples
    --------
    >>> base_signal = generate_sinusoidal(frequency=1.0, sampling_rate=100.0, duration=5.0)
    >>> noisy_signal = generate_noisy_signal(base_signal, noise_level=0.2)
    >>> print(noisy_signal)
    """
    noise = np.random.normal(0, noise_level, len(base_signal))
    noisy_signal = base_signal + noise
    return noisy_signal

def generate_ecg_signal(sampling_rate, duration):
    """
    Generate a synthetic ECG signal.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate in Hz.
    duration : float
        Duration of the signal in seconds.

    Returns
    -------
    ecg_signal : numpy.ndarray
        The generated synthetic ECG signal.

    Examples
    --------
    >>> ecg_signal = generate_ecg_signal(sampling_rate=1000.0, duration=5.0)
    >>> print(ecg_signal)
    """
    t = np.arange(0, duration, 1 / sampling_rate)
    heart_rate = 60  # beats per minute
    rr_interval = 60 / heart_rate
    qrs_duration = 0.1  # in seconds

    # PQRST waveform components
    p_wave = 0.25 * np.sin(np.pi * t / (rr_interval / 8))
    q_wave = -0.5 * np.sin(np.pi * t / (rr_interval / 20))
    r_wave = np.sin(np.pi * t / (rr_interval / 15))
    s_wave = -0.75 * np.sin(np.pi * t / (rr_interval / 30))
    t_wave = 0.2 * np.sin(np.pi * t / (rr_interval / 7))

    ecg_signal = np.zeros_like(t)
    for i in range(len(t)):
        if i % int(rr_interval * sampling_rate) < int(qrs_duration * sampling_rate):
            ecg_signal[i] = p_wave[i] + q_wave[i] + r_wave[i] + s_wave[i] + t_wave[i]

    return ecg_signal

def generate_resp_signal(sampling_rate, duration, frequency=0.2, amplitude=0.5):
    """
    Generate a synthetic respiratory signal.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate in Hz.
    duration : float
        Duration of the signal in seconds.
    frequency : float, optional (default=0.2)
        Frequency of the respiratory signal in Hz.
    amplitude : float, optional (default=0.5)
        Amplitude of the respiratory signal.

    Returns
    -------
    resp_signal : numpy.ndarray
        The generated synthetic respiratory signal.

    Examples
    --------
    >>> resp_signal = generate_resp_signal(sampling_rate=1000.0, duration=10.0)
    >>> print(resp_signal)
    """
    t = np.arange(0, duration, 1 / sampling_rate)
    resp_signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return resp_signal
