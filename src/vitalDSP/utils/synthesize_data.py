import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import ifft
from scipy.signal import resample
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from vitalDSP.utils.common import ecg_detect_peaks


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


def generate_square_wave(
    frequency, sampling_rate, duration, amplitude=1.0, duty_cycle=0.5
):
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


def generate_ecg_signal(
    sfecg=256,
    N=256,
    Anoise=0.0,
    hrmean=60,
    hrstd=1.0,
    lfhfratio=0.5,
    sfint=512,
    ti=None,
    ai=None,
    bi=None,
):
    """
    Generate a synthetic ECG signal using the dynamical model from McSharry et al.

    Parameters
    ----------
    sfecg : int, optional
        ECG sampling frequency in Hz (default is 256).
    N : int, optional
        Approximate number of heartbeats (default is 256).
    Anoise : float, optional
        Amplitude of additive uniformly distributed measurement noise (default is 0).
    hrmean : float, optional
        Mean heart rate in beats per minute (default is 60).
    hrstd : float, optional
        Standard deviation of heart rate in beats per minute (default is 1).
    lfhfratio : float, optional
        Low frequency to high frequency ratio (default is 0.5).
    sfint : int, optional
        Internal sampling frequency in Hz (default is 512).
    ti : list, optional
        Angles of extrema in degrees for PQRST (default is [-70, -15, 0, 15, 100]).
    ai : list, optional
        Z-position of extrema for PQRST (default is [1.2, -5, 30, -7.5, 0.75]).
    bi : list, optional
        Gaussian width of peaks for PQRST (default is [0.25, 0.1, 0.1, 0.1, 0.4]).

    Returns
    -------
    s : ndarray
        Generated ECG signal.
    ipeaks : ndarray
        Labels for PQRST peaks.

    Examples
    --------
    >>> signal = generate_ecg_signal(sfecg=256, N=256, Anoise=0.01, hrmean=70, sfint=512)
    >>> print(signal)
    """

    if ti is None:
        ti = [-70, -15, 0, 15, 100]
    if ai is None:
        ai = [1.2, -5, 30, -7.5, 0.75]
    if bi is None:
        bi = [0.25, 0.1, 0.1, 0.1, 0.4]

    ti = np.array(ti) * np.pi / 180.0

    hrfact = np.sqrt(hrmean / 60.0)
    hrfact2 = np.sqrt(hrfact)
    bi = hrfact * np.array(bi)
    ti = np.array([hrfact2, hrfact, 1, hrfact, hrfact2]) * ti

    q = round(sfint / sfecg)
    if sfint % sfecg != 0:
        raise ValueError("sfint must be an integer multiple of sfecg")

    flo = 0.1
    fhi = 0.25
    flostd = 0.01
    fhistd = 0.01

    rr0 = rrprocess(
        flo,
        fhi,
        flostd,
        fhistd,
        lfhfratio,
        hrmean,
        hrstd,
        1,
        2 ** int(np.ceil(np.log2(N * 60.0 / hrmean))),
    )

    rr = interp1d(np.linspace(0, len(rr0) - 1, len(rr0)), rr0, kind="cubic")(
        np.linspace(0, len(rr0) - 1, len(rr0) * sfint)
    )

    dt = 1.0 / sfint
    rrn = np.zeros_like(rr)
    tecg = 0
    i = 0
    while i < len(rr):
        tecg += rr[i]
        ip = int(round(tecg / dt))
        rrn[i:ip] = rr[i]
        i = ip

    Tspan = np.arange(0, len(rrn) * dt, dt)
    x0 = [1, 0, 0.04]

    X0 = odeint(derivsecgsyn, x0, Tspan, args=(rrn, sfint, ti, ai, bi))

    X = X0[::q, :]

    ipeaks = ecg_detect_peaks(X, ti, sfecg)

    z = X[:, 2]
    z = (z - np.min(z)) * 1.6 / (np.max(z) - np.min(z)) - 0.4

    s = z + Anoise * (2 * np.random.rand(len(z)) - 1)

    return s, ipeaks


def rrprocess(flo, fhi, flostd, fhistd, lfhfratio, hrmean, hrstd, sfrr, n):
    w1 = 2 * np.pi * flo
    w2 = 2 * np.pi * fhi
    c1 = 2 * np.pi * flostd
    c2 = 2 * np.pi * fhistd
    sig2 = 1
    sig1 = lfhfratio
    rrmean = 60.0 / hrmean
    rrstd = 60.0 * hrstd / (hrmean**2)

    df = sfrr / n
    w = np.arange(n) * 2 * np.pi * df
    Hw1 = sig1 * np.exp(-0.5 * ((w - w1) / c1) ** 2) / np.sqrt(2 * np.pi * c1**2)
    Hw2 = sig2 * np.exp(-0.5 * ((w - w2) / c2) ** 2) / np.sqrt(2 * np.pi * c2**2)
    Hw = Hw1 + Hw2

    # Resample Hw to ensure it has the correct length
    Hw_resampled = resample(Hw, n)

    # Ensure Hw_resampled is non-negative before taking the square root
    Hw_resampled = np.maximum(Hw_resampled, 0)

    Sw = (sfrr / 2) * np.sqrt(Hw_resampled)

    # Generate phase with correct length
    ph = 2 * np.pi * np.random.rand(n)
    SwC = Sw * np.exp(1j * ph)
    x = np.real(ifft(SwC)) / n

    rr = rrmean + x * (rrstd / np.std(x))

    # Handle potential NaN values in rr
    rr = np.nan_to_num(rr, nan=rrmean)

    return rr


def derivsecgsyn(x, t, rr, sfint, ti, ai, bi):
    # xi = np.cos(ti)
    # yi = np.sin(ti)
    ta = np.arctan2(x[1], x[0])
    r0 = 1.0
    a0 = 1.0 - np.sqrt(x[0] ** 2 + x[1] ** 2) / r0
    ip = int(np.floor(t * sfint))

    # Ensure ip is within the bounds of the rr array
    if ip >= len(rr):
        ip = len(rr) - 1

    w0 = 2 * np.pi / rr[ip]

    fresp = 0.25
    zbase = 0.005 * np.sin(2 * np.pi * fresp * t)

    dx1dt = a0 * x[0] - w0 * x[1]
    dx2dt = a0 * x[1] + w0 * x[0]

    dti = np.remainder(ta - ti, 2 * np.pi)
    dx3dt = -np.sum(ai * dti * np.exp(-0.5 * (dti / bi) ** 2)) - 1.0 * (x[2] - zbase)

    dxdt = [dx1dt, dx2dt, dx3dt]
    return dxdt


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


def generate_synthetic_ppg_reversed(
    duration=10, sampling_rate=1000, heart_rate=60, noise_level=0.01, display=False
):
    """
    Generate a synthetic PPG signal.

    Parameters
    ----------
    duration : float, optional
        Duration of the signal in seconds (default is 10).
    sampling_rate : int, optional
        Sampling rate in Hz (default is 1000).
    heart_rate : float, optional
        Heart rate in beats per minute (default is 60).
    noise_level : float, optional
        Standard deviation of the Gaussian noise to be added (default is 0.01).
    display : bool, optional
        If True, displays the generated PPG signal (default is False).

    Returns
    -------
    time : numpy.ndarray
        Array of time values.
    ppg_signal : numpy.ndarray
        The generated synthetic PPG signal.

    Example
    -------
    >>> time, ppg_signal = generate_synthetic_ppg(duration=10, heart_rate=60, display=True)
    """

    # Calculate the number of heartbeats and time array
    num_beats = int(duration * heart_rate / 60)
    time = np.linspace(0, duration, int(sampling_rate * duration))

    # Define PPG pulse using a combination of Gaussian functions
    def ppg_pulse(t):
        # Parameters for a typical PPG pulse waveform
        amplitude = 1.0
        width_systolic = 0.1  # Width of the systolic peak
        width_diastolic = 0.2  # Width of the diastolic component
        dicrotic_notch_depth = 0.4  # Depth of the dicrotic notch
        dicrotic_notch_delay = 0.1  # Time delay for the dicrotic notch

        # Systolic peak
        systolic_peak = amplitude * np.exp(-((t - 0.15) ** 2) / (2 * width_systolic**2))

        # Dicrotic notch and wave
        dicrotic_wave = dicrotic_notch_depth * np.exp(
            -((t - (0.15 + dicrotic_notch_delay)) ** 2) / (2 * width_diastolic**2)
        )

        return systolic_peak - dicrotic_wave

    # Generate one cycle of the PPG signal
    cycle_length = int(sampling_rate * 60 / heart_rate)
    ppg_cycle = ppg_pulse(np.linspace(0, 1, cycle_length))

    # Tile the cycle to create the full PPG signal
    ppg_signal = np.tile(ppg_cycle, num_beats)
    ppg_signal = ppg_signal[: len(time)]  # Trim to the exact length

    # Add Gaussian noise to simulate artifacts
    noise = np.random.normal(0, noise_level, len(ppg_signal))
    ppg_signal += noise

    # Display the signal if required
    if display:
        plt.figure(figsize=(10, 4))
        plt.plot(time, ppg_signal)
        plt.title("Synthetic PPG Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()

    return time, ppg_signal


def generate_synthetic_ppg(
    duration=10,
    sampling_rate=1000,
    heart_rate=60,
    noise_level=0.01,
    diastolic_amplitude=0.8,
    diastolic_width=0.15,
    dicrotic_notch_depth=0.6,
    dicrotic_notch_delay=0.2,
    randomize=False,
    display=False,
):
    """
    Generate a synthetic PPG signal with adjustable parameters.

    Parameters
    ----------
    duration : float, optional
        Duration of the signal in seconds (default is 10).
    sampling_rate : int, optional
        Sampling rate in Hz (default is 1000).
    heart_rate : float, optional
        Heart rate in beats per minute (default is 60).
    noise_level : float, optional
        Standard deviation of the Gaussian noise to be added (default is 0.01).
    diastolic_amplitude : float, optional
        Amplitude of the diastolic peak relative to the systolic peak (default is 0.8).
    diastolic_width : float, optional
        Width of the diastolic peak (default is 0.15).
    dicrotic_notch_depth : float, optional
        Depth of the dicrotic notch (default is 0.6).
    dicrotic_notch_delay : float, optional
        Time delay for the diastolic peak (default is 0.2).
    randomize : bool, optional
        If True, introduces randomness in the diastolic amplitude, width, and dicrotic notch depth (default is False).
    display : bool, optional
        If True, displays the generated PPG signal (default is False).

    Returns
    -------
    time : numpy.ndarray
        Array of time values.
    ppg_signal : numpy.ndarray
        The generated synthetic PPG signal.

    Example
    -------
    >>> time, ppg_signal = generate_synthetic_ppg(duration=10, heart_rate=60, randomize=True, display=True)
    """

    # Apply randomization if requested
    if randomize:
        diastolic_amplitude += np.random.uniform(-0.1, 0.1) * diastolic_amplitude
        diastolic_width += np.random.uniform(-0.05, 0.05) * diastolic_width
        dicrotic_notch_depth += np.random.uniform(-0.1, 0.1) * dicrotic_notch_depth
        dicrotic_notch_delay += np.random.uniform(-0.05, 0.05) * dicrotic_notch_delay

    # Calculate the number of heartbeats and time array
    num_beats = int(duration * heart_rate / 60)
    time = np.linspace(0, duration, int(sampling_rate * duration))

    # Define PPG pulse using a combination of Gaussian functions
    def ppg_pulse(t):
        amplitude = 1.0
        width_systolic = 0.08  # Width of the systolic peak

        # Systolic peak (tallest peak)
        systolic_peak = amplitude * np.exp(-((t - 0.2) ** 2) / (2 * width_systolic**2))

        # Diastolic peak (adjustable)
        diastolic_peak = (
            diastolic_amplitude
            * amplitude
            * np.exp(
                -((t - (0.2 + dicrotic_notch_delay)) ** 2) / (2 * diastolic_width**2)
            )
        )

        return systolic_peak + diastolic_peak

    # Generate one cycle of the PPG signal
    cycle_length = int(sampling_rate * 60 / heart_rate)
    ppg_cycle = ppg_pulse(np.linspace(0, 1, cycle_length))

    # Tile the cycle to create the full PPG signal
    ppg_signal = np.tile(ppg_cycle, num_beats)
    ppg_signal = ppg_signal[: len(time)]  # Trim to the exact length

    # Add Gaussian noise to simulate artifacts
    noise = np.random.normal(0, noise_level, len(ppg_signal))
    ppg_signal += noise

    # Display the signal if required
    if display:
        plt.figure(figsize=(10, 4))
        plt.plot(time, ppg_signal)
        plt.title("Synthetic PPG Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()

    return time, ppg_signal
