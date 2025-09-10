import numpy as np
from scipy.interpolate import interp1d

# from scipy.fft import ifft
# from scipy.signal import resample
import matplotlib.pyplot as plt

# from scipy.integrate import odeint
from scipy.integrate import solve_ivp

# from vitalDSP.utils.peak_detection import PeakDetection
# from vitalDSP.utils.common import ecg_detect_peaks


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
    N=None,
    duration=30,
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
        Approximate number of heartbeats (default is 60).
    duration : int, optional
        Approximate number of seconds (default is 30).
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

    # q = round(sfint / sfecg)
    if sfint % sfecg != 0:
        raise ValueError("sfint must be an integer multiple of sfecg")

    flo = 0.1
    fhi = 0.25
    flostd = 0.01
    fhistd = 0.01

    # calculate time scales for rr and total output
    sampfreqrr = 1
    trr = 1 / sampfreqrr
    rrmean = 60 / hrmean

    if duration is not None:
        Nrr = int(duration * hrmean / 60.0)
    else:
        Nrr = 2 ** (np.ceil(np.log2(N * rrmean / trr)))

    rr0 = rrprocess(
        flo,
        fhi,
        flostd,
        fhistd,
        lfhfratio,
        hrmean,
        hrstd,
        sampfreqrr,  # 1,
        Nrr,  # 2 ** int(np.ceil(np.log2(N * 60.0 / hrmean))),
    )

    # upsample rr time series from 1 Hz to sfint Hz
    rr = interp1d(np.linspace(0, len(rr0) - 1, len(rr0)), rr0, kind="cubic")(
        np.linspace(0, len(rr0) - 1, len(rr0) * sfint)
    )

    dt = 1.0 / sfint
    rrn = np.zeros_like(rr)
    tecg = 0
    i = 0
    while i < len(rr):
        tecg = tecg + rr[i]
        ip = int(np.round(tecg / dt))
        rrn[i : ip + 1] = rr[i]
        i = ip + 1
    Nt = ip
    x0 = [1, 0, 0.04]
    tspan = np.arange(0, (Nt - 1) * dt, dt)
    args = (rrn, sfint, ti, ai, bi)
    # solv_ode = solve_ivp(ordinary_differential_equation, [tspan[0], tspan[-1]],
    #                      x0, t_eval=np.arange(20.5, 21.5, 0.00001), args=args)
    solv_ode = solve_ivp(
        ordinary_differential_equation,
        [tspan[0], tspan[-1]],
        x0,
        t_eval=np.arange(tspan[0], tspan[-1], 1 / sfecg),
        args=args,
    )

    X = solv_ode.y
    z = X[2]
    z = (z - np.min(z)) * 1.6 / (np.max(z) - np.min(z)) - 0.4
    s = z + Anoise * (2 * np.random.rand(len(z)) - 1)

    # detector = PeakDetection(s, method="ecg_r_peak",
    #                          **{"distance":50,"window_size":7,"threshold_factor":1.8})
    # ipeaks = detector.detect_peaks()
    return s


def ordinary_differential_equation(
    t, x_equations, rr=None, sfint=None, ti=None, ai=None, bi=None
):
    x = x_equations[0]
    y = x_equations[1]
    z = x_equations[2]

    ta = np.arctan2(y, x)
    r0 = 1
    a0 = 1.0 - np.sqrt(x**2 + y**2) / r0
    ip = int(1 + np.floor(t * sfint))
    try:
        w0 = 2 * np.pi / rr[ip]
    except Exception:
        w0 = 2 * np.pi / rr[-1]

    fresp = 0.25
    zbase = 0.005 * np.sin(2 * np.pi * fresp * t)

    dx1dt = a0 * x - w0 * y
    dx2dt = a0 * y + w0 * x

    dti = np.fmod(ta - ti, 2 * np.pi)
    dx3dt = -np.sum(ai * dti * np.exp(-0.5 * np.divide(dti, bi) ** 2))
    dx3dt = dx3dt - 1.0 * (z - zbase)

    return [dx1dt, dx2dt, dx3dt]


def rrprocess(flo, fhi, flostd, fhistd, lfhfratio, hrmean, hrstd, sfrr, n):
    w1 = 2 * np.pi * flo
    w2 = 2 * np.pi * fhi
    c1 = 2 * np.pi * flostd
    c2 = 2 * np.pi * fhistd
    sig2 = 1
    sig1 = lfhfratio
    rrmean = 60 / hrmean
    rrstd = 60 * hrstd / (hrmean * hrmean)
    """
    Generating RR-intervals which have a bimodal power spectrum
    consisting of the sum of two Gaussian distributions
    """
    df = sfrr / n
    w = np.arange(0, n).T * 2 * np.pi * df
    dw1 = w - w1
    dw2 = w - w2

    # Resample Hw to ensure it has the correct length
    Hw1 = sig1 * np.exp(-0.5 * (dw1 / c1) ** 2) / np.sqrt(2 * np.pi * np.power(c1, 2))
    Hw2 = sig2 * np.exp(-0.5 * (dw2 / c2) ** 2) / np.sqrt(2 * np.pi * np.power(c2, 2))
    Hw = Hw1 + Hw2
    """
    An RR-interval time series T(t)
    with power spectrum is S(f)
    generated by taking the inverse Fourier transform of
    a sequence of complex numbers with amplitudes sqrt(S(f))
    and phases which are randomly distributed between 0 and 2pi
    """
    # Ensure Hw_resampled is non-negative before taking the square root
    Hw0_half = np.array(Hw[0 : int(n / 2)])
    Hw0 = np.append(Hw0_half, np.flip(Hw0_half))
    Sw = (sfrr / 2) * (Hw0**0.5)

    # Generate phase with correct length
    ph0 = 2 * np.pi * np.random.rand(int(n / 2) - 1, 1)
    # ph0 = 2 * np.pi * 0.001*np.arange(127).reshape(-1,1)
    ph = np.vstack((0, ph0, 0, -np.flip(ph0)))

    # create the complex number
    SwC = np.multiply(Sw.reshape(-1, 1), np.exp(1j * ph))
    inverse_res = np.fft.ifft(SwC.reshape(-1))
    x = (1 / n) * np.real(inverse_res)

    """
        By multiplying this time series by an appropriate scaling constant
        and adding an offset value, the resulting time series can be given
        any required mean and standard deviation
    """
    xstd = np.std(x)
    ratio = rrstd / xstd
    # Handle potential NaN values in rr
    rr = rrmean + x * ratio
    return rr


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
        if display:
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
        if display:
            plt.show()

    return time, ppg_signal


class SynthesizeData:
    """
    A class that provides access to all signal synthesis functions.
    This class wraps the individual functions for easier access and organization.
    """

    @staticmethod
    def generate_sinusoidal(
        frequency, sampling_rate, duration, amplitude=1.0, phase=0.0
    ):
        """Generate a sinusoidal wave."""
        return generate_sinusoidal(frequency, sampling_rate, duration, amplitude, phase)

    @staticmethod
    def generate_square_wave(
        frequency, sampling_rate, duration, amplitude=1.0, duty_cycle=0.5
    ):
        """Generate a square wave."""
        return generate_square_wave(
            frequency, sampling_rate, duration, amplitude, duty_cycle
        )

    @staticmethod
    def generate_noisy_signal(base_signal, noise_level=0.1):
        """Add Gaussian noise to a base signal."""
        return generate_noisy_signal(base_signal, noise_level)

    @staticmethod
    def generate_ecg_signal(
        duration=10,
        sampling_rate=1000,
        heart_rate=60,
        noise_level=0.01,
        qrs_width=0.08,
        t_wave_amplitude=0.3,
        p_wave_amplitude=0.25,
        randomize=False,
        display=False,
    ):
        """Generate a synthetic ECG signal."""
        return generate_ecg_signal(
            duration,
            sampling_rate,
            heart_rate,
            noise_level,
            qrs_width,
            t_wave_amplitude,
            p_wave_amplitude,
            randomize,
            display,
        )

    @staticmethod
    def generate_resp_signal(sampling_rate, duration, frequency=0.2, amplitude=0.5):
        """Generate a synthetic respiratory signal."""
        return generate_resp_signal(sampling_rate, duration, frequency, amplitude)

    @staticmethod
    def generate_synthetic_ppg_reversed(
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
        """Generate a synthetic PPG signal with reversed parameters."""
        return generate_synthetic_ppg_reversed(
            duration,
            sampling_rate,
            heart_rate,
            noise_level,
            diastolic_amplitude,
            diastolic_width,
            dicrotic_notch_depth,
            dicrotic_notch_delay,
            randomize,
            display,
        )

    @staticmethod
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
        """Generate a synthetic PPG signal."""
        return generate_synthetic_ppg(
            duration,
            sampling_rate,
            heart_rate,
            noise_level,
            diastolic_amplitude,
            diastolic_width,
            dicrotic_notch_depth,
            dicrotic_notch_delay,
            randomize,
            display,
        )

    @staticmethod
    def generate_ppg_data(duration=50, fs=1000):
        """Generate synthetic PPG data for testing."""
        t = np.linspace(0, duration, int(duration * fs))
        # Simple PPG-like signal
        signal = np.sin(2 * np.pi * 1.2 * t) * np.exp(-0.1 * t) + 0.5 * np.sin(
            2 * np.pi * 0.8 * t
        )
        signal += 0.1 * np.random.randn(len(signal))
        return t, signal
