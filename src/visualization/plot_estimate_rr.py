import numpy as np
from respiratory_analysis.estimate_rr import fft_based_rr, peak_detection_rr, time_domain_rr, frequency_domain_rr
import plotly.graph_objects as go
from utils.common import find_peaks
from plotly.subplots import make_subplots

def plot_rr_estimations(signal, sampling_rate, preprocess=None, **preprocess_kwargs):
    """
    Visualize respiratory rate estimation using multiple methods.

    Parameters
    ----------
    signal : numpy.ndarray
        The input respiratory signal.
    sampling_rate : float
        The sampling rate of the signal in Hz.
    preprocess : str, optional
        The preprocessing method to apply before estimation (e.g., "bandpass", "wavelet").
    preprocess_kwargs : dict, optional
        Additional arguments for the preprocessing function.
    """

    # Estimate RR using different methods
    rr_fft = fft_based_rr(signal, sampling_rate, preprocess=preprocess, **preprocess_kwargs)
    rr_peak = peak_detection_rr(signal, sampling_rate, preprocess=preprocess, **preprocess_kwargs)
    rr_time = time_domain_rr(signal, sampling_rate, preprocess=preprocess, **preprocess_kwargs)
    rr_freq = frequency_domain_rr(signal, sampling_rate, preprocess=preprocess, **preprocess_kwargs)

    # Prepare the subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f"FFT-Based RR Estimation: {rr_fft:.2f} BPM",
            f"Peak Detection RR Estimation: {rr_peak:.2f} BPM",
            f"Time-Domain RR Estimation: {rr_time:.2f} BPM",
            f"Frequency-Domain RR Estimation: {rr_freq:.2f} BPM"
        )
    )

    time_axis = np.arange(len(signal)) / sampling_rate

    # Plot original signal
    fig.add_trace(
        go.Scatter(x=time_axis, y=signal, mode='lines', name='Original Signal'),
        row=1, col=1
    )

    # Add FFT-based RR estimation
    freq_axis = np.fft.fftfreq(len(signal), d=1/sampling_rate)
    fft_spectrum = np.abs(np.fft.fft(signal))
    fig.add_trace(
        go.Scatter(x=freq_axis[:len(freq_axis)//2], y=fft_spectrum[:len(freq_axis)//2], mode='lines', name='FFT Spectrum'),
        row=1, col=1
    )

    # Add Peak Detection RR estimation
    peaks = find_peaks(signal)[0]
    fig.add_trace(
        go.Scatter(x=time_axis, y=signal, mode='lines', name='Original Signal'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_axis[peaks], y=signal[peaks], mode='markers', name='Detected Peaks', marker=dict(color='red')),
        row=2, col=1
    )

    # Add Time-Domain RR estimation (Autocorrelation)
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    fig.add_trace(
        go.Scatter(x=np.arange(len(autocorr)) / sampling_rate, y=autocorr, mode='lines', name='Autocorrelation'),
        row=3, col=1
    )

    # Add Frequency-Domain RR estimation (Welch method)
    from scipy.signal import welch
    freqs, psd = welch(signal, fs=sampling_rate)
    fig.add_trace(
        go.Scatter(x=freqs, y=psd, mode='lines', name='PSD (Welch)'),
        row=4, col=1
    )

    # Update layout for better visualization
    fig.update_layout(
        height=1000, width=800,
        title_text="Comparative Respiratory Rate Estimation Visualization",
        showlegend=True
    )

    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", row=3, col=1)
    fig.update_xaxes(title_text="Frequency [Hz]", row=1, col=1)
    fig.update_xaxes(title_text="Frequency [Hz]", row=4, col=1)

    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig.update_yaxes(title_text="Autocorrelation", row=3, col=1)
    fig.update_yaxes(title_text="Power Spectral Density", row=4, col=1)

    # Show plot
    fig.show()


# Example usage with a synthetic signal
signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01)) + 0.1 * np.random.normal(size=1000)
plot_rr_estimations(signal, sampling_rate=100, preprocess='bandpass', lowcut=0.1, highcut=0.5)
