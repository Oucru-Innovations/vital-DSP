"""
Transform Functions
Contains all signal transform implementation functions.
"""

import logging
import numpy as np
from dash import html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

logger = logging.getLogger(__name__)


def apply_fft_transform(
    time_data, signal_data, sampling_freq, options, window_type, n_points
):
    """Apply FFT transform and create visualizations."""
    from scipy import signal as scipy_signal
    from scipy.fft import fft, fftfreq

    # Apply window
    if window_type == "boxcar":
        window = np.ones(len(signal_data))
    elif window_type == "hamming":
        window = scipy_signal.hamming(len(signal_data))
    elif window_type == "hann":
        window = scipy_signal.hann(len(signal_data))
    elif window_type == "blackman":
        window = scipy_signal.blackman(len(signal_data))
    elif window_type == "kaiser":
        window = scipy_signal.kaiser(len(signal_data), beta=5)
    else:
        window = scipy_signal.hann(len(signal_data))

    windowed_signal = signal_data * window

    # Compute FFT
    n = min(n_points, len(windowed_signal))
    fft_values = fft(windowed_signal, n=n)
    frequencies = fftfreq(n, d=1 / sampling_freq)

    # Only positive frequencies
    positive_mask = frequencies >= 0
    frequencies = frequencies[positive_mask]
    fft_values = fft_values[positive_mask]

    # Calculate magnitude, phase, power
    magnitude = np.abs(fft_values)
    phase = np.angle(fft_values)
    power = magnitude**2

    # Create main plot (magnitude spectrum)
    main_fig = go.Figure()

    use_log = "log_scale" in (options or [])

    main_fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=magnitude if not use_log else np.log10(magnitude + 1e-10),
            mode="lines",
            name="Magnitude",
            line=dict(color="blue", width=1.5),
        )
    )

    main_fig.update_layout(
        title="FFT - Magnitude Spectrum",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude" + (" (log scale)" if use_log else ""),
        hovermode="x unified",
        template="plotly_white",
    )

    # Create analysis plots
    analysis_fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Phase Spectrum", "Power Spectrum"),
        vertical_spacing=0.15,
    )

    if "phase" in (options or []):
        analysis_fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=phase,
                mode="lines",
                name="Phase",
                line=dict(color="green"),
            ),
            row=1,
            col=1,
        )

    if "power" in (options or []):
        analysis_fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=power if not use_log else np.log10(power + 1e-10),
                mode="lines",
                name="Power",
                line=dict(color="red"),
            ),
            row=2,
            col=1,
        )

    analysis_fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
    analysis_fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    analysis_fig.update_yaxes(title_text="Phase (rad)", row=1, col=1)
    analysis_fig.update_yaxes(
        title_text="Power" + (" (log)" if use_log else ""), row=2, col=1
    )
    analysis_fig.update_layout(height=800, template="plotly_white", showlegend=False)

    # Results summary
    dominant_freq_idx = np.argmax(magnitude[1:]) + 1  # Skip DC component
    dominant_freq = frequencies[dominant_freq_idx]
    dominant_magnitude = magnitude[dominant_freq_idx]

    results = html.Div(
        [
            html.H6("FFT Analysis Results"),
            html.P(f"Dominant Frequency: {dominant_freq:.2f} Hz"),
            html.P(f"Dominant Magnitude: {dominant_magnitude:.2f}"),
            html.P(f"Total Power: {np.sum(power):.2e}"),
            html.P(f"Frequency Resolution: {frequencies[1] - frequencies[0]:.4f} Hz"),
        ]
    )

    # Peak analysis
    if "peak_detection" in (options or []):
        from scipy.signal import find_peaks

        peaks_idx, properties = find_peaks(
            magnitude, height=np.max(magnitude) * 0.1, distance=10
        )
        peak_freqs = frequencies[peaks_idx]
        peak_mags = magnitude[peaks_idx]

        # Sort by magnitude
        sorted_idx = np.argsort(peak_mags)[::-1][:5]  # Top 5 peaks

        peaks_content = html.Div(
            [
                html.H6("Top Frequency Peaks"),
                html.Ul(
                    [
                        html.Li(f"{peak_freqs[i]:.2f} Hz: {peak_mags[i]:.2f}")
                        for i in sorted_idx
                    ]
                ),
            ]
        )
    else:
        peaks_content = html.Div("Enable 'Peak Detection' to see peak analysis")

    # Frequency bands
    if "frequency_bands" in (options or []):
        bands_content = create_frequency_bands_analysis(frequencies, power)
    else:
        bands_content = html.Div("Enable 'Frequency Bands' to see band analysis")

    return main_fig, analysis_fig, results, peaks_content, bands_content


def apply_stft_transform(
    time_data,
    signal_data,
    sampling_freq,
    options,
    window_size,
    overlap_percent,
    window_type,
):
    """Apply STFT transform and create spectrogram."""
    from scipy import signal as scipy_signal

    overlap = int(window_size * overlap_percent / 100)

    f, t, Zxx = scipy_signal.stft(
        signal_data,
        fs=sampling_freq,
        window=window_type,
        nperseg=window_size,
        noverlap=overlap,
    )

    # Create spectrogram
    main_fig = go.Figure(
        data=go.Heatmap(
            x=t,
            y=f,
            z=np.abs(Zxx),
            colorscale="Jet",
            colorbar=dict(title="Magnitude"),
        )
    )

    main_fig.update_layout(
        title="STFT Spectrogram",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        template="plotly_white",
    )

    # Additional plots: magnitude over time for specific frequency bands
    analysis_fig = go.Figure()

    # Low freq band (0-10 Hz)
    low_mask = (f >= 0) & (f <= 10)
    low_energy = np.sum(np.abs(Zxx[low_mask, :]), axis=0)

    # Mid freq band (10-30 Hz)
    mid_mask = (f > 10) & (f <= 30)
    mid_energy = np.sum(np.abs(Zxx[mid_mask, :]), axis=0)

    # High freq band (>30 Hz)
    high_mask = f > 30
    high_energy = np.sum(np.abs(Zxx[high_mask, :]), axis=0)

    analysis_fig.add_trace(go.Scatter(x=t, y=low_energy, name="0-10 Hz", mode="lines"))
    analysis_fig.add_trace(go.Scatter(x=t, y=mid_energy, name="10-30 Hz", mode="lines"))
    analysis_fig.add_trace(go.Scatter(x=t, y=high_energy, name=">30 Hz", mode="lines"))

    analysis_fig.update_layout(
        title="Energy in Frequency Bands Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Energy",
        template="plotly_white",
    )

    results = html.Div(
        [
            html.H6("STFT Analysis Results"),
            html.P(f"Window Size: {window_size} samples"),
            html.P(f"Overlap: {overlap_percent}%"),
            html.P(f"Time Resolution: {t[1] - t[0]:.4f} s"),
            html.P(f"Frequency Resolution: {f[1] - f[0]:.4f} Hz"),
        ]
    )

    peaks_content = html.Div("Peak detection not applicable for STFT")
    bands_content = html.Div("See main spectrogram for frequency content")

    return main_fig, analysis_fig, results, peaks_content, bands_content


def apply_wavelet_transform(
    time_data, signal_data, sampling_freq, options, wavelet_type, n_scales
):
    """Apply wavelet transform."""
    import pywt

    scales = np.arange(1, n_scales + 1)

    try:
        coefficients, frequencies = pywt.cwt(
            signal_data, scales, wavelet_type, sampling_period=1 / sampling_freq
        )
    except Exception as e:
        logger.error(f"Wavelet transform failed: {e}")
        # Fallback to simple wavelet
        coefficients, frequencies = pywt.cwt(
            signal_data, scales, "morl", sampling_period=1 / sampling_freq
        )

    # Create scalogram
    main_fig = go.Figure(
        data=go.Heatmap(
            x=time_data,
            y=frequencies,
            z=np.abs(coefficients),
            colorscale="Jet",
            colorbar=dict(title="Magnitude"),
        )
    )

    main_fig.update_layout(
        title=f"Wavelet Transform Scalogram ({wavelet_type})",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        template="plotly_white",
    )

    # Analysis plot: global wavelet spectrum
    global_spectrum = np.sum(np.abs(coefficients), axis=1)

    analysis_fig = go.Figure(
        go.Scatter(
            x=frequencies, y=global_spectrum, mode="lines", name="Global Spectrum"
        )
    )

    analysis_fig.update_layout(
        title="Global Wavelet Spectrum",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Average Power",
        template="plotly_white",
    )

    results = html.Div(
        [
            html.H6("Wavelet Transform Results"),
            html.P(f"Wavelet Type: {wavelet_type}"),
            html.P(f"Number of Scales: {n_scales}"),
            html.P(f"Frequency Range: {frequencies[0]:.2f} - {frequencies[-1]:.2f} Hz"),
        ]
    )

    peaks_content = html.Div("Peak detection not applicable for wavelet transform")
    bands_content = html.Div("See scalogram for frequency-time content")

    return main_fig, analysis_fig, results, peaks_content, bands_content


def apply_hilbert_transform(time_data, signal_data, sampling_freq, options):
    """Apply Hilbert transform to extract instantaneous features."""
    from scipy.signal import hilbert

    analytic_signal = hilbert(signal_data)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (
        np.diff(instantaneous_phase) / (2.0 * np.pi) * sampling_freq
    )

    # Main plot: amplitude envelope
    main_fig = go.Figure()

    main_fig.add_trace(
        go.Scatter(
            x=time_data,
            y=signal_data,
            mode="lines",
            name="Original Signal",
            line=dict(color="blue", width=1),
            opacity=0.5,
        )
    )

    main_fig.add_trace(
        go.Scatter(
            x=time_data,
            y=amplitude_envelope,
            mode="lines",
            name="Amplitude Envelope",
            line=dict(color="red", width=2),
        )
    )

    main_fig.update_layout(
        title="Hilbert Transform - Amplitude Envelope",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
    )

    # Analysis plots: phase and instantaneous frequency
    analysis_fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Instantaneous Phase", "Instantaneous Frequency"),
        vertical_spacing=0.15,
    )

    analysis_fig.add_trace(
        go.Scatter(
            x=time_data,
            y=instantaneous_phase,
            mode="lines",
            name="Phase",
            line=dict(color="green"),
        ),
        row=1,
        col=1,
    )

    analysis_fig.add_trace(
        go.Scatter(
            x=time_data[:-1],
            y=instantaneous_frequency,
            mode="lines",
            name="Frequency",
            line=dict(color="purple"),
        ),
        row=2,
        col=1,
    )

    analysis_fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    analysis_fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    analysis_fig.update_yaxes(title_text="Phase (rad)", row=1, col=1)
    analysis_fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)
    analysis_fig.update_layout(height=800, template="plotly_white", showlegend=False)

    results = html.Div(
        [
            html.H6("Hilbert Transform Results"),
            html.P(f"Mean Amplitude: {np.mean(amplitude_envelope):.4f}"),
            html.P(
                f"Mean Instantaneous Frequency: {np.mean(instantaneous_frequency):.2f} Hz"
            ),
            html.P(f"Frequency Std Dev: {np.std(instantaneous_frequency):.2f} Hz"),
        ]
    )

    peaks_content = html.Div("See amplitude envelope for peak analysis")
    bands_content = html.Div("See instantaneous frequency for time-varying spectrum")

    return main_fig, analysis_fig, results, peaks_content, bands_content


def apply_mfcc_transform(time_data, signal_data, sampling_freq, options, n_mfcc, n_fft):
    """Apply MFCC transform (approximation using scipy)."""
    from scipy.fftpack import dct
    from scipy import signal as scipy_signal

    # Compute STFT
    f, t, Zxx = scipy_signal.stft(
        signal_data, fs=sampling_freq, nperseg=n_fft, noverlap=n_fft // 2
    )

    # Get power spectrum
    power_spectrum = np.abs(Zxx) ** 2

    # Apply mel filterbank (simplified)
    mel_filterbank = create_mel_filterbank(len(f), sampling_freq, n_mfcc)
    mel_spectrum = np.dot(mel_filterbank, power_spectrum)

    # Apply DCT to get MFCCs
    mfccs = dct(np.log(mel_spectrum + 1e-10), type=2, axis=0, norm="ortho")[:n_mfcc]

    # Main plot: MFCC features over time
    main_fig = go.Figure(
        data=go.Heatmap(
            x=t,
            y=np.arange(n_mfcc),
            z=mfccs,
            colorscale="RdBu",
            colorbar=dict(title="Coefficient Value"),
        )
    )

    main_fig.update_layout(
        title="MFCC Coefficients Over Time",
        xaxis_title="Time (s)",
        yaxis_title="MFCC Coefficient",
        template="plotly_white",
    )

    # Analysis plot: average MFCC values
    avg_mfccs = np.mean(mfccs, axis=1)

    analysis_fig = go.Figure(
        go.Bar(x=np.arange(n_mfcc), y=avg_mfccs, name="Average MFCCs")
    )

    analysis_fig.update_layout(
        title="Average MFCC Values",
        xaxis_title="MFCC Coefficient",
        yaxis_title="Average Value",
        template="plotly_white",
    )

    results = html.Div(
        [
            html.H6("MFCC Analysis Results"),
            html.P(f"Number of Coefficients: {n_mfcc}"),
            html.P(f"FFT Size: {n_fft}"),
            html.P(f"Dominant Coefficient: C{np.argmax(np.abs(avg_mfccs))}"),
        ]
    )

    peaks_content = html.Div("MFCC analysis focuses on cepstral features")
    bands_content = html.Div("See main plot for MFCC temporal evolution")

    return main_fig, analysis_fig, results, peaks_content, bands_content


def create_mel_filterbank(n_fft_bins, sampling_rate, n_filters):
    """Create simplified mel filterbank."""
    mel_min = 0
    mel_max = 2595 * np.log10(1 + sampling_rate / 2 / 700)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    bin_points = np.floor((n_fft_bins + 1) * hz_points / sampling_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft_bins - 1)

    filterbank = np.zeros((n_filters, n_fft_bins))
    for i in range(1, n_filters + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]

        for j in range(left, center):
            if center != left:
                filterbank[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                filterbank[i - 1, j] = (right - j) / (right - center)

    return filterbank


def create_frequency_bands_analysis(frequencies, power):
    """Create frequency bands analysis display."""
    # Define standard frequency bands
    bands = {
        "VLF (0-0.04 Hz)": (0, 0.04),
        "LF (0.04-0.15 Hz)": (0.04, 0.15),
        "HF (0.15-0.4 Hz)": (0.15, 0.4),
        "VHF (0.4-1 Hz)": (0.4, 1.0),
        "Low (1-10 Hz)": (1.0, 10.0),
        "Mid (10-30 Hz)": (10.0, 30.0),
        "High (>30 Hz)": (30.0, np.inf),
    }

    band_powers = {}
    for band_name, (f_low, f_high) in bands.items():
        mask = (frequencies >= f_low) & (frequencies < f_high)
        if np.any(mask):
            band_powers[band_name] = np.sum(power[mask])
        else:
            band_powers[band_name] = 0

    # Create display
    total_power = sum(band_powers.values())

    rows = []
    for band_name, band_power in band_powers.items():
        percentage = (band_power / total_power * 100) if total_power > 0 else 0
        rows.append(
            html.Tr(
                [
                    html.Td(band_name),
                    html.Td(f"{band_power:.2e}"),
                    html.Td(f"{percentage:.1f}%"),
                ]
            )
        )

    return html.Div(
        [
            html.H6("Frequency Band Power Distribution"),
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [html.Th("Band"), html.Th("Power"), html.Th("% of Total")]
                        )
                    ),
                    html.Tbody(rows),
                ],
                bordered=True,
                hover=True,
                responsive=True,
                striped=True,
            ),
        ]
    )
