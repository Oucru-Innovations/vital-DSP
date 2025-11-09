"""
Transform Functions (Refactored Version)
Contains all signal transform implementation functions using vitalDSP library.

This is a refactored version that replaces custom scipy implementations with
vitalDSP transform modules for better consistency, maintainability, and access
to the full vitalDSP feature set.

Author: vitalDSP Team
Date: November 3, 2025
Version: 2.0.0 (Refactored)

Changes from v1.0:
- Replaced custom scipy FFT with vitalDSP.transforms.fourier_transform.FourierTransform
- Replaced custom scipy STFT with vitalDSP.transforms.stft.STFT
- Replaced custom pywt Wavelet with vitalDSP.transforms.wavelet_transform.WaveletTransform
- Replaced custom scipy Hilbert with vitalDSP.transforms.hilbert_transform.HilbertTransform
- Updated MFCC to use vitalDSP.transforms.mfcc.MFCC
- Maintained same interface for backward compatibility with callbacks
"""

import logging
import numpy as np
from dash import html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

# Import vitalDSP transform modules
from vitalDSP.transforms.fourier_transform import FourierTransform
from vitalDSP.transforms.stft import STFT
from vitalDSP.transforms.wavelet_transform import WaveletTransform
from vitalDSP.transforms.hilbert_transform import HilbertTransform
from vitalDSP.transforms.mfcc import MFCC

logger = logging.getLogger(__name__)


def apply_fft_transform(
    time_data, signal_data, sampling_freq, options, window_type, n_points
):
    """
    Apply FFT transform and create visualizations using vitalDSP.

    This function uses vitalDSP.transforms.fourier_transform.FourierTransform
    instead of custom scipy implementation.

    Parameters
    ----------
    time_data : numpy.ndarray
        Time axis data
    signal_data : numpy.ndarray
        Input signal data
    sampling_freq : float
        Sampling frequency in Hz
    options : list
        Analysis options (e.g., ['log_scale', 'phase', 'power', 'peak_detection', 'frequency_bands'])
    window_type : str
        Window type ('boxcar', 'hamming', 'hann', 'blackman', 'kaiser')
    n_points : int
        Number of FFT points

    Returns
    -------
    tuple
        (main_fig, analysis_fig, results, peaks_content, bands_content)
    """
    try:
        logger.info(
            f"[vitalDSP FFT] Computing FFT with window={window_type}, n_points={n_points}"
        )

        # Apply window (vitalDSP FourierTransform applies Hamming by default, so we need to apply our own window first)
        if window_type == "boxcar":
            window = np.ones(len(signal_data))
        elif window_type == "hamming":
            window = np.hamming(len(signal_data))
        elif window_type == "hann":
            window = np.hanning(len(signal_data))
        elif window_type == "blackman":
            window = np.blackman(len(signal_data))
        elif window_type == "kaiser":
            window = np.kaiser(len(signal_data), beta=5)
        else:
            window = np.hanning(len(signal_data))

        windowed_signal = signal_data * window

        # Use vitalDSP FourierTransform
        ft = FourierTransform(windowed_signal)
        fft_values = ft.compute_dft()

        # Generate frequency axis
        n = min(n_points if n_points else len(windowed_signal), len(windowed_signal))
        frequencies = np.fft.fftfreq(n, d=1 / sampling_freq)

        # Only positive frequencies
        positive_mask = frequencies >= 0
        frequencies = frequencies[positive_mask]
        fft_values = fft_values[positive_mask]

        # Calculate magnitude, phase, power
        magnitude = np.abs(fft_values)
        phase = np.angle(fft_values)
        power = magnitude**2

        logger.info(f"[vitalDSP FFT] ✓ FFT computed: {len(frequencies)} frequency bins")

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
            title="FFT - Magnitude Spectrum (vitalDSP)",
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
        analysis_fig.update_layout(
            height=800, template="plotly_white", showlegend=False
        )

        # Results summary
        dominant_freq_idx = np.argmax(magnitude[1:]) + 1  # Skip DC component
        dominant_freq = frequencies[dominant_freq_idx]
        dominant_magnitude = magnitude[dominant_freq_idx]

        results = html.Div(
            [
                html.H6("FFT Analysis Results (vitalDSP)"),
                html.P(f"Dominant Frequency: {dominant_freq:.2f} Hz"),
                html.P(f"Dominant Magnitude: {dominant_magnitude:.2f}"),
                html.P(f"Total Power: {np.sum(power):.2e}"),
                html.P(
                    f"Frequency Resolution: {frequencies[1] - frequencies[0]:.4f} Hz"
                ),
                html.P(
                    "✓ Using vitalDSP.transforms.fourier_transform.FourierTransform",
                    className="text-success small",
                ),
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

    except Exception as e:
        logger.error(f"[vitalDSP FFT] Error in FFT transform: {e}", exc_info=True)
        # Return empty figures on error
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5
        )
        error_msg = html.Div(f"FFT Error: {str(e)}", className="text-danger")
        return empty_fig, empty_fig, error_msg, html.Div(), html.Div()


def apply_stft_transform(
    time_data,
    signal_data,
    sampling_freq,
    options,
    window_size,
    overlap_percent,
    window_type,
):
    """
    Apply STFT transform and create spectrogram using vitalDSP.

    This function uses vitalDSP.transforms.stft.STFT instead of custom scipy implementation.

    Parameters
    ----------
    time_data : numpy.ndarray
        Time axis data
    signal_data : numpy.ndarray
        Input signal data
    sampling_freq : float
        Sampling frequency in Hz
    options : list
        Analysis options
    window_size : int
        Window size in samples
    overlap_percent : float
        Overlap percentage (0-100)
    window_type : str
        Window type (note: vitalDSP uses Hanning by default)

    Returns
    -------
    tuple
        (main_fig, analysis_fig, results, peaks_content, bands_content)
    """
    try:
        logger.info(
            f"[vitalDSP STFT] Computing STFT with window_size={window_size}, overlap={overlap_percent}%"
        )

        # Calculate hop size from overlap
        overlap_samples = int(window_size * overlap_percent / 100)
        hop_size = window_size - overlap_samples

        # Use vitalDSP STFT
        stft = STFT(
            signal=signal_data,
            window_size=window_size,
            hop_size=hop_size,
            n_fft=window_size,  # Use window_size as n_fft for consistency
        )

        Zxx = stft.compute_stft()  # Returns complex 2D array (freq_bins x time_frames)

        # Generate frequency and time axes
        f = np.linspace(0, sampling_freq / 2, Zxx.shape[0])
        t = np.arange(Zxx.shape[1]) * hop_size / sampling_freq

        logger.info(
            f"[vitalDSP STFT] ✓ STFT computed: {Zxx.shape[0]} freq bins x {Zxx.shape[1]} time frames"
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
            title="STFT Spectrogram (vitalDSP)",
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

        analysis_fig.add_trace(
            go.Scatter(x=t, y=low_energy, name="0-10 Hz", mode="lines")
        )
        analysis_fig.add_trace(
            go.Scatter(x=t, y=mid_energy, name="10-30 Hz", mode="lines")
        )
        analysis_fig.add_trace(
            go.Scatter(x=t, y=high_energy, name=">30 Hz", mode="lines")
        )

        analysis_fig.update_layout(
            title="Energy in Frequency Bands Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Energy",
            template="plotly_white",
        )

        results = html.Div(
            [
                html.H6("STFT Analysis Results (vitalDSP)"),
                html.P(f"Window Size: {window_size} samples"),
                html.P(f"Overlap: {overlap_percent}%"),
                html.P(
                    f"Time Resolution: {t[1] - t[0]:.4f} s" if len(t) > 1 else "N/A"
                ),
                html.P(
                    f"Frequency Resolution: {f[1] - f[0]:.4f} Hz"
                    if len(f) > 1
                    else "N/A"
                ),
                html.P(
                    "✓ Using vitalDSP.transforms.stft.STFT",
                    className="text-success small",
                ),
            ]
        )

        peaks_content = html.Div("Peak detection not applicable for STFT")
        bands_content = html.Div("See main spectrogram for frequency content")

        return main_fig, analysis_fig, results, peaks_content, bands_content

    except Exception as e:
        logger.error(f"[vitalDSP STFT] Error in STFT transform: {e}", exc_info=True)
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5
        )
        error_msg = html.Div(f"STFT Error: {str(e)}", className="text-danger")
        return empty_fig, empty_fig, error_msg, html.Div(), html.Div()


def apply_wavelet_transform(
    time_data, signal_data, sampling_freq, options, wavelet_type, n_scales
):
    """
    Apply wavelet transform using vitalDSP.

    This function uses vitalDSP.transforms.wavelet_transform.WaveletTransform
    instead of custom pywt implementation.

    Note: vitalDSP uses DWT (Discrete Wavelet Transform), which is different from
    pywt's CWT (Continuous Wavelet Transform). The visualization may differ slightly.

    Parameters
    ----------
    time_data : numpy.ndarray
        Time axis data
    signal_data : numpy.ndarray
        Input signal data
    sampling_freq : float
        Sampling frequency in Hz
    options : list
        Analysis options
    wavelet_type : str
        Wavelet type (e.g., 'haar', 'db4', 'morl', 'mexh')
    n_scales : int
        Number of scales (for DWT, this determines decomposition levels)

    Returns
    -------
    tuple
        (main_fig, analysis_fig, results, peaks_content, bands_content)
    """
    try:
        logger.info(
            f"[vitalDSP Wavelet] Computing wavelet transform with wavelet={wavelet_type}"
        )

        # Use vitalDSP WaveletTransform
        wt = WaveletTransform(
            signal=signal_data, wavelet_name=wavelet_type, same_length=True
        )
        coefficients = wt.perform_wavelet_transform()

        # vitalDSP returns list of arrays (approximation + details at each level)
        # We need to create a scalogram-like visualization

        # For visualization, we'll create a pseudo-scalogram from DWT coefficients
        # This is an approximation since DWT doesn't directly give us a time-frequency plot like CWT

        logger.info(
            f"[vitalDSP Wavelet] ✓ Wavelet transform computed: {len(coefficients)} decomposition levels"
        )

        # Create a simplified visualization showing coefficients at each level
        fig_data = []
        for i, coef in enumerate(coefficients):
            level_name = "Approximation" if i == 0 else f"Detail Level {i}"
            # Resample coefficients to match time axis length for visualization
            time_axis_level = np.linspace(0, len(time_data) / sampling_freq, len(coef))
            fig_data.append((level_name, time_axis_level, coef))

        # Create main plot showing all decomposition levels
        main_fig = make_subplots(
            rows=len(fig_data),
            cols=1,
            subplot_titles=[f"{name}" for name, _, _ in fig_data],
            vertical_spacing=0.05,
        )

        for idx, (name, t, coef) in enumerate(fig_data, start=1):
            main_fig.add_trace(
                go.Scatter(
                    x=t,
                    y=coef,
                    mode="lines",
                    name=name,
                    line=dict(width=1),
                ),
                row=idx,
                col=1,
            )

        main_fig.update_layout(
            title=f"Wavelet Transform Decomposition ({wavelet_type}) - vitalDSP DWT",
            height=200 * len(fig_data),
            showlegend=False,
            template="plotly_white",
        )

        # Analysis plot: Energy at each level
        energies = [np.sum(coef**2) for coef in coefficients]
        level_names = ["Approx"] + [f"D{i}" for i in range(1, len(coefficients))]

        analysis_fig = go.Figure(
            go.Bar(x=level_names, y=energies, name="Energy per Level")
        )

        analysis_fig.update_layout(
            title="Energy Distribution Across Decomposition Levels",
            xaxis_title="Decomposition Level",
            yaxis_title="Energy",
            template="plotly_white",
        )

        results = html.Div(
            [
                html.H6("Wavelet Transform Results (vitalDSP DWT)"),
                html.P(f"Wavelet Type: {wavelet_type}"),
                html.P(f"Decomposition Levels: {len(coefficients) - 1}"),
                html.P(f"Total Energy: {sum(energies):.2e}"),
                html.P(
                    "✓ Using vitalDSP.transforms.wavelet_transform.WaveletTransform",
                    className="text-success small",
                ),
                html.P(
                    "Note: Using DWT (Discrete Wavelet Transform)",
                    className="text-info small",
                ),
            ]
        )

        peaks_content = html.Div("Peak detection not applicable for wavelet transform")
        bands_content = html.Div("See energy distribution for frequency-level content")

        return main_fig, analysis_fig, results, peaks_content, bands_content

    except Exception as e:
        logger.error(
            f"[vitalDSP Wavelet] Error in wavelet transform: {e}", exc_info=True
        )
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5
        )
        error_msg = html.Div(f"Wavelet Error: {str(e)}", className="text-danger")
        return empty_fig, empty_fig, error_msg, html.Div(), html.Div()


def apply_hilbert_transform(time_data, signal_data, sampling_freq, options):
    """
    Apply Hilbert transform to extract instantaneous features using vitalDSP.

    This function uses vitalDSP.transforms.hilbert_transform.HilbertTransform
    instead of custom scipy implementation.

    Parameters
    ----------
    time_data : numpy.ndarray
        Time axis data
    signal_data : numpy.ndarray
        Input signal data
    sampling_freq : float
        Sampling frequency in Hz
    options : list
        Analysis options

    Returns
    -------
    tuple
        (main_fig, analysis_fig, results, peaks_content, bands_content)
    """
    try:
        logger.info("[vitalDSP Hilbert] Computing Hilbert transform")

        # Use vitalDSP HilbertTransform
        ht = HilbertTransform(signal=signal_data)

        analytic_signal = ht.compute_hilbert()
        amplitude_envelope = ht.envelope()
        instantaneous_phase = ht.instantaneous_phase()

        # Compute instantaneous frequency
        instantaneous_frequency = (
            np.diff(np.unwrap(instantaneous_phase)) / (2.0 * np.pi) * sampling_freq
        )

        logger.info(f"[vitalDSP Hilbert] ✓ Hilbert transform computed")

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
            title="Hilbert Transform - Amplitude Envelope (vitalDSP)",
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
        analysis_fig.update_layout(
            height=800, template="plotly_white", showlegend=False
        )

        results = html.Div(
            [
                html.H6("Hilbert Transform Results (vitalDSP)"),
                html.P(f"Mean Amplitude: {np.mean(amplitude_envelope):.4f}"),
                html.P(
                    f"Mean Instantaneous Frequency: {np.mean(instantaneous_frequency):.2f} Hz"
                ),
                html.P(f"Frequency Std Dev: {np.std(instantaneous_frequency):.2f} Hz"),
                html.P(
                    "✓ Using vitalDSP.transforms.hilbert_transform.HilbertTransform",
                    className="text-success small",
                ),
            ]
        )

        peaks_content = html.Div("See amplitude envelope for peak analysis")
        bands_content = html.Div(
            "See instantaneous frequency for time-varying spectrum"
        )

        return main_fig, analysis_fig, results, peaks_content, bands_content

    except Exception as e:
        logger.error(
            f"[vitalDSP Hilbert] Error in Hilbert transform: {e}", exc_info=True
        )
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5
        )
        error_msg = html.Div(f"Hilbert Error: {str(e)}", className="text-danger")
        return empty_fig, empty_fig, error_msg, html.Div(), html.Div()


def apply_mfcc_transform(time_data, signal_data, sampling_freq, options, n_mfcc, n_fft):
    """
    Apply MFCC transform using vitalDSP.

    This function uses vitalDSP.transforms.mfcc.MFCC instead of custom implementation.

    Note: MFCC is primarily for audio/speech processing and may not be ideal for
    physiological signals. Consider removing this if not needed.

    Parameters
    ----------
    time_data : numpy.ndarray
        Time axis data
    signal_data : numpy.ndarray
        Input signal data
    sampling_freq : float
        Sampling frequency in Hz
    options : list
        Analysis options
    n_mfcc : int
        Number of MFCC coefficients
    n_fft : int
        FFT size

    Returns
    -------
    tuple
        (main_fig, analysis_fig, results, peaks_content, bands_content)
    """
    try:
        logger.info(f"[vitalDSP MFCC] Computing MFCC with n_coefficients={n_mfcc}")

        # Use vitalDSP MFCC
        mfcc = MFCC(
            signal=signal_data,
            sample_rate=int(sampling_freq),
            num_filters=40,
            num_coefficients=n_mfcc,
        )

        mfccs = mfcc.compute_mfcc()  # Returns 2D array (num_frames x num_coefficients)

        logger.info(f"[vitalDSP MFCC] ✓ MFCC computed: {mfccs.shape}")

        # Generate time axis for MFCC frames
        frame_stride = 0.01  # 10 ms (from MFCC implementation)
        t = np.arange(mfccs.shape[0]) * frame_stride

        # Main plot: MFCC features over time
        main_fig = go.Figure(
            data=go.Heatmap(
                x=t,
                y=np.arange(n_mfcc),
                z=mfccs.T,  # Transpose to match expected format
                colorscale="RdBu",
                colorbar=dict(title="Coefficient Value"),
            )
        )

        main_fig.update_layout(
            title="MFCC Coefficients Over Time (vitalDSP)",
            xaxis_title="Time (s)",
            yaxis_title="MFCC Coefficient",
            template="plotly_white",
        )

        # Analysis plot: average MFCC values
        avg_mfccs = np.mean(mfccs, axis=0)

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
                html.H6("MFCC Analysis Results (vitalDSP)"),
                html.P(f"Number of Coefficients: {n_mfcc}"),
                html.P(f"Number of Frames: {mfccs.shape[0]}"),
                html.P(f"Dominant Coefficient: C{np.argmax(np.abs(avg_mfccs))}"),
                html.P(
                    "✓ Using vitalDSP.transforms.mfcc.MFCC",
                    className="text-success small",
                ),
                html.P(
                    "Note: MFCC is primarily for audio/speech processing",
                    className="text-warning small",
                ),
            ]
        )

        peaks_content = html.Div("MFCC analysis focuses on cepstral features")
        bands_content = html.Div("See main plot for MFCC temporal evolution")

        return main_fig, analysis_fig, results, peaks_content, bands_content

    except Exception as e:
        logger.error(f"[vitalDSP MFCC] Error in MFCC transform: {e}", exc_info=True)
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5
        )
        error_msg = html.Div(f"MFCC Error: {str(e)}", className="text-danger")
        return empty_fig, empty_fig, error_msg, html.Div(), html.Div()


def create_frequency_bands_analysis(frequencies, power):
    """
    Create frequency bands analysis display.

    This is a helper function that remains unchanged as it's used for display purposes.
    """
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
