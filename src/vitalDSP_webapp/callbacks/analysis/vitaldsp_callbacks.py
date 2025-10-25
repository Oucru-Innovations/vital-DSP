"""
Analysis callbacks for vitalDSP core functionality.

This module handles core vitalDSP analysis callbacks including
frequency domain, and signal processing operations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html, dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from scipy import signal
import logging

# Import plot utilities for performance optimization
try:
    from vitalDSP_webapp.utils.plot_utils import limit_plot_data, check_plot_data_size
except ImportError:
    # Fallback if plot_utils not available
    def limit_plot_data(time_axis, signal_data, max_duration=300, max_points=10000, start_time=None):
        """Fallback implementation of limit_plot_data"""
        return time_axis, signal_data

    def check_plot_data_size(time_axis, signal_data):
        """Fallback implementation"""
        return True

# Import filtering functions from signal filtering callbacks
from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
    apply_traditional_filter,
)
from vitalDSP_webapp.callbacks.core.theme_callbacks import apply_plot_theme


# Helper function for formatting large numbers
def apply_filter(
    signal_data,
    sampling_freq,
    filter_family,
    filter_response,
    low_freq,
    high_freq,
    filter_order,
):
    """
    Apply filter to the signal using vitalDSP SignalFiltering class.

    This function now uses the validated vitalDSP SignalFiltering implementation
    for improved error handling, parameter validation, and consistency.

    Args:
        signal_data: Signal array
        sampling_freq: Sampling frequency in Hz
        filter_family: Filter type (butter, cheby1, cheby2, ellip, bessel)
        filter_response: Filter response (bandpass, lowpass, highpass, bandstop)
        low_freq: Low cutoff frequency in Hz
        high_freq: High cutoff frequency in Hz
        filter_order: Filter order

    Returns:
        np.ndarray: Filtered signal
    """
    try:
        from vitalDSP.filtering.signal_filtering import SignalFiltering

        # Create SignalFiltering instance
        sf = SignalFiltering(signal_data)

        # Determine cutoff and filter type
        if filter_response == "bandpass":
            cutoff = [low_freq, high_freq]
            filter_type = "band"
        elif filter_response == "lowpass":
            cutoff = high_freq
            filter_type = "low"
        elif filter_response == "highpass":
            cutoff = low_freq
            filter_type = "high"
        elif filter_response == "bandstop":
            cutoff = [low_freq, high_freq]
            filter_type = "bandstop"
        else:
            # Default to bandpass
            cutoff = [low_freq, high_freq]
            filter_type = "band"

        # Apply appropriate filter using vitalDSP
        if filter_family == "butter" or filter_family == "butterworth":
            return sf.butterworth(
                cutoff, fs=sampling_freq, order=filter_order, btype=filter_type
            )
        elif filter_family == "cheby1" or filter_family == "chebyshev1":
            return sf.chebyshev(
                cutoff,
                fs=sampling_freq,
                order=filter_order,
                btype=filter_type,
                ripple=0.5,
            )
        elif filter_family == "cheby2" or filter_family == "chebyshev2":
            # Use vitalDSP's Chebyshev Type II filter implementation
            return sf.chebyshev2(
                cutoff,
                fs=sampling_freq,
                order=filter_order,
                btype=filter_type,
                stopband_attenuation=40,
            )
        elif filter_family == "ellip" or filter_family == "elliptic":
            return sf.elliptic(
                cutoff,
                fs=sampling_freq,
                order=filter_order,
                btype=filter_type,
                ripple=0.5,
                stopband_attenuation=40,
            )
        elif filter_family == "bessel":
            # Use vitalDSP's Bessel filter implementation
            return sf.bessel(
                cutoff,
                fs=sampling_freq,
                order=filter_order,
                btype=filter_type,
            )
        else:
            # Default to Butterworth
            logger.warning(
                f"Unknown filter family '{filter_family}', defaulting to Butterworth"
            )
            return sf.butterworth(
                cutoff, fs=sampling_freq, order=filter_order, btype=filter_type
            )

    except Exception as e:
        logger.error(
            f"Error applying vitalDSP filter: {e}. Falling back to scipy implementation."
        )
        # Fallback to scipy if vitalDSP fails
        try:
            from scipy import signal as sp_signal

            nyquist = sampling_freq / 2
            low_freq_norm = low_freq / nyquist
            high_freq_norm = high_freq / nyquist

            if filter_response == "bandpass":
                btype = "band"
                cutoff = [low_freq_norm, high_freq_norm]
            elif filter_response == "lowpass":
                btype = "low"
                cutoff = high_freq_norm
            elif filter_response == "highpass":
                btype = "high"
                cutoff = low_freq_norm
            elif filter_response == "bandstop":
                btype = "bandstop"
                cutoff = [low_freq_norm, high_freq_norm]
            else:
                btype = "band"
                cutoff = [low_freq_norm, high_freq_norm]

            b, a = sp_signal.butter(filter_order, cutoff, btype=btype)
            return sp_signal.filtfilt(b, a, signal_data)
        except Exception as fallback_error:
            logger.error(f"Scipy fallback also failed: {fallback_error}")
            return signal_data


# def detect_peaks(signal_data, sampling_freq):
#     """Detect peaks in the signal."""
#     try:
#         # Calculate adaptive threshold
#         mean_val = np.mean(signal_data) if len(signal_data) > 0 else 0
#         std_val = np.std(signal_data) if len(signal_data) > 0 else 0
#         threshold = mean_val + 2 * std_val

#         # Find peaks with minimum distance constraint
#         min_distance = int(sampling_freq * 0.1)  # Minimum 0.1 seconds between peaks
#         peaks, _ = signal.find_peaks(
#             signal_data, height=threshold, distance=min_distance
#         )

#         return peaks

#     except Exception as e:
#         logger.error(f"Error detecting peaks: {e}")
#         return np.array([])



    @app.callback(
        [
            Output("fft-params", "style"),
            Output("psd-params", "style"),
            Output("stft-params", "style"),
            Output("wavelet-params", "style"),
        ],
        [Input("freq-analysis-type", "value")],
    )
    def toggle_frequency_params(analysis_type):
        """Show/hide parameter sections based on selected analysis type."""
        # Default style (hidden)
        hidden_style = {"display": "none"}
        visible_style = {"display": "block"}

        # Initialize all as hidden
        fft_style = hidden_style
        psd_style = hidden_style
        stft_style = hidden_style
        wavelet_style = hidden_style

        # Show relevant section based on analysis type
        if analysis_type == "fft":
            fft_style = visible_style
        elif analysis_type == "psd":
            psd_style = visible_style
        elif analysis_type == "stft":
            stft_style = visible_style
        elif analysis_type == "wavelet":
            wavelet_style = visible_style

        return fft_style, psd_style, stft_style, wavelet_style


def create_fft_plot(
    signal_data, sampling_freq, window_type, n_points, freq_min, freq_max
):
    """Create FFT plot using vitalDSP FourierTransform."""
    try:
        from vitalDSP.transforms.fourier_transform import FourierTransform

        # Apply window if specified
        if window_type and window_type != "none":
            if window_type == "hann":
                windowed_signal = signal_data * np.hanning(len(signal_data))
            elif window_type == "hamming":
                windowed_signal = signal_data * np.hamming(len(signal_data))
            elif window_type == "blackman":
                windowed_signal = signal_data * np.blackman(len(signal_data))
            else:
                windowed_signal = signal_data
        else:
            windowed_signal = signal_data

        # Use vitalDSP FourierTransform
        ft = FourierTransform(windowed_signal)
        fft_result = ft.compute_dft()

        # Compute frequency axis
        fft_freq = np.fft.fftfreq(len(fft_result), 1 / sampling_freq)

        # Get positive frequencies only
        positive_freq_mask = fft_freq > 0
        fft_freq = fft_freq[positive_freq_mask]
        fft_magnitude = np.abs(fft_result[positive_freq_mask])

        # Apply frequency range filter if specified
        if freq_min is not None and freq_max is not None:
            freq_mask = (fft_freq >= freq_min) & (fft_freq <= freq_max)
            fft_freq = fft_freq[freq_mask]
            fft_magnitude = fft_magnitude[freq_mask]

        # Create plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fft_freq,
                y=fft_magnitude,
                mode="lines",
                name="FFT Magnitude",
                line=dict(color="blue", width=2),
            )
        )

        fig.update_layout(
            title="Fast Fourier Transform (FFT)",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            showlegend=True,
            plot_bgcolor="white",
            template="plotly_white",
            # Add pan/zoom tools
            dragmode="pan",
            modebar=dict(
                add=[
                    "pan2d",
                    "zoom2d",
                    "select2d",
                    "lasso2d",
                    "zoomIn2d",
                    "zoomOut2d",
                    "autoScale2d",
                    "resetScale2d",
                ]
            ),
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating FFT plot: {e}")
        return create_empty_figure()


def create_stft_plot(
    signal_data, sampling_freq, window_size, hop_size, freq_min, freq_max
):
    """Create STFT plot using vitalDSP STFT."""
    try:
        from vitalDSP.transforms.stft import STFT

        # Use vitalDSP STFT
        stft_obj = STFT(signal_data, window_size=window_size, hop_size=hop_size)
        stft_result = stft_obj.compute_stft()

        # Get time and frequency axes
        time_axis = np.arange(stft_result.shape[1]) * hop_size / sampling_freq
        freq_axis = np.fft.rfftfreq(window_size, 1 / sampling_freq)

        # Apply frequency range filter if specified
        if freq_min is not None and freq_max is not None:
            freq_mask = (freq_axis >= freq_min) & (freq_axis <= freq_max)
            freq_axis = freq_axis[freq_mask]
            stft_result = stft_result[freq_mask, :]

        # Create plot
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=np.abs(stft_result),
                x=time_axis,
                y=freq_axis,
                colorscale="viridis",
                name="STFT Magnitude",
            )
        )

        fig.update_layout(
            title="Short-Time Fourier Transform (STFT)",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            showlegend=True,
            plot_bgcolor="white",
            template="plotly_white",
            # Add pan/zoom tools
            dragmode="pan",
            modebar=dict(
                add=[
                    "pan2d",
                    "zoom2d",
                    "select2d",
                    "lasso2d",
                    "zoomIn2d",
                    "zoomOut2d",
                    "autoScale2d",
                    "resetScale2d",
                ]
            ),
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating STFT plot: {e}")
        return create_empty_figure()


def create_wavelet_plot(
    signal_data, sampling_freq, wavelet_type, levels, freq_min, freq_max
):
    """Create wavelet plot using vitalDSP WaveletTransform."""
    try:
        from vitalDSP.transforms.wavelet_transform import WaveletTransform

        # Use vitalDSP WaveletTransform
        wt = WaveletTransform(signal_data, wavelet_name=wavelet_type)

        # Perform wavelet decomposition
        approximations = []
        details = []
        current_signal = signal_data.copy()

        for level in range(levels):
            approx, detail = wt._wavelet_decompose(current_signal)
            approximations.append(approx)
            details.append(detail)
            current_signal = approx

        # Create subplots for each level
        fig = make_subplots(
            rows=levels + 1,
            cols=1,
            subplot_titles=[f"Level {i+1} Detail" for i in range(levels)]
            + ["Final Approximation"],
            vertical_spacing=0.05,
        )

        # Add detail coefficients
        for i, detail in enumerate(details):
            time_axis = np.arange(len(detail)) / sampling_freq
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=detail,
                    mode="lines",
                    name=f"Level {i+1} Detail",
                    line=dict(width=1),
                ),
                row=i + 1,
                col=1,
            )

        # Add final approximation
        time_axis = np.arange(len(approximations[-1])) / sampling_freq
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=approximations[-1],
                mode="lines",
                name="Final Approximation",
                line=dict(color="red", width=2),
            ),
            row=levels + 1,
            col=1,
        )

        fig.update_layout(
            title=f"Wavelet Transform ({wavelet_type}) - {levels} Levels",
            height=300 * (levels + 1),
            showlegend=True,
            plot_bgcolor="white",
            template="plotly_white",
            # Add pan/zoom tools
            dragmode="pan",
            modebar=dict(
                add=[
                    "pan2d",
                    "zoom2d",
                    "select2d",
                    "lasso2d",
                    "zoomIn2d",
                    "zoomOut2d",
                    "autoScale2d",
                    "resetScale2d",
                ]
            ),
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating wavelet plot: {e}")
        return create_empty_figure()


def create_enhanced_psd_plot(
    signal_data,
    sampling_freq,
    window_sec,
    overlap,
    freq_max,
    log_scale,
    normalize,
    channel,
    column_mapping,
):
    """Create enhanced PSD plot using vitalDSP FrequencyDomainFeatures."""
    try:
        from vitalDSP.physiological_features.frequency_domain import (
            FrequencyDomainFeatures,
        )

        # Convert window from seconds to samples
        window_samples = int(window_sec * sampling_freq)

        # Use vitalDSP's SignalPowerAnalysis for PSD computation
        from vitalDSP.physiological_features.signal_power_analysis import SignalPowerAnalysis

        # Create SignalPowerAnalysis instance
        spa = SignalPowerAnalysis(signal_data)

        # Compute PSD using vitalDSP
        freqs, psd = spa.compute_psd(
            fs=sampling_freq,
            nperseg=min(window_samples, len(signal_data)),
            noverlap=int(min(window_samples, len(signal_data)) * overlap),
        )

        # Apply frequency range filter
        if freq_max:
            freq_mask = freqs <= freq_max
            freqs = freqs[freq_mask]
            psd = psd[freq_mask]

        # Apply log scale if requested
        if log_scale and "on" in log_scale:
            psd = 10 * np.log10(psd + 1e-10)  # Add small value to avoid log(0)
            y_label = "Power Spectral Density (dB/Hz)"
        else:
            y_label = "Power Spectral Density"

        # Normalize if requested
        if normalize and "on" in normalize:
            psd = psd / np.max(psd)
            y_label += " (Normalized)"

        # Create plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=psd,
                mode="lines",
                name="PSD",
                line=dict(color="green", width=2),
            )
        )

        fig.update_layout(
            title="Power Spectral Density (PSD)",
            xaxis_title="Frequency (Hz)",
            yaxis_title=y_label,
            showlegend=True,
            plot_bgcolor="white",
            template="plotly_white",
            # Add pan/zoom tools
            dragmode="pan",
            modebar=dict(
                add=[
                    "pan2d",
                    "zoom2d",
                    "select2d",
                    "lasso2d",
                    "zoomIn2d",
                    "zoomOut2d",
                    "autoScale2d",
                    "resetScale2d",
                ]
            ),
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating PSD plot: {e}")
        return create_empty_figure()


def create_enhanced_spectrogram_plot(
    signal_data,
    sampling_freq,
    window_size,
    overlap,
    freq_max,
    colormap,
    scaling,
    window_type,
):
    """Create enhanced spectrogram plot."""
    try:

        # Apply window function
        if window_type == "hann":
            window_func = np.hanning(window_size)
        elif window_type == "hamming":
            window_func = np.hamming(window_size)
        elif window_type == "blackman":
            window_func = np.blackman(window_size)
        elif window_type == "kaiser":
            window_func = np.kaiser(window_size, beta=14)
        elif window_type == "gaussian":
            window_func = np.exp(
                -0.5
                * ((np.arange(window_size) - window_size / 2) / (window_size / 6)) ** 2
            )
        else:
            window_func = np.ones(window_size)

        # Compute spectrogram using vitalDSP's STFT
        from vitalDSP.transforms.stft import STFT

        hop_size = window_size - int(window_size * overlap)
        stft_obj = STFT(signal_data, window_size=window_size, hop_size=hop_size, n_fft=window_size)
        stft_matrix = stft_obj.compute_stft()

        # Convert STFT to magnitude spectrogram
        Sxx = np.abs(stft_matrix) ** 2  # Power spectrogram

        # Create time and frequency axes
        times = np.arange(Sxx.shape[1]) * hop_size / sampling_freq
        freqs = np.fft.rfftfreq(window_size, 1/sampling_freq)

        # Apply frequency range filter
        if freq_max:
            freq_mask = freqs <= freq_max
            freqs = freqs[freq_mask]
            Sxx = Sxx[freq_mask, :]

        # Apply scaling
        if scaling == "density":
            Sxx = Sxx / sampling_freq
            title_suffix = " (Density)"
        else:
            title_suffix = " (Spectrum)"

        # Create plot
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=10 * np.log10(Sxx + 1e-10),  # Convert to dB
                x=times,
                y=freqs,
                colorscale=colormap,
                name="Spectrogram",
            )
        )

        fig.update_layout(
            title=f"Time-Frequency Spectrogram{title_suffix}",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            showlegend=True,
            plot_bgcolor="white",
            template="plotly_white",
            # Add pan/zoom tools
            dragmode="pan",
            modebar=dict(
                add=[
                    "pan2d",
                    "zoom2d",
                    "select2d",
                    "lasso2d",
                    "zoomIn2d",
                    "zoomOut2d",
                    "autoScale2d",
                    "resetScale2d",
                ]
            ),
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating spectrogram plot: {e}")
        return create_empty_figure()


def generate_frequency_analysis_results(
    signal_data, sampling_freq, analysis_type, analysis_options
):
    """Generate frequency analysis results summary."""
    try:
        results = []

        # Basic signal information
        results.append(html.H5("Signal Information"))
        results.append(html.P(f"Signal Length: {len(signal_data)} samples"))
        results.append(
            html.P(f"Duration: {len(signal_data) / sampling_freq:.2f} seconds")
        )
        results.append(html.P(f"Sampling Frequency: {sampling_freq} Hz"))
        results.append(html.P(f"Nyquist Frequency: {sampling_freq / 2:.2f} Hz"))

        # Analysis type specific results
        results.append(html.H5(f"Analysis Type: {analysis_type.upper()}"))

        if analysis_type == "fft":
            # FFT specific analysis
            fft_result = np.fft.fft(signal_data)
            fft_freq = np.fft.fftfreq(len(signal_data), 1 / sampling_freq)

            # Find dominant frequency
            positive_freq_mask = fft_freq > 0
            fft_freq_pos = fft_freq[positive_freq_mask]
            fft_magnitude_pos = np.abs(fft_result[positive_freq_mask])

            if len(fft_magnitude_pos) > 0:
                dominant_freq_idx = np.argmax(fft_magnitude_pos)
                dominant_freq = fft_freq_pos[dominant_freq_idx]
                results.append(html.P(f"Dominant Frequency: {dominant_freq:.2f} Hz"))
                results.append(
                    html.P(
                        f"Frequency Resolution: {fft_freq_pos[1] - fft_freq_pos[0]:.4f} Hz"
                    )
                )

        elif analysis_type == "psd":
            # PSD specific analysis
            from vitalDSP.physiological_features.signal_power_analysis import SignalPowerAnalysis

            spa = SignalPowerAnalysis(signal_data)
            freqs, psd = spa.compute_psd(fs=sampling_freq, nperseg=min(256, len(signal_data)//2))

            # Find peak frequency
            peak_freq_idx = np.argmax(psd)
            peak_freq = freqs[peak_freq_idx]
            results.append(html.P(f"Peak Frequency: {peak_freq:.2f} Hz"))
            results.append(html.P(f"Total Power: {np.sum(psd):.2e}"))

        elif analysis_type == "stft":
            # STFT specific analysis
            results.append(html.P("STFT provides time-frequency representation"))
            results.append(html.P("Check the spectrogram plot for detailed analysis"))

        elif analysis_type == "wavelet":
            # Wavelet specific analysis
            results.append(
                html.P("Wavelet transform provides multi-resolution analysis")
            )
            results.append(html.P("Check the wavelet plot for decomposition levels"))

        return html.Div(results)

    except Exception as e:
        logger.error(f"Error generating frequency analysis results: {e}")
        return html.Div(
            [html.H5("Error"), html.P(f"Failed to generate results: {str(e)}")]
        )


def create_frequency_peak_analysis_table(
    signal_data, sampling_freq, analysis_type, analysis_options, signal_type="unknown"
):
    """Create frequency peak analysis table."""
    try:

        if "peak_detection" not in analysis_options:
            return html.Div([html.P("Peak detection not selected")])

        # Perform peak detection on frequency domain
        if analysis_type == "fft":
            fft_result = np.fft.fft(signal_data)
            fft_freq = np.fft.fftfreq(len(signal_data), 1 / sampling_freq)

            # Get positive frequencies
            positive_freq_mask = fft_freq > 0
            fft_freq_pos = fft_freq[positive_freq_mask]
            fft_magnitude_pos = np.abs(fft_result[positive_freq_mask])

            # Find peaks
            # Use vitalDSP for ECG/PPG peak detection, scipy for others
            if signal_type and signal_type.lower() in ["ecg", "ppg"]:
                from vitalDSP.physiological_features.waveform import WaveformMorphology

                wm = WaveformMorphology(
                    fft_magnitude_pos, fs=sampling_freq, signal_type=signal_type.upper()
                )
                if signal_type.lower() == "ecg":
                    peaks = wm.r_peaks
                elif signal_type.lower() == "ppg":
                    peaks = wm.systolic_peaks
                properties = {}
            else:
                # Use scipy for other signal types
                from scipy.signal import find_peaks

                peaks, properties = find_peaks(
                    fft_magnitude_pos, height=np.max(fft_magnitude_pos) * 0.1
                )

            if len(peaks) > 0:
                # Create table
                table_data = []
                for i, peak_idx in enumerate(peaks[:10]):  # Limit to top 10 peaks
                    freq = fft_freq_pos[peak_idx]
                    magnitude = fft_magnitude_pos[peak_idx]
                    table_data.append(
                        [f"Peak {i+1}", f"{freq:.2f} Hz", f"{magnitude:.2e}"]
                    )

                return dbc.Table.from_dataframe(
                    pd.DataFrame(
                        table_data, columns=["Peak", "Frequency (Hz)", "Magnitude"]
                    ),
                    striped=True,
                    bordered=True,
                    hover=True,
                )
            else:
                return html.P("No significant peaks found")

        else:
            return html.P(f"Peak detection for {analysis_type} not implemented yet")

    except Exception as e:
        logger.error(f"Error creating peak analysis table: {e}")
        return html.Div(
            [html.H5("Error"), html.P(f"Failed to create peak table: {str(e)}")]
        )


def create_frequency_band_power_table(
    signal_data, sampling_freq, analysis_type, analysis_options
):
    """Create frequency band power analysis table."""
    try:
        if "band_power" not in analysis_options:
            return html.Div([html.P("Band power analysis not selected")])

        # Define frequency bands
        bands = {
            "Delta (0.5-4 Hz)": (0.5, 4),
            "Theta (4-8 Hz)": (4, 8),
            "Alpha (8-13 Hz)": (8, 13),
            "Beta (13-30 Hz)": (13, 30),
            "Gamma (30-100 Hz)": (30, 100),
        }

        # Compute PSD using vitalDSP's SignalPowerAnalysis
        from vitalDSP.physiological_features.signal_power_analysis import SignalPowerAnalysis

        spa = SignalPowerAnalysis(signal_data)
        freqs, psd = spa.compute_psd(fs=sampling_freq, nperseg=min(256, len(signal_data) // 2))

        # Calculate band powers
        band_powers = {}
        for band_name, (low_freq, high_freq) in bands.items():
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(freq_mask):
                power = np.trapz(psd[freq_mask], freqs[freq_mask])
                band_powers[band_name] = power

        # Create table
        table_data = []
        for band_name, power in band_powers.items():
            table_data.append([band_name, f"{power:.2e}"])

        return dbc.Table.from_dataframe(
            pd.DataFrame(table_data, columns=["Frequency Band", "Power"]),
            striped=True,
            bordered=True,
            hover=True,
        )

    except Exception as e:
        logger.error(f"Error creating band power table: {e}")
        return html.Div(
            [html.H5("Error"), html.P(f"Failed to create band power table: {str(e)}")]
        )


def create_frequency_stability_table(
    signal_data, sampling_freq, analysis_type, analysis_options
):
    """Create frequency stability analysis table."""
    try:
        if "stability" not in analysis_options:
            return html.Div([html.P("Stability analysis not selected")])

        # Basic stability metrics
        mean_value = np.mean(signal_data)
        std_value = np.std(signal_data)
        cv = std_value / abs(mean_value) if mean_value != 0 else float("inf")

        # Frequency stability (using FFT)
        fft_result = np.fft.fft(signal_data)
        fft_freq = np.fft.fftfreq(len(signal_data), 1 / sampling_freq)

        # Get positive frequencies
        positive_freq_mask = fft_freq > 0
        fft_freq_pos = fft_freq[positive_freq_mask]
        fft_magnitude_pos = np.abs(fft_result[positive_freq_mask])

        # Find dominant frequency
        if len(fft_magnitude_pos) > 0:
            dominant_freq_idx = np.argmax(fft_magnitude_pos)
            dominant_freq = fft_freq_pos[dominant_freq_idx]
        else:
            dominant_freq = 0

        # Create table
        table_data = [
            ["Mean Value", f"{mean_value:.4f}"],
            ["Standard Deviation", f"{std_value:.4f}"],
            ["Coefficient of Variation", f"{cv:.4f}"],
            ["Dominant Frequency", f"{dominant_freq:.2f} Hz"],
            ["Signal Length", f"{len(signal_data)} samples"],
        ]

        return dbc.Table.from_dataframe(
            pd.DataFrame(table_data, columns=["Metric", "Value"]),
            striped=True,
            bordered=True,
            hover=True,
        )

    except Exception as e:
        logger.error(f"Error creating stability table: {e}")
        return html.Div(
            [html.H5("Error"), html.P(f"Failed to create stability table: {str(e)}")]
        )


def create_frequency_harmonics_table(
    signal_data, sampling_freq, analysis_type, analysis_options
):
    """Create frequency harmonics analysis table."""
    try:
        if "harmonic_analysis" not in analysis_options:
            return html.Div([html.P("Harmonic analysis not selected")])

        # Perform FFT for harmonic analysis
        fft_result = np.fft.fft(signal_data)
        fft_freq = np.fft.fftfreq(len(signal_data), 1 / sampling_freq)

        # Get positive frequencies
        positive_freq_mask = fft_freq > 0
        fft_freq_pos = fft_freq[positive_freq_mask]
        fft_magnitude_pos = np.abs(fft_result[positive_freq_mask])

        if len(fft_magnitude_pos) > 0:
            # Find fundamental frequency (highest peak)
            # Use vitalDSP for ECG/PPG peak detection, scipy for others
            if signal_type and signal_type.lower() in ["ecg", "ppg"]:  # noqa: F821
                from vitalDSP.physiological_features.waveform import WaveformMorphology

                wm = WaveformMorphology(
                    fft_magnitude_pos, fs=sampling_freq, signal_type=signal_type.upper()  # noqa: F821
                )
                if signal_type.lower() == "ecg":  # noqa: F821
                    peaks = wm.r_peaks
                elif signal_type.lower() == "ppg":  # noqa: F821
                    peaks = wm.systolic_peaks
                properties = {}
            else:
                # Use scipy for other signal types
                from scipy.signal import find_peaks

                peaks, properties = find_peaks(
                    fft_magnitude_pos, height=np.max(fft_magnitude_pos) * 0.1
                )

            if len(peaks) > 0:
                fundamental_freq = fft_freq_pos[peaks[0]]

                # Find harmonics (multiples of fundamental)
                harmonics = []
                for i in range(1, 6):  # Look for up to 5th harmonic
                    harmonic_freq = fundamental_freq * i

                    # Find closest frequency in our data
                    freq_idx = np.argmin(np.abs(fft_freq_pos - harmonic_freq))
                    if freq_idx < len(fft_magnitude_pos):
                        harmonic_magnitude = fft_magnitude_pos[freq_idx]
                        harmonics.append(
                            [
                                f"{i}st",
                                f"{harmonic_freq:.2f} Hz",
                                f"{harmonic_magnitude:.2e}",
                            ]
                        )

                # Create table
                table_data = [
                    [
                        "Fundamental",
                        f"{fundamental_freq:.2f} Hz",
                        f"{fft_magnitude_pos[peaks[0]]:.2e}",
                    ]
                ]
                table_data.extend(harmonics)

                return dbc.Table.from_dataframe(
                    pd.DataFrame(
                        table_data, columns=["Harmonic", "Frequency (Hz)", "Magnitude"]
                    ),
                    striped=True,
                    bordered=True,
                    hover=True,
                )
            else:
                return html.P("No significant peaks found for harmonic analysis")
        else:
            return html.P("Insufficient data for harmonic analysis")

    except Exception as e:
        logger.error(f"Error creating harmonics table: {e}")
        return html.Div(
            [html.H5("Error"), html.P(f"Failed to create harmonics table: {str(e)}")]
        )


def register_vitaldsp_callbacks(app):
    """
    Register vitalDSP core callbacks.
    
    Note: Time domain callbacks have been migrated to time_domain_callbacks.py.
    This file now only contains helper functions for frequency domain analysis.
    No callbacks are registered here anymore.
    """
    # Time domain callbacks moved to register_time_domain_callbacks()
    # This function is kept for compatibility but registers nothing
    logger = logging.getLogger(__name__)
    logger.info("register_vitaldsp_callbacks called - no callbacks to register (migrated to time_domain_callbacks)")
    pass
