"""
Frequency domain analysis and advanced filtering callbacks for vitalDSP webapp.
Handles FFT, STFT, wavelet transforms, and advanced filtering techniques.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html
from dash.exceptions import PreventUpdate
from scipy import signal
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import dash_bootstrap_components as dbc
import logging

logger = logging.getLogger(__name__)


def register_frequency_filtering_callbacks(app):
    """Register all frequency domain analysis and filtering callbacks."""
    
    # Frequency Domain Analysis Callback
    @app.callback(
        [Output("freq-main-plot", "figure"),
         Output("freq-psd-plot", "figure"),
         Output("freq-spectrogram-plot", "figure"),
         Output("freq-analysis-results", "children"),
         Output("freq-peak-analysis-table", "children"),
         Output("freq-band-power-table", "children"),
         Output("freq-stability-table", "children"),
         Output("freq-harmonics-table", "children"),
         Output("store-frequency-data", "data"),
         Output("store-time-freq-data", "data")],
        [Input("url", "pathname"),
         Input("freq-btn-update-analysis", "n_clicks"),
         Input("freq-time-range-slider", "value"),
         Input("freq-btn-nudge-m10", "n_clicks"),
         Input("freq-btn-nudge-m1", "n_clicks"),
         Input("freq-btn-nudge-p1", "n_clicks"),
         Input("freq-btn-nudge-p10", "n_clicks")],
        [State("freq-start-time", "value"),
         State("freq-end-time", "value"),
         State("freq-analysis-type", "value"),
         State("fft-window-type", "value"),
         State("fft-n-points", "value"),
         # PSD Parameters
         State("psd-window", "value"),
         State("psd-overlap", "value"),
         State("psd-freq-max", "value"),
         State("psd-log-scale", "value"),
         State("psd-normalize", "value"),
         State("psd-channel", "value"),
         # STFT Parameters
         State("stft-window-size", "value"),
         State("stft-hop-size", "value"),
         State("stft-window-type", "value"),
         State("stft-overlap", "value"),
         State("stft-scaling", "value"),
         State("stft-freq-max", "value"),
         State("stft-colormap", "value"),
         State("wavelet-type", "value"),
         State("wavelet-levels", "value"),
         State("freq-min", "value"),
         State("freq-max", "value"),
         State("freq-analysis-options", "value")]
    )
    def frequency_domain_callback(pathname, n_clicks, slider_value, nudge_m10, nudge_m1, nudge_p1, nudge_p10,
                                start_time, end_time, analysis_type, fft_window, fft_n_points,
                                # PSD parameters
                                psd_window, psd_overlap, psd_freq_max, psd_log_scale, psd_normalize, psd_channel,
                                # STFT parameters
                                stft_window_size, stft_hop_size, stft_window_type, stft_overlap, stft_scaling,
                                stft_freq_max, stft_colormap, wavelet_type, wavelet_levels, freq_min, freq_max, analysis_options):
        """Unified callback for frequency domain analysis."""
        ctx = callback_context
        
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        logger.info(f"=== FREQUENCY DOMAIN CALLBACK TRIGGERED ===")
        logger.info(f"Trigger ID: {trigger_id}")
        logger.info(f"Pathname: {pathname}")
        
        # Only run this when we're on the frequency domain page
        if pathname != "/frequency":
            logger.info("Not on frequency page, returning empty figures")
            empty_figs = [create_empty_figure() for _ in range(3)]
            return empty_figs + ["Navigate to Frequency page", create_empty_figure(), None, None, None, None, None]
        
        # If this is the first time loading the page (no button clicks), show a message
        if not ctx.triggered or ctx.triggered[0]["prop_id"].split(".")[0] == "url":
            logger.info("First time loading frequency page, attempting to load data")
        
        try:
            # Get data from the data service
            logger.info("Attempting to import data service...")
            from vitalDSP_webapp.services.data.data_service import get_data_service
            logger.info("Data service imported successfully")
            
            data_service = get_data_service()
            data_store = data_service.get_all_data()
            
            if not data_store:
                logger.warning("No data available for frequency analysis")
                empty_figs = [create_empty_figure() for _ in range(3)]
                return empty_figs + ["No data available. Please upload data first.", create_empty_figure(), None, None, None, None, None]
            
            # Get the first available data
            data_id = list(data_store.keys())[0]
            df = data_service.get_data(data_id)
            column_mapping = data_service.get_column_mapping(data_id)
            data_info = data_service.get_data_info(data_id)
            
            if df is None or df.empty:
                logger.warning("Data is empty or None")
                empty_figs = [create_empty_figure() for _ in range(3)]
                return empty_figs + ["Data is empty. Please upload valid data.", create_empty_figure(), None, None, None, None, None]
            
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            logger.info(f"Column mapping: {column_mapping}")
            logger.info(f"Data info: {data_info}")
            
            # Extract time and signal columns
            time_col = column_mapping.get('time', df.columns[0])
            signal_col = column_mapping.get('signal', df.columns[1])
            
            # Get sampling frequency
            sampling_freq = data_info.get('sampling_freq', 1000)  # Default to 1000 Hz
            
            # Extract time and signal data
            time_data = df[time_col].values
            signal_data = df[signal_col].values
            
            # Handle time window
            if start_time is not None and end_time is not None:
                start_idx = np.searchsorted(time_data, start_time)
                end_idx = np.searchsorted(time_data, end_time)
                if end_idx > start_idx:
                    time_data = time_data[start_idx:end_idx]
                    signal_data = signal_data[start_idx:end_idx]
            
            # Set default values
            analysis_type = analysis_type or "FFT"
            fft_window = fft_window or "hann"
            fft_n_points = fft_n_points or len(signal_data)
            
            # Perform frequency domain analysis based on type
            if analysis_type == "FFT":
                main_fig, psd_fig, spectrogram_fig, results, peak_table, band_table, stability_table, harmonics_table, freq_data, time_freq_data = perform_fft_analysis(
                    time_data, signal_data, sampling_freq, fft_window, fft_n_points, freq_min, freq_max, analysis_options
                )
            elif analysis_type == "PSD":
                main_fig, psd_fig, spectrogram_fig, results, peak_table, band_table, stability_table, harmonics_table, freq_data, time_freq_data = perform_psd_analysis(
                    time_data, signal_data, sampling_freq, psd_window, psd_overlap, psd_freq_max, psd_log_scale, psd_normalize, psd_channel, freq_min, freq_max, analysis_options
                )
            elif analysis_type == "STFT":
                main_fig, psd_fig, spectrogram_fig, results, peak_table, band_table, stability_table, harmonics_table, freq_data, time_freq_data = perform_stft_analysis(
                    time_data, signal_data, sampling_freq, stft_window_size, stft_hop_size, stft_window_type, stft_overlap, stft_scaling, stft_freq_max, stft_colormap, freq_min, freq_max, analysis_options
                )
            elif analysis_type == "Wavelet":
                main_fig, psd_fig, spectrogram_fig, results, peak_table, band_table, stability_table, harmonics_table, freq_data, time_freq_data = perform_wavelet_analysis(
                    time_data, signal_data, sampling_freq, wavelet_type, wavelet_levels, freq_min, freq_max, analysis_options
                )
            else:
                # Default to FFT
                main_fig, psd_fig, spectrogram_fig, results, peak_table, band_table, stability_table, harmonics_table, freq_data, time_freq_data = perform_fft_analysis(
                    time_data, signal_data, sampling_freq, fft_window, fft_n_points, freq_min, freq_max, analysis_options
                )
            
            logger.info("Frequency domain analysis completed successfully")
            return main_fig, psd_fig, spectrogram_fig, results, peak_table, band_table, stability_table, harmonics_table, freq_data, time_freq_data
            
        except Exception as e:
            logger.error(f"Error in frequency domain analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            error_fig = create_empty_figure()
            error_fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            
            error_results = html.Div([
                html.H5("Error in Frequency Analysis"),
                html.P(f"Analysis failed: {str(e)}"),
                html.P("Please check your data and parameters.")
            ])
            
            empty_figs = [error_fig for _ in range(3)]
            empty_tables = [create_empty_figure() for _ in range(4)]
            
            return empty_figs + [error_results] + empty_tables + [None, None]

    # Advanced Filtering Callback
    @app.callback(
        [Output("frequency-filtered-signal-plot", "figure"),
         Output("filter-response-plot", "figure"),
         Output("filter-stats", "children")],
        [Input("btn-apply-filter", "n_clicks")],
        [State("store-uploaded-data", "data"),
         State("store-data-config", "data"),
         State("filter-type", "value"),
         State("cutoff-freq", "value"),
         State("filter-order", "value")]
    )
    def apply_filter(n_clicks, data_store, config_store, filter_type, cutoff_freq, filter_order):
        """Apply frequency filter to the signal."""
        if not n_clicks or not data_store or not config_store:
            raise PreventUpdate
            
        try:
            # Extract data
            df = pd.DataFrame(data_store["data"])
            sampling_freq = config_store["sampling_freq"]
            signal_data = df.iloc[:, 1].values
            
            # Set default values
            filter_type = filter_type or "lowpass"
            cutoff_freq = cutoff_freq or sampling_freq / 4
            filter_order = filter_order or 4
            
            # Normalize cutoff frequency
            nyquist = sampling_freq / 2
            normalized_cutoff = cutoff_freq / nyquist
            
            if normalized_cutoff >= 1.0:
                normalized_cutoff = 0.95
            
            # Design filter
            if filter_type == "lowpass":
                b, a = signal.butter(filter_order, normalized_cutoff, btype='low')
            elif filter_type == "highpass":
                b, a = signal.butter(filter_order, normalized_cutoff, btype='high')
            elif filter_type == "bandpass":
                # For bandpass, we need two cutoff frequencies
                if isinstance(cutoff_freq, (list, tuple)) and len(cutoff_freq) == 2:
                    low_cutoff = cutoff_freq[0] / nyquist
                    high_cutoff = cutoff_freq[1] / nyquist
                else:
                    low_cutoff = normalized_cutoff * 0.5
                    high_cutoff = normalized_cutoff
                b, a = signal.butter(filter_order, [low_cutoff, high_cutoff], btype='band')
            else:
                raise ValueError(f"Unsupported filter type: {filter_type}")
            
            # Apply filter
            filtered_signal = signal.filtfilt(b, a, signal_data)
            
            # Create time axis
            time_axis = np.arange(len(signal_data)) / sampling_freq
            
            # Create filtered signal plot
            signal_fig = go.Figure()
            signal_fig.add_trace(go.Scatter(
                x=time_axis,
                y=signal_data,
                mode='lines',
                name='Original Signal',
                line=dict(color='blue', opacity=0.7)
            ))
            signal_fig.add_trace(go.Scatter(
                x=time_axis,
                y=filtered_signal,
                mode='lines',
                name='Filtered Signal',
                line=dict(color='red')
            ))
            
            signal_fig.update_layout(
                title=f"{filter_type.title()} Filtered Signal",
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude",
                showlegend=True
            )
            
            # Create filter response plot
            w, h = signal.freqz(b, a, worN=1000)
            freq_response = w * sampling_freq / (2 * np.pi)
            
            response_fig = go.Figure()
            response_fig.add_trace(go.Scatter(
                x=freq_response,
                y=20 * np.log10(np.abs(h)),
                mode='lines',
                name='Filter Response',
                line=dict(color='green')
            ))
            
            response_fig.update_layout(
                title="Filter Frequency Response",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude (dB)",
                showlegend=True
            )
            
            # Create stats
            stats = html.Div([
                html.H5("Filter Statistics"),
                html.P(f"Filter Type: {filter_type.title()}"),
                html.P(f"Cutoff Frequency: {cutoff_freq:.2f} Hz"),
                html.P(f"Filter Order: {filter_order}"),
                html.P(f"Sampling Frequency: {sampling_freq} Hz"),
                html.P(f"Signal Length: {len(signal_data)} samples"),
                html.P(f"Duration: {len(signal_data)/sampling_freq:.2f} seconds")
            ])
            
            return signal_fig, response_fig, stats
            
        except Exception as e:
            logger.error(f"Error in frequency filtering: {e}")
            error_fig = create_empty_figure()
            error_fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            
            error_stats = html.Div([
                html.H5("Error"),
                html.P(f"Filtering failed: {str(e)}")
            ])
            
            return error_fig, error_fig, error_stats

    # Time input update callbacks
    @app.callback(
        [Output("freq-start-time", "value"),
         Output("freq-end-time", "value")],
        [Input("freq-time-range-slider", "value"),
         Input("freq-btn-nudge-m10", "n_clicks"),
         Input("freq-btn-nudge-m1", "n_clicks"),
         Input("freq-btn-nudge-p1", "n_clicks"),
         Input("freq-btn-nudge-p10", "n_clicks")],
        [State("freq-start-time", "value"),
         State("freq-end-time", "value")]
    )
    def update_freq_time_inputs(slider_value, nudge_m10, nudge_m1, nudge_p1, nudge_p10, start_time, end_time):
        """Update time inputs based on slider or nudge buttons."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if trigger_id == "freq-time-range-slider" and slider_value:
            return slider_value[0], slider_value[1]
        
        # Handle nudge buttons
        time_window = end_time - start_time if start_time and end_time else 10
        
        if trigger_id == "freq-btn-nudge-m10":
            new_start = max(0, start_time - 10) if start_time else 0
            new_end = new_start + time_window
            return new_start, new_end
        elif trigger_id == "freq-btn-nudge-m1":
            new_start = max(0, start_time - 1) if start_time else 0
            new_end = new_start + time_window
            return new_start, new_end
        elif trigger_id == "freq-btn-nudge-p1":
            new_start = start_time + 1 if start_time else 1
            new_end = new_start + time_window
            return new_start, new_end
        elif trigger_id == "freq-btn-nudge-p10":
            new_start = start_time + 10 if start_time else 10
            new_end = new_start + time_window
            return new_start, new_end
        
        return no_update, no_update

    # Time slider range update callback
    @app.callback(
        Output("freq-time-range-slider", "max"),
        [Input("store-uploaded-data", "data")]
    )
    def update_freq_time_slider_range(data_store):
        """Update time slider range based on uploaded data."""
        if not data_store:
            return 100
        
        try:
            df = pd.DataFrame(data_store["data"])
            if df.empty:
                return 100
            
            # Get time column (assume first column)
            time_data = df.iloc[:, 0].values
            max_time = np.max(time_data)
            return max_time
        except Exception as e:
            logger.error(f"Error updating time slider range: {e}")
            return 100


# Helper functions for frequency analysis
def create_empty_figure():
    """Create an empty figure for error handling."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    return fig


def perform_fft_analysis(time_data, signal_data, sampling_freq, window_type, n_points, freq_min, freq_max, options):
    """Perform FFT analysis on the signal."""
    # Apply window if specified
    if window_type and window_type != "none":
        window_func = getattr(signal.windows, window_type, signal.windows.hann)
        window = window_func(len(signal_data))
        windowed_signal = signal_data * window
    else:
        windowed_signal = signal_data
    
    # Perform FFT
    fft_result = fft(windowed_signal, n=n_points)
    freqs = fftfreq(n_points, 1/sampling_freq)
    
    # Get positive frequencies only
    positive_mask = freqs >= 0
    freqs = freqs[positive_mask]
    fft_magnitude = np.abs(fft_result[positive_mask])
    
    # Filter frequency range if specified
    if freq_min is not None and freq_max is not None:
        freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
        freqs = freqs[freq_mask]
        fft_magnitude = fft_magnitude[freq_mask]
    
    # Create main FFT plot
    main_fig = go.Figure()
    main_fig.add_trace(go.Scatter(
        x=freqs,
        y=fft_magnitude,
        mode='lines',
        name='FFT Magnitude',
        line=dict(color='blue')
    ))
    main_fig.update_layout(
        title="Frequency Domain Analysis (FFT)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        showlegend=True
    )
    
    # Create PSD plot (same as main for FFT)
    psd_fig = main_fig
    
    # Create spectrogram plot (not applicable for FFT)
    spectrogram_fig = create_empty_figure()
    spectrogram_fig.update_layout(title="Spectrogram (Not applicable for FFT)")
    
    # Generate analysis results
    results = generate_frequency_analysis_results(freqs, fft_magnitude, "FFT")
    
    # Generate tables
    peak_table = generate_peak_analysis_table(freqs, fft_magnitude)
    band_table = generate_band_power_table(freqs, fft_magnitude)
    stability_table = generate_stability_table(freqs, fft_magnitude)
    harmonics_table = generate_harmonics_table(freqs, fft_magnitude)
    
    # Store data
    freq_data = {
        "frequencies": freqs.tolist(),
        "magnitudes": fft_magnitude.tolist(),
        "analysis_type": "FFT"
    }
    time_freq_data = None
    
    return main_fig, psd_fig, spectrogram_fig, results, peak_table, band_table, stability_table, harmonics_table, freq_data, time_freq_data


def perform_psd_analysis(time_data, signal_data, sampling_freq, window, overlap, freq_max, log_scale, normalize, channel, freq_min, freq_max_range, options):
    """Perform Power Spectral Density analysis."""
    # Set default values
    window = window or "hann"
    overlap = overlap or 0.5
    freq_max = freq_max or sampling_freq / 2
    
    # Calculate PSD
    freqs, psd = signal.welch(signal_data, fs=sampling_freq, window=window, nperseg=min(256, len(signal_data)//4), noverlap=int(256*overlap))
    
    # Filter frequency range if specified
    if freq_min is not None and freq_max_range is not None:
        freq_mask = (freqs >= freq_min) & (freqs <= freq_max_range)
        freqs = freqs[freq_mask]
        psd = psd[freq_mask]
    
    # Apply log scale if requested
    if log_scale:
        psd = 10 * np.log10(psd + 1e-10)
        y_label = "Power Spectral Density (dB/Hz)"
    else:
        y_label = "Power Spectral Density"
    
    # Normalize if requested
    if normalize:
        psd = psd / np.max(psd)
        y_label += " (Normalized)"
    
    # Create main PSD plot
    main_fig = go.Figure()
    main_fig.add_trace(go.Scatter(
        x=freqs,
        y=psd,
        mode='lines',
        name='PSD',
        line=dict(color='red')
    ))
    main_fig.update_layout(
        title="Power Spectral Density Analysis",
        xaxis_title="Frequency (Hz)",
        yaxis_title=y_label,
        showlegend=True
    )
    
    # PSD plot is the same as main
    psd_fig = main_fig
    
    # Create spectrogram plot
    spectrogram_fig = create_empty_figure()
    spectrogram_fig.update_layout(title="Spectrogram (Not applicable for PSD)")
    
    # Generate results and tables
    results = generate_frequency_analysis_results(freqs, psd, "PSD")
    peak_table = generate_peak_analysis_table(freqs, psd)
    band_table = generate_band_power_table(freqs, psd)
    stability_table = generate_stability_table(freqs, psd)
    harmonics_table = generate_harmonics_table(freqs, psd)
    
    # Store data
    freq_data = {
        "frequencies": freqs.tolist(),
        "magnitudes": psd.tolist(),
        "analysis_type": "PSD"
    }
    time_freq_data = None
    
    return main_fig, psd_fig, spectrogram_fig, results, peak_table, band_table, stability_table, harmonics_table, freq_data, time_freq_data


def perform_stft_analysis(time_data, signal_data, sampling_freq, window_size, hop_size, window_type, overlap, scaling, freq_max, colormap, freq_min, freq_max_range, options):
    """Perform Short-Time Fourier Transform analysis."""
    # Set default values
    window_size = window_size or min(256, len(signal_data)//4)
    hop_size = hop_size or window_size // 4
    window_type = window_type or "hann"
    colormap = colormap or "viridis"
    
    # Calculate STFT
    freqs, times, Zxx = signal.stft(signal_data, fs=sampling_freq, nperseg=window_size, noverlap=overlap)
    
    # Filter frequency range if specified
    if freq_min is not None and freq_max_range is not None:
        freq_mask = (freqs >= freq_min) & (freqs <= freq_max_range)
        freqs = freqs[freq_mask]
        Zxx = Zxx[freq_mask, :]
    
    # Create main STFT plot (spectrogram)
    main_fig = go.Figure(data=go.Heatmap(
        z=20 * np.log10(np.abs(Zxx) + 1e-10),
        x=times,
        y=freqs,
        colorscale=colormap,
        name='STFT Magnitude'
    ))
    main_fig.update_layout(
        title="Short-Time Fourier Transform (STFT)",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        showlegend=False
    )
    
    # PSD plot (average over time)
    psd = np.mean(np.abs(Zxx)**2, axis=1)
    psd_fig = go.Figure()
    psd_fig.add_trace(go.Scatter(
        x=freqs,
        y=psd,
        mode='lines',
        name='Average PSD',
        line=dict(color='red')
    ))
    psd_fig.update_layout(
        title="Average Power Spectral Density (STFT)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power Spectral Density",
        showlegend=True
    )
    
    # Spectrogram plot is the same as main
    spectrogram_fig = main_fig
    
    # Generate results and tables
    results = generate_frequency_analysis_results(freqs, psd, "STFT")
    peak_table = generate_peak_analysis_table(freqs, psd)
    band_table = generate_band_power_table(freqs, psd)
    stability_table = generate_stability_table(freqs, psd)
    harmonics_table = generate_harmonics_table(freqs, psd)
    
    # Store data
    freq_data = {
        "frequencies": freqs.tolist(),
        "magnitudes": psd.tolist(),
        "analysis_type": "STFT"
    }
    time_freq_data = {
        "times": times.tolist(),
        "frequencies": freqs.tolist(),
        "magnitudes": (20 * np.log10(np.abs(Zxx) + 1e-10)).tolist()
    }
    
    return main_fig, psd_fig, spectrogram_fig, results, peak_table, band_table, stability_table, harmonics_table, freq_data, time_freq_data


def perform_wavelet_analysis(time_data, signal_data, sampling_freq, wavelet_type, levels, freq_min, freq_max, options):
    """Perform Wavelet analysis on the signal."""
    # This is a simplified wavelet analysis - in practice, you'd use pywt
    # For now, we'll create a placeholder
    main_fig = create_empty_figure()
    main_fig.update_layout(title="Wavelet Analysis (Not implemented)")
    
    psd_fig = create_empty_figure()
    psd_fig.update_layout(title="Wavelet PSD (Not implemented)")
    
    spectrogram_fig = create_empty_figure()
    spectrogram_fig.update_layout(title="Wavelet Spectrogram (Not implemented)")
    
    results = html.Div([
        html.H5("Wavelet Analysis"),
        html.P("Wavelet analysis is not yet implemented in this version.")
    ])
    
    empty_table = create_empty_figure()
    empty_table.update_layout(title="No Data")
    
    freq_data = {"analysis_type": "Wavelet"}
    time_freq_data = None
    
    return main_fig, psd_fig, spectrogram_fig, results, empty_table, empty_table, empty_table, empty_table, freq_data, time_freq_data


def generate_frequency_analysis_results(freqs, magnitudes, analysis_type):
    """Generate comprehensive frequency analysis results."""
    if len(freqs) == 0 or len(magnitudes) == 0:
        return html.Div("No data available for analysis")
    
    # Calculate basic statistics
    max_mag = np.max(magnitudes)
    max_freq = freqs[np.argmax(magnitudes)]
    mean_mag = np.mean(magnitudes)
    std_mag = np.std(magnitudes)
    
    # Calculate total power
    total_power = np.sum(magnitudes)
    
    # Find dominant frequency bands
    if len(freqs) > 1:
        freq_step = freqs[1] - freqs[0]
        power_density = magnitudes / freq_step
        total_power_density = np.sum(power_density)
    else:
        total_power_density = total_power
    
    return html.Div([
        html.H5(f"{analysis_type} Analysis Results"),
        html.Div([
            html.Div([
                html.H6("Peak Analysis"),
                html.P(f"Peak Frequency: {max_freq:.2f} Hz"),
                html.P(f"Peak Magnitude: {max_mag:.4f}"),
                html.P(f"Mean Magnitude: {mean_mag:.4f}"),
                html.P(f"Standard Deviation: {std_mag:.4f}")
            ], className="col-md-6"),
            html.Div([
                html.H6("Power Analysis"),
                html.P(f"Total Power: {total_power:.4f}"),
                html.P(f"Power Density: {total_power_density:.4f}"),
                html.P(f"Frequency Range: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz"),
                html.P(f"Frequency Resolution: {freqs[1] - freqs[0]:.4f} Hz")
            ], className="col-md-6")
        ], className="row")
    ])


def generate_peak_analysis_table(freqs, magnitudes):
    """Generate peak analysis table."""
    if len(freqs) == 0 or len(magnitudes) == 0:
        return create_empty_figure()
    
    # Find peaks
    peaks, _ = signal.find_peaks(magnitudes, height=np.max(magnitudes)*0.1)
    
    if len(peaks) == 0:
        return html.Div("No significant peaks found")
    
    # Create table
    table_data = []
    for i, peak_idx in enumerate(peaks[:10]):  # Top 10 peaks
        table_data.append({
            "Peak #": i + 1,
            "Frequency (Hz)": f"{freqs[peak_idx]:.2f}",
            "Magnitude": f"{magnitudes[peak_idx]:.4f}",
            "Relative Power (%)": f"{magnitudes[peak_idx]/np.max(magnitudes)*100:.1f}"
        })
    
    return html.Div([
        html.H6("Peak Analysis"),
        html.Table([
            html.Thead([
                html.Tr([html.Th(col) for col in table_data[0].keys()])
            ]),
            html.Tbody([
                html.Tr([html.Td(row[col]) for col in row.keys()])
                for row in table_data
            ])
        ], className="table table-sm")
    ])


def generate_band_power_table(freqs, magnitudes):
    """Generate frequency band power table."""
    if len(freqs) == 0 or len(magnitudes) == 0:
        return create_empty_figure()
    
    # Define frequency bands
    bands = [
        (0.1, 0.5, "Very Low Frequency (VLF)"),
        (0.5, 2.0, "Low Frequency (LF)"),
        (2.0, 5.0, "Medium Frequency (MF)"),
        (5.0, 10.0, "High Frequency (HF)"),
        (10.0, 50.0, "Very High Frequency (VHF)")
    ]
    
    table_data = []
    for low_freq, high_freq, band_name in bands:
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        if np.any(mask):
            band_power = np.sum(magnitudes[mask])
            total_power = np.sum(magnitudes)
            percentage = (band_power / total_power) * 100 if total_power > 0 else 0
            table_data.append({
                "Band": band_name,
                "Range (Hz)": f"{low_freq:.1f} - {high_freq:.1f}",
                "Power": f"{band_power:.4f}",
                "Percentage": f"{percentage:.1f}%"
            })
    
    if not table_data:
        return html.Div("No band power data available")
    
    return html.Div([
        html.H6("Frequency Band Power"),
        html.Table([
            html.Thead([
                html.Tr([html.Th(col) for col in table_data[0].keys()])
            ]),
            html.Tbody([
                html.Tr([html.Td(row[col]) for col in row.keys()])
                for row in table_data
            ])
        ], className="table table-sm")
    ])


def generate_stability_table(freqs, magnitudes):
    """Generate signal stability analysis table."""
    if len(freqs) == 0 or len(magnitudes) == 0:
        return create_empty_figure()
    
    # Calculate stability metrics
    mean_mag = np.mean(magnitudes)
    std_mag = np.std(magnitudes)
    cv = (std_mag / mean_mag) * 100 if mean_mag > 0 else 0
    
    # Calculate frequency stability
    if len(freqs) > 1:
        freq_variation = np.std(np.diff(freqs))
    else:
        freq_variation = 0
    
    return html.Div([
        html.H6("Signal Stability Analysis"),
        html.Table([
            html.Thead([
                html.Tr([html.Th("Metric"), html.Th("Value")])
            ]),
            html.Tbody([
                html.Tr([html.Td("Coefficient of Variation"), html.Td(f"{cv:.2f}%")]),
                html.Tr([html.Td("Frequency Variation"), html.Td(f"{freq_variation:.4f} Hz")]),
                html.Tr([html.Td("Magnitude Stability"), html.Td("Stable" if cv < 20 else "Variable")])
            ])
        ], className="table table-sm")
    ])


def generate_harmonics_table(freqs, magnitudes):
    """Generate harmonics analysis table."""
    if len(freqs) == 0 or len(magnitudes) == 0:
        return create_empty_figure()
    
    # Find fundamental frequency (highest peak)
    max_peak_idx = np.argmax(magnitudes)
    fundamental_freq = freqs[max_peak_idx]
    
    # Look for harmonics
    harmonics = []
    for i in range(2, 6):  # Check up to 5th harmonic
        harmonic_freq = fundamental_freq * i
        # Find closest frequency
        closest_idx = np.argmin(np.abs(freqs - harmonic_freq))
        if closest_idx < len(magnitudes):
            harmonic_magnitude = magnitudes[closest_idx]
            harmonic_ratio = harmonic_magnitude / magnitudes[max_peak_idx] if magnitudes[max_peak_idx] > 0 else 0
            harmonics.append({
                "Harmonic": f"{i}nd" if i == 2 else f"{i}rd" if i == 3 else f"{i}th",
                "Frequency (Hz)": f"{freqs[closest_idx]:.2f}",
                "Magnitude": f"{harmonic_magnitude:.4f}",
                "Ratio to Fundamental": f"{harmonic_ratio:.3f}"
            })
    
    if not harmonics:
        return html.Div("No significant harmonics detected")
    
    return html.Div([
        html.H6("Harmonics Analysis"),
        html.P(f"Fundamental Frequency: {fundamental_freq:.2f} Hz"),
        html.Table([
            html.Thead([
                html.Tr([html.Th(col) for col in harmonics[0].keys()])
            ]),
            html.Tbody([
                html.Tr([html.Td(row[col]) for col in row.keys()])
                for row in harmonics
            ])
        ], className="table table-sm")
    ])
