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
         Output("freq-time-freq-plot", "figure"),
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
         State("stft-window-size", "value"),
         State("stft-hop-size", "value"),
         State("wavelet-type", "value"),
         State("wavelet-levels", "value"),
         State("freq-min", "value"),
         State("freq-max", "value"),
         State("freq-analysis-options", "value")]
    )
    def frequency_domain_callback(pathname, n_clicks, slider_value, nudge_m10, nudge_m1, nudge_p1, nudge_p10,
                                start_time, end_time, analysis_type, fft_window, fft_n_points, stft_window_size,
                                stft_hop_size, wavelet_type, wavelet_levels, freq_min, freq_max, analysis_options):
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
            return create_empty_figure(), create_empty_figure(), "Navigate to Frequency page", create_empty_figure(), None, None
        
        # If this is the first time loading the page (no button clicks), show a message
        if not ctx.triggered or ctx.triggered[0]["prop_id"].split(".")[0] == "url":
            logger.info("First time loading frequency page, attempting to load data")
        
        try:
            # Get data from the data service
            logger.info("Attempting to import data service...")
            from ..services.data_service import get_data_service
            logger.info("Data service imported successfully")
            
            data_service = get_data_service()
            logger.info("Data service instance created")
            
            # Get the most recent data
            logger.info("Fetching all data from service...")
            all_data = data_service.get_all_data()
            logger.info(f"Retrieved data: {len(all_data) if all_data else 0} entries")
            
            if not all_data:
                logger.warning("No data found in service")
                return create_empty_figure(), create_empty_figure(), "No data available. Please upload and process data first.", create_empty_figure(), None, None
            
            # Get the most recent data entry
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            
            # Get column mapping
            column_mapping = data_service.get_column_mapping(latest_data_id)
            if not column_mapping:
                return create_empty_figure(), create_empty_figure(), "Please process your data on the Upload page first", create_empty_figure(), None, None
                
            df = data_service.get_data(latest_data_id)
            if df is None or df.empty:
                return create_empty_figure(), create_empty_figure(), "Data is empty or corrupted.", create_empty_figure(), None, None
            
            # Get sampling frequency
            sampling_freq = latest_data.get('info', {}).get('sampling_freq', 1000)
            
            # Handle time window adjustments
            if trigger_id in ["freq-btn-nudge-m10", "freq-btn-nudge-m1", "freq-btn-nudge-p1", "freq-btn-nudge-p10"]:
                if not start_time or not end_time:
                    start_time, end_time = 0, 10
                    
                if trigger_id == "freq-btn-nudge-m10":
                    start_time = max(0, start_time - 10)
                    end_time = max(10, end_time - 10)
                elif trigger_id == "freq-btn-nudge-m1":
                    start_time = max(0, start_time - 1)
                    end_time = max(1, end_time - 1)
                elif trigger_id == "freq-btn-nudge-p1":
                    start_time = start_time + 1
                    end_time = end_time + 1
                elif trigger_id == "freq-btn-nudge-p10":
                    start_time = start_time + 10
                    end_time = end_time + 10
            
            # Set default time window if not specified
            if not start_time or not end_time:
                start_time, end_time = 0, 10
            
            # Apply time window
            start_sample = int(start_time * sampling_freq)
            end_sample = int(end_time * sampling_freq)
            windowed_data = df.iloc[start_sample:end_sample].copy()
            
            # Create time axis
            time_axis = np.arange(len(windowed_data)) / sampling_freq
            
            # Get signal column
            signal_column = column_mapping.get('signal', df.columns[0])
            signal_data = windowed_data[signal_column].values
            
            # Perform frequency domain analysis based on selected type
            if analysis_type == "fft":
                main_plot = create_fft_plot(signal_data, sampling_freq, fft_window, fft_n_points, freq_min, freq_max)
                time_freq_plot = create_empty_figure()
            elif analysis_type == "stft":
                main_plot = create_stft_plot(signal_data, sampling_freq, stft_window_size, stft_hop_size, freq_min, freq_max)
                time_freq_plot = create_stft_spectrogram(signal_data, sampling_freq, stft_window_size, stft_hop_size)
            elif analysis_type == "wavelet":
                main_plot = create_wavelet_plot(signal_data, sampling_freq, wavelet_type, wavelet_levels, freq_min, freq_max)
                time_freq_plot = create_wavelet_scalogram(signal_data, sampling_freq, wavelet_type, wavelet_levels)
            else:
                main_plot = create_fft_plot(signal_data, sampling_freq, fft_window, fft_n_points, freq_min, freq_max)
                time_freq_plot = create_empty_figure()
            
            # Generate analysis results
            analysis_results = generate_frequency_analysis_results(signal_data, sampling_freq, analysis_type, analysis_options)
            
            # Generate detailed table components
            peak_table = create_frequency_peak_analysis_table(signal_data, sampling_freq, analysis_type, analysis_options)
            band_power_table = create_frequency_band_power_table(signal_data, sampling_freq, analysis_type, analysis_options)
            stability_table = create_frequency_stability_table(signal_data, sampling_freq, analysis_type, analysis_options)
            harmonics_table = create_frequency_harmonics_table(signal_data, sampling_freq, analysis_type, analysis_options)
            
            # Store processed data
            frequency_data = {
                "signal_data": signal_data.tolist(),
                "time_axis": time_axis.tolist(),
                "sampling_freq": sampling_freq,
                "window": [start_time, end_time],
                "analysis_type": analysis_type
            }
            
            time_freq_data = {
                "analysis_type": analysis_type,
                "parameters": {
                    "fft_window": fft_window,
                    "fft_n_points": fft_n_points,
                    "stft_window_size": stft_window_size,
                    "stft_hop_size": stft_hop_size,
                    "wavelet_type": wavelet_type,
                    "wavelet_levels": wavelet_levels
                }
            }
            
            return main_plot, time_freq_plot, analysis_results, peak_table, band_power_table, stability_table, harmonics_table, frequency_data, time_freq_data
            
        except Exception as e:
            logger.error(f"Error in frequency domain callback: {e}")
            import traceback
            traceback.print_exc()
            error_msg = f"Error in analysis: {str(e)}"
            return create_empty_figure(), create_empty_figure(), error_msg, error_msg, error_msg, error_msg, error_msg, None, None
    
    # Advanced Filtering Callback
    @app.callback(
        [Output("filter-original-plot", "figure"),
         Output("filter-filtered-plot", "figure"),
         Output("filter-comparison-plot", "figure"),
         Output("filter-quality-metrics", "children"),
         Output("filter-quality-plots", "figure"),
         Output("store-filtering-data", "data")],
        [Input("url", "pathname"),
         Input("filter-btn-apply", "n_clicks"),
         Input("filter-time-range-slider", "value"),
         Input("filter-btn-nudge-m10", "n_clicks"),
         Input("filter-btn-nudge-m1", "n_clicks"),
         Input("filter-btn-nudge-p1", "n_clicks"),
         Input("filter-btn-nudge-p10", "n_clicks")],
        [State("filter-start-time", "value"),
         State("filter-end-time", "value"),
         State("filter-type-select", "value"),
         State("filter-family-advanced", "value"),
         State("filter-response-advanced", "value"),
         State("filter-low-freq-advanced", "value"),
         State("filter-high-freq-advanced", "value"),
         State("filter-order-advanced", "value"),
         State("advanced-filter-method", "value"),
         State("advanced-noise-level", "value"),
         State("advanced-iterations", "value"),
         State("advanced-learning-rate", "value"),
         State("artifact-type", "value"),
         State("artifact-removal-strength", "value"),
         State("neural-network-type", "value"),
         State("neural-model-complexity", "value"),
         State("ensemble-method", "value"),
         State("ensemble-n-filters", "value"),
         State("filter-quality-options", "value")]
    )
    def advanced_filtering_callback(pathname, n_clicks, slider_value, nudge_m10, nudge_m1, nudge_p1, nudge_p10,
                                  start_time, end_time, filter_type, filter_family, filter_response, low_freq, high_freq,
                                  filter_order, advanced_method, noise_level, iterations, learning_rate, artifact_type,
                                  artifact_strength, neural_type, neural_complexity, ensemble_method, ensemble_n_filters,
                                  quality_options):
        """Unified callback for advanced filtering."""
        ctx = callback_context
        
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        logger.info(f"=== ADVANCED FILTERING CALLBACK TRIGGERED ===")
        logger.info(f"Trigger ID: {trigger_id}")
        logger.info(f"Pathname: {pathname}")
        
        # Only run this when we're on the filtering page
        if pathname != "/filtering":
            logger.info("Not on filtering page, returning empty figures")
            return create_empty_figure(), create_empty_figure(), create_empty_figure(), "Navigate to Filtering page", create_empty_figure(), None
        
        # If this is the first time loading the page (no button clicks), show a message
        if not ctx.triggered or ctx.triggered[0]["prop_id"].split(".")[0] == "url":
            logger.info("First time loading filtering page, attempting to load data")
        
        try:
            # Get data from the data service
            logger.info("Attempting to import data service for filtering...")
            from ..services.data_service import get_data_service
            logger.info("Data service imported successfully for filtering")
            
            data_service = get_data_service()
            logger.info("Data service instance created for filtering")
            
            # Get the most recent data
            logger.info("Fetching all data from service for filtering...")
            all_data = data_service.get_all_data()
            logger.info(f"Retrieved data for filtering: {len(all_data) if all_data else 0} entries")
            
            if not all_data:
                logger.warning("No data found in service for filtering")
                return create_empty_figure(), create_empty_figure(), create_empty_figure(), "No data available. Please upload and process data first.", create_empty_figure(), None
            
            # Get the most recent data entry
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            
            # Get column mapping
            column_mapping = data_service.get_column_mapping(latest_data_id)
            if not column_mapping:
                return create_empty_figure(), create_empty_figure(), create_empty_figure(), "Please process your data on the Upload page first", create_empty_figure(), None
                
            df = data_service.get_data(latest_data_id)
            if df is None or df.empty:
                return create_empty_figure(), create_empty_figure(), create_empty_figure(), "Data is empty or corrupted.", create_empty_figure(), None
            
            # Get sampling frequency
            sampling_freq = latest_data.get('info', {}).get('sampling_freq', 1000)
            
            # Handle time window adjustments
            if trigger_id in ["filter-btn-nudge-m10", "filter-btn-nudge-m1", "filter-btn-nudge-p1", "filter-btn-nudge-p10"]:
                if not start_time or not end_time:
                    start_time, end_time = 0, 10
                    
                if trigger_id == "filter-btn-nudge-m10":
                    start_time = max(0, start_time - 10)
                    end_time = max(10, end_time - 10)
                elif trigger_id == "filter-btn-nudge-m1":
                    start_time = max(0, start_time - 1)
                    end_time = max(1, end_time - 1)
                elif trigger_id == "filter-btn-nudge-p1":
                    start_time = start_time + 1
                    end_time = end_time + 1
                elif trigger_id == "filter-btn-nudge-p10":
                    start_time = start_time + 10
                    end_time = end_time + 10
            
            # Set default time window if not specified
            if not start_time or not end_time:
                start_time, end_time = 0, 10
            
            # Apply time window
            start_sample = int(start_time * sampling_freq)
            end_sample = int(end_time * sampling_freq)
            windowed_data = df.iloc[start_sample:end_sample].copy()
            
            # Create time axis
            time_axis = np.arange(len(windowed_data)) / sampling_freq
            
            # Get signal column
            signal_column = column_mapping.get('signal', df.columns[0])
            signal_data = windowed_data[signal_column].values
            
            # Create original signal plot
            original_plot = create_signal_plot(signal_data, time_axis, "Original Signal")
            
            # Apply filtering based on selected type
            if filter_type == "traditional":
                filtered_data = apply_traditional_filter(signal_data, sampling_freq, filter_family, filter_response, low_freq, high_freq, filter_order)
            elif filter_type == "advanced":
                filtered_data = apply_advanced_filter(signal_data, advanced_method, noise_level, iterations, learning_rate)
            elif filter_type == "artifact":
                filtered_data = apply_artifact_removal(signal_data, sampling_freq, artifact_type, artifact_strength)
            elif filter_type == "neural":
                filtered_data = apply_neural_filter(signal_data, neural_type, neural_complexity)
            elif filter_type == "ensemble":
                filtered_data = apply_ensemble_filter(signal_data, ensemble_method, ensemble_n_filters)
            else:
                filtered_data = signal_data
            
            # Create filtered signal plot
            filtered_plot = create_signal_plot(filtered_data, time_axis, "Filtered Signal")
            
            # Create comparison plot
            comparison_plot = create_filter_comparison_plot(signal_data, filtered_data, time_axis)
            
            # Generate quality metrics
            quality_metrics = generate_filter_quality_metrics(signal_data, filtered_data, sampling_freq, quality_options)
            
            # Create quality plots
            quality_plots = create_filter_quality_plots(signal_data, filtered_data, sampling_freq, quality_options)
            
            # Store filtering data
            filtering_data = {
                "original_signal": signal_data.tolist(),
                "filtered_signal": filtered_data.tolist(),
                "time_axis": time_axis.tolist(),
                "sampling_freq": sampling_freq,
                "filter_type": filter_type,
                "filter_parameters": {
                    "filter_family": filter_family,
                    "filter_response": filter_response,
                    "low_freq": low_freq,
                    "high_freq": high_freq,
                    "filter_order": filter_order
                }
            }
            
            return original_plot, filtered_plot, comparison_plot, quality_metrics, quality_plots, filtering_data
            
        except Exception as e:
            logger.error(f"Error in advanced filtering callback: {e}")
            import traceback
            traceback.print_exc()
            return create_empty_figure(), create_empty_figure(), create_empty_figure(), f"Error in filtering: {str(e)}", create_empty_figure(), None
    
    # Helper function callbacks for dynamic parameter visibility
    @app.callback(
        [Output("fft-params", "style"),
         Output("stft-params", "style"),
         Output("wavelet-params", "style")],
        [Input("freq-analysis-type", "value")]
    )
    def toggle_frequency_params(analysis_type):
        """Toggle visibility of parameter sections based on analysis type."""
        fft_style = {"display": "block" if analysis_type == "fft" else "none"}
        stft_style = {"display": "block" if analysis_type == "stft" else "none"}
        wavelet_style = {"display": "block" if analysis_type == "wavelet" else "none"}
        
        return fft_style, stft_style, wavelet_style
    
    @app.callback(
        [Output("traditional-filter-params", "style"),
         Output("advanced-filter-params", "style"),
         Output("artifact-removal-params", "style"),
         Output("neural-filter-params", "style"),
         Output("ensemble-filter-params", "style")],
        [Input("filter-type-select", "value")]
    )
    def toggle_filter_params(filter_type):
        """Toggle visibility of filter parameter sections based on filter type."""
        traditional_style = {"display": "block" if filter_type == "traditional" else "none"}
        advanced_style = {"display": "block" if filter_type == "advanced" else "none"}
        artifact_style = {"display": "block" if filter_type == "artifact" else "none"}
        neural_style = {"display": "block" if filter_type == "neural" else "none"}
        ensemble_style = {"display": "block" if filter_type == "ensemble" else "none"}
        
        return traditional_style, advanced_style, artifact_style, neural_style, ensemble_style
    
    # Time input update callbacks
    @app.callback(
        [Output("freq-start-time", "value"),
         Output("freq-end-time", "value")],
        [Input("freq-time-range-slider", "value")]
    )
    def update_freq_time_inputs(slider_value):
        """Update frequency time input fields based on slider."""
        if not slider_value:
            return no_update, no_update
        return slider_value[0], slider_value[1]
    
    @app.callback(
        [Output("filter-start-time", "value"),
         Output("filter-end-time", "value")],
        [Input("filter-time-range-slider", "value")]
    )
    def update_filter_time_inputs(slider_value):
        """Update filtering time input fields based on slider."""
        if not slider_value:
            return no_update, no_update
        return slider_value[0], slider_value[1]
    



# Helper functions for creating plots and analysis
def create_empty_figure():
    """Create an empty plotly figure."""
    return go.Figure().add_annotation(
        text="No data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False
    )

def create_fft_plot(signal_data, sampling_freq, window_type, n_points, freq_min, freq_max):
    """Create enhanced FFT plot with vitalDSP insights."""
    logger.info(f"Creating FFT plot with window: {window_type}, n_points: {n_points}")
    
    # Apply window function
    if window_type == "hamming":
        windowed_signal = signal_data * np.hamming(len(signal_data))
        logger.info("Applied Hamming window for optimal frequency resolution")
    elif window_type == "hanning":
        windowed_signal = signal_data * np.hanning(len(signal_data))
        logger.info("Applied Hanning window for reduced spectral leakage")
    elif window_type == "blackman":
        windowed_signal = signal_data * np.blackman(len(signal_data))
        logger.info("Applied Blackman window for excellent sidelobe suppression")
    elif window_type == "kaiser":
        windowed_signal = signal_data * np.kaiser(len(signal_data), beta=14)
        logger.info("Applied Kaiser window with beta=14 for optimal time-frequency trade-off")
    else:
        windowed_signal = signal_data
        logger.info("No window applied (rectangular window)")
    
    # Compute FFT
    fft_result = rfft(windowed_signal, n=n_points)
    freqs = rfftfreq(n_points, 1/sampling_freq)
    
    # Calculate vitalDSP metrics
    magnitude = np.abs(fft_result)
    power_spectrum = magnitude ** 2
    
    # Find dominant frequencies
    peak_idx = np.argmax(magnitude)
    peak_freq = freqs[peak_idx]
    peak_magnitude = magnitude[peak_idx]
    
    # Calculate frequency resolution
    freq_resolution = sampling_freq / n_points
    logger.info(f"Frequency resolution: {freq_resolution:.3f} Hz")
    logger.info(f"Peak frequency: {peak_freq:.2f} Hz with magnitude {peak_magnitude:.2e}")
    
    # Filter frequency range
    if freq_min is not None and freq_max is not None:
        mask = (freqs >= freq_min) & (freqs <= freq_max)
        freqs = freqs[mask]
        fft_result = fft_result[mask]
        magnitude = magnitude[mask]
        power_spectrum = power_spectrum[mask]
        logger.info(f"Filtered frequency range: {freq_min:.1f} - {freq_max:.1f} Hz")
    
    # Create enhanced plot with subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Frequency Spectrum (Magnitude)", "Power Spectral Density"),
        vertical_spacing=0.1
    )
    
    # Magnitude plot
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=magnitude,
            mode='lines',
            name=f'{window_type.title()} Window',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Frequency:</b> %{x:.2f} Hz<br><b>Magnitude:</b> %{y:.2e}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Mark peak frequency
    fig.add_trace(
        go.Scatter(
            x=[peak_freq],
            y=[peak_magnitude],
            mode='markers',
            name='Peak Frequency',
            marker=dict(color='red', size=10, symbol='diamond'),
            hovertemplate=f'<b>Peak:</b> {peak_freq:.2f} Hz<br><b>Magnitude:</b> {peak_magnitude:.2e}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Power spectral density plot
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=power_spectrum,
            mode='lines',
            name='Power Spectrum',
            line=dict(color='green', width=2),
            hovertemplate='<b>Frequency:</b> %{x:.2f} Hz<br><b>Power:</b> %{y:.2e}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add frequency band annotations
    if freq_max is None or freq_max > 30:
        # Add EEG frequency bands
        bands = {
            'Delta': (0.5, 4, 'purple'),
            'Theta': (4, 8, 'blue'),
            'Alpha': (8, 13, 'green'),
            'Beta': (13, 30, 'orange'),
            'Gamma': (30, 100, 'red')
        }
        
        for band_name, (low, high, color) in bands.items():
            if freq_max is None or high <= freq_max:
                fig.add_vrect(
                    x0=low, x1=high,
                    fillcolor=color, opacity=0.1,
                    layer="below", line_width=0,
                    annotation_text=band_name,
                    annotation_position="top left"
                )
    
    fig.update_layout(
        title=f"Enhanced FFT Analysis - {window_type.title()} Window (N={n_points})",
        template="plotly_white",
        height=600,
        showlegend=True,
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50, pad=4)
    )
    
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude", row=1, col=1)
    fig.update_yaxes(title_text="Power", row=2, col=1)
    
    logger.info("Enhanced FFT plot created successfully")
    return fig

def create_stft_plot(signal_data, sampling_freq, window_size, hop_size, freq_min, freq_max):
    """Create enhanced STFT plot with vitalDSP insights."""
    logger.info(f"Creating STFT plot with window_size: {window_size}, hop_size: {hop_size}")
    
    # Calculate STFT parameters
    overlap = window_size - hop_size
    time_resolution = hop_size / sampling_freq
    freq_resolution = sampling_freq / window_size
    
    logger.info(f"STFT Parameters - Time resolution: {time_resolution:.3f}s, Frequency resolution: {freq_resolution:.3f}Hz")
    logger.info(f"Overlap: {overlap} samples ({overlap/window_size*100:.1f}%)")
    
    # Compute STFT
    freqs, times, Zxx = signal.stft(signal_data, sampling_freq, nperseg=window_size, noverlap=overlap)
    
    # Calculate magnitude and power
    magnitude = np.abs(Zxx)
    power = magnitude ** 2
    
    # Find dominant time-frequency components
    max_power_idx = np.unravel_index(np.argmax(power), power.shape)
    max_freq = freqs[max_power_idx[0]]
    max_time = times[max_power_idx[1]]
    max_power = power[max_power_idx]
    
    logger.info(f"Dominant component: {max_freq:.2f} Hz at {max_time:.2f}s with power {max_power:.2e}")
    
    # Filter frequency range
    if freq_min is not None and freq_max is not None:
        mask = (freqs >= freq_min) & (freqs <= freq_max)
        freqs = freqs[mask]
        Zxx = Zxx[mask, :]
        magnitude = magnitude[mask, :]
        power = power[mask, :]
        logger.info(f"Filtered frequency range: {freq_min:.1f} - {freq_max:.1f} Hz")
    
    # Create enhanced plot with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("STFT Magnitude", "STFT Power", "Frequency Profile", "Time Profile"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # STFT Magnitude heatmap
    fig.add_trace(
        go.Heatmap(
            z=magnitude,
            x=times,
            y=freqs,
            colorscale='Viridis',
            name='Magnitude',
            hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Freq:</b> %{y:.2f}Hz<br><b>Magnitude:</b> %{z:.2e}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # STFT Power heatmap
    fig.add_trace(
        go.Heatmap(
            z=power,
            x=times,
            y=freqs,
            colorscale='Plasma',
            name='Power',
            hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Freq:</b> %{y:.2f}Hz<br><b>Power:</b> %{z:.2e}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Frequency profile at peak time
    peak_time_idx = np.argmax(np.max(power, axis=0))
    peak_time = times[peak_time_idx]
    freq_profile = magnitude[:, peak_time_idx]
    
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=freq_profile,
            mode='lines',
            name=f'Freq Profile at {peak_time:.2f}s',
            line=dict(color='red', width=2),
            hovertemplate='<b>Frequency:</b> %{x:.2f} Hz<br><b>Magnitude:</b> %{y:.2e}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Time profile at peak frequency
    peak_freq_idx = np.argmax(np.max(power, axis=1))
    peak_freq = freqs[peak_freq_idx]
    time_profile = magnitude[peak_freq_idx, :]
    
    fig.add_trace(
        go.Scatter(
            x=times,
            y=time_profile,
            mode='lines',
            name=f'Time Profile at {peak_freq:.2f}Hz',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Magnitude:</b> %{y:.2e}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Add frequency band annotations
    if freq_max is None or freq_max > 30:
        bands = {
            'Delta': (0.5, 4, 'purple'),
            'Theta': (4, 8, 'blue'),
            'Alpha': (8, 13, 'green'),
            'Beta': (13, 30, 'orange'),
            'Gamma': (30, 100, 'red')
        }
        
        for band_name, (low, high, color) in bands.items():
            if freq_max is None or high <= freq_max:
                fig.add_hline(
                    y=low, line_dash="dash", line_color=color,
                    annotation_text=f"{band_name} ({low}-{high}Hz)",
                    row=1, col=1
                )
                fig.add_hline(
                    y=low, line_dash="dash", line_color=color,
                    row=1, col=2
                )
    
    fig.update_layout(
        title=f"Enhanced STFT Analysis - Window: {window_size}, Hop: {hop_size}",
        template="plotly_white",
        height=700,
        showlegend=True,
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50, pad=4)
    )
    
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Magnitude", row=2, col=2)
    
    logger.info("Enhanced STFT plot created successfully")
    return fig

def create_wavelet_plot(signal_data, sampling_freq, wavelet_type, levels, freq_min, freq_max):
    """Create enhanced wavelet transform plot with vitalDSP insights."""
    logger.info(f"Creating wavelet plot with type: {wavelet_type}, levels: {levels}")
    
    try:
        # Import wavelet functions
        from scipy.signal import cwt, morlet2
        from scipy.signal import daub, qmf
        
        # Validate input parameters
        if levels <= 0 or levels > 100:
            logger.warning(f"Invalid levels value: {levels}, using default 20")
            levels = 20
        
        if len(signal_data) < 100:
            logger.warning(f"Signal too short for wavelet analysis: {len(signal_data)} samples")
            # Fallback to simple FFT
            return create_simple_frequency_plot(signal_data, sampling_freq, "FFT Fallback")
        
        # Create time-frequency representation
        widths = np.arange(1, levels + 1)
        
        # Apply different wavelet types based on vitalDSP capabilities
        if wavelet_type == "morlet":
            # Morlet wavelet with proper parameters
            def morlet_wavelet(x, width):
                try:
                    return morlet2(x, width, w=6.0)  # w=6.0 for good time-frequency localization
                except Exception as e:
                    logger.warning(f"Morlet wavelet failed with width {width}: {e}, using fallback")
                    # Fallback to simple Gaussian
                    return np.exp(-(x**2) / (2 * width**2))
            
            cwtmatr = cwt(signal_data, widths, morlet_wavelet)
            logger.info("Applied Morlet wavelet for optimal time-frequency localization")
            
        elif wavelet_type == "haar":
            # Haar wavelet implementation
            def haar_wavelet(x, width):
                try:
                    n = len(x)
                    wavelet = np.zeros(n)
                    mid = n // 2
                    wavelet[:mid] = 1
                    wavelet[mid:] = -1
                    return wavelet
                except Exception as e:
                    logger.warning(f"Haar wavelet failed: {e}, using fallback")
                    return np.ones(len(x)) * (1 if width % 2 == 0 else -1)
            
            cwtmatr = cwt(signal_data, widths, haar_wavelet)
            logger.info("Applied Haar wavelet for sharp transitions")
            
        elif wavelet_type == "daubechies":
            # Daubechies wavelet approximation using Ricker wavelet
            from scipy.signal import ricker
            
            def daubechies_wavelet(x, width):
                try:
                    return ricker(len(x), width)
                except Exception as e:
                    logger.warning(f"Daubechies wavelet failed: {e}, using fallback")
                    # Fallback to simple Gaussian
                    return np.exp(-(x**2) / (2 * width**2))
            
            cwtmatr = cwt(signal_data, widths, daubechies_wavelet)
            logger.info("Applied Daubechies wavelet approximation for smooth analysis")
            
        else:
            # Default to Morlet wavelet
            def morlet_wavelet(x, width):
                try:
                    return morlet2(x, width, w=6.0)
                except Exception as e:
                    logger.warning(f"Default Morlet wavelet failed: {e}, using fallback")
                    # Fallback to simple Gaussian
                    return np.exp(-(x**2) / (2 * width**2))
            
            cwtmatr = cwt(signal_data, widths, morlet_wavelet)
            logger.info("Applied default Morlet wavelet")
        
        # Create frequency axis (approximate)
        freqs = sampling_freq / (2 * np.pi * widths)
        
        # Validate wavelet results
        if cwtmatr is None or cwtmatr.size == 0:
            logger.error("Wavelet analysis produced empty results")
            raise ValueError("Wavelet analysis failed to produce valid results")
        
        # Calculate wavelet coefficients and power
        magnitude = np.abs(cwtmatr)
        power = magnitude ** 2
        
        # Additional validation
        if np.any(np.isnan(magnitude)) or np.any(np.isinf(magnitude)):
            logger.warning("Wavelet analysis produced NaN or Inf values, cleaning...")
            magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
            power = magnitude ** 2
        
        # Find dominant time-frequency components
        max_power_idx = np.unravel_index(np.argmax(power), power.shape)
        max_freq = freqs[max_power_idx[0]]
        max_time = max_power_idx[1] / sampling_freq
        max_power = power[max_power_idx]
        
        logger.info(f"Dominant wavelet component: {max_freq:.2f} Hz at {max_time:.2f}s with power {max_power:.2e}")
        
        # Calculate wavelet energy distribution
        energy_per_level = np.sum(power, axis=1)
        total_energy = np.sum(energy_per_level)
        energy_percentages = (energy_per_level / total_energy) * 100
        
        logger.info(f"Wavelet energy distribution across {levels} levels:")
        for i, (freq, energy_pct) in enumerate(zip(freqs, energy_percentages)):
            logger.info(f"  Level {i+1} ({freq:.2f} Hz): {energy_pct:.1f}%")
        
        # Filter frequency range
        if freq_min is not None and freq_max is not None:
            mask = (freqs >= freq_min) & (freqs <= freq_max)
            freqs = freqs[mask]
            cwtmatr = cwtmatr[mask, :]
            magnitude = magnitude[mask, :]
            power = power[mask, :]
            energy_per_level = energy_per_level[mask]
            logger.info(f"Filtered frequency range: {freq_min:.1f} - {freq_max:.1f} Hz")
        
        # Create enhanced plot with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Wavelet Scalogram", "Power Distribution", "Energy per Level", "Time Profile"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Wavelet scalogram
        fig.add_trace(
            go.Heatmap(
                z=magnitude,
                x=np.arange(len(signal_data)) / sampling_freq,
                y=freqs,
                colorscale='Viridis',
                name='Wavelet Magnitude',
                hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Freq:</b> %{y:.2f}Hz<br><b>Magnitude:</b> %{z:.2e}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Power distribution
        fig.add_trace(
            go.Heatmap(
                z=power,
                x=np.arange(len(signal_data)) / sampling_freq,
                y=freqs,
                colorscale='Plasma',
                name='Wavelet Power',
                hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Freq:</b> %{y:.2f}Hz<br><b>Power:</b> %{z:.2e}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Energy per level
        fig.add_trace(
            go.Bar(
                x=[f"{f:.1f} Hz" for f in freqs],
                y=energy_per_level,
                name='Energy per Level',
                marker_color='green',
                hovertemplate='<b>Frequency:</b> %{x}<br><b>Energy:</b> %{y:.2e}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Time profile at dominant frequency
        dominant_freq_idx = np.argmax(energy_per_level)
        dominant_freq = freqs[dominant_freq_idx]
        time_profile = magnitude[dominant_freq_idx, :]
        
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(signal_data)) / sampling_freq,
                y=time_profile,
                mode='lines',
                name=f'Time Profile at {dominant_freq:.2f}Hz',
                line=dict(color='red', width=2),
                hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Magnitude:</b> %{y:.2e}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Add frequency band annotations
        if freq_max is None or freq_max > 30:
            bands = {
                'Delta': (0.5, 4, 'purple'),
                'Theta': (4, 8, 'blue'),
                'Alpha': (8, 13, 'green'),
                'Beta': (13, 30, 'orange'),
                'Gamma': (30, 100, 'red')
            }
            
            for band_name, (low, high, color) in bands.items():
                if freq_max is None or high <= freq_max:
                    fig.add_hline(
                        y=low, line_dash="dash", line_color=color,
                        annotation_text=f"{band_name} ({low}-{high}Hz)",
                        row=1, col=1
                    )
                    fig.add_hline(
                        y=low, line_dash="dash", line_color=color,
                        row=1, col=2
                    )
        
        fig.update_layout(
            title=f"Enhanced Wavelet Analysis - {wavelet_type.title()} ({levels} levels)",
            template="plotly_white",
            height=700,
            showlegend=True,
            autosize=True,
            margin=dict(l=50, r=50, t=80, b=50, pad=4)
        )
        
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=2)
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig.update_yaxes(title_text="Energy", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Magnitude", row=2, col=2)
        
        logger.info("Enhanced wavelet plot created successfully")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating wavelet plot: {e}")
        logger.info("Falling back to simple frequency analysis")
        
        # Fallback to simple FFT visualization
        try:
            from scipy.fft import rfft, rfftfreq
            
            fft_result = np.abs(rfft(signal_data))
            freqs = rfftfreq(len(signal_data), 1/sampling_freq)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=freqs,
                y=fft_result,
                mode='lines',
                name='Frequency Spectrum (Fallback)',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=f"Wavelet Analysis Fallback - {wavelet_type.title()} ({levels} levels)",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude",
                template="plotly_white",
                height=400
            )
            
            return fig
            
        except Exception as fallback_error:
            logger.error(f"Fallback visualization also failed: {fallback_error}")
            return create_empty_figure()

def create_signal_plot(signal_data, time_axis, title):
    """Create enhanced signal plot with vitalDSP insights."""
    logger.info(f"Creating enhanced signal plot: {title}")
    
    # Calculate signal statistics
    signal_mean = np.mean(signal_data)
    signal_std = np.std(signal_data)
    signal_rms = np.sqrt(np.mean(signal_data ** 2))
    signal_peak = np.max(np.abs(signal_data))
    signal_crest_factor = signal_peak / signal_rms if signal_rms > 0 else 0
    
    # Calculate frequency content
    from scipy.fft import rfft, rfftfreq
    fft_result = np.abs(rfft(signal_data))
    freqs = rfftfreq(len(signal_data), time_axis[1] - time_axis[0] if len(time_axis) > 1 else 1)
    
    # Find dominant frequency
    peak_idx = np.argmax(fft_result)
    dominant_freq = freqs[peak_idx]
    
    # Calculate signal quality metrics
    snr_estimate = 20 * np.log10(signal_rms / signal_std) if signal_std > 0 else 0
    
    logger.info(f"Signal stats - Mean: {signal_mean:.3e}, RMS: {signal_rms:.3e}, Peak: {signal_peak:.3e}")
    logger.info(f"Dominant frequency: {dominant_freq:.2f} Hz, Crest factor: {signal_crest_factor:.2f}")
    
    # Create enhanced plot with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(f"{title}", "Signal Statistics", "Frequency Content", "Quality Metrics"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Main signal plot
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=signal_data,
            mode='lines',
            name=title,
            line=dict(color='blue', width=1.5),
            hovertemplate='<b>Time:</b> %{x:.3f}s<br><b>Amplitude:</b> %{y:.3e}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add mean and RMS lines
    fig.add_hline(y=signal_mean, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {signal_mean:.3e}", row=1, col=1)
    fig.add_hline(y=signal_mean + signal_rms, line_dash="dot", line_color="orange",
                  annotation_text=f"+RMS: {signal_mean + signal_rms:.3e}", row=1, col=1)
    fig.add_hline(y=signal_mean - signal_rms, line_dash="dot", line_color="orange",
                  annotation_text=f"-RMS: {signal_mean - signal_rms:.3e}", row=1, col=1)
    
    # Signal statistics
    stats_text = [
        f"Mean: {signal_mean:.3e}",
        f"Std Dev: {signal_std:.3e}",
        f"RMS: {signal_rms:.3e}",
        f"Peak: {signal_peak:.3e}",
        f"Crest Factor: {signal_crest_factor:.2f}",
        f"Dynamic Range: {20*np.log10(signal_peak/signal_std):.1f} dB"
    ]
    
    fig.add_trace(
        go.Scatter(
            x=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            y=[0.9, 0.75, 0.6, 0.45, 0.3, 0.15],
            mode='text',
            text=stats_text,
            textposition='middle left',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Frequency content
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=fft_result,
            mode='lines',
            name='Frequency Spectrum',
            line=dict(color='green', width=1.5),
            hovertemplate='<b>Frequency:</b> %{x:.2f} Hz<br><b>Magnitude:</b> %{y:.3e}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Mark dominant frequency
    fig.add_trace(
        go.Scatter(
            x=[dominant_freq],
            y=[fft_result[peak_idx]],
            mode='markers',
            name='Dominant Freq',
            marker=dict(color='red', size=10, symbol='diamond'),
            hovertemplate=f'<b>Dominant:</b> {dominant_freq:.2f} Hz<br><b>Magnitude:</b> {fft_result[peak_idx]:.3e}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Quality metrics visualization
    metrics = [
        ("SNR Estimate", snr_estimate, "dB"),
        ("Crest Factor", signal_crest_factor, ""),
        ("Dynamic Range", 20*np.log10(signal_peak/signal_std) if signal_std > 0 else 0, "dB"),
        ("Zero Crossings", np.sum(np.diff(np.sign(signal_data)) != 0), "count")
    ]
    
    metric_names = [m[0] for m in metrics]
    metric_values = [m[1] for m in metrics]
    metric_units = [m[2] for m in metrics]
    
    # Create bar chart for metrics
    colors = ['blue', 'green', 'orange', 'red']
    for i, (name, value, unit) in enumerate(metrics):
        fig.add_trace(
            go.Bar(
                x=[name],
                y=[value],
                name=f"{name} ({unit})",
                marker_color=colors[i],
                hovertemplate=f'<b>{name}:</b> {value:.2f} {unit}<extra></extra>'
            ),
            row=2, col=2
        )
    
    # Add frequency band annotations if applicable
    if dominant_freq < 100:  # Only for low frequencies
        bands = {
            'Delta': (0.5, 4, 'purple'),
            'Theta': (4, 8, 'blue'),
            'Alpha': (8, 13, 'green'),
            'Beta': (13, 30, 'orange'),
            'Gamma': (30, 100, 'red')
        }
        
        for band_name, (low, high, color) in bands.items():
            if low <= dominant_freq <= high:
                fig.add_vrect(
                    x0=low, x1=high,
                    fillcolor=color, opacity=0.1,
                    layer="below", line_width=0,
                    annotation_text=f"{band_name} Band",
                    annotation_position="top left",
                    row=2, col=1
                )
    
    fig.update_layout(
        title=f"Enhanced {title} Analysis",
        template="plotly_white",
        height=700,
        showlegend=True,
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50, pad=4)
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text="", row=1, col=2)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude", row=2, col=1)
    fig.update_xaxes(title_text="Metrics", row=2, col=2)
    fig.update_yaxes(title_text="Value", row=2, col=2)
    
    logger.info("Enhanced signal plot created successfully")
    return fig

def create_filter_comparison_plot(original_signal, filtered_signal, time_axis):
    """Create a comparison plot between original and filtered signals."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=original_signal,
        mode='lines',
        name='Original Signal',
        line=dict(color='blue', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=filtered_signal,
        mode='lines',
        name='Filtered Signal',
        line=dict(color='red', width=1)
    ))
    
    fig.update_layout(
        title="Signal Comparison",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        showlegend=True,
        template="plotly_white"
    )
    
    return fig

# Filtering functions
def apply_traditional_filter(signal_data, sampling_freq, filter_family, filter_response, low_freq, high_freq, filter_order):
    """Apply enhanced traditional filter with vitalDSP insights."""
    logger.info(f"Applying {filter_family} {filter_response} filter - Order: {filter_order}")
    logger.info(f"Frequency range: {low_freq:.1f} - {high_freq:.1f} Hz, Sampling: {sampling_freq} Hz")
    
    nyq = 0.5 * sampling_freq
    logger.info(f"Nyquist frequency: {nyq:.1f} Hz")
    
    # Calculate normalized cutoffs
    if filter_response == "bandpass":
        low_cutoff = low_freq / nyq
        high_cutoff = high_freq / nyq
        logger.info(f"Normalized cutoffs: {low_cutoff:.3f} - {high_cutoff:.3f}")
        
        if filter_family == "butter":
            b, a = signal.butter(filter_order, [low_cutoff, high_cutoff], btype='bandpass')
            logger.info("Applied Butterworth bandpass filter - maximally flat magnitude response")
        elif filter_family == "cheby1":
            b, a = signal.cheby1(filter_order, 1, [low_cutoff, high_cutoff], btype='bandpass')
            logger.info("Applied Chebyshev I bandpass filter - equiripple in passband")
        elif filter_family == "cheby2":
            b, a = signal.cheby2(filter_order, 20, [low_cutoff, high_cutoff], btype='bandpass')
            logger.info("Applied Chebyshev II bandpass filter - equiripple in stopband")
        elif filter_family == "ellip":
            b, a = signal.ellip(filter_order, 1, 20, [low_cutoff, high_cutoff], btype='bandpass')
            logger.info("Applied Elliptic bandpass filter - equiripple in both bands")
        elif filter_family == "bessel":
            b, a = signal.bessel(filter_order, [low_cutoff, high_cutoff], btype='bandpass')
            logger.info("Applied Bessel bandpass filter - maximally flat group delay")
        else:
            b, a = signal.butter(filter_order, [low_cutoff, high_cutoff], btype='bandpass')
            logger.info("Applied default Butterworth bandpass filter")
            
    elif filter_response == "lowpass":
        cutoff = high_freq / nyq
        logger.info(f"Normalized cutoff: {cutoff:.3f}")
        
        if filter_family == "butter":
            b, a = signal.butter(filter_order, cutoff, btype='lowpass')
        elif filter_family == "cheby1":
            b, a = signal.cheby1(filter_order, 1, cutoff, btype='lowpass')
        elif filter_family == "cheby2":
            b, a = signal.cheby2(filter_order, 20, cutoff, btype='lowpass')
        elif filter_family == "ellip":
            b, a = signal.ellip(filter_order, 1, 20, cutoff, btype='lowpass')
        elif filter_family == "bessel":
            b, a = signal.bessel(filter_order, cutoff, btype='lowpass')
        else:
            b, a = signal.butter(filter_order, cutoff, btype='lowpass')
            
    elif filter_response == "highpass":
        cutoff = low_freq / nyq
        logger.info(f"Normalized cutoff: {cutoff:.3f}")
        
        if filter_family == "butter":
            b, a = signal.butter(filter_order, cutoff, btype='highpass')
        elif filter_family == "cheby1":
            b, a = signal.cheby1(filter_order, 1, cutoff, btype='highpass')
        elif filter_family == "cheby2":
            b, a = signal.cheby2(filter_order, 20, cutoff, btype='highpass')
        elif filter_family == "ellip":
            b, a = signal.ellip(filter_order, 1, 20, cutoff, btype='highpass')
        elif filter_family == "bessel":
            b, a = signal.bessel(filter_order, cutoff, btype='highpass')
        else:
            b, a = signal.butter(filter_order, cutoff, btype='highpass')
    else:
        logger.warning(f"Unknown filter response: {filter_response}")
        return signal_data
    
    # Calculate filter characteristics
    w, h = signal.freqz(b, a, worN=1024)
    freqs = w * sampling_freq / (2 * np.pi)
    magnitude = np.abs(h)
    
    # Find -3dB point
    db_magnitude = 20 * np.log10(magnitude)
    minus_3db_idx = np.argmin(np.abs(db_magnitude + 3))
    actual_cutoff = freqs[minus_3db_idx]
    
    logger.info(f"Actual -3dB cutoff: {actual_cutoff:.2f} Hz")
    logger.info(f"Filter coefficients - b: {len(b)}, a: {len(a)}")
    
    # Apply zero-phase filtering
    filtered_signal = signal.filtfilt(b, a, signal_data)
    
    # Calculate filter performance metrics
    original_energy = np.sum(signal_data ** 2)
    filtered_energy = np.sum(filtered_signal ** 2)
    energy_preservation = (filtered_energy / original_energy) * 100
    
    logger.info(f"Energy preservation: {energy_preservation:.1f}%")
    logger.info(f"Original RMS: {np.sqrt(np.mean(signal_data**2)):.3e}")
    logger.info(f"Filtered RMS: {np.sqrt(np.mean(filtered_signal**2)):.3e}")
    
    return filtered_signal

def apply_advanced_filter(signal_data, method, noise_level, iterations, learning_rate):
    """Apply enhanced advanced filtering with vitalDSP insights."""
    logger.info(f"Applying advanced filter: {method} with {iterations} iterations")
    logger.info(f"Learning rate: {learning_rate}, Noise level: {noise_level}")
    
    n = len(signal_data)
    logger.info(f"Signal length: {n} samples")
    
    if method == "kalman":
        logger.info("Implementing Kalman filter for optimal state estimation")
        
        # Enhanced Kalman filter implementation
        filtered = np.zeros(n)
        filtered[0] = signal_data[0]
        
        # Kalman filter parameters
        Q = noise_level  # Process noise covariance
        R = noise_level * 0.1  # Measurement noise covariance
        P = 1.0  # Initial state covariance
        
        for i in range(1, n):
            # Prediction step
            x_pred = filtered[i-1]
            P_pred = P + Q
            
            # Update step
            K = P_pred / (P_pred + R)  # Kalman gain
            filtered[i] = x_pred + K * (signal_data[i] - x_pred)
            P = (1 - K) * P_pred
        
        logger.info("Kalman filter applied successfully")
        
    elif method == "wiener":
        logger.info("Implementing Wiener filter for optimal linear estimation")
        
        # Wiener filter implementation
        from scipy.signal import wiener
        filtered = wiener(signal_data, noise_level)
        logger.info("Wiener filter applied successfully")
        
    elif method == "savgol":
        logger.info("Implementing Savitzky-Golay filter for smooth derivatives")
        
        # Savitzky-Golay filter
        window_length = min(51, n // 4 * 2 + 1)  # Must be odd
        filtered = signal.savgol_filter(signal_data, window_length, 3)
        logger.info(f"Savitzky-Golay filter applied with window length {window_length}")
        
    elif method == "medfilt":
        logger.info("Implementing median filter for impulse noise removal")
        
        # Median filter
        kernel_size = max(3, int(noise_level * 10))
        # Ensure kernel size is odd for median filter
        if kernel_size % 2 == 0:
            kernel_size += 1
        # Ensure kernel size doesn't exceed signal length
        kernel_size = min(kernel_size, len(signal_data) - 1)
        if kernel_size < 3:
            kernel_size = 3
        filtered = signal.medfilt(signal_data, kernel_size)
        logger.info(f"Median filter applied with kernel size {kernel_size}")
        
    elif method == "gaussian":
        logger.info("Implementing Gaussian filter for smooth denoising")
        
        # Gaussian filter
        sigma = noise_level * 2
        filtered = signal.gaussian_filter1d(signal_data, sigma)
        logger.info(f"Gaussian filter applied with sigma {sigma:.2f}")
        
    else:
        logger.warning(f"Unknown advanced filter method: {method}")
        return signal_data
    
    # Calculate filter performance metrics
    original_energy = np.sum(signal_data ** 2)
    filtered_energy = np.sum(filtered ** 2)
    energy_preservation = (filtered_energy / original_energy) * 100
    
    # Calculate SNR improvement
    noise_original = np.std(signal_data - np.mean(signal_data))
    noise_filtered = np.std(filtered - np.mean(filtered))
    snr_improvement = 20 * np.log10(noise_original / noise_filtered) if noise_filtered > 0 else 0
    
    logger.info(f"Energy preservation: {energy_preservation:.1f}%")
    logger.info(f"SNR improvement: {snr_improvement:.2f} dB")
    logger.info(f"Original noise level: {noise_original:.3e}")
    logger.info(f"Filtered noise level: {noise_filtered:.3e}")
    
    return filtered

def apply_artifact_removal(signal_data, sampling_freq, artifact_type, strength):
    """Apply enhanced artifact removal with vitalDSP insights."""
    logger.info(f"Applying artifact removal: {artifact_type} with strength {strength}")
    logger.info(f"Signal length: {len(signal_data)} samples, Sampling: {sampling_freq} Hz")
    
    # Validate input parameters
    if strength <= 0 or strength > 10:
        logger.warning(f"Invalid strength value: {strength}, using default 1.0")
        strength = 1.0
    
    if sampling_freq <= 0:
        logger.error(f"Invalid sampling frequency: {sampling_freq}")
        return signal_data
    
    if len(signal_data) < 10:
        logger.warning(f"Signal too short for artifact removal: {len(signal_data)} samples")
        return signal_data

    if artifact_type == "baseline":
        logger.info("Removing baseline wander using median filtering")
        
        # Enhanced baseline correction
        kernel_size = max(3, int(sampling_freq / 10))
        # Ensure kernel size is odd for median filter
        if kernel_size % 2 == 0:
            kernel_size += 1
        # Ensure kernel size doesn't exceed signal length
        kernel_size = min(kernel_size, len(signal_data) - 1)
        if kernel_size < 3:
            kernel_size = 3
        baseline = signal.medfilt(signal_data, kernel_size)
        
        # Adaptive baseline correction
        corrected_signal = signal_data - strength * baseline
        
        # Calculate baseline metrics
        baseline_amplitude = np.max(baseline) - np.min(baseline)
        correction_factor = strength * baseline_amplitude / np.std(signal_data)
        
        logger.info(f"Baseline kernel size: {kernel_size}")
        logger.info(f"Baseline amplitude: {baseline_amplitude:.3e}")
        logger.info(f"Correction factor: {correction_factor:.2f}")
        
        return corrected_signal
        
    elif artifact_type == "powerline":
        logger.info("Removing power line interference (50/60 Hz)")
        
        # Notch filter for power line interference
        if sampling_freq > 100:  # Only if sampling is high enough
            # Detect power line frequency
            powerline_freq = 50 if sampling_freq < 120 else 60
            logger.info(f"Detected power line frequency: {powerline_freq} Hz")
            
            # Create notch filter
            nyq = sampling_freq / 2
            low_cutoff = (powerline_freq - 2) / nyq
            high_cutoff = (powerline_freq + 2) / nyq
            b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='bandstop')
            
            filtered_signal = signal.filtfilt(b, a, signal_data)
            logger.info("Power line interference removed with notch filter")
            
            return filtered_signal
        else:
            logger.warning("Sampling frequency too low for power line removal")
            return signal_data
            
    elif artifact_type == "motion":
        logger.info("Removing motion artifacts using high-pass filtering")
        
        # High-pass filter to remove slow motion artifacts
        cutoff = 0.5 / (sampling_freq / 2)  # 0.5 Hz cutoff
        b, a = signal.butter(4, cutoff, btype='highpass')
        
        filtered_signal = signal.filtfilt(b, a, signal_data)
        logger.info(f"Motion artifacts removed with {cutoff:.3f} Hz high-pass filter")
        
        return filtered_signal
        
    elif artifact_type == "muscle":
        logger.info("Removing muscle noise using bandpass filtering")
        
        # Bandpass filter for physiological signals (0.5-40 Hz)
        low_cutoff = 0.5 / (sampling_freq / 2)
        high_cutoff = 40 / (sampling_freq / 2)
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='bandpass')
        
        filtered_signal = signal.filtfilt(b, a, signal_data)
        logger.info("Muscle noise removed with 0.5-40 Hz bandpass filter")
        
        return filtered_signal
        
    elif artifact_type == "electrode":
        logger.info("Removing electrode noise using median filtering")
        
        # Median filter for impulse noise
        kernel_size = max(3, int(sampling_freq / 100))
        # Ensure kernel size is odd for median filter
        if kernel_size % 2 == 0:
            kernel_size += 1
        # Ensure kernel size doesn't exceed signal length
        kernel_size = min(kernel_size, len(signal_data) - 1)
        if kernel_size < 3:
            kernel_size = 3
        filtered_signal = signal.medfilt(signal_data, kernel_size)
        
        logger.info(f"Electrode noise removed with median filter (kernel: {kernel_size})")
        
        return filtered_signal
        
    else:
        logger.warning(f"Unknown artifact type: {artifact_type}")
        return signal_data

def apply_neural_filter(signal_data, network_type, complexity):
    """Apply enhanced neural network filtering with vitalDSP insights."""
    logger.info(f"Applying neural network filter: {network_type} with complexity {complexity}")
    logger.info(f"Signal length: {len(signal_data)} samples")
    
    if network_type == "autoencoder":
        logger.info("Implementing autoencoder-based denoising")
        
        # Simple autoencoder approximation using signal processing
        # In a real implementation, this would use actual neural networks
        window_size = max(3, complexity * 2)
        
        # Denoise using bilateral filtering (approximates neural network behavior)
        # Ensure window size is odd for median filter
        if window_size % 2 == 0:
            window_size += 1
        # Ensure window size doesn't exceed signal length
        window_size = min(window_size, len(signal_data) - 1)
        if window_size < 3:
            window_size = 3
        filtered_signal = signal.medfilt(signal_data, window_size)
        
        # Apply adaptive thresholding
        threshold = np.std(filtered_signal) * complexity * 0.1
        filtered_signal = np.where(np.abs(filtered_signal) < threshold, 0, filtered_signal)
        
        logger.info(f"Autoencoder filter applied with window size {window_size}")
        
    elif network_type == "cnn":
        logger.info("Implementing CNN-based filtering")
        
        # CNN approximation using convolution operations
        kernel_size = max(3, complexity * 3)
        
        # Apply multiple convolution kernels
        filtered_signal = signal_data.copy()
        for i in range(complexity):
            kernel = np.ones(kernel_size) / kernel_size
            filtered_signal = np.convolve(filtered_signal, kernel, mode='same')
        
        logger.info(f"CNN filter applied with {complexity} convolution layers")
        
    elif network_type == "lstm":
        logger.info("Implementing LSTM-based temporal filtering")
        
        # LSTM approximation using exponential smoothing
        alpha = 1.0 / (complexity + 1)
        filtered_signal = np.zeros_like(signal_data)
        filtered_signal[0] = signal_data[0]
        
        for i in range(1, len(signal_data)):
            filtered_signal[i] = alpha * signal_data[i] + (1 - alpha) * filtered_signal[i-1]
        
        logger.info(f"LSTM filter applied with smoothing factor {alpha:.3f}")
        
    elif network_type == "transformer":
        logger.info("Implementing Transformer-based attention filtering")
        
        # Transformer approximation using attention-like weighting
        window_size = max(5, complexity * 4)
        filtered_signal = np.zeros_like(signal_data)
        
        for i in range(len(signal_data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(signal_data), i + window_size // 2 + 1)
            
            # Attention weights based on distance
            weights = np.exp(-np.abs(np.arange(start_idx, end_idx) - i) / complexity)
            weights = weights / np.sum(weights)
            
            filtered_signal[i] = np.sum(signal_data[start_idx:end_idx] * weights)
        
        logger.info(f"Transformer filter applied with attention window {window_size}")
        
    else:
        logger.warning(f"Unknown neural network type: {network_type}")
        return signal_data
    
    # Calculate neural network performance metrics
    original_energy = np.sum(signal_data ** 2)
    filtered_energy = np.sum(filtered_signal ** 2)
    energy_preservation = (filtered_energy / original_energy) * 100
    
    # Calculate complexity-based performance
    complexity_score = complexity / 5.0  # Normalize to 0-1
    performance_metric = energy_preservation * complexity_score
    
    logger.info(f"Energy preservation: {energy_preservation:.1f}%")
    logger.info(f"Complexity score: {complexity_score:.2f}")
    logger.info(f"Performance metric: {performance_metric:.1f}")
    
    return filtered_signal

def apply_ensemble_filter(signal_data, method, n_filters):
    """Apply enhanced ensemble filtering with vitalDSP insights."""
    logger.info(f"Applying ensemble filter: {method} with {n_filters} filters")
    logger.info(f"Signal length: {len(signal_data)} samples")
    
    # Create multiple filtered versions using different approaches
    filtered_versions = []
    filter_types = []
    
    # Different filter families and parameters
    filter_configs = [
        ("butter", "lowpass", 0.1),
        ("cheby1", "lowpass", 0.15),
        ("ellip", "lowpass", 0.2),
        ("bessel", "lowpass", 0.25),
        ("butter", "bandpass", [0.05, 0.3]),
        ("cheby2", "highpass", 0.05),
        ("gaussian", None, 2.0),
        ("median", None, 5),
        ("savgol", None, 11)
    ]
    
    for i in range(min(n_filters, len(filter_configs))):
        filter_family, filter_type, params = filter_configs[i]
        filter_types.append(f"{filter_family}_{filter_type}")
        
        try:
            if filter_family == "butter":
                if isinstance(params, list):
                    b, a = signal.butter(4, params, btype=filter_type)
                else:
                    b, a = signal.butter(4, params, btype=filter_type)
                filtered = signal.filtfilt(b, a, signal_data)
                
            elif filter_family == "cheby1":
                if isinstance(params, list):
                    b, a = signal.cheby1(4, 1, params, btype=filter_type)
                else:
                    b, a = signal.cheby1(4, 1, params, btype=filter_type)
                filtered = signal.filtfilt(b, a, signal_data)
                
            elif filter_family == "ellip":
                if isinstance(params, list):
                    b, a = signal.ellip(4, 1, 20, params, btype=filter_type)
                else:
                    b, a = signal.ellip(4, 1, 20, params, btype=filter_type)
                filtered = signal.filtfilt(b, a, signal_data)
                
            elif filter_family == "bessel":
                if isinstance(params, list):
                    b, a = signal.bessel(4, params, btype=filter_type)
                else:
                    b, a = signal.bessel(4, params, btype=filter_type)
                filtered = signal.filtfilt(b, a, signal_data)
                
            elif filter_family == "gaussian":
                filtered = signal.gaussian_filter1d(signal_data, params)
                
            elif filter_family == "median":
                kernel_size = int(params)
                # Ensure kernel size is odd for median filter
                if kernel_size % 2 == 0:
                    kernel_size += 1
                # Ensure kernel size doesn't exceed signal length
                kernel_size = min(kernel_size, len(signal_data) - 1)
                if kernel_size < 3:
                    kernel_size = 3
                filtered = signal.medfilt(signal_data, kernel_size)
                
            elif filter_family == "savgol":
                filtered = signal.savgol_filter(signal_data, int(params), 3)
                
            else:
                filtered = signal_data
                
            filtered_versions.append(filtered)
            logger.info(f"Applied {filter_family} filter successfully")
            
        except Exception as e:
            logger.warning(f"Failed to apply {filter_family} filter: {e}")
            filtered_versions.append(signal_data)
    
    if not filtered_versions:
        logger.error("No filters were successfully applied")
        return signal_data
    
    logger.info(f"Successfully created {len(filtered_versions)} filtered versions")
    
    # Combine using selected method
    if method == "mean":
        logger.info("Combining filters using mean method")
        combined_signal = np.mean(filtered_versions, axis=0)
        
    elif method == "median":
        logger.info("Combining filters using median method")
        combined_signal = np.median(filtered_versions, axis=0)
        
    elif method == "weighted_mean":
        logger.info("Combining filters using weighted mean method")
        # Weight based on filter performance
        weights = []
        for filtered in filtered_versions:
            # Calculate weight based on energy preservation
            energy_ratio = np.sum(filtered ** 2) / np.sum(signal_data ** 2)
            weight = 1.0 / (1.0 + abs(energy_ratio - 1.0))  # Higher weight for better preservation
            weights.append(weight)
        
        weights = np.array(weights) / np.sum(weights)
        combined_signal = np.average(filtered_versions, axis=0, weights=weights)
        
        logger.info(f"Filter weights: {weights}")
        
    elif method == "bagging":
        logger.info("Combining filters using bagging method")
        # Bootstrap aggregation
        n_samples = len(signal_data)
        combined_signal = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Randomly select filters for each sample
            selected_filters = np.random.choice(len(filtered_versions), size=len(filtered_versions)//2, replace=False)
            combined_signal[i] = np.mean([filtered_versions[j][i] for j in selected_filters])
            
    elif method == "boosting":
        logger.info("Combining filters using boosting method")
        # Adaptive boosting
        weights = np.ones(len(filtered_versions))
        combined_signal = np.zeros_like(signal_data)
        
        for iteration in range(3):  # 3 boosting iterations
            # Weighted combination
            combined_signal = np.average(filtered_versions, axis=0, weights=weights)
            
            # Update weights based on performance
            for i, filtered in enumerate(filtered_versions):
                error = np.mean((filtered - combined_signal) ** 2)
                weights[i] *= np.exp(-error)
            
            weights = weights / np.sum(weights)
            
        logger.info(f"Final filter weights: {weights}")
        
    else:
        logger.warning(f"Unknown ensemble method: {method}, using mean")
        combined_signal = np.mean(filtered_versions, axis=0)
    
    # Calculate ensemble performance metrics
    original_energy = np.sum(signal_data ** 2)
    combined_energy = np.sum(combined_signal ** 2)
    energy_preservation = (combined_energy / original_energy) * 100
    
    # Calculate diversity metric
    diversity = np.mean([np.corrcoef(signal_data, filtered)[0, 1] for filtered in filtered_versions])
    
    logger.info(f"Ensemble energy preservation: {energy_preservation:.1f}%")
    logger.info(f"Filter diversity: {diversity:.3f}")
    logger.info(f"Combined {len(filtered_versions)} filters using {method} method")
    
    return combined_signal

# Analysis result functions
def generate_frequency_analysis_results(signal_data, sampling_freq, analysis_type, analysis_options):
    """Generate enhanced frequency analysis results with vitalDSP insights."""
    logger.info("Generating comprehensive frequency analysis results")
    
    results = []
    insights = []
    
    # Basic frequency analysis
    fft_result = np.abs(rfft(signal_data))
    freqs = rfftfreq(len(signal_data), 1/sampling_freq)
    power_spectrum = fft_result ** 2
    
    if "peak_freq" in analysis_options:
        # Find peak frequency
        peak_idx = np.argmax(fft_result)
        peak_freq = freqs[peak_idx]
        peak_magnitude = fft_result[peak_idx]
        results.append(f"Peak Frequency: {peak_freq:.2f} Hz")
        results.append(f"Peak Magnitude: {peak_magnitude:.2e}")
        
        # Frequency classification
        if peak_freq < 0.5:
            insights.append("Ultra-low frequency component detected")
        elif peak_freq < 4:
            insights.append("Delta wave dominance (deep sleep)")
        elif peak_freq < 8:
            insights.append("Theta wave dominance (light sleep)")
        elif peak_freq < 13:
            insights.append("Alpha wave dominance (relaxed state)")
        elif peak_freq < 30:
            insights.append("Beta wave dominance (active state)")
        else:
            insights.append("Gamma wave dominance (high cognitive activity)")
    
    if "dominant_freq" in analysis_options:
        # Calculate dominant frequency
        dominant_freq = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
        results.append(f"Dominant Frequency: {dominant_freq:.2f} Hz")
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
        results.append(f"Spectral Centroid: {spectral_centroid:.2f} Hz")
    
    if "band_power" in analysis_options:
        # Calculate band power with enhanced bands
        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 100),
            'Ultra-Low': (0.01, 0.5)
        }
        
        band_powers = {}
        total_power = np.sum(power_spectrum)
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                power = np.sum(power_spectrum[mask])
                percentage = (power / total_power) * 100
                band_powers[band_name] = (power, percentage)
                results.append(f"{band_name} Power ({low}-{high} Hz): {power:.2e} ({percentage:.1f}%)")
        
        # Find dominant band
        if band_powers:
            dominant_band = max(band_powers.items(), key=lambda x: x[1][1])
            insights.append(f"Dominant frequency band: {dominant_band[0]} ({dominant_band[1][1]:.1f}%)")
    
    if "spectral_features" in analysis_options:
        # Spectral rolloff
        cumulative_power = np.cumsum(power_spectrum)
        rolloff_threshold = 0.85  # 85% of total power
        rolloff_idx = np.argmax(cumulative_power >= rolloff_threshold * cumulative_power[-1])
        spectral_rolloff = freqs[rolloff_idx]
        results.append(f"Spectral Rolloff (85%): {spectral_rolloff:.2f} Hz")
        
        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power_spectrum) / np.sum(power_spectrum))
        results.append(f"Spectral Bandwidth: {spectral_bandwidth:.2f} Hz")
        
        # Spectral flatness
        geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))
        arithmetic_mean = np.mean(power_spectrum)
        spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        results.append(f"Spectral Flatness: {spectral_flatness:.3f}")
        
        # Spectral kurtosis
        spectral_kurtosis = np.mean(((power_spectrum - np.mean(power_spectrum)) / np.std(power_spectrum)) ** 4)
        results.append(f"Spectral Kurtosis: {spectral_kurtosis:.2f}")
    
    if "harmonic_analysis" in analysis_options:
        # Harmonic analysis
        fundamental_idx = np.argmax(fft_result)
        fundamental_freq = freqs[fundamental_idx]
        
        harmonics = []
        for i in range(2, 6):  # 2nd to 5th harmonic
            harmonic_freq = fundamental_freq * i
            harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
            harmonic_magnitude = fft_result[harmonic_idx]
            harmonics.append((i, harmonic_freq, harmonic_magnitude))
        
        results.append(f"Fundamental Frequency: {fundamental_freq:.2f} Hz")
        for i, freq, mag in harmonics:
            results.append(f"{i}nd Harmonic: {freq:.2f} Hz (mag: {mag:.2e})")
        
        # Harmonic distortion
        if harmonics:
            total_harmonic_power = sum(mag ** 2 for _, _, mag in harmonics)
            fundamental_power = fft_result[fundamental_idx] ** 2
            thd = np.sqrt(total_harmonic_power / fundamental_power) * 100
            results.append(f"Total Harmonic Distortion: {thd:.2f}%")
    
    if "stability_metrics" in analysis_options:
        # Frequency stability metrics
        if analysis_type == "stft":
            # For STFT, calculate frequency variation over time
            from scipy.signal import stft
            freqs_stft, times, Zxx = stft(signal_data, sampling_freq, nperseg=min(256, len(signal_data)//4))
            
            # Calculate frequency stability
            peak_freqs = []
            for i in range(Zxx.shape[1]):
                peak_idx = np.argmax(np.abs(Zxx[:, i]))
                peak_freqs.append(freqs_stft[peak_idx])
            
            freq_std = np.std(peak_freqs)
            freq_range = np.max(peak_freqs) - np.min(peak_freqs)
            
            results.append(f"Frequency Stability (std): {freq_std:.2f} Hz")
            results.append(f"Frequency Range: {freq_range:.2f} Hz")
            
            if freq_std < 1.0:
                insights.append("High frequency stability detected")
            elif freq_std < 5.0:
                insights.append("Moderate frequency stability")
            else:
                insights.append("Low frequency stability - signal may be non-stationary")
    
    # Create enhanced results display
    results_html = html.Div([
        html.H5("🎯 Frequency Analysis Results", className="mb-3"),
        
        # Results section
        html.Div([
            html.H6("📊 Quantitative Results", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Metric", className="text-center"),
                        html.Th("Value", className="text-center"),
                        html.Th("Unit", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(result.split(":")[0], className="fw-bold"),
                        html.Td(result.split(":")[1].strip(), className="text-end"),
                        html.Td("Hz" if "Hz" in result else "dB" if "dB" in result else "Count" if "Count" in result else "Ratio", className="text-muted")
                    ]) for result in results
                ])
            ], bordered=True, hover=True, responsive=True, className="mb-3")
        ], className="mb-4"),
        
        # Insights section
        html.Div([
            html.H6("💡 vitalDSP Insights", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Insight Type", className="text-center"),
                        html.Th("Description", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("🔍 Analysis", className="fw-bold text-info"),
                        html.Td(insight, className="text-start")
                    ]) for insight in insights
                ])
            ], bordered=True, hover=True, responsive=True, className="mb-3", color="info")
        ], className="mb-3"),
        
        # Analysis type info
        html.Div([
            html.H6("🔍 Analysis Configuration", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Parameter", className="text-center"),
                        html.Th("Value", className="text-center"),
                        html.Th("Unit", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Analysis Type", className="fw-bold"),
                        html.Td(analysis_type.upper(), className="text-center"),
                        html.Td("-", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Sampling Frequency", className="fw-bold"),
                        html.Td(f"{sampling_freq}", className="text-end"),
                        html.Td("Hz", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Signal Length", className="fw-bold"),
                        html.Td(f"{len(signal_data)}", className="text-end"),
                        html.Td("Samples", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Time Duration", className="fw-bold"),
                        html.Td(f"{len(signal_data)/sampling_freq:.2f}", className="text-end"),
                        html.Td("Seconds", className="text-muted")
                    ])
                ])
            ], bordered=True, hover=True, responsive=True, className="mb-3", color="light")
        ], className="text-muted")
    ])
    
    logger.info(f"Generated {len(results)} analysis results with {len(insights)} insights")
    return results_html

def generate_filter_quality_metrics(original_signal, filtered_signal, sampling_freq, quality_options):
    """Generate enhanced filter quality metrics with vitalDSP insights."""
    logger.info("Generating comprehensive filter quality metrics")
    
    metrics = []
    insights = []
    
    # Basic signal statistics
    original_rms = np.sqrt(np.mean(original_signal ** 2))
    filtered_rms = np.sqrt(np.mean(filtered_signal ** 2))
    original_energy = np.sum(original_signal ** 2)
    filtered_energy = np.sum(filtered_signal ** 2)
    
    logger.info(f"Original RMS: {original_rms:.3e}, Filtered RMS: {filtered_rms:.3e}")
    logger.info(f"Original Energy: {original_energy:.3e}, Filtered Energy: {filtered_energy:.3e}")
    
    if "snr" in quality_options:
        # Calculate SNR improvement
        noise_original = np.std(original_signal - np.mean(original_signal))
        noise_filtered = np.std(filtered_signal - np.mean(filtered_signal))
        
        if noise_filtered > 0:
            snr_improvement = 20 * np.log10(noise_original / noise_filtered)
            metrics.append(f"SNR Improvement: {snr_improvement:.2f} dB")
            
            # SNR classification
            if snr_improvement > 10:
                insights.append("Excellent noise reduction achieved")
            elif snr_improvement > 5:
                insights.append("Good noise reduction performance")
            elif snr_improvement > 0:
                insights.append("Moderate noise reduction")
            else:
                insights.append("Warning: Filter may be amplifying noise")
        else:
            metrics.append("SNR Improvement: Undefined (filtered noise too low)")
    
    if "artifact_reduction" in quality_options:
        # Calculate artifact reduction using multiple metrics
        # High-frequency content reduction
        artifact_original = np.sum(np.abs(np.diff(original_signal)))
        artifact_filtered = np.sum(np.abs(np.diff(filtered_signal)))
        reduction = (artifact_original - artifact_filtered) / artifact_original * 100
        metrics.append(f"Artifact Reduction (High-freq): {reduction:.1f}%")
        
        # Peak-to-peak reduction
        pp_original = np.max(original_signal) - np.min(original_signal)
        pp_filtered = np.max(filtered_signal) - np.min(filtered_signal)
        pp_reduction = (pp_original - pp_filtered) / pp_original * 100
        metrics.append(f"Peak-to-Peak Reduction: {pp_reduction:.1f}%")
        
        # Kurtosis change (measure of outlier reduction)
        kurt_original = np.mean(((original_signal - np.mean(original_signal)) / np.std(original_signal)) ** 4)
        kurt_filtered = np.mean(((filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)) ** 4)
        kurt_change = ((kurt_original - kurt_filtered) / kurt_original) * 100
        metrics.append(f"Kurtosis Change: {kurt_change:.1f}%")
        
        if reduction > 50:
            insights.append("Significant artifact removal achieved")
        elif reduction > 20:
            insights.append("Moderate artifact reduction")
        else:
            insights.append("Limited artifact reduction")
    
    if "distortion" in quality_options:
        # Calculate signal distortion using multiple metrics
        mse = np.mean((original_signal - filtered_signal) ** 2)
        metrics.append(f"Mean Square Error: {mse:.2e}")
        
        # Normalized MSE
        nmse = mse / np.var(original_signal)
        metrics.append(f"Normalized MSE: {nmse:.4f}")
        
        # Correlation coefficient
        correlation = np.corrcoef(original_signal, filtered_signal)[0, 1]
        metrics.append(f"Signal Correlation: {correlation:.3f}")
        
        # Phase distortion (using Hilbert transform)
        try:
            from scipy.signal import hilbert
            original_analytic = hilbert(original_signal)
            filtered_analytic = hilbert(filtered_signal)
            
            original_phase = np.angle(original_analytic)
            filtered_phase = np.angle(filtered_analytic)
            
            phase_diff = np.mean(np.abs(original_phase - filtered_phase))
            metrics.append(f"Phase Distortion: {phase_diff:.3f} rad")
            
            if phase_diff < 0.1:
                insights.append("Minimal phase distortion")
            elif phase_diff < 0.5:
                insights.append("Moderate phase distortion")
            else:
                insights.append("Significant phase distortion detected")
        except:
            metrics.append("Phase Distortion: Unable to calculate")
        
        # Distortion classification
        if nmse < 0.01:
            insights.append("Excellent signal preservation")
        elif nmse < 0.1:
            insights.append("Good signal preservation")
        elif nmse < 0.5:
            insights.append("Moderate signal distortion")
        else:
            insights.append("High signal distortion - filter may be too aggressive")
    
    if "energy_preservation" in quality_options:
        # Energy preservation metrics
        energy_preservation = (filtered_energy / original_energy) * 100
        metrics.append(f"Energy Preservation: {energy_preservation:.1f}%")
        
        # Power spectral density comparison
        from scipy.fft import rfft, rfftfreq
        original_psd = np.abs(rfft(original_signal)) ** 2
        filtered_psd = np.abs(rfft(filtered_signal)) ** 2
        
        psd_correlation = np.corrcoef(original_psd, filtered_psd)[0, 1]
        metrics.append(f"PSD Correlation: {psd_correlation:.3f}")
        
        if energy_preservation > 95:
            insights.append("Excellent energy preservation")
        elif energy_preservation > 80:
            insights.append("Good energy preservation")
        elif energy_preservation > 60:
            insights.append("Moderate energy loss")
        else:
            insights.append("Significant energy loss - filter may be too aggressive")
    
    if "computational_cost" in quality_options:
        # Computational cost estimation
        signal_length = len(original_signal)
        complexity_factor = signal_length / 1000  # Normalize to 1k samples
        
        if complexity_factor < 1:
            cost = "Low"
        elif complexity_factor < 10:
            cost = "Medium"
        else:
            cost = "High"
        
        metrics.append(f"Computational Cost: {cost}")
        metrics.append(f"Complexity Factor: {complexity_factor:.2f}")
        
        if cost == "Low":
            insights.append("Efficient processing - suitable for real-time applications")
        elif cost == "Medium":
            insights.append("Moderate computational load")
        else:
            insights.append("High computational load - consider optimization")
    
    if "filter_performance" in quality_options:
        # Overall filter performance score
        scores = []
        
        # SNR score (0-100)
        if "snr" in quality_options and 'snr_improvement' in locals():
            snr_score = min(100, max(0, (snr_improvement + 20) * 5))  # -20dB to 0dB maps to 0-100
            scores.append(snr_score)
        
        # Distortion score (0-100)
        if 'nmse' in locals():
            distortion_score = max(0, 100 - nmse * 1000)  # 0.1 NMSE = 0 score
            scores.append(distortion_score)
        
        # Energy preservation score
        if 'energy_preservation' in locals():
            scores.append(energy_preservation)
        
        if scores:
            overall_score = np.mean(scores)
            metrics.append(f"Overall Performance Score: {overall_score:.1f}/100")
            
            if overall_score > 80:
                insights.append("Excellent filter performance")
            elif overall_score > 60:
                insights.append("Good filter performance")
            elif overall_score > 40:
                insights.append("Moderate filter performance")
            else:
                insights.append("Poor filter performance - consider parameter adjustment")
    
    # Create enhanced metrics display
    metrics_html = html.Div([
        html.H5("🔍 Filter Quality Metrics", className="mb-3"),
        
        # Metrics section
        html.Div([
            html.H6("📊 Quantitative Metrics", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Metric", className="text-center"),
                        html.Th("Value", className="text-center"),
                        html.Th("Unit", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(metric.split(":")[0], className="fw-bold"),
                        html.Td(metric.split(":")[1].strip(), className="text-end"),
                        html.Td("dB" if "dB" in metric else "%" if "%" in metric else "Ratio" if "Ratio" in metric else "Score" if "Score" in metric else "Error" if "Error" in metric else "-", className="text-muted")
                    ]) for metric in metrics
                ])
            ], bordered=True, hover=True, responsive=True, className="mb-3")
        ], className="mb-4"),
        
        # Insights section
        html.Div([
            html.H6("💡 vitalDSP Insights", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Insight Type", className="text-center"),
                        html.Th("Description", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("🔍 Analysis", className="fw-bold text-info"),
                        html.Td(insight, className="text-start")
                    ]) for insight in insights
                ])
            ], bordered=True, hover=True, responsive=True, className="mb-3", color="info")
        ], className="mb-3"),
        
        # Signal comparison
        html.Div([
            html.H6("📈 Signal Comparison", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Signal Type", className="text-center"),
                        html.Th("Samples", className="text-center"),
                        html.Th("RMS", className="text-center"),
                        html.Th("Energy", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Original Signal", className="fw-bold"),
                        html.Td(f"{len(original_signal)}", className="text-end"),
                        html.Td(f"{original_rms:.3e}", className="text-end"),
                        html.Td(f"{original_energy:.3e}", className="text-end")
                    ]),
                    html.Tr([
                        html.Td("Filtered Signal", className="fw-bold"),
                        html.Td(f"{len(filtered_signal)}", className="text-end"),
                        html.Td(f"{filtered_rms:.3e}", className="text-end"),
                        html.Td(f"{filtered_energy:.3e}", className="text-end")
                    ]),
                    html.Tr([
                        html.Td("Processing Ratio", className="fw-bold"),
                        html.Td(f"{len(filtered_signal)/len(original_signal):.3f}", className="text-end"),
                        html.Td(f"{filtered_rms/original_rms:.3f}", className="text-end"),
                        html.Td(f"{filtered_energy/original_energy:.3f}", className="text-end")
                    ])
                ])
            ], bordered=True, hover=True, responsive=True, className="mb-3", color="light")
        ], className="text-muted")
    ])
    
    logger.info(f"Generated {len(metrics)} quality metrics with {len(insights)} insights")
    return metrics_html

# Additional plot functions
def create_stft_spectrogram(signal_data, sampling_freq, window_size, hop_size):
    """Create STFT spectrogram."""
    freqs, times, Zxx = signal.stft(signal_data, sampling_freq, nperseg=window_size, noverlap=window_size-hop_size)
    
    fig = go.Figure(data=go.Heatmap(
        z=np.abs(Zxx),
        x=times,
        y=freqs,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title="STFT Spectrogram",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        template="plotly_white"
    )
    
    return fig

def create_wavelet_scalogram(signal_data, sampling_freq, wavelet_type, levels):
    """Create enhanced wavelet scalogram with vitalDSP insights."""
    logger.info(f"Creating wavelet scalogram: {wavelet_type} with {levels} levels")
    
    try:
        from scipy.signal import cwt, morlet2, ricker, gausspulse
        from scipy.signal.windows import gaussian
        
        widths = np.arange(1, levels + 1)
        
        # Create appropriate wavelet function based on type
        if wavelet_type == "morlet":
            # Morlet wavelet with proper parameters
            def morlet_wavelet(x, width):
                return morlet2(x, width, w=6.0)  # w=6.0 for good time-frequency localization
            
            cwtmatr = cwt(signal_data, widths, morlet_wavelet)
            logger.info("Applied Morlet wavelet successfully")
            
        elif wavelet_type == "haar":
            # Haar wavelet implementation
            def haar_wavelet(x, width):
                n = len(x)
                wavelet = np.zeros(n)
                mid = n // 2
                wavelet[:mid] = 1
                wavelet[mid:] = -1
                return wavelet
            
            cwtmatr = cwt(signal_data, widths, haar_wavelet)
            logger.info("Applied Haar wavelet successfully")
            
        elif wavelet_type == "daubechies":
            # Daubechies wavelet approximation using Ricker wavelet
            def daubechies_wavelet(x, width):
                return ricker(len(x), width)
            
            cwtmatr = cwt(signal_data, widths, daubechies_wavelet)
            logger.info("Applied Daubechies wavelet approximation successfully")
            
        elif wavelet_type == "gaussian":
            # Gaussian wavelet
            def gaussian_wavelet(x, width):
                return gaussian(len(x), std=width)
            
            cwtmatr = cwt(signal_data, widths, gaussian_wavelet)
            logger.info("Applied Gaussian wavelet successfully")
            
        else:
            # Default to Morlet wavelet
            def morlet_wavelet(x, width):
                return morlet2(x, width, w=6.0)
            
            cwtmatr = cwt(signal_data, widths, morlet_wavelet)
            logger.info("Applied default Morlet wavelet successfully")
        
        # Calculate frequency axis
        freqs = sampling_freq / (2 * np.pi * widths)
        
        # Calculate wavelet coefficients and power
        magnitude = np.abs(cwtmatr)
        power = magnitude ** 2
        
        # Find dominant time-frequency components
        max_power_idx = np.unravel_index(np.argmax(power), power.shape)
        max_freq = freqs[max_power_idx[0]]
        max_time = max_power_idx[1] / sampling_freq
        max_power = power[max_power_idx]
        
        logger.info(f"Dominant wavelet component: {max_freq:.2f} Hz at {max_time:.2f}s with power {max_power:.2e}")
        
        # Create enhanced plot with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Wavelet Scalogram", "Power Distribution", "Energy per Level", "Time Profile"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Wavelet scalogram
        fig.add_trace(
            go.Heatmap(
                z=magnitude,
                x=np.arange(len(signal_data)) / sampling_freq,
                y=freqs,
                colorscale='Viridis',
                name='Wavelet Magnitude',
                hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Freq:</b> %{y:.2f}Hz<br><b>Magnitude:</b> %{z:.2e}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Power distribution
        fig.add_trace(
            go.Heatmap(
                z=power,
                x=np.arange(len(signal_data)) / sampling_freq,
                y=freqs,
                colorscale='Plasma',
                name='Wavelet Power',
                hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Freq:</b> %{y:.2f}Hz<br><b>Power:</b> %{z:.2e}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Energy per level
        energy_per_level = np.sum(power, axis=1)
        fig.add_trace(
            go.Bar(
                x=[f"{f:.1f} Hz" for f in freqs],
                y=energy_per_level,
                name='Energy per Level',
                marker_color='green',
                hovertemplate='<b>Frequency:</b> %{x}<br><b>Energy:</b> %{y:.2e}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Time profile at dominant frequency
        dominant_freq_idx = np.argmax(energy_per_level)
        dominant_freq = freqs[dominant_freq_idx]
        time_profile = magnitude[dominant_freq_idx, :]
        
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(signal_data)) / sampling_freq,
                y=time_profile,
                mode='lines',
                name=f'Time Profile at {dominant_freq:.2f}Hz',
                line=dict(color='red', width=2),
                hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Magnitude:</b> %{y:.2e}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Add frequency band annotations
        if np.max(freqs) > 30:
            bands = {
                'Delta': (0.5, 4, 'purple'),
                'Theta': (4, 8, 'blue'),
                'Alpha': (8, 13, 'green'),
                'Beta': (13, 30, 'orange'),
                'Gamma': (30, 100, 'red')
            }
            
            for band_name, (low, high, color) in bands.items():
                if np.max(freqs) >= low:
                    fig.add_hline(
                        y=low, line_dash="dash", line_color=color,
                        annotation_text=f"{band_name} ({low}-{high}Hz)",
                        row=1, col=1
                    )
                    fig.add_hline(
                        y=low, line_dash="dash", line_color=color,
                        row=1, col=2
                    )
        
        fig.update_layout(
            title=f"Enhanced Wavelet Scalogram - {wavelet_type.title()} ({levels} levels)",
            template="plotly_white",
            height=700,
            showlegend=True,
            autosize=True,
            margin=dict(l=50, r=50, t=80, b=50, pad=4)
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=2)
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig.update_yaxes(title_text="Energy", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Magnitude", row=2, col=2)
        
        logger.info("Enhanced wavelet scalogram created successfully")
        return fig
        
    except Exception as e:
        logger.error(f"Error creating wavelet scalogram: {e}")
        logger.info("Falling back to simple wavelet visualization")
        
        # Fallback to simple visualization
        try:
            # Simple FFT-based frequency analysis as fallback
            from scipy.fft import rfft, rfftfreq
            
            fft_result = np.abs(rfft(signal_data))
            freqs = rfftfreq(len(signal_data), 1/sampling_freq)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=freqs,
                y=fft_result,
                mode='lines',
                name='Frequency Spectrum (Fallback)',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=f"Wavelet Analysis Fallback - {wavelet_type.title()} ({levels} levels)",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude",
                template="plotly_white",
                height=400
            )
            
            return fig
            
        except Exception as fallback_error:
            logger.error(f"Fallback visualization also failed: {fallback_error}")
            return create_empty_figure()

def create_frequency_analysis_plots(signal_data, sampling_freq, analysis_type, analysis_options):
    """Create additional frequency analysis plots."""
    if "harmonics" in analysis_options:
        # Create harmonic analysis plot
        fft_result = np.abs(rfft(signal_data))
        freqs = rfftfreq(len(signal_data), 1/sampling_freq)
        
        # Find fundamental frequency
        peak_idx = np.argmax(fft_result)
        fundamental_freq = freqs[peak_idx]
        
        # Find harmonics
        harmonics = []
        for i in range(2, 6):  # 2nd to 5th harmonic
            harmonic_freq = fundamental_freq * i
            harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
            harmonics.append((harmonic_freq, fft_result[harmonic_idx]))
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freqs,
            y=fft_result,
            mode='lines',
            name='Spectrum',
            line=dict(color='blue', width=1)
        ))
        
        # Mark fundamental and harmonics
        fig.add_trace(go.Scatter(
            x=[fundamental_freq],
            y=[fft_result[peak_idx]],
            mode='markers',
            name='Fundamental',
            marker=dict(color='red', size=10)
        ))
        
        for freq, magnitude in harmonics:
            fig.add_trace(go.Scatter(
                x=[freq],
                y=[magnitude],
                mode='markers',
                name=f'Harmonic',
                marker=dict(color='orange', size=8)
            ))
        
        fig.update_layout(
            title="Harmonic Analysis",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            template="plotly_white"
        )
        
        return fig
    else:
        return create_empty_figure()

def create_filter_quality_plots(original_signal, filtered_signal, sampling_freq, quality_options):
    """Create filter quality plots."""
    if "snr" in quality_options:
        # Create SNR comparison plot
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Original Signal", "Filtered Signal"))
        
        time_axis = np.arange(len(original_signal)) / sampling_freq
        
        fig.add_trace(
            go.Scatter(x=time_axis, y=original_signal, mode='lines', name='Original'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time_axis, y=filtered_signal, mode='lines', name='Filtered'),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Signal Quality Comparison",
            template="plotly_white",
            height=400
        )
        
        return fig
    else:
        return create_empty_figure()

def create_simple_frequency_plot(signal_data, sampling_freq, title):
    """Create a simple frequency plot as fallback."""
    try:
        from scipy.fft import rfft, rfftfreq
        
        fft_result = np.abs(rfft(signal_data))
        freqs = rfftfreq(len(signal_data), 1/sampling_freq)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freqs,
            y=fft_result,
            mode='lines',
            name='Frequency Spectrum',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=f"{title} - Simple FFT",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            template="plotly_white",
            height=400
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating simple frequency plot: {e}")
        return create_empty_figure()


def create_frequency_peak_analysis_table(signal_data, sampling_freq, analysis_type, analysis_options):
    """Create a comprehensive peak frequency analysis table."""
    if "peak_freq" not in analysis_options:
        return html.Div([
            html.H6("🔍 Peak Frequency Analysis", className="text-muted"),
            html.P("Peak frequency detection not selected in analysis options.", className="text-muted")
        ])
    
    try:
        from scipy.fft import rfft, rfftfreq
        from scipy.signal import find_peaks
        
        # Perform FFT
        fft_result = np.abs(rfft(signal_data))
        freqs = rfftfreq(len(signal_data), 1/sampling_freq)
        
        # Find peaks in frequency domain
        peaks, properties = find_peaks(fft_result, prominence=0.1 * np.max(fft_result), distance=10)
        
        if len(peaks) == 0:
            return html.Div([
                html.H6("🔍 Peak Frequency Analysis", className="text-muted"),
                html.P("No significant frequency peaks detected.", className="text-muted")
            ])
        
        # Get top peaks
        top_peaks_idx = peaks[np.argsort(fft_result[peaks])[-10:]]  # Top 10 peaks
        top_peaks_freq = freqs[top_peaks_idx]
        top_peaks_mag = fft_result[top_peaks_idx]
        
        # Sort by frequency
        sorted_indices = np.argsort(top_peaks_freq)
        sorted_freqs = top_peaks_freq[sorted_indices]
        sorted_mags = top_peaks_mag[sorted_indices]
        
        return html.Div([
            html.H6("🔍 Peak Frequency Analysis", className="text-primary mb-3"),
            
            # Summary metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{len(peaks)}", className="text-center text-primary mb-0"),
                            html.Small("Total Peaks", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-primary")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{sorted_freqs[0]:.2f}", className="text-center text-success mb-0"),
                            html.Small("Lowest Peak (Hz)", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-success")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{sorted_freqs[-1]:.2f}", className="text-center text-warning mb-0"),
                            html.Small("Highest Peak (Hz)", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-warning")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{np.mean(sorted_mags):.2f}", className="text-center text-info mb-0"),
                            html.Small("Avg Magnitude", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-info")
                ], md=3)
            ], className="mb-3"),
            
            # Peak details table
            html.H6("Top Frequency Peaks", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Rank", className="text-center"),
                        html.Th("Frequency (Hz)", className="text-center"),
                        html.Th("Magnitude", className="text-center"),
                        html.Th("Period (s)", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(f"#{i+1}", className="fw-bold"),
                        html.Td(f"{freq:.2f}", className="text-center"),
                        html.Td(f"{mag:.2f}", className="text-center"),
                        html.Td(f"{1/freq:.3f}" if freq > 0 else "∞", className="text-center")
                    ]) for i, (freq, mag) in enumerate(zip(sorted_freqs, sorted_mags))
                ])
            ], bordered=True, hover=True, responsive=True, size="sm")
        ])
        
    except Exception as e:
        logger.error(f"Error creating peak frequency analysis table: {e}")
        return html.Div([
            html.H6("🔍 Peak Frequency Analysis", className="text-muted"),
            html.P(f"Error creating peak analysis: {str(e)}", className="text-danger")
        ])


def create_frequency_band_power_table(signal_data, sampling_freq, analysis_type, analysis_options):
    """Create a comprehensive band power analysis table."""
    if "band_power" not in analysis_options:
        return html.Div([
            html.H6("⚡ Band Power Analysis", className="text-muted"),
            html.P("Band power analysis not selected in analysis options.", className="text-muted")
        ])
    
    try:
        from scipy.fft import rfft, rfftfreq
        
        # Perform FFT
        fft_result = np.abs(rfft(signal_data))
        freqs = rfftfreq(len(signal_data), 1/sampling_freq)
        
        # Define frequency bands
        bands = {
            "Delta (0.5-4 Hz)": (0.5, 4),
            "Theta (4-8 Hz)": (4, 8),
            "Alpha (8-13 Hz)": (8, 13),
            "Beta (13-30 Hz)": (13, 30),
            "Gamma (30-100 Hz)": (30, 100)
        }
        
        band_powers = {}
        total_power = np.sum(fft_result**2)
        
        for band_name, (low_freq, high_freq) in bands.items():
            # Find frequencies in this band
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                band_power = np.sum(fft_result[band_mask]**2)
                band_percentage = (band_power / total_power) * 100
                band_powers[band_name] = (band_power, band_percentage)
            else:
                band_powers[band_name] = (0, 0)
        
        return html.Div([
            html.H6("⚡ Band Power Analysis", className="text-success mb-3"),
            
            # Summary metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{total_power:.2e}", className="text-center text-primary mb-0"),
                            html.Small("Total Power", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-primary")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{max(band_powers.values(), key=lambda x: x[1])[1]:.1f}%", className="text-center text-success mb-0"),
                            html.Small("Dominant Band", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-success")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{min(band_powers.values(), key=lambda x: x[1])[1]:.1f}%", className="text-center text-warning mb-0"),
                            html.Small("Weakest Band", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-warning")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{len(bands)}", className="text-center text-info mb-0"),
                            html.Small("Total Bands", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-info")
                ], md=3)
            ], className="mb-3"),
            
            # Band power table
            html.H6("Frequency Band Powers", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Frequency Band", className="text-center"),
                        html.Th("Power", className="text-center"),
                        html.Th("Percentage", className="text-center"),
                        html.Th("Status", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(band_name, className="fw-bold"),
                        html.Td(f"{power:.2e}", className="text-end"),
                        html.Td(f"{percentage:.1f}%", className="text-end"),
                        html.Td(html.Span(
                            'Dominant' if percentage > 30 else 'Strong' if percentage > 20 else 'Medium' if percentage > 10 else 'Weak',
                            className=f"badge {'bg-success' if percentage > 30 else 'bg-info' if percentage > 20 else 'bg-warning' if percentage > 10 else 'bg-secondary'}"
                        ), className="text-center")
                    ]) for band_name, (power, percentage) in band_powers.items()
                ])
            ], bordered=True, hover=True, responsive=True, size="sm")
        ])
        
    except Exception as e:
        logger.error(f"Error creating band power analysis table: {e}")
        return html.Div([
            html.H6("⚡ Band Power Analysis", className="text-muted"),
            html.P(f"Error creating band power analysis: {str(e)}", className="text-danger")
        ])


def create_frequency_stability_table(signal_data, sampling_freq, analysis_type, analysis_options):
    """Create a comprehensive frequency stability analysis table."""
    if "freq_stability" not in analysis_options:
        return html.Div([
            html.H6("📊 Frequency Stability", className="text-muted"),
            html.P("Frequency stability analysis not selected in analysis options.", className="text-muted")
        ])
    
    try:
        from scipy.fft import rfft, rfftfreq
        
        # Perform FFT
        fft_result = np.abs(rfft(signal_data))
        freqs = rfftfreq(len(signal_data), 1/sampling_freq)
        
        # Calculate stability metrics
        dominant_freq_idx = np.argmax(fft_result)
        dominant_freq = freqs[dominant_freq_idx]
        dominant_magnitude = fft_result[dominant_freq_idx]
        
        # Calculate bandwidth around dominant frequency
        half_max = dominant_magnitude / 2
        bandwidth_mask = fft_result >= half_max
        bandwidth = np.max(freqs[bandwidth_mask]) - np.min(freqs[bandwidth_mask])
        
        # Calculate frequency spread
        freq_spread = np.std(freqs)
        
        # Calculate stability score (inverse of spread)
        stability_score = 1 / (1 + freq_spread)
        
        # Calculate frequency consistency
        peak_freqs = freqs[fft_result > 0.5 * np.max(fft_result)]
        freq_consistency = 1 - (np.std(peak_freqs) / np.mean(peak_freqs)) if np.mean(peak_freqs) > 0 else 0
        
        return html.Div([
            html.H6("📊 Frequency Stability Analysis", className="text-info mb-3"),
            
            # Summary metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{dominant_freq:.2f} Hz", className="text-center text-primary mb-0"),
                            html.Small("Dominant Frequency", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-primary")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{bandwidth:.2f} Hz", className="text-center text-success mb-0"),
                            html.Small("Bandwidth", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-success")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{stability_score:.3f}", className="text-center text-warning mb-0"),
                            html.Small("Stability Score", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-warning")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{freq_consistency:.3f}", className="text-center text-info mb-0"),
                            html.Small("Consistency", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-info")
                ], md=3)
            ], className="mb-3"),
            
            # Stability metrics table
            html.H6("Stability Metrics", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Metric", className="text-center"),
                        html.Th("Value", className="text-center"),
                        html.Th("Interpretation", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Dominant Frequency", className="fw-bold"),
                        html.Td(f"{dominant_freq:.2f} Hz", className="text-end"),
                        html.Td("Primary frequency component", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Bandwidth", className="fw-bold"),
                        html.Td(f"{bandwidth:.2f} Hz", className="text-end"),
                        html.Td(html.Span(
                            'Narrow' if bandwidth < 5 else 'Medium' if bandwidth < 15 else 'Wide',
                            className=f"badge {'bg-success' if bandwidth < 5 else 'bg-warning' if bandwidth < 15 else 'bg-danger'}"
                        ), className="text-center")
                    ]),
                    html.Tr([
                        html.Td("Frequency Spread", className="fw-bold"),
                        html.Td(f"{freq_spread:.2f} Hz", className="text-end"),
                        html.Td(html.Span(
                            'Low' if freq_spread < 10 else 'Medium' if freq_spread < 25 else 'High',
                            className=f"badge {'bg-success' if freq_spread < 10 else 'bg-warning' if freq_spread < 25 else 'bg-danger'}"
                        ), className="text-center")
                    ]),
                    html.Tr([
                        html.Td("Stability Score", className="fw-bold"),
                        html.Td(f"{stability_score:.3f}", className="text-end"),
                        html.Td(html.Span(
                            'High' if stability_score > 0.8 else 'Medium' if stability_score > 0.5 else 'Low',
                            className=f"badge {'bg-success' if stability_score > 0.8 else 'bg-warning' if stability_score > 0.5 else 'bg-danger'}"
                        ), className="text-center")
                    ]),
                    html.Tr([
                        html.Td("Frequency Consistency", className="fw-bold"),
                        html.Td(f"{freq_consistency:.3f}", className="text-end"),
                        html.Td(html.Span(
                            'High' if freq_consistency > 0.8 else 'Medium' if freq_consistency > 0.5 else 'Low',
                            className=f"badge {'bg-success' if freq_consistency > 0.8 else 'bg-warning' if freq_consistency > 0.5 else 'bg-danger'}"
                        ), className="text-center")
                    ])
                ])
            ], bordered=True, hover=True, responsive=True, size="sm")
        ])
        
    except Exception as e:
        logger.error(f"Error creating frequency stability table: {e}")
        return html.Div([
            html.H6("📊 Frequency Stability", className="text-muted"),
            html.P(f"Error creating stability analysis: {str(e)}", className="text-danger")
        ])


def create_frequency_harmonics_table(signal_data, sampling_freq, analysis_type, analysis_options):
    """Create a comprehensive harmonic analysis table."""
    if "harmonics" not in analysis_options:
        return html.Div([
            html.H6("🎵 Harmonic Analysis", className="text-muted"),
            html.P("Harmonic analysis not selected in analysis options.", className="text-muted")
        ])
    
    try:
        from scipy.fft import rfft, rfftfreq
        
        # Perform FFT
        fft_result = np.abs(rfft(signal_data))
        freqs = rfftfreq(len(signal_data), 1/sampling_freq)
        
        # Find fundamental frequency
        peak_idx = np.argmax(fft_result)
        fundamental_freq = freqs[peak_idx]
        fundamental_mag = fft_result[peak_idx]
        
        # Find harmonics
        harmonics = []
        for i in range(2, 6):  # 2nd to 5th harmonic
            harmonic_freq = fundamental_freq * i
            # Find closest frequency bin
            harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
            harmonic_mag = fft_result[harmonic_idx]
            harmonics.append((i, harmonic_freq, harmonic_mag))
        
        # Calculate harmonic ratios
        harmonic_ratios = []
        for harmonic_num, harmonic_freq, harmonic_mag in harmonics:
            ratio = harmonic_mag / fundamental_mag if fundamental_mag > 0 else 0
            harmonic_ratios.append((harmonic_num, harmonic_freq, harmonic_mag, ratio))
        
        return html.Div([
            html.H6("🎵 Harmonic Analysis", className="text-secondary mb-3"),
            
            # Summary metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{fundamental_freq:.2f} Hz", className="text-center text-primary mb-0"),
                            html.Small("Fundamental Freq", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-primary")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{fundamental_mag:.2e}", className="text-center text-success mb-0"),
                            html.Small("Fundamental Mag", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-success")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{len(harmonics)}", className="text-center text-warning mb-0"),
                            html.Small("Harmonics Found", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-warning")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{np.mean([ratio for _, _, _, ratio in harmonic_ratios]):.3f}", className="text-center text-info mb-0"),
                            html.Small("Avg Harmonic Ratio", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-info")
                ], md=3)
            ], className="mb-3"),
            
            # Harmonic details table
            html.H6("Harmonic Components", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Harmonic", className="text-center"),
                        html.Th("Frequency (Hz)", className="text-center"),
                        html.Th("Magnitude", className="text-center"),
                        html.Th("Ratio to Fundamental", className="text-center"),
                        html.Th("Strength", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(f"Fundamental", className="fw-bold"),
                        html.Td(f"{fundamental_freq:.2f}", className="text-center"),
                        html.Td(f"{fundamental_mag:.2e}", className="text-center"),
                        html.Td("1.000", className="text-center"),
                        html.Td(html.Span("Strong", className="badge bg-success"), className="text-center")
                    ])
                ] + [
                    html.Tr([
                        html.Td(f"{harmonic_num}nd", className="fw-bold"),
                        html.Td(f"{harmonic_freq:.2f}", className="text-center"),
                        html.Td(f"{harmonic_mag:.2e}", className="text-center"),
                        html.Td(f"{ratio:.3f}", className="text-center"),
                        html.Td(html.Span(
                            'Strong' if ratio > 0.5 else 'Medium' if ratio > 0.2 else 'Weak',
                            className=f"badge {'bg-success' if ratio > 0.5 else 'bg-warning' if ratio > 0.2 else 'bg-secondary'}"
                        ), className="text-center")
                    ]) for harmonic_num, harmonic_freq, harmonic_mag, ratio in harmonic_ratios
                ])
            ], bordered=True, hover=True, responsive=True, size="sm")
        ])
        
    except Exception as e:
        logger.error(f"Error creating harmonic analysis table: {e}")
        return html.Div([
            html.H6("🎵 Harmonic Analysis", className="text-muted"),
            html.P(f"Error creating harmonic analysis: {str(e)}", className="text-danger")
        ])
