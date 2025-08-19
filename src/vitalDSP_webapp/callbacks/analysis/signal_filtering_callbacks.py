"""
Signal filtering callbacks for vitalDSP webapp.
Handles traditional filtering, advanced filtering, artifact removal, neural filtering, and ensemble filtering.
Uses actual vitalDSP functions for all computations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html
from dash.exceptions import PreventUpdate
from scipy import signal
import dash_bootstrap_components as dbc
import logging

logger = logging.getLogger(__name__)


def register_signal_filtering_callbacks(app):
    """Register all signal filtering callbacks."""
    
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
            from vitalDSP_webapp.services.data.data_service import get_data_service
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
            signal_column = column_mapping.get('signal', df.columns[1])
            signal_data = windowed_data[signal_column].values
            
            # Create original signal plot
            original_plot = create_signal_plot(signal_data, time_axis, "Original Signal")
            
            # Apply filtering based on selected type using vitalDSP functions
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
            
            logger.info("Advanced filtering completed successfully")
            return original_plot, filtered_plot, comparison_plot, quality_metrics, quality_plots, filtering_data
            
        except Exception as e:
            logger.error(f"Error in advanced filtering: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            error_fig = create_empty_figure()
            error_fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            
            error_results = html.Div([
                html.H5("Error in Advanced Filtering"),
                html.P(f"Filtering failed: {str(e)}"),
                html.P("Please check your data and parameters.")
            ])
            
            return error_fig, error_fig, error_fig, error_results, error_fig, None

    # Time input update callbacks
    @app.callback(
        [Output("filter-start-time", "value"),
         Output("filter-end-time", "value")],
        [Input("filter-time-range-slider", "value"),
         Input("filter-btn-nudge-m10", "n_clicks"),
         Input("filter-btn-nudge-m1", "n_clicks"),
         Input("filter-btn-nudge-p1", "n_clicks"),
         Input("filter-btn-nudge-p10", "n_clicks")],
        [State("filter-start-time", "value"),
         State("filter-end-time", "value")]
    )
    def update_filter_time_inputs(slider_value, nudge_m10, nudge_m1, nudge_p1, nudge_p10, start_time, end_time):
        """Update time inputs based on slider or nudge buttons."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if trigger_id == "filter-time-range-slider" and slider_value:
            return slider_value[0], slider_value[1]
        
        # Handle nudge buttons
        time_window = end_time - start_time if start_time and end_time else 10
        
        if trigger_id == "filter-btn-nudge-m10":
            new_start = max(0, start_time - 10) if start_time else 0
            new_end = new_start + time_window
            return new_start, new_end
        elif trigger_id == "filter-btn-nudge-m1":
            new_start = max(0, start_time - 1) if start_time else 0
            new_end = new_start + time_window
            return new_start, new_end
        elif trigger_id == "filter-btn-nudge-p1":
            new_start = start_time + 1 if start_time else 1
            new_end = new_start + time_window
            return new_start, new_end
        elif trigger_id == "filter-btn-nudge-p10":
            new_start = start_time + 10 if start_time else 10
            new_end = new_start + time_window
            return new_start, new_end
        
        return no_update, no_update

    # Time slider range update callback
    @app.callback(
        Output("filter-time-range-slider", "max"),
        [Input("store-uploaded-data", "data")]
    )
    def update_filter_time_slider_range(data_store):
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
            logger.error(f"Error updating filter time slider range: {e}")
            return 100

    # Filter type parameter visibility callback
    @app.callback(
        [Output("traditional-filter-params", "style"),
         Output("advanced-filter-params", "style"),
         Output("artifact-removal-params", "style"),
         Output("neural-network-params", "style"),
         Output("ensemble-params", "style")],
        [Input("filter-type-select", "value")]
    )
    def update_filter_parameter_visibility(filter_type):
        """Show/hide parameter sections based on selected filter type."""
        # Default style (hidden)
        hidden_style = {"display": "none"}
        visible_style = {"display": "block"}
        
        # Initialize all to hidden
        traditional_style = hidden_style
        advanced_style = hidden_style
        artifact_style = hidden_style
        neural_style = hidden_style
        ensemble_style = hidden_style
        
        # Show appropriate section based on selection
        if filter_type == "traditional":
            traditional_style = visible_style
        elif filter_type == "advanced":
            advanced_style = visible_style
        elif filter_type == "artifact":
            artifact_style = visible_style
        elif filter_type == "neural":
            neural_style = visible_style
        elif filter_type == "ensemble":
            ensemble_style = visible_style
        
        return traditional_style, advanced_style, artifact_style, neural_style, ensemble_style


# Helper functions for signal filtering
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


def create_signal_plot(signal_data, time_axis, title):
    """Create a signal plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=signal_data,
        mode='lines',
        name=title,
        line=dict(color='blue', width=1.5)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        showlegend=True,
        height=400
    )
    
    return fig


def create_filter_comparison_plot(original_signal, filtered_signal, time_axis):
    """Create a comparison plot between original and filtered signals."""
    fig = go.Figure()
    
    # Original signal
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=original_signal,
        mode='lines',
        name='Original Signal',
        line=dict(color='blue'),
        opacity=0.7
    ))
    
    # Filtered signal
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=filtered_signal,
        mode='lines',
        name='Filtered Signal',
        line=dict(color='red', width=1.5)
    ))
    
    fig.update_layout(
        title="Signal Comparison: Original vs Filtered",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        showlegend=True,
        height=400
    )
    
    return fig


def apply_traditional_filter(signal_data, sampling_freq, filter_family, filter_response, low_freq, high_freq, filter_order):
    """Apply traditional filtering using vitalDSP functions."""
    try:
        logger.info(f"Applying traditional filter: {filter_family} {filter_response} filter")
        
        # Import vitalDSP filtering functions
        from vitalDSP.filtering.signal_filtering import BandpassFilter
        
        # Set default values
        filter_family = filter_family or "butter"
        filter_response = filter_response or "low"
        low_freq = low_freq or 10
        high_freq = high_freq or 50
        filter_order = filter_order or 4
        
        # Create filter instance
        filter_obj = BandpassFilter(band_type=filter_family, fs=sampling_freq)
        
        # Apply filter based on response type
        if filter_response == "low":
            filtered_signal = filter_obj.signal_lowpass_filter(signal_data, low_freq, filter_order)
        elif filter_response == "high":
            filtered_signal = filter_obj.signal_highpass_filter(signal_data, high_freq, filter_order)
        elif filter_response == "bandpass":
            filtered_signal = filter_obj.signal_bandpass_filter(signal_data, low_freq, high_freq, filter_order)
        else:
            # Default to lowpass
            filtered_signal = filter_obj.signal_lowpass_filter(signal_data, low_freq, filter_order)
        
        logger.info(f"Traditional filter applied successfully: {filter_family} {filter_response}")
        return filtered_signal
        
    except Exception as e:
        logger.error(f"Error applying traditional filter: {e}")
        # Fallback to scipy
        try:
            if filter_response == "low":
                b, a = signal.butter(filter_order, low_freq/(sampling_freq/2), btype='low')
            elif filter_response == "high":
                b, a = signal.butter(filter_order, high_freq/(sampling_freq/2), btype='high')
            elif filter_response == "bandpass":
                b, a = signal.butter(filter_order, [low_freq/(sampling_freq/2), high_freq/(sampling_freq/2)], btype='band')
            else:
                b, a = signal.butter(filter_order, low_freq/(sampling_freq/2), btype='low')
            
            filtered_signal = signal.filtfilt(b, a, signal_data)
            logger.info("Fallback to scipy filtering successful")
            return filtered_signal
        except Exception as fallback_error:
            logger.error(f"Fallback filtering also failed: {fallback_error}")
            return signal_data


def apply_advanced_filter(signal_data, advanced_method, noise_level, iterations, learning_rate):
    """Apply advanced filtering using vitalDSP functions."""
    try:
        logger.info(f"Applying advanced filter: {advanced_method}")
        
        # Import vitalDSP advanced filtering functions
        from vitalDSP.filtering.advanced_signal_filtering import AdvancedSignalFiltering
        
        # Set default values
        advanced_method = advanced_method or "kalman"
        noise_level = noise_level or 0.1
        iterations = iterations or 100
        learning_rate = learning_rate or 0.01
        
        # Create advanced filter instance
        advanced_filter = AdvancedSignalFiltering(signal_data)
        
        # Apply filter based on method
        if advanced_method == "kalman":
            filtered_signal = advanced_filter.kalman_filter(R=noise_level, Q=1)
        elif advanced_method == "optimization":
            filtered_signal = advanced_filter.optimization_based_filtering(
                target=signal_data, 
                loss_type="mse", 
                learning_rate=learning_rate, 
                iterations=iterations
            )
        elif advanced_method == "gradient_descent":
            filtered_signal = advanced_filter.gradient_descent_filter(
                target=signal_data, 
                learning_rate=learning_rate, 
                iterations=iterations
            )
        elif advanced_method == "convolution":
            filtered_signal = advanced_filter.convolution_based_filter(
                kernel_type="smoothing", 
                kernel_size=5
            )
        elif advanced_method == "attention":
            filtered_signal = advanced_filter.attention_based_filter(
                attention_type="uniform", 
                size=5
            )
        else:
            # Default to Kalman filter
            filtered_signal = advanced_filter.kalman_filter(R=noise_level, Q=1)
        
        logger.info(f"Advanced filter applied successfully: {advanced_method}")
        return filtered_signal
        
    except Exception as e:
        logger.error(f"Error applying advanced filter: {e}")
        # Return original signal if advanced filtering fails
        return signal_data


def apply_artifact_removal(signal_data, sampling_freq, artifact_type, artifact_strength):
    """Apply artifact removal using vitalDSP functions."""
    try:
        logger.info(f"Applying artifact removal: {artifact_type}")
        
        # Import vitalDSP artifact removal functions
        from vitalDSP.filtering.artifact_removal import ArtifactRemoval
        
        # Set default values
        artifact_type = artifact_type or "baseline"
        artifact_strength = artifact_strength or 0.5
        
        # Create artifact removal instance
        artifact_remover = ArtifactRemoval(signal_data)
        
        # Apply artifact removal based on type
        if artifact_type == "baseline":
            filtered_signal = artifact_remover.baseline_correction(
                cutoff=artifact_strength, 
                fs=sampling_freq
            )
        elif artifact_type == "spike":
            filtered_signal = artifact_remover.median_filter_removal()
        elif artifact_type == "noise":
            filtered_signal = artifact_remover.wavelet_denoising()
        elif artifact_type == "powerline":
            filtered_signal = artifact_remover.notch_filter(fs=sampling_freq)
        elif artifact_type == "pca":
            filtered_signal = artifact_remover.pca_artifact_removal()
        elif artifact_type == "ica":
            filtered_signal = artifact_remover.ica_artifact_removal()
        else:
            # Default to baseline correction
            filtered_signal = artifact_remover.baseline_correction(
                cutoff=artifact_strength, 
                fs=sampling_freq
            )
        
        logger.info(f"Artifact removal applied successfully: {artifact_type}")
        return filtered_signal
        
    except Exception as e:
        logger.error(f"Error applying artifact removal: {e}")
        # Return original signal if artifact removal fails
        return signal_data


def apply_neural_filter(signal_data, neural_type, neural_complexity):
    """Apply neural network filtering using vitalDSP functions."""
    try:
        logger.info(f"Applying neural filter: {neural_type}")
        
        # Import vitalDSP neural filtering functions
        from vitalDSP.advanced_computation.neural_network_filtering import NeuralNetworkFiltering
        
        # Set default values
        neural_type = neural_type or "feedforward"
        neural_complexity = neural_complexity or "medium"
        
        # Set complexity-based parameters
        if neural_complexity == "low":
            hidden_layers = [32]
            epochs = 50
        elif neural_complexity == "high":
            hidden_layers = [128, 64, 32]
            epochs = 200
        else:  # medium
            hidden_layers = [64, 32]
            epochs = 100
        
        # Create neural filter instance
        neural_filter = NeuralNetworkFiltering(
            signal_data, 
            network_type=neural_type,
            hidden_layers=hidden_layers,
            epochs=epochs
        )
        
        # Train the neural network first
        neural_filter.train()
        
        # Apply the trained filter
        filtered_signal = neural_filter.apply_filter()
        
        logger.info(f"Neural filter applied successfully: {neural_type}")
        return filtered_signal
        
    except Exception as e:
        logger.error(f"Error applying neural filter: {e}")
        # Return original signal if neural filtering fails
        return signal_data


def apply_ensemble_filter(signal_data, ensemble_method, ensemble_n_filters):
    """Apply ensemble filtering using vitalDSP functions."""
    try:
        logger.info(f"Applying ensemble filter: {ensemble_method}")
        
        # Import vitalDSP ensemble filtering functions
        from vitalDSP.filtering.advanced_signal_filtering import AdvancedSignalFiltering
        
        # Set default values
        ensemble_method = ensemble_method or "mean"
        ensemble_n_filters = ensemble_n_filters or 3
        
        # Create advanced filter instance
        advanced_filter = AdvancedSignalFiltering(signal_data)
        
        # Create multiple filters for ensemble
        filters = []
        for i in range(ensemble_n_filters):
            # Create different filter configurations
            if i == 0:
                # Kalman filter
                filtered = advanced_filter.kalman_filter(R=0.1, Q=1)
            elif i == 1:
                # Convolution filter
                filtered = advanced_filter.convolution_based_filter(kernel_type="smoothing", kernel_size=3)
            else:
                # Optimization filter
                filtered = advanced_filter.optimization_based_filtering(
                    target=signal_data, 
                    loss_type="mse", 
                    learning_rate=0.01, 
                    iterations=50
                )
            filters.append(filtered)
        
        # Apply ensemble method
        if ensemble_method == "mean":
            filtered_signal = np.mean(filters, axis=0)
        elif ensemble_method == "median":
            filtered_signal = np.median(filters, axis=0)
        elif ensemble_method == "weighted":
            # Weight by filter performance (inverse of variance)
            weights = 1 / (np.var(filters, axis=0) + 1e-10)
            weights = weights / np.sum(weights)
            filtered_signal = np.average(filters, axis=0, weights=weights)
        else:
            # Default to mean
            filtered_signal = np.mean(filters, axis=0)
        
        logger.info(f"Ensemble filter applied successfully: {ensemble_method} with {ensemble_n_filters} filters")
        return filtered_signal
        
    except Exception as e:
        logger.error(f"Error applying ensemble filter: {e}")
        # Return original signal if ensemble filtering fails
        return signal_data


def generate_filter_quality_metrics(original_signal, filtered_signal, sampling_freq, quality_options):
    """Generate comprehensive filter quality metrics."""
    try:
        # Calculate basic metrics
        snr_improvement = calculate_snr_improvement(original_signal, filtered_signal)
        mse = calculate_mse(original_signal, filtered_signal)
        correlation = calculate_correlation(original_signal, filtered_signal)
        smoothness = calculate_smoothness(filtered_signal)
        
        # Calculate frequency domain metrics
        freq_metrics = calculate_frequency_metrics(original_signal, filtered_signal, sampling_freq)
        
        # Create metrics display
        metrics_html = html.Div([
            html.H5("Filter Quality Metrics", className="mb-3"),
            html.Div([
                html.Div([
                    html.H6("Signal Quality", className="mb-2"),
                    html.P(f"SNR Improvement: {snr_improvement:.2f} dB"),
                    html.P(f"Mean Square Error: {mse:.4f}"),
                    html.P(f"Correlation: {correlation:.3f}"),
                    html.P(f"Smoothness: {smoothness:.3f}")
                ], className="col-md-6"),
                html.Div([
                    html.H6("Frequency Analysis", className="mb-2"),
                    html.P(f"Peak Frequency: {freq_metrics['peak_freq']:.2f} Hz"),
                    html.P(f"Bandwidth: {freq_metrics['bandwidth']:.2f} Hz"),
                    html.P(f"Spectral Centroid: {freq_metrics['spectral_centroid']:.2f} Hz"),
                    html.P(f"Frequency Stability: {freq_metrics['freq_stability']:.3f}")
                ], className="col-md-6")
            ], className="row")
        ])
        
        return metrics_html
        
    except Exception as e:
        logger.error(f"Error generating quality metrics: {e}")
        return html.Div([
            html.H5("Filter Quality Metrics"),
            html.P(f"Error calculating metrics: {str(e)}")
        ])


def create_filter_quality_plots(original_signal, filtered_signal, sampling_freq, quality_options):
    """Create quality assessment plots."""
    try:
        # Create subplots for different quality metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Signal Comparison", "Frequency Response", "Error Analysis", "Quality Metrics"),
            vertical_spacing=0.15
        )
        
        # Time domain comparison
        time_axis = np.arange(len(original_signal)) / sampling_freq
        fig.add_trace(
            go.Scatter(x=time_axis, y=original_signal, mode='lines', name='Original', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_axis, y=filtered_signal, mode='lines', name='Filtered', line=dict(color='red')),
            row=1, col=1
        )
        
        # Frequency response
        freqs_orig, psd_orig = signal.welch(original_signal, fs=sampling_freq)
        freqs_filt, psd_filt = signal.welch(filtered_signal, fs=sampling_freq)
        
        fig.add_trace(
            go.Scatter(x=freqs_orig, y=10*np.log10(psd_orig), mode='lines', name='Original PSD', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=freqs_filt, y=10*np.log10(psd_filt), mode='lines', name='Filtered PSD', line=dict(color='red')),
            row=1, col=2
        )
        
        # Error analysis
        error = original_signal - filtered_signal
        fig.add_trace(
            go.Scatter(x=time_axis, y=error, mode='lines', name='Error', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Quality metrics over time
        window_size = min(100, len(original_signal) // 10)
        if window_size > 0:
            quality_metrics = []
            for i in range(0, len(original_signal) - window_size, window_size):
                orig_window = original_signal[i:i+window_size]
                filt_window = filtered_signal[i:i+window_size]
                snr = calculate_snr_improvement(orig_window, filt_window)
                quality_metrics.append(snr)
            
            time_centers = np.arange(len(quality_metrics)) * window_size / sampling_freq
            fig.add_trace(
                go.Scatter(x=time_centers, y=quality_metrics, mode='lines', name='SNR over Time', line=dict(color='green')),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Filter Quality Assessment",
            height=600,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating quality plots: {e}")
        return create_empty_figure()


# Quality metric calculation functions
def calculate_snr_improvement(original_signal, filtered_signal):
    """Calculate SNR improvement after filtering."""
    try:
        # Calculate noise as the difference between original and filtered
        noise = original_signal - filtered_signal
        
        # Calculate signal power (filtered signal)
        signal_power = np.mean(filtered_signal ** 2)
        
        # Calculate noise power
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            return snr
        else:
            return 0
    except Exception as e:
        logger.error(f"Error calculating SNR: {e}")
        return 0


def calculate_mse(original_signal, filtered_signal):
    """Calculate Mean Square Error between original and filtered signals."""
    try:
        return np.mean((original_signal - filtered_signal) ** 2)
    except Exception as e:
        logger.error(f"Error calculating MSE: {e}")
        return 0


def calculate_correlation(original_signal, filtered_signal):
    """Calculate correlation between original and filtered signals."""
    try:
        return np.corrcoef(original_signal, filtered_signal)[0, 1]
    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        return 0


def calculate_smoothness(signal_data):
    """Calculate signal smoothness using variance of first differences."""
    try:
        differences = np.diff(signal_data)
        return 1 / (1 + np.var(differences))
    except Exception as e:
        logger.error(f"Error calculating smoothness: {e}")
        return 0


def calculate_frequency_metrics(original_signal, filtered_signal, sampling_freq):
    """Calculate frequency domain metrics."""
    try:
        # Calculate PSD for both signals
        freqs_orig, psd_orig = signal.welch(original_signal, fs=sampling_freq)
        freqs_filt, psd_filt = signal.welch(filtered_signal, fs=sampling_freq)
        
        # Find peak frequency
        peak_idx = np.argmax(psd_filt)
        peak_freq = freqs_filt[peak_idx]
        
        # Calculate bandwidth (3dB down from peak)
        peak_power = psd_filt[peak_idx]
        threshold = peak_power / 2  # 3dB down
        above_threshold = psd_filt > threshold
        bandwidth = freqs_filt[-1] - freqs_filt[0] if np.any(above_threshold) else 0
        
        # Calculate spectral centroid
        spectral_centroid = np.sum(freqs_filt * psd_filt) / np.sum(psd_filt)
        
        # Calculate frequency stability (inverse of frequency variance)
        freq_stability = 1 / (1 + np.var(freqs_filt))
        
        return {
            'peak_freq': peak_freq,
            'bandwidth': bandwidth,
            'spectral_centroid': spectral_centroid,
            'freq_stability': freq_stability
        }
        
    except Exception as e:
        logger.error(f"Error calculating frequency metrics: {e}")
        return {
            'peak_freq': 0,
            'bandwidth': 0,
            'spectral_centroid': 0,
            'freq_stability': 0
        }
