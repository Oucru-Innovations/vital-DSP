"""
VitalDSP analysis callbacks for vitalDSP webapp.
Handles time domain analysis, signal filtering, and comprehensive signal processing.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html
from dash.exceptions import PreventUpdate
from scipy import signal
from scipy.signal import butter, cheby1, cheby2, ellip, bessel
import dash_bootstrap_components as dbc
import logging

logger = logging.getLogger(__name__)


def register_vitaldsp_callbacks(app):
    """Register all vitalDSP analysis callbacks."""
    
    @app.callback(
        [Output("main-signal-plot", "figure"),
         Output("filtered-signal-plot", "figure"),
         Output("analysis-results", "children"),
         Output("analysis-plots", "figure"),
         Output("store-time-domain-data", "data"),
         Output("store-filtered-data", "data")],
        [Input("url", "pathname"),
         Input("btn-update-analysis", "n_clicks"),
         Input("time-range-slider", "value"),
         Input("btn-nudge-m10", "n_clicks"),
         Input("btn-nudge-m1", "n_clicks"),
         Input("btn-nudge-p1", "n_clicks"),
         Input("btn-nudge-p10", "n_clicks")],
        [State("start-time", "value"),
         State("end-time", "value"),
         State("filter-family", "value"),
         State("filter-response", "value"),
         State("filter-low-freq", "value"),
         State("filter-high-freq", "value"),
         State("filter-order", "value"),
         State("analysis-options", "value")]
    )
    def time_domain_callback(pathname, n_clicks, slider_value, nudge_m10, nudge_m1, nudge_p1, nudge_p10,
                            start_time, end_time, filter_family, filter_response, low_freq, high_freq, filter_order, analysis_options):
        """Unified callback for time domain analysis - handles both page load and user interactions."""
        ctx = callback_context
        
        # Determine what triggered this callback
        if not ctx.triggered:
            logger.warning("No context triggered - raising PreventUpdate")
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        logger.info(f"=== TIME DOMAIN CALLBACK TRIGGERED ===")
        logger.info(f"Trigger ID: {trigger_id}")
        logger.info(f"Pathname: {pathname}")
        
        # Only run this when we're on the time domain page
        if pathname != "/time-domain":
            logger.info("Not on time domain page, returning empty figures")
            return create_empty_figure(), create_empty_figure(), "Navigate to Time Domain page", create_empty_figure(), None, None
        
        try:
            # Get data from the data service
            from ..services.data_service import get_data_service
            data_service = get_data_service()
            
            # Get the most recent data
            all_data = data_service.get_all_data()
            if not all_data:
                logger.warning("No data found in service")
                return create_empty_figure(), create_empty_figure(), "No data available. Please upload and process data first.", create_empty_figure(), None, None
            
            # Get the most recent data entry
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            
            logger.info(f"Found data: {latest_data_id}")
            
            # Get column mapping
            column_mapping = data_service.get_column_mapping(latest_data_id)
            if not column_mapping:
                logger.warning("Data has not been processed yet - no column mapping found")
                return create_empty_figure(), create_empty_figure(), "Please process your data on the Upload page first (configure column mapping)", create_empty_figure(), None, None
                
            logger.info(f"Column mapping found: {column_mapping}")
                
            df = data_service.get_data(latest_data_id)
            if df is None or df.empty:
                logger.warning("Data frame is empty")
                return create_empty_figure(), create_empty_figure(), "Data is empty or corrupted.", create_empty_figure(), None, None
            
            # Get sampling frequency from the data info
            sampling_freq = latest_data.get('info', {}).get('sampling_freq', 1000)
            logger.info(f"Sampling frequency: {sampling_freq}")
            
            # Handle time window adjustments for nudge buttons
            if trigger_id in ["btn-nudge-m10", "btn-nudge-m1", "btn-nudge-p1", "btn-nudge-p10"]:
                if not start_time or not end_time:
                    start_time, end_time = 0, 10
                    
                if trigger_id == "btn-nudge-m10":
                    start_time = max(0, start_time - 10)
                    end_time = max(10, end_time - 10)
                elif trigger_id == "btn-nudge-m1":
                    start_time = max(0, start_time - 1)
                    end_time = max(1, end_time - 1)
                elif trigger_id == "btn-nudge-p1":
                    start_time = start_time + 1
                    end_time = end_time + 1
                elif trigger_id == "btn-nudge-p10":
                    start_time = start_time + 10
                    end_time = end_time + 10
                
                logger.info(f"Time window adjusted: {start_time} to {end_time}")
            
            # Set default time window if not specified
            if not start_time or not end_time:
                start_time, end_time = 0, 10
                logger.info(f"Using default time window: {start_time} to {end_time}")
            
            # Apply time window
            start_sample = int(start_time * sampling_freq)
            end_sample = int(end_time * sampling_freq)
            windowed_data = df.iloc[start_sample:end_sample].copy()
            
            # Create time axis
            time_axis = np.arange(len(windowed_data)) / sampling_freq
            
            # Create main signal plot
            logger.info("Creating main signal plot...")
            main_plot = create_main_signal_plot(windowed_data, time_axis, sampling_freq, analysis_options or ["peaks", "hr", "quality"], column_mapping)
            logger.info("Main signal plot created successfully")
            
            # Apply filtering if user has selected options
            if filter_family and filter_response and low_freq and high_freq and filter_order:
                filtered_data = apply_filter(windowed_data, sampling_freq, filter_family, filter_response, 
                                           low_freq, high_freq, filter_order, column_mapping)
                logger.info("Filter applied successfully")
            else:
                filtered_data = windowed_data
                logger.info("No filter parameters, using raw data")
            
            # Create filtered signal plot                                   
            logger.info("Creating filtered signal plot...")
            filtered_plot = create_filtered_signal_plot(filtered_data, time_axis, sampling_freq, column_mapping)
            logger.info("Filtered signal plot created successfully")
            
            # Generate analysis results
            logger.info("Generating analysis results...")
            analysis_results = generate_analysis_results(windowed_data, filtered_data, time_axis, 
                                                       sampling_freq, analysis_options or ["peaks", "hr", "quality"], column_mapping)
            logger.info("Analysis results generated successfully")
            
            # Create analysis plots
            logger.info("Creating analysis plots...")
            analysis_plots = create_analysis_plots(windowed_data, filtered_data, time_axis, 
                                                  sampling_freq, analysis_options or ["peaks", "hr", "quality"], column_mapping)
            logger.info("Analysis plots created successfully")
            
            # Store processed data
            time_domain_data = {
                "raw_data": windowed_data.to_dict("records"),
                "time_axis": time_axis.tolist(),
                "sampling_freq": sampling_freq,
                "window": [start_time, end_time]
            }
            
            filtered_data_store = {
                "filtered_data": filtered_data.to_dict("records"),
                "filter_params": {
                    "family": filter_family,
                    "response": filter_response,
                    "low_freq": low_freq,
                    "high_freq": high_freq,
                    "order": filter_order
                }
            }
            
            logger.info("Time domain analysis completed successfully")
            return main_plot, filtered_plot, analysis_results, analysis_plots, time_domain_data, filtered_data_store
            
        except Exception as e:
            logger.error(f"Error in time domain callback: {e}")
            import traceback
            traceback.print_exc()
            return create_empty_figure(), create_empty_figure(), f"Error in analysis: {str(e)}", create_empty_figure(), None, None
    
    @app.callback(
        [Output("start-time", "value"),
         Output("end-time", "value")],
        [Input("time-range-slider", "value")]
    )
    def update_time_inputs(slider_value):
        """Update time input fields based on slider."""
        if not slider_value:
            return no_update, no_update
        return slider_value[0], slider_value[1]
    
    @app.callback(
        [Output("time-range-slider", "min"),
         Output("time-range-slider", "max"),
         Output("time-range-slider", "value")],
        [Input("url", "pathname")]
    )
    def update_time_slider_range(pathname):
        """Update time slider range based on data duration."""
        logger.info("=== UPDATE TIME SLIDER RANGE ===")
        logger.info(f"Pathname: {pathname}")
        
        # Only run this when we're on the time domain page
        if pathname != "/time-domain":
            return 0, 100, [0, 10]
        
        try:
            # Get data from the data service
            from ..services.data_service import get_data_service
            data_service = get_data_service()
            
            # Get the most recent data
            all_data = data_service.get_all_data()
            if not all_data:
                logger.warning("No data found in service")
                return 0, 100, [0, 10]
            
            # Get the most recent data entry
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            
            # Get the actual data
            df = data_service.get_data(latest_data_id)
            if df is None or df.empty:
                logger.warning("Data frame is empty")
                return 0, 100, [0, 10]
            
            # Get sampling frequency from the data info
            sampling_freq = latest_data.get('info', {}).get('sampling_freq', 1000)
            
            max_time = len(df) / sampling_freq
            logger.info(f"Max time: {max_time}, Sampling freq: {sampling_freq}")
            
            return 0, max_time, [0, min(10, max_time)]
            
        except Exception as e:
            logger.error(f"Error updating time slider range: {e}")
            return 0, 100, [0, 10]


def create_empty_figure():
    """Create an empty figure for when no data is available."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available<br>Please upload data first",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig


def create_main_signal_plot(data, time_axis, sampling_freq, analysis_options, column_mapping):
    """Create the main signal plot with optional annotations."""
    try:
        fig = go.Figure()
        
        # Use column mapping to get the correct columns
        time_col = column_mapping.get('time')
        signal_col = column_mapping.get('signal')
        
        if not time_col or not signal_col:
            logger.warning("Missing time or signal column in column mapping")
            return create_empty_figure()
        
        if time_col not in data.columns or signal_col not in data.columns:
            logger.warning(f"Columns {time_col} or {signal_col} not found in data")
            return create_empty_figure()
        
        # Create the main signal plot
        fig.add_trace(go.Scatter(
            x=data[time_col] if time_col != 'index' else time_axis,
            y=data[signal_col],
            mode='lines',
            name=f'Signal ({signal_col})',
            line=dict(color='#2E86AB', width=2),
            hovertemplate='<b>Time:</b> %{x}<br><b>Amplitude:</b> %{y}<extra></extra>'
        ))
        
        # Add peak detection if enabled
        if 'peaks' in analysis_options:
            try:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(data[signal_col], height=0.5*np.max(data[signal_col]), distance=int(0.5*sampling_freq))
                if len(peaks) > 0:
                    fig.add_trace(go.Scatter(
                        x=data[time_col].iloc[peaks] if time_col != 'index' else time_axis[peaks],
                        y=data[signal_col].iloc[peaks],
                        mode='markers',
                        name='Peaks',
                        marker=dict(color='red', size=8, symbol='diamond'),
                        hovertemplate='<b>Peak:</b> %{y}<extra></extra>'
                    ))
            except Exception as e:
                logger.error(f"Peak detection failed: {e}")
        
        fig.update_layout(
            title="Raw Signal with Analysis",
            xaxis_title=f"Time ({time_col})",
            yaxis_title=f"Amplitude ({signal_col})",
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=True,
            hovermode='closest'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating main signal plot: {e}")
        return create_empty_figure()


def create_filtered_signal_plot(filtered_data, time_axis, sampling_freq, column_mapping):
    """Create the filtered signal plot."""
    try:
        fig = go.Figure()
        
        # Use column mapping to get the correct columns
        time_col = column_mapping.get('time')
        signal_col = column_mapping.get('signal')
        
        if not time_col or not signal_col:
            logger.warning("Missing time or signal column in column mapping")
            return create_empty_figure()
        
        if time_col not in filtered_data.columns or signal_col not in filtered_data.columns:
            logger.warning(f"Columns {time_col} or {signal_col} not found in filtered data")
            return create_empty_figure()
        
        # Plot raw signal
        fig.add_trace(go.Scatter(
            x=filtered_data[time_col] if time_col != 'index' else time_axis,
            y=filtered_data[signal_col],
            mode='lines',
            name=f'Raw Signal ({signal_col})',
            line=dict(color='#2E86AB', width=1.5),
            opacity=0.7,
            hovertemplate='<b>Time:</b> %{x}<br><b>Raw:</b> %{y}<extra></extra>'
        ))
        
        # Plot filtered signal if available
        filtered_col = f"{signal_col}_filtered"
        if filtered_col in filtered_data.columns:
            fig.add_trace(go.Scatter(
                x=filtered_data[time_col] if time_col != 'index' else time_axis,
                y=filtered_data[filtered_col],
                mode='lines',
                name=f'Filtered Signal ({signal_col})',
                line=dict(color='#E63946', width=2),
                hovertemplate='<b>Time:</b> %{x}<br><b>Filtered:</b> %{y}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Signal Comparison: Raw vs Filtered",
            xaxis_title=f"Time ({time_col})",
            yaxis_title=f"Amplitude ({signal_col})",
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=True,
            hovermode='closest'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating filtered signal plot: {e}")
        return create_empty_figure()


def apply_filter(data, sampling_freq, filter_family, filter_response, low_freq, high_freq, filter_order, column_mapping):
    """Apply digital filter to the signal."""
    try:
        # Use column mapping to get the signal column
        signal_col = column_mapping.get('signal')
        if not signal_col or signal_col not in data.columns:
            return data
            
        signal_data = data[signal_col].values
        
        # Design filter
        if filter_response == "bandpass":
            b, a = butter(filter_order, [low_freq, high_freq], btype='band', fs=sampling_freq)
        elif filter_response == "lowpass":
            b, a = butter(filter_order, high_freq, btype='low', fs=sampling_freq)
        elif filter_response == "highpass":
            b, a = butter(filter_order, low_freq, btype='high', fs=sampling_freq)
        elif filter_response == "bandstop":
            b, a = butter(filter_order, [low_freq, high_freq], btype='bandstop', fs=sampling_freq)
        else:
            return data
        
        # Apply filter
        filtered_signal = signal.filtfilt(b, a, signal_data)
        
        # Create filtered dataframe
        filtered_data = data.copy()
        filtered_data[f"{signal_col}_filtered"] = filtered_signal
        
        return filtered_data
        
    except Exception as e:
        logger.error(f"Filtering failed: {e}")
        return data


def generate_analysis_results(raw_data, filtered_data, time_axis, sampling_freq, analysis_options, column_mapping):
    """Generate analysis results and insights."""
    if not analysis_options:
        return "No analysis options selected"
    
    results = []
    
    try:
        # Find signal column
        numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return "No numeric data available for analysis"
            
        signal_col = numeric_cols[0]
        raw_signal = raw_data[signal_col].values
        filtered_signal = filtered_data[signal_col].values
        
        # Basic statistics
        results.append(html.H6("📊 Signal Statistics", className="mb-2"))
        results.append(html.P([
            f"Mean: {np.mean(raw_signal):.2f} | ",
            f"Std: {np.std(raw_signal):.2f} | ",
            f"Min: {np.min(raw_signal):.2f} | ",
            f"Max: {np.max(raw_signal):.2f}"
        ], className="mb-2"))
        
        # Peak detection and heart rate
        if "peaks" in analysis_options:
            peaks, properties = signal.find_peaks(raw_signal, prominence=0.1 * np.std(raw_signal))
            if len(peaks) > 1:
                # Calculate RR intervals
                rr_intervals = np.diff(time_axis[peaks])
                heart_rate = 60 / np.mean(rr_intervals)
                
                results.append(html.Hr())
                results.append(html.H6("❤️ Heart Rate Analysis", className="mb-2"))
                results.append(html.P([
                    f"Detected {len(peaks)} peaks | ",
                    f"Average RR: {np.mean(rr_intervals):.3f}s | ",
                    f"Heart Rate: {heart_rate:.1f} BPM"
                ], className="mb-2"))
        
        # Signal quality assessment
        if "quality" in analysis_options:
            # Calculate SNR
            signal_power = np.mean(raw_signal**2)
            noise_power = np.var(raw_signal)
            snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
            
            # Artifact detection
            q75, q25 = np.percentile(raw_signal, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            artifact_count = np.sum((raw_signal < lower_bound) | (raw_signal > upper_bound))
            
            results.append(html.Hr())
            results.append(html.H6("🎯 Signal Quality", className="mb-2"))
            results.append(html.P([
                f"SNR: {snr_db:.1f} dB | ",
                f"Artifacts: {artifact_count} | ",
                f"Quality: {'Excellent' if snr_db > 20 else 'Good' if snr_db > 15 else 'Fair' if snr_db > 10 else 'Poor'}"
            ], className="mb-2"))
        
        # Filtering results
        if filtered_data is not None and not filtered_data.equals(raw_data):
            results.append(html.Hr())
            results.append(html.H6("🔧 Filtering Results", className="mb-2"))
            
            # Calculate power reduction
            raw_power = np.mean(raw_signal**2)
            filtered_power = np.mean(filtered_signal**2)
            power_reduction = (raw_power - filtered_power) / raw_power * 100
            
            results.append(html.P([
                f"Power reduction: {power_reduction:.1f}% | ",
                f"Signal preserved: {100 - power_reduction:.1f}%"
            ], className="mb-2"))
        
        return html.Div(results)
        
    except Exception as e:
        return html.Div([
            html.H6("❌ Analysis Error", className="text-danger"),
            html.P(f"Error during analysis: {str(e)}")
        ])


def create_analysis_plots(raw_data, filtered_data, time_axis, sampling_freq, analysis_options, column_mapping):
    """Create additional analysis plots."""
    try:
        if not analysis_options:
            return create_empty_figure()
        
        # Find signal column
        numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            logger.warning("No numeric data available for analysis plots")
            return create_empty_figure()
            
        signal_col = numeric_cols[0]
        raw_signal = raw_data[signal_col].values
        
        # Create subplots based on analysis options
        num_plots = sum([
            "peaks" in analysis_options,
            "hr" in analysis_options,
            "quality" in analysis_options
        ])
        
        if num_plots == 0:
            return create_empty_figure()
        
        fig = make_subplots(
            rows=num_plots, cols=1,
            subplot_titles=[opt.title() for opt in analysis_options if opt in ["peaks", "hr", "quality"]],
            vertical_spacing=0.1
        )
        
        plot_row = 1
        
        # Peak detection plot
        if "peaks" in analysis_options:
            try:
                peaks, properties = signal.find_peaks(raw_signal, prominence=0.1 * np.std(raw_signal))
                
                fig.add_trace(
                    go.Scatter(x=time_axis, y=raw_signal, mode='lines', name='Signal', 
                              line=dict(color='#3498db'), showlegend=False),
                    row=plot_row, col=1
                )
                
                if len(peaks) > 0:
                    fig.add_trace(
                        go.Scatter(x=time_axis[peaks], y=raw_signal[peaks], mode='markers', 
                                  name='Peaks', marker=dict(color='red', size=8, symbol='diamond'), showlegend=False),
                        row=plot_row, col=1
                    )
                
                plot_row += 1
            except Exception as e:
                logger.error(f"Error creating peak detection plot: {e}")
                plot_row += 1
        
        # Heart rate trend plot
        if "hr" in analysis_options and "peaks" in analysis_options:
            try:
                peaks, _ = signal.find_peaks(raw_signal, prominence=0.1 * np.std(raw_signal))
                if len(peaks) > 1:
                    rr_intervals = np.diff(time_axis[peaks])
                    heart_rates = 60 / rr_intervals
                    peak_times = time_axis[peaks][1:]  # Use second peak onwards
                    
                    fig.add_trace(
                        go.Scatter(x=peak_times, y=heart_rates, mode='lines+markers', 
                                  name='Heart Rate', line=dict(color='#e74c3c'), showlegend=False),
                        row=plot_row, col=1
                    )
                    
                    # Add mean line
                    mean_hr = np.mean(heart_rates)
                    fig.add_hline(y=mean_hr, line_dash="dash", line_color="gray", 
                                 annotation_text=f"Mean: {mean_hr:.1f} BPM", row=plot_row, col=1)
                
                plot_row += 1
            except Exception as e:
                logger.error(f"Error creating heart rate plot: {e}")
                plot_row += 1
        
        # Signal quality plot
        if "quality" in analysis_options:
            try:
                # Create moving window statistics
                window_size = int(0.5 * sampling_freq)  # 0.5 second window
                if len(raw_signal) > window_size:
                    moving_std = []
                    moving_mean = []
                    time_points = []
                    
                    for i in range(0, len(raw_signal) - window_size, window_size // 4):
                        window = raw_signal[i:i + window_size]
                        moving_std.append(np.std(window))
                        moving_mean.append(np.mean(window))
                        time_points.append(time_axis[i + window_size // 2])
                    
                    fig.add_trace(
                        go.Scatter(x=time_points, y=moving_std, mode='lines', 
                                  name='Moving Std', line=dict(color='#f39c12'), showlegend=False),
                        row=plot_row, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=time_points, y=moving_mean, mode='lines', 
                                  name='Moving Mean', line=dict(color='#27ae60'), showlegend=False),
                        row=plot_row, col=1
                    )
            except Exception as e:
                logger.error(f"Error creating signal quality plot: {e}")
        
        # Update layout
        fig.update_layout(
            title="Analysis Results",
            template="plotly_white",
            height=300 * num_plots,
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=False
        )
        
        # Update axes labels
        for i in range(1, num_plots + 1):
            fig.update_xaxes(title_text="Time (s)", row=i, col=1)
            if i == num_plots:  # Only show x-axis title for bottom plot
                fig.update_xaxes(title_text="Time (s)", row=i, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating analysis plots: {e}")
        return create_empty_figure()
