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


def process_signal_analysis(data_id: str, analysis_type: str, parameters: dict) -> dict:
    """
    Process signal analysis based on the given parameters.
    
    Parameters
    ----------
    data_id : str
        ID of the data to analyze
    analysis_type : str
        Type of analysis to perform
    parameters : dict
        Analysis parameters
        
    Returns
    -------
    dict
        Analysis results
    """
    try:
        # Get data from the data service
        from ..services.data_service import get_data_service
        data_service = get_data_service()
        
        # Get the data
        df = data_service.get_data(data_id)
        if df is None or df.empty:
            return {
                'status': 'error',
                'message': 'No data found for the given ID'
            }
        
        # Perform analysis based on type
        if analysis_type == 'hrv_analysis':
            results = perform_hrv_analysis(df, parameters)
        elif analysis_type == 'frequency_domain':
            results = perform_frequency_analysis(df, parameters)
        elif analysis_type == 'time_domain':
            results = perform_time_domain_analysis(df, parameters)
        elif analysis_type == 'wavelet_analysis':
            results = perform_wavelet_analysis(df, parameters)
        else:
            return {
                'status': 'error',
                'message': f'Unsupported analysis type: {analysis_type}'
            }
        
        return {
            'status': 'success',
            'results': results,
            'analysis_type': analysis_type,
            'parameters': parameters
        }
        
    except Exception as e:
        logger.error(f"Error in signal analysis: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }


def generate_analysis_report(analysis_results: dict, analysis_type: str) -> dict:
    """
    Generate a comprehensive analysis report.
    
    Parameters
    ----------
    analysis_results : dict
        Results from the analysis
    analysis_type : str
        Type of analysis performed
        
    Returns
    -------
    dict
        Generated report
    """
    try:
        if not analysis_results:
            return {
                'summary': 'No analysis results available',
                'details': {},
                'recommendations': 'No specific recommendations available'
            }
        
        # Create summary based on analysis type
        if analysis_type == 'hrv_analysis':
            summary = 'HRV Analysis Report'
            recommendations = 'Consider HRV trends over time for better insights'
        elif analysis_type == 'frequency_domain':
            summary = 'Frequency Domain Analysis Report'
            recommendations = 'Monitor frequency components for signal quality assessment'
        else:
            summary = f'{analysis_type.replace("_", " ").title()} Analysis Report'
            recommendations = 'Review results for any anomalies or patterns'
        
        # Extract details from results
        details = {}
        for key, value in analysis_results.items():
            if isinstance(value, dict):
                details[key] = value
            else:
                details[key] = str(value)
        
        return {
            'summary': summary,
            'details': details,
            'recommendations': recommendations
        }
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return {
            'summary': 'Error generating report',
            'details': {'error': str(e)},
            'recommendations': 'Please check the logs for more details'
        }


def create_visualization_components(analysis_results: dict, analysis_type: str) -> list:
    """
    Create visualization components for the analysis.
    
    Parameters
    ----------
    analysis_results : dict
        Results from the analysis
    analysis_type : str
        Type of analysis visualization
        
    Returns
    -------
    list
        List of visualization components
    """
    try:
        components = []
        
        # Add summary component
        if analysis_results:
            summary_component = html.Div([
                html.H4(f"{analysis_type.replace('_', ' ').title()} Results"),
                html.P(f"Analysis completed with {len(analysis_results)} result categories")
            ])
            components.append(summary_component)
        
        # Add specific components based on analysis type
        if analysis_type == 'hrv_analysis' and 'hrv_features' in analysis_results:
            hrv_component = html.Div([
                html.H5("HRV Metrics"),
                html.Ul([
                    html.Li(f"{key}: {value}") 
                    for key, value in analysis_results['hrv_features'].items()
                ])
            ])
            components.append(hrv_component)
        
        elif analysis_type == 'frequency_domain' and 'frequency_features' in analysis_results:
            freq_component = html.Div([
                html.H5("Frequency Domain Metrics"),
                html.Ul([
                    html.Li(f"{key}: {value}") 
                    for key, value in analysis_results['frequency_features'].items()
                ])
            ])
            components.append(freq_component)
        
        # Add plots if available
        if 'plots' in analysis_results:
            plots_component = html.Div([
                html.H5("Generated Plots"),
                html.P(f"Number of plots: {len(analysis_results['plots'])}")
            ])
            components.append(plots_component)
        
        # Always add at least one component
        if not components:
            components.append(html.Div([
                html.H5("Analysis Results"),
                html.P("No specific visualization components available")
            ]))
        
        return components
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return [html.Div([
            html.H5("Error"),
            html.P(f"Failed to create visualization: {str(e)}")
        ])]


def perform_hrv_analysis(df: pd.DataFrame, parameters: dict) -> dict:
    """Perform HRV analysis on the data."""
    # Placeholder implementation
    return {
        'hrv_features': {
            'mean_hr': 75.0,
            'sdnn': 45.2,
            'rmssd': 32.1,
            'pnn50': 28.5
        }
    }


def perform_frequency_analysis(df: pd.DataFrame, parameters: dict) -> dict:
    """Perform frequency domain analysis."""
    # Placeholder implementation
    return {
        'frequency_features': {
            'total_power': 1000.0,
            'vlf_power': 150.0,
            'lf_power': 400.0,
            'hf_power': 450.0
        }
    }


def perform_time_domain_analysis(df: pd.DataFrame, parameters: dict) -> dict:
    """Perform time domain analysis."""
    # Placeholder implementation
    return {
        'time_features': {
            'mean': float(df.mean().iloc[0]) if not df.empty else 0.0,
            'std': float(df.std().iloc[0]) if not df.empty else 0.0,
            'min': float(df.min().iloc[0]) if not df.empty else 0.0,
            'max': float(df.max().iloc[0]) if not df.empty else 0.0
        }
    }


def perform_wavelet_analysis(df: pd.DataFrame, parameters: dict) -> dict:
    """Perform wavelet analysis."""
    # Placeholder implementation
    return {
        'wavelet_features': {
            'wavelet_type': parameters.get('wavelet', 'db4'),
            'levels': parameters.get('levels', 6),
            'coefficients': len(df) if not df.empty else 0
        }
    }


def perform_filtering_analysis(df: pd.DataFrame, parameters: dict) -> dict:
    """Perform filtering analysis."""
    # Placeholder implementation
    return {
        'filter_type': parameters.get('filter_type', 'butterworth'),
        'cutoff_freq': parameters.get('cutoff_freq', 50.0),
        'filter_order': parameters.get('filter_order', 4)
    }


def create_analysis_summary(results: dict) -> str:
    """Create a summary of the analysis results."""
    return f"Analysis completed successfully with {len(results)} metrics"


def create_signal_plot(df: pd.DataFrame) -> go.Figure:
    """Create a basic signal plot."""
    fig = go.Figure()
    if not df.empty:
        time_col = df.columns[0] if len(df.columns) > 0 else 'time'
        signal_col = df.columns[1] if len(df.columns) > 1 else 'signal'
        
        fig.add_trace(go.Scatter(
            x=df[time_col] if time_col in df.columns else range(len(df)),
            y=df[signal_col] if signal_col in df.columns else df.iloc[:, 0],
            mode='lines',
            name='Signal'
        ))
    
    fig.update_layout(
        title='Signal Visualization',
        xaxis_title='Time',
        yaxis_title='Amplitude'
    )
    return fig


def create_analysis_charts(df: pd.DataFrame, analysis_type: str) -> list:
    """Create analysis-specific charts."""
    # Placeholder implementation
    return []


def create_summary_table(df: pd.DataFrame) -> html.Table:
    """Create a summary table of the data."""
    if df.empty:
        return html.Table([
            html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
            html.Tbody([html.Tr([html.Td("Status"), html.Td("No data")])])
        ])
    
    stats = {
        'Rows': len(df),
        'Columns': len(df.columns),
        'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
    }
    
    rows = [html.Tr([html.Td(k), html.Td(str(v))]) for k, v in stats.items()]
    
    return html.Table([
        html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
        html.Tbody(rows)
    ])


def register_vitaldsp_callbacks(app):
    """Register all vitalDSP analysis callbacks."""
    
    @app.callback(
        [Output("main-signal-plot", "figure"),
         Output("filtered-signal-plot", "figure"),
         Output("analysis-results", "children"),
         Output("peak-analysis-table", "children"),
         Output("signal-quality-table", "children"),
         Output("filtering-results-table", "children"),
         Output("additional-metrics-table", "children"),
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
            
            # Generate detailed table components
            logger.info("Generating detailed table components...")
            peak_table = create_peak_analysis_table(windowed_data, filtered_data, time_axis, 
                                                   sampling_freq, analysis_options or ["peaks", "hr", "quality"], column_mapping)
            quality_table = create_signal_quality_table(windowed_data, filtered_data, time_axis, 
                                                      sampling_freq, analysis_options or ["peaks", "hr", "quality"], column_mapping)
            filtering_table = create_filtering_results_table(windowed_data, filtered_data, time_axis, 
                                                           sampling_freq, analysis_options or ["peaks", "hr", "quality"], column_mapping)
            additional_table = create_additional_metrics_table(windowed_data, filtered_data, time_axis, 
                                                             sampling_freq, analysis_options or ["peaks", "hr", "quality"], column_mapping)
            logger.info("Table components generated successfully")
            
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
            return main_plot, filtered_plot, analysis_results, peak_table, quality_table, filtering_table, additional_table, time_domain_data, filtered_data_store
            
        except Exception as e:
            logger.error(f"Error in time domain callback: {e}")
            import traceback
            traceback.print_exc()
            error_msg = f"Error in analysis: {str(e)}"
            return create_empty_figure(), create_empty_figure(), error_msg, error_msg, error_msg, error_msg, error_msg, None, None
    
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
        results.append(dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Metric", className="text-center"),
                    html.Th("Value", className="text-center"),
                    html.Th("Unit", className="text-center")
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td("Mean", className="fw-bold"),
                    html.Td(f"{np.mean(raw_signal):.2f}", className="text-end"),
                    html.Td("Signal Units", className="text-muted")
                ]),
                html.Tr([
                    html.Td("Standard Deviation", className="fw-bold"),
                    html.Td(f"{np.std(raw_signal):.2f}", className="text-end"),
                    html.Td("Signal Units", className="text-muted")
                ]),
                html.Tr([
                    html.Td("Minimum", className="fw-bold"),
                    html.Td(f"{np.min(raw_signal):.2f}", className="text-end"),
                    html.Td("Signal Units", className="text-muted")
                ]),
                html.Tr([
                    html.Td("Maximum", className="fw-bold"),
                    html.Td(f"{np.max(raw_signal):.2f}", className="text-end"),
                    html.Td("Signal Units", className="text-muted")
                ]),
                html.Tr([
                    html.Td("RMS", className="fw-bold"),
                    html.Td(f"{np.sqrt(np.mean(raw_signal**2)):.2f}", className="text-end"),
                    html.Td("Signal Units", className="text-muted")
                ])
            ])
        ], bordered=True, hover=True, responsive=True, className="mb-3"))
        
        # Peak detection and heart rate
        if "peaks" in analysis_options:
            peaks, properties = signal.find_peaks(raw_signal, prominence=0.1 * np.std(raw_signal))
            if len(peaks) > 1:
                # Calculate RR intervals
                rr_intervals = np.diff(time_axis[peaks])
                heart_rate = 60 / np.mean(rr_intervals)
                
                results.append(html.Hr())
                results.append(html.H6("❤️ Heart Rate Analysis", className="mb-2"))
                results.append(dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Metric", className="text-center"),
                            html.Th("Value", className="text-center"),
                            html.Th("Unit", className="text-center")
                        ])
                    ]),
                    html.Tbody([
                        html.Tr([
                            html.Td("Detected Peaks", className="fw-bold"),
                            html.Td(f"{len(peaks)}", className="text-end"),
                            html.Td("Count", className="text-muted")
                        ]),
                        html.Tr([
                            html.Td("Average RR Interval", className="fw-bold"),
                            html.Td(f"{np.mean(rr_intervals):.3f}", className="text-end"),
                            html.Td("Seconds", className="text-muted")
                        ]),
                        html.Tr([
                            html.Td("Heart Rate", className="fw-bold"),
                            html.Td(f"{heart_rate:.1f}", className="text-end"),
                            html.Td("BPM", className="text-muted")
                        ]),
                        html.Tr([
                            html.Td("RR Standard Deviation", className="fw-bold"),
                            html.Td(f"{np.std(rr_intervals):.3f}", className="text-end"),
                            html.Td("Seconds", className="text-muted")
                        ])
                    ])
                ], bordered=True, hover=True, responsive=True, className="mb-3"))
        
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
            results.append(dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Metric", className="text-center"),
                        html.Th("Value", className="text-center"),
                        html.Th("Quality", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Signal-to-Noise Ratio", className="fw-bold"),
                        html.Td(f"{snr_db:.1f} dB", className="text-end"),
                        html.Td(html.Span(
                            'Excellent' if snr_db > 20 else 'Good' if snr_db > 15 else 'Fair' if snr_db > 10 else 'Poor',
                            className=f"badge {'bg-success' if snr_db > 20 else 'bg-info' if snr_db > 15 else 'bg-warning' if snr_db > 10 else 'bg-danger'}"
                        ), className="text-center")
                    ]),
                    html.Tr([
                        html.Td("Detected Artifacts", className="fw-bold"),
                        html.Td(f"{artifact_count}", className="text-end"),
                        html.Td(html.Span(
                            'Low' if artifact_count < 5 else 'Medium' if artifact_count < 15 else 'High',
                            className=f"badge {'bg-success' if artifact_count < 5 else 'bg-warning' if artifact_count < 15 else 'bg-danger'}"
                        ), className="text-center")
                    ]),
                    html.Tr([
                        html.Td("Signal Range", className="fw-bold"),
                        html.Td(f"{np.max(raw_signal) - np.min(raw_signal):.2f}", className="text-end"),
                        html.Td("Signal Units", className="text-muted")
                    ])
                ])
            ], bordered=True, hover=True, responsive=True, className="mb-3"))
        
        # Filtering results
        if filtered_data is not None and not filtered_data.equals(raw_data):
            results.append(html.Hr())
            results.append(html.H6("🔧 Filtering Results", className="mb-2"))
            
            # Calculate power reduction
            raw_power = np.mean(raw_signal**2)
            filtered_power = np.mean(filtered_signal**2)
            power_reduction = (raw_power - filtered_power) / raw_power * 100
            
            results.append(dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Metric", className="text-center"),
                        html.Th("Value", className="text-center"),
                        html.Th("Status", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Power Reduction", className="fw-bold"),
                        html.Td(f"{power_reduction:.1f}%", className="text-end"),
                        html.Td(html.Span(
                            'High' if power_reduction > 30 else 'Medium' if power_reduction > 15 else 'Low',
                            className=f"badge {'bg-success' if power_reduction > 30 else 'bg-warning' if power_reduction > 15 else 'bg-info'}"
                        ), className="text-center")
                    ]),
                    html.Tr([
                        html.Td("Signal Preserved", className="fw-bold"),
                        html.Td(f"{100 - power_reduction:.1f}%", className="text-end"),
                        html.Td(html.Span(
                            'Excellent' if (100 - power_reduction) > 85 else 'Good' if (100 - power_reduction) > 70 else 'Fair',
                            className=f"badge {'bg-success' if (100 - power_reduction) > 85 else 'bg-info' if (100 - power_reduction) > 70 else 'bg-warning'}"
                        ), className="text-center")
                    ]),
                    html.Tr([
                        html.Td("Filtering Efficiency", className="fw-bold"),
                        html.Td(f"{power_reduction / (100 - power_reduction):.2f}", className="text-end"),
                        html.Td("Ratio", className="text-muted")
                    ])
                ])
            ], bordered=True, hover=True, responsive=True, className="mb-3"))
        
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


def create_peak_analysis_table(raw_data, filtered_data, time_axis, sampling_freq, analysis_options, column_mapping):
    """Create a comprehensive peak analysis table."""
    if "peaks" not in analysis_options:
        return html.Div([
            html.H6("❤️ Peak Analysis", className="text-muted"),
            html.P("Peak analysis not selected in analysis options.", className="text-muted")
        ])
    
    try:
        # Find signal column
        numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return html.Div([
                html.H6("❤️ Peak Analysis", className="text-muted"),
                html.P("No numeric data available for peak analysis.", className="text-muted")
            ])
            
        signal_col = numeric_cols[0]
        raw_signal = raw_data[signal_col].values
        
        # Peak detection
        peaks, properties = signal.find_peaks(raw_signal, prominence=0.1 * np.std(raw_signal))
        
        if len(peaks) < 2:
            return html.Div([
                html.H6("❤️ Peak Analysis", className="text-muted"),
                html.P("Insufficient peaks detected for analysis.", className="text-muted")
            ])
        
        # Calculate metrics
        rr_intervals = np.diff(time_axis[peaks])
        heart_rates = 60 / rr_intervals
        peak_amplitudes = raw_signal[peaks]
        
        # Create peak details table
        peak_details = []
        for i, (peak_time, peak_amp, rr_int, hr) in enumerate(zip(time_axis[peaks], peak_amplitudes, 
                                                                   np.append(rr_intervals, np.nan), 
                                                                   np.append(heart_rates, np.nan))):
            peak_details.append([
                f"P{i+1}",
                f"{peak_time:.3f}s",
                f"{peak_amp:.3f}",
                f"{rr_int:.3f}s" if not np.isnan(rr_int) else "N/A",
                f"{hr:.1f} BPM" if not np.isnan(hr) else "N/A"
            ])
        
        return html.Div([
            html.H6("❤️ Peak Analysis", className="text-primary mb-3"),
            
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
                            html.H4(f"{np.mean(heart_rates):.1f}", className="text-center text-success mb-0"),
                            html.Small("Mean HR (BPM)", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-success")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{np.std(heart_rates):.1f}", className="text-center text-warning mb-0"),
                            html.Small("HR Std (BPM)", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-warning")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{np.mean(rr_intervals):.3f}", className="text-center text-info mb-0"),
                            html.Small("Mean RR (s)", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-info")
                ], md=3)
            ], className="mb-3"),
            
            # Peak details table
            html.H6("Peak Details", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Peak", className="text-center"),
                        html.Th("Time (s)", className="text-center"),
                        html.Th("Amplitude", className="text-center"),
                        html.Th("RR Interval (s)", className="text-center"),
                        html.Th("Heart Rate (BPM)", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(detail[0], className="fw-bold"),
                        html.Td(detail[1], className="text-center"),
                        html.Td(detail[2], className="text-center"),
                        html.Td(detail[3], className="text-center"),
                        html.Td(detail[4], className="text-center")
                    ]) for detail in peak_details[:20]  # Limit to first 20 peaks
                ])
            ], bordered=True, hover=True, responsive=True, size="sm", className="mb-3"),
            
            # Additional peak statistics
            html.H6("Peak Statistics", className="mb-2"),
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
                        html.Td("Peak Amplitude Range", className="fw-bold"),
                        html.Td(f"{np.max(peak_amplitudes) - np.min(peak_amplitudes):.3f}", className="text-end"),
                        html.Td("Signal Units", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Peak Amplitude Mean", className="fw-bold"),
                        html.Td(f"{np.mean(peak_amplitudes):.3f}", className="text-end"),
                        html.Td("Signal Units", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Peak Amplitude Std", className="fw-bold"),
                        html.Td(f"{np.std(peak_amplitudes):.3f}", className="text-end"),
                        html.Td("Signal Units", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("RR Interval Std", className="fw-bold"),
                        html.Td(f"{np.std(rr_intervals):.3f}", className="text-end"),
                        html.Td("Seconds", className="text-muted")
                    ])
                ])
            ], bordered=True, hover=True, responsive=True, size="sm")
        ])
        
    except Exception as e:
        logger.error(f"Error creating peak analysis table: {e}")
        return html.Div([
            html.H6("❤️ Peak Analysis", className="text-muted"),
            html.P(f"Error creating peak analysis: {str(e)}", className="text-danger")
        ])


def create_signal_quality_table(raw_data, filtered_data, time_axis, sampling_freq, analysis_options, column_mapping):
    """Create a comprehensive signal quality assessment table."""
    if "quality" not in analysis_options:
        return html.Div([
            html.H6("🎯 Signal Quality", className="text-muted"),
            html.P("Signal quality assessment not selected in analysis options.", className="text-muted")
        ])
    
    try:
        # Find signal column
        numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return html.Div([
                html.H6("🎯 Signal Quality", className="text-muted"),
                html.P("No numeric data available for quality assessment.", className="text-muted")
            ])
            
        signal_col = numeric_cols[0]
        raw_signal = raw_data[signal_col].values
        
        # Calculate quality metrics
        signal_power = np.mean(raw_signal**2)
        noise_power = np.var(raw_signal)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        
        # Artifact detection using IQR method
        q75, q25 = np.percentile(raw_signal, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        artifact_count = np.sum((raw_signal < lower_bound) | (raw_signal > upper_bound))
        artifact_percentage = (artifact_count / len(raw_signal)) * 100
        
        # Signal stability metrics
        moving_std = []
        window_size = int(0.5 * sampling_freq)
        for i in range(0, len(raw_signal) - window_size, window_size // 4):
            window = raw_signal[i:i + window_size]
            moving_std.append(np.std(window))
        
        stability_score = 1 - (np.std(moving_std) / np.mean(moving_std)) if np.mean(moving_std) > 0 else 0
        
        return html.Div([
            html.H6("🎯 Signal Quality Assessment", className="text-success mb-3"),
            
            # Quality score cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{snr_db:.1f} dB", className="text-center text-primary mb-0"),
                            html.Small("Signal-to-Noise Ratio", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-primary")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{artifact_percentage:.1f}%", className="text-center text-warning mb-0"),
                            html.Small("Artifact Percentage", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-warning")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{stability_score:.2f}", className="text-center text-success mb-0"),
                            html.Small("Stability Score", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-success")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{len(raw_signal)}", className="text-center text-info mb-0"),
                            html.Small("Total Samples", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-info")
                ], md=3)
            ], className="mb-3"),
            
            # Quality metrics table
            html.H6("Quality Metrics", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Metric", className="text-center"),
                        html.Th("Value", className="text-center"),
                        html.Th("Quality", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Signal-to-Noise Ratio", className="fw-bold"),
                        html.Td(f"{snr_db:.1f} dB", className="text-end"),
                        html.Td(html.Span(
                            'Excellent' if snr_db > 20 else 'Good' if snr_db > 15 else 'Fair' if snr_db > 10 else 'Poor',
                            className=f"badge {'bg-success' if snr_db > 20 else 'bg-info' if snr_db > 15 else 'bg-warning' if snr_db > 10 else 'bg-danger'}"
                        ), className="text-center")
                    ]),
                    html.Tr([
                        html.Td("Detected Artifacts", className="fw-bold"),
                        html.Td(f"{artifact_count} ({artifact_percentage:.1f}%)", className="text-end"),
                        html.Td(html.Span(
                            'Low' if artifact_percentage < 5 else 'Medium' if artifact_percentage < 15 else 'High',
                            className=f"badge {'bg-success' if artifact_percentage < 5 else 'bg-warning' if artifact_percentage < 15 else 'bg-danger'}"
                        ), className="text-center")
                    ]),
                    html.Tr([
                        html.Td("Signal Stability", className="fw-bold"),
                        html.Td(f"{stability_score:.2f}", className="text-end"),
                        html.Td(html.Span(
                            'High' if stability_score > 0.8 else 'Medium' if stability_score > 0.6 else 'Low',
                            className=f"badge {'bg-success' if stability_score > 0.8 else 'bg-warning' if stability_score > 0.6 else 'bg-danger'}"
                        ), className="text-center")
                    ]),
                    html.Tr([
                        html.Td("Signal Range", className="fw-bold"),
                        html.Td(f"{np.max(raw_signal) - np.min(raw_signal):.3f}", className="text-end"),
                        html.Td("Signal Units", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Signal Power", className="fw-bold"),
                        html.Td(f"{signal_power:.3f}", className="text-end"),
                        html.Td("Signal Units²", className="text-muted")
                    ])
                ])
            ], bordered=True, hover=True, responsive=True, size="sm")
        ])
        
    except Exception as e:
        logger.error(f"Error creating signal quality table: {e}")
        return html.Div([
            html.H6("🎯 Signal Quality", className="text-muted"),
            html.P(f"Error creating quality assessment: {str(e)}", className="text-danger")
        ])


def create_filtering_results_table(raw_data, filtered_data, time_axis, sampling_freq, analysis_options, column_mapping):
    """Create a comprehensive filtering results table."""
    try:
        # Check if filtering was applied
        if filtered_data is None or filtered_data.equals(raw_data):
            return html.Div([
                html.H6("🔧 Filtering Results", className="text-muted"),
                html.P("No filtering applied or filtering parameters not configured.", className="text-muted")
            ])
        
        # Find signal column
        numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return html.Div([
                html.H6("🔧 Filtering Results", className="text-muted"),
                html.P("No numeric data available for filtering analysis.", className="text-muted")
            ])
            
        signal_col = numeric_cols[0]
        raw_signal = raw_data[signal_col].values
        filtered_signal = filtered_data[signal_col].values
        
        # Calculate filtering metrics
        raw_power = np.mean(raw_signal**2)
        filtered_power = np.mean(filtered_signal**2)
        power_reduction = (raw_power - filtered_power) / raw_power * 100 if raw_power > 0 else 0
        
        # Frequency domain analysis
        raw_fft = np.abs(np.fft.rfft(raw_signal))
        filtered_fft = np.abs(np.fft.rfft(filtered_signal))
        freqs = np.fft.rfftfreq(len(raw_signal), 1/sampling_freq)
        
        # Calculate frequency-specific improvements
        low_freq_mask = freqs < 1.0  # Below 1 Hz
        mid_freq_mask = (freqs >= 1.0) & (freqs < 10.0)  # 1-10 Hz
        high_freq_mask = freqs >= 10.0  # Above 10 Hz
        
        low_freq_improvement = np.mean(filtered_fft[low_freq_mask]) / np.mean(raw_fft[low_freq_mask]) if np.mean(raw_fft[low_freq_mask]) > 0 else 1
        mid_freq_improvement = np.mean(filtered_fft[mid_freq_mask]) / np.mean(raw_fft[mid_freq_mask]) if np.mean(raw_fft[mid_freq_mask]) > 0 else 1
        high_freq_improvement = np.mean(filtered_fft[high_freq_mask]) / np.mean(raw_fft[high_freq_mask]) if np.mean(raw_fft[high_freq_mask]) > 0 else 1
        
        return html.Div([
            html.H6("🔧 Filtering Results", className="text-info mb-3"),
            
            # Filtering summary cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{power_reduction:.1f}%", className="text-center text-primary mb-0"),
                            html.Small("Power Reduction", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-primary")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{low_freq_improvement:.2f}x", className="text-center text-success mb-0"),
                            html.Small("Low Freq Improvement", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-success")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{mid_freq_improvement:.2f}x", className="text-center text-warning mb-0"),
                            html.Small("Mid Freq Improvement", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-warning")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{high_freq_improvement:.2f}x", className="text-center text-info mb-0"),
                            html.Small("High Freq Improvement", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-info")
                ], md=3)
            ], className="mb-3"),
            
            # Filtering metrics table
            html.H6("Filtering Metrics", className="mb-2"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Metric", className="text-center"),
                        html.Th("Raw Signal", className="text-center"),
                        html.Th("Filtered Signal", className="text-center"),
                        html.Th("Improvement", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Signal Power", className="fw-bold"),
                        html.Td(f"{raw_power:.3f}", className="text-end"),
                        html.Td(f"{filtered_power:.3f}", className="text-end"),
                        html.Td(f"{(filtered_power/raw_power)*100:.1f}%" if raw_power > 0 else "N/A", className="text-end")
                    ]),
                    html.Tr([
                        html.Td("Signal Variance", className="fw-bold"),
                        html.Td(f"{np.var(raw_signal):.3f}", className="text-end"),
                        html.Td(f"{np.var(filtered_signal):.3f}", className="text-end"),
                        html.Td(f"{(np.var(filtered_signal)/np.var(raw_signal))*100:.1f}%" if np.var(raw_signal) > 0 else "N/A", className="text-end")
                    ]),
                    html.Tr([
                        html.Td("Peak-to-Peak", className="fw-bold"),
                        html.Td(f"{np.max(raw_signal) - np.min(raw_signal):.3f}", className="text-end"),
                        html.Td(f"{np.max(filtered_signal) - np.min(filtered_signal):.3f}", className="text-end"),
                        html.Td("N/A", className="text-end")
                    ]),
                    html.Tr([
                        html.Td("Low Freq Content", className="fw-bold"),
                        html.Td(f"{np.mean(raw_fft[low_freq_mask]):.3f}", className="text-end"),
                        html.Td(f"{np.mean(filtered_fft[low_freq_mask]):.3f}", className="text-end"),
                        html.Td(f"{low_freq_improvement:.2f}x", className="text-end")
                    ]),
                    html.Tr([
                        html.Td("Mid Freq Content", className="fw-bold"),
                        html.Td(f"{np.mean(raw_fft[mid_freq_mask]):.3f}", className="text-end"),
                        html.Td(f"{np.mean(filtered_fft[mid_freq_mask]):.3f}", className="text-end"),
                        html.Td(f"{mid_freq_improvement:.2f}x", className="text-end")
                    ]),
                    html.Tr([
                        html.Td("High Freq Content", className="fw-bold"),
                        html.Td(f"{np.mean(raw_fft[high_freq_mask]):.3f}", className="text-end"),
                        html.Td(f"{np.mean(filtered_fft[high_freq_mask]):.3f}", className="text-end"),
                        html.Td(f"{high_freq_improvement:.2f}x", className="text-end")
                    ])
                ])
            ], bordered=True, hover=True, responsive=True, size="sm")
        ])
        
    except Exception as e:
        logger.error(f"Error creating filtering results table: {e}")
        return html.Div([
            html.H6("🔧 Filtering Results", className="text-muted"),
            html.P(f"Error creating filtering results: {str(e)}", className="text-danger")
        ])


def create_additional_metrics_table(raw_data, filtered_data, time_axis, sampling_freq, analysis_options, column_mapping):
    """Create additional metrics and insights table."""
    try:
        # Find signal column
        numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return html.Div([
                html.H6("📊 Additional Metrics", className="text-muted"),
                html.P("No numeric data available for additional metrics.", className="text-muted")
            ])
            
        signal_col = numeric_cols[0]
        raw_signal = raw_data[signal_col].values
        
        # Calculate additional metrics
        signal_skewness = np.mean(((raw_signal - np.mean(raw_signal)) / np.std(raw_signal))**3) if np.std(raw_signal) > 0 else 0
        signal_kurtosis = np.mean(((raw_signal - np.mean(raw_signal)) / np.std(raw_signal))**4) if np.std(raw_signal) > 0 else 0
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.signbit(raw_signal - np.mean(raw_signal))))
        zero_crossing_rate = zero_crossings / len(raw_signal)
        
        # Signal complexity (approximate entropy)
        def approximate_entropy(signal, m=2, r=0.2):
            try:
                N = len(signal)
                if N < m + 2:
                    return 0
                
                # Normalize signal
                signal_norm = (signal - np.mean(signal)) / np.std(signal)
                r = r * np.std(signal_norm)
                
                # Calculate phi(m) and phi(m+1)
                phi_m = 0
                phi_m1 = 0
                
                for i in range(N - m + 1):
                    pattern_m = signal_norm[i:i+m]
                    count_m = 0
                    count_m1 = 0
                    
                    for j in range(N - m + 1):
                        if np.all(np.abs(pattern_m - signal_norm[j:j+m]) <= r):
                            count_m += 1
                        if j < N - m:
                            if np.all(np.abs(pattern_m - signal_norm[j:j+m+1][:m]) <= r):
                                count_m1 += 1
                    
                    phi_m += np.log(count_m / (N - m + 1))
                    phi_m1 += np.log(count_m1 / (N - m + 1))
                
                phi_m /= (N - m + 1)
                phi_m1 /= (N - m + 1)
                
                return phi_m - phi_m1
            except:
                return 0
        
        complexity = approximate_entropy(raw_signal)
        
        return html.Div([
            html.H6("📊 Additional Metrics & Insights", className="text-secondary mb-3"),
            
            # Additional metrics cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{signal_skewness:.3f}", className="text-center text-primary mb-0"),
                            html.Small("Skewness", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-primary")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{signal_kurtosis:.3f}", className="text-center text-success mb-0"),
                            html.Small("Kurtosis", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-success")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{zero_crossing_rate:.3f}", className="text-center text-warning mb-0"),
                            html.Small("Zero Crossing Rate", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-warning")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{complexity:.3f}", className="text-center text-info mb-0"),
                            html.Small("Complexity", className="text-center d-block text-muted")
                        ])
                    ], className="text-center border-info")
                ], md=3)
            ], className="mb-3"),
            
            # Additional metrics table
            html.H6("Statistical Metrics", className="mb-2"),
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
                        html.Td("Signal Skewness", className="fw-bold"),
                        html.Td(f"{signal_skewness:.3f}", className="text-end"),
                        html.Td(html.Span(
                            'Symmetric' if abs(signal_skewness) < 0.5 else 'Right Skewed' if signal_skewness > 0.5 else 'Left Skewed',
                            className=f"badge {'bg-success' if abs(signal_skewness) < 0.5 else 'bg-warning'}"
                        ), className="text-center")
                    ]),
                    html.Tr([
                        html.Td("Signal Kurtosis", className="fw-bold"),
                        html.Td(f"{signal_kurtosis:.3f}", className="text-end"),
                        html.Td(html.Span(
                            'Normal' if abs(signal_kurtosis - 3) < 1 else 'Heavy Tailed' if signal_kurtosis > 4 else 'Light Tailed',
                            className=f"badge {'bg-success' if abs(signal_kurtosis - 3) < 1 else 'bg-warning'}"
                        ), className="text-center")
                    ]),
                    html.Tr([
                        html.Td("Zero Crossing Rate", className="fw-bold"),
                        html.Td(f"{zero_crossing_rate:.3f}", className="text-end"),
                        html.Td(html.Span(
                            'High Frequency' if zero_crossing_rate > 0.3 else 'Medium Frequency' if zero_crossing_rate > 0.1 else 'Low Frequency',
                            className=f"badge {'bg-danger' if zero_crossing_rate > 0.3 else 'bg-warning' if zero_crossing_rate > 0.1 else 'bg-success'}"
                        ), className="text-center")
                    ]),
                    html.Tr([
                        html.Td("Signal Complexity", className="fw-bold"),
                        html.Td(f"{complexity:.3f}", className="text-end"),
                        html.Td(html.Span(
                            'High' if complexity > 1.0 else 'Medium' if complexity > 0.5 else 'Low',
                            className=f"badge {'bg-danger' if complexity > 1.0 else 'bg-warning' if complexity > 0.5 else 'bg-success'}"
                        ), className="text-center")
                    ]),
                    html.Tr([
                        html.Td("Signal Duration", className="fw-bold"),
                        html.Td(f"{len(raw_signal)/sampling_freq:.2f}s", className="text-end"),
                        html.Td("Duration", className="text-muted")
                    ]),
                    html.Tr([
                        html.Td("Sampling Frequency", className="fw-bold"),
                        html.Td(f"{sampling_freq} Hz", className="text-end"),
                        html.Td("Frequency", className="text-muted")
                    ])
                ])
            ], bordered=True, hover=True, responsive=True, size="sm")
        ])
        
    except Exception as e:
        logger.error(f"Error creating additional metrics table: {e}")
        return html.Div([
            html.H6("📊 Additional Metrics", className="text-muted"),
            html.P(f"Error creating additional metrics: {str(e)}", className="text-danger")
        ])
