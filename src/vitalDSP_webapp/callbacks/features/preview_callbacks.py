"""
Preview and visualization callbacks for vitalDSP webapp.

This module handles data preview and visualization functionality.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html, dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import logging

logger = logging.getLogger(__name__)


def register_preview_callbacks(app):
    """Register all preview and visualization callbacks."""
    
    @app.callback(
        [Output("data-preview-plot", "figure"),
         Output("data-summary", "children")],
        [Input("btn-preview-data", "n_clicks"),
         Input("preview-window-size", "value")],
        [State("store-uploaded-data", "data"),
         State("store-data-config", "data")]
    )
    def preview_data(n_clicks, window_size, data_store, config_store):
        """Preview the uploaded data with configurable window size."""
        if not n_clicks or not data_store or not config_store:
            raise PreventUpdate
            
        try:
            # Extract data
            df = pd.DataFrame(data_store["data"])
            sampling_freq = config_store["sampling_freq"]
            
            # Set default window size
            window_size = window_size or 1000
            
            # Limit window size to data length
            if window_size > len(df):
                window_size = len(df)
            
            # Get data for preview
            preview_df = df.head(window_size)
            signal_data = preview_df.iloc[:, 1].values  # Assuming second column is signal
            
            # Create time axis
            time_axis = np.arange(len(signal_data)) / sampling_freq
            
            # Create preview plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=signal_data,
                mode='lines',
                name='Signal',
                line=dict(color='blue', width=1)
            ))
            
            fig.update_layout(
                title=f"Data Preview (First {window_size} samples)",
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude",
                showlegend=True,
                height=400
            )
            
            # Create data summary
            summary = html.Div([
                html.H5("Data Summary"),
                html.P(f"Total Samples: {len(df)}"),
                html.P(f"Preview Window: {window_size} samples"),
                html.P(f"Sampling Frequency: {sampling_freq} Hz"),
                html.P(f"Total Duration: {len(df)/sampling_freq:.2f} seconds"),
                html.P(f"Preview Duration: {window_size/sampling_freq:.2f} seconds"),
                html.P(f"Columns: {', '.join(df.columns)}"),
                html.Hr(),
                html.H6("Preview Statistics:"),
                html.P(f"Mean: {np.mean(signal_data):.4f}"),
                html.P(f"Std: {np.std(signal_data):.4f}"),
                html.P(f"Min: {np.min(signal_data):.4f}"),
                html.P(f"Max: {np.max(signal_data):.4f}")
            ])
            
            return fig, summary
            
        except Exception as e:
            logger.error(f"Error in data preview: {e}")
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            
            error_summary = html.Div([
                html.H5("Error"),
                html.P(f"Data preview failed: {str(e)}")
            ])
            
            return error_fig, error_summary


    @app.callback(
        Output("comparison-plot", "figure"),
        [Input("btn-compare-signals", "n_clicks")],
        [State("store-uploaded-data", "data"),
         State("store-data-config", "data"),
         State("comparison-type", "value")]
    )
    def compare_signals(n_clicks, data_store, config_store, comparison_type):
        """Compare different aspects of the signal."""
        if not n_clicks or not data_store or not config_store:
            raise PreventUpdate
            
        try:
            # Extract data
            df = pd.DataFrame(data_store["data"])
            sampling_freq = config_store["sampling_freq"]
            signal_data = df.iloc[:, 1].values
            
            # Set default comparison type
            comparison_type = comparison_type or "time_frequency"
            
            if comparison_type == "time_frequency":
                # Create subplot with time and frequency domains
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Time Domain", "Frequency Domain"),
                    vertical_spacing=0.1
                )
                
                # Time domain
                time_axis = np.arange(len(signal_data)) / sampling_freq
                fig.add_trace(
                    go.Scatter(x=time_axis, y=signal_data, mode='lines', name='Signal'),
                    row=1, col=1
                )
                
                # Frequency domain
                fft_result = np.fft.fft(signal_data)
                fft_freq = np.fft.fftfreq(len(signal_data), 1/sampling_freq)
                positive_mask = fft_freq > 0
                fft_freq = fft_freq[positive_mask]
                fft_magnitude = np.abs(fft_result[positive_mask])
                
                fig.add_trace(
                    go.Scatter(x=fft_freq, y=fft_magnitude, mode='lines', name='FFT'),
                    row=2, col=1
                )
                
                fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
                fig.update_yaxes(title_text="Amplitude", row=1, col=1)
                fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
                fig.update_yaxes(title_text="Magnitude", row=2, col=1)
                
            elif comparison_type == "statistical":
                # Create subplot with original signal and histogram
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Original Signal", "Signal Distribution"),
                    specs=[[{"type": "xy"}, {"type": "histogram"}]]
                )
                
                # Original signal
                time_axis = np.arange(len(signal_data)) / sampling_freq
                fig.add_trace(
                    go.Scatter(x=time_axis, y=signal_data, mode='lines', name='Signal'),
                    row=1, col=1
                )
                
                # Histogram
                fig.add_trace(
                    go.Histogram(x=signal_data, nbinsx=50, name='Distribution'),
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
                fig.update_yaxes(title_text="Amplitude", row=1, col=1)
                fig.update_xaxes(title_text="Amplitude", row=1, col=2)
                fig.update_yaxes(title_text="Count", row=1, col=2)
                
            else:
                raise ValueError(f"Unsupported comparison type: {comparison_type}")
            
            fig.update_layout(
                title=f"Signal Comparison - {comparison_type.replace('_', ' ').title()}",
                height=600,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error in signal comparison: {e}")
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return error_fig
