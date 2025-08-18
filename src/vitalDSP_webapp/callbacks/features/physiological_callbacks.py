"""
Physiological feature extraction callbacks for vitalDSP webapp.

This module handles extraction of physiological features from signals.
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

logger = logging.getLogger(__name__)


def register_physiological_callbacks(app):
    """Register all physiological analysis callbacks."""
    logger.info("=== REGISTERING PHYSIOLOGICAL CALLBACKS ===")
    
    # Import vitalDSP modules when callbacks are registered
    _import_vitaldsp_modules()
    
    @app.callback(
        [Output("physio-main-signal-plot", "figure"),
         Output("physio-analysis-results", "children"),
         Output("physio-analysis-plots", "figure"),
         Output("store-physio-data", "data"),
         Output("store-physio-features", "data")],
        [Input("url", "pathname"),
         Input("physio-btn-update-analysis", "n_clicks"),
         Input("physio-time-range-slider", "value"),
         Input("physio-btn-nudge-m10", "n_clicks"),
         Input("physio-btn-nudge-m1", "n_clicks"),
         Input("physio-btn-nudge-p1", "n_clicks"),
         Input("physio-btn-nudge-p10", "n_clicks")],
        [State("physio-start-time", "value"),
         State("physio-end-time", "value"),
         State("physio-signal-type", "value"),
         State("physio-analysis-categories", "value"),
         State("physio-hrv-options", "value"),
         State("physio-morphology-options", "value"),
         State("physio-advanced-options", "value")]
    )
    def physiological_analysis_callback(pathname, n_clicks, slider_value, nudge_m10, nudge_m1, nudge_p1, nudge_p10,
                                     start_time, end_time, signal_type, analysis_categories, hrv_options, 
                                     morphology_options, advanced_options):
        """Unified callback for physiological analysis - handles both page load and user interactions."""
        ctx = callback_context
        
        # Determine what triggered this callback
        if not ctx.triggered:
            logger.warning("No context triggered - raising PreventUpdate")
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        logger.info(f"=== PHYSIOLOGICAL ANALYSIS CALLBACK TRIGGERED ===")
        logger.info(f"Trigger ID: {trigger_id}")
        logger.info(f"Pathname: {pathname}")
        logger.info(f"Analysis categories: {analysis_categories}")
        logger.info(f"HRV options: {hrv_options}")
        logger.info(f"Morphology options: {morphology_options}")
        logger.info(f"Advanced options: {advanced_options}")
        
        # Only run this when we're on the physiological page
        if pathname != "/physiological":
            logger.info("Not on physiological page, returning empty figures")
            return create_empty_figure(), "Navigate to Physiological Features page", create_empty_figure(), None, None
        
        try:
            # Get data from the data service
            from vitalDSP_webapp.services.data.data_service import get_data_service
            data_service = get_data_service()
            
            # Get the most recent data
            all_data = data_service.get_all_data()
            if not all_data:
                logger.warning("No data found in service")
                return create_empty_figure(), "No data available. Please upload and process data first.", create_empty_figure(), None, None
            
            # Get the most recent data entry
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            
            logger.info(f"Found data: {latest_data_id}")
            
            # Get column mapping
            column_mapping = data_service.get_column_mapping(latest_data_id)
            if not column_mapping:
                logger.warning("Data has not been processed yet - no column mapping found")
                return create_empty_figure(), "Please process your data on the Upload page first (configure column mapping)", create_empty_figure(), None, None
                
            logger.info(f"Column mapping found: {column_mapping}")
                
            df = data_service.get_data(latest_data_id)
            if df is None or df.empty:
                logger.warning("Data frame is empty")
                return create_empty_figure(), "Data is empty or corrupted.", create_empty_figure(), None, None
            
            # Get sampling frequency from the data info
            sampling_freq = latest_data.get('info', {}).get('sampling_freq', 1000)
            logger.info(f"Sampling frequency: {sampling_freq}")
            
            # Handle time window adjustments for nudge buttons
            if trigger_id in ["physio-btn-nudge-m10", "physio-btn-nudge-m1", "physio-btn-nudge-p1", "physio-btn-nudge-p10"]:
                if not start_time or not end_time:
                    start_time, end_time = 0, 10
                    
                if trigger_id == "physio-btn-nudge-m10":
                    start_time = max(0, start_time - 10)
                    end_time = max(10, end_time - 10)
                elif trigger_id == "physio-btn-nudge-m1":
                    start_time = max(0, start_time - 1)
                    end_time = max(1, end_time - 1)
                elif trigger_id == "physio-btn-nudge-p1":
                    start_time = start_time + 1
                    end_time = end_time + 1
                elif trigger_id == "physio-btn-nudge-p10":
                    start_time = start_time + 10
                    end_time = end_time + 10
            
            # Handle time window from slider
            if trigger_id == "physio-time-range-slider" and slider_value:
                start_time, end_time = slider_value[0], slider_value[1]
            
            # Set default values if not provided
            start_time = start_time or 0
            end_time = end_time or 10
            signal_type = signal_type or "auto"
            analysis_categories = analysis_categories or ["hrv", "morphology"]
            hrv_options = hrv_options or ["time_domain"]
            morphology_options = morphology_options or ["peak_detection"]
            advanced_options = advanced_options or []
            
            logger.info(f"Time window: {start_time} - {end_time}")
            logger.info(f"Signal type: {signal_type}")
            logger.info(f"Analysis categories: {analysis_categories}")
            
            # Extract time and signal columns
            time_col = column_mapping.get('time', df.columns[0])
            signal_col = column_mapping.get('signal', df.columns[1])
            
            # Extract time and signal data
            time_data = df[time_col].values
            signal_data = df[signal_col].values
            
            # Apply time window
            start_idx = np.searchsorted(time_data, start_time)
            end_idx = np.searchsorted(time_data, end_time)
            if end_idx > start_idx:
                time_data = time_data[start_idx:end_idx]
                signal_data = signal_data[start_idx:end_idx]
            
            # Auto-detect signal type if needed
            if signal_type == "auto":
                signal_type = detect_physiological_signal_type(signal_data, sampling_freq)
                logger.info(f"Auto-detected signal type: {signal_type}")
            
            # Create main signal plot
            main_fig = create_physiological_signal_plot(time_data, signal_data, signal_type, sampling_freq)
            
            # Perform analysis based on selected categories
            analysis_results = perform_physiological_analysis(
                time_data, signal_data, signal_type, sampling_freq,
                analysis_categories, hrv_options, morphology_options, advanced_options
            )
            
            # Create analysis plots
            analysis_fig = create_physiological_analysis_plots(
                time_data, signal_data, signal_type, sampling_freq,
                analysis_categories, hrv_options, morphology_options, advanced_options
            )
            
            # Store data for other callbacks
            physio_data = {
                "time_data": time_data.tolist(),
                "signal_data": signal_data.tolist(),
                "signal_type": signal_type,
                "sampling_freq": sampling_freq,
                "analysis_categories": analysis_categories
            }
            
            physio_features = {
                "hrv_metrics": analysis_results.get("hrv_metrics", {}),
                "morphology_metrics": analysis_results.get("morphology_metrics", {}),
                "quality_metrics": analysis_results.get("quality_metrics", {}),
                "trend_metrics": analysis_results.get("trend_metrics", {})
            }
            
            logger.info("Physiological analysis completed successfully")
            return main_fig, analysis_results, analysis_fig, physio_data, physio_features
            
        except Exception as e:
            logger.error(f"Error in physiological analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            error_fig = create_empty_figure()
            error_fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            
            error_results = html.Div([
                html.H5("Error in Physiological Analysis"),
                html.P(f"Analysis failed: {str(e)}"),
                html.P("Please check your data and parameters.")
            ])
            
            return error_fig, error_results, error_fig, None, None

    # Time input update callbacks
    @app.callback(
        [Output("physio-start-time", "value"),
         Output("physio-end-time", "value")],
        [Input("physio-time-range-slider", "value"),
         Input("physio-btn-nudge-m10", "n_clicks"),
         Input("physio-btn-nudge-m1", "n_clicks"),
         Input("physio-btn-nudge-p1", "n_clicks"),
         Input("physio-btn-nudge-p10", "n_clicks")],
        [State("physio-start-time", "value"),
         State("physio-end-time", "value")]
    )
    def update_physio_time_inputs(slider_value, nudge_m10, nudge_m1, nudge_p1, nudge_p10, start_time, end_time):
        """Update time inputs based on slider or nudge buttons."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if trigger_id == "physio-time-range-slider" and slider_value:
            return slider_value[0], slider_value[1]
        
        # Handle nudge buttons
        time_window = end_time - start_time if start_time and end_time else 10
        
        if trigger_id == "physio-btn-nudge-m10":
            new_start = max(0, start_time - 10) if start_time else 0
            new_end = new_start + time_window
            return new_start, new_end
        elif trigger_id == "physio-btn-nudge-m1":
            new_start = max(0, start_time - 1) if start_time else 0
            new_end = new_start + time_window
            return new_start, new_end
        elif trigger_id == "physio-btn-nudge-p1":
            new_start = start_time + 1 if start_time else 1
            new_end = new_start + time_window
            return new_start, new_end
        elif trigger_id == "physio-btn-nudge-p10":
            new_start = start_time + 10 if start_time else 10
            new_end = new_start + time_window
            return new_start, new_end
        
        return no_update, no_update

    # Time slider range update callback
    @app.callback(
        Output("physio-time-range-slider", "max"),
        [Input("store-uploaded-data", "data")]
    )
    def update_physio_time_slider_range(data_store):
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


# Helper functions for physiological analysis
def _import_vitaldsp_modules():
    """Import vitalDSP modules when needed."""
    try:
        # This would import vitalDSP modules if available
        # For now, we'll use scipy and numpy
        pass
    except ImportError:
        logger.warning("vitalDSP modules not available, using scipy/numpy fallback")


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


def detect_physiological_signal_type(signal_data, sampling_freq):
    """Auto-detect the type of physiological signal."""
    try:
        # Simple heuristics for signal type detection
        # Calculate basic statistics
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        peak_to_peak = np.max(signal_data) - np.min(signal_data)
        
        # Find peaks for frequency analysis
        peaks, _ = signal.find_peaks(signal_data, height=mean_val + std_val, distance=int(sampling_freq * 0.3))
        
        if len(peaks) > 1:
            intervals = np.diff(peaks) / sampling_freq
            if np.mean(intervals) < 1.0:  # Less than 1 second between peaks
                return "ecg"  # Likely ECG (faster heart rate)
            else:
                return "ppg"  # Likely PPG (slower, more variable)
        else:
            return "ppg"  # Default to PPG if unclear
            
    except Exception as e:
        logger.warning(f"Error in signal type detection: {e}")
        return "ppg"  # Default fallback


def create_physiological_signal_plot(time_data, signal_data, signal_type, sampling_freq):
    """Create the main physiological signal plot."""
    fig = go.Figure()
    
    # Add main signal
    fig.add_trace(go.Scatter(
        x=time_data,
        y=signal_data,
        mode='lines',
        name=f'{signal_type.upper()} Signal',
        line=dict(color='blue', width=1)
    ))
    
    # Add peak detection if signal type supports it
    if signal_type in ["ecg", "ppg"]:
        try:
            peaks, _ = signal.find_peaks(signal_data, height=np.mean(signal_data) + np.std(signal_data), 
                                       distance=int(sampling_freq * 0.3))
            if len(peaks) > 0:
                fig.add_trace(go.Scatter(
                    x=time_data[peaks],
                    y=signal_data[peaks],
                    mode='markers',
                    name='Detected Peaks',
                    marker=dict(color='red', size=6, symbol='circle')
                ))
        except Exception as e:
            logger.warning(f"Error in peak detection: {e}")
    
    fig.update_layout(
        title=f"{signal_type.upper()} Signal Analysis",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def perform_physiological_analysis(time_data, signal_data, signal_type, sampling_freq, 
                                 analysis_categories, hrv_options, morphology_options, advanced_options):
    """Perform comprehensive physiological analysis."""
    results = {}
    
    try:
        # HRV Analysis
        if "hrv" in analysis_categories:
            results["hrv_metrics"] = analyze_hrv(signal_data, sampling_freq, hrv_options)
        
        # Morphology Analysis
        if "morphology" in analysis_categories:
            results["morphology_metrics"] = analyze_morphology(signal_data, sampling_freq, morphology_options)
        
        # Quality Analysis
        if "quality" in analysis_categories:
            results["quality_metrics"] = analyze_signal_quality(signal_data, sampling_freq)
        
        # Trend Analysis
        if "trend" in analysis_categories:
            results["trend_metrics"] = analyze_trends(signal_data, sampling_freq)
        
        # Create comprehensive results display
        return create_comprehensive_results_display(results, signal_type, sampling_freq)
        
    except Exception as e:
        logger.error(f"Error in physiological analysis: {e}")
        return html.Div([
            html.H5("Analysis Error"),
            html.P(f"Analysis failed: {str(e)}")
        ])


def create_physiological_analysis_plots(time_data, signal_data, signal_type, sampling_freq,
                                       analysis_categories, hrv_options, morphology_options, advanced_options):
    """Create detailed analysis plots."""
    try:
        if "hrv" in analysis_categories and "time_domain" in hrv_options:
            return create_hrv_plots(time_data, signal_data, sampling_freq)
        elif "morphology" in analysis_categories and "peak_detection" in morphology_options:
            return create_morphology_plots(time_data, signal_data, sampling_freq)
        else:
            return create_empty_figure()
    except Exception as e:
        logger.error(f"Error creating analysis plots: {e}")
        return create_empty_figure()


def analyze_hrv(signal_data, sampling_freq, hrv_options):
    """Analyze Heart Rate Variability."""
    try:
        # Find peaks
        peaks, _ = signal.find_peaks(signal_data, height=np.mean(signal_data) + np.std(signal_data), 
                                   distance=int(sampling_freq * 0.3))
        
        if len(peaks) < 2:
            return {"error": "Insufficient peaks for HRV analysis"}
        
        # Calculate RR intervals
        rr_intervals = np.diff(peaks) / sampling_freq * 1000  # Convert to milliseconds
        
        hrv_metrics = {}
        
        if "time_domain" in hrv_options:
            hrv_metrics.update({
                "mean_rr": np.mean(rr_intervals),
                "std_rr": np.std(rr_intervals),
                "rmssd": np.sqrt(np.mean(np.diff(rr_intervals) ** 2)),
                "nn50": np.sum(np.abs(np.diff(rr_intervals)) > 50),
                "pnn50": (np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals)) * 100
            })
        
        if "frequency_domain" in hrv_options:
            # Simple frequency domain analysis
            freqs, psd = signal.welch(rr_intervals, fs=1000/np.mean(rr_intervals))
            hrv_metrics.update({
                "total_power": np.sum(psd),
                "vlf_power": np.sum(psd[freqs < 0.04]),
                "lf_power": np.sum(psd[(freqs >= 0.04) & (freqs < 0.15)]),
                "hf_power": np.sum(psd[(freqs >= 0.15) & (freqs < 0.4)]),
                "lf_hf_ratio": np.sum(psd[(freqs >= 0.04) & (freqs < 0.15)]) / np.sum(psd[(freqs >= 0.15) & (freqs < 0.4)])
            })
        
        return hrv_metrics
        
    except Exception as e:
        logger.error(f"Error in HRV analysis: {e}")
        return {"error": f"HRV analysis failed: {str(e)}"}


def analyze_morphology(signal_data, sampling_freq, morphology_options):
    """Analyze signal morphology."""
    try:
        morphology_metrics = {}
        
        if "peak_detection" in morphology_options:
            peaks, properties = signal.find_peaks(signal_data, height=np.mean(signal_data) + np.std(signal_data), 
                                               distance=int(sampling_freq * 0.3))
            morphology_metrics.update({
                "num_peaks": len(peaks),
                "peak_heights": signal_data[peaks].tolist() if len(peaks) > 0 else [],
                "peak_positions": peaks.tolist() if len(peaks) > 0 else []
            })
        
        if "amplitude" in morphology_options:
            morphology_metrics.update({
                "mean_amplitude": np.mean(signal_data),
                "std_amplitude": np.std(signal_data),
                "min_amplitude": np.min(signal_data),
                "max_amplitude": np.max(signal_data),
                "peak_to_peak": np.max(signal_data) - np.min(signal_data)
            })
        
        if "duration" in morphology_options:
            morphology_metrics.update({
                "signal_duration": len(signal_data) / sampling_freq,
                "sampling_freq": sampling_freq,
                "num_samples": len(signal_data)
            })
        
        return morphology_metrics
        
    except Exception as e:
        logger.error(f"Error in morphology analysis: {e}")
        return {"error": f"Morphology analysis failed: {str(e)}"}


def analyze_signal_quality(signal_data, sampling_freq):
    """Analyze signal quality metrics."""
    try:
        # Calculate SNR estimate
        signal_power = np.mean(signal_data ** 2)
        noise_power = np.var(signal_data)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # Calculate other quality metrics
        quality_metrics = {
            "snr_db": snr,
            "dynamic_range": np.max(signal_data) - np.min(signal_data),
            "mean_value": np.mean(signal_data),
            "std_value": np.std(signal_data),
            "zero_crossings": np.sum(np.diff(np.sign(signal_data)) != 0)
        }
        
        return quality_metrics
        
    except Exception as e:
        logger.error(f"Error in quality analysis: {e}")
        return {"error": f"Quality analysis failed: {str(e)}"}


def analyze_trends(signal_data, sampling_freq):
    """Analyze signal trends."""
    try:
        # Simple trend analysis using linear regression
        time_axis = np.arange(len(signal_data)) / sampling_freq
        coeffs = np.polyfit(time_axis, signal_data, 1)
        trend_slope = coeffs[0]
        
        # Calculate trend strength
        trend_line = np.polyval(coeffs, time_axis)
        trend_strength = 1 - (np.sum((signal_data - trend_line) ** 2) / np.sum((signal_data - np.mean(signal_data)) ** 2))
        
        trend_metrics = {
            "trend_slope": trend_slope,
            "trend_strength": trend_strength,
            "trend_direction": "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "stable"
        }
        
        return trend_metrics
        
    except Exception as e:
        logger.error(f"Error in trend analysis: {e}")
        return {"error": f"Trend analysis failed: {str(e)}"}


def create_comprehensive_results_display(results, signal_type, sampling_freq):
    """Create comprehensive results display."""
    try:
        sections = []
        
        # HRV Results
        if "hrv_metrics" in results and "error" not in results["hrv_metrics"]:
            hrv_section = html.Div([
                html.H5("💓 Heart Rate Variability (HRV) Analysis"),
                html.Div([
                    html.Div([
                        html.H6("Time Domain Metrics"),
                        html.P(f"Mean RR: {results['hrv_metrics'].get('mean_rr', 0):.1f} ms"),
                        html.P(f"RMSSD: {results['hrv_metrics'].get('rmssd', 0):.1f} ms"),
                        html.P(f"pNN50: {results['hrv_metrics'].get('pnn50', 0):.1f}%")
                    ], className="col-md-6"),
                    html.Div([
                        html.H6("Frequency Domain Metrics"),
                        html.P(f"LF Power: {results['hrv_metrics'].get('lf_power', 0):.2f}"),
                        html.P(f"HF Power: {results['hrv_metrics'].get('hf_power', 0):.2f}"),
                        html.P(f"LF/HF Ratio: {results['hrv_metrics'].get('lf_hf_ratio', 0):.2f}")
                    ], className="col-md-6")
                ], className="row")
            ])
            sections.append(hrv_section)
        
        # Morphology Results
        if "morphology_metrics" in results and "error" not in results["morphology_metrics"]:
            morph_section = html.Div([
                html.H5("📊 Morphology Analysis"),
                html.Div([
                    html.Div([
                        html.H6("Peak Analysis"),
                        html.P(f"Number of Peaks: {results['morphology_metrics'].get('num_peaks', 0)}"),
                        html.P(f"Mean Peak Height: {np.mean(results['morphology_metrics'].get('peak_heights', [0])):.2f}")
                    ], className="col-md-6"),
                    html.Div([
                        html.H6("Amplitude Analysis"),
                        html.P(f"Peak-to-Peak: {results['morphology_metrics'].get('peak_to_peak', 0):.4f}"),
                        html.P(f"Standard Deviation: {results['morphology_metrics'].get('std_amplitude', 0):.4f}")
                    ], className="col-md-6")
                ], className="row")
            ])
            sections.append(morph_section)
        
        # Quality Results
        if "quality_metrics" in results and "error" not in results["quality_metrics"]:
            quality_section = html.Div([
                html.H5("⚖️ Signal Quality"),
                html.P(f"SNR: {results['quality_metrics'].get('snr_db', 0):.1f} dB"),
                html.P(f"Dynamic Range: {results['quality_metrics'].get('dynamic_range', 0):.4f}"),
                html.P(f"Zero Crossings: {results['quality_metrics'].get('zero_crossings', 0)}")
            ])
            sections.append(quality_section)
        
        # Trend Results
        if "trend_metrics" in results and "error" not in results["trend_metrics"]:
            trend_section = html.Div([
                html.H5("📈 Trend Analysis"),
                html.P(f"Trend Direction: {results['trend_metrics'].get('trend_direction', 'Unknown')}"),
                html.P(f"Trend Strength: {results['trend_metrics'].get('trend_strength', 0):.3f}")
            ])
            sections.append(trend_section)
        
        if not sections:
            return html.Div([
                html.H5("No Analysis Results"),
                html.P("Please select analysis categories and run the analysis.")
            ])
        
        return html.Div(sections)
        
    except Exception as e:
        logger.error(f"Error creating results display: {e}")
        return html.Div([
            html.H5("Error"),
            html.P(f"Failed to create results display: {str(e)}")
        ])


def create_hrv_plots(time_data, signal_data, sampling_freq):
    """Create HRV-specific plots."""
    try:
        # Find peaks for RR intervals
        peaks, _ = signal.find_peaks(signal_data, height=np.mean(signal_data) + np.std(signal_data), 
                                   distance=int(sampling_freq * 0.3))
        
        if len(peaks) < 2:
            return create_empty_figure()
        
        # Calculate RR intervals
        rr_intervals = np.diff(peaks) / sampling_freq * 1000  # Convert to milliseconds
        
        # Create subplot with RR intervals and histogram
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("RR Intervals", "RR Interval Distribution"),
            vertical_spacing=0.1
        )
        
        # RR intervals over time
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(rr_intervals)),
                y=rr_intervals,
                mode='lines+markers',
                name='RR Intervals',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # RR interval histogram
        fig.add_trace(
            go.Histogram(
                x=rr_intervals,
                nbinsx=20,
                name='RR Distribution',
                marker_color='green',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Heart Rate Variability Analysis",
            height=600,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating HRV plots: {e}")
        return create_empty_figure()


def create_morphology_plots(time_data, signal_data, sampling_freq):
    """Create morphology-specific plots."""
    try:
        # Find peaks
        peaks, properties = signal.find_peaks(signal_data, height=np.mean(signal_data) + np.std(signal_data), 
                                           distance=int(sampling_freq * 0.3))
        
        # Create subplot with signal and peak analysis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Signal with Peaks", "Peak Height Distribution"),
            vertical_spacing=0.1
        )
        
        # Main signal with peaks
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=signal_data,
                mode='lines',
                name='Signal',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        if len(peaks) > 0:
            fig.add_trace(
                go.Scatter(
                    x=time_data[peaks],
                    y=signal_data[peaks],
                    mode='markers',
                    name='Peaks',
                    marker=dict(color='red', size=8, symbol='circle')
                ),
                row=1, col=1
            )
            
            # Peak height distribution
            fig.add_trace(
                go.Histogram(
                    x=signal_data[peaks],
                    nbinsx=15,
                    name='Peak Heights',
                    marker_color='orange',
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title="Morphology Analysis",
            height=600,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating morphology plots: {e}")
        return create_empty_figure()
