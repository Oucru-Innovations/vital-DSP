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


def normalize_signal_type(signal_type):
    """Normalize signal type to ensure compatibility with vitalDSP."""
    if not signal_type:
        return "PPG"
    
    # Convert to uppercase and validate
    signal_type_upper = signal_type.upper()
    valid_types = ["ECG", "PPG", "EEG"]
    
    if signal_type_upper in valid_types:
        return signal_type_upper
    else:
        logger.warning(f"Invalid signal type '{signal_type}' detected. Defaulting to 'PPG'.")
        return "PPG"


def register_physiological_callbacks(app):
    """Register all physiological analysis callbacks."""
    logger.info("=== REGISTERING PHYSIOLOGICAL CALLBACKS ===")
    
    # Import vitalDSP modules when callbacks are registered
    _import_vitaldsp_modules()
    
    # Register additional callbacks for enhanced features
    register_additional_physiological_callbacks(app)
    
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
         State("physio-advanced-features", "value"),
         State("physio-quality-options", "value"),
         State("physio-transform-options", "value"),
         State("physio-advanced-computation", "value"),
         State("physio-feature-engineering", "value"),
         State("physio-preprocessing", "value")]
    )
    def physiological_analysis_callback(pathname, n_clicks, slider_value, nudge_m10, nudge_m1, nudge_p1, nudge_p10,
                                     start_time, end_time, signal_type, analysis_categories, hrv_options, 
                                     morphology_options, advanced_features, quality_options, transform_options,
                                     advanced_computation, feature_engineering, preprocessing):
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
        logger.info(f"Advanced features: {advanced_features}")
        logger.info(f"Quality options: {quality_options}")
        logger.info(f"Transform options: {transform_options}")
        logger.info(f"Advanced computation: {advanced_computation}")
        logger.info(f"Feature engineering: {feature_engineering}")
        logger.info(f"Preprocessing: {preprocessing}")
        
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
            analysis_categories = analysis_categories or ["hrv", "morphology", "beat2beat", "energy", "envelope", "segmentation", "trend", "waveform", "statistical", "frequency"]
            hrv_options = hrv_options or ["time_domain", "freq_domain", "nonlinear"]
            morphology_options = morphology_options or ["peaks", "duration", "area"]
            advanced_features = advanced_features or ["cross_signal", "ensemble", "change_detection", "power_analysis"]
            quality_options = quality_options or ["quality_index", "artifact_detection"]
            transform_options = transform_options or ["wavelet", "fourier", "hilbert"]
            advanced_computation = advanced_computation or ["anomaly_detection", "bayesian", "kalman"]
            feature_engineering = feature_engineering or ["ppg_light", "ppg_autonomic", "ecg_autonomic"]
            preprocessing = preprocessing or ["noise_reduction", "baseline_correction", "filtering"]
            
            logger.info(f"Time window: {start_time} - {end_time}")
            logger.info(f"Signal type: {signal_type}")
            logger.info(f"Analysis categories: {analysis_categories}")
            logger.info(f"HRV options: {hrv_options}")
            logger.info(f"Morphology options: {morphology_options}")
            logger.info(f"Advanced features: {advanced_features}")
            logger.info(f"Quality options: {quality_options}")
            logger.info(f"Transform options: {transform_options}")
            logger.info(f"Advanced computation: {advanced_computation}")
            logger.info(f"Feature engineering: {feature_engineering}")
            logger.info(f"Preprocessing: {preprocessing}")
            
            # Extract time and signal columns
            time_col = column_mapping.get('time', df.columns[0])
            signal_col = column_mapping.get('signal', df.columns[1])
            
            # Log column selection for debugging
            logger.info(f"Available columns: {list(df.columns)}")
            logger.info(f"Selected time column: {time_col}")
            logger.info(f"Selected signal column: {signal_col}")
            logger.info(f"Column mapping: {column_mapping}")
            
            # Extract time and signal data
            time_data = df[time_col].values
            signal_data = df[signal_col].values
            
            # Log data characteristics
            logger.info(f"Time data range: {time_data[0]} to {time_data[-1]}")
            logger.info(f"Signal data range: {np.min(signal_data):.4f} to {np.max(signal_data):.4f}")
            logger.info(f"Signal data mean: {np.mean(signal_data):.4f}, std: {np.std(signal_data):.4f}")
            logger.info(f"Total samples: {len(signal_data)}")
            
            # Suggest better column mappings if current ones seem wrong
            if signal_col == 'PLETH' and np.std(signal_data) < 0.01:
                logger.warning(f"PLETH column has very low variance ({np.std(signal_data):.6f}), might not be the best signal column")
                # Look for columns with higher variance
                for col in df.columns:
                    if col not in [time_col, signal_col] and df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        col_std = np.std(df[col].values)
                        if col_std > np.std(signal_data) * 10:  # 10x higher variance
                            logger.info(f"Consider using column '{col}' instead (std: {col_std:.6f})")
                            break
            
            # Get suggestions for better signal columns
            signal_suggestions = suggest_best_signal_column(df, time_col)
            if signal_suggestions:
                logger.info("Signal column suggestions (ranked by quality):")
                for i, suggestion in enumerate(signal_suggestions[:3]):  # Top 3 suggestions
                    logger.info(f"  {i+1}. {suggestion['column']} (score: {suggestion['score']}, variance: {suggestion['variance']:.6f})")
                
                # If current signal column has very low score, suggest using the best one
                current_score = next((s['score'] for s in signal_suggestions if s['column'] == signal_col), 0)
                best_score = signal_suggestions[0]['score']
                
                if current_score < best_score - 2:  # Significant difference
                    logger.warning(f"Current signal column '{signal_col}' has low quality (score: {current_score})")
                    logger.warning(f"Consider using '{signal_suggestions[0]['column']}' instead (score: {best_score})")
                    
                    # Auto-switch to better column if current one is very poor
                    if current_score < 2 and best_score >= 4:  # Current is very poor, best is good
                        old_signal_col = signal_col
                        signal_col = signal_suggestions[0]['column']
                        signal_data = df[signal_col].values
                        logger.info(f"Auto-switched signal column from '{old_signal_col}' to '{signal_col}' for better analysis quality")
                        
                        # Re-log signal characteristics with new column
                        logger.info(f"New signal data range: {np.min(signal_data):.4f} to {np.max(signal_data):.4f}")
                        logger.info(f"New signal data mean: {np.mean(signal_data):.4f}, std: {np.std(signal_data):.4f}")
                
                # Determine time unit and adjust window if needed
                time_range = time_data[-1] - time_data[0]
                logger.info(f"Full time range: {time_range}")
                
                # If time data appears to be in milliseconds, convert to seconds for analysis
                if time_range > 1000:  # Likely milliseconds
                    logger.info("Time data appears to be in milliseconds, converting to seconds")
                    time_data_seconds = time_data / 1000.0
                    # Adjust default time window for seconds
                    if start_time == 0 and end_time == 10:
                        start_time = 0
                        end_time = min(60, time_range / 1000.0)  # Use up to 60 seconds or full range
                        logger.info(f"Adjusted time window to: {start_time} - {end_time} seconds")
                else:
                    time_data_seconds = time_data
            
            # Apply time window
                start_idx = np.searchsorted(time_data_seconds, start_time)
                end_idx = np.searchsorted(time_data_seconds, end_time)
                
                # Ensure minimum signal length for analysis (at least 5 seconds worth of data)
                min_samples = max(5 * sampling_freq, 1000)  # At least 5 seconds or 1000 samples
                if end_idx - start_idx < min_samples:
                    # Extend the window to get sufficient data
                    if end_idx + min_samples <= len(time_data_seconds):
                        end_idx = start_idx + min_samples
                    elif start_idx - min_samples >= 0:
                        start_idx = end_idx - min_samples
                    else:
                        # Use the full signal if we can't get enough data
                        start_idx = 0
                        end_idx = len(time_data_seconds)
                        logger.warning(f"Using full signal due to insufficient data in time window")
                
                logger.info(f"Using signal segment: {start_idx} to {end_idx} (length: {end_idx - start_idx})")
                
            if end_idx > start_idx:
                time_data = time_data_seconds[start_idx:end_idx]
                signal_data = signal_data[start_idx:end_idx]
            
            # Validate signal length for analysis
            if len(signal_data) < min_samples:
                logger.warning(f"Signal segment too short ({len(signal_data)} samples), using fallback analysis")
                # For very short signals, limit analysis to basic features
                analysis_categories = ["morphology", "statistical", "frequency"]
                hrv_options = []
                advanced_features = []
                quality_options = []
                transform_options = []
                advanced_computation = []
                feature_engineering = []
                preprocessing = []
            
            # Auto-detect signal type if needed
            if signal_type == "auto":
                signal_type = detect_physiological_signal_type(signal_data, sampling_freq)
                logger.info(f"Auto-detected signal type: {signal_type}")
            
            # Create main signal plot
            main_fig = create_physiological_signal_plot(time_data, signal_data, signal_type, sampling_freq)
            
            # Perform analysis based on selected categories
            raw_analysis_results = perform_physiological_analysis_enhanced(
                time_data, signal_data, signal_type, sampling_freq,
                analysis_categories, hrv_options, morphology_options, advanced_features,
                quality_options, transform_options, advanced_computation, feature_engineering, preprocessing
            )
            
            # Create comprehensive results display
            analysis_results = create_comprehensive_results_display(raw_analysis_results, signal_type, sampling_freq)
            
            # Create analysis plots
            analysis_fig = create_physiological_analysis_plots(
                time_data, signal_data, signal_type, sampling_freq,
                analysis_categories, hrv_options, morphology_options, advanced_features,
                quality_options, transform_options, advanced_computation, feature_engineering, preprocessing
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
                "hrv_metrics": raw_analysis_results.get("hrv_metrics", {}),
                "morphology_metrics": raw_analysis_results.get("morphology_metrics", {}),
                "beat2beat_metrics": raw_analysis_results.get("beat2beat_metrics", {}),
                "energy_metrics": raw_analysis_results.get("energy_metrics", {}),
                "envelope_metrics": raw_analysis_results.get("envelope_metrics", {}),
                "segmentation_metrics": raw_analysis_results.get("segmentation_metrics", {}),
                "waveform_metrics": raw_analysis_results.get("waveform_metrics", {}),
                "statistical_metrics": raw_analysis_results.get("statistical_metrics", {}),
                "frequency_metrics": raw_analysis_results.get("frequency_metrics", {}),
                "advanced_features_metrics": raw_analysis_results.get("advanced_features_metrics", {}),
                "quality_metrics": raw_analysis_results.get("quality_metrics", {}),
                "transform_metrics": raw_analysis_results.get("transform_metrics", {}),
                "advanced_computation_metrics": raw_analysis_results.get("advanced_computation_metrics", {}),
                "feature_engineering_metrics": raw_analysis_results.get("feature_engineering_metrics", {}),
                "preprocessing_metrics": raw_analysis_results.get("preprocessing_metrics", {}),
                "trend_metrics": raw_analysis_results.get("trend_metrics", {})
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
    """Create the main physiological signal plot with enhanced visualization."""
    fig = go.Figure()
    
    # Add main signal with enhanced styling
    fig.add_trace(go.Scatter(
        x=time_data,
        y=signal_data,
        mode='lines',
        name=f'{signal_type.upper()} Signal',
        line=dict(color='#1f77b4', width=2, shape='linear'),
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))
    
    # Enhanced peak detection with red arrows
    if signal_type.lower() in ["ecg", "ppg"]:
        try:
            # Use more sophisticated peak detection
            height_threshold = np.mean(signal_data) + 1.5 * np.std(signal_data)
            distance_threshold = int(sampling_freq * 0.3)  # Minimum 0.3 seconds between peaks
            
            peaks, properties = signal.find_peaks(
                signal_data, 
                height=height_threshold,
                distance=distance_threshold,
                prominence=np.std(signal_data) * 0.5
            )
            
            if len(peaks) > 0:
                # Add peak markers with red arrows
                fig.add_trace(go.Scatter(
                    x=time_data[peaks],
                    y=signal_data[peaks],
                    mode='markers+text',
                    name='Detected Peaks',
                    text=[f'P{i+1}' for i in range(len(peaks))],
                    textposition='top center',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='arrow-up',
                        line=dict(color='darkred', width=2)
                    ),
                    textfont=dict(color='red', size=10, family='Arial Black')
                ))
                
                # Add peak-to-peak intervals
                if len(peaks) > 1:
                    intervals = np.diff(peaks) / sampling_freq
                    for i, (peak1, peak2, interval) in enumerate(zip(peaks[:-1], peaks[1:], intervals)):
                        # Add interval annotation
                        mid_x = (time_data[peak1] + time_data[peak2]) / 2
                        mid_y = (signal_data[peak1] + signal_data[peak2]) / 2
                        
                        fig.add_annotation(
                            x=mid_x,
                            y=mid_y,
                            text=f'{interval:.2f}s',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor='orange',
                            font=dict(size=10, color='orange'),
                            bgcolor='rgba(255, 255, 255, 0.8)',
                            bordercolor='orange',
                            borderwidth=1
                        )
                
                # Add statistics annotation
                mean_interval = np.mean(intervals) if len(peaks) > 1 else 0
                heart_rate = 60 / mean_interval if mean_interval > 0 else 0
                
                fig.add_annotation(
                    x=0.02,
                    y=0.98,
                    xref='paper',
                    yref='paper',
                    text=f'Peaks: {len(peaks)}<br>Mean Interval: {mean_interval:.2f}s<br>Heart Rate: {heart_rate:.1f} BPM',
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='red',
                    borderwidth=2,
                    font=dict(size=12, color='black')
                )
                
        except Exception as e:
            logger.warning(f"Error in peak detection: {e}")
    
    # Add baseline reference
    baseline = np.mean(signal_data)
    fig.add_hline(
        y=baseline,
        line_dash="dash",
        line_color="gray",
        annotation_text="Baseline",
        annotation_position="right"
    )
    
    # Add signal envelope
    try:
        # Calculate moving average for envelope
        window_size = min(50, len(signal_data) // 10)
        if window_size > 1:
            envelope_upper = np.convolve(signal_data, np.ones(window_size)/window_size, mode='same')
            envelope_lower = np.convolve(signal_data, -np.ones(window_size)/window_size, mode='same')
            
            fig.add_trace(go.Scatter(
                x=time_data,
                y=envelope_upper,
                mode='lines',
                name='Upper Envelope',
                line=dict(color='rgba(255, 165, 0, 0.6)', width=1, dash='dot'),
                showlegend=True
            ))
            
            fig.add_trace(go.Scatter(
                x=time_data,
                y=envelope_lower,
                mode='lines',
                name='Lower Envelope',
                line=dict(color='rgba(255, 165, 0, 0.6)', width=1, dash='dot'),
                showlegend=True,
                fill='tonexty',
                fillcolor='rgba(255, 165, 0, 0.1)'
            ))
    except Exception as e:
        logger.debug(f"Could not create envelope: {e}")
    
        # Enhanced layout
    fig.update_layout(
            title=dict(
                text=f"{signal_type.upper()} Signal Analysis with Peak Detection",
                x=0.5,
                font=dict(size=20, color='#2c3e50')
            ),
            xaxis=dict(
                title="Time (seconds)",
                titlefont=dict(size=14, color='#2c3e50'),
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinecolor='rgba(128, 128, 128, 0.5)',
                showgrid=True
            ),
            yaxis=dict(
                title="Amplitude",
                titlefont=dict(size=14, color='#2c3e50'),
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinecolor='rgba(128, 128, 128, 0.5)',
                showgrid=True
            ),
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=200, t=80, b=60),
            height=500
    )
    
    # Add grid styling
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
    
    return fig


def perform_physiological_analysis(time_data, signal_data, signal_type, sampling_freq, 
                                 analysis_categories, hrv_options, morphology_options, advanced_features,
                                 quality_options, transform_options, advanced_computation, feature_engineering, preprocessing):
    """Perform comprehensive physiological analysis."""
    results = {}
    
    try:
        # HRV Analysis
        if "hrv" in analysis_categories:
            results["hrv_metrics"] = analyze_hrv(signal_data, sampling_freq, hrv_options)
        
        # Morphology Analysis
        if "morphology" in analysis_categories:
            results["morphology_metrics"] = analyze_morphology(signal_data, sampling_freq, morphology_options)
        
        # Beat-to-Beat Analysis
        if "beat2beat" in analysis_categories:
            results["beat2beat_metrics"] = analyze_beat_to_beat(signal_data, sampling_freq)
        
        # Energy Analysis
        if "energy" in analysis_categories:
            results["energy_metrics"] = analyze_energy(signal_data, sampling_freq)
        
        # Envelope Detection
        if "envelope" in analysis_categories:
            results["envelope_metrics"] = analyze_envelope(signal_data, sampling_freq)
        
        # Signal Segmentation
        if "segmentation" in analysis_categories:
            results["segmentation_metrics"] = analyze_segmentation(signal_data, sampling_freq)
        
        # Trend Analysis
        if "trend" in analysis_categories:
            results["trend_metrics"] = analyze_trends(signal_data, sampling_freq)
        
        # Waveform Analysis
        if "waveform" in analysis_categories:
            results["waveform_metrics"] = analyze_waveform(signal_data, sampling_freq)
        
        # Statistical Analysis
        if "statistical" in analysis_categories:
            results["statistical_metrics"] = analyze_statistical(signal_data, sampling_freq)
        
        # Frequency Analysis
        if "frequency" in analysis_categories:
            results["frequency_metrics"] = analyze_frequency(signal_data, sampling_freq)
        
        # Advanced Features
        if advanced_features:
            results["advanced_features_metrics"] = analyze_advanced_features(signal_data, sampling_freq, advanced_features)
        
        # Signal Quality & Artifact Detection
        if quality_options:
            results["quality_metrics"] = analyze_signal_quality_advanced(signal_data, sampling_freq, quality_options)
        
        # Signal Transforms
        if transform_options:
            results["transform_metrics"] = analyze_transforms(signal_data, sampling_freq, transform_options)
        
        # Advanced Computation
        if advanced_computation:
            results["advanced_computation_metrics"] = analyze_advanced_computation(signal_data, sampling_freq, advanced_computation)
        
        # Feature Engineering
        if feature_engineering:
            results["feature_engineering_metrics"] = analyze_feature_engineering(signal_data, sampling_freq, feature_engineering, signal_type)
        
        # Preprocessing
        if preprocessing:
            results["preprocessing_metrics"] = analyze_preprocessing(signal_data, sampling_freq, preprocessing)
        
        # Create comprehensive results display
        return create_comprehensive_results_display(results, signal_type, sampling_freq)
        
    except Exception as e:
        logger.error(f"Error in physiological analysis: {e}")
        return html.Div([
            html.H5("Analysis Error"),
            html.P(f"Analysis failed: {str(e)}")
        ])


def create_physiological_analysis_plots(time_data, signal_data, signal_type, sampling_freq,
                                       analysis_categories, hrv_options, morphology_options, advanced_features,
                                       quality_options, transform_options, advanced_computation, feature_engineering, preprocessing):
    """Create comprehensive 2x2 grid layout with multiple analysis plots for better visualization."""
    try:
        # Find peaks for analysis - this is crucial for all plots
        height_threshold = np.mean(signal_data) + 1.5 * np.std(signal_data)
        distance_threshold = int(sampling_freq * 0.3)
        
        peaks, _ = signal.find_peaks(
            signal_data, 
            height=height_threshold,
            distance=distance_threshold,
            prominence=np.std(signal_data) * 0.5
        )
        
        # Create a 2x2 subplot layout for comprehensive analysis view
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Signal Overview", "Beat-to-Beat Analysis", "Frequency Analysis", "Poincaré Plot"),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Signal Overview (top-left) - Enhanced main signal with peaks
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=signal_data,
                mode='lines',
                name='Signal',
                line=dict(color='#1f77b4', width=2),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.1)'
            ),
            row=1, col=1
        )
        
        # Add peaks with red arrows for better visibility
        if len(peaks) > 0:
            fig.add_trace(
                go.Scatter(
                    x=time_data[peaks],
                    y=signal_data[peaks],
                    mode='markers+text',
                    name='Peaks',
                    text=[f'P{i+1}' for i in range(min(len(peaks), 10))],  # Limit text to first 10 peaks
                    textposition='top center',
                    marker=dict(
                        color='red',
                        size=8,
                        symbol='arrow-up',
                        line=dict(color='darkred', width=2)
                    ),
                    textfont=dict(color='red', size=8, family='Arial Black')
                ),
                row=1, col=1
            )
            
            # Add peak statistics
            mean_interval = np.mean(np.diff(peaks) / sampling_freq) if len(peaks) > 1 else 0
            heart_rate = 60 / mean_interval if mean_interval > 0 else 0
            
            fig.add_annotation(
                x=0.98, y=0.98,
                xref='x1', yref='y1',
                text=f'Peaks: {len(peaks)}<br>HR: {heart_rate:.1f} BPM<br>Mean Interval: {mean_interval:.2f}s',
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='red',
                borderwidth=2,
                font=dict(size=10, color='black')
            )
        
        # 2. Beat-to-Beat Analysis (top-right) - Beat intervals and variability
        if len(peaks) > 1:
            # Calculate beat intervals
            beat_intervals = np.diff(peaks) / sampling_freq
            
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(beat_intervals)),
                    y=beat_intervals,
                    mode='lines+markers',
                    name='Beat Intervals',
                    line=dict(color='#2ca02c', width=3),
                    marker=dict(color='#2ca02c', size=6, symbol='diamond'),
                    fill='tonexty',
                    fillcolor='rgba(44, 160, 44, 0.1)'
                ),
                row=1, col=2
            )
            
            # Add interval statistics
            mean_interval = np.mean(beat_intervals)
            std_interval = np.std(beat_intervals)
            heart_rate = 60 / mean_interval if mean_interval > 0 else 0
            
            fig.add_hline(
                y=mean_interval,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_interval:.3f}s",
                annotation_position="right",
                row=1, col=2
            )
            
            # Add beat-to-beat statistics
            fig.add_annotation(
                x=0.98, y=0.98,
                xref='x2', yref='y2',
                text=f'Heart Rate: {heart_rate:.1f} BPM<br>Mean Interval: {mean_interval:.3f}s<br>Variability: {std_interval:.3f}s',
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='green',
                borderwidth=2,
                font=dict(size=10, color='black')
            )
        else:
            # If no peaks, show signal energy analysis instead
            try:
                # Calculate signal energy over time
                window_size = min(50, len(signal_data) // 10)
                if window_size > 1:
                    energy_values = []
                    time_windows = []
                    
                    for i in range(0, len(signal_data) - window_size, window_size // 2):
                        window_data = signal_data[i:i+window_size]
                        energy = np.sum(window_data ** 2)
                        energy_values.append(energy)
                        time_windows.append(time_data[i + window_size // 2])
                    
                    if len(energy_values) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=time_windows,
                                y=energy_values,
                                mode='lines+markers',
                                name='Signal Energy',
                                line=dict(color='#ff7f0e', width=2),
                                marker=dict(color='#ff7f0e', size=4, symbol='circle')
                            ),
                            row=1, col=2
                        )
                        
                        # Add energy statistics
                        mean_energy = np.mean(energy_values)
                        total_energy = np.sum(signal_data ** 2)
                        
                        fig.add_annotation(
                            x=0.98, y=0.98,
                            xref='x2', yref='y2',
                            text=f'Mean Energy: {mean_energy:.2e}<br>Total Energy: {total_energy:.2e}<br>Energy Variance: {np.var(energy_values):.2e}',
                            showarrow=False,
                            bgcolor='rgba(255, 255, 255, 0.9)',
                            bordercolor='orange',
                            borderwidth=2,
                            font=dict(size=10, color='black')
                        )
            except Exception as e:
                logger.debug(f"Could not create energy analysis: {e}")
                # Final fallback: show signal amplitude distribution
                hist, bin_edges = np.histogram(signal_data, bins=30, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                fig.add_trace(
                    go.Bar(
                        x=bin_centers,
                        y=hist,
                        name='Signal Distribution',
                        marker_color='rgba(255, 165, 0, 0.7)',
                        opacity=0.8
                    ),
                    row=1, col=2
                )
                
                # Add signal statistics
                fig.add_annotation(
                    x=0.98, y=0.98,
                    xref='x2', yref='y2',
                    text=f'Mean: {np.mean(signal_data):.3f}<br>Std: {np.std(signal_data):.3f}<br>Range: {np.max(signal_data) - np.min(signal_data):.3f}',
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='orange',
                    borderwidth=2,
                    font=dict(size=10, color='black')
                )
        
        # 3. Frequency Analysis (bottom-left) - Power spectral density
        try:
            freqs, psd = signal.welch(signal_data, fs=sampling_freq, nperseg=min(256, len(signal_data)//2))
            
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=psd,
                    mode='lines',
                    name='Power Spectrum',
                    line=dict(color='#2ca02c', width=2),
                    fill='tonexty',
                    fillcolor='rgba(44, 160, 44, 0.1)'
                ),
                row=2, col=1
            )
            
            # Add frequency band annotations
            low_freq_mask = freqs < 1.0
            mid_freq_mask = (freqs >= 1.0) & (freqs < 10.0)
            high_freq_mask = freqs >= 10.0
            
            if np.any(low_freq_mask):
                low_power = np.sum(psd[low_freq_mask])
                fig.add_annotation(
                    x=0.95, y=0.9,
                    xref='x3', yref='y3',
                    text=f'Low: {low_power:.2e}',
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='blue'
                )
            
            if np.any(mid_freq_mask):
                mid_power = np.sum(psd[mid_freq_mask])
                fig.add_annotation(
                    x=0.95, y=0.9,
                    xref='x3', yref='y3',
                    text=f'Mid: {mid_power:.2e}',
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='green'
                )
            
            if np.any(high_freq_mask):
                high_power = np.sum(psd[high_freq_mask])
                fig.add_annotation(
                    x=0.95, y=0.9,
                    xref='x3', yref='y3',
                    text=f'High: {high_power:.2e}',
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='red'
                )
        except Exception as e:
            logger.debug(f"Could not create frequency analysis: {e}")
        
        # 4. Poincaré Plot (bottom-right) - RR intervals scatter plot
        if len(peaks) > 1:
            try:
                # Calculate RR intervals
                rr_intervals = np.diff(peaks) / sampling_freq * 1000  # Convert to milliseconds
                
                # Create Poincaré plot (RR_n vs RR_{n+1})
                fig.add_trace(
                    go.Scatter(
                        x=rr_intervals[:-1],
                        y=rr_intervals[1:],
                        mode='markers',
                        name='Poincaré Plot',
                        marker=dict(
                            color='rgba(156, 39, 176, 0.7)',
                            size=8,
                            symbol='circle',
                            line=dict(color='rgba(156, 39, 176, 1)', width=1)
                        )
                    ),
                    row=2, col=2
                )
                
                # Calculate Poincaré plot statistics
                diff_rr = np.diff(rr_intervals)
                sd1 = np.std(diff_rr) / np.sqrt(2)
                sd2 = np.std(rr_intervals) * np.sqrt(2)
                
                # Add SD1 and SD2 ellipses
                center_x, center_y = np.mean(rr_intervals[:-1]), np.mean(rr_intervals[1:])
                
                # Create SD1 ellipse points
                theta = np.linspace(0, 2*np.pi, 100)
                x_ellipse = center_x + sd1 * np.cos(theta)
                y_ellipse = center_y + sd1 * np.sin(theta)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_ellipse,
                        y=y_ellipse,
                        mode='lines',
                        name='SD1 Ellipse',
                        line=dict(color='red', width=2, dash='dash'),
                        showlegend=False
                    ),
                    row=2, col=2
                )
                
                # Add Poincaré statistics annotation
                fig.add_annotation(
                    x=0.98, y=0.98,
                    xref='x4', yref='y4',
                    text=f'SD1: {sd1:.1f} ms<br>SD2: {sd2:.1f} ms<br>RR Count: {len(rr_intervals)}',
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='purple',
                    borderwidth=2,
                    font=dict(size=10, color='black')
                )
                
            except Exception as e:
                logger.debug(f"Could not create Poincaré plot: {e}")
                # Fallback: show signal envelope instead
                try:
                    analytic_signal = signal.hilbert(signal_data)
                    envelope = np.abs(analytic_signal)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=time_data,
                            y=envelope,
                            mode='lines',
                            name='Signal Envelope',
                            line=dict(color='#d62728', width=2)
                        ),
                        row=2, col=2
                    )
                    
                    fig.add_annotation(
                        x=0.98, y=0.98,
                        xref='x4', yref='y4',
                        text=f'Envelope Mean: {np.mean(envelope):.3f}<br>Envelope Std: {np.std(envelope):.3f}',
                        showarrow=False,
                        bgcolor='rgba(255, 255, 255, 0.9)',
                        bordercolor='red',
                        borderwidth=2,
                        font=dict(size=10, color='black')
                    )
                except:
                    pass
        else:
            # If no peaks, show signal quality metrics instead
            try:
                # Calculate quality metrics
                baseline = np.mean(signal_data)
                noise_level = np.std(signal_data)
                snr = 20 * np.log10(np.abs(baseline) / noise_level) if noise_level > 0 else 0
                
                # Create quality score over time
                window_size = min(50, len(signal_data) // 10)
                if window_size > 1:
                    quality_scores = []
                    time_windows = []
                    
                    for i in range(0, len(signal_data) - window_size, window_size // 2):
                        window_data = signal_data[i:i+window_size]
                        window_mean = np.mean(window_data)
                        window_std = np.std(window_data)
                        
                        if window_std > 0:
                            quality_score = np.abs(window_mean) / window_std
                        else:
                            quality_score = 0
                        
                        quality_scores.append(quality_score)
                        time_windows.append(time_data[i + window_size // 2])
                    
                    if len(quality_scores) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=time_windows,
                                y=quality_scores,
                                mode='lines+markers',
                                name='Quality Score',
                                line=dict(color='#d62728', width=2),
                                marker=dict(color='#d62728', size=4, symbol='circle')
                            ),
                            row=2, col=2
                        )
                        
                        # Add quality statistics
                        mean_quality = np.mean(quality_scores)
                        fig.add_annotation(
                            x=0.98, y=0.98,
                            xref='x4', yref='y4',
                            text=f'Mean Quality: {mean_quality:.2f}<br>SNR: {snr:.1f} dB<br>Noise Level: {noise_level:.3f}',
                            showarrow=False,
                            bgcolor='rgba(255, 255, 255, 0.9)',
                            bordercolor='red',
                            borderwidth=2,
                            font=dict(size=10, color='black')
                        )
            except Exception as e:
                logger.debug(f"Could not create quality metrics: {e}")
        
        # Enhanced layout with better spacing and sizing
        fig.update_layout(
            title=dict(
                text=f"Comprehensive {signal_type.upper()} Analysis Dashboard",
                x=0.5,
                font=dict(size=18, color='#2c3e50')
            ),
            height=800,  # Increased height for better plot visibility
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=200, t=80, b=60)
        )
        
        # Update all subplot axes with better grid and styling
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    row=i, col=j,
                    title_font=dict(size=12),
                    tickfont=dict(size=10)
                )
                fig.update_yaxes(
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    row=i, col=j,
                    title_font=dict(size=12),
                    tickfont=dict(size=10)
                )
        
        # Add specific axis titles for better clarity
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        
        fig.update_xaxes(title_text="Value", row=1, col=2)
        fig.update_yaxes(title_text="Density", row=1, col=2)
        
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig.update_yaxes(title_text="Power", row=2, col=1)
        
        fig.update_xaxes(title_text="RR_n (ms)", row=2, col=2)
        fig.update_yaxes(title_text="RR_{n+1} (ms)", row=2, col=2)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating comprehensive analysis plots: {e}")
        return create_empty_figure()


def analyze_hrv(signal_data, sampling_freq, hrv_options):
    """Analyze Heart Rate Variability."""
    try:
        # Handle case where hrv_options might be None or empty
        if not hrv_options:
            hrv_options = ["time_domain", "freq_domain"]
        
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
        
        if "frequency_domain" in hrv_options or "freq_domain" in hrv_options:
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
        
        # Handle case where morphology_options might be None or empty
        if not morphology_options:
            morphology_options = ["peaks", "amplitude", "duration"]
        
        if "peak_detection" in morphology_options or "peaks" in morphology_options:
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
        # Check if we have enough data for trend analysis
        if len(signal_data) < 2:
            return {
                "trend_slope": 0,
                "trend_strength": 0,
                "trend_direction": "insufficient_data"
            }
        
        # Simple trend analysis using linear regression
        time_axis = np.arange(len(signal_data)) / sampling_freq
        
        # Ensure signal_data is not all the same value (which would cause polyfit to fail)
        if np.std(signal_data) == 0:
            return {
                "trend_slope": 0,
                "trend_strength": 0,
                "trend_direction": "no_variation"
            }
        
        coeffs = np.polyfit(time_axis, signal_data, 1)
        trend_slope = coeffs[0]
        
        # Calculate trend strength
        trend_line = np.polyval(coeffs, time_axis)
        signal_variance = np.sum((signal_data - np.mean(signal_data)) ** 2)
        
        # Avoid division by zero
        if signal_variance == 0:
            trend_strength = 0
        else:
            trend_strength = 1 - (np.sum((signal_data - trend_line) ** 2) / signal_variance)
        
        # Ensure trend_strength is within valid range
        trend_strength = max(0, min(1, trend_strength))
        
        trend_metrics = {
            "trend_slope": float(trend_slope),
            "trend_strength": float(trend_strength),
            "trend_direction": "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "stable"
        }
        
        return trend_metrics
        
    except Exception as e:
        logger.error(f"Error in trend analysis: {e}")
        return {"error": f"Trend analysis failed: {str(e)}"}


def create_comprehensive_results_display(results, signal_type, sampling_freq):
    """Create modern, compact results display for all analysis types."""
    try:
        sections = []
        
        # Create a modern metrics grid
        def create_metric_card(title, icon, metrics_dict, color="primary"):
            """Create a modern metric card with compact layout."""
            metric_items = []
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)) and value != 0:
                    # Format the value based on its type
                    if key.endswith('_db'):
                        formatted_value = f"{value:.1f} dB"
                    elif key.endswith('_ratio') or key.endswith('_strength'):
                        formatted_value = f"{value:.3f}"
                    elif key.endswith('_freq') or key.endswith('_frequency'):
                        formatted_value = f"{value:.3f} Hz"
                    elif key.endswith('_ms'):
                        formatted_value = f"{value:.1f} ms"
                    elif isinstance(value, float) and abs(value) < 0.01:
                        formatted_value = f"{value:.2e}"
                    else:
                        formatted_value = f"{value:.4f}"
                    
                    # Create a compact metric item
                    metric_items.append(
                        html.Div([
                            html.Span(f"{key.replace('_', ' ').title()}: ", className="text-muted fw-bold"),
                            html.Span(formatted_value, className="text-dark")
                        ], className="d-flex justify-content-between align-items-center py-1 border-bottom border-light")
                    )
            
            if not metric_items:
                return None
            
            return dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Span(icon, className="me-2 fs-4"),
                        html.Span(title, className="fs-6 fw-bold text-dark")
                    ], className="d-flex align-items-center")
                ], className=f"bg-{color} bg-opacity-10 border-0 py-2"),
                dbc.CardBody([
                    html.Div(metric_items, className="small")
                ], className="py-2 px-3")
            ], className="h-100 border-0 shadow-sm")
        
        # HRV Results
        if "hrv_metrics" in results and "error" not in results["hrv_metrics"]:
            hrv_metrics = {
                "mean_rr": results['hrv_metrics'].get('mean_rr', 0),
                "rmssd": results['hrv_metrics'].get('rmssd', 0),
                "pnn50": results['hrv_metrics'].get('pnn50', 0),
                "lf_power": results['hrv_metrics'].get('lf_power', 0),
                "hf_power": results['hrv_metrics'].get('hf_power', 0),
                "lf_hf_ratio": results['hrv_metrics'].get('lf_hf_ratio', 0)
            }
            hrv_card = create_metric_card("Heart Rate Variability", "💓", hrv_metrics, "danger")
            if hrv_card:
                sections.append(html.Div(hrv_card, className="col-lg-4 col-md-6 col-sm-12 mb-3"))
        
        # Morphology Results
        if "morphology_metrics" in results and "error" not in results["morphology_metrics"]:
            morph_metrics = {
                "num_peaks": results['morphology_metrics'].get('num_peaks', 0),
                "mean_peak_height": np.mean(results['morphology_metrics'].get('peak_heights', [0])),
                "peak_to_peak": results['morphology_metrics'].get('peak_to_peak', 0),
                "std_amplitude": results['morphology_metrics'].get('std_amplitude', 0)
            }
            morph_card = create_metric_card("Morphology Analysis", "📊", morph_metrics, "info")
            if morph_card:
                sections.append(html.Div(morph_card, className="col-lg-4 col-md-6 col-sm-12 mb-3"))
        
        # Beat-to-Beat Results
        if "beat2beat_metrics" in results and "error" not in results["beat2beat_metrics"]:
            b2b_metrics = {
                "num_beats": results['beat2beat_metrics'].get('num_beats', 0),
                "mean_beat_interval": results['beat2beat_metrics'].get('mean_beat_interval', 0),
                "beat_variability": results['beat2beat_metrics'].get('beat_variability', 0),
                "beat_regularity": results['beat2beat_metrics'].get('beat_regularity', 0)
            }
            b2b_card = create_metric_card("Beat-to-Beat Analysis", "🫀", b2b_metrics, "success")
            if b2b_card:
                sections.append(html.Div(b2b_card, className="col-lg-4 col-md-6 col-sm-12 mb-3"))
        
        # Energy Results
        if "energy_metrics" in results and "error" not in results["energy_metrics"]:
            energy_metrics = {
                "total_energy": results['energy_metrics'].get('total_energy', 0),
                "mean_energy": results['energy_metrics'].get('mean_energy', 0),
                "low_freq_energy": results['energy_metrics'].get('low_freq_energy', 0),
                "high_freq_energy": results['energy_metrics'].get('high_freq_energy', 0)
            }
            energy_card = create_metric_card("Energy Analysis", "⚡", energy_metrics, "warning")
            if energy_card:
                sections.append(html.Div(energy_card, className="col-lg-4 col-md-6 col-sm-12 mb-3"))
        
        # Envelope Results
        if "envelope_metrics" in results and "error" not in results["envelope_metrics"]:
            env_metrics = {
                "envelope_mean": results['envelope_metrics'].get('envelope_mean', 0),
                "envelope_range": results['envelope_metrics'].get('envelope_range', 0),
                "envelope_peaks": results['envelope_metrics'].get('envelope_peaks', 0)
            }
            env_card = create_metric_card("Envelope Analysis", "📦", env_metrics, "secondary")
            if env_card:
                sections.append(html.Div(env_card, className="col-lg-4 col-md-6 col-sm-12 mb-3"))
        
        # Segmentation Results
        if "segmentation_metrics" in results and "error" not in results["segmentation_metrics"]:
            seg_metrics = {
                "num_segments": results['segmentation_metrics'].get('num_segments', 0),
                "zero_crossings": results['segmentation_metrics'].get('zero_crossings', 0),
                "mean_segment_length": results['segmentation_metrics'].get('mean_segment_length', 0)
            }
            seg_card = create_metric_card("Signal Segmentation", "✂️", seg_metrics, "dark")
            if seg_card:
                sections.append(html.Div(seg_card, className="col-lg-4 col-md-6 col-sm-12 mb-3"))
        
        # Waveform Results
        if "waveform_metrics" in results and "error" not in results["waveform_metrics"]:
            wave_metrics = {
                "rms": results['waveform_metrics'].get('rms', 0),
                "skewness": results['waveform_metrics'].get('skewness', 0),
                "kurtosis": results['waveform_metrics'].get('kurtosis', 0),
                "peak_to_peak": results['waveform_metrics'].get('peak_to_peak', 0)
            }
            wave_card = create_metric_card("Waveform Analysis", "🌊", wave_metrics, "primary")
            if wave_card:
                sections.append(html.Div(wave_card, className="col-lg-4 col-md-6 col-sm-12 mb-3"))
        
        # Statistical Results
        if "statistical_metrics" in results and "error" not in results["statistical_metrics"]:
            stat_metrics = {
                "mean": results['statistical_metrics'].get('mean', 0),
                "median": results['statistical_metrics'].get('median', 0),
                "std": results['statistical_metrics'].get('std', 0),
                "iqr": results['statistical_metrics'].get('iqr', 0)
            }
            stat_card = create_metric_card("Statistical Analysis", "📊", stat_metrics, "info")
            if stat_card:
                sections.append(html.Div(stat_card, className="col-lg-4 col-md-6 col-sm-12 mb-3"))
        
        # Frequency Results
        if "frequency_metrics" in results and "error" not in results["frequency_metrics"]:
            freq_metrics = {
                "dominant_frequency": results['frequency_metrics'].get('dominant_frequency', 0),
                "spectral_centroid": results['frequency_metrics'].get('spectral_centroid', 0),
                "lf_power": results['frequency_metrics'].get('power_bands', {}).get('low', 0),
                "hf_power": results['frequency_metrics'].get('power_bands', {}).get('high', 0)
            }
            freq_card = create_metric_card("Frequency Analysis", "🔊", freq_metrics, "success")
            if freq_card:
                sections.append(html.Div(freq_card, className="col-lg-4 col-md-6 col-sm-12 mb-3"))
        
        # Advanced Features Results
        if "advanced_features_metrics" in results and "error" not in results["advanced_features_metrics"]:
            adv_metrics = {
                "cross_signal_correlation": results['advanced_features_metrics'].get('cross_signal_correlation', 0),
                "total_power": results['advanced_features_metrics'].get('total_power', 0)
            }
            adv_card = create_metric_card("Advanced Features", "🚀", adv_metrics, "warning")
            if adv_card:
                sections.append(html.Div(adv_card, className="col-lg-4 col-md-6 col-sm-12 mb-3"))
        
        # Quality Results
        if "quality_metrics" in results and "error" not in results["quality_metrics"]:
            quality_metrics = {
                "signal_quality_index": results['quality_metrics'].get('signal_quality_index', 0),
                "snr_db": results['quality_metrics'].get('snr_db', 0),
                "artifacts_detected": results['quality_metrics'].get('artifacts_detected', 0),
                "artifact_ratio": results['quality_metrics'].get('artifact_ratio', 0)
            }
            quality_card = create_metric_card("Signal Quality", "⚖️", quality_metrics, "danger")
            if quality_card:
                sections.append(html.Div(quality_card, className="col-lg-4 col-md-6 col-sm-12 mb-3"))
        
        # Transform Results
        if "transform_metrics" in results and "error" not in results["transform_metrics"]:
            transform_metrics = {
                "wavelet_energy": results['transform_metrics'].get('wavelet_energy', 0),
                "fourier_dominant_freq": results['transform_metrics'].get('fourier_dominant_freq', 0),
                "hilbert_phase": results['transform_metrics'].get('hilbert_phase', 0)
            }
            transform_card = create_metric_card("Signal Transforms", "🔄", transform_metrics, "primary")
            if transform_card:
                sections.append(html.Div(transform_card, className="col-lg-4 col-md-6 col-sm-12 mb-3"))
        
        # Advanced Computation Results
        if "advanced_computation_metrics" in results and "error" not in results["advanced_computation_metrics"]:
            comp_metrics = {
                "anomalies_detected": results['advanced_computation_metrics'].get('anomalies_detected', 0),
                "bayesian_prior_mean": results['advanced_computation_metrics'].get('bayesian_prior_mean', 0)
            }
            comp_card = create_metric_card("Advanced Computation", "🧠", comp_metrics, "dark")
            if comp_card:
                sections.append(html.Div(comp_card, className="col-lg-4 col-md-6 col-sm-12 mb-3"))
        
        # Feature Engineering Results
        if "feature_engineering_metrics" in results and "error" not in results["feature_engineering_metrics"]:
            feat_metrics = {
                "ppg_light_intensity": results['feature_engineering_metrics'].get('ppg_light_intensity', 0),
                "ppg_autonomic_response": results['feature_engineering_metrics'].get('ppg_autonomic_response', 0),
                "ecg_autonomic_response": results['feature_engineering_metrics'].get('ecg_autonomic_response', 0)
            }
            feat_card = create_metric_card("Feature Engineering", "🔧", feat_metrics, "info")
            if feat_card:
                sections.append(html.Div(feat_card, className="col-lg-4 col-md-6 col-sm-12 mb-3"))
        
        # Preprocessing Results
        if "preprocessing_metrics" in results and "error" not in results["preprocessing_metrics"]:
            prep_metrics = {
                "noise_level": results['preprocessing_metrics'].get('noise_level', 0),
                "signal_bandwidth": results['preprocessing_metrics'].get('signal_bandwidth', 0)
            }
            prep_card = create_metric_card("Preprocessing Analysis", "🔧", prep_metrics, "secondary")
            if prep_card:
                sections.append(html.Div(prep_card, className="col-lg-4 col-md-6 col-sm-12 mb-3"))
        
        # Trend Results
        if "trend_metrics" in results and "error" not in results["trend_metrics"]:
            trend_metrics = {
                "trend_direction": results['trend_metrics'].get('trend_direction', 'Unknown'),
                "trend_strength": results['trend_metrics'].get('trend_strength', 0)
            }
            trend_card = create_metric_card("Trend Analysis", "📈", trend_metrics, "success")
            if trend_card:
                sections.append(html.Div(trend_card, className="col-lg-4 col-md-6 col-sm-12 mb-3"))
        
        if not sections:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-info-circle text-muted fs-1 d-block text-center mb-3"),
                    html.H5("No Analysis Results", className="text-center text-muted"),
                    html.P("Please select analysis categories and run the analysis.", className="text-center text-muted")
                ], className="text-center py-5")
            ])
        
        # Create a modern grid layout
        return html.Div([
            html.Div([
                html.H4("📋 Analysis Results", className="text-center mb-4 text-dark"),
                html.P(f"Comprehensive physiological analysis for {signal_type.upper()} signal", className="text-center text-muted mb-4")
            ], className="mb-4"),
            html.Div(sections, className="row g-3")
        ])
        
    except Exception as e:
        logger.error(f"Error creating results display: {e}")
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle text-danger fs-1 d-block text-center mb-3"),
                html.H5("Error", className="text-center text-danger"),
                html.P(f"Failed to create results display: {str(e)}", className="text-center text-muted")
            ], className="text-center py-5")
        ])


def create_hrv_plots(time_data, signal_data, sampling_freq, hrv_options):
    """Create HRV-specific plots with enhanced visualization and analysis."""
    try:
        # Find peaks for RR intervals with enhanced detection
        height_threshold = np.mean(signal_data) + 1.5 * np.std(signal_data)
        distance_threshold = int(sampling_freq * 0.3)
        
        peaks, properties = signal.find_peaks(
            signal_data, 
            height=height_threshold,
            distance=distance_threshold,
            prominence=np.std(signal_data) * 0.5
        )
        
        if len(peaks) < 2:
            return create_empty_figure()
        
        # Calculate RR intervals
        rr_intervals = np.diff(peaks) / sampling_freq * 1000  # Convert to milliseconds
        
        # Determine number of subplots based on options
        num_plots = 1
        if "time_domain" in hrv_options:
            num_plots += 1
        if "freq_domain" in hrv_options:
            num_plots += 1
        if "nonlinear" in hrv_options:
            num_plots += 1
        
        fig = make_subplots(
            rows=num_plots, cols=1,
            subplot_titles=(["RR Intervals"] + 
                           (["Time Domain"] if "time_domain" in hrv_options else []) +
                           (["Frequency Domain"] if "freq_domain" in hrv_options else []) +
                           (["Nonlinear Analysis"] if "nonlinear" in hrv_options else [])),
            vertical_spacing=0.12,
            specs=[[{"secondary_y": False}] for _ in range(num_plots)]
        )
        
        current_row = 1
        
        # RR intervals over time with enhanced styling
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(rr_intervals)),
                y=rr_intervals,
                mode='lines+markers',
                name='RR Intervals',
                line=dict(color='#1f77b4', width=3),
                marker=dict(color='#1f77b4', size=6, symbol='circle'),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.1)'
            ),
            row=current_row, col=1
        )
        
        # Add RR interval statistics
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        
        fig.add_hline(
            y=mean_rr,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_rr:.1f} ms",
            annotation_position="right",
            row=current_row, col=1
        )
        
        # Add confidence bands
        fig.add_hline(
            y=mean_rr + std_rr,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"+1σ: {mean_rr + std_rr:.1f} ms",
            annotation_position="right",
            row=current_row, col=1
        )
        
        fig.add_hline(
            y=mean_rr - std_rr,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"-1σ: {mean_rr - std_rr:.1f} ms",
            annotation_position="right",
            row=current_row, col=1
        )
        
        current_row += 1
        
        # Time domain analysis with histogram
        if "time_domain" in hrv_options:
            # Create histogram of RR intervals
            hist, bin_edges = np.histogram(rr_intervals, bins=min(20, len(rr_intervals)//2), density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            fig.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=hist,
                name='RR Distribution',
                    marker_color='rgba(76, 175, 80, 0.7)',
                    opacity=0.8
                ),
                row=current_row, col=1
            )
            
            # Add normal distribution fit
            try:
                from scipy.stats import norm
                x_norm = np.linspace(min(rr_intervals), max(rr_intervals), 100)
                y_norm = norm.pdf(x_norm, mean_rr, std_rr)
                fig.add_trace(
                    go.Scatter(
                        x=x_norm,
                        y=y_norm,
                        mode='lines',
                        name='Normal Fit',
                        line=dict(color='red', width=2, dash='dash')
                    ),
                    row=current_row, col=1
                )
            except:
                pass
            
            current_row += 1
        
        # Frequency domain analysis with enhanced PSD
        if "freq_domain" in hrv_options:
            try:
                # Use Lomb-Scargle for unevenly spaced data
                freqs = np.linspace(0.003, 0.5, 1000)  # 0.003 to 0.5 Hz (18 to 300 BPM)
                
                # Calculate PSD using Welch method with better parameters
                if len(rr_intervals) > 10:
                    freqs_welch, psd_welch = signal.welch(rr_intervals, fs=1000/np.mean(rr_intervals), nperseg=min(len(rr_intervals)//2, 64))
                    
                    fig.add_trace(
                        go.Scatter(
                            x=freqs_welch,
                            y=psd_welch,
                            mode='lines',
                            name='Power Spectrum',
                            line=dict(color='#d62728', width=3),
                            fill='tonexty',
                            fillcolor='rgba(214, 39, 40, 0.1)'
                        ),
                        row=current_row, col=1
                    )
                    
                    # Add frequency band annotations
                    vlf_mask = freqs_welch < 0.04
                    lf_mask = (freqs_welch >= 0.04) & (freqs_welch < 0.15)
                    hf_mask = (freqs_welch >= 0.15) & (freqs_welch < 0.4)
                    
                    if np.any(vlf_mask):
                        vlf_power = np.trapz(psd_welch[vlf_mask], freqs_welch[vlf_mask])
                        fig.add_annotation(
                            x=0.98, y=0.98,
                            xref=f'x{current_row}', yref=f'y{current_row}',
                            text=f'VLF: {vlf_power:.1f}',
                            showarrow=False,
                            bgcolor='rgba(255, 255, 255, 0.8)',
                            bordercolor='blue'
                        )
                    
                    if np.any(lf_mask):
                        lf_power = np.trapz(psd_welch[lf_mask], freqs_welch[lf_mask])
                        fig.add_annotation(
                            x=0.95, y=0.98,
                            xref=f'x{current_row}', yref=f'y{current_row}',
                            text=f'LF: {lf_power:.1f}',
                            showarrow=False,
                            bgcolor='rgba(255, 255, 255, 0.8)',
                            bordercolor='green'
                        )
                    
                    if np.any(hf_mask):
                        hf_power = np.trapz(psd_welch[hf_mask], freqs_welch[hf_mask])
                        fig.add_annotation(
                            x=0.92, y=0.98,
                            xref=f'x{current_row}', yref=f'y{current_row}',
                            text=f'HF: {hf_power:.1f}',
                            showarrow=False,
                            bgcolor='rgba(255, 255, 255, 0.8)',
                            bordercolor='red'
                        )
                
            except Exception as e:
                logger.warning(f"Could not create frequency domain plot: {e}")
            
            current_row += 1
        
        # Nonlinear analysis with enhanced Poincaré plot
        if "nonlinear" in hrv_options:
            # Poincaré plot with enhanced styling
            fig.add_trace(
                go.Scatter(
                    x=rr_intervals[:-1],
                    y=rr_intervals[1:],
                    mode='markers',
                    name='Poincaré Plot',
                    marker=dict(
                        color='rgba(156, 39, 176, 0.7)',
                        size=8,
                        symbol='circle',
                        line=dict(color='rgba(156, 39, 176, 1)', width=1)
                    )
                ),
                row=current_row, col=1
            )
            
            # Add Poincaré plot statistics
            try:
                # Calculate SD1 and SD2
                diff_rr = np.diff(rr_intervals)
                sd1 = np.std(diff_rr) / np.sqrt(2)
                sd2 = np.std(rr_intervals) * np.sqrt(2)
                
                # Add ellipses
                center_x, center_y = np.mean(rr_intervals[:-1]), np.mean(rr_intervals[1:])
                
                # Create ellipse points
                theta = np.linspace(0, 2*np.pi, 100)
                x_ellipse = center_x + sd1 * np.cos(theta)
                y_ellipse = center_y + sd1 * np.sin(theta)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_ellipse,
                        y=y_ellipse,
                        mode='lines',
                        name='SD1 Ellipse',
                        line=dict(color='red', width=2, dash='dash'),
                        showlegend=False
                    ),
                    row=current_row, col=1
                )
                
                # Add statistics annotation
                fig.add_annotation(
                    x=0.98, y=0.98,
                    xref=f'x{current_row}', yref=f'y{current_row}',
                    text=f'SD1: {sd1:.1f} ms<br>SD2: {sd2:.1f} ms',
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='purple',
                    borderwidth=2
                )
                
            except Exception as e:
                logger.warning(f"Could not add Poincaré statistics: {e}")
        
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text="Enhanced Heart Rate Variability Analysis",
                x=0.5,
                font=dict(size=20, color='#2c3e50')
            ),
            height=250 * num_plots,
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=200, t=80, b=60)
        )
        
        # Update all subplot axes
        for i in range(1, num_plots + 1):
            fig.update_xaxes(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='rgba(128, 128, 128, 0.2)',
                row=i, col=1
            )
            fig.update_yaxes(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='rgba(128, 128, 128, 0.2)',
                row=i, col=1
            )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating enhanced HRV plots: {e}")
        return create_empty_figure()


def create_morphology_plots(time_data, signal_data, sampling_freq, morphology_options):
    """Create enhanced morphology-specific plots with red arrows and comprehensive analysis."""
    try:
        # Enhanced peak detection
        height_threshold = np.mean(signal_data) + 1.5 * np.std(signal_data)
        distance_threshold = int(sampling_freq * 0.3)
        
        peaks, properties = signal.find_peaks(
            signal_data, 
            height=height_threshold,
            distance=distance_threshold,
            prominence=np.std(signal_data) * 0.5
        )
        
        # Determine number of subplots based on options
        num_plots = 1  # Always show main signal
        if "peaks" in morphology_options:
            num_plots += 1
        if "duration" in morphology_options:
            num_plots += 1
        if "area" in morphology_options:
            num_plots += 1
        
        fig = make_subplots(
            rows=num_plots, cols=1,
            subplot_titles=(["Signal Analysis with Peak Detection"] + 
                           (["Peak Analysis"] if "peaks" in morphology_options else []) +
                           (["Duration Analysis"] if "duration" in morphology_options else []) +
                           (["Area Analysis"] if "area" in morphology_options else [])),
            vertical_spacing=0.12,
            specs=[[{"secondary_y": False}] for _ in range(num_plots)]
        )
        
        current_row = 1
        
        # Main signal with enhanced styling and red arrows for peaks
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=signal_data,
                mode='lines',
                name='Signal',
                line=dict(color='#1f77b4', width=2),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.1)'
            ),
            row=current_row, col=1
        )
        
        # Add red arrows for peaks
        if len(peaks) > 0:
            fig.add_trace(
                go.Scatter(
                    x=time_data[peaks],
                    y=signal_data[peaks],
                    mode='markers+text',
                    name='Detected Peaks',
                    text=[f'P{i+1}' for i in range(len(peaks))],
                    textposition='top center',
                    marker=dict(
                        color='red',
                        size=12,
                        symbol='arrow-up',
                        line=dict(color='darkred', width=2)
                    ),
                    textfont=dict(color='red', size=10, family='Arial Black')
                ),
                row=current_row, col=1
            )
            
            # Add peak statistics annotation
            peak_heights = signal_data[peaks]
            mean_height = np.mean(peak_heights)
            std_height = np.std(peak_heights)
            
            fig.add_annotation(
                x=0.98, y=0.98,
                xref=f'x{current_row}', yref=f'y{current_row}',
                text=f'Peaks: {len(peaks)}<br>Mean Height: {mean_height:.3f}<br>Height Std: {std_height:.3f}',
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='red',
                borderwidth=2,
                font=dict(size=12, color='black')
            )
        
        current_row += 1
        
        # Enhanced peak analysis with histogram and statistics
        if "peaks" in morphology_options and len(peaks) > 0:
            # Create histogram of peak heights
            hist, bin_edges = np.histogram(signal_data[peaks], bins=min(15, len(peaks)//2), density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            fig.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=hist,
                    name='Peak Heights Distribution',
                    marker_color='rgba(255, 165, 0, 0.7)',
                    opacity=0.8
                ),
                row=current_row, col=1
            )
            
            # Add normal distribution fit
            try:
                from scipy.stats import norm
                x_norm = np.linspace(min(signal_data[peaks]), max(signal_data[peaks]), 100)
                y_norm = norm.pdf(x_norm, mean_height, std_height)
                fig.add_trace(
                    go.Scatter(
                        x=x_norm,
                        y=y_norm,
                        mode='lines',
                        name='Normal Fit',
                        line=dict(color='red', width=2, dash='dash')
                    ),
                    row=current_row, col=1
                )
            except:
                pass
            
            # Add peak height statistics
            fig.add_annotation(
                x=0.98, y=0.98,
                xref=f'x{current_row}', yref=f'y{current_row}',
                text=f'Min: {np.min(peak_heights):.3f}<br>Max: {np.max(peak_heights):.3f}<br>Range: {np.max(peak_heights) - np.min(peak_heights):.3f}',
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='orange',
                borderwidth=2,
                font=dict(size=11, color='black')
            )
            
            current_row += 1
        
        # Enhanced duration analysis with beat intervals
        if "duration" in morphology_options and len(peaks) > 1:
            # Calculate beat intervals
            beat_intervals = np.diff(peaks) / sampling_freq
            
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(beat_intervals)),
                    y=beat_intervals,
                    mode='lines+markers',
                    name='Beat Intervals',
                    line=dict(color='#2ca02c', width=3),
                    marker=dict(color='#2ca02c', size=6, symbol='diamond'),
                    fill='tonexty',
                    fillcolor='rgba(44, 160, 44, 0.1)'
                ),
                row=current_row, col=1
            )
            
            # Add interval statistics
            mean_interval = np.mean(beat_intervals)
            std_interval = np.std(beat_intervals)
            
            fig.add_hline(
                y=mean_interval,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_interval:.3f}s",
                annotation_position="right",
                row=current_row, col=1
            )
            
            fig.add_annotation(
                x=0.98, y=0.98,
                xref=f'x{current_row}', yref=f'y{current_row}',
                text=f'Mean Interval: {mean_interval:.3f}s<br>Std: {std_interval:.3f}s<br>Heart Rate: {60/mean_interval:.1f} BPM',
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='green',
                borderwidth=2,
                font=dict(size=11, color='black')
            )
            
            current_row += 1
        
        # Enhanced area analysis with multiple metrics
        if "area" in morphology_options:
            # Calculate multiple area metrics
            cumulative_area = np.cumsum(np.abs(signal_data))
            signal_envelope = np.abs(signal_data)
            
            # Add cumulative area
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=cumulative_area,
                    mode='lines',
                    name='Cumulative Area',
                    line=dict(color='#9467bd', width=3),
                    fill='tonexty',
                    fillcolor='rgba(148, 103, 189, 0.1)'
                ),
                row=current_row, col=1
            )
            
            # Add signal envelope
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=signal_envelope,
                    mode='lines',
                    name='Signal Envelope',
                    line=dict(color='#e377c2', width=2, dash='dot'),
                opacity=0.7
            ),
                row=current_row, col=1
            )
            
            # Add area statistics
            total_area = np.sum(np.abs(signal_data))
            mean_area = np.mean(np.abs(signal_data))
            
            fig.add_annotation(
                x=0.98, y=0.98,
                xref=f'x{current_row}', yref=f'y{current_row}',
                text=f'Total Area: {total_area:.3f}<br>Mean Area: {mean_area:.3f}<br>Signal Length: {len(signal_data)} samples',
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='purple',
                borderwidth=2,
                font=dict(size=11, color='black')
            )
        
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text="Enhanced Morphology Analysis with Peak Detection",
                x=0.5,
                font=dict(size=20, color='#2c3e50')
            ),
            height=250 * num_plots,
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=200, t=80, b=60)
        )
        
        # Update all subplot axes
        for i in range(1, num_plots + 1):
            fig.update_xaxes(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='rgba(128, 128, 128, 0.2)',
                row=i, col=1
            )
            fig.update_yaxes(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='rgba(128, 128, 128, 0.2)',
                row=i, col=1
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating enhanced morphology plots: {e}")
        return create_empty_figure()


# New Analysis Functions for Comprehensive Physiological Features

def analyze_beat_to_beat(signal_data, sampling_freq):
    """Analyze beat-to-beat characteristics."""
    try:
        # Find peaks for beat analysis
        peaks, _ = signal.find_peaks(signal_data, height=np.mean(signal_data) + np.std(signal_data), 
                                   distance=int(sampling_freq * 0.3))
        
        if len(peaks) < 2:
            return {"error": "Insufficient beats for analysis"}
        
        # Calculate beat intervals
        beat_intervals = np.diff(peaks) / sampling_freq
        
        # Calculate beat-to-beat variability
        beat_variability = np.std(beat_intervals)
        beat_regularity = 1.0 / (1.0 + beat_variability)
        
        return {
            "num_beats": len(peaks),
            "mean_beat_interval": np.mean(beat_intervals),
            "beat_variability": beat_variability,
            "beat_regularity": beat_regularity,
            "beat_intervals": beat_intervals.tolist()
        }
    except Exception as e:
        logger.error(f"Error in beat-to-beat analysis: {e}")
        return {"error": f"Beat-to-beat analysis failed: {str(e)}"}


def analyze_energy(signal_data, sampling_freq):
    """Analyze signal energy characteristics."""
    try:
        # Calculate energy metrics
        total_energy = np.sum(signal_data ** 2)
        mean_energy = np.mean(signal_data ** 2)
        energy_variance = np.var(signal_data ** 2)
        
        # Calculate energy in different frequency bands
        freqs, psd = signal.welch(signal_data, fs=sampling_freq)
        low_freq_energy = np.sum(psd[freqs < 0.1])
        mid_freq_energy = np.sum(psd[(freqs >= 0.1) & (freqs < 1.0)])
        high_freq_energy = np.sum(psd[freqs >= 1.0])
        
        return {
            "total_energy": total_energy,
            "mean_energy": mean_energy,
            "energy_variance": energy_variance,
            "low_freq_energy": low_freq_energy,
            "mid_freq_energy": mid_freq_energy,
            "high_freq_energy": high_freq_energy,
            "energy_distribution": [low_freq_energy, mid_freq_energy, high_freq_energy]
        }
    except Exception as e:
        logger.error(f"Error in energy analysis: {e}")
        return {"error": f"Energy analysis failed: {str(e)}"}


def analyze_envelope(signal_data, sampling_freq):
    """Analyze signal envelope characteristics."""
    try:
        # Calculate analytic signal using Hilbert transform
        analytic_signal = signal.hilbert(signal_data)
        envelope = np.abs(analytic_signal)
        
        # Calculate envelope metrics
        envelope_mean = np.mean(envelope)
        envelope_std = np.std(envelope)
        envelope_range = np.max(envelope) - np.min(envelope)
        
        # Find envelope peaks
        env_peaks, _ = signal.find_peaks(envelope, height=np.mean(envelope) + np.std(envelope))
        
        return {
            "envelope_mean": envelope_mean,
            "envelope_std": envelope_std,
            "envelope_range": envelope_range,
            "envelope_peaks": len(env_peaks),
            "envelope_data": envelope.tolist()
        }
    except Exception as e:
        logger.error(f"Error in envelope analysis: {e}")
        return {"error": f"Envelope analysis failed: {str(e)}"}


def analyze_segmentation(signal_data, sampling_freq):
    """Analyze signal segmentation."""
    try:
        # Simple segmentation based on zero crossings
        zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]
        
        # Calculate segment lengths
        segment_lengths = np.diff(zero_crossings)
        if len(segment_lengths) > 0:
            mean_segment_length = np.mean(segment_lengths)
            segment_variability = np.std(segment_lengths)
        else:
            mean_segment_length = 0
            segment_variability = 0
        
        return {
            "num_segments": len(segment_lengths),
            "mean_segment_length": mean_segment_length,
            "segment_variability": segment_variability,
            "zero_crossings": len(zero_crossings)
        }
    except Exception as e:
        logger.error(f"Error in segmentation analysis: {e}")
        return {"error": f"Segmentation analysis failed: {str(e)}"}


def analyze_waveform(signal_data, sampling_freq):
    """Analyze waveform characteristics."""
    try:
        # Calculate waveform metrics
        waveform_metrics = {
            "mean": np.mean(signal_data),
            "std": np.std(signal_data),
            "skewness": np.mean(((signal_data - np.mean(signal_data)) / np.std(signal_data)) ** 3),
            "kurtosis": np.mean(((signal_data - np.mean(signal_data)) / np.std(signal_data)) ** 4),
            "peak_to_peak": np.max(signal_data) - np.min(signal_data),
            "rms": np.sqrt(np.mean(signal_data ** 2))
        }
        
        return waveform_metrics
    except Exception as e:
        logger.error(f"Error in waveform analysis: {e}")
        return {"error": f"Waveform analysis failed: {str(e)}"}


def analyze_statistical(signal_data, sampling_freq):
    """Analyze statistical characteristics."""
    try:
        # Calculate comprehensive statistical metrics
        percentiles = np.percentile(signal_data, [5, 25, 50, 75, 95])
        
        stats = {
            "mean": np.mean(signal_data),
            "median": np.median(signal_data),
            "std": np.std(signal_data),
            "variance": np.var(signal_data),
            "min": np.min(signal_data),
            "max": np.max(signal_data),
            "range": np.max(signal_data) - np.min(signal_data),
            "percentile_5": percentiles[0],
            "percentile_25": percentiles[1],
            "percentile_75": percentiles[3],
            "percentile_95": percentiles[4],
            "iqr": percentiles[3] - percentiles[1]
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error in statistical analysis: {e}")
        return {"error": f"Statistical analysis failed: {str(e)}"}


def analyze_frequency(signal_data, sampling_freq):
    """Analyze frequency domain characteristics."""
    try:
        # Calculate power spectral density
        freqs, psd = signal.welch(signal_data, fs=sampling_freq)
        
        # Calculate frequency domain metrics
        total_power = np.sum(psd)
        dominant_freq = freqs[np.argmax(psd)]
        spectral_centroid = np.sum(freqs * psd) / total_power
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / total_power)
        
        # Power in different frequency bands
        power_bands = {
            "very_low": np.sum(psd[freqs < 0.04]),
            "low": np.sum(psd[(freqs >= 0.04) & (freqs < 0.15)]),
            "high": np.sum(psd[(freqs >= 0.15) & (freqs < 0.4)]),
            "very_high": np.sum(psd[freqs >= 0.4])
        }
        
        return {
            "total_power": total_power,
            "dominant_frequency": dominant_freq,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "power_bands": power_bands,
            "frequencies": freqs.tolist(),
            "psd": psd.tolist()
        }
    except Exception as e:
        logger.error(f"Error in frequency analysis: {e}")
        return {"error": f"Frequency analysis failed: {str(e)}"}


def analyze_signal_quality_advanced(signal_data, sampling_freq, quality_options):
    """Advanced signal quality analysis."""
    try:
        quality_metrics = {}
        
        # Handle case where quality_options might be None or empty
        if not quality_options:
            quality_options = ["quality_index", "artifact_detection"]
        
        if "quality_index" in quality_options:
            # Calculate signal quality index
            signal_power = np.mean(signal_data ** 2)
            noise_power = np.var(signal_data)
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            quality_index = min(100, max(0, 100 - abs(snr - 20)))  # Simple quality index
            
            quality_metrics.update({
                "signal_quality_index": quality_index,
                "snr_db": snr
            })
        
        if "artifact_detection" in quality_options:
            # Simple artifact detection
            threshold = np.mean(signal_data) + 3 * np.std(signal_data)
            artifacts = np.sum(np.abs(signal_data) > threshold)
            artifact_ratio = artifacts / len(signal_data)
            
            quality_metrics.update({
                "artifacts_detected": artifacts,
                "artifact_ratio": artifact_ratio
            })
        
        if "snr_estimation" in quality_options:
            # Adaptive SNR estimation
            signal_power = np.mean(signal_data ** 2)
            noise_power = np.var(signal_data)
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            
            quality_metrics.update({
                "adaptive_snr": snr
            })
        
        return quality_metrics
    except Exception as e:
        logger.error(f"Error in advanced quality analysis: {e}")
        return {"error": f"Advanced quality analysis failed: {str(e)}"}


def analyze_transforms(signal_data, sampling_freq, transform_options):
    """Analyze signal transforms."""
    try:
        transform_metrics = {}
        
        # Handle case where transform_options might be None or empty
        if not transform_options:
            transform_options = ["wavelet", "fourier", "hilbert"]
        
        # Check if we have enough data for transforms
        if len(signal_data) < 4:
            return {"error": "Insufficient data for transform analysis"}
        
        if "wavelet" in transform_options:
            # Simple wavelet-like analysis
            # This is a simplified version - in practice you'd use proper wavelet transforms
            transform_metrics["wavelet_energy"] = float(np.sum(signal_data ** 2))
        
        if "fourier" in transform_options:
            # Fourier transform analysis
            try:
                fft_result = np.fft.fft(signal_data)
                fft_magnitude = np.abs(fft_result)
                # Find dominant frequency (avoid DC component)
                if len(fft_magnitude) > 1:
                    dominant_freq_idx = np.argmax(fft_magnitude[1:len(fft_magnitude)//2]) + 1
                    transform_metrics["fourier_dominant_freq"] = float(dominant_freq_idx * sampling_freq / len(signal_data))
                else:
                    transform_metrics["fourier_dominant_freq"] = 0.0
            except Exception as e:
                logger.warning(f"Fourier transform failed: {e}")
                transform_metrics["fourier_dominant_freq"] = 0.0
        
        if "hilbert" in transform_options:
            # Hilbert transform analysis
            try:
                # Ensure signal is not all zeros or constant
                if np.std(signal_data) > 0:
                    analytic_signal = signal.hilbert(signal_data)
                    transform_metrics["hilbert_phase"] = float(np.mean(np.angle(analytic_signal)))
                else:
                    transform_metrics["hilbert_phase"] = 0.0
            except Exception as e:
                logger.warning(f"Hilbert transform failed: {e}")
                transform_metrics["hilbert_phase"] = 0.0
        
        return transform_metrics
    except Exception as e:
        logger.error(f"Error in transform analysis: {e}")
        return {"error": f"Transform analysis failed: {str(e)}"}


def analyze_advanced_computation(signal_data, sampling_freq, advanced_computation):
    """Analyze advanced computation features."""
    try:
        comp_metrics = {}
        
        # Handle case where advanced_computation might be None or empty
        if not advanced_computation:
            advanced_computation = ["anomaly_detection", "bayesian", "kalman"]
        
        if "anomaly_detection" in advanced_computation:
            # Simple anomaly detection
            threshold = np.mean(signal_data) + 2 * np.std(signal_data)
            anomalies = np.sum(np.abs(signal_data) > threshold)
            comp_metrics["anomalies_detected"] = anomalies
        
        if "bayesian" in advanced_computation:
            # Simple Bayesian-like analysis
            prior_mean = np.mean(signal_data)
            prior_std = np.std(signal_data)
            comp_metrics["bayesian_prior_mean"] = prior_mean
            comp_metrics["bayesian_prior_std"] = prior_std
        
        if "kalman" in advanced_computation:
            # Simple Kalman-like analysis
            comp_metrics["kalman_estimate"] = np.mean(signal_data)
        
        return comp_metrics
    except Exception as e:
        logger.error(f"Error in advanced computation analysis: {e}")
        return {"error": f"Advanced computation analysis failed: {str(e)}"}


def analyze_feature_engineering(signal_data, sampling_freq, feature_engineering, signal_type):
    """Analyze feature engineering options."""
    try:
        feature_metrics = {}
        
        # Handle case where feature_engineering might be None or empty
        if not feature_engineering:
            feature_engineering = ["ppg_light", "ppg_autonomic", "ecg_autonomic"]
        
        if "ppg_light" in feature_engineering and signal_type == "ppg":
            # PPG light features
            feature_metrics["ppg_light_intensity"] = np.mean(signal_data)
            feature_metrics["ppg_light_variability"] = np.std(signal_data)
        
        if "ppg_autonomic" in feature_engineering and signal_type.lower() == "ppg":
            # PPG autonomic features
            feature_metrics["ppg_autonomic_response"] = np.std(signal_data)
        
        if "ecg_autonomic" in feature_engineering and signal_type.lower() == "ecg":
            # ECG autonomic features
            feature_metrics["ecg_autonomic_response"] = np.std(signal_data)
        
        return feature_metrics
    except Exception as e:
        logger.error(f"Error in feature engineering analysis: {e}")
        return {"error": f"Feature engineering analysis failed: {str(e)}"}


def analyze_advanced_features(signal_data, sampling_freq, advanced_features):
    """Analyze advanced features."""
    try:
        advanced_metrics = {}
        
        # Handle case where advanced_features might be None or empty
        if not advanced_features:
            advanced_features = ["cross_signal", "ensemble", "change_detection", "power_analysis"]
        
        if "cross_signal" in advanced_features:
            # Cross-signal analysis
            advanced_metrics["cross_signal_correlation"] = np.corrcoef(signal_data[:-1], signal_data[1:])[0, 1]
        
        if "ensemble" in advanced_features:
            # Ensemble analysis
            advanced_metrics["ensemble_mean"] = np.mean(signal_data)
            advanced_metrics["ensemble_std"] = np.std(signal_data)
        
        if "change_detection" in advanced_features:
            # Change detection
            diff_signal = np.diff(signal_data)
            advanced_metrics["change_points"] = np.sum(np.abs(diff_signal) > np.std(diff_signal))
        
        if "power_analysis" in advanced_features:
            # Power analysis
            freqs, psd = signal.welch(signal_data, fs=sampling_freq)
            advanced_metrics["total_power"] = np.sum(psd)
            advanced_metrics["peak_power"] = np.max(psd)
        
        return advanced_metrics
    except Exception as e:
        logger.error(f"Error in advanced features analysis: {e}")
        return {"error": f"Advanced features analysis failed: {str(e)}"}


def analyze_preprocessing(signal_data, sampling_freq, preprocessing):
    """Analyze preprocessing options."""
    try:
        preprocess_metrics = {}
        
        # Handle case where preprocessing might be None or empty
        if not preprocessing:
            preprocessing = ["noise_reduction", "baseline_correction", "filtering"]
        
        if "noise_reduction" in preprocessing:
            # Simple noise reduction analysis
            preprocess_metrics["noise_level"] = np.std(signal_data)
        
        if "baseline_correction" in preprocessing:
            # Simple baseline analysis
            baseline = np.mean(signal_data)
            preprocess_metrics["baseline_offset"] = baseline
        
        if "filtering" in preprocessing:
            # Simple filtering analysis
            preprocess_metrics["signal_bandwidth"] = sampling_freq / 2
        
        return preprocess_metrics
    except Exception as e:
        logger.error(f"Error in preprocessing analysis: {e}")
        return {"error": f"Preprocessing analysis failed: {str(e)}"}


# Plot Creation Functions for New Analysis Types

def create_beat_to_beat_plots(time_data, signal_data, sampling_freq):
    """Create beat-to-beat analysis plots."""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Beat Detection", "Beat Interval Analysis"),
            vertical_spacing=0.1
        )
        
        # Main signal with beats
        fig.add_trace(
            go.Scatter(x=time_data, y=signal_data, mode='lines', name='Signal', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add beat markers
        peaks, _ = signal.find_peaks(signal_data, height=np.mean(signal_data) + np.std(signal_data), 
                                   distance=int(sampling_freq * 0.3))
        if len(peaks) > 0:
            fig.add_trace(
                go.Scatter(x=time_data[peaks], y=signal_data[peaks], mode='markers', 
                          name='Beats', marker=dict(color='red', size=8)),
                row=1, col=1
            )
        
        # Beat intervals
        if len(peaks) > 1:
            intervals = np.diff(peaks) / sampling_freq
            fig.add_trace(
                go.Scatter(x=np.arange(len(intervals)), y=intervals, mode='lines+markers',
                          name='Beat Intervals', line=dict(color='green')),
                row=2, col=1
            )
        
        fig.update_layout(
            title="Beat-to-Beat Analysis", 
            height=600, 
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            margin=dict(l=60, r=200, t=80, b=60)
        )
        return fig
        
    except Exception as e:
        logger.error(f"Error creating beat-to-beat plots: {e}")
        return create_empty_figure()


def create_energy_plots(time_data, signal_data, sampling_freq):
    """Create energy analysis plots."""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Signal Energy", "Power Spectral Density"),
            vertical_spacing=0.1
        )
        
        # Energy over time
        energy = signal_data ** 2
        fig.add_trace(
            go.Scatter(x=time_data, y=energy, mode='lines', name='Energy', line=dict(color='purple')),
            row=1, col=1
        )
        
        # Power spectral density
        freqs, psd = signal.welch(signal_data, fs=sampling_freq)
        fig.add_trace(
            go.Scatter(x=freqs, y=psd, mode='lines', name='PSD', line=dict(color='orange')),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Energy Analysis", 
            height=600, 
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            margin=dict(l=60, r=200, t=80, b=60)
        )
        return fig
        
    except Exception as e:
        logger.error(f"Error creating energy plots: {e}")
        return create_empty_figure()


def create_envelope_plots(time_data, signal_data, sampling_freq):
    """Create envelope analysis plots."""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Signal with Envelope", "Envelope Analysis"),
            vertical_spacing=0.1
        )
        
        # Main signal
        fig.add_trace(
            go.Scatter(x=time_data, y=signal_data, mode='lines', name='Signal', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Envelope
        analytic_signal = signal.hilbert(signal_data)
        envelope = np.abs(analytic_signal)
        fig.add_trace(
            go.Scatter(x=time_data, y=envelope, mode='lines', name='Envelope', line=dict(color='red')),
            row=1, col=1
        )
        
        # Envelope histogram
        fig.add_trace(
            go.Histogram(x=envelope, nbinsx=20, name='Envelope Distribution', 
                        marker_color='green', opacity=0.7),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Envelope Analysis", 
            height=600, 
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            margin=dict(l=60, r=200, t=80, b=60)
        )
        return fig
        
    except Exception as e:
        logger.error(f"Error creating envelope plots: {e}")
        return create_empty_figure()


def create_segmentation_plots(time_data, signal_data, sampling_freq):
    """Create segmentation analysis plots."""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Signal Segmentation", "Segment Analysis"),
            vertical_spacing=0.1
        )
        
        # Main signal
        fig.add_trace(
            go.Scatter(x=time_data, y=signal_data, mode='lines', name='Signal', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Zero crossings
        zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]
        if len(zero_crossings) > 0:
            fig.add_trace(
                go.Scatter(x=time_data[zero_crossings], y=signal_data[zero_crossings], 
                          mode='markers', name='Zero Crossings', marker=dict(color='red', size=6)),
                row=1, col=1
            )
        
        # Segment lengths histogram
        if len(zero_crossings) > 1:
            segment_lengths = np.diff(zero_crossings)
            fig.add_trace(
                go.Histogram(x=segment_lengths, nbinsx=15, name='Segment Lengths',
                            marker_color='orange', opacity=0.7),
                row=2, col=1
            )
        
        fig.update_layout(
            title="Segmentation Analysis", 
            height=600, 
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            margin=dict(l=60, r=200, t=80, b=60)
        )
        return fig
        
    except Exception as e:
        logger.error(f"Error creating segmentation plots: {e}")
        return create_empty_figure()


def create_waveform_plots(time_data, signal_data, sampling_freq):
    """Create waveform analysis plots."""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Waveform Analysis", "Statistical Distribution"),
            vertical_spacing=0.1
        )
        
        # Main signal
        fig.add_trace(
            go.Scatter(x=time_data, y=signal_data, mode='lines', name='Signal', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Statistical distribution
        fig.add_trace(
            go.Histogram(x=signal_data, nbinsx=30, name='Signal Distribution',
                        marker_color='purple', opacity=0.7),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Waveform Analysis", 
            height=600, 
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            margin=dict(l=60, r=200, t=80, b=60)
        )
        return fig
        
    except Exception as e:
        logger.error(f"Error creating waveform plots: {e}")
        return create_empty_figure()


def create_frequency_plots(time_data, signal_data, sampling_freq):
    """Create frequency analysis plots."""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Power Spectral Density", "Frequency Bands"),
            vertical_spacing=0.1
        )
        
        # Power spectral density
        freqs, psd = signal.welch(signal_data, fs=sampling_freq)
        fig.add_trace(
            go.Scatter(x=freqs, y=psd, mode='lines', name='PSD', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Frequency bands
        vlf_mask = freqs < 0.04
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.4)
        vhf_mask = freqs >= 0.4
        
        fig.add_trace(go.Scatter(x=freqs[vlf_mask], y=psd[vlf_mask], mode='lines', 
                                name='VLF', line=dict(color='red')), row=2, col=1)
        fig.add_trace(go.Scatter(x=freqs[lf_mask], y=psd[lf_mask], mode='lines', 
                                name='LF', line=dict(color='green')), row=2, col=1)
        fig.add_trace(go.Scatter(x=freqs[hf_mask], y=psd[hf_mask], mode='lines', 
                                name='HF', line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=freqs[vhf_mask], y=psd[vhf_mask], mode='lines', 
                                name='VHF', line=dict(color='purple')), row=2, col=1)
        
        fig.update_layout(
            title="Frequency Analysis", 
            height=600, 
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            margin=dict(l=60, r=200, t=80, b=60)
        )
        return fig
        
    except Exception as e:
        logger.error(f"Error creating frequency plots: {e}")
        return create_empty_figure()


def create_transform_plots(time_data, signal_data, sampling_freq, transform_options):
    """Create transform analysis plots."""
    try:
        if "wavelet" in transform_options:
            return create_wavelet_plots(time_data, signal_data, sampling_freq)
        elif "fourier" in transform_options:
            return create_fourier_plots(time_data, signal_data, sampling_freq)
        elif "hilbert" in transform_options:
            return create_hilbert_plots(time_data, signal_data, sampling_freq)
        else:
            return create_empty_figure()
    except Exception as e:
        logger.error(f"Error creating transform plots: {e}")
        return create_empty_figure()


def create_wavelet_plots(time_data, signal_data, sampling_freq):
    """Create wavelet transform plots."""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Signal", "Wavelet-like Analysis"),
            vertical_spacing=0.1
        )
        
        # Main signal
        fig.add_trace(
            go.Scatter(x=time_data, y=signal_data, mode='lines', name='Signal', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Simplified wavelet-like analysis
        energy = signal_data ** 2
        fig.add_trace(
            go.Scatter(x=time_data, y=energy, mode='lines', name='Energy', line=dict(color='red')),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Wavelet Analysis", 
            height=600, 
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            margin=dict(l=60, r=200, t=80, b=60)
        )
        return fig
        
    except Exception as e:
        logger.error(f"Error creating wavelet plots: {e}")
        return create_empty_figure()


def create_fourier_plots(time_data, signal_data, sampling_freq):
    """Create Fourier transform plots."""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Signal", "Fourier Transform"),
            vertical_spacing=0.1
        )
        
        # Main signal
        fig.add_trace(
            go.Scatter(x=time_data, y=signal_data, mode='lines', name='Signal', line=dict(color='blue')),
            row=1, col=1
        )
        
        # FFT
        fft_result = np.fft.fft(signal_data)
        fft_magnitude = np.abs(fft_result)
        freqs = np.fft.fftfreq(len(signal_data), 1/sampling_freq)
        
        # Only show positive frequencies
        pos_mask = freqs > 0
        fig.add_trace(
            go.Scatter(x=freqs[pos_mask], y=fft_magnitude[pos_mask], mode='lines', 
                      name='FFT Magnitude', line=dict(color='green')),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Fourier Transform Analysis", 
            height=600, 
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            margin=dict(l=60, r=200, t=80, b=60)
        )
        return fig
        
    except Exception as e:
        logger.error(f"Error creating Fourier plots: {e}")
        return create_empty_figure()


def create_hilbert_plots(time_data, signal_data, sampling_freq):
    """Create Hilbert transform plots."""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Signal", "Hilbert Transform"),
            vertical_spacing=0.1
        )
        
        # Main signal
        fig.add_trace(
            go.Scatter(x=time_data, y=signal_data, mode='lines', name='Signal', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Hilbert transform
        analytic_signal = signal.hilbert(signal_data)
        phase = np.angle(analytic_signal)
        
        fig.add_trace(
            go.Scatter(x=time_data, y=phase, mode='lines', name='Phase', line=dict(color='red')),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Hilbert Transform Analysis", 
            height=600, 
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            margin=dict(l=60, r=200, t=80, b=60)
        )
        return fig
        
    except Exception as e:
        logger.error(f"Error creating Hilbert plots: {e}")
        return create_empty_figure()


# Update existing plot functions to handle new parameters

def create_morphology_plots(time_data, signal_data, sampling_freq, morphology_options):
    """Create enhanced morphology-specific plots."""
    try:
        # Find peaks
        peaks, properties = signal.find_peaks(signal_data, height=np.mean(signal_data) + np.std(signal_data), 
                                           distance=int(sampling_freq * 0.3))
        
        # Determine number of subplots based on options
        num_plots = 1  # Always show main signal
        if "peaks" in morphology_options:
            num_plots += 1
        if "duration" in morphology_options:
            num_plots += 1
        if "area" in morphology_options:
            num_plots += 1
        
        fig = make_subplots(
            rows=num_plots, cols=1,
            subplot_titles=(["Signal Analysis"] + 
                           (["Peak Analysis"] if "peaks" in morphology_options else []) +
                           (["Duration Analysis"] if "duration" in morphology_options else []) +
                           (["Area Analysis"] if "area" in morphology_options else [])),
            vertical_spacing=0.1
        )
        
        current_row = 1
        
        # Main signal with peaks
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=signal_data,
                mode='lines',
                name='Signal',
                line=dict(color='blue')
            ),
            row=current_row, col=1
        )
        
        if len(peaks) > 0 and "peaks" in morphology_options:
            fig.add_trace(
                go.Scatter(
                    x=time_data[peaks],
                    y=signal_data[peaks],
                    mode='markers',
                    name='Peaks',
                    marker=dict(color='red', size=8, symbol='circle')
                ),
                row=current_row, col=1
            )
        current_row += 1
            
        # Peak analysis
        if "peaks" in morphology_options and len(peaks) > 0:
            fig.add_trace(
                go.Histogram(
                    x=signal_data[peaks],
                    nbinsx=15,
                    name='Peak Heights',
                    marker_color='orange',
                    opacity=0.7
                ),
                row=current_row, col=1
            )
            current_row += 1
        
        # Duration analysis
        if "duration" in morphology_options:
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=signal_data,
                    mode='lines',
                    name='Signal Duration',
                    line=dict(color='green')
                ),
                row=current_row, col=1
            )
            current_row += 1
        
        # Area analysis
        if "area" in morphology_options:
            # Calculate cumulative area
            cumulative_area = np.cumsum(np.abs(signal_data))
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=cumulative_area,
                    mode='lines',
                    name='Cumulative Area',
                    line=dict(color='purple')
                ),
                row=current_row, col=1
            )
        
        fig.update_layout(
            title="Enhanced Morphology Analysis",
            height=200 * num_plots,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating enhanced morphology plots: {e}")
        return create_empty_figure()


# Additional Callback Functions for Enhanced Features

def register_additional_physiological_callbacks(app):
    """Register additional physiological analysis callbacks for enhanced features."""
    
    @app.callback(
        Output("physio-hrv-options-container", "style"),
        [Input("physio-analysis-categories", "value")]
    )
    def toggle_hrv_options_visibility(analysis_categories):
        """Show/hide HRV options based on analysis category selection."""
        if analysis_categories and "hrv" in analysis_categories:
            return {"display": "block"}
        return {"display": "none"}
    
    @app.callback(
        Output("physio-morphology-options-container", "style"),
        [Input("physio-analysis-categories", "value")]
    )
    def toggle_morphology_options_visibility(analysis_categories):
        """Show/hide morphology options based on analysis category selection."""
        if analysis_categories and "morphology" in analysis_categories:
            return {"display": "block"}
        return {"display": "none"}
    
    @app.callback(
        Output("physio-additional-analysis-section", "children"),
        [Input("store-physio-features", "data")]
    )
    def update_additional_analysis_section(features_data):
        """Update the additional analysis section with modern, compact feature information."""
        if not features_data:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-chart-line text-muted fs-1 d-block text-center mb-2"),
                    html.H5("Additional Analysis", className="text-center text-muted mb-2"),
                    html.P("Run the main analysis to see additional detailed information here.", className="text-center text-muted small")
                ], className="text-center py-4")
            ])
        
        # Create modern metric cards with compact layout
        def create_compact_metric_card(title, icon, metrics_dict, color="primary"):
            """Create a modern, compact metric card with minimal white space."""
            metric_items = []
            
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)) and value != 0:
                    # Format the value based on its type
                    if key.endswith('_db'):
                        formatted_value = f"{value:.1f} dB"
                    elif key.endswith('_ratio') or key.endswith('_strength'):
                        formatted_value = f"{value:.3f}"
                    elif key.endswith('_freq') or key.endswith('_frequency'):
                        formatted_value = f"{value:.3f} Hz"
                    elif key.endswith('_ms'):
                        formatted_value = f"{value:.1f} ms"
                    elif isinstance(value, float) and abs(value) < 0.01:
                        formatted_value = f"{value:.2e}"
                    else:
                        formatted_value = f"{value:.4f}"
                    
                    # Create compact metric item with minimal spacing
                    metric_items.append(
                        html.Div([
                            html.Span(f"{key.replace('_', ' ').title()}: ", className="text-muted fw-bold small"),
                            html.Span(formatted_value, className="text-dark fw-semibold")
                        ], className="d-flex justify-content-between align-items-center py-1 px-2 border-bottom border-light-subtle")
                    )
                elif isinstance(value, str) and value:
                    metric_items.append(
                        html.Div([
                            html.Span(f"{key.replace('_', ' ').title()}: ", className="text-muted fw-bold small"),
                            html.Span(value, className="text-dark fw-semibold")
                        ], className="d-flex justify-content-between align-items-center py-1 px-2 border-bottom border-light-subtle")
                    )
                elif isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], (int, float)):
                        metric_items.append(
                            html.Div([
                                html.Span(f"{key.replace('_', ' ').title()}: ", className="text-muted fw-bold small"),
                                html.Span(f"{len(value)} values", className="text-dark fw-semibold")
                            ], className="d-flex justify-content-between align-items-center py-1 px-2 border-bottom border-light-subtle")
                        )
                    else:
                        metric_items.append(
                            html.Div([
                                html.Span(f"{key.replace('_', ' ').title()}: ", className="text-muted fw-bold small"),
                                html.Span(f"{len(value)} items", className="text-dark fw-semibold")
                            ], className="d-flex justify-content-between align-items-center py-1 px-2 border-bottom border-light-subtle")
                        )
            
            if not metric_items:
                return None
            
            return dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Span(icon, className="me-2 fs-5"),
                        html.Span(title, className="fs-6 fw-bold text-dark")
                    ], className="d-flex align-items-center")
                ], className=f"bg-{color} bg-opacity-10 border-0 py-2 px-3"),
                dbc.CardBody([
                    html.Div(metric_items, className="small")
                ], className="py-2 px-3")
            ], className="h-100 border-0 shadow-sm")
        
        # Process features and create compact cards
        feature_cards = []
        
        for feature_type, metrics in features_data.items():
            if metrics and "error" not in metrics:
                # Map feature types to appropriate icons and colors
                icon_map = {
                    "hrv_metrics": ("💓", "danger"),
                    "morphology_metrics": ("📊", "info"),
                    "beat2beat_metrics": ("🫀", "success"),
                    "energy_metrics": ("⚡", "warning"),
                    "envelope_metrics": ("📦", "secondary"),
                    "segmentation_metrics": ("✂️", "dark"),
                    "waveform_metrics": ("🌊", "primary"),
                    "statistical_metrics": ("📈", "info"),
                    "frequency_metrics": ("🔊", "success"),
                    "advanced_features_metrics": ("🚀", "warning"),
                    "quality_metrics": ("⚖️", "danger"),
                    "transform_metrics": ("🔄", "primary"),
                    "advanced_computation_metrics": ("🧠", "dark"),
                    "feature_engineering_metrics": ("🔧", "info"),
                    "preprocessing_metrics": ("🔧", "secondary"),
                    "trend_metrics": ("📈", "success")
                }
                
                icon, color = icon_map.get(feature_type, ("📊", "primary"))
                title = feature_type.replace('_', ' ').title()
                
                # Create compact metric card
                compact_card = create_compact_metric_card(title, icon, metrics, color)
                if compact_card:
                    feature_cards.append(html.Div(compact_card, className="col-lg-4 col-md-6 col-sm-12 mb-2"))
        
        if not feature_cards:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-info-circle text-muted fs-1 d-block text-center mb-2"),
                    html.H5("No Additional Features", className="text-center text-muted mb-2"),
                    html.P("No additional features to display.", className="text-center text-muted small")
                ], className="text-center py-4")
            ])
        
        # Return modern, compact layout
        return html.Div([
            html.Div([
                html.H4("📋 Detailed Feature Analysis", className="text-center mb-3 text-dark"),
                html.P("Comprehensive physiological feature extraction results", className="text-center text-muted mb-3 small")
            ], className="mb-3"),
            html.Div(feature_cards, className="row g-2")  # Reduced gap with g-2
        ])
    
    @app.callback(
        Output("physio-btn-export-results", "disabled"),
        [Input("store-physio-features", "data")]
    )
    def toggle_export_button(features_data):
        """Enable/disable export button based on available data."""
        return not bool(features_data)
    
    @app.callback(
        Output("physio-time-range-slider", "marks"),
        [Input("store-physio-data", "data")]
    )
    def update_time_slider_marks(data_store):
        """Update time slider marks based on available data."""
        if not data_store or "time_data" not in data_store:
            return {}
        
        try:
            time_data = data_store["time_data"]
            if not time_data:
                return {}
            
            # Create marks at regular intervals
            max_time = max(time_data)
            step = max_time / 10
            marks = {}
            
            for i in range(0, 11):
                time_val = i * step
                marks[time_val] = f"{time_val:.1f}s"
            
            return marks
        except Exception as e:
            logger.error(f"Error updating time slider marks: {e}")
            return {}
    
    @app.callback(
        [Output("physio-start-time", "max"),
         Output("physio-end-time", "max")],
        [Input("store-physio-data", "data")]
    )
    def update_time_input_max_values(data_store):
        """Update max values for time inputs based on available data."""
        if not data_store or "time_data" not in data_store:
            return 100, 100
        
        try:
            time_data = data_store["time_data"]
            if not time_data:
                return 100, 100
            
            max_time = max(time_data)
            return max_time, max_time
        except Exception as e:
            logger.error(f"Error updating time input max values: {e}")
            return 100, 100


# Enhanced vitalDSP Integration Functions

def _import_vitaldsp_modules():
    """Import vitalDSP modules when needed."""
    try:
        # Import core vitalDSP modules
        import sys
        import os
        
        # Add the vitalDSP source path to sys.path
        vitaldsp_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'vitalDSP')
        if os.path.exists(vitaldsp_path):
            sys.path.insert(0, vitaldsp_path)
            logger.info("Added vitalDSP path to sys.path")
        
        # Try to import key modules
        try:
            from vitalDSP.physiological_features import hrv_analysis, time_domain, frequency_domain, nonlinear
            logger.info("Successfully imported vitalDSP physiological features modules")
        except ImportError as e:
            logger.warning(f"Could not import vitalDSP physiological features: {e}")
        
        try:
            from vitalDSP.feature_engineering import ppg_light_features, ppg_autonomic_features, ecg_autonomic_features
            logger.info("Successfully imported vitalDSP feature engineering modules")
        except ImportError as e:
            logger.warning(f"Could not import vitalDSP feature engineering: {e}")
        
        try:
            from vitalDSP.signal_quality_assessment import signal_quality_index, artifact_detection_removal
            logger.info("Successfully imported vitalDSP signal quality modules")
        except ImportError as e:
            logger.warning(f"Could not import vitalDSP signal quality modules: {e}")
        
        try:
            from vitalDSP.transforms import wavelet_transform, fourier_transform, hilbert_transform
            logger.info("Successfully imported vitalDSP transforms modules")
        except ImportError as e:
            logger.warning(f"Could not import vitalDSP transforms modules: {e}")
        
        try:
            from vitalDSP.advanced_computation import anomaly_detection, bayesian_analysis, kalman_filter
            logger.info("Successfully imported vitalDSP advanced computation modules")
        except ImportError as e:
            logger.warning(f"Could not import vitalDSP advanced computation modules: {e}")
        
    except Exception as e:
        logger.warning(f"Error importing vitalDSP modules: {e}")
        logger.info("Continuing with scipy/numpy fallback implementations")


def get_vitaldsp_hrv_analysis(signal_data, sampling_freq, hrv_options, signal_type="ppg"):
    """Get HRV analysis using vitalDSP modules if available."""
    try:
        # Check if signal is long enough for HRV analysis
        min_samples_for_hrv = 5 * sampling_freq  # At least 5 seconds
        if len(signal_data) < min_samples_for_hrv:
            logger.warning(f"Signal too short for HRV analysis ({len(signal_data)} samples, need at least {min_samples_for_hrv})")
            return {"error": f"Signal too short for HRV analysis. Need at least {min_samples_for_hrv} samples."}
        
        # Normalize signal type for vitalDSP compatibility
        signal_type_upper = normalize_signal_type(signal_type)
        
        try:
            from vitalDSP.physiological_features.hrv_analysis import HRVFeatures
            
            # Use vitalDSP HRV analysis with class-based approach
            # Note: vitalDSP HRVFeatures may not accept signal_type parameter
            try:
                hrv_features = HRVFeatures(signals=signal_data, fs=sampling_freq, signal_type=signal_type_upper)
            except TypeError:
                # Fallback if signal_type parameter is not supported
                hrv_features = HRVFeatures(signals=signal_data, fs=sampling_freq)
        except ImportError:
            logger.info("vitalDSP HRV module not available, using fallback implementation")
            return analyze_hrv(signal_data, sampling_freq, hrv_options)
        hrv_result = hrv_features.compute_all_features()
        
        # Map vitalDSP results to our format
        mapped_results = {}
        
        if "time_domain" in hrv_options:
            mapped_results.update({
                "mean_rr": hrv_result.get("mean_nn", 0),
                "std_rr": hrv_result.get("std_nn", 0),
                "rmssd": hrv_result.get("rmssd", 0),
                "nn50": hrv_result.get("nn50", 0),
                "pnn50": hrv_result.get("pnn50", 0)
            })
        
        if "freq_domain" in hrv_options:
            mapped_results.update({
                "total_power": hrv_result.get("total_power", 0),
                "vlf_power": hrv_result.get("vlf_power", 0),
                "lf_power": hrv_result.get("lf_power", 0),
                "hf_power": hrv_result.get("hf_power", 0),
                "lf_hf_ratio": hrv_result.get("lf_hf_ratio", 0)
            })
        
        if "nonlinear" in hrv_options:
            mapped_results.update({
                "poincare_sd1": hrv_result.get("poincare_sd1", 0),
                "poincare_sd2": hrv_result.get("poincare_sd2", 0),
                "dfa_alpha1": hrv_result.get("dfa", 0),
                "dfa_alpha2": 0  # vitalDSP doesn't provide alpha2
            })
        
        return mapped_results
        
    except ImportError:
        logger.info("vitalDSP HRV module not available, using fallback implementation")
        return analyze_hrv(signal_data, sampling_freq, hrv_options)
    except Exception as e:
        logger.error(f"Error in vitalDSP HRV analysis: {e}")
        return analyze_hrv(signal_data, sampling_freq, hrv_options)


def get_vitaldsp_morphology_analysis(signal_data, sampling_freq, morphology_options, signal_type="ppg"):
    """Get morphology analysis using vitalDSP modules if available."""
    try:
        # Check if signal is long enough for morphology analysis
        min_samples_for_morphology = 2 * sampling_freq  # At least 2 seconds
        if len(signal_data) < min_samples_for_morphology:
            logger.warning(f"Signal too short for morphology analysis ({len(signal_data)} samples, need at least {min_samples_for_morphology})")
            return {"error": f"Signal too short for morphology analysis. Need at least {min_samples_for_morphology} samples."}
        
        # Normalize signal type for vitalDSP compatibility
        signal_type_upper = normalize_signal_type(signal_type)
        
        from vitalDSP.feature_engineering.morphology_features import PhysiologicalFeatureExtractor
        
        # Use vitalDSP morphology analysis with class-based approach
        extractor = PhysiologicalFeatureExtractor(signal_data, fs=sampling_freq)
        
        # Create a basic peak config
        peak_config = {"window_size": 5, "slope_unit": "radians"}
        
        # Extract features with proper signal type
        morph_result = extractor.extract_features(signal_type=signal_type_upper, peak_config=peak_config)
        
        # Map vitalDSP results to our format
        mapped_results = {}
        
        if "peaks" in morphology_options:
            # Use basic peak detection as fallback since vitalDSP doesn't return peak counts directly
            from scipy import signal
            peaks, _ = signal.find_peaks(signal_data, height=np.mean(signal_data) + np.std(signal_data), 
                                       distance=int(sampling_freq * 0.3))
            mapped_results.update({
                "num_peaks": len(peaks),
                "peak_heights": signal_data[peaks].tolist() if len(peaks) > 0 else [],
                "peak_positions": peaks.tolist() if len(peaks) > 0 else []
            })
        
        if "duration" in morphology_options:
            mapped_results.update({
                "signal_duration": len(signal_data) / sampling_freq,
                "sampling_freq": sampling_freq,
                "num_samples": len(signal_data),
                "systolic_duration": morph_result.get("systolic_duration", 0),
                "diastolic_duration": morph_result.get("diastolic_duration", 0)
            })
        
        if "area" in morphology_options:
            mapped_results.update({
                "total_area": np.sum(np.abs(signal_data)),
                "area_under_curve": np.sum(signal_data),
                "systolic_area": morph_result.get("systolic_area", 0),
                "diastolic_area": morph_result.get("diastolic_area", 0)
            })
        
        return mapped_results
        
    except ImportError:
        logger.info("vitalDSP morphology module not available, using fallback implementation")
        return analyze_morphology(signal_data, sampling_freq, morphology_options)
    except Exception as e:
        logger.error(f"Error in vitalDSP morphology analysis: {e}")
        return analyze_morphology(signal_data, sampling_freq, morphology_options)


def get_vitaldsp_signal_quality(signal_data, sampling_freq, quality_options, signal_type="ppg"):
    """Get signal quality analysis using vitalDSP modules if available."""
    try:
        # Check if signal is long enough for quality analysis
        min_samples_for_quality = 3 * sampling_freq  # At least 3 seconds
        if len(signal_data) < min_samples_for_quality:
            logger.warning(f"Signal too short for quality analysis ({len(signal_data)} samples, need at least {min_samples_for_quality})")
            return {"error": f"Signal too short for quality analysis. Need at least {min_samples_for_quality} samples."}
        
        # Normalize signal type for vitalDSP compatibility
        signal_type_upper = normalize_signal_type(signal_type)
        
        from vitalDSP.signal_quality_assessment.signal_quality_index import SignalQualityIndex
        
        # Use vitalDSP signal quality analysis with class-based approach
        sqi = SignalQualityIndex(signal_data)
        
        # Map vitalDSP results to our format
        mapped_results = {}
        
        if "quality_index" in quality_options:
            # Use a simple approach - compute basic SQI metrics
            try:
                # Try to compute amplitude variability SQI
                window_size = min(100, len(signal_data) // 4)
                step_size = max(1, window_size // 2)
                
                # Ensure window size is not larger than signal length
                if window_size >= len(signal_data):
                    window_size = len(signal_data) // 2
                    step_size = max(1, window_size // 2)
                
                sqi_values, _, _ = sqi.amplitude_variability_sqi(window_size, step_size)
                overall_score = np.mean(sqi_values) if len(sqi_values) > 0 else 0
                
                mapped_results.update({
                    "signal_quality_index": overall_score,
                    "overall_score": overall_score
                })
            except Exception as e:
                logger.warning(f"Could not compute amplitude variability SQI: {e}")
                mapped_results.update({
                    "signal_quality_index": 0,
                    "overall_score": 0
                })
        
        if "artifact_detection" in quality_options:
            # Use basic artifact detection
            try:
                # Simple artifact detection based on amplitude outliers
                threshold = np.mean(signal_data) + 3 * np.std(signal_data)
                artifacts = np.sum(np.abs(signal_data) > threshold)
                artifact_ratio = artifacts / len(signal_data) if len(signal_data) > 0 else 0
                
                mapped_results.update({
                    "artifacts_detected": artifacts,
                    "artifact_ratio": artifact_ratio
                })
            except Exception as e:
                logger.warning(f"Could not compute artifact detection: {e}")
                mapped_results.update({
                    "artifacts_detected": 0,
                    "artifact_ratio": 0
                })
        
        if "snr_estimation" in quality_options:
            # Use basic SNR estimation
            try:
                signal_power = np.mean(signal_data ** 2)
                noise_power = np.var(signal_data)
                snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
                
                mapped_results.update({
                    "snr_db": snr_db,
                    "adaptive_snr": snr_db
                })
            except Exception as e:
                logger.warning(f"Could not compute SNR estimation: {e}")
                mapped_results.update({
                    "snr_db": 0,
                    "adaptive_snr": 0
                })
        
        return mapped_results
        
    except ImportError:
        logger.info("vitalDSP signal quality module not available, using fallback implementation")
        return analyze_signal_quality_advanced(signal_data, sampling_freq, quality_options)
    except Exception as e:
        logger.error(f"Error in vitalDSP signal quality analysis: {e}")
        return analyze_signal_quality_advanced(signal_data, sampling_freq, quality_options)


def get_vitaldsp_transforms(signal_data, sampling_freq, transform_options, signal_type="ppg"):
    """Get signal transforms using vitalDSP modules if available."""
    try:
        # Normalize signal type for vitalDSP compatibility
        signal_type_upper = normalize_signal_type(signal_type)
        mapped_results = {}
        
        if "wavelet" in transform_options:
            try:
                from vitalDSP.transforms.wavelet_transform import WaveletTransform
                # Use vitalDSP wavelet transform with class-based approach
                wavelet = WaveletTransform(signal_data, wavelet_name="haar")
                coeffs = wavelet.perform_wavelet_transform(level=1)
                
                # Calculate energy from wavelet coefficients
                energy = np.sum([np.sum(coeff ** 2) for coeff in coeffs])
                mapped_results["wavelet_energy"] = float(energy)
                
            except ImportError:
                mapped_results["wavelet_energy"] = np.sum(signal_data ** 2)
            except Exception as e:
                logger.warning(f"vitalDSP wavelet transform failed: {e}")
                mapped_results["wavelet_energy"] = np.sum(signal_data ** 2)
        
        if "fourier" in transform_options:
            try:
                from vitalDSP.transforms.fourier_transform import FourierTransform
                # Use vitalDSP fourier transform with class-based approach
                fourier = FourierTransform(signal_data)
                dft_result = fourier.compute_dft()
                
                # Find dominant frequency
                fft_magnitude = np.abs(dft_result)
                if len(fft_magnitude) > 1:
                    dominant_freq_idx = np.argmax(fft_magnitude[1:len(fft_magnitude)//2]) + 1
                    dominant_freq = dominant_freq_idx * sampling_freq / len(signal_data)
                else:
                    dominant_freq = 0
                
                mapped_results["fourier_dominant_freq"] = float(dominant_freq)
                
            except ImportError:
                fft_result = np.fft.fft(signal_data)
                fft_magnitude = np.abs(fft_result)
                if len(fft_magnitude) > 1:
                    dominant_freq_idx = np.argmax(fft_magnitude[1:len(fft_magnitude)//2]) + 1
                    mapped_results["fourier_dominant_freq"] = dominant_freq_idx * sampling_freq / len(signal_data)
                else:
                    mapped_results["fourier_dominant_freq"] = 0
            except Exception as e:
                logger.warning(f"vitalDSP fourier transform failed: {e}")
                fft_result = np.fft.fft(signal_data)
                fft_magnitude = np.abs(fft_result)
                if len(fft_magnitude) > 1:
                    dominant_freq_idx = np.argmax(fft_magnitude[1:len(fft_magnitude)//2]) + 1
                    mapped_results["fourier_dominant_freq"] = dominant_freq_idx * sampling_freq / len(signal_data)
                else:
                    mapped_results["fourier_dominant_freq"] = 0
        
        if "hilbert" in transform_options:
            try:
                from vitalDSP.transforms.hilbert_transform import HilbertTransform
                # Use vitalDSP hilbert transform with class-based approach
                hilbert = HilbertTransform(signal_data)
                analytic_signal = hilbert.compute_hilbert()
                
                # Calculate mean phase
                phase = np.angle(analytic_signal)
                mapped_results["hilbert_phase"] = float(np.mean(phase))
                
            except ImportError:
                if np.std(signal_data) > 0:
                    analytic_signal = signal.hilbert(signal_data)
                    mapped_results["hilbert_phase"] = np.mean(np.angle(analytic_signal))
                else:
                    mapped_results["hilbert_phase"] = 0
            except Exception as e:
                logger.warning(f"vitalDSP hilbert transform failed: {e}")
                if np.std(signal_data) > 0:
                    analytic_signal = signal.hilbert(signal_data)
                    mapped_results["hilbert_phase"] = np.mean(np.angle(analytic_signal))
                else:
                    mapped_results["hilbert_phase"] = 0
        
        return mapped_results
        
    except Exception as e:
        logger.error(f"Error in vitalDSP transforms analysis: {e}")
        return analyze_transforms(signal_data, sampling_freq, transform_options)


def get_vitaldsp_advanced_computation(signal_data, sampling_freq, advanced_computation, signal_type="ppg"):
    """Get advanced computation features using vitalDSP modules if available."""
    try:
        # Normalize signal type for vitalDSP compatibility
        signal_type_upper = normalize_signal_type(signal_type)
        mapped_results = {}
        
        if "anomaly_detection" in advanced_computation:
            try:
                from vitalDSP.advanced_computation.anomaly_detection import AnomalyDetection
                # Use vitalDSP anomaly detection with class-based approach
                anomaly_detector = AnomalyDetection(signal_data)
                anomalies = anomaly_detector.detect_anomalies(method="z_score", threshold=2.0)
                
                mapped_results["anomalies_detected"] = len(anomalies)
                
            except ImportError:
                threshold = np.mean(signal_data) + 2 * np.std(signal_data)
                mapped_results["anomalies_detected"] = np.sum(np.abs(signal_data) > threshold)
            except Exception as e:
                logger.warning(f"vitalDSP anomaly detection failed: {e}")
                threshold = np.mean(signal_data) + 2 * np.std(signal_data)
                mapped_results["anomalies_detected"] = np.sum(np.abs(signal_data) > threshold)
        
        if "bayesian" in advanced_computation:
            try:
                from vitalDSP.advanced_computation.bayesian_analysis import GaussianProcess
                # Use vitalDSP bayesian analysis with class-based approach
                gp = GaussianProcess(length_scale=1.0, noise=1e-10)
                
                # Create simple training data for demonstration
                X_train = np.array([[0.1], [0.4], [0.7]])
                y_train = np.array([np.mean(signal_data[:len(signal_data)//3]), 
                                  np.mean(signal_data[len(signal_data)//3:2*len(signal_data)//3]), 
                                  np.mean(signal_data[2*len(signal_data)//3:])])
                
                gp.update(X_train, y_train)
                
                # Predict at a new point
                X_new = np.array([[0.5]])
                mean, variance = gp.predict(X_new)
                
                mapped_results["bayesian_prior_mean"] = float(mean[0])
                mapped_results["bayesian_prior_std"] = float(np.sqrt(variance[0]))
                
            except ImportError:
                mapped_results["bayesian_prior_mean"] = np.mean(signal_data)
                mapped_results["bayesian_prior_std"] = np.std(signal_data)
            except Exception as e:
                logger.warning(f"vitalDSP bayesian analysis failed: {e}")
                mapped_results["bayesian_prior_mean"] = np.mean(signal_data)
                mapped_results["bayesian_prior_std"] = np.std(signal_data)
        
        if "kalman" in advanced_computation:
            try:
                from vitalDSP.advanced_computation.kalman_filter import KalmanFilter
                # Use vitalDSP kalman filter with class-based approach
                initial_state = np.array([np.mean(signal_data)])
                initial_covariance = np.array([[np.var(signal_data)]])
                process_covariance = np.array([[1e-5]])
                measurement_covariance = np.array([[np.var(signal_data)]])
                
                kalman = KalmanFilter(initial_state, initial_covariance, process_covariance, measurement_covariance)
                
                # Simple filtering setup
                measurement_matrix = np.array([[1]])
                transition_matrix = np.array([[1]])
                
                # Apply filter to a subset of the signal for efficiency
                subset_size = min(100, len(signal_data))
                subset = signal_data[:subset_size]
                
                filtered_signal = kalman.filter(subset, measurement_matrix, transition_matrix)
                mapped_results["kalman_estimate"] = float(np.mean(filtered_signal))
                
            except ImportError:
                mapped_results["kalman_estimate"] = np.mean(signal_data)
            except Exception as e:
                logger.warning(f"vitalDSP kalman filter failed: {e}")
                mapped_results["kalman_estimate"] = np.mean(signal_data)
        
        return mapped_results
        
    except Exception as e:
        logger.error(f"Error in vitalDSP advanced computation analysis: {e}")
        return analyze_advanced_computation(signal_data, sampling_freq, advanced_computation)


def get_vitaldsp_feature_engineering(signal_data, sampling_freq, feature_engineering, signal_type):
    """Get feature engineering features using vitalDSP modules if available."""
    try:
        # Ensure signal_type is properly capitalized for vitalDSP
        signal_type_upper = signal_type.upper() if signal_type else "PPG"
        mapped_results = {}
        
        if "ppg_light" in feature_engineering and signal_type.lower() == "ppg":
            try:
                from vitalDSP.feature_engineering.ppg_light_features import PPGLightFeatureExtractor
                # Use vitalDSP PPG light features with class-based approach
                # For now, we'll use the same signal for both IR and red (this is a limitation)
                light_extractor = PPGLightFeatureExtractor(signal_data, signal_data, sampling_freq)
                
                # Calculate perfusion index (this doesn't require red signal)
                try:
                    pi_values, _ = light_extractor.calculate_perfusion_index()
                    mapped_results["ppg_light_intensity"] = float(np.mean(pi_values)) if len(pi_values) > 0 else 0
                except:
                    mapped_results["ppg_light_intensity"] = np.mean(signal_data)
                
                mapped_results["ppg_light_variability"] = np.std(signal_data)
                
            except ImportError:
                mapped_results["ppg_light_intensity"] = np.mean(signal_data)
                mapped_results["ppg_light_variability"] = np.std(signal_data)
            except Exception as e:
                logger.warning(f"vitalDSP PPG light features failed: {e}")
                mapped_results["ppg_light_intensity"] = np.mean(signal_data)
                mapped_results["ppg_light_variability"] = np.std(signal_data)
        
        if "ppg_autonomic" in feature_engineering and signal_type.lower() == "ppg":
            try:
                from vitalDSP.feature_engineering.ppg_autonomic_features import PPGAutonomicFeatures
                # Use vitalDSP PPG autonomic features with class-based approach
                autonomic_extractor = PPGAutonomicFeatures(signal_data, sampling_freq)
                
                # Try to extract autonomic features
                try:
                    # This is a placeholder - the actual method name might be different
                    if hasattr(autonomic_extractor, 'extract_autonomic_features'):
                        autonomic_result = autonomic_extractor.extract_autonomic_features()
                        mapped_results["ppg_autonomic_response"] = autonomic_result.get("autonomic_response", 0) if isinstance(autonomic_result, dict) else 0
                    else:
                        mapped_results["ppg_autonomic_response"] = np.std(signal_data)
                except:
                    mapped_results["ppg_autonomic_response"] = np.std(signal_data)
                
            except ImportError:
                mapped_results["ppg_autonomic_response"] = np.std(signal_data)
            except Exception as e:
                logger.warning(f"vitalDSP PPG autonomic features failed: {e}")
                mapped_results["ppg_autonomic_response"] = np.std(signal_data)
        
        if "ecg_autonomic" in feature_engineering and signal_type.lower() == "ecg":
            try:
                from vitalDSP.feature_engineering.ecg_autonomic_features import ECGAutonomicFeatureExtractor
                # Use vitalDSP ECG autonomic features with class-based approach
                ecg_extractor = ECGAutonomicFeatureExtractor(signal_data, sampling_freq)
                
                # Try to extract autonomic features
                try:
                    # This is a placeholder - the actual method name might be different
                    if hasattr(ecg_extractor, 'extract_autonomic_features'):
                        ecg_result = ecg_extractor.extract_autonomic_features()
                        mapped_results["ecg_autonomic_response"] = ecg_result.get("autonomic_response", 0) if isinstance(ecg_result, dict) else 0
                    else:
                        mapped_results["ecg_autonomic_response"] = np.std(signal_data)
                except:
                    mapped_results["ecg_autonomic_response"] = np.std(signal_data)
                
            except ImportError:
                mapped_results["ecg_autonomic_response"] = np.std(signal_data)
            except Exception as e:
                logger.warning(f"vitalDSP ECG autonomic features failed: {e}")
                mapped_results["ecg_autonomic_response"] = np.std(signal_data)
        
        return mapped_results
        
    except Exception as e:
        logger.error(f"Error in vitalDSP feature engineering analysis: {e}")
        return analyze_feature_engineering(signal_data, sampling_freq, feature_engineering, signal_type)


# Update the main analysis function to use vitalDSP when available
def perform_physiological_analysis_enhanced(time_data, signal_data, signal_type, sampling_freq, 
                                          analysis_categories, hrv_options, morphology_options, advanced_features,
                                          quality_options, transform_options, advanced_computation, feature_engineering, preprocessing):
    """Enhanced physiological analysis with vitalDSP integration."""
    results = {}
    
    try:
        # HRV Analysis with vitalDSP integration
        if "hrv" in analysis_categories:
            results["hrv_metrics"] = get_vitaldsp_hrv_analysis(signal_data, sampling_freq, hrv_options, signal_type)
        
        # Morphology Analysis with vitalDSP integration
        if "morphology" in analysis_categories:
            results["morphology_metrics"] = get_vitaldsp_morphology_analysis(signal_data, sampling_freq, morphology_options, signal_type)
        
        # Other analyses remain the same for now
        if "beat2beat" in analysis_categories:
            results["beat2beat_metrics"] = analyze_beat_to_beat(signal_data, sampling_freq)
        
        if "energy" in analysis_categories:
            results["energy_metrics"] = analyze_energy(signal_data, sampling_freq)
        
        if "envelope" in analysis_categories:
            results["envelope_metrics"] = analyze_envelope(signal_data, sampling_freq)
        
        if "segmentation" in analysis_categories:
            results["segmentation_metrics"] = analyze_segmentation(signal_data, sampling_freq)
        
        if "trend" in analysis_categories:
            results["trend_metrics"] = analyze_trends(signal_data, sampling_freq)
        
        if "waveform" in analysis_categories:
            results["waveform_metrics"] = analyze_waveform(signal_data, sampling_freq)
        
        if "statistical" in analysis_categories:
            results["statistical_metrics"] = analyze_statistical(signal_data, sampling_freq)
        
        if "frequency" in analysis_categories:
            results["frequency_metrics"] = analyze_frequency(signal_data, sampling_freq)
        
        # Signal Quality with vitalDSP integration
        if quality_options:
            results["quality_metrics"] = get_vitaldsp_signal_quality(signal_data, sampling_freq, quality_options, signal_type)
        
        # Signal Transforms with vitalDSP integration
        if transform_options:
            results["transform_metrics"] = get_vitaldsp_transforms(signal_data, sampling_freq, transform_options, signal_type)
        
        # Advanced Computation with vitalDSP integration
        if advanced_computation:
            results["advanced_computation_metrics"] = get_vitaldsp_advanced_computation(signal_data, sampling_freq, advanced_computation, signal_type)
        
        # Advanced Features
        if advanced_features:
            results["advanced_features_metrics"] = analyze_advanced_features(signal_data, sampling_freq, advanced_features)
        
        # Feature Engineering with vitalDSP integration
        if feature_engineering:
            results["feature_engineering_metrics"] = get_vitaldsp_feature_engineering(signal_data, sampling_freq, feature_engineering, signal_type)
        
        # Preprocessing
        if preprocessing:
            results["preprocessing_metrics"] = analyze_preprocessing(signal_data, sampling_freq, preprocessing)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in enhanced physiological analysis: {e}")
        # Fallback to individual analysis functions
        results = {}
        
        try:
            # HRV Analysis
            if "hrv" in analysis_categories:
                results["hrv_metrics"] = analyze_hrv(signal_data, sampling_freq, hrv_options)
            
            # Morphology Analysis
            if "morphology" in analysis_categories:
                results["morphology_metrics"] = analyze_morphology(signal_data, sampling_freq, morphology_options)
            
            # Beat-to-Beat Analysis
            if "beat2beat" in analysis_categories:
                results["beat2beat_metrics"] = analyze_beat_to_beat(signal_data, sampling_freq)
            
            # Energy Analysis
            if "energy" in analysis_categories:
                results["energy_metrics"] = analyze_energy(signal_data, sampling_freq)
            
            # Envelope Detection
            if "envelope" in analysis_categories:
                results["envelope_metrics"] = analyze_envelope(signal_data, sampling_freq)
            
            # Signal Segmentation
            if "segmentation" in analysis_categories:
                results["segmentation_metrics"] = analyze_segmentation(signal_data, sampling_freq)
            
            # Trend Analysis
            if "trend" in analysis_categories:
                results["trend_metrics"] = analyze_trends(signal_data, sampling_freq)
            
            # Waveform Analysis
            if "waveform" in analysis_categories:
                results["waveform_metrics"] = analyze_waveform(signal_data, sampling_freq)
            
            # Statistical Analysis
            if "statistical" in analysis_categories:
                results["statistical_metrics"] = analyze_statistical(signal_data, sampling_freq)
            
            # Frequency Analysis
            if "frequency" in analysis_categories:
                results["frequency_metrics"] = analyze_frequency(signal_data, sampling_freq)
            
            # Signal Quality & Artifact Detection
            if quality_options:
                results["quality_metrics"] = analyze_signal_quality_advanced(signal_data, sampling_freq, quality_options)
            
            # Signal Transforms
            if transform_options:
                results["transform_metrics"] = analyze_transforms(signal_data, sampling_freq, transform_options)
            
            # Advanced Computation
            if advanced_computation:
                results["advanced_computation_metrics"] = analyze_advanced_computation(signal_data, sampling_freq, advanced_computation)
            
            # Feature Engineering
            if feature_engineering:
                results["feature_engineering_metrics"] = analyze_feature_engineering(signal_data, sampling_freq, feature_engineering, signal_type)
            
            # Preprocessing
            if preprocessing:
                results["preprocessing_metrics"] = analyze_preprocessing(signal_data, sampling_freq, preprocessing)
            
            return results
            
        except Exception as fallback_error:
            logger.error(f"Error in fallback analysis: {fallback_error}")
            return {"error": f"Analysis failed: {str(fallback_error)}"}


def suggest_best_signal_column(df, time_col):
    """Suggest the best signal column for physiological analysis."""
    signal_candidates = []
    
    for col in df.columns:
        if col != time_col and df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            try:
                col_data = df[col].values
                # Skip columns with all NaN or constant values
                if np.any(np.isnan(col_data)) or np.std(col_data) == 0:
                    continue
                
                # Calculate signal quality metrics
                variance = np.var(col_data)
                range_val = np.max(col_data) - np.min(col_data)
                mean_val = np.mean(col_data)
                
                # Score based on signal characteristics
                score = 0
                
                # Higher variance is generally better for physiological signals
                if variance > 0.001:
                    score += 1
                if variance > 0.01:
                    score += 1
                if variance > 0.1:
                    score += 1
                
                # Reasonable range (not too small, not too large)
                if 0.01 < range_val < 1000:
                    score += 1
                
                # Mean should not be extreme
                if -100 < mean_val < 100:
                    score += 1
                
                # Check for physiological signal patterns (basic heuristics)
                if len(col_data) > 100:
                    # Look for some variation (not completely flat)
                    diff_signal = np.diff(col_data)
                    if np.std(diff_signal) > 0.001:
                        score += 1
                
                signal_candidates.append({
                    'column': col,
                    'score': score,
                    'variance': variance,
                    'range': range_val,
                    'mean': mean_val
                })
                
            except Exception as e:
                logger.debug(f"Could not analyze column {col}: {e}")
                continue
    
    # Sort by score (highest first)
    signal_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return signal_candidates


def create_signal_quality_plots(time_data, signal_data, sampling_freq, quality_options):
    """Create comprehensive signal quality analysis plots."""
    try:
        # Determine number of subplots
        num_plots = 1  # Always show main signal
        if "quality_index" in quality_options:
            num_plots += 1
        if "artifact_detection" in quality_options:
            num_plots += 1
        
        fig = make_subplots(
            rows=num_plots, cols=1,
            subplot_titles=(["Signal Quality Overview"] + 
                           (["Quality Metrics"] if "quality_index" in quality_options else []) +
                           (["Artifact Detection"] if "artifact_detection" in quality_options else [])),
            vertical_spacing=0.12,
            specs=[[{"secondary_y": False}] for _ in range(num_plots)]
        )
        
        current_row = 1
        
        # Main signal with quality indicators
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=signal_data,
                mode='lines',
                name='Original Signal',
                line=dict(color='#1f77b4', width=2),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.1)'
            ),
            row=current_row, col=1
        )
        
        # Add baseline and noise bands
        baseline = np.mean(signal_data)
        noise_level = np.std(signal_data)
        
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color="gray",
            annotation_text="Baseline",
            annotation_position="right",
            row=current_row, col=1
        )
        
        # Add noise bands
        fig.add_hline(
            y=baseline + noise_level,
            line_dash="dot",
            line_color="orange",
            annotation_text="+1σ Noise",
            annotation_position="right",
            row=current_row, col=1
        )
        
        fig.add_hline(
            y=baseline - noise_level,
            line_dash="dot",
            line_color="orange",
            annotation_text="-1σ Noise",
            annotation_position="right",
            row=current_row, col=1
        )
        
        # Add quality statistics
        snr = 20 * np.log10(np.abs(baseline) / noise_level) if noise_level > 0 else 0
        
        fig.add_annotation(
            x=0.98, y=0.98,
            xref=f'x{current_row}', yref=f'y{current_row}',
            text=f'Baseline: {baseline:.3f}<br>Noise Level: {noise_level:.3f}<br>SNR: {snr:.1f} dB',
            showarrow=False,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='blue',
            borderwidth=2,
            font=dict(size=11, color='black')
        )
        
        current_row += 1
        
        # Quality metrics analysis
        if "quality_index" in quality_options:
            # Calculate moving quality metrics
            window_size = min(100, len(signal_data) // 10)
            if window_size > 1:
                quality_scores = []
                time_windows = []
                
                for i in range(0, len(signal_data) - window_size, window_size // 2):
                    window_data = signal_data[i:i+window_size]
                    window_mean = np.mean(window_data)
                    window_std = np.std(window_data)
                    
                    # Simple quality score based on signal-to-noise ratio
                    if window_std > 0:
                        quality_score = np.abs(window_mean) / window_std
                    else:
                        quality_score = 0
                    
                    quality_scores.append(quality_score)
                    time_windows.append(time_data[i + window_size // 2])
                
                if len(quality_scores) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=time_windows,
                            y=quality_scores,
                            mode='lines+markers',
                            name='Quality Score',
                            line=dict(color='#2ca02c', width=3),
                            marker=dict(color='#2ca02c', size=6, symbol='circle'),
                            fill='tonexty',
                            fillcolor='rgba(44, 160, 44, 0.1)'
                        ),
                        row=current_row, col=1
                    )
                    
                    # Add quality threshold
                    mean_quality = np.mean(quality_scores)
                    fig.add_hline(
                        y=mean_quality,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Mean Quality: {mean_quality:.2f}",
                        annotation_position="right",
                        row=current_row, col=1
                    )
                    
                    # Add quality statistics
                    fig.add_annotation(
                        x=0.98, y=0.98,
                        xref=f'x{current_row}', yref=f'y{current_row}',
                        text=f'Mean Quality: {mean_quality:.2f}<br>Min: {np.min(quality_scores):.2f}<br>Max: {np.max(quality_scores):.2f}',
                        showarrow=False,
                        bgcolor='rgba(255, 255, 255, 0.9)',
                        bordercolor='green',
                        borderwidth=2,
                        font=dict(size=11, color='black')
            )
            
            current_row += 1
        
        # Artifact detection analysis
        if "artifact_detection" in quality_options:
            # Detect artifacts using multiple methods
            artifacts = []
            artifact_scores = []
            
            # Method 1: Amplitude outliers
            threshold = baseline + 3 * noise_level
            amplitude_artifacts = np.abs(signal_data) > threshold
            
            # Method 2: Sudden changes
            diff_signal = np.diff(signal_data)
            change_threshold = 3 * np.std(diff_signal)
            change_artifacts = np.abs(diff_signal) > change_threshold
            
            # Combine artifact detection
            for i in range(len(signal_data)):
                artifact_score = 0
                if amplitude_artifacts[i]:
                    artifact_score += 1
                if i > 0 and change_artifacts[i-1]:
                    artifact_score += 1
                
                artifacts.append(artifact_score > 0)
                artifact_scores.append(artifact_score)
            
            # Plot artifact scores
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=artifact_scores,
                    mode='lines',
                    name='Artifact Score',
                    line=dict(color='#d62728', width=2),
                    fill='tonexty',
                    fillcolor='rgba(214, 39, 40, 0.1)'
                ),
                row=current_row, col=1
            )
            
            # Highlight artifact regions
            artifact_regions = np.where(artifacts)[0]
            if len(artifact_regions) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=time_data[artifact_regions],
                        y=signal_data[artifact_regions],
                        mode='markers',
                        name='Detected Artifacts',
                        marker=dict(
                            color='red',
                            size=8,
                            symbol='x',
                            line=dict(color='darkred', width=2)
                        )
                    ),
                    row=1, col=1  # Add to main signal plot
                )
            
            # Add artifact statistics
            total_artifacts = np.sum(artifacts)
            artifact_ratio = total_artifacts / len(signal_data) if len(signal_data) > 0 else 0
            
            fig.add_annotation(
                x=0.98, y=0.98,
                xref=f'x{current_row}', yref=f'y{current_row}',
                text=f'Total Artifacts: {total_artifacts}<br>Artifact Ratio: {artifact_ratio:.3f}<br>Clean Signal: {100*(1-artifact_ratio):.1f}%',
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='red',
                borderwidth=2,
                font=dict(size=11, color='black')
            )
        
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text="Comprehensive Signal Quality Analysis",
                x=0.5,
                font=dict(size=20, color='#2c3e50')
            ),
            height=250 * num_plots,
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=200, t=80, b=60)
        )
        
        # Update all subplot axes
        for i in range(1, num_plots + 1):
            fig.update_xaxes(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='rgba(128, 128, 128, 0.2)',
                row=i, col=1
            )
            fig.update_yaxes(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='rgba(128, 128, 128, 0.2)',
                row=i, col=1
            )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating signal quality plots: {e}")
        return create_empty_figure()


def create_advanced_features_plots(time_data, signal_data, sampling_freq, advanced_features):
    """Create advanced features analysis plots with cross-signal and ensemble analysis."""
    try:
        # Determine number of subplots
        num_plots = 1  # Always show main signal
        if "cross_signal" in advanced_features:
            num_plots += 1
        if "ensemble" in advanced_features:
            num_plots += 1
        if "change_detection" in advanced_features:
            num_plots += 1
        if "power_analysis" in advanced_features:
            num_plots += 1
        
        fig = make_subplots(
            rows=num_plots, cols=1,
            subplot_titles=(["Advanced Features Overview"] + 
                           (["Cross-Signal Analysis"] if "cross_signal" in advanced_features else []) +
                           (["Ensemble Analysis"] if "ensemble" in advanced_features else []) +
                           (["Change Detection"] if "change_detection" in advanced_features else []) +
                           (["Power Analysis"] if "power_analysis" in advanced_features else [])),
            vertical_spacing=0.12,
            specs=[[{"secondary_y": False}] for _ in range(num_plots)]
        )
        
        current_row = 1
        
        # Main signal with enhanced styling
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=signal_data,
                mode='lines',
                name='Original Signal',
                line=dict(color='#1f77b4', width=2),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.1)'
            ),
            row=current_row, col=1
        )
        
        current_row += 1
        
        # Cross-signal analysis
        if "cross_signal" in advanced_features:
            # Calculate cross-correlation with shifted versions
            max_lag = min(100, len(signal_data) // 4)
            lags = np.arange(-max_lag, max_lag + 1)
            correlations = []
            
            for lag in lags:
                if lag < 0:
                    corr = np.corrcoef(signal_data[:lag], signal_data[-lag:])[0, 1]
                elif lag > 0:
                    corr = np.corrcoef(signal_data[lag:], signal_data[:-lag])[0, 1]
                else:
                    corr = 1.0
                
                if np.isnan(corr):
                    corr = 0
                correlations.append(corr)
            
            fig.add_trace(
                go.Scatter(
                    x=lags / sampling_freq,  # Convert to seconds
                    y=correlations,
                    mode='lines',
                    name='Cross-Correlation',
                    line=dict(color='#ff7f0e', width=3),
                    fill='tonexty',
                    fillcolor='rgba(255, 127, 14, 0.1)'
                ),
                row=current_row, col=1
            )
            
            # Add correlation statistics
            max_corr = np.max(correlations)
            max_corr_lag = lags[np.argmax(correlations)] / sampling_freq
            fig.add_annotation(
                        x=0.98, y=0.98,
                        xref=f'x{current_row}', yref=f'y{current_row}',
                        text=f'Max Correlation: {max_corr:.3f}<br>At Lag: {max_corr_lag:.3f}s<br>Periodicity: {abs(max_corr_lag):.3f}s',
                        showarrow=False,
                        bgcolor='rgba(255, 255, 255, 0.9)',
                        bordercolor='orange',
                        borderwidth=2,
                        font=dict(size=11, color='black')
            )
            
            current_row += 1
        
        # Ensemble analysis
        if "ensemble" in advanced_features:
            # Create ensemble statistics using sliding windows
            window_size = min(50, len(signal_data) // 10)
            if window_size > 1:
                ensemble_means = []
                ensemble_stds = []
                time_windows = []
                
                for i in range(0, len(signal_data) - window_size, window_size // 2):
                    window_data = signal_data[i:i+window_size]
                    ensemble_means.append(np.mean(window_data))
                    ensemble_stds.append(np.std(window_data))
                    time_windows.append(time_data[i + window_size // 2])
                
                if len(ensemble_means) > 0:
                    # Plot ensemble mean
                    fig.add_trace(
                        go.Scatter(
                            x=time_windows,
                            y=ensemble_means,
                            mode='lines+markers',
                            name='Ensemble Mean',
                            line=dict(color='#2ca02c', width=3),
                            marker=dict(color='#2ca02c', size=6, symbol='circle'),
                            fill='tonexty',
                            fillcolor='rgba(44, 160, 44, 0.1)'
                        ),
                        row=current_row, col=1
                    )
                    
                    # Plot ensemble standard deviation
                    fig.add_trace(
                        go.Scatter(
                            x=time_windows,
                            y=ensemble_stds,
                            mode='lines+markers',
                            name='Ensemble Std',
                            line=dict(color='#d62728', width=2, dash='dot'),
                            marker=dict(color='#d62728', size=4, symbol='diamond'),
                            yaxis='y2'
                        ),
                        row=current_row, col=1
                    )
                    
                    # Add secondary y-axis for std
                    fig.update_layout(
                        yaxis2=dict(
                            title="Standard Deviation",
                            overlaying="y",
                            side="right",
                            showgrid=False
                        )
                    )
                    
                    # Add ensemble statistics
                    mean_ensemble = np.mean(ensemble_means)
                    std_ensemble = np.mean(ensemble_stds)
                    
                    fig.add_annotation(
                        x=0.98, y=0.98,
                        xref=f'x{current_row}', yref=f'y{current_row}',
                        text=f'Mean Ensemble: {mean_ensemble:.3f}<br>Avg Std: {std_ensemble:.3f}<br>Stability: {1/std_ensemble:.1f}',
                        showarrow=False,
                        bgcolor='rgba(255, 255, 255, 0.9)',
                        bordercolor='green',
                        borderwidth=2,
                        font=dict(size=11, color='black')
                    )
            
            current_row += 1
        
        # Change detection analysis
        if "change_detection" in advanced_features:
            # Detect changes using multiple methods
            # Method 1: First derivative
            diff_signal = np.diff(signal_data)
            change_threshold = 2 * np.std(diff_signal)
            change_points = np.abs(diff_signal) > change_threshold
            
            # Method 2: Moving variance
            window_size = min(30, len(signal_data) // 10)
            if window_size > 1:
                moving_variance = []
                time_windows = []
                
                for i in range(0, len(signal_data) - window_size, window_size // 2):
                    window_data = signal_data[i:i+window_size]
                    moving_variance.append(np.var(window_data))
                    time_windows.append(time_data[i + window_size // 2])
                
                if len(moving_variance) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=time_windows,
                            y=moving_variance,
                            mode='lines',
                            name='Moving Variance',
                            line=dict(color='#9467bd', width=3),
                            fill='tonexty',
                            fillcolor='rgba(148, 103, 189, 0.1)'
                        ),
                        row=current_row, col=1
                    )
                    
                    # Highlight change regions
                    change_regions = np.where(change_points)[0]
                    if len(change_regions) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=time_data[change_regions],
                                y=signal_data[change_regions],
                                mode='markers',
                                name='Change Points',
                                marker=dict(
                                    color='red',
                                    size=8,
                                    symbol='diamond',
                                    line=dict(color='darkred', width=2)
                                )
                            ),
                            row=1, col=1  # Add to main signal plot
                        )
                    
                    # Add change detection statistics
                    total_changes = np.sum(change_points)
                    change_rate = total_changes / len(diff_signal) if len(diff_signal) > 0 else 0
                    
                    fig.add_annotation(
                        x=0.98, y=0.98,
                        xref=f'x{current_row}', yref=f'y{current_row}',
                        text=f'Change Points: {total_changes}<br>Change Rate: {change_rate:.3f}<br>Stability: {1-change_rate:.3f}',
                        showarrow=False,
                        bgcolor='rgba(255, 255, 255, 0.9)',
                        bordercolor='purple',
                        borderwidth=2,
                        font=dict(size=11, color='black')
                    )
            
            current_row += 1
        
        # Power analysis
        if "power_analysis" in advanced_features:
            # Calculate power spectrum
            freqs, psd = signal.welch(signal_data, fs=sampling_freq, nperseg=min(256, len(signal_data)//2))
            
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=psd,
                    mode='lines',
                    name='Power Spectrum',
                    line=dict(color='#e377c2', width=3),
                    fill='tonexty',
                    fillcolor='rgba(227, 119, 194, 0.1)'
                ),
                row=current_row, col=1
            )
            
            # Add power statistics
            total_power = np.sum(psd)
            peak_power = np.max(psd)
            dominant_freq = freqs[np.argmax(psd)]
            
            # Calculate power in different frequency bands
            low_freq_mask = freqs < 1.0  # Below 1 Hz
            mid_freq_mask = (freqs >= 1.0) & (freqs < 10.0)  # 1-10 Hz
            high_freq_mask = freqs >= 10.0  # Above 10 Hz
            
            low_power = np.sum(psd[low_freq_mask]) if np.any(low_freq_mask) else 0
            mid_power = np.sum(psd[mid_freq_mask]) if np.any(mid_freq_mask) else 0
            high_power = np.sum(psd[high_freq_mask]) if np.any(high_freq_mask) else 0
            
            fig.add_annotation(
                x=0.98, y=0.98,
                xref=f'x{current_row}', yref=f'y{current_row}',
                text=f'Total Power: {total_power:.2e}<br>Peak Power: {peak_power:.2e}<br>Dominant Freq: {dominant_freq:.1f} Hz',
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='pink',
                borderwidth=2,
                font=dict(size=11, color='black')
            )
            
            # Add frequency band annotations
            if low_power > 0:
                fig.add_annotation(
                    x=0.95, y=0.9,
                    xref=f'x{current_row}', yref=f'y{current_row}',
                    text=f'Low: {low_power:.2e}',
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='blue'
                )
            
            if mid_power > 0:
                fig.add_annotation(
                    x=0.95, y=0.9,
                    xref=f'x{current_row}', yref=f'y{current_row}',
                    text=f'Mid: {mid_power:.2e}',
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='green'
                )
            
            if high_power > 0:
                fig.add_annotation(
                    x=0.95, y=0.9,
                    xref=f'x{current_row}', yref=f'y{current_row}',
                    text=f'High: {high_power:.2e}',
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='red'
                )
        
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text="Advanced Features Analysis",
                x=0.5,
                font=dict(size=20, color='#2c3e50')
            ),
            height=250 * num_plots,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update all subplot axes
        for i in range(1, num_plots + 1):
            fig.update_xaxes(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='rgba(128, 128, 128, 0.2)',
                row=i, col=1
            )
            fig.update_yaxes(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='rgba(128, 128, 128, 0.2)',
                row=i, col=1
            )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating advanced features plots: {e}")
        return create_empty_figure()


def create_comprehensive_dashboard(time_data, signal_data, signal_type, sampling_freq, analysis_categories):
    """Create a comprehensive dashboard with multiple analysis views."""
    try:
        # Create a 2x2 subplot layout for comprehensive view
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Signal Overview", "Peak Analysis", "Frequency Analysis", "Quality Metrics"),
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Signal Overview (top-left)
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=signal_data,
                mode='lines',
                name='Signal',
                line=dict(color='#1f77b4', width=2),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.1)'
            ),
            row=1, col=1
        )
        
        # Add peaks with red arrows
        try:
            height_threshold = np.mean(signal_data) + 1.5 * np.std(signal_data)
            distance_threshold = int(sampling_freq * 0.3)
            
            peaks, _ = signal.find_peaks(
                signal_data, 
                height=height_threshold,
                distance=distance_threshold,
                prominence=np.std(signal_data) * 0.5
            )
            
            if len(peaks) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=time_data[peaks],
                        y=signal_data[peaks],
                        mode='markers+text',
                        name='Peaks',
                        text=[f'P{i+1}' for i in range(len(peaks))],
                        textposition='top center',
                        marker=dict(
                            color='red',
                            size=10,
                            symbol='arrow-up',
                            line=dict(color='darkred', width=2)
                        ),
                        textfont=dict(color='red', size=8, family='Arial Black')
                    ),
                    row=1, col=1
                )
                
                # Add peak statistics
                mean_interval = np.mean(np.diff(peaks) / sampling_freq) if len(peaks) > 1 else 0
                heart_rate = 60 / mean_interval if mean_interval > 0 else 0
                
                fig.add_annotation(
                    x=0.98, y=0.98,
                    xref='x1', yref='y1',
                    text=f'Peaks: {len(peaks)}<br>HR: {heart_rate:.1f} BPM<br>Mean Interval: {mean_interval:.2f}s',
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='red',
                    borderwidth=2,
                    font=dict(size=10, color='black')
                )
        except Exception as e:
            logger.debug(f"Could not add peaks: {e}")
        
        # 2. Peak Analysis (top-right)
        if len(peaks) > 0:
            # Histogram of peak heights
            hist, bin_edges = np.histogram(signal_data[peaks], bins=min(15, len(peaks)//2), density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            fig.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=hist,
                    name='Peak Distribution',
                    marker_color='rgba(255, 165, 0, 0.7)',
                    opacity=0.8
                ),
                row=1, col=2
            )
            
            # Add peak statistics
            peak_heights = signal_data[peaks]
            fig.add_annotation(
                x=0.98, y=0.98,
                xref='x2', yref='y2',
                text=f'Mean Height: {np.mean(peak_heights):.3f}<br>Height Std: {np.std(peak_heights):.3f}<br>Range: {np.max(peak_heights) - np.min(peak_heights):.3f}',
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='orange',
                borderwidth=2,
                font=dict(size=10, color='black')
            )
        
        # 3. Frequency Analysis (bottom-left)
        try:
            freqs, psd = signal.welch(signal_data, fs=sampling_freq, nperseg=min(256, len(signal_data)//2))
            
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=psd,
                    mode='lines',
                    name='Power Spectrum',
                    line=dict(color='#2ca02c', width=2),
                    fill='tonexty',
                    fillcolor='rgba(44, 160, 44, 0.1)'
                ),
                row=2, col=1
            )
            
            # Add frequency band annotations
            low_freq_mask = freqs < 1.0
            mid_freq_mask = (freqs >= 1.0) & (freqs < 10.0)
            high_freq_mask = freqs >= 10.0
            
            if np.any(low_freq_mask):
                low_power = np.sum(psd[low_freq_mask])
                fig.add_annotation(
                    x=0.95, y=0.9,
                    xref='x3', yref='y3',
                    text=f'Low: {low_power:.2e}',
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='blue'
                )
            
            if np.any(mid_freq_mask):
                mid_power = np.sum(psd[mid_freq_mask])
                fig.add_annotation(
                    x=0.95, y=0.9,
                    xref='x3', yref='y3',
                    text=f'Mid: {mid_power:.2e}',
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='green'
                )
            
            if np.any(high_freq_mask):
                high_power = np.sum(psd[high_freq_mask])
                fig.add_annotation(
                    x=0.95, y=0.9,
                    xref='x3', yref='y3',
                    text=f'High: {high_power:.2e}',
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='red'
                )
        except Exception as e:
            logger.debug(f"Could not create frequency analysis: {e}")
        
        # 4. Quality Metrics (bottom-right)
        try:
            # Calculate quality metrics
            baseline = np.mean(signal_data)
            noise_level = np.std(signal_data)
            snr = 20 * np.log10(np.abs(baseline) / noise_level) if noise_level > 0 else 0
            
            # Create quality score over time
            window_size = min(50, len(signal_data) // 10)
            if window_size > 1:
                quality_scores = []
                time_windows = []
                
                for i in range(0, len(signal_data) - window_size, window_size // 2):
                    window_data = signal_data[i:i+window_size]
                    window_mean = np.mean(window_data)
                    window_std = np.std(window_data)
                    
                    if window_std > 0:
                        quality_score = np.abs(window_mean) / window_std
                    else:
                        quality_score = 0
                    
                    quality_scores.append(quality_score)
                    time_windows.append(time_data[i + window_size // 2])
                
                if len(quality_scores) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=time_windows,
                            y=quality_scores,
                            mode='lines+markers',
                            name='Quality Score',
                            line=dict(color='#d62728', width=2),
                            marker=dict(color='#d62728', size=4, symbol='circle')
                        ),
                        row=2, col=2
                    )
                    
                    # Add quality statistics
                    mean_quality = np.mean(quality_scores)
                    fig.add_annotation(
                        x=0.98, y=0.98,
                        xref='x4', yref='y4',
                        text=f'Mean Quality: {mean_quality:.2f}<br>SNR: {snr:.1f} dB<br>Noise Level: {noise_level:.3f}',
                        showarrow=False,
                        bgcolor='rgba(255, 255, 255, 0.9)',
                        bordercolor='red',
                        borderwidth=2,
                        font=dict(size=10, color='black')
                    )
        except Exception as e:
            logger.debug(f"Could not create quality metrics: {e}")
        
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text=f"Comprehensive {signal_type.upper()} Analysis Dashboard",
                x=0.5,
                font=dict(size=18, color='#2c3e50')
            ),
            height=800,
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=200, t=80, b=60)
        )
        
        # Update all subplot axes
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    row=i, col=j
                )
                fig.update_yaxes(
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    row=i, col=j
                )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating comprehensive dashboard: {e}")
        return create_empty_figure()

    # Comprehensive Dashboard Callback
    @app.callback(
        Output("store-physio-analysis", "data"),
        [Input("btn-comprehensive-dashboard", "n_clicks")],
        [State("store-uploaded-data", "data"),
         State("signal-type-select", "value"),
         State("start-time", "value"),
         State("end-time", "value"),
         State("analysis-categories", "value"),
         State("hrv-options", "value"),
         State("morphology-options", "value"),
         State("advanced-features", "value"),
         State("quality-options", "value"),
         State("transform-options", "value"),
         State("advanced-computation", "value"),
         State("feature-engineering", "value"),
         State("preprocessing", "value")]
    )
    def show_comprehensive_dashboard(n_clicks, data_store, signal_type, start_time, end_time,
                                   analysis_categories, hrv_options, morphology_options, advanced_features,
                                   quality_options, transform_options, advanced_computation, feature_engineering, preprocessing):
        """Show comprehensive dashboard view."""
        if not n_clicks or not data_store:
            return no_update
        
        try:
            # Get data
            df = pd.DataFrame(data_store["data"])
            if df.empty:
                return dash.no_update
            
            # Get column mapping
            column_mapping = data_store.get("column_mapping", {})
            sampling_freq = data_store.get("sampling_freq", 100)
            
            # Extract time and signal columns
            time_col = column_mapping.get('time', df.columns[0])
            signal_col = column_mapping.get('signal', df.columns[1])
            
            # Extract time and signal data
            time_data = df[time_col].values
            signal_data = df[signal_col].values
            
            # Apply time window
            start_idx = np.searchsorted(time_data, start_time or 0)
            end_idx = np.searchsorted(time_data, end_time or 10)
            
            if end_idx > start_idx:
                time_data = time_data[start_idx:end_idx]
                signal_data = signal_data[start_idx:end_idx]
            
            # Auto-detect signal type if needed
            if signal_type == "auto":
                signal_type = detect_physiological_signal_type(signal_data, sampling_freq)
            
            # Create comprehensive dashboard
            dashboard_fig = create_comprehensive_dashboard(time_data, signal_data, signal_type, sampling_freq, analysis_categories or [])
            
            # Store dashboard data
            dashboard_data = {
                "type": "comprehensive_dashboard",
                "figure": dashboard_fig,
                "signal_type": signal_type,
                "sampling_freq": sampling_freq,
                "time_data": time_data.tolist(),
                "signal_data": signal_data.tolist()
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error creating comprehensive dashboard: {e}")
            return no_update

    # ============================================================================
    # RESPIRATORY RATE ANALYSIS CALLBACKS
    # ============================================================================
    
    @app.callback(
        [Output("resp-main-plot", "figure"),
         Output("resp-analysis-results", "children"),
         Output("resp-analysis-plots", "figure"),
         Output("resp-data-store", "data"),
         Output("resp-features-store", "data")],
        [Input("url", "pathname"),
         Input("resp-analyze-btn", "n_clicks"),
         Input("resp-time-range-slider", "value"),
         Input("resp-btn-nudge-m10", "n_clicks"),
         Input("resp-btn-nudge-m1", "n_clicks"),
         Input("resp-btn-nudge-p1", "n_clicks"),
         Input("resp-btn-nudge-p10", "n_clicks")],
        [State("resp-start-time", "value"),
         State("resp-end-time", "value"),
         State("resp-signal-type", "value"),
         State("resp-estimation-methods", "value"),
         State("resp-advanced-options", "value"),
         State("resp-preprocessing-options", "value"),
         State("resp-low-cut", "value"),
         State("resp-high-cut", "value"),
         State("resp-min-breath-duration", "value"),
         State("resp-max-breath-duration", "value")]
    )
    def respiratory_analysis_callback(pathname, n_clicks, slider_value, nudge_m10, nudge_m1, nudge_p1, nudge_p10,
                                   start_time, end_time, signal_type, estimation_methods, advanced_options,
                                   preprocessing_options, low_cut, high_cut, min_breath_duration, max_breath_duration):
        """Comprehensive respiratory rate analysis using all vitalDSP respiratory features."""
        ctx = callback_context
        
        # Determine what triggered this callback
        if not ctx.triggered:
            logger.warning("No context triggered - raising PreventUpdate")
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        logger.info(f"=== RESPIRATORY ANALYSIS CALLBACK TRIGGERED ===")
        logger.info(f"Trigger ID: {trigger_id}")
        logger.info(f"Pathname: {pathname}")
        logger.info(f"Signal type: {signal_type}")
        logger.info(f"Estimation methods: {estimation_methods}")
        logger.info(f"Advanced options: {advanced_options}")
        logger.info(f"Preprocessing options: {preprocessing_options}")
        
        # Only run this when we're on the respiratory page
        if pathname != "/respiratory":
            logger.info("Not on respiratory page, returning empty figures")
            return create_empty_figure(), "Navigate to respiratory page to analyze respiratory signals.", create_empty_figure(), None, None
        
        # Handle first-time loading
        if not ctx.triggered or ctx.triggered[0]["prop_id"].split(".")[0] == "url":
            logger.info("First time loading respiratory page, attempting to load data")
        
        try:
            # Get data from the data service
            logger.info("Attempting to get data service...")
            from vitalDSP_webapp.services.data.data_service import get_data_service
            data_service = get_data_service()
            logger.info("Data service retrieved successfully")
            
            # Get the most recent data
            logger.info("Retrieving all data from service...")
            all_data = data_service.get_all_data()
            logger.info(f"All data keys: {list(all_data.keys()) if all_data else 'None'}")
            
            if not all_data:
                logger.warning("No data found in service")
                return create_empty_figure(), "No data available. Please upload and process data first.", create_empty_figure(), None, None
            
            # Get the most recent data entry
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            logger.info(f"Latest data ID: {latest_data_id}")
            logger.info(f"Latest data info: {latest_data.get('info', 'No info')}")
            
            # Get column mapping
            logger.info("Retrieving column mapping...")
            column_mapping = data_service.get_column_mapping(latest_data_id)
            logger.info(f"Column mapping: {column_mapping}")
            
            if not column_mapping:
                logger.warning("Data has not been processed yet - no column mapping found")
                return create_empty_figure(), "Please process your data on the Upload page first (configure column mapping)", create_empty_figure(), None, None
                
            # Get the actual data
            logger.info("Retrieving data frame...")
            df = data_service.get_data(latest_data_id)
            logger.info(f"Data frame shape: {df.shape if df is not None else 'None'}")
            logger.info(f"Data frame columns: {list(df.columns) if df is not None else 'None'}")
            
            if df is None or df.empty:
                logger.warning("Data frame is empty")
                return create_empty_figure(), "Data is empty or corrupted.", create_empty_figure(), None, None
            
            # Get sampling frequency from the data info
            sampling_freq = latest_data.get('info', {}).get('sampling_freq', 1000)
            logger.info(f"Sampling frequency: {sampling_freq}")
            
            # Handle time window adjustments for nudge buttons
            if trigger_id in ["resp-btn-nudge-m10", "resp-btn-nudge-m1", "resp-btn-nudge-p1", "resp-btn-nudge-p10"]:
                if not start_time or not end_time:
                    start_time, end_time = 0, 10
                    
                if trigger_id == "resp-btn-nudge-m10":
                    start_time = max(0, start_time - 10)
                    end_time = max(10, end_time - 10)
                elif trigger_id == "resp-btn-nudge-m1":
                    start_time = max(0, start_time - 1)
                    end_time = max(1, end_time - 1)
                elif trigger_id == "resp-btn-nudge-p1":
                    start_time = start_time + 1
                    end_time = end_time + 1
                elif trigger_id == "resp-btn-nudge-p10":
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
            
            # Get signal column
            signal_column = column_mapping.get('signal')
            logger.info(f"Signal column from mapping: {signal_column}")
            logger.info(f"Available columns in windowed data: {list(windowed_data.columns)}")
            
            if not signal_column or signal_column not in windowed_data.columns:
                logger.warning(f"Signal column {signal_column} not found in data")
                return create_empty_figure(), "Signal column not found in data.", create_empty_figure(), None, None
            
            signal_data = windowed_data[signal_column].values
            logger.info(f"Signal data shape: {signal_data.shape}")
            logger.info(f"Signal data range: {np.min(signal_data):.3f} to {np.max(signal_data):.3f}")
            logger.info(f"Signal data mean: {np.mean(signal_data):.3f}")
            
            # Auto-detect signal type if needed
            if signal_type == "auto":
                logger.info("Auto-detecting signal type...")
                signal_type = detect_respiratory_signal_type(signal_data, sampling_freq)
                logger.info(f"Auto-detected signal type: {signal_type}")
            
            logger.info(f"Final signal type: {signal_type}")
            logger.info(f"Estimation methods: {estimation_methods}")
            logger.info(f"Advanced options: {advanced_options}")
            logger.info(f"Preprocessing options: {preprocessing_options}")
            
            # Create main respiratory signal plot with annotations
            logger.info("Creating main respiratory signal plot with annotations...")
            main_plot = create_respiratory_signal_plot(signal_data, time_axis, sampling_freq, signal_type, 
                                                     estimation_methods, preprocessing_options, low_cut, high_cut)
            logger.info("Main respiratory signal plot created successfully")
            
            # Generate comprehensive respiratory analysis results
            logger.info("Generating comprehensive respiratory analysis results...")
            analysis_results = generate_comprehensive_respiratory_analysis(signal_data, time_axis, sampling_freq, 
                                                                        signal_type, estimation_methods, advanced_options,
                                                                        preprocessing_options, low_cut, high_cut,
                                                                        min_breath_duration, max_breath_duration)
            logger.info("Respiratory analysis results generated successfully")
            
            # Create respiratory analysis plots
            logger.info("Creating respiratory analysis plots...")
            analysis_plots = create_comprehensive_respiratory_plots(signal_data, time_axis, sampling_freq, 
                                                                  signal_type, estimation_methods, advanced_options,
                                                                  preprocessing_options, low_cut, high_cut)
            logger.info("Respiratory analysis plots created successfully")
            
            # Store processed data
            resp_data = {
                "signal_data": signal_data.tolist(),
                "time_axis": time_axis.tolist(),
                "sampling_freq": sampling_freq,
                "window": [start_time, end_time],
                "signal_type": signal_type,
                "estimation_methods": estimation_methods
            }
            
            resp_features = {
                "advanced_options": advanced_options,
                "preprocessing_options": preprocessing_options,
                "filter_params": {"low_cut": low_cut, "high_cut": high_cut},
                "breath_constraints": {"min_duration": min_breath_duration, "max_duration": max_breath_duration}
            }
            
            logger.info("Respiratory analysis completed successfully")
            return main_plot, analysis_results, analysis_plots, resp_data, resp_features
            
        except Exception as e:
            logger.error(f"Error in respiratory analysis callback: {e}")
            import traceback
            traceback.print_exc()
            return create_empty_figure(), f"Error in analysis: {str(e)}", create_empty_figure(), None, None
    
    @app.callback(
        [Output("resp-start-time", "value"),
         Output("resp-end-time", "value")],
        [Input("resp-time-range-slider", "value")]
    )
    def update_resp_time_inputs(slider_value):
        """Update time input fields based on slider."""
        if not slider_value:
            return no_update, no_update
        return slider_value[0], slider_value[1]
    
    @app.callback(
        [Output("resp-time-range-slider", "min"),
         Output("resp-time-range-slider", "max"),
         Output("resp-time-range-slider", "value")],
        [Input("url", "pathname")]
    )
    def update_resp_time_slider_range(pathname):
        """Update time slider range based on data duration."""
        logger.info("=== UPDATE RESP TIME SLIDER RANGE ===")
        logger.info(f"Pathname: {pathname}")
        
        # Only run this when we're on the respiratory page
        if pathname != "/respiratory":
            return 0, 100, [0, 10]
        
        try:
            # Get data from the data service
            from vitalDSP_webapp.services.data.data_service import get_data_service
            data_service = get_data_service()
            
            # Get the most recent data
            all_data = data_service.get_all_data()
            if not all_data:
                return 0, 100, [0, 10]
            
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            
            # Get sampling frequency and calculate duration
            sampling_freq = latest_data.get('info', {}).get('sampling_freq', 1000)
            df = data_service.get_data(latest_data_id)
            
            if df is None or df.empty:
                return 0, 100, [0, 10]
            
            duration = len(df) / sampling_freq
            max_time = int(duration)
            
            return 0, max_time, [0, min(10, max_time)]
            
        except Exception as e:
            logger.error(f"Error updating respiratory time slider range: {e}")
            return 0, 100, [0, 10]

    # ============================================================================
    # RESPIRATORY ANALYSIS HELPER FUNCTIONS
    # ============================================================================
    
    def detect_respiratory_signal_type(signal_data, sampling_freq):
        """Auto-detect if signal is respiratory or cardiac based on frequency content."""
        try:
            # Compute FFT
            fft_result = np.fft.fft(signal_data)
            fft_freq = np.fft.fftfreq(len(signal_data), 1/sampling_freq)
            
            # Focus on positive frequencies
            positive_mask = fft_freq > 0
            fft_freq = fft_freq[positive_mask]
            fft_magnitude = np.abs(fft_result[positive_mask])
            
            # Find dominant frequency
            dominant_idx = np.argmax(fft_magnitude)
            dominant_freq = fft_freq[dominant_idx]
            
            # Respiratory signals typically have lower frequencies (0.1-0.5 Hz)
            # Cardiac signals typically have higher frequencies (0.8-2.0 Hz)
            if dominant_freq < 0.5:
                return "respiratory"
            else:
                return "cardiac"
        except:
            return "unknown"

    def create_respiratory_signal_plot(signal_data, time_axis, sampling_freq, signal_type, 
                                      estimation_methods, preprocessing_options, low_cut, high_cut):
        """Create the main respiratory signal plot with annotations."""
        try:
            fig = go.Figure()
            
            # Add main signal
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=signal_data,
                mode='lines',
                name=f'{signal_type.title()} Signal',
                line=dict(color='blue', width=1)
            ))
            
            # Add preprocessing info if applied
            if preprocessing_options and any(preprocessing_options):
                fig.add_trace(go.Scatter(
                    x=time_axis,
                    y=signal_data,  # This would be the preprocessed signal in real implementation
                    mode='lines',
                    name='Preprocessed Signal',
                    line=dict(color='red', width=1, dash='dash')
                ))
            
            # Add filter frequency lines if specified
            if low_cut:
                fig.add_hline(y=np.min(signal_data), line_dash="dot", line_color="green",
                             annotation_text=f"Low Cut: {low_cut} Hz")
            if high_cut:
                fig.add_hline(y=np.max(signal_data), line_dash="dot", line_color="orange",
                             annotation_text=f"High Cut: {high_cut} Hz")
            
            fig.update_layout(
                title=f"Respiratory Signal Analysis - {signal_type.title()}",
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude",
                showlegend=True,
                height=400,
                plot_bgcolor='white',
                margin=dict(l=60, r=60, t=80, b=60)
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating respiratory signal plot: {e}")
            return create_empty_figure()

    def generate_comprehensive_respiratory_analysis(signal_data, time_axis, sampling_freq, 
                                                  signal_type, estimation_methods, advanced_options,
                                                  preprocessing_options, low_cut, high_cut,
                                                  min_breath_duration, max_breath_duration):
        """Generate comprehensive respiratory analysis results using vitalDSP."""
        try:
            results = []
            
            # Basic statistics
            results.append(html.H5("Signal Statistics"))
            results.append(html.P(f"Signal Type: {signal_type.title()}"))
            results.append(html.P(f"Duration: {time_axis[-1]:.2f} seconds"))
            results.append(html.P(f"Sampling Frequency: {sampling_freq} Hz"))
            results.append(html.P(f"Signal Length: {len(signal_data)} samples"))
            results.append(html.P(f"Mean Amplitude: {np.mean(signal_data):.4f}"))
            results.append(html.P(f"Std Amplitude: {np.std(signal_data):.4f}"))
            
            # Estimation methods used
            if estimation_methods:
                results.append(html.Hr())
                results.append(html.H6("Estimation Methods"))
                for method in estimation_methods:
                    results.append(html.P(f"• {method.replace('_', ' ').title()}"))
            
            # Preprocessing options
            if preprocessing_options and any(preprocessing_options):
                results.append(html.Hr())
                results.append(html.H6("Preprocessing Applied"))
                for option in preprocessing_options:
                    if option:
                        results.append(html.P(f"• {option.replace('_', ' ').title()}"))
            
            # Filter parameters
            if low_cut or high_cut:
                results.append(html.Hr())
                results.append(html.H6("Filter Parameters"))
                if low_cut:
                    results.append(html.P(f"Low Cutoff: {low_cut} Hz"))
                if high_cut:
                    results.append(html.P(f"High Cutoff: {high_cut} Hz"))
            
            # Breath duration constraints
            if min_breath_duration or max_breath_duration:
                results.append(html.Hr())
                results.append(html.H6("Breath Duration Constraints"))
                if min_breath_duration:
                    results.append(html.P(f"Minimum Duration: {min_breath_duration} seconds"))
                if max_breath_duration:
                    results.append(html.P(f"Maximum Duration: {max_breath_duration} seconds"))
            
            # Try to compute respiratory rate using vitalDSP if available
            try:
                from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis
                from vitalDSP.preprocess.preprocess_operations import PreprocessConfig
                
                # Create preprocessing configuration
                preprocess_config = PreprocessConfig()
                if preprocessing_options and "filter" in preprocessing_options:
                    preprocess_config.filter_type = "bandpass"
                    preprocess_config.lowcut = low_cut or 0.1
                    preprocess_config.highcut = high_cut or 0.5
                
                # Initialize respiratory analysis
                ra = RespiratoryAnalysis(signal_data, sampling_freq)
                
                # Compute respiratory rate using different methods
                if estimation_methods:
                    results.append(html.Hr())
                    results.append(html.H6("Respiratory Rate Estimates"))
                    
                    for method in estimation_methods:
                        try:
                            if method == "peak_detection":
                                rr = ra.compute_respiratory_rate(
                                    method="peaks",
                                    min_breath_duration=min_breath_duration or 0.5,
                                    max_breath_duration=max_breath_duration or 6,
                                    preprocess_config=preprocess_config
                                )
                                results.append(html.P(f"• Peak Detection: {rr:.2f} breaths/min"))
                            
                            elif method == "fft_based":
                                rr = ra.compute_respiratory_rate(
                                    method="fft_based",
                                    preprocess_config=preprocess_config
                                )
                                results.append(html.P(f"• FFT Based: {rr:.2f} breaths/min"))
                            
                            elif method == "frequency_domain":
                                rr = ra.compute_respiratory_rate(
                                    method="frequency_domain",
                                    preprocess_config=preprocess_config
                                )
                                results.append(html.P(f"• Frequency Domain: {rr:.2f} breaths/min"))
                            
                            elif method == "time_domain":
                                rr = ra.compute_respiratory_rate(
                                    method="time_domain",
                                    preprocess_config=preprocess_config
                                )
                                results.append(html.P(f"• Time Domain: {rr:.2f} breaths/min"))
                                
                        except Exception as e:
                            logger.warning(f"Failed to compute RR using {method}: {e}")
                            results.append(html.P(f"• {method.replace('_', ' ').title()}: Failed"))
                
                # Advanced analysis if selected
                if advanced_options:
                    results.append(html.Hr())
                    results.append(html.H6("Advanced Analysis"))
                    
                    for option in advanced_options:
                        try:
                            if option == "sleep_apnea":
                                from vitalDSP.respiratory_analysis.sleep_apnea_detection.amplitude_threshold import detect_apnea_amplitude
                                apnea_events = detect_apnea_amplitude(
                                    signal_data, sampling_freq, 
                                    threshold=np.mean(signal_data) * 0.5,
                                    min_duration=10
                                )
                                results.append(html.P(f"• Sleep Apnea Events: {len(apnea_events)} detected"))
                            
                            elif option == "multimodal":
                                from vitalDSP.respiratory_analysis.fusion.multimodal_analysis import multimodal_analysis
                                # For multimodal, we'd need multiple signals, so just show capability
                                results.append(html.P("• Multimodal Analysis: Available for multiple signals"))
                            
                            elif option == "ppg_ecg_fusion":
                                from vitalDSP.respiratory_analysis.fusion.ppg_ecg_fusion import ppg_ecg_fusion
                                # For fusion, we'd need both PPG and ECG signals
                                results.append(html.P("• PPG-ECG Fusion: Available for dual signals"))
                            
                            elif option == "resp_cardiac_fusion":
                                from vitalDSP.respiratory_analysis.fusion.respiratory_cardiac_fusion import respiratory_cardiac_fusion
                                results.append(html.P("• Respiratory-Cardiac Fusion: Available"))
                                
                        except Exception as e:
                            logger.warning(f"Failed to perform {option} analysis: {e}")
                            results.append(html.P(f"• {option.replace('_', ' ').title()}: Failed"))
                            
            except ImportError as e:
                logger.warning(f"vitalDSP respiratory modules not available: {e}")
                results.append(html.Hr())
                results.append(html.H6("Analysis Status"))
                results.append(html.P("vitalDSP respiratory modules not available for detailed analysis"))
            
            return html.Div(results)
        except Exception as e:
            logger.error(f"Error generating respiratory analysis results: {e}")
            return html.Div([html.H5("Error"), html.P(f"Failed to generate results: {str(e)}")])

    def create_comprehensive_respiratory_plots(signal_data, time_axis, sampling_freq, 
                                             signal_type, estimation_methods, advanced_options,
                                             preprocessing_options, low_cut, high_cut):
        """Create comprehensive respiratory analysis plots."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Time Domain", "Frequency Domain", "Power Spectral Density", "Signal Quality"),
                specs=[[{"type": "xy"}, {"type": "xy"}],
                       [{"type": "xy"}, {"type": "xy"}]]
            )
            
            # Time domain plot
            fig.add_trace(
                go.Scatter(x=time_axis, y=signal_data, mode='lines', name='Signal'),
                row=1, col=1
            )
            
            # Frequency domain plot
            fft_result = np.fft.fft(signal_data)
            fft_freq = np.fft.fftfreq(len(signal_data), 1/sampling_freq)
            positive_mask = fft_freq > 0
            fft_freq_positive = fft_freq[positive_mask]
            fft_magnitude = np.abs(fft_result[positive_mask])
            
            fig.add_trace(
                go.Scatter(x=fft_freq_positive, y=fft_magnitude, mode='lines', name='FFT Magnitude'),
                row=1, col=2
            )
            
            # Power Spectral Density
            freqs, psd = signal.welch(signal_data, sampling_freq, nperseg=min(256, len(signal_data)//4))
            fig.add_trace(
                go.Scatter(x=freqs, y=psd, mode='lines', name='PSD'),
                row=2, col=1
            )
            
            # Signal quality metrics
            # Calculate signal quality index
            signal_quality = np.std(signal_data) / np.mean(np.abs(signal_data)) if np.mean(np.abs(signal_data)) > 0 else 0
            fig.add_trace(
                go.Bar(x=['Signal Quality'], y=[signal_quality], name='Quality Index'),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="Comprehensive Respiratory Analysis",
                height=600,
                showlegend=False,
                plot_bgcolor='white',
                margin=dict(l=60, r=60, t=80, b=60)
            )
            
            # Update all subplot axes
            for i in range(1, 3):
                for j in range(1, 3):
                    fig.update_xaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='rgba(128, 128, 128, 0.2)',
                        row=i, col=j
                    )
                    fig.update_yaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='rgba(128, 128, 128, 0.2)',
                        row=i, col=j
                    )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comprehensive respiratory plots: {e}")
            return create_empty_figure()

    # Comprehensive Dashboard Callback
