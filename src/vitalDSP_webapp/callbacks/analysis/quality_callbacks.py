"""
Signal quality assessment callbacks for vitalDSP webapp.

This module handles comprehensive signal quality assessment including SNR, artifact detection, baseline wander, and other quality metrics.
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


def register_quality_callbacks(app):
    """Register all signal quality assessment callbacks."""
    logger.info("=== REGISTERING SIGNAL QUALITY CALLBACKS ===")
    
    # Import vitalDSP modules
    _import_vitaldsp_modules()
    
    @app.callback(
        [Output("quality-main-plot", "figure"),
         Output("quality-metrics-plot", "figure"),
         Output("quality-assessment-results", "children"),
         Output("quality-issues-recommendations", "children"),
         Output("quality-detailed-analysis", "children"),
         Output("quality-score-dashboard", "children"),
         Output("store-quality-data", "data"),
         Output("store-quality-results", "data")],
        [Input("quality-analyze-btn", "n_clicks"),
         Input("url", "pathname"),
         Input("quality-time-range-slider", "value"),
         Input("quality-btn-nudge-m10", "n_clicks"),
         Input("quality-btn-nudge-m1", "n_clicks"),
         Input("quality-btn-nudge-p1", "n_clicks"),
         Input("quality-btn-nudge-p10", "n_clicks")],
        [State("quality-start-time", "value"),
         State("quality-end-time", "value"),
         State("quality-signal-type", "value"),
         State("quality-metrics", "value"),
         State("quality-snr-threshold", "value"),
         State("quality-artifact-threshold", "value"),
         State("quality-advanced-options", "value")]
    )
    def quality_assessment_callback(n_clicks, pathname, slider_value, nudge_m10, nudge_m1, nudge_p1, nudge_p10,
                                  start_time, end_time, signal_type, quality_metrics, snr_threshold, 
                                  artifact_threshold, advanced_options):
        """Main callback for signal quality assessment."""
        logger.info("=== QUALITY ASSESSMENT CALLBACK TRIGGERED ===")
        
        # Only run this when we're on the quality page
        if pathname != "/quality":
            logger.info("Not on quality page, returning empty figures")
            return create_empty_figure(), create_empty_figure(), "Navigate to Quality page", "", "", "", None, None
        
        # Check if data is available first
        try:
            from vitalDSP_webapp.services.data.data_service import get_data_service
            data_service = get_data_service()
            
            if data_service is None:
                logger.error("Data service is None")
                return create_empty_figure(), create_empty_figure(), "Data service not available. Please restart the application.", "", "", "", None, None
            
            all_data = data_service.get_all_data()
            if not all_data:
                logger.info("No data available")
                return create_empty_figure(), create_empty_figure(), "No data available. Please upload and process data first.", "", "", "", None, None
            
            # Data is available, show instructions if no button click
            if n_clicks is None:
                logger.info("No button click - showing instructions")
                return create_empty_figure(), create_empty_figure(), "Data is available! Click '🎯 Assess Signal Quality' to start analysis.", "", "", "", None, None
            
            logger.info(f"Button clicked {n_clicks} times, starting quality assessment...")
            
            # Get the most recent data entry
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            
            logger.info(f"Found data: {latest_data_id}")
            
            # Get column mapping
            column_mapping = data_service.get_column_mapping(latest_data_id)
            if not column_mapping:
                logger.warning("Data has not been processed yet - no column mapping found")
                return create_empty_figure(), create_empty_figure(), "Please process your data on the Upload page first (configure column mapping)", "", "", "", None, None
                
            logger.info(f"Column mapping found: {column_mapping}")
                
            df = data_service.get_data(latest_data_id)
            if df is None or df.empty:
                logger.warning("Data frame is empty")
                return create_empty_figure(), create_empty_figure(), "Data is empty or corrupted.", "", "", "", None, None
            
            # Get sampling frequency from the data info
            sampling_freq = latest_data.get('info', {}).get('sampling_freq', 1000)
            logger.info(f"Sampling frequency: {sampling_freq}")
            
            # Get the main signal column
            signal_column = None
            for col_type, col_name in column_mapping.items():
                if col_type in ['ppg', 'ecg', 'signal']:
                    signal_column = col_name
                    break
            
            if signal_column is None:
                # Fallback to first column
                signal_column = df.columns[0]
            
            logger.info(f"Using signal column: {signal_column}")
            
            # Extract signal data
            signal_data = np.array(df[signal_column])
            
            # Handle time window
            start_time = start_time or 0
            end_time = end_time or 10
            
            # Convert time to sample indices
            start_idx = int(start_time * sampling_freq)
            end_idx = int(end_time * sampling_freq)
            
            # Ensure indices are within bounds
            start_idx = max(0, min(start_idx, len(signal_data) - 1))
            end_idx = max(start_idx + 1, min(end_idx, len(signal_data)))
            
            # Extract windowed data
            windowed_data = signal_data[start_idx:end_idx]
            time_axis = np.arange(len(windowed_data)) / sampling_freq
            
            # Set default values
            signal_type = signal_type or "auto"
            quality_metrics = quality_metrics or ["snr", "artifacts", "baseline_wander"]
            snr_threshold = snr_threshold or 10
            artifact_threshold = artifact_threshold or 0.1
            advanced_options = advanced_options or ["adaptive_threshold", "multiscale"]
            
            # Auto-detect signal type if needed
            if signal_type == "auto":
                signal_type = detect_signal_type(windowed_data, sampling_freq)
                logger.info(f"Auto-detected signal type: {signal_type}")
            
            # Perform quality assessment
            quality_results = assess_signal_quality(windowed_data, sampling_freq, quality_metrics, 
                                                 snr_threshold, artifact_threshold, advanced_options)
            
            # Create main quality plot
            main_plot = create_main_quality_plot(time_axis, windowed_data, quality_results, 
                                               signal_type, quality_metrics, sampling_freq)
            
            # Create quality metrics visualization
            metrics_plot = create_quality_metrics_plot(windowed_data, quality_results, 
                                                     quality_metrics, sampling_freq)
            
            # Create assessment results display
            assessment_results = create_quality_assessment_results(quality_results, signal_type, 
                                                                quality_metrics, sampling_freq)
            
            # Create issues and recommendations
            issues_recommendations = create_quality_issues_recommendations(quality_results, 
                                                                        quality_metrics, snr_threshold, 
                                                                        artifact_threshold)
            
            # Create detailed analysis
            detailed_analysis = create_quality_detailed_analysis(quality_results, quality_metrics)
            
            # Create quality score dashboard
            score_dashboard = create_quality_score_dashboard(quality_results, quality_metrics)
            
            # Store data for other callbacks
            quality_data = {
                "signal_data": windowed_data.tolist(),
                "signal_type": signal_type,
                "sampling_freq": sampling_freq,
                "quality_metrics": quality_metrics
            }
            
            quality_results_data = {
                "quality_results": quality_results,
                "signal_type": signal_type,
                "metrics": quality_metrics
            }
            
            logger.info("Quality assessment completed successfully")
            return main_plot, metrics_plot, assessment_results, issues_recommendations, detailed_analysis, score_dashboard, quality_data, quality_results_data
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            error_fig = create_empty_figure()
            error_fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            
            error_results = html.Div([
                html.H5("Error in Quality Assessment"),
                html.P(f"Assessment failed: {str(e)}"),
                html.P("Please check your data and parameters.")
            ])
            
            return error_fig, error_fig, error_results, "", "", "", None, None

    # Helper callbacks for time window updates
    @app.callback(
        [Output("quality-start-time", "value"),
         Output("quality-end-time", "value")],
        [Input("quality-btn-nudge-m10", "n_clicks"),
         Input("quality-btn-nudge-m1", "n_clicks"),
         Input("quality-btn-nudge-p1", "n_clicks"),
         Input("quality-btn-nudge-p10", "n_clicks")],
        [State("quality-start-time", "value"),
         State("quality-end-time", "value")]
    )
    def update_quality_time_inputs(nudge_m10, nudge_m1, nudge_p1, nudge_p10, start_time, end_time):
        """Update time inputs based on nudge button clicks."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        start_time = start_time or 0
        end_time = end_time or 10
        duration = end_time - start_time
        
        if button_id == "quality-btn-nudge-m10":
            start_time = max(0, start_time - 10)
            end_time = start_time + duration
        elif button_id == "quality-btn-nudge-m1":
            start_time = max(0, start_time - 1)
            end_time = start_time + duration
        elif button_id == "quality-btn-nudge-p1":
            start_time = start_time + 1
            end_time = start_time + duration
        elif button_id == "quality-btn-nudge-p10":
            start_time = start_time + 10
            end_time = start_time + duration
        
        return start_time, end_time
    
    @app.callback(
        Output("quality-time-range-slider", "value"),
        [Input("quality-start-time", "value"),
         Input("quality-end-time", "value")]
    )
    def update_quality_time_slider_range(start_time, end_time):
        """Update time slider range based on input values."""
        if start_time is not None and end_time is not None:
            return [start_time, end_time]
        return no_update


# Helper functions for signal quality assessment
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


def detect_signal_type(signal_data, sampling_freq):
    """Auto-detect the type of signal."""
    try:
        # Simple heuristics for signal type detection
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        
        # Find peaks for frequency analysis
        peaks, _ = signal.find_peaks(signal_data, height=mean_val + std_val, distance=int(sampling_freq * 0.3))
        
        if len(peaks) > 1:
            intervals = np.diff(peaks) / sampling_freq
            if np.mean(intervals) < 1.0:  # Less than 1 second between peaks
                return "ecg"  # Likely ECG (faster heart rate)
            else:
                return "ppg"  # Likely PPG (slower, more variable)
        else:
            return "general"  # Default to general if unclear
            
    except Exception as e:
        logger.warning(f"Error in signal type detection: {e}")
        return "general"  # Default fallback


def assess_signal_quality(signal_data, sampling_freq, quality_metrics, snr_threshold, artifact_threshold, advanced_options):
    """Perform comprehensive signal quality assessment."""
    try:
        quality_results = {}
        
        # SNR Assessment
        if "snr" in quality_metrics:
            quality_results["snr"] = calculate_snr(signal_data, sampling_freq)
        
        # Artifact Detection
        if "artifacts" in quality_metrics:
            quality_results["artifacts"] = detect_artifacts(signal_data, sampling_freq, artifact_threshold)
        
        # Baseline Wander
        if "baseline_wander" in quality_metrics:
            quality_results["baseline_wander"] = assess_baseline_wander(signal_data, sampling_freq)
        
        # Motion Artifacts
        if "motion_artifacts" in quality_metrics:
            quality_results["motion_artifacts"] = detect_motion_artifacts(signal_data, sampling_freq)
        
        # Signal Amplitude
        if "amplitude" in quality_metrics:
            quality_results["amplitude"] = assess_signal_amplitude(signal_data)
        
        # Signal Stability
        if "stability" in quality_metrics:
            quality_results["stability"] = assess_signal_stability(signal_data, sampling_freq)
        
        # Frequency Content
        if "frequency_content" in quality_metrics:
            quality_results["frequency_content"] = assess_frequency_content(signal_data, sampling_freq)
        
        # Peak Detection Quality
        if "peak_quality" in quality_metrics:
            quality_results["peak_quality"] = assess_peak_quality(signal_data, sampling_freq)
        
        # Signal Continuity
        if "continuity" in quality_metrics:
            quality_results["continuity"] = assess_signal_continuity(signal_data)
        
        # Outlier Detection
        if "outliers" in quality_metrics:
            quality_results["outliers"] = detect_outliers(signal_data)
        
        # Overall Quality Score
        quality_results["overall_score"] = calculate_overall_quality_score(quality_results)
        
        return quality_results
        
    except Exception as e:
        logger.error(f"Error in quality assessment: {e}")
        return {"error": f"Quality assessment failed: {str(e)}"}


def calculate_snr(signal_data, sampling_freq):
    """Calculate Signal-to-Noise Ratio."""
    try:
        # Estimate signal power (using peak detection)
        peaks, _ = signal.find_peaks(signal_data, height=np.mean(signal_data) + np.std(signal_data))
        if len(peaks) > 0:
            signal_power = np.mean(signal_data[peaks]**2)
        else:
            signal_power = np.mean(signal_data**2)
        
        # Estimate noise power (using residual after detrending)
        detrended = signal.detrend(signal_data)
        noise_power = np.mean(detrended**2)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = float('inf')
        
        return {
            "snr_db": snr_db,
            "signal_power": signal_power,
            "noise_power": noise_power,
            "quality": "excellent" if snr_db > 20 else "good" if snr_db > 10 else "poor"
        }
        
    except Exception as e:
        logger.error(f"Error calculating SNR: {e}")
        return {"error": f"SNR calculation failed: {str(e)}"}


def detect_artifacts(signal_data, sampling_freq, threshold):
    """Detect artifacts in the signal."""
    try:
        # Calculate signal statistics
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        
        # Find potential artifacts (outliers)
        artifact_threshold = mean_val + threshold * std_val
        artifacts = np.where(np.abs(signal_data - mean_val) > artifact_threshold)[0]
        
        # Calculate artifact metrics
        artifact_percentage = len(artifacts) / len(signal_data) * 100
        artifact_severity = np.mean(np.abs(signal_data[artifacts] - mean_val)) / std_val if len(artifacts) > 0 else 0
        
        return {
            "artifact_count": len(artifacts),
            "artifact_percentage": artifact_percentage,
            "artifact_severity": artifact_severity,
            "artifact_locations": artifacts.tolist(),
            "quality": "excellent" if artifact_percentage < 1 else "good" if artifact_percentage < 5 else "poor"
        }
        
    except Exception as e:
        logger.error(f"Error detecting artifacts: {e}")
        return {"error": f"Artifact detection failed: {str(e)}"}


def assess_baseline_wander(signal_data, sampling_freq):
    """Assess baseline wander in the signal."""
    try:
        # Apply high-pass filter to remove baseline wander
        nyquist = sampling_freq / 2
        cutoff = 0.5  # 0.5 Hz cutoff for baseline wander
        b, a = signal.butter(4, cutoff / nyquist, btype='high')
        filtered = signal.filtfilt(b, a, signal_data)
        
        # Calculate baseline wander as the difference between original and filtered
        baseline_wander = signal_data - filtered
        wander_magnitude = np.std(baseline_wander)
        wander_percentage = np.std(baseline_wander) / np.std(signal_data) * 100
        
        return {
            "wander_magnitude": wander_magnitude,
            "wander_percentage": wander_percentage,
            "quality": "excellent" if wander_percentage < 5 else "good" if wander_percentage < 15 else "poor"
        }
        
    except Exception as e:
        logger.error(f"Error assessing baseline wander: {e}")
        return {"error": f"Baseline wander assessment failed: {str(e)}"}


def detect_motion_artifacts(signal_data, sampling_freq):
    """Detect motion artifacts in the signal."""
    try:
        # Calculate signal envelope
        analytic_signal = signal.hilbert(signal_data)
        envelope = np.abs(analytic_signal)
        
        # Detect sudden changes in envelope (potential motion artifacts)
        envelope_diff = np.diff(envelope)
        motion_threshold = np.std(envelope_diff) * 2
        motion_artifacts = np.where(np.abs(envelope_diff) > motion_threshold)[0]
        
        motion_percentage = len(motion_artifacts) / len(envelope_diff) * 100
        
        return {
            "motion_artifact_count": len(motion_artifacts),
            "motion_percentage": motion_percentage,
            "quality": "excellent" if motion_percentage < 2 else "good" if motion_percentage < 10 else "poor"
        }
        
    except Exception as e:
        logger.error(f"Error detecting motion artifacts: {e}")
        return {"error": f"Motion artifact detection failed: {str(e)}"}


def assess_signal_amplitude(signal_data):
    """Assess signal amplitude characteristics."""
    try:
        amplitude_range = np.max(signal_data) - np.min(signal_data)
        amplitude_mean = np.mean(np.abs(signal_data))
        amplitude_cv = np.std(signal_data) / np.mean(np.abs(signal_data))  # Coefficient of variation
        
        return {
            "amplitude_range": amplitude_range,
            "amplitude_mean": amplitude_mean,
            "amplitude_cv": amplitude_cv,
            "quality": "excellent" if amplitude_cv < 0.3 else "good" if amplitude_cv < 0.6 else "poor"
        }
        
    except Exception as e:
        logger.error(f"Error assessing signal amplitude: {e}")
        return {"error": f"Amplitude assessment failed: {str(e)}"}


def assess_signal_stability(signal_data, sampling_freq):
    """Assess signal stability over time."""
    try:
        # Divide signal into segments and calculate stability
        segment_length = int(sampling_freq * 2)  # 2-second segments
        num_segments = len(signal_data) // segment_length
        
        if num_segments > 1:
            segment_means = []
            for i in range(num_segments):
                start_idx = i * segment_length
                end_idx = start_idx + segment_length
                segment_means.append(np.mean(signal_data[start_idx:end_idx]))
            
            stability_cv = np.std(segment_means) / np.mean(segment_means) if np.mean(segment_means) > 0 else 0
        else:
            stability_cv = 0
        
        return {
            "stability_cv": stability_cv,
            "num_segments": num_segments,
            "quality": "excellent" if stability_cv < 0.1 else "good" if stability_cv < 0.3 else "poor"
        }
        
    except Exception as e:
        logger.error(f"Error assessing signal stability: {e}")
        return {"error": f"Stability assessment failed: {str(e)}"}


def assess_frequency_content(signal_data, sampling_freq):
    """Assess frequency content of the signal."""
    try:
        # Calculate FFT
        fft_result = np.fft.fft(signal_data)
        fft_freq = np.fft.fftfreq(len(signal_data), 1/sampling_freq)
        
        # Get positive frequencies
        positive_mask = fft_freq > 0
        fft_freq = fft_freq[positive_mask]
        fft_magnitude = np.abs(fft_result[positive_mask])
        
        # Calculate spectral features
        spectral_centroid = np.sum(fft_freq * fft_magnitude) / np.sum(fft_magnitude) if np.sum(fft_magnitude) > 0 else 0
        spectral_bandwidth = np.sqrt(np.sum(((fft_freq - spectral_centroid) ** 2) * fft_magnitude) / np.sum(fft_magnitude)) if np.sum(fft_magnitude) > 0 else 0
        
        return {
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "dominant_frequency": fft_freq[np.argmax(fft_magnitude)] if len(fft_magnitude) > 0 else 0,
            "quality": "excellent" if spectral_bandwidth > 0 else "poor"
        }
        
    except Exception as e:
        logger.error(f"Error assessing frequency content: {e}")
        return {"error": f"Frequency content assessment failed: {str(e)}"}


def assess_peak_quality(signal_data, sampling_freq):
    """Assess quality of peak detection."""
    try:
        # Find peaks
        peaks, properties = signal.find_peaks(signal_data, height=np.mean(signal_data) + np.std(signal_data))
        
        if len(peaks) > 1:
            # Calculate peak intervals
            intervals = np.diff(peaks) / sampling_freq
            interval_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
            
            # Calculate peak heights
            peak_heights = signal_data[peaks]
            height_cv = np.std(peak_heights) / np.mean(peak_heights) if np.mean(peak_heights) > 0 else 0
            
            return {
                "peak_count": len(peaks),
                "interval_cv": interval_cv,
                "height_cv": height_cv,
                "quality": "excellent" if interval_cv < 0.1 and height_cv < 0.2 else "good" if interval_cv < 0.3 and height_cv < 0.5 else "poor"
            }
        else:
            return {
                "peak_count": len(peaks),
                "quality": "poor"
            }
        
    except Exception as e:
        logger.error(f"Error assessing peak quality: {e}")
        return {"error": f"Peak quality assessment failed: {str(e)}"}


def assess_signal_continuity(signal_data):
    """Assess signal continuity and gaps."""
    try:
        # Check for NaN or infinite values
        nan_count = np.sum(np.isnan(signal_data))
        inf_count = np.sum(np.isinf(signal_data))
        
        # Check for large gaps (sudden changes)
        signal_diff = np.diff(signal_data)
        gap_threshold = np.std(signal_diff) * 3
        gaps = np.where(np.abs(signal_diff) > gap_threshold)[0]
        
        continuity_score = 100 - (nan_count + inf_count + len(gaps)) / len(signal_data) * 100
        continuity_score = max(0, continuity_score)
        
        return {
            "nan_count": nan_count,
            "inf_count": inf_count,
            "gap_count": len(gaps),
            "continuity_score": continuity_score,
            "quality": "excellent" if continuity_score > 95 else "good" if continuity_score > 80 else "poor"
        }
        
    except Exception as e:
        logger.error(f"Error assessing signal continuity: {e}")
        return {"error": f"Continuity assessment failed: {str(e)}"}


def detect_outliers(signal_data):
    """Detect outliers in the signal."""
    try:
        # Use IQR method to detect outliers
        q1 = np.percentile(signal_data, 25)
        q3 = np.percentile(signal_data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = np.where((signal_data < lower_bound) | (signal_data > upper_bound))[0]
        outlier_percentage = len(outliers) / len(signal_data) * 100
        
        return {
            "outlier_count": len(outliers),
            "outlier_percentage": outlier_percentage,
            "quality": "excellent" if outlier_percentage < 1 else "good" if outlier_percentage < 5 else "poor"
        }
        
    except Exception as e:
        logger.error(f"Error detecting outliers: {e}")
        return {"error": f"Outlier detection failed: {str(e)}"}


def calculate_overall_quality_score(quality_results):
    """Calculate overall quality score from individual metrics."""
    try:
        if "error" in quality_results:
            return 0
        
        # Define quality weights
        quality_weights = {
            "snr": 0.25,
            "artifacts": 0.20,
            "baseline_wander": 0.15,
            "motion_artifacts": 0.10,
            "amplitude": 0.10,
            "stability": 0.10,
            "frequency_content": 0.05,
            "peak_quality": 0.05
        }
        
        # Calculate weighted score
        total_score = 0
        total_weight = 0
        
        for metric, weight in quality_weights.items():
            if metric in quality_results and "quality" in quality_results[metric]:
                quality = quality_results[metric]["quality"]
                if quality == "excellent":
                    score = 100
                elif quality == "good":
                    score = 70
                elif quality == "fair":
                    score = 50
                else:
                    score = 30
                
                total_score += score * weight
                total_weight += weight
        
        overall_score = total_score / total_weight if total_weight > 0 else 0
        
        # Determine overall quality
        if overall_score >= 90:
            overall_quality = "excellent"
        elif overall_score >= 70:
            overall_quality = "good"
        elif overall_score >= 50:
            overall_quality = "fair"
        else:
            overall_quality = "poor"
        
        return {
            "score": overall_score,
            "quality": overall_quality
        }
        
    except Exception as e:
        logger.error(f"Error calculating overall quality score: {e}")
        return {"error": f"Overall score calculation failed: {str(e)}"}


def create_main_quality_plot(time_axis, signal_data, quality_results, signal_type, quality_metrics, sampling_freq):
    """Create the main quality assessment plot."""
    try:
        if "error" in quality_results:
            return create_empty_figure()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Signal Quality Overview", "Quality Metrics"],
            vertical_spacing=0.1
        )
        
        # Original signal
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal_data,
                mode='lines',
                name='Signal',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add quality indicators
        if "artifacts" in quality_results and "artifact_locations" in quality_results["artifacts"]:
            artifacts = quality_results["artifacts"]["artifact_locations"]
            if len(artifacts) > 0:
                artifact_times = [time_axis[i] for i in artifacts if i < len(time_axis)]
                artifact_values = [signal_data[i] for i in artifacts if i < len(time_axis)]
                
                fig.add_trace(
                    go.Scatter(
                        x=artifact_times,
                        y=artifact_values,
                        mode='markers',
                        name='Artifacts',
                        marker=dict(color='red', size=8, symbol='x')
                    ),
                    row=1, col=1
                )
        
        # Quality metrics bar chart
        if "overall_score" in quality_results:
            metrics = list(quality_results.keys())
            metrics = [m for m in metrics if m != "overall_score" and "error" not in quality_results.get(m, {})]
            
            if metrics:
                scores = []
                for metric in metrics:
                    if "quality" in quality_results[metric]:
                        quality = quality_results[metric]["quality"]
                        if quality == "excellent":
                            scores.append(100)
                        elif quality == "good":
                            scores.append(70)
                        elif quality == "fair":
                            scores.append(50)
                        else:
                            scores.append(30)
                    else:
                        scores.append(50)
                
                fig.add_trace(
                    go.Bar(
                        x=metrics,
                        y=scores,
                        name='Quality Scores',
                        marker_color='green'
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            title=f"Signal Quality Assessment: {signal_type.upper()}",
            height=600,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating main quality plot: {e}")
        return create_empty_figure()


def create_quality_metrics_plot(signal_data, quality_results, quality_metrics, sampling_freq):
    """Create quality metrics visualization."""
    try:
        if "error" in quality_results:
            return create_empty_figure()
        
        # Create subplots based on selected metrics
        num_metrics = len(quality_metrics)
        if num_metrics == 0:
            num_metrics = 1
        
        fig = make_subplots(
            rows=num_metrics, cols=1,
            subplot_titles=[f"{metric.replace('_', ' ').title()} Analysis" for metric in quality_metrics],
            vertical_spacing=0.1
        )
        
        row = 1
        for metric in quality_metrics:
            if metric in quality_results and "error" not in quality_results[metric]:
                if metric == "snr":
                    # SNR visualization
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[quality_results[metric].get("snr_db", 0), quality_results[metric].get("snr_db", 0)],
                            mode='lines+markers',
                            name='SNR',
                            line=dict(color='blue', width=3)
                        ),
                        row=row, col=1
                    )
                    row += 1
                
                elif metric == "artifacts":
                    # Artifact visualization
                    if "artifact_locations" in quality_results[metric]:
                        artifacts = quality_results[metric]["artifact_locations"]
                        if len(artifacts) > 0:
                            fig.add_trace(
                                go.Scatter(
                                    x=artifacts,
                                    y=signal_data[artifacts],
                                    mode='markers',
                                    name='Artifacts',
                                    marker=dict(color='red', size=8, symbol='x')
                                ),
                                row=row, col=1
                            )
                    row += 1
        
        fig.update_layout(
            title="Quality Metrics Analysis",
            height=300 * num_metrics,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating quality metrics plot: {e}")
        return create_empty_figure()


def create_quality_assessment_results(quality_results, signal_type, quality_metrics, sampling_freq):
    """Create the quality assessment results display."""
    try:
        if "error" in quality_results:
            return html.Div([
                html.H5("Quality Assessment Error"),
                html.P(f"Assessment failed: {quality_results['error']}")
            ])
        
        sections = []
        
        # Overall quality score
        if "overall_score" in quality_results:
            overall = quality_results["overall_score"]
            sections.append(html.Div([
                html.H5(f"🎯 Overall Quality Assessment"),
                html.Div([
                    html.H6(f"Score: {overall.get('score', 0):.1f}/100"),
                    html.H6(f"Quality: {overall.get('quality', 'Unknown').title()}")
                ], className="text-center mb-3")
            ]))
        
        # Individual metric results
        for metric in quality_metrics:
            if metric in quality_results and "error" not in quality_results[metric]:
                metric_result = quality_results[metric]
                if "quality" in metric_result:
                    quality_color = {
                        "excellent": "success",
                        "good": "info",
                        "fair": "warning",
                        "poor": "danger"
                    }.get(metric_result["quality"], "secondary")
                    
                    sections.append(html.Div([
                        html.H6(f"{metric.replace('_', ' ').title()}"),
                        dbc.Badge(
                            metric_result["quality"].title(),
                            color=quality_color,
                            className="mb-2"
                        )
                    ], className="mb-3"))
        
        return html.Div(sections)
        
    except Exception as e:
        logger.error(f"Error creating assessment results: {e}")
        return html.Div([
            html.H5("Error"),
            html.P(f"Failed to create assessment results: {str(e)}")
        ])


def create_quality_issues_recommendations(quality_results, quality_metrics, snr_threshold, artifact_threshold):
    """Create quality issues and recommendations display."""
    try:
        if "error" in quality_results:
            return html.Div([
                html.H6("Quality Issues"),
                html.P("Assessment failed - no recommendations available.")
            ])
        
        issues = []
        recommendations = []
        
        # Check SNR
        if "snr" in quality_results and "snr_db" in quality_results["snr"]:
            snr_db = quality_results["snr"]["snr_db"]
            if snr_db < snr_threshold:
                issues.append("Low Signal-to-Noise Ratio")
                recommendations.append("Check sensor placement and reduce environmental noise")
        
        # Check artifacts
        if "artifacts" in quality_results and "artifact_percentage" in quality_results["artifacts"]:
            artifact_pct = quality_results["artifacts"]["artifact_percentage"]
            if artifact_pct > 5:
                issues.append("High artifact content")
                recommendations.append("Check for movement artifacts and improve signal acquisition")
        
        # Check baseline wander
        if "baseline_wander" in quality_results and "wander_percentage" in quality_results["baseline_wander"]:
            wander_pct = quality_results["baseline_wander"]["wander_percentage"]
            if wander_pct > 15:
                issues.append("Significant baseline wander")
                recommendations.append("Apply high-pass filtering or improve sensor stability")
        
        if not issues:
            issues.append("No significant quality issues detected")
            recommendations.append("Signal quality is acceptable for analysis")
        
        return html.Div([
            html.H6("⚠️ Quality Issues"),
            html.Ul([html.Li(issue) for issue in issues], className="mb-3"),
            html.H6("💡 Recommendations"),
            html.Ul([html.Li(rec) for rec in recommendations])
        ])
        
    except Exception as e:
        logger.error(f"Error creating issues and recommendations: {e}")
        return html.Div([
            html.H6("Error"),
            html.P(f"Failed to create issues and recommendations: {str(e)}")
        ])


def create_quality_detailed_analysis(quality_results, quality_metrics):
    """Create detailed quality analysis display."""
    try:
        if "error" in quality_results:
            return html.Div([
                html.H6("Detailed Analysis"),
                html.P("Assessment failed - no detailed analysis available.")
            ])
        
        sections = []
        
        for metric in quality_metrics:
            if metric in quality_results and "error" not in quality_results[metric]:
                metric_result = quality_results[metric]
                metric_title = metric.replace('_', ' ').title()
                
                metric_section = html.Div([
                    html.H6(f"📊 {metric_title}"),
                    html.Div([
                        html.P(f"{key.replace('_', ' ').title()}: {value}")
                        for key, value in metric_result.items()
                        if key not in ["quality", "error"]
                    ])
                ], className="mb-3")
                
                sections.append(metric_section)
        
        if not sections:
            return html.Div([
                html.H6("Detailed Analysis"),
                html.P("No detailed analysis available for selected metrics.")
            ])
        
        return html.Div(sections)
        
    except Exception as e:
        logger.error(f"Error creating detailed analysis: {e}")
        return html.Div([
            html.H6("Error"),
            html.P(f"Failed to create detailed analysis: {str(e)}")
        ])


def create_quality_score_dashboard(quality_results, quality_metrics):
    """Create quality score dashboard."""
    try:
        if "error" in quality_results:
            return html.Div([
                html.H6("Quality Dashboard"),
                html.P("Assessment failed - no dashboard available.")
            ])
        
        if "overall_score" not in quality_results:
            return html.Div([
                html.H6("Quality Dashboard"),
                html.P("Overall quality score not available.")
            ])
        
        overall = quality_results["overall_score"]
        score = overall.get("score", 0)
        quality = overall.get("quality", "Unknown")
        
        # Create score cards
        score_cards = []
        
        # Overall score card
        score_cards.append(html.Div([
            html.H4(f"{score:.1f}", className="text-center mb-2"),
            html.P("Overall Score", className="text-center text-muted mb-0")
        ], className="col-md-6"))
        
        # Quality level card
        quality_color = {
            "excellent": "success",
            "good": "info",
            "fair": "warning",
            "poor": "danger"
        }.get(quality, "secondary")
        
        score_cards.append(html.Div([
            dbc.Badge(
                quality.title(),
                color=quality_color,
                className="fs-4 p-3"
            ),
            html.P("Quality Level", className="text-center text-muted mb-0 mt-2")
        ], className="col-md-6"))
        
        return html.Div([
            html.H6("📊 Quality Score Dashboard"),
            html.Div(score_cards, className="row text-center")
        ])
        
    except Exception as e:
        logger.error(f"Error creating quality dashboard: {e}")
        return html.Div([
            html.H6("Error"),
            html.P(f"Failed to create quality dashboard: {str(e)}")
        ])
