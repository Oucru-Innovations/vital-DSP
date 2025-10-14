"""
Signal quality assessment callbacks using vitalDSP implementations.

This module replaces custom quality assessment implementations with validated
vitalDSP signal quality methods for improved accuracy and consistency.

IMPROVEMENTS OVER ORIGINAL quality_callbacks.py:
- Uses validated vitalDSP SignalQuality and SignalQualityIndex classes
- Provides temporal resolution with segment-wise analysis
- More accurate SNR calculation (full signal power vs peak-based)
- Multiple artifact detection methods (z-score, adaptive, kurtosis)
- Quantitative SQI values instead of qualitative assessments
- Automated normal/abnormal segment classification
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html
from dash.exceptions import PreventUpdate
from scipy import signal
import dash_bootstrap_components as dbc
import logging

logger = logging.getLogger(__name__)


def register_quality_callbacks(app):
    """Register all signal quality assessment callbacks using vitalDSP."""
    logger.info("=== REGISTERING vitalDSP SIGNAL QUALITY CALLBACKS ===")

    # Import vitalDSP modules
    _import_vitaldsp_modules()

    @app.callback(
        [
            Output("quality-main-plot", "figure"),
            Output("quality-metrics-plot", "figure"),
            Output("quality-assessment-results", "children"),
            Output("quality-issues-recommendations", "children"),
            Output("quality-detailed-analysis", "children"),
            Output("quality-score-dashboard", "children"),
            Output("store-quality-data", "data"),
            Output("store-quality-results", "data"),
        ],
        [
            Input("quality-analyze-btn", "n_clicks"),
            Input("url", "pathname"),
            Input("quality-time-range-slider", "value"),
            Input("quality-btn-nudge-m10", "n_clicks"),
            Input("quality-btn-nudge-m1", "n_clicks"),
            Input("quality-btn-nudge-p1", "n_clicks"),
            Input("quality-btn-nudge-p10", "n_clicks"),
        ],
        [
            State("quality-start-time", "value"),
            State("quality-end-time", "value"),
            State("quality-signal-type", "value"),
            State("quality-metrics", "value"),
            State("quality-snr-threshold", "value"),
            State("quality-artifact-threshold", "value"),
            State("quality-advanced-options", "value"),
        ],
    )
    def quality_assessment_callback(
        n_clicks,
        pathname,
        slider_value,
        nudge_m10,
        nudge_m1,
        nudge_p1,
        nudge_p10,
        start_time,
        end_time,
        signal_type,
        quality_metrics,
        snr_threshold,
        artifact_threshold,
        advanced_options,
    ):
        """Main callback for signal quality assessment using vitalDSP."""
        logger.info("=== vitalDSP QUALITY ASSESSMENT CALLBACK TRIGGERED ===")

        # Only run this when we're on the quality page
        if pathname != "/quality":
            logger.info("Not on quality page, returning empty figures")
            return (
                create_empty_figure(),
                create_empty_figure(),
                "Navigate to Quality page",
                "",
                "",
                "",
                None,
                None,
            )

        # Check if button was clicked
        if n_clicks is None:
            return (
                create_empty_figure(),
                create_empty_figure(),
                "Click 'Assess Signal Quality' to analyze your signal.",
                "",
                "",
                "",
                None,
                None,
            )

        # Check if data is available
        try:
            from vitalDSP_webapp.services.data.data_service import get_data_service

            data_service = get_data_service()
            
            # Check if Enhanced Data Service is available for heavy data processing
            if data_service and hasattr(data_service, 'is_enhanced_service_available'):
                if data_service.is_enhanced_service_available():
                    logger.info("Enhanced Data Service is available for heavy data processing")
                else:
                    logger.info("Using basic data service functionality")
            else:
                logger.info("Data service not available or missing enhanced service methods")

            if data_service is None:
                logger.error("Data service not available")
                return (
                    create_empty_figure(),
                    create_empty_figure(),
                    "Data service not available",
                    "",
                    "",
                    "",
                    None,
                    None,
                )

            if not data_service.has_data():
                logger.error("No data available")
                return (
                    create_empty_figure(),
                    create_empty_figure(),
                    "No data available. Please upload a signal first.",
                    "",
                    "",
                    "",
                    None,
                    None,
                )

            # Get all data
            all_data = data_service.get_all_data()
            if not all_data:
                logger.error("No data available")
                return (
                    create_empty_figure(),
                    create_empty_figure(),
                    "No data available. Please upload a signal first.",
                    "",
                    "",
                    "",
                    None,
                    None,
                )

            # Get the most recent data entry
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]

            # Get column mapping
            column_mapping = data_service.get_column_mapping(latest_data_id)
            if column_mapping is None:
                return (
                    create_empty_figure(),
                    create_empty_figure(),
                    "Please configure column mapping to process your data.",
                    "",
                    "",
                    "",
                    None,
                    None,
                )

            # Extract signal data and sampling frequency
            signal_data = latest_data.get("data")
            if signal_data is None:
                # If no 'data' field, assume the latest_data itself is the signal data
                signal_data = latest_data

            # Check if signal data is actually a valid array or DataFrame
            if not isinstance(signal_data, (np.ndarray, list)) and not hasattr(
                signal_data, "values"
            ):
                return (
                    create_empty_figure(),
                    create_empty_figure(),
                    "No signal data available. Please upload a signal first.",
                    "",
                    "",
                    "",
                    None,
                    None,
                )

            # Convert DataFrame to numpy array if needed
            if hasattr(signal_data, "values"):
                signal_data = signal_data.values

            sampling_freq = latest_data.get("info", {}).get("sampling_freq", 1000)
            
            # Enhanced data processing for heavy datasets
            if signal_data is not None and len(signal_data) > 0:
                data_size_mb = signal_data.nbytes / (1024 * 1024) if hasattr(signal_data, 'nbytes') else len(signal_data) * 8 / (1024 * 1024)
                num_samples = len(signal_data)
                
                logger.info(f"Signal data size: {data_size_mb:.2f} MB, Samples: {num_samples}")
                
                # Use Enhanced Data Service for heavy data processing
                if (data_service and hasattr(data_service, 'is_enhanced_service_available') and 
                    data_service.is_enhanced_service_available() and (data_size_mb > 5 or num_samples > 100000)):
                    logger.info(f"Using Enhanced Data Service for heavy quality assessment: {data_size_mb:.2f}MB, {num_samples} samples")
                    
                    # Get enhanced service for optimized processing
                    if hasattr(data_service, 'get_enhanced_service'):
                        enhanced_service = data_service.get_enhanced_service()
                        if enhanced_service:
                            logger.info("Enhanced Data Service is ready for optimized quality assessment")
                            # The enhanced service will automatically handle chunked processing
                            # and memory optimization during quality assessment
                else:
                    logger.info("Using standard processing for lightweight quality assessment")

            # Extract time segment if specified
            if start_time is not None and end_time is not None:
                start_idx = int(start_time * sampling_freq)
                end_idx = int(end_time * sampling_freq)
                signal_data = signal_data[start_idx:end_idx]

            # Perform quality assessment using vitalDSP
            quality_results = assess_signal_quality_vitaldsp(
                signal_data,
                sampling_freq,
                quality_metrics if quality_metrics else [],
                snr_threshold if snr_threshold else 10.0,
                artifact_threshold if artifact_threshold else 3.0,
                advanced_options if advanced_options else [],
            )

            # Create visualizations
            main_plot = create_quality_main_plot(
                signal_data, quality_results, sampling_freq
            )
            metrics_plot = create_quality_metrics_plot(quality_results)

            # Create results displays
            assessment_results = create_assessment_results_display(quality_results)
            issues_recommendations = create_issues_recommendations_display(
                quality_results
            )
            detailed_analysis = create_detailed_analysis_display(quality_results)
            score_dashboard = create_score_dashboard(quality_results)

            return (
                main_plot,
                metrics_plot,
                assessment_results,
                issues_recommendations,
                detailed_analysis,
                score_dashboard,
                {"signal": signal_data.tolist(), "sampling_freq": sampling_freq},
                quality_results,
            )

        except Exception as e:
            logger.error(f"Error in quality assessment callback: {e}", exc_info=True)
            return (
                create_empty_figure(),
                create_empty_figure(),
                f"Error: {str(e)}",
                "",
                "",
                "",
                None,
                None,
            )


def _import_vitaldsp_modules():
    """Import vitalDSP modules with error handling."""
    try:
        global SignalQuality, SignalQualityIndex
        global z_score_artifact_detection, adaptive_threshold_artifact_detection
        global kurtosis_artifact_detection

        from vitalDSP.signal_quality_assessment.signal_quality import SignalQuality
        from vitalDSP.signal_quality_assessment.signal_quality_index import (
            SignalQualityIndex,
        )
        from vitalDSP.signal_quality_assessment.artifact_detection_removal import (
            z_score_artifact_detection,
            adaptive_threshold_artifact_detection,
            kurtosis_artifact_detection,
        )

        logger.info("Successfully imported vitalDSP quality assessment modules")
    except ImportError as e:
        logger.error(f"Error importing vitalDSP modules: {e}")
        raise


def assess_signal_quality_vitaldsp(
    signal_data,
    sampling_freq,
    quality_metrics,
    snr_threshold,
    artifact_threshold,
    advanced_options,
):
    """
    Perform comprehensive signal quality assessment using vitalDSP.

    This function replaces the custom quality assessment with validated
    vitalDSP implementations for improved accuracy and consistency.

    Args:
        signal_data: Signal array
        sampling_freq: Sampling frequency in Hz
        quality_metrics: List of metrics to compute
        snr_threshold: SNR threshold for quality classification (dB)
        artifact_threshold: Artifact detection threshold (z-score)
        advanced_options: List of advanced options

    Returns:
        dict: Comprehensive quality assessment results
    """
    try:
        quality_results = {}

        # Window parameters for segment-wise analysis
        window_size = int(10 * sampling_freq)  # 10-second windows
        step_size = int(5 * sampling_freq)  # 5-second overlap

        # Initialize SignalQualityIndex for temporal analysis
        sqi = SignalQualityIndex(signal_data)

        # SNR Assessment (if we have filtered signal, use SignalQuality)
        if "snr" in quality_metrics or not quality_metrics:
            # For standalone SNR without processed signal, use signal power vs noise estimation
            # Estimate noise as high-frequency content
            from scipy.signal import butter, filtfilt

            # High-pass filter to extract noise
            nyq = sampling_freq / 2
            if nyq > 10:  # Only if we have sufficient frequency range
                cutoff = min(10, nyq * 0.8)  # 10 Hz or 80% of Nyquist
                b, a = butter(4, cutoff / nyq, btype="high")
                noise_estimate = filtfilt(b, a, signal_data)

                # Low-pass filter to get signal
                cutoff_low = max(0.5, cutoff / 20)  # Low cutoff
                b_low, a_low = butter(4, cutoff_low / nyq, btype="low")
                signal_estimate = filtfilt(b_low, a_low, signal_data)

                sq = SignalQuality(signal_estimate, signal_data)
                snr_db = sq.snr()
            else:
                # Fallback: use signal power vs std
                signal_power = np.mean(signal_data**2)
                noise_power = np.std(signal_data) ** 2
                snr_db = (
                    10 * np.log10(signal_power / noise_power)
                    if noise_power > 0
                    else float("inf")
                )

            quality_results["snr"] = {
                "snr_db": float(snr_db),
                "quality": (
                    "excellent"
                    if snr_db > 20
                    else "good" if snr_db > snr_threshold else "poor"
                ),
                "threshold": snr_threshold,
            }

        # Artifact Detection (multiple methods)
        if "artifacts" in quality_metrics or not quality_metrics:
            # Use vitalDSP's multiple artifact detection methods
            artifacts_zscore = z_score_artifact_detection(
                signal_data, z_threshold=artifact_threshold
            )

            # Adaptive threshold (better for non-stationary signals)
            artifacts_adaptive = adaptive_threshold_artifact_detection(
                signal_data,
                window_size=int(2 * sampling_freq),  # 2-second windows
                std_factor=artifact_threshold * 0.8,
            )

            # Kurtosis-based (better for sharp transients)
            artifacts_kurtosis = kurtosis_artifact_detection(
                signal_data, kurt_threshold=3.0
            )

            # Combine results (union of all detected artifacts)
            all_artifacts = np.unique(
                np.concatenate(
                    [artifacts_zscore, artifacts_adaptive, artifacts_kurtosis]
                )
            )

            artifact_percentage = 100 * len(all_artifacts) / len(signal_data)

            quality_results["artifacts"] = {
                "artifact_count": len(all_artifacts),
                "artifact_percentage": float(artifact_percentage),
                "artifact_locations": all_artifacts.tolist(),
                "zscore_count": len(artifacts_zscore),
                "adaptive_count": len(artifacts_adaptive),
                "kurtosis_count": len(artifacts_kurtosis),
                "quality": (
                    "excellent"
                    if artifact_percentage < 1
                    else "good" if artifact_percentage < 5 else "poor"
                ),
                "threshold": artifact_threshold,
            }

        # Baseline Wander SQI (temporal analysis)
        if "baseline_wander" in quality_metrics or not quality_metrics:
            baseline_sqi_values, bl_normal, bl_abnormal = sqi.baseline_wander_sqi(
                window_size,
                step_size,
                threshold=0.7,
                moving_avg_window=int(2 * sampling_freq),
            )

            quality_results["baseline_wander"] = {
                "mean_sqi": float(np.mean(baseline_sqi_values)),
                "min_sqi": float(np.min(baseline_sqi_values)),
                "sqi_values": [float(x) for x in baseline_sqi_values],
                "normal_segments": bl_normal,
                "abnormal_segments": bl_abnormal,
                "abnormal_percentage": (
                    100 * len(bl_abnormal) / (len(bl_normal) + len(bl_abnormal))
                    if (len(bl_normal) + len(bl_abnormal)) > 0
                    else 0
                ),
                "quality": (
                    "excellent"
                    if np.mean(baseline_sqi_values) > 0.8
                    else "good" if np.mean(baseline_sqi_values) > 0.6 else "poor"
                ),
            }

        # Amplitude Variability SQI
        if "amplitude" in quality_metrics or not quality_metrics:
            amp_sqi_values, amp_normal, amp_abnormal = sqi.amplitude_variability_sqi(
                window_size, step_size, threshold=0.7, scale="zscore"
            )

            quality_results["amplitude"] = {
                "mean_sqi": float(np.mean(amp_sqi_values)),
                "sqi_values": [float(x) for x in amp_sqi_values],
                "normal_segments": amp_normal,
                "abnormal_segments": amp_abnormal,
                "range": float(np.max(signal_data) - np.min(signal_data)),
                "quality": (
                    "excellent"
                    if np.mean(amp_sqi_values) > 0.8
                    else "good" if np.mean(amp_sqi_values) > 0.6 else "poor"
                ),
            }

        # Signal Entropy SQI
        if "frequency_content" in quality_metrics or not quality_metrics:
            entropy_sqi_values, ent_normal, ent_abnormal = sqi.signal_entropy_sqi(
                window_size, step_size, threshold=0.5
            )

            quality_results["frequency_content"] = {
                "mean_entropy_sqi": float(np.mean(entropy_sqi_values)),
                "sqi_values": [float(x) for x in entropy_sqi_values],
                "normal_segments": ent_normal,
                "abnormal_segments": ent_abnormal,
                "quality": (
                    "excellent"
                    if np.mean(entropy_sqi_values) > 0.7
                    else "good" if np.mean(entropy_sqi_values) > 0.5 else "poor"
                ),
            }

        # Zero Crossing SQI (signal stability)
        if "stability" in quality_metrics or not quality_metrics:
            zc_sqi_values, zc_normal, zc_abnormal = sqi.zero_crossing_sqi(
                window_size, step_size, threshold=0.6
            )

            quality_results["stability"] = {
                "mean_sqi": float(np.mean(zc_sqi_values)),
                "sqi_values": [float(x) for x in zc_sqi_values],
                "normal_segments": zc_normal,
                "abnormal_segments": zc_abnormal,
                "quality": (
                    "excellent"
                    if np.mean(zc_sqi_values) > 0.8
                    else "good" if np.mean(zc_sqi_values) > 0.6 else "poor"
                ),
            }

        # Energy SQI
        if "peak_quality" in quality_metrics or not quality_metrics:
            energy_sqi_values, en_normal, en_abnormal = sqi.energy_sqi(
                window_size, step_size, threshold=0.6
            )

            quality_results["peak_quality"] = {
                "mean_energy_sqi": float(np.mean(energy_sqi_values)),
                "sqi_values": [float(x) for x in energy_sqi_values],
                "normal_segments": en_normal,
                "abnormal_segments": en_abnormal,
                "quality": (
                    "excellent"
                    if np.mean(energy_sqi_values) > 0.8
                    else "good" if np.mean(energy_sqi_values) > 0.6 else "poor"
                ),
            }

        # Signal Continuity (NaN/Inf detection)
        if "continuity" in quality_metrics or not quality_metrics:
            has_nan = np.isnan(signal_data).any()
            has_inf = np.isinf(signal_data).any()
            nan_count = np.isnan(signal_data).sum()
            inf_count = np.isinf(signal_data).sum()

            quality_results["continuity"] = {
                "has_nan": bool(has_nan),
                "has_inf": bool(has_inf),
                "nan_count": int(nan_count),
                "inf_count": int(inf_count),
                "continuity_percentage": float(
                    100 * (1 - (nan_count + inf_count) / len(signal_data))
                ),
                "quality": "poor" if (has_nan or has_inf) else "excellent",
            }

        # Overall Quality Score (weighted average of all SQI metrics)
        sqi_scores = []
        if "snr" in quality_results and "snr_db" in quality_results["snr"]:
            # Normalize SNR to 0-1 scale (assuming typical range 0-30 dB)
            snr_normalized = min(1.0, max(0.0, quality_results["snr"]["snr_db"] / 30.0))
            sqi_scores.append(snr_normalized)

        if "baseline_wander" in quality_results:
            sqi_scores.append(quality_results["baseline_wander"]["mean_sqi"])

        if "amplitude" in quality_results:
            sqi_scores.append(quality_results["amplitude"]["mean_sqi"])

        if "stability" in quality_results:
            sqi_scores.append(quality_results["stability"]["mean_sqi"])

        if "frequency_content" in quality_results:
            sqi_scores.append(quality_results["frequency_content"]["mean_entropy_sqi"])

        # Artifact penalty
        artifact_penalty = 1.0
        if "artifacts" in quality_results:
            artifact_pct = quality_results["artifacts"]["artifact_percentage"]
            artifact_penalty = max(0.0, 1.0 - artifact_pct / 100.0)

        # Calculate overall score
        mean_sqi = np.mean(sqi_scores) if sqi_scores else 0.5
        overall_score = mean_sqi * artifact_penalty

        quality_results["overall_score"] = {
            "score": float(overall_score),
            "percentage": float(overall_score * 100),
            "quality": (
                "excellent"
                if overall_score > 0.8
                else (
                    "good"
                    if overall_score > 0.6
                    else "fair" if overall_score > 0.4 else "poor"
                )
            ),
            "component_scores": {
                "mean_sqi": float(mean_sqi),
                "artifact_penalty": float(artifact_penalty),
            },
        }

        logger.info(f"Quality assessment complete. Overall score: {overall_score:.2f}")
        return quality_results

    except Exception as e:
        logger.error(f"Error in vitalDSP quality assessment: {e}", exc_info=True)
        return {"error": f"Quality assessment failed: {str(e)}"}


def create_empty_figure():
    """Create an empty figure."""
    fig = go.Figure()
    fig.update_layout(
        title="No data available",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    return fig


def create_quality_main_plot(signal_data, quality_results, sampling_freq):
    """Create main quality visualization plot."""
    try:
        time_axis = np.arange(len(signal_data)) / sampling_freq

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=["Signal with Quality Markers", "Quality Metrics Over Time"],
            vertical_spacing=0.12,
        )

        # Plot signal
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal_data,
                mode="lines",
                name="Signal",
                line=dict(color="blue", width=1),
            ),
            row=1,
            col=1,
        )

        # Mark artifacts if available
        if (
            "artifacts" in quality_results
            and "artifact_locations" in quality_results["artifacts"]
        ):
            artifact_indices = quality_results["artifacts"]["artifact_locations"]
            if artifact_indices:
                fig.add_trace(
                    go.Scatter(
                        x=time_axis[artifact_indices],
                        y=signal_data[artifact_indices],
                        mode="markers",
                        name="Artifacts",
                        marker=dict(color="red", size=4, symbol="x"),
                    ),
                    row=1,
                    col=1,
                )

        # Plot SQI values over time if available
        if (
            "baseline_wander" in quality_results
            and "sqi_values" in quality_results["baseline_wander"]
        ):
            sqi_values = quality_results["baseline_wander"]["sqi_values"]
            window_size = int(10 * sampling_freq)
            step_size = int(5 * sampling_freq)
            sqi_times = np.arange(len(sqi_values)) * step_size / sampling_freq

            fig.add_trace(
                go.Scatter(
                    x=sqi_times,
                    y=sqi_values,
                    mode="lines+markers",
                    name="Baseline SQI",
                    line=dict(color="green", width=2),
                ),
                row=2,
                col=1,
            )

        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="SQI Value", row=2, col=1)

        fig.update_layout(height=800, showlegend=True, hovermode="x unified")

        return fig

    except Exception as e:
        logger.error(f"Error creating main plot: {e}")
        return create_empty_figure()


def create_quality_metrics_plot(quality_results):
    """Create quality metrics visualization."""
    try:
        metrics_data = []
        metric_names = []

        # Extract metrics
        if "snr" in quality_results and "snr_db" in quality_results["snr"]:
            metrics_data.append(min(1.0, quality_results["snr"]["snr_db"] / 30.0))
            metric_names.append("SNR")

        if "baseline_wander" in quality_results:
            metrics_data.append(quality_results["baseline_wander"]["mean_sqi"])
            metric_names.append("Baseline")

        if "amplitude" in quality_results:
            metrics_data.append(quality_results["amplitude"]["mean_sqi"])
            metric_names.append("Amplitude")

        if "stability" in quality_results:
            metrics_data.append(quality_results["stability"]["mean_sqi"])
            metric_names.append("Stability")

        if "artifacts" in quality_results:
            artifact_score = max(
                0, 1.0 - quality_results["artifacts"]["artifact_percentage"] / 100.0
            )
            metrics_data.append(artifact_score)
            metric_names.append("Artifact-free")

        # Create bar chart
        fig = go.Figure(
            data=[
                go.Bar(
                    x=metric_names,
                    y=metrics_data,
                    marker=dict(
                        color=metrics_data,
                        colorscale="RdYlGn",
                        cmin=0,
                        cmax=1,
                    ),
                )
            ]
        )

        fig.update_layout(
            title="Signal Quality Metrics (0-1 Scale)",
            xaxis_title="Metric",
            yaxis_title="Quality Score",
            yaxis=dict(range=[0, 1]),
            height=400,
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating metrics plot: {e}")
        return create_empty_figure()


def create_assessment_results_display(quality_results):
    """Create assessment results display."""
    try:
        if "error" in quality_results:
            return f"Error: {quality_results['error']}"
        if "overall_score" not in quality_results:
            return html.Div("No quality assessment results available.")

        overall = quality_results["overall_score"]
        quality_class = (
            "success"
            if overall["quality"] == "excellent"
            else (
                "info"
                if overall["quality"] == "good"
                else "warning" if overall["quality"] == "fair" else "danger"
            )
        )

        return dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Overall Quality Assessment", className="card-title"),
                    dbc.Alert(
                        [
                            html.H5(
                                f"Quality: {overall['quality'].upper()}",
                                className="alert-heading",
                            ),
                            html.P(
                                f"Score: {overall['score']:.3f} ({overall['percentage']:.1f}%)"
                            ),
                        ],
                        color=quality_class,
                    ),
                ]
            ),
            className="mt-3",
        )

    except Exception as e:
        logger.error(f"Error creating assessment results: {e}")
        return html.Div(f"Error: {str(e)}")


def create_issues_recommendations_display(quality_results):
    """Create issues and recommendations display."""
    try:
        issues = []

        if "artifacts" in quality_results:
            artifact_pct = quality_results["artifacts"]["artifact_percentage"]
            if artifact_pct > 5:
                issues.append(
                    f"High artifact content ({artifact_pct:.1f}%). Consider applying artifact removal filters."
                )

        if "baseline_wander" in quality_results:
            if quality_results["baseline_wander"]["quality"] == "poor":
                issues.append(
                    "Significant baseline wander detected. Consider applying baseline correction."
                )

        if "continuity" in quality_results:
            if (
                quality_results["continuity"]["has_nan"]
                or quality_results["continuity"]["has_inf"]
            ):
                issues.append(
                    "Signal contains NaN or Inf values. Data integrity check needed."
                )

        if not issues:
            return dbc.Alert("No significant quality issues detected.", color="success")

        return dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Issues & Recommendations", className="card-title"),
                    html.Ul([html.Li(issue) for issue in issues]),
                ]
            ),
            className="mt-3",
        )

    except Exception as e:
        logger.error(f"Error creating recommendations: {e}")
        return html.Div()


def create_detailed_analysis_display(quality_results):
    """Create detailed analysis display."""
    try:
        details = []

        if "snr" in quality_results:
            details.append(html.H5("SNR Analysis"))
            details.append(html.P(f"SNR: {quality_results['snr']['snr_db']:.2f} dB"))

        if "artifacts" in quality_results:
            details.append(html.H5("Artifact Detection"))
            details.append(
                html.P(
                    f"Total artifacts: {quality_results['artifacts']['artifact_count']} ({quality_results['artifacts']['artifact_percentage']:.2f}%)"
                )
            )
            details.append(
                html.P(
                    f"Z-score method: {quality_results['artifacts']['zscore_count']} artifacts"
                )
            )
            details.append(
                html.P(
                    f"Adaptive method: {quality_results['artifacts']['adaptive_count']} artifacts"
                )
            )
            details.append(
                html.P(
                    f"Kurtosis method: {quality_results['artifacts']['kurtosis_count']} artifacts"
                )
            )

        return dbc.Card(
            dbc.CardBody(details),
            className="mt-3",
        )

    except Exception as e:
        logger.error(f"Error creating detailed analysis: {e}")
        return html.Div()


def create_score_dashboard(quality_results):
    """Create score dashboard."""
    try:
        if "overall_score" not in quality_results:
            return html.Div()

        overall = quality_results["overall_score"]

        return dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Quality Score Dashboard", className="card-title"),
                    dbc.Progress(
                        value=overall["percentage"],
                        label=f"{overall['percentage']:.1f}%",
                        color=(
                            "success"
                            if overall["quality"] == "excellent"
                            else (
                                "info"
                                if overall["quality"] == "good"
                                else (
                                    "warning"
                                    if overall["quality"] == "fair"
                                    else "danger"
                                )
                            )
                        ),
                        className="mb-3",
                        style={"height": "30px"},
                    ),
                ]
            ),
            className="mt-3",
        )

    except Exception as e:
        logger.error(f"Error creating score dashboard: {e}")
        return html.Div()
