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
from dash import Input, Output, State, callback_context, no_update, html, dash, ALL
from dash.exceptions import PreventUpdate
from scipy import signal
import dash_bootstrap_components as dbc
import logging

# Import plot utilities for performance optimization
try:
    from vitalDSP_webapp.utils.plot_utils import limit_plot_data, check_plot_data_size
except ImportError:
    # Fallback if plot_utils not available
    def limit_plot_data(
        time_axis, signal_data, max_duration=300, max_points=10000, start_time=None
    ):
        """Fallback implementation of limit_plot_data"""
        return time_axis, signal_data

    def check_plot_data_size(time_axis, signal_data):
        """Fallback implementation"""
        return True


logger = logging.getLogger(__name__)


def register_quality_callbacks(app):
    """Register all signal quality assessment callbacks using vitalDSP."""
    logger.info("=== REGISTERING vitalDSP SIGNAL QUALITY CALLBACKS ===")

    # Import vitalDSP modules
    _import_vitaldsp_modules()

    # SQI parameter update callback
    @app.callback(
        Output("quality-sqi-parameters-container", "children"),
        [Input("quality-sqi-type", "value")],
    )
    def update_sqi_parameters(sqi_type):
        """Update SQI parameter inputs based on selected SQI type."""
        from .quality_sqi_functions import get_sqi_parameters_layout

        if not sqi_type:
            return []

        return get_sqi_parameters_layout(sqi_type)

    # SQI parameter storage callback
    @app.callback(
        Output("store-quality-sqi-params", "data"),
        [Input({"type": "sqi-param", "param": ALL}, "value")],
        [
            State({"type": "sqi-param", "param": ALL}, "id"),
            State("quality-sqi-type", "value"),
        ],
        prevent_initial_call=True,
    )
    def store_sqi_parameters(param_values, param_ids, sqi_type):
        """Store SQI parameters for use in analysis."""
        params = {}
        if param_ids and param_values:
            for param_id, value in zip(param_ids, param_values):
                param_name = param_id["param"]
                params[param_name] = value
        params["sqi_type"] = sqi_type
        return params

    # Conditional threshold inputs callback
    @app.callback(
        [
            Output({"type": "sqi-param-threshold-row", "param": "single"}, "style"),
            Output({"type": "sqi-param-threshold-row", "param": "range"}, "style"),
        ],
        [Input({"type": "sqi-param", "param": "threshold_type"}, "value")],
    )
    def update_threshold_inputs_visibility(threshold_type):
        """Show/hide threshold inputs based on threshold type."""
        if threshold_type == "range":
            return {"display": "none"}, {"display": "block"}
        else:
            return {"display": "block"}, {"display": "none"}

    # Navigation callback for position slider
    @app.callback(
        Output("quality-start-position", "value"),
        [
            Input("quality-btn-nudge-m10", "n_clicks"),
            Input("quality-btn-nudge-m5", "n_clicks"),
            Input("quality-btn-center", "n_clicks"),
            Input("quality-btn-nudge-p5", "n_clicks"),
            Input("quality-btn-nudge-p10", "n_clicks"),
        ],
        [State("quality-start-position", "value")],
    )
    def update_quality_position_nudge(
        nudge_m10, nudge_m5, center, nudge_p5, nudge_p10, start_position
    ):
        """Update time position based on nudge button clicks."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        start_position = start_position if start_position is not None else 0

        if trigger_id == "quality-btn-nudge-m10":
            return max(0, start_position - 10)
        elif trigger_id == "quality-btn-nudge-m5":
            return max(0, start_position - 5)
        elif trigger_id == "quality-btn-center":
            return 50  # Center at 50%
        elif trigger_id == "quality-btn-nudge-p5":
            return min(100, start_position + 5)
        elif trigger_id == "quality-btn-nudge-p10":
            return min(100, start_position + 10)

        return start_position

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
            Input("quality-metrics-bins", "value"),
            Input("quality-metrics-scope", "value"),
        ],
        [
            State("url", "pathname"),
            State("quality-start-position", "value"),  # Position slider (0-100%)
            State("quality-duration", "value"),  # Duration dropdown
            State("quality-signal-type", "value"),
            State("quality-signal-source", "value"),  # Signal source selector
            State("quality-sqi-type", "value"),  # NEW: SQI type selector
            State("quality-analysis-options", "value"),  # Analysis options
            State("store-quality-sqi-params", "data"),  # NEW: SQI parameters from store
            State("store-filtered-signal", "data"),  # Filtered signal data
            State("store-quality-data", "data"),  # Store full signal data
            State("store-quality-results", "data"),  # Store for quality results
        ],
    )
    def quality_assessment_callback(
        n_clicks,
        bins_count,  # NEW: Number of bins for histogram
        metrics_scope,  # NEW: segment vs entire
        pathname,
        start_position,  # Position slider (0-100%)
        duration,  # Duration dropdown
        signal_type,
        signal_source,  # Signal source selector
        sqi_type,  # NEW: SQI type
        analysis_options,  # Analysis options
        sqi_params,  # NEW: SQI parameters
        filtered_signal_data,  # Filtered signal data
        stored_quality_data,  # Store for quality data
        stored_quality_results,  # Store for quality results
    ):
        """Main callback for signal quality assessment using vitalDSP."""
        logger.info("=== vitalDSP QUALITY ASSESSMENT CALLBACK TRIGGERED ===")

        # Check which input triggered the callback
        ctx = callback_context
        trigger_id = None
        if ctx.triggered:
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

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

        # Check if button was clicked (for initial analysis)
        if n_clicks is None:
            # If metrics controls changed, we still need to check if we have stored results
            if trigger_id in ["quality-metrics-bins", "quality-metrics-scope"]:
                # Try to use stored results if available
                if stored_quality_data and stored_quality_data.get("signal"):
                    logger.info(
                        "Metrics controls changed - updating metrics plot with stored data"
                    )
                    # We'll update metrics plot with stored data below
                    pass
                else:
                    return (
                        create_empty_figure(),
                        create_empty_figure(),
                        "Please click 'Assess Signal Quality' first.",
                        "",
                        "",
                        "",
                        stored_quality_data if stored_quality_data else None,
                        None,
                    )
            else:
                return (
                    create_empty_figure(),
                    create_empty_figure(),
                    "Click 'Assess Signal Quality' to analyze your signal.",
                    "",
                    "",
                    "",
                    stored_quality_data if stored_quality_data else None,
                    None,
                )

        # If only metrics controls changed (bins or scope), reuse stored results if available
        if n_clicks is None and trigger_id in [
            "quality-metrics-bins",
            "quality-metrics-scope",
        ]:
            logger.info("Metrics controls updated - attempting to reuse stored results")
            logger.info(
                f"stored_quality_results exists: {stored_quality_results is not None}"
            )
            if stored_quality_results:
                logger.info(
                    f"stored_quality_results keys: {stored_quality_results.keys() if isinstance(stored_quality_results, dict) else 'not a dict'}"
                )
                logger.info(
                    f"sqi_values in results: {'sqi_values' in stored_quality_results if isinstance(stored_quality_results, dict) else False}"
                )

            # Check if we have stored results to reuse
            if (
                stored_quality_results
                and isinstance(stored_quality_results, dict)
                and stored_quality_results.get("sqi_values")
            ):
                logger.info(
                    f"Using stored results for metrics plot update with {len(stored_quality_results.get('sqi_values', []))} SQI values"
                )

                # Convert bins to int
                try:
                    bins = int(bins_count) if bins_count else 30
                except (ValueError, TypeError):
                    bins = 30

                # Get stored results
                stored_signal_data = (
                    stored_quality_data.get("signal") if stored_quality_data else None
                )
                stored_sampling_freq = (
                    stored_quality_data.get("sampling_freq")
                    if stored_quality_data
                    else 100
                )

                if stored_signal_data:
                    # Get stored filtered signal and signal type if available
                    stored_filtered_signal = (
                        stored_quality_data.get("filtered_signal")
                        if stored_quality_data
                        else None
                    )
                    stored_signal_type = (
                        stored_quality_data.get("signal_type", "PPG")
                        if stored_quality_data
                        else "PPG"
                    )

                    # Recreate plots with updated bins/scope
                    main_plot = create_quality_main_plot(
                        np.array(stored_signal_data),
                        stored_quality_results,
                        stored_sampling_freq,
                        filtered_signal=(
                            np.array(stored_filtered_signal)
                            if stored_filtered_signal
                            else None
                        ),
                        signal_type=stored_signal_type,
                    )

                    # For metrics plot, use appropriate data based on scope
                    scope = metrics_scope if metrics_scope else "segment"

                    # Check if we have stored entire signal quality results
                    if (
                        scope == "entire"
                        and stored_quality_data
                        and "entire_quality_results" in stored_quality_data
                    ):
                        entire_quality_results = stored_quality_data.get(
                            "entire_quality_results"
                        )
                        if entire_quality_results:
                            metrics_plot = create_quality_metrics_plot(
                                entire_quality_results, bins=bins
                            )
                        else:
                            # Fallback to segment results
                            metrics_plot = create_quality_metrics_plot(
                                stored_quality_results, bins=bins
                            )
                    else:
                        metrics_plot = create_quality_metrics_plot(
                            stored_quality_results, bins=bins
                        )

                    # Create results displays
                    assessment_results = create_assessment_results_display(
                        stored_quality_results
                    )
                    issues_recommendations = create_issues_recommendations_display(
                        stored_quality_results
                    )
                    detailed_analysis = create_detailed_analysis_display(
                        stored_quality_results
                    )
                    score_dashboard = create_score_dashboard(stored_quality_results)

                    return (
                        main_plot,
                        metrics_plot,
                        assessment_results,
                        issues_recommendations,
                        detailed_analysis,
                        score_dashboard,
                        stored_quality_data,
                        stored_quality_results,
                    )
                else:
                    return (
                        create_empty_figure(),
                        create_empty_figure(),
                        "Please click 'Assess Signal Quality' first.",
                        "",
                        "",
                        "",
                        stored_quality_data,
                        None,
                    )

        # Check if data is available
        try:
            from vitalDSP_webapp.services.data.enhanced_data_service import (
                get_enhanced_data_service,
            )

            data_service = get_enhanced_data_service()

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

            # Get the most recent data entry (data_service returns dict with IDs as keys)
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]
            logger.info(f"Latest data ID: {latest_data_id}")

            # Get column mapping
            column_mapping = data_service.get_column_mapping(latest_data_id)
            logger.info(f"Column mapping: {column_mapping}")

            if not column_mapping:
                return (
                    create_empty_figure(),
                    create_empty_figure(),
                    "No column mapping found. Please upload data first.",
                    "",
                    "",
                    "",
                    None,
                    None,
                )

            # Get the actual data DataFrame
            df = data_service.get_data(latest_data_id)
            if df is None or df.empty:
                return (
                    create_empty_figure(),
                    create_empty_figure(),
                    "Data is empty. Please upload valid data.",
                    "",
                    "",
                    "",
                    None,
                    None,
                )

            logger.info(f"Data shape: {df.shape}, columns: {df.columns.tolist()}")

            # Determine which signal to use (filtered or original)
            signal_data = None
            time_data = None

            if signal_source == "filtered" and filtered_signal_data:
                logger.info("Attempting to use filtered signal data")
                signal_data = np.array(filtered_signal_data.get("filtered_signal", []))
                if len(signal_data) == 0:
                    logger.warning("Filtered signal is empty, falling back to original")
                    signal_data = None

            # Fall back to original signal if filtered not available
            if signal_data is None:
                logger.info("Using original signal data from DataFrame")
                # Get time column
                time_col = column_mapping.get("time")
                if not time_col or time_col not in df.columns:
                    # Try default time column names
                    for col in ["time", "Time", "timestamp"]:
                        if col in df.columns:
                            time_col = col
                            break

                if not time_col:
                    return (
                        create_empty_figure(),
                        create_empty_figure(),
                        "No time column found in data",
                        "",
                        "",
                        "",
                        None,
                        None,
                    )

                time_data = df[time_col].values

                # Get signal column based on signal type
                signal_col = None
                if signal_type == "PPG":
                    signal_col = column_mapping.get("signal") or column_mapping.get(
                        "ppg"
                    )
                elif signal_type == "ECG":
                    signal_col = column_mapping.get("ecg")

                # Try common column names if mapping didn't work
                if not signal_col or signal_col not in df.columns:
                    for col in ["signal", "waveform", "ppg", "ecg", "RED", "value"]:
                        if col in df.columns:
                            signal_col = col
                            logger.info(f"Using column: {col}")
                            break

                if not signal_col:
                    return (
                        create_empty_figure(),
                        create_empty_figure(),
                        f"No signal column found for type {signal_type}",
                        "",
                        "",
                        "",
                        None,
                        None,
                    )

                signal_data = df[signal_col].values

            # If we have filtered signal but no time data yet, get time from df
            if time_data is None:
                time_col = column_mapping.get("time")
                if not time_col or time_col not in df.columns:
                    for col in ["time", "Time", "timestamp"]:
                        if col in df.columns:
                            time_col = col
                            break
                if time_col and time_col in df.columns:
                    time_data = df[time_col].values
                else:
                    # Generate time data based on sampling frequency
                    sampling_freq = latest_data.get("info", {}).get(
                        "sampling_freq", 100
                    )
                    time_data = np.arange(len(signal_data)) / sampling_freq

            # Ensure time and signal have same length
            min_len = min(len(time_data), len(signal_data))
            time_data = time_data[:min_len]
            signal_data = signal_data[:min_len]

            # Ensure time_data and signal_data are numeric arrays
            time_data = np.array(time_data, dtype=np.float64)
            signal_data = np.array(signal_data, dtype=np.float64)

            # Check if time data is in Unix timestamp format (very large values)
            # If so, normalize it to start from 0
            if time_data[0] > 1e9:  # Likely Unix timestamp
                logger.info(
                    f"Detected Unix timestamp format, normalizing time data (first value: {time_data[0]:.2e})"
                )

                # Normalize to start from 0 FIRST (to get relative time)
                time_data = time_data - time_data[0]

                # Calculate the time span
                time_span = time_data[-1]
                logger.info(f"Time span after normalization: {time_span:.2e}")

                # Check if we need to convert units based on the span
                if time_span > 1e9:  # Nanoseconds
                    logger.info("Converting from nanoseconds to seconds")
                    time_data = time_data / 1e9
                elif time_span > 1e6:  # Microseconds
                    logger.info("Converting from microseconds to seconds")
                    time_data = time_data / 1e6
                elif time_span > 1e3:  # Milliseconds
                    logger.info("Converting from milliseconds to seconds")
                    time_data = time_data / 1000.0

            total_duration = float(time_data[-1] - time_data[0])
            logger.info(
                f"Loaded data: {len(signal_data)} samples, duration: {total_duration:.2f}s"
            )

            # STORE ORIGINAL SIGNAL DATA BEFORE WINDOWING (for entire signal SQI computation)
            original_signal_data = signal_data.copy()
            original_time_data = time_data.copy()

            # Calculate sampling frequency from time data
            if len(time_data) > 1:
                sampling_freq = 1 / np.mean(np.diff(time_data))
            else:
                sampling_freq = latest_data.get("info", {}).get("sampling_freq", 100)

            # Calculate start time from position percentage
            start_position = float(start_position) if start_position is not None else 0
            duration = float(duration) if duration is not None else 60
            start_time = (start_position / 100.0) * total_duration
            end_time = start_time + duration

            # Extract time segment
            mask = (time_data >= start_time) & (time_data <= end_time)
            if not np.any(mask):
                return (
                    create_empty_figure(),
                    create_empty_figure(),
                    f"No data in time window {start_time:.1f}-{end_time:.1f}s",
                    "",
                    "",
                    "",
                    None,
                    None,
                )

            windowed_time = time_data[mask]
            windowed_signal = signal_data[mask]

            logger.info(
                f"Windowed data: {len(windowed_signal)} samples from {start_time:.1f}s to {end_time:.1f}s"
            )

            # Use windowed signal for quality assessment
            signal_data = windowed_signal

            # Perform quality assessment using vitalDSP SQI
            from .quality_sqi_functions import compute_sqi

            # Get SQI parameters (with defaults if not set)
            if not sqi_params:
                sqi_params = {"sqi_type": sqi_type or "snr_sqi"}

            logger.info(f"Computing SQI: {sqi_type}, params: {sqi_params}")

            quality_results = compute_sqi(
                signal_data,
                sqi_type or "snr_sqi",
                sqi_params,
                sampling_freq,
            )

            logger.info(
                f"Quality results computed: {len(quality_results.get('sqi_values', []))} SQI values"
            )
            logger.info(f"Quality results keys: {quality_results.keys()}")

            # Also compute SQI for entire signal (before windowing) for scope toggle
            entire_signal_data = None
            entire_sqi_results = None

            # Use original_signal_data that we stored before windowing
            if original_signal_data is not None and len(original_signal_data) > 0:
                entire_signal_data = original_signal_data

            # If we have entire signal, compute SQI for it
            window_size_param = sqi_params.get("window_size") if sqi_params else None
            window_size_param = (
                int(window_size_param)
                if window_size_param not in [None, "", 0]
                else 1000
            )
            if (
                entire_signal_data is not None
                and len(entire_signal_data) > window_size_param
            ):
                logger.info(
                    f"Computing SQI for entire signal: {len(entire_signal_data)} samples"
                )
                from .quality_sqi_functions import compute_sqi

                try:
                    entire_sqi_results = compute_sqi(
                        entire_signal_data,
                        sqi_type or "snr_sqi",
                        sqi_params,
                        sampling_freq,
                    )
                    logger.info(
                        f"Entire signal SQI computed: {len(entire_sqi_results.get('sqi_values', []))} values"
                    )
                except Exception as e:
                    logger.warning(f"Could not compute SQI for entire signal: {e}")

            # Get filtered signal if available and extract the same window
            filtered_signal = None
            logger.info(
                f"Checking for filtered signal data: {filtered_signal_data is not None}"
            )

            # PRIORITY 1: Try to get FULL filtered signal from Enhanced Data Service first
            # This ensures we have the complete filtered signal and can apply any windowing
            full_filtered_signal_from_service = data_service.get_filtered_data(
                latest_data_id
            )
            if (
                full_filtered_signal_from_service is not None
                and len(full_filtered_signal_from_service) > 0
            ):
                logger.info(
                    f"✅ Found FULL filtered signal from Enhanced Data Service: {len(full_filtered_signal_from_service)} samples"
                )

                # Apply the same windowing to the full filtered signal
                if len(full_filtered_signal_from_service) == len(original_signal_data):
                    # Both signals are the same length - apply the same windowing mask
                    filtered_signal = full_filtered_signal_from_service[mask]
                    logger.info(
                        f"✅ Successfully windowed full filtered signal from Data Service: {len(filtered_signal)} samples"
                    )

                    # Log signal value statistics for debugging
                    logger.info(
                        f"📊 FULL Filtered signal stats - min: {np.min(full_filtered_signal_from_service):.4f}, max: {np.max(full_filtered_signal_from_service):.4f}, mean: {np.mean(full_filtered_signal_from_service):.4f}"
                    )
                    logger.info(
                        f"📊 WINDOWED Filtered signal stats - min: {np.min(filtered_signal):.4f}, max: {np.max(filtered_signal):.4f}, mean: {np.mean(filtered_signal):.4f}"
                    )
                else:
                    logger.warning(
                        f"Full filtered signal length ({len(full_filtered_signal_from_service)}) doesn't match original signal ({len(original_signal_data)})"
                    )
                    # Still try to window it based on time indices if possible
                    if len(full_filtered_signal_from_service) >= len(mask):
                        filtered_signal = full_filtered_signal_from_service[mask]
                        logger.info(
                            f"Applied mask to full filtered signal: {len(filtered_signal)} samples"
                        )

            # PRIORITY 2: Fallback to Dash store if Enhanced Data Service doesn't have it
            if (
                filtered_signal is None
                and filtered_signal_data
                and isinstance(filtered_signal_data, dict)
            ):
                logger.info(f"Filtered signal data keys: {filtered_signal_data.keys()}")
                logger.info(
                    "Enhanced Data Service didn't have full filtered signal, checking Dash store..."
                )

                # Extract filtered signal from store
                if "signal" in filtered_signal_data:
                    signal_list = filtered_signal_data["signal"]

                    # Check if signal list is not empty
                    if signal_list and len(signal_list) > 0:
                        full_filtered_signal = np.array(signal_list)
                        logger.info(
                            f"Found filtered signal in Dash store with {len(full_filtered_signal)} samples"
                        )

                        # Extract the same window from filtered signal as we did for original signal
                        # The mask was created from the ORIGINAL signal's time_data before windowing
                        if len(full_filtered_signal) == len(original_signal_data):
                            # Both signals are the same length - apply the same windowing
                            filtered_signal = full_filtered_signal[mask]
                            logger.info(
                                f"Windowed filtered signal: {len(filtered_signal)} samples (same as original windowed: {len(signal_data)})"
                            )

                            # Verify the windowed signal is valid
                            if len(filtered_signal) > 0:
                                logger.info(
                                    f"✅ Successfully extracted filtered signal window: {len(filtered_signal)} samples"
                                )
                            else:
                                logger.warning(
                                    "Windowed filtered signal is empty after masking"
                                )
                                filtered_signal = None
                        else:
                            # Stored signal doesn't match - need to filter current window on-the-fly
                            # This ensures the filtered signal matches the current start position and duration
                            logger.info(
                                f"Filtered signal from Dash store is windowed ({len(full_filtered_signal)} samples), current window is {len(signal_data)} samples"
                            )
                            logger.info(
                                "Filtering current window on-the-fly to match start position and duration..."
                            )

                            try:
                                # Get filter info from data service to re-apply filter
                                filter_info = data_service.get_filter_info(
                                    latest_data_id
                                )
                                if filter_info:
                                    logger.info(f"Retrieved filter info: {filter_info}")
                                    # Also check if Dash store has additional filter parameters
                                    if (
                                        isinstance(filtered_signal_data, dict)
                                        and "filter_params" in filtered_signal_data
                                    ):
                                        filter_params = filtered_signal_data.get(
                                            "filter_params", {}
                                        )
                                        # Merge additional parameters into filter_info
                                        if "parameters" in filter_info:
                                            filter_info["parameters"].update(
                                                filter_params
                                            )
                                        logger.info(
                                            f"Enhanced filter info with Dash store params: {filter_info}"
                                        )

                                    # Filter the current windowed signal using the stored filter parameters
                                    filtered_signal = _apply_filter_to_signal(
                                        signal_data,
                                        filter_info,
                                        sampling_freq,
                                        signal_type,
                                    )
                                    if filtered_signal is not None:
                                        logger.info(
                                            f"✅ Successfully filtered current window on-the-fly: {len(filtered_signal)} samples"
                                        )
                                    else:
                                        logger.warning(
                                            "Failed to filter current window on-the-fly"
                                        )
                                        # Fallback: use stored signal if length matches (within tolerance)
                                        if (
                                            abs(
                                                len(full_filtered_signal)
                                                - len(signal_data)
                                            )
                                            <= 2
                                        ):
                                            filtered_signal = (
                                                full_filtered_signal.copy()
                                            )
                                            if len(filtered_signal) != len(signal_data):
                                                min_len = min(
                                                    len(filtered_signal),
                                                    len(signal_data),
                                                )
                                                filtered_signal = filtered_signal[
                                                    :min_len
                                                ]
                                                signal_data = signal_data[:min_len]
                                            logger.info(
                                                f"Using stored filtered signal as fallback: {len(filtered_signal)} samples"
                                            )
                                else:
                                    logger.warning(
                                        "No filter info available, cannot filter on-the-fly"
                                    )
                                    # Fallback: use stored signal if length matches
                                    if (
                                        abs(
                                            len(full_filtered_signal) - len(signal_data)
                                        )
                                        <= 2
                                    ):
                                        filtered_signal = full_filtered_signal.copy()
                                        if len(filtered_signal) != len(signal_data):
                                            min_len = min(
                                                len(filtered_signal), len(signal_data)
                                            )
                                            filtered_signal = filtered_signal[:min_len]
                                            signal_data = signal_data[:min_len]
                                        logger.info(
                                            f"Using stored filtered signal as fallback (no filter info): {len(filtered_signal)} samples"
                                        )
                            except Exception as e:
                                logger.warning(
                                    f"Error filtering current window on-the-fly: {e}",
                                    exc_info=True,
                                )
                                # Fallback: use stored signal if length matches
                                if (
                                    abs(len(full_filtered_signal) - len(signal_data))
                                    <= 2
                                ):
                                    filtered_signal = full_filtered_signal.copy()
                                    if len(filtered_signal) != len(signal_data):
                                        min_len = min(
                                            len(filtered_signal), len(signal_data)
                                        )
                                        filtered_signal = filtered_signal[:min_len]
                                        signal_data = signal_data[:min_len]
                                    logger.info(
                                        f"Using stored filtered signal as fallback after error: {len(filtered_signal)} samples"
                                    )
                    else:
                        logger.warning(
                            f"Filtered signal data 'signal' key exists but is empty or None (length: {len(signal_list) if signal_list else 0})"
                        )
                else:
                    logger.warning(
                        f"'signal' key not found in filtered_signal_data. Available keys: {list(filtered_signal_data.keys()) if isinstance(filtered_signal_data, dict) else 'Not a dict'}"
                    )

            # If not found in Dash store, try to get from data service
            if filtered_signal is None:
                try:
                    logger.info(
                        "Attempting to retrieve filtered signal from data service..."
                    )
                    # Use the same data_service that was already retrieved at the start of the callback
                    # latest_data_id is already available from earlier in the callback
                    logger.info(
                        f"Retrieving filtered signal from data service for data ID: {latest_data_id}"
                    )

                    # Check if filtered data exists
                    if data_service.has_filtered_data(latest_data_id):
                        stored_filtered_signal = data_service.get_filtered_data(
                            latest_data_id
                        )
                        if (
                            stored_filtered_signal is not None
                            and len(stored_filtered_signal) > 0
                        ):
                            logger.info(
                                f"✅ Found filtered signal in data service with {len(stored_filtered_signal)} samples"
                            )

                            # Determine how to use the stored filtered signal
                            # The stored filtered signal might be:
                            # 1. Full-length signal (same as original_signal_data)
                            # 2. Windowed signal (same as signal_data, or close to it)

                            stored_len = len(stored_filtered_signal)
                            windowed_len = len(signal_data)
                            full_len = len(original_signal_data)

                            logger.info(
                                f"Signal length comparison: stored={stored_len}, windowed={windowed_len}, full={full_len}"
                            )

                            if stored_len == full_len:
                                # Stored signal is full-length - apply the same windowing
                                filtered_signal = stored_filtered_signal[mask]
                                logger.info(
                                    f"Windowed full-length filtered signal: {len(filtered_signal)} samples (matches windowed original: {len(signal_data)})"
                                )
                            else:
                                # Stored signal doesn't match - need to filter current window on-the-fly
                                logger.info(
                                    f"Stored filtered signal is windowed ({stored_len} samples) but current window is {windowed_len} samples"
                                )
                                logger.info(
                                    "Filtering current window on-the-fly using stored filter parameters to match start position and duration..."
                                )

                                try:
                                    # Get filter info to re-apply filter
                                    filter_info = data_service.get_filter_info(
                                        latest_data_id
                                    )
                                    if filter_info:
                                        logger.info(
                                            f"Retrieved filter info: {filter_info}"
                                        )
                                        # Filter the current windowed signal using the stored filter parameters
                                        filtered_signal = _apply_filter_to_signal(
                                            signal_data,
                                            filter_info,
                                            sampling_freq,
                                            signal_type,
                                        )
                                        if filtered_signal is not None:
                                            logger.info(
                                                f"✅ Successfully filtered current window on-the-fly: {len(filtered_signal)} samples"
                                            )
                                        else:
                                            logger.warning(
                                                "Failed to filter current window on-the-fly"
                                            )
                                            # Fallback: use stored signal if length is very close (within 2 samples)
                                            if abs(stored_len - windowed_len) <= 2:
                                                filtered_signal = (
                                                    stored_filtered_signal.copy()
                                                )
                                                if len(filtered_signal) != len(
                                                    signal_data
                                                ):
                                                    min_len = min(
                                                        len(filtered_signal),
                                                        len(signal_data),
                                                    )
                                                    filtered_signal = filtered_signal[
                                                        :min_len
                                                    ]
                                                    signal_data = signal_data[:min_len]
                                                logger.info(
                                                    f"Using stored filtered signal as fallback: {len(filtered_signal)} samples"
                                                )
                                    else:
                                        logger.warning(
                                            "No filter info available, cannot filter on-the-fly"
                                        )
                                        # Fallback: use stored signal if length matches
                                        if abs(stored_len - windowed_len) <= 2:
                                            filtered_signal = (
                                                stored_filtered_signal.copy()
                                            )
                                            if len(filtered_signal) != len(signal_data):
                                                min_len = min(
                                                    len(filtered_signal),
                                                    len(signal_data),
                                                )
                                                filtered_signal = filtered_signal[
                                                    :min_len
                                                ]
                                                signal_data = signal_data[:min_len]
                                            logger.info(
                                                f"Using stored filtered signal as fallback (no filter info): {len(filtered_signal)} samples"
                                            )
                                except Exception as e:
                                    logger.warning(
                                        f"Error filtering current window on-the-fly: {e}",
                                        exc_info=True,
                                    )
                                    # Fallback: use stored signal if length is close
                                    if abs(stored_len - windowed_len) <= 2:
                                        filtered_signal = stored_filtered_signal.copy()
                                        if len(filtered_signal) != len(signal_data):
                                            min_len = min(
                                                len(filtered_signal), len(signal_data)
                                            )
                                            filtered_signal = filtered_signal[:min_len]
                                            signal_data = signal_data[:min_len]
                                        logger.info(
                                            f"Using stored filtered signal as fallback after error: {len(filtered_signal)} samples"
                                        )
                        else:
                            logger.warning(
                                "Filtered signal retrieved from data service but is None or empty"
                            )
                    else:
                        logger.info(
                            f"No filtered data available in data service for data ID: {latest_data_id}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Error retrieving filtered signal from data service: {e}",
                        exc_info=True,
                    )

            if filtered_signal is None:
                logger.info(
                    "No filtered signal data available in store or data service"
                )

            logger.info(
                f"Final filtered_signal status: {'Available' if filtered_signal is not None else 'None'} ({len(filtered_signal) if filtered_signal is not None else 0} samples)"
            )

            # Store windowed signal for segment analysis
            windowed_signal_store = {
                "signal": signal_data.tolist(),
                "sampling_freq": sampling_freq,
                "signal_type": signal_type,  # Store signal type
                "filtered_signal": (
                    filtered_signal.tolist() if filtered_signal is not None else None
                ),  # Store filtered signal
                "entire_signal": (
                    entire_signal_data.tolist()
                    if entire_signal_data is not None
                    else None
                ),
                "entire_sqi_values": (
                    entire_sqi_results.get("sqi_values", [])
                    if entire_sqi_results
                    else []
                ),
                "entire_quality_results": (
                    entire_sqi_results if entire_sqi_results else None
                ),  # Store full entire results
            }

            # Create visualizations
            try:
                logger.info(
                    f"Creating quality main plot with filtered_signal={'provided' if filtered_signal is not None else 'None'} ({len(filtered_signal) if filtered_signal is not None else 0} samples)"
                )
                main_plot = create_quality_main_plot(
                    signal_data,
                    quality_results,
                    sampling_freq,
                    filtered_signal=filtered_signal,
                    signal_type=signal_type,
                )
                logger.info(
                    f"Quality main plot created successfully. Plot has {len(main_plot.data) if hasattr(main_plot, 'data') else 'N/A'} traces"
                )
            except Exception as e:
                logger.error(
                    f"Error creating quality main plot with filtered signal: {e}",
                    exc_info=True,
                )
                # Fallback: try without filtered signal
                logger.info("Attempting to create plot without filtered signal")
                try:
                    main_plot = create_quality_main_plot(
                        signal_data,
                        quality_results,
                        sampling_freq,
                        filtered_signal=None,  # Force no filtered signal
                        signal_type=signal_type,
                    )
                    logger.info("Successfully created plot without filtered signal")
                except Exception as e2:
                    logger.error(
                        f"Error creating plot without filtered signal: {e2}",
                        exc_info=True,
                    )
                    main_plot = create_empty_figure()

            # For metrics plot, use appropriate data based on scope
            # Convert bins to int (handles string from Select component)
            try:
                bins = int(bins_count) if bins_count else 30
            except (ValueError, TypeError):
                bins = 30

            scope = metrics_scope if metrics_scope else "segment"

            # If entire signal requested, check if we have stored full signal data
            if (
                scope == "entire"
                and stored_quality_data
                and "entire_quality_results" in stored_quality_data
            ):
                # Use stored entire signal SQI results (includes all metrics, not just sqi_values)
                entire_quality_results = stored_quality_data.get(
                    "entire_quality_results"
                )
                if entire_quality_results:
                    metrics_plot = create_quality_metrics_plot(
                        entire_quality_results, bins=bins
                    )
                else:
                    # Fallback: use segment results if entire results not available
                    metrics_plot = create_quality_metrics_plot(
                        quality_results, bins=bins
                    )
            else:
                metrics_plot = create_quality_metrics_plot(quality_results, bins=bins)

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
                windowed_signal_store,
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

        # SNR Assessment - Enhanced with vitalDSP advanced methods
        if "snr" in quality_metrics or not quality_metrics:
            try:
                from vitalDSP.signal_quality_assessment.snr_computation import (
                    snr_power_ratio,
                    snr_peak_to_peak,
                    snr_mean_square,
                )

                # For standalone SNR without processed signal, use signal power vs noise estimation
                # Estimate noise as high-frequency content using vitalDSP
                from vitalDSP.filtering.signal_filtering import SignalFiltering

                # High-pass filter to extract noise
                nyq = sampling_freq / 2
                if nyq > 10:  # Only if we have sufficient frequency range
                    cutoff = min(10, nyq * 0.8)  # 10 Hz or 80% of Nyquist

                    # Use vitalDSP for filtering
                    sf_high = SignalFiltering(signal_data)
                    noise_estimate = sf_high.highpass(
                        cutoff=cutoff, fs=sampling_freq, order=4
                    )

                    # Low-pass filter to get signal
                    cutoff_low = max(0.5, cutoff / 20)  # Low cutoff
                    sf_low = SignalFiltering(signal_data)
                    signal_estimate = sf_low.lowpass(
                        cutoff=cutoff_low, fs=sampling_freq, order=4
                    )

                    # Calculate SNR using multiple methods
                    snr_db_power = snr_power_ratio(signal_estimate, noise_estimate)
                    snr_db_peak = snr_peak_to_peak(signal_estimate, noise_estimate)
                    snr_db_mean_sq = snr_mean_square(signal_estimate, noise_estimate)

                    # Use SignalQuality for additional validation
                    sq = SignalQuality(signal_estimate, signal_data)
                    snr_db_sq = sq.snr()

                    # Average the SNR estimates for robustness
                    snr_db = float(
                        np.mean([snr_db_power, snr_db_peak, snr_db_mean_sq, snr_db_sq])
                    )

                    quality_results["snr"] = {
                        "snr_db": snr_db,
                        "snr_power_ratio": float(snr_db_power),
                        "snr_peak_to_peak": float(snr_db_peak),
                        "snr_mean_square": float(snr_db_mean_sq),
                        "snr_signal_quality": float(snr_db_sq),
                        "quality": (
                            "excellent"
                            if snr_db > 20
                            else "good" if snr_db > snr_threshold else "poor"
                        ),
                        "threshold": snr_threshold,
                        "method": "vitalDSP_advanced_multi_method",
                    }
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
                        "method": "fallback_simple",
                    }

            except ImportError:
                logger.warning(
                    "vitalDSP snr_computation not available, using fallback method"
                )
                # Fallback to scipy method
                from scipy.signal import butter, filtfilt

                nyq = sampling_freq / 2
                if nyq > 10:
                    cutoff = min(10, nyq * 0.8)
                    b, a = butter(4, cutoff / nyq, btype="high")
                    noise_estimate = filtfilt(b, a, signal_data)

                    cutoff_low = max(0.5, cutoff / 20)
                    b_low, a_low = butter(4, cutoff_low / nyq, btype="low")
                    signal_estimate = filtfilt(b_low, a_low, signal_data)

                    sq = SignalQuality(signal_estimate, signal_data)
                    snr_db = sq.snr()
                else:
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
                    "method": "scipy_fallback",
                }
            except Exception as e:
                logger.warning(f"SNR calculation failed: {e}, using simple fallback")
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
                    "method": "error_fallback",
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


def _apply_filter_to_signal(signal_data, filter_info, sampling_freq, signal_type="PPG"):
    """Apply filter to signal using stored filter parameters.

    Args:
        signal_data: Signal data to filter (numpy array)
        filter_info: Filter info dictionary from data service
        sampling_freq: Sampling frequency
        signal_type: Signal type (for advanced filters)

    Returns:
        Filtered signal array or None if filtering fails
    """
    try:
        if not filter_info:
            logger.warning("No filter info provided")
            return None

        filter_type = filter_info.get("filter_type", "traditional")
        parameters = filter_info.get("parameters", {})

        logger.info(
            f"Applying filter on-the-fly: type={filter_type}, parameters={parameters}"
        )

        if filter_type == "traditional":
            # Extract traditional filter parameters
            filter_family = parameters.get("filter_family", "butter")
            filter_response = parameters.get("filter_response", "bandpass")
            low_freq = parameters.get("low_freq", 0.5)
            high_freq = parameters.get("high_freq", 5.0)
            filter_order = parameters.get("filter_order", 4)

            # Apply traditional filter using vitalDSP
            from vitalDSP.filtering.signal_filtering import SignalFiltering

            sf = SignalFiltering(signal_data)

            # Normalize frequencies
            nyquist = sampling_freq / 2
            low_freq_norm = low_freq / nyquist
            high_freq_norm = high_freq / nyquist

            # Ensure cutoff frequencies are within valid range
            low_freq_norm = max(0.001, min(low_freq_norm, 0.999))
            high_freq_norm = max(0.001, min(high_freq_norm, 0.999))

            # Apply filter based on response type
            if filter_response == "bandpass":
                filtered_signal = sf.bandpass(
                    lowcut=low_freq_norm,
                    highcut=high_freq_norm,
                    fs=sampling_freq,
                    order=filter_order,
                    filter_type=filter_family,
                )
            elif filter_response == "lowpass":
                filtered_signal = sf.lowpass(
                    cutoff=high_freq_norm,
                    fs=sampling_freq,
                    order=filter_order,
                    filter_type=filter_family,
                )
            elif filter_response == "highpass":
                filtered_signal = sf.highpass(
                    cutoff=low_freq_norm,
                    fs=sampling_freq,
                    order=filter_order,
                    filter_type=filter_family,
                )
            elif filter_response == "bandstop":
                filtered_signal = sf.bandstop(
                    lowcut=low_freq_norm,
                    highcut=high_freq_norm,
                    fs=sampling_freq,
                    order=filter_order,
                    filter_type=filter_family,
                )
            else:
                # Default to bandpass
                filtered_signal = sf.bandpass(
                    lowcut=low_freq_norm,
                    highcut=high_freq_norm,
                    fs=sampling_freq,
                    order=filter_order,
                    filter_type=filter_family,
                )

            logger.info(
                f"✅ Traditional filter applied successfully: {filter_family} {filter_response}"
            )
            return filtered_signal

        elif filter_type == "advanced":
            # For advanced filters, we need more complex logic
            advanced_method = parameters.get("advanced_method")

            if advanced_method == "convolution":
                # Use vitalDSP convolution-based filter for smoothing
                kernel_size = parameters.get(
                    "convolution_kernel_size"
                ) or parameters.get("kernel_size", 5)
                from vitalDSP.filtering.advanced_signal_filtering import (
                    AdvancedSignalFiltering,
                )

                logger.info(
                    f"[vitalDSP] Applying convolution-based filter: kernel_size={kernel_size}"
                )
                af = AdvancedSignalFiltering(signal_data)
                filtered_signal = af.convolution_based_filter(
                    kernel_type="smoothing", kernel_size=kernel_size
                )
                logger.info(f"✅ vitalDSP convolution filter applied successfully")
                return filtered_signal
            else:
                # For other advanced methods, fall back to traditional filter if params available
                filter_family = parameters.get("filter_family", "butter")
                filter_response = parameters.get("filter_response", "bandpass")
                low_freq = parameters.get("low_freq", 0.5)
                high_freq = parameters.get("high_freq", 5.0)
                filter_order = parameters.get("filter_order", 4)

                # Apply traditional filter as fallback
                from vitalDSP.filtering.signal_filtering import SignalFiltering

                sf = SignalFiltering(signal_data)
                nyquist = sampling_freq / 2
                low_freq_norm = max(0.001, min(low_freq / nyquist, 0.999))
                high_freq_norm = max(0.001, min(high_freq / nyquist, 0.999))

                filtered_signal = sf.bandpass(
                    lowcut=low_freq_norm,
                    highcut=high_freq_norm,
                    fs=sampling_freq,
                    order=filter_order,
                    filter_type=filter_family,
                )
                logger.info(
                    f"✅ Applied traditional filter as fallback for advanced method: {advanced_method}"
                )
                return filtered_signal
        else:
            logger.warning(f"Unknown filter type: {filter_type}")
            return None

    except Exception as e:
        logger.error(f"Error applying filter on-the-fly: {e}", exc_info=True)
        return None


def add_critical_points_to_plot(
    fig, signal_data, time_axis_plot, sampling_freq, signal_type, row_num
):
    """Add critical points detection to a subplot using WaveformMorphology.

    Args:
        fig: Plotly figure object
        signal_data: Full signal data (not limited)
        time_axis_plot: Limited time axis for plotting
        sampling_freq: Sampling frequency
        signal_type: Signal type (PPG, ECG, etc.)
        row_num: Row number in subplot to add traces to
    """
    try:
        from vitalDSP.physiological_features.waveform import WaveformMorphology

        # Create waveform morphology object (use FULL data for accurate peak detection)
        wm = WaveformMorphology(
            waveform=signal_data,
            fs=sampling_freq,
            signal_type=signal_type,
            simple_mode=True,
        )

        # Calculate plot range for filtering peaks
        plot_start_time = np.min(time_axis_plot)
        plot_end_time = np.max(time_axis_plot)
        plot_start_idx = int(plot_start_time * sampling_freq)
        plot_end_idx = int(plot_end_time * sampling_freq)

        # Detect critical points based on signal type
        if signal_type == "PPG":
            # For PPG: systolic peaks, dicrotic notches, diastolic peaks
            if hasattr(wm, "systolic_peaks") and wm.systolic_peaks is not None:
                # Filter peaks to only those within the plot range
                valid_peaks = wm.systolic_peaks[
                    (wm.systolic_peaks >= plot_start_idx)
                    & (wm.systolic_peaks < plot_end_idx)
                ]

                if len(valid_peaks) > 0:
                    peak_times = valid_peaks / sampling_freq
                    peak_values = signal_data[valid_peaks]

                    fig.add_trace(
                        go.Scatter(
                            x=peak_times,
                            y=peak_values,
                            mode="markers",
                            name="Systolic Peaks",
                            marker=dict(color="red", size=10, symbol="diamond"),
                            hovertemplate="<b>Systolic Peak:</b> %{y:.2f}<br><b>Time:</b> %{x:.3f}s<extra></extra>",
                            legendgroup="systolic",
                            showlegend=(
                                row_num == 1
                            ),  # Only show legend for first occurrence
                        ),
                        row=row_num,
                        col=1,
                    )

            # Detect and plot dicrotic notches
            try:
                dicrotic_notches = wm.detect_dicrotic_notches()
                if dicrotic_notches is not None and len(dicrotic_notches) > 0:
                    valid_notches = dicrotic_notches[
                        (dicrotic_notches >= plot_start_idx)
                        & (dicrotic_notches < plot_end_idx)
                    ]

                    if len(valid_notches) > 0:
                        notch_times = valid_notches / sampling_freq
                        notch_values = signal_data[valid_notches]

                        fig.add_trace(
                            go.Scatter(
                                x=notch_times,
                                y=notch_values,
                                mode="markers",
                                name="Dicrotic Notches",
                                marker=dict(color="orange", size=8, symbol="circle"),
                                hovertemplate="<b>Dicrotic Notch:</b> %{y:.2f}<br><b>Time:</b> %{x:.3f}s<extra></extra>",
                                legendgroup="dicrotic",
                                showlegend=(row_num == 1),
                            ),
                            row=row_num,
                            col=1,
                        )
            except Exception as e:
                logger.warning(f"Dicrotic notch detection failed: {e}")

            # Detect and plot diastolic peaks
            try:
                diastolic_peaks = wm.detect_diastolic_peak()
                if diastolic_peaks is not None and len(diastolic_peaks) > 0:
                    valid_peaks = diastolic_peaks[
                        (diastolic_peaks >= plot_start_idx)
                        & (diastolic_peaks < plot_end_idx)
                    ]

                    if len(valid_peaks) > 0:
                        peak_times = valid_peaks / sampling_freq
                        peak_values = signal_data[valid_peaks]

                        fig.add_trace(
                            go.Scatter(
                                x=peak_times,
                                y=peak_values,
                                mode="markers",
                                name="Diastolic Peaks",
                                marker=dict(color="green", size=8, symbol="square"),
                                hovertemplate="<b>Diastolic Peak:</b> %{y:.2f}<br><b>Time:</b> %{x:.3f}s<extra></extra>",
                                legendgroup="diastolic",
                                showlegend=(row_num == 1),
                            ),
                            row=row_num,
                            col=1,
                        )
            except Exception as e:
                logger.warning(f"Diastolic peak detection failed: {e}")

        elif signal_type == "ECG":
            # For ECG: R peaks, P peaks, T peaks
            if hasattr(wm, "r_peaks") and wm.r_peaks is not None:
                valid_peaks = wm.r_peaks[
                    (wm.r_peaks >= plot_start_idx) & (wm.r_peaks < plot_end_idx)
                ]

                if len(valid_peaks) > 0:
                    peak_times = valid_peaks / sampling_freq
                    peak_values = signal_data[valid_peaks]

                    fig.add_trace(
                        go.Scatter(
                            x=peak_times,
                            y=peak_values,
                            mode="markers",
                            name="R Peaks",
                            marker=dict(color="red", size=10, symbol="diamond"),
                            hovertemplate="<b>R Peak:</b> %{y:.2f}<br><b>Time:</b> %{x:.3f}s<extra></extra>",
                            legendgroup="r_peaks",
                            showlegend=(row_num == 1),
                        ),
                        row=row_num,
                        col=1,
                    )

            # Detect and plot P peaks
            try:
                p_peaks = wm.detect_p_peak()
                if p_peaks is not None and len(p_peaks) > 0:
                    valid_peaks = p_peaks[
                        (p_peaks >= plot_start_idx) & (p_peaks < plot_end_idx)
                    ]

                    if len(valid_peaks) > 0:
                        peak_times = valid_peaks / sampling_freq
                        peak_values = signal_data[valid_peaks]

                        fig.add_trace(
                            go.Scatter(
                                x=peak_times,
                                y=peak_values,
                                mode="markers",
                                name="P Peaks",
                                marker=dict(color="blue", size=8, symbol="circle"),
                                hovertemplate="<b>P Peak:</b> %{y:.2f}<br><b>Time:</b> %{x:.3f}s<extra></extra>",
                                legendgroup="p_peaks",
                                showlegend=(row_num == 1),
                            ),
                            row=row_num,
                            col=1,
                        )
            except Exception as e:
                logger.warning(f"P peak detection failed: {e}")

            # Detect and plot T peaks
            try:
                t_peaks = wm.detect_t_peak()
                if t_peaks is not None and len(t_peaks) > 0:
                    valid_peaks = t_peaks[
                        (t_peaks >= plot_start_idx) & (t_peaks < plot_end_idx)
                    ]

                    if len(valid_peaks) > 0:
                        peak_times = valid_peaks / sampling_freq
                        peak_values = signal_data[valid_peaks]

                        fig.add_trace(
                            go.Scatter(
                                x=peak_times,
                                y=peak_values,
                                mode="markers",
                                name="T Peaks",
                                marker=dict(color="green", size=8, symbol="square"),
                                hovertemplate="<b>T Peak:</b> %{y:.2f}<br><b>Time:</b> %{x:.3f}s<extra></extra>",
                                legendgroup="t_peaks",
                                showlegend=(row_num == 1),
                            ),
                            row=row_num,
                            col=1,
                        )
            except Exception as e:
                logger.warning(f"T peak detection failed: {e}")

        else:
            # For other signal types, use basic peak detection
            logger.info(f"Using basic peak detection for signal type: {signal_type}")
            try:
                from vitalDSP.physiological_features.peak_detection import detect_peaks

                peaks = detect_peaks(signal_data, sampling_freq)
                # Filter peaks to plot range
                valid_peaks = peaks[(peaks >= plot_start_idx) & (peaks < plot_end_idx)]

                if len(valid_peaks) > 0:
                    peak_times = valid_peaks / sampling_freq
                    peak_values = signal_data[valid_peaks]

                    fig.add_trace(
                        go.Scatter(
                            x=peak_times,
                            y=peak_values,
                            mode="markers",
                            name="Detected Peaks",
                            marker=dict(color="red", size=8, symbol="diamond"),
                            hovertemplate="<b>Peak:</b> %{y:.2f}<extra></extra>",
                            legendgroup="peaks",
                            showlegend=(row_num == 1),
                        ),
                        row=row_num,
                        col=1,
                    )
            except Exception as e:
                logger.warning(f"Basic peak detection failed: {e}")

    except Exception as e:
        logger.warning(f"Critical points detection failed: {e}")


def create_quality_main_plot(
    signal_data, quality_results, sampling_freq, filtered_signal=None, signal_type="PPG"
):
    """Create main quality visualization plot with segment classification.

    Args:
        signal_data: Original signal data
        quality_results: Quality assessment results
        sampling_freq: Sampling frequency
        filtered_signal: Optional filtered signal data (if None, won't show filtered plots)
        signal_type: Signal type for critical points detection (PPG, ECG, etc.)
    """
    try:
        time_axis = np.arange(len(signal_data)) / sampling_freq

        # PERFORMANCE OPTIMIZATION: Limit plot data to max 5 minutes and 10K points
        time_axis_plot, signal_data_plot = limit_plot_data(
            time_axis,
            signal_data,
            max_duration=300,  # 5 minutes max
            max_points=10000,  # 10K points max
        )

        logger.info(
            f"Quality plot data limited: {len(signal_data)} → {len(signal_data_plot)} points"
        )

        # Check if we have filtered signal - if yes, show 5 subplots; if no, show 2 subplots
        logger.info(
            f"Checking filtered signal: filtered_signal is {'not None' if filtered_signal is not None else 'None'}"
        )
        if filtered_signal is not None:
            logger.info(
                f"Filtered signal length: {len(filtered_signal)}, Original signal length: {len(signal_data)}"
            )

        if filtered_signal is not None and len(filtered_signal) == len(signal_data):
            logger.info("✅ Creating 5-subplot layout with filtered signal")
            # 5 subplots: original + classification, original + SQI, filtered + classification, filtered + SQI, SQI values
            num_rows = 5
            subplot_titles = [
                "Original Signal with Quality Classification",
                "Original Signal with SQI Values Over Time",
                "Filtered Signal with Quality Classification",
                "Filtered Signal with SQI Values Over Time",
                "SQI Values Over Time",
            ]
            row_heights = [0.20, 0.15, 0.20, 0.15, 0.30]
            vertical_spacing = 0.08

            # Also limit filtered signal data
            _, filtered_signal_plot = limit_plot_data(
                time_axis, filtered_signal, max_duration=300, max_points=10000
            )
            logger.info(
                f"Filtered signal plot limited: {len(filtered_signal)} → {len(filtered_signal_plot)} points"
            )
        else:
            logger.info(
                f"Using 2-subplot layout (filtered_signal: {'None' if filtered_signal is None else f'length {len(filtered_signal)} != {len(signal_data)}'})"
            )
            # 2 subplots: original signal + classification, SQI values
            num_rows = 2
            subplot_titles = [
                "Signal with Quality Classification",
                "SQI Values Over Time",
            ]
            row_heights = [0.60, 0.40]
            vertical_spacing = 0.12
            filtered_signal_plot = None

        fig = make_subplots(
            rows=num_rows,
            cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=vertical_spacing,
            row_heights=row_heights,
            specs=[[{"secondary_y": False}] for _ in range(num_rows)],
        )

        # Get SQI results for segment classification
        sqi_values = quality_results.get("sqi_values", [])
        threshold = quality_results.get("threshold", 0.5)
        threshold_type = quality_results.get("threshold_type", "above")

        # Helper function to add signal with quality classification
        def add_signal_with_classification(
            signal_plot, row_num, signal_full, signal_label="Signal"
        ):
            """Add signal with quality classification to a subplot."""
            # Plot signal base (will be colored based on segments)
            fig.add_trace(
                go.Scatter(
                    x=time_axis_plot,
                    y=signal_plot,
                    mode="lines",
                    name=signal_label,
                    line=dict(color="#95a5a6", width=1.5),  # Gray color
                    showlegend=(row_num == 1),
                ),
                row=row_num,
                col=1,
            )

            # Determine which segments are good/bad based on SQI values and threshold
            if sqi_values:
                window_size_samples = (
                    int(len(signal_data) / len(sqi_values))
                    if len(sqi_values) > 0
                    else len(signal_data)
                )
                segment_time_duration = window_size_samples / sampling_freq

                # Collect segments by quality type
                good_segments = []
                poor_segments = []

                for i, sqi_val in enumerate(sqi_values):
                    seg_start_time = i * segment_time_duration
                    seg_end_time = min(
                        (i + 1) * segment_time_duration, time_axis_plot[-1]
                    )

                    # Match vitalDSP's logic: threshold determines ABNORMAL segments
                    if isinstance(threshold, list):
                        # Range threshold: value should be between min and max to be NORMAL
                        is_good = threshold[0] <= sqi_val <= threshold[1]
                    elif threshold_type == "above":
                        # Values ABOVE threshold are ABNORMAL, so <= threshold is NORMAL
                        is_good = sqi_val <= threshold
                    elif threshold_type == "below":
                        # Values BELOW threshold are ABNORMAL, so >= threshold is NORMAL
                        is_good = sqi_val >= threshold
                    else:
                        is_good = (
                            sqi_val >= threshold
                        )  # Default to below logic (most common)

                    start_idx = max(0, int(seg_start_time * sampling_freq))
                    end_idx = min(len(signal_plot), int(seg_end_time * sampling_freq))

                    if end_idx > start_idx:
                        if is_good:
                            good_segments.append((start_idx, end_idx))
                        else:
                            poor_segments.append((start_idx, end_idx))

                # Add good quality segments (all as one trace with gaps)
                if good_segments:
                    x_good = []
                    y_good = []
                    x_good.append(time_axis_plot[good_segments[0][0]])
                    y_good.append(signal_plot[good_segments[0][0]])

                    for start, end in good_segments:
                        x_good.extend(time_axis_plot[start:end])
                        y_good.extend(signal_plot[start:end])
                        x_good.append(None)  # Add None to create gap
                        y_good.append(None)

                    fig.add_trace(
                        go.Scatter(
                            x=x_good,
                            y=y_good,
                            mode="lines",
                            name="Good Quality",
                            line=dict(color="#2ecc71", width=2.5),
                            showlegend=(row_num == 1),
                            legendgroup="quality_good",
                            hovertemplate="Good Quality Segment<br>Time: %{x:.2f}s<br>Amplitude: %{y:.2f}<extra></extra>",
                            opacity=0.85,
                        ),
                        row=row_num,
                        col=1,
                    )

                # Add poor quality segments (all as one trace with gaps)
                if poor_segments:
                    x_poor = []
                    y_poor = []

                    for start, end in poor_segments:
                        x_poor.extend(time_axis_plot[start:end])
                        y_poor.extend(signal_plot[start:end])
                        x_poor.append(None)  # Add None to create gap
                        y_poor.append(None)

                    fig.add_trace(
                        go.Scatter(
                            x=x_poor,
                            y=y_poor,
                            mode="lines",
                            name="Poor Quality",
                            line=dict(color="#e74c3c", width=2.5),
                            showlegend=(row_num == 1),
                            legendgroup="quality_poor",
                            hovertemplate="Poor Quality Segment<br>Time: %{x:.2f}s<br>Amplitude: %{y:.2f}<extra></extra>",
                            opacity=0.85,
                        ),
                        row=row_num,
                        col=1,
                    )

            # Add critical points detection
            add_critical_points_to_plot(
                fig, signal_full, time_axis_plot, sampling_freq, signal_type, row_num
            )

        # Helper function to add SQI values over time
        def add_sqi_over_time(row_num):
            """Add SQI values over time plot."""
            if not sqi_values:
                return

            window_size_samples = (
                int(len(signal_data) / len(sqi_values))
                if len(sqi_values) > 0
                else len(signal_data)
            )
            segment_time_duration = window_size_samples / sampling_freq

            # Create time points for SQI segments
            sqi_times = np.array(
                [i * segment_time_duration for i in range(len(sqi_values))]
            )

            # SQI values over time
            fig.add_trace(
                go.Scatter(
                    x=sqi_times,
                    y=sqi_values,
                    mode="lines+markers",
                    name="SQI Value",
                    line=dict(color="#3498db", width=2.5),  # Bright blue
                    marker=dict(size=5, color="#3498db"),
                    showlegend=(
                        row_num == 2 or row_num == 5
                    ),  # Show legend for SQI plots
                ),
                row=row_num,
                col=1,
            )

            # Add threshold line
            if len(sqi_times) > 0:
                # Handle both single threshold and range threshold
                if isinstance(threshold, list):
                    # Range threshold - add two lines
                    fig.add_trace(
                        go.Scatter(
                            x=[sqi_times[0], sqi_times[-1]],
                            y=[threshold[0], threshold[0]],
                            mode="lines",
                            name="Threshold Min",
                            line=dict(color="#e67e22", width=2, dash="dash"),
                            showlegend=(row_num == 2 or row_num == 5),
                        ),
                        row=row_num,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[sqi_times[0], sqi_times[-1]],
                            y=[threshold[1], threshold[1]],
                            mode="lines",
                            name="Threshold Max",
                            line=dict(color="#e67e22", width=2, dash="dash"),
                            showlegend=(row_num == 2 or row_num == 5),
                        ),
                        row=row_num,
                        col=1,
                    )
                else:
                    # Single threshold
                    fig.add_trace(
                        go.Scatter(
                            x=[sqi_times[0], sqi_times[-1]],
                            y=[threshold, threshold],
                            mode="lines",
                            name="Threshold",
                            line=dict(color="#e74c3c", width=2.5, dash="dash"),
                            showlegend=(row_num == 2 or row_num == 5),
                        ),
                        row=row_num,
                        col=1,
                    )

            # Add colored regions (green = good quality, red = poor quality)
            for i in range(len(sqi_values)):
                if i + 1 < len(sqi_times):
                    sqi_val = sqi_values[i]

                    # Match vitalDSP's logic: threshold determines ABNORMAL segments
                    if isinstance(threshold, list):
                        # Range threshold: value should be between min and max to be NORMAL
                        is_good = threshold[0] <= sqi_val <= threshold[1]
                    else:
                        # Single threshold
                        if threshold_type == "above":
                            # Values ABOVE threshold are ABNORMAL, so <= threshold is NORMAL
                            is_good = sqi_val <= threshold
                        elif threshold_type == "below":
                            # Values BELOW threshold are ABNORMAL, so >= threshold is NORMAL
                            is_good = sqi_val >= threshold
                        else:
                            is_good = (
                                sqi_val >= threshold
                            )  # Default to below logic (most common)

                    color = "#2ecc71" if is_good else "#e74c3c"  # Bright colors
                    opacity = 0.15 if is_good else 0.15

                    fig.add_shape(
                        type="rect",
                        x0=sqi_times[i],
                        x1=sqi_times[min(i + 1, len(sqi_times) - 1)],
                        y0=0,
                        y1=1,
                        fillcolor=color,
                        opacity=opacity,
                        layer="below",
                        line_width=0,
                        row=row_num,
                        col=1,
                    )

        # Now plot based on whether we have filtered signal or not
        if filtered_signal_plot is not None:
            # Plot 1: Original signal with quality classification
            add_signal_with_classification(
                signal_data_plot, 1, signal_data, "Original Signal"
            )

            # Plot 2: Original signal SQI values over time
            add_sqi_over_time(2)

            # Plot 3: Filtered signal with quality classification
            add_signal_with_classification(
                filtered_signal_plot, 3, filtered_signal, "Filtered Signal"
            )

            # Plot 4: Filtered signal SQI values over time
            add_sqi_over_time(4)

            # Plot 5: Main SQI values over time (larger plot)
            add_sqi_over_time(5)
        else:
            # Original 2-subplot layout
            # Plot 1: Original signal with quality classification
            add_signal_with_classification(signal_data_plot, 1, signal_data, "Signal")

            # Plot 2: SQI values over time
            add_sqi_over_time(2)

        # Update axes labels based on number of rows
        if filtered_signal_plot is not None:
            # 5 subplots
            fig.update_xaxes(title_text="Time (s)", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)

            fig.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig.update_yaxes(title_text="SQI Value", row=2, col=1, range=[0, 1])

            fig.update_xaxes(title_text="Time (s)", row=3, col=1)
            fig.update_yaxes(title_text="Amplitude", row=3, col=1)

            fig.update_xaxes(title_text="Time (s)", row=4, col=1)
            fig.update_yaxes(title_text="SQI Value", row=4, col=1, range=[0, 1])

            fig.update_xaxes(title_text="Time (s)", row=5, col=1)
            fig.update_yaxes(title_text="SQI Value", row=5, col=1, range=[0, 1])
        else:
            # 2 subplots (original)
            fig.update_xaxes(title_text="Time (s)", row=1, col=1)
            fig.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig.update_yaxes(title_text="SQI Value", row=2, col=1, range=[0, 1])

        # Add quality score annotation - positioned at top right
        quality_score = quality_results.get("quality_score", 0)
        overall_quality = quality_results.get("overall_quality", "UNKNOWN")

        # Choose color based on quality
        quality_colors = {
            "EXCELLENT": "#27ae60",  # Green
            "GOOD": "#2ecc71",  # Light green
            "FAIR": "#f39c12",  # Orange
            "POOR": "#e74c3c",  # Red
        }
        annotation_color = quality_colors.get(overall_quality, "#95a5a6")

        fig.add_annotation(
            text=f"<b>{overall_quality}</b><br>{quality_score:.1%}",
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            font=dict(size=16, color="white"),
            bgcolor=annotation_color,
            bordercolor="white",
            borderwidth=2,
            borderpad=8,
        )

        # Adjust height based on number of subplots
        plot_height = 1400 if filtered_signal_plot is not None else 700

        fig.update_layout(
            height=plot_height,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.0,
                xanchor="left",
                x=0.0,
                bgcolor="rgba(255, 255, 255, 0.9)",  # Semi-transparent background
                bordercolor="gray",
                borderwidth=1,
            ),
            hovermode="x unified",
            margin=dict(l=50, r=30, t=100, b=50),  # Optimized margins
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating main plot: {e}", exc_info=True)
        return create_empty_figure()


def create_quality_metrics_plot(quality_results, bins=30):
    """Create quality metrics visualization with histogram and quantiles.

    Args:
        quality_results: SQI results dictionary
        bins: Number of bins for histogram (default 30)
    """
    try:
        # Get SQI values from results
        sqi_values = quality_results.get("sqi_values", [])
        threshold = quality_results.get("threshold", 0.5)
        mean_sqi = quality_results.get("mean_sqi", 0)
        std_sqi = quality_results.get("std_sqi", 0)

        if not sqi_values:
            logger.warning("No SQI values found in results")
            return create_empty_figure()

        # Convert to numpy array for calculations
        sqi_array = np.array(sqi_values)

        # Calculate quantiles
        quantiles = {
            "Min": np.min(sqi_array),
            "Q1": np.percentile(sqi_array, 25),
            "Median": np.median(sqi_array),
            "Q3": np.percentile(sqi_array, 75),
            "Max": np.max(sqi_array),
        }

        # Create subplots: histogram and statistics
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "SQI Distribution",
                "Statistics",
                "SQI Over Time",
                "Classification",
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.15,
            specs=[
                [{"type": "xy"}, {"type": "xy"}],  # All xy plots for compatibility
                [{"type": "xy"}, {"type": "xy"}],
            ],
        )

        # 1. Histogram with quantile lines (use provided bins parameter)
        # Ensure bins is an integer
        try:
            bins_int = int(bins) if bins else 30
        except (ValueError, TypeError):
            bins_int = 30

        hist_data, bin_edges = np.histogram(sqi_array, bins=bins_int, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=hist_data,
                name="Distribution",
                marker=dict(
                    color=hist_data,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Count", len=0.9, y=0.75, x=1.02),
                ),
            ),
            row=1,
            col=1,
        )

        # Add quantile lines
        for q_name, q_value in quantiles.items():
            fig.add_vline(
                x=q_value,
                line_dash="dash",
                line_color="red" if "Min" in q_name or "Max" in q_name else "orange",
                annotation_text=q_name,
                annotation_position="top",
                row=1,
                col=1,
            )

        # Add threshold line(s) - handle both single and range thresholds
        if isinstance(threshold, list) and len(threshold) == 2:
            # Range threshold: add two lines
            fig.add_vline(
                x=threshold[0],
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"Min: {threshold[0]:.3f}",
                annotation_position="top right",
                row=1,
                col=1,
            )
            fig.add_vline(
                x=threshold[1],
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"Max: {threshold[1]:.3f}",
                annotation_position="top right",
                row=1,
                col=1,
            )
        else:
            # Single threshold
            threshold_val = float(threshold) if not isinstance(threshold, list) else 0.5
            fig.add_vline(
                x=threshold_val,
                line_dash="solid",
                line_color="blue",
                line_width=3,
                annotation_text=f"Threshold: {threshold_val:.3f}",
                annotation_position="top right",
                annotation_bgcolor="blue",
                annotation_font=dict(color="white"),
                row=1,
                col=1,
            )

        # 2. Statistics and quantiles table as a bar chart with actual values
        stats_labels = ["Mean", "Std", "Min", "Q1", "Median", "Q3", "Max"]
        stats_values = [
            mean_sqi,
            std_sqi,
            quantiles["Min"],
            quantiles["Q1"],
            quantiles["Median"],
            quantiles["Q3"],
            quantiles["Max"],
        ]

        fig.add_trace(
            go.Bar(
                x=stats_labels,
                y=stats_values,
                marker=dict(
                    color=stats_values,
                    colorscale="Viridis",
                    showscale=False,
                ),
                text=[f"{v:.3f}" for v in stats_values],
                textposition="outside",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Statistic", row=1, col=2)
        fig.update_yaxes(title_text="Value", row=1, col=2)

        # Add threshold as horizontal line (handle single and range)
        if isinstance(threshold, list) and len(threshold) == 2:
            # Range: add two horizontal lines
            fig.add_hline(
                y=threshold[0],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Min: {threshold[0]:.3f}",
                row=1,
                col=2,
            )
            fig.add_hline(
                y=threshold[1],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Max: {threshold[1]:.3f}",
                row=1,
                col=2,
            )
        else:
            # Single threshold
            threshold_val = float(threshold) if not isinstance(threshold, list) else 0.5
            fig.add_hline(
                y=threshold_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {threshold_val:.3f}",
                row=1,
                col=2,
            )

        # 3. SQI Over Time
        time_points = np.arange(len(sqi_values))
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=sqi_values,
                mode="lines+markers",
                name="SQI",
                line=dict(color="blue", width=2),
                marker=dict(size=4),
            ),
            row=2,
            col=1,
        )
        # Add threshold line for SQI Over Time plot (handle single and range)
        if isinstance(threshold, list) and len(threshold) == 2:
            # Range: add two horizontal lines
            fig.add_hline(
                y=threshold[0],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Min: {threshold[0]:.3f}",
                row=2,
                col=1,
            )
            fig.add_hline(
                y=threshold[1],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Max: {threshold[1]:.3f}",
                row=2,
                col=1,
            )
        else:
            # Single threshold
            threshold_val = float(threshold) if not isinstance(threshold, list) else 0.5
            fig.add_hline(
                y=threshold_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {threshold_val:.3f}",
                row=2,
                col=1,
            )
        fig.update_xaxes(title_text="Segment Index", row=2, col=1)
        fig.update_yaxes(title_text="SQI Value", row=2, col=1)

        # 4. Quality Classification (bar chart instead of pie for subplot compatibility)
        num_normal = quality_results.get("num_normal", 0)
        num_abnormal = quality_results.get("num_abnormal", 0)
        total_segments = num_normal + num_abnormal

        if total_segments > 0:
            normal_pct = (num_normal / total_segments) * 100
            abnormal_pct = (num_abnormal / total_segments) * 100

            fig.add_trace(
                go.Bar(
                    x=["Good Quality", "Poor Quality"],
                    y=[num_normal, num_abnormal],
                    text=[
                        f"{num_normal} ({normal_pct:.1f}%)",
                        f"{num_abnormal} ({abnormal_pct:.1f}%)",
                    ],
                    textposition="auto",
                    marker=dict(
                        color=["green", "red"],
                        line=dict(color="black", width=2),
                    ),
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_xaxes(title_text="SQI Value", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)

        # Update axes for quality classification bar chart
        fig.update_xaxes(title_text="", row=2, col=2)
        fig.update_yaxes(title_text="Number of Segments", row=2, col=2)

        fig.update_layout(
            height=600,  # Reduced from 800 to fit better
            showlegend=False,
            title_text="Signal Quality Index (SQI) Analysis",
            title_x=0.5,
            title_font=dict(size=14),
            margin=dict(l=50, r=50, t=80, b=50),  # Optimized margins
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating metrics plot: {e}", exc_info=True)
        return create_empty_figure()


def create_assessment_results_display(quality_results):
    """Create assessment results display."""
    try:
        if "error" in quality_results:
            return dbc.Alert(f"Error: {quality_results['error']}", color="danger")

        # Support both old format (overall_score) and new format (quality_score, overall_quality)
        if "overall_score" in quality_results:
            overall = quality_results["overall_score"]
            quality = overall.get("quality", "unknown")
            score = overall.get("score", 0)
            percentage = overall.get("percentage", 0)
        elif "quality_score" in quality_results:
            quality = quality_results.get("overall_quality", "UNKNOWN")
            score = quality_results.get("quality_score", 0)
            percentage = quality_results.get("quality_score", 0) * 100
        else:
            return dbc.Alert(
                "No quality assessment results available.", color="warning"
            )

        # Map quality labels
        quality_map = {
            "excellent": ("EXCELLENT", "success"),
            "EXCELLENT": ("EXCELLENT", "success"),
            "good": ("GOOD", "info"),
            "GOOD": ("GOOD", "info"),
            "fair": ("FAIR", "warning"),
            "FAIR": ("FAIR", "warning"),
            "poor": ("POOR", "danger"),
            "POOR": ("POOR", "danger"),
        }

        quality_label, quality_class = quality_map.get(
            quality, (quality.upper(), "secondary")
        )

        return dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Overall Quality Assessment", className="card-title"),
                    dbc.Alert(
                        [
                            html.H5(
                                f"Quality: {quality_label}",
                                className="alert-heading",
                            ),
                            html.P(f"Score: {score:.3f} ({percentage:.1f}%)"),
                            html.P(
                                f"SQI Type: {quality_results.get('sqi_type', 'N/A')}",
                                className="mb-0",
                            ),
                        ],
                        color=quality_class,
                    ),
                ]
            ),
            className="mt-3",
        )

    except Exception as e:
        logger.error(f"Error creating assessment results: {e}", exc_info=True)
        return dbc.Alert(f"Error: {str(e)}", color="danger")


def create_issues_recommendations_display(quality_results):
    """Create issues and recommendations display."""
    try:
        issues = []
        quality_score = quality_results.get("quality_score", 1.0)
        num_normal = quality_results.get("num_normal", 0)
        num_abnormal = quality_results.get("num_abnormal", 0)
        total_segments = num_normal + num_abnormal

        # Quality score based issues
        if quality_score < 0.4:
            issues.append(
                "Low overall quality score. Consider signal filtering or data collection improvement."
            )
        elif quality_score < 0.6:
            issues.append(
                "Moderate quality score. Some segments may need further processing."
            )

        # Segment-based issues
        if total_segments > 0:
            abnormal_percentage = (num_abnormal / total_segments) * 100
            if abnormal_percentage > 30:
                issues.append(
                    f"High percentage of poor quality segments ({abnormal_percentage:.1f}%). Consider adjusting processing parameters."
                )
            elif abnormal_percentage > 10:
                issues.append(
                    f"Some poor quality segments detected ({abnormal_percentage:.1f}%). Review signal quality thresholds."
                )

        # SQI-specific recommendations
        sqi_type = quality_results.get("sqi_type", "unknown")
        mean_sqi = quality_results.get("mean_sqi", 0)
        std_sqi = quality_results.get("std_sqi", 0)
        threshold = quality_results.get("threshold", 0.5)
        threshold_type = quality_results.get("threshold_type", "above")

        # Handle both single and range thresholds
        if isinstance(threshold, list) and len(threshold) == 2:
            # Range threshold: check if mean is outside the range
            if mean_sqi < threshold[0]:
                issues.append(
                    f"Mean SQI value ({mean_sqi:.3f}) is below the lower threshold ({threshold[0]:.3f}). Consider adjusting threshold or improving signal quality."
                )
            elif mean_sqi > threshold[1]:
                issues.append(
                    f"Mean SQI value ({mean_sqi:.3f}) is above the upper threshold ({threshold[1]:.3f}). Consider adjusting threshold."
                )
        else:
            # Single threshold
            threshold_val = float(threshold) if not isinstance(threshold, list) else 0.5
            if threshold_type == "above" and mean_sqi < threshold_val:
                issues.append(
                    f"Mean SQI value ({mean_sqi:.3f}) is below the threshold ({threshold_val:.3f}). Consider adjusting threshold or improving signal quality."
                )
            elif threshold_type == "below" and mean_sqi > threshold_val:
                issues.append(
                    f"Mean SQI value ({mean_sqi:.3f}) is above the threshold ({threshold_val:.3f}). Consider adjusting threshold."
                )

        if std_sqi > 0.2:
            issues.append(
                f"High variability in SQI values (std: {std_sqi:.3f}). Signal quality varies significantly over time."
            )

        if not issues:
            return dbc.Alert("No significant quality issues detected.", color="success")

        recommendations = [
            html.H4("Issues & Recommendations", className="card-title"),
        ]

        if issues:
            recommendations.append(html.Ul([html.Li(issue) for issue in issues]))

        # Add statistics
        recommendations.append(html.Hr(className="my-3"))
        recommendations.append(html.H5("Quality Statistics", className="card-title"))
        recommendations.append(
            html.P(
                f"Total Segments: {total_segments} (Normal: {num_normal}, Abnormal: {num_abnormal})"
            )
        )
        recommendations.append(html.P(f"Mean SQI: {mean_sqi:.3f} ± {std_sqi:.3f}"))
        recommendations.append(html.P(f"Quality Score: {quality_score:.1%}"))

        return dbc.Card(
            dbc.CardBody(recommendations),
            className="mt-3",
        )

    except Exception as e:
        logger.error(f"Error creating recommendations: {e}", exc_info=True)
        return dbc.Alert(f"Error: {str(e)}", color="danger")


def create_detailed_analysis_display(quality_results):
    """Create detailed analysis display."""
    try:
        details = []

        # Get SQI-specific details
        sqi_type = quality_results.get("sqi_type", "unknown")
        threshold = quality_results.get("threshold", 0.5)
        threshold_type = quality_results.get("threshold_type", "above")
        sqi_values = quality_results.get("sqi_values", [])

        # SQI Information
        details.append(html.H5("SQI Configuration"))
        details.append(html.P(f"SQI Type: {sqi_type.replace('_', ' ').title()}"))

        # Handle both single and range thresholds
        if isinstance(threshold, list) and len(threshold) == 2:
            details.append(
                html.P(
                    f"Threshold: {threshold[0]:.3f} - {threshold[1]:.3f} ({threshold_type})"
                )
            )
        else:
            threshold_val = float(threshold) if not isinstance(threshold, list) else 0.5
            details.append(html.P(f"Threshold: {threshold_val:.3f} ({threshold_type})"))

        if sqi_values:
            details.append(html.P(f"Number of SQI Values: {len(sqi_values)}"))

        # SQI Statistics
        details.append(html.Hr(className="my-3"))
        details.append(html.H5("SQI Statistics"))

        mean_sqi = quality_results.get("mean_sqi", 0)
        std_sqi = quality_results.get("std_sqi", 0)
        min_sqi = quality_results.get("min_sqi", 0)
        max_sqi = quality_results.get("max_sqi", 0)

        details.append(html.P(f"Mean SQI: {mean_sqi:.3f}"))
        details.append(html.P(f"Std Dev: {std_sqi:.3f}"))
        details.append(html.P(f"Min SQI: {min_sqi:.3f}"))
        details.append(html.P(f"Max SQI: {max_sqi:.3f}"))

        # Segment Information
        details.append(html.Hr(className="my-3"))
        details.append(html.H5("Segment Analysis"))

        num_normal = quality_results.get("num_normal", 0)
        num_abnormal = quality_results.get("num_abnormal", 0)
        num_segments = quality_results.get("num_segments", 0)

        details.append(html.P(f"Total Segments: {num_segments}"))
        details.append(
            html.P(
                f"Normal Segments: {num_normal} ({num_normal/num_segments*100:.1f}%)"
                if num_segments > 0
                else f"Normal Segments: {num_normal}"
            )
        )
        details.append(
            html.P(
                f"Abnormal Segments: {num_abnormal} ({num_abnormal/num_segments*100:.1f}%)"
                if num_segments > 0
                else f"Abnormal Segments: {num_abnormal}"
            )
        )

        # Quality Score Information
        details.append(html.Hr(className="my-3"))
        details.append(html.H5("Quality Assessment"))

        quality_score = quality_results.get("quality_score", 0)
        overall_quality = quality_results.get("overall_quality", "UNKNOWN")

        details.append(html.P(f"Overall Quality: {overall_quality}"))
        details.append(html.P(f"Quality Score: {quality_score:.1%}"))

        return dbc.Card(
            dbc.CardBody(details),
            className="mt-3",
        )

    except Exception as e:
        logger.error(f"Error creating detailed analysis: {e}", exc_info=True)
        return dbc.Alert(f"Error: {str(e)}", color="danger")


def create_score_dashboard(quality_results):
    """Create score dashboard."""
    try:
        # Support both old and new format
        if "overall_score" in quality_results:
            overall = quality_results["overall_score"]
            percentage = overall.get("percentage", 0)
            quality = overall.get("quality", "unknown")
        elif "quality_score" in quality_results:
            percentage = quality_results.get("quality_score", 0) * 100
            quality = quality_results.get("overall_quality", "UNKNOWN")
        else:
            return dbc.Alert("No score data available.", color="warning")

        # Map quality to progress bar color
        color_map = {
            "excellent": "success",
            "EXCELLENT": "success",
            "good": "info",
            "GOOD": "info",
            "fair": "warning",
            "FAIR": "warning",
            "poor": "danger",
            "POOR": "danger",
        }

        progress_color = color_map.get(quality, "secondary")

        return dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Quality Score Dashboard", className="card-title"),
                    dbc.Progress(
                        value=percentage,
                        label=f"{percentage:.1f}%",
                        color=progress_color,
                        className="mb-3",
                        style={"height": "30px"},
                    ),
                    html.P(f"Quality Level: {quality}", className="mb-0"),
                ]
            ),
            className="mt-3",
        )

    except Exception as e:
        logger.error(f"Error creating score dashboard: {e}", exc_info=True)
        return dbc.Alert(f"Error: {str(e)}", color="danger")
