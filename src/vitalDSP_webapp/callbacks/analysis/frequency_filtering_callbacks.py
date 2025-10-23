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
from scipy.fft import rfft, rfftfreq
import logging


# Helper function for formatting large numbers
def format_large_number(value, precision=3, use_scientific=False):
    """Format large numbers with appropriate scaling and units."""
    if value == 0:
        return "0"

    abs_value = abs(value)

    if use_scientific or abs_value >= 1e6:
        # Use scientific notation for very large numbers
        return f"{value:.{precision}e}"
    elif abs_value >= 1e3:
        # Use thousands (k) notation
        scaled_value = value / 1e3
        return f"{scaled_value:.{precision}f}k"
    elif abs_value >= 1:
        # Regular decimal notation
        return f"{value:.{precision}f}"
    elif abs_value >= 1e-3:
        # Use millis (m) notation
        scaled_value = value * 1e3
        return f"{scaled_value:.{precision}f}m"
    else:
        # Use scientific notation for very small numbers
        return f"{value:.{precision}e}"


logger = logging.getLogger(__name__)


def configure_plot_with_pan_zoom(fig, title="", height=400):
    """
    Configure plotly figure with pan/zoom tools and consistent styling.

    Args:
        fig: Plotly figure object
        title: Plot title
        height: Plot height in pixels

    Returns:
        Configured plotly figure
    """
    fig.update_layout(
        title=title,
        height=height,
        showlegend=True,
        template="plotly_white",
        # Enable pan and zoom - use simpler configuration
        dragmode="pan",  # Default to pan mode
        # Show the modebar with all tools
        modebar=dict(
            orientation="v",  # Vertical orientation
            bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
            color="rgba(0,0,0,0.5)",  # Dark icons
            activecolor="rgba(0,0,0,0.8)",  # Darker when active
        ),
    )
    return fig


def register_frequency_filtering_callbacks(app):
    """Register all frequency domain analysis and filtering callbacks."""

    # Auto-select signal type based on uploaded data
    @app.callback(
        [Output("freq-signal-source-select", "value")],
        [Input("url", "pathname")],
        prevent_initial_call=True,
    )
    def auto_select_freq_signal_source(pathname):
        """Auto-select signal source based on uploaded data."""
        if pathname != "/frequency":
            raise PreventUpdate

        try:
            from vitalDSP_webapp.services.data.enhanced_data_service import get_enhanced_data_service

            data_service = get_enhanced_data_service()
            if not data_service:
                logger.warning("Data service not available")
                return ["original"]

            # Get the latest data
            all_data = data_service.get_all_data()
            if not all_data:
                logger.info("No data available, using defaults")
                return ["original"]

            # Check if filtered data is available
            latest_data_id = max(
                all_data.keys(), key=lambda x: int(x.split("_")[1]) if "_" in x else 0
            )
            has_filtered = data_service.has_filtered_data(latest_data_id)

            if has_filtered:
                logger.info("Filtered data available, defaulting to filtered signal")
                return ["filtered"]
            else:
                logger.info("No filtered data available, using original signal")
                return ["original"]

        except Exception as e:
            logger.error(f"Error in auto-selection: {e}")
            return ["original"]

    # Frequency Domain Analysis Callback
    @app.callback(
        [
            Output("freq-main-plot", "figure"),
            Output("freq-psd-plot", "figure"),
            Output("freq-spectrogram-plot", "figure"),
            Output("freq-analysis-results", "children"),
            Output("freq-peak-analysis-table", "children"),
            Output("freq-band-power-table", "children"),
            Output("freq-stability-table", "children"),
            Output("freq-harmonics-table", "children"),
            Output("store-frequency-data", "data"),
            Output("store-time-freq-data", "data"),
        ],
        [
            # Input("url", "pathname"),  # REMOVED - was running full analysis on EVERY page load!
            Input("freq-btn-update-analysis", "n_clicks"),
            Input("freq-btn-nudge-m10", "n_clicks"),
            Input("freq-btn-nudge-m1", "n_clicks"),
            Input("freq-btn-nudge-p1", "n_clicks"),
            Input("freq-btn-nudge-p10", "n_clicks"),
        ],
        [
            State("url", "pathname"),  # MOVED to State - only read, doesn't trigger
            State("freq-time-range-slider", "value"),
            State("freq-start-time", "value"),
            State("freq-end-time", "value"),
            State(
                "freq-signal-source-select", "value"
            ),  # Added signal source selection
            State("freq-analysis-type", "value"),
            State("fft-window-type", "value"),
            State("fft-n-points", "value"),
            # PSD Parameters
            State("psd-window", "value"),
            State("psd-overlap", "value"),
            State("psd-freq-max", "value"),
            State("psd-log-scale", "value"),
            State("psd-normalize", "value"),
            State("psd-channel", "value"),
            # STFT Parameters
            State("stft-window-size", "value"),
            State("stft-hop-size", "value"),
            State("stft-window-type", "value"),
            State("stft-overlap", "value"),
            State("stft-scaling", "value"),
            State("stft-freq-max", "value"),
            State("stft-colormap", "value"),
            State("wavelet-type", "value"),
            State("wavelet-levels", "value"),
            State("freq-min", "value"),
            State("freq-max", "value"),
            State("freq-analysis-options", "value"),
        ],
    )
    def frequency_domain_callback(
        n_clicks,
        nudge_m10,
        nudge_m1,
        nudge_p1,
        nudge_p10,
        pathname,  # MOVED to correct position - this is a State parameter
        slider_value,
        start_time,
        end_time,
        signal_source,  # Added signal source parameter
        analysis_type,
        fft_window,
        fft_n_points,
        # PSD parameters
        psd_window,
        psd_overlap,
        psd_freq_max,
        psd_log_scale,
        psd_normalize,
        psd_channel,
        # STFT parameters
        stft_window_size,
        stft_hop_size,
        stft_window_type,
        stft_overlap,
        stft_scaling,
        stft_freq_max,
        stft_colormap,
        wavelet_type,
        wavelet_levels,
        freq_min,
        freq_max,
        analysis_options,
    ):
        """Unified callback for frequency domain analysis."""
        ctx = callback_context

        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        logger.info("=== FREQUENCY DOMAIN CALLBACK TRIGGERED ===")
        logger.info(f"Trigger ID: {trigger_id}")
        logger.info(f"Pathname: {pathname}")

        # Only run this when we're on the frequency domain page
        if pathname != "/frequency":
            logger.info("Not on frequency page, returning empty figures")
            empty_figs = [create_empty_figure() for _ in range(3)]
            return empty_figs + [
                "Navigate to Frequency page",
                create_empty_figure(),
                None,
                None,
                None,
                None,
                None,
            ]

        # If this is the first time loading the page (no button clicks), show a message
        if not ctx.triggered or ctx.triggered[0]["prop_id"].split(".")[0] == "url":
            logger.info("First time loading frequency page, attempting to load data")

        try:
            # Get data from the data service
            logger.info("Attempting to import data service...")
            from vitalDSP_webapp.services.data.enhanced_data_service import get_enhanced_data_service

            logger.info("Data service imported successfully")

            # Add vitalDSP to path if needed
            import sys
            import os

            current_dir = os.path.dirname(os.path.abspath(__file__))
            vitaldsp_path = os.path.join(current_dir, "..", "..", "..", "vitalDSP")
            if vitaldsp_path not in sys.path:
                sys.path.insert(0, vitaldsp_path)
                logger.info(f"Added vitalDSP path: {vitaldsp_path}")
            logger.info(f"Current sys.path: {sys.path[:3]}")

            data_service = get_enhanced_data_service()
            data_store = data_service.get_all_data()

            if not data_store:
                logger.warning("No data available for frequency analysis")
                empty_figs = [create_empty_figure() for _ in range(3)]
                return empty_figs + [
                    "No data available. Please upload data first.",
                    create_empty_figure(),
                    None,
                    None,
                    None,
                    None,
                    None,
                ]

            # Get the first available data
            data_id = list(data_store.keys())[0]
            df = data_service.get_data(data_id)
            column_mapping = data_service.get_column_mapping(data_id)
            data_info = data_service.get_data_info(data_id)

            logger.info("=== DATA LOADING DEBUG ===")
            logger.info(f"Data ID: {data_id}")
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns: {list(df.columns)}")
            logger.info(f"Column mapping: {column_mapping}")
            logger.info(f"Data info: {data_info}")
            logger.info(
                f"Data info keys: {list(data_info.keys()) if data_info else 'None'}"
            )
            logger.info(f"Data info: {data_info}")

            if df is None or df.empty:
                logger.warning("Data is empty or None")
                empty_figs = [create_empty_figure() for _ in range(3)]
                return empty_figs + [
                    "Data is empty. Please upload valid data.",
                    create_empty_figure(),
                    None,
                    None,
                    None,
                    None,
                    None,
                ]

            # Extract time and signal columns
            time_col = column_mapping.get("time", df.columns[0])
            signal_col = column_mapping.get("signal", df.columns[1])

            # Get sampling frequency
            sampling_freq = data_info.get("sampling_freq", 1000)  # Default to 1000 Hz

            logger.info("=== COLUMN SELECTION DEBUG ===")
            logger.info(
                f"Selected time column: '{time_col}' (index: {df.columns.get_loc(time_col) if time_col in df.columns else 'NOT FOUND'})"
            )
            logger.info(
                f"Selected signal column: '{signal_col}' (index: {df.columns.get_loc(signal_col) if signal_col in df.columns else 'NOT FOUND'})"
            )
            logger.info(f"Sampling frequency: {sampling_freq} Hz")

            # Extract time and signal data
            time_data = df[time_col].values
            signal_data = df[signal_col].values

            # Convert time data to seconds if it's in milliseconds
            # First check if time_data is datetime and convert to numeric
            if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                logger.info("Converting datetime time data to numeric seconds")
                first_timestamp = df[time_col].iloc[0]
                time_data = (df[time_col] - first_timestamp).dt.total_seconds().values
                logger.info("Time data converted to seconds from first timestamp")
            elif np.max(time_data) > 1000:  # Likely milliseconds
                time_data = time_data / 1000.0
                logger.info("Converted time data from milliseconds to seconds")

            # Load signal source (original or filtered) based on selection
            logger.info("=== SIGNAL SOURCE LOADING ===")
            logger.info(f"Signal source selection: {signal_source}")

            # Always keep the original signal_data for dynamic filtering
            original_signal_data = signal_data.copy()
            selected_signal = signal_data
            signal_source_info = "Original Signal"
            filter_info = None

            if signal_source == "filtered":
                # Try to load filtered data from filtering screen
                filtered_data = data_service.get_filtered_data(data_id)
                filter_info = data_service.get_filter_info(data_id)

                if filtered_data is not None:
                    logger.info(
                        f"Found filtered data with shape: {filtered_data.shape}"
                    )
                    selected_signal = filtered_data
                    signal_source_info = "Filtered Signal"
                else:
                    logger.info("No filtered data available, using original signal")
            else:
                logger.info("Using original signal as requested")

            logger.info(f"Selected signal shape: {selected_signal.shape}")
            logger.info(f"Signal source info: {signal_source_info}")

            logger.info(
                f"Time data after conversion: {time_data[:5]} (range: {time_data[0]:.3f}s to {time_data[-1]:.3f}s)"
            )

            logger.info("=== DATA EXTRACTION DEBUG ===")
            logger.info(f"Time data shape: {time_data.shape}")
            logger.info(f"Time data range: {time_data[0]:.3f}s to {time_data[-1]:.3f}s")
            logger.info(f"Signal data shape: {selected_signal.shape}")
            logger.info(
                f"Signal data range: {np.min(selected_signal):.3f} to {np.max(selected_signal):.3f}"
            )
            logger.info(
                f"Signal data summary: dtype={selected_signal.dtype}, count={len(selected_signal)}"
            )
            logger.info(
                f"Time data summary: dtype={time_data.dtype}, count={len(time_data)}"
            )
            logger.info(
                f"Signal data statistics - Mean: {np.mean(selected_signal):.3f}, Std: {np.std(selected_signal):.3f}"
            )
            logger.info(f"Signal data has NaN: {np.any(np.isnan(selected_signal))}")
            logger.info(f"Signal data has Inf: {np.any(np.isinf(selected_signal))}")
            logger.info(
                f"Signal data is constant: {np.all(selected_signal == selected_signal[0])}"
            )

            # Handle time window - use integer indexing like legacy code
            if start_time is not None and end_time is not None:
                start_sample = int(start_time * sampling_freq)
                end_sample = int(end_time * sampling_freq)

                # Ensure we have valid indices - use original signal length for bounds checking
                original_signal_length = len(original_signal_data)
                start_sample = max(0, min(start_sample, original_signal_length - 1))
                end_sample = max(
                    start_sample + 1, min(end_sample, original_signal_length)
                )

                if end_sample > start_sample:
                    # Apply time window to both original and selected signals
                    original_signal_data = original_signal_data[start_sample:end_sample]
                    selected_signal = selected_signal[start_sample:end_sample]
                    time_data = time_data[start_sample:end_sample]

                    # Check if we need to apply dynamic filtering for filtered signal
                    if signal_source == "filtered" and filter_info is not None:
                        # Check if the time range is within the original signal range
                        logger.info(
                            f"Getting original signal data for data_id: {data_id}"
                        )
                        logger.info(f"Data service type: {type(data_service)}")
                        logger.info(
                            f"Data service methods: {[method for method in dir(data_service) if not method.startswith('_')]}"
                        )
                        df = data_service.get_data(data_id)
                        logger.info(f"DataFrame type: {type(df)}")
                        logger.info(
                            f"DataFrame shape: {df.shape if df is not None else 'None'}"
                        )
                        # Use the already calculated original_signal_length from above
                        logger.info(
                            f"Using original signal length: {original_signal_length}"
                        )
                        expected_length = end_sample - start_sample

                        # If time range is outside original signal or we need dynamic filtering
                        if (
                            start_sample >= original_signal_length
                            or end_sample > original_signal_length
                            or expected_length != len(selected_signal)
                        ):
                            logger.info("=== DYNAMIC FILTERING ===")
                            logger.info(
                                f"Original signal length: {original_signal_length}"
                            )
                            logger.info(f"Time range: {start_sample} to {end_sample}")
                            logger.info(f"Expected window length: {expected_length}")
                            logger.info(
                                "Applying dynamic filtering to current window..."
                            )

                            # If time range is completely outside original signal, return error
                            if (
                                start_sample >= original_signal_length
                                or end_sample > original_signal_length
                            ):
                                logger.warning(
                                    f"Time range {start_sample}-{end_sample} is outside original signal range (0-{original_signal_length})"
                                )
                                empty_figs = [create_empty_figure() for _ in range(3)]
                                error_message = f"Time range is outside the available signal data. Please select a time range within 0 to {original_signal_length/sampling_freq:.1f} seconds."
                                return empty_figs + [
                                    error_message,
                                    create_empty_figure(),
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                ]

                            try:
                                # Import the same filtering function used in time domain
                                from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
                                    apply_traditional_filter,
                                )

                                # Get the full original signal for the current time window
                                logger.info(
                                    "Getting full original signal for dynamic filtering"
                                )
                                df_full = data_service.get_data(data_id)
                                logger.info(f"Full DataFrame type: {type(df_full)}")
                                logger.info(
                                    f"Full DataFrame shape: {df_full.shape if df_full is not None else 'None'}"
                                )
                                full_original_signal = (
                                    df_full[signal_col].values
                                    if df_full is not None
                                    else np.array([])
                                )
                                windowed_original_signal = full_original_signal[
                                    start_sample:end_sample
                                ]

                                # Get filter parameters
                                parameters = filter_info.get("parameters", {})
                                detrending_applied = filter_info.get(
                                    "detrending_applied", False
                                )

                                # Apply detrending if it was applied in the original filtering
                                if detrending_applied:
                                    from scipy import signal as scipy_signal

                                    signal_data_detrended = scipy_signal.detrend(
                                        windowed_original_signal
                                    )
                                    logger.info("Applied detrending to signal")
                                else:
                                    signal_data_detrended = windowed_original_signal

                                # Apply the same filter type as used in filtering screen
                                filter_type = filter_info.get(
                                    "filter_type", "traditional"
                                )

                                if filter_type == "traditional":
                                    # Extract traditional filter parameters
                                    filter_family = parameters.get(
                                        "filter_family", "butter"
                                    )
                                    filter_response = parameters.get(
                                        "filter_response", "bandpass"
                                    )
                                    low_freq = parameters.get("low_freq", 0.5)
                                    high_freq = parameters.get("high_freq", 5)
                                    filter_order = parameters.get("filter_order", 4)

                                    # Apply traditional filter
                                    selected_signal = apply_traditional_filter(
                                        signal_data_detrended,
                                        sampling_freq,
                                        filter_family,
                                        filter_response,
                                        low_freq,
                                        high_freq,
                                        filter_order,
                                    )
                                    logger.info("Applied dynamic traditional filter")

                                elif filter_type == "advanced":
                                    # Extract advanced filter parameters
                                    advanced_method = parameters.get(
                                        "advanced_method", "kalman"
                                    )
                                    noise_level = parameters.get("noise_level", 0.1)
                                    iterations = parameters.get("iterations", 100)
                                    learning_rate = parameters.get(
                                        "learning_rate", 0.01
                                    )

                                    # Import and apply advanced filter
                                    from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
                                        apply_advanced_filter,
                                    )

                                    selected_signal = apply_advanced_filter(
                                        signal_data_detrended,
                                        advanced_method,
                                        noise_level,
                                        iterations,
                                        learning_rate,
                                    )
                                    logger.info("Applied dynamic advanced filter")

                                elif filter_type == "artifact":
                                    # Extract artifact removal parameters
                                    artifact_type = parameters.get(
                                        "artifact_type", "baseline"
                                    )
                                    artifact_strength = parameters.get(
                                        "artifact_strength", 0.5
                                    )

                                    # Import and apply artifact removal
                                    from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
                                        apply_enhanced_artifact_removal,
                                    )

                                    selected_signal = apply_enhanced_artifact_removal(
                                        signal_data_detrended,
                                        sampling_freq,
                                        artifact_type,
                                        artifact_strength,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                    )
                                    logger.info("Applied dynamic artifact removal")

                                elif filter_type == "neural":
                                    # Extract neural filter parameters
                                    neural_type = parameters.get("neural_type", "lstm")
                                    neural_complexity = parameters.get(
                                        "neural_complexity", "medium"
                                    )

                                    # Import and apply neural filter
                                    from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
                                        apply_neural_filter,
                                    )

                                    selected_signal = apply_neural_filter(
                                        signal_data_detrended,
                                        neural_type,
                                        neural_complexity,
                                    )
                                    logger.info("Applied dynamic neural filter")

                                elif filter_type == "ensemble":
                                    # Extract ensemble filter parameters
                                    ensemble_method = parameters.get(
                                        "ensemble_method", "mean"
                                    )
                                    ensemble_n_filters = parameters.get(
                                        "ensemble_n_filters", 3
                                    )

                                    # Import and apply ensemble filter
                                    from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
                                        apply_enhanced_ensemble_filter,
                                    )

                                    selected_signal = apply_enhanced_ensemble_filter(
                                        signal_data_detrended,
                                        ensemble_method,
                                        ensemble_n_filters,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                    )
                                    logger.info("Applied dynamic ensemble filter")

                                else:
                                    logger.warning(
                                        f"Unknown filter type {filter_type}, using original signal"
                                    )
                                    selected_signal = signal_data_detrended

                                signal_source_info = "Filtered Signal (Dynamic)"
                                logger.info("Dynamic filtering completed successfully")

                            except Exception as e:
                                logger.error(f"Error in dynamic filtering: {e}")
                                logger.info("Falling back to original signal")
                                selected_signal = windowed_original_signal
                                signal_source_info = "Original Signal (Fallback)"

                    logger.info("=== TIME WINDOW DEBUG ===")
                    logger.info(f"Applied time window: {start_time}s to {end_time}s")
                    logger.info(
                        f"Start sample: {start_sample}, End sample: {end_sample}"
                    )
                    logger.info(f"Windowed data shape: {selected_signal.shape}")
                    logger.info(
                        f"Windowed time range: {time_data[0]:.3f}s to {time_data[-1]:.3f}s"
                    )
                    logger.info(
                        f"Windowed signal range: {np.min(selected_signal):.3f} to {np.max(selected_signal):.3f}"
                    )
                else:
                    logger.warning(
                        f"Invalid time window: start_sample={start_sample}, end_sample={end_sample}"
                    )
            else:
                logger.info("No time window specified, using full dataset")

            # Log final data state before analysis
            logger.info("=== FINAL DATA STATE FOR ANALYSIS ===")
            logger.info(f"Final signal data shape: {selected_signal.shape}")
            logger.info(f"Final time data shape: {time_data.shape}")
            logger.info(
                f"Final signal range: {np.min(selected_signal):.3f} to {np.max(selected_signal):.3f}"
            )
            logger.info(
                f"Final time range: {time_data[0]:.3f}s to {time_data[-1]:.3f}s"
            )
            logger.info(
                f"Signal data summary: dtype={selected_signal.dtype}, count={len(selected_signal)}"
            )
            logger.info(
                f"Time data summary: dtype={time_data.dtype}, count={len(time_data)}"
            )

            # Normalize selected signal for better analysis
            if np.std(selected_signal) > 0:
                selected_signal = (selected_signal - np.mean(selected_signal)) / np.std(
                    selected_signal
                )
                logger.info("Normalized selected signal to zero mean and unit variance")
                logger.info(
                    f"Normalized signal range: {np.min(selected_signal):.3f} to {np.max(selected_signal):.3f}"
                )
            else:
                logger.warning("Selected signal has zero variance, cannot normalize")

            # Check if signal is too short for analysis
            min_samples_required = 64  # Minimum samples needed for PSD analysis
            if len(selected_signal) < min_samples_required:
                logger.warning(
                    f"Signal too short for analysis: {len(selected_signal)} samples (minimum: {min_samples_required})"
                )
                empty_figs = [create_empty_figure() for _ in range(3)]
                error_message = f"Signal too short for frequency analysis. Current: {len(selected_signal)} samples, Minimum required: {min_samples_required} samples. Please select a larger time range."
                return empty_figs + [
                    error_message,
                    create_empty_figure(),
                    None,
                    None,
                    None,
                    None,
                    None,
                ]

            # Set default values
            analysis_type = analysis_type or "fft"
            fft_window = fft_window or "hann"
            fft_n_points = fft_n_points or len(selected_signal)

            logger.info("=== ANALYSIS PARAMETERS DEBUG ===")
            logger.info(f"Analysis type: {analysis_type}")
            logger.info(f"FFT window: {fft_window}")
            logger.info(f"FFT n_points: {fft_n_points}")
            logger.info(f"Frequency range: {freq_min} to {freq_max}")
            logger.info(f"Analysis options: {analysis_options}")

            # Perform frequency domain analysis based on type
            if analysis_type.lower() == "fft":
                (
                    main_fig,
                    psd_fig,
                    spectrogram_fig,
                    results,
                    peak_table,
                    band_table,
                    stability_table,
                    harmonics_table,
                    freq_data,
                    time_freq_data,
                ) = perform_fft_analysis(
                    time_data,
                    selected_signal,
                    sampling_freq,
                    fft_window,
                    fft_n_points,
                    freq_min,
                    freq_max,
                    analysis_options,
                )
            elif analysis_type.lower() == "psd":
                (
                    main_fig,
                    psd_fig,
                    spectrogram_fig,
                    results,
                    peak_table,
                    band_table,
                    stability_table,
                    harmonics_table,
                    freq_data,
                    time_freq_data,
                ) = perform_psd_analysis(
                    time_data,
                    selected_signal,
                    sampling_freq,
                    psd_window,
                    psd_overlap,
                    psd_freq_max,
                    psd_log_scale,
                    psd_normalize,
                    psd_channel,
                    freq_min,
                    freq_max,
                    analysis_options,
                )
            elif analysis_type.lower() == "stft":
                (
                    main_fig,
                    psd_fig,
                    spectrogram_fig,
                    results,
                    peak_table,
                    band_table,
                    stability_table,
                    harmonics_table,
                    freq_data,
                    time_freq_data,
                ) = perform_stft_analysis(
                    time_data,
                    selected_signal,
                    sampling_freq,
                    stft_window_size,
                    stft_hop_size,
                    stft_window_type,
                    stft_overlap,
                    stft_scaling,
                    stft_freq_max,
                    stft_colormap,
                    freq_min,
                    freq_max,
                    analysis_options,
                )
            elif analysis_type.lower() == "wavelet":
                (
                    main_fig,
                    psd_fig,
                    spectrogram_fig,
                    results,
                    peak_table,
                    band_table,
                    stability_table,
                    harmonics_table,
                    freq_data,
                    time_freq_data,
                ) = perform_wavelet_analysis(
                    time_data,
                    selected_signal,
                    sampling_freq,
                    wavelet_type,
                    wavelet_levels,
                    freq_min,
                    freq_max,
                    analysis_options,
                )
            else:
                # Default to FFT
                logger.info(
                    f"Unknown analysis type '{analysis_type}', defaulting to FFT"
                )
                (
                    main_fig,
                    psd_fig,
                    spectrogram_fig,
                    results,
                    peak_table,
                    band_table,
                    stability_table,
                    harmonics_table,
                    freq_data,
                    time_freq_data,
                ) = perform_fft_analysis(
                    time_data,
                    selected_signal,
                    sampling_freq,
                    fft_window,
                    fft_n_points,
                    freq_min,
                    freq_max,
                    analysis_options,
                )

            logger.info("Frequency domain analysis completed successfully")
            return (
                main_fig,
                psd_fig,
                spectrogram_fig,
                results,
                peak_table,
                band_table,
                stability_table,
                harmonics_table,
                freq_data,
                time_freq_data,
            )

        except Exception as e:
            logger.error(f"Error in frequency domain analysis: {e}")
            import traceback

            logger.error(traceback.format_exc())

            error_fig = create_empty_figure()
            error_fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

            error_results = html.Div(
                [
                    html.H5("Error in Frequency Analysis"),
                    html.P(f"Analysis failed: {str(e)}"),
                    html.P("Please check your data and parameters."),
                ]
            )

            empty_figs = [error_fig for _ in range(3)]
            empty_tables = [
                html.Div("Error occurred during analysis") for _ in range(4)
            ]

            return empty_figs + [error_results] + empty_tables + [None, None]

    # Advanced Filtering Callback
    @app.callback(
        [
            Output("frequency-filtered-signal-plot", "figure"),
            Output("filter-response-plot", "figure"),
            Output("filter-stats", "children"),
        ],
        [Input("btn-apply-filter", "n_clicks")],
        [
            State("store-uploaded-data", "data"),
            State("store-data-config", "data"),
            State("filter-type", "value"),
            State("cutoff-freq", "value"),
            State("filter-order", "value"),
        ],
    )
    def apply_filter(
        n_clicks, data_store, config_store, filter_type, cutoff_freq, filter_order
    ):
        """Apply frequency filter to the signal."""
        if not n_clicks or not data_store or not config_store:
            raise PreventUpdate

        try:
            # Extract data
            df = pd.DataFrame(data_store["data"])
            sampling_freq = config_store["sampling_freq"]
            selected_signal = df.iloc[:, 1].values

            # Set default values
            filter_type = filter_type or "lowpass"
            cutoff_freq = cutoff_freq or sampling_freq / 4
            filter_order = filter_order or 4

            # Normalize cutoff frequency
            nyquist = sampling_freq / 2
            normalized_cutoff = cutoff_freq / nyquist

            if normalized_cutoff >= 1.0:
                normalized_cutoff = 0.95

            # Design filter
            if filter_type == "lowpass":
                b, a = signal.butter(filter_order, normalized_cutoff, btype="low")
            elif filter_type == "highpass":
                b, a = signal.butter(filter_order, normalized_cutoff, btype="high")
            elif filter_type == "bandpass":
                # For bandpass, we need two cutoff frequencies
                if isinstance(cutoff_freq, (list, tuple)) and len(cutoff_freq) == 2:
                    low_cutoff = cutoff_freq[0] / nyquist
                    high_cutoff = cutoff_freq[1] / nyquist
                else:
                    low_cutoff = normalized_cutoff * 0.5
                    high_cutoff = normalized_cutoff
                b, a = signal.butter(
                    filter_order, [low_cutoff, high_cutoff], btype="band"
                )
            else:
                raise ValueError(f"Unsupported filter type: {filter_type}")

            # Apply filter
            filtered_signal = signal.filtfilt(b, a, selected_signal)

            # Create time axis
            time_axis = np.arange(len(selected_signal)) / sampling_freq

            # Create filtered signal plot
            signal_fig = go.Figure()
            signal_fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=selected_signal,
                    mode="lines",
                    name="Original Signal",
                    line=dict(color="blue"),
                    opacity=0.7,
                )
            )
            signal_fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=filtered_signal,
                    mode="lines",
                    name="Filtered Signal",
                    line=dict(color="red"),
                )
            )

            signal_fig.update_layout(
                title=f"{filter_type.title()} Filtered Signal",
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude",
                showlegend=True,
                template="plotly_white",
                # Enable pan and zoom
                dragmode="pan",
                modebar=dict(
                    orientation="v",
                    bgcolor="rgba(255,255,255,0.8)",
                    color="rgba(0,0,0,0.5)",
                    activecolor="rgba(0,0,0,0.8)",
                ),
            )

            # Create filter response plot
            w, h = signal.freqz(b, a, worN=1000)
            freq_response = w * sampling_freq / (2 * np.pi)

            response_fig = go.Figure()
            response_fig.add_trace(
                go.Scatter(
                    x=freq_response,
                    y=20 * np.log10(np.abs(h)),
                    mode="lines",
                    name="Filter Response",
                    line=dict(color="green"),
                )
            )

            response_fig.update_layout(
                title="Filter Frequency Response",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude (dB)",
                showlegend=True,
                template="plotly_white",
                # Enable pan and zoom
                dragmode="pan",
                modebar=dict(
                    orientation="v",
                    bgcolor="rgba(255,255,255,0.8)",
                    color="rgba(0,0,0,0.5)",
                    activecolor="rgba(0,0,0,0.8)",
                ),
            )

            # Create stats
            stats = html.Div(
                [
                    html.H5("Filter Statistics"),
                    html.P(f"Filter Type: {filter_type.title()}"),
                    html.P(f"Cutoff Frequency: {cutoff_freq:.2f} Hz"),
                    html.P(f"Filter Order: {filter_order}"),
                    html.P(f"Sampling Frequency: {sampling_freq} Hz"),
                    html.P(f"Signal Length: {len(selected_signal)} samples"),
                    html.P(
                        f"Duration: {len(selected_signal)/sampling_freq:.2f} seconds"
                    ),
                ]
            )

            return signal_fig, response_fig, stats

        except Exception as e:
            logger.error(f"Error in frequency filtering: {e}")
            error_fig = create_empty_figure()
            error_fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

            error_stats = html.Div(
                [html.H5("Error"), html.P(f"Filtering failed: {str(e)}")]
            )

            return error_fig, error_fig, error_stats

    # Time input update callbacks
    @app.callback(
        [Output("freq-start-time", "value"), Output("freq-end-time", "value")],
        [
            Input("freq-time-range-slider", "value"),
            Input("freq-btn-nudge-m10", "n_clicks"),
            Input("freq-btn-nudge-m1", "n_clicks"),
            Input("freq-btn-nudge-p1", "n_clicks"),
            Input("freq-btn-nudge-p10", "n_clicks"),
        ],
        [State("freq-start-time", "value"), State("freq-end-time", "value")],
    )
    def update_freq_time_inputs(
        slider_value, nudge_m10, nudge_m1, nudge_p1, nudge_p10, start_time, end_time
    ):
        """Update time inputs based on slider or nudge buttons."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "freq-time-range-slider" and slider_value:
            return slider_value[0], slider_value[1]

        # Handle nudge buttons
        time_window = end_time - start_time if start_time and end_time else 10

        if trigger_id == "freq-btn-nudge-m10":
            new_start = max(0, start_time - 10) if start_time else 0
            new_end = new_start + time_window
            return new_start, new_end
        elif trigger_id == "freq-btn-nudge-m1":
            new_start = max(0, start_time - 1) if start_time else 0
            new_end = new_start + time_window
            return new_start, new_end
        elif trigger_id == "freq-btn-nudge-p1":
            new_start = start_time + 1 if start_time else 1
            new_end = new_start + time_window
            return new_start, new_end
        elif trigger_id == "freq-btn-nudge-p10":
            new_start = start_time + 10 if start_time else 10
            new_end = new_start + time_window
            return new_start, new_end

        return no_update, no_update

    # Time slider range update callback
    @app.callback(
        Output("freq-time-range-slider", "max"),
        [Input("store-uploaded-data", "data")],
        prevent_initial_call=True,
    )
    def update_freq_time_slider_range(data_store):
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


# Helper functions for frequency analysis
def create_empty_figure():
    """Create an empty figure for error handling."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
    )
    return fig


def create_fft_plot(
    selected_signal, sampling_freq, window_type, n_points, freq_min, freq_max
):
    """Create enhanced FFT plot with vitalDSP insights."""
    logger.info(f"Creating FFT plot with window: {window_type}, n_points: {n_points}")

    # Apply window function
    if window_type == "hamming":
        windowed_signal = selected_signal * np.hamming(len(selected_signal))
        logger.info("Applied Hamming window for optimal frequency resolution")
    elif window_type == "hanning":
        windowed_signal = selected_signal * np.hanning(len(selected_signal))
        logger.info("Applied Hanning window for reduced spectral leakage")
    elif window_type == "blackman":
        windowed_signal = selected_signal * np.blackman(len(selected_signal))
        logger.info("Applied Blackman window for excellent sidelobe suppression")
    elif window_type == "kaiser":
        windowed_signal = selected_signal * np.kaiser(len(selected_signal), beta=14)
        logger.info(
            "Applied Kaiser window with beta=14 for optimal time-frequency trade-off"
        )
    else:
        windowed_signal = selected_signal
        logger.info("No window applied (rectangular window)")

    # Compute FFT
    fft_result = rfft(windowed_signal, n=n_points)
    freqs = rfftfreq(n_points, 1 / sampling_freq)

    # Calculate vitalDSP metrics
    magnitude = np.abs(fft_result)
    power_spectrum = magnitude**2

    # Find dominant frequencies
    peak_idx = np.argmax(magnitude)
    peak_freq = freqs[peak_idx]
    peak_magnitude = magnitude[peak_idx]

    # Calculate frequency resolution
    freq_resolution = sampling_freq / n_points
    logger.info(f"Frequency resolution: {freq_resolution:.3f} Hz")
    logger.info(
        f"Peak frequency: {peak_freq:.2f} Hz with magnitude {peak_magnitude:.2e}"
    )

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
        rows=2,
        cols=1,
        subplot_titles=("Frequency Spectrum (Magnitude)", "Power Spectral Density"),
        vertical_spacing=0.15,  # Increased from 0.1 to prevent overlap
    )

    # Magnitude plot
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=magnitude,
            mode="lines",
            name=f"{window_type.title()} Window",
            line=dict(color="blue", width=2),
            hovertemplate="<b>Frequency:</b> %{x:.2f} Hz<br><b>Magnitude:</b> %{y:.2e}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Mark peak frequency
    fig.add_trace(
        go.Scatter(
            x=[peak_freq],
            y=[peak_magnitude],
            mode="markers",
            name="Peak Frequency",
            marker=dict(color="red", size=10, symbol="diamond"),
            hovertemplate=f"<b>Peak:</b> {peak_freq:.2f} Hz<br><b>Magnitude:</b> {peak_magnitude:.2e}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Power spectral density plot
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=power_spectrum,
            mode="lines",
            name="Power Spectrum",
            line=dict(color="green", width=2),
            hovertemplate="<b>Frequency:</b> %{x:.2f} Hz<br><b>Power:</b> %{y:.2e}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Add frequency band annotations
    if freq_max is None or freq_max > 30:
        # Add EEG frequency bands
        bands = {
            "Delta": (0.5, 4, "purple"),
            "Theta": (4, 8, "blue"),
            "Alpha": (8, 13, "green"),
            "Beta": (13, 30, "orange"),
            "Gamma": (30, 100, "red"),
        }

        for band_name, (low, high, color) in bands.items():
            if freq_max is None or high <= freq_max:
                fig.add_vrect(
                    x0=low,
                    x1=high,
                    fillcolor=color,
                    opacity=0.1,
                    layer="below",
                    line_width=0,
                    annotation_text=band_name,
                    annotation_position="top left",
                )

    # Update layout
    fig.update_layout(
        title=f"Enhanced FFT Analysis - {window_type.title()} Window",
        height=800,  # Increased height to provide more space for the subplots
        showlegend=True,
        hovermode="closest",
        margin=dict(l=80, r=50, t=80, b=80),  # Add margins to prevent overlap
        template="plotly_white",
        # Add pan/zoom tools
        dragmode="pan",
        modebar=dict(
            add=[
                "pan2d",
                "zoom2d",
                "select2d",
                "lasso2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "resetScale2d",
            ]
        ),
    )

    # Update axes labels with better positioning
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude", row=1, col=1, title_standoff=20)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="Power", row=2, col=1, title_standoff=20)

    # Adjust subplot title positions to prevent overlap
    fig.update_annotations(dict(font_size=14, font_color="black"))

    return fig


def create_stft_plot(
    selected_signal, sampling_freq, window_size, hop_size, freq_min, freq_max
):
    """Create enhanced STFT plot with vitalDSP insights."""
    logger.info(
        f"Creating STFT plot with window_size: {window_size}, hop_size: {hop_size}"
    )

    # Calculate STFT parameters
    overlap = window_size - hop_size
    time_resolution = hop_size / sampling_freq
    freq_resolution = sampling_freq / window_size

    logger.info(
        f"STFT Parameters - Time resolution: {time_resolution:.3f}s, Frequency resolution: {freq_resolution:.3f}Hz"
    )
    logger.info(f"Overlap: {overlap} samples ({overlap/window_size*100:.1f}%)")

    # Compute STFT
    freqs, times, Zxx = signal.stft(
        selected_signal, sampling_freq, nperseg=window_size, noverlap=overlap
    )

    # Calculate magnitude and power
    magnitude = np.abs(Zxx)
    power = magnitude**2

    # Find dominant time-frequency components
    max_power_idx = np.unravel_index(np.argmax(power), power.shape)
    max_freq = freqs[max_power_idx[0]]
    max_time = times[max_power_idx[1]]
    max_power = power[max_power_idx]

    logger.info(
        f"Dominant component: {max_freq:.2f} Hz at {max_time:.2f}s with power {max_power:.2e}"
    )

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
        rows=2,
        cols=2,
        subplot_titles=(
            "STFT Magnitude",
            "STFT Power",
            "Frequency Profile",
            "Time Profile",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
        vertical_spacing=0.15,  # Increased from 0.1 to prevent overlap
        horizontal_spacing=0.1,
    )

    # STFT Magnitude heatmap
    fig.add_trace(
        go.Heatmap(
            z=magnitude,
            x=times,
            y=freqs,
            colorscale="Viridis",
            name="Magnitude",
            hovertemplate="<b>Time:</b> %{x:.2f}s<br><b>Freq:</b> %{y:.2f}Hz<br><b>Magnitude:</b> %{z:.2e}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # STFT Power heatmap
    fig.add_trace(
        go.Heatmap(
            z=power,
            x=times,
            y=freqs,
            colorscale="Plasma",
            name="Power",
            hovertemplate="<b>Time:</b> %{x:.2f}s<br><b>Freq:</b> %{y:.2f}Hz<br><b>Power:</b> %{z:.2e}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Frequency profile at peak time
    peak_time_idx = np.argmax(np.max(power, axis=0))
    peak_time = times[peak_time_idx]
    freq_profile = magnitude[:, peak_time_idx]

    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=freq_profile,
            mode="lines",
            name=f"Freq Profile at {peak_time:.2f}s",
            line=dict(color="red", width=2),
            hovertemplate="<b>Frequency:</b> %{x:.2f} Hz<br><b>Magnitude:</b> %{y:.2e}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Time profile at peak frequency
    peak_freq_idx = np.argmax(np.max(power, axis=1))
    peak_freq = freqs[peak_freq_idx]
    time_profile = magnitude[peak_freq_idx, :]

    fig.add_trace(
        go.Scatter(
            x=times,
            y=time_profile,
            mode="lines",
            name=f"Time Profile at {peak_freq:.2f}Hz",
            line=dict(color="orange", width=2),
            hovertemplate="<b>Time:</b> %{x:.2f}s<br><b>Magnitude:</b> %{y:.2e}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title=f"Enhanced STFT Analysis - Window: {window_size}, Hop: {hop_size}",
        height=700,
        showlegend=True,
        hovermode="closest",
        margin=dict(l=80, r=50, t=80, b=80),  # Add margins to prevent overlap
        template="plotly_white",
        # Add pan/zoom tools
        dragmode="pan",
        modebar=dict(
            add=[
                "pan2d",
                "zoom2d",
                "select2d",
                "lasso2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "resetScale2d",
            ]
        ),
    )

    # Update axes labels with better positioning
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1, title_standoff=20)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=2, title_standoff=20)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude", row=2, col=1, title_standoff=20)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Magnitude", row=2, col=2, title_standoff=20)

    # Adjust subplot title positions to prevent overlap
    fig.update_annotations(dict(font_size=14, font_color="black"))

    return fig


def create_wavelet_plot(
    selected_signal, sampling_freq, wavelet_type, levels, freq_min, freq_max
):
    """Create enhanced wavelet plot with vitalDSP insights."""
    logger.info(f"Creating wavelet plot with type: {wavelet_type}, levels: {levels}")

    try:
        from vitalDSP.transforms.wavelet_transform import WaveletTransform

        # Validate input parameters
        if levels <= 0:
            logger.warning(f"Invalid levels value: {levels}, using default 4")
            levels = 4

        # Ensure levels is not too high to avoid subplot issues
        max_reasonable_levels = min(10, int(np.log2(len(selected_signal) / 10)))
        if levels > max_reasonable_levels:
            logger.warning(
                f"Levels value {levels} is too high for signal length {len(selected_signal)}. Using {max_reasonable_levels}"
            )
            levels = max_reasonable_levels

        if len(selected_signal) < 10:
            logger.warning(
                f"Signal too short for wavelet analysis: {len(selected_signal)} samples"
            )
            # Create a simple plot instead
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(selected_signal)) / sampling_freq,
                    y=selected_signal,
                    mode="lines",
                    name="Original Signal",
                    line=dict(color="black", width=2),
                )
            )
            fig.update_layout(
                title=f"Wavelet Analysis - Signal too short ({len(selected_signal)} samples)",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
            )
            return fig

        # Validate wavelet type
        wt = WaveletTransform(selected_signal, wavelet_name=wavelet_type)
        # Note: vitalDSP WaveletTransform handles validation internally

        # Validate signal data
        if np.any(np.isnan(selected_signal)) or np.any(np.isinf(selected_signal)):
            logger.warning("Signal contains NaN or Inf values, cleaning...")
            selected_signal = np.nan_to_num(
                selected_signal, nan=0.0, posinf=0.0, neginf=0.0
            )

        # Ensure signal data is finite
        if not np.all(np.isfinite(selected_signal)):
            logger.error("Signal data contains non-finite values after cleaning")
            # Create a simple plot instead
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(selected_signal)) / sampling_freq,
                    y=selected_signal,
                    mode="lines",
                    name="Original Signal",
                    line=dict(color="black", width=2),
                )
            )
            fig.update_layout(
                title="Wavelet Analysis - Invalid signal data",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
            )
            return fig

        # Validate sampling frequency
        if sampling_freq <= 0:
            logger.warning(
                f"Invalid sampling frequency: {sampling_freq}, using default 1000 Hz"
            )
            sampling_freq = 1000.0

        # Perform wavelet decomposition
        logger.info(
            f"Performing wavelet decomposition with {wavelet_type} wavelet, {levels} levels"
        )
        try:
            coefficients = wt.perform_wavelet_transform()
            # Convert to list format for compatibility
            coeffs = (
                [coefficients] if isinstance(coefficients, np.ndarray) else coefficients
            )
            logger.info(
                f"Wavelet decomposition completed. Number of coefficients: {len(coeffs)}"
            )

            # Validate coefficients
            if not coeffs or len(coeffs) == 0:
                logger.warning("Wavelet decomposition produced empty coefficients")
                # Create fallback plots
                main_fig = create_empty_figure()
                main_fig.update_layout(
                    title="Wavelet Analysis - No coefficients available"
                )

                psd_fig = create_empty_figure()
                psd_fig.update_layout(title="Wavelet PSD - No coefficients available")

                spectrogram_fig = create_empty_figure()
                spectrogram_fig.update_layout(
                    title="Wavelet Spectrogram - No coefficients available"
                )

                results = html.Div(
                    [
                        html.H5(f"Wavelet Analysis - {wavelet_type.upper()}"),
                        html.P(f"Decomposition levels: {levels}"),
                        html.P(f"Signal length: {len(selected_signal)} samples"),
                        html.P(f"Sampling frequency: {sampling_freq} Hz"),
                        html.P("Warning: No wavelet coefficients were produced"),
                    ]
                )

                # Create empty tables
                empty_table = create_empty_figure()
                empty_table.update_layout(title="No Data Available")

                freq_data = {"analysis_type": f"Wavelet_{wavelet_type.upper()}"}
                time_freq_data = {
                    "levels": levels,
                    "wavelet_type": wavelet_type,
                    "coefficients": [],
                }

                return (
                    main_fig,
                    psd_fig,
                    spectrogram_fig,
                    results,
                    empty_table,
                    empty_table,
                    empty_table,
                    empty_table,
                    freq_data,
                    time_freq_data,
                )

            # Check if we have the expected number of coefficients
            expected_coeffs = levels + 1
            if len(coeffs) != expected_coeffs:
                logger.warning(
                    f"Expected {expected_coeffs} coefficients, got {len(coeffs)}. Adjusting levels."
                )
                levels = len(coeffs) - 1
                total_rows = levels + 1
                logger.info(f"Adjusted levels to {levels}, total rows to {total_rows}")

        except Exception as e:
            logger.error(f"Error in wavelet decomposition: {e}")
            # Fallback to simple plot
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(selected_signal)) / sampling_freq,
                    y=selected_signal,
                    mode="lines",
                    name="Original Signal",
                    line=dict(color="black", width=2),
                )
            )
            fig.update_layout(
                title=f"Wavelet Analysis Error - Decomposition failed: {str(e)}",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
            )
            return fig

        # Calculate frequency bands for each level
        freq_bands = []
        for i in range(levels):
            # Approximate frequency for each level
            freq = sampling_freq / (2 ** (i + 1))
            freq_bands.append(freq)

        # Create enhanced plot with subplots
        # We need levels + 1 rows: 1 for original signal + levels for wavelet coefficients
        total_rows = levels + 1
        logger.info(f"Creating subplot with {total_rows} rows")

        # Create subplot titles - handle the case where levels might have been adjusted
        subplot_titles = ["Original Signal"]
        for i in range(levels):
            if i < len(freq_bands):
                subplot_titles.append(f"Level {i+1} ({freq_bands[i]:.1f} Hz)")
            else:
                subplot_titles.append(f"Level {i+1}")

        try:
            fig = make_subplots(
                rows=total_rows,
                cols=1,
                subplot_titles=subplot_titles,
                vertical_spacing=0.12,  # Increased from 0.05 to prevent overlap
            )
        except Exception as e:
            logger.error(f"Error creating subplots: {e}")
            # Fallback to simple plot
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(selected_signal)) / sampling_freq,
                    y=selected_signal,
                    mode="lines",
                    name="Original Signal",
                    line=dict(color="black", width=2),
                )
            )
            fig.update_layout(
                title=f"Wavelet Analysis - Subplot creation failed: {str(e)}",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
            )
            return fig

        # Original signal
        time_axis = np.arange(len(selected_signal)) / sampling_freq
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=selected_signal,
                mode="lines",
                name="Original Signal",
                line=dict(color="black", width=2),
            ),
            row=1,
            col=1,
        )

        # Wavelet coefficients - start from index 1 to skip approximation coefficients
        # since we already have the original signal plot
        logger.info(
            f"Adding wavelet coefficient traces. Coefficients available: {len(coeffs)}"
        )
        traces_added = 0

        for i in range(1, len(coeffs)):
            if i <= levels:  # Ensure we don't exceed the number of levels
                color = ["red", "green", "orange", "purple", "brown"][(i - 1) % 5]
                name = f"Detail Level {i}"

                # Final safety check for row index
                row_idx = i + 1
                if row_idx > total_rows:
                    logger.warning(
                        f"Row index {row_idx} exceeds total rows {total_rows} for level {i}, skipping"
                    )
                    continue

                try:
                    # Use a simpler approach - just use the coefficients directly
                    # This avoids potential issues with pywt.upcoef
                    logger.info(
                        f"Processing coefficients for level {i}, length: {len(coeffs[i])}"
                    )

                    # Pad or truncate coefficients to match signal length
                    if len(coeffs[i]) < len(selected_signal):
                        # Pad with zeros
                        padded_coeff = np.pad(
                            coeffs[i],
                            (0, len(selected_signal) - len(coeffs[i])),
                            mode="constant",
                        )
                        logger.info(
                            f"Padded coefficients from {len(coeffs[i])} to {len(padded_coeff)}"
                        )
                    else:
                        # Truncate
                        padded_coeff = coeffs[i][: len(selected_signal)]
                        logger.info(
                            f"Truncated coefficients from {len(coeffs[i])} to {len(padded_coeff)}"
                        )

                    logger.info(f"Adding trace for level {i} at row {row_idx}")
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=padded_coeff,
                            mode="lines",
                            name=name,
                            line=dict(color=color, width=1.5),
                        ),
                        row=row_idx,
                        col=1,
                    )
                    traces_added += 1
                except Exception as e:
                    logger.error(
                        f"Error processing wavelet coefficients for level {i}: {e}"
                    )
                    # Continue with other levels instead of failing completely
                    continue

        # Check if we successfully added any traces
        if traces_added == 0:
            logger.warning("No wavelet coefficient traces were added successfully")
            # Adjust the subplot to only show the original signal
            fig = make_subplots(
                rows=1,
                cols=1,
                subplot_titles=["Original Signal"],
                vertical_spacing=0.12,
            )
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=selected_signal,
                    mode="lines",
                    name="Original Signal",
                    line=dict(color="black", width=2),
                ),
                row=1,
                col=1,
            )
            fig.update_layout(
                title=f"Wavelet Analysis - {wavelet_type.upper()} Wavelet (Simplified)",
                height=400,
                showlegend=True,
                hovermode="closest",
                margin=dict(l=80, r=50, t=80, b=80),
            )
            fig.update_xaxes(title_text="Time (s)", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=1, col=1, title_standoff=20)
            return fig

        # Update layout
        fig.update_layout(
            title=f"Enhanced Wavelet Analysis - {wavelet_type.upper()} Wavelet, {levels} Levels",
            height=200 * total_rows,
            showlegend=True,
            hovermode="closest",
            margin=dict(l=80, r=50, t=80, b=80),  # Add margins to prevent overlap
            template="plotly_white",
            # Add pan/zoom tools
            dragmode="pan",
            modebar=dict(
                add=[
                    "pan2d",
                    "zoom2d",
                    "select2d",
                    "lasso2d",
                    "zoomIn2d",
                    "zoomOut2d",
                    "autoScale2d",
                    "resetScale2d",
                ]
            ),
        )

        # Update axes labels with better positioning
        for i in range(total_rows):
            fig.update_xaxes(title_text="Time (s)", row=i + 1, col=1)
            if i == 0:
                fig.update_yaxes(
                    title_text="Amplitude", row=i + 1, col=1, title_standoff=20
                )
            else:
                fig.update_yaxes(
                    title_text=f"Level {i} Coefficients",
                    row=i + 1,
                    col=1,
                    title_standoff=20,
                )

        # Adjust subplot title positions to prevent overlap
        fig.update_annotations(dict(font_size=14, font_color="black"))

        logger.info("Wavelet plot created successfully")
        return fig

    except ImportError:
        logger.warning("PyWavelets not available, creating placeholder plot")
        fig = create_empty_figure()
        fig.update_layout(title="Wavelet Analysis - PyWavelets not available")
        return fig
    except Exception as e:
        logger.error(f"Unexpected error in create_wavelet_plot: {e}")
        import traceback

        logger.error(traceback.format_exc())
        # Create a simple error plot instead of failing completely
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=(
                    np.arange(len(selected_signal)) / sampling_freq
                    if sampling_freq > 0
                    else np.arange(len(selected_signal))
                ),
                y=selected_signal,
                mode="lines",
                name="Original Signal",
                line=dict(color="black", width=2),
            )
        )
        fig.update_layout(
            title=f"Wavelet Analysis Error - {str(e)}",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
        )
        return fig


def perform_fft_analysis(
    time_data,
    selected_signal,
    sampling_freq,
    window_type,
    n_points,
    freq_min,
    freq_max,
    options,
):
    """Perform FFT analysis on the signal using scipy functions."""
    logger.info("=== FFT ANALYSIS DEBUG ===")
    logger.info(f"Input signal shape: {selected_signal.shape}")
    logger.info(f"Input time shape: {time_data.shape}")
    logger.info(f"Sampling frequency: {sampling_freq} Hz")
    logger.info(f"Window type: {window_type}")
    logger.info(f"N points: {n_points}")
    logger.info(f"Frequency range: {freq_min} to {freq_max}")

    # Apply window if specified
    if window_type and window_type != "none":
        if window_type == "hann":
            windowed_signal = selected_signal * signal.windows.hann(
                len(selected_signal)
            )
            logger.info("Applied Hann window for reduced spectral leakage")
        elif window_type == "hamming":
            windowed_signal = selected_signal * signal.windows.hamming(
                len(selected_signal)
            )
            logger.info("Applied Hamming window for optimal frequency resolution")
        elif window_type == "blackman":
            windowed_signal = selected_signal * signal.windows.blackman(
                len(selected_signal)
            )
            logger.info("Applied Blackman window for excellent sidelobe suppression")
        elif window_type == "kaiser":
            windowed_signal = selected_signal * signal.windows.kaiser(
                len(selected_signal), beta=14
            )
            logger.info(
                "Applied Kaiser window with beta=14 for optimal time-frequency trade-off"
            )
        else:
            windowed_signal = selected_signal
            logger.info("No window applied, using original signal")
    else:
        windowed_signal = selected_signal
        logger.info("No window applied, using original signal")

    # Compute FFT using scipy
    fft_result = rfft(windowed_signal, n=n_points)
    freqs = rfftfreq(n_points, 1 / sampling_freq)

    # Calculate magnitude and power spectrum
    fft_magnitude = np.abs(fft_result)
    power_spectrum = fft_magnitude**2

    logger.info(f"FFT result shape: {fft_result.shape}")
    logger.info(f"Frequencies shape: {freqs.shape}")
    logger.info(f"Magnitude shape: {fft_magnitude.shape}")
    logger.info(f"Frequency range: {freqs[0]:.3f} to {freqs[-1]:.3f} Hz")
    logger.info(
        f"Magnitude range: {np.min(fft_magnitude):.3e} to {np.max(fft_magnitude):.3e}"
    )

    # Filter frequency range if specified
    if freq_min is not None and freq_max is not None:
        freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
        freqs = freqs[freq_mask]
        fft_magnitude = fft_magnitude[freq_mask]
        power_spectrum = power_spectrum[freq_mask]
        logger.info(f"Filtered frequency range: {freq_min:.1f} - {freq_max:.1f} Hz")
        logger.info(f"Filtered frequencies shape: {freqs.shape}")
        logger.info(f"Filtered magnitude shape: {fft_magnitude.shape}")

    # Create enhanced FFT plot
    main_fig = create_fft_plot(
        selected_signal, sampling_freq, window_type, n_points, freq_min, freq_max
    )

    # PSD plot is the same as main for FFT
    psd_fig = main_fig

    # Create spectrogram plot for FFT (showing frequency vs magnitude as heatmap)
    # For FFT, we'll create a 2D representation showing frequency vs magnitude
    spectrogram_fig = go.Figure(
        data=go.Heatmap(
            z=[fft_magnitude],  # Single row for FFT
            x=freqs,
            y=["FFT Magnitude"],
            colorscale="Viridis",
            name="FFT Magnitude Spectrum",
        )
    )
    spectrogram_fig.update_layout(
        title=f"FFT Magnitude Spectrum - {window_type.title()} Window",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Analysis Type",
        height=400,
        showlegend=False,
    )

    logger.info("FFT analysis completed successfully")

    # Generate analysis results using the computed data
    results = generate_frequency_analysis_results(freqs, fft_magnitude, "FFT")

    # Generate tables
    peak_table = generate_peak_analysis_table(freqs, fft_magnitude)
    band_table = generate_band_power_table(freqs, fft_magnitude)
    stability_table = generate_stability_table(freqs, fft_magnitude)
    harmonics_table = generate_harmonics_table(freqs, fft_magnitude)

    # Store data
    freq_data = {
        "frequencies": freqs.tolist(),
        "magnitudes": fft_magnitude.tolist(),
        "analysis_type": "FFT",
    }
    time_freq_data = None

    return (
        main_fig,
        psd_fig,
        spectrogram_fig,
        results,
        peak_table,
        band_table,
        stability_table,
        harmonics_table,
        freq_data,
        time_freq_data,
    )


def perform_psd_analysis(
    time_data,
    selected_signal,
    sampling_freq,
    window,
    overlap,
    freq_max,
    log_scale,
    normalize,
    channel,
    freq_min,
    freq_max_range,
    options,
):
    """Perform Power Spectral Density analysis using vitalDSP functions."""
    logger.info("=== PSD ANALYSIS DEBUG ===")
    logger.info(f"Input signal shape: {selected_signal.shape}")
    logger.info(f"Input time shape: {time_data.shape}")
    logger.info(f"Sampling frequency: {sampling_freq} Hz")

    # Check if signal is too short for PSD analysis
    min_samples_required = 64  # Minimum samples needed for PSD analysis
    if len(selected_signal) < min_samples_required:
        logger.warning(
            f"Signal too short for PSD analysis: {len(selected_signal)} samples (minimum: {min_samples_required})"
        )
        # Return empty figures with error message
        empty_fig = create_empty_figure()
        empty_fig.add_annotation(
            text=f"Signal too short for PSD analysis<br>Current: {len(selected_signal)} samples<br>Minimum required: {min_samples_required} samples",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="red"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=2,
        )
        return (
            empty_fig,
            empty_fig,
            empty_fig,
            "Signal too short for PSD analysis",
            None,
            None,
            None,
            None,
            None,
            None,
        )
    logger.info(f"Window parameter: {window} (type: {type(window)})")
    logger.info(f"Overlap: {overlap}%")
    logger.info(f"Frequency range: {freq_min} to {freq_max_range}")

    # Set default values
    window_type = window or "hann"
    if isinstance(window, int):
        # Convert integer window to string representation
        window_map = {1: "hann", 2: "hamming", 3: "blackman", 4: "kaiser"}
        window_type = window_map.get(window, "hann")
        logger.info(f"Converted integer window {window} to window type: {window_type}")

    overlap = overlap or 50  # Now comes as percentage (0-100)
    freq_max = freq_max or sampling_freq / 2

    # Convert overlap from percentage to decimal
    overlap_decimal = overlap / 100.0 if overlap is not None else 0.5
    logger.info(f"Overlap decimal: {overlap_decimal}")

    # Calculate PSD using vitalDSP approach
    nperseg = max(64, min(256, len(selected_signal) // 4))
    noverlap = int(nperseg * overlap_decimal)
    logger.info(f"PSD parameters: nperseg={nperseg}, noverlap={noverlap}")

    # Ensure noverlap is less than nperseg
    if noverlap >= nperseg:
        noverlap = max(0, nperseg - 1)
        logger.info(f"Adjusted noverlap to {noverlap} to ensure noverlap < nperseg")

    # Use scipy.signal.welch with proper window function
    if window_type == "hann":
        window_func = signal.windows.hann(nperseg)
    elif window_type == "hamming":
        window_func = signal.windows.hamming(nperseg)
    elif window_type == "blackman":
        window_func = signal.windows.blackman(nperseg)
    else:
        window_func = signal.windows.hann(nperseg)

    freqs, psd = signal.welch(
        selected_signal,
        fs=sampling_freq,
        window=window_func,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    logger.info(f"PSD result: freqs shape={freqs.shape}, psd shape={psd.shape}")
    logger.info(f"PSD frequency range: {freqs[0]:.3f} to {freqs[-1]:.3f} Hz")
    logger.info(f"PSD power range: {np.min(psd):.3e} to {np.max(psd):.3e}")

    # Filter frequency range if specified
    if freq_min is not None and freq_max_range is not None:
        freq_mask = (freqs >= freq_min) & (freqs <= freq_max_range)
        freqs = freqs[freq_mask]
        psd = psd[freq_mask]
        logger.info(f"Filtered PSD: freqs shape={freqs.shape}, psd shape={psd.shape}")

    # Apply log scale if requested
    if log_scale and "on" in log_scale:
        psd_display = 10 * np.log10(psd + 1e-10)
        y_label = "Power Spectral Density (dB/Hz)"
        logger.info("Applied log scale to PSD")
    else:
        psd_display = psd
        y_label = "Power Spectral Density"
        logger.info("Using linear scale for PSD")

    # Normalize if requested
    if normalize and "on" in normalize:
        psd_display = psd_display / np.max(psd_display)
        y_label += " (Normalized)"
        logger.info("Normalized PSD")

    logger.info("PSD analysis completed successfully")

    # Create enhanced PSD plot with subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Power Spectral Density", "Cumulative Power"),
        vertical_spacing=0.15,  # Increased from 0.1 to prevent overlap
    )

    # Main PSD plot
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=psd_display,
            mode="lines",
            name=f"PSD ({window_type} window)",
            line=dict(color="red", width=2),
            hovertemplate="<b>Frequency:</b> %{x:.2f} Hz<br><b>PSD:</b> %{y:.2e}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Mark peak frequency
    peak_idx = np.argmax(psd_display)
    peak_freq = freqs[peak_idx]
    peak_psd = psd_display[peak_idx]

    fig.add_trace(
        go.Scatter(
            x=[peak_freq],
            y=[peak_psd],
            mode="markers",
            name="Peak Frequency",
            marker=dict(color="blue", size=10, symbol="diamond"),
            hovertemplate=f"<b>Peak:</b> {peak_freq:.2f} Hz<br><b>PSD:</b> {peak_psd:.2e}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Cumulative power plot
    cumulative_power = np.cumsum(psd) / np.sum(psd)
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=cumulative_power,
            mode="lines",
            name="Cumulative Power",
            line=dict(color="green", width=2),
            hovertemplate="<b>Frequency:</b> %{x:.2f} Hz<br><b>Cumulative Power:</b> %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Add frequency band annotations
    if freq_max_range is None or freq_max_range > 30:
        # Add physiological frequency bands
        bands = {
            "Delta": (0.5, 4, "purple"),
            "Theta": (4, 8, "blue"),
            "Alpha": (8, 13, "green"),
            "Beta": (13, 30, "orange"),
            "Gamma": (30, 100, "red"),
        }

        for band_name, (low, high, color) in bands.items():
            if freq_max_range is None or high <= freq_max_range:
                fig.add_vrect(
                    x0=low,
                    x1=high,
                    fillcolor=color,
                    opacity=0.1,
                    layer="below",
                    line_width=0,
                    annotation_text=band_name,
                    annotation_position="top left",
                )

    # Update layout
    fig.update_layout(
        title=f"Enhanced PSD Analysis - {window_type.title()} Window, {overlap}% Overlap",
        height=800,  # Increased height to provide more space for the second plot
        showlegend=True,
        hovermode="closest",
        margin=dict(l=80, r=50, t=80, b=80),  # Add margins to prevent overlap
        template="plotly_white",
        # Add pan/zoom tools
        dragmode="pan",
        modebar=dict(
            add=[
                "pan2d",
                "zoom2d",
                "select2d",
                "lasso2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "resetScale2d",
            ]
        ),
    )

    # Update axes labels with better positioning
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(title_text=y_label, row=1, col=1, title_standoff=20)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(
        title_text="Cumulative Power (Normalized)", row=2, col=1, title_standoff=20
    )

    # Adjust subplot title positions to prevent overlap
    fig.update_annotations(dict(font_size=14, font_color="black"))

    # Main PSD plot is the same as the enhanced plot
    main_fig = fig

    # PSD plot is the same as main
    psd_fig = main_fig

    # Create spectrogram plot
    # For PSD, we'll create a 2D representation showing frequency vs power
    spectrogram_fig = go.Figure(
        data=go.Heatmap(
            z=[psd_display],  # Single row for PSD
            x=freqs,
            y=["PSD Power"],
            colorscale="Plasma",
            name="PSD Power Spectrum",
        )
    )
    spectrogram_fig.update_layout(
        title=f"PSD Power Spectrum - {window_type.title()} Window",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Analysis Type",
        height=400,
        showlegend=False,
    )

    # Generate results and tables
    results = generate_frequency_analysis_results(freqs, psd_display, "PSD")
    peak_table = generate_peak_analysis_table(freqs, psd_display)
    band_table = generate_band_power_table(freqs, psd_display)
    stability_table = generate_stability_table(freqs, psd_display)
    harmonics_table = generate_harmonics_table(freqs, psd_display)

    # Store data
    freq_data = {
        "frequencies": freqs.tolist(),
        "magnitudes": psd_display.tolist(),
        "analysis_type": "PSD",
    }
    time_freq_data = None

    return (
        main_fig,
        psd_fig,
        spectrogram_fig,
        results,
        peak_table,
        band_table,
        stability_table,
        harmonics_table,
        freq_data,
        time_freq_data,
    )


def perform_stft_analysis(
    time_data,
    selected_signal,
    sampling_freq,
    window_size,
    hop_size,
    window_type,
    overlap,
    scaling,
    freq_max,
    colormap,
    freq_min,
    freq_max_range,
    options,
):
    """Perform Short-Time Fourier Transform analysis using vitalDSP functions."""
    # Check if signal is too short for STFT analysis
    min_samples_required = 64  # Minimum samples needed for STFT analysis
    if len(selected_signal) < min_samples_required:
        logger.warning(
            f"Signal too short for STFT analysis: {len(selected_signal)} samples (minimum: {min_samples_required})"
        )
        # Return empty figures with error message
        empty_fig = create_empty_figure()
        empty_fig.add_annotation(
            text=f"Signal too short for STFT analysis<br>Current: {len(selected_signal)} samples<br>Minimum required: {min_samples_required} samples",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="red"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=2,
        )
        return (
            empty_fig,
            empty_fig,
            empty_fig,
            "Signal too short for STFT analysis",
            None,
            None,
            None,
            None,
            None,
            None,
        )

    # Set default values
    window_size = max(64, min(256, len(selected_signal) // 4))
    hop_size = hop_size or max(16, window_size // 4)
    window_type = window_type or "hann"
    colormap = colormap or "viridis"
    overlap = overlap or 50  # Now comes as percentage (0-100)

    # Convert overlap from percentage to decimal
    overlap_decimal = overlap / 100.0 if overlap is not None else 0.5

    # Calculate noverlap and ensure it's less than window_size
    noverlap = int(window_size * overlap_decimal)
    if noverlap >= window_size:
        noverlap = max(0, window_size - 1)
        logger.info(
            f"Adjusted STFT noverlap to {noverlap} to ensure noverlap < window_size"
        )

    # Use vitalDSP STFT class
    try:
        logger.info("Attempting to import vitalDSP STFT...")
        from vitalDSP.transforms.stft import STFT

        logger.info("Successfully imported vitalDSP STFT")

        logger.info(
            f"Creating STFT object with window_size={window_size}, hop_size={hop_size}, n_fft={window_size}"
        )
        stft_obj = STFT(
            selected_signal,
            window_size=window_size,
            hop_size=hop_size,
            n_fft=window_size,
        )
        logger.info("Computing STFT using vitalDSP...")
        Zxx = stft_obj.compute_stft()
        logger.info(f"vitalDSP STFT completed, result shape: {Zxx.shape}")

        # Create frequency and time arrays
        freqs = np.fft.rfftfreq(window_size, 1 / sampling_freq)
        times = np.arange(Zxx.shape[1]) * hop_size / sampling_freq

        logger.info(f"STFT result shape: {Zxx.shape}")
        logger.info(f"Frequencies shape: {freqs.shape}")
        logger.info(f"Times shape: {times.shape}")

    except ImportError as e:
        logger.warning(f"vitalDSP not available: {e}, falling back to scipy")
        # Fallback to scipy implementation
        freqs, times, Zxx = signal.stft(
            selected_signal, fs=sampling_freq, nperseg=window_size, noverlap=noverlap
        )

    # Filter frequency range if specified
    if freq_min is not None and freq_max_range is not None:
        freq_mask = (freqs >= freq_min) & (freqs <= freq_max_range)
        freqs = freqs[freq_mask]
        Zxx = Zxx[freq_mask, :]

    # Create enhanced STFT plot
    main_fig = create_stft_plot(
        selected_signal, sampling_freq, window_size, hop_size, freq_min, freq_max_range
    )

    # PSD plot (average over time)
    psd = np.mean(np.abs(Zxx) ** 2, axis=1)
    psd_fig = go.Figure()
    psd_fig.add_trace(
        go.Scatter(
            x=freqs, y=psd, mode="lines", name="Average PSD", line=dict(color="red")
        )
    )
    psd_fig.update_layout(
        title="Average Power Spectral Density (STFT)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power Spectral Density",
        showlegend=True,
    )

    # Spectrogram plot is the same as main
    spectrogram_fig = main_fig

    # Generate results and tables
    results = generate_frequency_analysis_results(freqs, psd, "STFT")
    peak_table = generate_peak_analysis_table(freqs, psd)
    band_table = generate_band_power_table(freqs, psd)
    stability_table = generate_stability_table(freqs, psd)
    harmonics_table = generate_harmonics_table(freqs, psd)

    # Store data
    freq_data = {
        "frequencies": freqs.tolist(),
        "magnitudes": psd.tolist(),
        "analysis_type": "STFT",
    }
    time_freq_data = {
        "times": times.tolist(),
        "frequencies": freqs.tolist(),
        "magnitudes": (20 * np.log10(np.abs(Zxx) + 1e-10)).tolist(),
    }

    return (
        main_fig,
        psd_fig,
        spectrogram_fig,
        results,
        peak_table,
        band_table,
        stability_table,
        harmonics_table,
        freq_data,
        time_freq_data,
    )


def perform_wavelet_analysis(
    time_data,
    selected_signal,
    sampling_freq,
    wavelet_type,
    levels,
    freq_min,
    freq_max,
    options,
):
    """Perform Wavelet analysis on the signal."""
    # Check if signal is too short for Wavelet analysis
    min_samples_required = 32  # Minimum samples needed for Wavelet analysis
    if len(selected_signal) < min_samples_required:
        logger.warning(
            f"Signal too short for Wavelet analysis: {len(selected_signal)} samples (minimum: {min_samples_required})"
        )
        # Return empty figures with error message
        empty_fig = create_empty_figure()
        empty_fig.add_annotation(
            text=f"Signal too short for Wavelet analysis<br>Current: {len(selected_signal)} samples<br>Minimum required: {min_samples_required} samples",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="red"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=2,
        )
        return (
            empty_fig,
            empty_fig,
            empty_fig,
            "Signal too short for Wavelet analysis",
            None,
            None,
            None,
            None,
            None,
            None,
        )

    # Create enhanced wavelet plot
    main_fig = create_wavelet_plot(
        selected_signal, sampling_freq, wavelet_type, levels, freq_min, freq_max
    )

    # For wavelet analysis, we'll create a simplified PSD from the approximation coefficients
    try:
        from vitalDSP.transforms.wavelet_transform import WaveletTransform

        # Perform wavelet decomposition using vitalDSP
        wt = WaveletTransform(selected_signal, wavelet_name=wavelet_type)
        coefficients = wt.perform_wavelet_transform()
        # Convert to list format for compatibility
        coeffs = (
            [coefficients] if isinstance(coefficients, np.ndarray) else coefficients
        )

        # Validate coefficients
        if not coeffs or len(coeffs) == 0:
            logger.warning("Wavelet decomposition produced empty coefficients")
            # Create fallback plots
            psd_fig = create_empty_figure()
            psd_fig.update_layout(title="Wavelet PSD - No coefficients available")

            spectrogram_fig = create_empty_figure()
            spectrogram_fig.update_layout(
                title="Wavelet Spectrogram - No coefficients available"
            )

            results = html.Div(
                [
                    html.H5(f"Wavelet Analysis - {wavelet_type.upper()}"),
                    html.P(f"Decomposition levels: {levels}"),
                    html.P(f"Signal length: {len(selected_signal)} samples"),
                    html.P(f"Sampling frequency: {sampling_freq} Hz"),
                    html.P("Warning: No wavelet coefficients were produced"),
                ]
            )

            # Create empty tables
            empty_table = create_empty_figure()
            empty_table.update_layout(title="No Data Available")

            freq_data = {"analysis_type": f"Wavelet_{wavelet_type.upper()}"}
            time_freq_data = {
                "levels": levels,
                "wavelet_type": wavelet_type,
                "coefficients": [],
            }

            return (
                main_fig,
                psd_fig,
                spectrogram_fig,
                results,
                empty_table,
                empty_table,
                empty_table,
                empty_table,
                freq_data,
                time_freq_data,
            )

        # Use approximation coefficients for PSD-like analysis
        approx_coeffs = coeffs[0]

        # Validate approximation coefficients
        if approx_coeffs is None or len(approx_coeffs) == 0:
            logger.warning("Approximation coefficients are empty or None")
            # Create fallback plots
            psd_fig = create_empty_figure()
            psd_fig.update_layout(title="Wavelet PSD - No approximation coefficients")

            spectrogram_fig = create_empty_figure()
            spectrogram_fig.update_layout(title="Wavelet Spectrogram - No coefficients")

            results = html.Div(
                [
                    html.H5(f"Wavelet Analysis - {wavelet_type.upper()}"),
                    html.P(f"Decomposition levels: {levels}"),
                    html.P(f"Signal length: {len(selected_signal)} samples"),
                    html.P(f"Sampling frequency: {sampling_freq} Hz"),
                    html.P("Warning: No approximation coefficients were produced"),
                ]
            )

            # Create empty tables
            empty_table = create_empty_figure()
            empty_table.update_layout(title="No Data Available")

            freq_data = {"analysis_type": f"Wavelet_{wavelet_type.upper()}"}
            time_freq_data = {
                "levels": levels,
                "wavelet_type": wavelet_type,
                "coefficients": [],
            }

            return (
                main_fig,
                psd_fig,
                spectrogram_fig,
                results,
                empty_table,
                empty_table,
                empty_table,
                empty_table,
                freq_data,
                time_freq_data,
            )

        # Create PSD plot from approximation coefficients
        psd_fig = go.Figure()
        psd_fig.add_trace(
            go.Scatter(
                x=np.arange(len(approx_coeffs)),
                y=np.abs(approx_coeffs),
                mode="lines",
                name=f"Wavelet {wavelet_type.upper()}",
                line=dict(color="blue"),
            )
        )
        psd_fig.update_layout(
            title=f"Wavelet Approximation Coefficients - {wavelet_type.upper()}",
            xaxis_title="Coefficient Index",
            yaxis_title="Magnitude",
            showlegend=True,
        )

        # Spectrogram plot for wavelet (showing coefficient levels)
        spectrogram_fig = go.Figure()

        # Create a heatmap of wavelet coefficients
        try:
            # Pad coefficients to same length first, then create array
            max_len = max(len(coeff) for coeff in coeffs[1:]) if len(coeffs) > 1 else 0
            if max_len > 0:
                padded_coeffs = []
                for coeff in coeffs[1:]:
                    padded = np.pad(coeff, (0, max_len - len(coeff)), mode="constant")
                    padded_coeffs.append(padded)

                coeff_matrix = np.array(padded_coeffs)

                spectrogram_fig.add_trace(
                    go.Heatmap(
                        z=np.abs(coeff_matrix),
                        x=np.arange(coeff_matrix.shape[1]),
                        y=[f"Level {i+1}" for i in range(coeff_matrix.shape[0])],
                        colorscale="Viridis",
                        name="Wavelet Coefficients",
                    )
                )
                spectrogram_fig.update_layout(
                    title=f"Wavelet Coefficient Levels - {wavelet_type.upper()}",
                    xaxis_title="Coefficient Index",
                    yaxis_title="Decomposition Level",
                    showlegend=False,
                )
            else:
                spectrogram_fig.update_layout(title="No wavelet coefficients available")
        except Exception as e:
            logger.warning(f"Error creating wavelet coefficient heatmap: {e}")
            spectrogram_fig.update_layout(
                title=f"Wavelet Coefficients - Error: {str(e)}"
            )

        # Generate results and tables
        results = html.Div(
            [
                html.H5(f"Wavelet Analysis - {wavelet_type.upper()}"),
                html.P(f"Decomposition levels: {levels}"),
                html.P(f"Signal length: {len(selected_signal)} samples"),
                html.P(f"Sampling frequency: {sampling_freq} Hz"),
            ]
        )

        # Create tables from approximation coefficients
        freqs = np.arange(len(approx_coeffs))
        magnitudes = np.abs(approx_coeffs)

        # Validate frequency and magnitude arrays
        if len(freqs) == 0 or len(magnitudes) == 0:
            logger.warning("Frequency or magnitude arrays are empty")
            # Create empty tables
            empty_table = create_empty_figure()
            empty_table.update_layout(title="No Data Available")

            peak_table = empty_table
            band_table = empty_table
            stability_table = empty_table
            harmonics_table = empty_table
        else:
            peak_table = generate_peak_analysis_table(freqs, magnitudes)
            band_table = generate_band_power_table(freqs, magnitudes)
            stability_table = generate_stability_table(freqs, magnitudes)
            harmonics_table = generate_harmonics_table(freqs, magnitudes)

        # Store data
        freq_data = {
            "frequencies": freqs.tolist(),
            "magnitudes": magnitudes.tolist(),
            "analysis_type": f"Wavelet_{wavelet_type.upper()}",
        }
        time_freq_data = {
            "levels": levels,
            "wavelet_type": wavelet_type,
            "coefficients": [
                coeff.tolist()
                for coeff in coeffs
                if coeff is not None and len(coeff) > 0
            ],
        }

    except ImportError:
        logger.warning("PyWavelets not available, creating placeholder plots")

        psd_fig = create_empty_figure()
        psd_fig.update_layout(title="Wavelet PSD - PyWavelets not available")

        spectrogram_fig = create_empty_figure()
        spectrogram_fig.update_layout(
            title="Wavelet Spectrogram - PyWavelets not available"
        )

        results = html.Div(
            [
                html.H5("Wavelet Analysis"),
                html.P("Wavelet analysis is not yet implemented in this version."),
            ]
        )

        empty_table = create_empty_figure()
        empty_table.update_layout(title="No Data")

        freq_data = {"analysis_type": "Wavelet"}
        time_freq_data = None

        peak_table = empty_table
        band_table = empty_table
        stability_table = empty_table
        harmonics_table = empty_table

    return (
        main_fig,
        psd_fig,
        spectrogram_fig,
        results,
        peak_table,
        band_table,
        stability_table,
        harmonics_table,
        freq_data,
        time_freq_data,
    )


def generate_frequency_analysis_results(freqs, magnitudes, analysis_type):
    """Generate comprehensive frequency analysis results."""
    if len(freqs) == 0 or len(magnitudes) == 0:
        return html.Div("No data available for analysis")

    # Calculate basic statistics
    max_mag = np.max(magnitudes)
    max_freq = freqs[np.argmax(magnitudes)]
    mean_mag = np.mean(magnitudes)
    std_mag = np.std(magnitudes)

    # Calculate total power
    total_power = np.sum(magnitudes)

    # Find dominant frequency bands
    if len(freqs) > 1:
        freq_step = freqs[1] - freqs[0]
        power_density = magnitudes / freq_step
        total_power_density = np.sum(power_density)
    else:
        total_power_density = total_power

    return html.Div(
        [
            html.H5(f"{analysis_type} Analysis Results"),
            html.Div(
                [
                    html.Div(
                        [
                            html.H6("Peak Analysis"),
                            html.P(f"Peak Frequency: {max_freq:.2f} Hz"),
                            html.P(f"Peak Magnitude: {format_large_number(max_mag)}"),
                            html.P(f"Mean Magnitude: {format_large_number(mean_mag)}"),
                            html.P(
                                f"Standard Deviation: {format_large_number(std_mag)}"
                            ),
                        ],
                        className="col-md-6",
                    ),
                    html.Div(
                        [
                            html.H6("Power Analysis"),
                            html.P(f"Total Power: {format_large_number(total_power)}"),
                            html.P(
                                f"Power Density: {format_large_number(total_power_density)}"
                            ),
                            html.P(
                                f"Frequency Range: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz"
                            ),
                            html.P(
                                f"Frequency Resolution: {freqs[1] - freqs[0]:.4f} Hz"
                            ),
                        ],
                        className="col-md-6",
                    ),
                ],
                className="row",
            ),
        ]
    )


def generate_peak_analysis_table(freqs, magnitudes):
    """Generate peak analysis table."""
    if len(freqs) == 0 or len(magnitudes) == 0:
        return html.Div("No data available for peak analysis")

    # Find peaks
    peaks, _ = signal.find_peaks(magnitudes, height=np.max(magnitudes) * 0.1)

    if len(peaks) == 0:
        return html.Div("No significant peaks found")

    # Create table
    table_data = []
    for i, peak_idx in enumerate(peaks[:10]):  # Top 10 peaks
        table_data.append(
            {
                "Peak #": i + 1,
                "Frequency (Hz)": f"{freqs[peak_idx]:.2f}",
                "Magnitude": f"{magnitudes[peak_idx]:.4f}",
                "Relative Power (%)": f"{magnitudes[peak_idx]/np.max(magnitudes)*100:.1f}",
            }
        )

    return html.Div(
        [
            html.H6("Peak Analysis"),
            html.Table(
                [
                    html.Thead(
                        [html.Tr([html.Th(col) for col in table_data[0].keys()])]
                    ),
                    html.Tbody(
                        [
                            html.Tr([html.Td(row[col]) for col in row.keys()])
                            for row in table_data
                        ]
                    ),
                ],
                className="table table-sm",
            ),
        ]
    )


def generate_band_power_table(freqs, magnitudes):
    """Generate frequency band power table."""
    if len(freqs) == 0 or len(magnitudes) == 0:
        return html.Div("No data available for band power analysis")

    # Define frequency bands
    bands = [
        (0.1, 0.5, "Very Low Frequency (VLF)"),
        (0.5, 2.0, "Low Frequency (LF)"),
        (2.0, 5.0, "Medium Frequency (MF)"),
        (5.0, 10.0, "High Frequency (HF)"),
        (10.0, 50.0, "Very High Frequency (VHF)"),
    ]

    table_data = []
    for low_freq, high_freq, band_name in bands:
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        if np.any(mask):
            band_power = np.sum(magnitudes[mask])
            total_power = np.sum(magnitudes)

            # Handle negative power values by using absolute values for percentage calculation
            # This ensures percentage is always meaningful regardless of power sign
            if total_power != 0:
                # Use absolute values for percentage calculation to handle negative powers
                percentage = (np.abs(band_power) / np.abs(total_power)) * 100
            else:
                percentage = 0

            table_data.append(
                {
                    "Band": band_name,
                    "Range (Hz)": f"{low_freq:.1f} - {high_freq:.1f}",
                    "Power": f"{band_power:.4f}",
                    "Percentage": f"{percentage:.1f}%",
                }
            )

    if not table_data:
        return html.Div("No band power data available")

    return html.Div(
        [
            html.H6("Frequency Band Power"),
            html.Table(
                [
                    html.Thead(
                        [html.Tr([html.Th(col) for col in table_data[0].keys()])]
                    ),
                    html.Tbody(
                        [
                            html.Tr([html.Td(row[col]) for col in row.keys()])
                            for row in table_data
                        ]
                    ),
                ],
                className="table table-sm",
            ),
        ]
    )


def generate_stability_table(freqs, magnitudes):
    """Generate signal stability analysis table."""
    if len(freqs) == 0 or len(magnitudes) == 0:
        return html.Div("No data available for stability analysis")

    # Calculate stability metrics
    mean_mag = np.mean(magnitudes)
    std_mag = np.std(magnitudes)
    cv = (std_mag / mean_mag) * 100 if mean_mag > 0 else 0

    # Calculate frequency stability
    if len(freqs) > 1:
        freq_variation = np.std(np.diff(freqs))
    else:
        freq_variation = 0

    return html.Div(
        [
            html.H6("Signal Stability Analysis"),
            html.Table(
                [
                    html.Thead([html.Tr([html.Th("Metric"), html.Th("Value")])]),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td("Coefficient of Variation"),
                                    html.Td(f"{cv:.2f}%"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Frequency Variation"),
                                    html.Td(f"{freq_variation:.4f} Hz"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Magnitude Stability"),
                                    html.Td("Stable" if cv < 20 else "Variable"),
                                ]
                            ),
                        ]
                    ),
                ],
                className="table table-sm",
            ),
        ]
    )


def generate_harmonics_table(freqs, magnitudes):
    """Generate harmonics analysis table."""
    if len(freqs) == 0 or len(magnitudes) == 0:
        return html.Div("No data available for harmonics analysis")

    # Find fundamental frequency (highest peak)
    max_peak_idx = np.argmax(magnitudes)
    fundamental_freq = freqs[max_peak_idx]

    # Look for harmonics
    harmonics = []
    for i in range(2, 6):  # Check up to 5th harmonic
        harmonic_freq = fundamental_freq * i
        # Find closest frequency
        closest_idx = np.argmin(np.abs(freqs - harmonic_freq))
        if closest_idx < len(magnitudes):
            harmonic_magnitude = magnitudes[closest_idx]
            harmonic_ratio = (
                harmonic_magnitude / magnitudes[max_peak_idx]
                if magnitudes[max_peak_idx] > 0
                else 0
            )
            harmonics.append(
                {
                    "Harmonic": (
                        f"{i}nd" if i == 2 else f"{i}rd" if i == 3 else f"{i}th"
                    ),
                    "Frequency (Hz)": f"{freqs[closest_idx]:.2f}",
                    "Magnitude": f"{harmonic_magnitude:.4f}",
                    "Ratio to Fundamental": f"{harmonic_ratio:.3f}",
                }
            )

    if not harmonics:
        return html.Div("No significant harmonics detected")

    return html.Div(
        [
            html.H6("Harmonics Analysis"),
            html.P(f"Fundamental Frequency: {fundamental_freq:.2f} Hz"),
            html.Table(
                [
                    html.Thead(
                        [html.Tr([html.Th(col) for col in harmonics[0].keys()])]
                    ),
                    html.Tbody(
                        [
                            html.Tr([html.Td(row[col]) for col in row.keys()])
                            for row in harmonics
                        ]
                    ),
                ],
                className="table table-sm",
            ),
        ]
    )
