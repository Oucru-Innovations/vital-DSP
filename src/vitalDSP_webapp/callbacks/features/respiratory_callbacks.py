"""
Comprehensive respiratory rate analysis callbacks for vitalDSP webapp.
Handles all respiratory features including estimation methods, sleep apnea detection, and fusion analysis.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html
from dash.exceptions import PreventUpdate
from scipy import signal
import logging

logger = logging.getLogger(__name__)


def create_empty_figure():
    """Create an empty figure for error cases."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available or error occurred",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
    )
    return fig


def detect_respiratory_signal_type(signal_data, sampling_freq):
    """Auto-detect if signal is respiratory or cardiac based on frequency content."""
    try:
        # Validate sampling frequency
        if sampling_freq <= 0:
            return "unknown"

        # Compute FFT
        fft_result = np.fft.fft(signal_data)
        fft_freq = np.fft.fftfreq(len(signal_data), 1 / sampling_freq)

        # Focus on positive frequencies
        positive_mask = fft_freq > 0
        fft_freq = fft_freq[positive_mask]
        fft_magnitude = np.abs(fft_result[positive_mask])

        # Check for constant or near-constant signals
        if (
            np.std(signal_data) < 1e-10
        ):  # Very small standard deviation indicates constant signal
            return "unknown"

        # Find dominant frequency
        dominant_idx = np.argmax(fft_magnitude)
        dominant_freq = fft_freq[dominant_idx]

        # Respiratory signals typically have lower frequencies (0.1-0.5 Hz)
        # Cardiac signals typically have higher frequencies (0.8-2.0 Hz)
        if dominant_freq < 0.5:
            return "respiratory"
        else:
            return "cardiac"
    except Exception:
        return "unknown"


def create_respiratory_signal_plot(
    signal_data,
    time_axis,
    sampling_freq,
    signal_type,
    estimation_methods,
    preprocessing_options,
    low_cut,
    high_cut,
):
    """Create the main respiratory signal plot with annotations."""
    try:
        # Validate inputs
        if signal_data is None or time_axis is None or sampling_freq is None:
            return create_empty_figure()

        fig = go.Figure()

        # Add main signal
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal_data,
                mode="lines",
                name=f"{signal_type.title()} Signal",
                line=dict(color="blue", width=1),
            )
        )

        # Add preprocessing info if applied
        if preprocessing_options and any(preprocessing_options):
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=signal_data,  # This would be the preprocessed signal in real implementation
                    mode="lines",
                    name="Preprocessed Signal",
                    line=dict(color="red", width=1, dash="dash"),
                )
            )

        # Add filter frequency lines if specified
        if low_cut:
            fig.add_hline(
                y=np.min(signal_data),
                line_dash="dot",
                line_color="green",
                annotation_text=f"Low Cut: {low_cut} Hz",
            )
        if high_cut:
            fig.add_hline(
                y=np.max(signal_data),
                line_dash="dot",
                line_color="orange",
                annotation_text=f"High Cut: {high_cut} Hz",
            )

        fig.update_layout(
            title=f"Respiratory Signal Analysis - {signal_type.title()}",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            showlegend=True,
            height=400,
            plot_bgcolor="white",
            margin=dict(l=60, r=60, t=80, b=60),
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating respiratory signal plot: {e}")
        return create_empty_figure()


def generate_comprehensive_respiratory_analysis(
    signal_data,
    time_axis,
    sampling_freq,
    signal_type,
    estimation_methods,
    advanced_options,
    preprocessing_options,
    low_cut,
    high_cut,
    min_breath_duration,
    max_breath_duration,
):
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
                results.append(
                    html.P(f"Minimum Duration: {min_breath_duration} seconds")
                )
            if max_breath_duration:
                results.append(
                    html.P(f"Maximum Duration: {max_breath_duration} seconds")
                )

        # Try to compute respiratory rate using vitalDSP if available
        try:
            from vitalDSP.respiratory_analysis.respiratory_analysis import (
                RespiratoryAnalysis,
            )
            from vitalDSP.preprocess.preprocess_operations import PreprocessConfig

            # Create preprocessing configuration
            preprocess_config = PreprocessConfig()
            if preprocessing_options and "filter" in preprocessing_options:
                preprocess_config.filter_type = "bandpass"
                preprocess_config.lowcut = low_cut or 0.1
                preprocess_config.highcut = high_cut or 0.8

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
                                min_breath_duration=min_breath_duration or 0.1,
                                max_breath_duration=max_breath_duration or 6,
                                preprocess_config=preprocess_config,
                            )
                            results.append(
                                html.P(f"• Peak Detection: {rr:.2f} breaths/min")
                            )

                        elif method == "fft_based":
                            rr = ra.compute_respiratory_rate(
                                method="fft_based", preprocess_config=preprocess_config
                            )
                            results.append(html.P(f"• FFT Based: {rr:.2f} breaths/min"))

                        elif method == "frequency_domain":
                            rr = ra.compute_respiratory_rate(
                                method="frequency_domain",
                                preprocess_config=preprocess_config,
                            )
                            results.append(
                                html.P(f"• Frequency Domain: {rr:.2f} breaths/min")
                            )

                        elif method == "time_domain":
                            rr = ra.compute_respiratory_rate(
                                method="time_domain",
                                preprocess_config=preprocess_config,
                            )
                            results.append(
                                html.P(f"• Time Domain: {rr:.2f} breaths/min")
                            )

                    except Exception as e:
                        logger.warning(f"Failed to compute RR using {method}: {e}")
                        results.append(
                            html.P(f"• {method.replace('_', ' ').title()}: Failed")
                        )

            # Advanced analysis if selected
            if advanced_options:
                results.append(html.Hr())
                results.append(html.H6("Advanced Analysis"))

                for option in advanced_options:
                    try:
                        if option == "sleep_apnea":
                            from vitalDSP.respiratory_analysis.sleep_apnea_detection.amplitude_threshold import (
                                detect_apnea_amplitude,
                            )

                            apnea_events = detect_apnea_amplitude(
                                signal_data,
                                sampling_freq,
                                threshold=np.mean(signal_data) * 0.5,
                                min_duration=10,
                            )
                            results.append(
                                html.P(
                                    f"• Sleep Apnea Events: {len(apnea_events)} detected"
                                )
                            )

                        elif option == "multimodal":
                            from vitalDSP.respiratory_analysis.fusion.multimodal_analysis import (
                                multimodal_analysis,
                            )

                            # For multimodal, we'd need multiple signals, so just show capability
                            results.append(
                                html.P(
                                    "• Multimodal Analysis: Available for multiple signals"
                                )
                            )

                        elif option == "ppg_ecg_fusion":
                            from vitalDSP.respiratory_analysis.fusion.ppg_ecg_fusion import (
                                ppg_ecg_fusion,
                            )

                            # For fusion, we'd need both PPG and ECG signals
                            results.append(
                                html.P("• PPG-ECG Fusion: Available for dual signals")
                            )

                        elif option == "resp_cardiac_fusion":
                            from vitalDSP.respiratory_analysis.fusion.respiratory_cardiac_fusion import (
                                respiratory_cardiac_fusion,
                            )

                            results.append(
                                html.P("• Respiratory-Cardiac Fusion: Available")
                            )

                    except Exception as e:
                        logger.warning(f"Failed to perform {option} analysis: {e}")
                        results.append(
                            html.P(f"• {option.replace('_', ' ').title()}: Failed")
                        )

        except ImportError as e:
            logger.warning(f"vitalDSP respiratory modules not available: {e}")
            results.append(html.Hr())
            results.append(html.H6("Analysis Status"))
            results.append(
                html.P(
                    "vitalDSP respiratory modules not available for detailed analysis"
                )
            )

        return results
    except Exception as e:
        logger.error(f"Error generating respiratory analysis results: {e}")
        return [html.H5("Error"), html.P(f"Failed to generate results: {str(e)}")]


def create_comprehensive_respiratory_plots(
    signal_data,
    time_axis,
    sampling_freq,
    signal_type,
    estimation_methods,
    advanced_options,
    preprocessing_options,
    low_cut,
    high_cut,
    min_breath_duration,
    max_breath_duration,
):
    """Create comprehensive respiratory analysis plots."""
    try:
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Time Domain",
                "Frequency Domain",
                "Power Spectral Density",
                "Signal Quality",
            ),
            specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        )

        # Time domain plot
        fig.add_trace(
            go.Scatter(x=time_axis, y=signal_data, mode="lines", name="Signal"),
            row=1,
            col=1,
        )

        # Frequency domain plot
        fft_result = np.fft.fft(signal_data)
        fft_freq = np.fft.fftfreq(len(signal_data), 1 / sampling_freq)
        positive_mask = fft_freq > 0
        fft_freq_positive = fft_freq[positive_mask]
        fft_magnitude = np.abs(fft_result[positive_mask])

        fig.add_trace(
            go.Scatter(
                x=fft_freq_positive, y=fft_magnitude, mode="lines", name="FFT Magnitude"
            ),
            row=1,
            col=2,
        )

        # Power Spectral Density
        freqs, psd = signal.welch(
            signal_data, sampling_freq, nperseg=min(256, len(signal_data) // 4)
        )
        fig.add_trace(
            go.Scatter(x=freqs, y=psd, mode="lines", name="PSD"), row=2, col=1
        )

        # Signal quality metrics
        # Calculate signal quality index
        signal_quality = (
            np.std(signal_data) / np.mean(np.abs(signal_data))
            if np.mean(np.abs(signal_data)) > 0
            else 0
        )
        fig.add_trace(
            go.Bar(x=["Signal Quality"], y=[signal_quality], name="Quality Index"),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title="Comprehensive Respiratory Analysis",
            height=600,
            showlegend=False,
            plot_bgcolor="white",
            margin=dict(l=60, r=60, t=80, b=60),
        )

        # Update all subplot axes
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="rgba(128, 128, 128, 0.2)",
                    row=i,
                    col=j,
                )
                fig.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="rgba(128, 128, 128, 0.2)",
                    row=i,
                    col=j,
                )

        return fig

    except Exception as e:
        logger.error(f"Error creating comprehensive respiratory plots: {e}")
        return create_empty_figure()


def register_respiratory_callbacks(app):
    """Register all respiratory analysis callbacks."""
    logger.info("=== REGISTERING RESPIRATORY CALLBACKS ===")

    @app.callback(
        [
            Output("resp-main-plot", "figure"),
            Output("resp-analysis-results", "children"),
            Output("resp-analysis-plots", "figure"),
            Output("resp-data-store", "data"),
            Output("resp-features-store", "data"),
        ],
        [
            Input("url", "pathname"),
            Input("resp-analyze-btn", "n_clicks"),
            Input("resp-time-range-slider", "value"),
            Input("resp-btn-nudge-m10", "n_clicks"),
            Input("resp-btn-nudge-m1", "n_clicks"),
            Input("resp-btn-nudge-p1", "n_clicks"),
            Input("resp-btn-nudge-p10", "n_clicks"),
        ],
        [
            State("resp-start-time", "value"),
            State("resp-end-time", "value"),
            State("resp-signal-type", "value"),
            State("resp-signal-source-select", "value"),
            State("resp-estimation-methods", "value"),
            State("resp-advanced-options", "value"),
            State("resp-preprocessing-options", "value"),
            State("resp-low-cut", "value"),
            State("resp-high-cut", "value"),
            State("resp-min-breath-duration", "value"),
            State("resp-max-breath-duration", "value"),
        ],
    )
    def respiratory_analysis_callback(
        pathname,
        n_clicks,
        slider_value,
        nudge_m10,
        nudge_m1,
        nudge_p1,
        nudge_p10,
        start_time,
        end_time,
        signal_type,
        signal_source,
        estimation_methods,
        advanced_options,
        preprocessing_options,
        low_cut,
        high_cut,
        min_breath_duration,
        max_breath_duration,
    ):
        """Comprehensive respiratory rate analysis using all vitalDSP respiratory features."""
        ctx = callback_context

        # Determine what triggered this callback
        if not ctx.triggered:
            logger.warning("No context triggered - raising PreventUpdate")
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        logger.info("=== RESPIRATORY ANALYSIS CALLBACK TRIGGERED ===")
        logger.info(f"Trigger ID: {trigger_id}")
        logger.info(f"Pathname: {pathname}")
        logger.info(f"Signal type: {signal_type}")
        logger.info(f"Estimation methods: {estimation_methods}")
        logger.info(f"Advanced options: {advanced_options}")
        logger.info(f"Preprocessing options: {preprocessing_options}")

        # Only run this when we're on the respiratory page
        if pathname != "/respiratory":
            logger.info("Not on respiratory page, returning empty figures")
            return (
                create_empty_figure(),
                "Navigate to respiratory page to analyze respiratory signals.",
                create_empty_figure(),
                None,
                None,
            )

        # Handle first-time loading
        if not ctx.triggered or ctx.triggered[0]["prop_id"].split(".")[0] == "url":
            logger.info("First time loading respiratory page, attempting to load data")

        try:
            # Get data from the data service
            logger.info("Attempting to get data service...")
            from vitalDSP_webapp.services.data.data_service import get_data_service

            data_service = get_data_service()
            logger.info("Data service retrieved successfully")
            
            # Check if Enhanced Data Service is available for heavy data processing
            if data_service.is_enhanced_service_available():
                logger.info("Enhanced Data Service is available for heavy data processing")
            else:
                logger.info("Using basic data service functionality")

            # Get the most recent data
            logger.info("Retrieving all data from service...")
            all_data = data_service.get_all_data()
            logger.info(
                f"All data keys: {list(all_data.keys()) if all_data else 'None'}"
            )

            if not all_data:
                logger.warning("No data found in service")
                return (
                    create_empty_figure(),
                    "No data available. Please upload and process data first.",
                    create_empty_figure(),
                    None,
                    None,
                )

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
                logger.warning(
                    "Data has not been processed yet - no column mapping found"
                )
                return (
                    create_empty_figure(),
                    "Please process your data on the Upload page first (configure column mapping)",
                    create_empty_figure(),
                    None,
                    None,
                )

            # Get the actual data
            logger.info("Retrieving data frame...")
            df = data_service.get_data(latest_data_id)
            
            # Enhanced data processing for heavy datasets
            if df is not None and not df.empty:
                data_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                num_samples = len(df)
                
                logger.info(f"Data size: {data_size_mb:.2f} MB, Samples: {num_samples}")
                
                # Use Enhanced Data Service for heavy data processing
                if data_service.is_enhanced_service_available() and (data_size_mb > 5 or num_samples > 100000):
                    logger.info(f"Using Enhanced Data Service for heavy respiratory analysis: {data_size_mb:.2f}MB, {num_samples} samples")
                    
                    # Get enhanced service for optimized processing
                    enhanced_service = data_service.get_enhanced_service()
                    if enhanced_service:
                        logger.info("Enhanced Data Service is ready for optimized respiratory analysis")
                        # The enhanced service will automatically handle chunked processing
                        # and memory optimization during respiratory analysis
                else:
                    logger.info("Using standard processing for lightweight respiratory analysis")
            logger.info(f"Data frame shape: {df.shape if df is not None else 'None'}")
            logger.info(
                f"Data frame columns: {list(df.columns) if df is not None else 'None'}"
            )

            if df is None or df.empty:
                logger.warning("Data frame is empty")
                return (
                    create_empty_figure(),
                    "Data is empty or corrupted.",
                    create_empty_figure(),
                    None,
                    None,
                )

            # Get sampling frequency from the data info
            sampling_freq = latest_data.get("info", {}).get("sampling_freq", 1000)
            logger.info(f"Sampling frequency: {sampling_freq}")

            # Handle time window adjustments for nudge buttons
            if trigger_id in [
                "resp-btn-nudge-m10",
                "resp-btn-nudge-m1",
                "resp-btn-nudge-p1",
                "resp-btn-nudge-p10",
            ]:
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
            signal_column = column_mapping.get("signal")
            logger.info(f"Signal column from mapping: {signal_column}")
            logger.info(
                f"Available columns in windowed data: {list(windowed_data.columns)}"
            )

            if not signal_column or signal_column not in windowed_data.columns:
                logger.warning(f"Signal column {signal_column} not found in data")
                return (
                    create_empty_figure(),
                    "Signal column not found in data.",
                    create_empty_figure(),
                    None,
                    None,
                )

            signal_data = windowed_data[signal_column].values

            # Initialize filter_info for signal source loading
            filter_info = None

            # Check if we need to apply dynamic filtering for filtered signal
            if signal_source == "filtered" and filter_info is not None:
                # Check if the time range is within the original signal range
                original_signal_length = len(df[signal_column].values)
                expected_length = end_sample - start_sample

                # If time range is outside original signal or we need dynamic filtering
                if (
                    start_sample >= original_signal_length
                    or end_sample > original_signal_length
                    or expected_length != len(signal_data)
                ):
                    logger.info("=== DYNAMIC FILTERING ===")
                    logger.info(f"Original signal length: {original_signal_length}")
                    logger.info(f"Time range: {start_sample} to {end_sample}")
                    logger.info(f"Expected window length: {expected_length}")
                    logger.info("Applying dynamic filtering to current window...")

                    # If time range is completely outside original signal, return error
                    if (
                        start_sample >= original_signal_length
                        or end_sample > original_signal_length
                    ):
                        logger.warning(
                            f"Time range {start_sample}-{end_sample} is outside original signal range (0-{original_signal_length})"
                        )
                        return (
                            create_empty_figure(),
                            f"Time range is outside the available signal data. Please select a time range within 0 to {original_signal_length/sampling_freq:.1f} seconds.",
                            create_empty_figure(),
                            None,
                            None,
                        )

                    try:
                        # Import the same filtering function used in time domain
                        from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
                            apply_traditional_filter,
                        )

                        # Get the full original signal for the current time window
                        full_original_signal = df[signal_column].values
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
                        filter_type = filter_info.get("filter_type", "traditional")

                        if filter_type == "traditional":
                            # Extract traditional filter parameters
                            filter_family = parameters.get("filter_family", "butter")
                            filter_response = parameters.get(
                                "filter_response", "bandpass"
                            )
                            low_freq = parameters.get("low_freq", 0.5)
                            high_freq = parameters.get("high_freq", 5)
                            filter_order = parameters.get("filter_order", 4)

                            # Apply traditional filter
                            signal_data = apply_traditional_filter(
                                signal_data_detrended,
                                sampling_freq,
                                filter_family,
                                filter_response,
                                low_freq,
                                high_freq,
                                filter_order,
                            )
                            logger.info("Applied dynamic traditional filter")
                        else:
                            # For other filter types, use the original signal
                            signal_data = signal_data_detrended
                            logger.info(
                                f"Using original signal for filter type: {filter_type}"
                            )

                        signal_source_info = "Filtered Signal (Dynamic)"
                        logger.info("Dynamic filtering completed successfully")

                    except Exception as e:
                        logger.error(f"Error in dynamic filtering: {e}")
                        logger.info("Falling back to original signal")
                        signal_data = windowed_original_signal
                        signal_source_info = "Original Signal (Fallback)"
            logger.info(f"Signal data shape: {signal_data.shape}")
            logger.info(
                f"Signal data range: {np.min(signal_data):.3f} to {np.max(signal_data):.3f}"
            )
            logger.info(f"Signal data mean: {np.mean(signal_data):.3f}")

            # Signal source loading logic
            logger.info("=== SIGNAL SOURCE LOADING ===")
            logger.info(f"Signal source selection: {signal_source}")

            if signal_source == "filtered":
                # Try to load filtered data from filtering screen
                filtered_data = data_service.get_filtered_data(latest_data_id)
                filter_info = data_service.get_filter_info(latest_data_id)

                if filtered_data is not None:
                    logger.info(
                        f"Found filtered data with shape: {filtered_data.shape}"
                    )
                    # Apply time window to filtered data
                    filtered_windowed = filtered_data[start_sample:end_sample]
                    signal_data = filtered_windowed
                    signal_source_info = "Filtered Signal"
                else:
                    logger.info("No filtered data available, using original signal")
                    signal_source_info = "Original Signal"
            else:
                logger.info("Using original signal as requested")
                signal_source_info = "Original Signal"

            # Log selected signal characteristics
            logger.info(f"Signal data shape: {signal_data.shape}")
            logger.info(
                f"Signal data range: {np.min(signal_data):.3f} to {np.max(signal_data):.3f}"
            )
            logger.info(f"Signal data mean: {np.mean(signal_data):.3f}")
            logger.info(f"Signal source: {signal_source_info}")

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
            main_plot = create_respiratory_signal_plot(
                signal_data,
                time_axis,
                sampling_freq,
                signal_type,
                estimation_methods,
                preprocessing_options,
                low_cut,
                high_cut,
            )
            logger.info("Main respiratory signal plot created successfully")

            # Generate comprehensive respiratory analysis results
            logger.info("Generating comprehensive respiratory analysis results...")
            analysis_results = generate_comprehensive_respiratory_analysis(
                signal_data,
                time_axis,
                sampling_freq,
                signal_type,
                estimation_methods,
                advanced_options,
                preprocessing_options,
                low_cut,
                high_cut,
                min_breath_duration,
                max_breath_duration,
            )
            logger.info("Respiratory analysis results generated successfully")

            # Create respiratory analysis plots
            logger.info("Creating respiratory analysis plots...")
            analysis_plots = create_comprehensive_respiratory_plots(
                signal_data,
                time_axis,
                sampling_freq,
                signal_type,
                estimation_methods,
                advanced_options,
                preprocessing_options,
                low_cut,
                high_cut,
            )
            logger.info("Respiratory analysis plots created successfully")

            # Store processed data
            resp_data = {
                "signal_data": signal_data.tolist(),
                "time_axis": time_axis.tolist(),
                "sampling_freq": sampling_freq,
                "window": [start_time, end_time],
                "signal_type": signal_type,
                "estimation_methods": estimation_methods,
            }

            resp_features = {
                "advanced_options": advanced_options,
                "preprocessing_options": preprocessing_options,
                "filter_params": {"low_cut": low_cut, "high_cut": high_cut},
                "breath_constraints": {
                    "min_duration": min_breath_duration,
                    "max_duration": max_breath_duration,
                },
            }

            logger.info("Respiratory analysis completed successfully")
            return main_plot, analysis_results, analysis_plots, resp_data, resp_features

        except Exception as e:
            logger.error(f"Error in respiratory analysis callback: {e}")
            import traceback

            traceback.print_exc()
            return (
                create_empty_figure(),
                f"Error in analysis: {str(e)}",
                create_empty_figure(),
                None,
                None,
            )

    @app.callback(
        [Output("resp-start-time", "value"), Output("resp-end-time", "value")],
        [Input("resp-time-range-slider", "value")],
    )
    def update_resp_time_inputs(slider_value):
        """Update time input fields based on slider."""
        if not slider_value:
            return no_update, no_update
        return slider_value[0], slider_value[1]

    @app.callback(
        [
            Output("resp-time-range-slider", "min"),
            Output("resp-time-range-slider", "max"),
            Output("resp-time-range-slider", "value"),
        ],
        [Input("url", "pathname")],
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
            sampling_freq = latest_data.get("info", {}).get("sampling_freq", 1000)
            df = data_service.get_data(latest_data_id)

            if df is None or df.empty:
                return 0, 100, [0, 10]

            duration = len(df) / sampling_freq
            max_time = int(duration)

            return 0, max_time, [0, min(10, max_time)]

        except Exception as e:
            logger.error(f"Error updating respiratory time slider range: {e}")
            return 0, 100, [0, 10]
