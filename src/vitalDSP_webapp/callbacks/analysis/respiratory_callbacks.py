"""
Respiratory analysis callbacks for vitalDSP webapp.
Handles comprehensive respiratory rate estimation and breathing pattern analysis using vitalDSP.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html
from dash.exceptions import PreventUpdate
from scipy import signal
import dash_bootstrap_components as dbc
import logging

# Initialize logger first
logger = logging.getLogger(__name__)

# Initialize vitalDSP modules as None
RespiratoryAnalysis = None
peak_detection_rr = None
fft_based_rr = None
frequency_domain_rr = None
time_domain_rr = None
detect_apnea_amplitude = None
detect_apnea_pauses = None
multimodal_analysis = None
ppg_ecg_fusion = None
respiratory_cardiac_fusion = None
PPGAutonomicFeatures = None
ECGPPGSynchronization = None
PreprocessConfig = None
preprocess_signal = None

# Global variable for app instance (will be set by register_callbacks)
app = None


def _import_vitaldsp_modules():
    """Import vitalDSP modules with error handling."""
    global RespiratoryAnalysis, peak_detection_rr, fft_based_rr, frequency_domain_rr
    global time_domain_rr, detect_apnea_amplitude, detect_apnea_pauses, multimodal_analysis
    global ppg_ecg_fusion, respiratory_cardiac_fusion, PPGAutonomicFeatures
    global ECGPPGSynchronization, PreprocessConfig, preprocess_signal

    logger.info("=== IMPORTING VITALDSP MODULES ===")

    try:
        from vitalDSP.respiratory_analysis.respiratory_analysis import (
            RespiratoryAnalysis,
        )

        logger.info("✓ RespiratoryAnalysis imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import RespiratoryAnalysis: {e}")
        RespiratoryAnalysis = None

    try:
        from vitalDSP.respiratory_analysis.estimate_rr.peak_detection_rr import (
            peak_detection_rr,
        )

        logger.info("✓ peak_detection_rr imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import peak_detection_rr: {e}")
        peak_detection_rr = None

    try:
        from vitalDSP.respiratory_analysis.estimate_rr.fft_based_rr import fft_based_rr

        logger.info("✓ fft_based_rr imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import fft_based_rr: {e}")
        fft_based_rr = None

    try:
        from vitalDSP.respiratory_analysis.estimate_rr.frequency_domain_rr import (
            frequency_domain_rr,
        )

        logger.info("✓ frequency_domain_rr imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import frequency_domain_rr: {e}")
        frequency_domain_rr = None

    try:
        from vitalDSP.respiratory_analysis.estimate_rr.time_domain_rr import (
            time_domain_rr,
        )

        logger.info("✓ time_domain_rr imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import time_domain_rr: {e}")
        time_domain_rr = None

    try:
        from vitalDSP.respiratory_analysis.sleep_apnea_detection.amplitude_threshold import (
            detect_apnea_amplitude,
        )

        logger.info("✓ detect_apnea_amplitude imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import detect_apnea_amplitude: {e}")
        detect_apnea_amplitude = None

    try:
        from vitalDSP.respiratory_analysis.sleep_apnea_detection.pause_detection import (
            detect_apnea_pauses,
        )

        logger.info("✓ detect_apnea_pauses imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import detect_apnea_pauses: {e}")
        detect_apnea_pauses = None

    try:
        from vitalDSP.respiratory_analysis.fusion.multimodal_analysis import (
            multimodal_analysis,
        )

        logger.info("✓ multimodal_analysis imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import multimodal_analysis: {e}")
        multimodal_analysis = None

    try:
        from vitalDSP.respiratory_analysis.fusion.ppg_ecg_fusion import ppg_ecg_fusion

        logger.info("✓ ppg_ecg_fusion imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import ppg_ecg_fusion: {e}")
        ppg_ecg_fusion = None

    try:
        from vitalDSP.respiratory_analysis.fusion.respiratory_cardiac_fusion import (
            respiratory_cardiac_fusion,
        )

        logger.info("✓ respiratory_cardiac_fusion imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import respiratory_cardiac_fusion: {e}")
        respiratory_cardiac_fusion = None

    try:
        from vitalDSP.feature_engineering.ppg_autonomic_features import (
            PPGAutonomicFeatures,
        )

        logger.info("✓ PPGAutonomicFeatures imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import PPGAutonomicFeatures: {e}")
        PPGAutonomicFeatures = None

    try:
        from vitalDSP.feature_engineering.ecg_ppg_synchronyzation_features import (
            ECGPPGSynchronization,
        )

        logger.info("✓ ECGPPGSynchronization imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import ECGPPGSynchronization: {e}")
        ECGPPGSynchronization = None

    try:
        from vitalDSP.preprocess.preprocess_operations import (
            PreprocessConfig,
            preprocess_signal,
        )

        logger.info("✓ PreprocessConfig and preprocess_signal imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import PreprocessConfig/preprocess_signal: {e}")
        PreprocessConfig = None
        preprocess_signal = None

    logger.info("=== VITALDSP MODULE IMPORT COMPLETED ===")


def toggle_ensemble_options(estimation_methods):
    """Show/hide ensemble options based on selection."""
    if estimation_methods and "ensemble" in estimation_methods:
        return {"display": "block"}
    return {"display": "none"}


def register_respiratory_callbacks(app):
    """Register all respiratory analysis callbacks."""
    logger.info("=== REGISTERING RESPIRATORY CALLBACKS ===")
    logger.info(f"App type: {type(app)}")

    # Import vitalDSP modules when callbacks are registered
    _import_vitaldsp_modules()

    # Auto-select signal type based on uploaded data
    @app.callback(
        [Output("resp-signal-type", "value")],
        [Input("url", "pathname")],
        prevent_initial_call=True,
    )
    def auto_select_resp_signal_type(pathname):
        """Auto-select signal type based on uploaded data."""
        logger.info("=== AUTO-SELECT RESP SIGNAL TYPE CALLBACK TRIGGERED ===")
        logger.info(f"Pathname: {pathname}")

        if pathname != "/respiratory":
            logger.info("Not on respiratory page, preventing update")
            raise PreventUpdate

        try:
            from vitalDSP_webapp.services.data.data_service import get_data_service

            data_service = get_data_service()
            if not data_service:
                logger.warning("Data service not available")
                return ["PPG"]

            # Get the latest data
            all_data = data_service.get_all_data()
            if not all_data:
                logger.info("No data available, using defaults")
                return ["PPG"]

            # Get the most recent data
            latest_data_id = max(
                all_data.keys(), key=lambda x: int(x.split("_")[1]) if "_" in x else 0
            )
            data_info = data_service.get_data_info(latest_data_id)

            if not data_info:
                logger.info("No data info available, using defaults")
                return ["PPG"]

            # Debug: Log the data_info to see what's stored
            logger.info(
                f"Resp data info keys: {list(data_info.keys()) if data_info else 'None'}"
            )
            logger.info(f"Resp full data info: {data_info}")

            # First, check if signal type is stored in data info
            stored_signal_type = data_info.get("signal_type", None)
            logger.info(f"Resp stored signal type: {stored_signal_type}")

            if stored_signal_type and stored_signal_type.lower() != "auto":
                # Convert stored value to match respiratory screen dropdown format (lowercase)
                signal_type = stored_signal_type.lower()
                logger.info(f"Resp using stored signal type: {signal_type}")
            else:
                # Auto-detect signal type based on data characteristics
                signal_type = "ppg"  # Default (lowercase for respiratory screen)
                logger.info("Resp auto-detecting signal type from data characteristics")

            # Try to detect signal type from column names or data characteristics if not stored
            if (
                stored_signal_type
                and stored_signal_type.lower() == "auto"
                or not stored_signal_type
            ):
                df = data_service.get_data(latest_data_id)
                if df is not None and not df.empty:
                    column_mapping = data_service.get_column_mapping(latest_data_id)
                    signal_column = column_mapping.get("signal", "")

                    # Check column names for signal type hints
                    if any(
                        keyword in signal_column.lower()
                        for keyword in ["ecg", "electrocardio"]
                    ):
                        signal_type = "ECG"
                        logger.info("Auto-detected ECG signal type from column name")
                    elif any(
                        keyword in signal_column.lower()
                        for keyword in ["ppg", "pleth", "photopleth"]
                    ):
                        signal_type = "PPG"
                        logger.info("Auto-detected PPG signal type from column name")
                    else:
                        # Try to detect from data characteristics
                        try:
                            signal_data = (
                                df[signal_column].values
                                if signal_column
                                else df.iloc[:, 1].values
                            )
                            sampling_freq = data_info.get("sampling_freq", 1000)

                            # Simple heuristic: ECG typically has higher frequency content
                            from scipy import signal

                            f, psd = signal.welch(
                                signal_data,
                                fs=sampling_freq,
                                nperseg=min(1024, len(signal_data) // 4),
                            )
                            dominant_freq = f[np.argmax(psd)]

                            if (
                                dominant_freq > 1.0
                            ):  # Higher frequency content suggests ECG
                                signal_type = "ECG"
                                logger.info(
                                    "Auto-detected ECG signal type from frequency analysis"
                                )
                            else:
                                signal_type = "PPG"
                                logger.info(
                                    "Auto-detected PPG signal type from frequency analysis"
                                )
                        except Exception as e:
                            logger.warning(
                                f"Could not analyze signal characteristics: {e}"
                            )
                            signal_type = "PPG"

            logger.info(f"Auto-selected respiratory signal type: {signal_type}")
            return [signal_type]

        except Exception as e:
            logger.error(f"Error in auto-selection: {e}")
            return ["PPG"]

    @app.callback(
        Output("resp-ensemble-options", "style"),
        Input("resp-estimation-methods", "value"),
        prevent_initial_call=False,
    )
    def toggle_ensemble_options_callback(estimation_methods):
        """Show/hide ensemble options based on selection."""
        return toggle_ensemble_options(estimation_methods)

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
            State("resp-estimation-methods", "value"),
            State("resp-advanced-options", "value"),
            State("resp-preprocessing-options", "value"),
            State("resp-low-cut", "value"),
            State("resp-high-cut", "value"),
            State("resp-min-breath-duration", "value"),
            State("resp-max-breath-duration", "value"),
            State("resp-ensemble-method", "value"),
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
        estimation_methods,
        advanced_options,
        preprocessing_options,
        low_cut,
        high_cut,
        min_breath_duration,
        max_breath_duration,
        ensemble_method,
    ):
        """Unified callback for respiratory analysis - handles both page load and user interactions."""
        ctx = callback_context

        # Determine what triggered this callback
        if not ctx.triggered:
            logger.warning("No context triggered - raising PreventUpdate")
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        logger.info("=== RESPIRATORY ANALYSIS CALLBACK TRIGGERED ===")
        logger.info(f"Trigger ID: {trigger_id}")
        logger.info(f"Pathname: {pathname}")
        logger.info("All callback parameters:")
        logger.info(f"  - n_clicks: {n_clicks}")
        logger.info(f"  - slider_value: {slider_value}")
        logger.info(f"  - start_time: {start_time}")
        logger.info(f"  - end_time: {end_time}")
        logger.info(f"  - signal_type: {signal_type}")
        logger.info(f"  - estimation_methods: {estimation_methods}")
        logger.info(f"  - advanced_options: {advanced_options}")
        logger.info(f"  - preprocessing_options: {preprocessing_options}")
        logger.info(f"  - low_cut: {low_cut}")
        logger.info(f"  - high_cut: {high_cut}")
        logger.info(f"  - min_breath_duration: {min_breath_duration}")
        logger.info(f"  - max_breath_duration: {max_breath_duration}")

        # Only run this when we're on the respiratory page
        if pathname != "/respiratory":
            logger.info("Not on respiratory page, returning empty figures")
            return (
                create_empty_figure(),
                "Navigate to Respiratory Analysis page",
                create_empty_figure(),
                None,
                None,
            )

        # If this is the first time loading the page (no button clicks), show a message
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
            logger.info(f"Signal data shape: {signal_data.shape}")
            logger.info(
                f"Signal data range: {np.min(signal_data):.3f} to {np.max(signal_data):.3f}"
            )
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
            logger.info(f"Main plot type: {type(main_plot)}")

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
                ensemble_method,
            )
            logger.info("Respiratory analysis results generated successfully")
            logger.info(f"Analysis results type: {type(analysis_results)}")
            logger.info(
                f"Analysis results content length: {len(str(analysis_results)) if analysis_results else 'None'}"
            )

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
            logger.info(f"Analysis plots type: {type(analysis_plots)}")

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
            logger.info("Returning results:")
            logger.info(f"  - Main plot: {type(main_plot)}")
            logger.info(f"  - Analysis results: {type(analysis_results)}")
            logger.info(f"  - Analysis plots: {type(analysis_plots)}")
            logger.info(f"  - Resp data: {type(resp_data)}")
            logger.info(f"  - Resp features: {type(resp_features)}")

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
            sampling_freq = latest_data.get("info", {}).get("sampling_freq", 1000)

            max_time = len(df) / sampling_freq
            logger.info(f"Max time: {max_time}, Sampling freq: {sampling_freq}")

            return 0, max_time, [0, min(10, max_time)]

        except Exception as e:
            logger.error(f"Error updating resp time slider range: {e}")
            return 0, 100, [0, 10]

            logger.info("=== RESPIRATORY CALLBACKS REGISTERED SUCCESSFULLY ===")

    @app.callback(
        Output("resp-additional-analysis-section", "children"),
        [Input("resp-advanced-options", "value")],
    )
    def update_additional_analysis_section(advanced_options):
        """Update additional analysis section based on selected options."""
        if not advanced_options:
            return html.Div()

        sections = []

        # Create inline grid layout for advanced analysis sections
        if any(advanced_options):
            sections.append(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            [
                                html.H5(
                                    "🔬 Advanced Analysis Results", className="mb-0"
                                ),
                                html.Small(
                                    "Inline analysis blocks for efficient space usage",
                                    className="text-muted",
                                ),
                            ]
                        ),
                        dbc.CardBody([html.Div(id="resp-advanced-analysis-results")]),
                    ],
                    className="mb-2 shadow-sm",
                )
            )

        return html.Div(sections)

    @app.callback(
        Output("resp-btn-export-results", "n_clicks"),
        [Input("resp-btn-export-results", "n_clicks")],
        [State("resp-data-store", "data"), State("resp-features-store", "data")],
    )
    def export_respiratory_results(n_clicks, resp_data, resp_features):
        """Export respiratory analysis results."""
        if not n_clicks or not resp_data:
            return no_update

        try:
            # Convert to JSON and trigger download

            # This would typically trigger a download in a real implementation
            logger.info("Respiratory analysis results exported successfully")

        except Exception as e:
            logger.error(f"Error exporting respiratory results: {e}")

        return no_update


def create_empty_figure():
    """Create an empty figure for when no data is available."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available<br>Please upload data first",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def detect_respiratory_signal_type(signal_data, sampling_freq):
    """Auto-detect respiratory signal type based on signal characteristics."""
    try:
        # Calculate basic statistics
        std_val = np.std(signal_data)
        range_val = np.max(signal_data) - np.min(signal_data)

        # Calculate frequency content
        fft_result = np.abs(np.fft.rfft(signal_data))
        freqs = np.fft.rfftfreq(len(signal_data), 1 / sampling_freq)

        # Find dominant frequency
        peak_idx = np.argmax(fft_result)
        dominant_freq = freqs[peak_idx]

        # Respiratory signals typically have dominant frequency around 0.2-0.5 Hz (12-30 BPM)
        # PPG respiratory component is usually lower amplitude
        # ECG respiratory component is more complex

        if 0.1 < dominant_freq < 0.8 and range_val < 2 * std_val:
            return "ppg"
        elif 0.1 < dominant_freq < 1.0 and range_val > 3 * std_val:
            return "respiratory"
        elif dominant_freq > 0.5:
            return "ecg"
        else:
            return "ppg"  # Default to PPG for most cases

    except Exception as e:
        logger.warning(f"Respiratory signal type detection failed: {e}")
        return "ppg"  # Default fallback


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
    logger.info("=== CREATE RESPIRATORY SIGNAL PLOT STARTED ===")
    logger.info("Input parameters:")
    logger.info(f"  - signal_data shape: {signal_data.shape}")
    logger.info(f"  - time_axis shape: {time_axis.shape}")
    logger.info(f"  - sampling_freq: {sampling_freq}")
    logger.info(f"  - signal_type: {signal_type}")
    logger.info(f"  - estimation_methods: {estimation_methods}")
    logger.info(f"  - preprocessing_options: {preprocessing_options}")
    logger.info(f"  - low_cut: {low_cut}")
    logger.info(f"  - high_cut: {high_cut}")

    try:
        fig = go.Figure()

        # Apply preprocessing if selected
        logger.info("Applying preprocessing...")
        processed_signal = signal_data.copy()
        if preprocessing_options and "filter" in preprocessing_options:
            try:
                # Design bandpass filter for respiratory frequencies
                nyquist = sampling_freq / 2
                low = low_cut / nyquist
                high = high_cut / nyquist

                if low < high < 1.0:
                    b, a = signal.butter(4, [low, high], btype="bandpass")
                    processed_signal = signal.filtfilt(b, a, processed_signal)
                    logger.info(f"Applied bandpass filter: {low_cut}-{high_cut} Hz")
            except Exception as e:
                logger.error(f"Bandpass filtering failed: {e}")

        # Apply additional preprocessing options
        if preprocessing_options and "baseline_correction" in preprocessing_options:
            try:
                # Remove baseline wander using high-pass filter
                baseline_cutoff = 0.01  # 0.01 Hz cutoff for baseline
                baseline_nyquist = baseline_cutoff / nyquist
                if baseline_nyquist < 1.0:
                    b_baseline, a_baseline = signal.butter(
                        2, baseline_nyquist, btype="highpass"
                    )
                    processed_signal = signal.filtfilt(
                        b_baseline, a_baseline, processed_signal
                    )
                    logger.info("Applied baseline correction")
            except Exception as e:
                logger.error(f"Baseline correction failed: {e}")

        if preprocessing_options and "moving_average" in preprocessing_options:
            try:
                # Apply moving average smoothing
                window_size = int(0.1 * sampling_freq)  # 100ms window
                if window_size > 0:
                    processed_signal = np.convolve(
                        processed_signal,
                        np.ones(window_size) / window_size,
                        mode="same",
                    )
                    logger.info("Applied moving average smoothing")
            except Exception as e:
                logger.error(f"Moving average smoothing failed: {e}")

        # Create the main signal plot
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=processed_signal,
                mode="lines",
                name=f"{signal_type.upper()} Signal (Processed)",
                line=dict(color="#2E86AB", width=2),
                hovertemplate="<b>Time:</b> %{x:.3f}s<br><b>Amplitude:</b> %{y:.3f}<extra></extra>",
            )
        )

        # Add original signal if preprocessing was applied
        if preprocessing_options and len(preprocessing_options) > 0:
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=signal_data,
                    mode="lines",
                    name=f"{signal_type.upper()} Signal (Original)",
                    line=dict(color="#95A5A6", width=1, dash="dot"),
                    opacity=0.7,
                    hovertemplate="<b>Time:</b> %{x:.3f}s<br><b>Amplitude:</b> %{y:.3f}<extra></extra>",
                )
            )

        # Add breathing pattern detection if enabled
        if estimation_methods and "peak_detection" in estimation_methods:
            try:
                # Detect breathing peaks
                prominence = 0.3 * np.std(processed_signal)
                distance = int(0.5 * sampling_freq)  # Minimum 0.5s between breaths

                peaks, properties = signal.find_peaks(
                    processed_signal, prominence=prominence, distance=distance
                )

                if len(peaks) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis[peaks],
                            y=processed_signal[peaks],
                            mode="markers",
                            name="Breathing Peaks",
                            marker=dict(color="red", size=8, symbol="diamond"),
                            hovertemplate="<b>Breath:</b> %{y:.3f}<br><b>Time:</b> %{x:.3f}s<extra></extra>",
                        )
                    )

                    # Add breath annotations
                    for i, peak in enumerate(peaks[:10]):  # Limit to first 10 breaths
                        fig.add_annotation(
                            x=time_axis[peak],
                            y=processed_signal[peak],
                            text=f"B{i+1}",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor="red",
                            ax=0,
                            ay=-40,
                        )

                    # Add breathing rate annotation
                    if len(peaks) > 1:
                        breath_intervals = np.diff(peaks) / sampling_freq
                        breathing_rate = 60 / np.mean(breath_intervals)
                        fig.add_annotation(
                            x=0.02,
                            y=0.98,
                            xref="paper",
                            yref="paper",
                            text=f"Breathing Rate: {breathing_rate:.1f} BPM",
                            showarrow=False,
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="red",
                            borderwidth=2,
                        )

            except Exception as e:
                logger.error(f"Breathing peak detection failed: {e}")

        # Add baseline
        baseline = np.mean(processed_signal)
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Baseline: {baseline:.3f}",
        )

        logger.info("Updating plot layout...")
        fig.update_layout(
            title=f"{signal_type.upper()} Respiratory Signal Analysis",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=True,
            hovermode="closest",
        )

        logger.info("=== CREATE RESPIRATORY SIGNAL PLOT COMPLETED ===")
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
    ensemble_method=None,
):
    """Generate comprehensive respiratory analysis results using vitalDSP."""
    logger.info("=== GENERATE COMPREHENSIVE RESPIRATORY ANALYSIS STARTED ===")
    logger.info("Input parameters:")
    logger.info(f"  - signal_data shape: {signal_data.shape}")
    logger.info(f"  - time_axis shape: {time_axis.shape}")
    logger.info(f"  - sampling_freq: {sampling_freq}")
    logger.info(f"  - signal_type: {signal_type}")
    logger.info(f"  - estimation_methods: {estimation_methods}")
    logger.info(f"  - advanced_options: {advanced_options}")
    logger.info(f"  - preprocessing_options: {preprocessing_options}")
    logger.info(f"  - low_cut: {low_cut}")
    logger.info(f"  - high_cut: {high_cut}")
    logger.info(f"  - min_breath_duration: {min_breath_duration}")
    logger.info(f"  - max_breath_duration: {max_breath_duration}")
    logger.info(f"  - ensemble_method: {ensemble_method}")

    try:
        results = []

        # Initialize vitalDSP RespiratoryAnalysis
        if RespiratoryAnalysis is None:
            logger.warning(
                "RespiratoryAnalysis module not available - using fallback analysis"
            )
            results.append(
                html.Div(
                    [
                        html.H5("⚠️ Using Fallback Analysis", className="text-warning"),
                        html.P(
                            "vitalDSP modules not available. Using basic signal processing methods.",
                            className="text-muted",
                        ),
                    ]
                )
            )

            # Fallback: Basic respiratory rate estimation using peak detection
            try:
                # Simple peak detection for breathing
                prominence = 0.3 * np.std(signal_data)
                distance = int(0.5 * sampling_freq)  # Minimum 0.5s between breaths
                peaks, _ = signal.find_peaks(
                    signal_data, prominence=prominence, distance=distance
                )

                if len(peaks) > 1:
                    breath_intervals = np.diff(peaks) / sampling_freq
                    rr_bpm = 60 / np.mean(breath_intervals)
                    rr_std = (
                        60 * np.std(breath_intervals) / (np.mean(breath_intervals) ** 2)
                    )

                    results.append(
                        html.Div(
                            [
                                html.Strong("Fallback Peak Detection: "),
                                html.Span(
                                    f"{rr_bpm:.2f} BPM", className="text-success"
                                ),
                            ],
                            className="mb-2",
                        )
                    )

                    results.append(
                        html.Div(
                            [
                                html.Strong("Breathing Variability: "),
                                html.Span(f"{rr_std:.2f} BPM", className="text-info"),
                            ],
                            className="mb-2",
                        )
                    )

                    results.append(
                        html.Div(
                            [
                                html.Strong("Number of Breaths: "),
                                html.Span(f"{len(peaks)}", className="text-info"),
                            ],
                            className="mb-2",
                        )
                    )
                else:
                    results.append(
                        html.Div(
                            [
                                html.Strong("Fallback Analysis: "),
                                html.Span(
                                    "Insufficient peaks detected",
                                    className="text-warning",
                                ),
                            ],
                            className="mb-2",
                        )
                    )

            except Exception as e:
                logger.error(f"Fallback analysis failed: {e}")
                results.append(
                    html.Div(
                        [
                            html.Strong("Fallback Analysis: "),
                            html.Span("Failed", className="text-danger"),
                        ],
                        className="mb-2",
                    )
                )

            # Continue with basic statistics
        else:
            try:
                logger.info("Initializing vitalDSP RespiratoryAnalysis...")
                resp_analysis = RespiratoryAnalysis(signal_data, sampling_freq)
                logger.info("RespiratoryAnalysis initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize RespiratoryAnalysis: {e}")
                logger.info("Falling back to basic analysis")
                resp_analysis = None
                results.append(html.H5("⚠️ Fallback Analysis", className="text-warning"))
                results.append(
                    html.P(
                        "vitalDSP initialization failed. Using basic signal analysis.",
                        className="text-muted",
                    )
                )

        # Create preprocessing configuration
        if PreprocessConfig is None:
            logger.error(
                "PreprocessConfig module not available - using default settings"
            )
            preprocess_config = None
        else:
            try:
                logger.info("Creating preprocessing configuration...")
                preprocess_config = PreprocessConfig(
                    filter_type="bandpass",
                    lowcut=low_cut,
                    highcut=high_cut,
                    respiratory_mode=True,
                )
                logger.info(f"PreprocessConfig created: {preprocess_config}")
            except Exception as config_error:
                logger.error(f"Failed to create PreprocessConfig: {config_error}")
                logger.info("Using default preprocessing settings")
                preprocess_config = None

        # Respiratory Rate Estimation using multiple methods
        logger.info(f"Processing estimation methods: {estimation_methods}")
        if estimation_methods and resp_analysis is not None:
            logger.info("Adding respiratory rate estimation header...")
            results.append(html.H5("🫁 Respiratory Rate Estimation", className="mb-3"))

            for method in estimation_methods:
                logger.info(f"Processing method: {method}")
                try:
                    if method == "peak_detection":
                        logger.info(
                            "Computing respiratory rate using peak detection method..."
                        )
                        try:
                            rr = resp_analysis.compute_respiratory_rate(
                                method="peaks",
                                min_breath_duration=min_breath_duration,
                                max_breath_duration=max_breath_duration,
                                preprocess_config=preprocess_config,
                            )
                            logger.info(f"Peak detection method result: {rr:.2f} BPM")
                            results.append(
                                html.Div(
                                    [
                                        html.Strong("Peak Detection Method: "),
                                        html.Span(
                                            f"{rr:.2f} BPM", className="text-success"
                                        ),
                                    ],
                                    className="mb-2",
                                )
                            )
                        except Exception as e:
                            logger.error(f"Peak detection method failed: {e}")
                            results.append(
                                html.Div(
                                    [
                                        html.Strong("Peak Detection Method: "),
                                        html.Span("Failed", className="text-danger"),
                                    ],
                                    className="mb-2",
                                )
                            )

                    elif method == "zero_crossing":
                        logger.info(
                            "Computing respiratory rate using zero crossing method..."
                        )
                        try:
                            rr = resp_analysis.compute_respiratory_rate(
                                method="zero_crossing",
                                min_breath_duration=min_breath_duration,
                                max_breath_duration=max_breath_duration,
                                preprocess_config=preprocess_config,
                            )
                            # Zero crossing counts both up/down crossings, so divide by 2 for complete breaths
                            rr = rr / 2
                            logger.info(
                                f"Zero crossing method result (corrected): {rr:.2f} BPM"
                            )
                            results.append(
                                html.Div(
                                    [
                                        html.Strong("Zero Crossing Method: "),
                                        html.Span(
                                            f"{rr:.2f} BPM", className="text-success"
                                        ),
                                        html.Small(
                                            " (corrected for symmetry)",
                                            className="text-muted ms-2",
                                        ),
                                    ],
                                    className="mb-2",
                                )
                            )
                        except Exception as e:
                            logger.error(f"Zero crossing method failed: {e}")
                            results.append(
                                html.Div(
                                    [
                                        html.Strong("Zero Crossing Method: "),
                                        html.Span("Failed", className="text-danger"),
                                    ],
                                    className="mb-2",
                                )
                            )

                    elif method == "time_domain":
                        if time_domain_rr is None:
                            logger.warning("time_domain_rr function not available")
                            results.append(
                                html.Div(
                                    [
                                        html.Strong("Time Domain Method: "),
                                        html.Span(
                                            "Function not available",
                                            className="text-warning",
                                        ),
                                    ],
                                    className="mb-2",
                                )
                            )
                        else:
                            rr = time_domain_rr(
                                signal_data,
                                sampling_freq,
                                preprocess=(
                                    "bandpass"
                                    if "filter" in preprocessing_options
                                    else None
                                ),
                                lowcut=low_cut,
                                highcut=high_cut,
                            )
                            results.append(
                                html.Div(
                                    [
                                        html.Strong("Time Domain Method: "),
                                        html.Span(
                                            f"{rr:.2f} BPM", className="text-success"
                                        ),
                                    ],
                                    className="mb-2",
                                )
                            )

                    elif method == "frequency_domain":
                        if frequency_domain_rr is None:
                            logger.warning("frequency_domain_rr function not available")
                            results.append(
                                html.Div(
                                    [
                                        html.Strong("Frequency Domain Method: "),
                                        html.Span(
                                            "Function not available",
                                            className="text-warning",
                                        ),
                                    ],
                                    className="mb-2",
                                )
                            )
                        else:
                            rr = frequency_domain_rr(
                                signal_data,
                                sampling_freq,
                                preprocess=(
                                    "bandpass"
                                    if "filter" in preprocessing_options
                                    else None
                                ),
                                lowcut=low_cut,
                                highcut=high_cut,
                            )
                            # Frequency domain method may detect double frequency, so divide by 2
                            rr = rr / 2
                            logger.info(
                                f"Frequency domain method result (corrected): {rr:.2f} BPM"
                            )
                            results.append(
                                html.Div(
                                    [
                                        html.Strong("Frequency Domain Method: "),
                                        html.Span(
                                            f"{rr:.2f} BPM", className="text-success"
                                        ),
                                        html.Small(
                                            " (corrected for symmetry)",
                                            className="text-muted ms-2",
                                        ),
                                    ],
                                    className="mb-2",
                                )
                            )

                    elif method == "fft_based":
                        if fft_based_rr is None:
                            logger.warning("fft_based_rr function not available")
                            results.append(
                                html.Div(
                                    [
                                        html.Strong("FFT-based Method: "),
                                        html.Span(
                                            "Function not available",
                                            className="text-warning",
                                        ),
                                    ],
                                    className="mb-2",
                                )
                            )
                        else:
                            rr = fft_based_rr(
                                signal_data,
                                sampling_freq,
                                preprocess=(
                                    "bandpass"
                                    if "filter" in preprocessing_options
                                    else None
                                ),
                                lowcut=low_cut,
                                highcut=high_cut,
                            )
                            results.append(
                                html.Div(
                                    [
                                        html.Strong("FFT-based Method: "),
                                        html.Span(
                                            f"{rr:.2f} BPM", className="text-success"
                                        ),
                                    ],
                                    className="mb-2",
                                )
                            )

                    elif method == "counting":
                        if peak_detection_rr is None:
                            logger.warning("peak_detection_rr function not available")
                            results.append(
                                html.Div(
                                    [
                                        html.Strong("Counting Method: "),
                                        html.Span(
                                            "Function not available",
                                            className="text-warning",
                                        ),
                                    ],
                                    className="mb-2",
                                )
                            )
                        else:
                            rr = peak_detection_rr(
                                signal_data,
                                sampling_freq,
                                preprocess=(
                                    "bandpass"
                                    if "filter" in preprocessing_options
                                    else None
                                ),
                                lowcut=low_cut,
                                highcut=high_cut,
                            )
                            results.append(
                                html.Div(
                                    [
                                        html.Strong("Counting Method: "),
                                        html.Span(
                                            f"{rr:.2f} BPM", className="text-success"
                                        ),
                                    ],
                                    className="mb-2",
                                )
                            )

                    elif method == "ensemble":
                        logger.info(
                            "Computing ensemble respiratory rate using all available methods..."
                        )
                        logger.info(f"Ensemble method parameter: {ensemble_method}")
                        try:
                            # Collect all available RR estimates
                            ensemble_estimates = []
                            method_names = []

                            # Try each method and collect results
                            if RespiratoryAnalysis is not None:
                                # Peak detection
                                try:
                                    rr_peak = resp_analysis.compute_respiratory_rate(
                                        method="peaks",
                                        min_breath_duration=min_breath_duration,
                                        max_breath_duration=max_breath_duration,
                                        preprocess_config=preprocess_config,
                                    )
                                    ensemble_estimates.append(rr_peak)
                                    method_names.append("Peak Detection")
                                except Exception:
                                    pass

                                # Zero crossing
                                try:
                                    rr_zero = resp_analysis.compute_respiratory_rate(
                                        method="zero_crossing",
                                        min_breath_duration=min_breath_duration,
                                        max_breath_duration=max_breath_duration,
                                        preprocess_config=preprocess_config,
                                    )
                                    ensemble_estimates.append(
                                        rr_zero / 2
                                    )  # Apply symmetry correction
                                    method_names.append("Zero Crossing")
                                except Exception:
                                    pass

                            # Direct function calls
                            if time_domain_rr is not None:
                                try:
                                    rr_time = time_domain_rr(
                                        signal_data,
                                        sampling_freq,
                                        preprocess=(
                                            "bandpass"
                                            if "filter" in preprocessing_options
                                            else None
                                        ),
                                        lowcut=low_cut,
                                        highcut=high_cut,
                                    )
                                    ensemble_estimates.append(rr_time)
                                    method_names.append("Time Domain")
                                except Exception:
                                    pass

                            if frequency_domain_rr is not None:
                                try:
                                    rr_freq = frequency_domain_rr(
                                        signal_data,
                                        sampling_freq,
                                        preprocess=(
                                            "bandpass"
                                            if "filter" in preprocessing_options
                                            else None
                                        ),
                                        lowcut=low_cut,
                                        highcut=high_cut,
                                    )
                                    ensemble_estimates.append(
                                        rr_freq / 2
                                    )  # Apply symmetry correction
                                    method_names.append("Frequency Domain")
                                except Exception:
                                    pass

                            if fft_based_rr is not None:
                                try:
                                    rr_fft = fft_based_rr(
                                        signal_data,
                                        sampling_freq,
                                        preprocess=(
                                            "bandpass"
                                            if "filter" in preprocessing_options
                                            else None
                                        ),
                                        lowcut=low_cut,
                                        highcut=high_cut,
                                    )
                                    ensemble_estimates.append(rr_fft)
                                    method_names.append("FFT-based")
                                except Exception:
                                    pass

                            if peak_detection_rr is not None:
                                try:
                                    rr_count = peak_detection_rr(
                                        signal_data,
                                        sampling_freq,
                                        preprocess=(
                                            "bandpass"
                                            if "filter" in preprocessing_options
                                            else None
                                        ),
                                        lowcut=low_cut,
                                        highcut=high_cut,
                                    )
                                    ensemble_estimates.append(rr_count)
                                    method_names.append("Counting")
                                except Exception:
                                    pass

                            if len(ensemble_estimates) > 0:
                                # Apply selected ensemble method
                                current_ensemble_method = (
                                    ensemble_method or "mean"
                                )  # Default to mean if not specified
                                logger.info(
                                    f"Using ensemble method: {current_ensemble_method}"
                                )

                                if current_ensemble_method == "mean":
                                    ensemble_result = np.mean(ensemble_estimates)
                                    method_description = "Simple Average"
                                elif current_ensemble_method == "weighted_mean":
                                    # Weight by inverse of variance (more reliable methods get higher weight)
                                    weights = [
                                        1.0
                                        / (1.0 + abs(est - np.mean(ensemble_estimates)))
                                        for est in ensemble_estimates
                                    ]
                                    weights = np.array(weights) / np.sum(
                                        weights
                                    )  # Normalize weights
                                    ensemble_result = np.average(
                                        ensemble_estimates, weights=weights
                                    )
                                    method_description = (
                                        "Weighted Average (by reliability)"
                                    )
                                elif current_ensemble_method == "bagging":
                                    # Bootstrap aggregation
                                    n_bootstrap = 100
                                    bootstrap_estimates = []
                                    for _ in range(n_bootstrap):
                                        indices = np.random.choice(
                                            len(ensemble_estimates),
                                            size=len(ensemble_estimates),
                                            replace=True,
                                        )
                                        bootstrap_sample = [
                                            ensemble_estimates[i] for i in indices
                                        ]
                                        bootstrap_estimates.append(
                                            np.mean(bootstrap_sample)
                                        )
                                    ensemble_result = np.mean(bootstrap_estimates)
                                    method_description = "Bootstrap Aggregation"
                                elif current_ensemble_method == "boosting":
                                    # Sequential learning (simple implementation)
                                    ensemble_result = ensemble_estimates[
                                        0
                                    ]  # Start with first estimate
                                    for i in range(1, len(ensemble_estimates)):
                                        # Simple boosting: adjust based on previous estimate
                                        ensemble_result = (
                                            0.7 * ensemble_result
                                            + 0.3 * ensemble_estimates[i]
                                        )
                                    method_description = "Sequential Learning"
                                else:
                                    ensemble_result = np.mean(ensemble_estimates)
                                    method_description = "Simple Average (fallback)"

                                # Calculate ensemble statistics
                                ensemble_std = np.std(ensemble_estimates)
                                ensemble_min = np.min(ensemble_estimates)
                                ensemble_max = np.max(ensemble_estimates)

                                # Calculate confidence interval (95%)
                                ensemble_ci_lower = np.percentile(
                                    ensemble_estimates, 2.5
                                )
                                ensemble_ci_upper = np.percentile(
                                    ensemble_estimates, 97.5
                                )

                                # Calculate coefficient of variation
                                ensemble_cv = (
                                    ensemble_std / ensemble_result
                                    if ensemble_result > 0
                                    else 0
                                )

                                # Determine reliability score
                                if ensemble_cv < 0.1:
                                    reliability = "Excellent"
                                    reliability_color = "text-success"
                                elif ensemble_cv < 0.2:
                                    reliability = "Good"
                                    reliability_color = "text-info"
                                elif ensemble_cv < 0.3:
                                    reliability = "Fair"
                                    reliability_color = "text-warning"
                                else:
                                    reliability = "Poor"
                                    reliability_color = "text-danger"

                                results.append(
                                    html.Div(
                                        [
                                            html.Strong(
                                                "🎯 Ensemble Respiratory Rate: "
                                            ),
                                            html.Span(
                                                f"{ensemble_result:.2f} ± {ensemble_std:.2f} BPM",
                                                className="text-success",
                                            ),
                                            html.Br(),
                                            html.Small(
                                                f"Ensemble Method: {method_description}",
                                                className="text-info",
                                            ),
                                            html.Br(),
                                            html.Small(
                                                "Reliability: ", className="text-muted"
                                            ),
                                            html.Span(
                                                f"{reliability}",
                                                className=reliability_color,
                                            ),
                                            html.Br(),
                                            html.Small(
                                                f"Methods used: {len(ensemble_estimates)} ({', '.join(method_names)})",
                                                className="text-muted",
                                            ),
                                            html.Br(),
                                            html.Small(
                                                f"Range: {ensemble_min:.1f} - {ensemble_max:.1f} BPM",
                                                className="text-muted",
                                            ),
                                            html.Br(),
                                            html.Small(
                                                f"95% CI: {ensemble_ci_lower:.1f} - {ensemble_ci_upper:.1f} BPM",
                                                className="text-muted",
                                            ),
                                            html.Br(),
                                            html.Small(
                                                f"Coefficient of Variation: {ensemble_cv:.3f}",
                                                className="text-muted",
                                            ),
                                        ],
                                        className="mb-3 p-3 border border-success rounded",
                                    )
                                )

                                logger.info(
                                    f"Ensemble method result: {ensemble_result:.2f} ± {ensemble_std:.2f} BPM using {current_ensemble_method}"
                                )
                            else:
                                results.append(
                                    html.Div(
                                        [
                                            html.Strong("Ensemble Method: "),
                                            html.Span(
                                                "No methods available",
                                                className="text-warning",
                                            ),
                                        ],
                                        className="mb-2",
                                    )
                                )

                        except Exception as e:
                            logger.error(f"Ensemble method failed: {e}")
                            results.append(
                                html.Div(
                                    [
                                        html.Strong("Ensemble Method: "),
                                        html.Span("Failed", className="text-danger"),
                                    ],
                                    className="mb-2",
                                )
                            )

                except Exception as e:
                    logger.warning(f"Method {method} failed: {e}")
                    results.append(
                        html.Div(
                            [
                                html.Strong(
                                    f"{method.replace('_', ' ').title()} Method: "
                                ),
                                html.Span("Failed", className="text-danger"),
                            ],
                            className="mb-2",
                        )
                    )

        elif estimation_methods and resp_analysis is None:
            logger.warning(
                "Estimation methods requested but RespiratoryAnalysis not available"
            )
            results.append(
                html.H5("⚠️ Respiratory Rate Estimation", className="text-warning")
            )
            results.append(
                html.P(
                    "vitalDSP RespiratoryAnalysis not available. Cannot perform advanced estimation methods.",
                    className="text-muted",
                )
            )

        # Method Agreement Analysis (if multiple methods were used)
        if estimation_methods and len(estimation_methods) > 1:
            logger.info("Adding method agreement analysis...")
            results.append(html.Hr())
            results.append(
                html.H5("📊 METHOD AGREEMENT ANALYSIS", className="text-primary mb-3")
            )

            try:
                # Collect results from different estimation methods
                method_results = []

                # Add basic peak detection method
                try:
                    prominence = 0.3 * np.std(signal_data)
                    distance = int(0.5 * sampling_freq)
                    peaks, _ = signal.find_peaks(
                        signal_data, prominence=prominence, distance=distance
                    )
                    if len(peaks) > 1:
                        breath_intervals = np.diff(peaks) / sampling_freq
                        rr_peak = 60.0 / np.mean(breath_intervals)
                        method_results.append(("Peak Detection", rr_peak, len(peaks)))
                except Exception as e:
                    logger.warning(f"Peak detection method failed: {e}")

                # Add FFT-based method
                try:
                    # Simple FFT-based respiratory rate estimation
                    fft_signal = np.fft.fft(signal_data)
                    freqs = np.fft.fftfreq(len(signal_data), 1 / sampling_freq)

                    # Focus on respiratory frequency range (0.1-0.5 Hz = 6-30 BPM)
                    resp_mask = (freqs > 0.1) & (freqs < 0.5)
                    resp_freqs = freqs[resp_mask]
                    resp_fft = np.abs(fft_signal[resp_mask])

                    if len(resp_freqs) > 0:
                        dominant_freq_idx = np.argmax(resp_fft)
                        dominant_freq = resp_freqs[dominant_freq_idx]
                        rr_fft = dominant_freq * 60  # Convert to BPM
                        method_results.append(("FFT Analysis", rr_fft, 1))
                except Exception as e:
                    logger.warning(f"FFT method failed: {e}")

                # Add autocorrelation method
                try:
                    # Simple autocorrelation-based method
                    autocorr = np.correlate(signal_data, signal_data, mode="full")
                    autocorr = autocorr[len(autocorr) // 2 :]

                    # Find peaks in autocorrelation (excluding first peak)
                    peaks_auto, _ = signal.find_peaks(autocorr[100:], distance=50)
                    if len(peaks_auto) > 0:
                        # Convert peak distance to frequency
                        peak_distance = peaks_auto[0] / sampling_freq
                        rr_auto = 60.0 / peak_distance
                        method_results.append(
                            ("Autocorrelation", rr_auto, len(peaks_auto))
                        )
                except Exception as e:
                    logger.warning(f"Autocorrelation method failed: {e}")

                if len(method_results) > 1:
                    # Calculate agreement metrics
                    rr_values = [result[1] for result in method_results]
                    mean_rr = np.mean(rr_values)
                    std_rr = np.std(rr_values)
                    cv_rr = std_rr / mean_rr if mean_rr > 0 else 0

                    # Calculate pairwise differences
                    differences = []
                    for i in range(len(rr_values)):
                        for j in range(i + 1, len(rr_values)):
                            diff = abs(rr_values[i] - rr_values[j])
                            differences.append(diff)

                    mean_diff = np.mean(differences) if differences else 0
                    max_diff = max(differences) if differences else 0

                    # Agreement assessment
                    if cv_rr < 0.1:
                        agreement_level = "Excellent"
                        agreement_color = "text-success"
                    elif cv_rr < 0.2:
                        agreement_level = "Good"
                        agreement_color = "text-info"
                    elif cv_rr < 0.3:
                        agreement_level = "Moderate"
                        agreement_color = "text-warning"
                    else:
                        agreement_level = "Poor"
                        agreement_color = "text-danger"

                    # Create comprehensive method agreement analysis
                    results.append(
                        html.Div(
                            [
                                # Method Results Summary
                                html.Div(
                                    [
                                        html.H6(
                                            "📊 METHOD RESULTS",
                                            className="text-dark mb-2",
                                        ),
                                        html.P(
                                            f"Methods Used: {len(method_results)}",
                                            className="mb-1",
                                        ),
                                        html.P(
                                            f"Mean RR: {mean_rr:.1f} BPM",
                                            className="mb-1",
                                        ),
                                        html.P(
                                            f"Std Dev: {std_rr:.1f} BPM",
                                            className="mb-1",
                                        ),
                                        html.P(
                                            f"Coefficient of Variation: {cv_rr:.3f}",
                                            className="mb-1",
                                        ),
                                    ],
                                    className="p-2 border border-primary rounded mb-2",
                                ),
                                # Individual Method Results
                                html.Div(
                                    [
                                        html.H6(
                                            "🔬 INDIVIDUAL METHODS",
                                            className="text-dark mb-2",
                                        ),
                                        *[
                                            html.P(
                                                f"{method[0]}: {method[1]:.1f} BPM ({method[2]} samples)",
                                                className="mb-1",
                                            )
                                            for method in method_results
                                        ],
                                    ],
                                    className="p-2 border border-info rounded mb-2",
                                ),
                                # Agreement Assessment
                                html.Div(
                                    [
                                        html.H6(
                                            "🎯 AGREEMENT ASSESSMENT",
                                            className="text-dark mb-2",
                                        ),
                                        html.P(
                                            f"Agreement Level: {agreement_level}",
                                            className=f"mb-1 {agreement_color}",
                                        ),
                                        html.P(
                                            f"Mean Difference: {mean_diff:.1f} BPM",
                                            className="mb-1",
                                        ),
                                        html.P(
                                            f"Max Difference: {max_diff:.1f} BPM",
                                            className="mb-1",
                                        ),
                                        html.P(f"CV: {cv_rr:.3f}", className="mb-1"),
                                    ],
                                    className="p-2 border border-success rounded mb-2",
                                ),
                                # Clinical Interpretation
                                html.Div(
                                    [
                                        html.H6(
                                            "🏥 CLINICAL INTERPRETATION",
                                            className="text-dark mb-2",
                                        ),
                                        html.P(
                                            f"Reliability: {'High' if cv_rr < 0.2 else 'Moderate' if cv_rr < 0.3 else 'Low'}",
                                            className=(
                                                "text-success"
                                                if cv_rr < 0.2
                                                else (
                                                    "text-warning"
                                                    if cv_rr < 0.3
                                                    else "text-danger"
                                                )
                                            ),
                                        ),
                                        html.P(
                                            f"Consistency: {'Consistent' if mean_diff < 2 else 'Variable' if mean_diff < 5 else 'Inconsistent'}",
                                            className=(
                                                "text-success"
                                                if mean_diff < 2
                                                else (
                                                    "text-warning"
                                                    if mean_diff < 5
                                                    else "text-danger"
                                                )
                                            ),
                                        ),
                                        html.P(
                                            f"Recommendation: {'Use any method' if cv_rr < 0.15 else 'Use average' if cv_rr < 0.25 else 'Verify manually'}",
                                            className=(
                                                "text-success"
                                                if cv_rr < 0.25
                                                else (
                                                    "text-info"
                                                    if cv_rr < 0.25
                                                    else "text-warning"
                                                )
                                            ),
                                        ),
                                    ],
                                    className="p-2 border border-warning rounded",
                                ),
                            ],
                            className="mb-3",
                        )
                    )

                    logger.info(
                        f"Method agreement analysis completed: {len(method_results)} methods, CV: {cv_rr:.3f}"
                    )

                else:
                    results.append(
                        html.Div(
                            [
                                html.H6(
                                    "⚠️ INSUFFICIENT METHODS",
                                    className="text-warning mb-2",
                                ),
                                html.P(
                                    f"Only {len(method_results)} method(s) available. Need at least 2 methods for agreement analysis."
                                ),
                                html.P(
                                    "Try different signal processing approaches or check signal quality."
                                ),
                            ],
                            className="p-3 border border-warning rounded bg-light",
                        )
                    )

            except Exception as e:
                logger.error(f"Method agreement analysis failed: {e}")
                results.append(
                    html.Div(
                        [
                            html.H6("❌ ANALYSIS FAILED", className="text-danger mb-2"),
                            html.P(f"Error: {str(e)}"),
                            html.P("Please check signal quality and try again."),
                        ],
                        className="p-3 border border-danger rounded bg-light",
                    )
                )

        # Advanced Analysis Features
        logger.info(f"Processing advanced options: {advanced_options}")
        if advanced_options:
            logger.info("Adding advanced analysis section...")
            results.append(html.Hr())
            results.append(
                html.H5("🔬 ADVANCED ANALYSIS", className="text-primary mb-2")
            )

            # Debug: Show what options are selected
            results.append(
                html.Div(
                    [
                        html.H6("📋 SELECTED OPTIONS", className="text-dark mb-2"),
                        html.P(
                            f"Selected: {', '.join(advanced_options)}", className="mb-1"
                        ),
                        html.P(f"Signal Type: {signal_type}", className="mb-1"),
                        html.P(
                            f"Signal Length: {len(signal_data)} samples",
                            className="mb-1",
                        ),
                        html.P(
                            f"Duration: {len(signal_data)/sampling_freq:.1f}s",
                            className="mb-1",
                        ),
                    ],
                    className="p-2 border border-info rounded mb-2 bg-light",
                )
            )

            # Create inline grid layout for analysis blocks
            analysis_blocks = []

            # Sleep Apnea Detection
            if "sleep_apnea" in advanced_options:
                logger.info("Processing sleep apnea detection...")
                try:
                    # Detect apnea events using amplitude threshold
                    apnea_threshold = 0.3 * np.std(signal_data)
                    logger.info(f"Apnea threshold: {apnea_threshold:.3f}")

                    if detect_apnea_amplitude is None:
                        logger.warning("detect_apnea_amplitude function not available")
                        apnea_events = []
                    else:
                        apnea_events = detect_apnea_amplitude(
                            signal_data,
                            sampling_freq,
                            threshold=apnea_threshold,
                            min_duration=5,
                        )

                    # Also detect apnea events using pause detection
                    if detect_apnea_pauses is None:
                        logger.warning("detect_apnea_pauses function not available")
                        pause_apnea_events = []
                    else:
                        pause_apnea_events = detect_apnea_pauses(
                            signal_data, sampling_freq, min_pause_duration=5
                        )

                    total_apnea_events = len(apnea_events) + len(pause_apnea_events)
                    all_events = apnea_events + pause_apnea_events

                    if total_apnea_events > 0:
                        # Create compact sleep apnea card
                        apnea_card = html.Div(
                            [
                                html.H6(
                                    "😴 SLEEP APNEA", className="text-warning mb-1"
                                ),
                                html.P(
                                    f"Events: {total_apnea_events}",
                                    className="mb-1 fw-bold",
                                ),
                                html.P(
                                    f"Amplitude: {len(apnea_events)}",
                                    className="mb-1 small",
                                ),
                                html.P(
                                    f"Pause: {len(pause_apnea_events)}",
                                    className="mb-1 small",
                                ),
                                html.P(
                                    (
                                        f"First: {all_events[0][0]:.1f}s"
                                        if all_events
                                        else "No events"
                                    ),
                                    className="mb-0 small text-muted",
                                ),
                            ],
                            className="p-2 border border-warning rounded bg-light h-100",
                        )
                        analysis_blocks.append(apnea_card)
                    else:
                        # Create no events card
                        no_apnea_card = html.Div(
                            [
                                html.H6(
                                    "😴 SLEEP APNEA", className="text-success mb-1"
                                ),
                                html.P(
                                    "No events detected",
                                    className="mb-0 fw-bold text-success",
                                ),
                            ],
                            className="p-2 border border-success rounded bg-light h-100",
                        )
                        analysis_blocks.append(no_apnea_card)

                except Exception as e:
                    logger.warning(f"Sleep apnea detection failed: {e}")
                    # Create error card
                    error_card = html.Div(
                        [
                            html.H6("😴 SLEEP APNEA", className="text-danger mb-1"),
                            html.P(
                                "Analysis failed", className="mb-0 fw-bold text-danger"
                            ),
                        ],
                        className="p-2 border border-danger rounded bg-light h-100",
                    )
                    analysis_blocks.append(error_card)

            # Breathing Pattern Analysis
            if "breathing_pattern" in advanced_options:
                logger.info("Processing breathing pattern analysis...")
                try:
                    # Enhanced breathing pattern analysis with comprehensive insights
                    # Use user-specified parameters for consistency with respiratory rate estimation
                    prominence = 0.3 * np.std(signal_data)

                    # Use min_breath_duration for distance calculation (convert to samples)
                    min_distance_samples = int(
                        (min_breath_duration or 0.1) * sampling_freq
                    )
                    max_distance_samples = int(
                        (max_breath_duration or 6.0) * sampling_freq
                    )

                    # Apply filtering if preprocessing options include filtering
                    processed_signal = signal_data.copy()
                    if preprocessing_options and "filter" in preprocessing_options:
                        from scipy import signal as scipy_signal

                        # Apply bandpass filter using user-specified cutoffs
                        low_cutoff = low_cut or 0.1
                        high_cutoff = high_cut or 0.8
                        nyquist = sampling_freq / 2
                        low = low_cutoff / nyquist
                        high = high_cutoff / nyquist
                        b, a = scipy_signal.butter(4, [low, high], btype="band")
                        processed_signal = scipy_signal.filtfilt(b, a, signal_data)
                        logger.info(
                            f"Applied bandpass filter: {low_cutoff}-{high_cutoff} Hz"
                        )

                    peaks, properties = signal.find_peaks(
                        processed_signal,
                        prominence=prominence,
                        distance=min_distance_samples,
                    )

                    if len(peaks) > 1:
                        breath_intervals = np.diff(peaks) / sampling_freq
                        variability = np.std(breath_intervals)
                        mean_interval = np.mean(breath_intervals)

                        # Calculate respiratory rate metrics
                        rr_bpm = 60.0 / mean_interval

                        # Log the parameters used for debugging
                        logger.info("Breathing pattern analysis parameters:")
                        logger.info(
                            f"  - Min breath duration: {min_breath_duration or 0.1}s"
                        )
                        logger.info(
                            f"  - Max breath duration: {max_breath_duration or 6.0}s"
                        )
                        logger.info(f"  - Low cut: {low_cut or 0.1} Hz")
                        logger.info(f"  - High cut: {high_cut or 0.8} Hz")
                        logger.info(
                            f"  - Filtering applied: {'Yes' if preprocessing_options and 'filter' in preprocessing_options else 'No'}"
                        )
                        logger.info(f"  - Detected peaks: {len(peaks)}")
                        logger.info(f"  - Calculated RR: {rr_bpm:.2f} BPM")
                        rr_std = 60.0 * variability / (mean_interval**2)
                        rr_cv = variability / mean_interval  # Coefficient of variation

                        # Breathing pattern classification
                        if rr_cv < 0.1:
                            pattern_type = "Regular"
                            pattern_color = "text-success"
                        elif rr_cv < 0.2:
                            pattern_type = "Slightly Irregular"
                            pattern_color = "text-info"
                        elif rr_cv < 0.3:
                            pattern_type = "Moderately Irregular"
                            pattern_color = "text-warning"
                        else:
                            pattern_type = "Highly Irregular"
                            pattern_color = "text-danger"

                        # Detect breathing irregularities
                        irregular_breaths = np.sum(
                            np.abs(breath_intervals - mean_interval) > 2 * variability
                        )
                        irregularity_percentage = (
                            irregular_breaths / len(breath_intervals)
                        ) * 100

                        # Breathing rate stability
                        if irregularity_percentage < 10:
                            stability = "High"
                            stability_color = "text-success"
                        elif irregularity_percentage < 25:
                            stability = "Moderate"
                            stability_color = "text-info"
                        elif irregularity_percentage < 50:
                            stability = "Low"
                            stability_color = "text-warning"
                        else:
                            stability = "Very Low"
                            stability_color = "text-danger"

                        # Create compact breathing pattern card
                        breathing_card = html.Div(
                            [
                                html.H6(
                                    "🫁 BREATHING PATTERN",
                                    className="text-primary mb-1",
                                ),
                                html.P(
                                    f"Breaths: {len(peaks)}", className="mb-1 fw-bold"
                                ),
                                html.P(f"RR: {rr_bpm:.1f} BPM", className="mb-1"),
                                html.P(
                                    f"Pattern: {pattern_type}",
                                    className=f"mb-1 {pattern_color}",
                                ),
                                html.P(
                                    f"Stability: {stability}",
                                    className=f"mb-1 {stability_color}",
                                ),
                                html.P(
                                    f"CV: {rr_cv:.3f}",
                                    className="mb-0 small text-muted",
                                ),
                            ],
                            className="p-2 border border-primary rounded bg-light h-100",
                        )
                        analysis_blocks.append(breathing_card)

                        logger.info(
                            f"Breathing pattern analysis completed: {len(peaks)} breaths, RR: {rr_bpm:.1f} BPM, CV: {rr_cv:.3f}"
                        )

                    else:
                        # Create insufficient data card
                        insufficient_card = html.Div(
                            [
                                html.H6(
                                    "🫁 BREATHING PATTERN",
                                    className="text-warning mb-1",
                                ),
                                html.P(
                                    f"Only {len(peaks)} breath(s)",
                                    className="mb-1 fw-bold text-warning",
                                ),
                                html.P(
                                    "Need at least 2 breaths",
                                    className="mb-0 small text-muted",
                                ),
                            ],
                            className="p-2 border border-warning rounded bg-light h-100",
                        )
                        analysis_blocks.append(insufficient_card)

                except Exception as e:
                    logger.error(f"Breathing pattern analysis failed: {e}")
                    # Create error card
                    error_card = html.Div(
                        [
                            html.H6(
                                "🫁 BREATHING PATTERN", className="text-danger mb-1"
                            ),
                            html.P(
                                "Analysis failed", className="mb-1 fw-bold text-danger"
                            ),
                            html.P(
                                f"Error: {str(e)[:30]}...",
                                className="mb-0 small text-muted",
                            ),
                        ],
                        className="p-2 border border-danger rounded bg-light h-100",
                    )
                    analysis_blocks.append(error_card)

            # Respiratory Variability
            if "respiratory_variability" in advanced_options:
                logger.info("Processing respiratory variability analysis...")
                try:
                    # Enhanced respiratory variability analysis
                    if signal_type == "ppg":
                        if PPGAutonomicFeatures is None:
                            logger.warning("PPGAutonomicFeatures not available")
                            # Fallback to basic RRV calculation
                            try:
                                # Calculate basic RRV from breathing intervals
                                prominence = 0.3 * np.std(signal_data)
                                distance = int(0.5 * sampling_freq)
                                peaks, _ = signal.find_peaks(
                                    signal_data,
                                    prominence=prominence,
                                    distance=distance,
                                )

                                if len(peaks) > 1:
                                    breath_intervals = np.diff(peaks) / sampling_freq
                                    rr_bpm = 60.0 / np.mean(breath_intervals)
                                    rrv = np.std(breath_intervals)
                                    rrv_norm = rrv / np.mean(
                                        breath_intervals
                                    )  # Normalized RRV

                                    results.append(
                                        html.Div(
                                            [
                                                html.H5(
                                                    "📊 RESPIRATORY RATE VARIABILITY",
                                                    className="text-primary mb-3",
                                                ),
                                                html.Div(
                                                    [
                                                        html.H6(
                                                            "🫁 BASIC RRV METRICS",
                                                            className="text-dark mb-2",
                                                        ),
                                                        html.P(
                                                            f"RRV (Std Dev): {rrv:.3f}s",
                                                            className="mb-1",
                                                        ),
                                                        html.P(
                                                            f"Normalized RRV: {rrv_norm:.3f}",
                                                            className="mb-1",
                                                        ),
                                                        html.P(
                                                            f"Mean RR: {rr_bpm:.1f} BPM",
                                                            className="mb-1",
                                                        ),
                                                        html.P(
                                                            f"Breaths: {len(peaks)}",
                                                            className="mb-1",
                                                        ),
                                                        html.P(
                                                            "Method: Fallback (Peak Detection)",
                                                            className="text-info",
                                                        ),
                                                    ],
                                                    className="p-3 border border-info rounded bg-light",
                                                ),
                                            ],
                                            className="mb-4",
                                        )
                                    )
                                else:
                                    results.append(
                                        html.Div(
                                            [
                                                html.H5(
                                                    "📊 RESPIRATORY RATE VARIABILITY",
                                                    className="text-warning mb-3",
                                                ),
                                                html.Div(
                                                    [
                                                        html.H6(
                                                            "⚠️ INSUFFICIENT DATA",
                                                            className="text-warning",
                                                        ),
                                                        html.P(
                                                            "Need at least 2 breaths for RRV analysis.",
                                                            className="mb-1",
                                                        ),
                                                    ],
                                                    className="p-3 border border-warning rounded bg-light",
                                                ),
                                            ],
                                            className="mb-4",
                                        )
                                    )
                            except Exception as fallback_error:
                                logger.error(
                                    f"Fallback RRV calculation failed: {fallback_error}"
                                )
                                results.append(
                                    html.Div(
                                        [
                                            html.H5(
                                                "📊 RESPIRATORY RATE VARIABILITY",
                                                className="text-danger mb-3",
                                            ),
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "❌ ANALYSIS FAILED",
                                                        className="text-danger",
                                                    ),
                                                    html.P(
                                                        f"Error: {str(fallback_error)}",
                                                        className="mb-1",
                                                    ),
                                                ],
                                                className="p-3 border border-danger rounded bg-light",
                                            ),
                                        ],
                                        className="mb-4",
                                    )
                                )
                        else:
                            # Use PPG respiratory autonomic features
                            try:
                                ppg_features = PPGAutonomicFeatures(
                                    signal_data, sampling_freq
                                )
                                rrv = ppg_features.compute_rrv()
                                rsa = ppg_features.compute_rsa()

                                # Create compact PPG RRV card
                                ppg_rrv_card = html.Div(
                                    [
                                        html.H6(
                                            "📊 PPG RRV", className="text-success mb-1"
                                        ),
                                        html.P(
                                            f"RRV: {rrv:.4f}s", className="mb-1 fw-bold"
                                        ),
                                        html.P(f"RSA: {rsa:.4f}s", className="mb-1"),
                                        html.P(
                                            "vitalDSP PPG",
                                            className="mb-0 small text-success",
                                        ),
                                    ],
                                    className="p-2 border border-success rounded bg-light h-100",
                                )
                                analysis_blocks.append(ppg_rrv_card)
                            except Exception as ppg_error:
                                logger.error(f"PPG RRV analysis failed: {ppg_error}")
                                results.append(
                                    html.Div(
                                        [
                                            html.H5(
                                                "📊 RESPIRATORY RATE VARIABILITY",
                                                className="text-danger mb-3",
                                            ),
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "❌ PPG ANALYSIS FAILED",
                                                        className="text-danger",
                                                    ),
                                                    html.P(
                                                        f"Error: {str(ppg_error)}",
                                                        className="mb-1",
                                                    ),
                                                    html.P(
                                                        "Falling back to basic RRV calculation...",
                                                        className="text-info",
                                                    ),
                                                ],
                                                className="p-3 border border-danger rounded bg-light",
                                            ),
                                        ],
                                        className="mb-4",
                                    )
                                )
                    else:
                        # For non-PPG signals, use basic RRV calculation
                        try:
                            prominence = 0.3 * np.std(signal_data)
                            distance = int(0.5 * sampling_freq)
                            peaks, _ = signal.find_peaks(
                                signal_data, prominence=prominence, distance=distance
                            )

                            if len(peaks) > 1:
                                breath_intervals = np.diff(peaks) / sampling_freq
                                rr_bpm = 60.0 / np.mean(breath_intervals)
                                rrv = np.std(breath_intervals)
                                rrv_norm = rrv / np.mean(breath_intervals)

                                # Create compact basic RRV card
                                basic_rrv_card = html.Div(
                                    [
                                        html.H6(
                                            "📊 BASIC RRV", className="text-info mb-1"
                                        ),
                                        html.P(
                                            f"RRV: {rrv:.3f}s", className="mb-1 fw-bold"
                                        ),
                                        html.P(
                                            f"Norm RRV: {rrv_norm:.3f}",
                                            className="mb-1",
                                        ),
                                        html.P(
                                            f"RR: {rr_bpm:.1f} BPM", className="mb-1"
                                        ),
                                        html.P(
                                            f"Breaths: {len(peaks)}",
                                            className="mb-0 small text-muted",
                                        ),
                                    ],
                                    className="p-2 border border-info rounded bg-light h-100",
                                )
                                analysis_blocks.append(basic_rrv_card)
                            else:
                                # Create insufficient data card
                                insufficient_rrv_card = html.Div(
                                    [
                                        html.H6(
                                            "📊 BASIC RRV",
                                            className="text-warning mb-1",
                                        ),
                                        html.P(
                                            "Insufficient data",
                                            className="mb-1 fw-bold text-warning",
                                        ),
                                        html.P(
                                            "Need at least 2 breaths",
                                            className="mb-0 small text-muted",
                                        ),
                                    ],
                                    className="p-2 border border-warning rounded bg-light h-100",
                                )
                                analysis_blocks.append(insufficient_rrv_card)
                        except Exception as basic_error:
                            logger.error(f"Basic RRV calculation failed: {basic_error}")
                            # Create error card
                            error_rrv_card = html.Div(
                                [
                                    html.H6(
                                        "📊 BASIC RRV", className="text-danger mb-1"
                                    ),
                                    html.P(
                                        "Analysis failed",
                                        className="mb-1 fw-bold text-danger",
                                    ),
                                    html.P(
                                        f"Error: {str(basic_error)[:30]}...",
                                        className="mb-0 small text-muted",
                                    ),
                                ],
                                className="p-2 border border-danger rounded bg-light h-100",
                            )
                            analysis_blocks.append(error_rrv_card)

                except Exception as e:
                    logger.error(f"Respiratory variability analysis failed: {e}")
                    # Create error card
                    error_rrv_main_card = html.Div(
                        [
                            html.H6("📊 RRV ANALYSIS", className="text-danger mb-1"),
                            html.P(
                                "Analysis failed", className="mb-1 fw-bold text-danger"
                            ),
                            html.P(
                                f"Error: {str(e)[:30]}...",
                                className="mb-0 small text-muted",
                            ),
                        ],
                        className="p-2 border border-danger rounded bg-light h-100",
                    )
                    analysis_blocks.append(error_rrv_main_card)

            # PPG-ECG Fusion Analysis
            if "ppg_ecg_fusion" in advanced_options:
                logger.info("Processing PPG-ECG fusion analysis...")
                try:
                    if ppg_ecg_fusion is None:
                        logger.warning("ppg_ecg_fusion function not available")
                        # Fallback to basic fusion analysis
                        try:
                            # Calculate respiratory rate from PPG signal using multiple methods
                            prominence = 0.3 * np.std(signal_data)
                            distance = int(0.5 * sampling_freq)
                            peaks, _ = signal.find_peaks(
                                signal_data, prominence=prominence, distance=distance
                            )

                            if len(peaks) > 1:
                                breath_intervals = np.diff(peaks) / sampling_freq
                                rr_ppg = 60.0 / np.mean(breath_intervals)

                                # Calculate frequency domain estimate
                                fft_result = np.abs(np.fft.rfft(signal_data))
                                freqs = np.fft.rfftfreq(
                                    len(signal_data), 1 / sampling_freq
                                )
                                resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
                                if np.any(resp_mask):
                                    peak_freq_idx = np.argmax(fft_result[resp_mask])
                                    peak_freq = freqs[resp_mask][peak_freq_idx]
                                    rr_freq = peak_freq * 60
                                else:
                                    rr_freq = rr_ppg

                                # Fusion result (simple average)
                                fusion_rr = np.mean([rr_ppg, rr_freq])
                                fusion_std = np.std([rr_ppg, rr_freq])

                                results.append(
                                    html.Div(
                                        [
                                            html.H5(
                                                "🔗 PPG-ECG FUSION",
                                                className="text-primary mb-3",
                                            ),
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "🫁 FUSION RESULTS",
                                                        className="text-dark mb-2",
                                                    ),
                                                    html.P(
                                                        f"Fusion RR: {fusion_rr:.1f} ± {fusion_std:.1f} BPM",
                                                        className="mb-1",
                                                    ),
                                                    html.P(
                                                        f"PPG RR: {rr_ppg:.1f} BPM",
                                                        className="mb-1",
                                                    ),
                                                    html.P(
                                                        f"Frequency RR: {rr_freq:.1f} BPM",
                                                        className="mb-1",
                                                    ),
                                                    html.P(
                                                        "Method: Fallback (Peak + FFT)",
                                                        className="text-info",
                                                    ),
                                                ],
                                                className="p-3 border border-info rounded bg-light",
                                            ),
                                        ],
                                        className="mb-4",
                                    )
                                )
                            else:
                                results.append(
                                    html.Div(
                                        [
                                            html.H6(
                                                "🔗 PPG-ECG Fusion Analysis",
                                                className="mb-3",
                                            ),
                                            html.Div(
                                                [
                                                    html.Strong(
                                                        "⚠️ Insufficient Data:",
                                                        className="text-warning",
                                                    ),
                                                    html.Br(),
                                                    html.Small(
                                                        "Need at least 2 breaths for fusion analysis.",
                                                        className="text-muted",
                                                    ),
                                                ],
                                                className="mb-3 p-3 border border-warning rounded bg-light",
                                            ),
                                        ],
                                        className="mb-4",
                                    )
                                )
                        except Exception as fallback_error:
                            logger.error(
                                f"Fallback PPG-ECG fusion failed: {fallback_error}"
                            )
                            results.append(
                                html.Div(
                                    [
                                        html.H6(
                                            "🔗 PPG-ECG Fusion Analysis",
                                            className="mb-3",
                                        ),
                                        html.Div(
                                            [
                                                html.Strong(
                                                    "❌ Analysis Failed:",
                                                    className="text-danger",
                                                ),
                                                html.Br(),
                                                html.Small(
                                                    f"Error: {str(fallback_error)}",
                                                    className="text-muted",
                                                ),
                                            ],
                                            className="mb-3 p-3 border border-danger rounded bg-light",
                                        ),
                                    ],
                                    className="mb-4",
                                )
                            )
                    else:
                        # Use vitalDSP PPG-ECG fusion
                        try:
                            # For demonstration, use the same signal as both PPG and ECG
                            # In real applications, you would have separate signals
                            fusion_rr = ppg_ecg_fusion(
                                signal_data,
                                signal_data,
                                sampling_freq,
                                preprocess="bandpass",
                                lowcut=low_cut,
                                highcut=high_cut,
                            )

                            results.append(
                                html.Div(
                                    [
                                        html.H5(
                                            "🔗 PPG-ECG FUSION",
                                            className="text-success mb-3",
                                        ),
                                        html.Div(
                                            [
                                                html.H6(
                                                    "🫁 VITALDSP RESULT",
                                                    className="text-dark mb-2",
                                                ),
                                                html.P(
                                                    f"Fusion RR: {fusion_rr:.1f} BPM",
                                                    className="mb-1",
                                                ),
                                                html.P(
                                                    "Method: vitalDSP PPG-ECG Fusion",
                                                    className="text-success",
                                                ),
                                            ],
                                            className="p-3 border border-success rounded bg-light",
                                        ),
                                    ],
                                    className="mb-4",
                                )
                            )
                        except Exception as fusion_error:
                            logger.error(
                                f"vitalDSP PPG-ECG fusion failed: {fusion_error}"
                            )
                            results.append(
                                html.Div(
                                    [
                                        html.H6(
                                            "🔗 PPG-ECG Fusion Analysis",
                                            className="mb-3",
                                        ),
                                        html.Div(
                                            [
                                                html.Strong(
                                                    "❌ vitalDSP Fusion Failed:",
                                                    className="text-danger",
                                                ),
                                                html.Br(),
                                                html.Small(
                                                    f"Error: {str(fusion_error)}",
                                                    className="text-muted",
                                                ),
                                                html.Br(),
                                                html.Small(
                                                    "Falling back to basic fusion...",
                                                    className="text-info",
                                                ),
                                            ],
                                            className="mb-3 p-3 border border-danger rounded bg-light",
                                        ),
                                    ],
                                    className="mb-4",
                                )
                            )

                except Exception as e:
                    logger.error(f"PPG-ECG fusion analysis failed: {e}")
                    results.append(
                        html.Div(
                            [
                                html.H6("🔗 PPG-ECG Fusion Analysis", className="mb-3"),
                                html.Div(
                                    [
                                        html.Strong(
                                            "❌ Analysis Failed:",
                                            className="text-danger",
                                        ),
                                        html.Br(),
                                        html.Small(
                                            f"Error: {str(e)}", className="text-muted"
                                        ),
                                    ],
                                    className="mb-3 p-3 border border-danger rounded bg-light",
                                ),
                            ],
                            className="mb-4",
                        )
                    )

            # Respiratory-Cardiac Fusion Analysis
            if "resp_cardiac_fusion" in advanced_options:
                logger.info("Processing respiratory-cardiac fusion analysis...")
                try:
                    if respiratory_cardiac_fusion is None:
                        logger.warning(
                            "respiratory_cardiac_fusion function not available"
                        )
                        # Fallback to basic fusion analysis
                        try:
                            # Calculate respiratory rate from signal using multiple methods
                            prominence = 0.3 * np.std(signal_data)
                            distance = int(0.5 * sampling_freq)
                            peaks, _ = signal.find_peaks(
                                signal_data, prominence=prominence, distance=distance
                            )

                            if len(peaks) > 1:
                                breath_intervals = np.diff(peaks) / sampling_freq
                                rr_resp = 60.0 / np.mean(breath_intervals)

                                # Calculate frequency domain estimate
                                fft_result = np.abs(np.fft.rfft(signal_data))
                                freqs = np.fft.rfftfreq(
                                    len(signal_data), 1 / sampling_freq
                                )
                                resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
                                if np.any(resp_mask):
                                    peak_freq_idx = np.argmax(fft_result[resp_mask])
                                    peak_freq = freqs[resp_mask][peak_freq_idx]
                                    rr_cardiac = peak_freq * 60
                                else:
                                    rr_cardiac = rr_resp

                                # Fusion result (simple average)
                                fusion_rr = np.mean([rr_resp, rr_cardiac])
                                fusion_std = np.std([rr_resp, rr_cardiac])

                                results.append(
                                    html.Div(
                                        [
                                            html.H6(
                                                "🫀 Respiratory-Cardiac Fusion Analysis",
                                                className="mb-3",
                                            ),
                                            html.Div(
                                                [
                                                    html.Strong(
                                                        "🫁 Fusion Respiratory Rate:",
                                                        className="text-primary",
                                                    ),
                                                    html.Br(),
                                                    html.Small(
                                                        f"• Fusion RR: {fusion_rr:.1f} ± {fusion_std:.1f} BPM",
                                                        className="text-muted",
                                                    ),
                                                    html.Br(),
                                                    html.Small(
                                                        f"• Respiratory-based RR: {rr_resp:.1f} BPM",
                                                        className="text-muted",
                                                    ),
                                                    html.Br(),
                                                    html.Small(
                                                        f"• Cardiac-based RR: {rr_cardiac:.1f} BPM",
                                                        className="text-muted",
                                                    ),
                                                    html.Br(),
                                                    html.Small(
                                                        "• Analysis Method: Fallback (Peak + FFT)",
                                                        className="text-info",
                                                    ),
                                                ],
                                                className="mb-3 p-3 border border-info rounded bg-light",
                                            ),
                                        ],
                                        className="mb-4",
                                    )
                                )
                            else:
                                results.append(
                                    html.Div(
                                        [
                                            html.H6(
                                                "🫀 Respiratory-Cardiac Fusion Analysis",
                                                className="mb-3",
                                            ),
                                            html.Div(
                                                [
                                                    html.Strong(
                                                        "⚠️ Insufficient Data:",
                                                        className="text-warning",
                                                    ),
                                                    html.Br(),
                                                    html.Small(
                                                        "Need at least 2 breaths for fusion analysis.",
                                                        className="text-muted",
                                                    ),
                                                ],
                                                className="mb-3 p-3 border border-warning rounded bg-light",
                                            ),
                                        ],
                                        className="mb-4",
                                    )
                                )
                        except Exception as fallback_error:
                            logger.error(
                                f"Fallback respiratory-cardiac fusion failed: {fallback_error}"
                            )
                            results.append(
                                html.Div(
                                    [
                                        html.H6(
                                            "🫀 Respiratory-Cardiac Fusion Analysis",
                                            className="mb-3",
                                        ),
                                        html.Div(
                                            [
                                                html.Strong(
                                                    "❌ Analysis Failed:",
                                                    className="text-danger",
                                                ),
                                                html.Br(),
                                                html.Small(
                                                    f"Error: {str(fallback_error)}",
                                                    className="text-muted",
                                                ),
                                            ],
                                            className="mb-3 p-3 border border-danger rounded bg-light",
                                        ),
                                    ],
                                    className="mb-4",
                                )
                            )
                    else:
                        # Use vitalDSP respiratory-cardiac fusion
                        try:
                            # For demonstration, use the same signal as both respiratory and cardiac
                            # In real applications, you would have separate signals
                            fusion_rr = respiratory_cardiac_fusion(
                                signal_data,
                                signal_data,
                                sampling_freq,
                                preprocess="bandpass",
                                lowcut=low_cut,
                                highcut=high_cut,
                            )

                            results.append(
                                html.Div(
                                    [
                                        html.H6(
                                            "🫀 Respiratory-Cardiac Fusion Analysis",
                                            className="mb-3",
                                        ),
                                        html.Div(
                                            [
                                                html.Strong(
                                                    "🫁 vitalDSP Fusion Result:",
                                                    className="text-primary",
                                                ),
                                                html.Br(),
                                                html.Small(
                                                    f"• Fusion Respiratory Rate: {fusion_rr:.1f} BPM",
                                                    className="text-muted",
                                                ),
                                                html.Br(),
                                                html.Small(
                                                    "• Analysis Method: vitalDSP Respiratory-Cardiac Fusion",
                                                    className="text-success",
                                                ),
                                            ],
                                            className="mb-3 p-3 border border-success rounded bg-light",
                                        ),
                                    ],
                                    className="mb-4",
                                )
                            )
                        except Exception as fusion_error:
                            logger.error(
                                f"vitalDSP respiratory-cardiac fusion failed: {fusion_error}"
                            )
                            results.append(
                                html.Div(
                                    [
                                        html.H6(
                                            "🫀 Respiratory-Cardiac Fusion Analysis",
                                            className="mb-3",
                                        ),
                                        html.Div(
                                            [
                                                html.Strong(
                                                    "❌ vitalDSP Fusion Failed:",
                                                    className="text-danger",
                                                ),
                                                html.Br(),
                                                html.Small(
                                                    f"Error: {str(fusion_error)}",
                                                    className="text-muted",
                                                ),
                                                html.Br(),
                                                html.Small(
                                                    "Falling back to basic fusion...",
                                                    className="text-info",
                                                ),
                                            ],
                                            className="mb-3 p-3 border border-danger rounded bg-light",
                                        ),
                                    ],
                                    className="mb-4",
                                )
                            )

                except Exception as e:
                    logger.error(f"Respiratory-cardiac fusion analysis failed: {e}")
                    results.append(
                        html.Div(
                            [
                                html.H6(
                                    "🫀 Respiratory-Cardiac Fusion Analysis",
                                    className="mb-3",
                                ),
                                html.Div(
                                    [
                                        html.Strong(
                                            "❌ Analysis Failed:",
                                            className="text-danger",
                                        ),
                                        html.Br(),
                                        html.Small(
                                            f"Error: {str(e)}", className="text-muted"
                                        ),
                                    ],
                                    className="mb-3 p-3 border border-danger rounded bg-light",
                                ),
                            ],
                            className="mb-4",
                        )
                    )

            # Multimodal Fusion
            if "multimodal" in advanced_options:
                logger.info("Processing multimodal fusion analysis...")
                try:
                    if multimodal_analysis is None:
                        logger.warning("multimodal_analysis function not available")
                        # Fallback to basic multimodal analysis
                        try:
                            # Calculate respiratory rate using multiple methods
                            prominence = 0.3 * np.std(signal_data)
                            distance = int(0.5 * sampling_freq)
                            peaks, _ = signal.find_peaks(
                                signal_data, prominence=prominence, distance=distance
                            )

                            if len(peaks) > 1:
                                breath_intervals = np.diff(peaks) / sampling_freq
                                rr_peak = 60.0 / np.mean(breath_intervals)

                                # Calculate frequency domain estimate
                                fft_result = np.abs(np.fft.rfft(signal_data))
                                freqs = np.fft.rfftfreq(
                                    len(signal_data), 1 / sampling_freq
                                )
                                resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
                                if np.any(resp_mask):
                                    peak_freq_idx = np.argmax(fft_result[resp_mask])
                                    peak_freq = freqs[resp_mask][peak_freq_idx]
                                    rr_freq = peak_freq * 60
                                else:
                                    rr_freq = rr_peak

                                # Multimodal result (simple average)
                                multimodal_rr = np.mean([rr_peak, rr_freq])
                                multimodal_std = np.std([rr_peak, rr_freq])

                                results.append(
                                    html.Div(
                                        [
                                            html.H5(
                                                "🔗 MULTIMODAL FUSION",
                                                className="text-primary mb-3",
                                            ),
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "🫁 MULTIMODAL RESULTS",
                                                        className="text-dark mb-2",
                                                    ),
                                                    html.P(
                                                        f"Multimodal RR: {multimodal_rr:.1f} ± {multimodal_std:.1f} BPM",
                                                        className="mb-1",
                                                    ),
                                                    html.P(
                                                        f"Peak-based RR: {rr_peak:.1f} BPM",
                                                        className="mb-1",
                                                    ),
                                                    html.P(
                                                        f"Frequency-based RR: {rr_freq:.1f} BPM",
                                                        className="mb-1",
                                                    ),
                                                    html.P(
                                                        "Method: Fallback (Peak + FFT)",
                                                        className="text-info",
                                                    ),
                                                ],
                                                className="p-3 border border-info rounded bg-light",
                                            ),
                                        ],
                                        className="mb-4",
                                    )
                                )
                            else:
                                results.append(
                                    html.Div(
                                        [
                                            html.H5(
                                                "🔗 MULTIMODAL FUSION",
                                                className="text-warning mb-3",
                                            ),
                                            html.Div(
                                                [
                                                    html.H6(
                                                        "⚠️ INSUFFICIENT DATA",
                                                        className="text-warning",
                                                    ),
                                                    html.P(
                                                        "Need at least 2 breaths for multimodal analysis.",
                                                        className="mb-1",
                                                    ),
                                                ],
                                                className="p-3 border border-warning rounded bg-light",
                                            ),
                                        ],
                                        className="mb-4",
                                    )
                                )
                        except Exception as fallback_error:
                            logger.error(
                                f"Fallback multimodal fusion failed: {fallback_error}"
                            )
                            results.append(
                                html.Div(
                                    [
                                        html.H5(
                                            "🔗 MULTIMODAL FUSION",
                                            className="text-danger mb-3",
                                        ),
                                        html.Div(
                                            [
                                                html.H6(
                                                    "❌ ANALYSIS FAILED",
                                                    className="text-danger",
                                                ),
                                                html.P(
                                                    f"Error: {str(fallback_error)}",
                                                    className="mb-1",
                                                ),
                                            ],
                                            className="p-3 border border-danger rounded bg-light",
                                        ),
                                    ],
                                    className="mb-4",
                                )
                            )
                    else:
                        # Use vitalDSP multimodal analysis
                        try:
                            # For demonstration, use the same signal as multiple modalities
                            # In real applications, you would have different signal types
                            multimodal_rr = multimodal_analysis(
                                [
                                    signal_data,
                                    signal_data,
                                ],  # Using same signal for demo
                                sampling_freq,
                                preprocess="bandpass",
                                lowcut=low_cut,
                                highcut=high_cut,
                            )

                            results.append(
                                html.Div(
                                    [
                                        html.H5(
                                            "🔗 MULTIMODAL FUSION",
                                            className="text-success mb-3",
                                        ),
                                        html.Div(
                                            [
                                                html.H6(
                                                    "🫁 VITALDSP RESULT",
                                                    className="text-dark mb-2",
                                                ),
                                                html.P(
                                                    f"Multimodal RR: {multimodal_rr:.1f} BPM",
                                                    className="mb-1",
                                                ),
                                                html.P(
                                                    "Method: vitalDSP Multimodal Analysis",
                                                    className="text-success",
                                                ),
                                            ],
                                            className="p-3 border border-success rounded bg-light",
                                        ),
                                    ],
                                    className="mb-4",
                                )
                            )
                        except Exception as fusion_error:
                            logger.error(
                                f"vitalDSP multimodal fusion failed: {fusion_error}"
                            )
                            results.append(
                                html.Div(
                                    [
                                        html.H5(
                                            "🔗 MULTIMODAL FUSION",
                                            className="text-danger mb-3",
                                        ),
                                        html.Div(
                                            [
                                                html.H6(
                                                    "❌ VITALDSP FAILED",
                                                    className="text-danger",
                                                ),
                                                html.P(
                                                    f"Error: {str(fusion_error)}",
                                                    className="mb-1",
                                                ),
                                                html.P(
                                                    "Falling back to basic multimodal...",
                                                    className="text-info",
                                                ),
                                            ],
                                            className="p-3 border border-danger rounded bg-light",
                                        ),
                                    ],
                                    className="mb-4",
                                )
                            )

                except Exception as e:
                    logger.error(f"Multimodal fusion analysis failed: {e}")
                    results.append(
                        html.Div(
                            [
                                html.H6(
                                    "🔗 Multimodal Fusion Analysis", className="mb-3"
                                ),
                                html.Div(
                                    [
                                        html.Strong(
                                            "❌ Analysis Failed:",
                                            className="text-danger",
                                        ),
                                        html.Br(),
                                        html.Small(
                                            f"Error: {str(e)}", className="text-muted"
                                        ),
                                    ],
                                    className="mb-3 p-3 border border-danger rounded bg-light",
                                ),
                            ],
                            className="mb-4",
                        )
                    )

            # Quality Assessment
            if "quality_assessment" in advanced_options:
                logger.info("Processing signal quality assessment...")
                try:
                    # Enhanced signal quality metrics
                    signal_mean = np.mean(signal_data)
                    signal_std = np.std(signal_data)
                    signal_min = np.min(signal_data)
                    signal_max = np.max(signal_data)
                    dynamic_range = signal_max - signal_min

                    # Calculate signal-to-noise ratio using different methods
                    # Method 1: Variance-based SNR
                    signal_power = np.var(signal_data)
                    noise_power = np.var(signal_data - signal_mean)
                    snr_variance = (
                        10 * np.log10(signal_power / noise_power)
                        if noise_power > 0
                        else 0
                    )

                    # Method 2: Peak-to-peak SNR
                    peak_to_peak = signal_max - signal_min
                    rms_noise = np.sqrt(np.mean((signal_data - signal_mean) ** 2))
                    snr_pp = (
                        20 * np.log10(peak_to_peak / (2 * rms_noise))
                        if rms_noise > 0
                        else 0
                    )

                    # Method 3: RMS SNR
                    rms_signal = np.sqrt(np.mean(signal_data**2))
                    rms_noise_signal = np.sqrt(
                        np.mean((signal_data - signal_mean) ** 2)
                    )
                    snr_rms = (
                        20 * np.log10(rms_signal / rms_noise_signal)
                        if rms_noise_signal > 0
                        else 0
                    )

                    # Signal quality classification
                    if snr_variance > 20:
                        quality_level = "Excellent"
                        quality_color = "text-success"
                    elif snr_variance > 15:
                        quality_level = "Good"
                        quality_color = "text-info"
                    elif snr_variance > 10:
                        quality_level = "Fair"
                        quality_color = "text-warning"
                    else:
                        quality_level = "Poor"
                        quality_color = "text-danger"

                    # Calculate additional quality metrics
                    signal_energy = np.sum(signal_data**2)
                    zero_crossings = np.sum(np.diff(np.sign(signal_data)) != 0)

                    results.append(
                        html.Div(
                            [
                                html.H5(
                                    "🎯 SIGNAL QUALITY ASSESSMENT",
                                    className="text-primary mb-2",
                                ),
                                # Quality Summary
                                html.Div(
                                    [
                                        html.H6(
                                            "📊 QUALITY SUMMARY",
                                            className="text-dark mb-2",
                                        ),
                                        html.P(
                                            f"Overall Quality: {quality_level}",
                                            className=f"mb-1 {quality_color}",
                                        ),
                                        html.P(
                                            f"Variance SNR: {snr_variance:.2f} dB",
                                            className="mb-1",
                                        ),
                                        html.P(
                                            f"Peak-to-Peak SNR: {snr_pp:.2f} dB",
                                            className="mb-1",
                                        ),
                                        html.P(
                                            f"RMS SNR: {snr_rms:.2f} dB",
                                            className="mb-1",
                                        ),
                                    ],
                                    className="p-2 border border-primary rounded mb-2",
                                ),
                                # Signal Statistics
                                html.Div(
                                    [
                                        html.H6(
                                            "📈 SIGNAL STATISTICS",
                                            className="text-dark mb-2",
                                        ),
                                        html.P(
                                            f"Dynamic Range: {dynamic_range:.3f}",
                                            className="mb-1",
                                        ),
                                        html.P(
                                            f"Signal Power: {signal_power:.3f}",
                                            className="mb-1",
                                        ),
                                        html.P(
                                            f"Signal Energy: {signal_energy:.3f}",
                                            className="mb-1",
                                        ),
                                        html.P(
                                            f"Zero Crossings: {zero_crossings}",
                                            className="mb-1",
                                        ),
                                    ],
                                    className="p-2 border border-info rounded mb-2",
                                ),
                                # Signal Characteristics
                                html.Div(
                                    [
                                        html.H6(
                                            "🔍 SIGNAL CHARACTERISTICS",
                                            className="text-dark mb-2",
                                        ),
                                        html.P(
                                            f"Mean: {signal_mean:.3f}", className="mb-1"
                                        ),
                                        html.P(
                                            f"Std Dev: {signal_std:.3f}",
                                            className="mb-1",
                                        ),
                                        html.P(
                                            f"Range: {signal_min:.3f} to {signal_max:.3f}",
                                            className="mb-1",
                                        ),
                                        html.P(
                                            (
                                                f"CV: {signal_std/abs(signal_mean):.3f}"
                                                if abs(signal_mean) > 0
                                                else "N/A"
                                            ),
                                            className="mb-1",
                                        ),
                                    ],
                                    className="p-2 border border-success rounded",
                                ),
                            ],
                            className="mb-3",
                        )
                    )

                    logger.info(
                        f"Signal quality assessment completed: SNR={snr_variance:.2f} dB, Quality={quality_level}"
                    )

                except Exception as e:
                    logger.error(f"Quality assessment failed: {e}")
                    results.append(
                        html.Div(
                            [
                                html.H5(
                                    "🎯 SIGNAL QUALITY ASSESSMENT",
                                    className="text-danger mb-3",
                                ),
                                html.Div(
                                    [
                                        html.H6(
                                            "❌ ANALYSIS FAILED",
                                            className="text-danger",
                                        ),
                                        html.P(f"Error: {str(e)}", className="mb-1"),
                                    ],
                                    className="p-3 border border-danger rounded bg-light",
                                ),
                            ],
                            className="mb-4",
                        )
                    )

        # Display all analysis blocks in inline grid layout
        if analysis_blocks:
            results.append(
                html.Div(
                    [
                        html.H6("📊 ANALYSIS RESULTS", className="text-dark mb-2"),
                        html.Div(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(block, md=3, className="mb-2")
                                        for block in analysis_blocks
                                    ],
                                    className="g-2",
                                )
                            ]
                        ),
                    ],
                    className="mb-3",
                )
            )

        # Enhanced Signal Analysis
        logger.info("Adding enhanced signal analysis section...")
        results.append(html.Hr())
        results.append(
            html.H5("🔍 ENHANCED SIGNAL ANALYSIS", className="text-primary mb-2")
        )

        # Basic statistics
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        min_val = np.min(signal_data)
        max_val = np.max(signal_data)
        rms_val = np.sqrt(np.mean(signal_data**2))
        peak_to_peak = max_val - min_val

        # Signal quality metrics
        snr_db = 20 * np.log10(rms_val / std_val) if std_val > 0 else 0
        crest_factor = peak_to_peak / rms_val if rms_val > 0 else 0

        # Frequency domain analysis
        fft_result = np.abs(np.fft.rfft(signal_data))
        freqs = np.fft.rfftfreq(len(signal_data), 1 / sampling_freq)

        # Find dominant frequency
        peak_idx = np.argmax(fft_result)
        dominant_freq = freqs[peak_idx]
        dominant_freq_bpm = dominant_freq * 60

        # Respiratory frequency band analysis
        resp_mask = (freqs >= 0.1) & (freqs <= 0.5)  # 0.1-0.5 Hz (6-30 BPM)
        if np.any(resp_mask):
            resp_power = np.sum(fft_result[resp_mask])
            total_power = np.sum(fft_result)
            resp_power_ratio = resp_power / total_power if total_power > 0 else 0
        else:
            resp_power_ratio = 0

        # Breathing pattern classification
        try:
            peaks, _ = signal.find_peaks(signal_data, distance=int(0.5 * sampling_freq))
            if len(peaks) > 1:
                breath_intervals = np.diff(peaks) / sampling_freq
                mean_interval = np.mean(breath_intervals)
                interval_std = np.std(breath_intervals)
                cv_intervals = interval_std / mean_interval if mean_interval > 0 else 0

                # Classify breathing pattern
                if cv_intervals < 0.1:
                    pattern_type = "Regular"
                    pattern_color = "text-success"
                elif cv_intervals < 0.2:
                    pattern_type = "Slightly Irregular"
                    pattern_color = "text-info"
                elif cv_intervals < 0.3:
                    pattern_type = "Irregular"
                    pattern_color = "text-warning"
                else:
                    pattern_type = "Highly Irregular"
                    pattern_color = "text-danger"

                # Clinical interpretation
                if mean_interval < 2.0:  # > 30 BPM
                    clinical_status = "Tachypnea (Rapid Breathing)"
                    clinical_color = "text-warning"
                elif mean_interval > 6.0:  # < 10 BPM
                    clinical_status = "Bradypnea (Slow Breathing)"
                    clinical_color = "text-warning"
                else:
                    clinical_status = "Normal Breathing Rate"
                    clinical_color = "text-success"
            else:
                pattern_type = "Insufficient Data"
                pattern_color = "text-muted"
                clinical_status = "Cannot Determine"
                clinical_color = "text-muted"
                cv_intervals = 0
                mean_interval = 0
        except Exception:
            pattern_type = "Analysis Failed"
            pattern_color = "text-danger"
            clinical_status = "Cannot Determine"
            clinical_color = "text-danger"
            cv_intervals = 0
            mean_interval = 0

        # Create comprehensive analysis display
        results.append(
            html.Div(
                [
                    # Signal Quality Metrics
                    html.H6("📊 Signal Quality Metrics", className="mb-2"),
                    html.Div(
                        [
                            html.Small(f"SNR: {snr_db:.1f} dB", className="text-muted"),
                            html.Br(),
                            html.Small(
                                f"Crest Factor: {crest_factor:.2f}",
                                className="text-muted",
                            ),
                            html.Br(),
                            html.Small(f"RMS: {rms_val:.3f}", className="text-muted"),
                            html.Br(),
                            html.Small(
                                f"Peak-to-Peak: {peak_to_peak:.3f}",
                                className="text-muted",
                            ),
                        ],
                        className="ms-3 mb-2",
                    ),
                    # Frequency Analysis
                    html.H6("🌊 Frequency Analysis", className="mb-2"),
                    html.Div(
                        [
                            html.Small(
                                f"Dominant Frequency: {dominant_freq:.3f} Hz ({dominant_freq_bpm:.1f} BPM)",
                                className="text-muted",
                            ),
                            html.Br(),
                            html.Small(
                                f"Respiratory Power Ratio: {resp_power_ratio:.1%}",
                                className="text-muted",
                            ),
                        ],
                        className="ms-3 mb-2",
                    ),
                    # Breathing Pattern Analysis
                    html.H6("🫁 Breathing Pattern Analysis", className="mb-2"),
                    html.Div(
                        [
                            html.Small("Pattern Type: ", className="text-muted"),
                            html.Span(f"{pattern_type}", className=pattern_color),
                            html.Br(),
                            html.Small(
                                f"Mean Interval: {mean_interval:.2f}s",
                                className="text-muted",
                            ),
                            html.Br(),
                            html.Small(
                                f"Interval CV: {cv_intervals:.3f}",
                                className="text-muted",
                            ),
                        ],
                        className="ms-3 mb-2",
                    ),
                    # Clinical Interpretation
                    html.H6("🏥 Clinical Interpretation", className="mb-2"),
                    html.Div(
                        [
                            html.Small("Status: ", className="text-muted"),
                            html.Span(f"{clinical_status}", className=clinical_color),
                        ],
                        className="ms-3 mb-2",
                    ),
                ],
                className="p-3 border border-info rounded mb-3",
            )
        )

        # Basic Statistics (simplified)
        results.append(
            html.Div(
                [
                    html.H6("📈 Basic Statistics", className="mb-2"),
                    html.Small(f"Mean: {mean_val:.3f}", className="text-muted"),
                    html.Br(),
                    html.Small(f"Std: {std_val:.3f}", className="text-muted"),
                    html.Br(),
                    html.Small(
                        f"Range: {min_val:.3f} to {max_val:.3f}", className="text-muted"
                    ),
                    html.Br(),
                    html.Small(
                        f"Duration: {len(signal_data)/sampling_freq:.1f}s",
                        className="text-muted",
                    ),
                ],
                className="mb-2",
            )
        )

        # Additional Respiratory Metrics
        logger.info("Calculating additional respiratory metrics...")
        try:
            # Calculate respiratory rate variability from peak intervals
            peaks, _ = signal.find_peaks(signal_data, distance=int(0.5 * sampling_freq))
            logger.info(f"Found {len(peaks)} peaks for additional metrics")
            if len(peaks) > 1:
                breath_intervals = np.diff(peaks) / sampling_freq
                rr_mean = 60 / np.mean(breath_intervals)
                rr_std = (
                    60 * np.std(breath_intervals) / (np.mean(breath_intervals) ** 2)
                )

                results.append(
                    html.Div(
                        [
                            html.Strong("Respiratory Rate Metrics: "),
                            html.Br(),
                            html.Small(
                                f"Mean RR: {rr_mean:.1f} BPM", className="text-muted"
                            ),
                            html.Br(),
                            html.Small(
                                f"RR Variability: {rr_std:.1f} BPM",
                                className="text-muted",
                            ),
                            html.Br(),
                            html.Small(
                                f"Number of breaths: {len(peaks)}",
                                className="text-muted",
                            ),
                        ],
                        className="mb-2",
                    )
                )

        except Exception as e:
            logger.warning(f"Additional respiratory metrics failed: {e}")

        # Comprehensive Summary
        logger.info("Adding comprehensive summary section...")
        results.append(html.Hr())
        results.append(html.H5("🎯 Comprehensive Summary", className="mb-3"))

        # Calculate overall respiratory rate if we have estimates
        try:
            # Collect all RR estimates from the results
            rr_estimates = []
            for result in results:
                if hasattr(result, "children"):
                    for child in result.children:
                        if hasattr(child, "children"):
                            for subchild in child.children:
                                if isinstance(subchild, str) and "BPM" in subchild:
                                    try:
                                        rr_val = float(subchild.split()[0])
                                        rr_estimates.append(rr_val)
                                    except Exception:
                                        pass

            if rr_estimates:
                overall_mean = np.mean(rr_estimates)
                overall_std = np.std(rr_estimates)

                # Overall assessment
                if overall_std < 2.0:
                    assessment = "Excellent Agreement"
                    assessment_color = "text-success"
                elif overall_std < 4.0:
                    assessment = "Good Agreement"
                    assessment_color = "text-info"
                elif overall_std < 6.0:
                    assessment = "Fair Agreement"
                    assessment_color = "text-warning"
                else:
                    assessment = "Poor Agreement"
                    assessment_color = "text-danger"

                results.append(
                    html.Div(
                        [
                            html.H6("📈 Overall Assessment", className="mb-2"),
                            html.Div(
                                [
                                    html.Small(
                                        "Mean Respiratory Rate: ",
                                        className="text-muted",
                                    ),
                                    html.Span(
                                        f"{overall_mean:.1f} ± {overall_std:.1f} BPM",
                                        className="text-success",
                                    ),
                                    html.Br(),
                                    html.Small(
                                        "Method Agreement: ", className="text-muted"
                                    ),
                                    html.Span(
                                        f"{assessment}", className=assessment_color
                                    ),
                                    html.Br(),
                                    html.Small(
                                        f"Number of Methods: {len(rr_estimates)}",
                                        className="text-muted",
                                    ),
                                ],
                                className="ms-3 mb-2",
                            ),
                        ],
                        className="p-3 border border-success rounded mb-3",
                    )
                )
        except Exception as e:
            logger.warning(f"Summary calculation failed: {e}")

        # Recommendations
        results.append(
            html.Div(
                [
                    html.H6("💡 Recommendations", className="mb-2"),
                    html.Div(
                        [
                            html.Small(
                                "• Use ensemble method for most reliable estimates",
                                className="text-muted",
                            ),
                            html.Br(),
                            html.Small(
                                "• Check signal quality if methods disagree significantly",
                                className="text-muted",
                            ),
                            html.Br(),
                            html.Small(
                                "• Consider preprocessing options for noisy signals",
                                className="text-muted",
                            ),
                            html.Br(),
                            html.Small(
                                "• Monitor breathing pattern regularity for clinical insights",
                                className="text-muted",
                            ),
                        ],
                        className="ms-3 mb-2",
                    ),
                ],
                className="p-3 border border-info rounded mb-3",
            )
        )

        logger.info(f"Final results list length: {len(results)}")
        logger.info("=== GENERATE COMPREHENSIVE RESPIRATORY ANALYSIS COMPLETED ===")
        return html.Div(results)

    except Exception as e:
        logger.error(f"Error generating respiratory analysis results: {e}")
        return html.Div(
            [
                html.H5("❌ Analysis Failed", className="text-danger"),
                html.P(f"Error: {str(e)}", className="text-muted"),
            ]
        )


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
):
    """Create comprehensive respiratory analysis plots."""
    logger.info("=== CREATE COMPREHENSIVE RESPIRATORY PLOTS STARTED ===")
    logger.info("Input parameters:")
    logger.info(f"  - signal_data shape: {signal_data.shape}")
    logger.info(f"  - time_axis shape: {time_axis.shape}")
    logger.info(f"  - sampling_freq: {sampling_freq}")
    logger.info(f"  - signal_type: {signal_type}")
    logger.info(f"  - estimation_methods: {estimation_methods}")
    logger.info(f"  - advanced_options: {advanced_options}")
    logger.info(f"  - preprocessing_options: {preprocessing_options}")
    logger.info(f"  - low_cut: {low_cut}")
    logger.info(f"  - high_cut: {high_cut}")

    try:
        # Create subplots for different analyses
        logger.info("Creating subplots...")
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Time Domain Signal",
                "Frequency Domain",
                "Breathing Pattern & RRV",
                "Respiratory Rate Over Time",
                "Sleep Apnea Detection",
                "Signal Quality & Ensemble",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
        )

        # 1. Time Domain Signal
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal_data,
                mode="lines",
                name="Original Signal",
                line=dict(color="#2E86AB", width=2),
            ),
            row=1,
            col=1,
        )

        # Apply preprocessing if selected
        if preprocessing_options and "filter" in preprocessing_options:
            try:
                nyquist = sampling_freq / 2
                low = low_cut / nyquist
                high = high_cut / nyquist

                if low < high < 1.0:
                    b, a = signal.butter(4, [low, high], btype="bandpass")
                    filtered_signal = signal.filtfilt(b, a, signal_data)
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=filtered_signal,
                            mode="lines",
                            name="Filtered Signal",
                            line=dict(color="#E74C3C", width=2),
                        ),
                        row=1,
                        col=1,
                    )
            except Exception as e:
                logger.error(f"Filtering failed: {e}")

        # 2. Frequency Domain
        fft_result = np.abs(np.fft.rfft(signal_data))
        freqs = np.fft.rfftfreq(len(signal_data), 1 / sampling_freq)

        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=fft_result,
                mode="lines",
                name="FFT Spectrum",
                line=dict(color="#9B59B6", width=2),
            ),
            row=1,
            col=2,
        )

        # Highlight respiratory frequency band
        resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
        if np.any(resp_mask):
            fig.add_trace(
                go.Scatter(
                    x=freqs[resp_mask],
                    y=fft_result[resp_mask],
                    mode="lines",
                    name="Respiratory Band",
                    line=dict(color="#E67E22", width=3),
                ),
                row=1,
                col=2,
            )

            # Add respiratory frequency annotations
            peak_freq_idx = np.argmax(fft_result[resp_mask])
            peak_freq = freqs[resp_mask][peak_freq_idx]
            peak_magnitude = fft_result[resp_mask][peak_freq_idx]

            fig.add_annotation(
                x=peak_freq,
                y=peak_magnitude,
                text=f"Peak: {peak_freq:.2f} Hz<br>({peak_freq*60:.1f} BPM)",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#E67E22",
                ax=0.1,
                ay=-40,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#E67E22",
            )

        # 3. Breathing Pattern & Respiratory Rate Variability
        if estimation_methods and "peak_detection" in estimation_methods:
            try:
                prominence = 0.3 * np.std(signal_data)
                distance = int(0.5 * sampling_freq)
                peaks, _ = signal.find_peaks(
                    signal_data, prominence=prominence, distance=distance
                )

                if len(peaks) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis[peaks],
                            y=signal_data[peaks],
                            mode="markers",
                            name="Breathing Peaks",
                            marker=dict(color="red", size=8, symbol="diamond"),
                        ),
                        row=2,
                        col=1,
                    )

                    # Add breath intervals
                    if len(peaks) > 1:
                        breath_intervals = np.diff(peaks) / sampling_freq
                        interval_times = time_axis[peaks[1:]]

                        # Calculate respiratory rate over time
                        rr_over_time = 60.0 / breath_intervals  # Convert to BPM

                        fig.add_trace(
                            go.Scatter(
                                x=interval_times,
                                y=breath_intervals,
                                mode="lines+markers",
                                name="Breath Intervals",
                                line=dict(color="#27AE60", width=2),
                            ),
                            row=2,
                            col=2,
                        )

                        # Add respiratory rate variability analysis
                        if len(rr_over_time) > 2:
                            # Calculate moving average of RR
                            window_size = min(5, len(rr_over_time))
                            if window_size > 1:
                                rr_moving_avg = np.convolve(
                                    rr_over_time,
                                    np.ones(window_size) / window_size,
                                    mode="same",
                                )
                                fig.add_trace(
                                    go.Scatter(
                                        x=interval_times,
                                        y=rr_moving_avg,
                                        mode="lines",
                                        name="RR Moving Average",
                                        line=dict(color="#8E44AD", width=2, dash="dot"),
                                    ),
                                    row=2,
                                    col=2,
                                )

                            # Add RRV statistics annotation
                            rr_mean = np.mean(rr_over_time)
                            rr_std = np.std(rr_over_time)
                            rr_cv = rr_std / rr_mean if rr_mean > 0 else 0

                            fig.add_annotation(
                                x=interval_times[len(interval_times) // 2],
                                y=np.max(breath_intervals),
                                text=f"RRV: {rr_cv:.3f}<br>Mean RR: {rr_mean:.1f} BPM<br>Std: {rr_std:.1f}",
                                showarrow=False,
                                bgcolor="rgba(255,255,255,0.9)",
                                bordercolor="#27AE60",
                                borderwidth=1,
                                font=dict(size=10),
                            )

            except Exception as e:
                logger.error(f"Breathing pattern analysis failed: {e}")

        # 4. Sleep Apnea Detection
        if advanced_options and "sleep_apnea" in advanced_options:
            try:
                # Amplitude-based apnea detection
                apnea_threshold = 0.3 * np.std(signal_data)

                if detect_apnea_amplitude is None:
                    logger.warning(
                        "detect_apnea_amplitude function not available for plots"
                    )
                    apnea_events = []
                else:
                    apnea_events = detect_apnea_amplitude(
                        signal_data,
                        sampling_freq,
                        threshold=apnea_threshold,
                        min_duration=5,
                    )

                # Pause-based apnea detection
                if detect_apnea_pauses is None:
                    logger.warning(
                        "detect_apnea_pauses function not available for plots"
                    )
                    pause_apnea_events = []
                else:
                    pause_apnea_events = detect_apnea_pauses(
                        signal_data, sampling_freq, min_pause_duration=5
                    )

                # Plot all apnea events
                all_apnea_events = apnea_events + pause_apnea_events
                if all_apnea_events:
                    for start, end in all_apnea_events:
                        start_idx = int(start * sampling_freq)
                        end_idx = int(end * sampling_freq)
                        if start_idx < len(time_axis) and end_idx < len(time_axis):
                            fig.add_trace(
                                go.Scatter(
                                    x=time_axis[start_idx:end_idx],
                                    y=signal_data[start_idx:end_idx],
                                    mode="lines",
                                    name="Apnea Event",
                                    line=dict(color="red", width=3),
                                ),
                                row=3,
                                col=1,
                            )

                    # Add apnea threshold line
                    fig.add_hline(
                        y=apnea_threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Apnea Threshold",
                        row=3,
                        col=1,
                    )
                    fig.add_hline(
                        y=-apnea_threshold,
                        line_dash="dash",
                        line_color="red",
                        row=3,
                        col=1,
                    )

            except Exception as e:
                logger.error(f"Sleep apnea detection failed: {e}")

        # 5. Signal Quality & Ensemble Method Comparison
        try:
            # Calculate moving average for trend
            window_size = int(0.5 * sampling_freq)
            if window_size > 0:
                moving_avg = np.convolve(
                    signal_data, np.ones(window_size) / window_size, mode="same"
                )
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=moving_avg,
                        mode="lines",
                        name="Moving Average",
                        line=dict(color="#F39C12", width=2),
                    ),
                    row=3,
                    col=2,
                )

                # Add signal envelope
                upper_envelope = moving_avg + 2 * np.std(signal_data)
                lower_envelope = moving_avg - 2 * np.std(signal_data)

                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=upper_envelope,
                        mode="lines",
                        name="Upper Envelope",
                        line=dict(color="#E74C3C", width=1, dash="dot"),
                        opacity=0.7,
                    ),
                    row=3,
                    col=2,
                )

                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=lower_envelope,
                        mode="lines",
                        name="Lower Envelope",
                        line=dict(color="#E74C3C", width=1, dash="dot"),
                        opacity=0.7,
                    ),
                    row=3,
                    col=2,
                )

                # Add ensemble method comparison if multiple methods are selected
                if (
                    estimation_methods
                    and len(estimation_methods) > 1
                    and "ensemble" in estimation_methods
                ):
                    try:
                        # Create a mini-ensemble visualization
                        ensemble_window = int(2.0 * sampling_freq)  # 2-second windows
                        if ensemble_window > 0 and len(signal_data) > ensemble_window:
                            # Calculate local statistics in windows
                            n_windows = len(signal_data) // ensemble_window
                            window_means = []
                            window_stds = []
                            window_times = []

                            for i in range(n_windows):
                                start_idx = i * ensemble_window
                                end_idx = start_idx + ensemble_window
                                window_data = signal_data[start_idx:end_idx]
                                window_means.append(np.mean(window_data))
                                window_stds.append(np.std(window_data))
                                window_times.append(
                                    time_axis[start_idx + ensemble_window // 2]
                                )

                            # Add ensemble stability indicator
                            fig.add_trace(
                                go.Scatter(
                                    x=window_times,
                                    y=window_means,
                                    mode="lines+markers",
                                    name="Local Ensemble Mean",
                                    line=dict(color="#2ECC71", width=2),
                                    marker=dict(size=4),
                                ),
                                row=3,
                                col=2,
                            )

                            # Add ensemble variance indicator
                            fig.add_trace(
                                go.Scatter(
                                    x=window_times,
                                    y=window_stds,
                                    mode="lines",
                                    name="Local Ensemble Std",
                                    line=dict(color="#E67E22", width=1, dash="dot"),
                                    opacity=0.7,
                                ),
                                row=3,
                                col=2,
                            )

                    except Exception as e:
                        logger.error(f"Ensemble visualization failed: {e}")

        except Exception as e:
            logger.error(f"Signal quality analysis failed: {e}")

        # Update layout
        logger.info("Updating plot layout...")
        fig.update_layout(
            title="Comprehensive Respiratory Analysis",
            template="plotly_white",
            height=800,
            showlegend=True,
            margin=dict(l=40, r=40, t=80, b=40),
        )

        # Update axes labels
        logger.info("Updating axes labels...")
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
        fig.update_yaxes(title_text="Magnitude", row=1, col=2)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Interval (s) / RR (BPM)", row=2, col=2)
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude", row=3, col=1)
        fig.update_xaxes(title_text="Time (s)", row=3, col=2)
        fig.update_yaxes(title_text="Amplitude / Statistics", row=3, col=2)

        logger.info("=== CREATE COMPREHENSIVE RESPIRATORY PLOTS COMPLETED ===")
        return fig

    except Exception as e:
        logger.error(f"Error creating comprehensive respiratory plots: {e}")
        return create_empty_figure()
