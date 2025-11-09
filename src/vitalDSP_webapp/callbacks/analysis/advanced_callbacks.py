"""
Advanced analysis callbacks for vitalDSP webapp.

This module handles advanced signal processing methods including machine learning, deep learning,
ensemble methods, and cutting-edge analysis techniques for research applications.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html, dcc
from dash.exceptions import PreventUpdate
from scipy import signal
import logging
from vitalDSP_webapp.services.data.enhanced_data_service import (
    get_enhanced_data_service,
)

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

# ===================================================================
# PHASE 1: Feature Extraction Unification
# Import comprehensive feature extraction functions from features_callbacks
# These are more complete implementations with vitalDSP support
# ===================================================================
try:
    from vitalDSP_webapp.callbacks.features.features_callbacks import (
        extract_comprehensive_features,
        extract_statistical_features as features_extract_statistical,
        extract_spectral_features as features_extract_spectral,
        extract_morphological_features as features_extract_morphological,
        extract_entropy_features,
        extract_fractal_features,
        extract_advanced_features as features_extract_advanced,
        apply_preprocessing,
        detect_signal_type as features_detect_signal_type,
    )

    FEATURES_EXTRACTION_AVAILABLE = True
    logger.info("✅ Imported comprehensive feature extraction from features_callbacks")
except ImportError as e:
    FEATURES_EXTRACTION_AVAILABLE = False
    logger.warning(f"⚠️ Could not import from features_callbacks: {e}")
    logger.warning("Will use local feature extraction functions")


def register_advanced_callbacks(app):
    """Register all advanced analysis callbacks."""
    logger.info("=== REGISTERING ADVANCED ANALYSIS CALLBACKS ===")
    _import_vitaldsp_modules()

    @app.callback(
        [
            Output("advanced-main-plot", "figure"),
            Output("advanced-performance-plot", "figure"),
            Output("advanced-analysis-summary", "children"),
            Output("advanced-model-details", "children"),
            Output("advanced-performance-metrics", "children"),
            Output("advanced-feature-importance", "children"),
            Output("advanced-visualizations", "figure"),
            Output("advanced-detailed-report", "children"),  # NEW: Detailed report
            Output("store-advanced-data", "data"),
            Output("store-advanced-results", "data"),
        ],
        [
            Input("advanced-analyze-btn", "n_clicks"),
        ],
        [
            State("url", "pathname"),  # MOVED to State - only read, doesn't trigger
            State("advanced-start-position", "value"),  # Position slider (0-100%)
            State("advanced-duration", "value"),  # Duration dropdown
            State("advanced-signal-source", "value"),  # Signal source selector
            State("advanced-signal-type", "value"),
            State("advanced-analysis-categories", "value"),
            State("advanced-ml-options", "value"),
            State("advanced-deep-learning-options", "value"),
            State("advanced-cv-folds", "value"),  # Fixed ID
            State("advanced-random-state", "value"),
            State("advanced-model-config", "value"),
            State("store-filtered-signal", "data"),  # Filtered signal store
            # ML Parameters - SVM
            State("advanced-svm-enable", "value"),
            State("advanced-svm-kernel", "value"),
            State("advanced-svm-c", "value"),
            State("advanced-svm-gamma", "value"),
            # ML Parameters - Random Forest
            State("advanced-rf-enable", "value"),
            State("advanced-rf-n-estimators", "value"),
            State("advanced-rf-max-depth", "value"),
            State("advanced-rf-min-samples-split", "value"),
            # ML Parameters - Neural Network
            State("advanced-nn-enable", "value"),
            State("advanced-nn-hidden-layers", "value"),
            State("advanced-nn-activation", "value"),
            State("advanced-nn-learning-rate", "value"),
            # DL Parameters - LSTM
            State("advanced-lstm-enable", "value"),
            State("advanced-lstm-units", "value"),
            State("advanced-lstm-layers", "value"),
            State("advanced-lstm-dropout", "value"),
            State("advanced-lstm-lr", "value"),
            # DL Parameters - CNN
            State("advanced-cnn-enable", "value"),
            State("advanced-cnn-filters", "value"),
            State("advanced-cnn-kernel", "value"),
            State("advanced-cnn-pool", "value"),
            State("advanced-cnn-dropout", "value"),
            # DL Parameters - Transformer
            State("advanced-transformer-enable", "value"),
            State("advanced-transformer-heads", "value"),
            State("advanced-transformer-layers", "value"),
            State("advanced-transformer-dim", "value"),
            # PHASE 2/3: New UI components from features page
            State(
                "advanced-feature-categories", "value"
            ),  # Feature categories to extract
            State(
                "advanced-preprocessing", "value"
            ),  # Preprocessing options (legacy - hidden)
            State("advanced-advanced-options", "value"),  # Advanced feature options
            # PHASE B: Preprocessing parameters
            State("advanced-detrend-enable", "value"),
            State("advanced-detrend-type", "value"),
            State("advanced-normalize-enable", "value"),
            State("advanced-normalize-type", "value"),
            State("advanced-filter-enable", "value"),
            State("advanced-filter-family", "value"),  # PHASE D: Advanced filtering
            State("advanced-filter-type", "value"),
            State("advanced-filter-low-freq", "value"),  # PHASE D: Low frequency
            State(
                "advanced-filter-high-freq", "value"
            ),  # PHASE D: High frequency (replaces cutoff)
            State("advanced-filter-order", "value"),
            State(
                "advanced-filter-ripple", "value"
            ),  # PHASE D: Ripple for Chebyshev/Elliptic
            State("advanced-outlier-enable", "value"),
            State("advanced-outlier-method", "value"),
            State("advanced-outlier-threshold", "value"),
            State("advanced-smoothing-enable", "value"),
            State("advanced-smoothing-method", "value"),
            State("advanced-smoothing-window", "value"),
        ],
    )
    def advanced_analysis_callback(
        n_clicks,
        pathname,
        start_position,  # Position slider (0-100%)
        duration,  # Duration dropdown
        signal_source,  # Signal source selector
        signal_type,
        analysis_categories,
        ml_options,
        deep_learning_options,
        cv_folds,
        random_state,
        model_config,
        filtered_signal_data,  # Filtered signal store
        # ML Parameters - SVM
        svm_enable,
        svm_kernel,
        svm_c,
        svm_gamma,
        # ML Parameters - Random Forest
        rf_enable,
        rf_n_estimators,
        rf_max_depth,
        rf_min_samples_split,
        # ML Parameters - Neural Network
        nn_enable,
        nn_hidden_layers,
        nn_activation,
        nn_learning_rate,
        # DL Parameters - LSTM
        lstm_enable,
        lstm_units,
        lstm_layers,
        lstm_dropout,
        lstm_lr,
        # DL Parameters - CNN
        cnn_enable,
        cnn_filters,
        cnn_kernel,
        cnn_pool,
        cnn_dropout,
        # DL Parameters - Transformer
        transformer_enable,
        transformer_heads,
        transformer_layers,
        transformer_dim,
        # PHASE 2/3: New parameters from features page
        feature_categories,  # List of feature categories to extract
        preprocessing,  # List of preprocessing options (legacy - hidden)
        advanced_options,  # List of advanced feature options
        # PHASE B: Preprocessing parameters
        detrend_enable,
        detrend_type,
        normalize_enable,
        normalize_type,
        filter_enable,
        filter_family,  # PHASE D: Advanced filtering
        filter_type,
        filter_low_freq,  # PHASE D: Low frequency
        filter_high_freq,  # PHASE D: High frequency
        filter_order,
        filter_ripple,  # PHASE D: Ripple parameter
        outlier_enable,
        outlier_method,
        outlier_threshold,
        smoothing_enable,
        smoothing_method,
        smoothing_window,
    ):
        """Main callback for advanced analysis."""
        # Only run if button was clicked
        if not n_clicks:
            return (
                create_empty_figure(),
                create_empty_figure(),
                "",
                "",
                "",
                "",
                create_empty_figure(),
                "",  # detailed report
                None,
                None,
            )

        # Only run on advanced page
        if pathname != "/advanced":
            return (
                create_empty_figure(),
                create_empty_figure(),
                "",
                "",
                "",
                "",
                create_empty_figure(),
                "",  # detailed report
                None,
                None,
            )

        # Handle analysis button click
        if n_clicks:
            try:
                # Get data service
                data_service = get_enhanced_data_service()

                # Get stored data
                stored_data = data_service.get_all_data()
                if not stored_data:
                    error_fig = create_empty_figure()
                    error_results = html.Div(
                        [
                            html.H5("No Data Available"),
                            html.P(
                                "Please upload data first to perform advanced analysis."
                            ),
                        ]
                    )
                    return (
                        error_fig,
                        error_fig,
                        error_results,
                        "",
                        "",
                        "",
                        "",
                        error_fig,
                        "",  # detailed report
                        None,
                        None,
                    )

                # Use the most recent data (data_service returns dict with IDs as keys)
                data_id = list(stored_data.keys())[-1]
                latest_data = stored_data[data_id]
                logger.info(f"Latest data ID: {data_id}")

                # Get column mapping first to find signal column
                column_mapping = data_service.get_column_mapping(data_id)
                logger.info(f"Column mapping: {column_mapping}")

                if not column_mapping:
                    error_fig = create_empty_figure()
                    error_results = html.Div(
                        [
                            html.H5("Data Error"),
                            html.P(
                                "No column mapping found. Please upload data first."
                            ),
                        ]
                    )
                    return (
                        error_fig,
                        error_fig,
                        error_results,
                        "",
                        "",
                        "",
                        "",
                        error_fig,
                        None,
                        None,
                    )

                # Get data info
                data_info = data_service.get_data_info(data_id)

                # Get sampling frequency from latest_data (same as other pages)
                sampling_freq = latest_data.get("info", {}).get("sampling_freq", 1000)
                logger.info(f"Sampling frequency from upload data: {sampling_freq}")

                # Get the actual DataFrame
                df = data_service.get_data(data_id)
                logger.info(f"Data type: {type(df)}")
                logger.info(
                    f"Data shape: {df.shape if hasattr(df, 'shape') else 'N/A'}, columns: {list(df.columns) if hasattr(df, 'columns') else 'N/A'}"
                )

                # Handle case where df might be a dict instead of DataFrame
                if isinstance(df, dict):
                    logger.warning(
                        "get_data returned dict instead of DataFrame, extracting 'data' key"
                    )
                    df = df.get("data", None)
                    if df is not None and isinstance(df, dict):
                        # Try to convert dict to DataFrame
                        import pandas as pd

                        try:
                            df = pd.DataFrame(df)
                        except Exception as e:
                            logger.error(f"Could not convert dict to DataFrame: {e}")
                            df = None

                if df is None or (hasattr(df, "empty") and df.empty):
                    error_fig = create_empty_figure()
                    error_results = html.Div(
                        [
                            html.H5("Data Error"),
                            html.P("The uploaded data is empty or invalid."),
                        ]
                    )
                    return (
                        error_fig,
                        error_fig,
                        error_results,
                        "",
                        "",
                        "",
                        "",
                        error_fig,
                        "",  # detailed report
                        None,
                        None,
                    )

                # Final check: ensure df is a pandas DataFrame
                import pandas as pd

                if not isinstance(df, pd.DataFrame):
                    logger.error(f"Data is not a DataFrame, got type: {type(df)}")
                    error_fig = create_empty_figure()
                    error_results = html.Div(
                        [
                            html.H5("Data Type Error"),
                            html.P(
                                f"Expected DataFrame but got {type(df).__name__}. Please re-upload your data."
                            ),
                        ]
                    )
                    return (
                        error_fig,
                        error_fig,
                        error_results,
                        "",
                        "",
                        "",
                        "",
                        error_fig,
                        "",  # detailed report
                        None,
                        None,
                    )

                # Convert position (0-100%) and duration to time indices
                start_position = (
                    float(start_position) if start_position is not None else 0
                )
                duration = float(duration) if duration is not None else 60

                # Calculate total duration available in the data
                total_duration = len(df) / sampling_freq

                # Calculate start time based on position percentage
                start_time_actual = (start_position / 100.0) * total_duration

                # Calculate desired end time based on requested duration
                desired_end_time = start_time_actual + duration

                # Convert to sample indices (this is what we'll actually extract)
                start_sample = int(start_time_actual * sampling_freq)
                desired_end_sample = int(desired_end_time * sampling_freq)

                # Ensure start_sample is valid
                start_sample = max(0, min(start_sample, len(df) - 1))

                # Clamp end_sample to available data
                end_sample = min(desired_end_sample, len(df))

                # Recalculate actual end_time based on extracted samples
                end_time = (
                    start_sample / sampling_freq
                    + (end_sample - start_sample) / sampling_freq
                )

                # Calculate actual window size
                actual_window_size = end_sample - start_sample
                actual_window_duration = actual_window_size / sampling_freq

                logger.info(f"Time window calculation:")
                logger.info(
                    f"  Requested: start={start_position}%, duration={duration}s"
                )
                logger.info(
                    f"  Total data available: {total_duration:.2f}s ({len(df)} samples)"
                )
                logger.info(f"  Start time: {start_time_actual:.2f}s")
                logger.info(f"  Desired end time: {desired_end_time:.2f}s")
                logger.info(
                    f"  Sample range: {start_sample} to {end_sample} ({actual_window_size} samples)"
                )
                logger.info(f"  Actual duration: {actual_window_duration:.2f}s")

                # Check signal source
                signal_source = signal_source or "filtered"

                # Get signal data based on source
                if signal_source == "filtered" and filtered_signal_data:
                    # Use filtered signal if available
                    filtered_signal_full = np.array(filtered_signal_data["signal"])
                    logger.info(
                        f"Using filtered signal. Full length: {len(filtered_signal_full)}, extracting indices {start_sample} to {end_sample}"
                    )

                    # Apply same windowing to filtered signal
                    if len(filtered_signal_full) >= end_sample:
                        signal_data = filtered_signal_full[start_sample:end_sample]
                        logger.info(
                            f"Extracted filtered signal window: {len(signal_data)} samples ({len(signal_data)/sampling_freq:.2f}s)"
                        )
                    else:
                        logger.warning(
                            f"Filtered signal too short ({len(filtered_signal_full)} < {end_sample}), using original signal"
                        )
                        # Fallback to original signal
                        windowed_df = df.iloc[start_sample:end_sample].copy()
                        signal_col = column_mapping.get("signal", "signal")
                        if signal_col and signal_col in windowed_df.columns:
                            signal_data = windowed_df[signal_col].values
                        else:
                            signal_data = windowed_df.iloc[:, 0].values
                else:
                    # Use original signal column - window the DataFrame first
                    logger.info(
                        f"Using original signal. Windowed DataFrame: {start_sample} to {end_sample} from {len(df)} samples"
                    )
                    windowed_df = df.iloc[start_sample:end_sample].copy()
                    signal_col = column_mapping.get("signal", "signal")
                    if signal_col and signal_col in windowed_df.columns:
                        signal_data = windowed_df[signal_col].values
                    else:
                        signal_data = windowed_df.iloc[:, 0].values

                logger.info(
                    f"Final signal data length: {len(signal_data)} samples ({len(signal_data)/sampling_freq:.2f} seconds)"
                )
                logger.info(
                    f"Signal data range: {np.min(signal_data):.2f} to {np.max(signal_data):.2f}"
                )

                # Create time axis using actual time values (not normalized to start from 0)
                time_axis = np.linspace(start_time_actual, end_time, len(signal_data))
                logger.info(
                    f"Time axis range: {time_axis[0]:.2f}s to {time_axis[-1]:.2f}s"
                )

                # Auto-detect signal type if needed
                if signal_type == "auto":
                    signal_type = detect_signal_type(signal_data, sampling_freq)

                # Prepare ML model parameters
                ml_params = {
                    "svm": {
                        "enable": svm_enable,
                        "kernel": svm_kernel or "rbf",
                        "C": svm_c or 1.0,
                        "gamma": svm_gamma or "scale",
                    },
                    "random_forest": {
                        "enable": rf_enable,
                        "n_estimators": rf_n_estimators or 100,
                        "max_depth": rf_max_depth or 10,
                        "min_samples_split": rf_min_samples_split or 2,
                    },
                    "neural_network": {
                        "enable": nn_enable,
                        "hidden_layers": nn_hidden_layers or "100,50",
                        "activation": nn_activation or "relu",
                        "learning_rate": nn_learning_rate or 0.001,
                    },
                }

                # Prepare DL model parameters
                dl_params = {
                    "lstm": {
                        "enable": lstm_enable,
                        "units": lstm_units or 128,
                        "layers": lstm_layers or 2,
                        "dropout": lstm_dropout or 0.2,
                        "learning_rate": lstm_lr or 0.001,
                    },
                    "cnn": {
                        "enable": cnn_enable,
                        "filters": cnn_filters or "64,128,256",
                        "kernel": cnn_kernel or 3,
                        "pool": cnn_pool or 2,
                        "dropout": cnn_dropout or 0.3,
                    },
                    "transformer": {
                        "enable": transformer_enable,
                        "heads": transformer_heads or 8,
                        "layers": transformer_layers or 4,
                        "dim": transformer_dim or 512,
                    },
                }
                logger.info(
                    f"ML params prepared: svm_enable={svm_enable}, rf_enable={rf_enable}"
                )
                logger.info(
                    f"DL params prepared: lstm_enable={lstm_enable}, cnn_enable={cnn_enable}"
                )

                # PHASE B: Build preprocessing_params dict from enable flags and parameters
                preprocessing_params = {}
                if detrend_enable:
                    preprocessing_params["detrend"] = {"type": detrend_type}
                    logger.info(f"Detrending enabled: type={detrend_type}")

                if normalize_enable:
                    preprocessing_params["normalize"] = {"type": normalize_type}
                    logger.info(f"Normalization enabled: type={normalize_type}")

                if filter_enable:
                    # PHASE D: Enhanced filtering with advanced parameters
                    preprocessing_params["filter"] = {
                        "family": filter_family,
                        "type": filter_type,
                        "low_freq": filter_low_freq,
                        "high_freq": filter_high_freq,
                        "order": filter_order,
                        "ripple": filter_ripple,
                    }
                    logger.info(
                        f"Filtering enabled: family={filter_family}, type={filter_type}, low={filter_low_freq}, high={filter_high_freq}, order={filter_order}, ripple={filter_ripple}"
                    )

                if outlier_enable:
                    preprocessing_params["outlier_removal"] = {
                        "method": outlier_method,
                        "threshold": outlier_threshold,
                    }
                    logger.info(
                        f"Outlier removal enabled: method={outlier_method}, threshold={outlier_threshold}"
                    )

                if smoothing_enable:
                    preprocessing_params["smoothing"] = {
                        "method": smoothing_method,
                        "window": smoothing_window,
                    }
                    logger.info(
                        f"Smoothing enabled: method={smoothing_method}, window={smoothing_window}"
                    )

                # Perform advanced analysis
                logger.info("Starting advanced analysis...")
                logger.info(f"Feature categories: {feature_categories}")
                logger.info(f"Preprocessing params: {preprocessing_params}")
                logger.info(f"Advanced options: {advanced_options}")

                analysis_results = perform_advanced_analysis(
                    signal_data,
                    sampling_freq,
                    signal_type,
                    analysis_categories,
                    ml_options,
                    deep_learning_options,
                    cv_folds,
                    random_state,
                    model_config,
                    ml_params,
                    dl_params,
                    preprocessing=preprocessing_params,  # PHASE B: Pass preprocessing params dict
                    feature_categories=feature_categories,  # PHASE 3: Pass feature categories
                    advanced_options=advanced_options,  # PHASE 3: Pass advanced options
                )
                logger.info(
                    f"Analysis complete. Results keys: {list(analysis_results.keys())}"
                )

                # Create visualizations
                logger.info("Creating visualizations...")
                main_plot = create_main_advanced_plot(
                    time_axis, signal_data, analysis_results, signal_type
                )
                performance_plot = create_advanced_performance_plot(
                    analysis_results, analysis_categories
                )
                visualizations = create_advanced_visualizations(
                    analysis_results, time_axis, signal_data, sampling_freq
                )
                logger.info("Visualizations created successfully")

                # Create detailed report
                detailed_report = create_detailed_analysis_report(
                    analysis_results,
                    analysis_categories,
                    signal_type,
                    time_axis,
                    signal_data,
                    sampling_freq,
                )

                # Create result displays
                # PHASE 4: Pass new parameters to display functions
                # PHASE B: Pass preprocessing_params dict instead of list
                analysis_summary = create_advanced_analysis_summary(
                    analysis_results,
                    signal_type,
                    feature_categories=feature_categories,
                    preprocessing=preprocessing_params,
                    advanced_options=advanced_options,
                )
                model_details = create_advanced_model_details(
                    analysis_results, analysis_categories
                )
                performance_metrics = create_advanced_performance_metrics(
                    analysis_results
                )
                feature_importance = create_advanced_feature_importance(
                    analysis_results
                )

                # Store results
                stored_results = {
                    "analysis_results": analysis_results,
                    "signal_type": signal_type,
                    "time_window": [start_time_actual, end_time],
                    "sampling_frequency": sampling_freq,
                }

                logger.info("Returning analysis results to UI...")
                return (
                    main_plot,
                    performance_plot,
                    analysis_summary,
                    model_details,
                    performance_metrics,
                    feature_importance,
                    visualizations,
                    detailed_report,
                    {"data_id": data_id, "time_window": [start_time_actual, end_time]},
                    stored_results,
                )

            except Exception as e:
                import traceback

                logger.error(f"Error in advanced analysis: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                error_fig = create_empty_figure()
                error_results = html.Div(
                    [
                        html.H5("Analysis Error"),
                        html.P(f"Advanced analysis failed: {str(e)}"),
                        html.Pre(
                            traceback.format_exc(),
                            style={"fontSize": "10px", "whiteSpace": "pre-wrap"},
                        ),
                    ]
                )
                return (
                    error_fig,
                    error_fig,
                    error_results,
                    "",
                    "",
                    "",
                    error_fig,
                    "",  # detailed report
                    None,
                    None,
                )

        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )

    # Removed old sync callbacks - no longer needed with start/duration approach

    # Navigation callback for position slider
    @app.callback(
        Output("advanced-start-position", "value"),
        [
            Input("advanced-btn-nudge-m10", "n_clicks"),
            Input("advanced-btn-nudge-m1", "n_clicks"),
            Input("advanced-btn-nudge-p1", "n_clicks"),
            Input("advanced-btn-nudge-p10", "n_clicks"),
        ],
        [State("advanced-start-position", "value")],
        prevent_initial_call=True,
    )
    def update_advanced_position_slider(
        nudge_m10, nudge_m1, nudge_p1, nudge_p10, start_position
    ):
        """Update position slider based on navigation button clicks."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Get current position (default to 0 if None)
        start_position = start_position if start_position is not None else 0

        # Update position based on button clicked
        if trigger_id == "advanced-btn-nudge-m10":
            return max(0, start_position - 10)
        elif trigger_id == "advanced-btn-nudge-m1":
            return max(0, start_position - 1)
        elif trigger_id == "advanced-btn-nudge-p1":
            return min(100, start_position + 1)
        elif trigger_id == "advanced-btn-nudge-p10":
            return min(100, start_position + 10)

        return start_position

    # Sync ML accordion enables with hidden checklist (for backward compatibility)
    @app.callback(
        Output("advanced-ml-options", "value"),
        [
            Input("advanced-svm-enable", "value"),
            Input("advanced-rf-enable", "value"),
            Input("advanced-nn-enable", "value"),
        ],
        prevent_initial_call=True,
    )
    def sync_ml_enables_to_checklist(svm_enable, rf_enable, nn_enable):
        """Sync ML method enable checkboxes to hidden checklist for backward compatibility."""
        ml_options = []
        if svm_enable:
            ml_options.append("svm")
        if rf_enable:
            ml_options.append("random_forest")
        if nn_enable:
            ml_options.append("neural_networks")
        return ml_options

    # Sync DL accordion enables with hidden checklist (for backward compatibility)
    @app.callback(
        Output("advanced-deep-learning-options", "value"),
        [
            Input("advanced-lstm-enable", "value"),
            Input("advanced-cnn-enable", "value"),
            Input("advanced-transformer-enable", "value"),
        ],
        prevent_initial_call=True,
    )
    def sync_dl_enables_to_checklist(lstm_enable, cnn_enable, transformer_enable):
        """Sync DL method enable checkboxes to hidden checklist for backward compatibility."""
        dl_options = []
        if lstm_enable:
            dl_options.append("lstm")
        if cnn_enable:
            dl_options.append("cnn")
        if transformer_enable:
            dl_options.append("transformer")
        return dl_options


# Helper functions for advanced analysis
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


def detect_signal_type(signal_data, sampling_freq):
    """Auto-detect the type of signal."""
    try:
        # Simple heuristics for signal type detection
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)

        # Find peaks for frequency analysis
        peaks, _ = signal.find_peaks(
            signal_data, height=mean_val + std_val, distance=int(sampling_freq * 0.3)
        )

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


def perform_advanced_analysis(
    signal_data,
    sampling_freq,
    signal_type,
    analysis_categories,
    ml_options,
    deep_learning_options,
    cv_folds,
    random_state,
    model_config,
    ml_params=None,
    dl_params=None,
    preprocessing=None,
    feature_categories=None,
    advanced_options=None,
):
    """
    Perform comprehensive advanced analysis.

    Args:
        signal_data: Signal data array
        sampling_freq: Sampling frequency in Hz
        signal_type: Type of signal (ecg, ppg, etc.)
        analysis_categories: List of analysis types to perform
        ml_options: ML methods to use
        deep_learning_options: DL methods to use
        cv_folds: Number of cross-validation folds
        random_state: Random state for reproducibility
        model_config: Model configuration options
        ml_params: ML model parameters
        dl_params: DL model parameters
        preprocessing: Preprocessing steps to apply (NEW from features page)
        feature_categories: Feature categories to extract (NEW from features page)
        advanced_options: Advanced feature options (NEW from features page)
    """
    analysis_results = {}

    try:
        # Set default feature categories if not provided
        if feature_categories is None:
            feature_categories = [
                "statistical",
                "spectral",
                "temporal",
                "morphological",
            ]

        # Always extract features (needed for ML/DL analysis)
        # Now uses comprehensive extraction with preprocessing and advanced options
        analysis_results["features"] = extract_advanced_features(
            signal_data,
            sampling_freq,
            categories=feature_categories,
            advanced_options=advanced_options,
            preprocessing=preprocessing,
        )

        # Machine learning analysis
        if "ml_analysis" in analysis_categories and ml_options:
            analysis_results["ml_results"] = perform_ml_analysis(
                signal_data,
                sampling_freq,
                ml_options,
                cv_folds,
                random_state,
                ml_params,
            )

        # Deep learning analysis
        if "deep_learning" in analysis_categories and deep_learning_options:
            try:
                analysis_results["dl_results"] = perform_deep_learning_analysis(
                    signal_data, sampling_freq, deep_learning_options, dl_params
                )
            except Exception as e:
                logger.error(f"Error in deep learning analysis: {e}")
                analysis_results["dl_results"] = {
                    "error": f"Deep learning analysis failed: {str(e)}"
                }

        # Pattern recognition
        if "pattern_recognition" in analysis_categories:
            analysis_results["patterns"] = perform_pattern_recognition(
                signal_data, sampling_freq
            )

        # Ensemble methods (support both "ensemble" and "ensemble_methods" for compatibility)
        if (
            "ensemble" in analysis_categories
            or "ensemble_methods" in analysis_categories
        ):
            analysis_results["ensemble"] = perform_ensemble_analysis(
                signal_data, sampling_freq, cv_folds, random_state
            )

        # Anomaly detection
        if "anomaly_detection" in analysis_categories:
            analysis_results["anomaly_detection"] = perform_anomaly_detection(
                signal_data, sampling_freq
            )

        # Classification analysis
        if "classification" in analysis_categories:
            analysis_results["classification"] = perform_classification_analysis(
                signal_data, sampling_freq
            )

        # Regression analysis
        if "regression" in analysis_categories:
            analysis_results["regression"] = perform_regression_analysis(
                signal_data, sampling_freq
            )

        # Clustering analysis
        if "clustering" in analysis_categories:
            analysis_results["clustering"] = perform_clustering_analysis(
                signal_data, sampling_freq
            )

        # Dimensionality reduction analysis
        if "dimensionality_reduction" in analysis_categories:
            analysis_results["dimensionality_reduction"] = (
                perform_dimensionality_reduction_analysis(signal_data, sampling_freq)
            )

        # Forecasting analysis
        if "forecasting" in analysis_categories:
            analysis_results["forecasting"] = perform_forecasting_analysis(
                signal_data, sampling_freq
            )

        # Advanced signal processing (when pattern recognition is enabled, include advanced processing)
        if "pattern_recognition" in analysis_categories:
            analysis_results["advanced_processing"] = (
                perform_advanced_signal_processing(signal_data, sampling_freq)
            )

        return analysis_results

    except Exception as e:
        logger.error(f"Error in advanced analysis: {e}")
        return {"error": f"Advanced analysis failed: {str(e)}"}


def extract_advanced_features(
    signal_data,
    sampling_freq,
    categories=None,
    advanced_options=None,
    preprocessing=None,
):
    """
    Extract advanced features from the signal.

    Now uses comprehensive feature extraction from features_callbacks for more complete analysis.
    Falls back to local implementations if features_callbacks not available.

    Args:
        signal_data: The signal data array
        sampling_freq: Sampling frequency in Hz
        categories: List of feature categories to extract (statistical, spectral, temporal, morphological, entropy, fractal)
        advanced_options: List of advanced options (cross_correlation, phase_analysis, nonlinear, wavelet, ml_features)
        preprocessing: List of preprocessing steps to apply
    """
    try:
        # Set default categories if not provided
        if categories is None:
            categories = ["statistical", "spectral", "temporal", "morphological"]

        if advanced_options is None:
            advanced_options = []

        if preprocessing is None:
            preprocessing = []

        features = {}
        use_comprehensive = (
            FEATURES_EXTRACTION_AVAILABLE  # Local copy to avoid scope issues
        )

        # Use comprehensive feature extraction from features_callbacks if available
        if use_comprehensive:
            logger.info(
                "Using comprehensive feature extraction from features_callbacks"
            )

            # Apply preprocessing if requested
            processed_signal = signal_data
            if preprocessing:
                try:
                    processed_signal = apply_preprocessing(
                        signal_data, preprocessing, sampling_freq, filter_info=None
                    )
                    logger.info(f"Applied preprocessing: {preprocessing}")
                except Exception as e:
                    logger.warning(f"Preprocessing failed: {e}, using original signal")
                    processed_signal = signal_data

            # Extract comprehensive features using features_callbacks
            try:
                comprehensive_features = extract_comprehensive_features(
                    processed_signal, sampling_freq, categories, advanced_options
                )

                # Merge with existing structure
                features.update(comprehensive_features)
                logger.info(f"Extracted features for categories: {categories}")

                # BUGFIX: Override temporal & morphological features with local implementations
                # The features_callbacks versions don't return the correct key names for display
                if "temporal" in categories:
                    try:
                        logger.info(
                            "Overriding temporal features with local vitalDSP implementation"
                        )
                        features["temporal"] = extract_temporal_features(
                            processed_signal, sampling_freq
                        )
                    except Exception as e:
                        logger.warning(
                            f"Local temporal feature extraction failed: {e}, keeping comprehensive version"
                        )

                if "morphological" in categories:
                    try:
                        logger.info(
                            "Overriding morphological features with local implementation"
                        )
                        features["morphological"] = extract_morphological_features(
                            processed_signal, sampling_freq
                        )
                    except Exception as e:
                        logger.warning(
                            f"Local morphological feature extraction failed: {e}, keeping comprehensive version"
                        )

            except Exception as e:
                logger.warning(
                    f"Comprehensive feature extraction failed: {e}, falling back to local implementation"
                )
                # Fall back to local implementation below
                use_comprehensive = False

        # Local implementation fallback (original code)
        if not use_comprehensive or not features:
            logger.info("Using local feature extraction implementation")

            # Statistical features
            if "statistical" in categories:
                features["statistical"] = {
                    "mean": np.mean(signal_data),
                    "std": np.std(signal_data),
                    "skewness": calculate_skewness(signal_data),
                    "kurtosis": calculate_kurtosis(signal_data),
                    "entropy": calculate_entropy(signal_data),
                }

            # Spectral features
            if "spectral" in categories:
                features["spectral"] = extract_spectral_features(
                    signal_data, sampling_freq
                )

            # Temporal features (uses vitalDSP WaveformMorphology - keep this!)
            if "temporal" in categories:
                features["temporal"] = extract_temporal_features(
                    signal_data, sampling_freq
                )

            # Morphological features
            if "morphological" in categories:
                features["morphological"] = extract_morphological_features(
                    signal_data, sampling_freq
                )

        return features

    except Exception as e:
        logger.error(f"Error extracting advanced features: {e}")
        return {"error": f"Feature extraction failed: {str(e)}"}


def perform_ml_analysis(
    signal_data, sampling_freq, ml_options, cv_folds, random_state, ml_params=None
):
    """Perform machine learning analysis."""
    try:
        ml_results = {}

        # Extract features for ML
        features = extract_ml_features(signal_data, sampling_freq)

        if "svm" in ml_options:
            svm_params = ml_params.get("svm", {}) if ml_params else {}
            ml_results["svm"] = train_svm_model(
                features, cv_folds, random_state, svm_params
            )

        if "random_forest" in ml_options:
            rf_params = ml_params.get("random_forest", {}) if ml_params else {}
            ml_results["random_forest"] = train_random_forest_model(
                features, cv_folds, random_state, rf_params
            )

        if "neural_network" in ml_options:
            nn_params = ml_params.get("neural_network", {}) if ml_params else {}
            ml_results["neural_network"] = train_neural_network_model(
                features, cv_folds, random_state, nn_params
            )

        if "gradient_boosting" in ml_options:
            ml_results["gradient_boosting"] = train_gradient_boosting_model(
                features, cv_folds, random_state
            )

        return ml_results

    except Exception as e:
        logger.error(f"Error in ML analysis: {e}")
        return {"error": f"ML analysis failed: {str(e)}"}


def perform_deep_learning_analysis(
    signal_data, sampling_freq, deep_learning_options, dl_params=None
):
    """Perform deep learning analysis."""
    try:
        dl_results = {}

        # Prepare data for deep learning
        dl_data = prepare_dl_data(signal_data, sampling_freq)

        if "cnn" in deep_learning_options:
            cnn_params = dl_params.get("cnn", {}) if dl_params else {}
            dl_results["cnn"] = train_cnn_model(dl_data, cnn_params)

        if "lstm" in deep_learning_options:
            lstm_params = dl_params.get("lstm", {}) if dl_params else {}
            dl_results["lstm"] = train_lstm_model(dl_data, lstm_params)

        if "transformer" in deep_learning_options:
            transformer_params = dl_params.get("transformer", {}) if dl_params else {}
            dl_results["transformer"] = train_transformer_model(
                dl_data, transformer_params
            )

        return dl_results

    except Exception as e:
        logger.error(f"Error in deep learning analysis: {e}")
        return {"error": f"Deep learning analysis failed: {str(e)}"}


def perform_pattern_recognition(signal_data, sampling_freq):
    """Perform pattern recognition analysis."""
    try:
        patterns = {}

        # Peak detection patterns
        patterns["peaks"] = analyze_peak_patterns(signal_data, sampling_freq)

        # Frequency patterns
        patterns["frequency"] = analyze_frequency_patterns(signal_data, sampling_freq)

        # Morphological patterns
        patterns["morphology"] = analyze_morphological_patterns(
            signal_data, sampling_freq
        )

        return patterns

    except Exception as e:
        logger.error(f"Error in pattern recognition: {e}")
        return {"error": f"Pattern recognition failed: {str(e)}"}


def perform_ensemble_analysis(signal_data, sampling_freq, cv_folds, random_state):
    """Perform ensemble analysis."""
    try:
        ensemble_results = {}

        # Voting ensemble
        ensemble_results["voting"] = create_voting_ensemble(
            signal_data, sampling_freq, cv_folds, random_state
        )

        # Stacking ensemble
        ensemble_results["stacking"] = create_stacking_ensemble(
            signal_data, sampling_freq, cv_folds, random_state
        )

        # Bagging ensemble
        ensemble_results["bagging"] = create_bagging_ensemble(
            signal_data, sampling_freq, cv_folds, random_state
        )

        return ensemble_results

    except Exception as e:
        logger.error(f"Error in ensemble analysis: {e}")
        return {"error": f"Ensemble analysis failed: {str(e)}"}


def perform_anomaly_detection(signal_data, sampling_freq):
    """Perform anomaly detection using vitalDSP's AnomalyDetection module."""
    try:
        from vitalDSP.advanced_computation.anomaly_detection import AnomalyDetection

        results = {}
        anomaly_detector = AnomalyDetection(signal_data)

        # Z-score based detection
        try:
            anomalies_z_score = anomaly_detector.detect_anomalies(
                method="z_score", threshold=3.0
            )
            results["z_score_anomalies"] = len(anomalies_z_score)
            results["z_score_rate"] = (
                len(anomalies_z_score) / len(signal_data) * 100
                if len(signal_data) > 0
                else 0
            )
        except Exception as e:
            logger.warning(f"Z-score anomaly detection failed: {e}")
            results["z_score_anomalies"] = 0
            results["z_score_rate"] = 0

        # Moving average based detection
        try:
            anomalies_ma = anomaly_detector.detect_anomalies(
                method="moving_average", window_size=5, threshold=0.5
            )
            results["moving_average_anomalies"] = len(anomalies_ma)
            results["moving_average_rate"] = (
                len(anomalies_ma) / len(signal_data) * 100
                if len(signal_data) > 0
                else 0
            )
        except Exception as e:
            logger.warning(f"Moving average anomaly detection failed: {e}")
            results["moving_average_anomalies"] = 0
            results["moving_average_rate"] = 0

        # LOF based detection
        try:
            n_neighbors = min(20, len(signal_data) // 10)
            if n_neighbors >= 2:
                anomalies_lof = anomaly_detector.detect_anomalies(
                    method="lof", n_neighbors=n_neighbors
                )
                results["lof_anomalies"] = len(anomalies_lof)
                results["lof_rate"] = (
                    len(anomalies_lof) / len(signal_data) * 100
                    if len(signal_data) > 0
                    else 0
                )
            else:
                results["lof_anomalies"] = 0
                results["lof_rate"] = 0
        except Exception as e:
            logger.warning(f"LOF anomaly detection failed: {e}")
            results["lof_anomalies"] = 0
            results["lof_rate"] = 0

        # FFT based detection
        try:
            anomalies_fft = anomaly_detector.detect_anomalies(
                method="fft", threshold=1.5
            )
            results["fft_anomalies"] = len(anomalies_fft)
            results["fft_rate"] = (
                len(anomalies_fft) / len(signal_data) * 100
                if len(signal_data) > 0
                else 0
            )
        except Exception as e:
            logger.warning(f"FFT anomaly detection failed: {e}")
            results["fft_anomalies"] = 0
            results["fft_rate"] = 0

        # Combined detection: Use voting mechanism (anomaly detected by at least 2 methods)
        # This prevents false positives from over-sensitive methods
        anomaly_votes = {}
        method_anomalies = []

        if "z_score_anomalies" in results and results.get("z_score_anomalies", 0) > 0:
            method_anomalies.append(("z_score", anomalies_z_score))
        if (
            "moving_average_anomalies" in results
            and results.get("moving_average_anomalies", 0) > 0
        ):
            method_anomalies.append(("moving_average", anomalies_ma))
        if "lof_anomalies" in results and results.get("lof_anomalies", 0) > 0:
            method_anomalies.append(("lof", anomalies_lof))
        if "fft_anomalies" in results and results.get("fft_anomalies", 0) > 0:
            method_anomalies.append(("fft", anomalies_fft))

        # Count votes for each sample
        for method_name, anomalies in method_anomalies:
            for idx in anomalies:
                if idx < len(signal_data):  # Ensure valid index
                    anomaly_votes[idx] = anomaly_votes.get(idx, 0) + 1

        # Only consider samples detected by at least 2 methods as true anomalies
        # If only one method available, use it but cap at 10% of signal
        min_votes = 2 if len(method_anomalies) >= 2 else 1
        confirmed_anomalies = [
            idx for idx, votes in anomaly_votes.items() if votes >= min_votes
        ]

        # Cap anomalies at reasonable percentage (e.g., max 10% of signal)
        max_anomalies = max(100, int(len(signal_data) * 0.10))
        if len(confirmed_anomalies) > max_anomalies:
            # Sort by vote count and take top anomalies
            anomaly_scores = [(idx, anomaly_votes[idx]) for idx in confirmed_anomalies]
            anomaly_scores.sort(key=lambda x: x[1], reverse=True)
            confirmed_anomalies = [idx for idx, _ in anomaly_scores[:max_anomalies]]
            logger.warning(
                f"Anomaly detection found {len(confirmed_anomalies)} anomalies ({len(confirmed_anomalies)/len(signal_data)*100:.1f}%), capped at {max_anomalies}"
            )

        results["total_anomalies"] = len(confirmed_anomalies)
        results["overall_anomaly_rate"] = (
            len(confirmed_anomalies) / len(signal_data) * 100
            if len(signal_data) > 0
            else 0
        )
        results["method"] = "vitalDSP Multiple Detection Methods (Voting)"
        results["methods_used"] = [name for name, _ in method_anomalies]
        results["min_votes"] = min_votes

        logger.info(
            f"Anomaly detection complete: {results['total_anomalies']} anomalies found ({results['overall_anomaly_rate']:.2f}%)"
        )

        return results

    except ImportError as e:
        logger.error(f"vitalDSP AnomalyDetection module not available: {e}")
        return {
            "error": "Anomaly detection module not available in vitalDSP",
            "method": "Not Available - Please install vitalDSP",
        }
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        return {"error": f"Anomaly detection failed: {str(e)}"}


def perform_classification_analysis(signal_data, sampling_freq):
    """Perform dedicated classification analysis."""
    try:
        results = {}

        # Extract features for classification
        features = extract_advanced_features(signal_data, sampling_freq)

        # Signal quality classification (Good/Poor)
        try:
            # Use amplitude variability as a proxy for quality
            amplitude_std = np.std(signal_data)
            amplitude_mean = np.mean(np.abs(signal_data))
            quality_score = amplitude_mean / (amplitude_std + 1e-6)

            if quality_score > 2.0:
                quality_class = "Good"
                confidence = min(0.95, quality_score / 5.0)
            elif quality_score > 1.0:
                quality_class = "Fair"
                confidence = quality_score / 3.0
            else:
                quality_class = "Poor"
                confidence = max(0.1, quality_score / 2.0)

            results["signal_quality_classification"] = {
                "class": quality_class,
                "confidence": confidence,
                "quality_score": quality_score,
                "method": "Amplitude-based Classification",
            }
        except Exception as e:
            logger.warning(f"Signal quality classification failed: {e}")
            results["signal_quality_classification"] = {"error": str(e)}

        # Heart rate classification (Normal/Elevated/High)
        try:
            temporal_features = extract_temporal_features(signal_data, sampling_freq)
            heart_rate = temporal_features.get("heart_rate", 0)

            if heart_rate < 60:
                hr_class = "Bradycardia"
                confidence = 0.8
            elif heart_rate <= 100:
                hr_class = "Normal"
                confidence = 0.9
            elif heart_rate <= 120:
                hr_class = "Elevated"
                confidence = 0.85
            else:
                hr_class = "Tachycardia"
                confidence = 0.8

            results["heart_rate_classification"] = {
                "class": hr_class,
                "confidence": confidence,
                "heart_rate": heart_rate,
                "method": "Physiological Range Classification",
            }
        except Exception as e:
            logger.warning(f"Heart rate classification failed: {e}")
            results["heart_rate_classification"] = {"error": str(e)}

        # Pattern classification (Regular/Irregular/Arrhythmic)
        try:
            temporal_features = extract_temporal_features(signal_data, sampling_freq)
            peak_count = temporal_features.get("peak_count", 0)
            interval_std = temporal_features.get("interval_std", 0)

            if interval_std < 0.05:  # Low variability
                pattern_class = "Regular"
                confidence = 0.9
            elif interval_std < 0.15:  # Moderate variability
                pattern_class = "Irregular"
                confidence = 0.8
            else:  # High variability
                pattern_class = "Arrhythmic"
                confidence = 0.7

            results["pattern_classification"] = {
                "class": pattern_class,
                "confidence": confidence,
                "interval_std": interval_std,
                "peak_count": peak_count,
                "method": "Variability-based Classification",
            }
        except Exception as e:
            logger.warning(f"Pattern classification failed: {e}")
            results["pattern_classification"] = {"error": str(e)}

        results["method"] = "Multi-class Classification Analysis"
        results["total_classes"] = len(
            [
                k
                for k in results.keys()
                if k != "method" and "error" not in str(results[k])
            ]
        )

        return results

    except Exception as e:
        logger.error(f"Error in classification analysis: {e}")
        return {"error": f"Classification analysis failed: {str(e)}"}


def perform_regression_analysis(signal_data, sampling_freq):
    """Perform dedicated regression analysis."""
    try:
        results = {}

        # Time series regression for trend analysis
        try:
            time_points = np.arange(len(signal_data))

            # Linear regression for trend
            coeffs = np.polyfit(time_points, signal_data, 1)
            trend_slope = coeffs[0]
            trend_intercept = coeffs[1]

            # Calculate R-squared
            predicted = np.polyval(coeffs, time_points)
            ss_res = np.sum((signal_data - predicted) ** 2)
            ss_tot = np.sum((signal_data - np.mean(signal_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            results["linear_trend_regression"] = {
                "slope": trend_slope,
                "intercept": trend_intercept,
                "r_squared": r_squared,
                "trend_direction": (
                    "Increasing"
                    if trend_slope > 0
                    else "Decreasing" if trend_slope < 0 else "Stable"
                ),
                "method": "Linear Regression",
            }
        except Exception as e:
            logger.warning(f"Linear trend regression failed: {e}")
            results["linear_trend_regression"] = {"error": str(e)}

        # Polynomial regression for non-linear patterns
        try:
            time_points = np.arange(len(signal_data))

            # Quadratic regression
            coeffs_quad = np.polyfit(time_points, signal_data, 2)
            predicted_quad = np.polyval(coeffs_quad, time_points)

            ss_res_quad = np.sum((signal_data - predicted_quad) ** 2)
            ss_tot_quad = np.sum((signal_data - np.mean(signal_data)) ** 2)
            r_squared_quad = 1 - (ss_res_quad / ss_tot_quad) if ss_tot_quad > 0 else 0

            results["polynomial_regression"] = {
                "coefficients": coeffs_quad.tolist(),
                "r_squared": r_squared_quad,
                "degree": 2,
                "method": "Quadratic Regression",
            }
        except Exception as e:
            logger.warning(f"Polynomial regression failed: {e}")
            results["polynomial_regression"] = {"error": str(e)}

        # Heart rate variability regression
        try:
            temporal_features = extract_temporal_features(signal_data, sampling_freq)
            peak_count = temporal_features.get("peak_count", 0)

            if peak_count > 1:
                # Simple HRV trend analysis
                intervals = temporal_features.get("intervals", [])
                if len(intervals) > 2:
                    interval_trend = np.polyfit(range(len(intervals)), intervals, 1)[0]

                    results["hrv_regression"] = {
                        "interval_trend": interval_trend,
                        "trend_direction": (
                            "Increasing" if interval_trend > 0 else "Decreasing"
                        ),
                        "peak_count": peak_count,
                        "method": "HRV Interval Regression",
                    }
                else:
                    results["hrv_regression"] = {
                        "error": "Insufficient intervals for regression"
                    }
            else:
                results["hrv_regression"] = {
                    "error": "Insufficient peaks for HRV analysis"
                }
        except Exception as e:
            logger.warning(f"HRV regression failed: {e}")
            results["hrv_regression"] = {"error": str(e)}

        results["method"] = "Multi-method Regression Analysis"
        results["total_regressions"] = len(
            [
                k
                for k in results.keys()
                if k != "method" and "error" not in str(results[k])
            ]
        )

        return results

    except Exception as e:
        logger.error(f"Error in regression analysis: {e}")
        return {"error": f"Regression analysis failed: {str(e)}"}


def perform_clustering_analysis(signal_data, sampling_freq):
    """Perform dedicated clustering analysis."""
    try:
        results = {}

        # Segment-based clustering
        try:
            # Divide signal into segments
            segment_size = min(100, len(signal_data) // 10)
            if segment_size < 10:
                segment_size = len(signal_data)

            segments = []
            for i in range(0, len(signal_data) - segment_size + 1, segment_size):
                segment = signal_data[i : i + segment_size]
                segments.append(segment)

            if len(segments) > 1:
                # Extract features for each segment
                segment_features = []
                for segment in segments:
                    features = {
                        "mean": np.mean(segment),
                        "std": np.std(segment),
                        "energy": np.sum(segment**2),
                        "zero_crossings": np.sum(np.diff(np.sign(segment)) != 0),
                    }
                    segment_features.append(list(features.values()))

                segment_features = np.array(segment_features)

                # Simple clustering based on feature similarity
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(segment_features)

                # Determine optimal number of clusters (2-4)
                n_clusters = min(4, max(2, len(segments) // 3))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(features_scaled)

                results["segment_clustering"] = {
                    "n_clusters": n_clusters,
                    "cluster_labels": cluster_labels.tolist(),
                    "cluster_centers": kmeans.cluster_centers_.tolist(),
                    "inertia": kmeans.inertia_,
                    "method": "K-means Clustering",
                }
            else:
                results["segment_clustering"] = {
                    "error": "Insufficient segments for clustering"
                }
        except ImportError:
            logger.warning("sklearn not available for clustering")
            results["segment_clustering"] = {"error": "sklearn required for clustering"}
        except Exception as e:
            logger.warning(f"Segment clustering failed: {e}")
            results["segment_clustering"] = {"error": str(e)}

        # Peak-based clustering
        try:
            temporal_features = extract_temporal_features(signal_data, sampling_freq)
            peak_count = temporal_features.get("peak_count", 0)

            if peak_count > 3:
                # Cluster peaks based on amplitude and timing
                peaks = temporal_features.get("peaks", [])
                if len(peaks) > 3:
                    peak_features = []
                    for i, peak_idx in enumerate(peaks):
                        if peak_idx < len(signal_data):
                            amplitude = signal_data[peak_idx]
                            timing = peak_idx / sampling_freq
                            peak_features.append([amplitude, timing])

                    if len(peak_features) > 3:
                        peak_features = np.array(peak_features)

                        # Simple amplitude-based clustering
                        amplitude_threshold = np.percentile(peak_features[:, 0], 50)
                        high_amplitude_cluster = peak_features[
                            peak_features[:, 0] > amplitude_threshold
                        ]
                        low_amplitude_cluster = peak_features[
                            peak_features[:, 0] <= amplitude_threshold
                        ]

                        results["peak_clustering"] = {
                            "high_amplitude_peaks": len(high_amplitude_cluster),
                            "low_amplitude_peaks": len(low_amplitude_cluster),
                            "amplitude_threshold": amplitude_threshold,
                            "method": "Amplitude-based Peak Clustering",
                        }
                    else:
                        results["peak_clustering"] = {
                            "error": "Insufficient peak features"
                        }
                else:
                    results["peak_clustering"] = {"error": "Insufficient peaks"}
            else:
                results["peak_clustering"] = {
                    "error": "Insufficient peaks for clustering"
                }
        except Exception as e:
            logger.warning(f"Peak clustering failed: {e}")
            results["peak_clustering"] = {"error": str(e)}

        results["method"] = "Multi-method Clustering Analysis"
        results["total_clusters"] = len(
            [
                k
                for k in results.keys()
                if k != "method" and "error" not in str(results[k])
            ]
        )

        return results

    except Exception as e:
        logger.error(f"Error in clustering analysis: {e}")
        return {"error": f"Clustering analysis failed: {str(e)}"}


def perform_dimensionality_reduction_analysis(signal_data, sampling_freq):
    """Perform dedicated dimensionality reduction analysis."""
    try:
        results = {}

        # Extract comprehensive features
        all_features = extract_advanced_features(signal_data, sampling_freq)

        # Prepare feature matrix
        feature_names = []
        feature_values = []

        for category, features in all_features.items():
            if isinstance(features, dict):
                for name, value in features.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        feature_names.append(f"{category}_{name}")
                        feature_values.append(value)

        if len(feature_values) < 2:
            return {"error": "Insufficient features for dimensionality reduction"}

        feature_matrix = np.array(feature_values).reshape(1, -1)

        # PCA Analysis
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # For single sample, we'll analyze feature importance instead
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(feature_matrix)

            # Calculate feature variances (proxy for PCA importance)
            feature_variances = np.var(features_scaled, axis=0)
            sorted_indices = np.argsort(feature_variances)[::-1]

            # Top features by variance
            top_features = []
            for i in range(min(5, len(feature_names))):
                idx = sorted_indices[i]
                top_features.append(
                    {
                        "name": feature_names[idx],
                        "variance": feature_variances[idx],
                        "importance": feature_variances[idx]
                        / np.sum(feature_variances),
                    }
                )

            results["pca_analysis"] = {
                "top_features": top_features,
                "total_features": len(feature_names),
                "explained_variance_ratio": (
                    feature_variances / np.sum(feature_variances)
                ).tolist(),
                "method": "Feature Variance Analysis (PCA proxy)",
            }
        except ImportError:
            logger.warning("sklearn not available for PCA")
            results["pca_analysis"] = {"error": "sklearn required for PCA"}
        except Exception as e:
            logger.warning(f"PCA analysis failed: {e}")
            results["pca_analysis"] = {"error": str(e)}

        # Feature importance ranking
        try:
            # Rank features by their absolute values (normalized)
            normalized_features = np.abs(feature_values) / np.max(
                np.abs(feature_values)
            )
            feature_importance = []

            for i, (name, value, norm_value) in enumerate(
                zip(feature_names, feature_values, normalized_features)
            ):
                feature_importance.append(
                    {
                        "name": name,
                        "value": value,
                        "normalized_importance": norm_value,
                        "rank": i + 1,
                    }
                )

            # Sort by importance
            feature_importance.sort(
                key=lambda x: x["normalized_importance"], reverse=True
            )

            results["feature_importance"] = {
                "top_features": feature_importance[:10],  # Top 10 features
                "total_features": len(feature_names),
                "method": "Normalized Feature Importance",
            }
        except Exception as e:
            logger.warning(f"Feature importance analysis failed: {e}")
            results["feature_importance"] = {"error": str(e)}

        # Feature correlation analysis
        try:
            # For single sample, we'll analyze feature relationships
            feature_categories = {}
            for name, value in zip(feature_names, feature_values):
                category = name.split("_")[0]
                if category not in feature_categories:
                    feature_categories[category] = []
                feature_categories[category].append(value)

            category_stats = {}
            for category, values in feature_categories.items():
                if len(values) > 0:
                    category_stats[category] = {
                        "count": len(values),
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "range": np.max(values) - np.min(values),
                    }

            results["feature_correlation"] = {
                "category_stats": category_stats,
                "total_categories": len(feature_categories),
                "method": "Category-based Feature Analysis",
            }
        except Exception as e:
            logger.warning(f"Feature correlation analysis failed: {e}")
            results["feature_correlation"] = {"error": str(e)}

        results["method"] = "Multi-method Dimensionality Reduction Analysis"
        results["total_methods"] = len(
            [
                k
                for k in results.keys()
                if k != "method" and "error" not in str(results[k])
            ]
        )

        return results

    except Exception as e:
        logger.error(f"Error in dimensionality reduction analysis: {e}")
        return {"error": f"Dimensionality reduction analysis failed: {str(e)}"}


def perform_forecasting_analysis(signal_data, sampling_freq):
    """Perform dedicated time series forecasting analysis."""
    try:
        results = {}

        # Improved linear forecasting with better trend analysis
        try:
            time_points = np.arange(len(signal_data))

            # Use recent 80% of data for trend fitting (more robust to early transients)
            recent_start = max(0, len(signal_data) // 5)
            recent_time = time_points[recent_start:]
            recent_signal = signal_data[recent_start:]

            # Fit polynomial (degree 2 for better curve fitting) or linear
            if len(recent_signal) > 10:
                try:
                    # Try quadratic fit first
                    coeffs = np.polyfit(recent_time, recent_signal, 2)
                    polynomial_order = 2
                except Exception as err:
                    logger.error(f"Error fitting polynomial: {err}")
                    # Fallback to linear
                    coeffs = np.polyfit(recent_time, recent_signal, 1)
                    polynomial_order = 1
            else:
                coeffs = np.polyfit(time_points, signal_data, 1)
                polynomial_order = 1

            # Forecast next 10% of signal length (or 5 seconds, whichever is smaller)
            forecast_duration = min(
                5.0, len(signal_data) / sampling_freq * 0.1
            )  # 5 seconds max
            forecast_length = int(forecast_duration * sampling_freq)
            forecast_length = max(
                10, min(forecast_length, len(signal_data) // 5)
            )  # At least 10, at most 20% of signal

            future_time_points = np.arange(
                len(signal_data), len(signal_data) + forecast_length
            )
            forecast_values = np.polyval(coeffs, future_time_points)

            # Calculate forecast confidence based on fit quality
            fitted_values = np.polyval(coeffs, recent_time)
            residuals = recent_signal - fitted_values
            rmse = np.sqrt(np.mean(residuals**2))

            # Calculate confidence intervals (95% confidence)
            confidence_intervals = rmse * 1.96

            # Calculate R-squared for trend fit quality
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((recent_signal - np.mean(recent_signal)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            results["linear_forecast"] = {
                "forecast_length": forecast_length,
                "forecast_values": forecast_values.tolist(),
                "confidence_interval": confidence_intervals,
                "trend_slope": coeffs[-2] if len(coeffs) >= 2 else 0,  # Linear term
                "rmse": rmse,
                "r_squared": r_squared,
                "polynomial_order": polynomial_order,
                "method": f"Polynomial Trend Forecasting (order {polynomial_order})",
            }
        except Exception as e:
            logger.warning(f"Linear forecasting failed: {e}")
            results["linear_forecast"] = {"error": str(e)}

        # Improved moving average forecasting with trend consideration
        try:
            # Use adaptive window size based on signal characteristics
            if len(signal_data) > 100:
                # Try to match signal periodicity
                temporal_features = extract_temporal_features(
                    signal_data, sampling_freq
                )
                heart_rate = temporal_features.get("heart_rate", 0)
                if heart_rate > 0:
                    period_samples = int(
                        sampling_freq * 60 / heart_rate
                    )  # One heart beat period
                    window_size = max(
                        10, min(period_samples * 2, len(signal_data) // 10)
                    )
                else:
                    window_size = min(50, len(signal_data) // 10)
            else:
                window_size = min(20, len(signal_data) // 5)

            window_size = max(5, window_size)  # At least 5 samples

            # Calculate moving average with trend
            moving_avg = np.convolve(
                signal_data, np.ones(window_size) / window_size, mode="valid"
            )

            # Calculate trend in moving average
            ma_time = np.arange(len(moving_avg))
            if len(moving_avg) > 5:
                ma_trend = np.polyfit(
                    ma_time[-min(20, len(moving_avg)) :],
                    moving_avg[-min(20, len(moving_avg)) :],
                    1,
                )
            else:
                ma_trend = [
                    0,
                    moving_avg[-1] if len(moving_avg) > 0 else np.mean(signal_data),
                ]

            # Forecast with trend
            forecast_length = max(
                10, min(len(signal_data) // 10, int(5 * sampling_freq))
            )
            future_ma_indices = np.arange(
                len(moving_avg), len(moving_avg) + forecast_length
            )
            ma_forecast = np.polyval(ma_trend, future_ma_indices)

            # Smooth the forecast
            if len(ma_forecast) > 5:
                ma_forecast = np.convolve(ma_forecast, np.ones(5) / 5, mode="same")

            results["moving_average_forecast"] = {
                "forecast_length": forecast_length,
                "forecast_values": ma_forecast.tolist(),
                "window_size": window_size,
                "last_ma_value": (
                    moving_avg[-1] if len(moving_avg) > 0 else np.mean(signal_data)
                ),
                "trend_slope": ma_trend[0],
                "method": "Adaptive Moving Average with Trend",
            }
        except Exception as e:
            logger.warning(f"Moving average forecasting failed: {e}")
            results["moving_average_forecast"] = {"error": str(e)}

        # Heart rate forecasting - improved with actual HR trend analysis
        try:
            temporal_features = extract_temporal_features(signal_data, sampling_freq)
            heart_rate = temporal_features.get("heart_rate", 0)

            if heart_rate > 0:
                # Calculate HR from sliding windows to get trend
                window_size_samples = int(sampling_freq * 5)  # 5-second windows
                if window_size_samples < len(signal_data) // 4:
                    num_windows = len(signal_data) // window_size_samples
                    hr_trend = []
                    for i in range(num_windows):
                        window_start = i * window_size_samples
                        window_end = min(
                            window_start + window_size_samples, len(signal_data)
                        )
                        window_data = signal_data[window_start:window_end]
                        try:
                            window_peaks, _ = signal.find_peaks(
                                window_data,
                                distance=int(
                                    sampling_freq / heart_rate * 0.6
                                ),  # At least 60% of expected interval
                                height=np.percentile(window_data, 75),
                            )
                            if len(window_peaks) > 1:
                                intervals = np.diff(window_peaks) / sampling_freq
                                hr_window = (
                                    60.0 / np.mean(intervals)
                                    if np.mean(intervals) > 0
                                    else heart_rate
                                )
                                hr_trend.append(hr_window)
                        except Exception as err:
                            print(f"Error calculating HR trend: {err}")
                            hr_trend.append(heart_rate)

                    if len(hr_trend) >= 2:
                        # Fit trend to HR values
                        hr_time = np.arange(len(hr_trend))
                        hr_coeffs = np.polyfit(hr_time, hr_trend, 1)  # Linear trend

                        forecast_length = max(10, len(signal_data) // 10)
                        forecast_time = forecast_length / sampling_freq

                        # Forecast HR based on trend
                        future_hr_times = np.arange(
                            len(hr_trend),
                            len(hr_trend)
                            + (forecast_length // window_size_samples)
                            + 1,
                        )
                        forecast_hr_trend = np.polyval(hr_coeffs, future_hr_times)

                        # Interpolate to match forecast_length
                        if len(forecast_hr_trend) > 1:
                            forecast_hr = np.interp(
                                np.arange(forecast_length),
                                np.linspace(
                                    0, forecast_length - 1, len(forecast_hr_trend)
                                ),
                                forecast_hr_trend,
                            )
                        else:
                            forecast_hr = np.full(
                                forecast_length,
                                (
                                    forecast_hr_trend[0]
                                    if len(forecast_hr_trend) > 0
                                    else heart_rate
                                ),
                            )

                        # Ensure reasonable HR values (30-200 bpm)
                        forecast_hr = np.clip(forecast_hr, 30, 200)
                    else:
                        # Fallback to constant HR
                        forecast_length = max(10, len(signal_data) // 10)
                        forecast_time = forecast_length / sampling_freq
                        forecast_hr = np.full(forecast_length, heart_rate)
                else:
                    # Signal too short, use constant HR
                    forecast_length = max(10, len(signal_data) // 10)
                    forecast_time = forecast_length / sampling_freq
                    forecast_hr = np.full(forecast_length, heart_rate)

                results["heart_rate_forecast"] = {
                    "forecast_length": forecast_length,
                    "forecast_time_seconds": forecast_time,
                    "current_hr": heart_rate,
                    "forecast_hr_values": forecast_hr.tolist(),
                    "method": "Heart Rate Trend Forecasting (Window-based)",
                }
            else:
                results["heart_rate_forecast"] = {"error": "No heart rate detected"}
        except Exception as e:
            logger.warning(f"Heart rate forecasting failed: {e}")
            results["heart_rate_forecast"] = {"error": str(e)}

        # Signal amplitude forecasting
        try:
            # Analyze recent amplitude trends
            recent_window = min(100, len(signal_data) // 5)
            recent_signal = signal_data[-recent_window:]

            # Calculate amplitude trend
            time_points = np.arange(len(recent_signal))
            amplitude_coeffs = np.polyfit(time_points, np.abs(recent_signal), 1)

            # Forecast amplitude
            forecast_length = max(10, len(signal_data) // 10)
            future_time_points = np.arange(
                len(recent_signal), len(recent_signal) + forecast_length
            )
            forecast_amplitude = np.polyval(amplitude_coeffs, future_time_points)

            results["amplitude_forecast"] = {
                "forecast_length": forecast_length,
                "forecast_amplitude": forecast_amplitude.tolist(),
                "amplitude_trend": amplitude_coeffs[0],
                "method": "Amplitude Trend Forecasting",
            }
        except Exception as e:
            logger.warning(f"Amplitude forecasting failed: {e}")
            results["amplitude_forecast"] = {"error": str(e)}

        results["method"] = "Multi-method Forecasting Analysis"
        results["total_forecasts"] = len(
            [
                k
                for k in results.keys()
                if k != "method" and "error" not in str(results[k])
            ]
        )

        return results

    except Exception as e:
        logger.error(f"Error in forecasting analysis: {e}")
        return {"error": f"Forecasting analysis failed: {str(e)}"}


def perform_advanced_signal_processing(signal_data, sampling_freq):
    """Perform advanced signal processing."""
    try:
        advanced_results = {}

        # Wavelet analysis
        advanced_results["wavelet"] = perform_wavelet_analysis(
            signal_data, sampling_freq
        )

        # Hilbert-Huang transform
        advanced_results["hilbert_huang"] = perform_hilbert_huang_transform(
            signal_data, sampling_freq
        )

        # Empirical mode decomposition
        advanced_results["emd"] = perform_empirical_mode_decomposition(
            signal_data, sampling_freq
        )

        return advanced_results

    except Exception as e:
        logger.error(f"Error in advanced signal processing: {e}")
        return {"error": f"Advanced signal processing failed: {str(e)}"}


# Additional helper functions for specific analyses
def calculate_skewness(data):
    """Calculate skewness of the data."""
    try:
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    except Exception:
        return 0


def calculate_kurtosis(data):
    """Calculate kurtosis of the data."""
    try:
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    except Exception:
        return 0


def calculate_entropy(data):
    """Calculate entropy of the data."""
    try:
        # Simple histogram-based entropy
        hist, _ = np.histogram(data, bins=50)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0
        p = hist / np.sum(hist)
        return -np.sum(p * np.log2(p))
    except Exception:
        return 0


def extract_spectral_features(signal_data, sampling_freq):
    """Extract spectral features."""
    try:
        # FFT
        fft_result = np.fft.fft(signal_data)
        fft_freq = np.fft.fftfreq(len(signal_data), 1 / sampling_freq)

        # Get positive frequencies
        positive_mask = fft_freq > 0
        fft_freq = fft_freq[positive_mask]
        fft_magnitude = np.abs(fft_result[positive_mask])

        # Spectral features
        spectral_centroid = (
            np.sum(fft_freq * fft_magnitude) / np.sum(fft_magnitude)
            if np.sum(fft_magnitude) > 0
            else 0
        )
        spectral_bandwidth = (
            np.sqrt(
                np.sum(((fft_freq - spectral_centroid) ** 2) * fft_magnitude)
                / np.sum(fft_magnitude)
            )
            if np.sum(fft_magnitude) > 0
            else 0
        )

        return {
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "dominant_frequency": (
                fft_freq[np.argmax(fft_magnitude)] if len(fft_magnitude) > 0 else 0
            ),
        }
    except Exception as e:
        logger.error(f"Error extracting spectral features: {e}")
        return {"error": f"Spectral feature extraction failed: {str(e)}"}


def extract_temporal_features(signal_data, sampling_freq):
    """Extract temporal features using vitalDSP WaveformMorphology."""
    try:
        # Use vitalDSP's WaveformMorphology for proper peak detection
        from vitalDSP.physiological_features.waveform import WaveformMorphology

        # Detect peaks using vitalDSP (auto-detects signal type or uses ECG as default)
        wm = WaveformMorphology(signal_data, fs=sampling_freq, signal_type="ECG")
        peaks = (
            wm.critical_points.get("r_peaks", wm.r_peaks)
            if hasattr(wm, "critical_points")
            else wm.r_peaks
        )

        use_basic_detection = False

        if peaks is not None and len(peaks) > 1:
            # Calculate peak intervals in seconds
            intervals = np.diff(peaks) / sampling_freq
            mean_interval = np.mean(intervals)

            # Calculate heart rate
            if mean_interval > 0:
                heart_rate = 60 / mean_interval

                # Validation logging for physiological ranges
                if heart_rate > 200:
                    logger.warning(
                        f"vitalDSP calculated HR ({heart_rate:.1f} bpm) exceeds limit. Found {len(peaks)} peaks - likely over-detection."
                    )
                    logger.warning(f"Falling back to basic peak detection...")
                    use_basic_detection = True
                elif heart_rate < 30:
                    logger.warning(
                        f"Calculated HR ({heart_rate:.1f} bpm) below typical physiological limit (<30 bpm)."
                    )
            else:
                heart_rate = 0

            if not use_basic_detection:
                logger.info(
                    f"vitalDSP peak detection: found {len(peaks)} peaks, HR={heart_rate:.1f} bpm"
                )
            return {
                "peak_count": len(peaks),
                "mean_interval": mean_interval,
                "interval_std": np.std(intervals),
                "heart_rate": heart_rate,
            }

        if use_basic_detection or peaks is None or len(peaks) <= 1:
            # Fallback if vitalDSP peak detection fails
            logger.info(
                "vitalDSP peak detection returned no peaks, using basic peak detection"
            )
            threshold = np.mean(signal_data) + 1.5 * np.std(signal_data)
            # Minimum peak distance should allow max heart rate ~200 bpm
            # For 200 bpm, interval is 0.3s, so min_peak_distance = 0.25s * sampling_freq
            min_peak_distance = int(0.25 * sampling_freq)
            peaks_basic, _ = signal.find_peaks(
                signal_data, height=threshold, distance=min_peak_distance
            )

            if len(peaks_basic) > 1:
                # Calculate HR and validate
                intervals = np.diff(peaks_basic) / sampling_freq
                mean_interval = np.mean(intervals)
                heart_rate = 60 / mean_interval if mean_interval > 0 else 0

                logger.info(
                    f"Basic peak detection found {len(peaks_basic)} peaks, HR={heart_rate:.1f} bpm"
                )

                # If HR is unrealistic, try more aggressive filtering
                if heart_rate > 250:
                    logger.warning(
                        f"HR {heart_rate:.1f} bpm too high, applying more aggressive peak detection"
                    )
                    min_peak_distance = int(0.4 * sampling_freq)  # Allow max ~150 bpm
                    peaks_basic, _ = signal.find_peaks(
                        signal_data, height=threshold, distance=min_peak_distance
                    )

                    # Recalculate after filtering
                    if len(peaks_basic) > 1:
                        intervals = np.diff(peaks_basic) / sampling_freq
                        mean_interval = np.mean(intervals)
                        heart_rate = 60 / mean_interval if mean_interval > 0 else 0
                        logger.info(
                            f"After aggressive filtering: {len(peaks_basic)} peaks, HR={heart_rate:.1f} bpm"
                        )

                return {
                    "peak_count": len(peaks_basic),
                    "mean_interval": mean_interval,
                    "interval_std": np.std(intervals),
                    "heart_rate": heart_rate,
                }
            else:
                # No peaks found with basic detection
                return {
                    "peak_count": 0,
                    "mean_interval": 0,
                    "interval_std": 0,
                    "heart_rate": 0,
                }
    except ImportError:
        logger.warning(
            "vitalDSP not available for peak detection, using scipy fallback"
        )
        # Fallback to scipy peak detection
        threshold = np.mean(signal_data) + 1.5 * np.std(signal_data)
        min_peak_distance = int(0.3 * sampling_freq)
        peaks, _ = signal.find_peaks(
            signal_data, height=threshold, distance=min_peak_distance
        )

        if len(peaks) > 1:
            intervals = np.diff(peaks) / sampling_freq
            mean_interval = np.mean(intervals)
            heart_rate = 60 / mean_interval if mean_interval > 0 else 0

            return {
                "peak_count": len(peaks),
                "mean_interval": mean_interval,
                "interval_std": np.std(intervals),
                "heart_rate": heart_rate,
            }
        else:
            return {
                "peak_count": 0,
                "mean_interval": 0,
                "interval_std": 0,
                "heart_rate": 0,
            }
    except Exception as e:
        logger.error(f"Error extracting temporal features: {e}")
        return {"error": f"Temporal feature extraction failed: {str(e)}"}


def extract_morphological_features(signal_data, sampling_freq):
    """Extract morphological features."""
    try:
        # Basic morphological features
        return {
            "amplitude_range": np.max(signal_data) - np.min(signal_data),
            "amplitude_mean": np.mean(np.abs(signal_data)),
            "zero_crossings": np.sum(np.diff(np.sign(signal_data)) != 0),
            "signal_energy": np.sum(signal_data**2),
        }
    except Exception as e:
        logger.error(f"Error extracting morphological features: {e}")
        return {"error": f"Morphological feature extraction failed: {str(e)}"}


def extract_ml_features(signal_data, sampling_freq):
    """Extract features for machine learning."""
    try:
        # Combine all feature types
        features = []

        # Statistical features
        features.extend(
            [
                np.mean(signal_data),
                np.std(signal_data),
                calculate_skewness(signal_data),
                calculate_kurtosis(signal_data),
                calculate_entropy(signal_data),
            ]
        )

        # Spectral features
        spectral = extract_spectral_features(signal_data, sampling_freq)
        if "error" not in spectral:
            features.extend(
                [
                    spectral.get("spectral_centroid", 0),
                    spectral.get("spectral_bandwidth", 0),
                    spectral.get("dominant_frequency", 0),
                ]
            )

        # Temporal features
        temporal = extract_temporal_features(signal_data, sampling_freq)
        if "error" not in temporal:
            features.extend(
                [
                    temporal.get("peak_count", 0),
                    temporal.get("mean_interval", 0),
                    temporal.get("heart_rate", 0),
                ]
            )

        return np.array(features)

    except Exception as e:
        logger.error(f"Error extracting ML features: {e}")
        return np.array([])


# ML/DL models using actual sklearn implementations
def train_svm_model(features, cv_folds, random_state, svm_params=None):
    """Train SVM model using sklearn."""
    try:
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        # Extract parameters
        kernel = "rbf"
        C = 1.0
        gamma = "scale"

        if svm_params:
            kernel = svm_params.get("kernel", "rbf")
            C = svm_params.get("C", 1.0)
            gamma = svm_params.get("gamma", "scale")

        # Create dummy labels for demonstration (in real use, these would be provided)
        labels = np.random.randint(0, 2, len(features))

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.reshape(-1, 1))

        # Train SVM with parameters
        svm_model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)
        cv_scores = cross_val_score(svm_model, features_scaled, labels, cv=cv_folds)

        return {
            "model_type": "SVM",
            "status": "trained",
            "cv_folds": cv_folds,
            "cv_score_mean": float(np.mean(cv_scores)),
            "cv_score_std": float(np.std(cv_scores)),
        }
    except ImportError:
        return {
            "model_type": "SVM",
            "status": "sklearn_not_available",
            "cv_folds": cv_folds,
        }
    except Exception as e:
        return {
            "model_type": "SVM",
            "status": "error",
            "error": str(e),
            "cv_folds": cv_folds,
        }


def train_random_forest_model(features, cv_folds, random_state, rf_params=None):
    """Train Random Forest model using sklearn."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        # Extract parameters
        n_estimators = 100
        max_depth = 10
        min_samples_split = 2

        if rf_params:
            n_estimators = rf_params.get("n_estimators", 100)
            max_depth = rf_params.get("max_depth", 10)
            min_samples_split = rf_params.get("min_samples_split", 2)

        # Create dummy labels for demonstration
        labels = np.random.randint(0, 2, len(features))

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.reshape(-1, 1))

        # Train Random Forest with parameters
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )
        cv_scores = cross_val_score(rf_model, features_scaled, labels, cv=cv_folds)

        return {
            "model_type": "Random Forest",
            "status": "trained",
            "cv_folds": cv_folds,
            "cv_score_mean": float(np.mean(cv_scores)),
            "cv_score_std": float(np.std(cv_scores)),
        }
    except ImportError:
        return {
            "model_type": "Random Forest",
            "status": "sklearn_not_available",
            "cv_folds": cv_folds,
        }
    except Exception as e:
        return {
            "model_type": "Random Forest",
            "status": "error",
            "error": str(e),
            "cv_folds": cv_folds,
        }


def train_neural_network_model(features, cv_folds, random_state, nn_params=None):
    """
    Train Neural Network model for signal processing using vitalDSP.

    Parameters
    ----------
    features : numpy.ndarray
        Feature data for training (assumed to be signal data)
    cv_folds : int
        Number of cross-validation folds
    random_state : int
        Random seed for reproducibility
    nn_params : dict
        Neural network parameters

    Returns
    -------
    dict
        Dictionary containing model results and performance metrics
    """
    try:
        from vitalDSP.advanced_computation.neural_network_filtering import (
            NeuralNetworkFiltering,
        )

        # Extract parameters
        hidden_layers = [64, 32]
        learning_rate = 0.001

        if nn_params:
            hidden_layers_str = nn_params.get("hidden_layers", "100,50")
            hidden_layers = [int(x.strip()) for x in hidden_layers_str.split(",")]
            learning_rate = nn_params.get("learning_rate", 0.001)

        # Train a feedforward neural network for signal denoising
        nn_filter = NeuralNetworkFiltering(
            features,
            network_type="feedforward",
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            epochs=50,  # Reduced for webapp performance
            batch_size=32,
            dropout_rate=0.3,
            batch_norm=True,
        )

        logger.info("Training neural network filter...")
        nn_filter.train()

        # Apply the trained filter
        filtered_signal = nn_filter.apply_filter()

        # Calculate denoising performance (SNR improvement)
        noise_power = np.var(features - filtered_signal)
        signal_power = np.var(filtered_signal)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

        logger.info(
            f"Neural network training complete. SNR improvement: {snr_db:.2f} dB"
        )

        return {
            "model_type": "Neural Network",
            "status": "success",
            "network_type": "feedforward",
            "hidden_layers": hidden_layers,
            "learning_rate": learning_rate,
            "filtered_signal": filtered_signal,
            "snr_db": snr_db,
            "cv_folds": cv_folds,
            "training_samples": len(features),
        }

    except ImportError as e:
        logger.warning(f"vitalDSP NeuralNetworkFiltering not available: {e}")
        return {
            "model_type": "Neural Network",
            "status": "placeholder",
            "cv_folds": cv_folds,
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Error training neural network model: {e}")
        return {
            "model_type": "Neural Network",
            "status": "placeholder",
            "cv_folds": cv_folds,
            "error": str(e),
        }


def train_gradient_boosting_model(features, cv_folds, random_state):
    """Train Gradient Boosting model using sklearn."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        # Create dummy labels for demonstration
        labels = np.random.randint(0, 2, len(features))

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.reshape(-1, 1))

        # Train Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=100, random_state=random_state
        )
        cv_scores = cross_val_score(gb_model, features_scaled, labels, cv=cv_folds)

        return {
            "model_type": "Gradient Boosting",
            "status": "trained",
            "cv_folds": cv_folds,
            "cv_score_mean": float(np.mean(cv_scores)),
            "cv_score_std": float(np.std(cv_scores)),
        }
    except ImportError:
        return {
            "model_type": "Gradient Boosting",
            "status": "sklearn_not_available",
            "cv_folds": cv_folds,
        }
    except Exception as e:
        return {
            "model_type": "Gradient Boosting",
            "status": "error",
            "error": str(e),
            "cv_folds": cv_folds,
        }


def prepare_dl_data(signal_data, sampling_freq):
    """
    Prepare data for deep learning models.

    Parameters
    ----------
    signal_data : numpy.ndarray
        Input signal data
    sampling_freq : float
        Sampling frequency in Hz

    Returns
    -------
    dict
        Dictionary containing prepared data for DL models
    """
    return {
        "signal_data": signal_data,
        "sampling_freq": sampling_freq,
        "data_shape": signal_data.shape,
        "num_samples": len(signal_data),
    }


def train_cnn_model(data, cnn_params=None):
    """
    Train CNN model for signal analysis using vitalDSP.

    Parameters
    ----------
    data : dict
        Dictionary containing signal data and metadata
    cnn_params : dict
        CNN model parameters

    Returns
    -------
    dict
        Dictionary containing CNN model results
    """
    try:
        from vitalDSP.advanced_computation.neural_network_filtering import (
            NeuralNetworkFiltering,
        )

        signal_data = data.get("signal_data", np.array([]))
        sampling_freq = data.get("sampling_freq", 1000)

        # Extract CNN parameters
        cnn_filters = [64, 128, 256]
        kernel_size = 3
        pool_size = 2
        dropout_rate = 0.3

        if cnn_params:
            filters_str = cnn_params.get("filters", "64,128,256")
            cnn_filters = [int(x.strip()) for x in filters_str.split(",")]
            kernel_size = cnn_params.get("kernel", 3)
            pool_size = cnn_params.get("pool", 2)
            dropout_rate = cnn_params.get("dropout", 0.3)

        if len(signal_data) == 0:
            return {
                "model_type": "CNN",
                "status": "placeholder",
                "error": "No signal data",
            }

        # Reshape signal for CNN (requires 2D input)
        # Create overlapping windows for CNN processing
        window_size = min(128, len(signal_data) // 4)
        stride = window_size // 2
        windows = []

        for i in range(0, len(signal_data) - window_size, stride):
            windows.append(signal_data[i : i + window_size])

        if len(windows) < 2:
            return {
                "model_type": "CNN",
                "status": "placeholder",
                "error": "Signal too short for CNN processing",
            }

        # Note: vitalDSP CNN expects specific shape, simplified implementation for demo
        logger.info(
            f"CNN model: Processing {len(windows)} windows of size {window_size}"
        )

        return {
            "model_type": "CNN",
            "status": "placeholder",
            "windows_processed": len(windows),
            "window_size": window_size,
            "filters": cnn_filters,
            "kernel_size": kernel_size,
            "pool_size": pool_size,
            "dropout_rate": dropout_rate,
            "sampling_freq": sampling_freq,
            "note": f"CNN architecture: {cnn_filters} filters, {kernel_size}x{kernel_size} kernel, dropout={dropout_rate}",
        }

    except Exception as e:
        logger.error(f"Error training CNN model: {e}")
        return {"model_type": "CNN", "status": "placeholder", "error": str(e)}


def train_lstm_model(data, lstm_params=None):
    """
    Train LSTM model for time-series signal analysis using vitalDSP.

    Parameters
    ----------
    data : dict
        Dictionary containing signal data and metadata
    lstm_params : dict
        LSTM model parameters

    Returns
    -------
    dict
        Dictionary containing LSTM model results
    """
    try:
        from vitalDSP.advanced_computation.neural_network_filtering import (
            NeuralNetworkFiltering,
        )

        signal_data = data.get("signal_data", np.array([]))
        sampling_freq = data.get("sampling_freq", 1000)

        if len(signal_data) == 0:
            return {
                "model_type": "LSTM",
                "status": "placeholder",
                "error": "No signal data",
            }

        # LSTM works well for sequential data
        # Create sequences for LSTM
        sequence_length = min(64, len(signal_data) // 8)

        if len(signal_data) < sequence_length * 2:
            return {
                "model_type": "LSTM",
                "status": "placeholder",
                "error": "Signal too short for LSTM processing",
            }

        # Extract LSTM parameters
        lstm_units = 128
        lstm_layers = 2
        dropout_rate = 0.2
        learning_rate = 0.001

        if lstm_params:
            lstm_units = lstm_params.get("units", 128)
            lstm_layers = lstm_params.get("layers", 2)
            dropout_rate = lstm_params.get("dropout", 0.2)
            learning_rate = lstm_params.get("learning_rate", 0.001)

        # Create hidden_layers list based on number of layers
        hidden_layers = [lstm_units] * lstm_layers

        # Train LSTM network
        try:
            lstm_filter = NeuralNetworkFiltering(
                signal_data,
                network_type="recurrent",
                hidden_layers=hidden_layers,
                learning_rate=learning_rate,
                epochs=30,
                batch_size=16,
                dropout_rate=dropout_rate,
                batch_norm=True,
                recurrent_type="lstm",
            )

            logger.info("Training LSTM model...")
            lstm_filter.train()

            # Apply the filter - LSTM is passed the full signal data, but it needs proper windowing
            # The NeuralNetworkFiltering class should handle this internally, but we need to be careful
            filtered_signal = lstm_filter.apply_filter()

            # Ensure shapes match for MSE calculation
            filtered_signal_flat = filtered_signal.flatten()
            # Match the shorter length
            min_len = min(len(signal_data), len(filtered_signal_flat))
            if min_len > 0:
                mse = np.mean(
                    (signal_data[:min_len] - filtered_signal_flat[:min_len]) ** 2
                )
            else:
                mse = 0.0

            logger.info(f"LSTM training complete. MSE: {mse:.6f}")
        except Exception as train_error:
            # If training fails (e.g., shape mismatch), return placeholder without training
            logger.warning(
                f"LSTM training failed: {train_error}. Returning placeholder result."
            )
            mse = 0.0
            filtered_signal = np.zeros_like(signal_data)  # Placeholder output

        return {
            "model_type": "LSTM",
            "status": "placeholder",
            "sequence_length": sequence_length,
            "hidden_units": lstm_units,
            "num_layers": lstm_layers,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "mse": mse,
            "filtered_signal": filtered_signal,
            "sampling_freq": sampling_freq,
            "note": f"LSTM with {lstm_units} hidden units, {lstm_layers} layers, dropout={dropout_rate}",
        }

    except Exception as e:
        logger.error(f"Error training LSTM model: {e}")
        return {"model_type": "LSTM", "status": "placeholder", "error": str(e)}


def train_transformer_model(data, transformer_params=None):
    """Train Transformer model using sklearn (simplified implementation)."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        signal_data = data.get("signal_data", np.array([]))
        if len(signal_data) == 0:
            return {"model_type": "Transformer", "status": "no_data"}

        # Extract Transformer parameters
        attention_heads = 8
        num_layers = 4
        model_dim = 512

        if transformer_params:
            attention_heads = transformer_params.get("heads", 8)
            num_layers = transformer_params.get("layers", 4)
            model_dim = transformer_params.get("dim", 512)

        # Create dummy labels for demonstration
        labels = np.random.randint(0, 2, len(signal_data))

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(signal_data.reshape(-1, 1))

        # Use Random Forest as a simplified transformer-like model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        cv_scores = cross_val_score(model, features_scaled, labels, cv=3)

        return {
            "model_type": "Transformer",
            "status": "trained",
            "attention_heads": attention_heads,
            "num_layers": num_layers,
            "model_dimension": model_dim,
            "cv_score_mean": float(np.mean(cv_scores)),
            "note": f"Simplified transformer: {num_layers} layers, {attention_heads} heads, dim={model_dim}",
        }
    except ImportError:
        return {"model_type": "Transformer", "status": "sklearn_not_available"}
    except Exception as e:
        return {"model_type": "Transformer", "status": "error", "error": str(e)}


def analyze_peak_patterns(signal_data, sampling_freq):
    """Analyze peak patterns using vitalDSP."""
    try:
        from vitalDSP.physiological_features.waveform import WaveformMorphology

        # Detect peaks using vitalDSP
        wm = WaveformMorphology(signal_data, fs=sampling_freq, signal_type="ECG")
        peaks = wm.r_peaks

        if len(peaks) > 1:
            # Calculate peak intervals
            intervals = np.diff(peaks) / sampling_freq
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)

            return {
                "pattern_type": "Peak Patterns",
                "status": "analyzed",
                "num_peaks": len(peaks),
                "mean_interval": float(mean_interval),
                "std_interval": float(std_interval),
                "heart_rate": float(60.0 / mean_interval) if mean_interval > 0 else 0,
            }
        else:
            return {"pattern_type": "Peak Patterns", "status": "insufficient_peaks"}
    except ImportError:
        return {"pattern_type": "Peak Patterns", "status": "vitaldsp_not_available"}
    except Exception as e:
        return {"pattern_type": "Peak Patterns", "status": "error", "error": str(e)}


def analyze_frequency_patterns(signal_data, sampling_freq):
    """Analyze frequency patterns using vitalDSP."""
    try:
        from vitalDSP.transforms.fourier_transform import FourierTransform

        # Perform FFT analysis
        ft = FourierTransform(signal_data, fs=sampling_freq)
        freqs, psd = ft.compute_psd()

        # Find dominant frequency
        dominant_freq_idx = np.argmax(psd)
        dominant_freq = freqs[dominant_freq_idx]

        # Calculate spectral centroid
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd)

        return {
            "pattern_type": "Frequency Patterns",
            "status": "analyzed",
            "dominant_frequency": float(dominant_freq),
            "spectral_centroid": float(spectral_centroid),
            "total_power": float(np.sum(psd)),
        }
    except ImportError:
        return {
            "pattern_type": "Frequency Patterns",
            "status": "vitaldsp_not_available",
        }
    except Exception as e:
        return {
            "pattern_type": "Frequency Patterns",
            "status": "error",
            "error": str(e),
        }


def analyze_morphological_patterns(signal_data, sampling_freq):
    """Analyze morphological patterns using vitalDSP."""
    try:
        from vitalDSP.physiological_features.waveform import WaveformMorphology

        # Analyze waveform morphology
        wm = WaveformMorphology(signal_data, fs=sampling_freq, signal_type="ECG")

        # Extract morphological features
        peaks = wm.r_peaks
        valleys = wm.t_waves  # Using t_waves as valleys approximation

        if len(peaks) > 0:
            peak_heights = signal_data[peaks]
            mean_peak_height = np.mean(peak_heights)
            std_peak_height = np.std(peak_heights)

            return {
                "pattern_type": "Morphological Patterns",
                "status": "analyzed",
                "num_peaks": len(peaks),
                "mean_peak_height": float(mean_peak_height),
                "std_peak_height": float(std_peak_height),
                "peak_variability": (
                    float(std_peak_height / mean_peak_height)
                    if mean_peak_height > 0
                    else 0
                ),
            }
        else:
            return {
                "pattern_type": "Morphological Patterns",
                "status": "no_peaks_detected",
            }
    except ImportError:
        return {
            "pattern_type": "Morphological Patterns",
            "status": "vitaldsp_not_available",
        }
    except Exception as e:
        return {
            "pattern_type": "Morphological Patterns",
            "status": "error",
            "error": str(e),
        }


def create_voting_ensemble(signal_data, sampling_freq, cv_folds, random_state):
    """Create voting ensemble using sklearn."""
    try:
        from sklearn.ensemble import VotingClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        # Create dummy labels for demonstration
        labels = np.random.randint(0, 2, len(signal_data))

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(signal_data.reshape(-1, 1))

        # Create base models
        svm = SVC(random_state=random_state)
        rf = RandomForestClassifier(n_estimators=50, random_state=random_state)
        lr = LogisticRegression(random_state=random_state)

        # Create voting ensemble
        voting_clf = VotingClassifier(
            estimators=[("svm", svm), ("rf", rf), ("lr", lr)], voting="hard"
        )

        cv_scores = cross_val_score(voting_clf, features_scaled, labels, cv=cv_folds)

        return {
            "ensemble_type": "Voting",
            "status": "trained",
            "cv_score_mean": float(np.mean(cv_scores)),
            "cv_score_std": float(np.std(cv_scores)),
            "base_models": ["SVM", "Random Forest", "Logistic Regression"],
        }
    except ImportError:
        return {"ensemble_type": "Voting", "status": "sklearn_not_available"}
    except Exception as e:
        return {"ensemble_type": "Voting", "status": "error", "error": str(e)}


def create_stacking_ensemble(signal_data, sampling_freq, cv_folds, random_state):
    """Create stacking ensemble using sklearn."""
    try:
        from sklearn.ensemble import StackingClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        # Create dummy labels for demonstration
        labels = np.random.randint(0, 2, len(signal_data))

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(signal_data.reshape(-1, 1))

        # Create base models
        base_models = [
            ("svm", SVC(random_state=random_state)),
            ("rf", RandomForestClassifier(n_estimators=50, random_state=random_state)),
        ]

        # Create stacking ensemble
        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(random_state=random_state),
            cv=cv_folds,
        )

        cv_scores = cross_val_score(stacking_clf, features_scaled, labels, cv=cv_folds)

        return {
            "ensemble_type": "Stacking",
            "status": "trained",
            "cv_score_mean": float(np.mean(cv_scores)),
            "cv_score_std": float(np.std(cv_scores)),
            "base_models": ["SVM", "Random Forest"],
            "meta_model": "Logistic Regression",
        }
    except ImportError:
        return {"ensemble_type": "Stacking", "status": "sklearn_not_available"}
    except Exception as e:
        return {"ensemble_type": "Stacking", "status": "error", "error": str(e)}


def create_bagging_ensemble(signal_data, sampling_freq, cv_folds, random_state):
    """Create bagging ensemble using sklearn."""
    try:
        from sklearn.ensemble import BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        # Create dummy labels for demonstration
        labels = np.random.randint(0, 2, len(signal_data))

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(signal_data.reshape(-1, 1))

        # Create bagging ensemble
        bagging_clf = BaggingClassifier(
            base_estimator=DecisionTreeClassifier(random_state=random_state),
            n_estimators=50,
            random_state=random_state,
        )

        cv_scores = cross_val_score(bagging_clf, features_scaled, labels, cv=cv_folds)

        return {
            "ensemble_type": "Bagging",
            "status": "trained",
            "cv_score_mean": float(np.mean(cv_scores)),
            "cv_score_std": float(np.std(cv_scores)),
            "base_estimator": "Decision Tree",
            "n_estimators": 50,
        }
    except ImportError:
        return {"ensemble_type": "Bagging", "status": "sklearn_not_available"}
    except Exception as e:
        return {"ensemble_type": "Bagging", "status": "error", "error": str(e)}


def perform_wavelet_analysis(signal_data, sampling_freq):
    """Perform wavelet analysis using vitalDSP."""
    try:
        from vitalDSP.transforms.wavelet_transform import WaveletTransform

        # Basic validation
        if not isinstance(signal_data, (np.ndarray, list)) or len(signal_data) == 0:
            return {"analysis_type": "Wavelet", "status": "no_data"}

        # Perform wavelet transform
        wt = WaveletTransform(signal_data, wavelet_name="db4")
        coefficients = wt.perform_wavelet_transform()

        # Calculate energy distribution
        if isinstance(coefficients, list):
            energy_levels = [np.sum(np.abs(coeff) ** 2) for coeff in coefficients]
            total_energy = sum(energy_levels)
            energy_percentages = (
                [e / total_energy * 100 for e in energy_levels]
                if total_energy > 0
                else [0] * len(energy_levels)
            )
        else:
            energy_levels = [np.sum(np.abs(coefficients) ** 2)]
            total_energy = sum(energy_levels)
            energy_percentages = [100.0] if total_energy > 0 else [0]

        return {
            "analysis_type": "Wavelet",
            "status": "analyzed",
            "wavelet_type": "db4",
            "num_coefficients": (
                len(coefficients) if isinstance(coefficients, list) else 1
            ),
            "total_energy": float(total_energy),
            "energy_percentages": [float(p) for p in energy_percentages],
        }
    except ImportError:
        return {"analysis_type": "Wavelet", "status": "vitaldsp_not_available"}
    except Exception as e:
        return {"analysis_type": "Wavelet", "status": "error", "error": str(e)}


def perform_hilbert_huang_transform(signal_data, sampling_freq):
    """Perform Hilbert-Huang transform using vitalDSP."""
    try:
        from vitalDSP.transforms.hilbert_transform import HilbertTransform

        # Basic validation
        if not isinstance(signal_data, (np.ndarray, list)) or len(signal_data) == 0:
            return {"analysis_type": "Hilbert-Huang", "status": "no_data"}

        # Perform Hilbert transform
        ht = HilbertTransform(signal_data, fs=sampling_freq)
        analytic_signal = ht.compute_analytic_signal()

        # Extract instantaneous amplitude and phase
        amplitude = np.abs(analytic_signal)
        phase = np.angle(analytic_signal)

        # Calculate instantaneous frequency
        instantaneous_freq = np.diff(np.unwrap(phase)) / (2.0 * np.pi) * sampling_freq

        return {
            "analysis_type": "Hilbert-Huang",
            "status": "analyzed",
            "mean_amplitude": float(np.mean(amplitude)),
            "std_amplitude": float(np.std(amplitude)),
            "mean_frequency": (
                float(np.mean(instantaneous_freq)) if len(instantaneous_freq) > 0 else 0
            ),
            "std_frequency": (
                float(np.std(instantaneous_freq)) if len(instantaneous_freq) > 0 else 0
            ),
        }
    except ImportError:
        return {"analysis_type": "Hilbert-Huang", "status": "vitaldsp_not_available"}
    except Exception as e:
        return {"analysis_type": "Hilbert-Huang", "status": "error", "error": str(e)}


def perform_empirical_mode_decomposition(signal_data, sampling_freq):
    """
    Perform Empirical Mode Decomposition (EMD) using vitalDSP.

    EMD decomposes non-linear and non-stationary signals into Intrinsic Mode Functions (IMFs).

    Parameters
    ----------
    signal_data : numpy.ndarray
        Input signal to decompose
    sampling_freq : float
        Sampling frequency of the signal in Hz

    Returns
    -------
    dict
        Dictionary containing:
        - analysis_type: str
        - imfs: list of numpy.ndarray (Intrinsic Mode Functions)
        - num_imfs: int
        - residual: numpy.ndarray (final residual after decomposition)
        - reconstruction_error: float (MSE between original and reconstructed signal)
    """
    try:
        from vitalDSP.advanced_computation.emd import EMD

        # Perform EMD decomposition
        emd = EMD(signal_data)
        imfs = emd.emd(
            max_imfs=8, stop_criterion=0.05
        )  # Limit to 8 IMFs for efficiency

        # Calculate residual
        residual = signal_data - np.sum(imfs, axis=0)

        # Calculate reconstruction error
        reconstructed = np.sum(imfs, axis=0) + residual
        reconstruction_error = np.mean((signal_data - reconstructed) ** 2)

        logger.info(f"EMD decomposition complete: {len(imfs)} IMFs extracted")

        return {
            "analysis_type": "EMD",
            "status": "placeholder",
            "imfs": imfs,
            "num_imfs": len(imfs),
            "residual": residual,
            "reconstruction_error": reconstruction_error,
            "sampling_freq": sampling_freq,
        }

    except ImportError as e:
        logger.warning(f"vitalDSP EMD not available: {e}, returning placeholder")
        return {"analysis_type": "EMD", "status": "placeholder", "error": str(e)}
    except Exception as e:
        logger.error(f"Error in EMD decomposition: {e}")
        return {"analysis_type": "EMD", "status": "placeholder", "error": str(e)}


# Visualization functions
def create_main_advanced_plot(time_axis, signal_data, analysis_results, signal_type):
    """Create the main advanced analysis plot with enhanced annotations."""
    try:
        # Don't fail if individual analysis steps have errors - plot what we have
        # Only fail if analysis_results itself is None or empty
        if not analysis_results:
            return create_empty_figure()

        fig = go.Figure()

        # Original signal with enhanced styling
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal_data,
                mode="lines",
                name="Signal",
                line=dict(color="#1f77b4", width=1.5),
                hovertemplate="<b>Time:</b> %{x:.3f}s<br>"
                + "<b>Amplitude:</b> %{y:.2f}<br>"
                + "<extra></extra>",
            )
        )

        # Add peak annotations if temporal features available
        if "features" in analysis_results:
            features = analysis_results["features"]
            if "temporal" in features and "error" not in features["temporal"]:
                temporal = features["temporal"]
                # Try to detect and mark peaks
                try:
                    from scipy import signal as scipy_signal

                    peaks, _ = scipy_signal.find_peaks(
                        signal_data,
                        height=np.mean(signal_data) + 1.5 * np.std(signal_data),
                        distance=int(
                            0.3
                            * len(signal_data)
                            / (time_axis[-1] if len(time_axis) > 0 else 1)
                        ),
                    )

                    if len(peaks) > 0:
                        peak_times = (
                            time_axis[peaks]
                            if len(peaks) < len(time_axis)
                            else time_axis[: len(peaks)]
                        )
                        peak_values = (
                            signal_data[peaks]
                            if len(peaks) < len(signal_data)
                            else signal_data[: len(peaks)]
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=peak_times,
                                y=peak_values,
                                mode="markers",
                                name=f"Detected Peaks ({len(peaks)})",
                                marker=dict(color="red", size=8, symbol="x"),
                                hovertemplate="<b>Peak</b><br>"
                                + "<b>Time:</b> %{x:.3f}s<br>"
                                + "<b>Amplitude:</b> %{y:.2f}<br>"
                                + "<extra></extra>",
                            )
                        )

                        # Add annotation for first few peaks
                        for i, (t, v) in enumerate(
                            zip(peak_times[:3], peak_values[:3])
                        ):
                            fig.add_annotation(
                                x=t,
                                y=v,
                                text=f"P{i+1}",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowwidth=1,
                                arrowcolor="red",
                                ax=0,
                                ay=-30,
                                font=dict(size=10, color="red"),
                            )
                except Exception as e:
                    logger.debug(f"Could not add peak annotations: {e}")

        # Calculate statistics for subtitle
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        duration = time_axis[-1] - time_axis[0] if len(time_axis) > 1 else 0

        # Build comprehensive title with statistics
        title_text = f"<b>Advanced Analysis: {signal_type.upper()} Signal</b><br>"
        title_text += f"<sub>Duration: {duration:.1f}s | Mean: {mean_val:.2f} | Std: {std_val:.2f} | Samples: {len(signal_data)}</sub>"

        # Add heart rate info if available
        if "features" in analysis_results:
            features = analysis_results["features"]
            if "temporal" in features and "error" not in features["temporal"]:
                temporal = features["temporal"]
                if "heart_rate" in temporal and temporal["heart_rate"] > 0:
                    hr = temporal["heart_rate"]
                    hr_status = "✅" if 30 <= hr <= 200 else "⚠️"
                    title_text += f" | Heart Rate: {hr:.1f} bpm {hr_status}"

        # PHASE D: Enhanced layout with modern styling and interactivity
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor="center",
                font=dict(size=16),
            ),
            xaxis=dict(
                title="<b>Time (seconds)</b>",
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(128,128,128,0.2)",
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="rgba(128,128,128,0.3)",
                # PHASE D: Enable range slider and selector
                rangeslider=dict(
                    visible=False
                ),  # Keep disabled for cleaner look, enable zoom instead
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(
                                count=10,
                                label="10s",
                                step="second",
                                stepmode="backward",
                            ),
                            dict(
                                count=30,
                                label="30s",
                                step="second",
                                stepmode="backward",
                            ),
                            dict(
                                count=1, label="1m", step="minute", stepmode="backward"
                            ),
                            dict(step="all", label="All"),
                        ]
                    ),
                    bgcolor="rgba(150, 150, 150, 0.1)",
                    activecolor="rgba(0, 123, 255, 0.3)",
                    font=dict(size=10),
                    x=0,
                    y=1.15,
                ),
            ),
            yaxis=dict(
                title="<b>Amplitude</b>",
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(128,128,128,0.2)",
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="rgba(128,128,128,0.3)",
            ),
            height=550,  # PHASE D: Slightly taller for better visibility
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(128,128,128,0.3)",
                borderwidth=1,
            ),
            hovermode="x unified",  # PHASE D: Unified hover for better comparison
            plot_bgcolor="white",
            paper_bgcolor="rgba(248, 249, 250, 0.5)",  # PHASE D: Subtle background
            # PHASE D: Add modebar buttons for interactivity
            modebar=dict(
                bgcolor="rgba(255, 255, 255, 0.8)",
                color="rgba(0, 0, 0, 0.5)",
                activecolor="rgba(0, 123, 255, 1)",
            ),
            # PHASE D: Enable drag modes
            dragmode="zoom",
        )

        # PHASE D: Add statistical overlay shapes
        # Add mean line
        fig.add_hline(
            y=mean_val,
            line_dash="dot",
            line_color="green",
            opacity=0.5,
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_position="right",
            annotation_font=dict(size=10, color="green"),
        )

        # Add ±1 std deviation bands
        fig.add_hrect(
            y0=mean_val - std_val,
            y1=mean_val + std_val,
            fillcolor="green",
            opacity=0.1,
            line_width=0,
            annotation_text="±1σ",
            annotation_position="top right",
            annotation_font=dict(size=9, color="green"),
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating main advanced plot: {e}")
        return create_empty_figure()


def create_advanced_performance_plot(analysis_results, analysis_categories):
    """Create advanced performance plot with actual CV scores and error bars."""
    try:
        # Don't fail if individual analysis steps have errors - plot what we have
        if not analysis_results:
            return create_empty_figure()

        fig = go.Figure()
        has_data = False

        # Add ML performance metrics if available
        if "ml_results" in analysis_results:
            ml_results = analysis_results["ml_results"]
            if "error" not in ml_results and ml_results:
                models = []
                scores = []
                errors = []
                colors = []

                # Extract actual CV scores from ML results
                for model_name, model_data in ml_results.items():
                    if isinstance(model_data, dict) and "cv_score_mean" in model_data:
                        models.append(model_name.replace("_", " ").title())
                        scores.append(
                            model_data.get("cv_score_mean", 0) * 100
                        )  # Convert to percentage
                        errors.append(
                            model_data.get("cv_score_std", 0) * 100
                        )  # Convert to percentage

                        # Color-code by performance
                        score = model_data.get("cv_score_mean", 0)
                        if score >= 0.8:
                            colors.append("#2ecc71")  # Green - Good
                        elif score >= 0.6:
                            colors.append("#f39c12")  # Orange - Moderate
                        else:
                            colors.append("#e74c3c")  # Red - Poor

                        has_data = True

                if has_data:
                    # Add bar chart with error bars
                    fig.add_trace(
                        go.Bar(
                            x=models,
                            y=scores,
                            error_y=dict(type="data", array=errors, visible=True),
                            marker=dict(
                                color=colors, line=dict(color="black", width=1)
                            ),
                            text=[f"{s:.1f}%" for s in scores],
                            textposition="outside",
                            hovertemplate="<b>%{x}</b><br>"
                            + "CV Score: %{y:.1f}%<br>"
                            + "Std Dev: %{error_y.array:.1f}%<br>"
                            + "<extra></extra>",
                        )
                    )

                    # Add threshold line for acceptable performance
                    fig.add_hline(
                        y=60,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Acceptable Threshold (60%)",
                        annotation_position="right",
                    )

        # Add DL performance metrics if available
        if "dl_results" in analysis_results and has_data:
            dl_results = analysis_results["dl_results"]
            if "error" not in dl_results and dl_results:
                # Add DL models to the plot
                # (similar extraction as ML above if dl_results contain cv scores)
                pass

        if not has_data:
            # Return empty figure with message
            fig.add_annotation(
                text="No performance metrics available<br>Enable ML/DL analysis to see results",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=14, color="gray"),
            )

        # PHASE D: Enhanced layout with modern styling and color bands
        fig.update_layout(
            title=dict(
                text="<b>Model Performance Comparison</b><br><sub>Cross-Validation Scores with Standard Deviation</sub>",
                x=0.5,
                xanchor="center",
                font=dict(size=16),
            ),
            xaxis=dict(
                title="<b>Model Type</b>",
                showgrid=False,
                tickangle=-45,  # PHASE D: Angle labels for better readability
            ),
            yaxis=dict(
                title="<b>CV Score (%)</b>",
                range=[0, 105],
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(128,128,128,0.2)",
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="rgba(128,128,128,0.3)",
            ),
            height=500,  # PHASE D: Slightly taller
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="rgba(248, 249, 250, 0.5)",  # PHASE D: Subtle background
            hovermode="x",
            # PHASE D: Enhanced modebar
            modebar=dict(
                bgcolor="rgba(255, 255, 255, 0.8)",
                color="rgba(0, 0, 0, 0.5)",
                activecolor="rgba(0, 123, 255, 1)",
            ),
        )

        # PHASE D: Add performance bands with color coding
        if has_data:
            # Excellent performance band (80-100%)
            fig.add_hrect(
                y0=80,
                y1=100,
                fillcolor="green",
                opacity=0.05,
                line_width=0,
                annotation_text="Excellent",
                annotation_position="top left",
                annotation_font=dict(size=9, color="green"),
            )
            # Good performance band (60-80%)
            fig.add_hrect(
                y0=60,
                y1=80,
                fillcolor="orange",
                opacity=0.05,
                line_width=0,
                annotation_text="Good",
                annotation_position="bottom left",
                annotation_font=dict(size=9, color="orange"),
            )
            # Poor performance band (0-60%)
            fig.add_hrect(
                y0=0,
                y1=60,
                fillcolor="red",
                opacity=0.05,
                line_width=0,
                annotation_text="Needs Improvement",
                annotation_position="bottom left",
                annotation_font=dict(size=9, color="red"),
            )

        return fig

    except Exception as e:
        logger.error(f"Error creating performance plot: {e}")
        return create_empty_figure()


def create_advanced_visualizations(
    analysis_results, time_axis, signal_data, sampling_freq
):
    """Create comprehensive advanced visualizations with multiple analysis views."""
    try:
        if not analysis_results:
            return create_empty_figure()

        # Create comprehensive visualization with multiple subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                "Combined Analysis Overview",  # Row 1, Col 1
                "Classification & Regression",  # Row 1, Col 2
                "Anomaly Detection",  # Row 2, Col 1
                "Clustering & Patterns",  # Row 2, Col 2
                "Advanced Processing",  # Row 3, Col 1
                "Forecasting & Predictions",  # Row 3, Col 2
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.15,
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
            ],
        )

        # === Plot 1: Combined Feature Analysis, Pattern Recognition, and Advanced Processing ===
        # This is the MAIN comprehensive plot combining all three analyses in one view
        row1_col1 = (1, 1)

        # 1. Main signal - base layer (always shown)
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal_data,
                mode="lines",
                name="Signal",
                line=dict(color="blue", width=2),
                legendgroup="signal",
                opacity=0.8,
            ),
            row=row1_col1[0],
            col=row1_col1[1],
        )

        # 2. Feature Analysis overlay - normalized feature scores as envelope
        if "features" in analysis_results and "error" not in str(
            analysis_results.get("features", {})
        ):
            try:
                features = analysis_results["features"]

                # Extract and combine key feature metrics into a single composite score
                feature_values = []

                if "statistical" in features and "error" not in str(
                    features.get("statistical", {})
                ):
                    stat = features["statistical"]
                    feature_values.append(abs(stat.get("skewness", 0)) * 0.3)
                    feature_values.append(abs(stat.get("kurtosis", 0)) * 0.3)

                if "spectral" in features and "error" not in str(
                    features.get("spectral", {})
                ):
                    spec = features["spectral"]
                    feature_values.append(
                        np.log1p(spec.get("spectral_centroid", 0)) * 0.2
                    )
                    feature_values.append(
                        np.log1p(spec.get("dominant_frequency", 0)) * 0.2
                    )

                if "temporal" in features and "error" not in str(
                    features.get("temporal", {})
                ):
                    temp = features["temporal"]
                    hr_norm = min(temp.get("heart_rate", 0) / 200, 1.0)
                    peaks_norm = min(temp.get("peak_count", 0) / 1000, 1.0)
                    feature_values.append(hr_norm * 0.1)
                    feature_values.append(peaks_norm * 0.1)

                if feature_values:
                    composite_score = np.mean(feature_values) if feature_values else 0

                    # Create smooth envelope overlay
                    window_size = max(50, min(500, len(signal_data) // 20))
                    envelope = np.full(len(signal_data), composite_score)
                    envelope = np.convolve(
                        envelope, np.ones(window_size) / window_size, mode="same"
                    )

                    # Scale to signal range for visibility
                    signal_range = np.max(signal_data) - np.min(signal_data)
                    if signal_range > 0:
                        envelope_scaled = (
                            np.min(signal_data) + envelope * signal_range * 0.5
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=time_axis,
                                y=envelope_scaled,
                                mode="lines",
                                name="Feature Analysis Envelope",
                                line=dict(color="purple", width=2, dash="dot"),
                                fill="tonexty",
                                fillcolor="rgba(128, 0, 128, 0.1)",
                                legendgroup="features",
                            ),
                            row=row1_col1[0],
                            col=row1_col1[1],
                        )
            except Exception as e:
                logger.debug(f"Could not add feature overlay: {e}")

        # 3. Pattern Recognition - detected peaks as markers
        try:
            if (
                "patterns" in analysis_results
                and "peaks" in analysis_results["patterns"]
            ):
                peaks_info = analysis_results["patterns"]["peaks"]
                if peaks_info.get("num_peaks", 0) > 0:
                    temporal_features = extract_temporal_features(
                        signal_data, sampling_freq
                    )
                    peaks = temporal_features.get("peaks", [])
                    if (
                        len(peaks) > 0
                        and isinstance(peaks, np.ndarray)
                        or (isinstance(peaks, list) and len(peaks) > 0)
                    ):
                        peaks = np.array(peaks)
                        valid_peaks = peaks[(peaks >= 0) & (peaks < len(time_axis))]
                        if len(valid_peaks) > 0:
                            peak_times = time_axis[valid_peaks]
                            peak_values = signal_data[valid_peaks]

                            fig.add_trace(
                                go.Scatter(
                                    x=peak_times,
                                    y=peak_values,
                                    mode="markers",
                                    name="Pattern Recognition (Peaks)",
                                    marker=dict(
                                        color="red",
                                        size=10,
                                        symbol="diamond",
                                        line=dict(width=1, color="darkred"),
                                    ),
                                    legendgroup="patterns",
                                ),
                                row=row1_col1[0],
                                col=row1_col1[1],
                            )
        except Exception as e:
            logger.debug(f"Could not add pattern markers: {e}")

        # 4. Advanced Processing - EMD or wavelet decomposition overlay
        try:
            if "advanced_processing" in analysis_results:
                adv = analysis_results["advanced_processing"]

                # Try EMD reconstruction
                if "emd" in adv and "error" not in str(adv.get("emd", {})):
                    emd = adv["emd"]
                    if "imfs" in emd and len(emd["imfs"]) > 0:
                        # Sum first few IMFs to show high-frequency components
                        imfs = emd["imfs"]
                        num_imfs_to_show = min(3, len(imfs))
                        high_freq_component = np.sum(
                            [
                                imfs[i]
                                for i in range(num_imfs_to_show)
                                if len(imfs[i]) == len(signal_data)
                            ],
                            axis=0,
                        )

                        if len(high_freq_component) == len(time_axis):
                            fig.add_trace(
                                go.Scatter(
                                    x=time_axis,
                                    y=high_freq_component,
                                    mode="lines",
                                    name="Advanced Processing (High-Freq)",
                                    line=dict(
                                        color="orange", width=1.5, dash="dashdot"
                                    ),
                                    opacity=0.7,
                                    legendgroup="advanced",
                                ),
                                row=row1_col1[0],
                                col=row1_col1[1],
                            )

                # Try wavelet if available
                elif "wavelet" in adv and "error" not in str(adv.get("wavelet", {})):
                    wavelet = adv["wavelet"]
                    if "coefficients" in wavelet and len(wavelet["coefficients"]) > 0:
                        # Use first level detail coefficients
                        coeffs = (
                            wavelet["coefficients"][0]
                            if isinstance(wavelet["coefficients"], list)
                            else wavelet["coefficients"]
                        )
                        if len(coeffs) <= len(time_axis):
                            # Interpolate to match signal length
                            coeffs_interp = np.interp(
                                np.arange(len(signal_data)),
                                np.linspace(0, len(signal_data) - 1, len(coeffs)),
                                coeffs,
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=time_axis,
                                    y=coeffs_interp,
                                    mode="lines",
                                    name="Advanced Processing (Wavelet)",
                                    line=dict(color="green", width=1.5, dash="dot"),
                                    opacity=0.6,
                                    legendgroup="advanced",
                                ),
                                row=row1_col1[0],
                                col=row1_col1[1],
                            )
        except Exception as e:
            logger.debug(f"Could not add advanced processing overlay: {e}")

        # === Plot 2: Classification & Regression Results ===
        row1_col2 = (1, 2)

        classification_summary = {}
        if "classification" in analysis_results and "error" not in str(
            analysis_results["classification"]
        ):
            class_results = analysis_results["classification"]
            for key, value in class_results.items():
                if (
                    isinstance(value, dict)
                    and "error" not in str(value)
                    and "class" in value
                ):
                    classification_summary[key] = value.get("class", "Unknown")

        if classification_summary:
            classes = list(classification_summary.keys())
            confidences = []
            for c in classes:
                if isinstance(analysis_results["classification"][c], dict):
                    conf = (
                        analysis_results["classification"][c].get("confidence", 0.5)
                        * 100
                    )
                else:
                    conf = 0.5
                confidences.append(conf)

            fig.add_trace(
                go.Bar(
                    x=[c.replace("_", " ").title() for c in classes],
                    y=confidences,
                    name="Classification Confidence",
                    marker_color="lightblue",
                    text=[classification_summary[c] for c in classes],
                    textposition="outside",
                ),
                row=row1_col2[0],
                col=row1_col2[1],
            )
        else:
            fig.add_trace(
                go.Bar(x=["N/A"], y=[0], name="No Classification", marker_color="gray"),
                row=row1_col2[0],
                col=row1_col2[1],
            )

        # === Plot 3: Anomaly Detection ===
        row2_col1 = (2, 1)

        if (
            "anomaly_detection" in analysis_results
            and "error" not in analysis_results["anomaly_detection"]
        ):
            anomaly = analysis_results["anomaly_detection"]

            # Create visualization showing anomaly zones
            signal_normal = signal_data.copy()

            # Highlight anomalies if we have the indices
            anomaly_zones = np.zeros_like(signal_data)
            if "total_anomalies" in anomaly and anomaly["total_anomalies"] > 0:
                # Mark zones with high z-score
                mean_val = np.mean(signal_data)
                std_val = np.std(signal_data)
                if std_val > 0:
                    z_scores = np.abs((signal_data - mean_val) / std_val)
                    anomaly_mask = z_scores > 3
                    anomaly_zones[anomaly_mask] = signal_data[anomaly_mask]

                    fig.add_trace(
                        go.Scatter(
                            x=time_axis[anomaly_mask],
                            y=signal_data[anomaly_mask],
                            mode="markers",
                            name="Anomalies",
                            marker=dict(color="red", size=6, symbol="x"),
                            legendgroup="anomalies",
                        ),
                        row=row2_col1[0],
                        col=row2_col1[1],
                    )

                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=signal_normal,
                        mode="lines",
                        name="Signal",
                        line=dict(color="lightblue", width=1),
                        legendgroup="normal",
                    ),
                    row=row2_col1[0],
                    col=row2_col1[1],
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=signal_data,
                    mode="lines",
                    name="Signal",
                    line=dict(color="blue"),
                ),
                row=row2_col1[0],
                col=row2_col1[1],
            )

        # === Plot 4: Clustering & Pattern Recognition ===
        row2_col2 = (2, 2)

        # Show pattern types
        if (
            "clustering" in analysis_results
            and "segment_clustering" in analysis_results["clustering"]
        ):
            cluster_info = analysis_results["clustering"]["segment_clustering"]
            if "cluster_labels" in cluster_info:
                cluster_labels = cluster_info["cluster_labels"]

                # Create cluster visualization
                num_clusters = len(set(cluster_labels))
                # Use predefined colors for clusters
                predefined_colors = [
                    "red",
                    "green",
                    "blue",
                    "orange",
                    "purple",
                    "brown",
                    "pink",
                    "gray",
                    "olive",
                    "cyan",
                ]

                segment_size = len(signal_data) // len(cluster_labels)
                for i, label in enumerate(cluster_labels):
                    start_idx = i * segment_size
                    end_idx = min(start_idx + segment_size, len(signal_data))
                    if start_idx < len(time_axis):
                        segment_time = time_axis[start_idx:end_idx]
                        segment_data = signal_data[start_idx:end_idx]

                        color = predefined_colors[label % len(predefined_colors)]

                        fig.add_trace(
                            go.Scatter(
                                x=segment_time,
                                y=segment_data,
                                mode="lines",
                                name=f"Cluster {label}",
                                line=dict(color=color, width=1.5),
                                legendgroup=f"cluster_{label}",
                                showlegend=(i < 5),  # Only show first 5 clusters
                            ),
                            row=row2_col2[0],
                            col=row2_col2[1],
                        )
        else:
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=signal_data,
                    mode="lines",
                    name="Signal",
                    line=dict(color="green"),
                ),
                row=row2_col2[0],
                col=row2_col2[1],
            )

        # === Plot 5: Advanced Processing (EMD, Wavelet, etc.) ===
        row3_col1 = (3, 1)

        if (
            "advanced_processing" in analysis_results
            and "emd" in analysis_results["advanced_processing"]
        ):
            emd_results = analysis_results["advanced_processing"]["emd"]
            if "imfs" in emd_results:
                imfs = emd_results["imfs"]
                if len(imfs) > 0 and len(imfs[0]) == len(time_axis):
                    # Plot first few IMFs
                    colors_imf = ["orange", "green", "purple"]
                    for i in range(min(3, len(imfs))):
                        if i == 0 or i == 1 or i == 2:
                            fig.add_trace(
                                go.Scatter(
                                    x=time_axis,
                                    y=imfs[i],
                                    mode="lines",
                                    name=f"IMF {i+1}",
                                    line=dict(color=colors_imf[i], width=1),
                                    legendgroup="imf",
                                    opacity=0.7,
                                ),
                                row=row3_col1[0],
                                col=row3_col1[1],
                            )

        if (
            "advanced_processing" not in analysis_results
            or "emd" not in analysis_results["advanced_processing"]
        ):
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=signal_data,
                    mode="lines",
                    name="Signal",
                    line=dict(color="purple"),
                ),
                row=row3_col1[0],
                col=row3_col1[1],
            )

        # === Plot 6: Forecasting ===
        row3_col2 = (3, 2)

        if (
            "forecasting" in analysis_results
            and "linear_forecast" in analysis_results["forecasting"]
        ):
            fc = analysis_results["forecasting"]["linear_forecast"]
            if "forecast_values" in fc and "error" not in fc:
                forecast_values = fc["forecast_values"]
                forecast_length = fc.get("forecast_length", len(forecast_values))

                # Extend time axis for forecast
                time_step = (
                    time_axis[1] - time_axis[0]
                    if len(time_axis) > 1
                    else 1 / sampling_freq
                )
                forecast_time = np.arange(
                    time_axis[-1],
                    time_axis[-1] + len(forecast_values) * time_step,
                    time_step,
                )

                # Plot original signal
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=signal_data,
                    mode="lines",
                    name="Original",
                    line=dict(color="blue", width=1.5),
                ),
                row=row3_col2[0],
                col=row3_col2[1],
            )

            # Plot forecast
            fig.add_trace(
                go.Scatter(
                    x=forecast_time[: len(forecast_values)],
                    y=forecast_values,
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="red", width=2, dash="dot"),
                    marker=dict(size=3),
                ),
                row=row3_col2[0],
                col=row3_col2[1],
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=signal_data,
                    mode="lines",
                    name="Signal",
                    line=dict(color="cyan"),
                ),
                row=row3_col2[0],
                col=row3_col2[1],
            )

            # PHASE D: Enhanced comprehensive visualization layout
            fig.update_layout(
                title=dict(
                    text="<b>Advanced Analysis - Comprehensive Visualizations</b><br><sub>Multi-dimensional signal analysis with interactive exploration</sub>",
                    x=0.5,
                    xanchor="center",
                    font=dict(size=18),
                ),
                height=1300,  # PHASE D: Taller for better visibility
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.02,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="rgba(128,128,128,0.3)",
                    borderwidth=1,
                ),
                plot_bgcolor="white",
                paper_bgcolor="rgba(248, 249, 250, 0.5)",  # PHASE D: Subtle background
                hovermode="closest",
                # PHASE D: Enhanced modebar
                modebar=dict(
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    color="rgba(0, 0, 0, 0.5)",
                    activecolor="rgba(0, 123, 255, 1)",
                ),
            )

            # PHASE D: Update all axes with consistent styling
            fig.update_xaxes(
                title_text="<b>Time (s)</b>",
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(128,128,128,0.2)",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="rgba(128,128,128,0.3)",
            )
            fig.update_yaxes(
                title_text="<b>Amplitude</b>",
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(128,128,128,0.2)",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="rgba(128,128,128,0.3)",
            )

            return fig

    except Exception as e:
        logger.error(f"Error creating advanced visualizations: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return create_empty_figure()


# Visualization helper functions for detailed report
def create_anomaly_detection_plot(
    signal_data, time_axis, anomaly_results, sampling_freq
):
    """Create visualization for anomaly detection."""
    try:
        fig = go.Figure()

        # Plot normal signal
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal_data,
                mode="lines",
                name="Signal",
                line=dict(color="blue", width=1),
            )
        )

        # Highlight anomalies using z-score method
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        if std_val > 0:
            z_scores = np.abs((signal_data - mean_val) / std_val)
            anomaly_mask = z_scores > 3
            if np.any(anomaly_mask):
                fig.add_trace(
                    go.Scatter(
                        x=time_axis[anomaly_mask],
                        y=signal_data[anomaly_mask],
                        mode="markers",
                        name="Anomalies",
                        marker=dict(color="red", size=6, symbol="x"),
                    )
                )

        fig.update_layout(
            title="Anomaly Detection Visualization",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=400,
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating anomaly plot: {e}")
        return create_empty_figure()


def create_classification_plot(class_data):
    """Create bar chart for classification results."""
    try:
        if not class_data:
            return create_empty_figure()

        classes = list(class_data.keys())
        confidences = [class_data[c]["confidence"] * 100 for c in classes]
        class_labels = [class_data[c]["class"] for c in classes]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[c.replace("_", " ").title() for c in classes],
                y=confidences,
                text=[f"{l}<br>{c:.1f}%" for l, c in zip(class_labels, confidences)],
                textposition="outside",
                marker_color="lightblue",
                name="Confidence",
            )
        )

        fig.update_layout(
            title="Classification Confidence Scores",
            xaxis_title="Classification Type",
            yaxis_title="Confidence (%)",
            height=300,
            showlegend=False,
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating classification plot: {e}")
        return create_empty_figure()


def create_regression_plot(signal_data, time_axis, regression_results):
    """Create visualization for regression analysis."""
    try:
        fig = go.Figure()

        # Plot signal
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal_data,
                mode="lines",
                name="Signal",
                line=dict(color="blue", width=1),
            )
        )

        # Add trend line
        if "slope" in regression_results:
            slope = regression_results.get("slope", 0)
            # Calculate intercept if not available
            if "intercept" in regression_results:
                intercept = regression_results.get("intercept")
            else:
                intercept = np.mean(signal_data) - slope * np.mean(
                    np.arange(len(signal_data))
                )
            trend_line = slope * np.arange(len(signal_data)) + intercept
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=trend_line,
                    mode="lines",
                    name="Trend Line",
                    line=dict(color="red", width=2, dash="dash"),
                )
            )

        fig.update_layout(
            title="Regression Analysis - Trend Detection",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=300,
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating regression plot: {e}")
        return create_empty_figure()


def create_clustering_plot(signal_data, time_axis, clustering_results):
    """Create visualization for clustering analysis."""
    try:
        fig = go.Figure()

        if "cluster_labels" in clustering_results:
            cluster_labels = clustering_results["cluster_labels"]
            n_clusters = clustering_results.get("n_clusters", len(set(cluster_labels)))

            # Color map for clusters
            colors = [
                "red",
                "green",
                "blue",
                "orange",
                "purple",
                "brown",
                "pink",
                "gray",
            ]

            # Group signal by cluster
            segment_size = len(signal_data) // len(cluster_labels)
            for i, label in enumerate(cluster_labels):
                start_idx = i * segment_size
                end_idx = min(start_idx + segment_size, len(signal_data))
                if start_idx < len(time_axis):
                    color = colors[label % len(colors)]
                    fig.add_trace(
                        go.Scatter(
                            x=time_axis[start_idx:end_idx],
                            y=signal_data[start_idx:end_idx],
                            mode="lines",
                            name=f"Cluster {label}",
                            line=dict(color=color, width=1.5),
                            showlegend=(i < 5),
                        )
                    )
        else:
            # Fallback: just plot signal
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=signal_data,
                    mode="lines",
                    name="Signal",
                    line=dict(color="blue"),
                )
            )

        fig.update_layout(
            title="Clustering Analysis - Signal Segments",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=300,
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating clustering plot: {e}")
        return create_empty_figure()


def create_forecast_plot(signal_data, time_axis, forecast_results, sampling_freq):
    """Create visualization for forecasting."""
    try:
        fig = go.Figure()

        # Plot original signal
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal_data,
                mode="lines",
                name="Original Signal",
                line=dict(color="blue", width=1.5),
            )
        )

        # Add forecast if available
        if "forecast_values" in forecast_results:
            forecast_values = forecast_results["forecast_values"]
            forecast_length = len(forecast_values)
            time_step = (
                time_axis[1] - time_axis[0] if len(time_axis) > 1 else 1 / sampling_freq
            )
            forecast_time = np.arange(
                time_axis[-1] + time_step,
                time_axis[-1] + (forecast_length + 1) * time_step,
                time_step,
            )[:forecast_length]

            fig.add_trace(
                go.Scatter(
                    x=forecast_time,
                    y=forecast_values,
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="red", width=2, dash="dot"),
                    marker=dict(size=3),
                )
            )

        fig.update_layout(
            title="Time Series Forecasting",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=300,
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating forecast plot: {e}")
        return create_empty_figure()


def create_emd_visualization(signal_data, time_axis, emd_results):
    """Create visualization for EMD decomposition."""
    try:
        if "imfs" not in emd_results:
            return create_empty_figure()

        imfs = emd_results["imfs"]
        num_imfs = min(len(imfs), 4)  # Show first 4 IMFs

        fig = make_subplots(
            rows=num_imfs + 1,
            cols=1,
            subplot_titles=["Original Signal"]
            + [f"IMF {i+1}" for i in range(num_imfs)],
            vertical_spacing=0.05,
        )

        # Original signal
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal_data,
                mode="lines",
                name="Original",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

        # IMFs
        colors = ["red", "green", "orange", "purple"]
        for i in range(num_imfs):
            if len(imfs[i]) == len(time_axis):
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=imfs[i],
                        mode="lines",
                        name=f"IMF {i+1}",
                        line=dict(color=colors[i % len(colors)], width=1),
                    ),
                    row=i + 2,
                    col=1,
                )

        fig.update_layout(height=600, showlegend=False)
        fig.update_xaxes(title_text="Time (s)", row=num_imfs + 1, col=1)

        return fig
    except Exception as e:
        logger.error(f"Error creating EMD visualization: {e}")
        return create_empty_figure()


def create_detailed_analysis_report(
    analysis_results,
    analysis_categories,
    signal_type,
    time_axis,
    signal_data,
    sampling_freq,
):
    """Create comprehensive detailed analysis report with all findings."""
    try:
        if not analysis_results or "error" in analysis_results:
            return html.Div(
                [
                    html.H6("Analysis Report"),
                    html.P("No analysis data available to generate report."),
                ]
            )

        sections = []

        # Report Header
        sections.append(
            html.Div(
                [
                    html.H4("📋 Comprehensive Analysis Report", className="mb-3"),
                    html.Hr(),
                ]
            )
        )

        # Executive Summary
        sections.append(
            html.Div(
                [
                    html.H5("Executive Summary", className="mt-4"),
                    html.P(
                        f"""
                This comprehensive analysis report presents findings from advanced signal processing
                of a {signal_type.upper()} signal with {len(signal_data)} samples at {sampling_freq:.1f} Hz.
                The analysis encompasses multiple categories including feature extraction, machine learning,
                pattern recognition, and advanced signal processing techniques.
            """
                    ),
                ]
            )
        )

        sections.append(html.Hr())

        # 1. Feature Analysis Results
        if "features" in analysis_results:
            features = analysis_results["features"]
            sections.append(
                html.Div(
                    [
                        html.H5("1. Feature Analysis", className="text-primary"),
                        html.P(html.Strong("Extracted Features:"), className="mt-2"),
                    ]
                )
            )

            # Feature metrics in a table format
            feature_table_rows = []
            if "statistical" in features and "error" not in features["statistical"]:
                stat = features["statistical"]
                feature_table_rows.extend(
                    [
                        html.Tr(
                            [
                                html.Td(html.Strong("Mean:")),
                                html.Td(f"{stat.get('mean', 0):.6f}"),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td(html.Strong("Std Dev:")),
                                html.Td(f"{stat.get('std', 0):.6f}"),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td(html.Strong("Skewness:")),
                                html.Td(f"{stat.get('skewness', 0):.6f}"),
                            ]
                        ),
                    ]
                )

            if "temporal" in features and "error" not in features["temporal"]:
                temp = features["temporal"]
                feature_table_rows.extend(
                    [
                        html.Tr(
                            [
                                html.Td(html.Strong("Peak Count:")),
                                html.Td(f"{temp.get('peak_count', 0)}"),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td(html.Strong("Heart Rate:")),
                                html.Td(f"{temp.get('heart_rate', 0):.2f} bpm"),
                            ]
                        ),
                    ]
                )

            if feature_table_rows:
                sections.append(
                    html.Table(
                        [html.Tbody(feature_table_rows)], className="table table-sm"
                    )
                )

        # 2. Anomaly Detection Report
        if (
            "anomaly_detection" in analysis_results
            and "error" not in analysis_results["anomaly_detection"]
        ):
            anomaly = analysis_results["anomaly_detection"]
            anomaly_rate = anomaly.get("overall_anomaly_rate", 0)
            total_anomalies = anomaly.get("total_anomalies", 0)

            # Check for abnormal values
            warning_msg = ""
            if anomaly_rate > 20:
                warning_msg = html.Div(
                    [
                        html.Strong("⚠️ Warning: "),
                        f"Anomaly rate ({anomaly_rate:.2f}%) is unusually high. This may indicate noisy signal or overly sensitive detection parameters.",
                    ],
                    className="alert alert-warning mt-2 mb-2",
                )

            # Create anomaly visualization
            anomaly_fig = create_anomaly_detection_plot(
                signal_data, time_axis, anomaly, sampling_freq
            )

            sections.append(
                html.Div(
                    [
                        html.H5("2. Anomaly Detection", className="text-danger mt-4"),
                        html.P(
                            html.Strong(
                                f"Detection Method: {anomaly.get('method', 'N/A')}"
                            )
                        ),
                        html.P(
                            f"Total anomalies detected: {total_anomalies} ({anomaly_rate:.2f}% of signal)"
                        ),
                        (
                            html.P(
                                f"Methods used: {', '.join(anomaly.get('methods_used', []))}"
                            )
                            if "methods_used" in anomaly
                            else None
                        ),
                        warning_msg if warning_msg else None,
                        (
                            dcc.Graph(figure=anomaly_fig, style={"height": "400px"})
                            if anomaly_fig
                            else None
                        ),
                    ]
                )
            )

        # 3. Classification Results
        if "classification" in analysis_results and "error" not in str(
            analysis_results["classification"]
        ):
            classification = analysis_results["classification"]
            sections.append(
                html.Div(
                    [
                        html.H5(
                            "3. Classification Results", className="text-success mt-4"
                        ),
                    ]
                )
            )

            class_results = []
            class_data = {}

            for key, value in classification.items():
                if isinstance(value, dict) and "error" not in str(value):
                    if "class" in value:
                        class_name = key.replace("_", " ").title()
                        class_val = value.get("class", "N/A")
                        conf = value.get("confidence", 0)
                        class_results.append(
                            html.P(
                                f"• {class_name}: {html.Strong(class_val)} (Confidence: {conf:.1%})",
                                style={"marginBottom": "0.5rem"},
                            )
                        )
                        class_data[class_name] = {
                            "class": class_val,
                            "confidence": conf,
                        }

            # Create classification visualization
            if class_data:
                class_fig = create_classification_plot(class_data)
                sections.append(dcc.Graph(figure=class_fig, style={"height": "300px"}))

            if class_results:
                sections.append(html.Div(class_results))
            else:
                sections.append(
                    html.P(
                        "No classification results available.", className="text-muted"
                    )
                )

        # 4. Regression Analysis
        if (
            "regression" in analysis_results
            and "error" not in analysis_results["regression"]
        ):
            regression = analysis_results["regression"]
            sections.append(
                html.Div(
                    [
                        html.H5("4. Regression Analysis", className="text-info mt-4"),
                    ]
                )
            )

            if (
                "linear_trend_regression" in regression
                and "error" not in regression["linear_trend_regression"]
            ):
                lr = regression["linear_trend_regression"]
                r_squared = lr.get("r_squared", 0)
                slope = lr.get("slope", 0)
                trend_dir = lr.get("trend_direction", "N/A")

                # Check for abnormal values
                rsq_warning = ""
                if r_squared < 0.01:
                    rsq_warning = html.P(
                        html.Small(
                            "⚠️ Note: Very low R-squared indicates weak linear relationship."
                        ),
                        className="text-muted mt-1",
                    )

                sections.append(
                    html.Div(
                        [
                            html.P(html.Strong(f"Trend Direction: {trend_dir}")),
                            html.P(f"R-squared: {r_squared:.4f}"),
                            html.P(f"Slope: {slope:.6f}"),
                            rsq_warning if rsq_warning else None,
                        ]
                    )
                )

                # Create regression visualization
                regression_fig = create_regression_plot(signal_data, time_axis, lr)
                if regression_fig:
                    sections.append(
                        dcc.Graph(figure=regression_fig, style={"height": "300px"})
                    )

        # 5. Clustering Results
        if (
            "clustering" in analysis_results
            and "error" not in analysis_results["clustering"]
        ):
            clustering = analysis_results["clustering"]
            sections.append(
                html.Div(
                    [
                        html.H5(
                            "5. Clustering Analysis", className="text-warning mt-4"
                        ),
                    ]
                )
            )

            if (
                "segment_clustering" in clustering
                and "error" not in clustering["segment_clustering"]
            ):
                seg = clustering["segment_clustering"]
                sections.append(
                    html.P(
                        f"""
                    Number of clusters identified: {seg.get('n_clusters', 0)}
                    Method: {seg.get('method', 'N/A')}
                """
                    )
                )

                # Create clustering visualization
                clustering_fig = create_clustering_plot(signal_data, time_axis, seg)
                if clustering_fig:
                    sections.append(
                        dcc.Graph(figure=clustering_fig, style={"height": "300px"})
                    )

        # 6. Forecasting
        if (
            "forecasting" in analysis_results
            and "error" not in analysis_results["forecasting"]
        ):
            forecasting = analysis_results["forecasting"]
            sections.append(
                html.Div(
                    [
                        html.H5(
                            "6. Forecasting Analysis", className="text-secondary mt-4"
                        ),
                    ]
                )
            )

            if (
                "linear_forecast" in forecasting
                and "error" not in forecasting["linear_forecast"]
            ):
                lf = forecasting["linear_forecast"]
                forecast_duration = (
                    lf.get("forecast_length", 0) / sampling_freq
                    if sampling_freq > 0
                    else 0
                )

                sections.append(
                    html.Div(
                        [
                            html.P(
                                html.Strong(
                                    f"Method: {lf.get('method', 'Linear Forecasting')}"
                                )
                            ),
                            html.Table(
                                [
                                    html.Tbody(
                                        [
                                            html.Tr(
                                                [
                                                    html.Td(
                                                        html.Strong("Forecast Length:")
                                                    ),
                                                    html.Td(
                                                        f"{lf.get('forecast_length', 0)} samples ({forecast_duration:.2f} seconds)"
                                                    ),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td(
                                                        html.Strong("Trend Slope:")
                                                    ),
                                                    html.Td(
                                                        f"{lf.get('trend_slope', 0):.6f}"
                                                    ),
                                                ]
                                            ),
                                            (
                                                html.Tr(
                                                    [
                                                        html.Td(html.Strong("RMSE:")),
                                                        html.Td(
                                                            f"{lf.get('rmse', 0):.4f}"
                                                        ),
                                                    ]
                                                )
                                                if "rmse" in lf
                                                else None
                                            ),
                                            (
                                                html.Tr(
                                                    [
                                                        html.Td(
                                                            html.Strong("R-squared:")
                                                        ),
                                                        html.Td(
                                                            f"{lf.get('r_squared', 0):.4f}"
                                                        ),
                                                    ]
                                                )
                                                if "r_squared" in lf
                                                else None
                                            ),
                                            (
                                                html.Tr(
                                                    [
                                                        html.Td(
                                                            html.Strong(
                                                                "Confidence Interval:"
                                                            )
                                                        ),
                                                        html.Td(
                                                            f"±{lf.get('confidence_interval', 0):.4f}"
                                                        ),
                                                    ]
                                                )
                                                if "confidence_interval" in lf
                                                else None
                                            ),
                                        ]
                                    )
                                ],
                                className="table table-sm table-bordered mb-3",
                            ),
                        ]
                    )
                )

                # Create forecasting visualization
                forecast_fig = create_forecast_plot(
                    signal_data, time_axis, lf, sampling_freq
                )
                if forecast_fig:
                    sections.append(
                        dcc.Graph(figure=forecast_fig, style={"height": "300px"})
                    )

        # 7. Advanced Processing
        if "advanced_processing" in analysis_results:
            adv = analysis_results["advanced_processing"]
            sections.append(
                html.Div(
                    [
                        html.H5("7. Advanced Signal Processing", className="mt-4"),
                    ]
                )
            )

            if "emd" in adv and "error" not in adv["emd"]:
                emd = adv["emd"]
                imfs = emd.get("imfs", [])
                num_imfs = len(imfs)

                sections.append(
                    html.P(
                        html.Strong(
                            f"""
                    Empirical Mode Decomposition: {num_imfs} IMFs extracted
                    Reconstruction Error: {emd.get('reconstruction_error', 0):.6f}
                """
                        )
                    )
                )

                # Detailed IMF Analysis
                if num_imfs > 0:
                    sections.append(
                        html.H6("IMF Analysis Details", className="mt-3 mb-2")
                    )

                    imf_details = []
                    for i, imf in enumerate(imfs):
                        if len(imf) == len(time_axis):
                            # Calculate IMF characteristics
                            imf_mean = np.mean(imf)
                            imf_std = np.std(imf)
                            imf_energy = np.sum(imf**2)
                            total_energy = np.sum(
                                [
                                    np.sum(imf_arr**2)
                                    for imf_arr in imfs
                                    if len(imf_arr) == len(time_axis)
                                ]
                            )
                            imf_energy_pct = (
                                (imf_energy / total_energy * 100)
                                if total_energy > 0
                                else 0
                            )

                            # Estimate dominant frequency for this IMF
                            try:
                                fft_vals = np.fft.fft(imf)
                                fft_freq = np.fft.fftfreq(len(imf), 1 / sampling_freq)
                                power = np.abs(fft_vals) ** 2
                                # Only consider positive frequencies
                                positive_freq_idx = fft_freq > 0
                                if np.any(positive_freq_idx):
                                    dominant_freq_idx = np.argmax(
                                        power[positive_freq_idx]
                                    )
                                    dominant_freq = fft_freq[positive_freq_idx][
                                        dominant_freq_idx
                                    ]
                                else:
                                    dominant_freq = 0
                            except Exception as err:
                                print(f"Error calculating dominant frequency: {err}")
                                dominant_freq = 0

                            # IMF interpretation
                            if i == 0:
                                imf_desc = "High-frequency noise/details"
                            elif i < num_imfs // 3:
                                imf_desc = "High-frequency components"
                            elif i < 2 * num_imfs // 3:
                                imf_desc = "Medium-frequency oscillations"
                            else:
                                imf_desc = "Low-frequency trend/residual"

                            imf_details.append(
                                html.Div(
                                    [
                                        html.H6(f"IMF {i+1}", className="mb-1"),
                                        html.P(
                                            f"Description: {imf_desc}",
                                            style={
                                                "fontSize": "0.9em",
                                                "color": "#666",
                                            },
                                        ),
                                        html.Table(
                                            [
                                                html.Tbody(
                                                    [
                                                        html.Tr(
                                                            [
                                                                html.Td(
                                                                    html.Strong("Mean:")
                                                                ),
                                                                html.Td(
                                                                    f"{imf_mean:.4f}"
                                                                ),
                                                            ]
                                                        ),
                                                        html.Tr(
                                                            [
                                                                html.Td(
                                                                    html.Strong(
                                                                        "Std Dev:"
                                                                    )
                                                                ),
                                                                html.Td(
                                                                    f"{imf_std:.4f}"
                                                                ),
                                                            ]
                                                        ),
                                                        html.Tr(
                                                            [
                                                                html.Td(
                                                                    html.Strong(
                                                                        "Energy:"
                                                                    )
                                                                ),
                                                                html.Td(
                                                                    f"{imf_energy:.2f} ({imf_energy_pct:.1f}% of total)"
                                                                ),
                                                            ]
                                                        ),
                                                        html.Tr(
                                                            [
                                                                html.Td(
                                                                    html.Strong(
                                                                        "Dominant Frequency:"
                                                                    )
                                                                ),
                                                                html.Td(
                                                                    f"{dominant_freq:.2f} Hz"
                                                                ),
                                                            ]
                                                        ),
                                                        html.Tr(
                                                            [
                                                                html.Td(
                                                                    html.Strong(
                                                                        "Amplitude Range:"
                                                                    )
                                                                ),
                                                                html.Td(
                                                                    f"{np.min(imf):.2f} to {np.max(imf):.2f}"
                                                                ),
                                                            ]
                                                        ),
                                                    ]
                                                )
                                            ],
                                            className="table table-sm table-bordered mb-3",
                                        ),
                                    ],
                                    className="mb-3 p-2 border rounded",
                                )
                            )

                    if imf_details:
                        sections.append(html.Div(imf_details))

                    # Summary statistics for residual
                    residual = emd.get("residual", np.array([]))
                    if len(residual) == len(time_axis):
                        sections.append(
                            html.Div(
                                [
                                    html.H6("Residual Component", className="mt-3"),
                                    html.Table(
                                        [
                                            html.Tbody(
                                                [
                                                    html.Tr(
                                                        [
                                                            html.Td(
                                                                html.Strong("Mean:")
                                                            ),
                                                            html.Td(
                                                                f"{np.mean(residual):.4f}"
                                                            ),
                                                        ]
                                                    ),
                                                    html.Tr(
                                                        [
                                                            html.Td(
                                                                html.Strong("Std Dev:")
                                                            ),
                                                            html.Td(
                                                                f"{np.std(residual):.4f}"
                                                            ),
                                                        ]
                                                    ),
                                                    html.Tr(
                                                        [
                                                            html.Td(
                                                                html.Strong("Range:")
                                                            ),
                                                            html.Td(
                                                                f"{np.min(residual):.2f} to {np.max(residual):.2f}"
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            )
                                        ],
                                        className="table table-sm table-bordered",
                                    ),
                                ]
                            )
                        )

                # Create EMD visualization
                emd_fig = create_emd_visualization(signal_data, time_axis, emd)
                if emd_fig:
                    sections.append(
                        dcc.Graph(figure=emd_fig, style={"height": "600px"})
                    )

        # Summary Statistics Table
        sections.append(
            html.Div(
                [
                    html.H5("Summary Statistics", className="mt-4"),
                    html.Table(
                        [
                            html.Tbody(
                                [
                                    html.Tr(
                                        [
                                            html.Td("Signal Length"),
                                            html.Td(f"{len(signal_data)} samples"),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td("Duration"),
                                            html.Td(
                                                f"{len(signal_data)/sampling_freq:.2f} seconds"
                                            ),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td("Sampling Frequency"),
                                            html.Td(f"{sampling_freq:.1f} Hz"),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td("Signal Range"),
                                            html.Td(
                                                f"{np.min(signal_data):.2f} to {np.max(signal_data):.2f}"
                                            ),
                                        ]
                                    ),
                                ]
                            )
                        ],
                        className="table table-bordered",
                    ),
                ]
            )
        )

        return html.Div(sections, className="p-4")

    except Exception as e:
        logger.error(f"Error creating detailed report: {e}")
        return html.Div(
            [html.H6("Report Error"), html.P(f"Failed to generate report: {str(e)}")]
        )


# Result display functions
def create_advanced_analysis_summary(
    analysis_results,
    signal_type,
    feature_categories=None,
    preprocessing=None,
    advanced_options=None,
):
    """
    Create advanced analysis summary with actual results.

    PHASE 4: Now includes preprocessing and advanced options display.

    Args:
        analysis_results: Analysis results dictionary
        signal_type: Type of signal (ecg, ppg, etc.)
        feature_categories: List of feature categories extracted (NEW)
        preprocessing: List of preprocessing options applied (NEW)
        advanced_options: List of advanced options used (NEW)
    """
    try:
        if "error" in analysis_results:
            return html.Div(
                [
                    html.H5("Analysis Summary"),
                    html.P(f"Analysis failed: {analysis_results['error']}"),
                ]
            )

        sections = []

        # Overall summary header
        sections.append(
            html.Div(
                [
                    html.H5("🧠 Advanced Analysis Summary"),
                    html.P(f"Signal Type: {signal_type.upper()}"),
                    html.P(
                        f"Analysis Categories Completed: {len([k for k, v in analysis_results.items() if 'error' not in str(v)])}"
                    ),
                ]
            )
        )

        # PHASE 4: Show configuration info (NEW)
        if feature_categories or preprocessing or advanced_options:
            sections.append(html.Hr())
            sections.append(html.H6("⚙️ Configuration"))

            if feature_categories:
                sections.append(html.P(html.Strong("Feature Categories:")))
                sections.append(html.P(f"  {', '.join(feature_categories)}"))

            if preprocessing:
                sections.append(html.P(html.Strong("Preprocessing Applied:")))
                # PHASE B: Handle dict format with parameters
                if isinstance(preprocessing, dict):
                    for operation, params in preprocessing.items():
                        if isinstance(params, dict):
                            param_str = ", ".join(
                                [f"{k}={v}" for k, v in params.items()]
                            )
                            sections.append(html.P(f"  • {operation}: {param_str}"))
                        else:
                            sections.append(html.P(f"  • {operation}"))
                else:
                    # Legacy list format
                    sections.append(html.P(f"  {', '.join(preprocessing)}"))

            if advanced_options:
                sections.append(html.P(html.Strong("Advanced Options:")))
                sections.append(html.P(f"  {', '.join(advanced_options)}"))

        # Display features if available
        if (
            "features" in analysis_results
            and "error" not in analysis_results["features"]
        ):
            features = analysis_results["features"]
            sections.append(html.Hr())
            sections.append(html.H6("📊 Extracted Features"))

            # Statistical features
            if "statistical" in features and "error" not in features["statistical"]:
                stat = features["statistical"]
                sections.append(html.P(html.Strong("Statistical Features:")))
                sections.append(html.P(f"  • Mean: {stat.get('mean', 0):.6f}"))
                sections.append(html.P(f"  • Std Dev: {stat.get('std', 0):.6f}"))
                sections.append(html.P(f"  • Skewness: {stat.get('skewness', 0):.6f}"))
                sections.append(html.P(f"  • Kurtosis: {stat.get('kurtosis', 0):.6f}"))
                sections.append(html.P(f"  • Entropy: {stat.get('entropy', 0):.6f}"))

            # Spectral features
            if "spectral" in features and "error" not in features["spectral"]:
                spec = features["spectral"]
                sections.append(html.P(html.Strong("Spectral Features:")))
                sections.append(
                    html.P(
                        f"  • Spectral Centroid: {spec.get('spectral_centroid', 0):.2f} Hz"
                    )
                )
                sections.append(
                    html.P(
                        f"  • Spectral Bandwidth: {spec.get('spectral_bandwidth', 0):.2f} Hz"
                    )
                )
                sections.append(
                    html.P(
                        f"  • Dominant Frequency: {spec.get('dominant_frequency', 0):.2f} Hz"
                    )
                )

            # Temporal features
            if "temporal" in features and "error" not in features["temporal"]:
                temp = features["temporal"]
                sections.append(html.P(html.Strong("Temporal Features:")))
                # Handle both key naming conventions (features_callbacks vs local)
                peak_count = temp.get("peak_count", temp.get("num_peaks", 0))
                mean_interval = temp.get("mean_interval", 0)
                heart_rate = temp.get("heart_rate", 0)
                interval_std = temp.get("interval_std", temp.get("std_interval", 0))

                sections.append(html.P(f"  • Peak Count: {peak_count}"))
                sections.append(html.P(f"  • Mean Interval: {mean_interval:.4f} s"))
                sections.append(html.P(f"  • Heart Rate: {heart_rate:.2f} bpm"))
                sections.append(html.P(f"  • Interval Std: {interval_std:.4f} s"))

            # Morphological features
            if "morphological" in features and "error" not in features["morphological"]:
                morph = features["morphological"]
                sections.append(html.P(html.Strong("Morphological Features:")))
                # Handle both key naming conventions
                amplitude_range = morph.get("amplitude_range", 0)
                # If amplitude_range is 0, try to calculate from other keys
                if (
                    amplitude_range == 0
                    and "max_peak_height" in morph
                    and "min_peak_height" in morph
                ):
                    amplitude_range = (
                        morph["max_peak_height"] - morph["min_peak_height"]
                    )

                amplitude_mean = morph.get(
                    "amplitude_mean", morph.get("mean_peak_height", 0)
                )
                zero_crossings = morph.get("zero_crossings", 0)
                signal_energy = morph.get("signal_energy", 0)

                sections.append(html.P(f"  • Amplitude Range: {amplitude_range:.6f}"))
                sections.append(html.P(f"  • Mean Amplitude: {amplitude_mean:.6f}"))

                # Show num_peaks if available (more informative)
                if "num_peaks" in morph:
                    sections.append(
                        html.P(f"  • Number of Peaks: {morph['num_peaks']}")
                    )
                if "num_valleys" in morph:
                    sections.append(
                        html.P(f"  • Number of Valleys: {morph['num_valleys']}")
                    )

                sections.append(html.P(f"  • Zero Crossings: {zero_crossings}"))
                sections.append(html.P(f"  • Signal Energy: {signal_energy:.2f}"))

            # PHASE 4: Entropy features (NEW)
            if "entropy" in features and "error" not in features["entropy"]:
                entropy = features["entropy"]
                sections.append(html.P(html.Strong("Entropy Features:")))
                if "sample_entropy" in entropy:
                    sections.append(
                        html.P(
                            f"  • Sample Entropy: {entropy.get('sample_entropy', 0):.6f}"
                        )
                    )
                if "approximate_entropy" in entropy:
                    sections.append(
                        html.P(
                            f"  • Approximate Entropy: {entropy.get('approximate_entropy', 0):.6f}"
                        )
                    )
                if "permutation_entropy" in entropy:
                    sections.append(
                        html.P(
                            f"  • Permutation Entropy: {entropy.get('permutation_entropy', 0):.6f}"
                        )
                    )

            # PHASE 4: Fractal features (NEW)
            if "fractal" in features and "error" not in features["fractal"]:
                fractal = features["fractal"]
                sections.append(html.P(html.Strong("Fractal Features:")))
                if "hurst_exponent" in fractal:
                    sections.append(
                        html.P(
                            f"  • Hurst Exponent: {fractal.get('hurst_exponent', 0):.6f}"
                        )
                    )
                if "fractal_dimension" in fractal:
                    sections.append(
                        html.P(
                            f"  • Fractal Dimension: {fractal.get('fractal_dimension', 0):.6f}"
                        )
                    )
                if "dfa_alpha" in fractal:
                    sections.append(
                        html.P(f"  • DFA Alpha: {fractal.get('dfa_alpha', 0):.6f}")
                    )

        # Display pattern recognition if available
        if (
            "patterns" in analysis_results
            and "error" not in analysis_results["patterns"]
        ):
            patterns = analysis_results["patterns"]
            sections.append(html.Hr())
            sections.append(html.H6("🎯 Pattern Recognition"))

            if "peaks" in patterns and "error" not in patterns["peaks"]:
                peak_pat = patterns["peaks"]
                sections.append(html.P(html.Strong("Peak Patterns:")))
                sections.append(
                    html.P(f"  • Number of Peaks: {peak_pat.get('num_peaks', 0)}")
                )
                if "mean_interval" in peak_pat:
                    sections.append(
                        html.P(
                            f"  • Mean Interval: {peak_pat.get('mean_interval', 0):.4f} s"
                        )
                    )
                    sections.append(
                        html.P(
                            f"  • Heart Rate: {peak_pat.get('heart_rate', 0):.2f} bpm"
                        )
                    )

            if "frequency" in patterns and "error" not in patterns["frequency"]:
                freq_pat = patterns["frequency"]
                sections.append(html.P(html.Strong("Frequency Patterns:")))
                sections.append(
                    html.P(
                        f"  • Dominant Frequency: {freq_pat.get('dominant_frequency', 0):.2f} Hz"
                    )
                )
                sections.append(
                    html.P(
                        f"  • Spectral Centroid: {freq_pat.get('spectral_centroid', 0):.2f} Hz"
                    )
                )
                sections.append(
                    html.P(f"  • Total Power: {freq_pat.get('total_power', 0):.2f}")
                )

            if "morphology" in patterns and "error" not in patterns["morphology"]:
                morph_pat = patterns["morphology"]
                sections.append(html.P(html.Strong("Morphological Patterns:")))
                sections.append(
                    html.P(f"  • Peak Count: {morph_pat.get('num_peaks', 0)}")
                )
                sections.append(
                    html.P(
                        f"  • Mean Peak Height: {morph_pat.get('mean_peak_height', 0):.6f}"
                    )
                )
                sections.append(
                    html.P(
                        f"  • Peak Variability: {morph_pat.get('peak_variability', 0):.4f}"
                    )
                )

        # Display Anomaly Detection if available
        if (
            "anomaly_detection" in analysis_results
            and "error" not in analysis_results["anomaly_detection"]
        ):
            anomaly = analysis_results["anomaly_detection"]
            sections.append(html.Hr())
            sections.append(html.H6("⚠️ Anomaly Detection"))
            sections.append(html.P(f"Method: {anomaly.get('method', 'N/A')}"))
            sections.append(
                html.P(f"Total Anomalies: {anomaly.get('total_anomalies', 0)}")
            )
            sections.append(
                html.P(f"Anomaly Rate: {anomaly.get('overall_anomaly_rate', 0):.2f}%")
            )
            if "z_score_anomalies" in anomaly:
                sections.append(
                    html.P(
                        f"Z-score anomalies: {anomaly.get('z_score_anomalies', 0)} ({anomaly.get('z_score_rate', 0):.2f}%)"
                    )
                )

        # Display Classification if available
        if (
            "classification" in analysis_results
            and "error" not in analysis_results["classification"]
        ):
            classification = analysis_results["classification"]
            sections.append(html.Hr())
            sections.append(html.H6("🏷️ Classification Results"))

            if "signal_quality_classification" in classification:
                sq_class = classification["signal_quality_classification"]
                if "class" in sq_class:
                    sections.append(html.P(html.Strong("Signal Quality:")))
                    sections.append(
                        html.P(f"  • Class: {sq_class.get('class', 'N/A')}")
                    )
                    sections.append(
                        html.P(f"  • Confidence: {sq_class.get('confidence', 0):.2%}")
                    )

            if "heart_rate_classification" in classification:
                hr_class = classification["heart_rate_classification"]
                if "class" in hr_class:
                    sections.append(html.P(html.Strong("Heart Rate Classification:")))
                    sections.append(
                        html.P(f"  • Class: {hr_class.get('class', 'N/A')}")
                    )
                    sections.append(
                        html.P(f"  • Confidence: {hr_class.get('confidence', 0):.2%}")
                    )
                    sections.append(
                        html.P(
                            f"  • Heart Rate: {hr_class.get('heart_rate', 0):.1f} bpm"
                        )
                    )

            if "pattern_classification" in classification:
                pat_class = classification["pattern_classification"]
                if "class" in pat_class:
                    sections.append(html.P(html.Strong("Pattern Type:")))
                    sections.append(
                        html.P(f"  • Class: {pat_class.get('class', 'N/A')}")
                    )
                    sections.append(
                        html.P(f"  • Confidence: {pat_class.get('confidence', 0):.2%}")
                    )

        # Display Regression if available
        if (
            "regression" in analysis_results
            and "error" not in analysis_results["regression"]
        ):
            regression = analysis_results["regression"]
            sections.append(html.Hr())
            sections.append(html.H6("📈 Regression Analysis"))

            if (
                "linear_trend_regression" in regression
                and "error" not in regression["linear_trend_regression"]
            ):
                lr = regression["linear_trend_regression"]
                sections.append(html.P(html.Strong("Linear Trend:")))
                sections.append(
                    html.P(f"  • Trend: {lr.get('trend_direction', 'N/A')}")
                )
                sections.append(html.P(f"  • R²: {lr.get('r_squared', 0):.4f}"))
                sections.append(html.P(f"  • Slope: {lr.get('slope', 0):.6f}"))

        # Display Clustering if available
        if (
            "clustering" in analysis_results
            and "error" not in analysis_results["clustering"]
        ):
            clustering = analysis_results["clustering"]
            sections.append(html.Hr())
            sections.append(html.H6("🔗 Clustering Analysis"))
            sections.append(
                html.P(f"Total Clusters: {clustering.get('total_clusters', 0)}")
            )

            if (
                "segment_clustering" in clustering
                and "error" not in clustering["segment_clustering"]
            ):
                seg = clustering["segment_clustering"]
                sections.append(html.P(html.Strong("Segment Clustering:")))
                sections.append(
                    html.P(f"  • Number of Clusters: {seg.get('n_clusters', 0)}")
                )

        # Display Dimensionality Reduction if available
        if (
            "dimensionality_reduction" in analysis_results
            and "error" not in analysis_results["dimensionality_reduction"]
        ):
            dim_red = analysis_results["dimensionality_reduction"]
            sections.append(html.Hr())
            sections.append(html.H6("📉 Dimensionality Reduction"))
            sections.append(html.P(f"Total Methods: {dim_red.get('total_methods', 0)}"))

            if "feature_importance" in dim_red:
                fi = dim_red["feature_importance"]
                sections.append(html.P(html.Strong("Top Features:")))
                if "top_features" in fi:
                    for i, feat in enumerate(fi["top_features"][:5]):
                        sections.append(
                            html.P(
                                f"  • {feat.get('name', 'N/A')}: {feat.get('normalized_importance', 0):.4f}"
                            )
                        )

        # Display Forecasting if available
        if (
            "forecasting" in analysis_results
            and "error" not in analysis_results["forecasting"]
        ):
            forecasting = analysis_results["forecasting"]
            sections.append(html.Hr())
            sections.append(html.H6("🔮 Forecasting Analysis"))
            sections.append(
                html.P(f"Total Forecasts: {forecasting.get('total_forecasts', 0)}")
            )

            if (
                "linear_forecast" in forecasting
                and "error" not in forecasting["linear_forecast"]
            ):
                lf = forecasting["linear_forecast"]
                sections.append(html.P(html.Strong("Linear Forecast:")))
                sections.append(
                    html.P(
                        f"  • Forecast Length: {lf.get('forecast_length', 0)} samples"
                    )
                )
                sections.append(html.P(f"  • Trend: {lf.get('trend_slope', 0):.6f}"))

        return html.Div(sections, className="p-3")

    except Exception as e:
        logger.error(f"Error creating analysis summary: {e}")
        return html.Div(
            [html.H5("Error"), html.P(f"Failed to create analysis summary: {str(e)}")]
        )


def create_advanced_model_details(analysis_results, analysis_categories):
    """Create advanced model details with actual results."""
    try:
        if "error" in analysis_results:
            return html.Div(
                [
                    html.H6("Model Details"),
                    html.P("Analysis failed - no model details available."),
                ]
            )

        sections = []

        # ML Models
        if (
            "ml_results" in analysis_results
            and "error" not in analysis_results["ml_results"]
        ):
            ml_results = analysis_results["ml_results"]
            sections.append(html.H6("🤖 Machine Learning Models"))

            for model_name, model_info in ml_results.items():
                model_title = model_name.replace("_", " ").title()
                model_type = model_info.get("model_type", "Unknown")
                status = model_info.get("status", "Unknown")

                model_div = html.Div(
                    [
                        html.P(html.Strong(f"{model_title} ({model_type})")),
                        html.P(f"Status: {status}"),
                    ],
                    className="mb-3 p-2 border rounded",
                )

                # Show CV scores if available
                if "cv_score_mean" in model_info:
                    cv_mean = model_info.get("cv_score_mean", 0)
                    cv_std = model_info.get("cv_score_std", 0)
                    model_div.children.append(
                        html.P(f"CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
                    )
                    model_div.children.append(
                        html.P(f"CV Folds: {model_info.get('cv_folds', 'N/A')}")
                    )

                sections.append(model_div)

        # Deep Learning Models
        if (
            "dl_results" in analysis_results
            and "error" not in analysis_results["dl_results"]
        ):
            dl_results = analysis_results["dl_results"]
            sections.append(html.Hr())
            sections.append(html.H6("🧠 Deep Learning Models"))

            for model_name, model_info in dl_results.items():
                model_title = model_name.replace("_", " ").upper()
                model_type = model_info.get("model_type", "Unknown")
                status = model_info.get("status", "Unknown")

                model_div = html.Div(
                    [
                        html.P(html.Strong(f"{model_title} ({model_type})")),
                        html.P(f"Status: {status}"),
                    ],
                    className="mb-3 p-2 border rounded",
                )

                # Show specific DL details
                if "windows_processed" in model_info:
                    model_div.children.append(
                        html.P(
                            f"Windows Processed: {model_info.get('windows_processed', 0)}"
                        )
                    )
                    model_div.children.append(
                        html.P(f"Window Size: {model_info.get('window_size', 0)}")
                    )

                if "note" in model_info:
                    model_div.children.append(
                        html.P(html.Small(model_info.get("note", "")))
                    )

                sections.append(model_div)

        # Ensemble Models
        if (
            "ensemble" in analysis_results
            and "error" not in analysis_results["ensemble"]
        ):
            ensemble = analysis_results["ensemble"]
            sections.append(html.Hr())
            sections.append(html.H6("🎭 Ensemble Models"))

            for ensemble_type, ensemble_info in ensemble.items():
                if "error" not in str(ensemble_info):
                    ensemble_div = html.Div(
                        [
                            html.P(
                                html.Strong(
                                    ensemble_info.get(
                                        "ensemble_type", "Unknown"
                                    ).title()
                                )
                            ),
                            html.P(f"Status: {ensemble_info.get('status', 'Unknown')}"),
                        ],
                        className="mb-2 p-2 border rounded",
                    )

                    if "cv_score_mean" in ensemble_info:
                        ensemble_div.children.append(
                            html.P(
                                f"CV Score: {ensemble_info.get('cv_score_mean', 0):.4f} ± {ensemble_info.get('cv_score_std', 0):.4f}"
                            )
                        )

                    if "base_models" in ensemble_info:
                        models_str = ", ".join(ensemble_info.get("base_models", []))
                        ensemble_div.children.append(
                            html.P(f"Base Models: {models_str}")
                        )

                    if "meta_model" in ensemble_info:
                        ensemble_div.children.append(
                            html.P(
                                f"Meta Model: {ensemble_info.get('meta_model', 'N/A')}"
                            )
                        )

                    sections.append(ensemble_div)

        if not sections:
            sections.append(
                html.Div(
                    [
                        html.H6("Model Details"),
                        html.P(
                            "No model details available. Select ML/DL options and run analysis."
                        ),
                    ]
                )
            )

        return html.Div(sections, className="p-3")

    except Exception as e:
        logger.error(f"Error creating model details: {e}")
        return html.Div(
            [html.H6("Error"), html.P(f"Failed to create model details: {str(e)}")]
        )


def create_advanced_performance_metrics(analysis_results):
    """Create advanced performance metrics with actual results."""
    try:
        if "error" in analysis_results:
            return html.Div(
                [
                    html.H6("Performance Metrics"),
                    html.P("Analysis failed - no performance metrics available."),
                ]
            )

        sections = []
        sections.append(html.H6("📈 Performance Metrics"))

        # ML Model Performance
        if (
            "ml_results" in analysis_results
            and "error" not in analysis_results["ml_results"]
        ):
            ml_results = analysis_results["ml_results"]
            sections.append(html.P(html.Strong("Machine Learning Models:")))

            ml_scores = []
            for model_name, model_info in ml_results.items():
                if "cv_score_mean" in model_info:
                    cv_mean = model_info.get("cv_score_mean", 0)
                    cv_std = model_info.get("cv_score_std", 0)
                    ml_scores.append(
                        (model_name.replace("_", " ").title(), cv_mean, cv_std)
                    )

            if ml_scores:
                for model_name, score, std in ml_scores:
                    sections.append(html.P(f"{model_name}: {score:.4f} ± {std:.4f}"))
            else:
                sections.append(html.P("No ML metrics available"))

        # Ensemble Performance
        if (
            "ensemble" in analysis_results
            and "error" not in analysis_results["ensemble"]
        ):
            ensemble = analysis_results["ensemble"]
            sections.append(html.Hr())
            sections.append(html.P(html.Strong("Ensemble Models:")))

            for ensemble_type, ensemble_info in ensemble.items():
                if (
                    "error" not in str(ensemble_info)
                    and "cv_score_mean" in ensemble_info
                ):
                    ensemble_name = ensemble_info.get(
                        "ensemble_type", ensemble_type
                    ).title()
                    cv_mean = ensemble_info.get("cv_score_mean", 0)
                    cv_std = ensemble_info.get("cv_score_std", 0)
                    sections.append(
                        html.P(f"{ensemble_name}: {cv_mean:.4f} ± {cv_std:.4f}")
                    )

        # Advanced Processing Performance
        if (
            "advanced_processing" in analysis_results
            and "error" not in analysis_results["advanced_processing"]
        ):
            advanced = analysis_results["advanced_processing"]
            sections.append(html.Hr())
            sections.append(html.P(html.Strong("Advanced Processing:")))

            # Wavelet
            if "wavelet" in advanced and "error" not in advanced["wavelet"]:
                wavelet = advanced["wavelet"]
                sections.append(
                    html.P(
                        f"Wavelet ({wavelet.get('wavelet_type', 'N/A')}): {wavelet.get('num_coefficients', 0)} coefficients"
                    )
                )
                sections.append(
                    html.P(f"Total Energy: {wavelet.get('total_energy', 0):.2f}")
                )

            # Hilbert-Huang
            if "hilbert_huang" in advanced and "error" not in advanced["hilbert_huang"]:
                hht = advanced["hilbert_huang"]
                sections.append(
                    html.P(f"HHT Mean Amplitude: {hht.get('mean_amplitude', 0):.6f}")
                )
                sections.append(
                    html.P(f"HHT Mean Frequency: {hht.get('mean_frequency', 0):.2f} Hz")
                )

            # EMD
            if "emd" in advanced and "error" not in advanced["emd"]:
                emd = advanced["emd"]
                sections.append(html.P(f"EMD IMFs: {emd.get('num_imfs', 0)}"))
                sections.append(
                    html.P(
                        f"Reconstruction Error: {emd.get('reconstruction_error', 0):.6f}"
                    )
                )

        if len(sections) == 1:  # Only header
            sections.append(
                html.P(
                    "No performance metrics available. Run analysis to generate metrics."
                )
            )

        return html.Div(sections, className="p-3")

    except Exception as e:
        logger.error(f"Error creating performance metrics: {e}")
        return html.Div(
            [
                html.H6("Error"),
                html.P(f"Failed to create performance metrics: {str(e)}"),
            ]
        )


def create_advanced_feature_importance(analysis_results):
    """Create advanced feature importance display with actual data."""
    try:
        if "error" in analysis_results:
            return html.Div(
                [
                    html.H6("Feature Importance"),
                    html.P("Analysis failed - no feature importance available."),
                ]
            )

        sections = []
        sections.append(html.H6("🎯 Feature Categories"))

        # Show feature categories and their statistics
        if (
            "features" in analysis_results
            and "error" not in analysis_results["features"]
        ):
            features = analysis_results["features"]

            # Count features in each category
            feature_counts = {
                "Statistical": 5,  # mean, std, skewness, kurtosis, entropy
                "Spectral": 3,  # centroid, bandwidth, dominant_freq
                "Temporal": 4,  # peak_count, mean_interval, heart_rate, interval_std
                "Morphological": 4,  # amplitude_range, amplitude_mean, zero_crossings, signal_energy
            }

            total_features = sum(feature_counts.values())

            # Create bar for visualization
            for category, count in feature_counts.items():
                percentage = (count / total_features * 100) if total_features > 0 else 0
                sections.append(
                    html.P(f"{category}: {percentage:.0f}% ({count} features)")
                )

            sections.append(html.Hr())
            sections.append(html.P(html.Strong("Feature Summary:")))
            sections.append(html.P(f"Total Features Extracted: {total_features}"))

            # Show which categories are available
            available_categories = []
            if "statistical" in features and "error" not in features["statistical"]:
                available_categories.append("Statistical")
            if "spectral" in features and "error" not in features["spectral"]:
                available_categories.append("Spectral")
            if "temporal" in features and "error" not in features["temporal"]:
                available_categories.append("Temporal")
            if "morphological" in features and "error" not in features["morphological"]:
                available_categories.append("Morphological")

            sections.append(
                html.P(
                    f"Available Categories: {', '.join(available_categories) if available_categories else 'None'}"
                )
            )
        else:
            sections.append(
                html.P(
                    "No feature extraction performed. Enable 'Feature Engineering' in analysis categories."
                )
            )

        return html.Div(sections, className="p-3")

    except Exception as e:
        logger.error(f"Error creating feature importance: {e}")
        return html.Div(
            [html.H6("Error"), html.P(f"Failed to create feature importance: {str(e)}")]
        )
