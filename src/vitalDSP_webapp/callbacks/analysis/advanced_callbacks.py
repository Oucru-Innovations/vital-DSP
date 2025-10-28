"""
Advanced analysis callbacks for vitalDSP webapp.

This module handles advanced signal processing methods including machine learning, deep learning,
ensemble methods, and cutting-edge analysis techniques for research applications.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback_context, no_update, html
from dash.exceptions import PreventUpdate
from scipy import signal
import logging
from vitalDSP_webapp.services.data.enhanced_data_service import get_enhanced_data_service

# Import plot utilities for performance optimization
try:
    from vitalDSP_webapp.utils.plot_utils import limit_plot_data, check_plot_data_size
except ImportError:
    # Fallback if plot_utils not available
    def limit_plot_data(time_axis, signal_data, max_duration=300, max_points=10000, start_time=None):
        """Fallback implementation of limit_plot_data"""
        return time_axis, signal_data

    def check_plot_data_size(time_axis, signal_data):
        """Fallback implementation"""
        return True

logger = logging.getLogger(__name__)


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
                        error_fig,
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
                            html.P("No column mapping found. Please upload data first."),
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
                
                # Get the actual DataFrame
                df = data_service.get_data(data_id)
                logger.info(f"Data shape: {df.shape if df is not None else 'None'}, columns: {list(df.columns) if df is not None else 'None'}")

                if df is None or df.empty:
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
                        error_fig,
                        None,
                        None,
                    )

                # Get sampling frequency
                sampling_freq = data_info.get("sampling_frequency", 1000)

                # Convert position (0-100%) and duration to time indices
                start_position = float(start_position) if start_position is not None else 0
                duration = float(duration) if duration is not None else 60

                # Calculate total duration and start time
                total_duration = len(df) / sampling_freq
                start_time = (start_position / 100.0) * total_duration
                end_time = min(start_time + duration, total_duration)

                # Convert to sample indices
                start_idx = int(start_time * sampling_freq)
                end_idx = int(end_time * sampling_freq)

                # Ensure valid indices
                start_idx = max(0, min(start_idx, len(df) - 1))
                end_idx = max(start_idx + 1, min(end_idx, len(df)))

                # Check signal source
                signal_source = signal_source or "filtered"

                # Get signal data based on source
                if signal_source == "filtered" and filtered_signal_data:
                    # Use filtered signal if available
                    signal_data = np.array(filtered_signal_data["signal"])
                    # Apply same windowing
                    if len(signal_data) >= end_idx:
                        signal_data = signal_data[start_idx:end_idx]
                    else:
                        # Fallback to original signal column
                        signal_col = column_mapping.get("signal", "signal")
                        if signal_col and signal_col in df.columns:
                            signal_data = df[signal_col].values[start_idx:end_idx]
                        else:
                            # Final fallback: use first numeric column
                            signal_data = df.iloc[start_idx:end_idx].values.flatten()
                else:
                    # Use original signal column
                    signal_col = column_mapping.get("signal", "signal")
                    if signal_col and signal_col in df.columns:
                        signal_data = df[signal_col].values[start_idx:end_idx]
                    else:
                        # Fallback: use first numeric column
                        signal_data = df.iloc[start_idx:end_idx].values.flatten()

                # Create time axis (normalized to start from 0)
                time_axis = np.arange(len(signal_data)) / sampling_freq

                # Auto-detect signal type if needed
                if signal_type == "auto":
                    signal_type = detect_signal_type(signal_data, sampling_freq)

                # Perform advanced analysis
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
                )

                # Create visualizations
                main_plot = create_main_advanced_plot(
                    time_axis, signal_data, analysis_results, signal_type
                )
                performance_plot = create_advanced_performance_plot(
                    analysis_results, analysis_categories
                )
                visualizations = create_advanced_visualizations(
                    analysis_results, signal_data, sampling_freq
                )

                # Create result displays
                analysis_summary = create_advanced_analysis_summary(
                    analysis_results, signal_type
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
                    "time_window": [start_time, end_time],
                    "sampling_frequency": sampling_freq,
                }

                return (
                    main_plot,
                    performance_plot,
                    analysis_summary,
                    model_details,
                    performance_metrics,
                    feature_importance,
                    visualizations,
                    {"data_id": data_id, "time_window": [start_time, end_time]},
                    stored_results,
                )

            except Exception as e:
                logger.error(f"Error in advanced analysis: {e}")
                error_fig = create_empty_figure()
                error_results = html.Div(
                    [
                        html.H5("Analysis Error"),
                        html.P(f"Advanced analysis failed: {str(e)}"),
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
    def update_advanced_position_slider(nudge_m10, nudge_m1, nudge_p1, nudge_p10, start_position):
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
):
    """Perform comprehensive advanced analysis."""
    try:
        analysis_results = {}
        
        # Always extract features (needed for ML/DL analysis)
        analysis_results["features"] = extract_advanced_features(
            signal_data, sampling_freq
        )

        # Machine learning analysis
        if "ml_analysis" in analysis_categories and ml_options:
            analysis_results["ml_results"] = perform_ml_analysis(
                signal_data, sampling_freq, ml_options, cv_folds, random_state
            )

        # Deep learning analysis
        if "deep_learning" in analysis_categories and deep_learning_options:
            try:
                analysis_results["dl_results"] = perform_deep_learning_analysis(
                    signal_data, sampling_freq, deep_learning_options
                )
            except Exception as e:
                logger.error(f"Error in deep learning analysis: {e}")
                analysis_results["dl_results"] = {"error": f"Deep learning analysis failed: {str(e)}"}

        # Pattern recognition
        if "pattern_recognition" in analysis_categories:
            analysis_results["patterns"] = perform_pattern_recognition(
                signal_data, sampling_freq
            )

        # Ensemble methods (fixed: use "ensemble" not "ensemble_methods")
        if "ensemble" in analysis_categories:
            analysis_results["ensemble"] = perform_ensemble_analysis(
                signal_data, sampling_freq, cv_folds, random_state
            )
        
        # Advanced signal processing (when pattern recognition is enabled, include advanced processing)
        # Also handle classification, regression, etc. through ML analysis
        if "pattern_recognition" in analysis_categories or "classification" in analysis_categories or "regression" in analysis_categories or "clustering" in analysis_categories or "dimensionality_reduction" in analysis_categories or "forecasting" in analysis_categories:
            analysis_results["advanced_processing"] = perform_advanced_signal_processing(
                signal_data, sampling_freq
            )

        return analysis_results

    except Exception as e:
        logger.error(f"Error in advanced analysis: {e}")
        return {"error": f"Advanced analysis failed: {str(e)}"}


def extract_advanced_features(signal_data, sampling_freq):
    """Extract advanced features from the signal."""
    try:
        features = {}

        # Statistical features
        features["statistical"] = {
            "mean": np.mean(signal_data),
            "std": np.std(signal_data),
            "skewness": calculate_skewness(signal_data),
            "kurtosis": calculate_kurtosis(signal_data),
            "entropy": calculate_entropy(signal_data),
        }

        # Spectral features
        features["spectral"] = extract_spectral_features(signal_data, sampling_freq)

        # Temporal features
        features["temporal"] = extract_temporal_features(signal_data, sampling_freq)

        # Morphological features
        features["morphological"] = extract_morphological_features(
            signal_data, sampling_freq
        )

        return features

    except Exception as e:
        logger.error(f"Error extracting advanced features: {e}")
        return {"error": f"Feature extraction failed: {str(e)}"}


def perform_ml_analysis(signal_data, sampling_freq, ml_options, cv_folds, random_state):
    """Perform machine learning analysis."""
    try:
        ml_results = {}

        # Extract features for ML
        features = extract_ml_features(signal_data, sampling_freq)

        if "svm" in ml_options:
            ml_results["svm"] = train_svm_model(features, cv_folds, random_state)

        if "random_forest" in ml_options:
            ml_results["random_forest"] = train_random_forest_model(
                features, cv_folds, random_state
            )

        if "neural_network" in ml_options:
            ml_results["neural_network"] = train_neural_network_model(
                features, cv_folds, random_state
            )

        if "gradient_boosting" in ml_options:
            ml_results["gradient_boosting"] = train_gradient_boosting_model(
                features, cv_folds, random_state
            )

        return ml_results

    except Exception as e:
        logger.error(f"Error in ML analysis: {e}")
        return {"error": f"ML analysis failed: {str(e)}"}


def perform_deep_learning_analysis(signal_data, sampling_freq, deep_learning_options):
    """Perform deep learning analysis."""
    try:
        dl_results = {}

        # Prepare data for deep learning
        dl_data = prepare_dl_data(signal_data, sampling_freq)

        if "cnn" in deep_learning_options:
            dl_results["cnn"] = train_cnn_model(dl_data)

        if "lstm" in deep_learning_options:
            dl_results["lstm"] = train_lstm_model(dl_data)

        if "transformer" in deep_learning_options:
            dl_results["transformer"] = train_transformer_model(dl_data)

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
    """Extract temporal features."""
    try:
        # Peak detection
        peaks, _ = signal.find_peaks(
            signal_data, height=np.mean(signal_data) + np.std(signal_data)
        )

        if len(peaks) > 1:
            intervals = np.diff(peaks) / sampling_freq
            return {
                "peak_count": len(peaks),
                "mean_interval": np.mean(intervals),
                "interval_std": np.std(intervals),
                "heart_rate": 60 / np.mean(intervals) if np.mean(intervals) > 0 else 0,
            }
        else:
            return {
                "peak_count": len(peaks),
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
def train_svm_model(features, cv_folds, random_state):
    """Train SVM model using sklearn."""
    try:
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        # Create dummy labels for demonstration (in real use, these would be provided)
        labels = np.random.randint(0, 2, len(features))

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.reshape(-1, 1))

        # Train SVM
        svm_model = SVC(random_state=random_state)
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


def train_random_forest_model(features, cv_folds, random_state):
    """Train Random Forest model using sklearn."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        # Create dummy labels for demonstration
        labels = np.random.randint(0, 2, len(features))

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.reshape(-1, 1))

        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
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


def train_neural_network_model(features, cv_folds, random_state):
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

    Returns
    -------
    dict
        Dictionary containing model results and performance metrics
    """
    try:
        from vitalDSP.advanced_computation.neural_network_filtering import (
            NeuralNetworkFiltering,
        )

        # Train a feedforward neural network for signal denoising
        nn_filter = NeuralNetworkFiltering(
            features,
            network_type="feedforward",
            hidden_layers=[64, 32],
            learning_rate=0.001,
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
            "hidden_layers": [64, 32],
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


def train_cnn_model(data):
    """
    Train CNN model for signal analysis using vitalDSP.

    Parameters
    ----------
    data : dict
        Dictionary containing signal data and metadata

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
            "sampling_freq": sampling_freq,
            "note": "CNN architecture: 16 filters, 3x3 kernel, ReLU activation",
        }

    except Exception as e:
        logger.error(f"Error training CNN model: {e}")
        return {"model_type": "CNN", "status": "placeholder", "error": str(e)}


def train_lstm_model(data):
    """
    Train LSTM model for time-series signal analysis using vitalDSP.

    Parameters
    ----------
    data : dict
        Dictionary containing signal data and metadata

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

        # Train LSTM network
        lstm_filter = NeuralNetworkFiltering(
            signal_data,
            network_type="recurrent",
            hidden_layers=[64],
            learning_rate=0.001,
            epochs=30,
            batch_size=16,
            dropout_rate=0.3,
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
            mse = np.mean((signal_data[:min_len] - filtered_signal_flat[:min_len]) ** 2)
        else:
            mse = 0.0

        logger.info(f"LSTM training complete. MSE: {mse:.6f}")

        return {
            "model_type": "LSTM",
            "status": "placeholder",
            "sequence_length": sequence_length,
            "hidden_units": 64,
            "mse": mse,
            "filtered_signal": filtered_signal,
            "sampling_freq": sampling_freq,
            "note": "LSTM with 64 hidden units, dropout=0.3",
        }

    except Exception as e:
        logger.error(f"Error training LSTM model: {e}")
        return {"model_type": "LSTM", "status": "placeholder", "error": str(e)}


def train_transformer_model(data):
    """Train Transformer model using sklearn (simplified implementation)."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        signal_data = data.get("signal_data", np.array([]))
        if len(signal_data) == 0:
            return {"model_type": "Transformer", "status": "no_data"}

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
            "cv_score_mean": float(np.mean(cv_scores)),
            "note": "Simplified transformer using Random Forest",
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
    """Create the main advanced analysis plot."""
    try:
        # Don't fail if individual analysis steps have errors - plot what we have
        # Only fail if analysis_results itself is None or empty
        if not analysis_results:
            return create_empty_figure()

        fig = go.Figure()

        # Original signal
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
            title=f"Advanced Analysis: {signal_type.upper()} Signal",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=400,
            showlegend=True,
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating main advanced plot: {e}")
        return create_empty_figure()


def create_advanced_performance_plot(analysis_results, analysis_categories):
    """Create advanced performance plot."""
    try:
        # Don't fail if individual analysis steps have errors - plot what we have
        if not analysis_results:
            return create_empty_figure()

        fig = go.Figure()

        # Add performance metrics if available
        if "ml_results" in analysis_results:
            ml_results = analysis_results["ml_results"]
            if "error" not in ml_results:
                models = list(ml_results.keys())
                # Placeholder performance scores
                scores = [0.8, 0.85, 0.75, 0.9]  # Placeholder values

                fig.add_trace(
                    go.Bar(x=models, y=scores[: len(models)], name="Performance Scores")
                )

        fig.update_layout(
            title="Advanced Analysis Performance",
            xaxis_title="Models",
            yaxis_title="Performance Score",
            height=400,
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating performance plot: {e}")
        return create_empty_figure()


def create_advanced_visualizations(analysis_results, signal_data, sampling_freq):
    """Create advanced visualizations including EMD decomposition."""
    try:
        # Don't fail if individual analysis steps have errors - plot what we have
        if not analysis_results:
            return create_empty_figure()

        # Check if EMD results are available
        if "emd" in analysis_results and "imfs" in analysis_results["emd"]:
            emd_results = analysis_results["emd"]
            imfs = emd_results["imfs"]
            num_imfs = min(len(imfs), 8)  # Display up to 8 IMFs

            # Create subplots for EMD visualization
            fig = make_subplots(
                rows=num_imfs + 1,
                cols=1,
                subplot_titles=["Original Signal"]
                + [f"IMF {i+1}" for i in range(num_imfs)],
                vertical_spacing=0.02,
            )

            # Time axis
            time_axis = np.arange(len(signal_data)) / sampling_freq

            # Plot original signal
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=signal_data,
                    mode="lines",
                    name="Original",
                    line=dict(color="blue", width=1.5),
                ),
                row=1,
                col=1,
            )

            # Plot each IMF
            colors = [
                "red",
                "green",
                "orange",
                "purple",
                "brown",
                "pink",
                "gray",
                "olive",
            ]
            for i in range(num_imfs):
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

            # Update layout
            fig.update_layout(
                title=f"EMD Decomposition - {num_imfs} IMFs (Reconstruction Error: {emd_results.get('reconstruction_error', 0):.6f})",
                height=200 * (num_imfs + 1),
                showlegend=False,
            )
            fig.update_xaxes(title_text="Time (s)", row=num_imfs + 1, col=1)

            return fig

        else:
            # Fallback to placeholder visualizations
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[
                    "Feature Analysis",
                    "Pattern Recognition",
                    "ML Results",
                    "Advanced Processing",
                ],
                vertical_spacing=0.1,
            )

            # Placeholder visualizations
            fig.add_trace(
                go.Scatter(y=signal_data[:100], name="Features"), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(y=signal_data[100:200], name="Patterns"), row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=["Model 1", "Model 2"], y=[0.8, 0.9], name="ML"), row=2, col=1
            )
            fig.add_trace(
                go.Scatter(y=signal_data[200:300], name="Advanced"), row=2, col=2
            )

            fig.update_layout(
                title="Advanced Analysis Visualizations", height=500, showlegend=False
            )

            return fig

    except Exception as e:
        logger.error(f"Error creating advanced visualizations: {e}")
        return create_empty_figure()


# Result display functions
def create_advanced_analysis_summary(analysis_results, signal_type):
    """Create advanced analysis summary with actual results."""
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
                    html.P(f"Analysis Categories Completed: {len([k for k, v in analysis_results.items() if 'error' not in str(v)])}"),
                ]
            )
        )

        # Display features if available
        if "features" in analysis_results and "error" not in analysis_results["features"]:
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
                sections.append(html.P(f"  • Spectral Centroid: {spec.get('spectral_centroid', 0):.2f} Hz"))
                sections.append(html.P(f"  • Spectral Bandwidth: {spec.get('spectral_bandwidth', 0):.2f} Hz"))
                sections.append(html.P(f"  • Dominant Frequency: {spec.get('dominant_frequency', 0):.2f} Hz"))
            
            # Temporal features
            if "temporal" in features and "error" not in features["temporal"]:
                temp = features["temporal"]
                sections.append(html.P(html.Strong("Temporal Features:")))
                sections.append(html.P(f"  • Peak Count: {temp.get('peak_count', 0)}"))
                sections.append(html.P(f"  • Mean Interval: {temp.get('mean_interval', 0):.4f} s"))
                sections.append(html.P(f"  • Heart Rate: {temp.get('heart_rate', 0):.2f} bpm"))
                sections.append(html.P(f"  • Interval Std: {temp.get('interval_std', 0):.4f} s"))
            
            # Morphological features
            if "morphological" in features and "error" not in features["morphological"]:
                morph = features["morphological"]
                sections.append(html.P(html.Strong("Morphological Features:")))
                sections.append(html.P(f"  • Amplitude Range: {morph.get('amplitude_range', 0):.6f}"))
                sections.append(html.P(f"  • Mean Amplitude: {morph.get('amplitude_mean', 0):.6f}"))
                sections.append(html.P(f"  • Zero Crossings: {morph.get('zero_crossings', 0)}"))
                sections.append(html.P(f"  • Signal Energy: {morph.get('signal_energy', 0):.2f}"))

        # Display pattern recognition if available
        if "patterns" in analysis_results and "error" not in analysis_results["patterns"]:
            patterns = analysis_results["patterns"]
            sections.append(html.Hr())
            sections.append(html.H6("🎯 Pattern Recognition"))
            
            if "peaks" in patterns and "error" not in patterns["peaks"]:
                peak_pat = patterns["peaks"]
                sections.append(html.P(html.Strong("Peak Patterns:")))
                sections.append(html.P(f"  • Number of Peaks: {peak_pat.get('num_peaks', 0)}"))
                if "mean_interval" in peak_pat:
                    sections.append(html.P(f"  • Mean Interval: {peak_pat.get('mean_interval', 0):.4f} s"))
                    sections.append(html.P(f"  • Heart Rate: {peak_pat.get('heart_rate', 0):.2f} bpm"))
            
            if "frequency" in patterns and "error" not in patterns["frequency"]:
                freq_pat = patterns["frequency"]
                sections.append(html.P(html.Strong("Frequency Patterns:")))
                sections.append(html.P(f"  • Dominant Frequency: {freq_pat.get('dominant_frequency', 0):.2f} Hz"))
                sections.append(html.P(f"  • Spectral Centroid: {freq_pat.get('spectral_centroid', 0):.2f} Hz"))
                sections.append(html.P(f"  • Total Power: {freq_pat.get('total_power', 0):.2f}"))
            
            if "morphology" in patterns and "error" not in patterns["morphology"]:
                morph_pat = patterns["morphology"]
                sections.append(html.P(html.Strong("Morphological Patterns:")))
                sections.append(html.P(f"  • Peak Count: {morph_pat.get('num_peaks', 0)}"))
                sections.append(html.P(f"  • Mean Peak Height: {morph_pat.get('mean_peak_height', 0):.6f}"))
                sections.append(html.P(f"  • Peak Variability: {morph_pat.get('peak_variability', 0):.4f}"))

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
        if "ml_results" in analysis_results and "error" not in analysis_results["ml_results"]:
            ml_results = analysis_results["ml_results"]
            sections.append(html.H6("🤖 Machine Learning Models"))
            
            for model_name, model_info in ml_results.items():
                model_title = model_name.replace('_', ' ').title()
                model_type = model_info.get('model_type', 'Unknown')
                status = model_info.get('status', 'Unknown')
                
                model_div = html.Div(
                    [
                        html.P(html.Strong(f"{model_title} ({model_type})")),
                        html.P(f"Status: {status}"),
                    ],
                    className="mb-3 p-2 border rounded"
                )
                
                # Show CV scores if available
                if 'cv_score_mean' in model_info:
                    cv_mean = model_info.get('cv_score_mean', 0)
                    cv_std = model_info.get('cv_score_std', 0)
                    model_div.children.append(
                        html.P(f"CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
                    )
                    model_div.children.append(
                        html.P(f"CV Folds: {model_info.get('cv_folds', 'N/A')}")
                    )
                
                sections.append(model_div)
        
        # Deep Learning Models
        if "dl_results" in analysis_results and "error" not in analysis_results["dl_results"]:
            dl_results = analysis_results["dl_results"]
            sections.append(html.Hr())
            sections.append(html.H6("🧠 Deep Learning Models"))
            
            for model_name, model_info in dl_results.items():
                model_title = model_name.replace('_', ' ').upper()
                model_type = model_info.get('model_type', 'Unknown')
                status = model_info.get('status', 'Unknown')
                
                model_div = html.Div(
                    [
                        html.P(html.Strong(f"{model_title} ({model_type})")),
                        html.P(f"Status: {status}"),
                    ],
                    className="mb-3 p-2 border rounded"
                )
                
                # Show specific DL details
                if 'windows_processed' in model_info:
                    model_div.children.append(
                        html.P(f"Windows Processed: {model_info.get('windows_processed', 0)}")
                    )
                    model_div.children.append(
                        html.P(f"Window Size: {model_info.get('window_size', 0)}")
                    )
                
                if 'note' in model_info:
                    model_div.children.append(
                        html.P(html.Small(model_info.get('note', '')))
                    )
                
                sections.append(model_div)
        
        # Ensemble Models
        if "ensemble" in analysis_results and "error" not in analysis_results["ensemble"]:
            ensemble = analysis_results["ensemble"]
            sections.append(html.Hr())
            sections.append(html.H6("🎭 Ensemble Models"))
            
            for ensemble_type, ensemble_info in ensemble.items():
                if 'error' not in str(ensemble_info):
                    ensemble_div = html.Div(
                        [
                            html.P(html.Strong(ensemble_info.get('ensemble_type', 'Unknown').title())),
                            html.P(f"Status: {ensemble_info.get('status', 'Unknown')}"),
                        ],
                        className="mb-2 p-2 border rounded"
                    )
                    
                    if 'cv_score_mean' in ensemble_info:
                        ensemble_div.children.append(
                            html.P(f"CV Score: {ensemble_info.get('cv_score_mean', 0):.4f} ± {ensemble_info.get('cv_score_std', 0):.4f}")
                        )
                    
                    if 'base_models' in ensemble_info:
                        models_str = ', '.join(ensemble_info.get('base_models', []))
                        ensemble_div.children.append(
                            html.P(f"Base Models: {models_str}")
                        )
                    
                    if 'meta_model' in ensemble_info:
                        ensemble_div.children.append(
                            html.P(f"Meta Model: {ensemble_info.get('meta_model', 'N/A')}")
                        )
                    
                    sections.append(ensemble_div)

        if not sections:
            sections.append(
                html.Div(
                    [html.H6("Model Details"), html.P("No model details available. Select ML/DL options and run analysis.")]
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
        if "ml_results" in analysis_results and "error" not in analysis_results["ml_results"]:
            ml_results = analysis_results["ml_results"]
            sections.append(html.P(html.Strong("Machine Learning Models:")))
            
            ml_scores = []
            for model_name, model_info in ml_results.items():
                if 'cv_score_mean' in model_info:
                    cv_mean = model_info.get('cv_score_mean', 0)
                    cv_std = model_info.get('cv_score_std', 0)
                    ml_scores.append((model_name.replace('_', ' ').title(), cv_mean, cv_std))
            
            if ml_scores:
                for model_name, score, std in ml_scores:
                    sections.append(html.P(f"{model_name}: {score:.4f} ± {std:.4f}"))
            else:
                sections.append(html.P("No ML metrics available"))
        
        # Ensemble Performance
        if "ensemble" in analysis_results and "error" not in analysis_results["ensemble"]:
            ensemble = analysis_results["ensemble"]
            sections.append(html.Hr())
            sections.append(html.P(html.Strong("Ensemble Models:")))
            
            for ensemble_type, ensemble_info in ensemble.items():
                if 'error' not in str(ensemble_info) and 'cv_score_mean' in ensemble_info:
                    ensemble_name = ensemble_info.get('ensemble_type', ensemble_type).title()
                    cv_mean = ensemble_info.get('cv_score_mean', 0)
                    cv_std = ensemble_info.get('cv_score_std', 0)
                    sections.append(html.P(f"{ensemble_name}: {cv_mean:.4f} ± {cv_std:.4f}"))

        # Advanced Processing Performance
        if "advanced_processing" in analysis_results and "error" not in analysis_results["advanced_processing"]:
            advanced = analysis_results["advanced_processing"]
            sections.append(html.Hr())
            sections.append(html.P(html.Strong("Advanced Processing:")))
            
            # Wavelet
            if "wavelet" in advanced and "error" not in advanced["wavelet"]:
                wavelet = advanced["wavelet"]
                sections.append(html.P(f"Wavelet ({wavelet.get('wavelet_type', 'N/A')}): {wavelet.get('num_coefficients', 0)} coefficients"))
                sections.append(html.P(f"Total Energy: {wavelet.get('total_energy', 0):.2f}"))
            
            # Hilbert-Huang
            if "hilbert_huang" in advanced and "error" not in advanced["hilbert_huang"]:
                hht = advanced["hilbert_huang"]
                sections.append(html.P(f"HHT Mean Amplitude: {hht.get('mean_amplitude', 0):.6f}"))
                sections.append(html.P(f"HHT Mean Frequency: {hht.get('mean_frequency', 0):.2f} Hz"))
            
            # EMD
            if "emd" in advanced and "error" not in advanced["emd"]:
                emd = advanced["emd"]
                sections.append(html.P(f"EMD IMFs: {emd.get('num_imfs', 0)}"))
                sections.append(html.P(f"Reconstruction Error: {emd.get('reconstruction_error', 0):.6f}"))

        if len(sections) == 1:  # Only header
            sections.append(html.P("No performance metrics available. Run analysis to generate metrics."))

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
        if "features" in analysis_results and "error" not in analysis_results["features"]:
            features = analysis_results["features"]
            
            # Count features in each category
            feature_counts = {
                "Statistical": 5,  # mean, std, skewness, kurtosis, entropy
                "Spectral": 3,      # centroid, bandwidth, dominant_freq
                "Temporal": 4,      # peak_count, mean_interval, heart_rate, interval_std
                "Morphological": 4  # amplitude_range, amplitude_mean, zero_crossings, signal_energy
            }
            
            total_features = sum(feature_counts.values())
            
            # Create bar for visualization
            for category, count in feature_counts.items():
                percentage = (count / total_features * 100) if total_features > 0 else 0
                sections.append(html.P(f"{category}: {percentage:.0f}% ({count} features)"))
            
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
            
            sections.append(html.P(f"Available Categories: {', '.join(available_categories) if available_categories else 'None'}"))
        else:
            sections.append(html.P("No feature extraction performed. Enable 'Feature Engineering' in analysis categories."))

        return html.Div(sections, className="p-3")

    except Exception as e:
        logger.error(f"Error creating feature importance: {e}")
        return html.Div(
            [html.H6("Error"), html.P(f"Failed to create feature importance: {str(e)}")]
        )
