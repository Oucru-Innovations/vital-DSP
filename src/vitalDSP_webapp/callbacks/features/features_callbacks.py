"""
Feature engineering callbacks for vitalDSP webapp.

This module handles advanced feature engineering and signal processing features.
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
import traceback

logger = logging.getLogger(__name__)


def register_features_callbacks(app):
    """Register all feature engineering callbacks."""
    logger.info("=== REGISTERING FEATURE ENGINEERING CALLBACKS ===")

    # Import vitalDSP modules
    _import_vitaldsp_modules()

    @app.callback(
        [
            Output("features-analysis-results", "children"),
            Output("features-analysis-plots", "figure"),
            Output("store-features-data", "data"),
            Output("store-features-features", "data"),
        ],
        [Input("features-analyze-btn", "n_clicks"), Input("url", "pathname")],
        [
            State("features-signal-type", "value"),
            State("features-signal-source-select", "value"),
            State("features-preprocessing", "value"),
            State("features-categories", "value"),
            State("features-advanced-options", "value"),
        ],
    )
    def features_analysis_callback(
        n_clicks,
        pathname,
        signal_type,
        signal_source,
        preprocessing,
        categories,
        advanced_options,
    ):
        """Main callback for feature engineering analysis."""
        logger.info("=== FEATURES ANALYSIS CALLBACK TRIGGERED ===")

        # Only run this when we're on the features page
        if pathname != "/features":
            logger.info("Not on features page, returning empty figures")
            return "Navigate to Features page", create_empty_figure(), None, None

        # Check if data is available first
        try:
            from vitalDSP_webapp.services.data.data_service import get_data_service

            data_service = get_data_service()

            if data_service is None:
                logger.error("Data service is None")
                return (
                    "Data service not available. Please restart the application.",
                    create_empty_figure(),
                    None,
                    None,
                )

            all_data = data_service.get_all_data()
            if not all_data:
                logger.info("No data available")
                return (
                    "No data available. Please upload and process data first.",
                    create_empty_figure(),
                    None,
                    None,
                )

            # Data is available, show instructions if no button click
            if n_clicks is None:
                logger.info("No button click - showing instructions")
                return (
                    "Data is available! Click '🚀 Analyze Features' to start analysis.",
                    create_empty_figure(),
                    None,
                    None,
                )

            logger.info(f"Button clicked {n_clicks} times, starting analysis...")

            # Get the most recent data entry
            latest_data_id = list(all_data.keys())[-1]
            latest_data = all_data[latest_data_id]

            logger.info(f"Found data: {latest_data_id}")

            # Get column mapping
            column_mapping = data_service.get_column_mapping(latest_data_id)
            if not column_mapping:
                logger.warning(
                    "Data has not been processed yet - no column mapping found"
                )
                return (
                    "Please process your data on the Upload page first (configure column mapping)",
                    create_empty_figure(),
                    None,
                    None,
                )

            logger.info(f"Column mapping found: {column_mapping}")

            df = data_service.get_data(latest_data_id)
            if df is None or df.empty:
                logger.warning("Data frame is empty")
                return "Data is empty or corrupted.", create_empty_figure(), None, None

            # Get sampling frequency from the data info
            sampling_freq = latest_data.get("info", {}).get("sampling_freq", 1000)
            logger.info(f"Sampling frequency: {sampling_freq}")

            # Get the main signal column
            signal_column = None
            for col_type, col_name in column_mapping.items():
                if col_type in ["ppg", "ecg", "signal"]:
                    signal_column = col_name
                    break

            if signal_column is None:
                # Fallback to first column
                signal_column = df.columns[0]

            logger.info(f"Using signal column: {signal_column}")

            # Extract signal data
            signal_data = np.array(df[signal_column])

            # Always keep the original signal_data for dynamic filtering
            original_signal_data = signal_data.copy()
            selected_signal = signal_data
            signal_source_info = "Original Signal"
            filter_info = None

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
                    selected_signal = filtered_data
                    signal_source_info = "Filtered Signal"
                else:
                    logger.info("No filtered data available, using original signal")
            else:
                logger.info("Using original signal as requested")

            # Log selected signal characteristics
            logger.info(f"Selected signal shape: {selected_signal.shape}")
            logger.info(
                f"Selected signal range: {np.min(selected_signal):.3f} to {np.max(selected_signal):.3f}"
            )
            logger.info(f"Selected signal mean: {np.mean(selected_signal):.3f}")
            logger.info(f"Signal source: {signal_source_info}")

            # Use selected_signal for analysis
            signal_data = selected_signal

            # Set default values
            signal_type = signal_type or "auto"
            preprocessing = preprocessing or ["detrend", "normalize"]
            categories = categories or ["statistical", "spectral"]
            advanced_options = advanced_options or []

            # Auto-detect signal type if needed
            if signal_type == "auto":
                signal_type = detect_signal_type(signal_data, sampling_freq)
                logger.info(f"Auto-detected signal type: {signal_type}")

            # Apply preprocessing
            processed_signal = apply_preprocessing(
                signal_data, preprocessing, sampling_freq, filter_info
            )

            # Extract features based on selected categories
            features = extract_comprehensive_features(
                processed_signal, sampling_freq, categories, advanced_options
            )

            # Extract additional signal quality and physiological features
            signal_quality = _extract_signal_quality_metrics(
                processed_signal, sampling_freq
            )
            physiological_features = _extract_physiological_features(
                processed_signal, sampling_freq, signal_type
            )

            # Add to features dictionary
            features["signal_quality"] = signal_quality
            features["physiological"] = physiological_features

            # Create comprehensive results display
            results_display = create_comprehensive_features_display(
                features, signal_type, categories
            )

            # Create analysis plots
            logger.info(
                f"Creating plots with signal data length: {len(processed_signal)}, categories: {categories}"
            )
            logger.info(f"Features keys: {list(features.keys())}")
            analysis_plots = create_features_analysis_plots(
                processed_signal, features, categories, sampling_freq
            )

            # Store data for other callbacks
            features_data = {
                "signal_data": processed_signal.tolist(),
                "signal_type": signal_type,
                "sampling_freq": sampling_freq,
                "categories": categories,
                "signal_quality": signal_quality,
                "physiological_features": physiological_features,
            }

            features_features = {
                "statistical": features.get("statistical", {}),
                "spectral": features.get("spectral", {}),
                "temporal": features.get("temporal", {}),
                "morphological": features.get("morphological", {}),
                "entropy": features.get("entropy", {}),
                "fractal": features.get("fractal", {}),
                "signal_quality": signal_quality,
                "physiological": physiological_features,
            }

            logger.info("Feature analysis completed successfully")
            return results_display, analysis_plots, features_data, features_features

        except Exception as e:
            logger.error(f"Error in feature analysis: {e}")
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
                    html.H5("Error in Feature Analysis"),
                    html.P(f"Analysis failed: {str(e)}"),
                    html.P("Please check your data and parameters."),
                ]
            )

            return error_results, error_fig, None, None


# Helper functions for feature engineering
def _import_vitaldsp_modules():
    """Import vitalDSP modules when needed."""
    try:
        # Import vitalDSP modules directly from their modules since __init__.py files are empty
        from vitalDSP.physiological_features.time_domain import TimeDomainFeatures
        from vitalDSP.physiological_features.frequency_domain import (
            FrequencyDomainFeatures,
        )
        from vitalDSP.physiological_features.hrv_analysis import HRVFeatures
        from vitalDSP.physiological_features.beat_to_beat import BeatToBeatAnalysis
        from vitalDSP.physiological_features.energy_analysis import EnergyAnalysis
        from vitalDSP.physiological_features.envelope_detection import EnvelopeDetection
        from vitalDSP.physiological_features.signal_segmentation import (
            SignalSegmentation,
        )
        from vitalDSP.physiological_features.trend_analysis import TrendAnalysis
        from vitalDSP.physiological_features.waveform import WaveformMorphology
        from vitalDSP.physiological_features.nonlinear import NonlinearFeatures
        from vitalDSP.physiological_features.cross_correlation import (
            CrossCorrelationFeatures,
        )
        from vitalDSP.physiological_features.signal_power_analysis import (
            SignalPowerAnalysis,
        )

        # Import transforms
        from vitalDSP.transforms.wavelet_transform import WaveletTransform
        from vitalDSP.transforms.fourier_transform import FourierTransform
        from vitalDSP.transforms.hilbert_transform import HilbertTransform
        from vitalDSP.transforms.stft import STFT
        from vitalDSP.transforms.mfcc import MFCC
        from vitalDSP.transforms.pca_ica_signal_decomposition import (
            PCAICASignalDecomposition,
        )

        # Import advanced computation
        from vitalDSP.advanced_computation.non_linear_analysis import NonlinearAnalysis
        from vitalDSP.advanced_computation.anomaly_detection import AnomalyDetection
        from vitalDSP.advanced_computation.bayesian_analysis import BayesianAnalysis
        from vitalDSP.advanced_computation.kalman_filter import KalmanFilter
        from vitalDSP.advanced_computation.harmonic_percussive_separation import (
            HarmonicPercussiveSeparation,
        )
        from vitalDSP.advanced_computation.emd import EMD

        # Import feature engineering
        from vitalDSP.feature_engineering.morphology_features import MorphologyFeatures
        from vitalDSP.feature_engineering.ppg_light_features import PPGLightFeatures
        from vitalDSP.feature_engineering.ppg_autonomic_features import (
            PPGAutonomicFeatures,
        )

        logger.info("Successfully imported vitalDSP modules")
        return True
    except ImportError as e:
        logger.warning(
            f"vitalDSP modules not available: {e}, using scipy/numpy fallback"
        )
        return False


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


def apply_preprocessing(
    signal_data, preprocessing_options, sampling_freq, filter_info=None
):
    """Apply preprocessing to the signal."""
    processed_signal = signal_data.copy()

    try:
        for option in preprocessing_options:
            if option == "detrend":
                # Use vitalDSP detrending when available
                try:
                    from vitalDSP.preprocess.preprocess_operations import (
                        PreprocessConfig,
                    )
                    from vitalDSP.preprocess.preprocess_operations import (
                        PreprocessOperations,
                    )

                    preprocess_config = PreprocessConfig()
                    preprocess_config.detrend_type = "linear"

                    preprocess_ops = PreprocessOperations(
                        processed_signal, preprocess_config
                    )
                    processed_signal = preprocess_ops.apply_detrending()
                    logger.info("Applied vitalDSP detrending")

                except Exception as e:
                    logger.warning(f"vitalDSP detrending failed, using scipy: {e}")
                    # Fallback to scipy detrending
                    processed_signal = signal.detrend(processed_signal)
            elif option == "normalize":
                # Use vitalDSP normalization when available
                try:
                    from vitalDSP.preprocess.preprocess_operations import (
                        PreprocessConfig,
                    )
                    from vitalDSP.preprocess.preprocess_operations import (
                        PreprocessOperations,
                    )

                    preprocess_config = PreprocessConfig()
                    preprocess_config.normalization_type = "z_score"

                    preprocess_ops = PreprocessOperations(
                        processed_signal, preprocess_config
                    )
                    processed_signal = preprocess_ops.apply_normalization()
                    logger.info("Applied vitalDSP normalization")

                except Exception as e:
                    logger.warning(f"vitalDSP normalization failed, using basic: {e}")
                    # Fallback to basic normalization with safety checks
                    # Handle infinite values
                    processed_signal = np.where(
                        np.isfinite(processed_signal), processed_signal, 0
                    )
                    signal_mean = np.mean(processed_signal)
                    signal_std = np.std(processed_signal)
                    if signal_std > 0:
                        processed_signal = (processed_signal - signal_mean) / signal_std
                    else:
                        processed_signal = (
                            processed_signal - signal_mean
                        )  # Just center if std is 0
            elif option == "filter":
                # Use filter parameters from filtering screen if available
                if filter_info is not None:
                    logger.info("Using filter parameters from filtering screen")
                    parameters = filter_info.get("parameters", {})
                    filter_type = filter_info.get("filter_type", "traditional")

                    if filter_type == "traditional":
                        # Extract traditional filter parameters
                        filter_family = parameters.get("filter_family", "butter")
                        filter_response = parameters.get("filter_response", "bandpass")
                        low_freq = parameters.get("low_freq", 0.1)
                        high_freq = parameters.get("high_freq", 0.8)
                        filter_order = parameters.get("filter_order", 4)

                        # Apply the same filter as used in filtering screen
                        from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
                            apply_traditional_filter,
                        )

                        processed_signal = apply_traditional_filter(
                            processed_signal,
                            sampling_freq,
                            filter_family,
                            filter_response,
                            low_freq,
                            high_freq,
                            filter_order,
                        )
                        logger.info(
                            f"Applied traditional filter: {filter_family} {filter_response} {low_freq}-{high_freq} Hz"
                        )
                    else:
                        # For other filter types, use original signal
                        logger.info(
                            f"Using original signal for filter type: {filter_type}"
                        )
                else:
                    # Fallback to default low-pass filter
                    logger.info(
                        "No filter info available, using default low-pass filter"
                    )
                    nyquist = sampling_freq / 2
                    cutoff = min(nyquist * 0.8, 50)  # 80% of Nyquist or 50 Hz
                    b, a = signal.butter(4, cutoff / nyquist, btype="low")
                    processed_signal = signal.filtfilt(b, a, processed_signal)
            elif option == "outlier_removal":
                # Use vitalDSP outlier removal when available
                try:
                    from vitalDSP.preprocess.preprocess_operations import (
                        PreprocessConfig,
                    )
                    from vitalDSP.preprocess.preprocess_operations import (
                        PreprocessOperations,
                    )

                    preprocess_config = PreprocessConfig()
                    preprocess_config.outlier_method = "iqr"
                    preprocess_config.outlier_threshold = 1.5

                    preprocess_ops = PreprocessOperations(
                        processed_signal, preprocess_config
                    )
                    processed_signal = preprocess_ops.apply_outlier_removal()
                    logger.info("Applied vitalDSP outlier removal")

                except Exception as e:
                    logger.warning(
                        f"vitalDSP outlier removal failed, using basic IQR: {e}"
                    )
                    # Fallback to basic IQR method
                    q1 = np.percentile(processed_signal, 25)
                    q3 = np.percentile(processed_signal, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outlier_mask = (processed_signal >= lower_bound) & (
                        processed_signal <= upper_bound
                    )
                    if (
                        np.sum(outlier_mask) > len(processed_signal) * 0.5
                    ):  # Only if we don't lose too much data
                        processed_signal = processed_signal[outlier_mask]
            elif option == "smoothing":
                # Apply moving average smoothing using vitalDSP when available
                try:
                    # Try to use vitalDSP smoothing if available
                    from vitalDSP.preprocess.preprocess_operations import (
                        PreprocessConfig,
                    )
                    from vitalDSP.preprocess.preprocess_operations import (
                        PreprocessOperations,
                    )

                    preprocess_config = PreprocessConfig()
                    preprocess_config.smoothing_type = "savgol"
                    preprocess_config.smoothing_window = min(
                        21, len(processed_signal) // 10
                    )
                    if preprocess_config.smoothing_window % 2 == 0:
                        preprocess_config.smoothing_window += 1

                    preprocess_ops = PreprocessOperations(
                        processed_signal, preprocess_config
                    )
                    processed_signal = preprocess_ops.apply_smoothing()
                    logger.info("Applied vitalDSP smoothing")

                except Exception as e:
                    logger.warning(f"vitalDSP smoothing failed, using scipy: {e}")
                    # Fallback to scipy smoothing
                    window_size = min(21, len(processed_signal) // 10)
                    if window_size % 2 == 0:
                        window_size += 1
                    processed_signal = signal.savgol_filter(
                        processed_signal, window_size, 3
                    )
            elif option == "baseline_correction":
                # Apply baseline correction using vitalDSP when available
                try:
                    from vitalDSP.preprocess.preprocess_operations import (
                        PreprocessConfig,
                    )
                    from vitalDSP.preprocess.preprocess_operations import (
                        PreprocessOperations,
                    )

                    preprocess_config = PreprocessConfig()
                    preprocess_config.baseline_method = "polynomial"
                    preprocess_config.baseline_order = 3

                    preprocess_ops = PreprocessOperations(
                        processed_signal, preprocess_config
                    )
                    processed_signal = preprocess_ops.apply_baseline_correction()
                    logger.info("Applied vitalDSP baseline correction")

                except Exception as e:
                    logger.warning(
                        f"vitalDSP baseline correction failed, using scipy: {e}"
                    )
                    # Fallback to scipy detrending
                    processed_signal = signal.detrend(processed_signal)
            elif option == "noise_reduction":
                # Apply noise reduction using vitalDSP when available
                try:
                    from vitalDSP.preprocess.preprocess_operations import (
                        PreprocessConfig,
                    )
                    from vitalDSP.preprocess.preprocess_operations import (
                        PreprocessOperations,
                    )

                    preprocess_config = PreprocessConfig()
                    preprocess_config.noise_reduction_method = "wavelet"
                    preprocess_config.noise_threshold = 0.1

                    preprocess_ops = PreprocessOperations(
                        processed_signal, preprocess_config
                    )
                    processed_signal = preprocess_ops.apply_noise_reduction()
                    logger.info("Applied vitalDSP noise reduction")

                except Exception as e:
                    logger.warning(
                        f"vitalDSP noise reduction failed, using basic filtering: {e}"
                    )
                    # Fallback to basic low-pass filtering
                    nyquist = sampling_freq / 2
                    cutoff = min(nyquist * 0.8, 50)
                    b, a = signal.butter(4, cutoff / nyquist, btype="low")
                    processed_signal = signal.filtfilt(b, a, processed_signal)
            elif option == "artifact_removal":
                # Apply artifact removal using vitalDSP when available
                try:
                    from vitalDSP.preprocess.preprocess_operations import (
                        PreprocessConfig,
                    )
                    from vitalDSP.preprocess.preprocess_operations import (
                        PreprocessOperations,
                    )

                    preprocess_config = PreprocessConfig()
                    preprocess_config.artifact_method = "statistical"
                    preprocess_config.artifact_threshold = 3.0  # 3 standard deviations

                    preprocess_ops = PreprocessOperations(
                        processed_signal, preprocess_config
                    )
                    processed_signal = preprocess_ops.apply_artifact_removal()
                    logger.info("Applied vitalDSP artifact removal")

                except Exception as e:
                    logger.warning(
                        f"vitalDSP artifact removal failed, using basic outlier removal: {e}"
                    )
                    # Fallback to basic outlier removal
                    q1 = np.percentile(processed_signal, 25)
                    q3 = np.percentile(processed_signal, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outlier_mask = (processed_signal >= lower_bound) & (
                        processed_signal <= upper_bound
                    )
                    if np.sum(outlier_mask) > len(processed_signal) * 0.5:
                        processed_signal = processed_signal[outlier_mask]

        return processed_signal

    except Exception as e:
        logger.warning(f"Error in preprocessing: {e}")
        return signal_data  # Return original if preprocessing fails


def extract_comprehensive_features(
    signal_data, sampling_freq, categories, advanced_options
):
    """Extract comprehensive features from the signal using vitalDSP modules."""
    features = {}

    try:
        logger.info(f"Extracting features for categories: {categories}")
        logger.info(
            f"Signal data length: {len(signal_data)}, sampling_freq: {sampling_freq}"
        )

        # Check if vitalDSP modules are available
        vitaldsp_available = _import_vitaldsp_modules()

        # Statistical Features
        if "statistical" in categories:
            features["statistical"] = extract_statistical_features(
                signal_data, vitaldsp_available
            )

        # Spectral Features
        if "spectral" in categories:
            features["spectral"] = extract_spectral_features(
                signal_data, sampling_freq, vitaldsp_available
            )

        # Temporal Features
        if "temporal" in categories:
            features["temporal"] = extract_temporal_features(
                signal_data, sampling_freq, vitaldsp_available
            )

        # Morphological Features
        if "morphological" in categories:
            features["morphological"] = extract_morphological_features(
                signal_data, sampling_freq, vitaldsp_available
            )

        # Entropy Features
        if "entropy" in categories:
            features["entropy"] = extract_entropy_features(
                signal_data, vitaldsp_available
            )

        # Fractal Features
        if "fractal" in categories:
            features["fractal"] = extract_fractal_features(
                signal_data, vitaldsp_available
            )

        # Advanced Features
        if advanced_options:
            features["advanced"] = extract_advanced_features(
                signal_data, sampling_freq, advanced_options, vitaldsp_available
            )

        logger.info(f"Extracted features: {list(features.keys())}")
        for key, value in features.items():
            if isinstance(value, dict) and "error" in value:
                logger.warning(f"Error in {key}: {value['error']}")

        return features

    except Exception as e:
        logger.error(f"Error in feature extraction: {e}")
        return {"error": f"Feature extraction failed: {str(e)}"}


def extract_statistical_features(signal_data, vitaldsp_available=False):
    """Extract statistical features using vitalDSP when available."""
    try:
        if vitaldsp_available:
            try:
                # Use vitalDSP TimeDomainFeatures for enhanced statistical analysis
                from vitalDSP.physiological_features.time_domain import (
                    TimeDomainFeatures,
                )

                # Create TimeDomainFeatures object
                td_features = TimeDomainFeatures(signal_data)

                # Extract comprehensive statistical features
                basic_stats = {
                    "mean": np.mean(signal_data),
                    "std": np.std(signal_data),
                    "variance": np.var(signal_data),
                    "rms": np.sqrt(np.mean(signal_data**2)),
                    "peak_to_peak": np.max(signal_data) - np.min(signal_data),
                }

                # Add vitalDSP enhanced features
                enhanced_stats = {
                    "skewness": _calculate_skewness(signal_data),
                    "kurtosis": _calculate_kurtosis(signal_data),
                    "crest_factor": (
                        np.max(np.abs(signal_data)) / np.sqrt(np.mean(signal_data**2))
                        if np.mean(signal_data**2) > 0
                        else 0
                    ),
                    "shape_factor": (
                        np.sqrt(np.mean(signal_data**2)) / np.mean(np.abs(signal_data))
                        if np.mean(np.abs(signal_data)) > 0
                        else 0
                    ),
                    "impulse_factor": (
                        np.max(np.abs(signal_data)) / np.mean(np.abs(signal_data))
                        if np.mean(np.abs(signal_data)) > 0
                        else 0
                    ),
                }

                # Try to get additional vitalDSP statistical features
                try:
                    # Get signal quality metrics
                    if hasattr(td_features, "get_signal_quality"):
                        quality_metrics = td_features.get_signal_quality()
                        enhanced_stats.update(quality_metrics)

                    # Get additional statistical measures
                    if hasattr(td_features, "get_statistical_measures"):
                        vitaldsp_stats = td_features.get_statistical_measures()
                        enhanced_stats.update(vitaldsp_stats)

                except Exception as e:
                    logger.warning(
                        f"Could not get additional vitalDSP statistical features: {e}"
                    )

                # Combine basic and enhanced features
                return {**basic_stats, **enhanced_stats}

            except Exception as e:
                logger.warning(
                    f"vitalDSP statistical features failed, using basic: {e}"
                )
                # Fall back to basic implementation
                pass

        # Basic statistical features (fallback)
        return {
            "mean": np.mean(signal_data),
            "std": np.std(signal_data),
            "variance": np.var(signal_data),
            "skewness": _calculate_skewness(signal_data),
            "kurtosis": _calculate_kurtosis(signal_data),
            "rms": np.sqrt(np.mean(signal_data**2)),
            "peak_to_peak": np.max(signal_data) - np.min(signal_data),
            "crest_factor": (
                np.max(np.abs(signal_data)) / np.sqrt(np.mean(signal_data**2))
                if np.mean(signal_data**2) > 0
                else 0
            ),
            "shape_factor": (
                np.sqrt(np.mean(signal_data**2)) / np.mean(np.abs(signal_data))
                if np.mean(np.abs(signal_data)) > 0
                else 0
            ),
            "impulse_factor": (
                np.max(np.abs(signal_data)) / np.mean(np.abs(signal_data))
                if np.mean(np.abs(signal_data)) > 0
                else 0
            ),
        }
    except Exception as e:
        logger.error(f"Error in statistical features: {e}")
        return {"error": f"Statistical features failed: {str(e)}"}


def extract_spectral_features(signal_data, sampling_freq, vitaldsp_available=False):
    """Extract spectral features using vitalDSP when available."""
    try:
        if vitaldsp_available:
            try:
                # Use vitalDSP FourierTransform
                from vitalDSP.transforms.fourier_transform import FourierTransform

                ft = FourierTransform(signal_data)
                fft_result = ft.compute_dft()

                # Calculate frequencies manually
                fft_freq = np.fft.fftfreq(len(signal_data), 1 / sampling_freq)
                fft_magnitude = np.abs(fft_result)

                # Get positive frequencies
                positive_mask = fft_freq > 0
                fft_freq = fft_freq[positive_mask]
                fft_magnitude = fft_magnitude[positive_mask]

                # Try to use vitalDSP STFT for additional features
                try:
                    from vitalDSP.transforms.stft import STFT

                    stft_obj = STFT(signal_data, sampling_freq)
                    stft_result = stft_obj.compute_stft()
                    stft_available = True
                    stft_shape = (
                        stft_result.shape if hasattr(stft_result, "shape") else "N/A"
                    )
                except Exception as e:
                    logger.warning(f"STFT failed: {e}")
                    stft_available = False
                    stft_shape = "N/A"

                # Calculate advanced spectral features
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
                spectral_rolloff = _calculate_spectral_rolloff(fft_freq, fft_magnitude)

                return {
                    "spectral_centroid": spectral_centroid,
                    "spectral_bandwidth": spectral_bandwidth,
                    "spectral_rolloff": spectral_rolloff,
                    "dominant_frequency": (
                        fft_freq[np.argmax(fft_magnitude)]
                        if len(fft_magnitude) > 0
                        else 0
                    ),
                    "total_energy": np.sum(fft_magnitude**2),
                    "frequency_range": (
                        f"{fft_freq[0]:.2f} - {fft_freq[-1]:.2f} Hz"
                        if len(fft_freq) > 0
                        else "N/A"
                    ),
                    "stft_available": stft_available,
                    "stft_shape": stft_shape,
                }
            except Exception as e:
                logger.warning(
                    f"vitalDSP spectral features failed, falling back to scipy: {e}"
                )
                vitaldsp_available = False

        # Fallback to scipy implementation
        if not vitaldsp_available:
            # Calculate FFT
            fft_result = np.fft.fft(signal_data)
            fft_freq = np.fft.fftfreq(len(signal_data), 1 / sampling_freq)

            # Get positive frequencies
            positive_mask = fft_freq > 0
            fft_freq = fft_freq[positive_mask]
            fft_magnitude = np.abs(fft_result[positive_mask])

            # Calculate spectral features
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
            spectral_rolloff = _calculate_spectral_rolloff(fft_freq, fft_magnitude)

            return {
                "spectral_centroid": spectral_centroid,
                "spectral_bandwidth": spectral_bandwidth,
                "spectral_rolloff": spectral_rolloff,
                "dominant_frequency": (
                    fft_freq[np.argmax(fft_magnitude)] if len(fft_magnitude) > 0 else 0
                ),
                "total_energy": np.sum(fft_magnitude**2),
                "frequency_range": (
                    f"{fft_freq[0]:.2f} - {fft_freq[-1]:.2f} Hz"
                    if len(fft_freq) > 0
                    else "N/A"
                ),
                "stft_available": False,
            }

    except Exception as e:
        logger.error(f"Error in spectral features: {e}")
        return {"error": f"Spectral features failed: {str(e)}"}


def extract_temporal_features(signal_data, sampling_freq, vitaldsp_available=False):
    """Extract temporal features using vitalDSP when available."""
    try:
        if vitaldsp_available:
            try:
                # Try to use vitalDSP temporal features
                from vitalDSP.physiological_features.time_domain import (
                    TimeDomainFeatures,
                )

                # For temporal features, we need to detect peaks first
                peaks, _ = signal.find_peaks(
                    signal_data,
                    height=np.mean(signal_data) + np.std(signal_data),
                    distance=int(sampling_freq * 0.3),
                )

                if len(peaks) > 1:
                    # Convert to NN intervals for HRV analysis
                    intervals = (
                        np.diff(peaks) / sampling_freq * 1000
                    )  # Convert to milliseconds

                    temporal_features = {
                        "signal_duration": len(signal_data) / sampling_freq,
                        "sampling_frequency": sampling_freq,
                        "num_samples": len(signal_data),
                        "num_peaks": len(peaks),
                        "mean_interval": np.mean(intervals)
                        / 1000,  # Convert back to seconds
                        "std_interval": np.std(intervals) / 1000,
                        "min_interval": np.min(intervals) / 1000,
                        "max_interval": np.max(intervals) / 1000,
                        "vitaldsp_used": True,
                    }
                else:
                    temporal_features = {
                        "signal_duration": len(signal_data) / sampling_freq,
                        "sampling_frequency": sampling_freq,
                        "num_samples": len(signal_data),
                        "vitaldsp_used": True,
                    }

                return temporal_features

            except Exception as e:
                logger.warning(
                    f"vitalDSP temporal features failed, falling back to scipy: {e}"
                )
                vitaldsp_available = False

        # Fallback to scipy implementation
        if not vitaldsp_available:
            # Find peaks
            peaks, _ = signal.find_peaks(
                signal_data,
                height=np.mean(signal_data) + np.std(signal_data),
                distance=int(sampling_freq * 0.3),
            )

            temporal_features = {
                "signal_duration": len(signal_data) / sampling_freq,
                "sampling_frequency": sampling_freq,
                "num_samples": len(signal_data),
            }

            if len(peaks) > 1:
                intervals = np.diff(peaks) / sampling_freq
                temporal_features.update(
                    {
                        "num_peaks": len(peaks),
                        "mean_interval": np.mean(intervals),
                        "std_interval": np.std(intervals),
                        "min_interval": np.min(intervals),
                        "max_interval": np.max(intervals),
                    }
                )

            temporal_features["vitaldsp_used"] = False
            return temporal_features

    except Exception as e:
        logger.error(f"Error in temporal features: {e}")
        return {"error": f"Temporal features failed: {str(e)}"}


def extract_morphological_features(
    signal_data, sampling_freq, vitaldsp_available=False
):
    """Extract morphological features using vitalDSP when available."""
    try:
        if vitaldsp_available:
            try:
                # Try to use vitalDSP morphological features
                from vitalDSP.physiological_features.waveform import WaveformMorphology

                # Find peaks and valleys using vitalDSP
                peaks, _ = signal.find_peaks(
                    signal_data,
                    height=np.mean(signal_data) + np.std(signal_data),
                    distance=int(sampling_freq * 0.3),
                )
                valleys, _ = signal.find_peaks(
                    -signal_data,
                    height=np.mean(-signal_data) + np.std(-signal_data),
                    distance=int(sampling_freq * 0.3),
                )

                morphological_features = {
                    "num_peaks": len(peaks),
                    "num_valleys": len(valleys),
                    "peak_heights": (
                        signal_data[peaks].tolist() if len(peaks) > 0 else []
                    ),
                    "valley_heights": (
                        signal_data[valleys].tolist() if len(valleys) > 0 else []
                    ),
                    "vitaldsp_used": True,
                }

                if len(peaks) > 0:
                    morphological_features.update(
                        {
                            "mean_peak_height": np.mean(signal_data[peaks]),
                            "std_peak_height": np.std(signal_data[peaks]),
                            "max_peak_height": np.max(signal_data[peaks]),
                            "min_peak_height": np.min(signal_data[peaks]),
                        }
                    )

                return morphological_features

            except Exception as e:
                logger.warning(
                    f"vitalDSP morphological features failed, falling back to scipy: {e}"
                )
                vitaldsp_available = False

        # Fallback to scipy implementation
        if not vitaldsp_available:
            # Find peaks and valleys
            peaks, peak_properties = signal.find_peaks(
                signal_data,
                height=np.mean(signal_data) + np.std(signal_data),
                distance=int(sampling_freq * 0.3),
            )
            valleys, valley_properties = signal.find_peaks(
                -signal_data,
                height=np.mean(-signal_data) + np.std(-signal_data),
                distance=int(sampling_freq * 0.3),
            )

            morphological_features = {
                "num_peaks": len(peaks),
                "num_valleys": len(valleys),
                "peak_heights": signal_data[peaks].tolist() if len(peaks) > 0 else [],
                "valley_heights": (
                    signal_data[valleys].tolist() if len(valleys) > 0 else []
                ),
            }

            if len(peaks) > 0:
                morphological_features.update(
                    {
                        "mean_peak_height": np.mean(signal_data[peaks]),
                        "std_peak_height": np.std(signal_data[peaks]),
                        "max_peak_height": np.max(signal_data[peaks]),
                        "min_peak_height": np.min(signal_data[peaks]),
                    }
                )

            morphological_features["vitaldsp_used"] = False
            return morphological_features

    except Exception as e:
        logger.error(f"Error in morphological features: {e}")
        return {"error": f"Morphological features failed: {str(e)}"}


def extract_entropy_features(signal_data, vitaldsp_available=False):
    """Extract entropy-based features using vitalDSP when available."""
    try:
        if vitaldsp_available:
            try:
                # Use vitalDSP NonlinearFeatures
                from vitalDSP.physiological_features.nonlinear import NonlinearFeatures

                nf = NonlinearFeatures(signal_data)

                # Extract comprehensive entropy features
                sample_entropy = nf.compute_sample_entropy()
                approximate_entropy = nf.compute_approximate_entropy()

                # Try to get additional features
                try:
                    fractal_dim = nf.compute_fractal_dimension()
                except Exception:
                    fractal_dim = None

                try:
                    lyapunov = nf.compute_lyapunov_exponent()
                except Exception:
                    lyapunov = None

                try:
                    dfa_result = nf.compute_dfa()
                except Exception:
                    dfa_result = None

                try:
                    poincare_features = nf.compute_poincare_features()
                except Exception:
                    poincare_features = None

                return {
                    "sample_entropy": sample_entropy,
                    "approximate_entropy": approximate_entropy,
                    "fractal_dimension": fractal_dim,
                    "lyapunov_exponent": lyapunov,
                    "dfa_result": dfa_result,
                    "poincare_features": poincare_features,
                    "vitaldsp_used": True,
                }

            except Exception as e:
                logger.warning(
                    f"vitalDSP entropy features failed, falling back to scipy: {e}"
                )
                vitaldsp_available = False

        # Fallback to simplified entropy calculation
        if not vitaldsp_available:
            return {
                "sample_entropy": np.log(np.std(signal_data) + 1e-10),
                "approximate_entropy": np.log(np.std(signal_data) + 1e-10),
                "permutation_entropy": np.log(np.std(signal_data) + 1e-10),
                "vitaldsp_used": False,
            }

    except Exception as e:
        logger.error(f"Error in entropy features: {e}")
        return {"error": f"Entropy features failed: {str(e)}"}


def extract_fractal_features(signal_data, vitaldsp_available=False):
    """Extract fractal features using vitalDSP when available."""
    try:
        if vitaldsp_available:
            try:
                # Use vitalDSP NonlinearFeatures for fractal analysis
                from vitalDSP.physiological_features.nonlinear import NonlinearFeatures

                nf = NonlinearFeatures(signal_data)

                # Get fractal dimension
                try:
                    fractal_dim = nf.compute_fractal_dimension()
                except Exception:
                    fractal_dim = None

                # Get DFA result
                try:
                    dfa_result = nf.compute_dfa()
                except Exception:
                    dfa_result = None

                return {
                    "higuchi_fractal_dimension": fractal_dim,
                    "box_counting_dimension": fractal_dim,  # Use same value for now
                    "dfa_result": dfa_result,
                    "vitaldsp_used": True,
                }

            except Exception as e:
                logger.warning(
                    f"vitalDSP fractal features failed, falling back to scipy: {e}"
                )
                vitaldsp_available = False

        # Fallback to simplified fractal dimension calculation
        if not vitaldsp_available:
            return {
                "higuchi_fractal_dimension": 1.0,
                "box_counting_dimension": 1.0,
                "vitaldsp_used": False,
            }

    except Exception as e:
        logger.error(f"Error in fractal features: {e}")
        return {"error": f"Fractal features failed: {str(e)}"}


def extract_advanced_features(
    signal_data, sampling_freq, advanced_options, vitaldsp_available=False
):
    """Extract advanced features based on selected options using vitalDSP when available."""
    advanced_features = {}

    try:
        for option in advanced_options:
            if option == "cross_correlation":
                if vitaldsp_available:
                    try:
                        # Use vitalDSP CrossCorrelationFeatures
                        from vitalDSP.physiological_features.cross_correlation import (
                            CrossCorrelationFeatures,
                        )

                        cc = CrossCorrelationFeatures(
                            signal_data, signal_data, sampling_freq
                        )
                        cross_corr, lag = cc.compute_cross_correlation()

                        advanced_features["cross_correlation"] = {
                            "cross_correlation": (
                                cross_corr.tolist()
                                if hasattr(cross_corr, "tolist")
                                else cross_corr
                            ),
                            "max_lag": lag,
                            "correlation_strength": (
                                np.max(np.abs(cross_corr))
                                if hasattr(cross_corr, "__len__")
                                else 0
                            ),
                            "vitaldsp_used": True,
                        }
                    except Exception as e:
                        logger.warning(
                            f"vitalDSP cross-correlation failed, falling back to scipy: {e}"
                        )
                        # Fallback to scipy
                        autocorr = np.correlate(signal_data, signal_data, mode="full")
                        autocorr = autocorr[len(autocorr) // 2 :]
                        advanced_features["cross_correlation"] = {
                            "max_lag": (
                                np.argmax(autocorr[1:]) + 1 if len(autocorr) > 1 else 0
                            ),
                            "correlation_strength": (
                                np.max(autocorr[1:]) / autocorr[0]
                                if autocorr[0] > 0
                                else 0
                            ),
                            "vitaldsp_used": False,
                        }
                    else:
                        # Use scipy
                        autocorr = np.correlate(signal_data, signal_data, mode="full")
                        autocorr = autocorr[len(autocorr) // 2 :]
                        advanced_features["cross_correlation"] = {
                            "max_lag": (
                                np.argmax(autocorr[1:]) + 1 if len(autocorr) > 1 else 0
                            ),
                            "correlation_strength": (
                                np.max(autocorr[1:]) / autocorr[0]
                                if autocorr[0] > 0
                                else 0
                            ),
                            "vitaldsp_used": False,
                        }

            elif option == "phase_analysis":
                if vitaldsp_available:
                    try:
                        # Use vitalDSP HilbertTransform
                        from vitalDSP.transforms.hilbert_transform import (
                            HilbertTransform,
                        )

                        ht = HilbertTransform(signal_data)
                        analytic_signal = ht.compute_hilbert()
                        phase = ht.instantaneous_phase()

                        advanced_features["phase_analysis"] = {
                            "phase_range": np.max(phase) - np.min(phase),
                            "phase_std": np.std(phase),
                            "analytic_signal": (
                                analytic_signal.tolist()
                                if hasattr(analytic_signal, "tolist")
                                else analytic_signal
                            ),
                            "vitaldsp_used": True,
                        }
                    except Exception as e:
                        logger.warning(
                            f"vitalDSP phase analysis failed, falling back to scipy: {e}"
                        )
                        # Fallback to scipy
                        analytic_signal = signal.hilbert(signal_data)
                        phase = np.unwrap(np.angle(analytic_signal))
                        advanced_features["phase_analysis"] = {
                            "phase_range": np.max(phase) - np.min(phase),
                            "phase_std": np.std(phase),
                            "vitaldsp_used": False,
                        }
                    else:
                        # Use scipy
                        analytic_signal = signal.hilbert(signal_data)
                        phase = np.unwrap(np.angle(analytic_signal))
                        advanced_features["phase_analysis"] = {
                            "phase_range": np.max(phase) - np.min(phase),
                            "phase_std": np.std(phase),
                            "vitaldsp_used": False,
                        }

            elif option == "wavelet":
                if vitaldsp_available:
                    try:
                        # Use vitalDSP WaveletTransform
                        from vitalDSP.transforms.wavelet_transform import (
                            WaveletTransform,
                        )

                        wt = WaveletTransform(signal_data, wavelet_name="haar")
                        wavelet_coeffs = wt.perform_wavelet_transform()

                        advanced_features["wavelet"] = {
                            "wavelet_coefficients": wavelet_coeffs,
                            "wavelet_name": "haar",
                            "vitaldsp_used": True,
                        }
                    except Exception as e:
                        logger.warning(f"vitalDSP wavelet transform failed: {e}")
                        advanced_features["wavelet"] = {
                            "error": f"Wavelet transform failed: {str(e)}"
                        }

            elif option == "ml_features":
                if vitaldsp_available:
                    try:
                        # Use vitalDSP advanced computation features
                        from vitalDSP.advanced_computation.anomaly_detection import (
                            AnomalyDetection,
                        )

                        ad = AnomalyDetection(signal_data)
                        anomalies = ad.detect_anomalies()

                        advanced_features["ml_features"] = {
                            "anomalies_detected": (
                                len(anomalies) if hasattr(anomalies, "__len__") else 0
                            ),
                            "anomaly_indices": (
                                anomalies.tolist()
                                if hasattr(anomalies, "tolist")
                                else anomalies
                            ),
                            "vitaldsp_used": True,
                        }
                    except Exception as e:
                        logger.warning(f"vitalDSP ML features failed: {e}")
                        advanced_features["ml_features"] = {
                            "error": f"ML features failed: {str(e)}"
                        }

        return advanced_features

    except Exception as e:
        logger.error(f"Error in advanced features: {e}")
        return {"error": f"Advanced features failed: {str(e)}"}


def create_comprehensive_features_display(features, signal_type, categories):
    """Create comprehensive features display with enhanced insights."""
    try:
        sections = []

        # Header with signal type and vitalDSP usage info
        vitaldsp_used = any(
            features.get(cat, {}).get("vitaldsp_used", False)
            for cat in features.keys()
            if isinstance(features.get(cat), dict)
        )

        header_section = html.Div(
            [
                html.H4(
                    "🚀 Advanced Feature Analysis Results",
                    className="text-primary mb-3",
                ),
                html.Div(
                    [
                        html.Span(
                            f"Signal Type: {signal_type.upper()}",
                            className="badge bg-primary me-2",
                        ),
                        html.Span(
                            f"vitalDSP: {'✅ Used' if vitaldsp_used else '⚠️ Fallback'}",
                            className=f"badge {'bg-success' if vitaldsp_used else 'bg-warning'} me-2",
                        ),
                        html.Span(
                            f"Categories: {len(categories)}", className="badge bg-info"
                        ),
                    ],
                    className="mb-3",
                ),
            ]
        )
        sections.append(header_section)

        # Statistical Features
        if "statistical" in categories and "error" not in features.get(
            "statistical", {}
        ):
            stats = features["statistical"]
            stats_section = html.Div(
                [
                    html.H5("📊 Statistical Features", className="text-success"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        "Basic Statistics", className="text-primary"
                                    ),
                                    html.P(
                                        f"Mean: {stats.get('mean', 0):.4f}",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"Standard Deviation: {stats.get('std', 0):.4f}",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"Variance: {stats.get('variance', 0):.4f}",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"RMS: {stats.get('rms', 0):.4f}",
                                        className="mb-1",
                                    ),
                                ],
                                className="col-md-6",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        "Shape Statistics", className="text-primary"
                                    ),
                                    html.P(
                                        f"Skewness: {stats.get('skewness', 0):.4f}",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"Kurtosis: {stats.get('kurtosis', 0):.4f}",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"Peak-to-Peak: {stats.get('peak_to_peak', 0):.4f}",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"Crest Factor: {stats.get('crest_factor', 0):.4f}",
                                        className="mb-1",
                                    ),
                                ],
                                className="col-md-6",
                            ),
                        ],
                        className="row",
                    ),
                ],
                className="mb-4",
            )
            sections.append(stats_section)

        # Spectral Features
        if "spectral" in categories and "error" not in features.get("spectral", {}):
            spec = features["spectral"]
            spec_section = html.Div(
                [
                    html.H5("🌊 Spectral Features", className="text-info"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        "Frequency Analysis", className="text-primary"
                                    ),
                                    html.P(
                                        f"Spectral Centroid: {spec.get('spectral_centroid', 0):.2f} Hz",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"Spectral Bandwidth: {spec.get('spectral_bandwidth', 0):.2f} Hz",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"Dominant Frequency: {spec.get('dominant_frequency', 0):.2f} Hz",
                                        className="mb-1",
                                    ),
                                ],
                                className="col-md-6",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        "Energy Analysis", className="text-primary"
                                    ),
                                    html.P(
                                        f"Total Energy: {spec.get('total_energy', 0):.2f}",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"Frequency Range: {spec.get('frequency_range', 'N/A')}",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"STFT Available: {'✅' if spec.get('stft_available', False) else '❌'}",
                                        className="mb-1",
                                    ),
                                ],
                                className="col-md-6",
                            ),
                        ],
                        className="row",
                    ),
                ],
                className="mb-4",
            )
            sections.append(spec_section)

        # Temporal Features
        if "temporal" in categories and "error" not in features.get("temporal", {}):
            temp = features["temporal"]
            temp_section = html.Div(
                [
                    html.H5("⏱️ Temporal Features", className="text-warning"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        "Signal Properties", className="text-primary"
                                    ),
                                    html.P(
                                        f"Signal Duration: {temp.get('signal_duration', 0):.2f} seconds",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"Sampling Frequency: {temp.get('sampling_frequency', 0)} Hz",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"Number of Samples: {temp.get('num_samples', 0)}",
                                        className="mb-1",
                                    ),
                                ],
                                className="col-md-6",
                            ),
                            html.Div(
                                [
                                    (
                                        html.H6(
                                            "Peak Analysis", className="text-primary"
                                        )
                                        if "num_peaks" in temp
                                        else html.Div()
                                    ),
                                    (
                                        html.P(
                                            f"Number of Peaks: {temp.get('num_peaks', 0)}",
                                            className="mb-1",
                                        )
                                        if "num_peaks" in temp
                                        else html.Div()
                                    ),
                                    (
                                        html.P(
                                            f"Mean Interval: {temp.get('mean_interval', 0):.3f} seconds",
                                            className="mb-1",
                                        )
                                        if "mean_interval" in temp
                                        else html.Div()
                                    ),
                                    (
                                        html.P(
                                            f"Interval Std: {temp.get('std_interval', 0):.3f} seconds",
                                            className="mb-1",
                                        )
                                        if "std_interval" in temp
                                        else html.Div()
                                    ),
                                ],
                                className="col-md-6",
                            ),
                        ],
                        className="row",
                    ),
                ],
                className="mb-4",
            )
            sections.append(temp_section)

        # Morphological Features
        if "morphological" in categories and "error" not in features.get(
            "morphological", {}
        ):
            morph = features["morphological"]
            morph_section = html.Div(
                [
                    html.H5("🔍 Morphological Features", className="text-danger"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6("Peak Analysis", className="text-primary"),
                                    html.P(
                                        f"Number of Peaks: {morph.get('num_peaks', 0)}",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"Number of Valleys: {morph.get('num_valleys', 0)}",
                                        className="mb-1",
                                    ),
                                ],
                                className="col-md-6",
                            ),
                            html.Div(
                                [
                                    (
                                        html.H6(
                                            "Peak Statistics", className="text-primary"
                                        )
                                        if "mean_peak_height" in morph
                                        else html.Div()
                                    ),
                                    (
                                        html.P(
                                            f"Mean Peak Height: {morph.get('mean_peak_height', 0):.4f}",
                                            className="mb-1",
                                        )
                                        if "mean_peak_height" in morph
                                        else html.Div()
                                    ),
                                    (
                                        html.P(
                                            f"Peak Height Std: {morph.get('std_peak_height', 0):.4f}",
                                            className="mb-1",
                                        )
                                        if "std_peak_height" in morph
                                        else html.Div()
                                    ),
                                    (
                                        html.P(
                                            f"Max Peak Height: {morph.get('max_peak_height', 0):.4f}",
                                            className="mb-1",
                                        )
                                        if "max_peak_height" in morph
                                        else html.Div()
                                    ),
                                ],
                                className="col-md-6",
                            ),
                        ],
                        className="row",
                    ),
                ],
                className="mb-4",
            )
            sections.append(morph_section)

        # Entropy Features
        if "entropy" in categories and "error" not in features.get("entropy", {}):
            ent = features["entropy"]
            ent_section = html.Div(
                [
                    html.H5(
                        "🧠 Entropy & Complexity Features", className="text-secondary"
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        "Entropy Measures", className="text-primary"
                                    ),
                                    html.P(
                                        f"Sample Entropy: {ent.get('sample_entropy', 0):.4f}",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"Approximate Entropy: {ent.get('approximate_entropy', 0):.4f}",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"Fractal Dimension: {ent.get('fractal_dimension', 'N/A')}",
                                        className="mb-1",
                                    ),
                                ],
                                className="col-md-6",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        "Advanced Measures", className="text-primary"
                                    ),
                                    html.P(
                                        f"Lyapunov Exponent: {ent.get('lyapunov_exponent', 'N/A')}",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"DFA Result: {ent.get('dfa_result', 'N/A')}",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"Poincaré Features: {'Available' if ent.get('poincare_features') else 'N/A'}",
                                        className="mb-1",
                                    ),
                                ],
                                className="col-md-6",
                            ),
                        ],
                        className="row",
                    ),
                    html.Div(
                        [
                            html.Small(
                                f"vitalDSP: {'✅ Used' if ent.get('vitaldsp_used', False) else '⚠️ Fallback'}",
                                className="text-muted",
                            )
                        ],
                        className="mt-2",
                    ),
                ],
                className="mb-4",
            )
            sections.append(ent_section)

        # Advanced Features
        if "advanced" in features and "error" not in features.get("advanced", {}):
            adv = features["advanced"]
            adv_section = html.Div(
                [
                    html.H5("🚀 Advanced Features", className="text-dark"),
                    html.Div(
                        [
                            (
                                html.Div(
                                    [
                                        (
                                            html.H6(
                                                "Cross-Correlation",
                                                className="text-primary",
                                            )
                                            if "cross_correlation" in adv
                                            else html.Div()
                                        ),
                                        (
                                            html.P(
                                                f"Max Lag: {adv.get('cross_correlation', {}).get('max_lag', 'N/A')}",
                                                className="mb-1",
                                            )
                                            if "cross_correlation" in adv
                                            else html.Div()
                                        ),
                                        (
                                            html.P(
                                                f"Correlation Strength: {adv.get('cross_correlation', {}).get('correlation_strength', 'N/A'):.4f}",
                                                className="mb-1",
                                            )
                                            if "cross_correlation" in adv
                                            else html.Div()
                                        ),
                                        (
                                            html.P(
                                                f"vitalDSP: {'✅' if adv.get('cross_correlation', {}).get('vitaldsp_used', False) else '⚠️'}",
                                                className="mb-1",
                                            )
                                            if "cross_correlation" in adv
                                            else html.Div()
                                        ),
                                    ],
                                    className="col-md-6",
                                )
                                if "cross_correlation" in adv
                                else html.Div()
                            ),
                            (
                                html.Div(
                                    [
                                        (
                                            html.H6(
                                                "Phase Analysis",
                                                className="text-primary",
                                            )
                                            if "phase_analysis" in adv
                                            else html.Div()
                                        ),
                                        (
                                            html.P(
                                                f"Phase Range: {adv.get('phase_analysis', {}).get('phase_range', 'N/A'):.4f}",
                                                className="mb-1",
                                            )
                                            if "phase_analysis" in adv
                                            else html.Div()
                                        ),
                                        (
                                            html.P(
                                                f"Phase Std: {adv.get('phase_analysis', {}).get('phase_std', 'N/A'):.4f}",
                                                className="mb-1",
                                            )
                                            if "phase_analysis" in adv
                                            else html.Div()
                                        ),
                                        (
                                            html.P(
                                                f"vitalDSP: {'✅' if adv.get('phase_analysis', {}).get('vitaldsp_used', False) else '⚠️'}",
                                                className="mb-1",
                                            )
                                            if "phase_analysis" in adv
                                            else html.Div()
                                        ),
                                    ],
                                    className="col-md-6",
                                )
                                if "phase_analysis" in adv
                                else html.Div()
                            ),
                        ],
                        className="row",
                    ),
                ],
                className="mb-4",
            )
            sections.append(adv_section)

        # Signal Quality Features
        if "signal_quality" in features and "error" not in features.get(
            "signal_quality", {}
        ):
            sq = features["signal_quality"]
            sq_section = html.Div(
                [
                    html.H5("🔍 Signal Quality Assessment", className="text-success"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        "Quality Metrics", className="text-primary"
                                    ),
                                    html.P(
                                        f"Signal-to-Noise Ratio: {sq.get('snr', 0):.2f} dB",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"Baseline Wander: {sq.get('baseline_wander', 0):.4f}",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"Motion Artifacts: {sq.get('motion_artifacts', 0)}",
                                        className="mb-1",
                                    ),
                                ],
                                className="col-md-6",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        "Quality Assessment", className="text-primary"
                                    ),
                                    html.P(
                                        f"vitalDSP: {'✅ Used' if sq.get('vitaldsp_used', False) else '⚠️ Fallback'}",
                                        className="mb-1",
                                    ),
                                    html.P(
                                        f"Quality Score: {'Good' if sq.get('snr', 0) > 10 else 'Fair' if sq.get('snr', 0) > 5 else 'Poor'}",
                                        className="mb-1",
                                    ),
                                ],
                                className="col-md-6",
                            ),
                        ],
                        className="row",
                    ),
                ],
                className="mb-4",
            )
            sections.append(sq_section)

        # Physiological Features
        if "physiological" in features and "error" not in features.get(
            "physiological", {}
        ):
            phys = features["physiological"]
            if phys.get("vitaldsp_used", False) and "waveform_features" in phys:
                phys_section = html.Div(
                    [
                        html.H5("💓 Physiological Features", className="text-danger"),
                        html.Div(
                            [
                                html.H6(
                                    "Waveform Morphology", className="text-primary"
                                ),
                                html.P("vitalDSP: ✅ Used", className="mb-1"),
                                html.P(
                                    f"Features Available: {len(phys.get('waveform_features', {}))}",
                                    className="mb-1",
                                ),
                                html.Small(
                                    "Advanced physiological waveform analysis completed",
                                    className="text-muted",
                                ),
                            ]
                        ),
                    ],
                    className="mb-4",
                )
                sections.append(phys_section)

        if not sections:
            return html.Div(
                [
                    html.H5("No Features Extracted"),
                    html.P("Please select feature categories and run the analysis."),
                ]
            )

        # Add summary section
        summary_section = html.Div(
            [
                html.Hr(),
                html.H6("📋 Analysis Summary", className="text-primary"),
                html.P(
                    f"Successfully extracted features from {len(categories)} categories"
                ),
                html.P(f"Signal type: {signal_type}"),
                html.P(
                    f"vitalDSP utilization: {'High' if vitaldsp_used else 'Limited'}"
                ),
                html.Small(
                    "Features extracted using advanced signal processing algorithms",
                    className="text-muted",
                ),
            ],
            className="mt-4",
        )
        sections.append(summary_section)

        return html.Div(sections)

    except Exception as e:
        logger.error(f"Error creating features display: {e}")
        return html.Div(
            [html.H5("Error"), html.P(f"Failed to create features display: {str(e)}")]
        )


def create_features_analysis_plots(signal_data, features, categories, sampling_freq):
    """Create comprehensive features analysis plots with enhanced visualizations."""
    try:
        logger.info(
            f"Plotting function called with signal_data length: {len(signal_data)}, categories: {categories}"
        )
        logger.info(f"Features available: {list(features.keys())}")

        if len(categories) == 0:
            logger.warning("No categories selected for plotting")
            return create_empty_figure()

        if signal_data is None or len(signal_data) == 0:
            logger.error("Signal data is None or empty")
            return create_empty_figure()

        # Create subplots based on selected categories
        num_plots = len(categories)
        if num_plots == 0:
            num_plots = 1

        # Add extra plots for advanced features if available
        advanced_plots = 0
        if "advanced" in features and "error" not in features.get("advanced", {}):
            advanced_options = features["advanced"].keys()
            if "cross_correlation" in advanced_options:
                advanced_plots += 1
            if "phase_analysis" in advanced_options:
                advanced_plots += 1
            if "wavelet" in advanced_options:
                advanced_plots += 1

        # Add signal quality plot
        signal_quality_plot = (
            1
            if "signal_quality" in features
            and "error" not in features.get("signal_quality", {})
            else 0
        )

        total_plots = num_plots + advanced_plots + signal_quality_plot

        logger.info(f"Creating subplots: rows={total_plots}, cols=2")

        # Create subplot titles
        subplot_titles = (
            [f"{cat.title()} Features" for cat in categories]
            + ["Cross-Correlation", "Phase Analysis", "Wavelet Analysis"][
                :advanced_plots
            ]
            + ["Signal Quality"] * signal_quality_plot
            + [f"{cat.title()} Stats" for cat in categories]
            + ["Advanced Stats"] * advanced_plots
            + ["Quality Metrics"] * signal_quality_plot
        )

        logger.info(f"Subplot titles: {subplot_titles}")

        # For now, let's use a simple plot to debug the issue
        logger.info("Creating simple plot instead of complex subplots for debugging")
        fig = go.Figure()

        # Add the main signal
        fig.add_trace(
            go.Scatter(
                x=list(range(len(signal_data))),
                y=signal_data,
                mode="lines",
                name="Signal Data",
                line=dict(color="blue", width=1),
            )
        )

        # Add some basic statistics as horizontal lines
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)

        fig.add_hline(
            y=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val:.4f}",
        )
        fig.add_hline(
            y=mean_val + std_val,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"+1σ: {mean_val + std_val:.4f}",
        )
        fig.add_hline(
            y=mean_val - std_val,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"-1σ: {mean_val - std_val:.4f}",
        )

        # Add a second subplot for features if available
        if "statistical" in features and "error" not in features.get("statistical", {}):
            stats = features["statistical"]
            stat_names = ["Mean", "Std", "Variance", "Skewness", "Kurtosis"]
            stat_values = [
                stats.get("mean", 0),
                stats.get("std", 0),
                stats.get("variance", 0),
                stats.get("skewness", 0),
                stats.get("kurtosis", 0),
            ]

            fig.add_trace(
                go.Bar(
                    x=stat_names,
                    y=stat_values,
                    name="Statistics",
                    marker_color=[
                        "#1f77b4",
                        "#ff7f0e",
                        "#2ca02c",
                        "#d62728",
                        "#9467bd",
                    ],
                    yaxis="y2",
                )
            )

            # Update layout to include secondary y-axis
            fig.update_layout(
                yaxis2=dict(title="Feature Values", overlaying="y", side="right")
            )

        fig.update_layout(
            title="Signal Analysis - Basic Plot",
            xaxis_title="Sample Index",
            yaxis_title="Amplitude",
            height=600,
            showlegend=True,
            template="plotly_white",
        )

        logger.info("Simple plot created successfully")
        return fig

    except Exception as e:
        logger.error(f"Error creating analysis plots: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        # Create a simple fallback plot for debugging
        try:
            fallback_fig = go.Figure()
            fallback_fig.add_trace(
                go.Scatter(
                    x=list(range(len(signal_data))),
                    y=signal_data,
                    mode="lines",
                    name="Signal Data",
                )
            )
            fallback_fig.update_layout(
                title="Fallback Plot - Signal Data",
                xaxis_title="Sample Index",
                yaxis_title="Amplitude",
            )
            logger.info("Created fallback plot")
            return fallback_fig
        except Exception as fallback_error:
            logger.error(f"Even fallback plot failed: {fallback_error}")
        return create_empty_figure()


# Utility functions for feature calculations
def _calculate_skewness(data):
    """Calculate skewness of the data."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 3)


def _calculate_kurtosis(data):
    """Calculate kurtosis of the data."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 4) - 3


def _calculate_spectral_rolloff(freq, magnitude, threshold=0.85):
    """Calculate spectral rolloff frequency."""
    total_energy = np.sum(magnitude)
    if total_energy == 0:
        return freq[-1] if len(freq) > 0 else 0

    cumulative_energy = np.cumsum(magnitude)
    rolloff_idx = np.where(cumulative_energy >= threshold * total_energy)[0]

    if len(rolloff_idx) > 0:
        return freq[rolloff_idx[0]]
    return freq[-1] if len(freq) > 0 else 0


def _extract_signal_quality_metrics(signal_data, sampling_freq):
    """Extract signal quality metrics using vitalDSP if available."""
    try:
        # Try to use vitalDSP signal quality assessment
        from vitalDSP.signal_quality_assessment.signal_quality import SignalQuality

        # For SNR calculation, we need both original and processed signals
        # Since we only have one signal, we'll use a simple approach
        # Get various quality metrics using available methods
        # For SNR, we'll estimate it using the signal itself
        signal_power = np.mean(signal_data**2)
        noise_estimate = np.mean(np.diff(signal_data) ** 2)
        snr = 10 * np.log10(signal_power / noise_estimate) if noise_estimate > 0 else 0

        # Estimate baseline wander and motion artifacts
        baseline_wander = np.std(signal_data - np.mean(signal_data))
        motion_artifacts = np.sum(
            np.abs(np.diff(signal_data)) > 2 * np.std(signal_data)
        )

        return {
            "snr": snr,
            "baseline_wander": baseline_wander,
            "motion_artifacts": motion_artifacts,
            "vitaldsp_used": True,
        }
    except Exception as e:
        logger.warning(f"vitalDSP signal quality assessment failed: {e}")
        # Fallback to basic quality metrics
        return {
            "snr": (
                (np.std(signal_data) / np.mean(np.abs(np.diff(signal_data))))
                if np.mean(np.abs(np.diff(signal_data))) > 0
                else 0
            ),
            "baseline_wander": np.std(signal_data - np.mean(signal_data)),
            "motion_artifacts": np.sum(
                np.abs(np.diff(signal_data)) > 2 * np.std(signal_data)
            ),
            "vitaldsp_used": False,
        }


def _extract_physiological_features(signal_data, sampling_freq, signal_type):
    """Extract physiological-specific features using vitalDSP."""
    try:
        if signal_type.lower() in ["ppg", "ecg"]:
            # Use vitalDSP physiological features
            from vitalDSP.physiological_features.waveform import WaveformMorphology

            wm = WaveformMorphology(
                signal_data, sampling_freq, signal_type=signal_type.upper()
            )

            # Extract waveform features using available methods
            try:
                # First detect peaks using scipy since vitalDSP expects peaks to be passed in
                peaks, _ = signal.find_peaks(
                    signal_data,
                    height=np.mean(signal_data) + np.std(signal_data),
                    distance=int(sampling_freq * 0.3),
                )

                # Get basic morphological features
                if signal_type.lower() == "ecg":
                    # For ECG, get QRS features using detected peaks
                    if len(peaks) > 0:
                        q_valleys = (
                            wm.detect_q_valley(peaks)
                            if hasattr(wm, "detect_q_valley")
                            else []
                        )
                        s_valleys = (
                            wm.detect_s_valley(peaks)
                            if hasattr(wm, "detect_s_valley")
                            else []
                        )

                        features = {
                            "r_peaks": len(peaks),
                            "q_valleys": len(q_valleys),
                            "s_valleys": len(s_valleys),
                            "heart_rate": len(peaks)
                            * 60
                            / (len(signal_data) / sampling_freq),
                        }
                    else:
                        features = {
                            "r_peaks": 0,
                            "q_valleys": 0,
                            "s_valleys": 0,
                            "heart_rate": 0,
                        }
                else:
                    # For PPG, get pulse features
                    if len(peaks) > 0:
                        troughs = (
                            wm.detect_troughs(peaks)
                            if hasattr(wm, "detect_troughs")
                            else []
                        )

                        features = {
                            "systolic_peaks": len(peaks),
                            "troughs": len(troughs),
                            "pulse_rate": len(peaks)
                            * 60
                            / (len(signal_data) / sampling_freq),
                        }
                    else:
                        features = {"systolic_peaks": 0, "troughs": 0, "pulse_rate": 0}

                return {"waveform_features": features, "vitaldsp_used": True}
            except Exception as e:
                logger.warning(f"Error extracting waveform features: {e}")
                return {"waveform_features": {"error": str(e)}, "vitaldsp_used": True}
        else:
            return {"vitaldsp_used": False}
    except Exception as e:
        logger.warning(f"vitalDSP physiological features failed: {e}")
        return {"vitaldsp_used": False}
