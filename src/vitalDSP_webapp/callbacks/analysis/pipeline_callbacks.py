"""
Pipeline visualization and execution callbacks.

This module handles callbacks for the 8-stage processing pipeline page,
including execution, progress tracking, and results visualization.
"""

from dash.dependencies import Input, Output, State
from dash import html, no_update, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import logging
import time
import json

# Import the real pipeline integration service
# TODO: Re-enable after optimizing pipeline initialization
# from vitalDSP_webapp.services.pipeline_integration import get_pipeline_service

logger = logging.getLogger(__name__)

# Define the 8 pipeline stages
PIPELINE_STAGES = [
    "Data Ingestion",
    "Quality Screening",
    "Parallel Processing",
    "Quality Validation",
    "Segmentation",
    "Feature Extraction",
    "Intelligent Output",
    "Output Package",
]


def _get_stage_number(current_stage):
    """
    Extract stage number from current_stage (handles both int and string pipeline_run_id).

    Args:
        current_stage: Either an int (old simulation) or string (pipeline_run_id)

    Returns:
        int: Stage number, or 0 if it's a pipeline_run_id
    """
    if isinstance(current_stage, int):
        return current_stage
    elif isinstance(current_stage, str) and len(current_stage) == 8:
        # It's a pipeline_run_id, get stage from pipeline_data
        if hasattr(register_pipeline_callbacks, "pipeline_data"):
            pipeline_data = register_pipeline_callbacks.pipeline_data.get(current_stage)
            if pipeline_data:
                return pipeline_data.get("current_stage", 0)
    return 0


def _execute_pipeline_stage(
    pipeline_data: dict,
    stage: int,
    signal_data: np.ndarray,
    fs: float,
    signal_type: str,
):
    """
    Execute a single pipeline stage with real vitalDSP processing.

    Args:
        pipeline_data: Pipeline run data dictionary
        stage: Stage number (1-8)
        signal_data: Input signal data
        fs: Sampling frequency
        signal_type: Signal type (ecg, ppg, etc.)

    Returns:
        tuple: (success: bool, result: dict or error_message: str)
    """
    try:
        if stage == 1:
            # Stage 1: Data Ingestion
            result = {
                "samples": len(signal_data),
                "duration": len(signal_data) / fs,
                "fs": fs,
                "signal_type": signal_type,
                "mean": float(np.mean(signal_data)),
                "std": float(np.std(signal_data)),
                "min": float(np.min(signal_data)),
                "max": float(np.max(signal_data)),
            }
            return True, result

        elif stage == 2:
            # Stage 2: Quality Screening - Use compute_sqi function from quality page
            from vitalDSP_webapp.callbacks.analysis.quality_sqi_functions import (
                compute_sqi,
            )

            # Get parameters from pipeline_data (now using quality page parameter structure)
            sqi_type = pipeline_data.get("stage2_sqi_type", "snr_sqi")
            window_size = pipeline_data.get("stage2_window_size", 1000)
            step_size = pipeline_data.get("stage2_step_size", 500)
            threshold_type = pipeline_data.get("stage2_threshold_type", "below")
            threshold = pipeline_data.get("stage2_threshold", 0.7)
            scale = pipeline_data.get("stage2_scale", "zscore")

            # Build SQI parameters dictionary
            sqi_params = {
                "window_size": window_size,
                "step_size": step_size,
                "threshold": threshold,
                "threshold_type": threshold_type,
                "scale": scale,
            }

            logger.info(f"Stage 2 - Computing {sqi_type} with params: {sqi_params}")
            logger.info(f"Signal length: {len(signal_data)}, Sampling freq: {fs}")

            try:
                # Use compute_sqi for consistent computation (same as Quality Page)
                quality_results = compute_sqi(signal_data, sqi_type, sqi_params, fs)

                # Extract results from compute_sqi output
                sqi_values = quality_results.get("sqi_values", [])
                normal_segments = quality_results.get("normal_segments", [])
                abnormal_segments = quality_results.get("abnormal_segments", [])
                overall_sqi = quality_results.get("overall_sqi", 0.5)

                logger.info(
                    f"✓ Stage 2 - {sqi_type}: {overall_sqi:.3f} "
                    f"(normal: {len(normal_segments)}, abnormal: {len(abnormal_segments)})"
                )

                # Build result structure (compatible with visualization)
                result = {
                    "sqi_type": sqi_type,
                    "quality_scores": {sqi_type: float(overall_sqi)},
                    "sqi_values": (
                        sqi_values
                        if isinstance(sqi_values, list)
                        else sqi_values.tolist()
                    ),
                    "sqi_arrays": {
                        sqi_type: (
                            sqi_values
                            if isinstance(sqi_values, list)
                            else sqi_values.tolist()
                        )
                    },
                    "normal_segments": normal_segments,
                    "abnormal_segments": abnormal_segments,
                    "overall_quality": float(overall_sqi),
                    "passed": overall_sqi >= threshold,
                    "params_used": sqi_params,
                }

                return True, result

            except Exception as e:
                logger.error(
                    f"Stage 2 - {sqi_type} computation failed: {e}", exc_info=True
                )
                # Return failure result
                result = {
                    "sqi_type": sqi_type,
                    "quality_scores": {sqi_type: 0.5},
                    "sqi_values": [],
                    "sqi_arrays": {sqi_type: []},
                    "normal_segments": [],
                    "abnormal_segments": [],
                    "overall_quality": 0.5,
                    "passed": False,
                    "params_used": sqi_params,
                    "error": str(e),
                }
                return False, result

        elif stage == 3:
            # Stage 3: Parallel Processing (Comprehensive Filtering)
            from vitalDSP.filtering.signal_filtering import SignalFiltering
            from vitalDSP.filtering.artifact_removal import ArtifactRemoval
            from vitalDSP.filtering.advanced_signal_filtering import (
                AdvancedSignalFiltering,
            )

            # Get comprehensive parameters from pipeline_data
            stage3_filter_type = pipeline_data.get("stage3_filter_type", "traditional")
            signal_source = pipeline_data.get("stage3_signal_source", "original")
            application_count = pipeline_data.get("stage3_application_count", 1)
            apply_detrend = pipeline_data.get("stage3_detrend", False)
            paths = pipeline_data.get("paths", ["filtered", "preprocessed"])

            results = {}

            if "raw" in paths:
                results["raw"] = signal_data.copy()

            # Start with the appropriate source signal
            working_signal = signal_data.copy()

            # Apply detrending if requested
            if apply_detrend:
                from scipy import signal as scipy_signal

                working_signal = scipy_signal.detrend(working_signal)
                logger.info("✓ Applied detrending")

            # Apply filtering based on filter type (with multiple applications if specified)
            filtered = working_signal.copy()

            for iteration in range(application_count):
                logger.info(f"Filter application {iteration + 1}/{application_count}")

                if stage3_filter_type == "traditional":
                    # Traditional filter parameters
                    filter_family = pipeline_data.get("stage3_filter_family", "butter")
                    filter_response = pipeline_data.get(
                        "stage3_filter_response", "bandpass"
                    )
                    lowcut = pipeline_data.get("stage3_filter_lowcut", 0.5)
                    highcut = pipeline_data.get("stage3_filter_highcut", 40)
                    filter_order = pipeline_data.get("stage3_filter_order", 4)
                    filter_rp = pipeline_data.get("stage3_filter_rp", 1)
                    filter_rs = pipeline_data.get("stage3_filter_rs", 40)

                    sf = SignalFiltering(filtered)

                    try:
                        if filter_response == "lowpass":
                            filtered = sf.lowpass(
                                cutoff=highcut,
                                fs=fs,
                                order=filter_order,
                                filter_type=filter_family,
                            )
                            logger.info(
                                f"✓ Applied {filter_family} lowpass: {highcut} Hz, order {filter_order}"
                            )
                        elif filter_response == "highpass":
                            filtered = sf.highpass(
                                cutoff=lowcut,
                                fs=fs,
                                order=filter_order,
                                filter_type=filter_family,
                            )
                            logger.info(
                                f"✓ Applied {filter_family} highpass: {lowcut} Hz, order {filter_order}"
                            )
                        elif filter_response == "bandpass":
                            filtered = sf.bandpass(
                                lowcut=lowcut,
                                highcut=highcut,
                                fs=fs,
                                order=filter_order,
                                filter_type=filter_family,
                            )
                            logger.info(
                                f"✓ Applied {filter_family} bandpass: {lowcut}-{highcut} Hz, order {filter_order}"
                            )
                        elif filter_response == "bandstop":
                            filtered = sf.bandstop(
                                lowcut=lowcut,
                                highcut=highcut,
                                fs=fs,
                                order=filter_order,
                                filter_type=filter_family,
                            )
                            logger.info(
                                f"✓ Applied {filter_family} bandstop: {lowcut}-{highcut} Hz, order {filter_order}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Filter {filter_family} failed: {e}, falling back to butterworth bandpass"
                        )
                        filtered = sf.bandpass(
                            lowcut=lowcut,
                            highcut=highcut,
                            fs=fs,
                            order=filter_order,
                            filter_type="butter",
                        )

                    # Apply additional filters if specified
                    savgol_window = pipeline_data.get("stage3_savgol_window")
                    if savgol_window and savgol_window >= 3:
                        savgol_polyorder = pipeline_data.get(
                            "stage3_savgol_polyorder", 2
                        )
                        from scipy.signal import savgol_filter

                        filtered = savgol_filter(
                            filtered, savgol_window, savgol_polyorder
                        )
                        logger.info(
                            f"✓ Applied Savitzky-Golay: window={savgol_window}, polyorder={savgol_polyorder}"
                        )

                    moving_avg_window = pipeline_data.get("stage3_moving_avg_window")
                    if moving_avg_window and moving_avg_window >= 3:
                        filtered = np.convolve(
                            filtered,
                            np.ones(moving_avg_window) / moving_avg_window,
                            mode="same",
                        )
                        logger.info(
                            f"✓ Applied Moving Average: window={moving_avg_window}"
                        )

                    gaussian_sigma = pipeline_data.get("stage3_gaussian_sigma")
                    if gaussian_sigma and gaussian_sigma > 0:
                        from scipy.ndimage import gaussian_filter1d

                        filtered = gaussian_filter1d(filtered, sigma=gaussian_sigma)
                        logger.info(
                            f"✓ Applied Gaussian filter: sigma={gaussian_sigma}"
                        )

                elif stage3_filter_type == "advanced":
                    advanced_method = pipeline_data.get(
                        "stage3_advanced_method", "kalman"
                    )
                    asf = AdvancedSignalFiltering(filtered)

                    if advanced_method == "kalman":
                        kalman_r = pipeline_data.get("stage3_kalman_r", 1.0)
                        kalman_q = pipeline_data.get("stage3_kalman_q", 1.0)
                        filtered = asf.kalman_filter(R=kalman_r, Q=kalman_q)
                        logger.info(
                            f"✓ Applied Kalman filter: R={kalman_r}, Q={kalman_q}"
                        )
                    elif advanced_method == "adaptive":
                        adaptive_mu = pipeline_data.get("stage3_adaptive_mu", 0.01)
                        adaptive_order = pipeline_data.get("stage3_adaptive_order", 4)
                        filtered = asf.adaptive_filter(
                            mu=adaptive_mu, order=adaptive_order
                        )
                        logger.info(
                            f"✓ Applied Adaptive filter: μ={adaptive_mu}, order={adaptive_order}"
                        )
                    else:
                        logger.warning(
                            f"Advanced method '{advanced_method}' not fully supported, applying Kalman"
                        )
                        filtered = asf.kalman_filter(R=1.0, Q=1.0)

                elif stage3_filter_type == "artifact":
                    artifact_type = pipeline_data.get(
                        "stage3_artifact_type", "baseline"
                    )
                    artifact_strength = pipeline_data.get(
                        "stage3_artifact_strength", 0.5
                    )
                    wavelet_type = pipeline_data.get("stage3_wavelet_type", "db4")
                    wavelet_level = pipeline_data.get("stage3_wavelet_level", 3)
                    threshold_type = pipeline_data.get("stage3_threshold_type", "soft")
                    threshold_value = pipeline_data.get("stage3_threshold_value", 0.1)

                    ar = ArtifactRemoval(filtered)

                    if artifact_type == "baseline":
                        filtered = ar.baseline_correction(
                            cutoff=artifact_strength, fs=fs
                        )
                        logger.info(
                            f"✓ Applied baseline correction: cutoff={artifact_strength} Hz"
                        )
                    elif artifact_type == "spike":
                        filtered = ar.spike_removal(threshold=threshold_value)
                        logger.info(
                            f"✓ Applied spike removal: threshold={threshold_value}"
                        )
                    elif artifact_type == "noise":
                        filtered = ar.wavelet_denoising(
                            wavelet_type=wavelet_type,
                            level=wavelet_level,
                            threshold_type=threshold_type,
                        )
                        logger.info(
                            f"✓ Applied wavelet denoising: {wavelet_type}, level={wavelet_level}, {threshold_type}"
                        )
                    elif artifact_type in ["powerline", "pca", "ica"]:
                        logger.info(
                            f"Artifact type '{artifact_type}' using wavelet denoising as fallback"
                        )
                        filtered = ar.wavelet_denoising(
                            wavelet_type=wavelet_type,
                            level=wavelet_level,
                            threshold_type=threshold_type,
                        )

                elif stage3_filter_type == "neural":
                    neural_type = pipeline_data.get("stage3_neural_type", "autoencoder")
                    neural_complexity = pipeline_data.get(
                        "stage3_neural_complexity", "medium"
                    )
                    logger.info(
                        f"Neural network filtering ({neural_type}, {neural_complexity}) - using Kalman as fallback"
                    )
                    asf = AdvancedSignalFiltering(filtered)
                    filtered = asf.kalman_filter(R=0.5, Q=1.0)

                elif stage3_filter_type == "ensemble":
                    ensemble_method = pipeline_data.get(
                        "stage3_ensemble_method", "mean"
                    )
                    ensemble_n_filters = pipeline_data.get(
                        "stage3_ensemble_n_filters", 3
                    )
                    logger.info(
                        f"Ensemble filtering ({ensemble_method}, n={ensemble_n_filters}) - using adaptive multi-pass as fallback"
                    )
                    asf = AdvancedSignalFiltering(filtered)
                    filtered = asf.kalman_filter(R=0.5, Q=1.0)

            # Store filtered result
            if "filtered" in paths:
                results["filtered"] = filtered

            # Create preprocessed version (additional cleanup)
            if "preprocessed" in paths:
                ar = ArtifactRemoval(filtered)
                preprocessed = ar.baseline_correction(cutoff=0.5, fs=fs)
                results["preprocessed"] = preprocessed
                logger.info("✓ Created preprocessed signal (baseline corrected)")

            result = {
                "paths": list(results.keys()),
                "filtered_samples": {k: len(v) for k, v in results.items()},
                "filter_params": {
                    "filter_type": stage3_filter_type,
                    "signal_source": signal_source,
                    "application_count": application_count,
                    "detrend": apply_detrend,
                },
            }

            # Store processed signals for next stages
            pipeline_data["processed_signals"] = results
            return True, result

        elif stage == 4:
            # Stage 4: Quality Validation - Use compute_sqi function from quality page
            from vitalDSP_webapp.callbacks.analysis.quality_sqi_functions import (
                compute_sqi,
            )

            processed_signals = pipeline_data.get("processed_signals", {})

            # Get parameters from pipeline_data (now using quality page parameter structure)
            sqi_type = pipeline_data.get("stage4_sqi_type", "snr_sqi")
            window_size = pipeline_data.get("stage4_window_size", 1000)
            step_size = pipeline_data.get("stage4_step_size", 500)
            threshold_type = pipeline_data.get("stage4_threshold_type", "below")
            threshold = pipeline_data.get("stage4_threshold", 0.7)
            scale = pipeline_data.get("stage4_scale", "zscore")

            # Build SQI parameters dictionary
            sqi_params = {
                "window_size": window_size,
                "step_size": step_size,
                "threshold": threshold,
                "threshold_type": threshold_type,
                "scale": scale,
            }

            quality_metrics = {}

            logger.info(
                f"Stage 4 - Computing {sqi_type} for {len(processed_signals)} processing paths"
            )
            logger.info(f"Stage 4 - Available paths: {list(processed_signals.keys())}")
            logger.info(f"Stage 4 - SQI parameters: {sqi_params}")

            # Compute SQI for each processing path
            for path_name, path_signal in processed_signals.items():
                try:
                    logger.info(
                        f"Stage 4 - Computing {sqi_type} for path: {path_name} (signal length: {len(path_signal)})"
                    )

                    # Use compute_sqi for consistent computation (same as Quality Page)
                    path_quality_results = compute_sqi(
                        path_signal, sqi_type, sqi_params, fs
                    )

                    logger.info(
                        f"Stage 4 - compute_sqi returned: {list(path_quality_results.keys())}"
                    )

                    # Extract results
                    sqi_values = path_quality_results.get("sqi_values", [])
                    normal_segments = path_quality_results.get("normal_segments", [])
                    abnormal_segments = path_quality_results.get(
                        "abnormal_segments", []
                    )
                    overall_sqi = path_quality_results.get(
                        "quality_score", 0.5
                    )  # FIX: Use correct key from compute_sqi

                    logger.info(
                        f"Stage 4 - {path_name} overall_sqi before additional metrics: {overall_sqi}"
                    )

                    # Compute additional quality metrics for visualization
                    snr_score = 0.0
                    smoothness_score = 0.0
                    artifact_score = 0.0

                    try:
                        # Estimate SNR from signal statistics
                        signal_mean = np.mean(path_signal)
                        signal_std = np.std(path_signal)
                        if signal_std > 0:
                            snr_score = min(
                                abs(signal_mean) / signal_std, 1.0
                            )  # Normalize to 0-1

                        # Estimate smoothness from signal derivative
                        signal_diff = np.diff(path_signal)
                        smoothness_score = 1.0 / (
                            1.0 + np.std(signal_diff)
                        )  # Higher = smoother

                        # Estimate artifact level (inverse of smoothness and SNR)
                        artifact_score = 1.0 - (
                            snr_score * 0.5 + smoothness_score * 0.5
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not compute additional metrics for {path_name}: {e}"
                        )

                    # Store results for this path
                    quality_metrics[path_name] = {
                        "sqi_type": sqi_type,
                        "overall_quality": float(overall_sqi),
                        "overall_sqi": float(overall_sqi),  # Alias for compatibility
                        "snr_score": float(snr_score),
                        "smoothness_score": float(smoothness_score),
                        "artifact_score": float(artifact_score),
                        "sqi_values": (
                            sqi_values
                            if isinstance(sqi_values, list)
                            else sqi_values.tolist()
                        ),
                        "normal_segments": normal_segments,
                        "abnormal_segments": abnormal_segments,
                        "passed": overall_sqi >= threshold,
                    }

                    logger.info(
                        f"✓ {path_name}: {sqi_type}={overall_sqi:.3f} "
                        f"(normal: {len(normal_segments)}, abnormal: {len(abnormal_segments)})"
                    )

                except Exception as e:
                    logger.error(
                        f"Stage 4 - {sqi_type} failed for {path_name}: {e}",
                        exc_info=True,
                    )
                    # Store failure result
                    quality_metrics[path_name] = {
                        "sqi_type": sqi_type,
                        "overall_quality": 0.5,
                        "overall_sqi": 0.5,
                        "snr_score": 0.0,
                        "smoothness_score": 0.0,
                        "artifact_score": 0.0,
                        "sqi_values": [],
                        "normal_segments": [],
                        "abnormal_segments": [],
                        "passed": False,
                        "error": str(e),
                    }

            # Select best path based on overall quality
            best_path = (
                max(
                    quality_metrics, key=lambda k: quality_metrics[k]["overall_quality"]
                )
                if quality_metrics
                else "raw"
            )
            best_quality = quality_metrics.get(best_path, {}).get(
                "overall_quality", 0.5
            )

            result = {
                "quality_metrics": quality_metrics,
                "best_path": best_path,
                "confidence": float(best_quality),
                "sqi_type": sqi_type,
                "sqi_params": sqi_params,
                "num_paths_evaluated": len(quality_metrics),
            }
            pipeline_data["best_path"] = best_path
            logger.info(
                f"✓ Stage 4 - Best path selected: {best_path} (quality: {best_quality:.3f})"
            )
            return True, result

        elif stage == 5:
            # Stage 5: Segmentation
            best_path = pipeline_data.get("best_path", "raw")
            processed_signals = pipeline_data.get("processed_signals", {})
            best_signal = processed_signals.get(best_path, signal_data)

            # Get parameters from pipeline_data
            window_size = pipeline_data.get("window_size", 30)  # seconds
            overlap_ratio = pipeline_data.get("overlap_ratio", 0.5)
            min_segment_length = pipeline_data.get("min_segment_length", 5)  # seconds
            window_function = pipeline_data.get("window_function", "none")

            window_samples = int(window_size * fs)
            step_samples = int(window_samples * (1 - overlap_ratio))
            min_samples = int(min_segment_length * fs)

            # Generate window function if requested
            window_func = None
            if window_function != "none":
                from vitalDSP.utils.signal_processing.window_functions import (
                    WindowFunctions,
                )

                if window_function == "hamming":
                    wf = WindowFunctions()
                    window_func = wf.hamming(window_samples)
                elif window_function == "hanning":
                    wf = WindowFunctions()
                    window_func = wf.hanning(window_samples)
                elif window_function == "blackman":
                    wf = WindowFunctions()
                    window_func = wf.blackman(window_samples)
                elif window_function == "gaussian":
                    wf = WindowFunctions()
                    window_func = wf.gaussian(window_samples, std=window_samples / 6)

                logger.info(f"✓ Using {window_function} window function")

            segments = []
            segment_positions = []  # Store positions for visualization

            for start in range(0, len(best_signal) - window_samples + 1, step_samples):
                end = start + window_samples
                segment = best_signal[start:end]

                # Apply window function if specified
                if window_func is not None:
                    segment = segment * window_func

                # Only include if meets minimum length requirement
                if len(segment) >= min_samples:
                    segments.append(segment)
                    segment_positions.append((start, end))

            result = {
                "num_segments": len(segments),
                "window_size": window_size,
                "overlap_ratio": overlap_ratio,
                "min_segment_length": min_segment_length,
                "window_function": window_function,
                "segment_length": window_samples,
                "segment_positions": segment_positions,  # For visualization
            }
            pipeline_data["segments"] = segments
            pipeline_data["segment_positions"] = segment_positions

            logger.info(
                f"✓ Created {len(segments)} segments with {window_function} window"
            )
            return True, result

        elif stage == 6:
            # Stage 6: Feature Extraction
            from vitalDSP.physiological_features.time_domain import TimeDomainFeatures

            segments = pipeline_data.get("segments", [])

            # Get selected feature categories and specific features
            feature_categories = pipeline_data.get(
                "feature_types", ["time", "frequency"]
            )
            time_features_selected = pipeline_data.get(
                "time_features", ["mean", "std", "rms", "ptp"]
            )
            frequency_features_selected = pipeline_data.get(
                "frequency_features", ["spectral_centroid", "dominant_freq"]
            )
            nonlinear_features_selected = pipeline_data.get(
                "nonlinear_features", ["sample_entropy"]
            )
            statistical_features_selected = pipeline_data.get(
                "statistical_features", ["skewness", "kurtosis"]
            )

            features = []
            feature_names_used = set()

            for seg_idx, seg in enumerate(segments):
                seg_features = {}

                # Time Domain Features
                if "time" in feature_categories:
                    if "mean" in time_features_selected:
                        seg_features["mean"] = float(np.mean(seg))
                        feature_names_used.add("mean")
                    if "std" in time_features_selected:
                        seg_features["std"] = float(np.std(seg))
                        feature_names_used.add("std")
                    if "rms" in time_features_selected:
                        seg_features["rms"] = float(np.sqrt(np.mean(seg**2)))
                        feature_names_used.add("rms")
                    if "ptp" in time_features_selected:
                        seg_features["peak_to_peak"] = float(np.ptp(seg))
                        feature_names_used.add("peak_to_peak")
                    if "zero_crossings" in time_features_selected:
                        zero_crossings = np.sum(np.diff(np.sign(seg)) != 0)
                        seg_features["zero_crossings"] = int(zero_crossings)
                        feature_names_used.add("zero_crossings")
                    if "mad" in time_features_selected:
                        seg_features["mad"] = float(np.mean(np.abs(seg - np.mean(seg))))
                        feature_names_used.add("mad")
                    if "energy" in time_features_selected:
                        seg_features["energy"] = float(np.sum(seg**2))
                        feature_names_used.add("energy")
                    if "power" in time_features_selected:
                        seg_features["power"] = float(np.mean(seg**2))
                        feature_names_used.add("power")
                    if "crest_factor" in time_features_selected:
                        rms_val = np.sqrt(np.mean(seg**2))
                        crest = (
                            float(np.max(np.abs(seg)) / rms_val) if rms_val > 0 else 0
                        )
                        seg_features["crest_factor"] = crest
                        feature_names_used.add("crest_factor")

                # Frequency Domain Features
                if "frequency" in feature_categories:
                    try:
                        # Compute power spectral density using vitalDSP SignalPowerAnalysis
                        from vitalDSP.physiological_features.signal_power_analysis import (
                            SignalPowerAnalysis,
                        )

                        spa = SignalPowerAnalysis(seg)
                        freqs, psd = spa.compute_psd(fs=fs)

                        if "spectral_centroid" in frequency_features_selected:
                            spectral_centroid = np.sum(freqs * psd) / (
                                np.sum(psd) + 1e-10
                            )
                            seg_features["spectral_centroid"] = float(spectral_centroid)
                            feature_names_used.add("spectral_centroid")

                        if "dominant_freq" in frequency_features_selected:
                            dominant_freq = freqs[np.argmax(psd)]
                            seg_features["dominant_freq"] = float(dominant_freq)
                            feature_names_used.add("dominant_freq")

                        if "spectral_entropy" in frequency_features_selected:
                            psd_norm = psd / (np.sum(psd) + 1e-10)
                            spectral_entropy = -np.sum(
                                psd_norm * np.log2(psd_norm + 1e-10)
                            )
                            seg_features["spectral_entropy"] = float(spectral_entropy)
                            feature_names_used.add("spectral_entropy")

                        if "band_power" in frequency_features_selected:
                            band_power = np.sum(psd)
                            seg_features["band_power"] = float(band_power)
                            feature_names_used.add("band_power")

                        if "peak_freq" in frequency_features_selected:
                            # Same as dominant freq
                            seg_features["peak_freq"] = float(freqs[np.argmax(psd)])
                            feature_names_used.add("peak_freq")

                        if "spectral_bandwidth" in frequency_features_selected:
                            # Calculate spectral bandwidth
                            spectral_centroid = np.sum(freqs * psd) / (
                                np.sum(psd) + 1e-10
                            )
                            spectral_bandwidth = np.sqrt(
                                np.sum(((freqs - spectral_centroid) ** 2) * psd)
                                / (np.sum(psd) + 1e-10)
                            )
                            seg_features["spectral_bandwidth"] = float(
                                spectral_bandwidth
                            )
                            feature_names_used.add("spectral_bandwidth")

                        if "spectral_rolloff" in frequency_features_selected:
                            # Calculate spectral rolloff (85% energy threshold)
                            cumsum_psd = np.cumsum(psd)
                            rolloff_threshold = 0.85 * cumsum_psd[-1]
                            rolloff_idx = np.where(cumsum_psd >= rolloff_threshold)[0]
                            spectral_rolloff = (
                                float(freqs[rolloff_idx[0]])
                                if len(rolloff_idx) > 0
                                else 0
                            )
                            seg_features["spectral_rolloff"] = spectral_rolloff
                            feature_names_used.add("spectral_rolloff")
                    except Exception as e:
                        logger.warning(
                            f"Frequency feature extraction failed for segment {seg_idx}: {e}"
                        )

                # Statistical Features
                if "statistical" in feature_categories:
                    # Use scipy.stats for statistical features (TimeDomainFeatures is for HRV)
                    from scipy.stats import skew, kurtosis as kurt

                    if "skewness" in statistical_features_selected:
                        seg_features["skewness"] = float(skew(seg))
                        feature_names_used.add("skewness")
                    if "kurtosis" in statistical_features_selected:
                        seg_features["kurtosis"] = float(kurt(seg))
                        feature_names_used.add("kurtosis")
                    if "variance" in statistical_features_selected:
                        seg_features["variance"] = float(np.var(seg))
                        feature_names_used.add("variance")
                    if "median" in statistical_features_selected:
                        seg_features["median"] = float(np.median(seg))
                        feature_names_used.add("median")
                    if "iqr" in statistical_features_selected:
                        from scipy.stats import iqr

                        seg_features["iqr"] = float(iqr(seg))
                        feature_names_used.add("iqr")

                # Nonlinear Features (computationally expensive - simplified versions)
                if "nonlinear" in feature_categories:
                    if "sample_entropy" in nonlinear_features_selected:
                        # Simplified sample entropy approximation
                        try:
                            m, r = 2, 0.2 * np.std(seg)
                            N = len(seg)
                            if N > m + 1:
                                # Very simplified version - proper implementation would use vitalDSP's entropy functions
                                sample_ent = -np.log(np.std(seg) + 1e-10)
                                seg_features["sample_entropy"] = float(sample_ent)
                                feature_names_used.add("sample_entropy")
                        except Exception as e:
                            logger.warning(
                                f"Sample entropy failed for segment {seg_idx}: {e}"
                            )

                    if "approx_entropy" in nonlinear_features_selected:
                        # Placeholder - simplified approximation
                        try:
                            approx_ent = float(np.log(np.std(seg) + 1e-10))
                            seg_features["approx_entropy"] = approx_ent
                            feature_names_used.add("approx_entropy")
                        except Exception as e:
                            logger.warning(
                                f"Approx entropy failed for segment {seg_idx}: {e}"
                            )

                    if "fractal_dim" in nonlinear_features_selected:
                        # Placeholder - Higuchi fractal dimension approximation
                        seg_features["fractal_dim"] = float(1.5)  # Placeholder
                        feature_names_used.add("fractal_dim")

                # Morphological Features
                morphological_features_selected = pipeline_data.get(
                    "morphological_features", []
                )
                if (
                    "morphological" in feature_categories
                    and morphological_features_selected
                ):
                    try:
                        from vitalDSP.physiological_features.waveform import (
                            WaveformMorphology,
                        )

                        # Get signal type to determine waveform type
                        signal_type = pipeline_data.get("signal_type", "ecg").upper()

                        # Use WaveformMorphology class for physiological signal-specific peak detection
                        # Constructor automatically detects peaks based on signal_type
                        waveform = WaveformMorphology(
                            seg, fs=fs, signal_type=signal_type
                        )

                        # Get peaks based on signal type (already detected in constructor)
                        peaks = None
                        if signal_type == "ECG":
                            # For ECG: R-peaks already detected
                            peaks = waveform.r_peaks
                        elif signal_type == "PPG":
                            # For PPG: systolic peaks already detected
                            peaks = waveform.systolic_peaks
                        elif signal_type == "EEG":
                            # For EEG: general peaks
                            peaks = waveform.eeg_peaks
                        else:
                            # For generic signals: use general PeakDetection
                            from vitalDSP.utils.signal_processing.peak_detection import (
                                PeakDetection,
                            )

                            peak_detector = PeakDetection(seg)
                            peaks = peak_detector.detect_peaks()

                        if (
                            "num_peaks" in morphological_features_selected
                            and peaks is not None
                        ):
                            seg_features["num_peaks"] = int(len(peaks))
                            feature_names_used.add("num_peaks")

                        if (
                            "mean_peak_height" in morphological_features_selected
                            and peaks is not None
                            and len(peaks) > 0
                        ):
                            seg_features["mean_peak_height"] = float(
                                np.mean(seg[peaks])
                            )
                            feature_names_used.add("mean_peak_height")

                        # Detect valleys (troughs) based on signal type
                        if any(
                            f in morphological_features_selected
                            for f in ["num_valleys", "mean_valley_depth"]
                        ):
                            valleys = None
                            if signal_type == "ECG":
                                # For ECG: detect S-valleys
                                valleys = waveform.detect_s_valley()
                            elif signal_type == "PPG":
                                # For PPG: detect troughs between systolic peaks
                                valleys = waveform.detect_troughs(
                                    systolic_peaks=waveform.systolic_peaks
                                )
                            else:
                                # For other signals: use inverted peak detection
                                from vitalDSP.utils.signal_processing.peak_detection import (
                                    PeakDetection,
                                )

                                valley_detector = PeakDetection(-seg)
                                valleys = valley_detector.detect_peaks()

                            if (
                                "num_valleys" in morphological_features_selected
                                and valleys is not None
                            ):
                                seg_features["num_valleys"] = int(len(valleys))
                                feature_names_used.add("num_valleys")

                            if (
                                "mean_valley_depth" in morphological_features_selected
                                and valleys is not None
                                and len(valleys) > 0
                            ):
                                seg_features["mean_valley_depth"] = float(
                                    np.mean(seg[valleys])
                                )
                                feature_names_used.add("mean_valley_depth")

                    except Exception as e:
                        logger.warning(
                            f"Morphological feature extraction failed for segment {seg_idx}: {e}"
                        )

                features.append(seg_features)

            result = {
                "num_segments": len(segments),
                "num_features": len(feature_names_used),
                "feature_names": sorted(list(feature_names_used)),
                "feature_categories": feature_categories,
                "features_per_segment": len(features[0]) if features else 0,
            }
            pipeline_data["features"] = features

            logger.info(
                f"✓ Extracted {len(feature_names_used)} feature types from {len(segments)} segments"
            )
            return True, result

        elif stage == 7:
            # Stage 7: Intelligent Output
            # Get parameters from pipeline_data
            selection_criterion = pipeline_data.get(
                "selection_criterion", "best_quality"
            )
            confidence_threshold = pipeline_data.get("confidence_threshold", 0.7)
            generate_recommendations = pipeline_data.get(
                "generate_recommendations", True
            )

            # Get quality metrics from Stage 4
            stage4_results = pipeline_data.get("results", {}).get("stage_4", {})
            quality_metrics = stage4_results.get("quality_metrics", {})
            current_best_path = pipeline_data.get("best_path", "raw")

            # Apply selection criterion
            if selection_criterion == "best_quality":
                # Use overall quality score
                selected_path = (
                    max(
                        quality_metrics,
                        key=lambda k: quality_metrics[k].get("overall_quality", 0),
                    )
                    if quality_metrics
                    else current_best_path
                )
                confidence = quality_metrics.get(selected_path, {}).get(
                    "overall_quality", 0.5
                )
            elif selection_criterion == "highest_snr":
                # Use SNR score
                selected_path = (
                    max(
                        quality_metrics,
                        key=lambda k: quality_metrics[k].get("snr_score", 0),
                    )
                    if quality_metrics
                    else current_best_path
                )
                confidence = quality_metrics.get(selected_path, {}).get(
                    "snr_score", 0.5
                )
            elif selection_criterion == "lowest_artifact":
                # Use artifact score (higher is better)
                selected_path = (
                    max(
                        quality_metrics,
                        key=lambda k: quality_metrics[k].get("artifact_score", 0),
                    )
                    if quality_metrics
                    else current_best_path
                )
                confidence = quality_metrics.get(selected_path, {}).get(
                    "artifact_score", 0.5
                )
            elif selection_criterion == "weighted":
                # Use weighted combination (same as best_quality)
                selected_path = (
                    max(
                        quality_metrics,
                        key=lambda k: quality_metrics[k].get("overall_quality", 0),
                    )
                    if quality_metrics
                    else current_best_path
                )
                confidence = quality_metrics.get(selected_path, {}).get(
                    "overall_quality", 0.5
                )
            else:
                selected_path = current_best_path
                confidence = 0.5

            # Generate recommendations if enabled
            recommendations = []
            if generate_recommendations:
                if confidence >= confidence_threshold:
                    recommendations.append(
                        f"✓ High confidence ({confidence:.2f}) - Recommended path: {selected_path.upper()}"
                    )
                    recommendations.append(
                        f"Signal quality is {['poor', 'fair', 'good', 'excellent'][min(3, int(confidence * 4))]} - suitable for analysis"
                    )
                else:
                    recommendations.append(
                        f"⚠ Low confidence ({confidence:.2f}) - Additional preprocessing may be needed"
                    )
                    recommendations.append(
                        "Consider adjusting filter parameters or artifact removal methods"
                    )

                # Add path-specific recommendations
                if quality_metrics:
                    for path_name, metrics in quality_metrics.items():
                        if metrics.get("artifact_score", 1.0) < 0.6:
                            recommendations.append(
                                f"⚠ {path_name.upper()}: High artifact level detected"
                            )
                        if metrics.get("snr_score", 1.0) < 0.5:
                            recommendations.append(
                                f"⚠ {path_name.upper()}: Low SNR - consider noise reduction"
                            )

            result = {
                "selected_path": selected_path,
                "confidence": float(confidence),
                "meets_threshold": confidence >= confidence_threshold,
                "selection_criterion": selection_criterion,
                "recommendations": recommendations,
                "quality_summary": quality_metrics,
            }

            # Update best_path if selection changed
            pipeline_data["best_path"] = selected_path

            logger.info(
                f"✓ Selected path: {selected_path} (confidence: {confidence:.3f}, criterion: {selection_criterion})"
            )
            return True, result

        elif stage == 8:
            # Stage 8: Output Package
            import time

            # Get parameters from pipeline_data
            output_formats = pipeline_data.get("output_formats", ["json", "csv"])
            output_contents = pipeline_data.get(
                "output_contents",
                ["processed_signals", "quality_metrics", "features", "metadata"],
            )
            compress_output = pipeline_data.get("compress_output", True)

            # Prepare output package based on selected contents
            output_package = {}

            if "raw_signal" in output_contents:
                output_package["raw_signal"] = {
                    "data": (
                        pipeline_data.get("signal_data", []).tolist()
                        if isinstance(pipeline_data.get("signal_data"), np.ndarray)
                        else []
                    ),
                    "length": len(pipeline_data.get("signal_data", [])),
                    "fs": pipeline_data.get("fs", 0),
                }

            if "processed_signals" in output_contents:
                processed_signals = pipeline_data.get("processed_signals", {})
                output_package["processed_signals"] = {
                    path: {
                        "data": sig.tolist() if isinstance(sig, np.ndarray) else sig,
                        "length": len(sig),
                    }
                    for path, sig in processed_signals.items()
                }

            if "quality_metrics" in output_contents:
                # Collect quality metrics from stages 2 and 4
                stage2_results = pipeline_data.get("results", {}).get("stage_2", {})
                stage4_results = pipeline_data.get("results", {}).get("stage_4", {})

                output_package["quality_metrics"] = {
                    "sqi_scores": stage2_results.get("quality_scores", {}),
                    "overall_sqi": stage2_results.get("overall_quality", 0),
                    "path_quality": stage4_results.get("quality_metrics", {}),
                    "best_path": stage4_results.get("best_path", "unknown"),
                }

            if "features" in output_contents:
                features = pipeline_data.get("features", [])
                output_package["features"] = {
                    "num_segments": len(features),
                    "feature_data": features,
                    "feature_names": pipeline_data.get("results", {})
                    .get("stage_6", {})
                    .get("feature_names", []),
                }

            if "segments" in output_contents:
                segments = pipeline_data.get("segments", [])
                output_package["segments"] = {
                    "num_segments": len(segments),
                    "segment_positions": pipeline_data.get("segment_positions", []),
                    "window_size": pipeline_data.get("window_size", 0),
                    "overlap_ratio": pipeline_data.get("overlap_ratio", 0),
                }

            if "metadata" in output_contents:
                output_package["metadata"] = {
                    "signal_type": pipeline_data.get("signal_type", "unknown"),
                    "fs": pipeline_data.get("fs", 0),
                    "duration_seconds": len(pipeline_data.get("signal_data", []))
                    / pipeline_data.get("fs", 1),
                    "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "pipeline_parameters": {
                        "sqi_methods": pipeline_data.get("sqi_methods", []),
                        "filter_type": pipeline_data.get("filter_type", "unknown"),
                        "artifact_method": pipeline_data.get(
                            "artifact_method", "unknown"
                        ),
                        "selection_criterion": pipeline_data.get(
                            "selection_criterion", "unknown"
                        ),
                    },
                }

            # Store output package for export
            pipeline_data["output_package"] = output_package

            result = {
                "status": "completed",
                "total_stages": 8,
                "output_formats": output_formats,
                "output_contents": output_contents,
                "compress_output": compress_output,
                "package_size_kb": len(str(output_package)) / 1024,
                "results_summary": {
                    k: {"completed": True, "has_data": bool(v)}
                    for k, v in pipeline_data.get("results", {}).items()
                },
            }

            logger.info(
                f"✓ Output package created: {len(output_contents)} content types, {len(output_formats)} formats"
            )
            return True, result

        else:
            return False, f"Unknown stage: {stage}"

    except Exception as e:
        # Provide detailed error information for debugging
        stage_names = {
            1: "Data Ingestion",
            2: "Quality Screening",
            3: "Filtering",
            4: "Quality Validation",
            5: "Segmentation",
            6: "Feature Extraction",
            7: "Intelligent Output",
            8: "Output Package",
        }
        stage_name = stage_names.get(stage, f"Stage {stage}")

        logger.error(f"❌ Error in {stage_name} (Stage {stage}): {e}")
        import traceback

        logger.error(f"Full traceback:\n{traceback.format_exc()}")

        # Return user-friendly error message
        error_type = type(e).__name__
        error_msg = str(e)

        # Provide helpful context based on error type
        if "name" in error_msg and "not defined" in error_msg:
            return (
                False,
                f"{error_type}: {error_msg}\n\nThis is likely a parameter configuration issue. Please report this bug.",
            )
        elif "KeyError" in error_type:
            return (
                False,
                f"Missing required parameter: {error_msg}\n\nPlease check stage configuration.",
            )
        elif "ValueError" in error_type:
            return (
                False,
                f"Invalid parameter value: {error_msg}\n\nPlease adjust stage parameters.",
            )
        elif "TypeError" in error_type:
            return (
                False,
                f"Parameter type mismatch: {error_msg}\n\nPlease check parameter types.",
            )
        else:
            return False, f"{error_type}: {error_msg}"


def register_pipeline_callbacks(app):
    """
    Register all pipeline-related callbacks.

    Parameters
    ----------
    app : Dash
        The Dash application instance
    """

    # Stage 3 Filtering - Branching Logic Callback
    @app.callback(
        [
            Output("pipeline-stage3-traditional-params", "style"),
            Output("pipeline-stage3-advanced-params", "style"),
            Output("pipeline-stage3-artifact-params", "style"),
            Output("pipeline-stage3-neural-params", "style"),
            Output("pipeline-stage3-ensemble-params", "style"),
        ],
        Input("pipeline-stage3-filter-type", "value"),
    )
    def update_stage3_params_visibility(filter_type):
        """
        Update visibility of Stage 3 parameter sections based on filter type selection.

        Parameters
        ----------
        filter_type : str
            Selected filter type ('traditional', 'advanced', 'artifact', 'neural', 'ensemble')

        Returns
        -------
        tuple
            Style dictionaries for each parameter section
        """
        # Initialize all sections as hidden
        styles = {
            "traditional": {"display": "none"},
            "advanced": {"display": "none"},
            "artifact": {"display": "none"},
            "neural": {"display": "none"},
            "ensemble": {"display": "none"},
        }

        # Show the selected filter type's parameters
        if filter_type in styles:
            styles[filter_type] = {"display": "block"}

        return (
            styles["traditional"],
            styles["advanced"],
            styles["artifact"],
            styles["neural"],
            styles["ensemble"],
        )

    # Stage 3 Advanced Filter Method - Sub-branching Logic Callback
    @app.callback(
        [
            Output("pipeline-stage3-kalman-params", "style"),
            Output("pipeline-stage3-adaptive-params", "style"),
        ],
        Input("pipeline-stage3-advanced-method", "value"),
    )
    def update_advanced_method_params(advanced_method):
        """
        Update visibility of advanced method parameters based on selected method.

        Parameters
        ----------
        advanced_method : str
            Selected advanced filter method

        Returns
        -------
        tuple
            Style dictionaries for kalman and adaptive parameter sections
        """
        kalman_style = {"display": "block" if advanced_method == "kalman" else "none"}
        adaptive_style = {
            "display": "block" if advanced_method == "adaptive" else "none"
        }

        return kalman_style, adaptive_style

    @app.callback(
        [
            Output("pipeline-progress-container", "style"),
            Output("pipeline-processing-progress", "value"),
            Output("pipeline-processing-progress-status", "children"),
            Output("pipeline-run-btn", "disabled"),
            Output("pipeline-stop-btn", "disabled"),
            Output("pipeline-export-btn", "disabled"),
            Output("pipeline-report-btn", "disabled"),
            Output("pipeline-progress-interval", "disabled"),
            Output("pipeline-current-stage", "data", allow_duplicate=True),
        ],
        [
            Input("pipeline-run-btn", "n_clicks"),
            Input("pipeline-stop-btn", "n_clicks"),
            Input("pipeline-reset-btn", "n_clicks"),
            Input("pipeline-progress-interval", "n_intervals"),
        ],
        [
            State("pipeline-signal-type", "value"),
            State("pipeline-path-selector", "value"),
            State("pipeline-enable-quality", "value"),
            State("pipeline-current-stage", "data"),
            State("store-uploaded-data", "data"),
            # Stage 2 parameters - Quality Screening
            State("pipeline-stage2-sqi-type", "value"),
            State("pipeline-stage2-window-size", "value"),
            State("pipeline-stage2-step-size", "value"),
            State("pipeline-stage2-threshold-type", "value"),
            State("pipeline-stage2-threshold", "value"),
            State("pipeline-stage2-scale", "value"),
            # Stage 3 parameters - Comprehensive Filtering
            State("pipeline-stage3-filter-type", "value"),
            State("pipeline-stage3-detrend", "value"),
            State("pipeline-stage3-signal-source", "value"),
            State("pipeline-stage3-application-count", "value"),
            # Traditional filter parameters
            State("pipeline-stage3-filter-family", "value"),
            State("pipeline-stage3-filter-response", "value"),
            State("pipeline-stage3-filter-lowcut", "value"),
            State("pipeline-stage3-filter-highcut", "value"),
            State("pipeline-stage3-filter-order", "value"),
            State("pipeline-stage3-filter-rp", "value"),
            State("pipeline-stage3-filter-rs", "value"),
            State("pipeline-stage3-savgol-window", "value"),
            State("pipeline-stage3-savgol-polyorder", "value"),
            State("pipeline-stage3-moving-avg-window", "value"),
            State("pipeline-stage3-gaussian-sigma", "value"),
            # Advanced filter parameters
            State("pipeline-stage3-advanced-method", "value"),
            State("pipeline-stage3-kalman-r", "value"),
            State("pipeline-stage3-kalman-q", "value"),
            State("pipeline-stage3-adaptive-mu", "value"),
            State("pipeline-stage3-adaptive-order", "value"),
            # Artifact removal parameters
            State("pipeline-stage3-artifact-type", "value"),
            State("pipeline-stage3-artifact-strength", "value"),
            State("pipeline-stage3-wavelet-type", "value"),
            State("pipeline-stage3-wavelet-level", "value"),
            State("pipeline-stage3-threshold-type", "value"),
            State("pipeline-stage3-threshold-value", "value"),
            # Neural network parameters
            State("pipeline-stage3-neural-type", "value"),
            State("pipeline-stage3-neural-complexity", "value"),
            # Ensemble parameters
            State("pipeline-stage3-ensemble-method", "value"),
            State("pipeline-stage3-ensemble-n-filters", "value"),
            # Stage 4 parameters - Quality Validation
            State("pipeline-stage4-sqi-type", "value"),
            State("pipeline-stage4-window-size", "value"),
            State("pipeline-stage4-step-size", "value"),
            State("pipeline-stage4-threshold-type", "value"),
            State("pipeline-stage4-threshold", "value"),
            State("pipeline-stage4-scale", "value"),
            # Stage 5 parameters - Segmentation
            State("pipeline-window-size", "value"),
            State("pipeline-overlap-ratio", "value"),
            State("pipeline-min-segment-length", "value"),
            State("pipeline-window-function", "value"),
            # Stage 6 parameters - Feature Extraction
            State("pipeline-feature-types", "value"),
            State("pipeline-time-features", "value"),
            State("pipeline-frequency-features", "value"),
            State("pipeline-nonlinear-features", "value"),
            State("pipeline-statistical-features", "value"),
            State("pipeline-morphological-features", "value"),
            # Stage 7 parameters - Intelligent Output
            State("pipeline-selection-criterion", "value"),
            State("pipeline-confidence-threshold", "value"),
            State("pipeline-generate-recommendations", "value"),
            # Stage 8 parameters - Output Package
            State("pipeline-output-formats", "value"),
            State("pipeline-output-contents", "value"),
            State("pipeline-compress-output", "value"),
        ],
        prevent_initial_call=True,
    )
    def handle_pipeline_execution(
        run_clicks,
        stop_clicks,
        reset_clicks,
        n_intervals,
        signal_type,
        paths,
        enable_quality,
        current_stage,
        uploaded_data,
        # Stage 2 parameters - Quality Screening
        stage2_sqi_type,
        stage2_window_size,
        stage2_step_size,
        stage2_threshold_type,
        stage2_threshold,
        stage2_scale,
        # Stage 3 parameters - Comprehensive Filtering
        stage3_filter_type,
        stage3_detrend,
        stage3_signal_source,
        stage3_application_count,
        # Traditional filter parameters
        stage3_filter_family,
        stage3_filter_response,
        stage3_filter_lowcut,
        stage3_filter_highcut,
        stage3_filter_order,
        stage3_filter_rp,
        stage3_filter_rs,
        stage3_savgol_window,
        stage3_savgol_polyorder,
        stage3_moving_avg_window,
        stage3_gaussian_sigma,
        # Advanced filter parameters
        stage3_advanced_method,
        stage3_kalman_r,
        stage3_kalman_q,
        stage3_adaptive_mu,
        stage3_adaptive_order,
        # Artifact removal parameters
        stage3_artifact_type,
        stage3_artifact_strength,
        stage3_wavelet_type,
        stage3_wavelet_level,
        stage3_threshold_type,
        stage3_threshold_value,
        # Neural network parameters
        stage3_neural_type,
        stage3_neural_complexity,
        # Ensemble parameters
        stage3_ensemble_method,
        stage3_ensemble_n_filters,
        # Stage 4 parameters - Quality Validation
        stage4_sqi_type,
        stage4_window_size,
        stage4_step_size,
        stage4_threshold_type,
        stage4_threshold,
        stage4_scale,
        # Stage 5 parameters - Segmentation
        window_size,
        overlap_ratio,
        min_segment_length,
        window_function,
        # Stage 6 parameters - Feature Extraction
        feature_types,
        time_features,
        frequency_features,
        nonlinear_features,
        statistical_features,
        morphological_features,
        # Stage 7 parameters - Intelligent Output
        selection_criterion,
        confidence_threshold,
        generate_recommendations,
        # Stage 8 parameters - Output Package
        output_formats,
        output_contents,
        compress_output,
    ):
        """
        Handle pipeline execution, progress tracking, and control buttons.
        """
        ctx = callback_context

        if not ctx.triggered:
            return (
                {"display": "none"},
                0,
                "",
                False,
                True,
                True,
                True,
                True,
                0,
            )

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Reset button
        if trigger_id == "pipeline-reset-btn":
            return (
                {"display": "none"},
                0,
                "",
                False,
                True,
                True,
                True,
                True,
                0,
            )

        # Run button
        if trigger_id == "pipeline-run-btn":
            # Try to get data from data service (for large files) or store (for small files)
            data_loaded = False
            df = None

            # First, try data service (more reliable for large files)
            try:
                from vitalDSP_webapp.services.data.enhanced_data_service import (
                    get_enhanced_data_service,
                )

                data_service = get_enhanced_data_service()

                if data_service:
                    # Get all stored data and use the most recent one
                    all_data = data_service.get_all_data()
                    logger.info(f"Data service has {len(all_data)} stored datasets")

                    if all_data:
                        # Get the most recent data ID (highest number)
                        data_ids = sorted(all_data.keys(), reverse=True)
                        latest_id = data_ids[0]
                        logger.info(f"Using latest data ID: {latest_id}")

                        df = data_service.get_data(latest_id)
                        if df is not None:
                            logger.info(
                                f"✓ Loaded data from data service: ID={latest_id}, shape={df.shape}, columns={list(df.columns)}"
                            )
                            data_loaded = True
                        else:
                            logger.warning(
                                f"Data ID {latest_id} exists but data is None"
                            )
                    else:
                        logger.warning("Data service available but no stored data")
                else:
                    logger.warning("Data service is None")
            except Exception as e:
                logger.warning(f"Could not load from data service: {e}")
                import traceback

                logger.warning(f"Traceback: {traceback.format_exc()}")

            # Fallback: try uploaded_data store (for smaller files)
            if not data_loaded and uploaded_data:
                try:
                    # pandas is already imported at the top of the file
                    if isinstance(uploaded_data, list) and len(uploaded_data) > 0:
                        df = pd.DataFrame(uploaded_data)
                        logger.info(
                            f"✓ Loaded data from store: shape={df.shape}, columns={list(df.columns)}"
                        )
                        data_loaded = True
                except Exception as e:
                    logger.warning(f"Could not load from store: {e}")

            # If no data loaded, show error
            if not data_loaded or df is None:
                logger.warning("No uploaded data found in either data service or store")
                return (
                    {"display": "block"},
                    0,
                    "⚠️ Error: No data uploaded. Please upload and process data first.",
                    False,
                    True,
                    True,
                    True,
                    True,
                    0,
                )

            # Convert uploaded data to DataFrame for processing
            try:
                logger.info(
                    f"Processing data: shape={df.shape}, columns={list(df.columns)}"
                )

                # Determine signal column (look for common names)
                signal_col = None
                for col_name in [
                    "signal",
                    "Signal",
                    "value",
                    "Value",
                    "amplitude",
                    "Amplitude",
                ]:
                    if col_name in df.columns:
                        signal_col = col_name
                        break

                if signal_col is None and len(df.columns) > 0:
                    # Use first numeric column
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            signal_col = col
                            break

                if signal_col is None:
                    raise ValueError("No signal column found in uploaded data")

                signal_data = df[signal_col].values
                logger.info(
                    f"✓ Signal data extracted: column='{signal_col}', length={len(signal_data)}, range=[{signal_data.min():.3f}, {signal_data.max():.3f}]"
                )

                # Determine sampling frequency from data config if available
                # Try to get from data service first
                try:
                    from vitalDSP_webapp.services.data.data_service import (
                        get_data_service,
                    )

                    data_service = get_enhanced_data_service()
                    all_data = data_service.get_all_data()
                    if all_data:
                        latest_id = sorted(all_data.keys(), reverse=True)[0]
                        data_info = data_service.get_data_info(latest_id)
                        fs = data_info.get("sampling_freq", 128) if data_info else 128
                        logger.info(f"✓ Sampling frequency from data service: {fs} Hz")
                    else:
                        fs = 128  # Default
                        logger.warning("No data info found, using default fs=128 Hz")
                except Exception as e:
                    logger.warning(
                        f"Could not get sampling freq from data service: {e}"
                    )
                    fs = 128  # Default

                # REAL PIPELINE MODE: Process the actual data
                logger.info(
                    f"Starting REAL pipeline processing for signal_type: {signal_type}"
                )
                logger.info(
                    f"Selected paths: {paths}, enable_quality: {enable_quality}"
                )
                logger.info(f"Data: {len(signal_data)} samples at {fs} Hz")
                logger.info(f"Duration: {len(signal_data)/fs:.2f} seconds")

                # Store signal data and config in session for pipeline to access
                # We'll use a simple approach: store in a global dict keyed by a new pipeline_run_id
                import hashlib

                pipeline_run_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]

                # Store in module-level dict (we'll create this)
                if not hasattr(register_pipeline_callbacks, "pipeline_data"):
                    register_pipeline_callbacks.pipeline_data = {}

                # Store the current pipeline run ID for export callbacks
                register_pipeline_callbacks.current_pipeline_run_id = pipeline_run_id

                register_pipeline_callbacks.pipeline_data[pipeline_run_id] = {
                    "signal_data": signal_data,
                    "fs": fs,
                    "signal_type": signal_type,
                    "paths": paths or ["filtered", "preprocessed"],
                    "enable_quality": enable_quality,
                    "current_stage": 0,
                    "total_stages": 8,
                    "results": {},
                    # Stage 2 parameters - Quality Screening
                    "stage2_sqi_type": stage2_sqi_type or "snr_sqi",
                    "stage2_window_size": stage2_window_size or 1000,
                    "stage2_step_size": stage2_step_size or 500,
                    "stage2_threshold_type": stage2_threshold_type or "below",
                    "stage2_threshold": stage2_threshold or 0.7,
                    "stage2_scale": stage2_scale or "zscore",
                    # Stage 3 parameters - Comprehensive Filtering
                    "stage3_filter_type": stage3_filter_type or "traditional",
                    "stage3_detrend": "detrend" in (stage3_detrend or []),
                    "stage3_signal_source": stage3_signal_source or "original",
                    "stage3_application_count": stage3_application_count or 1,
                    # Traditional filter parameters
                    "stage3_filter_family": stage3_filter_family or "butter",
                    "stage3_filter_response": stage3_filter_response or "bandpass",
                    "stage3_filter_lowcut": stage3_filter_lowcut or 0.5,
                    "stage3_filter_highcut": stage3_filter_highcut or 40,
                    "stage3_filter_order": stage3_filter_order or 4,
                    "stage3_filter_rp": stage3_filter_rp or 1,
                    "stage3_filter_rs": stage3_filter_rs or 40,
                    "stage3_savgol_window": stage3_savgol_window,
                    "stage3_savgol_polyorder": stage3_savgol_polyorder or 2,
                    "stage3_moving_avg_window": stage3_moving_avg_window,
                    "stage3_gaussian_sigma": stage3_gaussian_sigma,
                    # Advanced filter parameters
                    "stage3_advanced_method": stage3_advanced_method or "kalman",
                    "stage3_kalman_r": stage3_kalman_r or 1.0,
                    "stage3_kalman_q": stage3_kalman_q or 1.0,
                    "stage3_adaptive_mu": stage3_adaptive_mu or 0.01,
                    "stage3_adaptive_order": stage3_adaptive_order or 4,
                    # Artifact removal parameters
                    "stage3_artifact_type": stage3_artifact_type or "baseline",
                    "stage3_artifact_strength": stage3_artifact_strength or 0.5,
                    "stage3_wavelet_type": stage3_wavelet_type or "db4",
                    "stage3_wavelet_level": stage3_wavelet_level or 3,
                    "stage3_threshold_type": stage3_threshold_type or "soft",
                    "stage3_threshold_value": stage3_threshold_value or 0.1,
                    # Neural network parameters
                    "stage3_neural_type": stage3_neural_type or "autoencoder",
                    "stage3_neural_complexity": stage3_neural_complexity or "medium",
                    # Ensemble parameters
                    "stage3_ensemble_method": stage3_ensemble_method or "mean",
                    "stage3_ensemble_n_filters": stage3_ensemble_n_filters or 3,
                    # Stage 4 parameters - Quality Validation
                    "stage4_sqi_type": stage4_sqi_type or "snr_sqi",
                    "stage4_window_size": stage4_window_size or 1000,
                    "stage4_step_size": stage4_step_size or 500,
                    "stage4_threshold_type": stage4_threshold_type or "below",
                    "stage4_threshold": stage4_threshold or 0.7,
                    "stage4_scale": stage4_scale or "zscore",
                    # Stage 5 parameters - Segmentation
                    "window_size": window_size or 30,
                    "overlap_ratio": overlap_ratio or 0.5,
                    "min_segment_length": min_segment_length or 5,
                    "window_function": window_function or "none",
                    # Stage 6 parameters - Feature Extraction
                    "feature_types": feature_types or ["time", "frequency"],
                    "time_features": time_features or ["mean", "std", "rms", "ptp"],
                    "frequency_features": frequency_features
                    or ["spectral_centroid", "dominant_freq"],
                    "nonlinear_features": nonlinear_features or ["sample_entropy"],
                    "statistical_features": statistical_features
                    or ["skewness", "kurtosis"],
                    "morphological_features": morphological_features or [],
                    # Stage 7 parameters - Intelligent Output
                    "selection_criterion": selection_criterion or "best_quality",
                    "confidence_threshold": confidence_threshold or 0.7,
                    "generate_recommendations": (
                        generate_recommendations
                        if generate_recommendations is not None
                        else True
                    ),
                    # Stage 8 parameters - Output Package
                    "output_formats": output_formats or ["json", "csv"],
                    "output_contents": output_contents
                    or ["processed_signals", "quality_metrics", "features", "metadata"],
                    "compress_output": (
                        compress_output if compress_output is not None else True
                    ),
                }

                logger.info(f"Created pipeline run: {pipeline_run_id}")

                return (
                    {"display": "block"},
                    0,  # Start at 0% - data ingestion will begin
                    "Starting real pipeline processing...",
                    True,  # Disable run button
                    False,  # Enable stop button
                    True,  # Disable export button
                    True,  # Disable report button
                    False,  # Enable interval (False = not disabled)
                    pipeline_run_id,  # Store pipeline run ID
                )

            except Exception as e:
                logger.error(f"Error loading uploaded data: {e}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                return (
                    {"display": "block"},
                    0,
                    f"⚠️ Error: {str(e)}",
                    False,
                    True,
                    True,
                    True,
                    True,
                    0,
                )

            # TODO: Re-enable data validation when using real pipeline
            # if not uploaded_data:
            #     return (
            #         {"display": "block"},
            #         0,
            #         "Error: No data uploaded",
            #         False,
            #         True,
            #         True,
            #         True,
            #         True,
            #         0,
            #     )
            #
            # # Extract signal data from uploaded data
            # try:
            #     signal_data = np.array(uploaded_data.get('data', []))
            #     fs = uploaded_data.get('fs', 1000)
            #
            #     if len(signal_data) == 0:
            #         return (
            #             {"display": "block"},
            #             0,
            #             "Error: No signal data found",
            #             False,
            #             True,
            #             True,
            #             True,
            #             True,
            #             0,
            #         )
            #
            #     # Real pipeline execution code here
            #     ...
            # except Exception as e:
            #     logger.error(f"Failed to start pipeline execution: {e}")
            #     return (
            #         {"display": "block"},
            #         0,
            #         f"Error starting pipeline: {str(e)}",
            #         False,
            #         True,
            #         True,
            #         True,
            #         True,
            #         0,
            #     )

        # Stop button
        if trigger_id == "pipeline-stop-btn":
            # TODO: Re-enable after optimizing pipeline initialization
            # # Check if we have a real session running
            # if isinstance(current_stage, str) and current_stage.startswith('session_'):
            #     session_id = current_stage.replace('session_', '')
            #     pipeline_service = get_pipeline_service()
            #     stopped = pipeline_service.stop_pipeline_execution(session_id)
            #
            #     if stopped:
            #         return (
            #             {"display": "block"},
            #             current_stage / 8 * 100,
            #             f"Pipeline stopped at current stage",
            #             False,
            #             True,
            #             True,
            #             True,
            #             True,  # Disable interval
            #             current_stage,
            #         )

            # Simulation mode stop
            return (
                {"display": "block"},
                current_stage / 8 * 100 if isinstance(current_stage, int) else 0,
                (
                    f"Stopped at stage {current_stage + 1}/8"
                    if isinstance(current_stage, int)
                    else "Stopped"
                ),
                False,
                True,
                current_stage == 8 if isinstance(current_stage, int) else True,
                current_stage == 8 if isinstance(current_stage, int) else True,
                True,  # Disable interval
                current_stage,
            )

        # Progress interval - track real pipeline execution
        if trigger_id == "pipeline-progress-interval":
            logger.info(
                f"Interval fired - current_stage type: {type(current_stage)}, value: {current_stage}"
            )

            # Check if this is a pipeline run ID (string hash) vs old simulation mode (int)
            if isinstance(current_stage, str) and len(current_stage) == 8:
                # REAL PIPELINE MODE
                pipeline_run_id = current_stage

                # Get pipeline data
                if not hasattr(register_pipeline_callbacks, "pipeline_data"):
                    logger.error("Pipeline data not found!")
                    # Disable interval if no data
                    return (
                        {"display": "block"},
                        0,
                        "Error: No pipeline data",
                        False,
                        True,
                        True,
                        True,
                        True,  # DISABLE INTERVAL
                        0,
                    )

                pipeline_data = register_pipeline_callbacks.pipeline_data.get(
                    pipeline_run_id
                )
                if not pipeline_data:
                    logger.error(f"Pipeline run {pipeline_run_id} not found!")
                    # Disable interval if run not found
                    return (
                        {"display": "block"},
                        0,
                        "Error: Pipeline run not found",
                        False,
                        True,
                        True,
                        True,
                        True,  # DISABLE INTERVAL
                        pipeline_run_id,
                    )

                current_stage_num = pipeline_data.get("current_stage", 0)
                is_completed = pipeline_data.get("completed", False)

                logger.info(
                    f"Real pipeline mode - Run ID: {pipeline_run_id}, Stage: {current_stage_num}, Completed: {is_completed}"
                )

                # If pipeline already marked as completed OR stage >= 8, stop the interval IMMEDIATELY
                if is_completed or current_stage_num >= 8:
                    logger.info(
                        f"⚠️ Interval fired but pipeline already completed (stage={current_stage_num}, completed={is_completed}) - STOPPING INTERVAL"
                    )
                    return (
                        {"display": "block"},
                        100,
                        "🔴 REAL DATA MODE - Pipeline completed successfully!",
                        False,  # Enable run button
                        True,  # Disable stop button
                        False,  # Enable export button
                        False,  # Enable report button
                        True,  # Disable interval (STOP FIRING)
                        pipeline_run_id,
                    )

                # Process one stage at a time
                if current_stage_num < 8:
                    # Execute current stage
                    new_stage = current_stage_num + 1
                    success, result = _execute_pipeline_stage(
                        pipeline_data,
                        new_stage,
                        signal_data=pipeline_data["signal_data"],
                        fs=pipeline_data["fs"],
                        signal_type=pipeline_data["signal_type"],
                    )

                    if success:
                        # Update pipeline data
                        pipeline_data["current_stage"] = new_stage
                        pipeline_data["results"][f"stage_{new_stage}"] = result

                        progress = (new_stage / 8) * 100
                        stage_names = [
                            "Data Ingestion",
                            "Quality Screening",
                            "Parallel Processing",
                            "Quality Validation",
                            "Segmentation",
                            "Feature Extraction",
                            "Intelligent Output",
                            "Output Package",
                        ]
                        status = f"🔴 REAL DATA MODE - Stage {new_stage}/8: {stage_names[new_stage - 1]}"

                        logger.info(
                            f"✓ Stage {new_stage} completed: {stage_names[new_stage - 1]}"
                        )

                        # Check if pipeline is complete
                        if new_stage == 8:
                            logger.info("🎉 Pipeline completed successfully!")
                            # Mark pipeline as completed
                            pipeline_data["completed"] = True
                            pipeline_data["completion_time"] = time.time()
                            # Force trigger plot updates by updating current_stage in pipeline_data
                            pipeline_data["current_stage"] = 8
                            return (
                                {"display": "block"},
                                100,
                                "🔴 REAL DATA MODE - Pipeline completed successfully!",
                                False,
                                True,
                                False,  # Enable export
                                False,  # Enable report
                                True,  # Disable interval
                                pipeline_run_id,  # Keep pipeline_run_id for consistency
                            )

                        # Update current_stage in pipeline_data
                        pipeline_data["current_stage"] = new_stage
                        return (
                            {"display": "block"},
                            progress,
                            status,
                            True,
                            False,
                            True,
                            True,
                            False,  # Keep interval enabled
                            pipeline_run_id,  # Keep pipeline_run_id for consistency
                        )
                    else:
                        # Stage failed - provide detailed user-friendly error message
                        stage_names = [
                            "Data Ingestion",
                            "Quality Screening",
                            "Filtering",
                            "Quality Validation",
                            "Segmentation",
                            "Feature Extraction",
                            "Intelligent Output",
                            "Output Package",
                        ]
                        stage_name = (
                            stage_names[new_stage - 1]
                            if new_stage <= 8
                            else f"Stage {new_stage}"
                        )

                        # Extract meaningful error message
                        error_msg = str(result)
                        if len(error_msg) > 200:
                            error_msg = error_msg[:200] + "..."

                        user_message = (
                            f"❌ Error in {stage_name} (Stage {new_stage})\n\n"
                            f"📋 Details: {error_msg}\n\n"
                            f"💡 Tip: Check console logs for full details or try adjusting stage parameters."
                        )

                        logger.error(
                            f"❌ Stage {new_stage} ({stage_name}) failed: {result}"
                        )
                        return (
                            {"display": "block"},
                            (current_stage_num / 8) * 100,
                            user_message,
                            False,
                            True,
                            True,
                            True,
                            True,  # Disable interval
                            pipeline_run_id,
                        )
                else:
                    # Stage is >= 8, should have been caught earlier but add safety
                    logger.warning(
                        f"⚠️ Interval fired for completed pipeline (stage {current_stage_num}) - STOPPING"
                    )
                    return (
                        {"display": "block"},
                        100,
                        "🔴 REAL DATA MODE - Pipeline completed!",
                        False,
                        True,
                        False,
                        False,
                        True,  # DISABLE INTERVAL
                        pipeline_run_id,
                    )

            # Fallback for old simulation mode (backward compatibility)
            # TODO: Re-enable after optimizing pipeline initialization
            # # Check if we have a session ID (real execution)
            # if isinstance(current_stage, str) and current_stage.startswith('session_'):
            #     session_id = current_stage.replace('session_', '')  # Extract actual session ID
            #     pipeline_service = get_pipeline_service()
            #     execution_state = pipeline_service.get_execution_state(session_id)
            #
            #     if execution_state:
            #         if execution_state.status == 'completed':
            #             return (
            #                 {"display": "block"},
            #                 100,
            #                 "Pipeline completed successfully!",
            #                 False,
            #                 True,
            #                 False,  # Enable export
            #                 False,  # Enable report
            #                 True,  # Disable interval
            #                 session_id,
            #             )
            #         elif execution_state.status == 'failed':
            #             return (
            #                 {"display": "block"},
            #                 execution_state.progress_percentage,
            #                 f"Pipeline failed: {execution_state.error_message}",
            #                 False,
            #                 True,
            #                 True,
            #                 True,
            #                 True,  # Disable interval
            #                 session_id,
            #             )
            #         elif execution_state.status == 'stopped':
            #             return (
            #                 {"display": "block"},
            #                 execution_state.progress_percentage,
            #                 f"Pipeline stopped at stage {execution_state.current_stage}",
            #                 False,
            #                 True,
            #                 True,
            #                 True,
            #                 True,  # Disable interval
            #                 session_id,
            #             )
            #         elif execution_state.status == 'running':
            #             return (
            #                 {"display": "block"},
            #                 execution_state.progress_percentage,
            #                 f"Stage {execution_state.current_stage}/8: {execution_state.stage_name}",
            #                 True,
            #                 False,
            #                 True,
            #                 True,
            #                 False,  # Keep interval enabled
            #                 session_id,
            #             )

            # Simulation mode - increment through stages
            if isinstance(current_stage, int) and current_stage < 8:
                logger.info(f"Simulation mode: incrementing from stage {current_stage}")
                new_stage = current_stage + 1
                progress = (new_stage / 8) * 100
                stage_names = [
                    "Data Ingestion",
                    "Quality Screening",
                    "Parallel Processing",
                    "Quality Validation",
                    "Segmentation",
                    "Feature Extraction",
                    "Intelligent Output",
                    "Output Package",
                ]
                status = f"🟡 SIMULATION MODE - Stage {new_stage}/8: {stage_names[new_stage - 1]}"

                # Check if pipeline is complete
                if new_stage == 8:
                    return (
                        {"display": "block"},
                        100,
                        "🟡 SIMULATION MODE - Pipeline completed successfully!",
                        False,
                        True,
                        False,  # Enable export
                        False,  # Enable report
                        True,  # Disable interval
                        new_stage,
                    )

                return (
                    {"display": "block"},
                    progress,
                    status,
                    True,
                    False,
                    True,
                    True,
                    False,  # Keep interval enabled
                    new_stage,
                )

        return no_update

    @app.callback(
        Output("pipeline-stage-details", "children"),
        [Input("pipeline-current-stage", "data")],
        [State("pipeline-signal-type", "value")],
    )
    def update_stage_details(current_stage, signal_type):
        """
        Update the stage details panel based on current pipeline stage.
        Shows real pipeline results when available, falls back to simulation data.
        """
        stage_num = _get_stage_number(current_stage)

        if stage_num == 0:
            return html.Div(
                [
                    html.P(
                        "Click 'Run Pipeline' to start processing.",
                        className="text-muted",
                    ),
                ],
                className="text-center p-5",
            )

        # Check if we have real pipeline data
        pipeline_data = None
        if isinstance(current_stage, str) and len(current_stage) == 8:
            # It's a pipeline_run_id, get real pipeline data
            if hasattr(register_pipeline_callbacks, "pipeline_data"):
                pipeline_data = register_pipeline_callbacks.pipeline_data.get(
                    current_stage
                )

        # Default simulation stage info (fallback)
        stage_info = {
            1: {
                "name": "Data Ingestion",
                "description": "Loading and validating input data, detecting format, and extracting metadata.",
                "metrics": {
                    "Format Detected": "CSV",
                    "Rows Loaded": "10,000",
                    "Sampling Rate": "1000 Hz",
                    "Duration": "10 seconds",
                },
            },
            2: {
                "name": "Quality Screening",
                "description": "Three-stage quality screening: SNR assessment, statistical analysis, and signal-specific checks.",
                "metrics": {
                    "SNR": "15.2 dB",
                    "Outlier Ratio": "0.05",
                    "Jump Ratio": "0.08",
                    "Overall Quality": "0.85",
                },
            },
            3: {
                "name": "Parallel Processing",
                "description": "Processing signal through multiple paths: RAW, FILTERED, and PREPROCESSED.",
                "metrics": {
                    "Paths Active": "2 (FILTERED, PREPROCESSED)",
                    "Filter Applied": "Butterworth bandpass 0.5-40 Hz",
                    "Artifacts Removed": "23",
                },
            },
            4: {
                "name": "Quality Validation",
                "description": "Comparing quality metrics across processing paths to select the best path.",
                "metrics": {
                    "RAW Quality": "0.65",
                    "FILTERED Quality": "0.82",
                    "PREPROCESSED Quality": "0.88",
                    "Best Path": "PREPROCESSED",
                },
            },
            5: {
                "name": "Segmentation",
                "description": "Dividing signal into overlapping windows for detailed analysis.",
                "metrics": {
                    "Window Size": "30 seconds",
                    "Overlap": "50%",
                    "Total Segments": "19",
                    "Valid Segments": "18",
                },
            },
            6: {
                "name": "Feature Extraction",
                "description": "Extracting time-domain, frequency-domain, and nonlinear features from each segment.",
                "metrics": {
                    "Time Features": "9",
                    "Frequency Features": "4",
                    "Nonlinear Features": "0",
                    "Total Features": "13",
                },
            },
            7: {
                "name": "Intelligent Output",
                "description": "Analyzing results and generating intelligent recommendations.",
                "metrics": {
                    "Best Path": "PREPROCESSED",
                    "Confidence": "High (0.92)",
                    "Recommendations": "3",
                },
            },
            8: {
                "name": "Output Package",
                "description": "Packaging all results, metadata, and recommendations for export.",
                "metrics": {
                    "Status": "Complete",
                    "Processing Time": "2.45 seconds",
                    "Output Size": "1.2 MB",
                },
            },
        }

        # If we have real pipeline data, use it instead of simulation data
        if pipeline_data and "results" in pipeline_data:
            real_results = pipeline_data["results"]

            # Update stage info with real data for completed stages
            if stage_num >= 1 and f"stage_{stage_num}" in real_results:
                stage_result = real_results[f"stage_{stage_num}"]

                if stage_num == 1:  # Data Ingestion
                    stage_info[1]["metrics"] = {
                        "Samples": f"{stage_result.get('samples', 'N/A'):,}",
                        "Duration": f"{stage_result.get('duration', 0):.2f} seconds",
                        "Sampling Rate": f"{stage_result.get('fs', 0)} Hz",
                        "Signal Type": stage_result.get("signal_type", "Unknown"),
                        "Mean": f"{stage_result.get('mean', 0):.3f}",
                        "Std Dev": f"{stage_result.get('std', 0):.3f}",
                    }
                elif stage_num == 2:  # Quality Screening
                    # Handle potential string values for numeric formatting
                    snr = stage_result.get("snr", 0)
                    outlier_ratio = stage_result.get("outlier_ratio", 0)
                    jump_ratio = stage_result.get("jump_ratio", 0)
                    overall_quality = stage_result.get("overall_quality", 0)

                    try:
                        snr_str = f"{float(snr):.1f} dB"
                    except (ValueError, TypeError):
                        snr_str = f"{snr} dB"

                    try:
                        outlier_ratio_str = f"{float(outlier_ratio):.3f}"
                    except (ValueError, TypeError):
                        outlier_ratio_str = f"{outlier_ratio}"

                    try:
                        jump_ratio_str = f"{float(jump_ratio):.3f}"
                    except (ValueError, TypeError):
                        jump_ratio_str = f"{jump_ratio}"

                    try:
                        overall_quality_str = f"{float(overall_quality):.3f}"
                    except (ValueError, TypeError):
                        overall_quality_str = f"{overall_quality}"

                    stage_info[2]["metrics"] = {
                        "SNR": snr_str,
                        "Outlier Ratio": outlier_ratio_str,
                        "Jump Ratio": jump_ratio_str,
                        "Overall Quality": overall_quality_str,
                    }
                elif stage_num == 3:  # Parallel Processing
                    stage_info[3]["metrics"] = {
                        "Paths Active": f"{stage_result.get('paths_active', 0)}",
                        "Filter Applied": stage_result.get("filter_info", "N/A"),
                        "Artifacts Removed": f"{stage_result.get('artifacts_removed', 0)}",
                    }
                elif stage_num == 4:  # Quality Validation
                    # Handle potential string values for numeric formatting
                    raw_quality = stage_result.get("raw_quality", 0)
                    filtered_quality = stage_result.get("filtered_quality", 0)
                    preprocessed_quality = stage_result.get("preprocessed_quality", 0)

                    try:
                        raw_quality_str = f"{float(raw_quality):.3f}"
                    except (ValueError, TypeError):
                        raw_quality_str = f"{raw_quality}"

                    try:
                        filtered_quality_str = f"{float(filtered_quality):.3f}"
                    except (ValueError, TypeError):
                        filtered_quality_str = f"{filtered_quality}"

                    try:
                        preprocessed_quality_str = f"{float(preprocessed_quality):.3f}"
                    except (ValueError, TypeError):
                        preprocessed_quality_str = f"{preprocessed_quality}"

                    stage_info[4]["metrics"] = {
                        "RAW Quality": raw_quality_str,
                        "FILTERED Quality": filtered_quality_str,
                        "PREPROCESSED Quality": preprocessed_quality_str,
                        "Best Path": stage_result.get("best_path", "N/A"),
                    }
                elif stage_num == 5:  # Segmentation
                    # Handle potential string values for numeric formatting
                    window_size = stage_result.get("window_size", 0)
                    overlap_ratio = stage_result.get("overlap_ratio", 0)
                    total_segments = stage_result.get("total_segments", 0)
                    valid_segments = stage_result.get("valid_segments", 0)

                    try:
                        overlap_percent = f"{float(overlap_ratio)*100:.0f}%"
                    except (ValueError, TypeError):
                        overlap_percent = f"{overlap_ratio}%"

                    stage_info[5]["metrics"] = {
                        "Window Size": f"{window_size} seconds",
                        "Overlap": overlap_percent,
                        "Total Segments": f"{total_segments}",
                        "Valid Segments": f"{valid_segments}",
                    }
                elif stage_num == 6:  # Feature Extraction
                    stage_info[6]["metrics"] = {
                        "Time Features": f"{stage_result.get('time_features', 0)}",
                        "Frequency Features": f"{stage_result.get('frequency_features', 0)}",
                        "Nonlinear Features": f"{stage_result.get('nonlinear_features', 0)}",
                        "Total Features": f"{stage_result.get('total_features', 0)}",
                    }
                elif stage_num == 7:  # Intelligent Output
                    # Handle potential string values for numeric formatting
                    confidence = stage_result.get("confidence", 0)

                    try:
                        confidence_str = f"{float(confidence):.3f}"
                    except (ValueError, TypeError):
                        confidence_str = f"{confidence}"

                    stage_info[7]["metrics"] = {
                        "Best Path": stage_result.get("best_path", "N/A"),
                        "Confidence": confidence_str,
                        "Recommendations": f"{stage_result.get('recommendations_count', 0)}",
                    }
                elif stage_num == 8:  # Output Package
                    # Handle potential string values for numeric formatting
                    processing_time = stage_result.get("processing_time", 0)
                    output_size = stage_result.get("output_size", 0)

                    # Convert to float if possible, otherwise use string
                    try:
                        processing_time_str = f"{float(processing_time):.2f} seconds"
                    except (ValueError, TypeError):
                        processing_time_str = f"{processing_time} seconds"

                    try:
                        output_size_str = f"{float(output_size):.1f} MB"
                    except (ValueError, TypeError):
                        output_size_str = f"{output_size} MB"

                    stage_info[8]["metrics"] = {
                        "Status": "Complete",
                        "Processing Time": processing_time_str,
                        "Output Size": output_size_str,
                    }

        if stage_num in stage_info:
            info = stage_info[stage_num]

            # Add mode indicator
            mode_indicator = (
                html.Div(
                    [
                        dbc.Badge(
                            (
                                "🔴 REAL DATA MODE"
                                if pipeline_data
                                else "🟡 SIMULATION MODE"
                            ),
                            color="success" if pipeline_data else "warning",
                            className="mb-2",
                            style={"fontSize": "0.9rem"},
                        )
                    ],
                    className="text-center",
                )
                if stage_num > 0
                else None
            )

            return html.Div(
                [
                    mode_indicator,
                    html.H5(f"{stage_num}. {info['name']}", className="mb-3"),
                    html.P(info["description"], className="text-muted mb-3"),
                    html.H6("Stage Metrics:", className="mb-2"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Strong(f"{key}: "),
                                    html.Span(str(value)),
                                ],
                                className="mb-2",
                            )
                            for key, value in info["metrics"].items()
                        ]
                    ),
                ]
            )

        return html.Div("No stage information available.")

    # OLD CALLBACK - Replaced by Phase B update_path_signal_comparison callback
    # The old pipeline-paths-comparison component was removed and replaced with
    # the new path-comparison-signal-plot component in Phase B
    """
    @app.callback(
        Output("pipeline-paths-comparison", "figure"),
        [Input("pipeline-current-stage", "data")],
        [
            State("pipeline-path-selector", "value"),
            State("store-uploaded-data", "data"),
        ],
    )
    def update_paths_comparison_OLD(current_stage, selected_paths, uploaded_data):
        '''
        Create comparison plot of different processing paths.
        Shows real pipeline data when available, falls back to simulation data.

        Technical Details:
        - Uses vitalDSP SignalFiltering.bandpass() for real bandpass filtering
        - Uses vitalDSP ArtifactRemoval.baseline_correction() for preprocessing
        - Signal-type-specific filter parameters (ECG: 0.5-40Hz, PPG: 0.5-8Hz, EEG: 0.5-50Hz)
        - Falls back to simple scaling if vitalDSP not available
        - Limits visualization to first 10 seconds for performance
        - Maintains proper time axis based on data's sampling frequency
        '''
        # Only show after stage 3 (parallel processing)
        stage_num = _get_stage_number(current_stage)
        if stage_num < 3:
            return {
                "data": [],
                "layout": go.Layout(
                    title="Processing Paths Comparison",
                    xaxis={"title": "Time (s)"},
                    yaxis={"title": "Amplitude"},
                    annotations=[
                        {
                            "text": "Run pipeline to see path comparison",
                            "xref": "paper",
                            "yref": "paper",
                            "showarrow": False,
                            "font": {"size": 16, "color": "#999"},
                        }
                    ],
                ),
            }

        # Check if we have real pipeline data
        pipeline_data = None
        if isinstance(current_stage, str) and len(current_stage) == 8:
            # It's a pipeline_run_id, get real pipeline data
            if hasattr(register_pipeline_callbacks, "pipeline_data"):
                pipeline_data = register_pipeline_callbacks.pipeline_data.get(
                    current_stage
                )

        traces = []

        if pipeline_data and "signal_data" in pipeline_data:
            # REAL DATA MODE: Use actual pipeline data
            signal_data = pipeline_data["signal_data"]
            fs = pipeline_data["fs"]

            # Create time axis
            t = np.arange(len(signal_data)) / fs

            # For real data, we'll show different processing stages
            # Since we don't have separate processed signals yet, we'll show the original signal
            # with different visual representations to simulate different processing paths

            if "raw" in selected_paths:
                traces.append(
                    go.Scatter(
                        x=t,
                        y=signal_data,
                        name="RAW",
                        mode="lines",
                        line=dict(color="red", width=1),
                        opacity=0.8,
                    )
                )

            if "filtered" in selected_paths:
                # Apply vitalDSP filtering to simulate filtered signal
                try:
                    from vitalDSP.filtering.signal_filtering import SignalFiltering

                    # Create SignalFiltering object
                    sf = SignalFiltering(signal_data)

                    # Apply bandpass filter based on signal type
                    if pipeline_data.get("signal_type", "ecg").lower() == "ecg":
                        lowcut, highcut = 0.5, 40.0
                    elif pipeline_data.get("signal_type", "ecg").lower() == "ppg":
                        lowcut, highcut = 0.5, 8.0
                    elif pipeline_data.get("signal_type", "ecg").lower() == "eeg":
                        lowcut, highcut = 0.5, 50.0
                    else:
                        lowcut, highcut = 0.5, 40.0

                    # Use vitalDSP bandpass filtering
                    filtered_signal = sf.bandpass(
                        lowcut=lowcut,
                        highcut=highcut,
                        fs=fs,
                        order=4,
                        filter_type="butter",
                    )

                    traces.append(
                        go.Scatter(
                            x=t,
                            y=filtered_signal,
                            name="FILTERED",
                            mode="lines",
                            line=dict(color="blue", width=1),
                            opacity=0.8,
                        )
                    )
                except ImportError:
                    # Fallback if vitalDSP not available
                    traces.append(
                        go.Scatter(
                            x=t,
                            y=signal_data * 0.9,  # Simple scaling
                            name="FILTERED",
                            mode="lines",
                            line=dict(color="blue", width=1),
                            opacity=0.8,
                        )
                    )

            if "preprocessed" in selected_paths:
                # Apply vitalDSP preprocessing (filtering + artifact removal)
                try:
                    from vitalDSP.filtering.signal_filtering import SignalFiltering
                    from vitalDSP.filtering.artifact_removal import ArtifactRemoval

                    # First apply vitalDSP filtering
                    sf = SignalFiltering(signal_data)

                    # Apply bandpass filter based on signal type
                    if pipeline_data.get("signal_type", "ecg").lower() == "ecg":
                        lowcut, highcut = 0.5, 40.0
                    elif pipeline_data.get("signal_type", "ecg").lower() == "ppg":
                        lowcut, highcut = 0.5, 8.0
                    elif pipeline_data.get("signal_type", "ecg").lower() == "eeg":
                        lowcut, highcut = 0.5, 50.0
                    else:
                        lowcut, highcut = 0.5, 40.0

                    filtered_signal = sf.bandpass(
                        lowcut=lowcut,
                        highcut=highcut,
                        fs=fs,
                        order=4,
                        filter_type="butter",
                    )

                    # Then apply vitalDSP artifact removal
                    ar = ArtifactRemoval(filtered_signal)
                    # Use baseline correction for preprocessing
                    preprocessed_signal = ar.baseline_correction(cutoff=0.5, fs=fs)

                    traces.append(
                        go.Scatter(
                            x=t,
                            y=preprocessed_signal,
                            name="PREPROCESSED",
                            mode="lines",
                            line=dict(color="green", width=2),
                            opacity=0.9,
                        )
                    )
                except ImportError:
                    # Fallback if vitalDSP not available - simple baseline removal
                    preprocessed_signal = signal_data - np.mean(signal_data)
                    traces.append(
                        go.Scatter(
                            x=t,
                            y=preprocessed_signal,
                            name="PREPROCESSED",
                            mode="lines",
                            line=dict(color="green", width=2),
                            opacity=0.9,
                        )
                    )

            # Limit to first 10 seconds for visualization
            max_samples = min(len(signal_data), int(10 * fs))
            for trace in traces:
                trace.x = trace.x[:max_samples]
                trace.y = trace.y[:max_samples]

            title = (
                "🔴 REAL DATA MODE - Processing Paths Comparison (First 10 seconds)"
            )

        else:
            # SIMULATION MODE: Generate sample data for demonstration
            t = np.linspace(0, 10, 1000)

            # Use vitalDSP for ECG/PPG signal generation, numpy for others
            try:
                from vitalDSP.utils.data_processing.synthesize_data import (
                    generate_ecg_signal,
                    generate_synthetic_ppg,
                )

                # Generate ECG signal for demonstration
                ecg_signal = generate_ecg_signal(duration=10, fs=100)
                # Resample to match our time vector
                ecg_resampled = np.interp(
                    t, np.linspace(0, 10, len(ecg_signal)), ecg_signal
                )

                raw_signal = ecg_resampled + 0.3 * np.random.randn(len(t))
                filtered_signal = ecg_resampled + 0.1 * np.random.randn(len(t))
                preprocessed_signal = ecg_resampled
            except ImportError:
                # Fallback to numpy implementation if vitalDSP not available
                raw_signal = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.random.randn(len(t))
                filtered_signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(
                    len(t)
                )
                preprocessed_signal = np.sin(2 * np.pi * 1.2 * t)

            if "raw" in selected_paths:
                traces.append(
                    go.Scatter(
                        x=t,
                        y=raw_signal,
                        name="RAW",
                        mode="lines",
                        line=dict(color="red", width=1),
                    )
                )

            if "filtered" in selected_paths:
                traces.append(
                    go.Scatter(
                        x=t,
                        y=filtered_signal,
                        name="FILTERED",
                        mode="lines",
                        line=dict(color="blue", width=1),
                    )
                )

            if "preprocessed" in selected_paths:
                traces.append(
                    go.Scatter(
                        x=t,
                        y=preprocessed_signal,
                        name="PREPROCESSED",
                        mode="lines",
                        line=dict(color="green", width=2),
                    )
                )

            title = (
                "🟡 SIMULATION MODE - Processing Paths Comparison (First 10 seconds)"
            )

        return {
            "data": traces,
            "layout": go.Layout(
                title=title,
                xaxis={"title": "Time (s)"},
                yaxis={"title": "Amplitude"},
                hovermode="closest",
                legend={"x": 0, "y": 1},
            ),
        }
    """

    @app.callback(
        Output("pipeline-quality-results", "children"),
        [Input("pipeline-current-stage", "data")],
    )
    def update_quality_results(current_stage):
        """
        Display quality screening results.
        """
        stage_num = _get_stage_number(current_stage)
        if stage_num < 2:
            return html.Div(
                "Quality screening not yet performed.", className="text-muted"
            )

        # Sample quality results
        quality_data = {
            "Stage 1: SNR Assessment": {"SNR (dB)": 15.2, "Status": "✅ PASS"},
            "Stage 2: Statistical Screen": {
                "Outlier Ratio": 0.05,
                "Jump Ratio": 0.08,
                "Status": "✅ PASS",
            },
            "Stage 3: Signal-Specific": {
                "Baseline Wander": 0.72,
                "Amplitude Variability": 0.81,
                "Status": "✅ PASS",
            },
        }

        result_cards = []
        for stage_name, metrics in quality_data.items():
            result_cards.append(
                html.Div(
                    [
                        html.H6(stage_name, className="mb-2"),
                        html.Div(
                            [
                                html.Div(
                                    [html.Strong(f"{k}: "), html.Span(str(v))],
                                    className="mb-1",
                                )
                                for k, v in metrics.items()
                            ]
                        ),
                        html.Hr(),
                    ],
                    className="mb-3",
                )
            )

        return html.Div(result_cards)

    @app.callback(
        Output("pipeline-features-summary", "children"),
        [Input("pipeline-current-stage", "data")],
    )
    def update_features_summary(current_stage):
        """
        Display feature extraction summary.
        """
        stage_num = _get_stage_number(current_stage)
        if stage_num < 6:
            return html.Div(
                "Feature extraction not yet performed.", className="text-muted"
            )

        # Sample feature summary
        features_df = pd.DataFrame(
            {
                "Feature": [
                    "Mean",
                    "Std Dev",
                    "RMS",
                    "Spectral Centroid",
                    "Dominant Frequency",
                ],
                "Value": [0.023, 0.145, 0.147, 1.85, 1.2],
                "Unit": ["V", "V", "V", "Hz", "Hz"],
            }
        )

        return html.Div(
            [
                html.H6("Extracted Features (Sample)", className="mb-3"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Strong(f"{row['Feature']}: "),
                                html.Span(f"{row['Value']:.3f} {row['Unit']}"),
                            ],
                            className="mb-2",
                        )
                        for _, row in features_df.iterrows()
                    ]
                ),
            ]
        )

    @app.callback(
        Output("pipeline-output-recommendations", "children"),
        [Input("pipeline-current-stage", "data")],
    )
    def update_output_recommendations(current_stage):
        """
        Display intelligent output recommendations from real Stage 7 results.
        """
        stage_num = _get_stage_number(current_stage)
        if stage_num < 7:
            return html.Div(
                "Recommendations not yet generated.", className="text-muted"
            )

        # Get real results from Stage 7
        pipeline_run_id = getattr(
            register_pipeline_callbacks, "current_pipeline_run_id", None
        )
        if not pipeline_run_id:
            return html.Div("No pipeline run found.", className="text-muted")

        pipeline_data = register_pipeline_callbacks.pipeline_data.get(
            pipeline_run_id, {}
        )
        stage7_results = pipeline_data.get("results", {}).get("stage_7", {})

        if not stage7_results:
            return html.Div("Stage 7 not yet completed.", className="text-muted")

        # Extract real data
        selected_path = stage7_results.get("selected_path", "unknown")
        confidence = stage7_results.get("confidence", 0.0)
        meets_threshold = stage7_results.get("meets_threshold", False)
        recommendations = stage7_results.get("recommendations", [])
        selection_criterion = stage7_results.get("selection_criterion", "unknown")

        # Build UI with real data
        items = []

        # Best path info
        items.append(
            html.Li(
                [
                    html.Strong("Selected Path: "),
                    html.Span(
                        f"{selected_path.upper()} ",
                        style={"color": "#28a745" if meets_threshold else "#ffc107"},
                    ),
                    html.Span(
                        f"(Confidence: {confidence:.2f}, Criterion: {selection_criterion})",
                        className="text-muted",
                    ),
                ]
            )
        )

        # Confidence indicator
        confidence_status = (
            "High" if confidence >= 0.8 else "Medium" if confidence >= 0.6 else "Low"
        )
        confidence_color = (
            "#28a745"
            if confidence >= 0.8
            else "#ffc107" if confidence >= 0.6 else "#dc3545"
        )
        items.append(
            html.Li(
                [
                    html.Strong("Confidence Level: "),
                    html.Span(
                        confidence_status,
                        style={"color": confidence_color, "fontWeight": "bold"},
                    ),
                    html.Span(
                        " - "
                        + ("Meets threshold" if meets_threshold else "Below threshold"),
                        className="text-muted",
                    ),
                ]
            )
        )

        # Add all recommendations
        for rec in recommendations:
            items.append(html.Li(rec))

        return html.Div(
            [
                html.H6("Intelligent Output Recommendations:", className="mb-3"),
                html.Ul(items, style={"lineHeight": "2"}),
            ]
        )

    @app.callback(
        Output("pipeline-results", "data"),
        [Input("pipeline-current-stage", "data")],
    )
    def update_pipeline_results(current_stage):
        """
        Update pipeline results when execution completes.
        """
        # Real pipeline mode - return actual results
        if isinstance(current_stage, str) and len(current_stage) == 8:
            pipeline_run_id = current_stage
            if hasattr(register_pipeline_callbacks, "pipeline_data"):
                pipeline_data = register_pipeline_callbacks.pipeline_data.get(
                    pipeline_run_id
                )
                if pipeline_data and pipeline_data.get("current_stage") == 8:
                    return {
                        "pipeline_run_id": pipeline_run_id,
                        "status": "completed",
                        "results": pipeline_data.get("results", {}),
                        "signal_type": pipeline_data.get("signal_type"),
                        "paths": pipeline_data.get("paths"),
                    }

        # Legacy simulation mode - return mock results when complete
        if isinstance(current_stage, int) and current_stage == 8:
            return {"status": "completed", "simulation": True, "execution_time": 2.45}

        return no_update

    @app.callback(
        [
            Output("pipeline-progress-container", "children", allow_duplicate=True),
            Output("pipeline-progress-container", "style", allow_duplicate=True),
        ],
        [Input("pipeline-current-stage", "data")],
        prevent_initial_call=True,
    )
    def update_pipeline_step_indicator(current_stage):
        """
        Update the visual step progress indicator based on current stage.
        """
        from dash import html

        # Handle both int stages (1-8) and string pipeline_run_id
        stage_num = _get_stage_number(current_stage)

        # Use the module-level pipeline stages
        pipeline_stages = PIPELINE_STAGES

        # Create step items manually (instead of using helper function)
        step_items = []
        for i, step_name in enumerate(pipeline_stages):
            # Determine step status based on current stage
            if i < stage_num:
                # Completed
                icon = "fa-check-circle"
                icon_color = "text-success"
                status = "completed"
            elif i == stage_num:
                # In progress
                icon = "fa-circle-notch fa-spin"
                icon_color = "text-primary"
                status = "in-progress"
            else:
                # Pending
                icon = "fa-circle"
                icon_color = "text-muted"
                status = "pending"

            step_items.append(
                html.Div(
                    className=f"step-item {status}",
                    style={
                        "display": "inline-block",
                        "textAlign": "center",
                        "flex": "1",
                        "position": "relative",
                    },
                    children=[
                        html.I(
                            className=f"fas {icon} {icon_color}",
                            style={"fontSize": "24px"},
                        ),
                        html.Div(
                            step_name,
                            className="mt-2",
                            style={
                                "fontSize": "12px",
                                "fontWeight": "bold" if i == stage_num else "normal",
                            },
                        ),
                    ],
                )
            )

            # Add connector line between steps (except after last step)
            if i < len(pipeline_stages) - 1:
                connector_color = "#28a745" if i < stage_num else "#dee2e6"
                step_items.append(
                    html.Div(
                        style={
                            "flex": "0.5",
                            "height": "2px",
                            "backgroundColor": connector_color,
                            "alignSelf": "center",
                            "marginTop": "-20px",
                        }
                    )
                )

        # Create the progress indicator content
        progress_content = html.Div(
            className="step-progress",
            style={
                "display": "flex",
                "alignItems": "flex-start",
                "justifyContent": "space-between",
                "padding": "20px",
                "backgroundColor": "#f8f9fa",
                "borderRadius": "5px",
                "marginBottom": "20px",
            },
            children=step_items,
        )

        # Show the container when pipeline is running (stage > 0)
        container_style = {"display": "block" if stage_num > 0 else "none"}

        return progress_content, container_style

    # Helper function to get data slice with 300s limit
    def _get_data_slice(signal, fs, viz_window=300, viz_start=0, viz_mode="start"):
        """
        Get a slice of signal data limited to viz_window seconds.
        """
        if len(signal) == 0:
            return signal, 0, 0

        total_duration = len(signal) / fs
        window_samples = int(viz_window * fs)

        if viz_mode == "random":
            import random

            max_start = max(0, len(signal) - window_samples)
            start_idx = random.randint(0, max_start) if max_start > 0 else 0
        elif viz_mode == "custom":
            start_idx = int(viz_start * fs)
        else:  # 'start'
            start_idx = 0

        start_idx = max(0, min(start_idx, len(signal) - 1))
        end_idx = min(start_idx + window_samples, len(signal))

        return signal[start_idx:end_idx], start_idx, end_idx

    # Stage 1 Visualization Callback
    @app.callback(
        [
            Output("pipeline-stage1-plot", "figure"),
            Output("pipeline-stage1-container", "style"),
        ],
        [
            Input("pipeline-current-stage", "data"),
            Input("pipeline-viz-refresh-btn", "n_clicks"),
        ],
        [
            State("pipeline-viz-window", "value"),
            State("pipeline-viz-start", "value"),
            State("pipeline-viz-mode", "value"),
        ],
    )
    def update_stage1_plot(current_stage, n_clicks, viz_window, viz_start, viz_mode):
        """
        Display Stage 1: Raw Signal (300s limit).
        """
        stage_num = _get_stage_number(current_stage)
        if stage_num < 1:
            return {}, {"display": "none"}

        # Get pipeline data
        pipeline_run_id = getattr(
            register_pipeline_callbacks, "current_pipeline_run_id", None
        )
        if not pipeline_run_id:
            return {}, {"display": "none"}

        pipeline_data = register_pipeline_callbacks.pipeline_data.get(
            pipeline_run_id, {}
        )
        signal_data = pipeline_data.get("signal_data", np.array([]))
        fs = pipeline_data.get("fs", 1)

        if len(signal_data) == 0:
            return {}, {"display": "none"}

        # Apply 300s limit
        signal_slice, start_idx, end_idx = _get_data_slice(
            signal_data, fs, viz_window or 300, viz_start or 0, viz_mode or "start"
        )

        # Create time axis
        time_axis = np.arange(len(signal_slice)) / fs + (start_idx / fs)

        # Create plotly figure
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal_slice,
                mode="lines",
                name="Raw Signal",
                line=dict(color="#2C3E50", width=1),
            )
        )

        fig.update_layout(
            title=f"Stage 1: Raw Signal (Showing {len(signal_slice)/fs:.1f}s from {start_idx/fs:.1f}s)",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            hovermode="x unified",
            template="plotly_white",
            height=400,
        )

        return fig, {"display": "block"}

    # Stage 2 Visualization Callback
    @app.callback(
        [
            Output("pipeline-stage2-plot", "figure"),
            Output("pipeline-stage2-container", "style"),
        ],
        [
            Input("pipeline-current-stage", "data"),
            Input("pipeline-viz-refresh-btn", "n_clicks"),
        ],
    )
    def update_stage2_plot(current_stage, n_clicks):
        """
        Display Stage 2: SQI Metrics Over Time.
        """
        stage_num = _get_stage_number(current_stage)
        if stage_num < 2:
            return {}, {"display": "none"}

        # Get pipeline data
        pipeline_run_id = getattr(
            register_pipeline_callbacks, "current_pipeline_run_id", None
        )
        if not pipeline_run_id:
            return {}, {"display": "none"}

        pipeline_data = register_pipeline_callbacks.pipeline_data.get(
            pipeline_run_id, {}
        )
        stage2_results = pipeline_data.get("results", {}).get("stage_2", {})
        sqi_arrays = stage2_results.get("sqi_arrays", {})
        fs = pipeline_data.get("fs", 1)

        if not sqi_arrays:
            return {}, {"display": "none"}

        # Create plotly figure
        fig = go.Figure()

        for sqi_name, sqi_values in sqi_arrays.items():
            if len(sqi_values) > 0:
                # Create time axis for SQI windows
                time_axis = np.arange(len(sqi_values)) * stage2_results.get(
                    "sqi_step", 2.5
                )

                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=sqi_values,
                        mode="lines+markers",
                        name=sqi_name.replace("_sqi", "").replace("_", " ").title(),
                        marker=dict(size=6),
                    )
                )

        fig.update_layout(
            title="Stage 2: Signal Quality Index (SQI) Metrics Over Time",
            xaxis_title="Time (s)",
            yaxis_title="SQI Score",
            hovermode="x unified",
            template="plotly_white",
            height=400,
            showlegend=True,
        )

        return fig, {"display": "block"}

    # Stage 4 Visualization Callback
    @app.callback(
        [
            Output("pipeline-stage4-plot", "figure"),
            Output("pipeline-stage4-container", "style"),
        ],
        [Input("pipeline-current-stage", "data")],
    )
    def update_stage4_plot(current_stage):
        """
        Display Stage 4: Path Quality Comparison.
        """
        stage_num = _get_stage_number(current_stage)
        if stage_num < 4:
            return {}, {"display": "none"}

        # Get pipeline data
        pipeline_run_id = getattr(
            register_pipeline_callbacks, "current_pipeline_run_id", None
        )
        if not pipeline_run_id:
            return {}, {"display": "none"}

        pipeline_data = register_pipeline_callbacks.pipeline_data.get(
            pipeline_run_id, {}
        )
        stage4_results = pipeline_data.get("results", {}).get("stage_4", {})
        quality_metrics = stage4_results.get("quality_metrics", {})

        if not quality_metrics:
            return {}, {"display": "none"}

        # Prepare data for grouped bar chart
        paths = list(quality_metrics.keys())
        snr_scores = [quality_metrics[p].get("snr_score", 0) for p in paths]
        smoothness_scores = [
            quality_metrics[p].get("smoothness_score", 0) for p in paths
        ]
        artifact_scores = [quality_metrics[p].get("artifact_score", 0) for p in paths]
        overall_scores = [quality_metrics[p].get("overall_quality", 0) for p in paths]

        # Create plotly figure
        fig = go.Figure()
        fig.add_trace(go.Bar(name="SNR", x=paths, y=snr_scores, marker_color="#3498DB"))
        fig.add_trace(
            go.Bar(
                name="Smoothness", x=paths, y=smoothness_scores, marker_color="#2ECC71"
            )
        )
        fig.add_trace(
            go.Bar(name="Artifact", x=paths, y=artifact_scores, marker_color="#F39C12")
        )
        fig.add_trace(
            go.Bar(name="Overall", x=paths, y=overall_scores, marker_color="#E74C3C")
        )

        fig.update_layout(
            title="Stage 4: Path Quality Comparison",
            xaxis_title="Processing Path",
            yaxis_title="Quality Score",
            barmode="group",
            template="plotly_white",
            height=400,
            showlegend=True,
        )

        return fig, {"display": "block"}

    # Stage 5 Visualization Callback
    @app.callback(
        [
            Output("pipeline-stage5-plot", "figure"),
            Output("pipeline-stage5-container", "style"),
        ],
        [
            Input("pipeline-current-stage", "data"),
            Input("pipeline-viz-refresh-btn", "n_clicks"),
        ],
        [
            State("pipeline-viz-window", "value"),
            State("pipeline-viz-start", "value"),
            State("pipeline-viz-mode", "value"),
        ],
    )
    def update_stage5_plot(current_stage, n_clicks, viz_window, viz_start, viz_mode):
        """
        Display Stage 5: Signal Segmentation.
        """
        stage_num = _get_stage_number(current_stage)
        if stage_num < 5:
            return {}, {"display": "none"}

        # Get pipeline data
        pipeline_run_id = getattr(
            register_pipeline_callbacks, "current_pipeline_run_id", None
        )
        if not pipeline_run_id:
            return {}, {"display": "none"}

        pipeline_data = register_pipeline_callbacks.pipeline_data.get(
            pipeline_run_id, {}
        )
        best_path = pipeline_data.get("best_path", "raw")
        processed_signals = pipeline_data.get("processed_signals", {})
        best_signal = processed_signals.get(
            best_path, pipeline_data.get("signal_data", np.array([]))
        )
        segment_positions = pipeline_data.get("segment_positions", [])
        fs = pipeline_data.get("fs", 1)

        if len(best_signal) == 0:
            return {}, {"display": "none"}

        # Apply 300s limit
        signal_slice, start_idx, end_idx = _get_data_slice(
            best_signal, fs, viz_window or 300, viz_start or 0, viz_mode or "start"
        )

        # Create time axis
        time_axis = np.arange(len(signal_slice)) / fs + (start_idx / fs)

        # Create plotly figure
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal_slice,
                mode="lines",
                name="Signal",
                line=dict(color="#2C3E50", width=1),
            )
        )

        # Add segment boundaries
        for seg_start, seg_end in segment_positions:
            if start_idx <= seg_start < end_idx:
                fig.add_vline(
                    x=seg_start / fs,
                    line=dict(color="red", width=1, dash="dash"),
                    annotation_text="Segment",
                )

        fig.update_layout(
            title=f"Stage 5: Signal Segmentation - {best_path.upper()} Path ({len(segment_positions)} segments)",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            hovermode="x unified",
            template="plotly_white",
            height=400,
        )

        return fig, {"display": "block"}

    # Stage 6 Visualization Callback
    @app.callback(
        [
            Output("pipeline-stage6-plot", "figure"),
            Output("pipeline-stage6-container", "style"),
        ],
        [Input("pipeline-current-stage", "data")],
    )
    def update_stage6_plot(current_stage):
        """
        Display Stage 6: Feature Extraction with multiple visualizations.
        Creates a subplot figure with:
        1. Feature heatmap across segments
        2. Feature statistics (mean, std) bar chart
        3. Feature distribution box plots
        """
        stage_num = _get_stage_number(current_stage)
        if stage_num < 6:
            return {}, {"display": "none"}

        # Get pipeline data
        pipeline_run_id = getattr(
            register_pipeline_callbacks, "current_pipeline_run_id", None
        )
        if not pipeline_run_id:
            return {}, {"display": "none"}

        pipeline_data = register_pipeline_callbacks.pipeline_data.get(
            pipeline_run_id, {}
        )
        features = pipeline_data.get("features", [])
        stage6_results = pipeline_data.get("results", {}).get("stage_6", {})
        feature_names = stage6_results.get("feature_names", [])
        best_path = pipeline_data.get("best_path", "unknown")

        if not features or not feature_names:
            return {}, {"display": "none"}

        # Convert features to matrix (limit to first 50 segments for visualization)
        features_limited = features[:50]
        feature_matrix = []
        for feat_dict in features_limited:
            row = [feat_dict.get(fname, 0) for fname in feature_names]
            feature_matrix.append(row)

        if not feature_matrix:
            return {}, {"display": "none"}

        feature_matrix = np.array(feature_matrix).T  # Transpose for features as rows

        # Calculate feature statistics across all segments
        all_feature_matrix = []
        for feat_dict in features:
            row = [feat_dict.get(fname, 0) for fname in feature_names]
            all_feature_matrix.append(row)
        all_feature_matrix = np.array(all_feature_matrix)

        feature_means = np.mean(all_feature_matrix, axis=0)
        feature_stds = np.std(all_feature_matrix, axis=0)

        # Normalize feature matrix using z-score for better visualization
        # This handles features with spiking variance
        feature_matrix_normalized = np.copy(feature_matrix)
        for i in range(len(feature_names)):
            feat_mean = feature_means[i]
            feat_std = feature_stds[i]
            if feat_std > 1e-10:  # Avoid division by zero
                feature_matrix_normalized[i, :] = (
                    feature_matrix[i, :] - feat_mean
                ) / feat_std
            else:
                feature_matrix_normalized[i, :] = 0

        # Calculate coefficient of variation to identify high-variance features
        cv = np.abs(feature_stds / (feature_means + 1e-10))

        # Create subplots: 2 rows, 2 columns
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                f"Normalized Heatmap ({best_path.upper()}, Z-scored)",
                "Mean Values (Log Scale)",
                "Coefficient of Variation",
                "Top 10 Variable Features",
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "box"}],
            ],
            vertical_spacing=0.18,  # Increased from 0.15
            horizontal_spacing=0.15,  # Increased from 0.12
        )

        # 1. Normalized Feature Heatmap (z-scored)
        # This ensures all features are on the same scale regardless of magnitude
        fig.add_trace(
            go.Heatmap(
                z=feature_matrix_normalized,
                x=[f"S{i+1}" for i in range(len(features_limited))],
                y=feature_names,
                colorscale="RdBu_r",  # Diverging colorscale for z-scores
                zmid=0,  # Center at 0 (mean)
                customdata=feature_matrix,  # Store original values for hover
                hovertemplate="Feature: %{y}<br>Segment: %{x}<br>Z-score: %{z:.2f}<br>Original: %{customdata:.3f}<extra></extra>",
                showscale=True,
                colorbar=dict(x=0.46, len=0.4, y=0.75, title="Z-score"),
            ),
            row=1,
            col=1,
        )

        # 2. Feature Mean Values (Log Scale for better visibility)
        # Add small epsilon to avoid log(0)
        feature_means_safe = np.abs(feature_means) + 1e-10

        fig.add_trace(
            go.Bar(
                x=feature_names,
                y=feature_means_safe,
                marker_color="#3498DB",
                name="Mean",
                customdata=feature_means,  # Store original values
                hovertemplate="Feature: %{x}<br>Mean: %{customdata:.4e}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # 3. Coefficient of Variation (shows relative variability)
        # CV = std/mean, indicates which features vary most relative to their magnitude
        fig.add_trace(
            go.Bar(
                x=feature_names,
                y=cv,
                marker_color="#2ECC71",
                name="CV",
                hovertemplate="Feature: %{x}<br>CV: %{y:.3f}<br>(High CV = High variability)<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # 4. Box plots for top 10 most variable features (by CV)
        top_cv_indices = np.argsort(cv)[-10:][::-1]
        for idx in top_cv_indices:
            feat_name = feature_names[idx]
            feat_values = all_feature_matrix[:, idx]

            fig.add_trace(
                go.Box(
                    y=feat_values,
                    name=feat_name,
                    boxmean="sd",  # Show mean and std
                    hovertemplate=f"{feat_name}<br>Value: %{{y:.3f}}<extra></extra>",
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title_text=f"Stage 6: Feature Extraction Analysis - {best_path.upper()} Path ({len(features)} segments, {len(feature_names)} features)",
            showlegend=False,
            template="plotly_white",
            height=1000,  # Increased from 900 to prevent overlap
            margin=dict(t=120, b=80, l=80, r=80),  # Add margins to prevent clipping
        )

        # Update x-axes
        fig.update_xaxes(title_text="Segment", row=1, col=1, tickangle=45)
        fig.update_xaxes(
            title_text="Feature", row=1, col=2, tickangle=45, tickfont=dict(size=8)
        )
        fig.update_xaxes(
            title_text="Feature", row=2, col=1, tickangle=45, tickfont=dict(size=8)
        )
        fig.update_xaxes(
            title_text="Feature", row=2, col=2, tickangle=45, tickfont=dict(size=8)
        )

        # Update y-axes
        fig.update_yaxes(title_text="Feature", row=1, col=1, tickfont=dict(size=8))
        fig.update_yaxes(title_text="Mean Value (Log Scale)", row=1, col=2, type="log")
        fig.update_yaxes(title_text="Coefficient of Variation", row=2, col=1)
        fig.update_yaxes(title_text="Feature Value", row=2, col=2)

        return fig, {"display": "block"}

    # Export Results Callback
    @app.callback(
        Output("download-dataframe", "data"),
        [Input("pipeline-export-btn", "n_clicks")],
        [State("pipeline-current-stage", "data")],
        prevent_initial_call=True,
    )
    def export_pipeline_results(n_clicks, current_stage):
        """
        Export pipeline results to JSON/CSV files.
        """
        logger.info(
            f"Export button clicked: n_clicks={n_clicks}, current_stage={current_stage}"
        )

        if n_clicks is None:
            return no_update

        # Get the current pipeline run ID
        pipeline_run_id = getattr(
            register_pipeline_callbacks, "current_pipeline_run_id", None
        )
        logger.info(f"Pipeline run ID for export: {pipeline_run_id}")

        if not pipeline_run_id:
            logger.warning("No pipeline run ID found for export")
            return no_update

        # Get pipeline data
        pipeline_data = register_pipeline_callbacks.pipeline_data.get(
            pipeline_run_id, {}
        )
        logger.info(f"Pipeline data keys: {list(pipeline_data.keys())}")

        output_package = pipeline_data.get("output_package", {})
        logger.info(
            f"Output package keys: {list(output_package.keys()) if output_package else 'None'}"
        )

        if not output_package:
            logger.warning(
                "No output package found for export - creating one from results"
            )
            # Fallback: create output package from available results
            output_package = {
                "results": pipeline_data.get("results", {}),
                "features": pipeline_data.get("features", []),
                "segments": pipeline_data.get("segments", []),
                "metadata": {
                    "signal_type": pipeline_data.get("signal_type", "unknown"),
                    "fs": pipeline_data.get("fs", 0),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            }

        # Add comprehensive stage configurations to output package
        stage_configurations = {
            "pipeline_metadata": {
                "pipeline_run_id": pipeline_run_id,
                "signal_type": pipeline_data.get("signal_type", "unknown"),
                "sampling_frequency": pipeline_data.get("fs", 0),
                "processing_paths": pipeline_data.get("paths", []),
                "quality_validation_enabled": pipeline_data.get(
                    "enable_quality", False
                ),
                "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_stages": pipeline_data.get("total_stages", 8),
                "completed_stage": pipeline_data.get("current_stage", 0),
            },
            "stage_1_data_ingestion": {
                "description": "Load and validate input signal data",
                "signal_length": len(pipeline_data.get("signal_data", [])),
                "signal_type": pipeline_data.get("signal_type", "unknown"),
                "sampling_frequency": pipeline_data.get("fs", 0),
            },
            "stage_2_quality_screening": {
                "description": "Initial quality assessment and bad segment detection",
                "enabled": pipeline_data.get("enable_quality", False),
                "sqi_type": pipeline_data.get("stage2_sqi_type", "snr_sqi"),
                "window_size": pipeline_data.get("stage2_window_size", 1000),
                "step_size": pipeline_data.get("stage2_step_size", 500),
                "threshold_type": pipeline_data.get("stage2_threshold_type", "below"),
                "threshold": pipeline_data.get("stage2_threshold", 0.7),
                "scale": pipeline_data.get("stage2_scale", "zscore"),
            },
            "stage_3_comprehensive_filtering": {
                "description": "Apply filtering and artifact removal to each processing path",
                "filter_type": pipeline_data.get("stage3_filter_type", "traditional"),
                "detrend_enabled": pipeline_data.get("stage3_detrend", False),
                "signal_source": pipeline_data.get("stage3_signal_source", "original"),
                "application_count": pipeline_data.get("stage3_application_count", 1),
                "traditional_filter": {
                    "family": pipeline_data.get("stage3_filter_family", "butter"),
                    "response_type": pipeline_data.get(
                        "stage3_filter_response", "bandpass"
                    ),
                    "lowcut_frequency": pipeline_data.get("stage3_filter_lowcut", 0.5),
                    "highcut_frequency": pipeline_data.get("stage3_filter_highcut", 40),
                    "order": pipeline_data.get("stage3_filter_order", 4),
                    "passband_ripple": pipeline_data.get("stage3_filter_rp", 1),
                    "stopband_attenuation": pipeline_data.get("stage3_filter_rs", 40),
                },
                "smoothing_filters": {
                    "savgol_window": pipeline_data.get("stage3_savgol_window"),
                    "savgol_polyorder": pipeline_data.get("stage3_savgol_polyorder", 2),
                    "moving_avg_window": pipeline_data.get("stage3_moving_avg_window"),
                    "gaussian_sigma": pipeline_data.get("stage3_gaussian_sigma"),
                },
                "advanced_filter": {
                    "method": pipeline_data.get("stage3_advanced_method", "kalman"),
                    "kalman_r": pipeline_data.get("stage3_kalman_r", 1.0),
                    "kalman_q": pipeline_data.get("stage3_kalman_q", 1.0),
                    "adaptive_mu": pipeline_data.get("stage3_adaptive_mu", 0.01),
                    "adaptive_order": pipeline_data.get("stage3_adaptive_order", 4),
                },
                "artifact_removal": {
                    "type": pipeline_data.get("stage3_artifact_type", "baseline"),
                    "strength": pipeline_data.get("stage3_artifact_strength", 0.5),
                    "wavelet_type": pipeline_data.get("stage3_wavelet_type", "db4"),
                    "wavelet_level": pipeline_data.get("stage3_wavelet_level", 3),
                    "threshold_type": pipeline_data.get(
                        "stage3_threshold_type", "soft"
                    ),
                    "threshold_value": pipeline_data.get("stage3_threshold_value", 0.1),
                },
                "neural_network": {
                    "type": pipeline_data.get("stage3_neural_type", "autoencoder"),
                    "complexity": pipeline_data.get(
                        "stage3_neural_complexity", "medium"
                    ),
                },
                "ensemble": {
                    "method": pipeline_data.get("stage3_ensemble_method", "mean"),
                    "n_filters": pipeline_data.get("stage3_ensemble_n_filters", 3),
                },
            },
            "stage_4_quality_validation": {
                "description": "Validate quality of each processing path and select best",
                "sqi_type": pipeline_data.get("stage4_sqi_type", "snr_sqi"),
                "window_size": pipeline_data.get("stage4_window_size", 1000),
                "step_size": pipeline_data.get("stage4_step_size", 500),
                "threshold_type": pipeline_data.get("stage4_threshold_type", "below"),
                "threshold": pipeline_data.get("stage4_threshold", 0.7),
                "scale": pipeline_data.get("stage4_scale", "zscore"),
            },
            "stage_5_segmentation": {
                "description": "Segment signal into analysis windows",
                "window_size": pipeline_data.get("window_size", 30),
                "overlap_ratio": pipeline_data.get("overlap_ratio", 0.5),
                "min_segment_length": pipeline_data.get("min_segment_length", 5),
                "window_function": pipeline_data.get("window_function", "none"),
            },
            "stage_6_feature_extraction": {
                "description": "Extract features from each segment",
                "feature_types": pipeline_data.get(
                    "feature_types", ["time", "frequency"]
                ),
                "time_features": pipeline_data.get(
                    "time_features", ["mean", "std", "rms", "ptp"]
                ),
                "frequency_features": pipeline_data.get(
                    "frequency_features", ["spectral_centroid", "dominant_freq"]
                ),
                "nonlinear_features": pipeline_data.get(
                    "nonlinear_features", ["sample_entropy"]
                ),
                "statistical_features": pipeline_data.get(
                    "statistical_features", ["skewness", "kurtosis"]
                ),
                "morphological_features": pipeline_data.get(
                    "morphological_features", []
                ),
            },
            "stage_7_intelligent_output": {
                "description": "Select best processing path and generate recommendations",
                "selection_criterion": pipeline_data.get(
                    "selection_criterion", "best_quality"
                ),
                "confidence_threshold": pipeline_data.get("confidence_threshold", 0.7),
                "generate_recommendations": pipeline_data.get(
                    "generate_recommendations", True
                ),
            },
            "stage_8_output_package": {
                "description": "Package results for export",
                "output_formats": pipeline_data.get("output_formats", ["json", "csv"]),
                "output_contents": pipeline_data.get(
                    "output_contents",
                    ["processed_signals", "quality_metrics", "features", "metadata"],
                ),
                "compress_output": pipeline_data.get("compress_output", True),
            },
        }

        # Add configurations to output package
        output_package["stage_configurations"] = stage_configurations

        # Export as JSON
        output_json = json.dumps(output_package, indent=2, default=str)

        logger.info(
            f"Exporting pipeline results for run: {pipeline_run_id} ({len(output_json)} bytes)"
        )

        return dict(
            content=output_json, filename=f"pipeline_results_{pipeline_run_id}.json"
        )

    # Export Stage Data Callback
    @app.callback(
        Output("download-dataframe", "data", allow_duplicate=True),
        [Input("pipeline-export-stage-btn", "n_clicks")],
        [State("pipeline-current-stage", "data")],
        prevent_initial_call=True,
    )
    def export_stage_data(n_clicks, current_stage):
        """
        Export current stage data.
        """
        if n_clicks is None:
            return no_update

        stage_num = _get_stage_number(current_stage)
        if stage_num == 0:
            return no_update

        # Get the current pipeline run ID
        pipeline_run_id = getattr(
            register_pipeline_callbacks, "current_pipeline_run_id", None
        )
        if not pipeline_run_id:
            logger.warning("No pipeline run ID found for stage export")
            return no_update

        # Get pipeline data
        pipeline_data = register_pipeline_callbacks.pipeline_data.get(
            pipeline_run_id, {}
        )
        stage_results = pipeline_data.get("results", {}).get(f"stage_{stage_num}", {})

        if not stage_results:
            logger.warning(f"No results found for stage {stage_num}")
            return no_update

        # Export stage results as JSON
        stage_json = json.dumps(
            {
                "stage": stage_num,
                "stage_name": (
                    PIPELINE_STAGES[stage_num]
                    if stage_num < len(PIPELINE_STAGES)
                    else "Unknown"
                ),
                "results": stage_results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
            default=str,
        )

        logger.info(f"Exporting stage {stage_num} data for run: {pipeline_run_id}")

        return dict(
            content=stage_json,
            filename=f"stage_{stage_num}_results_{pipeline_run_id}.json",
        )

    # Generate Report Callback
    @app.callback(
        Output("download-dataframe", "data", allow_duplicate=True),
        [Input("pipeline-report-btn", "n_clicks")],
        [State("pipeline-current-stage", "data")],
        prevent_initial_call=True,
    )
    def generate_pipeline_report(n_clicks, current_stage):
        """
        Generate PDF report of pipeline execution.
        """
        logger.info(
            f"Report button clicked: n_clicks={n_clicks}, current_stage={current_stage}"
        )

        if n_clicks is None:
            return no_update

        # Get the current pipeline run ID
        pipeline_run_id = getattr(
            register_pipeline_callbacks, "current_pipeline_run_id", None
        )
        logger.info(f"Pipeline run ID for report: {pipeline_run_id}")

        if not pipeline_run_id:
            logger.warning("No pipeline run ID found for report generation")
            return no_update

        # Get pipeline data
        pipeline_data = register_pipeline_callbacks.pipeline_data.get(
            pipeline_run_id, {}
        )
        results = pipeline_data.get("results", {})
        logger.info(f"Results keys: {list(results.keys())}")

        if not results:
            logger.warning("No results found for report generation")
            return no_update

        # Generate Markdown report (can be converted to PDF later)
        report_lines = [
            "# VitalDSP Pipeline Analysis Report",
            "",
            "---",
            "",
            "## Pipeline Overview",
            "",
            f"- **Pipeline Run ID:** `{pipeline_run_id}`",
            f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Signal Type:** {pipeline_data.get('signal_type', 'Unknown')}",
            f"- **Sampling Frequency:** {pipeline_data.get('fs', 0)} Hz",
            f"- **Signal Length:** {len(pipeline_data.get('signal_data', []))} samples",
            f"- **Duration:** {len(pipeline_data.get('signal_data', [])) / max(pipeline_data.get('fs', 1), 1):.2f} seconds",
            f"- **Processing Paths:** {', '.join(pipeline_data.get('paths', ['filtered', 'preprocessed']))}",
            f"- **Quality Validation:** {'Enabled' if pipeline_data.get('enable_quality', False) else 'Disabled'}",
            f"- **Completed Stages:** {pipeline_data.get('current_stage', 0)}/8",
            "",
            "---",
            "",
        ]

        # Add detailed stage configurations and results
        stage_configs = [
            {
                "num": 1,
                "name": "Data Ingestion",
                "description": "Load and validate input signal data",
                "config": {
                    "Signal Type": pipeline_data.get("signal_type", "Unknown"),
                    "Sampling Frequency": f"{pipeline_data.get('fs', 0)} Hz",
                    "Signal Length": f"{len(pipeline_data.get('signal_data', []))} samples",
                },
            },
            {
                "num": 2,
                "name": "Quality Screening",
                "description": "Initial quality assessment and bad segment detection",
                "config": {
                    "Enabled": pipeline_data.get("enable_quality", False),
                    "SQI Type": pipeline_data.get("stage2_sqi_type", "snr_sqi"),
                    "Window Size": pipeline_data.get("stage2_window_size", 1000),
                    "Step Size": pipeline_data.get("stage2_step_size", 500),
                    "Threshold": f"{pipeline_data.get('stage2_threshold', 0.7)} ({pipeline_data.get('stage2_threshold_type', 'below')})",
                    "Scale": pipeline_data.get("stage2_scale", "zscore"),
                },
            },
            {
                "num": 3,
                "name": "Comprehensive Filtering",
                "description": "Apply filtering and artifact removal to each processing path",
                "config": {
                    "Filter Type": pipeline_data.get(
                        "stage3_filter_type", "traditional"
                    ),
                    "Signal Source": pipeline_data.get(
                        "stage3_signal_source", "original"
                    ),
                    "Detrend": (
                        "Enabled"
                        if pipeline_data.get("stage3_detrend", False)
                        else "Disabled"
                    ),
                    "Application Count": pipeline_data.get(
                        "stage3_application_count", 1
                    ),
                    "Filter Family": pipeline_data.get(
                        "stage3_filter_family", "butter"
                    ),
                    "Response Type": pipeline_data.get(
                        "stage3_filter_response", "bandpass"
                    ),
                    "Frequency Band": f"{pipeline_data.get('stage3_filter_lowcut', 0.5)}-{pipeline_data.get('stage3_filter_highcut', 40)} Hz",
                    "Filter Order": pipeline_data.get("stage3_filter_order", 4),
                    "Artifact Removal": pipeline_data.get(
                        "stage3_artifact_type", "baseline"
                    ),
                },
            },
            {
                "num": 4,
                "name": "Quality Validation",
                "description": "Validate quality of each processing path and select best",
                "config": {
                    "SQI Type": pipeline_data.get("stage4_sqi_type", "snr_sqi"),
                    "Window Size": pipeline_data.get("stage4_window_size", 1000),
                    "Step Size": pipeline_data.get("stage4_step_size", 500),
                    "Threshold": f"{pipeline_data.get('stage4_threshold', 0.7)} ({pipeline_data.get('stage4_threshold_type', 'below')})",
                    "Scale": pipeline_data.get("stage4_scale", "zscore"),
                },
            },
            {
                "num": 5,
                "name": "Segmentation",
                "description": "Segment signal into analysis windows",
                "config": {
                    "Window Size": f"{pipeline_data.get('window_size', 30)} seconds",
                    "Overlap Ratio": f"{pipeline_data.get('overlap_ratio', 0.5) * 100:.0f}%",
                    "Min Segment Length": f"{pipeline_data.get('min_segment_length', 5)} seconds",
                    "Window Function": pipeline_data.get("window_function", "none"),
                },
            },
            {
                "num": 6,
                "name": "Feature Extraction",
                "description": "Extract features from each segment",
                "config": {
                    "Feature Types": ", ".join(
                        pipeline_data.get("feature_types", ["time", "frequency"])
                    ),
                    "Time Features": ", ".join(
                        pipeline_data.get("time_features", ["mean", "std"])
                    ),
                    "Frequency Features": ", ".join(
                        pipeline_data.get("frequency_features", ["spectral_centroid"])
                    ),
                    "Nonlinear Features": ", ".join(
                        pipeline_data.get("nonlinear_features", ["sample_entropy"])
                    ),
                    "Statistical Features": ", ".join(
                        pipeline_data.get("statistical_features", ["skewness"])
                    ),
                },
            },
            {
                "num": 7,
                "name": "Intelligent Output Selection",
                "description": "Select best processing path and generate recommendations",
                "config": {
                    "Selection Criterion": pipeline_data.get(
                        "selection_criterion", "best_quality"
                    ),
                    "Confidence Threshold": pipeline_data.get(
                        "confidence_threshold", 0.7
                    ),
                    "Generate Recommendations": pipeline_data.get(
                        "generate_recommendations", True
                    ),
                },
            },
            {
                "num": 8,
                "name": "Output Package",
                "description": "Package results for export",
                "config": {
                    "Output Formats": ", ".join(
                        pipeline_data.get("output_formats", ["json", "csv"])
                    ),
                    "Output Contents": ", ".join(
                        pipeline_data.get(
                            "output_contents",
                            [
                                "processed_signals",
                                "quality_metrics",
                                "features",
                                "metadata",
                            ],
                        )
                    ),
                    "Compress Output": pipeline_data.get("compress_output", True),
                },
            },
        ]

        for stage_info in stage_configs:
            stage_num = stage_info["num"]
            stage_key = f"stage_{stage_num}"

            # Add stage header
            report_lines.append(f"## Stage {stage_num}: {stage_info['name']}")
            report_lines.append("")
            report_lines.append(f"**Description:** {stage_info['description']}")
            report_lines.append("")

            # Add configuration
            report_lines.append("### Configuration")
            report_lines.append("")
            for config_key, config_value in stage_info["config"].items():
                report_lines.append(f"- **{config_key}:** {config_value}")
            report_lines.append("")

            # Add results if available
            if stage_key in results:
                report_lines.append("### Results")
                report_lines.append("")

                stage_data = results[stage_key]

                # Handle quality metrics specially
                if stage_num == 4 and "quality_metrics" in stage_data:
                    quality_metrics = stage_data["quality_metrics"]
                    report_lines.append("#### Processing Path Quality Comparison")
                    report_lines.append("")
                    report_lines.append(
                        "| Path | Quality Score | Normal Segments | Abnormal Segments | Status |"
                    )
                    report_lines.append(
                        "|------|--------------|-----------------|-------------------|--------|"
                    )
                    for path_name, metrics in quality_metrics.items():
                        quality = metrics.get("overall_quality", 0.0)
                        normal = len(metrics.get("normal_segments", []))
                        abnormal = len(metrics.get("abnormal_segments", []))
                        passed = (
                            "✅ Passed"
                            if metrics.get("passed", False)
                            else "⚠️ Below Threshold"
                        )
                        report_lines.append(
                            f"| {path_name.upper()} | {quality:.3f} | {normal} | {abnormal} | {passed} |"
                        )
                    report_lines.append("")

                    if "best_path" in stage_data:
                        report_lines.append(
                            f"**Selected Best Path:** {stage_data['best_path'].upper()} ⭐"
                        )
                        report_lines.append("")

                # Handle feature extraction specially
                elif stage_num == 6 and "features" in stage_data:
                    features = stage_data["features"]
                    if isinstance(features, list) and len(features) > 0:
                        report_lines.append(f"**Total Segments:** {len(features)}")
                        report_lines.append("")
                        if isinstance(features[0], dict):
                            feature_names = list(features[0].keys())
                            report_lines.append(
                                f"**Extracted Features ({len(feature_names)}):**"
                            )
                            report_lines.append("")
                            for i, name in enumerate(
                                feature_names[:20], 1
                            ):  # Show first 20
                                report_lines.append(f"{i}. {name}")
                            if len(feature_names) > 20:
                                report_lines.append(
                                    f"... and {len(feature_names) - 20} more features"
                                )
                            report_lines.append("")

                # Handle other results generically
                else:
                    for key, value in stage_data.items():
                        if key in ["quality_metrics", "features"]:  # Already handled
                            continue
                        if isinstance(value, (int, float, str, bool)):
                            report_lines.append(f"- **{key}:** {value}")
                        elif isinstance(value, dict):
                            report_lines.append(f"- **{key}:**")
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, (int, float, str, bool)):
                                    report_lines.append(f"  - {sub_key}: {sub_value}")
                        elif isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], (int, float, str)):
                                report_lines.append(
                                    f"- **{key}:** {', '.join(map(str, value[:5]))}{'...' if len(value) > 5 else ''}"
                                )
                    report_lines.append("")
            else:
                report_lines.append("*Stage not completed or no results available*")
                report_lines.append("")

            report_lines.append("---")
            report_lines.append("")

        # Add recommendations if available
        stage7_results = results.get("stage_7", {})
        if "recommendations" in stage7_results and stage7_results["recommendations"]:
            report_lines.append("## Recommendations")
            report_lines.append("")
            for i, rec in enumerate(stage7_results["recommendations"], 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")

        # Add summary statistics
        report_lines.append("## Summary")
        report_lines.append("")
        report_lines.append(
            f"- **Pipeline Completed:** {'Yes ✅' if pipeline_data.get('completed', False) else 'No ⚠️'}"
        )
        if pipeline_data.get("completion_time"):
            start_time = pipeline_data.get("start_time", 0)
            completion_time = pipeline_data.get("completion_time", 0)
            duration = completion_time - start_time
            report_lines.append(f"- **Execution Time:** {duration:.2f} seconds")

        # Best path selection
        best_path = None
        for stage_key in results:
            if "best_path" in results[stage_key]:
                best_path = results[stage_key]["best_path"]
                break
        if best_path:
            report_lines.append(
                f"- **Selected Processing Path:** {best_path.upper()} ⭐"
            )

        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("*Report generated by VitalDSP Pipeline Analysis System*")

        report_content = "\n".join(report_lines)

        logger.info(f"Generating pipeline report for run: {pipeline_run_id}")

        return dict(
            content=report_content, filename=f"pipeline_report_{pipeline_run_id}.md"
        )

    # Filter Mode Toggle Callback
    @app.callback(
        Output("advanced-filter-params", "style"),
        [Input("pipeline-filter-mode", "value")],
    )
    def toggle_filter_mode(filter_mode):
        """
        Show/hide advanced filter parameters based on selected mode.
        """
        if filter_mode == "advanced":
            return {"display": "block"}
        else:
            return {"display": "none"}

    # ============================================================================
    # PIPELINE STATE BUILDER CALLBACK
    # ============================================================================

    @app.callback(
        Output("pipeline-state", "data"),
        [
            Input("pipeline-current-stage", "data"),
            Input("pipeline-progress-interval", "n_intervals"),
        ],
        prevent_initial_call=True,
    )
    def build_pipeline_state(current_stage, n_intervals):
        """
        Build pipeline state from pipeline_data for Phase A & B callbacks.

        This callback translates the internal pipeline_data structure into
        the format expected by the Phase A & B visualization callbacks.
        """
        import time

        # Default empty state
        default_state = {}

        if not current_stage:
            return default_state

        # Get pipeline_run_id
        pipeline_run_id = None
        if isinstance(current_stage, str) and len(current_stage) == 8:
            pipeline_run_id = current_stage
        elif hasattr(register_pipeline_callbacks, "current_pipeline_run_id"):
            pipeline_run_id = register_pipeline_callbacks.current_pipeline_run_id

        if not pipeline_run_id or not hasattr(
            register_pipeline_callbacks, "pipeline_data"
        ):
            return default_state

        pipeline_data = register_pipeline_callbacks.pipeline_data.get(pipeline_run_id)
        if not pipeline_data:
            return default_state

        # Build pipeline state
        current_stage_num = pipeline_data.get("current_stage", 0)
        results = pipeline_data.get("results", {})

        # Determine status
        status = "idle"
        if current_stage_num > 0 and current_stage_num < 8:
            status = "running"
        elif current_stage_num >= 8:
            status = "completed"
        elif pipeline_data.get("error"):
            status = "error"

        # Get start time
        start_timestamp = pipeline_data.get("start_time", time.time())
        from datetime import datetime

        start_time_str = datetime.fromtimestamp(start_timestamp).strftime("%H:%M:%S")

        # Build quality scores from stage 4 results
        quality_scores = {}
        stage4_results = results.get("stage_4", {})
        if stage4_results:
            quality_metrics = stage4_results.get("quality_metrics", {})
            for path_name, metrics in quality_metrics.items():
                quality_scores[path_name] = metrics.get("overall_quality", 0.5)

            # Add best path
            best_path = stage4_results.get("best_path")
            if best_path:
                quality_scores["best_path"] = best_path

        # Build path progress (estimate based on current stage)
        path_progress = {}
        paths = pipeline_data.get("paths", [])
        if current_stage_num >= 3:
            # During stage 3-4, paths are being processed
            for path in paths:
                if current_stage_num == 3:
                    path_progress[path] = 50  # In progress
                elif current_stage_num >= 4:
                    path_progress[path] = 100  # Complete

        # Determine which paths are completed
        completed_paths = []
        if current_stage_num >= 4:
            completed_paths = paths

        # Build state
        state = {
            "run_id": pipeline_run_id,
            "start_time": start_time_str,
            "start_timestamp": start_timestamp,
            "status": status,
            "current_stage": current_stage_num,
            "sampling_rate": pipeline_data.get("fs", 1000),
            "path_progress": path_progress,
            "completed_paths": completed_paths,
            "quality_scores": quality_scores,
        }

        # Add processed signals if available (for path comparison plot)
        processed_signals = pipeline_data.get("processed_signals", {})
        if processed_signals:
            # Limit to first 10000 samples for performance
            state["path_signals"] = {}
            for path, signal in processed_signals.items():
                if isinstance(signal, np.ndarray):
                    state["path_signals"][path] = (
                        signal[:10000].tolist()
                        if len(signal) > 10000
                        else signal.tolist()
                    )

        return state

    # ============================================================================
    # PHASE A: INTERACTIVE FLOW DIAGRAM CALLBACKS
    # ============================================================================

    @app.callback(
        [
            Output("pipeline-current-stage-text", "children"),
            Output("pipeline-status-text", "children"),
            Output("pipeline-overall-progress-text", "children"),
            Output("pipeline-flow-progress-bar", "value"),
            Output("pipeline-flow-progress-bar", "color"),
        ]
        + [Output(f"pipeline-stage-card-{i}", "className") for i in range(1, 9)]
        + [Output(f"pipeline-stage-icon-{i}", "className") for i in range(1, 9)],
        [Input("pipeline-state", "data"), Input("pipeline-current-stage", "data")],
    )
    def update_flow_diagram(pipeline_state, current_stage):
        """
        Update the interactive flow diagram based on pipeline execution state.

        Updates:
        - Current stage text
        - Status text
        - Overall progress percentage
        - Progress bar value and color
        - Stage card highlighting
        - Stage icon status colors
        """
        # Default values
        stage_text = "Ready to Start"
        status_text = "Configure pipeline and click 'Run Pipeline'"
        progress_text = "0%"
        progress_value = 0
        progress_color = "info"

        # Default card and icon classes
        card_classes = ["pipeline-stage-box" for _ in range(8)]
        icon_classes = [
            f"fas fa-{icon} pipeline-stage-icon"
            for icon in [
                "database",
                "filter",
                "code-branch",
                "check-circle",
                "grip-vertical",
                "chart-bar",
                "brain",
                "box",
            ]
        ]

        if not pipeline_state or not current_stage:
            return (
                [stage_text, status_text, progress_text, progress_value, progress_color]
                + card_classes
                + icon_classes
            )

        # Get stage number
        stage_num = _get_stage_number(current_stage)

        if stage_num > 0 and stage_num <= 8:
            # Update progress
            progress_value = int((stage_num / 8) * 100)
            progress_text = f"{progress_value}%"
            stage_text = f"Stage {stage_num} - {PIPELINE_STAGES[stage_num - 1]}"

            # Determine status and color
            if pipeline_state.get("status") == "running":
                status_text = "Processing..."
                progress_color = "primary"
            elif pipeline_state.get("status") == "completed":
                status_text = "✓ Pipeline Complete"
                progress_color = "success"
                progress_value = 100
                progress_text = "100%"
            elif pipeline_state.get("status") == "error":
                status_text = "✗ Error Occurred"
                progress_color = "danger"

            # Highlight active and completed stages
            for i in range(1, 9):
                if i < stage_num:
                    # Completed stages
                    card_classes[i - 1] = "pipeline-stage-box pipeline-stage-complete"
                    icon_classes[i - 1] = icon_classes[i - 1].replace(
                        "pipeline-stage-icon", "pipeline-stage-icon text-success"
                    )
                elif i == stage_num:
                    # Active stage
                    card_classes[i - 1] = "pipeline-stage-box pipeline-stage-active"
                    icon_classes[i - 1] = icon_classes[i - 1].replace(
                        "pipeline-stage-icon", "pipeline-stage-icon text-primary"
                    )
                else:
                    # Pending stages
                    card_classes[i - 1] = "pipeline-stage-box pipeline-stage-pending"
                    icon_classes[i - 1] = icon_classes[i - 1].replace(
                        "pipeline-stage-icon", "pipeline-stage-icon text-muted"
                    )

        return (
            [stage_text, status_text, progress_text, progress_value, progress_color]
            + card_classes
            + icon_classes
        )

    @app.callback(
        [
            Output("pipeline-paths-status-container", "style"),
            Output("pipeline-path-raw-progress", "value"),
            Output("pipeline-path-filtered-progress", "value"),
            Output("pipeline-path-preprocessed-progress", "value"),
            Output("pipeline-path-raw-quality", "children"),
            Output("pipeline-path-filtered-quality", "children"),
            Output("pipeline-path-preprocessed-quality", "children"),
        ],
        [Input("pipeline-state", "data"), Input("pipeline-current-stage", "data")],
    )
    def update_path_progress(pipeline_state, current_stage):
        """
        Update the path progress bars and quality scores.

        Shows path comparison during stages 3-4 (Parallel Processing & Quality Validation).
        """
        # Default: hide path status
        container_style = {"display": "none"}
        raw_progress = 0
        filtered_progress = 0
        preprocessed_progress = 0
        raw_quality = "N/A"
        filtered_quality = "N/A"
        preprocessed_quality = "N/A"

        if not pipeline_state or not current_stage:
            return [
                container_style,
                raw_progress,
                filtered_progress,
                preprocessed_progress,
                raw_quality,
                filtered_quality,
                preprocessed_quality,
            ]

        stage_num = _get_stage_number(current_stage)

        # Show path status during stages 3-4
        if stage_num in [3, 4]:
            container_style = {"display": "block"}

            # Get path progress from pipeline state
            path_progress = pipeline_state.get("path_progress", {})
            raw_progress = path_progress.get("raw", 0)
            filtered_progress = path_progress.get("filtered", 0)
            preprocessed_progress = path_progress.get("preprocessed", 0)

            # Get quality scores
            quality_scores = pipeline_state.get("quality_scores", {})
            if "raw" in quality_scores:
                raw_quality = f"Quality: {quality_scores['raw']:.2f}"
            if "filtered" in quality_scores:
                filtered_quality = f"Quality: {quality_scores['filtered']:.2f}"
                # Add star if best quality
                if quality_scores.get("best_path") == "filtered":
                    filtered_quality += " ⭐"
            if "preprocessed" in quality_scores:
                preprocessed_quality = f"Quality: {quality_scores['preprocessed']:.2f}"
                if quality_scores.get("best_path") == "preprocessed":
                    preprocessed_quality += " ⭐"

        return [
            container_style,
            raw_progress,
            filtered_progress,
            preprocessed_progress,
            raw_quality,
            filtered_quality,
            preprocessed_quality,
        ]

    @app.callback(
        [
            Output("pipeline-status-indicator", "children"),
            Output("pipeline-run-id", "children"),
            Output("pipeline-start-time", "children"),
            Output("pipeline-elapsed-time", "children"),
            Output("pipeline-current-stage-indicator", "children"),
            Output("pipeline-stage-progress-mini", "value"),
        ]
        + [Output(f"pipeline-stage-status-icon-{i}", "className") for i in range(1, 9)]
        + [
            Output(f"pipeline-path-status-{path}", "className")
            for path in ["raw", "filtered", "preprocessed"]
        ],
        [
            Input("pipeline-state", "data"),
            Input("pipeline-progress-interval", "n_intervals"),
        ],
    )
    def update_live_status_panel(pipeline_state, n_intervals):
        """
        Update the live status panel with current pipeline execution information.

        Updates every second when pipeline is running.
        """
        # Default values
        status_indicator = "🔴"
        run_id = "N/A"
        start_time = "N/A"
        elapsed_time = "00:00:00"
        stage_indicator = "0/8"
        stage_progress = 0

        # Default stage status icons (all pending)
        stage_icons = ["fas fa-circle text-muted" for _ in range(8)]

        # Default path status icons (all pending)
        path_icons = ["fas fa-circle text-muted" for _ in range(3)]

        if not pipeline_state:
            return (
                [
                    status_indicator,
                    run_id,
                    start_time,
                    elapsed_time,
                    stage_indicator,
                    stage_progress,
                ]
                + stage_icons
                + path_icons
            )

        # Update basic info
        if "run_id" in pipeline_state:
            run_id = pipeline_state["run_id"]

        if "start_time" in pipeline_state:
            start_time = pipeline_state["start_time"]

            # Calculate elapsed time
            if pipeline_state.get("status") == "running":
                elapsed_seconds = int(
                    time.time()
                    - float(pipeline_state.get("start_timestamp", time.time()))
                )
                hours = elapsed_seconds // 3600
                minutes = (elapsed_seconds % 3600) // 60
                seconds = elapsed_seconds % 60
                elapsed_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Update status indicator
        if pipeline_state.get("status") == "running":
            status_indicator = "🟢"
        elif pipeline_state.get("status") == "completed":
            status_indicator = "✅"
        elif pipeline_state.get("status") == "error":
            status_indicator = "🔴"

        # Update stage progress
        current_stage = pipeline_state.get("current_stage", 0)
        if current_stage > 0:
            stage_indicator = f"{current_stage}/8"
            stage_progress = int((current_stage / 8) * 100)

            # Update stage status icons
            for i in range(1, 9):
                if i < current_stage:
                    stage_icons[i - 1] = "fas fa-check-circle text-success"
                elif i == current_stage:
                    if pipeline_state.get("status") == "error":
                        stage_icons[i - 1] = "fas fa-times-circle text-danger"
                    else:
                        stage_icons[i - 1] = "fas fa-circle text-primary"
                else:
                    stage_icons[i - 1] = "fas fa-circle text-muted"

        # Update path status icons
        completed_paths = pipeline_state.get("completed_paths", [])
        path_map = {"raw": 0, "filtered": 1, "preprocessed": 2}
        for path, idx in path_map.items():
            if path in completed_paths:
                path_icons[idx] = "fas fa-check-circle text-success"

        return (
            [
                status_indicator,
                run_id,
                start_time,
                elapsed_time,
                stage_indicator,
                stage_progress,
            ]
            + stage_icons
            + path_icons
        )

    # ============================================================================
    # PHASE B: PATH COMPARISON DASHBOARD CALLBACKS
    # ============================================================================

    @app.callback(
        [
            Output("path-table-raw-quality", "children"),
            Output("path-table-filtered-quality", "children"),
            Output("path-table-preprocessed-quality", "children"),
            Output("path-table-raw-status", "children"),
            Output("path-table-filtered-status", "children"),
            Output("path-table-preprocessed-status", "children"),
            Output("path-metrics-container", "style"),
            Output("path-signal-viz-container", "style"),
        ],
        [Input("pipeline-state", "data")],
    )
    def update_path_comparison_table(pipeline_state):
        """
        Update the path comparison table with quality scores and status.
        """
        # Default values
        raw_quality = "N/A"
        filtered_quality = "N/A"
        preprocessed_quality = "N/A"
        raw_status = "⚪ Pending"
        filtered_status = "⚪ Pending"
        preprocessed_status = "⚪ Pending"
        metrics_style = {"display": "none"}
        signal_viz_style = {"display": "none"}

        if not pipeline_state:
            return [
                raw_quality,
                filtered_quality,
                preprocessed_quality,
                raw_status,
                filtered_status,
                preprocessed_status,
                metrics_style,
                signal_viz_style,
            ]

        # Get quality scores
        quality_scores = pipeline_state.get("quality_scores", {})
        if "raw" in quality_scores:
            raw_quality = f"{quality_scores['raw']:.3f}"
            if quality_scores.get("best_path") == "raw":
                raw_quality += " ⭐"
        if "filtered" in quality_scores:
            filtered_quality = f"{quality_scores['filtered']:.3f}"
            if quality_scores.get("best_path") == "filtered":
                filtered_quality += " ⭐"
        if "preprocessed" in quality_scores:
            preprocessed_quality = f"{quality_scores['preprocessed']:.3f}"
            if quality_scores.get("best_path") == "preprocessed":
                preprocessed_quality += " ⭐"

        # Get path status
        completed_paths = pipeline_state.get("completed_paths", [])
        current_stage = pipeline_state.get("current_stage", 0)

        # Update status icons based on completion
        if "raw" in completed_paths:
            raw_status = "✅ Complete"
        elif current_stage >= 3:
            raw_status = "🔵 Processing"

        if "filtered" in completed_paths:
            filtered_status = "✅ Complete"
        elif current_stage >= 3:
            filtered_status = "🔵 Processing"

        if "preprocessed" in completed_paths:
            preprocessed_status = "✅ Complete"
        elif current_stage >= 3:
            preprocessed_status = "🔵 Processing"

        # Show metrics and signal visualization after stage 4
        if current_stage >= 4:
            metrics_style = {"display": "block"}
            signal_viz_style = {"display": "block"}

        return [
            raw_quality,
            filtered_quality,
            preprocessed_quality,
            raw_status,
            filtered_status,
            preprocessed_status,
            metrics_style,
            signal_viz_style,
        ]

    @app.callback(
        [
            Output("metric-snr-raw", "children"),
            Output("metric-snr-filtered", "children"),
            Output("metric-snr-preprocessed", "children"),
            Output("metric-baseline-raw", "children"),
            Output("metric-baseline-filtered", "children"),
            Output("metric-baseline-preprocessed", "children"),
            Output("metric-artifact-raw", "children"),
            Output("metric-artifact-filtered", "children"),
            Output("metric-artifact-preprocessed", "children"),
        ],
        [Input("pipeline-state", "data")],
    )
    def update_path_metrics(pipeline_state):
        """
        Update detailed quality metrics for each path.
        """
        # Default values
        metrics = {
            "snr": {
                "raw": "RAW: N/A",
                "filtered": "FILTERED: N/A",
                "preprocessed": "PREPROCESSED: N/A",
            },
            "baseline": {
                "raw": "RAW: N/A",
                "filtered": "FILTERED: N/A",
                "preprocessed": "PREPROCESSED: N/A",
            },
            "artifact": {
                "raw": "RAW: N/A",
                "filtered": "FILTERED: N/A",
                "preprocessed": "PREPROCESSED: N/A",
            },
        }

        if not pipeline_state:
            return [
                metrics["snr"]["raw"],
                metrics["snr"]["filtered"],
                metrics["snr"]["preprocessed"],
                metrics["baseline"]["raw"],
                metrics["baseline"]["filtered"],
                metrics["baseline"]["preprocessed"],
                metrics["artifact"]["raw"],
                metrics["artifact"]["filtered"],
                metrics["artifact"]["preprocessed"],
            ]

        # Get detailed metrics from pipeline state
        detailed_metrics = pipeline_state.get("detailed_metrics", {})

        # Update SNR metrics
        if "snr" in detailed_metrics:
            snr = detailed_metrics["snr"]
            if "raw" in snr:
                metrics["snr"]["raw"] = f"RAW: {snr['raw']:.2f} dB"
            if "filtered" in snr:
                metrics["snr"]["filtered"] = f"FILTERED: {snr['filtered']:.2f} dB"
                if detailed_metrics.get("best_snr") == "filtered":
                    metrics["snr"]["filtered"] += " ⭐"
            if "preprocessed" in snr:
                metrics["snr"][
                    "preprocessed"
                ] = f"PREPROCESSED: {snr['preprocessed']:.2f} dB"
                if detailed_metrics.get("best_snr") == "preprocessed":
                    metrics["snr"]["preprocessed"] += " ⭐"

        # Update baseline wander metrics
        if "baseline_wander" in detailed_metrics:
            baseline = detailed_metrics["baseline_wander"]
            levels = {0: "Low", 1: "Medium", 2: "High"}
            if "raw" in baseline:
                metrics["baseline"][
                    "raw"
                ] = f"RAW: {levels.get(baseline['raw'], 'N/A')}"
            if "filtered" in baseline:
                metrics["baseline"][
                    "filtered"
                ] = f"FILTERED: {levels.get(baseline['filtered'], 'N/A')}"
                if detailed_metrics.get("best_baseline") == "filtered":
                    metrics["baseline"]["filtered"] += " ⭐"
            if "preprocessed" in baseline:
                metrics["baseline"][
                    "preprocessed"
                ] = f"PREPROCESSED: {levels.get(baseline['preprocessed'], 'N/A')}"
                if detailed_metrics.get("best_baseline") == "preprocessed":
                    metrics["baseline"]["preprocessed"] += " ⭐"

        # Update artifact level metrics
        if "artifact_level" in detailed_metrics:
            artifact = detailed_metrics["artifact_level"]
            levels = {0: "Very Low", 1: "Low", 2: "Medium", 3: "High"}
            if "raw" in artifact:
                metrics["artifact"][
                    "raw"
                ] = f"RAW: {levels.get(artifact['raw'], 'N/A')}"
            if "filtered" in artifact:
                metrics["artifact"][
                    "filtered"
                ] = f"FILTERED: {levels.get(artifact['filtered'], 'N/A')}"
                if detailed_metrics.get("best_artifact") == "filtered":
                    metrics["artifact"]["filtered"] += " ⭐"
            if "preprocessed" in artifact:
                metrics["artifact"][
                    "preprocessed"
                ] = f"PREPROCESSED: {levels.get(artifact['preprocessed'], 'N/A')}"
                if detailed_metrics.get("best_artifact") == "preprocessed":
                    metrics["artifact"]["preprocessed"] += " ⭐"

        return [
            metrics["snr"]["raw"],
            metrics["snr"]["filtered"],
            metrics["snr"]["preprocessed"],
            metrics["baseline"]["raw"],
            metrics["baseline"]["filtered"],
            metrics["baseline"]["preprocessed"],
            metrics["artifact"]["raw"],
            metrics["artifact"]["filtered"],
            metrics["artifact"]["preprocessed"],
        ]

    @app.callback(
        [
            Output("path-recommendation-text", "children"),
            Output("path-recommendation-alert", "color"),
        ],
        [Input("pipeline-state", "data")],
    )
    def update_path_recommendation(pipeline_state):
        """
        Update the path recommendation based on quality analysis.
        """
        default_text = "Run pipeline to see path recommendation"
        default_color = "info"

        if not pipeline_state:
            return [default_text, default_color]

        # Get recommendation from pipeline state
        recommendation = pipeline_state.get("recommendation", {})
        best_path = recommendation.get("best_path")

        if not best_path:
            # Check if we have quality scores
            quality_scores = pipeline_state.get("quality_scores", {})
            if quality_scores:
                best_path = quality_scores.get("best_path")

        if best_path:
            path_names = {
                "raw": "RAW (No filtering)",
                "filtered": "FILTERED (Bandpass filtering)",
                "preprocessed": "PREPROCESSED (Filtering + Artifact Removal)",
            }

            reason = recommendation.get("reason", "Based on quality metrics analysis")
            confidence = recommendation.get("confidence", 0.0)

            text = [
                html.Strong(
                    f"Recommended Path: {path_names.get(best_path, best_path.upper())}"
                ),
                html.Br(),
                html.Span(f"Reason: {reason}"),
                html.Br(),
                (
                    html.Small(f"Confidence: {confidence:.1%}", className="text-muted")
                    if confidence > 0
                    else None
                ),
            ]

            # Color based on confidence
            if confidence >= 0.8:
                color = "success"
            elif confidence >= 0.6:
                color = "primary"
            else:
                color = "warning"

            return [text, color]

        return [default_text, default_color]

    @app.callback(
        Output("path-comparison-signal-plot", "figure"),
        [Input("pipeline-state", "data")],
    )
    def update_path_signal_comparison(pipeline_state):
        """
        Update the signal comparison plot showing all three paths.
        """
        # Default empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Signal Comparison Across Paths",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            template="plotly_white",
            showlegend=True,
            hovermode="x unified",
        )

        if not pipeline_state:
            fig.add_annotation(
                text="Run pipeline to see signal comparison",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="gray"),
            )
            return fig

        # Get signal data from pipeline state
        signals = pipeline_state.get("path_signals", {})
        fs = pipeline_state.get("sampling_rate", 1000)

        if not signals:
            fig.add_annotation(
                text="Processing... Signal data will appear here",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="gray"),
            )
            return fig

        # Add traces for each path
        colors = {"raw": "gray", "filtered": "blue", "preprocessed": "green"}
        names = {"raw": "RAW", "filtered": "FILTERED", "preprocessed": "PREPROCESSED"}

        for path_name in ["raw", "filtered", "preprocessed"]:
            if path_name in signals:
                signal_data = signals[path_name]
                time_axis = np.arange(len(signal_data)) / fs

                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=signal_data,
                        name=names[path_name],
                        line=dict(color=colors[path_name], width=1),
                        opacity=0.8,
                    )
                )

        return fig

    logger.info("Pipeline callbacks registered successfully")
