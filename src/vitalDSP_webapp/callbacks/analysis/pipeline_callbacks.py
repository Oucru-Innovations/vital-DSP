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
            # Stage 2: Quality Screening - Use SignalQualityIndex individual methods
            from vitalDSP.signal_quality_assessment.signal_quality_index import (
                SignalQualityIndex,
            )

            # Initialize SignalQualityIndex
            sqi = SignalQualityIndex(signal_data)

            # Get parameters from pipeline_data
            sqi_methods = pipeline_data.get(
                "sqi_methods", ["amplitude_variability", "baseline_wander", "snr"]
            )
            sqi_window_seconds = pipeline_data.get("sqi_window", 5)
            sqi_step_seconds = pipeline_data.get("sqi_step", 2.5)
            quality_threshold = pipeline_data.get("quality_threshold", 0.7)
            sqi_scale = pipeline_data.get("sqi_scale", "zscore")

            # Define window parameters for segment-based SQIs
            window_size = int(sqi_window_seconds * fs)
            step_size = int(sqi_step_seconds * fs)

            # Compute selected SQI metrics
            quality_results = {}
            sqi_arrays = {}  # Store full arrays for visualization

            # Map of SQI method names to their functions
            sqi_function_map = {
                "amplitude_variability": lambda: sqi.amplitude_variability_sqi(
                    window_size, step_size, threshold=quality_threshold, scale=sqi_scale
                ),
                "baseline_wander": lambda: sqi.baseline_wander_sqi(
                    window_size, step_size, threshold=quality_threshold, scale=sqi_scale
                ),
                "snr": lambda: sqi.snr_sqi(
                    window_size, step_size, threshold=quality_threshold, scale=sqi_scale
                ),
                "zero_crossing": lambda: sqi.zero_crossing_consistency_sqi(
                    window_size, step_size, threshold=quality_threshold, scale=sqi_scale
                ),
                "entropy": lambda: sqi.entropy_sqi(
                    window_size, step_size, threshold=quality_threshold, scale=sqi_scale
                ),
                "kurtosis": lambda: sqi.kurtosis_sqi(
                    window_size, step_size, threshold=quality_threshold, scale=sqi_scale
                ),
                "skewness": lambda: sqi.skewness_sqi(
                    window_size, step_size, threshold=quality_threshold, scale=sqi_scale
                ),
            }

            # Compute selected SQI methods
            for method in sqi_methods:
                if method in sqi_function_map:
                    try:
                        sqi_array, normal_segs, abnormal_segs = sqi_function_map[
                            method
                        ]()
                        quality_results[f"{method}_sqi"] = float(np.mean(sqi_array))
                        sqi_arrays[f"{method}_sqi"] = (
                            sqi_array.tolist()
                        )  # Store for visualization
                        logger.info(
                            f"✓ {method} SQI: {quality_results[f'{method}_sqi']:.3f} (normal: {len(normal_segs)}, abnormal: {len(abnormal_segs)})"
                        )
                    except Exception as e:
                        logger.warning(f"{method} SQI failed: {e}")
                        quality_results[f"{method}_sqi"] = 0.5
                        sqi_arrays[f"{method}_sqi"] = []

            # Compute overall quality score (average of available SQIs)
            sqi_values = [
                v
                for v in quality_results.values()
                if isinstance(v, (int, float)) and not np.isnan(v)
            ]
            overall_quality = np.mean(sqi_values) if sqi_values else 0.5

            result = {
                "quality_scores": quality_results,
                "sqi_arrays": sqi_arrays,  # Full arrays for visualization
                "overall_quality": float(overall_quality),
                "passed": overall_quality >= quality_threshold,
                "sqi_window": sqi_window_seconds,
                "sqi_step": sqi_step_seconds,
                "threshold_used": quality_threshold,
                "scale_used": sqi_scale,
                "methods_used": sqi_methods,
            }
            return True, result

        elif stage == 3:
            # Stage 3: Parallel Processing (Filtering + Artifact Removal)
            from vitalDSP.filtering.signal_filtering import (
                SignalFiltering,
                BandpassFilter,
            )
            from vitalDSP.filtering.artifact_removal import ArtifactRemoval

            # Get parameters from pipeline_data
            filter_type = pipeline_data.get("filter_type", "butter")
            lowcut = pipeline_data.get("filter_lowcut", 0.5)
            highcut = pipeline_data.get("filter_highcut", 40)
            filter_order = pipeline_data.get("filter_order", 4)
            filter_rp = pipeline_data.get("filter_rp", 1)
            filter_rs = pipeline_data.get("filter_rs", 40)

            artifact_method = pipeline_data.get(
                "artifact_method", "baseline_correction"
            )
            baseline_cutoff = pipeline_data.get("baseline_cutoff", 0.5)
            median_kernel = pipeline_data.get("median_kernel", 5)
            wavelet_type = pipeline_data.get("wavelet_type", "db")
            wavelet_order = pipeline_data.get("wavelet_order", 4)

            paths = pipeline_data.get("paths", ["filtered", "preprocessed"])
            results = {}

            if "raw" in paths:
                results["raw"] = signal_data.copy()

            if "filtered" in paths or "preprocessed" in paths:
                # Apply bandpass filtering with selected filter type
                sf = SignalFiltering(signal_data)

                # Use the bandpass method with filter_type parameter
                try:
                    filtered = sf.bandpass(
                        lowcut=lowcut,
                        highcut=highcut,
                        fs=fs,
                        order=filter_order,
                        filter_type=filter_type,
                    )
                    logger.info(
                        f"✓ Applied {filter_type} bandpass filter: {lowcut}-{highcut} Hz, order {filter_order}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Bandpass filtering with {filter_type} failed: {e}, falling back to butterworth"
                    )
                    filtered = sf.bandpass(
                        lowcut=lowcut,
                        highcut=highcut,
                        fs=fs,
                        order=filter_order,
                        filter_type="butter",
                    )

                if "filtered" in paths:
                    results["filtered"] = filtered

                if "preprocessed" in paths:
                    # Apply selected artifact removal method
                    ar = ArtifactRemoval(filtered)

                    if artifact_method == "baseline_correction":
                        preprocessed = ar.baseline_correction(
                            cutoff=baseline_cutoff, fs=fs
                        )
                        logger.info(
                            f"✓ Applied baseline correction: cutoff={baseline_cutoff} Hz"
                        )
                    elif artifact_method == "mean_subtraction":
                        preprocessed = ar.mean_subtraction()
                        logger.info("✓ Applied mean subtraction")
                    elif artifact_method == "median_filter":
                        preprocessed = ar.median_filter_removal(
                            kernel_size=median_kernel
                        )
                        logger.info(
                            f"✓ Applied median filter: kernel_size={median_kernel}"
                        )
                    elif artifact_method == "wavelet":
                        preprocessed = ar.wavelet_denoising(
                            wavelet_type=wavelet_type, order=wavelet_order, level=1
                        )
                        logger.info(
                            f"✓ Applied wavelet denoising: {wavelet_type}{wavelet_order}"
                        )
                    else:
                        # Fallback to baseline correction
                        preprocessed = ar.baseline_correction(
                            cutoff=baseline_cutoff, fs=fs
                        )
                        logger.warning(
                            f"Unknown artifact method '{artifact_method}', using baseline correction"
                        )

                    results["preprocessed"] = preprocessed

            result = {
                "paths": list(results.keys()),
                "filtered_samples": {k: len(v) for k, v in results.items()},
                "filter_params": {
                    "type": filter_type,
                    "lowcut": lowcut,
                    "highcut": highcut,
                    "order": filter_order,
                    "rp": filter_rp,
                    "rs": filter_rs,
                },
                "artifact_params": {
                    "method": artifact_method,
                    "baseline_cutoff": baseline_cutoff,
                    "median_kernel": median_kernel,
                    "wavelet_type": wavelet_type,
                    "wavelet_order": wavelet_order,
                },
            }
            # Store processed signals for next stages
            pipeline_data["processed_signals"] = results
            return True, result

        elif stage == 4:
            # Stage 4: Quality Validation (compare paths using SQI + traditional metrics)
            from vitalDSP.signal_quality_assessment.signal_quality_index import (
                SignalQualityIndex,
            )

            processed_signals = pipeline_data.get("processed_signals", {})

            # Get parameters from pipeline_data
            stage4_sqi_methods = pipeline_data.get(
                "stage4_sqi_methods",
                ["snr", "amplitude_variability", "baseline_wander"],
            )
            sqi_metrics_weight = pipeline_data.get("sqi_metrics_weight", 0.5)
            snr_method = pipeline_data.get("snr_method", "standard")
            snr_weight = pipeline_data.get("snr_weight", 0.25)
            smoothness_weight = pipeline_data.get("smoothness_weight", 0.15)
            artifact_weight = pipeline_data.get("artifact_weight", 0.1)

            quality_metrics = {}

            # Define SQI window parameters
            sqi_window_seconds = 5
            window_size = int(sqi_window_seconds * fs)
            step_size = window_size // 2

            for path_name, path_signal in processed_signals.items():
                # Initialize SignalQualityIndex for this path
                sqi = SignalQualityIndex(path_signal)

                # Calculate SQI scores for selected methods
                sqi_scores = {}
                sqi_function_map = {
                    "amplitude_variability": lambda: sqi.amplitude_variability_sqi(
                        window_size, step_size, threshold=0.7
                    ),
                    "baseline_wander": lambda: sqi.baseline_wander_sqi(
                        window_size, step_size, threshold=0.7
                    ),
                    "snr": lambda: sqi.snr_sqi(window_size, step_size, threshold=0.7),
                    "zero_crossing": lambda: sqi.zero_crossing_consistency_sqi(
                        window_size, step_size, threshold=0.7
                    ),
                    "entropy": lambda: sqi.entropy_sqi(
                        window_size, step_size, threshold=0.7
                    ),
                    "kurtosis": lambda: sqi.kurtosis_sqi(
                        window_size, step_size, threshold=0.7
                    ),
                    "skewness": lambda: sqi.skewness_sqi(
                        window_size, step_size, threshold=0.7
                    ),
                }

                for method in stage4_sqi_methods:
                    if method in sqi_function_map:
                        try:
                            sqi_array, normal_segs, abnormal_segs = sqi_function_map[
                                method
                            ]()
                            sqi_scores[method] = float(np.mean(sqi_array))
                        except Exception as e:
                            logger.warning(
                                f"Stage 4 - {method} SQI failed for {path_name}: {e}"
                            )
                            sqi_scores[method] = 0.5

                # Calculate average SQI score
                avg_sqi_score = (
                    np.mean(list(sqi_scores.values())) if sqi_scores else 0.5
                )

                # Calculate traditional SNR based on selected method
                if snr_method == "standard":
                    noise = path_signal - signal_data[: len(path_signal)]
                    snr = float(np.var(path_signal) / (np.var(noise) + 1e-10))
                elif snr_method == "rms":
                    noise = path_signal - signal_data[: len(path_signal)]
                    snr = float(
                        np.sqrt(np.mean(path_signal**2))
                        / (np.sqrt(np.mean(noise**2)) + 1e-10)
                    )
                elif snr_method == "peak":
                    noise = path_signal - signal_data[: len(path_signal)]
                    snr = float(
                        np.max(np.abs(path_signal)) / (np.max(np.abs(noise)) + 1e-10)
                    )
                else:
                    snr = 1.0

                snr_score = min(1.0, max(0.0, snr / 20.0))

                # Calculate smoothness
                diff2 = np.diff(np.diff(path_signal))
                smoothness = 1.0 / (1.0 + np.std(diff2))
                smoothness_score = min(1.0, max(0.0, smoothness * 10))

                # Calculate artifact level
                diff1 = np.diff(path_signal)
                artifact_ratio = np.sum(np.abs(diff1) > 3 * np.std(diff1)) / len(diff1)
                artifact_score = 1.0 - min(1.0, artifact_ratio * 5)

                # Weighted combination of all metrics
                overall_quality = (
                    sqi_metrics_weight * avg_sqi_score
                    + snr_weight * snr_score
                    + smoothness_weight * smoothness_score
                    + artifact_weight * artifact_score
                )

                quality_metrics[path_name] = {
                    "sqi_scores": sqi_scores,
                    "avg_sqi_score": float(avg_sqi_score),
                    "snr": float(snr),
                    "snr_score": float(snr_score),
                    "smoothness_score": float(smoothness_score),
                    "artifact_score": float(artifact_score),
                    "overall_quality": float(overall_quality),
                }

                logger.info(
                    f"✓ {path_name}: Avg SQI={avg_sqi_score:.3f}, SNR={snr:.2f}, Overall Quality={overall_quality:.3f}"
                )

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
                "snr_method": snr_method,
                "weights": {
                    "snr": snr_weight,
                    "smoothness": smoothness_weight,
                    "artifact": artifact_weight,
                },
            }
            pipeline_data["best_path"] = best_path
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
            from vitalDSP.transforms.fourier_transform import FourierTransform

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

                # Frequency Domain Features
                if "frequency" in feature_categories:
                    try:
                        # Compute power spectral density using vitalDSP
                        ft = FourierTransform(seg, fs=fs)
                        freqs, psd = ft.compute_psd()

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
                    except Exception as e:
                        logger.warning(
                            f"Frequency feature extraction failed for segment {seg_idx}: {e}"
                        )

                # Statistical Features
                if "statistical" in feature_categories:
                    tdf = TimeDomainFeatures(seg)
                    stats_features = tdf.extract_all()

                    if "skewness" in statistical_features_selected:
                        seg_features["skewness"] = float(
                            stats_features.get("skewness", 0)
                        )
                        feature_names_used.add("skewness")
                    if "kurtosis" in statistical_features_selected:
                        seg_features["kurtosis"] = float(
                            stats_features.get("kurtosis", 0)
                        )
                        feature_names_used.add("kurtosis")
                    if "variance" in statistical_features_selected:
                        seg_features["variance"] = float(np.var(seg))
                        feature_names_used.add("variance")
                    if "median" in statistical_features_selected:
                        seg_features["median"] = float(np.median(seg))
                        feature_names_used.add("median")
                    if "iqr" in statistical_features_selected:
                        seg_features["iqr"] = float(stats_features.get("iqr", 0))
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
        logger.error(f"Error in stage {stage}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False, str(e)


def register_pipeline_callbacks(app):
    """
    Register all pipeline-related callbacks.

    Parameters
    ----------
    app : Dash
        The Dash application instance
    """

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
            State("pipeline-paths", "value"),
            State("pipeline-enable-quality", "value"),
            State("pipeline-current-stage", "data"),
            State("store-uploaded-data", "data"),
            # Stage 2 parameters - Quality Screening
            State("pipeline-sqi-methods", "value"),
            State("pipeline-sqi-window", "value"),
            State("pipeline-sqi-step", "value"),
            State("pipeline-quality-threshold", "value"),
            State("pipeline-sqi-scale", "value"),
            # Stage 3 parameters - Filtering
            State("pipeline-filter-type", "value"),
            State("pipeline-filter-lowcut", "value"),
            State("pipeline-filter-highcut", "value"),
            State("pipeline-filter-order", "value"),
            State("pipeline-filter-rp", "value"),
            State("pipeline-filter-rs", "value"),
            # Stage 3 parameters - Artifact Removal
            State("pipeline-artifact-method", "value"),
            State("pipeline-baseline-cutoff", "value"),
            State("pipeline-median-kernel", "value"),
            State("pipeline-wavelet-type", "value"),
            State("pipeline-wavelet-order", "value"),
            # Stage 4 parameters - Quality Validation
            State("pipeline-stage4-sqi-methods", "value"),
            State("pipeline-sqi-metrics-weight", "value"),
            State("pipeline-snr-method", "value"),
            State("pipeline-snr-weight", "value"),
            State("pipeline-smoothness-weight", "value"),
            State("pipeline-artifact-weight", "value"),
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
        sqi_methods,
        sqi_window,
        sqi_step,
        quality_threshold,
        sqi_scale,
        # Stage 3 parameters - Filtering
        filter_type,
        filter_lowcut,
        filter_highcut,
        filter_order,
        filter_rp,
        filter_rs,
        # Stage 3 parameters - Artifact Removal
        artifact_method,
        baseline_cutoff,
        median_kernel,
        wavelet_type,
        wavelet_order,
        # Stage 4 parameters - Quality Validation
        stage4_sqi_methods,
        sqi_metrics_weight,
        snr_method,
        snr_weight,
        smoothness_weight,
        artifact_weight,
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
                from vitalDSP_webapp.services.data.enhanced_data_service import get_enhanced_data_service

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
                    import pandas as pd

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
                import time

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
                    "sqi_methods": sqi_methods
                    or ["amplitude_variability", "baseline_wander", "snr"],
                    "sqi_window": sqi_window or 5,
                    "sqi_step": sqi_step or 2.5,
                    "quality_threshold": quality_threshold or 0.7,
                    "sqi_scale": sqi_scale or "zscore",
                    # Stage 3 parameters - Filtering
                    "filter_type": filter_type or "butter",
                    "filter_lowcut": filter_lowcut or 0.5,
                    "filter_highcut": filter_highcut or 40,
                    "filter_order": filter_order or 4,
                    "filter_rp": filter_rp or 1,
                    "filter_rs": filter_rs or 40,
                    # Stage 3 parameters - Artifact Removal
                    "artifact_method": artifact_method or "baseline_correction",
                    "baseline_cutoff": baseline_cutoff or 0.5,
                    "median_kernel": median_kernel or 5,
                    "wavelet_type": wavelet_type or "db",
                    "wavelet_order": wavelet_order or 4,
                    # Stage 4 parameters - Quality Validation
                    "stage4_sqi_methods": stage4_sqi_methods
                    or ["snr", "amplitude_variability", "baseline_wander"],
                    "sqi_metrics_weight": sqi_metrics_weight or 0.5,
                    "snr_method": snr_method or "standard",
                    "snr_weight": snr_weight or 0.25,
                    "smoothness_weight": smoothness_weight or 0.15,
                    "artifact_weight": artifact_weight or 0.1,
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
                            return (
                                {"display": "block"},
                                100,
                                "🔴 REAL DATA MODE - Pipeline completed successfully!",
                                False,
                                True,
                                False,  # Enable export
                                False,  # Enable report
                                True,  # Disable interval
                                pipeline_run_id,
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
                            pipeline_run_id,
                        )
                    else:
                        # Stage failed
                        logger.error(f"❌ Stage {new_stage} failed!")
                        return (
                            {"display": "block"},
                            (current_stage_num / 8) * 100,
                            f"Error at stage {new_stage}: {result}",
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

    @app.callback(
        Output("pipeline-paths-comparison", "figure"),
        [Input("pipeline-current-stage", "data")],
        [
            State("pipeline-paths", "value"),
            State("store-uploaded-data", "data"),
        ],
    )
    def update_paths_comparison(current_stage, selected_paths, uploaded_data):
        """
        Create comparison plot of different processing paths.
        Shows real pipeline data when available, falls back to simulation data.

        Technical Details:
        - Uses vitalDSP SignalFiltering.bandpass() for real bandpass filtering
        - Uses vitalDSP ArtifactRemoval.baseline_correction() for preprocessing
        - Signal-type-specific filter parameters (ECG: 0.5-40Hz, PPG: 0.5-8Hz, EEG: 0.5-50Hz)
        - Falls back to simple scaling if vitalDSP not available
        - Limits visualization to first 10 seconds for performance
        - Maintains proper time axis based on data's sampling frequency
        """
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
            title=f"Stage 5: Signal Segmentation ({len(segment_positions)} segments)",
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
        Display Stage 6: Feature Extraction Heatmap.
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

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=feature_matrix,
                x=[f"Seg {i+1}" for i in range(len(features_limited))],
                y=feature_names,
                colorscale="Viridis",
                hovertemplate="Feature: %{y}<br>Segment: %{x}<br>Value: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"Stage 6: Feature Extraction Heatmap ({len(features)} segments total, showing first 50)",
            xaxis_title="Segment",
            yaxis_title="Feature",
            template="plotly_white",
            height=max(400, len(feature_names) * 20),
        )

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
        if n_clicks is None:
            return no_update

        # Get the current pipeline run ID
        pipeline_run_id = getattr(
            register_pipeline_callbacks, "current_pipeline_run_id", None
        )
        if not pipeline_run_id:
            logger.warning("No pipeline run ID found for export")
            return no_update

        # Get pipeline data
        pipeline_data = register_pipeline_callbacks.pipeline_data.get(
            pipeline_run_id, {}
        )
        output_package = pipeline_data.get("output_package", {})

        if not output_package:
            logger.warning("No output package found for export")
            return no_update

        # Export as JSON
        output_json = json.dumps(output_package, indent=2, default=str)

        logger.info(f"Exporting pipeline results for run: {pipeline_run_id}")

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
        if n_clicks is None:
            return no_update

        # Get the current pipeline run ID
        pipeline_run_id = getattr(
            register_pipeline_callbacks, "current_pipeline_run_id", None
        )
        if not pipeline_run_id:
            logger.warning("No pipeline run ID found for report generation")
            return no_update

        # Get pipeline data
        pipeline_data = register_pipeline_callbacks.pipeline_data.get(
            pipeline_run_id, {}
        )
        results = pipeline_data.get("results", {})

        if not results:
            logger.warning("No results found for report generation")
            return no_update

        # Generate Markdown report (can be converted to PDF later)
        report_lines = [
            "# VitalDSP Pipeline Analysis Report",
            f"\n**Pipeline Run ID:** {pipeline_run_id}",
            f"\n**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Signal Type:** {pipeline_data.get('signal_type', 'Unknown')}",
            f"\n**Sampling Frequency:** {pipeline_data.get('fs', 0)} Hz",
            "\n---\n",
        ]

        # Add summary for each stage
        for stage_num in range(1, 9):
            stage_key = f"stage_{stage_num}"
            if stage_key in results:
                stage_name = (
                    PIPELINE_STAGES[stage_num]
                    if stage_num < len(PIPELINE_STAGES)
                    else f"Stage {stage_num}"
                )
                report_lines.append(f"\n## {stage_name}\n")

                stage_data = results[stage_key]
                for key, value in stage_data.items():
                    if isinstance(value, (int, float, str, bool)):
                        report_lines.append(f"- **{key}:** {value}")
                    elif isinstance(value, dict):
                        report_lines.append(f"- **{key}:**")
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float, str, bool)):
                                report_lines.append(f"  - {sub_key}: {sub_value}")
                    elif (
                        isinstance(value, list)
                        and len(value) > 0
                        and isinstance(value[0], (int, float, str))
                    ):
                        report_lines.append(
                            f"- **{key}:** {', '.join(map(str, value[:5]))}{'...' if len(value) > 5 else ''}"
                        )

                report_lines.append("\n")

        # Add recommendations if available
        stage7_results = results.get("stage_7", {})
        if "recommendations" in stage7_results:
            report_lines.append("\n## Recommendations\n")
            for rec in stage7_results["recommendations"]:
                report_lines.append(f"- {rec}")

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

    logger.info("Pipeline callbacks registered successfully")
