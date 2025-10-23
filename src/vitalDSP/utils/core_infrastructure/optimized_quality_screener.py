"""
Optimized Quality Screener for Large-Scale Physiological Signal Processing

This module implements an optimized 3-stage quality screening system with
dynamic configuration and performance optimization.

Author: vitalDSP Team
Date: 2025-10-12
Phase: 1 - Core Infrastructure (Optimized)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import threading
from functools import lru_cache
import gc

from vitalDSP.utils.config_utilities.dynamic_config import (
    get_config,
    DynamicConfig,
    Environment,
)
from vitalDSP.utils.core_infrastructure.optimized_data_loaders import ProgressInfo


class QualityLevel(Enum):
    """Quality levels for signal segments."""

    EXCELLENT = "excellent"  # > 0.8
    GOOD = "good"  # 0.6 - 0.8
    FAIR = "fair"  # 0.4 - 0.6
    POOR = "poor"  # 0.2 - 0.4
    UNUSABLE = "unusable"  # < 0.2


@dataclass
class QualityMetrics:
    """
    Comprehensive quality metrics for signal segments.
    """

    snr_db: float
    artifact_ratio: float
    baseline_drift: float
    signal_power: float
    noise_power: float
    peak_detection_rate: float
    frequency_domain_score: float
    temporal_consistency: float
    overall_quality: float
    quality_level: QualityLevel
    processing_recommendation: str
    confidence_score: float = 0.0
    processing_time_ms: float = 0.0


@dataclass
class ScreeningResult:
    """
    Result of quality screening operation.
    """

    segment_id: str
    start_idx: int
    end_idx: int
    duration_seconds: float
    quality_metrics: QualityMetrics
    passed_screening: bool
    screening_time: float
    warnings: List[str]
    performance_stats: Dict[str, Any] = None


class OptimizedQualityScreener:
    """
    Optimized 3-stage quality screening system for physiological signals.
    """

    def __init__(
        self,
        signal_type: str = "generic",
        sampling_rate: float = None,
        segment_duration: float = None,
        overlap_ratio: float = None,
        enable_parallel: bool = None,
        max_workers: Optional[int] = None,
        config: Optional[DynamicConfig] = None,
        **kwargs,
    ):
        """
        Initialize OptimizedQualityScreener.
        """
        self.config = config or get_config()
        self.signal_type = signal_type.lower()

        # Use config defaults if not provided
        self.sampling_rate = (
            sampling_rate or self.config.quality_screener.default_sampling_rate
        )
        self.segment_duration = (
            segment_duration or self.config.quality_screener.default_segment_duration
        )
        self.overlap_ratio = (
            overlap_ratio or self.config.quality_screener.default_overlap_ratio
        )
        self.enable_parallel = (
            enable_parallel
            if enable_parallel is not None
            else self.config.quality_screener.parallel_processing["enable_by_default"]
        )

        # Calculate optimal worker count
        if max_workers is None:
            max_workers_factor = self.config.quality_screener.parallel_processing[
                "max_workers_factor"
            ]
            max_workers_cap = self.config.quality_screener.parallel_processing[
                "max_workers_cap"
            ]
            self.max_workers = min(
                int(self.config.system_resources.cpu_count * max_workers_factor),
                max_workers_cap,
            )
        else:
            self.max_workers = max_workers

        # Calculate segment size in samples
        self.segment_size = int(self.segment_duration * self.sampling_rate)
        self.overlap_samples = int(self.segment_size * self.overlap_ratio)

        # Get thresholds for this signal type
        self.thresholds = self._get_signal_thresholds()

        # Performance tracking
        self.screening_stats = {
            "total_segments": 0,
            "passed_segments": 0,
            "total_time": 0.0,
            "avg_time_per_segment": 0.0,
            "parallel_speedup": 0.0,
            "memory_usage_mb": 0.0,
        }

        # Cached computations
        self._cached_fft_freqs = {}
        self._cached_window_sizes = {}

    def _get_signal_thresholds(self) -> Dict[str, Any]:
        """Get thresholds for the current signal type."""
        base_thresholds = self.config.quality_screener.quality_thresholds.get(
            "generic", {}
        )
        signal_thresholds = self.config.quality_screener.quality_thresholds.get(
            self.signal_type, {}
        )

        # Merge with signal-specific overrides
        thresholds = {**base_thresholds, **signal_thresholds}

        # Add signal-specific parameters
        signal_params = self.config.quality_screener.signal_parameters.get(
            self.signal_type, {}
        )
        thresholds.update(signal_params)

        return thresholds

    def screen_signal(
        self,
        signal_data: Union[np.ndarray, pd.DataFrame],
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
    ) -> List[ScreeningResult]:
        """
        Screen signal quality using optimized 3-stage approach.
        """
        start_time = time.time()

        # Convert to numpy array if needed
        if isinstance(signal_data, pd.DataFrame):
            signal_array = signal_data.values.flatten()
        else:
            signal_array = signal_data.flatten()

        # Generate segments
        segments = self._generate_segments(signal_array)

        # Determine if parallel processing should be used
        use_parallel = (
            self.enable_parallel
            and len(segments)
            >= self.config.quality_screener.parallel_processing[
                "min_segments_for_parallel"
            ]
        )

        if use_parallel:
            results = self._screen_segments_parallel(segments, progress_callback)
        else:
            results = self._screen_segments_sequential(segments, progress_callback)

        # Update statistics
        total_time = time.time() - start_time
        self._update_statistics(results, total_time)

        return results

    def _generate_segments(self, signal_array: np.ndarray) -> List[Dict[str, Any]]:
        """Generate overlapping segments from signal."""
        segments = []
        step_size = self.segment_size - self.overlap_samples

        for i in range(0, len(signal_array) - self.segment_size + 1, step_size):
            end_idx = min(i + self.segment_size, len(signal_array))
            segment_data = signal_array[i:end_idx]

            segments.append(
                {
                    "segment_id": f"seg_{i}_{end_idx}",
                    "start_idx": i,
                    "end_idx": end_idx,
                    "data": segment_data,
                    "duration_seconds": len(segment_data) / self.sampling_rate,
                }
            )

        return segments

    def _screen_segments_sequential(
        self,
        segments: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
    ) -> List[ScreeningResult]:
        """Screen segments sequentially with optimized processing."""
        results = []
        last_progress_update = time.time()

        for i, segment in enumerate(segments):
            # Progress callback with throttling
            current_time = time.time()
            if (
                progress_callback
                and current_time - last_progress_update
                >= self.config.data_loader.progress_update_interval
            ):

                progress_info = ProgressInfo(
                    bytes_processed=i * self.segment_size * 8,
                    total_bytes=len(segments) * self.segment_size * 8,
                    chunks_processed=i,
                    total_chunks=len(segments),
                    elapsed_time=current_time - last_progress_update,
                    estimated_remaining=0.0,
                    current_chunk_size=self.segment_size,
                    loading_strategy="sequential_quality_screening",
                )
                progress_callback(progress_info)
                last_progress_update = current_time

            result = self._screen_single_segment_optimized(segment)
            results.append(result)

        return results

    def _screen_segments_parallel(
        self,
        segments: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
    ) -> List[ScreeningResult]:
        """Screen segments in parallel with optimized worker management."""
        results = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all segments for processing
            future_to_segment = {
                executor.submit(self._screen_single_segment_optimized, segment): segment
                for segment in segments
            }

            # Collect results as they complete
            completed_count = 0
            last_progress_update = time.time()

            for future in future_to_segment:
                # Progress callback with throttling
                current_time = time.time()
                if (
                    progress_callback
                    and current_time - last_progress_update
                    >= self.config.data_loader.progress_update_interval
                ):

                    progress_info = ProgressInfo(
                        bytes_processed=completed_count * self.segment_size * 8,
                        total_bytes=len(segments) * self.segment_size * 8,
                        chunks_processed=completed_count,
                        total_chunks=len(segments),
                        elapsed_time=current_time - last_progress_update,
                        estimated_remaining=0.0,
                        current_chunk_size=self.segment_size,
                        loading_strategy="parallel_quality_screening",
                    )
                    progress_callback(progress_info)
                    last_progress_update = current_time

                result = future.result()
                results.append(result)
                completed_count += 1

        # Sort results by segment start index
        results.sort(key=lambda x: x.start_idx)
        return results

    def _screen_single_segment_optimized(
        self, segment: Dict[str, Any]
    ) -> ScreeningResult:
        """Screen a single segment using optimized 3-stage approach."""
        start_time = time.time()
        warnings_list = []
        performance_stats = {}

        try:
            # Stage 1: Quick SNR Check (optimized)
            snr_result = self._stage1_snr_check_optimized(segment["data"])
            if not snr_result["passed"]:
                warnings_list.append(
                    f"Stage 1 failed: SNR too low ({snr_result['snr_db']:.1f} dB)"
                )

            # Stage 2: Statistical Screen (optimized)
            stats_result = self._stage2_statistical_screen_optimized(segment["data"])
            if not stats_result["passed"]:
                warnings_list.append("Stage 2 failed: Statistical anomalies detected")

            # Stage 3: Signal-Specific Screen (optimized)
            signal_result = self._stage3_signal_specific_screen_optimized(
                segment["data"]
            )
            if not signal_result["passed"]:
                warnings_list.append("Stage 3 failed: Signal-specific quality issues")

            # Combine results
            passed_screening = all(
                [snr_result["passed"], stats_result["passed"], signal_result["passed"]]
            )

            # Calculate overall quality metrics
            quality_metrics = self._calculate_overall_metrics_optimized(
                snr_result, stats_result, signal_result
            )

            screening_time = time.time() - start_time
            quality_metrics.processing_time_ms = screening_time * 1000

            # Performance stats
            performance_stats = {
                "snr_calculation_time": snr_result.get("calculation_time", 0),
                "statistical_calculation_time": stats_result.get("calculation_time", 0),
                "signal_specific_calculation_time": signal_result.get(
                    "calculation_time", 0
                ),
                "total_calculation_time": screening_time,
            }

            return ScreeningResult(
                segment_id=segment["segment_id"],
                start_idx=segment["start_idx"],
                end_idx=segment["end_idx"],
                duration_seconds=segment["duration_seconds"],
                quality_metrics=quality_metrics,
                passed_screening=passed_screening,
                screening_time=screening_time,
                warnings=warnings_list,
                performance_stats=performance_stats,
            )

        except Exception as e:
            warnings_list.append(f"Screening error: {str(e)}")
            return ScreeningResult(
                segment_id=segment["segment_id"],
                start_idx=segment["start_idx"],
                end_idx=segment["end_idx"],
                duration_seconds=segment["duration_seconds"],
                quality_metrics=self._create_failed_metrics(),
                passed_screening=False,
                screening_time=time.time() - start_time,
                warnings=warnings_list,
                performance_stats={"error": str(e)},
            )

    def _stage1_snr_check_optimized(self, signal: np.ndarray) -> Dict[str, Any]:
        """Optimized Stage 1: Quick SNR Check."""
        start_time = time.time()

        try:
            # Optimized signal power calculation using vectorized operations
            signal_power = np.mean(signal**2)

            # Optimized noise estimation using diff
            signal_diff = np.diff(signal)
            noise_power = np.var(signal_diff)

            # Calculate SNR in dB
            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
            else:
                snr_db = float("inf")

            passed = snr_db >= self.thresholds["snr_min_db"]

            return {
                "passed": passed,
                "snr_db": snr_db,
                "signal_power": signal_power,
                "noise_power": noise_power,
                "calculation_time": time.time() - start_time,
            }

        except Exception as e:
            return {
                "passed": False,
                "snr_db": -float("inf"),
                "signal_power": 0.0,
                "noise_power": float("inf"),
                "error": str(e),
                "calculation_time": time.time() - start_time,
            }

    def _stage2_statistical_screen_optimized(
        self, signal: np.ndarray
    ) -> Dict[str, Any]:
        """Optimized Stage 2: Statistical Screen."""
        start_time = time.time()

        try:
            # Vectorized statistical calculations
            mean_val = np.mean(signal)
            std_val = np.std(signal)

            # Optimized outlier detection
            outlier_threshold = self.config.quality_screener.statistical_thresholds[
                "outlier_sigma_multiplier"
            ]
            outliers = np.abs(signal - mean_val) > outlier_threshold * std_val
            outlier_ratio = np.sum(outliers) / len(signal)

            # Optimized constant value detection
            unique_values = len(np.unique(signal))
            constant_ratio = 1.0 - (unique_values / len(signal))

            # Optimized jump detection
            signal_diff = np.abs(np.diff(signal))
            jump_threshold = self.config.quality_screener.statistical_thresholds[
                "jump_sigma_multiplier"
            ] * np.std(signal_diff)
            jumps = signal_diff > jump_threshold
            jump_ratio = np.sum(jumps) / len(signal_diff)

            # Pass if all ratios are below thresholds
            passed = (
                outlier_ratio
                < self.config.quality_screener.statistical_thresholds[
                    "outlier_max_ratio"
                ]
                and constant_ratio
                < self.config.quality_screener.statistical_thresholds[
                    "constant_max_ratio"
                ]
                and jump_ratio
                < self.config.quality_screener.statistical_thresholds["jump_max_ratio"]
            )

            return {
                "passed": passed,
                "outlier_ratio": outlier_ratio,
                "constant_ratio": constant_ratio,
                "jump_ratio": jump_ratio,
                "mean": mean_val,
                "std": std_val,
                "calculation_time": time.time() - start_time,
            }

        except Exception as e:
            return {
                "passed": False,
                "outlier_ratio": 1.0,
                "constant_ratio": 1.0,
                "jump_ratio": 1.0,
                "error": str(e),
                "calculation_time": time.time() - start_time,
            }

    def _stage3_signal_specific_screen_optimized(
        self, signal: np.ndarray
    ) -> Dict[str, Any]:
        """Optimized Stage 3: Signal-Specific Screen."""
        start_time = time.time()

        try:
            # Calculate baseline drift (optimized)
            baseline_drift = self._calculate_baseline_drift_optimized(signal)

            # Calculate peak detection rate (optimized)
            peak_rate = self._calculate_peak_detection_rate_optimized(signal)

            # Calculate frequency domain score (optimized)
            freq_score = self._calculate_frequency_score_optimized(signal)

            # Calculate temporal consistency (optimized)
            temporal_score = self._calculate_temporal_consistency_optimized(signal)

            # Calculate artifact ratio (simplified for performance)
            artifact_ratio = self._estimate_artifact_ratio_optimized(signal)

            # Pass if all metrics meet thresholds
            passed = (
                artifact_ratio <= self.thresholds["artifact_max_ratio"]
                and baseline_drift <= self.thresholds["baseline_max_drift"]
                and peak_rate >= self.thresholds["peak_detection_min_rate"]
                and freq_score >= self.thresholds["frequency_score_min"]
                and temporal_score >= self.thresholds["temporal_consistency_min"]
            )

            return {
                "passed": passed,
                "artifact_ratio": artifact_ratio,
                "baseline_drift": baseline_drift,
                "peak_detection_rate": peak_rate,
                "frequency_score": freq_score,
                "temporal_consistency": temporal_score,
                "calculation_time": time.time() - start_time,
            }

        except Exception as e:
            return {
                "passed": False,
                "artifact_ratio": 1.0,
                "baseline_drift": float("inf"),
                "peak_detection_rate": 0.0,
                "frequency_score": 0.0,
                "temporal_consistency": 0.0,
                "error": str(e),
                "calculation_time": time.time() - start_time,
            }

    def _calculate_baseline_drift_optimized(self, signal: np.ndarray) -> float:
        """Optimized baseline drift calculation."""
        # Use linear regression for efficiency
        x = np.arange(len(signal))
        coeffs = np.polyfit(x, signal, 1)
        trend_line = np.polyval(coeffs, x)
        drift = np.max(np.abs(trend_line - np.mean(signal)))
        return drift

    def _calculate_peak_detection_rate_optimized(self, signal: np.ndarray) -> float:
        """Optimized peak detection rate calculation."""
        if self.signal_type == "ecg":
            return self._detect_ecg_peaks_optimized(signal)
        elif self.signal_type == "ppg":
            return self._detect_ppg_peaks_optimized(signal)
        else:
            return self._detect_generic_peaks_optimized(signal)

    def _detect_ecg_peaks_optimized(self, signal: np.ndarray) -> float:
        """Optimized ECG peak detection."""
        try:
            from scipy.signal import find_peaks

            # Use signal-specific parameters
            max_hr = self.thresholds.get("max_heart_rate_bpm", 150)
            min_distance = int(
                60 / max_hr * self.sampling_rate
            )  # Minimum distance between peaks

            peaks, _ = find_peaks(signal, distance=min_distance, height=np.mean(signal))

            # Expected peaks based on duration and typical heart rate
            duration = len(signal) / self.sampling_rate
            expected_hr = self.thresholds.get("expected_heart_rate_bpm", 72)
            expected_peaks = duration * expected_hr / 60

            if expected_peaks > 0:
                return min(len(peaks) / expected_peaks, 1.0)
            return 0.0

        except ImportError:
            # Fallback to simple peak detection
            return self._simple_peak_detection(signal)

    def _detect_ppg_peaks_optimized(self, signal: np.ndarray) -> float:
        """Optimized PPG peak detection."""
        try:
            from scipy.signal import find_peaks

            # Use signal-specific parameters
            max_pr = self.thresholds.get("max_pulse_rate_bpm", 120)
            min_distance = int(60 / max_pr * self.sampling_rate)

            peaks, _ = find_peaks(signal, distance=min_distance, height=np.mean(signal))

            # Expected peaks based on duration and typical pulse rate
            duration = len(signal) / self.sampling_rate
            expected_pr = self.thresholds.get("expected_pulse_rate_bpm", 60)
            expected_peaks = duration * expected_pr / 60

            if expected_peaks > 0:
                return min(len(peaks) / expected_peaks, 1.0)
            return 0.0

        except ImportError:
            return self._simple_peak_detection(signal)

    def _detect_generic_peaks_optimized(self, signal: np.ndarray) -> float:
        """Optimized generic peak detection."""
        # Simple threshold-based peak detection
        threshold = np.mean(signal) + 0.5 * np.std(signal)
        peaks = signal > threshold

        # Count consecutive peaks as single peaks
        peak_count = 0
        in_peak = False
        for peak in peaks:
            if peak and not in_peak:
                peak_count += 1
                in_peak = True
            elif not peak:
                in_peak = False

        # Return ratio of peaks found
        return min(peak_count / (len(signal) / 100), 1.0)

    def _simple_peak_detection(self, signal: np.ndarray) -> float:
        """Simple peak detection fallback."""
        # Find local maxima
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
                peaks.append(i)

        return min(len(peaks) / (len(signal) / 100), 1.0)

    def _calculate_frequency_score_optimized(self, signal: np.ndarray) -> float:
        """Optimized frequency domain quality score."""
        try:
            from scipy.fft import fft, fftfreq

            # Use cached FFT frequencies if available
            signal_length = len(signal)
            if signal_length not in self._cached_fft_freqs:
                self._cached_fft_freqs[signal_length] = fftfreq(
                    signal_length, 1 / self.sampling_rate
                )

            freqs = self._cached_fft_freqs[signal_length]

            # Compute FFT
            fft_signal = fft(signal)

            # Focus on positive frequencies
            positive_mask = freqs >= 0
            positive_freqs = freqs[positive_mask]
            positive_fft = np.abs(fft_signal[positive_mask])

            # Calculate spectral centroid
            if np.sum(positive_fft) > 0:
                spectral_centroid = np.sum(positive_freqs * positive_fft) / np.sum(
                    positive_fft
                )
                normalization_factor = self.config.quality_screener.frequency_analysis[
                    "spectral_centroid_normalization_factor"
                ]
                score = min(
                    spectral_centroid / (self.sampling_rate / normalization_factor), 1.0
                )
            else:
                score = 0.0

            return score

        except ImportError:
            # Fallback to simple frequency analysis
            return self._simple_frequency_analysis(signal)

    def _simple_frequency_analysis(self, signal: np.ndarray) -> float:
        """Simple frequency analysis fallback."""
        # Calculate signal variance as a proxy for frequency content
        signal_var = np.var(signal)
        max_var = np.var(np.random.randn(len(signal)))  # Reference variance
        return min(signal_var / max_var, 1.0)

    def _calculate_temporal_consistency_optimized(self, signal: np.ndarray) -> float:
        """Optimized temporal consistency calculation."""
        try:
            # Calculate autocorrelation at lag 1
            autocorr = np.corrcoef(signal[:-1], signal[1:])[0, 1]

            # Calculate local variance consistency with optimized windowing
            window_size = self._get_optimal_window_size(len(signal))
            local_vars = []

            for i in range(0, len(signal) - window_size, window_size):
                local_vars.append(np.var(signal[i : i + window_size]))

            if len(local_vars) > 1:
                var_consistency = 1.0 - (np.std(local_vars) / np.mean(local_vars))
            else:
                var_consistency = 1.0

            # Combine metrics
            consistency_score = (max(autocorr, 0) + var_consistency) / 2
            return max(consistency_score, 0.0)

        except Exception:
            return 0.0

    def _get_optimal_window_size(self, signal_length: int) -> int:
        """Get optimal window size for temporal analysis."""
        if signal_length not in self._cached_window_sizes:
            window_factor = self.config.quality_screener.frequency_analysis[
                "window_size_factor"
            ]
            min_window = self.config.quality_screener.frequency_analysis[
                "min_window_size"
            ]
            window_size = max(min_window, int(signal_length * window_factor))
            self._cached_window_sizes[signal_length] = window_size

        return self._cached_window_sizes[signal_length]

    def _estimate_artifact_ratio_optimized(self, signal: np.ndarray) -> float:
        """Optimized artifact ratio estimation."""
        # Simple artifact detection based on signal amplitude
        signal_std = np.std(signal)
        signal_mean = np.mean(signal)

        # Count samples that are more than 3 standard deviations from mean
        artifact_threshold = 3 * signal_std
        artifacts = np.abs(signal - signal_mean) > artifact_threshold

        return np.sum(artifacts) / len(signal)

    def _calculate_overall_metrics_optimized(
        self,
        snr_result: Dict[str, Any],
        stats_result: Dict[str, Any],
        signal_result: Dict[str, Any],
    ) -> QualityMetrics:
        """Calculate overall quality metrics with confidence scoring."""
        # Calculate individual scores
        scores = []

        # SNR score (0-1)
        snr_score = (
            min(snr_result["snr_db"] / 30.0, 1.0)
            if snr_result["snr_db"] != float("inf")
            else 1.0
        )
        scores.append(snr_score)

        # Statistical score (0-1)
        stats_score = 1.0 - max(
            stats_result.get("outlier_ratio", 0),
            stats_result.get("constant_ratio", 0),
            stats_result.get("jump_ratio", 0),
        )
        scores.append(stats_score)

        # Signal-specific score (0-1)
        signal_score = (
            signal_result.get("peak_detection_rate", 0)
            + signal_result.get("frequency_score", 0)
            + signal_result.get("temporal_consistency", 0)
        ) / 3
        scores.append(signal_score)

        # Calculate weighted average
        overall_quality = np.mean(scores)

        # Calculate confidence score based on consistency
        confidence_score = 1.0 - np.std(scores) if len(scores) > 1 else 1.0

        # Determine quality level
        quality_levels = self.config.quality_screener.quality_levels
        if overall_quality > quality_levels["excellent"]:
            quality_level = QualityLevel.EXCELLENT
            recommendation = "Full processing recommended"
        elif overall_quality > quality_levels["good"]:
            quality_level = QualityLevel.GOOD
            recommendation = "Standard processing recommended"
        elif overall_quality > quality_levels["fair"]:
            quality_level = QualityLevel.FAIR
            recommendation = "Conservative processing recommended"
        elif overall_quality > quality_levels["poor"]:
            quality_level = QualityLevel.POOR
            recommendation = "Minimal processing or manual review"
        else:
            quality_level = QualityLevel.UNUSABLE
            recommendation = "Skip processing or manual review"

        return QualityMetrics(
            snr_db=snr_result.get("snr_db", -float("inf")),
            artifact_ratio=signal_result.get("artifact_ratio", 1.0),
            baseline_drift=signal_result.get("baseline_drift", float("inf")),
            signal_power=snr_result.get("signal_power", 0.0),
            noise_power=snr_result.get("noise_power", float("inf")),
            peak_detection_rate=signal_result.get("peak_detection_rate", 0.0),
            frequency_domain_score=signal_result.get("frequency_score", 0.0),
            temporal_consistency=signal_result.get("temporal_consistency", 0.0),
            overall_quality=overall_quality,
            quality_level=quality_level,
            processing_recommendation=recommendation,
            confidence_score=confidence_score,
        )

    def _create_failed_metrics(self) -> QualityMetrics:
        """Create metrics for failed screening."""
        return QualityMetrics(
            snr_db=-float("inf"),
            artifact_ratio=1.0,
            baseline_drift=float("inf"),
            signal_power=0.0,
            noise_power=float("inf"),
            peak_detection_rate=0.0,
            frequency_domain_score=0.0,
            temporal_consistency=0.0,
            overall_quality=0.0,
            quality_level=QualityLevel.UNUSABLE,
            processing_recommendation="Skip processing",
            confidence_score=0.0,
        )

    def _update_statistics(self, results: List[ScreeningResult], total_time: float):
        """Update screening statistics."""
        self.screening_stats["total_segments"] += len(results)
        self.screening_stats["passed_segments"] += sum(
            1 for r in results if r.passed_screening
        )
        self.screening_stats["total_time"] += total_time

        if self.screening_stats["total_segments"] > 0:
            self.screening_stats["avg_time_per_segment"] = (
                self.screening_stats["total_time"]
                / self.screening_stats["total_segments"]
            )

        # Calculate parallel speedup
        if len(results) > 1 and self.enable_parallel:
            sequential_time = sum(
                r.performance_stats.get("total_calculation_time", 0) for r in results
            )
            parallel_time = total_time
            if parallel_time > 0:
                self.screening_stats["parallel_speedup"] = (
                    sequential_time / parallel_time
                )

    def get_statistics(self) -> Dict[str, Any]:
        """Get screening statistics."""
        stats = self.screening_stats.copy()
        if stats["total_segments"] > 0:
            stats["pass_rate"] = stats["passed_segments"] / stats["total_segments"]
        else:
            stats["pass_rate"] = 0.0
        return stats

    def reset_statistics(self):
        """Reset screening statistics."""
        self.screening_stats = {
            "total_segments": 0,
            "passed_segments": 0,
            "total_time": 0.0,
            "avg_time_per_segment": 0.0,
            "parallel_speedup": 0.0,
            "memory_usage_mb": 0.0,
        }


# Backward compatibility alias
QualityScreener = OptimizedQualityScreener

# Example usage and tests
if __name__ == "__main__":
    print("Optimized Quality Screener Module")
    print("=" * 50)
    print("\nThis module provides optimized 3-stage quality screening")
    print("with dynamic configuration and performance optimization.")
    print("\nFeatures:")
    print("  - Dynamic configuration system")
    print("  - Optimized algorithms")
    print("  - Performance monitoring")
    print("  - Cached computations")
