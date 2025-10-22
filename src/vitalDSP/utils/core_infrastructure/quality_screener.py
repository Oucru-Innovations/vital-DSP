"""
Quality Screener for Large-Scale Physiological Signal Processing

This module implements a 3-stage quality screening system for efficient
preprocessing of large physiological datasets:

Stage 1: Quick SNR Check - Fast signal-to-noise ratio estimation
Stage 2: Statistical Screen - Statistical anomaly detection
Stage 3: Signal-Specific Screen - Domain-specific quality metrics using
         comprehensive SignalQualityIndex methods from signal_quality_assessment

Features:
- Multi-stage screening with configurable thresholds
- Deep integration with SignalQualityIndex for 14+ SQI metrics
- Signal-type-specific quality assessment (ECG, PPG, EEG, generic)
- Batch processing for large datasets with parallel support
- Quality-aware processing recommendations
- Configuration-driven threshold management

SignalQualityIndex Integration:
- ECG: amplitude_variability, baseline_wander, zero_crossing, HRV
- PPG: ppg_signal_quality, baseline_wander, waveform_similarity
- EEG: eeg_band_power, signal_entropy, skewness, kurtosis
- Generic: energy, amplitude_variability, baseline_wander

Author: vitalDSP Team
Date: 2025-10-12
Phase: 1 - Core Infrastructure (Week 2)
Updated: 2025-10-17 (Phase 4 - Enhanced SQI Integration)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import time
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Configure logging
logger = logging.getLogger(__name__)

# Import existing quality assessment modules
from vitalDSP.signal_quality_assessment.signal_quality_index import SignalQualityIndex

# from vitalDSP.signal_quality_assessment.artifact_detection_removal import ArtifactDetector
from vitalDSP.utils.core_infrastructure.data_loaders import (
    ChunkedDataLoader,
    MemoryMappedLoader,
    ProgressInfo,
)


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

    Attributes:
        snr_db: Signal-to-noise ratio in dB
        artifact_ratio: Ratio of artifact-contaminated samples
        baseline_drift: Baseline drift magnitude
        signal_power: Signal power
        noise_power: Noise power
        peak_detection_rate: Rate of successful peak detection
        frequency_domain_score: Frequency domain quality score
        temporal_consistency: Temporal consistency score
        overall_quality: Overall quality score (0-1)
        quality_level: Quality level classification
        processing_recommendation: Recommended processing approach
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


@dataclass
class ScreeningResult:
    """
    Result of quality screening operation.

    Attributes:
        segment_id: Unique identifier for the segment
        start_idx: Starting index in original signal
        end_idx: Ending index in original signal
        duration_seconds: Duration in seconds
        quality_metrics: Detailed quality metrics
        passed_screening: Whether segment passed all screening stages
        screening_time: Time taken for screening (seconds)
        warnings: List of warnings generated during screening
    """

    segment_id: str
    start_idx: int
    end_idx: int
    duration_seconds: float
    quality_metrics: QualityMetrics
    passed_screening: bool
    screening_time: float
    warnings: List[str]


class QualityScreener:
    """
    3-stage quality screening system for physiological signals.

    The screener implements a hierarchical approach:
    1. Quick SNR Check: Fast signal-to-noise ratio estimation
    2. Statistical Screen: Statistical anomaly detection
    3. Signal-Specific Screen: Domain-specific quality metrics

    Features:
    - Configurable thresholds for each stage
    - Batch processing for efficiency
    - Integration with existing quality modules
    - Parallel processing support
    - Quality-aware recommendations

    Example:
        >>> screener = QualityScreener(signal_type='ECG', sampling_rate=250)
        >>> results = screener.screen_signal(signal_data)
        >>> for result in results:
        ...     if result.passed_screening:
        ...         process_good_segment(result)
    """

    def __init__(
        self,
        signal_type: str = "generic",
        sampling_rate: float = 100.0,
        segment_duration: float = 10.0,
        overlap_ratio: float = 0.1,
        enable_parallel: bool = True,
        max_workers: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize QualityScreener.

        Args:
            signal_type: Type of signal ('ECG', 'PPG', 'EEG', 'generic')
            sampling_rate: Sampling rate in Hz
            segment_duration: Duration of segments for screening (seconds)
            overlap_ratio: Overlap ratio between segments (0-1)
            enable_parallel: Enable parallel processing
            max_workers: Maximum number of worker processes
            **kwargs: Additional configuration parameters
        """
        self.signal_type = signal_type.lower()
        self.sampling_rate = sampling_rate
        self.segment_duration = segment_duration
        self.overlap_ratio = overlap_ratio
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers or min(mp.cpu_count(), 8)

        # Calculate segment size in samples
        self.segment_size = int(segment_duration * sampling_rate)
        self.overlap_samples = int(self.segment_size * overlap_ratio)

        # Initialize quality assessment modules (will be created per signal)
        self.quality_index = None
        # self.artifact_detector = ArtifactDetector()  # Commented out due to import issues

        # Configure thresholds based on signal type
        self.thresholds = self._configure_thresholds(**kwargs)

        # Performance tracking
        self.screening_stats = {
            "total_segments": 0,
            "passed_segments": 0,
            "total_time": 0.0,
            "avg_time_per_segment": 0.0,
        }

    def _configure_thresholds(self, **kwargs) -> Dict[str, Any]:
        """Configure quality thresholds based on signal type."""
        base_thresholds = {
            # Stage 1: SNR thresholds
            "snr_min_db": 10.0,
            # Stage 2: Statistical thresholds
            "artifact_max_ratio": 0.3,
            "outlier_std_factor": 3.0,  # Standard deviations for outlier detection
            "outlier_max_ratio": 0.1,  # Maximum ratio of outliers allowed
            "constant_max_ratio": 0.5,  # Maximum ratio of constant values allowed
            "jump_std_factor": 3.0,  # Standard deviations for jump detection
            "jump_max_ratio": 0.15,  # Maximum ratio of jumps allowed
            # Stage 3: Signal-specific SQI thresholds
            "baseline_max_drift": 0.5,
            "peak_detection_min_rate": 0.7,
            "frequency_score_min": 0.5,
            "temporal_consistency_min": 0.5,
            "overall_quality_min": 0.4,
            # SignalQualityIndex method thresholds (0-1 normalized scores)
            "amplitude_variability_min": 0.5,
            "baseline_wander_min": 0.5,
            "zero_crossing_min": 0.5,
            "waveform_similarity_min": 0.5,
            "signal_entropy_min": 0.5,
            "skewness_min": 0.5,
            "kurtosis_min": 0.5,
            "peak_to_peak_min": 0.5,
            "energy_min": 0.5,
            "hrv_min": 0.5,
            "ppg_quality_min": 0.5,
            "eeg_band_power_min": 0.5,
            "respiratory_quality_min": 0.5,
            # Peak detection parameters (for legacy methods)
            "ecg_min_peak_distance_factor": 0.4,  # 0.4 seconds = 150 BPM max
            "ppg_min_peak_distance_factor": 0.5,  # 0.5 seconds = 120 BPM max
            "ecg_expected_heart_rate_bpm": 72,  # Expected average heart rate for ECG
            "ppg_expected_pulse_rate_bpm": 60,  # Expected average pulse rate for PPG
        }

        # Signal-specific adjustments
        if self.signal_type == "ecg":
            base_thresholds.update(
                {
                    "snr_min_db": 15.0,
                    "artifact_max_ratio": 0.2,
                    "peak_detection_min_rate": 0.8,
                    "amplitude_variability_min": 0.6,
                    "baseline_wander_min": 0.6,
                    "zero_crossing_min": 0.6,
                    "hrv_min": 0.5,
                }
            )
        elif self.signal_type == "ppg":
            base_thresholds.update(
                {
                    "snr_min_db": 12.0,
                    "artifact_max_ratio": 0.25,
                    "baseline_max_drift": 0.3,
                    "ppg_quality_min": 0.6,
                    "baseline_wander_min": 0.6,
                    "waveform_similarity_min": 0.5,
                }
            )
        elif self.signal_type == "eeg":
            base_thresholds.update(
                {
                    "snr_min_db": 8.0,
                    "artifact_max_ratio": 0.4,
                    "baseline_max_drift": 0.6,
                    "eeg_band_power_min": 0.5,
                    "signal_entropy_min": 0.5,
                    "skewness_min": 0.4,
                    "kurtosis_min": 0.4,
                }
            )

        # Override with user-provided thresholds
        base_thresholds.update(kwargs)
        return base_thresholds

    def screen_signal(
        self,
        signal_data: Union[np.ndarray, pd.DataFrame],
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
    ) -> List[ScreeningResult]:
        """
        Screen signal quality using 3-stage approach.

        Args:
            signal_data: Input signal data
            progress_callback: Optional progress callback

        Returns:
            List of screening results for each segment
        """
        start_time = time.time()

        # Convert to numpy array if needed
        if isinstance(signal_data, pd.DataFrame):
            signal_array = signal_data.values.flatten()
        else:
            signal_array = signal_data.flatten()

        # Generate segments
        segments = self._generate_segments(signal_array)

        if self.enable_parallel and len(segments) > 1:
            results = self._screen_segments_parallel(segments, progress_callback)
        else:
            results = self._screen_segments_sequential(segments, progress_callback)

        # Update statistics
        self._update_statistics(results, time.time() - start_time)

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
        """Screen segments sequentially."""
        results = []

        for i, segment in enumerate(segments):
            if progress_callback:
                progress_info = ProgressInfo(
                    bytes_processed=i * self.segment_size * 8,  # Estimate
                    total_bytes=len(segments) * self.segment_size * 8,
                    chunks_processed=i,
                    total_chunks=len(segments),
                    elapsed_time=0.0,
                    estimated_remaining=0.0,
                    current_chunk_size=self.segment_size,
                    loading_strategy="quality_screening",
                )
                progress_callback(progress_info)

            result = self._screen_single_segment(segment)
            results.append(result)

        return results

    def _screen_segments_parallel(
        self,
        segments: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
    ) -> List[ScreeningResult]:
        """Screen segments in parallel using ThreadPoolExecutor for better compatibility."""
        results = []

        try:
            # Use ThreadPoolExecutor instead of ProcessPoolExecutor for better compatibility
            # with unpicklable objects (SignalQualityIndex instances)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all segments for processing
                future_to_segment = {
                    executor.submit(self._screen_single_segment, segment): segment
                    for segment in segments
                }

                # Collect results as they complete
                for i, future in enumerate(future_to_segment):
                    if progress_callback:
                        progress_info = ProgressInfo(
                            bytes_processed=i * self.segment_size * 8,
                            total_bytes=len(segments) * self.segment_size * 8,
                            chunks_processed=i,
                            total_chunks=len(segments),
                            elapsed_time=0.0,
                            estimated_remaining=0.0,
                            current_chunk_size=self.segment_size,
                            loading_strategy="parallel_quality_screening",
                        )
                        progress_callback(progress_info)

                    result = future.result()
                    results.append(result)

        except Exception as e:
            # If parallel processing fails, fall back to sequential processing
            logger.warning(
                f"Parallel screening failed ({e}), falling back to sequential processing"
            )
            results = self._screen_segments_sequential(segments, progress_callback)

        # Sort results by segment start index
        results.sort(key=lambda x: x.start_idx)
        return results

    def _screen_single_segment(self, segment: Dict[str, Any]) -> ScreeningResult:
        """Screen a single segment using 3-stage approach."""
        start_time = time.time()
        warnings_list = []

        try:
            # Stage 1: Quick SNR Check
            snr_result = self._stage1_snr_check(segment["data"])
            if not snr_result["passed"]:
                warnings_list.append(
                    f"Stage 1 failed: SNR too low ({snr_result['snr_db']:.1f} dB)"
                )

            # Stage 2: Statistical Screen
            stats_result = self._stage2_statistical_screen(segment["data"])
            if not stats_result["passed"]:
                warnings_list.append("Stage 2 failed: Statistical anomalies detected")

            # Stage 3: Signal-Specific Screen
            signal_result = self._stage3_signal_specific_screen(segment["data"])
            if not signal_result["passed"]:
                warnings_list.append("Stage 3 failed: Signal-specific quality issues")

            # Combine results
            passed_screening = all(
                [snr_result["passed"], stats_result["passed"], signal_result["passed"]]
            )

            # Calculate overall quality metrics
            quality_metrics = self._calculate_overall_metrics(
                snr_result, stats_result, signal_result
            )

            screening_time = time.time() - start_time

            return ScreeningResult(
                segment_id=segment["segment_id"],
                start_idx=segment["start_idx"],
                end_idx=segment["end_idx"],
                duration_seconds=segment["duration_seconds"],
                quality_metrics=quality_metrics,
                passed_screening=passed_screening,
                screening_time=screening_time,
                warnings=warnings_list,
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
            )

    def _stage1_snr_check(self, signal: np.ndarray) -> Dict[str, Any]:
        """Stage 1: Quick SNR Check."""
        try:
            # Estimate signal power (using RMS)
            signal_power = np.mean(signal**2)

            # Estimate noise power (using high-frequency components)
            # Simple approach: use standard deviation of differences
            signal_diff = np.diff(signal)
            noise_power = np.var(signal_diff)

            # Calculate SNR in dB
            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
            else:
                snr_db = float("inf")

            passed = bool(snr_db >= self.thresholds["snr_min_db"])

            return {
                "passed": passed,
                "snr_db": snr_db,
                "signal_power": signal_power,
                "noise_power": noise_power,
            }

        except Exception as e:
            return {
                "passed": False,
                "snr_db": -float("inf"),
                "signal_power": 0.0,
                "noise_power": float("inf"),
                "error": str(e),
            }

    def _stage2_statistical_screen(self, signal: np.ndarray) -> Dict[str, Any]:
        """Stage 2: Statistical Screen using configurable thresholds."""
        try:
            # Check for statistical anomalies
            mean_val = np.mean(signal)
            std_val = np.std(signal)

            # Check for outliers using configurable std_factor
            outlier_threshold = self.thresholds["outlier_std_factor"] * std_val
            outliers = np.abs(signal - mean_val) > outlier_threshold
            outlier_ratio = np.sum(outliers) / len(signal)

            # Check for constant values (potential sensor failure)
            unique_values = len(np.unique(signal))
            constant_ratio = 1.0 - (unique_values / len(signal))

            # Check for sudden jumps (potential artifacts) using configurable std_factor
            signal_diff = np.abs(np.diff(signal))
            jump_threshold = self.thresholds["jump_std_factor"] * np.std(signal_diff)
            jumps = signal_diff > jump_threshold
            jump_ratio = np.sum(jumps) / len(signal_diff)

            # Pass if all ratios are below configurable thresholds
            passed = bool(
                outlier_ratio < self.thresholds["outlier_max_ratio"]
                and constant_ratio < self.thresholds["constant_max_ratio"]
                and jump_ratio < self.thresholds["jump_max_ratio"]
            )

            return {
                "passed": passed,
                "outlier_ratio": outlier_ratio,
                "constant_ratio": constant_ratio,
                "jump_ratio": jump_ratio,
                "mean": mean_val,
                "std": std_val,
            }

        except Exception as e:
            return {
                "passed": False,
                "outlier_ratio": 1.0,
                "constant_ratio": 1.0,
                "jump_ratio": 1.0,
                "error": str(e),
            }

    def _stage3_signal_specific_screen(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Stage 3: Signal-Specific Screen using SignalQualityIndex methods.

        Applies signal-type-specific quality metrics from vitalDSP's
        signal_quality_assessment module based on configuration.
        """
        try:
            # Initialize warnings list
            warnings_list = []

            # Create quality index for this signal
            if self.quality_index is None:
                self.quality_index = SignalQualityIndex(signal)
            else:
                # Update with new signal
                self.quality_index.signal = signal

            # Calculate window and step sizes from configuration
            window_size = min(
                int(self.segment_duration * self.sampling_rate / 4), len(signal) // 2
            )
            step_size = int(window_size * (1 - self.overlap_ratio))

            # Ensure minimum sizes
            window_size = max(window_size, 50)
            step_size = max(step_size, 25)

            # Collect SQI metrics based on signal type
            sqi_scores = {}
            passed_checks = []

            # Common metrics for all signal types
            # SNR SQI
            snr_sqi_result = self.quality_index.snr_sqi(
                window_size=window_size,
                step_size=step_size,
                threshold=self.thresholds["snr_min_db"],
                threshold_type="above",
            )
            sqi_scores["snr_sqi"] = self._extract_sqi_score(snr_sqi_result)

            # Baseline wander SQI
            baseline_wander_result = self.quality_index.baseline_wander_sqi(
                window_size=window_size,
                step_size=step_size,
                threshold=self.thresholds["baseline_wander_min"],
                threshold_type="above",
            )
            sqi_scores["baseline_wander"] = self._extract_sqi_score(
                baseline_wander_result
            )
            passed_checks.append(
                sqi_scores["baseline_wander"] >= self.thresholds["baseline_wander_min"]
            )

            # Amplitude variability SQI
            amp_var_result = self.quality_index.amplitude_variability_sqi(
                window_size=window_size,
                step_size=step_size,
                threshold=self.thresholds["amplitude_variability_min"],
                threshold_type="above",
            )
            sqi_scores["amplitude_variability"] = self._extract_sqi_score(
                amp_var_result
            )
            passed_checks.append(
                sqi_scores["amplitude_variability"]
                >= self.thresholds["amplitude_variability_min"]
            )

            # Signal-specific metrics
            if self.signal_type == "ecg":
                # ECG-specific SQI methods
                zero_cross_result = self.quality_index.zero_crossing_sqi(
                    window_size=window_size,
                    step_size=step_size,
                    threshold=self.thresholds["zero_crossing_min"],
                    threshold_type="above",
                )
                sqi_scores["zero_crossing"] = self._extract_sqi_score(zero_cross_result)
                passed_checks.append(
                    sqi_scores["zero_crossing"] >= self.thresholds["zero_crossing_min"]
                )

                # Heart rate variability SQI (only for ECG signals with RR intervals)
                try:
                    # Extract RR intervals from ECG signal
                    from vitalDSP.physiological_features.waveform import (
                        WaveformMorphology,
                    )

                    # Create waveform morphology object for peak detection
                    wm = WaveformMorphology(
                        waveform=signal, fs=self.sampling_rate, signal_type="ECG"
                    )

                    # Get R-peaks
                    r_peaks = wm.r_peaks

                    if len(r_peaks) > 1:
                        # Calculate RR intervals in milliseconds
                        rr_intervals = np.diff(r_peaks) * 1000 / self.sampling_rate

                        # Filter out unrealistic RR intervals (300-2000 ms)
                        valid_rr = rr_intervals[
                            (rr_intervals >= 300) & (rr_intervals <= 2000)
                        ]

                        if len(valid_rr) > 2:  # Need at least 3 intervals for HRV
                            # Calculate HRV SQI
                            hrv_result = self.quality_index.heart_rate_variability_sqi(
                                rr_intervals=valid_rr,
                                window_size=window_size,
                                step_size=step_size,
                                threshold=self.thresholds["hrv_min"],
                                threshold_type="above",
                            )
                            sqi_scores["hrv"] = self._extract_sqi_score(hrv_result)
                            passed_checks.append(
                                sqi_scores["hrv"] >= self.thresholds["hrv_min"]
                            )
                        else:
                            warnings_list.append(
                                "Insufficient valid RR intervals for HRV analysis"
                            )
                            sqi_scores["hrv"] = 0.3  # Low score for insufficient data
                            passed_checks.append(False)
                    else:
                        warnings_list.append("No R-peaks detected for HRV analysis")
                        sqi_scores["hrv"] = 0.3  # Low score for no peaks
                        passed_checks.append(False)

                except Exception as e:
                    warnings_list.append(f"HRV calculation failed: {str(e)}")
                    sqi_scores["hrv"] = 0.3  # Low score for failed calculation
                    passed_checks.append(False)

            elif self.signal_type == "ppg":
                # PPG-specific SQI method
                ppg_quality_result = self.quality_index.ppg_signal_quality_sqi(
                    window_size=window_size,
                    step_size=step_size,
                    threshold=self.thresholds["ppg_quality_min"],
                    threshold_type="above",
                )
                sqi_scores["ppg_quality"] = self._extract_sqi_score(ppg_quality_result)
                passed_checks.append(
                    sqi_scores["ppg_quality"] >= self.thresholds["ppg_quality_min"]
                )

                # Waveform similarity SQI
                waveform_sim_result = self.quality_index.waveform_similarity_sqi(
                    window_size=window_size,
                    step_size=step_size,
                    threshold=self.thresholds["waveform_similarity_min"],
                    threshold_type="above",
                )
                sqi_scores["waveform_similarity"] = self._extract_sqi_score(
                    waveform_sim_result
                )
                passed_checks.append(
                    sqi_scores["waveform_similarity"]
                    >= self.thresholds["waveform_similarity_min"]
                )

                # Heart rate variability SQI for PPG signals (using systolic peaks)
                try:
                    # Extract RR intervals from PPG signal using systolic peaks
                    from vitalDSP.physiological_features.waveform import (
                        WaveformMorphology,
                    )

                    # Create waveform morphology object for peak detection
                    wm = WaveformMorphology(
                        waveform=signal, fs=self.sampling_rate, signal_type="PPG"
                    )

                    # Get systolic peaks (equivalent to R-peaks for PPG)
                    systolic_peaks = wm.systolic_peaks

                    if len(systolic_peaks) > 1:
                        # Calculate RR intervals in milliseconds
                        rr_intervals = (
                            np.diff(systolic_peaks) * 1000 / self.sampling_rate
                        )

                        # Filter out unrealistic RR intervals (300-2000 ms)
                        valid_rr = rr_intervals[
                            (rr_intervals >= 300) & (rr_intervals <= 2000)
                        ]

                        if len(valid_rr) > 2:  # Need at least 3 intervals for HRV
                            # Calculate HRV SQI
                            hrv_result = self.quality_index.heart_rate_variability_sqi(
                                rr_intervals=valid_rr,
                                window_size=window_size,
                                step_size=step_size,
                                threshold=self.thresholds["hrv_min"],
                                threshold_type="above",
                            )
                            sqi_scores["hrv"] = self._extract_sqi_score(hrv_result)
                            passed_checks.append(
                                sqi_scores["hrv"] >= self.thresholds["hrv_min"]
                            )
                        else:
                            warnings_list.append(
                                "Insufficient valid RR intervals for PPG HRV analysis"
                            )
                            sqi_scores["hrv"] = 0.3  # Low score for insufficient data
                            passed_checks.append(False)
                    else:
                        warnings_list.append(
                            "No systolic peaks detected for PPG HRV analysis"
                        )
                        sqi_scores["hrv"] = 0.3  # Low score for no peaks
                        passed_checks.append(False)

                except Exception as e:
                    warnings_list.append(f"PPG HRV calculation failed: {str(e)}")
                    sqi_scores["hrv"] = 0.3  # Low score for failed calculation
                    passed_checks.append(False)

            elif self.signal_type == "eeg":
                # EEG-specific SQI methods
                eeg_band_result = self.quality_index.eeg_band_power_sqi(
                    window_size=window_size,
                    step_size=step_size,
                    threshold=self.thresholds["eeg_band_power_min"],
                    threshold_type="above",
                )
                sqi_scores["eeg_band_power"] = self._extract_sqi_score(eeg_band_result)
                passed_checks.append(
                    sqi_scores["eeg_band_power"]
                    >= self.thresholds["eeg_band_power_min"]
                )

                # Signal entropy SQI
                entropy_result = self.quality_index.signal_entropy_sqi(
                    window_size=window_size,
                    step_size=step_size,
                    threshold=self.thresholds["signal_entropy_min"],
                    threshold_type="above",
                )
                sqi_scores["signal_entropy"] = self._extract_sqi_score(entropy_result)
                passed_checks.append(
                    sqi_scores["signal_entropy"]
                    >= self.thresholds["signal_entropy_min"]
                )

                # Skewness and kurtosis SQI
                skewness_result = self.quality_index.skewness_sqi(
                    window_size=window_size,
                    step_size=step_size,
                    threshold=self.thresholds["skewness_min"],
                    threshold_type="above",
                )
                sqi_scores["skewness"] = self._extract_sqi_score(skewness_result)
                passed_checks.append(
                    sqi_scores["skewness"] >= self.thresholds["skewness_min"]
                )

                kurtosis_result = self.quality_index.kurtosis_sqi(
                    window_size=window_size,
                    step_size=step_size,
                    threshold=self.thresholds["kurtosis_min"],
                    threshold_type="above",
                )
                sqi_scores["kurtosis"] = self._extract_sqi_score(kurtosis_result)
                passed_checks.append(
                    sqi_scores["kurtosis"] >= self.thresholds["kurtosis_min"]
                )

            else:
                # Generic signal - use common metrics only
                energy_result = self.quality_index.energy_sqi(
                    window_size=window_size,
                    step_size=step_size,
                    threshold=self.thresholds["energy_min"],
                    threshold_type="above",
                )
                sqi_scores["energy"] = self._extract_sqi_score(energy_result)
                passed_checks.append(
                    sqi_scores["energy"] >= self.thresholds["energy_min"]
                )

            # Legacy metrics (for backward compatibility)
            artifact_ratio = 0.0  # Placeholder
            baseline_drift = self._calculate_baseline_drift(signal)
            peak_rate = self._calculate_peak_detection_rate(signal)
            freq_score = self._calculate_frequency_score(signal)
            temporal_score = self._calculate_temporal_consistency(signal)

            # Add legacy checks
            passed_checks.extend(
                [
                    baseline_drift <= self.thresholds["baseline_max_drift"],
                    peak_rate >= self.thresholds["peak_detection_min_rate"],
                    freq_score >= self.thresholds["frequency_score_min"],
                    temporal_score >= self.thresholds["temporal_consistency_min"],
                ]
            )

            # Overall pass decision: all checks must pass
            passed = bool(all(passed_checks))

            # Calculate overall quality score from SQI metrics
            overall_sqi_score = (
                np.mean(list(sqi_scores.values())) if sqi_scores else 0.0
            )

            return {
                "passed": passed,
                "quality_score": overall_sqi_score,
                "sqi_scores": sqi_scores,
                "artifact_ratio": artifact_ratio,
                "baseline_drift": baseline_drift,
                "peak_detection_rate": peak_rate,
                "frequency_score": freq_score,
                "temporal_consistency": temporal_score,
                "warnings": warnings_list,
            }

        except Exception as e:
            logger.warning(f"Stage 3 screening failed: {e}")
            return {
                "passed": False,
                "quality_score": 0.0,
                "sqi_scores": {},
                "artifact_ratio": 1.0,
                "baseline_drift": float("inf"),
                "peak_detection_rate": 0.0,
                "frequency_score": 0.0,
                "temporal_consistency": 0.0,
                "error": str(e),
            }

    def _extract_sqi_score(self, sqi_result) -> float:
        """
        Extract normalized score from SignalQualityIndex result.

        SignalQualityIndex methods return tuple of (scores_array, indices_array).
        We extract the mean of the scores array as a single quality score.
        """
        if isinstance(sqi_result, tuple) and len(sqi_result) > 0:
            scores_array = sqi_result[0]
            if isinstance(scores_array, np.ndarray) and len(scores_array) > 0:
                # Return mean score, normalized to 0-1 range
                return float(np.mean(scores_array))
        return 0.0

    def _calculate_baseline_drift(self, signal: np.ndarray) -> float:
        """Calculate baseline drift magnitude."""
        # Simple approach: use linear trend
        x = np.arange(len(signal))
        coeffs = np.polyfit(x, signal, 1)
        trend_line = np.polyval(coeffs, x)
        drift = np.max(np.abs(trend_line - np.mean(signal)))
        return drift

    def _calculate_peak_detection_rate(self, signal: np.ndarray) -> float:
        """Calculate peak detection rate (signal-specific)."""
        if self.signal_type == "ecg":
            # For ECG, look for R-peaks
            return self._detect_ecg_peaks(signal)
        elif self.signal_type == "ppg":
            # For PPG, look for pulse peaks
            return self._detect_ppg_peaks(signal)
        else:
            # Generic peak detection
            return self._detect_generic_peaks(signal)

    def _detect_ecg_peaks(self, signal: np.ndarray) -> float:
        """Detect ECG R-peaks using configurable parameters."""
        # Simple peak detection based on local maxima
        from scipy.signal import find_peaks

        # Find peaks with configurable minimum distance
        min_distance = int(
            self.thresholds["ecg_min_peak_distance_factor"] * self.sampling_rate
        )
        peaks, _ = find_peaks(signal, distance=min_distance, height=np.mean(signal))

        # Expected number of peaks based on duration and configurable heart rate
        duration = len(signal) / self.sampling_rate
        expected_heart_rate_hz = self.thresholds["ecg_expected_heart_rate_bpm"] / 60.0
        expected_peaks = duration * expected_heart_rate_hz

        if expected_peaks > 0:
            return min(len(peaks) / expected_peaks, 1.0)
        return 0.0

    def _detect_ppg_peaks(self, signal: np.ndarray) -> float:
        """Detect PPG pulse peaks using configurable parameters."""
        from scipy.signal import find_peaks

        # Find peaks with configurable minimum distance
        min_distance = int(
            self.thresholds["ppg_min_peak_distance_factor"] * self.sampling_rate
        )
        peaks, _ = find_peaks(signal, distance=min_distance, height=np.mean(signal))

        # Expected number of peaks based on duration and configurable pulse rate
        duration = len(signal) / self.sampling_rate
        expected_pulse_rate_hz = self.thresholds["ppg_expected_pulse_rate_bpm"] / 60.0
        expected_peaks = duration * expected_pulse_rate_hz

        if expected_peaks > 0:
            return min(len(peaks) / expected_peaks, 1.0)
        return 0.0

    def _detect_generic_peaks(self, signal: np.ndarray) -> float:
        """Generic peak detection."""
        from scipy.signal import find_peaks

        # Find peaks with adaptive threshold
        threshold = np.mean(signal) + 0.5 * np.std(signal)
        peaks, _ = find_peaks(signal, height=threshold)

        # Return ratio of peaks found
        return min(len(peaks) / (len(signal) / 100), 1.0)

    def _calculate_frequency_score(self, signal: np.ndarray) -> float:
        """Calculate frequency domain quality score."""
        try:
            from scipy.fft import fft, fftfreq

            # Compute FFT
            fft_signal = fft(signal)
            freqs = fftfreq(len(signal), 1 / self.sampling_rate)

            # Focus on positive frequencies
            positive_freqs = freqs[: len(freqs) // 2]
            positive_fft = np.abs(fft_signal[: len(fft_signal) // 2])

            # Calculate spectral centroid (indicator of signal quality)
            if np.sum(positive_fft) > 0:
                spectral_centroid = np.sum(positive_freqs * positive_fft) / np.sum(
                    positive_fft
                )
                # Normalize to 0-1 range
                score = min(spectral_centroid / (self.sampling_rate / 4), 1.0)
            else:
                score = 0.0

            return score

        except Exception:
            return 0.0

    def _calculate_temporal_consistency(self, signal: np.ndarray) -> float:
        """Calculate temporal consistency score."""
        try:
            # Calculate autocorrelation at lag 1
            autocorr = np.corrcoef(signal[:-1], signal[1:])[0, 1]

            # Calculate local variance consistency
            window_size = min(100, len(signal) // 10)
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

    def _calculate_overall_metrics(
        self,
        snr_result: Dict[str, Any],
        stats_result: Dict[str, Any],
        signal_result: Dict[str, Any],
    ) -> QualityMetrics:
        """Calculate overall quality metrics."""
        # Calculate overall quality score
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

        # Signal-specific score (0-1) - extract mean from SNR SQI result
        quality_score_data = signal_result.get("quality_score", 0.0)
        if isinstance(quality_score_data, tuple) and len(quality_score_data) > 0:
            # Extract the first element (array of scores) and take mean
            signal_score = (
                np.mean(quality_score_data[0])
                if len(quality_score_data[0]) > 0
                else 0.0
            )
        else:
            signal_score = 0.0
        scores.append(signal_score)

        # Calculate weighted average
        overall_quality = np.mean(scores)

        # Determine quality level
        if overall_quality > 0.8:
            quality_level = QualityLevel.EXCELLENT
            recommendation = "Full processing recommended"
        elif overall_quality > 0.6:
            quality_level = QualityLevel.GOOD
            recommendation = "Standard processing recommended"
        elif overall_quality > 0.4:
            quality_level = QualityLevel.FAIR
            recommendation = "Conservative processing recommended"
        elif overall_quality > 0.2:
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
        }


# Configuration dataclass for compatibility
@dataclass
class QualityScreeningConfig:
    """
    Configuration for quality screening (compatibility wrapper).

    This provides a dataclass interface that maps to QualityScreener constructor parameters.
    """

    segment_duration: float = 10.0
    overlap: float = 0.5  # Mapped to overlap_ratio
    snr_threshold: float = 10.0  # Mapped to snr_min_db
    max_workers: int = 2
    min_quality_score: float = 0.7  # Mapped to overall_quality_min
    conservative: bool = True


class SignalQualityScreener(QualityScreener):
    """
    Compatibility wrapper for QualityScreener that accepts QualityScreeningConfig.

    This allows using the documented API from Phase 4 documentation while
    maintaining compatibility with the existing QualityScreener implementation.
    """

    def __init__(self, config: Optional[QualityScreeningConfig] = None, **kwargs):
        """
        Initialize SignalQualityScreener with config.

        Args:
            config: QualityScreeningConfig object (optional)
            **kwargs: Additional parameters passed to QualityScreener
        """
        if config is None:
            config = QualityScreeningConfig()

        # Map config parameters to QualityScreener parameters
        super().__init__(
            segment_duration=config.segment_duration,
            overlap_ratio=config.overlap,
            max_workers=config.max_workers,
            snr_min_db=config.snr_threshold,
            overall_quality_min=config.min_quality_score,
            **kwargs,
        )


# Example usage and tests
if __name__ == "__main__":
    print("Quality Screener Module")
    print("=" * 50)
    print("\nThis module provides 3-stage quality screening")
    print("for physiological signals.")
    print("\nFeatures:")
    print("  - Stage 1: Quick SNR Check")
    print("  - Stage 2: Statistical Screen")
    print("  - Stage 3: Signal-Specific Screen")
    print("  - Parallel processing support")
    print("  - Quality-aware recommendations")
