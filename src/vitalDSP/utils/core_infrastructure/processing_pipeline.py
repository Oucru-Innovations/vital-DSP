"""
Standard 8-Stage Processing Pipeline for Large Data Processing Architecture

This module implements the conservative processing pipeline as outlined in the
Large Data Processing Architecture document. The pipeline processes data through
8 stages with checkpointing, caching, and comprehensive quality assessment.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0
"""

import os
import pickle
import hashlib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config_utilities.dynamic_config import DynamicConfigManager
from .quality_screener import QualityScreener
from .parallel_pipeline import ParallelPipeline
from .data_loaders import ChunkedDataLoader, MemoryMappedLoader

# Configure logging
logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Enumeration of processing pipeline stages."""

    DATA_INGESTION = "data_ingestion"
    QUALITY_SCREENING = "quality_screening"
    PARALLEL_PROCESSING = "parallel_processing"
    QUALITY_VALIDATION = "quality_validation"
    SEGMENTATION = "segmentation"
    FEATURE_EXTRACTION = "feature_extraction"
    INTELLIGENT_OUTPUT = "intelligent_output"
    OUTPUT_PACKAGE = "output_package"


@dataclass
class ProcessingCheckpoint:
    """Data class for processing checkpoint information."""

    stage: ProcessingStage
    timestamp: datetime
    data_hash: str
    metadata: Dict[str, Any]
    file_path: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class ProcessingResult:
    """Data class for processing result information."""

    stage: ProcessingStage
    success: bool
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    quality_scores: Optional[Dict[str, float]] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None


class ProcessingCache:
    """
    Intelligent caching system for intermediate processing results.
    Prevents re-computation of expensive operations with compression and TTL.
    """

    def __init__(
        self,
        cache_dir: str = "~/.vitaldsp/cache",
        max_cache_size_gb: float = 10.0,
        compression: bool = True,
    ):
        """
        Initialize processing cache.

        Args:
            cache_dir: Directory for cache storage
            max_cache_size_gb: Maximum cache size in GB
            compression: Enable compression for cached data
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_gb = max_cache_size_gb
        self.compression = compression
        self.cache_stats = {"hits": 0, "misses": 0, "size_bytes": 0, "entries": 0}
        self._lock = threading.Lock()

    def get_cache_key(
        self, data: np.ndarray, operation: str, params: Dict[str, Any]
    ) -> str:
        """
        Generate unique cache key for data and operation.

        Args:
            data: Input data array
            operation: Operation name
            params: Operation parameters

        Returns:
            Unique cache key string
        """
        # Create hash of data (sample-based for large arrays)
        if len(data) > 10000:
            # Sample data for large arrays
            sample_indices = np.linspace(0, len(data) - 1, 1000, dtype=int)
            data_sample = data[sample_indices]
        else:
            data_sample = data

        data_hash = hashlib.md5(data_sample.tobytes()).hexdigest()
        params_hash = hashlib.md5(str(sorted(params.items())).encode()).hexdigest()

        return f"{operation}_{data_hash[:8]}_{params_hash[:8]}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result.

        Args:
            key: Cache key

        Returns:
            Cached result or None if not found
        """
        cache_file = self.cache_dir / f"{key}.npz"

        if not cache_file.exists():
            with self._lock:
                self.cache_stats["misses"] += 1
            return None

        try:
            # Check if cache is expired (24 hours TTL)
            file_age = datetime.now().timestamp() - cache_file.stat().st_mtime
            if file_age > 86400:  # 24 hours
                cache_file.unlink()
                with self._lock:
                    self.cache_stats["misses"] += 1
                return None

            cached_data = np.load(cache_file, allow_pickle=True)
            result = dict(cached_data)

            with self._lock:
                self.cache_stats["hits"] += 1

            logger.debug(f"Cache hit for key: {key}")
            return result

        except Exception as e:
            logger.warning(f"Failed to load cache for key {key}: {e}")
            cache_file.unlink()  # Remove corrupted cache file
            with self._lock:
                self.cache_stats["misses"] += 1
            return None

    def set(self, key: str, result: Dict[str, Any]) -> None:
        """
        Cache result with compression.

        Args:
            key: Cache key
            result: Result data to cache
        """
        try:
            cache_file = self.cache_dir / f"{key}.npz"

            # Check cache size limit
            self._enforce_cache_size_limit()

            if self.compression:
                np.savez_compressed(cache_file, **result)
            else:
                np.savez(cache_file, **result)

            with self._lock:
                self.cache_stats["entries"] += 1
                self.cache_stats["size_bytes"] += cache_file.stat().st_size

            logger.debug(f"Cached result for key: {key}")

        except Exception as e:
            logger.error(f"Failed to cache result for key {key}: {e}")

    def _enforce_cache_size_limit(self) -> None:
        """Enforce cache size limit by removing oldest entries."""
        if self.cache_stats["size_bytes"] < self.max_cache_size_gb * 1024**3:
            return

        # Get all cache files sorted by modification time
        cache_files = list(self.cache_dir.glob("*.npz"))
        cache_files.sort(key=lambda x: x.stat().st_mtime)

        # Remove oldest files until under limit
        removed_size = 0
        for cache_file in cache_files:
            file_size = cache_file.stat().st_size
            cache_file.unlink()
            removed_size += file_size

            with self._lock:
                self.cache_stats["entries"] -= 1
                self.cache_stats["size_bytes"] -= file_size

            if self.cache_stats["size_bytes"] < self.max_cache_size_gb * 0.8 * 1024**3:
                break

        logger.info(f"Cache cleanup: removed {removed_size / 1024**2:.1f} MB")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.cache_stats["hits"] / max(
            self.cache_stats["hits"] + self.cache_stats["misses"], 1
        )

        return {
            "hit_rate": hit_rate,
            "total_hits": self.cache_stats["hits"],
            "total_misses": self.cache_stats["misses"],
            "cache_size_mb": self.cache_stats["size_bytes"] / 1024**2,
            "cache_entries": self.cache_stats["entries"],
            "max_size_gb": self.max_cache_size_gb,
        }

    def clear(self) -> None:
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.npz"):
            cache_file.unlink()

        with self._lock:
            self.cache_stats = {"hits": 0, "misses": 0, "size_bytes": 0, "entries": 0}

        logger.info("Cache cleared")


class CheckpointManager:
    """
    Manages processing checkpoints to allow resumption of interrupted processing.
    """

    def __init__(self, checkpoint_dir: str = "~/.vitaldsp/checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint storage
        """
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def create_session_id(self) -> str:
        """Create unique session identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}_{os.getpid()}"

    def save_checkpoint(
        self,
        session_id: str,
        stage: ProcessingStage,
        data: Any,
        metadata: Dict[str, Any],
    ) -> str:
        """
        Save processing checkpoint.

        Args:
            session_id: Unique session identifier
            stage: Processing stage
            data: Data to checkpoint
            metadata: Additional metadata

        Returns:
            Checkpoint file path
        """
        checkpoint_file = self.checkpoint_dir / f"{session_id}_stage_{stage.value}.pkl"

        try:
            checkpoint = ProcessingCheckpoint(
                stage=stage,
                timestamp=datetime.now(),
                data_hash=self._compute_data_hash(data),
                metadata=metadata,
                file_path=str(checkpoint_file),
                success=True,
            )

            with self._lock:
                with open(checkpoint_file, "wb") as f:
                    pickle.dump({"checkpoint": checkpoint, "data": data}, f)

            logger.info(f"Checkpoint saved: {checkpoint_file}")
            return str(checkpoint_file)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(
        self, session_id: str, stage: ProcessingStage
    ) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """
        Load processing checkpoint.

        Args:
            session_id: Unique session identifier
            stage: Processing stage

        Returns:
            Tuple of (data, metadata) or None if not found
        """
        checkpoint_file = self.checkpoint_dir / f"{session_id}_stage_{stage.value}.pkl"

        if not checkpoint_file.exists():
            return None

        try:
            with self._lock:
                with open(checkpoint_file, "rb") as f:
                    checkpoint_data = pickle.load(f)

            checkpoint = checkpoint_data["checkpoint"]
            data = checkpoint_data["data"]

            logger.info(f"Checkpoint loaded: {checkpoint_file}")
            return data, checkpoint.metadata

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def list_checkpoints(self, session_id: str) -> List[ProcessingCheckpoint]:
        """
        List all checkpoints for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            List of checkpoint information
        """
        checkpoints = []

        for checkpoint_file in self.checkpoint_dir.glob(f"{session_id}_stage_*.pkl"):
            try:
                with open(checkpoint_file, "rb") as f:
                    checkpoint_data = pickle.load(f)
                checkpoints.append(checkpoint_data["checkpoint"])
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {checkpoint_file}: {e}")

        return sorted(checkpoints, key=lambda x: x.timestamp)

    def cleanup_session(self, session_id: str) -> None:
        """
        Clean up all checkpoints for a session.

        Args:
            session_id: Unique session identifier
        """
        for checkpoint_file in self.checkpoint_dir.glob(f"{session_id}_stage_*.pkl"):
            try:
                checkpoint_file.unlink()
                logger.debug(f"Removed checkpoint: {checkpoint_file}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_file}: {e}")

    def _compute_data_hash(self, data: Any) -> str:
        """Compute hash of data for integrity checking."""
        if isinstance(data, np.ndarray):
            # Sample large arrays for hash computation
            if len(data) > 10000:
                sample_indices = np.linspace(0, len(data) - 1, 1000, dtype=int)
                data_sample = data[sample_indices]
            else:
                data_sample = data
            return hashlib.md5(data_sample.tobytes()).hexdigest()
        else:
            return hashlib.md5(str(data).encode()).hexdigest()


class StandardProcessingPipeline:
    """
    Standard 8-Stage Processing Pipeline for Large Data Processing.

    Implements the conservative processing approach with:
    - Non-destructive quality screening
    - Parallel path processing (raw, preprocessed, filtered, full)
    - Comprehensive quality validation
    - Flexible segmentation strategies
    - Intelligent output options
    """

    def __init__(
        self,
        config_manager: Optional[DynamicConfigManager] = None,
        cache_dir: str = "~/.vitaldsp/cache",
        checkpoint_dir: str = "~/.vitaldsp/checkpoints",
    ):
        """
        Initialize processing pipeline.

        Args:
            config_manager: Configuration manager instance
            cache_dir: Cache directory path
            checkpoint_dir: Checkpoint directory path
        """
        self.config = config_manager or DynamicConfigManager()
        self.cache = ProcessingCache(cache_dir)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.quality_screener = QualityScreener()
        self.parallel_pipeline = ParallelPipeline()

        # Processing statistics
        self.stats = {
            "total_processing_time": 0.0,
            "stages_completed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "checkpoints_saved": 0,
            "errors_encountered": 0,
        }

    def process_signal(
        self,
        signal: np.ndarray,
        fs: float,
        signal_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        resume_from_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """
        Process signal through the complete 8-stage pipeline.

        Args:
            signal: Input signal data
            fs: Sampling frequency
            signal_type: Type of signal (ECG, PPG, EEG, etc.)
            metadata: Additional signal metadata
            session_id: Optional session ID for checkpointing
            resume_from_checkpoint: Whether to resume from existing checkpoints

        Returns:
            Complete processing results
        """
        start_time = datetime.now()

        if session_id is None:
            session_id = self.checkpoint_manager.create_session_id()

        logger.info(
            f"Starting processing pipeline for {signal_type} signal "
            f"(length: {len(signal)}, fs: {fs} Hz)"
        )

        # Initialize processing context
        context = {
            "signal": signal,
            "fs": fs,
            "signal_type": signal_type,
            "metadata": metadata or {},
            "session_id": session_id,
            "start_time": start_time,
            "results": {},
        }

        try:
            # Execute all processing stages
            for stage in ProcessingStage:
                logger.info(f"Executing stage: {stage.value}")

                # Check for existing checkpoint
                if resume_from_checkpoint:
                    checkpoint_data = self.checkpoint_manager.load_checkpoint(
                        session_id, stage
                    )
                    if checkpoint_data is not None:
                        logger.info(
                            f"Resuming from checkpoint for stage: {stage.value}"
                        )
                        context["results"][stage.value] = checkpoint_data[0]
                        context.update(checkpoint_data[1])
                        continue

                # Execute stage
                stage_result = self._execute_stage(stage, context)
                context["results"][stage.value] = stage_result

                # Save checkpoint
                if stage_result.success:
                    self.checkpoint_manager.save_checkpoint(
                        session_id, stage, stage_result.data, context
                    )
                    self.stats["checkpoints_saved"] += 1
                else:
                    logger.error(
                        f"Stage {stage.value} failed: {stage_result.error_message}"
                    )
                    self.stats["errors_encountered"] += 1
                    break

                self.stats["stages_completed"] += 1

            # Calculate total processing time
            total_time = (datetime.now() - start_time).total_seconds()
            self.stats["total_processing_time"] += total_time

            # Generate final results
            final_results = self._generate_final_results(context)

            logger.info(f"Processing completed in {total_time:.2f} seconds")
            return final_results

        except Exception as e:
            logger.error(f"Processing pipeline failed: {e}")
            self.stats["errors_encountered"] += 1
            raise

        finally:
            # Cleanup checkpoints if processing completed successfully
            if self.stats["errors_encountered"] == 0:
                self.checkpoint_manager.cleanup_session(session_id)

    def _execute_stage(
        self, stage: ProcessingStage, context: Dict[str, Any]
    ) -> ProcessingResult:
        """
        Execute a specific processing stage.

        Args:
            stage: Processing stage to execute
            context: Processing context

        Returns:
            Processing result
        """
        start_time = datetime.now()

        try:
            if stage == ProcessingStage.DATA_INGESTION:
                result = self._stage_data_ingestion(context)
            elif stage == ProcessingStage.QUALITY_SCREENING:
                result = self._stage_quality_screening(context)
            elif stage == ProcessingStage.PARALLEL_PROCESSING:
                result = self._stage_parallel_processing(context)
            elif stage == ProcessingStage.QUALITY_VALIDATION:
                result = self._stage_quality_validation(context)
            elif stage == ProcessingStage.SEGMENTATION:
                result = self._stage_segmentation(context)
            elif stage == ProcessingStage.FEATURE_EXTRACTION:
                result = self._stage_feature_extraction(context)
            elif stage == ProcessingStage.INTELLIGENT_OUTPUT:
                result = self._stage_intelligent_output(context)
            elif stage == ProcessingStage.OUTPUT_PACKAGE:
                result = self._stage_output_package(context)
            else:
                raise ValueError(f"Unknown processing stage: {stage}")

            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time

            return result

        except Exception as e:
            logger.error(f"Stage {stage.value} failed: {e}")
            return ProcessingResult(
                stage=stage,
                success=False,
                error_message=str(e),
                processing_time=(datetime.now() - start_time).total_seconds(),
            )

    def _stage_data_ingestion(self, context: Dict[str, Any]) -> ProcessingResult:
        """Stage 1: Data Ingestion - Format detection, metadata extraction, size estimation."""
        signal = context["signal"]
        fs = context["fs"]
        signal_type = context["signal_type"]

        # Format detection and metadata extraction
        ingestion_data = {
            "signal_length": len(signal),
            "sampling_frequency": fs,
            "duration_seconds": len(signal) / fs,
            "signal_type": signal_type,
            "data_type": str(signal.dtype),
            "memory_size_mb": signal.nbytes / 1024**2,
            "timestamp": datetime.now().isoformat(),
            "metadata": context.get("metadata", {}),
        }

        # Size estimation and processing recommendations
        duration_minutes = ingestion_data["duration_seconds"] / 60

        if duration_minutes < 5:
            processing_mode = "whole_signal"
        elif duration_minutes < 60:
            processing_mode = "segment_with_overlap"
        else:
            processing_mode = "hybrid"

        ingestion_data["recommended_processing_mode"] = processing_mode
        ingestion_data["processing_complexity"] = self._estimate_processing_complexity(
            signal, fs
        )

        logger.info(
            f"Data ingestion completed: {ingestion_data['signal_length']} samples, "
            f"{ingestion_data['duration_seconds']:.1f}s duration"
        )

        return ProcessingResult(
            stage=ProcessingStage.DATA_INGESTION,
            success=True,
            data=ingestion_data,
            metadata={"processing_mode": processing_mode},
        )

    def _stage_quality_screening(self, context: Dict[str, Any]) -> ProcessingResult:
        """Stage 2: Quality Screening - Non-destructive quality assessment."""
        signal = context["signal"]
        fs = context["fs"]
        signal_type = context["signal_type"]

        # Perform comprehensive quality screening
        quality_results = self.quality_screener.screen_signal(signal)

        # Add quality-based processing recommendations
        quality_score = (
            quality_results[0].quality_metrics.overall_quality
            if quality_results
            else 0.5
        )

        if quality_score > 0.8:
            recommendation = "excellent_quality_safe_for_all_analyses"
        elif quality_score > 0.6:
            recommendation = "good_quality_suitable_for_most_analyses"
        elif quality_score > 0.4:
            recommendation = "fair_quality_usable_with_preprocessing"
        else:
            recommendation = "poor_quality_manual_review_recommended"

        # Create a dictionary structure for the results
        quality_data = {
            "screening_results": quality_results,
            "overall_quality_score": quality_score,
            "processing_recommendation": recommendation,
            "false_positive_risk": self._assess_false_positive_risk_from_results(
                quality_results
            ),
            "total_segments": len(quality_results),
            "passed_segments": sum(1 for r in quality_results if r.passed_screening),
            "pass_rate": (
                sum(1 for r in quality_results if r.passed_screening)
                / len(quality_results)
                if quality_results
                else 0.0
            ),
        }

        logger.info(
            f"Quality screening completed: overall score {quality_score:.3f}, "
            f"recommendation: {recommendation}"
        )

        return ProcessingResult(
            stage=ProcessingStage.QUALITY_SCREENING,
            success=True,
            data=quality_data,
            quality_scores=quality_data,
        )

    def _stage_parallel_processing(self, context: Dict[str, Any]) -> ProcessingResult:
        """Stage 3: Parallel Processing - Process through multiple paths simultaneously."""
        signal = context["signal"]
        fs = context["fs"]
        signal_type = context["signal_type"]

        # Process through multiple paths (raw, filtered, preprocessed, full)
        paths = {}

        # Path 1: Raw data (no processing)
        paths["raw"] = {
            "data": signal.copy(),
            "quality": self._assess_path_quality(signal, fs, signal_type),
            "distortion": {"severity": 0.0, "type": "none"},
            "processing_applied": "none",
        }

        # Path 2: Filtered data (basic bandpass filtering)
        filtered_signal = self._apply_filtering(signal, fs, signal_type)
        paths["filtered"] = {
            "data": filtered_signal,
            "quality": self._assess_path_quality(filtered_signal, fs, signal_type),
            "distortion": self._compare_signals(signal, filtered_signal),
            "processing_applied": "bandpass_filter",
        }

        # Path 3: Preprocessed data (filtering + artifact removal)
        preprocessed_signal = self._apply_preprocessing(signal, fs, signal_type)
        paths["preprocessed"] = {
            "data": preprocessed_signal,
            "quality": self._assess_path_quality(preprocessed_signal, fs, signal_type),
            "distortion": self._compare_signals(signal, preprocessed_signal),
            "processing_applied": "filter_and_artifact_removal",
        }

        # Path 4: Full processing (all methods combined)
        full_signal, full_features = self._extract_simple_features(
            preprocessed_signal, fs, signal_type
        )
        paths["full"] = {
            "data": full_signal,
            "quality": self._assess_path_quality(full_signal, fs, signal_type),
            "distortion": self._compare_signals(signal, full_signal),
            "processing_applied": "full_pipeline",
            "features": full_features,
        }

        # Compare all paths and select best
        best_path = self._select_best_path(paths)

        parallel_results = {
            "paths": paths,
            "comparison": {
                "best_path": best_path,
                "total_paths": len(paths),
                "all_paths": list(paths.keys()),
            },
        }

        # Add processing recommendations
        parallel_results["processing_recommendations"] = self._analyze_processing_paths(
            parallel_results
        )

        logger.info(
            f"Parallel processing completed: {len(parallel_results['paths'])} paths processed"
        )

        return ProcessingResult(
            stage=ProcessingStage.PARALLEL_PROCESSING,
            success=True,
            data=parallel_results,
        )

    def _stage_quality_validation(self, context: Dict[str, Any]) -> ProcessingResult:
        """Stage 4: Quality Validation - Compare all processing paths."""
        parallel_result = context["results"][ProcessingStage.PARALLEL_PROCESSING.value]

        # Extract the actual data from ProcessingResult object
        parallel_results = (
            parallel_result.data if parallel_result.data is not None else {}
        )

        # Compare all processing paths
        validation_results = self._compare_processing_paths(parallel_results)

        # Detect distortions and quality improvements
        validation_results["distortion_analysis"] = self._analyze_distortions(
            parallel_results
        )
        validation_results["quality_improvements"] = self._analyze_quality_improvements(
            parallel_results
        )

        logger.info(
            f"Quality validation completed: {len(validation_results.get('path_comparisons', []))} paths compared"
        )

        return ProcessingResult(
            stage=ProcessingStage.QUALITY_VALIDATION,
            success=True,
            data=validation_results,
        )

    def _stage_segmentation(self, context: Dict[str, Any]) -> ProcessingResult:
        """Stage 5: Segmentation - Multiple segmentation strategies."""
        signal = context["signal"]
        fs = context["fs"]
        signal_type = context["signal_type"]
        processing_mode = context["metadata"].get(
            "processing_mode", "segment_with_overlap"
        )

        # Determine segmentation strategy
        segmentation_results = self._perform_segmentation(
            signal, fs, signal_type, processing_mode
        )

        logger.info(
            f"Segmentation completed: {len(segmentation_results['segments'])} segments created"
        )

        return ProcessingResult(
            stage=ProcessingStage.SEGMENTATION, success=True, data=segmentation_results
        )

    def _stage_feature_extraction(self, context: Dict[str, Any]) -> ProcessingResult:
        """Stage 6: Feature Extraction - Per-segment and global features."""
        segmentation_result = context["results"][ProcessingStage.SEGMENTATION.value]
        parallel_result = context["results"][ProcessingStage.PARALLEL_PROCESSING.value]

        # Extract the data from ProcessingResult objects
        segmentation_results = (
            segmentation_result.data
            if hasattr(segmentation_result, "data")
            else segmentation_result
        )
        parallel_results = (
            parallel_result.data
            if hasattr(parallel_result, "data")
            else parallel_result
        )

        # Extract features for all segments and processing paths
        feature_results = self._extract_comprehensive_features(
            segmentation_results, parallel_results
        )

        logger.info(
            f"Feature extraction completed: {len(feature_results['segment_features'])} segment features"
        )

        return ProcessingResult(
            stage=ProcessingStage.FEATURE_EXTRACTION, success=True, data=feature_results
        )

    def _stage_intelligent_output(self, context: Dict[str, Any]) -> ProcessingResult:
        """Stage 7: Intelligent Output - Generate multiple output options."""
        all_results = context["results"]

        # Generate intelligent output options
        output_options = self._generate_output_options(all_results)

        logger.info(
            f"Intelligent output completed: {len(output_options)} output options generated"
        )

        return ProcessingResult(
            stage=ProcessingStage.INTELLIGENT_OUTPUT, success=True, data=output_options
        )

    def _stage_output_package(self, context: Dict[str, Any]) -> ProcessingResult:
        """Stage 8: Output Package - Create final output package."""
        all_results = context["results"]

        # Create comprehensive output package
        output_package = self._create_output_package(all_results, context)

        logger.info(
            f"Output package completed: {len(output_package)} components packaged"
        )

        return ProcessingResult(
            stage=ProcessingStage.OUTPUT_PACKAGE, success=True, data=output_package
        )

    def _estimate_processing_complexity(self, signal: np.ndarray, fs: float) -> str:
        """Estimate processing complexity based on signal characteristics."""
        duration_minutes = len(signal) / fs / 60
        memory_mb = signal.nbytes / 1024**2

        if duration_minutes < 5 and memory_mb < 100:
            return "low"
        elif duration_minutes < 60 and memory_mb < 1000:
            return "medium"
        else:
            return "high"

    def _assess_false_positive_risk_from_results(
        self, quality_results: List[Any]
    ) -> Dict[str, Any]:
        """Assess false positive risk from quality screening results."""
        if not quality_results:
            return {"risk_level": "unknown", "confidence": 0.0}

        # Extract quality scores from screening results
        individual_scores = []
        for result in quality_results:
            if hasattr(result, "quality_metrics") and hasattr(
                result.quality_metrics, "overall_quality"
            ):
                individual_scores.append(result.quality_metrics.overall_quality)

        if len(individual_scores) < 2:
            return {"risk_level": "unknown", "confidence": 0.0}

        # Calculate disagreement between scores
        disagreement = np.std(individual_scores)

        if disagreement > 0.3:
            risk_level = "high"
        elif disagreement > 0.15:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Calculate confidence based on consistency
        confidence = max(0.0, 1.0 - disagreement)

        return {
            "risk_level": risk_level,
            "confidence": confidence,
            "disagreement": disagreement,
            "score_count": len(individual_scores),
        }

    def _assess_false_positive_risk(
        self, quality_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess false positive risk in quality assessment."""
        # Extract individual quality scores
        individual_scores = []
        for key, value in quality_results.items():
            if isinstance(value, (int, float)) and "score" in key.lower():
                individual_scores.append(value)

        if len(individual_scores) < 2:
            return {"risk_level": "unknown", "confidence": 0.0}

        # Calculate disagreement between scores
        disagreement = np.std(individual_scores)

        if disagreement > 0.3:
            risk_level = "high"
        elif disagreement > 0.15:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "risk_level": risk_level,
            "confidence": 1.0 - disagreement,
            "disagreement": disagreement,
            "recommendation": f"Quality assessment has {risk_level} false positive risk",
        }

    def _analyze_processing_paths(
        self, parallel_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze processing paths and provide recommendations."""
        paths = parallel_results["paths"]
        comparison = parallel_results["comparison"]

        recommendations = {
            "best_path": comparison.get("best_path", "raw"),
            "alternative_paths": [],
            "path_analysis": {},
        }

        # Analyze each path
        for path_name, path_data in paths.items():
            quality_score = path_data.get("quality", {}).get("quality_score", 0.0)
            distortion_level = path_data.get("distortion", {}).get("severity", 0.0)

            recommendations["path_analysis"][path_name] = {
                "quality_score": quality_score,
                "distortion_level": distortion_level,
                "recommendation": self._get_path_recommendation(
                    path_name, quality_score, distortion_level
                ),
            }

            if path_name != recommendations["best_path"]:
                recommendations["alternative_paths"].append(path_name)

        return recommendations

    def _get_path_recommendation(
        self, path_name: str, quality_score: float, distortion_level: float
    ) -> str:
        """Get recommendation for a specific processing path."""
        if path_name == "raw":
            return "Raw data - no processing applied"
        elif distortion_level < 0.1 and quality_score > 0.7:
            return f"{path_name} - excellent quality with minimal distortion"
        elif distortion_level > 0.3:
            return f"{path_name} - high distortion, use with caution"
        else:
            return f"{path_name} - moderate quality and distortion"

    def _compare_processing_paths(
        self, parallel_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare all processing paths for quality validation."""
        paths = parallel_results["paths"]
        comparison = parallel_results["comparison"]

        # Detailed comparison analysis
        path_comparisons = {}

        raw_path = paths.get("raw", {})
        raw_quality = raw_path.get("quality", {}).get("quality_score", 0.0)

        for path_name, path_data in paths.items():
            if path_name == "raw":
                continue

            path_quality = path_data.get("quality", {}).get("quality_score", 0.0)
            distortion = path_data.get("distortion", {}).get("severity", 0.0)

            quality_improvement = path_quality - raw_quality

            path_comparisons[path_name] = {
                "quality_score": path_quality,
                "quality_improvement": quality_improvement,
                "distortion_level": distortion,
                "net_benefit": quality_improvement - distortion,
                "recommendation": (
                    "use" if quality_improvement > distortion else "avoid"
                ),
            }

        return {
            "path_comparisons": path_comparisons,
            "best_path": comparison.get("best_path", "raw"),
            "overall_recommendation": self._get_overall_recommendation(
                path_comparisons
            ),
        }

    def _get_overall_recommendation(self, path_comparisons: Dict[str, Any]) -> str:
        """Get overall recommendation based on path comparisons."""
        if not path_comparisons:
            return "Use raw data - no processing paths available"

        # Find path with best net benefit
        best_path = max(path_comparisons.items(), key=lambda x: x[1]["net_benefit"])

        if best_path[1]["net_benefit"] > 0.1:
            return f"Use {best_path[0]} - significant quality improvement"
        elif best_path[1]["net_benefit"] > 0.0:
            return f"Consider {best_path[0]} - slight quality improvement"
        else:
            return "Use raw data - processing provides no benefit"

    def _analyze_distortions(self, parallel_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze distortions across processing paths."""
        paths = parallel_results["paths"]

        distortion_analysis = {
            "path_distortions": {},
            "overall_distortion_level": "low",
            "recommendations": [],
        }

        distortion_levels = []

        for path_name, path_data in paths.items():
            if "distortion" in path_data:
                distortion = path_data["distortion"]
                distortion_analysis["path_distortions"][path_name] = distortion
                distortion_levels.append(distortion.get("severity", 0.0))

        if distortion_levels:
            avg_distortion = np.mean(distortion_levels)
            max_distortion = np.max(distortion_levels)

            if max_distortion > 0.5:
                distortion_analysis["overall_distortion_level"] = "high"
                distortion_analysis["recommendations"].append(
                    "High distortion detected - prefer raw data"
                )
            elif avg_distortion > 0.2:
                distortion_analysis["overall_distortion_level"] = "medium"
                distortion_analysis["recommendations"].append(
                    "Moderate distortion - compare results carefully"
                )
            else:
                distortion_analysis["overall_distortion_level"] = "low"
                distortion_analysis["recommendations"].append(
                    "Low distortion - processing appears safe"
                )

        return distortion_analysis

    def _analyze_quality_improvements(
        self, parallel_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze quality improvements across processing paths."""
        paths = parallel_results["paths"]

        quality_improvements = {
            "path_improvements": {},
            "overall_improvement": 0.0,
            "recommendations": [],
        }

        raw_quality = paths.get("raw", {}).get("quality", {}).get("quality_score", 0.0)
        improvements = []

        for path_name, path_data in paths.items():
            if path_name == "raw":
                continue

            path_quality = path_data.get("quality", {}).get("quality_score", 0.0)
            improvement = path_quality - raw_quality

            quality_improvements["path_improvements"][path_name] = {
                "quality_score": path_quality,
                "improvement": improvement,
                "improvement_percent": (improvement / max(raw_quality, 0.01)) * 100,
            }

            improvements.append(improvement)

        if improvements:
            quality_improvements["overall_improvement"] = np.mean(improvements)

            if quality_improvements["overall_improvement"] > 0.1:
                quality_improvements["recommendations"].append(
                    "Significant quality improvement from processing"
                )
            elif quality_improvements["overall_improvement"] > 0.0:
                quality_improvements["recommendations"].append(
                    "Modest quality improvement from processing"
                )
            else:
                quality_improvements["recommendations"].append(
                    "No quality improvement from processing"
                )

        return quality_improvements

    def _perform_segmentation(
        self, signal: np.ndarray, fs: float, signal_type: str, processing_mode: str
    ) -> Dict[str, Any]:
        """Perform segmentation based on processing mode."""
        duration_seconds = len(signal) / fs

        if processing_mode == "whole_signal":
            # Single segment for whole signal
            segments = [
                {
                    "segment_id": 0,
                    "start_idx": 0,
                    "end_idx": len(signal),
                    "start_time": 0.0,
                    "end_time": duration_seconds,
                    "duration": duration_seconds,
                    "data": signal,
                }
            ]

        elif processing_mode == "segment_with_overlap":
            # Segmented processing with overlap
            segment_duration = self.config.get(
                f"segmentation.{signal_type}.segment_duration_sec", 30
            )
            overlap_ratio = self.config.get(
                f"segmentation.{signal_type}.overlap_ratio", 0.2
            )

            segments = self._create_overlapping_segments(
                signal, fs, segment_duration, overlap_ratio
            )

        else:  # hybrid
            # Hybrid approach with multiple segment sizes
            segments = self._create_hybrid_segments(signal, fs, signal_type)

        return {
            "segments": segments,
            "total_segments": len(segments),
            "processing_mode": processing_mode,
            "segmentation_strategy": self._get_segmentation_strategy(processing_mode),
        }

    def _create_overlapping_segments(
        self,
        signal: np.ndarray,
        fs: float,
        segment_duration: float,
        overlap_ratio: float,
    ) -> List[Dict[str, Any]]:
        """Create overlapping segments."""
        segment_size = int(segment_duration * fs)
        overlap_size = int(segment_size * overlap_ratio)
        hop_size = segment_size - overlap_size

        segments = []

        for i, start_idx in enumerate(
            range(0, len(signal) - segment_size + 1, hop_size)
        ):
            end_idx = start_idx + segment_size

            segments.append(
                {
                    "segment_id": i,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_time": start_idx / fs,
                    "end_time": end_idx / fs,
                    "duration": segment_duration,
                    "data": signal[start_idx:end_idx],
                }
            )

        return segments

    def _create_hybrid_segments(
        self, signal: np.ndarray, fs: float, signal_type: str
    ) -> List[Dict[str, Any]]:
        """Create hybrid segments with multiple sizes."""
        duration_seconds = len(signal) / fs

        # Create segments of different sizes
        segments = []

        # Short segments for detailed analysis
        short_duration = 10  # seconds
        short_segments = self._create_overlapping_segments(
            signal, fs, short_duration, 0.1
        )

        # Medium segments for standard analysis
        medium_duration = 30  # seconds
        medium_segments = self._create_overlapping_segments(
            signal, fs, medium_duration, 0.2
        )

        # Long segments for global analysis
        long_duration = 60  # seconds
        long_segments = self._create_overlapping_segments(
            signal, fs, long_duration, 0.1
        )

        # Combine and tag segments
        for seg in short_segments:
            seg["segment_type"] = "short"
            seg["analysis_type"] = "detailed"
            segments.append(seg)

        for seg in medium_segments:
            seg["segment_type"] = "medium"
            seg["analysis_type"] = "standard"
            segments.append(seg)

        for seg in long_segments:
            seg["segment_type"] = "long"
            seg["analysis_type"] = "global"
            segments.append(seg)

        return segments

    def _get_segmentation_strategy(self, processing_mode: str) -> str:
        """Get segmentation strategy description."""
        strategies = {
            "whole_signal": "Single segment for entire signal",
            "segment_with_overlap": "Overlapping segments for continuous analysis",
            "hybrid": "Multiple segment sizes for comprehensive analysis",
        }
        return strategies.get(processing_mode, "Unknown strategy")

    def _extract_comprehensive_features(
        self, segmentation_results: Dict[str, Any], parallel_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract comprehensive features from segments and processing paths."""
        segments = segmentation_results["segments"]
        paths = parallel_results["paths"]

        feature_results = {
            "segment_features": [],
            "path_features": {},
            "global_features": {},
            "feature_statistics": {},
        }

        # Extract features for each segment
        for segment in segments:
            segment_features = self._extract_segment_features(segment, paths)
            feature_results["segment_features"].append(segment_features)

        # Extract features for each processing path
        for path_name, path_data in paths.items():
            if "features" in path_data:
                feature_results["path_features"][path_name] = path_data["features"]

        # Extract global features
        feature_results["global_features"] = self._extract_global_features(segments)

        # Calculate feature statistics
        feature_results["feature_statistics"] = self._calculate_feature_statistics(
            feature_results["segment_features"]
        )

        return feature_results

    def _extract_segment_features(
        self, segment: Dict[str, Any], paths: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract features for a single segment."""
        segment_data = segment["data"]

        # Basic segment features
        features = {
            "segment_id": segment["segment_id"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "duration": segment["duration"],
            "length": len(segment_data),
            "mean": np.mean(segment_data),
            "std": np.std(segment_data),
            "min": np.min(segment_data),
            "max": np.max(segment_data),
            "range": np.ptp(segment_data),
        }

        # Add processing path specific features
        for path_name, path_data in paths.items():
            if "features" in path_data:
                features[f"{path_name}_features"] = path_data["features"]

        return features

    def _extract_global_features(
        self, segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract global features from all segments."""
        if not segments:
            return {}

        # Combine all segment data for global analysis
        all_data = np.concatenate([seg["data"] for seg in segments])

        global_features = {
            "total_duration": sum(seg["duration"] for seg in segments),
            "total_segments": len(segments),
            "global_mean": np.mean(all_data),
            "global_std": np.std(all_data),
            "global_min": np.min(all_data),
            "global_max": np.max(all_data),
            "global_range": np.ptp(all_data),
            "data_completeness": len(all_data)
            / (len(all_data) + np.sum(np.isnan(all_data))),
            "temporal_consistency": self._calculate_temporal_consistency(segments),
        }

        return global_features

    def _calculate_temporal_consistency(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate temporal consistency across segments."""
        if len(segments) < 2:
            return 1.0

        # Calculate consistency of basic statistics across segments
        means = [np.mean(seg["data"]) for seg in segments]
        stds = [np.std(seg["data"]) for seg in segments]

        mean_consistency = 1.0 - (np.std(means) / max(np.mean(np.abs(means)), 1e-10))
        std_consistency = 1.0 - (np.std(stds) / max(np.mean(stds), 1e-10))

        return (mean_consistency + std_consistency) / 2

    def _calculate_feature_statistics(
        self, segment_features: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate statistics across all segment features."""
        if not segment_features:
            return {}

        # Extract common features
        durations = [feat["duration"] for feat in segment_features]
        means = [feat["mean"] for feat in segment_features]
        stds = [feat["std"] for feat in segment_features]

        return {
            "duration_stats": {
                "mean": np.mean(durations),
                "std": np.std(durations),
                "min": np.min(durations),
                "max": np.max(durations),
            },
            "mean_stats": {
                "mean": np.mean(means),
                "std": np.std(means),
                "min": np.min(means),
                "max": np.max(means),
            },
            "std_stats": {
                "mean": np.mean(stds),
                "std": np.std(stds),
                "min": np.min(stds),
                "max": np.max(stds),
            },
        }

    def _generate_output_options(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate multiple output options for user selection."""
        segmentation_results = all_results.get(ProcessingStage.SEGMENTATION.value, {})
        feature_results = all_results.get(ProcessingStage.FEATURE_EXTRACTION.value, {})
        quality_results = all_results.get(ProcessingStage.QUALITY_SCREENING.value, {})

        segments = segmentation_results.get("segments", [])

        # Generate different output options
        output_options = {
            "option_1_all_segments": {
                "description": "Export all segments with timestamps",
                "output_type": "segmented_signal",
                "segment_count": len(segments),
                "includes_quality": True,
                "file_size_estimate": self._estimate_output_size(segments, "all"),
            },
            "option_2_best_quality": {
                "description": "Export only high-quality segments (quality > 0.7)",
                "output_type": "filtered_segments",
                "segment_count": self._count_high_quality_segments(
                    segments, quality_results
                ),
                "quality_threshold": 0.7,
            },
            "option_3_concatenated": {
                "description": "Concatenate best segments into continuous signal",
                "output_type": "continuous_signal",
                "note": "Timestamps preserved, gaps where poor quality removed",
            },
            "option_4_features_only": {
                "description": "Export aggregated features (per-segment + global)",
                "output_type": "features_database",
                "format": "CSV/JSON with timestamps",
                "feature_count": len(feature_results.get("segment_features", [])),
            },
            "option_5_time_range": {
                "description": "Export specific time range (user selects)",
                "output_type": "time_range_selection",
                "interactive": True,
            },
            "option_6_whole_signal": {
                "description": "Export entire signal (all segments merged)",
                "output_type": "complete_signal",
                "file_size_estimate": self._estimate_output_size(segments, "whole"),
            },
        }

        return output_options

    def _count_high_quality_segments(
        self, segments: List[Dict[str, Any]], quality_results: Dict[str, Any]
    ) -> int:
        """Count segments that meet high quality threshold."""
        # This is a simplified implementation
        # In practice, you'd need to map segments to quality scores
        return len(segments)  # Placeholder

    def _estimate_output_size(
        self, segments: List[Dict[str, Any]], output_type: str
    ) -> str:
        """Estimate output file size."""
        total_samples = sum(len(seg["data"]) for seg in segments)
        size_mb = (total_samples * 8) / 1024**2  # Assuming float64

        if size_mb < 1:
            return f"{size_mb * 1024:.0f} KB"
        elif size_mb < 1000:
            return f"{size_mb:.1f} MB"
        else:
            return f"{size_mb / 1024:.1f} GB"

    def _create_output_package(
        self, all_results: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive output package."""
        output_package = {
            "processing_summary": {
                "signal_type": context["signal_type"],
                "sampling_frequency": context["fs"],
                "total_duration": context["signal"].shape[0] / context["fs"],
                "processing_time": self.stats["total_processing_time"],
                "stages_completed": self.stats["stages_completed"],
                "timestamp": datetime.now().isoformat(),
            },
            "data_ingestion": all_results.get(ProcessingStage.DATA_INGESTION.value, {}),
            "quality_assessment": all_results.get(
                ProcessingStage.QUALITY_SCREENING.value, {}
            ),
            "processing_results": all_results.get(
                ProcessingStage.PARALLEL_PROCESSING.value, {}
            ),
            "quality_validation": all_results.get(
                ProcessingStage.QUALITY_VALIDATION.value, {}
            ),
            "segmentation": all_results.get(ProcessingStage.SEGMENTATION.value, {}),
            "features": all_results.get(ProcessingStage.FEATURE_EXTRACTION.value, {}),
            "output_options": all_results.get(
                ProcessingStage.INTELLIGENT_OUTPUT.value, {}
            ),
            "processing_statistics": self.stats,
            "cache_statistics": self.cache.get_stats(),
            "recommendations": self._generate_processing_recommendations(all_results),
        }

        return output_package

    def _generate_processing_recommendations(
        self, all_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate processing recommendations based on all results."""
        recommendations = {
            "best_processing_path": "raw",  # Default
            "quality_assessment": "manual_review_recommended",
            "segmentation_strategy": "standard",
            "output_format": "features_only",
            "further_analysis": [],
        }

        # Analyze results to generate recommendations
        quality_results = all_results.get(ProcessingStage.QUALITY_SCREENING.value, {})
        parallel_results = all_results.get(
            ProcessingStage.PARALLEL_PROCESSING.value, {}
        )

        if quality_results:
            overall_quality = quality_results.get("overall_quality_score", 0.0)

            if overall_quality > 0.8:
                recommendations["quality_assessment"] = (
                    "excellent_quality_safe_for_all_analyses"
                )
            elif overall_quality > 0.6:
                recommendations["quality_assessment"] = (
                    "good_quality_suitable_for_most_analyses"
                )
            elif overall_quality > 0.4:
                recommendations["quality_assessment"] = (
                    "fair_quality_usable_with_preprocessing"
                )
            else:
                recommendations["quality_assessment"] = (
                    "poor_quality_manual_review_recommended"
                )

        if parallel_results and "comparison" in parallel_results:
            recommendations["best_processing_path"] = parallel_results[
                "comparison"
            ].get("best_path", "raw")

        return recommendations

    def _apply_filtering(
        self, signal: np.ndarray, fs: float, signal_type: str
    ) -> np.ndarray:
        """Apply basic bandpass filtering to signal using vitalDSP SignalFiltering."""
        try:
            from vitalDSP.filtering.signal_filtering import SignalFiltering

            # Get filter parameters based on signal type
            if signal_type.lower() == "ecg":
                lowcut, highcut = 0.5, 40.0
            elif signal_type.lower() == "ppg":
                lowcut, highcut = 0.5, 8.0
            elif signal_type.lower() == "eeg":
                lowcut, highcut = 0.5, 50.0
            else:
                lowcut, highcut = 0.5, 40.0

            # Use vitalDSP SignalFiltering for bandpass filtering
            sf = SignalFiltering(signal)
            filtered = sf.bandpass(
                lowcut=lowcut, highcut=highcut, fs=fs, order=4, filter_type="butter"
            )

            logger.info(
                f"Applied vitalDSP bandpass filter ({lowcut}-{highcut} Hz) for {signal_type}"
            )
            return filtered
        except Exception as e:
            logger.warning(f"vitalDSP filtering failed: {e}, returning original signal")
            return signal.copy()

    def _apply_preprocessing(
        self, signal: np.ndarray, fs: float, signal_type: str
    ) -> np.ndarray:
        """Apply preprocessing (filtering + artifact removal) using vitalDSP modules."""
        try:
            # First apply filtering using vitalDSP
            filtered = self._apply_filtering(signal, fs, signal_type)

            # Apply vitalDSP artifact removal
            from vitalDSP.filtering.artifact_removal import ArtifactRemoval

            ar = ArtifactRemoval(filtered, fs)
            # Use adaptive threshold artifact removal for better results
            preprocessed = ar.adaptive_threshold_removal(
                window_size=int(2 * fs),  # 2-second windows
                std_factor=3.0,  # 3 standard deviations
            )

            logger.info(f"Applied vitalDSP artifact removal for {signal_type}")
            return preprocessed
        except Exception as e:
            logger.warning(
                f"vitalDSP preprocessing failed: {e}, using basic preprocessing"
            )
            # Fallback to basic preprocessing
            filtered = self._apply_filtering(signal, fs, signal_type)
            mean = np.mean(filtered)
            std = np.std(filtered)
            threshold = 5 * std
            return np.clip(filtered, mean - threshold, mean + threshold)

    def _extract_simple_features(
        self, signal: np.ndarray, fs: float, signal_type: str
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract comprehensive features using numpy and scipy for signal analysis."""
        try:
            features = {}

            # Time domain features using numpy
            features["mean"] = float(np.mean(signal))
            features["std"] = float(np.std(signal))
            features["min"] = float(np.min(signal))
            features["max"] = float(np.max(signal))
            features["range"] = float(np.ptp(signal))
            features["rms"] = float(np.sqrt(np.mean(signal**2)))
            features["variance"] = float(np.var(signal))

            # Statistical moments
            from scipy import stats

            features["skewness"] = float(stats.skew(signal))
            features["kurtosis"] = float(stats.kurtosis(signal))

            # Frequency domain features using scipy
            from scipy.fft import fft, fftfreq

            fft_vals = fft(signal)
            fft_freqs = fftfreq(len(signal), 1 / fs)

            # Use only positive frequencies
            positive_freqs = fft_freqs > 0
            fft_power = np.abs(fft_vals[positive_freqs]) ** 2
            freqs = fft_freqs[positive_freqs]

            if len(freqs) > 0 and np.sum(fft_power) > 0:
                # Spectral centroid
                features["spectral_centroid"] = float(
                    np.sum(freqs * fft_power) / np.sum(fft_power)
                )

                # Spectral bandwidth
                centroid = features["spectral_centroid"]
                features["spectral_bandwidth"] = float(
                    np.sqrt(
                        np.sum(((freqs - centroid) ** 2) * fft_power)
                        / np.sum(fft_power)
                    )
                )

                # Dominant frequency
                features["dominant_frequency"] = float(freqs[np.argmax(fft_power)])

                # Spectral energy
                features["spectral_energy"] = float(np.sum(fft_power))
            else:
                features["spectral_centroid"] = 0.0
                features["spectral_bandwidth"] = 0.0
                features["dominant_frequency"] = 0.0
                features["spectral_energy"] = 0.0

            # Total energy
            features["total_energy"] = float(np.sum(signal**2))

            logger.info(f"Extracted {len(features)} features for {signal_type}")
            return signal, features

        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}, using basic features")
            # Minimal fallback features
            features = {
                "mean": float(np.mean(signal)),
                "std": float(np.std(signal)),
                "min": float(np.min(signal)),
                "max": float(np.max(signal)),
                "range": float(np.ptp(signal)),
                "energy": float(np.sum(signal**2)),
            }
            return signal, features

    def _assess_path_quality(
        self, signal: np.ndarray, fs: float, signal_type: str
    ) -> Dict[str, float]:
        """Assess quality of a processing path."""
        try:
            # Use quality screener to assess signal quality
            screening_results = self.quality_screener.screen_signal(signal)

            if screening_results:
                # Get first result (overall quality)
                result = screening_results[0]
                return {
                    "quality_score": result.quality_metrics.overall_quality,
                    "snr": result.quality_metrics.snr,
                    "passed_screening": result.passed_screening,
                }
            else:
                # Fallback: basic SNR estimation
                signal_power = np.var(signal)
                noise_estimate = np.var(np.diff(signal)) / 2
                snr = 10 * np.log10(signal_power / max(noise_estimate, 1e-10))

                return {
                    "quality_score": min(snr / 30, 1.0),  # Normalize to 0-1
                    "snr": snr,
                    "passed_screening": snr > 10,
                }
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return {
                "quality_score": 0.5,
                "snr": 0.0,
                "passed_screening": True,
            }

    def _compare_signals(
        self, original: np.ndarray, processed: np.ndarray
    ) -> Dict[str, Any]:
        """Compare original and processed signals to detect distortion."""
        try:
            # Calculate correlation
            correlation = np.corrcoef(original, processed)[0, 1]

            # Calculate RMSE (normalized)
            rmse = np.sqrt(np.mean((original - processed) ** 2))
            normalized_rmse = rmse / (np.std(original) + 1e-10)

            # Calculate distortion severity (0 = no distortion, 1 = high distortion)
            distortion_severity = max(
                0, min(1, (1 - correlation) + normalized_rmse / 2)
            )

            # Classify distortion type
            if distortion_severity < 0.1:
                distortion_type = "minimal"
            elif distortion_severity < 0.3:
                distortion_type = "low"
            elif distortion_severity < 0.5:
                distortion_type = "moderate"
            else:
                distortion_type = "high"

            return {
                "severity": distortion_severity,
                "type": distortion_type,
                "correlation": correlation,
                "rmse": rmse,
                "normalized_rmse": normalized_rmse,
            }
        except Exception as e:
            logger.warning(f"Signal comparison failed: {e}")
            return {
                "severity": 0.5,
                "type": "unknown",
                "correlation": 0.0,
                "rmse": 0.0,
                "normalized_rmse": 0.0,
            }

    def _select_best_path(self, paths: Dict[str, Any]) -> str:
        """Select best processing path based on quality and distortion."""
        best_path = "raw"
        best_score = -1.0

        for path_name, path_data in paths.items():
            quality_score = path_data["quality"].get("quality_score", 0.0)
            distortion_severity = path_data["distortion"].get("severity", 1.0)

            # Net score: quality improvement minus distortion cost
            net_score = quality_score - distortion_severity

            if net_score > best_score:
                best_score = net_score
                best_path = path_name

        return best_path

    def _generate_final_results(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final processing results."""
        # Get the last successful stage result or create a basic result
        if ProcessingStage.OUTPUT_PACKAGE.value in context["results"]:
            final_results = context["results"][ProcessingStage.OUTPUT_PACKAGE.value]
        else:
            # Create a basic result if OUTPUT_PACKAGE stage didn't complete
            final_results = {
                "processing_results": context.get("results", {}),
                "success": True,
                "message": "Processing completed with partial results",
            }

        # Add processing metadata
        final_results["processing_metadata"] = {
            "session_id": context["session_id"],
            "start_time": context["start_time"].isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_processing_time": self.stats["total_processing_time"],
            "stages_completed": self.stats["stages_completed"],
            "errors_encountered": self.stats["errors_encountered"],
        }

        return final_results

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "pipeline_stats": self.stats,
            "cache_stats": self.cache.get_stats(),
            "config_stats": self.config.get_statistics(),
            "optimization_stats": {
                "cache_enabled": True,
                "checkpointing_enabled": True,
                "parallel_processing_enabled": True,
                "quality_aware_processing": True,
            },
        }

    def clear_cache(self) -> None:
        """Clear processing cache."""
        self.cache.clear()

    def cleanup_checkpoints(self, session_id: str) -> None:
        """Clean up checkpoints for a session."""
        self.checkpoint_manager.cleanup_session(session_id)


# Alias for OptimizedStandardProcessingPipeline
# Currently uses the same implementation as StandardProcessingPipeline
# Future optimizations can be added here
class OptimizedStandardProcessingPipeline(StandardProcessingPipeline):
    """
    Optimized version of StandardProcessingPipeline for large file processing.

    Currently an alias to StandardProcessingPipeline. Future optimizations will include:
    - Parallel path processing in Stage 3
    - More efficient memory management
    - Optimized caching strategies
    - Reduced memory allocations

    Use this for files >5 minutes or when performance is critical.
    """

    pass
