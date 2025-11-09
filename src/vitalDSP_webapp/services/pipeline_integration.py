"""
Pipeline Integration Service for vitalDSP Webapp

This module provides integration between the webapp frontend and the core vitalDSP
StandardProcessingPipeline, handling real pipeline execution with progress tracking.
"""

import logging
import threading
import time
import uuid
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd

# Import core vitalDSP components
from vitalDSP.utils.core_infrastructure.processing_pipeline import (
    StandardProcessingPipeline,
    ProcessingStage,
    ProcessingResult,
)
from vitalDSP.utils.core_infrastructure.quality_screener import QualityScreener
from vitalDSP.utils.config_utilities.dynamic_config import DynamicConfigManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineExecutionState:
    """State tracking for pipeline execution."""

    session_id: str
    status: str  # 'running', 'completed', 'failed', 'stopped'
    current_stage: int
    total_stages: int
    progress_percentage: float
    stage_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    quality_scores: Optional[Dict[str, Any]] = None
    processing_metadata: Optional[Dict[str, Any]] = None


class PipelineIntegrationService:
    """
    Service for integrating webapp frontend with core vitalDSP pipeline.

    Handles:
    - Real pipeline execution using StandardProcessingPipeline
    - Progress tracking and state management
    - Background processing with threading
    - Result caching and retrieval
    - Error handling and recovery
    """

    def __init__(self):
        """Initialize the pipeline integration service."""
        self.active_executions: Dict[str, PipelineExecutionState] = {}
        self.execution_threads: Dict[str, threading.Thread] = {}
        self.progress_callbacks: Dict[str, Callable] = {}

        # Initialize core pipeline components
        self.config_manager = DynamicConfigManager()
        self.pipeline = StandardProcessingPipeline(
            config_manager=self.config_manager,
            cache_dir="~/.vitaldsp/webapp_cache",
            checkpoint_dir="~/.vitaldsp/webapp_checkpoints",
        )

        logger.info("Pipeline Integration Service initialized")

    def start_pipeline_execution(
        self,
        signal_data: np.ndarray,
        fs: float,
        signal_type: str,
        processing_paths: List[str],
        quality_threshold: float = 0.5,
        window_size: int = 30,
        overlap_ratio: float = 0.5,
        feature_types: List[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """
        Start pipeline execution in background thread.

        Args:
            signal_data: Input signal data
            fs: Sampling frequency
            signal_type: Type of signal (ecg, ppg, etc.)
            processing_paths: List of processing paths to use
            quality_threshold: Quality screening threshold
            window_size: Segmentation window size
            overlap_ratio: Overlap ratio for segmentation
            feature_types: Types of features to extract
            progress_callback: Optional callback for progress updates

        Returns:
            session_id: Unique session identifier for tracking
        """
        session_id = str(uuid.uuid4())

        # Create execution state
        execution_state = PipelineExecutionState(
            session_id=session_id,
            status="running",
            current_stage=0,
            total_stages=8,
            progress_percentage=0.0,
            stage_name="Initializing",
            start_time=datetime.now(),
        )

        # Store execution state
        self.active_executions[session_id] = execution_state

        # Store progress callback if provided
        if progress_callback:
            self.progress_callbacks[session_id] = progress_callback

        # Start background thread
        thread = threading.Thread(
            target=self._execute_pipeline_thread,
            args=(
                session_id,
                signal_data,
                fs,
                signal_type,
                processing_paths,
                quality_threshold,
                window_size,
                overlap_ratio,
                feature_types or ["time_domain", "frequency_domain"],
            ),
            daemon=True,
        )

        self.execution_threads[session_id] = thread
        thread.start()

        logger.info(f"Started pipeline execution with session_id: {session_id}")
        return session_id

    def _execute_pipeline_thread(
        self,
        session_id: str,
        signal_data: np.ndarray,
        fs: float,
        signal_type: str,
        processing_paths: List[str],
        quality_threshold: float,
        window_size: int,
        overlap_ratio: float,
        feature_types: List[str],
    ):
        """Execute pipeline in background thread with progress tracking."""
        try:
            execution_state = self.active_executions[session_id]

            # Update progress: Stage 1 - Data Ingestion
            self._update_progress(session_id, 1, "Data Ingestion", 12.5)

            # Prepare metadata
            metadata = {
                "signal_type": signal_type,
                "sampling_rate": fs,
                "processing_paths": processing_paths,
                "quality_threshold": quality_threshold,
                "window_size": window_size,
                "overlap_ratio": overlap_ratio,
                "feature_types": feature_types,
                "session_id": session_id,
            }

            # Execute the real pipeline
            logger.info(
                f"Executing StandardProcessingPipeline for session {session_id}"
            )

            # Update progress: Stage 2 - Quality Screening
            self._update_progress(session_id, 2, "Quality Screening", 25.0)

            # Update progress: Stage 3 - Parallel Processing
            self._update_progress(session_id, 3, "Parallel Processing", 37.5)

            # Update progress: Stage 4 - Quality Validation
            self._update_progress(session_id, 4, "Quality Validation", 50.0)

            # Update progress: Stage 5 - Segmentation
            self._update_progress(session_id, 5, "Segmentation", 62.5)

            # Update progress: Stage 6 - Feature Extraction
            self._update_progress(session_id, 6, "Feature Extraction", 75.0)

            # Update progress: Stage 7 - Intelligent Output
            self._update_progress(session_id, 7, "Intelligent Output", 87.5)

            # Execute the actual pipeline
            results = self.pipeline.process_signal(
                signal=signal_data,
                fs=fs,
                signal_type=signal_type,
                metadata=metadata,
                session_id=session_id,
                resume_from_checkpoint=False,
            )

            # Update progress: Stage 8 - Output Package
            self._update_progress(session_id, 8, "Output Package", 100.0)

            # Mark as completed
            execution_state.status = "completed"
            execution_state.end_time = datetime.now()
            execution_state.results = results
            execution_state.quality_scores = results.get("quality_scores", {})
            execution_state.processing_metadata = results.get("processing_metadata", {})

            logger.info(f"Pipeline execution completed for session {session_id}")

        except Exception as e:
            logger.error(f"Pipeline execution failed for session {session_id}: {e}")
            execution_state = self.active_executions[session_id]
            execution_state.status = "failed"
            execution_state.end_time = datetime.now()
            execution_state.error_message = str(e)

        finally:
            # Clean up thread reference
            if session_id in self.execution_threads:
                del self.execution_threads[session_id]

    def _update_progress(
        self, session_id: str, stage: int, stage_name: str, progress_percentage: float
    ):
        """Update execution progress and notify callbacks."""
        if session_id in self.active_executions:
            execution_state = self.active_executions[session_id]
            execution_state.current_stage = stage
            execution_state.stage_name = stage_name
            execution_state.progress_percentage = progress_percentage

            # Notify progress callback if registered
            if session_id in self.progress_callbacks:
                try:
                    self.progress_callbacks[session_id](execution_state)
                except Exception as e:
                    logger.warning(
                        f"Progress callback failed for session {session_id}: {e}"
                    )

    def stop_pipeline_execution(self, session_id: str) -> bool:
        """
        Stop pipeline execution.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if stopped successfully
        """
        if session_id in self.active_executions:
            execution_state = self.active_executions[session_id]
            if execution_state.status == "running":
                execution_state.status = "stopped"
                execution_state.end_time = datetime.now()

                # Note: We can't actually stop the pipeline mid-execution
                # but we can mark it as stopped for UI purposes
                logger.info(
                    f"Pipeline execution marked as stopped for session {session_id}"
                )
                return True

        return False

    def get_execution_state(self, session_id: str) -> Optional[PipelineExecutionState]:
        """
        Get current execution state.

        Args:
            session_id: Session identifier

        Returns:
            PipelineExecutionState or None if not found
        """
        return self.active_executions.get(session_id)

    def get_execution_results(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get pipeline execution results.

        Args:
            session_id: Session identifier

        Returns:
            Results dictionary or None if not completed
        """
        execution_state = self.active_executions.get(session_id)
        if execution_state and execution_state.status == "completed":
            return execution_state.results
        return None

    def cleanup_session(self, session_id: str):
        """Clean up session data."""
        if session_id in self.active_executions:
            del self.active_executions[session_id]
        if session_id in self.progress_callbacks:
            del self.progress_callbacks[session_id]
        if session_id in self.execution_threads:
            del self.execution_threads[session_id]

        logger.info(f"Cleaned up session {session_id}")

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.active_executions.keys())

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get pipeline execution statistics."""
        total_executions = len(self.active_executions)
        running_executions = sum(
            1 for state in self.active_executions.values() if state.status == "running"
        )
        completed_executions = sum(
            1
            for state in self.active_executions.values()
            if state.status == "completed"
        )
        failed_executions = sum(
            1 for state in self.active_executions.values() if state.status == "failed"
        )

        return {
            "total_executions": total_executions,
            "running_executions": running_executions,
            "completed_executions": completed_executions,
            "failed_executions": failed_executions,
            "active_sessions": self.get_active_sessions(),
        }


# Global service instance
pipeline_service = PipelineIntegrationService()


def get_pipeline_service() -> PipelineIntegrationService:
    """Get the global pipeline integration service instance."""
    return pipeline_service
