"""
Progress Tracking Service for vitalDSP Webapp

This service provides real-time progress tracking for long-running operations
like filtering, quality assessment, and feature extraction.
"""

import logging
import threading
import time
import uuid
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ProgressStatus(Enum):
    """Status of progress tracking."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressInfo:
    """Information about a progress tracking session."""
    task_id: str
    operation_name: str
    status: ProgressStatus
    progress_percentage: float
    current_step: str
    total_steps: int
    current_step_number: int
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ProgressTracker:
    """
    Service for tracking progress of long-running operations.
    
    Features:
    - Real-time progress updates
    - Multiple concurrent operations
    - Progress callbacks
    - Error handling and recovery
    - Cancellation support
    """
    
    def __init__(self):
        """Initialize the progress tracker."""
        self.active_tasks: Dict[str, ProgressInfo] = {}
        self.progress_callbacks: Dict[str, Callable] = {}
        self.cancellation_flags: Dict[str, bool] = {}
        
        logger.info("Progress Tracker initialized")
    
    def start_task(
        self,
        operation_name: str,
        total_steps: int = 100,
        progress_callback: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start tracking a new task.
        
        Args:
            operation_name: Name of the operation
            total_steps: Total number of steps
            progress_callback: Optional callback for progress updates
            metadata: Optional metadata about the task
            
        Returns:
            task_id: Unique identifier for the task
        """
        task_id = str(uuid.uuid4())
        
        progress_info = ProgressInfo(
            task_id=task_id,
            operation_name=operation_name,
            status=ProgressStatus.PENDING,
            progress_percentage=0.0,
            current_step="Initializing",
            total_steps=total_steps,
            current_step_number=0,
            start_time=datetime.now(),
            metadata=metadata or {}
        )
        
        self.active_tasks[task_id] = progress_info
        self.cancellation_flags[task_id] = False
        
        if progress_callback:
            self.progress_callbacks[task_id] = progress_callback
        
        logger.info(f"Started task {task_id}: {operation_name}")
        return task_id
    
    def update_progress(
        self,
        task_id: str,
        progress_percentage: float,
        current_step: str,
        step_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update progress for a task.
        
        Args:
            task_id: Task identifier
            progress_percentage: Progress percentage (0-100)
            current_step: Description of current step
            step_number: Optional step number
            metadata: Optional additional metadata
            
        Returns:
            bool: True if update successful, False if task not found or cancelled
        """
        if task_id not in self.active_tasks:
            logger.warning(f"Task {task_id} not found for progress update")
            return False
        
        if self.cancellation_flags.get(task_id, False):
            logger.info(f"Task {task_id} was cancelled, stopping progress updates")
            return False
        
        progress_info = self.active_tasks[task_id]
        progress_info.progress_percentage = min(100.0, max(0.0, progress_percentage))
        progress_info.current_step = current_step
        progress_info.status = ProgressStatus.RUNNING
        
        if step_number is not None:
            progress_info.current_step_number = step_number
        
        if metadata:
            progress_info.metadata.update(metadata)
        
        # Notify callback if registered
        if task_id in self.progress_callbacks:
            try:
                self.progress_callbacks[task_id](progress_info)
            except Exception as e:
                logger.warning(f"Progress callback failed for task {task_id}: {e}")
        
        logger.debug(f"Updated progress for task {task_id}: {progress_percentage:.1f}% - {current_step}")
        return True
    
    def complete_task(
        self,
        task_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Mark a task as completed or failed.
        
        Args:
            task_id: Task identifier
            success: Whether the task completed successfully
            error_message: Error message if failed
            metadata: Optional final metadata
        """
        if task_id not in self.active_tasks:
            logger.warning(f"Task {task_id} not found for completion")
            return
        
        progress_info = self.active_tasks[task_id]
        progress_info.end_time = datetime.now()
        
        if success:
            progress_info.status = ProgressStatus.COMPLETED
            progress_info.progress_percentage = 100.0
            progress_info.current_step = "Completed"
        else:
            progress_info.status = ProgressStatus.FAILED
            progress_info.error_message = error_message
            progress_info.current_step = "Failed"
        
        if metadata:
            progress_info.metadata.update(metadata)
        
        # Notify callback if registered
        if task_id in self.progress_callbacks:
            try:
                self.progress_callbacks[task_id](progress_info)
            except Exception as e:
                logger.warning(f"Progress callback failed for task {task_id}: {e}")
        
        logger.info(f"Task {task_id} completed with status: {progress_info.status.value}")
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            bool: True if task was cancelled, False if not found
        """
        if task_id not in self.active_tasks:
            logger.warning(f"Task {task_id} not found for cancellation")
            return False
        
        self.cancellation_flags[task_id] = True
        
        progress_info = self.active_tasks[task_id]
        progress_info.status = ProgressStatus.CANCELLED
        progress_info.end_time = datetime.now()
        progress_info.current_step = "Cancelled"
        
        logger.info(f"Task {task_id} cancelled")
        return True
    
    def get_task_progress(self, task_id: str) -> Optional[ProgressInfo]:
        """
        Get progress information for a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            ProgressInfo or None if not found
        """
        return self.active_tasks.get(task_id)
    
    def get_all_active_tasks(self) -> List[ProgressInfo]:
        """Get all active tasks."""
        return list(self.active_tasks.values())
    
    def cleanup_task(self, task_id: str):
        """Clean up task data."""
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        if task_id in self.progress_callbacks:
            del self.progress_callbacks[task_id]
        if task_id in self.cancellation_flags:
            del self.cancellation_flags[task_id]
        
        logger.debug(f"Cleaned up task {task_id}")
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Clean up completed tasks older than specified age."""
        current_time = datetime.now()
        tasks_to_remove = []
        
        for task_id, progress_info in self.active_tasks.items():
            if progress_info.end_time:
                age_hours = (current_time - progress_info.end_time).total_seconds() / 3600
                if age_hours > max_age_hours:
                    tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            self.cleanup_task(task_id)
        
        if tasks_to_remove:
            logger.info(f"Cleaned up {len(tasks_to_remove)} old completed tasks")


# Global progress tracker instance
progress_tracker = ProgressTracker()


def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker instance."""
    return progress_tracker


def track_progress(task_id: str, operation_name: str, total_steps: int = 100):
    """
    Context manager for tracking progress.
    
    Usage:
        with track_progress(task_id, "Filtering Signal", 5) as tracker:
            tracker.update_progress(20, "Loading data", 1)
            # ... do work ...
            tracker.update_progress(40, "Applying filter", 2)
            # ... do work ...
    """
    class ProgressContext:
        def __init__(self, task_id: str, operation_name: str, total_steps: int):
            self.task_id = task_id
            self.operation_name = operation_name
            self.total_steps = total_steps
            self.tracker = get_progress_tracker()
        
        def __enter__(self):
            return self.tracker
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                self.tracker.complete_task(self.task_id, success=True)
            else:
                self.tracker.complete_task(
                    self.task_id, 
                    success=False, 
                    error_message=str(exc_val)
                )
    
    return ProgressContext(task_id, operation_name, total_steps)
