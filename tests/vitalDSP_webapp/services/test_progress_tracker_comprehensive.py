"""
Comprehensive tests for progress_tracker.py to improve coverage.

This file adds extensive coverage for progress tracking service.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from vitalDSP_webapp.services.progress_tracker import (
    ProgressTracker,
    ProgressStatus,
    ProgressInfo,
)


@pytest.fixture
def progress_tracker():
    """Create a ProgressTracker instance for testing."""
    return ProgressTracker()


@pytest.fixture
def sample_callback():
    """Create a sample callback function."""
    callback = Mock()
    return callback


class TestProgressTrackerInit:
    """Test ProgressTracker initialization."""

    def test_init(self):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker()
        assert tracker.active_tasks == {}
        assert tracker.progress_callbacks == {}
        assert tracker.cancellation_flags == {}


class TestProgressTrackerStartTask:
    """Test start_task method."""

    def test_start_task_basic(self, progress_tracker):
        """Test basic task start."""
        task_id = progress_tracker.start_task("test_operation", total_steps=100)
        
        assert isinstance(task_id, str)
        assert task_id in progress_tracker.active_tasks
        task = progress_tracker.active_tasks[task_id]
        assert task.operation_name == "test_operation"
        assert task.status == ProgressStatus.PENDING
        assert task.progress_percentage == 0.0

    def test_start_task_with_callback(self, progress_tracker, sample_callback):
        """Test start_task with callback."""
        task_id = progress_tracker.start_task(
            "test_operation", 
            total_steps=50,
            progress_callback=sample_callback
        )
        
        assert task_id in progress_tracker.progress_callbacks
        assert progress_tracker.progress_callbacks[task_id] == sample_callback

    def test_start_task_with_metadata(self, progress_tracker):
        """Test start_task with metadata."""
        metadata = {"key1": "value1", "key2": 42}
        task_id = progress_tracker.start_task(
            "test_operation",
            metadata=metadata
        )
        
        task = progress_tracker.active_tasks[task_id]
        assert task.metadata == metadata

    def test_start_task_default_total_steps(self, progress_tracker):
        """Test start_task with default total_steps."""
        task_id = progress_tracker.start_task("test_operation")
        task = progress_tracker.active_tasks[task_id]
        assert task.total_steps == 100


class TestProgressTrackerUpdateProgress:
    """Test update_progress method."""

    def test_update_progress_basic(self, progress_tracker):
        """Test basic progress update."""
        task_id = progress_tracker.start_task("test_operation")
        result = progress_tracker.update_progress(
            task_id, 
            progress_percentage=50.0,
            current_step="Halfway done"
        )
        
        assert result is True
        task = progress_tracker.active_tasks[task_id]
        assert task.progress_percentage == 50.0
        assert task.current_step == "Halfway done"
        assert task.status == ProgressStatus.RUNNING

    def test_update_progress_with_step_number(self, progress_tracker):
        """Test progress update with step number."""
        task_id = progress_tracker.start_task("test_operation", total_steps=10)
        progress_tracker.update_progress(
            task_id,
            progress_percentage=30.0,
            current_step="Step 3",
            step_number=3
        )
        
        task = progress_tracker.active_tasks[task_id]
        assert task.current_step_number == 3

    def test_update_progress_with_metadata(self, progress_tracker):
        """Test progress update with metadata."""
        task_id = progress_tracker.start_task("test_operation")
        metadata = {"processed_items": 50}
        progress_tracker.update_progress(
            task_id,
            progress_percentage=50.0,
            current_step="Processing",
            metadata=metadata
        )
        
        task = progress_tracker.active_tasks[task_id]
        assert "processed_items" in task.metadata
        assert task.metadata["processed_items"] == 50

    def test_update_progress_clamps_percentage(self, progress_tracker):
        """Test that progress percentage is clamped to 0-100."""
        task_id = progress_tracker.start_task("test_operation")
        
        # Test over 100
        progress_tracker.update_progress(task_id, 150.0, "Over 100")
        assert progress_tracker.active_tasks[task_id].progress_percentage == 100.0
        
        # Test under 0
        progress_tracker.update_progress(task_id, -10.0, "Under 0")
        assert progress_tracker.active_tasks[task_id].progress_percentage == 0.0

    def test_update_progress_nonexistent_task(self, progress_tracker):
        """Test update_progress with nonexistent task."""
        result = progress_tracker.update_progress(
            "nonexistent_id",
            progress_percentage=50.0,
            current_step="Test"
        )
        assert result is False

    def test_update_progress_cancelled_task(self, progress_tracker):
        """Test update_progress on cancelled task."""
        task_id = progress_tracker.start_task("test_operation")
        progress_tracker.cancel_task(task_id)
        
        result = progress_tracker.update_progress(
            task_id,
            progress_percentage=50.0,
            current_step="Should not update"
        )
        assert result is False

    def test_update_progress_callback(self, progress_tracker, sample_callback):
        """Test that callback is called on progress update."""
        task_id = progress_tracker.start_task(
            "test_operation",
            progress_callback=sample_callback
        )
        progress_tracker.update_progress(
            task_id,
            progress_percentage=25.0,
            current_step="Quarter done"
        )
        
        sample_callback.assert_called_once()
        call_args = sample_callback.call_args[0][0]
        assert isinstance(call_args, ProgressInfo)
        assert call_args.progress_percentage == 25.0

    def test_update_progress_callback_exception(self, progress_tracker):
        """Test that callback exceptions are handled."""
        def failing_callback(progress_info):
            raise Exception("Callback error")
        
        task_id = progress_tracker.start_task(
            "test_operation",
            progress_callback=failing_callback
        )
        # Should not raise exception
        result = progress_tracker.update_progress(
            task_id,
            progress_percentage=50.0,
            current_step="Test"
        )
        assert result is True


class TestProgressTrackerCompleteTask:
    """Test complete_task method."""

    def test_complete_task_success(self, progress_tracker):
        """Test completing a task successfully."""
        task_id = progress_tracker.start_task("test_operation")
        progress_tracker.complete_task(task_id, success=True)
        
        task = progress_tracker.active_tasks[task_id]
        assert task.status == ProgressStatus.COMPLETED
        assert task.progress_percentage == 100.0
        assert task.current_step == "Completed"
        assert task.end_time is not None

    def test_complete_task_failure(self, progress_tracker):
        """Test completing a task with failure."""
        task_id = progress_tracker.start_task("test_operation")
        error_message = "Test error"
        progress_tracker.complete_task(task_id, success=False, error_message=error_message)
        
        task = progress_tracker.active_tasks[task_id]
        assert task.status == ProgressStatus.FAILED
        assert task.error_message == error_message
        assert task.current_step == "Failed"

    def test_complete_task_with_metadata(self, progress_tracker):
        """Test completing a task with metadata."""
        task_id = progress_tracker.start_task("test_operation")
        metadata = {"result_count": 100}
        progress_tracker.complete_task(task_id, success=True, metadata=metadata)
        
        task = progress_tracker.active_tasks[task_id]
        assert "result_count" in task.metadata
        assert task.metadata["result_count"] == 100

    def test_complete_task_nonexistent(self, progress_tracker):
        """Test completing a nonexistent task."""
        # Should not raise exception
        progress_tracker.complete_task("nonexistent_id", success=True)

    def test_complete_task_callback(self, progress_tracker, sample_callback):
        """Test that callback is called on completion."""
        task_id = progress_tracker.start_task(
            "test_operation",
            progress_callback=sample_callback
        )
        progress_tracker.complete_task(task_id, success=True)
        
        assert sample_callback.call_count >= 1


class TestProgressTrackerCancelTask:
    """Test cancel_task method."""

    def test_cancel_task_basic(self, progress_tracker):
        """Test basic task cancellation."""
        task_id = progress_tracker.start_task("test_operation")
        result = progress_tracker.cancel_task(task_id)
        
        assert result is True
        assert progress_tracker.cancellation_flags[task_id] is True
        task = progress_tracker.active_tasks[task_id]
        assert task.status == ProgressStatus.CANCELLED
        assert task.end_time is not None

    def test_cancel_task_nonexistent(self, progress_tracker):
        """Test cancelling a nonexistent task."""
        result = progress_tracker.cancel_task("nonexistent_id")
        assert result is False


class TestProgressTrackerGetTaskProgress:
    """Test get_task_progress method."""

    def test_get_task_progress_existing(self, progress_tracker):
        """Test getting progress for existing task."""
        task_id = progress_tracker.start_task("test_operation")
        progress_tracker.update_progress(task_id, 50.0, "Halfway")
        
        progress = progress_tracker.get_task_progress(task_id)
        assert progress is not None
        assert isinstance(progress, ProgressInfo)
        assert progress.progress_percentage == 50.0

    def test_get_task_progress_nonexistent(self, progress_tracker):
        """Test getting progress for nonexistent task."""
        progress = progress_tracker.get_task_progress("nonexistent_id")
        assert progress is None


class TestProgressTrackerGetAllActiveTasks:
    """Test get_all_active_tasks method."""

    def test_get_all_active_tasks_empty(self, progress_tracker):
        """Test getting all tasks when none exist."""
        tasks = progress_tracker.get_all_active_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) == 0

    def test_get_all_active_tasks_multiple(self, progress_tracker):
        """Test getting all tasks with multiple tasks."""
        task_id1 = progress_tracker.start_task("operation1")
        task_id2 = progress_tracker.start_task("operation2")
        
        tasks = progress_tracker.get_all_active_tasks()
        assert len(tasks) == 2
        task_ids = [task.task_id for task in tasks]
        assert task_id1 in task_ids
        assert task_id2 in task_ids


class TestProgressTrackerCleanupTask:
    """Test cleanup_task method."""

    def test_cleanup_task(self, progress_tracker):
        """Test cleaning up a task."""
        task_id = progress_tracker.start_task("test_operation")
        progress_tracker.complete_task(task_id, success=True)
        
        progress_tracker.cleanup_task(task_id)
        
        assert task_id not in progress_tracker.active_tasks
        assert task_id not in progress_tracker.progress_callbacks
        assert task_id not in progress_tracker.cancellation_flags


class TestProgressTrackerCleanupCompletedTasks:
    """Test cleanup_completed_tasks method."""

    def test_cleanup_completed_tasks_old(self, progress_tracker):
        """Test cleaning up old completed tasks."""
        task_id = progress_tracker.start_task("test_operation")
        progress_tracker.complete_task(task_id, success=True)
        
        # Mock old end_time
        with patch.object(progress_tracker.active_tasks[task_id], 'end_time', 
                         datetime.now() - timedelta(hours=25)):
            progress_tracker.cleanup_completed_tasks(max_age_hours=24)
        
        # Task should be cleaned up
        assert task_id not in progress_tracker.active_tasks

    def test_cleanup_completed_tasks_recent(self, progress_tracker):
        """Test that recent completed tasks are not cleaned up."""
        task_id = progress_tracker.start_task("test_operation")
        progress_tracker.complete_task(task_id, success=True)
        
        # Should not clean up recent tasks
        progress_tracker.cleanup_completed_tasks(max_age_hours=24)
        
        # Task should still exist
        assert task_id in progress_tracker.active_tasks

    def test_cleanup_completed_tasks_running(self, progress_tracker):
        """Test that running tasks are not cleaned up."""
        task_id = progress_tracker.start_task("test_operation")
        progress_tracker.update_progress(task_id, 50.0, "Running")
        
        progress_tracker.cleanup_completed_tasks(max_age_hours=24)
        
        # Running task should still exist
        assert task_id in progress_tracker.active_tasks


class TestProgressTrackerLoadingProgress:
    """Test to_loading_progress and from_loading_progress methods."""

    def test_to_loading_progress(self, progress_tracker):
        """Test converting to LoadingProgress."""
        task_id = progress_tracker.start_task("test_operation", total_steps=100)
        progress_tracker.update_progress(task_id, 50.0, "Halfway", step_number=50)
        
        loading_progress = progress_tracker.to_loading_progress(task_id)
        
        # Should return dict or LoadingProgress object
        assert loading_progress is not None
        if isinstance(loading_progress, dict):
            # LoadingProgress has various fields, just check it's a valid dict
            assert len(loading_progress) > 0

    def test_from_loading_progress(self, progress_tracker):
        """Test creating from LoadingProgress."""
        # Create a mock LoadingProgress-like dict
        loading_data = {
            "operation": "test_operation",
            "progress": 50.0,
            "current_step": "Halfway",
            "total_steps": 100
        }
        
        task_id = progress_tracker.from_loading_progress(loading_data)
        
        assert isinstance(task_id, str)
        assert task_id in progress_tracker.active_tasks

