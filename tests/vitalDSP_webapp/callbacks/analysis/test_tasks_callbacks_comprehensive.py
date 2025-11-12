"""
Comprehensive unit tests for tasks_callbacks.py module.

This test file adds extensive coverage for task management callbacks.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dash import Dash
from datetime import datetime

# Import the module to test
from vitalDSP_webapp.callbacks.analysis.tasks_callbacks import (
    register_tasks_callbacks,
)


@pytest.fixture
def mock_app():
    """Create a mock Dash app for testing."""
    app = Mock(spec=Dash)
    app.callback = Mock()
    return app


@pytest.fixture
def mock_task():
    """Create a mock task object."""
    task = Mock()
    task.task_id = "test_task_123"
    task.operation_name = "test_operation"
    task.status = Mock()
    task.status.value = "running"
    task.progress_percentage = 50.0
    task.current_step = "Processing"
    task.start_time = datetime.now()
    task.end_time = None
    task.error_message = None
    task.metadata = {}
    return task


class TestTasksCallbacksRegistration:
    """Test task callback registration."""

    def test_register_tasks_callbacks(self, mock_app):
        """Test that task callbacks are properly registered."""
        register_tasks_callbacks(mock_app)
        
        # Verify that callback decorator was called
        assert mock_app.callback.called
        assert mock_app.callback.call_count >= 1

    def test_update_tasks_display_callback_registered(self, mock_app):
        """Test that update_tasks_display callback is registered."""
        register_tasks_callbacks(mock_app)
        
        callback_found = False
        for call in mock_app.callback.call_args_list:
            outputs = call[0][0]
            if isinstance(outputs, list) and len(outputs) >= 5:
                if any("active-tasks-list" in str(out) for out in outputs):
                    callback_found = True
                    break
        
        assert callback_found, "update_tasks_display callback should be registered"


class TestTasksCallbacksFunctionality:
    """Test task callback functionality."""

    @patch('vitalDSP_webapp.callbacks.analysis.tasks_callbacks.get_progress_tracker')
    def test_update_tasks_display_no_tasks(self, mock_get_tracker, mock_app):
        """Test update_tasks_display with no active tasks."""
        mock_tracker = Mock()
        mock_tracker.get_all_active_tasks.return_value = []
        mock_get_tracker.return_value = mock_tracker
        
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_tasks_callbacks(mock_app)
        
        # Find update_tasks_display callback
        update_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_tasks_display':
                update_callback = func
                break
        
        if update_callback:
            result = update_callback(0, 0)
            assert len(result) == 5
            assert result[1] == "0"  # total_count
            assert result[2] == "0"  # running_count
            assert result[3] == "0"  # completed_count
            assert result[4] == "0"  # failed_count

    @patch('vitalDSP_webapp.callbacks.analysis.tasks_callbacks.get_progress_tracker')
    @patch('vitalDSP_webapp.callbacks.analysis.tasks_callbacks.create_task_item')
    def test_update_tasks_display_with_tasks(self, mock_create_item, mock_get_tracker, mock_app, mock_task):
        """Test update_tasks_display with active tasks."""
        mock_tracker = Mock()
        mock_tracker.get_all_active_tasks.return_value = [mock_task]
        mock_get_tracker.return_value = mock_tracker
        
        mock_create_item.return_value = Mock()
        
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_tasks_callbacks(mock_app)
        
        # Find update_tasks_display callback
        update_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_tasks_display':
                update_callback = func
                break
        
        if update_callback:
            result = update_callback(0, 0)
            assert len(result) == 5
            assert result[1] == "1"  # total_count
            assert result[2] == "1"  # running_count
            assert result[3] == "0"  # completed_count
            assert result[4] == "0"  # failed_count
            mock_create_item.assert_called_once()

    @patch('vitalDSP_webapp.callbacks.analysis.tasks_callbacks.get_progress_tracker')
    @patch('vitalDSP_webapp.callbacks.analysis.tasks_callbacks.create_task_item')
    def test_update_tasks_display_multiple_statuses(self, mock_create_item, mock_get_tracker, mock_app):
        """Test update_tasks_display with tasks of different statuses."""
        # Create tasks with different statuses
        running_task = Mock()
        running_task.task_id = "task1"
        running_task.operation_name = "op1"
        running_task.status = Mock()
        running_task.status.value = "running"
        running_task.progress_percentage = 50.0
        running_task.current_step = "Processing"
        running_task.start_time = datetime.now()
        running_task.end_time = None
        running_task.error_message = None
        running_task.metadata = {}
        
        completed_task = Mock()
        completed_task.task_id = "task2"
        completed_task.operation_name = "op2"
        completed_task.status = Mock()
        completed_task.status.value = "completed"
        completed_task.progress_percentage = 100.0
        completed_task.current_step = "Done"
        completed_task.start_time = datetime.now()
        completed_task.end_time = datetime.now()
        completed_task.error_message = None
        completed_task.metadata = {}
        
        failed_task = Mock()
        failed_task.task_id = "task3"
        failed_task.operation_name = "op3"
        failed_task.status = Mock()
        failed_task.status.value = "failed"
        failed_task.progress_percentage = 30.0
        failed_task.current_step = "Error"
        failed_task.start_time = datetime.now()
        failed_task.end_time = datetime.now()
        failed_task.error_message = "Test error"
        failed_task.metadata = {}
        
        mock_tracker = Mock()
        mock_tracker.get_all_active_tasks.return_value = [running_task, completed_task, failed_task]
        mock_get_tracker.return_value = mock_tracker
        
        mock_create_item.return_value = Mock()
        
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_tasks_callbacks(mock_app)
        
        # Find update_tasks_display callback
        update_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_tasks_display':
                update_callback = func
                break
        
        if update_callback:
            result = update_callback(0, 0)
            assert len(result) == 5
            assert result[1] == "3"  # total_count
            assert result[2] == "1"  # running_count
            assert result[3] == "1"  # completed_count
            assert result[4] == "1"  # failed_count

    @patch('vitalDSP_webapp.callbacks.analysis.tasks_callbacks.get_progress_tracker')
    def test_update_tasks_display_exception_handling(self, mock_get_tracker, mock_app):
        """Test update_tasks_display exception handling."""
        mock_tracker = Mock()
        mock_tracker.get_all_active_tasks.side_effect = Exception("Tracker error")
        mock_get_tracker.return_value = mock_tracker
        
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_tasks_callbacks(mock_app)
        
        # Find update_tasks_display callback
        update_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_tasks_display':
                update_callback = func
                break
        
        if update_callback:
            # Should handle exception gracefully
            result = update_callback(0, 0)
            assert len(result) == 5
            # Should return default values on error
            assert result[1] == "0"  # total_count

