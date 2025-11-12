"""
Comprehensive tests for header_monitoring_callbacks.py module.

This test file adds extensive coverage to reach 60%+ coverage.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dash import Dash

# Import the module to test
from vitalDSP_webapp.callbacks.core.header_monitoring_callbacks import (
    register_header_monitoring_callbacks,
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
    task.status = Mock()
    task.status.value = "running"
    return task


class TestHeaderMonitoringCallbacksRegistration:
    """Test callback registration."""

    def test_register_header_monitoring_callbacks(self, mock_app):
        """Test that callbacks are properly registered."""
        register_header_monitoring_callbacks(mock_app)
        assert mock_app.callback.called
        assert mock_app.callback.call_count >= 1


class TestHeaderMonitoringCallbacks:
    """Test header monitoring callback functionality."""

    @patch('vitalDSP_webapp.callbacks.core.header_monitoring_callbacks.get_progress_tracker')
    @patch('vitalDSP_webapp.callbacks.core.header_monitoring_callbacks.psutil')
    def test_update_header_monitoring_low_memory(self, mock_psutil, mock_get_tracker, mock_app):
        """Test update_header_monitoring with low memory usage."""
        # Mock psutil
        mock_memory = Mock()
        mock_memory.percent = 50.0
        mock_memory.used = 4 * (1024**3)  # 4 GB
        mock_memory.total = 8 * (1024**3)  # 8 GB
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.disk_usage.return_value = Mock(used=100*(1024**3), total=500*(1024**3))
        
        # Mock progress tracker
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
        register_header_monitoring_callbacks(mock_app)
        
        # Find update_header_monitoring callback
        monitoring_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_header_monitoring':
                monitoring_callback = func
                break
        
        if monitoring_callback:
            result = monitoring_callback(0)
            assert len(result) == 6
            assert result[1] == "success"  # Memory color should be success for < 70%
            assert result[3] == "info"  # Tasks color should be info for 0 tasks
            assert result[5] == "success"  # System color should be success

    @patch('vitalDSP_webapp.callbacks.core.header_monitoring_callbacks.get_progress_tracker')
    @patch('vitalDSP_webapp.callbacks.core.header_monitoring_callbacks.psutil')
    def test_update_header_monitoring_high_memory(self, mock_psutil, mock_get_tracker, mock_app):
        """Test update_header_monitoring with high memory usage."""
        # Mock psutil
        mock_memory = Mock()
        mock_memory.percent = 90.0
        mock_memory.used = 7.2 * (1024**3)  # 7.2 GB
        mock_memory.total = 8 * (1024**3)  # 8 GB
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.disk_usage.return_value = Mock(used=100*(1024**3), total=500*(1024**3))
        
        # Mock progress tracker
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
        register_header_monitoring_callbacks(mock_app)
        
        # Find update_header_monitoring callback
        monitoring_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_header_monitoring':
                monitoring_callback = func
                break
        
        if monitoring_callback:
            result = monitoring_callback(0)
            assert len(result) == 6
            assert result[1] == "danger"  # Memory color should be danger for >= 85%

    @patch('vitalDSP_webapp.callbacks.core.header_monitoring_callbacks.get_progress_tracker')
    @patch('vitalDSP_webapp.callbacks.core.header_monitoring_callbacks.psutil')
    def test_update_header_monitoring_with_tasks(self, mock_psutil, mock_get_tracker, mock_app, mock_task):
        """Test update_header_monitoring with active tasks."""
        # Mock psutil
        mock_memory = Mock()
        mock_memory.percent = 50.0
        mock_memory.used = 4 * (1024**3)
        mock_memory.total = 8 * (1024**3)
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.disk_usage.return_value = Mock(used=100*(1024**3), total=500*(1024**3))
        
        # Mock progress tracker with running tasks
        mock_tracker = Mock()
        mock_tracker.get_all_active_tasks.return_value = [mock_task, mock_task]
        mock_get_tracker.return_value = mock_tracker
        
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_header_monitoring_callbacks(mock_app)
        
        # Find update_header_monitoring callback
        monitoring_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_header_monitoring':
                monitoring_callback = func
                break
        
        if monitoring_callback:
            result = monitoring_callback(0)
            assert len(result) == 6
            assert "2" in result[2]  # Should show 2 tasks
            assert result[3] == "warning"  # Tasks color should be warning for < 3 tasks

    @patch('vitalDSP_webapp.callbacks.core.header_monitoring_callbacks.get_progress_tracker')
    @patch('vitalDSP_webapp.callbacks.core.header_monitoring_callbacks.psutil')
    def test_update_header_monitoring_overloaded(self, mock_psutil, mock_get_tracker, mock_app):
        """Test update_header_monitoring with overloaded system."""
        # Mock psutil - overloaded system
        mock_memory = Mock()
        mock_memory.percent = 96.0
        mock_memory.used = 7.68 * (1024**3)
        mock_memory.total = 8 * (1024**3)
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_percent.return_value = 96.0
        mock_psutil.disk_usage.return_value = Mock(used=480*(1024**3), total=500*(1024**3))
        
        # Mock progress tracker
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
        register_header_monitoring_callbacks(mock_app)
        
        # Find update_header_monitoring callback
        monitoring_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_header_monitoring':
                monitoring_callback = func
                break
        
        if monitoring_callback:
            result = monitoring_callback(0)
            assert len(result) == 6
            assert result[4] == "System: Overloaded"  # System status should be Overloaded
            assert result[5] == "danger"  # System color should be danger

    @patch('vitalDSP_webapp.callbacks.core.header_monitoring_callbacks.get_progress_tracker')
    @patch('vitalDSP_webapp.callbacks.core.header_monitoring_callbacks.psutil')
    def test_update_header_monitoring_exception_handling(self, mock_psutil, mock_get_tracker, mock_app):
        """Test update_header_monitoring exception handling."""
        mock_psutil.virtual_memory.side_effect = Exception("psutil error")
        
        captured_callbacks = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_header_monitoring_callbacks(mock_app)
        
        # Find update_header_monitoring callback
        monitoring_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'update_header_monitoring':
                monitoring_callback = func
                break
        
        if monitoring_callback:
            # Should handle exception gracefully
            result = monitoring_callback(0)
            assert len(result) == 6
            # Should return default values on error

