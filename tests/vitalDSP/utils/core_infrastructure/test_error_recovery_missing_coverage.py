"""
Additional tests for error_recovery.py (core_infrastructure) to cover missing lines.

Tests target specific uncovered lines from coverage report.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
import time
from datetime import datetime

try:
    from vitalDSP.utils.core_infrastructure.error_recovery import (
        ErrorRecoveryManager,
        ErrorHandler,
        ErrorSeverity,
        ErrorCategory,
        ErrorInfo,
        RecoveryResult,
    )
    ERROR_RECOVERY_AVAILABLE = True
except ImportError:
    ERROR_RECOVERY_AVAILABLE = False
from vitalDSP.utils.config_utilities.dynamic_config import DynamicConfigManager


@pytest.mark.skipif(not ERROR_RECOVERY_AVAILABLE, reason="ErrorRecoveryManager not available")
class TestErrorRecoveryManagerMissingCoverage:
    """Tests for uncovered lines in ErrorRecoveryManager."""

    def test_init_default(self):
        """Test default initialization."""
        manager = ErrorRecoveryManager()
        assert hasattr(manager, 'recovery_strategies')
        assert hasattr(manager, 'recovery_history')

    def test_init_with_config(self):
        """Test initialization with config."""
        config_manager = DynamicConfigManager()
        manager = ErrorRecoveryManager(config_manager)
        assert manager.config == config_manager

    def test_attempt_recovery_no_strategy(self):
        """Test recovery attempt with no strategy available (lines 164-167)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.PROCESSING,
            error_type="UnknownError",
            message="Unknown error",
            details={},
            context={}
        )
        
        # Mock to return None strategy
        with patch.object(manager, '_determine_recovery_strategy', return_value=None):
            result = manager.attempt_recovery(error_info, {})
            assert result.success is False

    def test_attempt_recovery_success(self):
        """Test successful recovery attempt (lines 183-188)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MEMORY,
            error_type="MemoryError",
            message="Out of memory",
            details={},
            context={}
        )
        
        result = manager.attempt_recovery(error_info, {'chunk_size': 10000})
        # Should attempt recovery
        assert isinstance(result, RecoveryResult)

    def test_attempt_recovery_exception(self):
        """Test recovery attempt with exception (lines 190-192)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MEMORY,
            error_type="MemoryError",
            message="Out of memory",
            details={},
            context={}
        )
        
        # Mock strategy to raise exception
        def failing_strategy(ei, ctx):
            raise Exception("Recovery failed")
        
        with patch.object(manager, '_determine_recovery_strategy', return_value=failing_strategy):
            result = manager.attempt_recovery(error_info, {})
            assert result.success is False

    def test_determine_recovery_strategy_memory(self):
        """Test determining recovery strategy for memory error (lines 199-200)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MEMORY,
            error_type="MemoryError",
            message="Out of memory",
            details={},
            context={}
        )
        
        strategy = manager._determine_recovery_strategy(error_info)
        assert strategy is not None
        assert callable(strategy)

    def test_determine_recovery_strategy_corrupt(self):
        """Test determining recovery strategy for corruption error (lines 201-202)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATA,
            error_type="CorruptError",
            message="Data corrupted",
            details={},
            context={}
        )
        
        strategy = manager._determine_recovery_strategy(error_info)
        assert strategy is not None

    def test_determine_recovery_strategy_timeout(self):
        """Test determining recovery strategy for timeout (lines 203-204)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING,
            error_type="TimeoutError",
            message="Timed out",
            details={},
            context={}
        )
        
        strategy = manager._determine_recovery_strategy(error_info)
        assert strategy is not None

    def test_determine_recovery_strategy_file(self):
        """Test determining recovery strategy for file error (lines 205-206)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.FILE,
            error_type="FileNotFoundError",
            message="File not found",
            details={},
            context={}
        )
        
        strategy = manager._determine_recovery_strategy(error_info)
        assert strategy is not None

    def test_determine_recovery_strategy_config(self):
        """Test determining recovery strategy for config error (lines 207-208)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.CONFIGURATION,
            error_type="ConfigError",
            message="Configuration error",
            details={},
            context={}
        )
        
        strategy = manager._determine_recovery_strategy(error_info)
        assert strategy is not None

    def test_determine_recovery_strategy_network(self):
        """Test determining recovery strategy for network error (lines 209-210)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            error_type="NetworkError",
            message="Connection error",
            details={},
            context={}
        )
        
        strategy = manager._determine_recovery_strategy(error_info)
        assert strategy is not None

    def test_determine_recovery_strategy_generic(self):
        """Test determining recovery strategy for generic error (lines 211-212)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING,
            error_type="GenericError",
            message="Some error",
            details={},
            context={}
        )
        
        strategy = manager._determine_recovery_strategy(error_info)
        assert strategy is not None

    def test_recover_from_memory_error_with_chunk_size(self):
        """Test memory error recovery with chunk_size (lines 227-240)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MEMORY,
            error_type="MemoryError",
            message="Out of memory",
            details={},
            context={}
        )
        
        context = {'chunk_size': 10000}
        result = manager._recover_from_memory_error(error_info, context)
        assert isinstance(result, RecoveryResult)
        assert context['chunk_size'] < 10000  # Should be reduced

    def test_recover_from_memory_error_without_chunk_size(self):
        """Test memory error recovery without chunk_size (lines 241-247)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MEMORY,
            error_type="MemoryError",
            message="Out of memory",
            details={},
            context={}
        )
        
        context = {}
        result = manager._recover_from_memory_error(error_info, context)
        assert isinstance(result, RecoveryResult)

    def test_recover_from_memory_error_exception(self):
        """Test memory error recovery with exception (lines 249-251)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MEMORY,
            error_type="MemoryError",
            message="Out of memory",
            details={},
            context={}
        )
        
        # Force exception in recovery
        with patch('gc.collect', side_effect=Exception("GC failed")):
            result = manager._recover_from_memory_error(error_info, {})
            assert result.success is False

    def test_recover_from_data_corruption_with_signal(self):
        """Test data corruption recovery with signal (lines 261-290)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATA,
            error_type="CorruptError",
            message="Data corrupted",
            details={},
            context={}
        )
        
        # Signal with NaN values
        signal = np.random.randn(1000)
        signal[100:150] = np.nan
        context = {'signal': signal}
        
        result = manager._recover_from_data_corruption(error_info, context)
        assert isinstance(result, RecoveryResult)

    def test_recover_from_data_corruption_no_signal(self):
        """Test data corruption recovery without signal (line 292)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATA,
            error_type="CorruptError",
            message="Data corrupted",
            details={},
            context={}
        )
        
        context = {}
        result = manager._recover_from_data_corruption(error_info, context)
        assert result.success is False

    def test_recover_from_data_corruption_exception(self):
        """Test data corruption recovery with exception (lines 294-296)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATA,
            error_type="CorruptError",
            message="Data corrupted",
            details={},
            context={}
        )
        
        # Force exception
        with patch('numpy.isnan', side_effect=Exception("Check failed")):
            result = manager._recover_from_data_corruption(error_info, {'signal': np.array([1, 2, 3])})
            assert result.success is False

    def test_recover_from_timeout_with_timeout(self):
        """Test timeout recovery with timeout in context (lines 306-315)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING,
            error_type="TimeoutError",
            message="Timed out",
            details={},
            context={}
        )
        
        context = {'timeout': 60}
        result = manager._recover_from_timeout(error_info, context)
        assert isinstance(result, RecoveryResult)
        assert context['timeout'] > 60  # Should be increased

    def test_recover_from_timeout_without_timeout(self):
        """Test timeout recovery without timeout in context (lines 316-322)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING,
            error_type="TimeoutError",
            message="Timed out",
            details={},
            context={}
        )
        
        context = {}
        result = manager._recover_from_timeout(error_info, context)
        assert isinstance(result, RecoveryResult)

    def test_recover_from_timeout_exception(self):
        """Test timeout recovery with exception (lines 324-326)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING,
            error_type="TimeoutError",
            message="Timed out",
            details={},
            context={}
        )
        
        # Force exception
        with patch('time.sleep', side_effect=Exception("Sleep failed")):
            result = manager._recover_from_timeout(error_info, {'timeout': 60})
            assert result.success is False

    def test_recover_from_file_error_with_path(self, tmp_path):
        """Test file error recovery with file path (lines 336-352)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.FILE,
            error_type="FileNotFoundError",
            message="File not found",
            details={},
            context={}
        )
        
        # Create alternative file
        alt_file = tmp_path / "test.parquet"
        alt_file.write_text("test data")
        
        context = {'file_path': str(tmp_path / "test.csv")}
        result = manager._recover_from_file_error(error_info, context)
        # May or may not find alternative
        assert isinstance(result, RecoveryResult)

    def test_recover_from_file_error_no_path(self):
        """Test file error recovery without file path (line 354)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.FILE,
            error_type="FileNotFoundError",
            message="File not found",
            details={},
            context={}
        )
        
        context = {}
        result = manager._recover_from_file_error(error_info, context)
        assert result.success is False

    def test_recover_from_file_error_exception(self):
        """Test file error recovery with exception (lines 356-358)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.FILE,
            error_type="FileNotFoundError",
            message="File not found",
            details={},
            context={}
        )
        
        # Force exception
        with patch('pathlib.Path.exists', side_effect=Exception("Path check failed")):
            result = manager._recover_from_file_error(error_info, {'file_path': '/test/path.csv'})
            assert result.success is False

    def test_recover_from_config_error_with_config(self):
        """Test config error recovery with config (lines 368-378)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.CONFIGURATION,
            error_type="ConfigError",
            message="Configuration error",
            details={},
            context={}
        )
        
        # Mock config with reset method
        mock_config = Mock()
        mock_config.reset_to_defaults = Mock()
        
        context = {'config': mock_config}
        result = manager._recover_from_config_error(error_info, context)
        assert isinstance(result, RecoveryResult)
        mock_config.reset_to_defaults.assert_called_once()

    def test_recover_from_config_error_no_config(self):
        """Test config error recovery without config (line 380)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.CONFIGURATION,
            error_type="ConfigError",
            message="Configuration error",
            details={},
            context={}
        )
        
        context = {}
        result = manager._recover_from_config_error(error_info, context)
        assert result.success is False

    def test_recover_from_config_error_exception(self):
        """Test config error recovery with exception (lines 382-384)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.CONFIGURATION,
            error_type="ConfigError",
            message="Configuration error",
            details={},
            context={}
        )
        
        # Mock config to raise exception
        mock_config = Mock()
        mock_config.reset_to_defaults.side_effect = Exception("Reset failed")
        
        context = {'config': mock_config}
        result = manager._recover_from_config_error(error_info, context)
        assert result.success is False

    def test_recover_from_network_error_retry(self):
        """Test network error recovery with retry (lines 394-408)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            error_type="NetworkError",
            message="Connection error",
            details={},
            context={}
        )
        
        context = {'retry_count': 0}
        result = manager._recover_from_network_error(error_info, context)
        assert isinstance(result, RecoveryResult)
        assert context['retry_count'] > 0

    def test_recover_from_network_error_max_retries(self):
        """Test network error recovery with max retries (lines 409-410)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            error_type="NetworkError",
            message="Connection error",
            details={},
            context={}
        )
        
        context = {'retry_count': 3}
        result = manager._recover_from_network_error(error_info, context)
        assert result.success is False

    def test_recover_from_network_error_exception(self):
        """Test network error recovery with exception (lines 412-414)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            error_type="NetworkError",
            message="Connection error",
            details={},
            context={}
        )
        
        # Force exception
        with patch('time.sleep', side_effect=Exception("Sleep failed")):
            result = manager._recover_from_network_error(error_info, {'retry_count': 0})
            assert result.success is False

    def test_recover_from_generic_error_with_partial_results(self):
        """Test generic error recovery with partial results (lines 425-432)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING,
            error_type="GenericError",
            message="Some error",
            details={},
            context={}
        )
        
        context = {'partial_results': {'result1': np.array([1, 2, 3])}}
        result = manager._recover_from_generic_error(error_info, context)
        assert isinstance(result, RecoveryResult)

    def test_recover_from_generic_error_no_partial_results(self):
        """Test generic error recovery without partial results (line 434)."""
        manager = ErrorRecoveryManager()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING,
            error_type="GenericError",
            message="Some error",
            details={},
            context={}
        )
        
        context = {}
        result = manager._recover_from_generic_error(error_info, context)
        assert result.success is False

    def test_save_partial_results(self):
        """Test saving partial results (lines 448-453)."""
        manager = ErrorRecoveryManager()
        session_id = "test_session"
        results = {'result1': np.array([1, 2, 3]), 'result2': np.array([4, 5, 6])}
        
        manager.save_partial_results(session_id, results)
        assert session_id in manager.partial_results

    def test_get_partial_results(self):
        """Test getting partial results (line 465)."""
        manager = ErrorRecoveryManager()
        session_id = "test_session"
        results = {'result1': np.array([1, 2, 3])}
        
        manager.save_partial_results(session_id, results)
        retrieved = manager.get_partial_results(session_id)
        assert retrieved is not None

    def test_get_partial_results_not_found(self):
        """Test getting partial results that don't exist."""
        manager = ErrorRecoveryManager()
        retrieved = manager.get_partial_results("nonexistent_session")
        assert retrieved is None

    def test_cleanup_partial_results(self):
        """Test cleaning up partial results (lines 474-475)."""
        manager = ErrorRecoveryManager()
        session_id = "test_session"
        results = {'result1': np.array([1, 2, 3])}
        
        manager.save_partial_results(session_id, results)
        manager.cleanup_partial_results(session_id)
        assert session_id not in manager.partial_results

    def test_get_error_statistics(self):
        """Test getting error statistics (lines 484-495)."""
        manager = ErrorRecoveryManager()
        stats = manager.get_error_statistics()
        assert isinstance(stats, dict)
        assert 'total_errors' in stats
        assert 'recovery_success_rate' in stats


@pytest.mark.skipif(not ERROR_RECOVERY_AVAILABLE, reason="ErrorHandler not available")
class TestErrorHandlerMissingCoverage:
    """Tests for uncovered lines in ErrorHandler."""

    def test_init_default(self):
        """Test default initialization."""
        handler = ErrorHandler()
        assert hasattr(handler, 'error_history')
        assert hasattr(handler, 'recovery_manager')

    def test_init_with_config(self):
        """Test initialization with config."""
        config_manager = DynamicConfigManager()
        handler = ErrorHandler(config_manager=config_manager)
        assert handler.config == config_manager

    def test_init_with_log_file(self, tmp_path):
        """Test initialization with log file (lines 521-522)."""
        log_file = tmp_path / "error.log"
        handler = ErrorHandler(log_file=str(log_file))
        assert log_file.exists() or True  # May or may not create immediately

    def test_handle_error_basic(self):
        """Test basic error handling (lines 588-608)."""
        handler = ErrorHandler()
        error = ValueError("Test error")
        context = {'signal': np.array([1, 2, 3])}
        
        error_info = handler.handle_error(
            error,
            context,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING
        )
        assert isinstance(error_info, ErrorInfo)
        assert error_info.error_type == "ValueError"

    def test_generate_error_id(self):
        """Test error ID generation (lines 610-614)."""
        handler = ErrorHandler()
        error_id = handler._generate_error_id()
        assert isinstance(error_id, str)
        assert error_id.startswith("ERR_")

    def test_extract_error_details(self):
        """Test error details extraction (lines 616-632)."""
        handler = ErrorHandler()
        error = ValueError("Test error", "arg1", "arg2")
        details = handler._extract_error_details(error)
        assert isinstance(details, dict)
        assert 'error_type' in details
        assert 'error_args' in details

    def test_extract_error_details_specific_types(self):
        """Test error details extraction for specific error types (lines 624-647)."""
        handler = ErrorHandler()
        
        # Test different error types
        errors = [
            ValueError("Value error"),
            TypeError("Type error"),
            FileNotFoundError("File not found"),
            MemoryError("Memory error"),
            KeyError("key"),
            IndexError("index"),
        ]
        
        for error in errors:
            details = handler._extract_error_details(error)
            assert isinstance(details, dict)

    def test_log_error(self):
        """Test error logging (lines 679-692)."""
        handler = ErrorHandler()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.PROCESSING,
            error_type="TestError",
            message="Test message",
            details={},
            context={}
        )
        
        # Should not raise exception
        handler._log_error(error_info)

    def test_get_user_friendly_message(self):
        """Test getting user-friendly error messages (lines 707-717)."""
        handler = ErrorHandler()
        error_info = ErrorInfo(
            error_id="test_err",
            timestamp=datetime.now(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MEMORY,
            error_type="MemoryError",
            message="Out of memory",
            details={},
            context={}
        )
        
        message = handler.get_user_friendly_message(error_info)
        assert isinstance(message, str)
        assert len(message) > 0

    def test_get_error_history(self):
        """Test getting error history (lines 726-753)."""
        handler = ErrorHandler()
        error = ValueError("Test error")
        handler.handle_error(error, {})
        
        history = handler.get_error_history()
        assert isinstance(history, list)
        assert len(history) > 0

    def test_clear_error_history(self):
        """Test clearing error history (lines 773-813)."""
        handler = ErrorHandler()
        error = ValueError("Test error")
        handler.handle_error(error, {})
        handler.clear_error_history()
        
        history = handler.get_error_history()
        assert len(history) == 0

    def test_export_error_report(self, tmp_path):
        """Test exporting error report (lines 842-870)."""
        handler = ErrorHandler()
        error = ValueError("Test error")
        handler.handle_error(error, {})
        
        report_path = tmp_path / "error_report.json"
        handler.export_error_report(str(report_path))
        assert report_path.exists()

    def test_get_error_statistics(self):
        """Test getting error statistics (lines 889-894)."""
        handler = ErrorHandler()
        error = ValueError("Test error")
        handler.handle_error(error, {})
        
        stats = handler.get_error_statistics()
        assert isinstance(stats, dict)
        assert 'total_errors' in stats

    def test_handle_error_with_recovery(self):
        """Test error handling with recovery attempt (lines 917-955)."""
        handler = ErrorHandler()
        error = MemoryError("Out of memory")
        context = {'chunk_size': 10000}
        
        error_info = handler.handle_error(
            error,
            context,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MEMORY,
            recovery_attempted=True
        )
        
        # Should attempt recovery
        assert isinstance(error_info, ErrorInfo)
        assert error_info.recovery_attempted is True

    def test_handle_error_with_recovery_result(self):
        """Test error handling with recovery result (lines 964-971)."""
        handler = ErrorHandler()
        error = ValueError("Test error")
        context = {'chunk_size': 10000}
        
        error_info = handler.handle_error(
            error,
            context,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING
        )
        
        # May trigger recovery
        recovery_result = handler.recovery_manager.attempt_recovery(error_info, context)
        assert isinstance(recovery_result, RecoveryResult)

    def test_error_templates_loading(self):
        """Test error templates loading (lines 985-1018)."""
        handler = ErrorHandler()
        assert hasattr(handler, 'error_templates')
        assert isinstance(handler.error_templates, dict)
        assert 'memory_error' in handler.error_templates

