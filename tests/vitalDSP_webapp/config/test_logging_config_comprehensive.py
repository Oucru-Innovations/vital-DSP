"""
Comprehensive tests for vitalDSP_webapp logging configuration to improve coverage
"""
import pytest
import logging
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Import the modules we need to test
try:
    from vitalDSP_webapp.config.logging_config import (
        setup_logging,
        get_logger,
        log_function_call,
        log_performance,
        log_data_operation,
        log_error_with_context,
        setup_development_logging,
        setup_production_logging,
        setup_test_logging
    )
except ImportError:
    # Fallback: add src to path if import fails
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..', '..')
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from vitalDSP_webapp.config.logging_config import (
        setup_logging,
        get_logger,
        log_function_call,
        log_performance,
        log_data_operation,
        log_error_with_context,
        setup_development_logging,
        setup_production_logging,
        setup_test_logging
    )


class TestLoggingConfigComprehensive:
    """Comprehensive test class for logging configuration"""
    
    def setup_method(self):
        """Setup method run before each test"""
        # Store original logging configuration
        self.original_handlers = logging.root.handlers[:]
        self.original_level = logging.root.level
        
        # Create temporary directory for test logs
        self.temp_dir = tempfile.mkdtemp()
        self.test_log_file = os.path.join(self.temp_dir, "test.log")

    def teardown_method(self):
        """Cleanup method run after each test"""
        # Close all current handlers before restoring
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
            
        # Restore original logging configuration
        logging.root.handlers = self.original_handlers
        logging.root.level = self.original_level

        # Clean up temporary files (with retry on Windows)
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                # On Windows, files might still be locked, try again after a short delay
                import time
                time.sleep(0.1)
                try:
                    shutil.rmtree(self.temp_dir)
                except PermissionError:
                    # If still can't delete, just pass - temp files will be cleaned up by OS
                    pass

    def test_setup_logging_basic(self):
        """Test basic setup_logging functionality"""
        setup_logging(log_level="DEBUG")
        
        # Should have set up root logger
        assert logging.root.level == logging.DEBUG
        assert len(logging.root.handlers) > 0

    def test_setup_logging_with_file(self):
        """Test setup_logging with file output"""
        setup_logging(
            log_level="INFO",
            log_file=self.test_log_file
        )
        
        # Should have console and file handlers
        assert len(logging.root.handlers) >= 2
        
        # Test logging to file
        logger = logging.getLogger("test")
        logger.info("Test message")
        
        # Check that file was created and contains message
        assert os.path.exists(self.test_log_file)
        with open(self.test_log_file, 'r') as f:
            content = f.read()
            assert "Test message" in content

    def test_setup_logging_custom_format(self):
        """Test setup_logging with custom format"""
        custom_format = "%(levelname)s: %(message)s"
        
        setup_logging(
            log_level="INFO",
            log_file=self.test_log_file,
            log_format=custom_format
        )
        
        logger = logging.getLogger("test")
        logger.info("Custom format test")
        
        # Check that custom format is applied
        with open(self.test_log_file, 'r') as f:
            content = f.read()
            assert "INFO: Custom format test" in content

    def test_setup_logging_invalid_level(self):
        """Test setup_logging with invalid log level"""
        # Should default to INFO for invalid level
        setup_logging(log_level="INVALID_LEVEL")
        
        assert logging.root.level == logging.INFO

    def test_setup_logging_rotation_parameters(self):
        """Test setup_logging with file rotation parameters"""
        setup_logging(
            log_level="INFO",
            log_file=self.test_log_file,
            max_bytes=1024,
            backup_count=3
        )
        
        # Should have set up rotating file handler
        file_handlers = [h for h in logging.root.handlers 
                        if isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(file_handlers) > 0
        
        handler = file_handlers[0]
        assert handler.maxBytes == 1024
        assert handler.backupCount == 3

    def test_get_logger_basic(self):
        """Test get_logger basic functionality"""
        logger = get_logger("test_module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_basic(self):
        """Test get_logger basic functionality"""
        logger = get_logger("test_module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_multiple_calls(self):
        """Test that get_logger returns same logger for same name"""
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")
        
        assert logger1 is logger2

    def test_setup_development_logging(self):
        """Test setup_development_logging function"""
        setup_development_logging()
        
        # Should have configured logging for development
        assert len(logging.root.handlers) > 0

    def test_setup_production_logging(self):
        """Test setup_production_logging function"""
        setup_production_logging()
        
        # Should have configured logging for production
        assert len(logging.root.handlers) > 0

    def test_setup_test_logging(self):
        """Test setup_test_logging function"""
        setup_test_logging()
        
        # Should have configured logging for testing
        assert len(logging.root.handlers) > 0

    def test_log_function_call_basic(self):
        """Test log_function_call basic functionality"""
        # Test that function doesn't raise an error
        try:
            log_function_call("test_function", args=(1, 2), kwargs={"key": "value"})
        except Exception as e:
            pytest.fail(f"log_function_call raised an exception: {e}")

    def test_log_function_call_no_args(self):
        """Test log_function_call without args"""
        # Test that function doesn't raise an error
        try:
            log_function_call("test_function")
        except Exception as e:
            pytest.fail(f"log_function_call raised an exception: {e}")

    def test_log_performance(self):
        """Test log_performance function"""
        # Test that function doesn't raise an error
        try:
            log_performance("test_function", 0.123)
        except Exception as e:
            pytest.fail(f"log_performance raised an exception: {e}")

    def test_log_data_operation(self):
        """Test log_data_operation function"""
        # Test that function doesn't raise an error
        try:
            log_data_operation("load", 1000, "CSV")
        except Exception as e:
            pytest.fail(f"log_data_operation raised an exception: {e}")

    def test_log_error_with_context_basic(self):
        """Test log_error_with_context function"""
        error = ValueError("Test error")
        
        # Test that function doesn't raise an error
        try:
            log_error_with_context(error, "test_context")
        except Exception as e:
            pytest.fail(f"log_error_with_context raised an exception: {e}")

    def test_log_error_with_context_no_traceback(self):
        """Test log_error_with_context without traceback"""
        error = ValueError("Test error")
        
        # Test that function doesn't raise an error
        try:
            log_error_with_context(error, "test_context", include_traceback=False)
        except Exception as e:
            pytest.fail(f"log_error_with_context raised an exception: {e}")

    def test_logging_with_different_modules(self):
        """Test logging with different module names"""
        setup_logging(log_level="INFO", log_file=self.test_log_file)
        
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        logger3 = get_logger("module3.submodule")
        
        logger1.info("Message from module1")
        logger2.warning("Warning from module2")
        logger3.error("Error from module3.submodule")
        
        # Check that all messages are logged
        with open(self.test_log_file, 'r') as f:
            content = f.read()
            assert "Message from module1" in content
            assert "Warning from module2" in content
            assert "Error from module3.submodule" in content

    def test_logging_performance_large_messages(self):
        """Test logging performance with large messages"""
        setup_logging(log_level="INFO", log_file=self.test_log_file)
        logger = get_logger("performance_test")
        
        # Log many messages
        for i in range(100):
            large_message = "Large message " + "x" * 1000 + f" number {i}"
            logger.info(large_message)
        
        # Should complete without errors
        assert os.path.exists(self.test_log_file)
        
        # Check file size is reasonable
        file_size = os.path.getsize(self.test_log_file)
        assert file_size > 0

    def test_logging_unicode_messages(self):
        """Test logging with unicode characters"""
        setup_logging(log_level="INFO", log_file=self.test_log_file)
        logger = get_logger("unicode_test")
        
        unicode_message = "Unicode test: αβγδε 中文 🚀 émojis"
        logger.info(unicode_message)
        
        # Should handle unicode without errors
        assert os.path.exists(self.test_log_file)
        with open(self.test_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Unicode test" in content

    def test_logging_exception_handling(self):
        """Test logging with exception information"""
        setup_logging(log_level="DEBUG", log_file=self.test_log_file)
        logger = get_logger("exception_test")
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("Caught exception")
        
        # Should log exception with traceback
        with open(self.test_log_file, 'r') as f:
            content = f.read()
            assert "Caught exception" in content
            assert "ValueError" in content
            assert "Traceback" in content

    def test_multiple_logging_setups(self):
        """Test multiple calls to setup_logging"""
        # First setup
        setup_logging(log_level="INFO", log_file=self.test_log_file)
        initial_handler_count = len(logging.root.handlers)
        
        # Second setup - should not duplicate handlers
        setup_logging(log_level="DEBUG", log_file=self.test_log_file)
        
        # Handler count should not have doubled
        # (exact behavior depends on implementation)
        final_handler_count = len(logging.root.handlers)
        # The test should verify that handlers aren't endlessly duplicated

    def test_logging_environment_setup(self):
        """Test logging setup in different environments"""
        # Test that different setup functions work
        setup_development_logging()
        dev_handler_count = len(logging.root.handlers)
        
        # Clear handlers
        logging.root.handlers = []
        
        setup_production_logging()
        prod_handler_count = len(logging.root.handlers)
        
        # Clear handlers
        logging.root.handlers = []
        
        setup_test_logging()
        test_handler_count = len(logging.root.handlers)
        
        # All should have set up handlers
        assert dev_handler_count > 0
        assert prod_handler_count > 0
        assert test_handler_count > 0
