"""
Tests for vitalDSP_webapp utils error_handler
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from dash import html
from datetime import datetime

# Import the modules we need to test
try:
    from vitalDSP_webapp.utils.error_handler import (
        WebappError,
        DataProcessingError,
        FileUploadError,
        AnalysisError,
        ValidationError,
        handle_error,
        create_error_alert,
        create_success_alert,
        create_warning_alert,
        create_info_alert
    )
except ImportError:
    # Fallback: add src to path if import fails
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..', '..')
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from vitalDSP_webapp.utils.error_handler import (
        WebappError,
        DataProcessingError,
        FileUploadError,
        AnalysisError,
        ValidationError,
        handle_error,
        create_error_alert,
        create_success_alert,
        create_warning_alert,
        create_info_alert
    )


class TestWebappError:
    """Test class for WebappError base exception"""
    
    def test_webapp_error_basic(self):
        """Test basic WebappError creation"""
        error = WebappError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.details == {}
        assert hasattr(error, 'timestamp')
        assert isinstance(error.timestamp, datetime)

    def test_webapp_error_with_code(self):
        """Test WebappError with error code"""
        error = WebappError("Test error message", error_code="TEST_001")
        
        assert str(error) == "[TEST_001] Test error message"
        assert error.error_code == "TEST_001"
        assert error.error_type == "TEST_001"  # Backward compatibility

    def test_webapp_error_with_details(self):
        """Test WebappError with details"""
        details = {"user_id": 123, "action": "upload"}
        error = WebappError("Test error", details=details)
        
        assert error.details == details
        assert error.context == details  # Backward compatibility

    def test_webapp_error_inheritance(self):
        """Test that WebappError inherits from Exception"""
        error = WebappError("Test error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, WebappError)

    def test_webapp_error_timestamp(self):
        """Test that WebappError has a timestamp"""
        error = WebappError("Test error")
        
        assert hasattr(error, 'timestamp')
        assert isinstance(error.timestamp, datetime)
        # Should be recent
        assert (datetime.now() - error.timestamp).total_seconds() < 1


class TestSpecificErrors:
    """Test class for specific error types"""
    
    def test_data_processing_error(self):
        """Test DataProcessingError"""
        error = DataProcessingError("Data processing failed", error_code="DP_001")
        
        assert isinstance(error, WebappError)
        assert isinstance(error, DataProcessingError)
        assert str(error) == "[DP_001] Data processing failed"

    def test_file_upload_error(self):
        """Test FileUploadError"""
        error = FileUploadError("File upload failed", error_code="FU_001")
        
        assert isinstance(error, WebappError)
        assert isinstance(error, FileUploadError)
        assert str(error) == "[FU_001] File upload failed"

    def test_analysis_error(self):
        """Test AnalysisError"""
        error = AnalysisError("Analysis failed", error_code="AN_001")
        
        assert isinstance(error, WebappError)
        assert isinstance(error, AnalysisError)
        assert str(error) == "[AN_001] Analysis failed"

    def test_validation_error(self):
        """Test ValidationError"""
        error = ValidationError("Validation failed", error_code="VAL_001")
        
        assert isinstance(error, WebappError)
        assert isinstance(error, ValidationError)
        assert str(error) == "[VAL_001] Validation failed"


class TestErrorHandlerFunctions:
    """Test class for error handler utility functions"""
    
    def test_handle_error_basic(self):
        """Test basic error handling"""
        try:
            raise ValueError("Test error")
        except Exception as e:
            result = handle_error(e, "test_context")
            
            assert isinstance(result, dict)
            assert "type" in result
            assert "message" in result
            assert "context" in result
            assert result["context"] == "test_context"

    def test_handle_error_webapp_error(self):
        """Test handling of WebappError"""
        error = WebappError("Custom error", error_code="CUSTOM_001", details={"key": "value"})
        
        result = handle_error(error, "test_context")
        
        assert isinstance(result, dict)
        assert result["message"] == "[CUSTOM_001] Custom error"  # Implementation includes error code in message
        assert result["type"] == "WebappError"

    def test_handle_error_standard_exception(self):
        """Test handling of standard Python exceptions"""
        error = ValueError("Standard error")
        
        result = handle_error(error, "test_context")
        
        assert isinstance(result, dict)
        assert result["message"] == "Standard error"
        assert result["type"] == "ValueError"

    def test_create_error_alert_basic(self):
        """Test creating basic error alert"""
        result = create_error_alert("Test error message")
        
        # The function returns a Dash component
        assert result is not None
        # Check that the error message is included
        assert "Test error message" in str(result)

    def test_create_error_alert_with_title(self):
        """Test creating error alert with custom title"""
        result = create_error_alert("Test error", title="Custom Title")
        
        assert result is not None
        assert "Custom Title" in str(result)
        assert "Test error" in str(result)

    def test_create_error_alert_dismissible(self):
        """Test creating dismissible error alert"""
        result = create_error_alert("Test error", dismissible=True)
        
        assert result is not None
        # Should contain dismissible elements
        assert "Test error" in str(result)

    def test_create_success_alert_basic(self):
        """Test creating basic success alert"""
        result = create_success_alert("Success message")
        
        assert result is not None
        assert "Success message" in str(result)

    def test_create_warning_alert_basic(self):
        """Test creating basic warning alert"""
        result = create_warning_alert("Warning message")
        
        assert result is not None
        assert "Warning message" in str(result)

    def test_create_info_alert_basic(self):
        """Test creating basic info alert"""
        result = create_info_alert("Info message")
        
        assert result is not None
        assert "Info message" in str(result)

    @patch('vitalDSP_webapp.utils.error_handler.logger')
    def test_logging_integration(self, mock_logger):
        """Test that logging is properly integrated"""
        error = ValueError("Test error")
        handle_error(error, "test_context")
        
        # Should have logged something
        assert mock_logger.error.called

    def test_error_handling_robustness(self):
        """Test error handling robustness with edge cases"""
        # Test with empty string error
        try:
            raise ValueError("")
        except Exception as e:
            result = handle_error(e, "test_context")
            assert isinstance(result, dict)

    def test_alert_creation_robustness(self):
        """Test alert creation robustness"""
        # Test with empty message
        result = create_error_alert("")
        assert result is not None
        
        # Test with very long message
        long_message = "x" * 1000
        result = create_error_alert(long_message)
        assert result is not None

    def test_error_context_preservation(self):
        """Test that error context is preserved through handling"""
        original_error = ValueError("Original error")
        
        result = handle_error(original_error, "test_context")
        
        assert result["context"] == "test_context"
        assert result["type"] == "ValueError"
        assert result["message"] == "Original error"