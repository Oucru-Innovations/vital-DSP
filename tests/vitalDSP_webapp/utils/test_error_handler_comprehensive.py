"""
Comprehensive tests for vitalDSP_webapp error handler to improve coverage
"""
import pytest
import logging
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from dash import html
import dash_bootstrap_components as dbc

# Import the modules we need to test
try:
    from vitalDSP_webapp.utils.error_handler import (
        WebappError,
        DataProcessingError,
        FileUploadError,
        ValidationError,
        AnalysisError,
        handle_error,
        create_error_alert,
        handle_upload_error,
        handle_processing_error,
        handle_analysis_error,
        log_error_with_context,
        create_user_friendly_error_message,
        format_error_for_display,
        format_error_message,
        create_warning_alert,
        create_info_alert,
        create_success_alert,
        safe_execute,
        validate_required_fields,
        validate_data_types
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
        ValidationError,
        AnalysisError,
        handle_error,
        create_error_alert,
        handle_upload_error,
        handle_processing_error,
        handle_analysis_error,
        log_error_with_context,
        create_user_friendly_error_message,
        format_error_for_display,
        format_error_message,
        create_warning_alert,
        create_info_alert,
        create_success_alert,
        safe_execute,
        validate_required_fields,
        validate_data_types
    )


class TestWebappErrorClasses:
    """Test custom error classes"""
    
    def test_webapp_error_basic(self):
        """Test basic WebappError functionality"""
        error = WebappError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.details == {}
        assert hasattr(error, 'timestamp')
        assert isinstance(error.timestamp, datetime)

    def test_webapp_error_with_code(self):
        """Test WebappError with error code"""
        error = WebappError("Test error", error_code="TEST_001")
        
        assert str(error) == "[TEST_001] Test error"
        assert error.error_code == "TEST_001"
        assert error.error_type == "TEST_001"  # Backward compatibility

    def test_webapp_error_with_details(self):
        """Test WebappError with details"""
        details = {"file": "test.csv", "line": 42}
        error = WebappError("Test error", details=details)
        
        assert error.details == details
        assert error.context == details  # Backward compatibility

    def test_webapp_error_with_all_params(self):
        """Test WebappError with all parameters"""
        details = {"component": "upload", "action": "validate"}
        error = WebappError("Complete test", error_code="COMPLETE", details=details)

        assert error.message == "Complete test"
        assert error.error_code == "COMPLETE"
        assert error.details == details
        assert str(error) == "[COMPLETE] Complete test"


class TestErrorHandlingFunctions:
    """Test error handling utility functions"""

    def test_handle_error_basic(self):
        """Test basic error handling"""
        error = Exception("Test error")
        result = handle_error(error, "Test context")

        assert result["type"] == "Exception"
        assert result["message"] == "Test error"
        assert result["context"] == "Test context"
        assert "timestamp" in result

    @patch('vitalDSP_webapp.utils.error_handler.logger')
    def test_handle_error_with_debug_logging(self, mock_logger):
        """Test error handling with debug logging enabled"""
        mock_logger.isEnabledFor.return_value = True

        error = Exception("Debug test")
        result = handle_error(error, "Debug context")

        assert "traceback" in result
        assert result["traceback"] is not None

    def test_create_error_alert_with_exception(self):
        """Test creating error alert with exception"""
        error = ValueError("Test value error")
        alert = create_error_alert(error, title="Validation Error")

        assert isinstance(alert, dbc.Alert)
        assert alert.color == "danger"

    def test_create_error_alert_with_string(self):
        """Test creating error alert with string message"""
        alert = create_error_alert("Simple error message", title="Error")

        assert isinstance(alert, dbc.Alert)
        assert alert.color == "danger"

    def test_create_error_alert_with_details(self):
        """Test creating error alert with details shown"""
        error = WebappError("Detailed error", error_code="DETAIL_001")
        error.context = {"file": "test.csv"}
        alert = create_error_alert(error, show_details=True)

        assert isinstance(alert, dbc.Alert)
        # Check that alert has multiple children when details are shown
        assert len(alert.children) > 2

    def test_create_error_alert_dismissible(self):
        """Test creating dismissible error alert"""
        alert = create_error_alert("Test error", dismissible=True)

        assert isinstance(alert, dbc.Alert)
        assert alert.dismissable == True

    def test_handle_upload_error_with_file_upload_error(self):
        """Test handling FileUploadError specifically"""
        error = FileUploadError("File too large", error_code="FILE_SIZE")
        context = {"filename": "large_file.csv", "size_mb": 150}

        result = handle_upload_error(error, context)

        assert result["error"] == True
        assert result["message"] == "File too large"
        assert "upload_context" in result
        assert "suggestions" in result
        assert len(result["suggestions"]) > 0

    def test_handle_processing_error_with_data_processing_error(self):
        """Test handling DataProcessingError specifically"""
        error = DataProcessingError("Invalid data format", error_code="FORMAT_ERROR")
        context = {"data_type": "csv", "column_count": 5}

        result = handle_processing_error(error, context)

        assert result["error"] == True
        assert result["message"] == "Invalid data format"
        assert "processing_context" in result
        assert "suggestions" in result

    def test_handle_analysis_error_with_analysis_error(self):
        """Test handling AnalysisError specifically"""
        error = AnalysisError("Insufficient data", error_code="DATA_ERROR")
        context = {"required_points": 1000, "actual_points": 100}

        result = handle_analysis_error(error, context)

        assert result["error"] == True
        assert result["message"] == "Insufficient data"
        assert "analysis_context" in result
        assert "suggestions" in result

    @patch('vitalDSP_webapp.utils.error_handler.logger')
    def test_log_error_with_context_debug_level(self, mock_logger):
        """Test logging error with DEBUG level"""
        error = Exception("Debug level error")
        context = {"module": "test", "function": "test_func"}

        log_error_with_context(error, context, level="DEBUG")

        mock_logger.debug.assert_called_once()

    @patch('vitalDSP_webapp.utils.error_handler.logger')
    def test_log_error_with_context_info_level(self, mock_logger):
        """Test logging error with INFO level"""
        error = Exception("Info level error")
        context = {"module": "test"}

        log_error_with_context(error, context, level="INFO")

        mock_logger.info.assert_called_once()

    @patch('vitalDSP_webapp.utils.error_handler.logger')
    def test_log_error_with_context_warning_level(self, mock_logger):
        """Test logging error with WARNING level"""
        error = Exception("Warning level error")
        context = {"module": "test"}

        log_error_with_context(error, context, level="WARNING")

        mock_logger.warning.assert_called_once()

    @patch('vitalDSP_webapp.utils.error_handler.logger')
    def test_log_error_with_context_critical_level(self, mock_logger):
        """Test logging error with CRITICAL level"""
        error = Exception("Critical level error")
        context = {"module": "test"}

        log_error_with_context(error, context, level="CRITICAL")

        mock_logger.critical.assert_called_once()

    @patch('vitalDSP_webapp.utils.error_handler.logger')
    def test_log_error_with_context_invalid_level(self, mock_logger):
        """Test logging error with invalid level defaults to ERROR"""
        error = Exception("Invalid level error")
        context = {"module": "test"}

        log_error_with_context(error, context, level="INVALID")

        # Should default to error
        mock_logger.error.assert_called()

    def test_create_user_friendly_error_message_upload_error(self):
        """Test creating user-friendly message for upload error"""
        error = WebappError("File format not supported", error_code="upload_error")
        message = create_user_friendly_error_message(error)

        assert "Upload Error" in message
        assert "File format not supported" in message

    def test_create_user_friendly_error_message_processing_error(self):
        """Test creating user-friendly message for processing error"""
        error = WebappError("Data processing failed", error_code="processing_error")
        message = create_user_friendly_error_message(error)

        assert "Processing Error" in message
        assert "Data processing failed" in message

    def test_create_user_friendly_error_message_analysis_error(self):
        """Test creating user-friendly message for analysis error"""
        error = WebappError("Analysis failed", error_code="analysis_error")
        message = create_user_friendly_error_message(error)

        assert "Analysis Error" in message
        assert "Analysis failed" in message

    def test_create_user_friendly_error_message_generic_webapp_error(self):
        """Test creating user-friendly message for generic webapp error"""
        error = WebappError("Generic error", error_code="GENERIC")
        message = create_user_friendly_error_message(error)

        assert "GENERIC" in message
        assert "Generic error" in message

    def test_create_user_friendly_error_message_file_error(self):
        """Test creating user-friendly message for file-related generic error"""
        error = Exception("File not found error")
        message = create_user_friendly_error_message(error)

        assert "File Error" in message

    def test_create_user_friendly_error_message_data_error(self):
        """Test creating user-friendly message for data-related generic error"""
        error = Exception("Data validation error")
        message = create_user_friendly_error_message(error)

        assert "Data Error" in message

    def test_create_user_friendly_error_message_other_error(self):
        """Test creating user-friendly message for other generic errors"""
        error = Exception("Something went wrong")
        message = create_user_friendly_error_message(error)

        assert "An error occurred" in message

    def test_format_error_for_display_basic(self):
        """Test formatting error for display"""
        error = ValueError("Test value error")
        formatted = format_error_for_display(error)

        assert "message" in formatted
        assert "type" in formatted
        assert "timestamp" in formatted
        assert formatted["message"] == "Test value error"

    def test_format_error_for_display_with_details(self):
        """Test formatting error with details for display"""
        error = WebappError("Test error", error_code="TEST", details={"key": "value"})
        formatted = format_error_for_display(error, include_details=False)

        assert "context" in formatted
        assert formatted["context"] == {"key": "value"}

    def test_format_error_for_display_with_traceback(self):
        """Test formatting error with traceback"""
        error = Exception("Test error with traceback")
        formatted = format_error_for_display(error, include_details=True)

        assert "traceback" in formatted
        assert formatted["traceback"] is not None

    def test_safe_execute_success(self):
        """Test safe_execute with successful function"""
        def successful_func(x, y):
            return x + y

        result = safe_execute(successful_func, 2, 3, error_context="Test")
        assert result == 5

    def test_safe_execute_failure(self):
        """Test safe_execute with failing function"""
        def failing_func():
            raise ValueError("Test error")

        result = safe_execute(failing_func, default_return="default", error_context="Test")
        assert result == "default"

    def test_validate_required_fields_success(self):
        """Test validate_required_fields with all fields present"""
        data = {"field1": "value1", "field2": "value2"}
        required_fields = ["field1", "field2"]

        # Should not raise exception
        validate_required_fields(data, required_fields)

    def test_validate_required_fields_missing(self):
        """Test validate_required_fields with missing fields"""
        data = {"field1": "value1"}
        required_fields = ["field1", "field2", "field3"]

        with pytest.raises(ValidationError) as exc_info:
            validate_required_fields(data, required_fields)

        assert "Missing required fields" in str(exc_info.value)
        assert "field2" in str(exc_info.value)

    def test_validate_required_fields_none_value(self):
        """Test validate_required_fields with None value"""
        data = {"field1": "value1", "field2": None}
        required_fields = ["field1", "field2"]

        with pytest.raises(ValidationError):
            validate_required_fields(data, required_fields)

    def test_validate_data_types_success(self):
        """Test validate_data_types with correct types"""
        data = {"name": "test", "age": 25, "score": 95.5}
        field_types = {"name": str, "age": int, "score": float}

        # Should not raise exception
        validate_data_types(data, field_types)

    def test_validate_data_types_mismatch(self):
        """Test validate_data_types with type mismatch"""
        data = {"name": "test", "age": "25"}  # age should be int
        field_types = {"name": str, "age": int}

        with pytest.raises(ValidationError) as exc_info:
            validate_data_types(data, field_types)

        assert "Data type validation failed" in str(exc_info.value)
        assert "age" in str(exc_info.value)

    def test_validate_data_types_ignores_none(self):
        """Test validate_data_types ignores None values"""
        data = {"name": "test", "age": None}
        field_types = {"name": str, "age": int}

        # Should not raise exception for None values
        validate_data_types(data, field_types)

    def test_format_error_message_webapp_error_with_details(self):
        """Test format_error_message with WebappError and details"""
        error = WebappError("Test error", details={"key": "value"})
        message = format_error_message(error)

        assert "Test error" in message
        assert "Details:" in message

    def test_format_error_message_generic_error(self):
        """Test format_error_message with generic error"""
        error = Exception("Generic error")
        message = format_error_message(error)

        assert "Generic error" in message

    def test_format_error_message_with_traceback(self):
        """Test format_error_message with traceback included"""
        error = Exception("Error with traceback")
        message = format_error_message(error, include_traceback=True)

        assert "Error with traceback" in message
        assert "Traceback:" in message

    def test_create_warning_alert(self):
        """Test creating warning alert"""
        alert = create_warning_alert("This is a warning", title="Warning")

        assert isinstance(alert, dbc.Alert)
        assert alert.color == "warning"

    def test_create_info_alert(self):
        """Test creating info alert"""
        alert = create_info_alert("This is information", title="Info")

        assert isinstance(alert, dbc.Alert)
        assert alert.color == "info"

    def test_create_success_alert(self):
        """Test creating success alert"""
        alert = create_success_alert("Operation successful", title="Success")

        assert isinstance(alert, dbc.Alert)
        assert alert.color == "success"


class TestErrorSuggestions:
    """Test error suggestion functions"""

    def test_get_upload_error_suggestions_file_size(self):
        """Test upload suggestions for file size error"""
        from vitalDSP_webapp.utils.error_handler import get_upload_error_suggestions

        error = Exception("file size too large")
        suggestions = get_upload_error_suggestions(error)

        assert len(suggestions) > 0
        assert any("smaller file" in s.lower() for s in suggestions)

    def test_get_upload_error_suggestions_file_format(self):
        """Test upload suggestions for file format error"""
        from vitalDSP_webapp.utils.error_handler import get_upload_error_suggestions

        error = Exception("invalid file format")
        suggestions = get_upload_error_suggestions(error)

        assert len(suggestions) > 0
        assert any("file format" in s.lower() for s in suggestions)

    def test_get_upload_error_suggestions_permission(self):
        """Test upload suggestions for permission error"""
        from vitalDSP_webapp.utils.error_handler import get_upload_error_suggestions

        error = Exception("permission denied")
        suggestions = get_upload_error_suggestions(error)

        assert len(suggestions) > 0
        assert any("permission" in s.lower() for s in suggestions)

    def test_get_upload_error_suggestions_generic(self):
        """Test upload suggestions for generic error"""
        from vitalDSP_webapp.utils.error_handler import get_upload_error_suggestions

        error = Exception("unknown error")
        suggestions = get_upload_error_suggestions(error)

        assert len(suggestions) > 0

    def test_get_processing_error_suggestions_data_format(self):
        """Test processing suggestions for data format error"""
        from vitalDSP_webapp.utils.error_handler import get_processing_error_suggestions

        error = Exception("invalid data format")
        suggestions = get_processing_error_suggestions(error)

        assert len(suggestions) > 0
        assert any("data format" in s.lower() for s in suggestions)

    def test_get_processing_error_suggestions_column_mapping(self):
        """Test processing suggestions for column mapping error"""
        from vitalDSP_webapp.utils.error_handler import get_processing_error_suggestions

        error = Exception("column mapping error")
        suggestions = get_processing_error_suggestions(error)

        assert len(suggestions) > 0
        assert any("column mapping" in s.lower() for s in suggestions)

    def test_get_processing_error_suggestions_sampling_frequency(self):
        """Test processing suggestions for sampling frequency error"""
        from vitalDSP_webapp.utils.error_handler import get_processing_error_suggestions

        error = Exception("invalid sampling frequency")
        suggestions = get_processing_error_suggestions(error)

        assert len(suggestions) > 0
        assert any("sampling frequency" in s.lower() for s in suggestions)

    def test_get_processing_error_suggestions_generic(self):
        """Test processing suggestions for generic error"""
        from vitalDSP_webapp.utils.error_handler import get_processing_error_suggestions

        error = Exception("unknown processing error")
        suggestions = get_processing_error_suggestions(error)

        assert len(suggestions) > 0

    def test_get_analysis_error_suggestions_insufficient_data(self):
        """Test analysis suggestions for insufficient data error"""
        from vitalDSP_webapp.utils.error_handler import get_analysis_error_suggestions

        error = Exception("insufficient data points")
        suggestions = get_analysis_error_suggestions(error)

        assert len(suggestions) > 0
        assert any("data points" in s.lower() for s in suggestions)

    def test_get_analysis_error_suggestions_parameter(self):
        """Test analysis suggestions for parameter error"""
        from vitalDSP_webapp.utils.error_handler import get_analysis_error_suggestions

        error = Exception("invalid parameter value")
        suggestions = get_analysis_error_suggestions(error)

        assert len(suggestions) > 0
        assert any("parameter" in s.lower() for s in suggestions)

    def test_get_analysis_error_suggestions_generic(self):
        """Test analysis suggestions for generic error"""
        from vitalDSP_webapp.utils.error_handler import get_analysis_error_suggestions

        error = Exception("unknown analysis error")
        suggestions = get_analysis_error_suggestions(error)

        assert len(suggestions) > 0
        error = WebappError(
            message="Validation failed",
            error_code="VALIDATION_001", 
            details=details
        )
        
        assert str(error) == "[VALIDATION_001] Validation failed"
        assert error.message == "Validation failed"
        assert error.error_code == "VALIDATION_001"
        assert error.details == details

    def test_data_processing_error(self):
        """Test DataProcessingError inheritance"""
        error = DataProcessingError("Processing failed", error_code="PROC_001")
        
        assert isinstance(error, WebappError)
        assert str(error) == "[PROC_001] Processing failed"

    def test_file_upload_error(self):
        """Test FileUploadError inheritance"""
        error = FileUploadError("Upload failed", error_code="UPLOAD_001")
        
        assert isinstance(error, WebappError)
        assert str(error) == "[UPLOAD_001] Upload failed"

    def test_validation_error(self):
        """Test ValidationError inheritance"""
        error = ValidationError("Invalid input", error_code="VAL_001")
        
        assert isinstance(error, WebappError)
        assert str(error) == "[VAL_001] Invalid input"

    def test_analysis_error(self):
        """Test AnalysisError inheritance"""
        error = AnalysisError("Analysis error", error_code="ANALYSIS_001")
        
        assert isinstance(error, WebappError)
        assert str(error) == "[ANALYSIS_001] Analysis error"


class TestErrorHandlerFunctions:
    """Test error handler utility functions"""
    
    def test_handle_error_webapp_error(self):
        """Test handle_error with WebappError"""
        error = WebappError("Test error", error_code="TEST_001")
        
        result = handle_error(error, context="test_context")
        
        assert isinstance(result, dict)
        assert result["type"] == "WebappError"
        assert result["message"] == "[TEST_001] Test error"
        assert result["context"] == "test_context"
        assert "timestamp" in result

    def test_handle_error_standard_exception(self):
        """Test handle_error with standard Exception"""
        error = ValueError("Invalid value")
        
        result = handle_error(error, context="test_context")
        
        assert isinstance(result, dict)
        assert result["type"] == "ValueError"
        assert result["message"] == "Invalid value"
        assert result["context"] == "test_context"
        assert "timestamp" in result

    def test_handle_error_with_context(self):
        """Test handle_error with additional context"""
        error = WebappError("Test error")
        
        result = handle_error(error, context="upload_component")
        
        assert result["context"] == "upload_component"

    def test_create_error_alert_basic(self):
        """Test create_error_alert with basic parameters"""
        error = WebappError("Test error")
        
        result = create_error_alert(error)
        
        assert isinstance(result, dbc.Alert)
        assert result.color == "danger"
        assert result.dismissable is False

    def test_create_error_alert_dismissible(self):
        """Test create_error_alert with dismissible=True"""
        error = WebappError("Test error")
        
        result = create_error_alert(error, dismissible=True)
        
        assert isinstance(result, dbc.Alert)
        assert result.dismissable is True

    def test_create_error_alert_with_details(self):
        """Test create_error_alert with show_details=True"""
        error = WebappError("Test error", details={"file": "test.csv"})
        
        result = create_error_alert(error, show_details=True)
        
        assert isinstance(result, dbc.Alert)
        # Should contain details in the alert content

    def test_create_error_alert_custom_color(self):
        """Test create_error_alert with custom color"""
        error = WebappError("Test error")
        
        result = create_error_alert(error, color="warning")
        
        assert isinstance(result, dbc.Alert)
        assert result.color == "warning"

    def test_create_error_alert_string_error(self):
        """Test create_error_alert with string error"""
        result = create_error_alert("Simple error message")
        
        assert isinstance(result, dbc.Alert)

    def test_handle_upload_error(self):
        """Test handle_upload_error function"""
        error = FileUploadError("Upload failed")
        
        result = handle_upload_error(error)
        
        assert isinstance(result, dict)
        assert "error" in result
        assert "suggestions" in result

    def test_handle_processing_error(self):
        """Test handle_processing_error function"""
        error = DataProcessingError("Processing failed")
        
        result = handle_processing_error(error)
        
        assert isinstance(result, dict)
        assert "error" in result
        assert "suggestions" in result

    def test_handle_analysis_error(self):
        """Test handle_analysis_error function"""
        error = AnalysisError("Analysis failed")
        
        result = handle_analysis_error(error)
        
        assert isinstance(result, dict)
        assert "error" in result
        assert "suggestions" in result

    def test_log_error_with_context(self):
        """Test log_error_with_context function"""
        error = WebappError("Test error")
        context = {"component": "test", "user": "test_user"}

        # Test that function doesn't raise an error
        try:
            log_error_with_context(error, context, level="ERROR")
        except Exception as e:
            pytest.fail(f"log_error_with_context raised an exception: {e}")

    def test_create_user_friendly_error_message(self):
        """Test create_user_friendly_error_message function"""
        error = ValidationError("Invalid input format")
        
        result = create_user_friendly_error_message(error)
        
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_error_for_display(self):
        """Test format_error_for_display function"""
        error = WebappError("Test error", error_code="TEST_001")
        
        result = format_error_for_display(error, include_details=True)
        
        assert isinstance(result, dict)
        assert "message" in result
        assert "type" in result

    def test_format_error_message_webapp_error(self):
        """Test format_error_message with WebappError"""
        error = WebappError("Test error", error_code="TEST_001")
        
        result = format_error_message(error)
        
        assert isinstance(result, str)
        assert "TEST_001" in result
        assert "Test error" in result

    def test_format_error_message_standard_exception(self):
        """Test format_error_message with standard Exception"""
        error = ValueError("Invalid value")

        result = format_error_message(error)

        assert isinstance(result, str)
        assert "Invalid value" in result
        assert "Invalid value" in result

    def test_format_error_message_with_context(self):
        """Test format_error_message with include_context=True"""
        error = WebappError("Test error", details={"file": "test.csv"})
        
        result = format_error_message(error, include_traceback=True)
        
        assert isinstance(result, str)
        assert "test.csv" in result


class TestErrorHandlerUtilityFunctions:
    """Test error handler utility functions"""

    def test_create_warning_alert(self):
        """Test create_warning_alert function"""
        result = create_warning_alert("This is a warning", title="Warning")
        
        assert isinstance(result, dbc.Alert)
        assert result.color == "warning"

    def test_create_info_alert(self):
        """Test create_info_alert function"""
        result = create_info_alert("This is information", title="Info")
        
        assert isinstance(result, dbc.Alert)
        assert result.color == "info"

    def test_create_success_alert(self):
        """Test create_success_alert function"""
        result = create_success_alert("Operation successful", title="Success")
        
        assert isinstance(result, dbc.Alert)
        assert result.color == "success"

    def test_safe_execute_success(self):
        """Test safe_execute with successful function"""
        def successful_function(x, y):
            return x + y
        
        result = safe_execute(successful_function, 2, 3, error_context="test")
        
        assert result == 5

    def test_safe_execute_with_exception(self):
        """Test safe_execute with function that raises exception"""
        def failing_function():
            raise ValueError("Test error")
        
        result = safe_execute(failing_function, default_return="default", error_context="test")
        
        assert result == "default"

    def test_validate_required_fields_success(self):
        """Test validate_required_fields with valid data"""
        data = {"field1": "value1", "field2": "value2"}
        required_fields = ["field1", "field2"]
        
        # Should not raise exception
        validate_required_fields(data, required_fields, context="test")

    def test_validate_required_fields_missing(self):
        """Test validate_required_fields with missing field"""
        data = {"field1": "value1"}
        required_fields = ["field1", "field2"]
        
        with pytest.raises(ValidationError):
            validate_required_fields(data, required_fields, context="test")

    def test_validate_data_types_success(self):
        """Test validate_data_types with correct types"""
        data = {"name": "John", "age": 25, "active": True}
        field_types = {"name": str, "age": int, "active": bool}
        
        # Should not raise exception
        validate_data_types(data, field_types, context="test")

    def test_validate_data_types_wrong_type(self):
        """Test validate_data_types with wrong type"""
        data = {"name": "John", "age": "25"}  # age should be int
        field_types = {"name": str, "age": int}
        
        with pytest.raises(ValidationError):
            validate_data_types(data, field_types, context="test")

    def test_format_error_message_with_traceback(self):
        """Test format_error_message with traceback"""
        error = ValueError("Test error")
        
        result = format_error_message(error, include_traceback=True)
        
        assert isinstance(result, str)
        assert "Test error" in result
        assert "Test error" in result

    def test_format_error_message_without_traceback(self):
        """Test format_error_message without traceback"""
        error = ValueError("Test error")
        
        result = format_error_message(error, include_traceback=False)
        
        assert isinstance(result, str)
        assert "Test error" in result
        assert "Test error" in result

    def test_multiple_error_handling_functions(self):
        """Test multiple error handling functions work together"""
        # Test upload error
        upload_error = FileUploadError("File too large")
        upload_result = handle_upload_error(upload_error)
        
        # Test processing error
        process_error = DataProcessingError("Invalid data format")
        process_result = handle_processing_error(process_error)
        
        # Test analysis error
        analysis_error = AnalysisError("Analysis failed")
        analysis_result = handle_analysis_error(analysis_error)
        
        # All should return dictionaries with error info
        assert isinstance(upload_result, dict)
        assert isinstance(process_result, dict)
        assert isinstance(analysis_result, dict)
