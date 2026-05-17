"""
Tests for vitalDSP_webapp core page routing callbacks
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from dash import html

# Import the modules we need to test
try:
    from vitalDSP_webapp.callbacks.core.page_routing_callbacks import (
        display_page,
        register_page_routing_callbacks,
        _get_welcome_layout,
        _get_error_layout
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
    from vitalDSP_webapp.callbacks.core.page_routing_callbacks import (
        display_page,
        register_page_routing_callbacks,
        _get_welcome_layout,
        _get_error_layout
    )


class TestPageRoutingCallbacks:
    """Test class for page routing callbacks"""

    def setup_method(self):
        """Setup method run before each test"""
        self.mock_app = Mock()
        self.mock_app.callback = Mock(return_value=Mock())

    def test_register_page_routing_callbacks(self):
        """Test that register_page_routing_callbacks registers the expected callback"""
        register_page_routing_callbacks(self.mock_app)

        # Should register exactly 1 callback
        assert self.mock_app.callback.call_count == 1

        # Check callback registration
        call_args = self.mock_app.callback.call_args_list[0]
        output = call_args[0][0]  # First positional argument (output)
        inputs = call_args[0][1]  # Second positional argument (inputs)

        assert output.component_id == "page-content"
        assert output.component_property == "children"

        assert len(inputs) == 1
        assert inputs[0].component_id == "url"
        assert inputs[0].component_property == "pathname"

    @patch('vitalDSP_webapp.callbacks.core.page_routing_callbacks.upload_layout')
    def test_display_page_upload(self, mock_upload_layout):
        """Test display_page function for upload path"""
        mock_layout = html.Div("Upload Layout")
        mock_upload_layout.return_value = mock_layout

        result = display_page("/upload")

        mock_upload_layout.assert_called_once()
        assert result == mock_layout

    @patch('vitalDSP_webapp.callbacks.core.page_routing_callbacks.time_domain_layout')
    def test_display_page_time_domain(self, mock_time_domain_layout):
        """Test display_page function for time-domain path"""
        mock_layout = html.Div("Time Domain Layout")
        mock_time_domain_layout.return_value = mock_layout

        result = display_page("/time-domain")

        mock_time_domain_layout.assert_called_once()
        assert result == mock_layout

    @patch('vitalDSP_webapp.callbacks.core.page_routing_callbacks.frequency_layout')
    def test_display_page_frequency(self, mock_frequency_layout):
        """Test display_page function for frequency path"""
        mock_layout = html.Div("Frequency Layout")
        mock_frequency_layout.return_value = mock_layout

        result = display_page("/frequency")

        mock_frequency_layout.assert_called_once()
        assert result == mock_layout

    @patch('vitalDSP_webapp.callbacks.core.page_routing_callbacks.filtering_layout')
    def test_display_page_filtering(self, mock_filtering_layout):
        """Test display_page function for filtering path"""
        mock_layout = html.Div("Filtering Layout")
        mock_filtering_layout.return_value = mock_layout

        result = display_page("/filtering")

        mock_filtering_layout.assert_called_once()
        assert result == mock_layout

    def test_display_page_preview(self):
        """Test display_page function for preview path returns welcome layout"""
        result = display_page("/preview")

        assert isinstance(result, html.Div)

    def test_display_page_root_path(self):
        """Test display_page function for root path"""
        result = display_page("/")

        # Should return welcome layout
        assert isinstance(result, html.Div)
        # Check that it contains expected welcome content
        assert any("Welcome to vitalDSP" in str(child) for child in result.children if hasattr(child, 'children'))

    def test_display_page_none_path(self):
        """Test display_page function for None path"""
        result = display_page(None)

        # Should return welcome layout
        assert isinstance(result, html.Div)
        # Check that it contains expected welcome content
        assert any("Welcome to vitalDSP" in str(child) for child in result.children if hasattr(child, 'children'))

    def test_display_page_unknown_path(self):
        """Test display_page function for unknown path"""
        result = display_page("/unknown-path")

        # Should return welcome layout for unknown paths
        assert isinstance(result, html.Div)
        # Check that it contains expected welcome content
        assert any("Welcome to vitalDSP" in str(child) for child in result.children if hasattr(child, 'children'))

    @patch('vitalDSP_webapp.callbacks.core.page_routing_callbacks.upload_layout')
    def test_display_page_exception_handling(self, mock_upload_layout):
        """Test display_page function handles exceptions properly"""
        # Make the layout function raise an exception
        mock_upload_layout.side_effect = Exception("Test error")

        result = display_page("/upload")

        # Should return error layout
        assert isinstance(result, html.Div)
        # Check that it contains error content
        assert any("Error Loading Page" in str(child) for child in result.children if hasattr(child, 'children'))

    def test_get_welcome_layout(self):
        """Test _get_welcome_layout function"""
        result = _get_welcome_layout()

        assert isinstance(result, html.Div)
        # Check for key elements
        assert any("Welcome to vitalDSP" in str(child) for child in result.children if hasattr(child, 'children'))

    def test_get_error_layout(self):
        """Test _get_error_layout function"""
        error_message = "Test error message"
        result = _get_error_layout(error_message)

        assert isinstance(result, html.Div)
        # Check that error message is included
        assert error_message in str(result)

    def test_display_page_callback_integration(self):
        """Test the callback integration with mock app"""
        mock_app = Mock()

        # Mock the callback decorator to capture the function
        captured_func = None
        def mock_callback(*args, **kwargs):
            def decorator(func):
                nonlocal captured_func
                captured_func = func
                return func
            return decorator

        mock_app.callback = mock_callback
        register_page_routing_callbacks(mock_app)

        # Test the captured function
        assert captured_func is not None

        # Test with different paths
        result = captured_func("/upload")
        assert isinstance(result, html.Div)

        result = captured_func("/")
        assert isinstance(result, html.Div)

    @patch('vitalDSP_webapp.callbacks.core.page_routing_callbacks.logger')
    def test_display_page_logging(self, mock_logger):
        """Test that display_page logs appropriate messages"""
        display_page("/upload")

        # Check that logging was called
        assert mock_logger.info.call_count >= 2  # At least the trigger and the specific page

        # Check specific log messages
        mock_logger.info.assert_any_call("=== PAGE ROUTING CALLBACK TRIGGERED ===")
        mock_logger.info.assert_any_call("Page routing callback triggered with pathname: /upload")
        mock_logger.info.assert_any_call("Returning upload layout")

    @patch('vitalDSP_webapp.callbacks.core.page_routing_callbacks.logger')
    @patch('vitalDSP_webapp.callbacks.core.page_routing_callbacks.upload_layout')
    def test_display_page_error_logging(self, mock_upload_layout, mock_logger):
        """Test that display_page logs errors properly"""
        # Make the layout function raise an exception
        test_error = Exception("Test error")
        mock_upload_layout.side_effect = test_error

        result = display_page("/upload")

        # Check that error was logged
        mock_logger.error.assert_called_once_with("Error in page routing callback: Test error")

        # Should still return a valid layout (error layout)
        assert isinstance(result, html.Div)
