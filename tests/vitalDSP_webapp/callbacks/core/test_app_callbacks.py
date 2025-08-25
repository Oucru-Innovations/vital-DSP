"""
Tests for vitalDSP_webapp core app callbacks
"""
import sys
import os
import pytest
from unittest.mock import Mock, patch
from dash import Dash
from dash.dependencies import Input, Output

# Ensure we can import the module
try:
    from vitalDSP_webapp.callbacks.core.app_callbacks import register_sidebar_callbacks
except ImportError:
    # Fallback: add src to path if import fails
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..', '..')
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from vitalDSP_webapp.callbacks.core.app_callbacks import register_sidebar_callbacks


class TestAppCallbacks:
    """Test class for app callbacks"""
    
    def setup_method(self):
        """Setup method run before each test"""
        self.mock_app = Mock()
        self.mock_app.callback = Mock(return_value=Mock())

    def test_register_sidebar_callbacks(self):
        """Test that register_sidebar_callbacks registers the expected callbacks"""
        register_sidebar_callbacks(self.mock_app)
        
        # Should register exactly 2 callbacks
        assert self.mock_app.callback.call_count == 2
        
        # Check first callback (toggle_sidebar)
        first_call = self.mock_app.callback.call_args_list[0]
        first_outputs = first_call[0][0]  # First positional argument (outputs)
        first_inputs = first_call[0][1]   # Second positional argument (inputs)
        
        assert len(first_outputs) == 2
        assert first_outputs[0].component_id == "sidebar"
        assert first_outputs[0].component_property == "className"
        assert first_outputs[1].component_id == "sidebar-toggle-icon"
        assert first_outputs[1].component_property == "className"
        
        assert len(first_inputs) == 1
        assert first_inputs[0].component_id == "sidebar-toggle"
        assert first_inputs[0].component_property == "n_clicks"
        
        # Check second callback (adjust_page_content_position)
        second_call = self.mock_app.callback.call_args_list[1]
        second_output = second_call[0][0]  # First positional argument (output)
        second_inputs = second_call[0][1]  # Second positional argument (inputs)
        
        assert second_output.component_id == "page-content"
        assert second_output.component_property == "style"
        
        assert len(second_inputs) == 1
        assert second_inputs[0].component_id == "sidebar"
        assert second_inputs[0].component_property == "className"

    def test_toggle_sidebar_expanded(self, capsys=None):
        """Test toggle_sidebar function returns expanded state for even clicks"""
        mock_app = Mock()
        
        # Mock the callback decorator to capture only the first function
        captured_funcs = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_funcs.append(func)
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_sidebar_callbacks(mock_app)
        
        # Get the first function (toggle_sidebar)
        toggle_sidebar_func = captured_funcs[0]
        
        # Test with 0 clicks (even - expanded state)
        sidebar_class, icon_class = toggle_sidebar_func(0)
        assert sidebar_class == "sidebar sidebar-expanded"
        assert icon_class == "fas fa-bars"

    def test_toggle_sidebar_collapsed(self, capsys=None):
        """Test toggle_sidebar function returns collapsed state for odd clicks"""
        mock_app = Mock()
        
        # Mock the callback decorator to capture the function
        captured_funcs = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_funcs.append(func)
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_sidebar_callbacks(mock_app)
        
        # Get the first function (toggle_sidebar)
        toggle_sidebar_func = captured_funcs[0]
        
        # Test with 1 click (odd - collapsed state)
        sidebar_class, icon_class = toggle_sidebar_func(1)
        assert sidebar_class == "sidebar sidebar-collapsed"
        assert icon_class == "fas fa-arrow-right"

    def test_toggle_sidebar_none_clicks(self, capsys=None):
        """Test toggle_sidebar function handles None n_clicks"""
        mock_app = Mock()
        
        # Mock the callback decorator to capture the function
        captured_funcs = []
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_funcs.append(func)
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_sidebar_callbacks(mock_app)
        
        # Get the first function (toggle_sidebar)
        toggle_sidebar_func = captured_funcs[0]
        
        # Test with None clicks (should default to 0)
        sidebar_class, icon_class = toggle_sidebar_func(None)
        assert sidebar_class == "sidebar sidebar-expanded"
        assert icon_class == "fas fa-bars"

    def test_adjust_page_content_position_expanded(self):
        """Test adjust_page_content_position for expanded sidebar"""
        with patch('vitalDSP_webapp.config.settings.app_config') as mock_app_config:
            mock_app_config.SIDEBAR_WIDTH = 250
            mock_app_config.SIDEBAR_COLLAPSED_WIDTH = 60
            mock_app_config.HEADER_HEIGHT = 70
            
            mock_app = Mock()
            
            # Mock the callback decorator to capture the second function
            captured_funcs = []
            def mock_callback(*args, **kwargs):
                def decorator(func):
                    captured_funcs.append(func)
                    return func
                return decorator
            
            mock_app.callback = mock_callback
            register_sidebar_callbacks(mock_app)
            
            # Get the second registered function (adjust_page_content_position)
            adjust_page_content_func = captured_funcs[1]
            
            # Test with expanded sidebar class
            style = adjust_page_content_func("sidebar sidebar-expanded")
            
            assert style["left"] == "250px"
            assert style["top"] == "70px"
            assert style["position"] == "absolute"
            assert style["right"] == "0"
            assert style["padding"] == "2rem"
            assert style["backgroundColor"] == "#ffffff"
            assert style["minHeight"] == "calc(100vh - 70px)"
            assert style["zIndex"] == 100
            assert style["transition"] == "left 0.3s cubic-bezier(0.4, 0, 0.2, 1)"

    def test_adjust_page_content_position_collapsed(self):
        """Test adjust_page_content_position for collapsed sidebar"""
        with patch('vitalDSP_webapp.config.settings.app_config') as mock_app_config:
            mock_app_config.SIDEBAR_WIDTH = 250
            mock_app_config.SIDEBAR_COLLAPSED_WIDTH = 60
            mock_app_config.HEADER_HEIGHT = 70
            
            mock_app = Mock()
            
            # Mock the callback decorator to capture the second function
            captured_funcs = []
            def mock_callback(*args, **kwargs):
                def decorator(func):
                    captured_funcs.append(func)
                    return func
                return decorator
            
            mock_app.callback = mock_callback
            register_sidebar_callbacks(mock_app)
            
            # Get the second registered function (adjust_page_content_position)
            adjust_page_content_func = captured_funcs[1]
            
            # Test with collapsed sidebar class
            style = adjust_page_content_func("sidebar sidebar-collapsed")
            
            assert style["left"] == "60px"
            assert style["top"] == "70px"
            assert style["position"] == "absolute"
            assert style["right"] == "0"
            assert style["padding"] == "2rem"
            assert style["backgroundColor"] == "#ffffff"
            assert style["minHeight"] == "calc(100vh - 70px)"
            assert style["zIndex"] == 100
            assert style["transition"] == "left 0.3s cubic-bezier(0.4, 0, 0.2, 1)"
