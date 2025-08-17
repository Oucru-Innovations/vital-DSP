"""
Tests for page routing callbacks.
"""

import pytest
from unittest.mock import Mock, patch
from dash import html

from vitalDSP_webapp.callbacks.page_routing_callbacks import (
    register_page_routing_callbacks,
    display_page,
    test_routing
)


class TestPageRoutingCallbacks:
    """Test class for page routing callbacks."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_app = Mock()
        self.mock_app.callback = Mock(return_value=Mock())
    
    def test_register_page_routing_callbacks(self):
        """Test that page routing callbacks are registered correctly."""
        register_page_routing_callbacks(self.mock_app)
        
        # Should register 2 callbacks
        assert self.mock_app.callback.call_count == 2
        
        # Check first callback (display_page)
        first_call = self.mock_app.callback.call_args_list[0]
        # The Output and Input are passed as Output and Input objects
        first_output = first_call[0][0]
        first_input = first_call[0][1]
        assert first_output.component_id == "page-content"
        assert first_output.component_property == "children"
        assert len(first_input) == 1
        assert first_input[0].component_id == "url"
        assert first_input[0].component_property == "pathname"
        
        # Check second callback (test_routing)
        second_call = self.mock_app.callback.call_args_list[1]
        second_output = second_call[0][0]
        second_input = second_call[0][1]
        assert second_output.component_id == "page-content"
        assert second_output.component_property == "style"
        assert len(second_input) == 1
        assert second_input[0].component_id == "url"
        assert second_input[0].component_property == "pathname"
    
    def test_display_page_home(self):
        """Test display_page function for home page."""
        result = display_page("/")
        
        assert isinstance(result, html.Div)
        assert "Welcome to vitalDSP Comprehensive Dashboard" in str(result)
        assert "Digital Signal Processing for Vital Signs" in str(result)
    
    def test_display_page_upload(self):
        """Test display_page function for upload page."""
        with patch('vitalDSP_webapp.callbacks.page_routing_callbacks.upload_layout') as mock_upload:
            mock_upload.return_value = html.Div("Upload Layout")
            result = display_page("/upload")
            
            assert isinstance(result, html.Div)
            assert "Upload Layout" in str(result)
            mock_upload.assert_called_once()
    
    def test_display_page_time_domain(self):
        """Test display_page function for time domain page."""
        with patch('vitalDSP_webapp.callbacks.page_routing_callbacks.time_domain_layout') as mock_layout:
            mock_layout.return_value = html.Div("Time Domain Layout")
            result = display_page("/time-domain")
            
            assert isinstance(result, html.Div)
            assert "Time Domain Layout" in str(result)
            mock_layout.assert_called_once()
    
    def test_display_page_frequency(self):
        """Test display_page function for frequency page."""
        with patch('vitalDSP_webapp.callbacks.page_routing_callbacks.frequency_layout') as mock_layout:
            mock_layout.return_value = html.Div("Frequency Layout")
            result = display_page("/frequency")
            
            assert isinstance(result, html.Div)
            assert "Frequency Layout" in str(result)
            mock_layout.assert_called_once()
    
    def test_display_page_filtering(self):
        """Test display_page function for filtering page."""
        with patch('vitalDSP_webapp.callbacks.page_routing_callbacks.filtering_layout') as mock_layout:
            mock_layout.return_value = html.Div("Filtering Layout")
            result = display_page("/filtering")
            
            assert isinstance(result, html.Div)
            assert "Filtering Layout" in str(result)
            mock_layout.assert_called_once()
    
    def test_display_page_physiological(self):
        """Test display_page function for physiological page."""
        with patch('vitalDSP_webapp.callbacks.page_routing_callbacks.physiological_layout') as mock_layout:
            mock_layout.return_value = html.Div("Physiological Layout")
            result = display_page("/physiological")
            
            assert isinstance(result, html.Div)
            assert "Physiological Layout" in str(result)
            mock_layout.assert_called_once()
    
    def test_display_page_respiratory(self):
        """Test display_page function for respiratory page."""
        with patch('vitalDSP_webapp.callbacks.page_routing_callbacks.respiratory_layout') as mock_layout:
            mock_layout.return_value = html.Div("Respiratory Layout")
            result = display_page("/respiratory")
            
            assert isinstance(result, html.Div)
            assert "Respiratory Layout" in str(result)
            mock_layout.assert_called_once()
    
    def test_display_page_features(self):
        """Test display_page function for features page."""
        with patch('vitalDSP_webapp.callbacks.page_routing_callbacks.features_layout') as mock_layout:
            mock_layout.return_value = html.Div("Features Layout")
            result = display_page("/features")
            
            assert isinstance(result, html.Div)
            assert "Features Layout" in str(result)
            mock_layout.assert_called_once()
    
    def test_display_page_transforms(self):
        """Test display_page function for transforms page."""
        with patch('vitalDSP_webapp.callbacks.page_routing_callbacks.transforms_layout') as mock_layout:
            mock_layout.return_value = html.Div("Transforms Layout")
            result = display_page("/transforms")
            
            assert isinstance(result, html.Div)
            assert "Transforms Layout" in str(result)
            mock_layout.assert_called_once()
    
    def test_display_page_quality(self):
        """Test display_page function for quality page."""
        with patch('vitalDSP_webapp.callbacks.page_routing_callbacks.quality_layout') as mock_layout:
            mock_layout.return_value = html.Div("Quality Layout")
            result = display_page("/quality")
            
            assert isinstance(result, html.Div)
            assert "Quality Layout" in str(result)
            mock_layout.assert_called_once()
    
    def test_display_page_advanced(self):
        """Test display_page function for advanced page."""
        with patch('vitalDSP_webapp.callbacks.page_routing_callbacks.advanced_layout') as mock_layout:
            mock_layout.return_value = html.Div("Advanced Layout")
            result = display_page("/advanced")
            
            assert isinstance(result, html.Div)
            assert "Advanced Layout" in str(result)
            mock_layout.assert_called_once()
    
    def test_display_page_health_report(self):
        """Test display_page function for health report page."""
        with patch('vitalDSP_webapp.callbacks.page_routing_callbacks.health_report_layout') as mock_layout:
            mock_layout.return_value = html.Div("Health Report Layout")
            result = display_page("/health-report")
            
            assert isinstance(result, html.Div)
            assert "Health Report Layout" in str(result)
            mock_layout.assert_called_once()
    
    def test_display_page_settings(self):
        """Test display_page function for settings page."""
        with patch('vitalDSP_webapp.callbacks.page_routing_callbacks.settings_layout') as mock_layout:
            mock_layout.return_value = html.Div("Settings Layout")
            result = display_page("/settings")
            
            assert isinstance(result, html.Div)
            assert "Settings Layout" in str(result)
            mock_layout.assert_called_once()
    
    def test_display_page_unknown(self):
        """Test display_page function for unknown page."""
        result = display_page("/unknown-page")
        
        assert isinstance(result, html.Div)
        assert "Welcome to vitalDSP Comprehensive Dashboard" in str(result)
    
    def test_display_page_none(self):
        """Test display_page function for None pathname."""
        result = display_page(None)
        
        assert isinstance(result, html.Div)
        assert "Welcome to vitalDSP Comprehensive Dashboard" in str(result)
    
    def test_display_page_exception_handling(self):
        """Test display_page function handles exceptions gracefully."""
        with patch('vitalDSP_webapp.callbacks.page_routing_callbacks.upload_layout') as mock_upload:
            mock_upload.side_effect = Exception("Layout error")
            result = display_page("/upload")
            
            assert isinstance(result, html.Div)
            assert "Error Loading Page" in str(result)
            assert "Layout error" in str(result)
    
    def test_test_routing(self):
        """Test test_routing function."""
        from dash import no_update
        result = test_routing("/test")
        
        assert result == no_update
