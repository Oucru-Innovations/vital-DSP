"""
Edge cases and remaining uncovered lines test cases for physiological_callbacks.py
Focuses on the remaining uncovered branches and complex edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from dash import callback_context
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go


@pytest.fixture
def sample_noisy_data():
    """Create sample noisy data"""
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.random.randn(1000)
    return pd.DataFrame({'time': t, 'signal': signal})


@pytest.fixture
def sample_short_data():
    """Create sample short data"""
    np.random.seed(42)
    t = np.linspace(0, 1, 100)  # Only 1 second of data
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(100)
    return pd.DataFrame({'time': t, 'signal': signal})


@pytest.fixture
def sample_empty_data():
    """Create sample empty data"""
    return pd.DataFrame({'time': [], 'signal': []})


@pytest.fixture
def mock_data_service():
    """Create a mock data service"""
    service = Mock()
    service.get_all_data.return_value = {}
    service.get_data_info.return_value = {}
    service.current_data = None
    return service


class TestPhysiologicalCallbacksEdgeCases:
    """Test edge cases and error conditions"""

    def test_analyze_hrv_callback_empty_dataframe(self, mock_data_service, sample_empty_data):
        """Test with empty dataframe (line 1515)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with empty dataframe
            mock_data_service.current_data = sample_empty_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                result = hrv_callback(
                    n_clicks=1,
                    selected_metrics=["rmssd", "sdnn"],
                    time_domain_options=["rmssd"],
                    freq_domain_options=["lf", "hf"],
                    nonlinear_options=["poincare"],
                    preprocessing_options=["detrend"],
                    advanced_options={}
                )
                # Should handle empty dataframe gracefully
                assert isinstance(result, list)
                assert len(result) > 0

    def test_analyze_hrv_callback_short_data(self, mock_data_service, sample_short_data):
        """Test with short data (line 1530-1532)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with short data
            mock_data_service.current_data = sample_short_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 15.2,  # Lower values due to short data
                        "sdnn": 25.8,
                        "warning": "Short data segment"
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options={}
                    )
                    # Should handle short data
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_noisy_data(self, mock_data_service, sample_noisy_data):
        """Test with noisy data (line 1537-1542)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with noisy data
            mock_data_service.current_data = sample_noisy_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 35.2,  # Higher due to noise
                        "sdnn": 48.5,
                        "noise_level": 0.5,
                        "quality_warning": "High noise detected"
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend", "filter"],
                        advanced_options={}
                    )
                    # Should handle noisy data
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_missing_columns(self, mock_data_service):
        """Test with missing columns (line 1563-1565)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with missing columns
            mock_data_service.current_data = pd.DataFrame({'time': [1, 2, 3]})  # Missing signal column
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                result = hrv_callback(
                    n_clicks=1,
                    selected_metrics=["rmssd", "sdnn"],
                    time_domain_options=["rmssd"],
                    freq_domain_options=["lf", "hf"],
                    nonlinear_options=["poincare"],
                    preprocessing_options=["detrend"],
                    advanced_options={}
                )
                # Should handle missing columns gracefully
                assert isinstance(result, list)
                assert len(result) > 0

    def test_analyze_hrv_callback_invalid_sampling_freq(self, mock_data_service, sample_noisy_data):
        """Test with invalid sampling frequency (line 1604)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with invalid sampling frequency
            mock_data_service.current_data = sample_noisy_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 0,  # Invalid sampling frequency
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                result = hrv_callback(
                    n_clicks=1,
                    selected_metrics=["rmssd", "sdnn"],
                    time_domain_options=["rmssd"],
                    freq_domain_options=["lf", "hf"],
                    nonlinear_options=["poincare"],
                    preprocessing_options=["detrend"],
                    advanced_options={}
                )
                # Should handle invalid sampling frequency
                assert isinstance(result, list)
                assert len(result) > 0

    def test_analyze_hrv_callback_negative_values(self, mock_data_service, sample_noisy_data):
        """Test with negative values (line 1621-1623)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with negative values
            mock_data_service.current_data = sample_noisy_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": -15.2,  # Negative value
                        "sdnn": 25.8,
                        "warning": "Negative values detected"
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options={}
                    )
                    # Should handle negative values
                    assert isinstance(result, list)
                    assert len(result) > 0


class TestPhysiologicalCallbacksAdvancedEdgeCases:
    """Test advanced edge cases and complex scenarios"""

    def test_analyze_hrv_callback_extreme_values(self, mock_data_service, sample_noisy_data):
        """Test with extreme values (line 1775-1776)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with extreme values
            mock_data_service.current_data = sample_noisy_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 1e6,  # Extreme value
                        "sdnn": 1e-6,  # Extreme value
                        "warning": "Extreme values detected"
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options={}
                    )
                    # Should handle extreme values
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_nan_values(self, mock_data_service, sample_noisy_data):
        """Test with NaN values (line 1792->1827)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with NaN values
            mock_data_service.current_data = sample_noisy_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": np.nan,  # NaN value
                        "sdnn": 25.8,
                        "warning": "NaN values detected"
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options={}
                    )
                    # Should handle NaN values
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_inf_values(self, mock_data_service, sample_noisy_data):
        """Test with infinite values (line 1823-1824)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test with infinite values
            mock_data_service.current_data = sample_noisy_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": np.inf,  # Infinite value
                        "sdnn": 25.8,
                        "warning": "Infinite values detected"
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options={}
                    )
                    # Should handle infinite values
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_memory_error(self, mock_data_service, sample_noisy_data):
        """Test memory error handling (line 1889->1895)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test memory error
            mock_data_service.current_data = sample_noisy_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv', side_effect=MemoryError("Out of memory")):
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options={}
                    )
                    # Should handle memory error
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_timeout_error(self, mock_data_service, sample_noisy_data):
        """Test timeout error handling (line 1895->1901)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test timeout error
            mock_data_service.current_data = sample_noisy_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv', side_effect=TimeoutError("Operation timed out")):
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options={}
                    )
                    # Should handle timeout error
                    assert isinstance(result, list)
                    assert len(result) > 0


class TestPhysiologicalCallbacksComplexErrorHandling:
    """Test complex error handling scenarios"""

    def test_analyze_hrv_callback_multiple_errors(self, mock_data_service, sample_noisy_data):
        """Test multiple error conditions (line 1902, 1908, 1912, 1916)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test multiple error conditions
            mock_data_service.current_data = sample_noisy_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "errors": [
                            "Warning: Low signal quality",
                            "Warning: Insufficient data length",
                            "Warning: High noise level"
                        ]
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options={}
                    )
                    # Should handle multiple errors
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_recovery_mechanisms(self, mock_data_service, sample_noisy_data):
        """Test recovery mechanisms (line 1922, 1926, 1940)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test recovery mechanisms
            mock_data_service.current_data = sample_noisy_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "recovery_info": {
                            "fallback_method_used": True,
                            "recovery_successful": True,
                            "original_error": "Analysis failed"
                        }
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options={}
                    )
                    # Should handle recovery mechanisms
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_fallback_methods(self, mock_data_service, sample_noisy_data):
        """Test fallback methods (line 1952, 1958, 1964, 1970)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test fallback methods
            mock_data_service.current_data = sample_noisy_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "fallback_methods": {
                            "primary_method": "failed",
                            "fallback_method": "basic_analysis",
                            "fallback_successful": True
                        }
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options={}
                    )
                    # Should handle fallback methods
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_validation_failure(self, mock_data_service, sample_noisy_data):
        """Test validation failure (line 1977-1979)"""
        from vitalDSP_webapp.callbacks.features.physiological_callbacks import register_physiological_callbacks
        
        mock_app = Mock()
        captured_callbacks = []
        
        def mock_callback(*args, **kwargs):
            def decorator(func):
                captured_callbacks.append((args, kwargs, func))
                return func
            return decorator
        
        mock_app.callback = mock_callback
        register_physiological_callbacks(mock_app)
        
        # Find HRV analysis callback
        hrv_callback = None
        for args, kwargs, func in captured_callbacks:
            if func.__name__ == 'analyze_hrv':
                hrv_callback = func
                break
        
        if hrv_callback:
            # Test validation failure
            mock_data_service.current_data = sample_noisy_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "signal"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "validation_failed": True,
                        "validation_errors": [
                            "Data quality below threshold",
                            "Insufficient signal length"
                        ]
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options={}
                    )
                    # Should handle validation failure
                    assert isinstance(result, list)
                    assert len(result) > 0
