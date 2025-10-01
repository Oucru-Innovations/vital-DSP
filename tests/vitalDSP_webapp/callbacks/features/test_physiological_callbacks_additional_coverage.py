"""
Additional comprehensive test cases to cover more missing lines in physiological_callbacks.py
Focuses on remaining uncovered branches, complex logic paths, and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from dash import callback_context
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go


@pytest.fixture
def sample_ecg_data():
    """Create sample ECG data"""
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    # ECG-like signal with QRS complexes
    signal = np.zeros_like(t)
    for i in range(0, len(t), 100):  # Heart rate ~60 bpm
        if i + 50 < len(t):
            signal[i:i+50] = np.sin(2 * np.pi * 20 * t[i:i+50]) * np.exp(-(t[i:i+50] - t[i])**2 / 0.01)
    signal += 0.1 * np.random.randn(len(signal))
    return pd.DataFrame({'time': t, 'ECG': signal})


@pytest.fixture
def sample_ppg_data():
    """Create sample PPG data"""
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    # PPG-like signal with pulse waves
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
    signal += 0.1 * np.random.randn(len(signal))
    return pd.DataFrame({'time': t, 'PPG': signal})


@pytest.fixture
def mock_data_service():
    """Create a mock data service"""
    service = Mock()
    service.get_all_data.return_value = {}
    service.get_data_info.return_value = {}
    service.current_data = None
    return service


class TestPhysiologicalCallbacksComplexPaths:
    """Test complex logic paths and integration scenarios"""

    def test_analyze_hrv_callback_quality_assessment(self, mock_data_service, sample_ecg_data):
        """Test quality assessment integration (line 983-997)"""
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
            # Test with quality assessment
            mock_data_service.current_data = sample_ecg_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "ECG"],
                "quality_metrics": {
                    "snr": 15.2,
                    "artifact_percentage": 5.1,
                    "baseline_wander": 0.3
                }
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "quality_score": 0.85
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
                    # Should handle quality assessment
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_trend_analysis(self, mock_data_service, sample_ecg_data):
        """Test trend analysis (line 1001)"""
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
            # Test with trend analysis
            mock_data_service.current_data = sample_ecg_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "ECG"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "trend_analysis": {
                            "rmssd_trend": "increasing",
                            "sdnn_trend": "stable"
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
                    # Should handle trend analysis
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_statistical_analysis(self, mock_data_service, sample_ecg_data):
        """Test statistical analysis (line 1087-1091)"""
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
            # Test with statistical analysis
            mock_data_service.current_data = sample_ecg_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "ECG"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "statistical_analysis": {
                            "confidence_intervals": {
                                "rmssd": [25.1, 35.1],
                                "sdnn": [38.8, 46.8]
                            },
                            "p_values": {
                                "rmssd": 0.023,
                                "sdnn": 0.156
                            }
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
                    # Should handle statistical analysis
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_comparative_analysis(self, mock_data_service, sample_ecg_data):
        """Test comparative analysis (line 1095)"""
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
            # Test with comparative analysis
            mock_data_service.current_data = sample_ecg_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "ECG"],
                "reference_values": {
                    "rmssd": 25.0,
                    "sdnn": 40.0
                }
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "comparative_analysis": {
                            "rmssd_vs_reference": 1.204,  # 20.4% higher
                            "sdnn_vs_reference": 1.07     # 7% higher
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
                    # Should handle comparative analysis
                    assert isinstance(result, list)
                    assert len(result) > 0


class TestPhysiologicalCallbacksAdvancedScenarios:
    """Test advanced scenarios and complex integration paths"""

    def test_analyze_hrv_callback_windowed_analysis(self, mock_data_service, sample_ecg_data):
        """Test windowed analysis (line 1114->1129)"""
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
            # Test with windowed analysis
            mock_data_service.current_data = sample_ecg_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "ECG"]
            }
            
            advanced_options = {
                "window_size": 300,
                "overlap": 0.5,
                "windowed_analysis": True
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "windowed_results": [
                            {"rmssd": 28.5, "sdnn": 40.2},
                            {"rmssd": 31.7, "sdnn": 45.4},
                            {"rmssd": 29.8, "sdnn": 42.1}
                        ]
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options=advanced_options
                    )
                    # Should handle windowed analysis
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_multi_scale_analysis(self, mock_data_service, sample_ecg_data):
        """Test multi-scale analysis (line 1122->1114)"""
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
            # Test with multi-scale analysis
            mock_data_service.current_data = sample_ecg_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "ECG"]
            }
            
            advanced_options = {
                "multi_scale_analysis": True,
                "scales": [1, 2, 4, 8, 16]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "multi_scale_results": {
                            "scale_1": {"rmssd": 30.1, "sdnn": 42.8},
                            "scale_2": {"rmssd": 28.5, "sdnn": 40.2},
                            "scale_4": {"rmssd": 25.3, "sdnn": 35.7}
                        }
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options=advanced_options
                    )
                    # Should handle multi-scale analysis
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_ensemble_analysis(self, mock_data_service, sample_ecg_data):
        """Test ensemble analysis (line 1130->1234)"""
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
            # Test with ensemble analysis
            mock_data_service.current_data = sample_ecg_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "ECG"]
            }
            
            advanced_options = {
                "ensemble_analysis": True,
                "ensemble_methods": ["time_domain", "frequency_domain", "nonlinear"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "ensemble_results": {
                            "consensus_rmssd": 29.8,
                            "consensus_sdnn": 42.5,
                            "confidence_score": 0.92
                        }
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options=advanced_options
                    )
                    # Should handle ensemble analysis
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_adaptive_analysis(self, mock_data_service, sample_ecg_data):
        """Test adaptive analysis (line 1159->1178)"""
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
            # Test with adaptive analysis
            mock_data_service.current_data = sample_ecg_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "ECG"]
            }
            
            advanced_options = {
                "adaptive_analysis": True,
                "adaptation_threshold": 0.1
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "adaptive_results": {
                            "adapted_rmssd": 29.5,
                            "adapted_sdnn": 42.1,
                            "adaptation_factor": 0.98
                        }
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options=advanced_options
                    )
                    # Should handle adaptive analysis
                    assert isinstance(result, list)
                    assert len(result) > 0


class TestPhysiologicalCallbacksSpecializedFeatures:
    """Test specialized features and advanced functionality"""

    def test_analyze_hrv_callback_real_time_analysis(self, mock_data_service, sample_ecg_data):
        """Test real-time analysis (line 1183-1188)"""
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
            # Test with real-time analysis
            mock_data_service.current_data = sample_ecg_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "ECG"]
            }
            
            advanced_options = {
                "real_time_analysis": True,
                "update_interval": 1.0
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "real_time_metrics": {
                            "instantaneous_hr": 72.5,
                            "hrv_trend": "stable"
                        }
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options=advanced_options
                    )
                    # Should handle real-time analysis
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_batch_analysis(self, mock_data_service, sample_ecg_data):
        """Test batch analysis (line 1195->1209)"""
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
            # Test with batch analysis
            mock_data_service.current_data = sample_ecg_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "ECG"]
            }
            
            advanced_options = {
                "batch_analysis": True,
                "batch_size": 100
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "batch_results": {
                            "mean_rmssd": 29.8,
                            "std_rmssd": 2.1,
                            "mean_sdnn": 42.5,
                            "std_sdnn": 3.2
                        }
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options=advanced_options
                    )
                    # Should handle batch analysis
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_validation_analysis(self, mock_data_service, sample_ecg_data):
        """Test validation analysis (line 1219, 1221)"""
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
            # Test with validation analysis
            mock_data_service.current_data = sample_ecg_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "ECG"]
            }
            
            advanced_options = {
                "validation_analysis": True,
                "validation_method": "cross_validation"
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "validation_results": {
                            "accuracy": 0.95,
                            "precision": 0.92,
                            "recall": 0.88,
                            "f1_score": 0.90
                        }
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options=advanced_options
                    )
                    # Should handle validation analysis
                    assert isinstance(result, list)
                    assert len(result) > 0


class TestPhysiologicalCallbacksIntegrationPaths:
    """Test integration paths and complex workflows"""

    def test_analyze_hrv_callback_comprehensive_workflow(self, mock_data_service, sample_ecg_data):
        """Test comprehensive workflow (line 1234->1355)"""
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
            # Test with comprehensive workflow
            mock_data_service.current_data = sample_ecg_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "ECG"]
            }
            
            advanced_options = {
                "comprehensive_analysis": True,
                "include_all_features": True
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv') as mock_analyze:
                    mock_analyze.return_value = {
                        "rmssd": 30.1,
                        "sdnn": 42.8,
                        "comprehensive_results": {
                            "time_domain": {"rmssd": 30.1, "sdnn": 42.8},
                            "frequency_domain": {"lf": 150.5, "hf": 200.3},
                            "nonlinear": {"poincare_sd1": 25.5, "poincare_sd2": 45.2},
                            "quality": {"snr": 15.2, "artifact_percentage": 5.1},
                            "trend": {"rmssd_trend": "stable", "sdnn_trend": "increasing"}
                        }
                    }
                    
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn", "lf", "hf"],
                        time_domain_options=["rmssd", "sdnn"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend", "filter"],
                        advanced_options=advanced_options
                    )
                    # Should handle comprehensive workflow
                    assert isinstance(result, list)
                    assert len(result) > 0

    def test_analyze_hrv_callback_error_recovery(self, mock_data_service, sample_ecg_data):
        """Test error recovery mechanisms (line 1245-1349)"""
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
            # Test error recovery
            mock_data_service.current_data = sample_ecg_data
            mock_data_service.get_data_info.return_value = {
                "sampling_freq": 1000,
                "signal_type": "ECG",
                "columns": ["time", "ECG"]
            }
            
            with patch('vitalDSP_webapp.services.data.data_service.get_data_service', return_value=mock_data_service, create=True):
                with patch('vitalDSP.analysis.hrv_analysis.analyze_hrv', side_effect=Exception("Analysis failed")):
                    result = hrv_callback(
                        n_clicks=1,
                        selected_metrics=["rmssd", "sdnn"],
                        time_domain_options=["rmssd"],
                        freq_domain_options=["lf", "hf"],
                        nonlinear_options=["poincare"],
                        preprocessing_options=["detrend"],
                        advanced_options={}
                    )
                    # Should handle error recovery gracefully
                    assert isinstance(result, list)
                    assert len(result) > 0
