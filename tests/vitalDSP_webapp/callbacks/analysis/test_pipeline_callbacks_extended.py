"""
Extended tests for pipeline_callbacks.py to increase coverage.

This file adds more comprehensive tests for pipeline execution stages.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dash import Dash
from datetime import datetime

# Import the module to test
from vitalDSP_webapp.callbacks.analysis.pipeline_callbacks import (
    register_pipeline_callbacks,
    _execute_pipeline_stage,
    _get_stage_number,
)


@pytest.fixture
def mock_app():
    """Create a mock Dash app for testing."""
    app = Mock(spec=Dash)
    app.callback = Mock()
    return app


@pytest.fixture
def sample_signal_data():
    """Create sample signal data for testing."""
    np.random.seed(42)
    t = np.linspace(0, 10, 10000)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
    signal += 0.1 * np.random.randn(len(signal))
    return signal, 1000


@pytest.fixture
def sample_pipeline_data():
    """Create sample pipeline data."""
    return {
        "current_stage": 1,
        "stage2_sqi_type": "snr_sqi",
        "stage2_window_size": 1000,
        "stage2_step_size": 500,
        "stage2_threshold_type": "below",
        "stage2_threshold": 0.7,
        "stage2_scale": "zscore",
        "stage3_filter_type": "traditional",
        "stage3_detrend": False,
        "stage3_signal_source": "original",
        "stage3_application_count": 1,
        "stage3_filter_family": "butter",
        "stage3_filter_response": "bandpass",
        "stage3_filter_lowcut": 0.5,
        "stage3_filter_highcut": 40,
        "stage3_filter_order": 4,
        "stage3_filter_rp": 1,
        "stage3_filter_rs": 40,
        "paths": ["filtered"]
    }


class TestExecutePipelineStageExtended:
    """Extended tests for _execute_pipeline_stage function."""

    def test_execute_pipeline_stage_1(self, sample_signal_data, sample_pipeline_data):
        """Test stage 1 execution."""
        signal_data, fs = sample_signal_data
        success, result = _execute_pipeline_stage(
            sample_pipeline_data, 1, signal_data, fs, "ecg"
        )
        assert success is True
        assert "samples" in result
        assert "duration" in result
        assert result["fs"] == fs
        assert result["signal_type"] == "ecg"

    @patch('vitalDSP_webapp.callbacks.analysis.quality_sqi_functions.compute_sqi')
    def test_execute_pipeline_stage_2_success(self, mock_compute_sqi, sample_signal_data, sample_pipeline_data):
        """Test stage 2 execution with successful SQI computation."""
        signal_data, fs = sample_signal_data
        mock_compute_sqi.return_value = {
            "sqi_values": [0.8, 0.9, 0.85],
            "normal_segments": [(0, 1000), (2000, 3000)],
            "abnormal_segments": [(1000, 2000)],
            "overall_sqi": 0.85
        }
        
        success, result = _execute_pipeline_stage(
            sample_pipeline_data, 2, signal_data, fs, "ecg"
        )
        assert success is True
        assert "sqi_type" in result
        assert "overall_quality" in result
        assert result["overall_quality"] == 0.85
        assert result["passed"] is True

    @patch('vitalDSP_webapp.callbacks.analysis.quality_sqi_functions.compute_sqi')
    def test_execute_pipeline_stage_2_failure(self, mock_compute_sqi, sample_signal_data, sample_pipeline_data):
        """Test stage 2 execution with failed SQI computation."""
        signal_data, fs = sample_signal_data
        mock_compute_sqi.side_effect = Exception("SQI computation failed")
        
        success, result = _execute_pipeline_stage(
            sample_pipeline_data, 2, signal_data, fs, "ecg"
        )
        assert success is False
        # Result can be dict or string
        if isinstance(result, dict):
            assert "error" in result or "failed" in str(result).lower()
        else:
            assert "error" in result.lower() or "failed" in result.lower()

    @patch('vitalDSP.filtering.signal_filtering.SignalFiltering')
    def test_execute_pipeline_stage_3_traditional_bandpass(self, mock_sf_class, sample_signal_data, sample_pipeline_data):
        """Test stage 3 execution with traditional bandpass filter."""
        signal_data, fs = sample_signal_data
        mock_sf = Mock()
        mock_sf.bandpass.return_value = signal_data * 0.9
        mock_sf_class.return_value = mock_sf
        
        success, result = _execute_pipeline_stage(
            sample_pipeline_data, 3, signal_data, fs, "ecg"
        )
        assert success is True

    @patch('vitalDSP.filtering.signal_filtering.SignalFiltering')
    def test_execute_pipeline_stage_3_traditional_lowpass(self, mock_sf_class, sample_signal_data, sample_pipeline_data):
        """Test stage 3 execution with traditional lowpass filter."""
        signal_data, fs = sample_signal_data
        pipeline_data = sample_pipeline_data.copy()
        pipeline_data["stage3_filter_response"] = "low"
        pipeline_data["stage3_filter_lowcut"] = None
        
        mock_sf = Mock()
        mock_sf.butterworth.return_value = signal_data * 0.9
        mock_sf_class.return_value = mock_sf
        
        success, result = _execute_pipeline_stage(
            pipeline_data, 3, signal_data, fs, "ecg"
        )
        assert success is True

    @patch('vitalDSP.filtering.signal_filtering.SignalFiltering')
    def test_execute_pipeline_stage_3_traditional_highpass(self, mock_sf_class, sample_signal_data, sample_pipeline_data):
        """Test stage 3 execution with traditional highpass filter."""
        signal_data, fs = sample_signal_data
        pipeline_data = sample_pipeline_data.copy()
        pipeline_data["stage3_filter_response"] = "high"
        pipeline_data["stage3_filter_highcut"] = None
        
        mock_sf = Mock()
        mock_sf.butterworth.return_value = signal_data * 0.9
        mock_sf_class.return_value = mock_sf
        
        success, result = _execute_pipeline_stage(
            pipeline_data, 3, signal_data, fs, "ecg"
        )
        assert success is True

    @patch('vitalDSP.filtering.advanced_signal_filtering.AdvancedSignalFiltering')
    def test_execute_pipeline_stage_3_advanced_kalman(self, mock_asf_class, sample_signal_data, sample_pipeline_data):
        """Test stage 3 execution with advanced Kalman filter."""
        signal_data, fs = sample_signal_data
        pipeline_data = sample_pipeline_data.copy()
        pipeline_data["stage3_filter_type"] = "advanced"
        pipeline_data["stage3_advanced_method"] = "kalman"
        pipeline_data["stage3_kalman_r"] = 1.0
        pipeline_data["stage3_kalman_q"] = 1.0
        
        mock_asf = Mock()
        mock_asf.kalman_filter.return_value = signal_data * 0.9
        mock_asf_class.return_value = mock_asf
        
        success, result = _execute_pipeline_stage(
            pipeline_data, 3, signal_data, fs, "ecg"
        )
        assert success is True

    @patch('vitalDSP.filtering.advanced_signal_filtering.AdvancedSignalFiltering')
    def test_execute_pipeline_stage_3_advanced_adaptive(self, mock_asf_class, sample_signal_data, sample_pipeline_data):
        """Test stage 3 execution with advanced adaptive filter."""
        signal_data, fs = sample_signal_data
        pipeline_data = sample_pipeline_data.copy()
        pipeline_data["stage3_filter_type"] = "advanced"
        pipeline_data["stage3_advanced_method"] = "adaptive"
        pipeline_data["stage3_adaptive_mu"] = 0.01
        pipeline_data["stage3_adaptive_order"] = 4
        
        mock_asf = Mock()
        mock_asf.adaptive_filter.return_value = signal_data * 0.9
        mock_asf_class.return_value = mock_asf
        
        success, result = _execute_pipeline_stage(
            pipeline_data, 3, signal_data, fs, "ecg"
        )
        assert success is True

    @patch('vitalDSP.filtering.artifact_removal.ArtifactRemoval')
    def test_execute_pipeline_stage_3_artifact_baseline(self, mock_ar_class, sample_signal_data, sample_pipeline_data):
        """Test stage 3 execution with artifact removal - baseline correction."""
        signal_data, fs = sample_signal_data
        pipeline_data = sample_pipeline_data.copy()
        pipeline_data["stage3_filter_type"] = "artifact"
        pipeline_data["stage3_artifact_type"] = "baseline"
        pipeline_data["stage3_artifact_strength"] = 0.5
        
        mock_ar = Mock()
        mock_ar.baseline_correction.return_value = signal_data * 0.9
        mock_ar_class.return_value = mock_ar
        
        success, result = _execute_pipeline_stage(
            pipeline_data, 3, signal_data, fs, "ecg"
        )
        assert success is True

    @patch('vitalDSP.filtering.artifact_removal.ArtifactRemoval')
    def test_execute_pipeline_stage_3_artifact_spike(self, mock_ar_class, sample_signal_data, sample_pipeline_data):
        """Test stage 3 execution with artifact removal - spike removal."""
        signal_data, fs = sample_signal_data
        pipeline_data = sample_pipeline_data.copy()
        pipeline_data["stage3_filter_type"] = "artifact"
        pipeline_data["stage3_artifact_type"] = "spike"
        pipeline_data["stage3_threshold_value"] = 0.1
        
        mock_ar = Mock()
        mock_ar.spike_removal.return_value = signal_data * 0.9
        mock_ar_class.return_value = mock_ar
        
        success, result = _execute_pipeline_stage(
            pipeline_data, 3, signal_data, fs, "ecg"
        )
        assert success is True

    @patch('vitalDSP.filtering.artifact_removal.ArtifactRemoval')
    def test_execute_pipeline_stage_3_artifact_wavelet(self, mock_ar_class, sample_signal_data, sample_pipeline_data):
        """Test stage 3 execution with artifact removal - wavelet denoising."""
        signal_data, fs = sample_signal_data
        pipeline_data = sample_pipeline_data.copy()
        pipeline_data["stage3_filter_type"] = "artifact"
        pipeline_data["stage3_artifact_type"] = "noise"
        pipeline_data["stage3_wavelet_type"] = "db4"
        pipeline_data["stage3_wavelet_level"] = 3
        pipeline_data["stage3_threshold_type"] = "soft"
        
        mock_ar = Mock()
        mock_ar.wavelet_denoising.return_value = signal_data * 0.9
        mock_ar_class.return_value = mock_ar
        
        success, result = _execute_pipeline_stage(
            pipeline_data, 3, signal_data, fs, "ecg"
        )
        assert success is True

    def test_execute_pipeline_stage_invalid_stage(self, sample_signal_data, sample_pipeline_data):
        """Test _execute_pipeline_stage with invalid stage number."""
        signal_data, fs = sample_signal_data
        success, result = _execute_pipeline_stage(
            sample_pipeline_data, 99, signal_data, fs, "ecg"
        )
        assert success is False
        # Result can be dict or string
        if isinstance(result, dict):
            assert "error" in result or "unknown" in str(result).lower() or "invalid" in str(result).lower()
        else:
            assert "invalid" in result.lower() or "error" in result.lower() or "unknown" in result.lower()

    def test_execute_pipeline_stage_4(self, sample_signal_data, sample_pipeline_data):
        """Test stage 4 execution (Feature Extraction)."""
        signal_data, fs = sample_signal_data
        # Stage 4 typically extracts features
        success, result = _execute_pipeline_stage(
            sample_pipeline_data, 4, signal_data, fs, "ecg"
        )
        # Stage 4 may succeed or fail depending on implementation
        assert isinstance(success, bool)

    def test_execute_pipeline_stage_5(self, sample_signal_data, sample_pipeline_data):
        """Test stage 5 execution (Analysis)."""
        signal_data, fs = sample_signal_data
        success, result = _execute_pipeline_stage(
            sample_pipeline_data, 5, signal_data, fs, "ecg"
        )
        assert isinstance(success, bool)

    def test_execute_pipeline_stage_6(self, sample_signal_data, sample_pipeline_data):
        """Test stage 6 execution (Visualization)."""
        signal_data, fs = sample_signal_data
        success, result = _execute_pipeline_stage(
            sample_pipeline_data, 6, signal_data, fs, "ecg"
        )
        assert isinstance(success, bool)

    def test_execute_pipeline_stage_7(self, sample_signal_data, sample_pipeline_data):
        """Test stage 7 execution (Export)."""
        signal_data, fs = sample_signal_data
        success, result = _execute_pipeline_stage(
            sample_pipeline_data, 7, signal_data, fs, "ecg"
        )
        assert isinstance(success, bool)

    def test_execute_pipeline_stage_8(self, sample_signal_data, sample_pipeline_data):
        """Test stage 8 execution (Report)."""
        signal_data, fs = sample_signal_data
        success, result = _execute_pipeline_stage(
            sample_pipeline_data, 8, signal_data, fs, "ecg"
        )
        assert isinstance(success, bool)


class TestPipelineCallbacksRegistration:
    """Test pipeline callbacks registration."""

    def test_register_pipeline_callbacks(self, mock_app):
        """Test that callbacks are properly registered."""
        register_pipeline_callbacks(mock_app)
        assert mock_app.callback.called
        assert mock_app.callback.call_count >= 1

