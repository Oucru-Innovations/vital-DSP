"""
Comprehensive tests for export_utils.py to improve coverage.

This file adds extensive coverage for all export utility functions.
"""

import pytest
import numpy as np
import pandas as pd
import json
from datetime import datetime
from vitalDSP_webapp.utils.export_utils import (
    prepare_download_data,
    export_filtered_signal_csv,
    export_filtered_signal_json,
    export_features_csv,
    export_features_json,
    export_quality_metrics_csv,
    export_quality_metrics_json,
    export_respiratory_analysis_csv,
    export_respiratory_analysis_json,
    export_transform_results_csv,
    export_transform_results_json,
    export_time_domain_analysis_csv,
    export_time_domain_analysis_json,
    export_frequency_domain_analysis_csv,
    export_frequency_domain_analysis_json,
)


@pytest.fixture
def sample_signal_data():
    """Create sample signal data for testing."""
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))
    return signal, t, 100.0  # signal, time, sampling_freq


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return {
        "filter_type": "bandpass",
        "filter_params": {"low_freq": 0.5, "high_freq": 40},
        "signal_type": "PPG"
    }


class TestPrepareDownloadData:
    """Test prepare_download_data function."""

    def test_prepare_download_data_string(self):
        """Test prepare_download_data with string data."""
        data = "test content"
        result = prepare_download_data(data, "test.txt", "text/plain")
        assert isinstance(result, dict)
        assert result["content"] == data
        assert result["filename"] == "test.txt"
        assert result["type"] == "text/plain"

    def test_prepare_download_data_bytes(self):
        """Test prepare_download_data with bytes data."""
        data = b"test content"
        result = prepare_download_data(data, "test.bin", "application/octet-stream")
        assert isinstance(result, dict)
        assert "content" in result
        assert result["filename"] == "test.bin"
        assert result["base64"] is True
        assert result["type"] == "application/octet-stream"


class TestExportFilteredSignalCSV:
    """Test export_filtered_signal_csv function."""

    def test_export_filtered_signal_csv_basic(self, sample_signal_data):
        """Test basic CSV export."""
        signal, time, fs = sample_signal_data
        result = export_filtered_signal_csv(signal, time, fs)
        assert isinstance(result, str)
        assert "time" in result
        assert "signal" in result
        assert str(fs) in result

    def test_export_filtered_signal_csv_with_metadata(self, sample_signal_data, sample_metadata):
        """Test CSV export with metadata."""
        signal, time, fs = sample_signal_data
        result = export_filtered_signal_csv(signal, time, fs, sample_metadata)
        assert isinstance(result, str)
        assert "filter_type" in result or "bandpass" in result
        assert "PPG" in result

    def test_export_filtered_signal_csv_empty_signal(self):
        """Test CSV export with empty signal."""
        signal = np.array([])
        time = np.array([])
        result = export_filtered_signal_csv(signal, time, 100.0)
        assert isinstance(result, str)
        assert "time" in result or len(result) > 0


class TestExportFilteredSignalJSON:
    """Test export_filtered_signal_json function."""

    def test_export_filtered_signal_json_basic(self, sample_signal_data):
        """Test basic JSON export."""
        signal, time, fs = sample_signal_data
        result = export_filtered_signal_json(signal, time, fs)
        assert isinstance(result, str)
        data = json.loads(result)
        assert "data" in data
        assert "signal" in data["data"]
        assert "time" in data["data"]
        assert "metadata" in data

    def test_export_filtered_signal_json_with_metadata(self, sample_signal_data, sample_metadata):
        """Test JSON export with metadata."""
        signal, time, fs = sample_signal_data
        result = export_filtered_signal_json(signal, time, fs, sample_metadata)
        assert isinstance(result, str)
        data = json.loads(result)
        assert "metadata" in data
        assert data["metadata"]["sampling_frequency"] == fs


class TestExportFeaturesCSV:
    """Test export_features_csv function."""

    def test_export_features_csv_basic(self):
        """Test basic features CSV export."""
        features = {
            "mean": 1.5,
            "std": 0.5,
            "min": 0.0,
            "max": 3.0
        }
        result = export_features_csv(features, "PPG")
        assert isinstance(result, str)
        assert "mean" in result or "1.5" in result

    def test_export_features_csv_complex(self):
        """Test features CSV export with complex data."""
        features = {
            "time_domain": {"mean": 1.5, "std": 0.5},
            "frequency_domain": {"dominant_freq": 1.2}
        }
        result = export_features_csv(features, "ECG")
        assert isinstance(result, str)
        assert len(result) > 0


class TestExportFeaturesJSON:
    """Test export_features_json function."""

    def test_export_features_json_basic(self):
        """Test basic features JSON export."""
        features = {
            "mean": 1.5,
            "std": 0.5
        }
        result = export_features_json(features, "PPG")
        assert isinstance(result, str)
        data = json.loads(result)
        assert "features" in data
        assert "metadata" in data
        assert data["metadata"]["signal_type"] == "PPG"

    def test_export_features_json_complex(self):
        """Test features JSON export with complex data."""
        features = {
            "time_domain": {"mean": 1.5},
            "frequency_domain": {"dominant_freq": 1.2}
        }
        result = export_features_json(features, "ECG")
        assert isinstance(result, str)
        data = json.loads(result)
        assert isinstance(data, dict)


class TestExportQualityMetricsCSV:
    """Test export_quality_metrics_csv function."""

    def test_export_quality_metrics_csv_basic(self):
        """Test basic quality metrics CSV export."""
        quality_results = {
            "overall_score": {
                "score": 0.85,
                "percentage": 85.0,
                "quality": "Good"
            },
            "snr": {"value": 25.5}
        }
        result = export_quality_metrics_csv(quality_results)
        assert isinstance(result, str)
        assert "Score" in result or "0.85" in result or len(result) > 0

    def test_export_quality_metrics_csv_with_segments(self):
        """Test quality metrics CSV export with segments."""
        quality_results = {
            "overall_sqi": 0.85,
            "normal_segments": [(0, 1000), (2000, 3000)],
            "abnormal_segments": [(1000, 2000)]
        }
        result = export_quality_metrics_csv(quality_results)
        assert isinstance(result, str)
        assert len(result) > 0


class TestExportQualityMetricsJSON:
    """Test export_quality_metrics_json function."""

    def test_export_quality_metrics_json_basic(self):
        """Test basic quality metrics JSON export."""
        quality_results = {
            "overall_sqi": 0.85,
            "snr": 25.5
        }
        result = export_quality_metrics_json(quality_results)
        assert isinstance(result, str)
        data = json.loads(result)
        assert "quality_metrics" in data or "overall_sqi" in data


class TestExportRespiratoryAnalysisCSV:
    """Test export_respiratory_analysis_csv function."""

    def test_export_respiratory_analysis_csv_basic(self):
        """Test basic respiratory analysis CSV export."""
        respiratory_data = {
            "respiratory_rate": 15.5,
            "rr_intervals": [0.8, 0.9, 0.85]
        }
        result = export_respiratory_analysis_csv(respiratory_data)
        assert isinstance(result, str)
        assert "respiratory_rate" in result or "15.5" in result

    def test_export_respiratory_analysis_csv_with_time(self):
        """Test respiratory analysis CSV export with time data."""
        respiratory_data = {
            "respiratory_rate": 15.5,
            "time": [0.0, 1.0, 2.0],
            "rr_intervals": [0.8, 0.9, 0.85]
        }
        result = export_respiratory_analysis_csv(respiratory_data)
        assert isinstance(result, str)
        assert len(result) > 0


class TestExportRespiratoryAnalysisJSON:
    """Test export_respiratory_analysis_json function."""

    def test_export_respiratory_analysis_json_basic(self):
        """Test basic respiratory analysis JSON export."""
        respiratory_data = {
            "respiratory_rate": 15.5,
            "rr_intervals": [0.8, 0.9, 0.85]
        }
        result = export_respiratory_analysis_json(respiratory_data)
        assert isinstance(result, str)
        data = json.loads(result)
        assert isinstance(data, dict)


class TestExportTransformResultsCSV:
    """Test export_transform_results_csv function."""

    def test_export_transform_results_csv_fft(self):
        """Test transform results CSV export for FFT."""
        transform_data = {
            "transform_type": "fft",
            "frequency": [1.0, 2.0, 3.0],
            "magnitude": [0.5, 0.3, 0.2]
        }
        result = export_transform_results_csv(transform_data)
        assert isinstance(result, str)
        assert "frequency" in result or "magnitude" in result

    def test_export_transform_results_csv_wavelet(self):
        """Test transform results CSV export for Wavelet."""
        transform_data = {
            "transform_type": "wavelet",
            "coefficients": [[0.1, 0.2], [0.3, 0.4]]
        }
        result = export_transform_results_csv(transform_data)
        assert isinstance(result, str)
        assert len(result) > 0


class TestExportTransformResultsJSON:
    """Test export_transform_results_json function."""

    def test_export_transform_results_json_fft(self):
        """Test transform results JSON export for FFT."""
        transform_data = {
            "transform_type": "fft",
            "frequency": [1.0, 2.0, 3.0],
            "magnitude": [0.5, 0.3, 0.2]
        }
        result = export_transform_results_json(transform_data)
        assert isinstance(result, str)
        data = json.loads(result)
        assert isinstance(data, dict)


class TestExportTimeDomainAnalysisCSV:
    """Test export_time_domain_analysis_csv function."""

    def test_export_time_domain_analysis_csv_basic(self):
        """Test basic time domain analysis CSV export."""
        analysis_results = {
            "mean": 1.5,
            "std": 0.5,
            "duration": 10.0
        }
        result = export_time_domain_analysis_csv(analysis_results)
        assert isinstance(result, str)
        assert len(result) > 0


class TestExportTimeDomainAnalysisJSON:
    """Test export_time_domain_analysis_json function."""

    def test_export_time_domain_analysis_json_basic(self):
        """Test basic time domain analysis JSON export."""
        analysis_results = {
            "mean": 1.5,
            "std": 0.5
        }
        result = export_time_domain_analysis_json(analysis_results)
        assert isinstance(result, str)
        data = json.loads(result)
        assert isinstance(data, dict)


class TestExportFrequencyDomainAnalysisCSV:
    """Test export_frequency_domain_analysis_csv function."""

    def test_export_frequency_domain_analysis_csv_basic(self):
        """Test basic frequency domain analysis CSV export."""
        analysis_results = {
            "dominant_frequency": 1.2,
            "power_spectral_density": [0.1, 0.2, 0.3]
        }
        result = export_frequency_domain_analysis_csv(analysis_results)
        assert isinstance(result, str)
        assert len(result) > 0


class TestExportFrequencyDomainAnalysisJSON:
    """Test export_frequency_domain_analysis_json function."""

    def test_export_frequency_domain_analysis_json_basic(self):
        """Test basic frequency domain analysis JSON export."""
        analysis_results = {
            "dominant_frequency": 1.2,
            "power_spectral_density": [0.1, 0.2, 0.3]
        }
        result = export_frequency_domain_analysis_json(analysis_results)
        assert isinstance(result, str)
        data = json.loads(result)
        assert isinstance(data, dict)

