"""
Test configuration and fixtures for vitalDSP_webapp
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os


@pytest.fixture
def sample_ppg_data():
    """Sample PPG data for testing"""
    # Generate synthetic PPG-like data
    fs = 100  # 100 Hz sampling rate
    duration = 10  # 10 seconds
    t = np.linspace(0, duration, fs * duration)
    
    # Create a realistic PPG signal with heart rate around 60-80 BPM
    heart_rate = 70  # BPM
    signal = np.sin(2 * np.pi * heart_rate / 60 * t)
    
    # Add some noise and artifacts
    noise = 0.1 * np.random.randn(len(t))
    artifacts = 0.3 * np.exp(-((t - 5) / 0.5) ** 2)  # Gaussian artifact at 5 seconds
    
    ppg_signal = signal + noise + artifacts
    
    return pd.DataFrame({
        'timestamp': t,
        'ppg': ppg_signal,
        'quality': np.ones(len(t)) * 0.9  # High quality signal
    })


@pytest.fixture
def sample_ecg_data():
    """Sample ECG data for testing"""
    # Generate synthetic ECG-like data
    fs = 250  # 250 Hz sampling rate
    duration = 10  # 10 seconds
    t = np.linspace(0, duration, fs * duration)
    
    # Create a realistic ECG signal with heart rate around 60-80 BPM
    heart_rate = 75  # BPM
    signal = np.zeros_like(t)
    
    # Add QRS complexes
    for i in range(int(duration * heart_rate / 60)):
        peak_time = i * 60 / heart_rate
        if peak_time < duration:
            peak_idx = int(peak_time * fs)
            if peak_idx < len(signal):
                # QRS complex shape
                signal[max(0, peak_idx-10):min(len(signal), peak_idx+10)] = 1.0
                signal[max(0, peak_idx-5):min(len(signal), peak_idx+5)] = 2.0
    
    # Add some noise
    noise = 0.05 * np.random.randn(len(t))
    ecg_signal = signal + noise
    
    return pd.DataFrame({
        'timestamp': t,
        'ecg': ecg_signal,
        'quality': np.ones(len(t)) * 0.95  # Very high quality signal
    })


@pytest.fixture
def sample_respiratory_data():
    """Sample respiratory data for testing"""
    # Generate synthetic respiratory data
    fs = 50  # 50 Hz sampling rate
    duration = 30  # 30 seconds
    t = np.linspace(0, duration, fs * duration)
    
    # Create a realistic respiratory signal with breathing rate around 12-20 BPM
    breathing_rate = 15  # BPM
    signal = np.sin(2 * np.pi * breathing_rate / 60 * t)
    
    # Add some variability and noise
    variability = 0.2 * np.sin(2 * np.pi * 0.1 * t)  # Slow variability
    noise = 0.05 * np.random.randn(len(t))
    resp_signal = signal + variability + noise
    
    return pd.DataFrame({
        'timestamp': t,
        'respiratory': resp_signal,
        'quality': np.ones(len(t)) * 0.85  # Good quality signal
    })


@pytest.fixture
def mock_dash_app():
    """Mock Dash app for testing"""
    app = Mock()
    app.layout = Mock()
    app.callback = Mock()
    app.clientside_callback = Mock()
    return app


@pytest.fixture
def temp_upload_dir():
    """Temporary directory for testing file uploads"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_file_upload():
    """Mock file upload for testing"""
    upload = Mock()
    upload.filename = "test_signal.csv"
    upload.content_type = "text/csv"
    upload.read.return_value = b"timestamp,value\n0,1.0\n1,2.0\n2,3.0"
    return upload


@pytest.fixture
def sample_analysis_parameters():
    """Sample analysis parameters for testing"""
    return {
        'filter_type': 'bandpass',
        'low_freq': 0.5,
        'high_freq': 40.0,
        'filter_order': 4,
        'window_size': 1024,
        'overlap': 0.5,
        'feature_types': ['hrv', 'morphology', 'frequency'],
        'quality_threshold': 0.8
    }


@pytest.fixture
def mock_signal_processor():
    """Mock signal processor for testing"""
    processor = Mock()
    processor.load_signal.return_value = pd.DataFrame({'timestamp': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})
    processor.preprocess_signal.return_value = pd.DataFrame({'timestamp': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})
    processor.apply_filter.return_value = pd.DataFrame({'timestamp': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})
    processor.extract_features.return_value = {'hrv': 75.0, 'quality': 0.9}
    return processor
