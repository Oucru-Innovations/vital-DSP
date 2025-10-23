"""
vitalDSP - Digital Signal Processing for Vital Signs

A comprehensive library for processing and analyzing physiological signals
including ECG, PPG, and other vital signs data. This library provides advanced
signal processing capabilities with optimized performance for large-scale data
processing, real-time analysis, and health monitoring applications.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Advanced signal filtering and preprocessing
- Comprehensive physiological feature extraction
- Real-time respiratory analysis
- Signal quality assessment and artifact detection
- Machine learning models for health analysis
- Optimized parallel processing pipelines
- Interactive visualization tools

Modules:
- filtering: Signal filtering and artifact removal
- physiological_features: Feature extraction from vital signs
- respiratory_analysis: Respiratory rate estimation and analysis
- transforms: Signal transformations (FFT, Wavelet, etc.)
- signal_quality_assessment: Quality metrics and artifact detection
- health_analysis: Health report generation and interpretation
- advanced_computation: Advanced algorithms (EMD, Kalman, Neural Networks)
- ml_models: Machine learning models and feature extraction
- utils: Core infrastructure and utility functions
- visualization: Plotting and visualization tools

Examples:
--------
Basic signal filtering:
    >>> from vitalDSP import SignalFiltering
    >>> filter_obj = SignalFiltering(signal, fs=250)
    >>> filtered = filter_obj.bandpass_filter(low=0.5, high=40)

Feature extraction:
    >>> from vitalDSP import TimeDomainFeatures
    >>> features = TimeDomainFeatures(signal, fs=250)
    >>> hr_features = features.extract_hr_features()

Respiratory analysis:
    >>> from vitalDSP import RespiratoryAnalysis
    >>> resp_analyzer = RespiratoryAnalysis(signal, fs=250)
    >>> rr_rate = resp_analyzer.estimate_respiratory_rate()

Signal quality assessment:
    >>> from vitalDSP import SignalQuality
    >>> sq = SignalQuality(original_signal, processed_signal)
    >>> snr_db = sq.snr()
"""

# Configure warnings before importing modules
from .utils.warning_config import configure_warnings, suppress_test_warnings

configure_warnings()
suppress_test_warnings()

# Version information
__version__ = "1.0.0"
__author__ = "vitalDSP Team"
__description__ = "Digital Signal Processing for Vital Signs"

# Import main modules
try:
    from .filtering.signal_filtering import SignalFiltering, BandpassFilter
    from .physiological_features.time_domain import TimeDomainFeatures
    from .physiological_features.frequency_domain import FrequencyDomainFeatures
    from .physiological_features.hrv_analysis import HRVFeatures
    from .respiratory_analysis.respiratory_analysis import RespiratoryAnalysis
    from .transforms.fourier_transform import FourierTransform
    from .transforms.wavelet_transform import WaveletTransform
    from .signal_quality_assessment.signal_quality import SignalQuality
    from .health_analysis.health_report_generator import HealthReportGenerator
    from .utils.signal_processing.peak_detection import PeakDetection
    from .utils.signal_processing.scaler import StandardScaler
    from .utils.data_processing.synthesize_data import SynthesizeData
    from .advanced_computation.anomaly_detection import AnomalyDetection
    from .advanced_computation.bayesian_analysis import BayesianOptimization
    from .advanced_computation.emd import EMD
    from .advanced_computation.neural_network_filtering import NeuralNetworkFiltering
    from .advanced_computation.reinforcement_learning_filter import (
        ReinforcementLearningFilter,
    )
    from .advanced_computation.sparse_signal_processing import SparseSignalProcessing

    __all__ = [
        "SignalFiltering",
        "BandpassFilter",
        "TimeDomainFeatures",
        "FrequencyDomainFeatures",
        "HRVFeatures",
        "RespiratoryAnalysis",
        "FourierTransform",
        "WaveletTransform",
        "SignalQuality",
        "HealthReportGenerator",
        "PeakDetection",
        "StandardScaler",
        "SynthesizeData",
        "AnomalyDetection",
        "BayesianOptimization",
        "EMD",
        "NeuralNetworkFiltering",
        "ReinforcementLearningFilter",
        "SparseSignalProcessing",
    ]

except ImportError as e:
    print(f"Warning: Some vitalDSP modules could not be imported: {e}")
    __all__ = []
