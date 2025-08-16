"""
vitalDSP - Digital Signal Processing for Vital Signs

A comprehensive library for processing and analyzing physiological signals
including ECG, PPG, and other vital signs data.
"""

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
    from .utils.peak_detection import PeakDetection
    from .utils.scaler import StandardScaler
    from .utils.synthesize_data import SynthesizeData
    from .advanced_computation.anomaly_detection import AnomalyDetection
    from .advanced_computation.bayesian_analysis import BayesianOptimization
    from .advanced_computation.emd import EMD
    from .advanced_computation.neural_network_filtering import NeuralNetworkFiltering
    from .advanced_computation.reinforcement_learning_filter import ReinforcementLearningFilter
    from .advanced_computation.sparse_signal_processing import SparseSignalProcessing
    
    __all__ = [
        'SignalFiltering', 'BandpassFilter', 'TimeDomainFeatures', 'FrequencyDomainFeatures',
        'HRVFeatures', 'RespiratoryAnalysis', 'FourierTransform', 'WaveletTransform',
        'SignalQuality', 'HealthReportGenerator', 'PeakDetection', 'StandardScaler',
        'SynthesizeData', 'AnomalyDetection', 'BayesianOptimization', 'EMD',
        'NeuralNetworkFiltering', 'ReinforcementLearningFilter', 'SparseSignalProcessing'
    ]
    
except ImportError as e:
    print(f"Warning: Some vitalDSP modules could not be imported: {e}")
    __all__ = []
