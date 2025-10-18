"""
Advanced Computation Module for Physiological Signal Processing

This module provides advanced computational algorithms for physiological
signal analysis including ECG, PPG, EEG, and other vital signs. It implements
state-of-the-art methods for signal processing, machine learning, and
advanced mathematical analysis.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Anomaly detection and pattern recognition
- Bayesian analysis and optimization
- Empirical Mode Decomposition (EMD)
- Kalman filtering for real-time processing
- Neural network-based filtering
- Reinforcement learning algorithms
- Sparse signal processing techniques
- Generative signal synthesis
- Harmonic-percussive separation
- Multimodal signal fusion
- Non-linear analysis methods
- Pitch shifting and audio processing
- Real-time anomaly detection

Examples:
--------
Basic anomaly detection:
    >>> from vitalDSP.advanced_computation import AnomalyDetection
    >>> detector = AnomalyDetection(signal)
    >>> anomalies = detector.detect_anomalies(method="z_score")

EMD decomposition:
    >>> from vitalDSP.advanced_computation import EMD
    >>> emd = EMD(signal)
    >>> imfs = emd.emd()

Kalman filtering:
    >>> from vitalDSP.advanced_computation import KalmanFilter
    >>> kf = KalmanFilter(signal)
    >>> filtered = kf.filter()
"""

from .anomaly_detection import AnomalyDetection
from .bayesian_analysis import BayesianOptimization
from .emd import EMD
from .kalman_filter import KalmanFilter
from .neural_network_filtering import NeuralNetworkFiltering
from .reinforcement_learning_filter import ReinforcementLearningFilter
from .sparse_signal_processing import SparseSignalProcessing
from .generative_signal_synthesis import GenerativeSignalSynthesis
from .harmonic_percussive_separation import HarmonicPercussiveSeparation
from .multimodal_fusion import MultimodalFusion
from .non_linear_analysis import NonlinearAnalysis
from .pitch_shift import PitchShift
from .real_time_anomaly_detection import RealTimeAnomalyDetection

__all__ = [
    "AnomalyDetection",
    "BayesianOptimization", 
    "EMD",
    "KalmanFilter",
    "NeuralNetworkFiltering",
    "ReinforcementLearningFilter",
    "SparseSignalProcessing",
    "GenerativeSignalSynthesis",
    "HarmonicPercussiveSeparation",
    "MultimodalFusion",
    "NonlinearAnalysis",
    "PitchShift",
    "RealTimeAnomalyDetection",
]
