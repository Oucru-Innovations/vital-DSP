"""
Feature Engineering Module for Physiological Signal Processing

This module provides comprehensive feature engineering capabilities for
physiological signals including ECG, PPG, EEG, and other vital signs. It
implements advanced algorithms for extracting meaningful features from
raw signals for machine learning and analysis applications.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- ECG autonomic feature extraction
- PPG autonomic feature analysis
- ECG-PPG synchronization features
- Morphology feature extraction
- PPG light-based feature analysis
- Comprehensive feature engineering pipeline
- Signal-specific feature optimization
- Advanced morphological analysis

Examples:
--------
ECG feature extraction:
    >>> from vitalDSP.feature_engineering import ECGExtractor
    >>> ecg_features = ECGExtractor(ecg_signal, fs=250)
    >>> p_wave_duration = ecg_features.compute_p_wave_duration()

PPG feature analysis:
    >>> from vitalDSP.feature_engineering import PPGExtractor
    >>> ppg_features = PPGExtractor(ppg_signal, fs=128)
    >>> autonomic_features = ppg_features.extract_autonomic_features()

Morphology analysis:
    >>> from vitalDSP.feature_engineering import MorphologyFeatures
    >>> morph_features = MorphologyFeatures(signal, fs=250)
    >>> waveform_features = morph_features.extract_waveform_features()
"""

from .ecg_autonomic_features import ECGExtractor
from .ppg_autonomic_features import PPGAutonomicFeatures
from .ecg_ppg_synchronyzation_features import ECGPPGSynchronization
from .morphology_features import PhysiologicalFeatureExtractor
from .ppg_light_features import PPGLightFeatureExtractor

__all__ = [
    "ECGExtractor",
    "PPGAutonomicFeatures",
    "ECGPPGSynchronization",
    "PhysiologicalFeatureExtractor",
    "PPGLightFeatureExtractor",
]
