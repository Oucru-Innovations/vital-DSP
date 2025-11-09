"""
Physiological Features Module for Signal Processing

This module provides comprehensive physiological feature extraction capabilities
for signals including ECG, PPG, EEG, and other vital signs. It implements
advanced algorithms for time-domain, frequency-domain, and non-linear analysis
of physiological signals.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Time-domain feature extraction (SDNN, RMSSD, pNN50)
- Frequency-domain analysis (LF, HF, LF/HF ratio)
- Heart Rate Variability (HRV) analysis
- Beat-to-beat interval analysis
- Energy analysis and power computation
- Envelope detection and signal morphology
- Signal segmentation and trend analysis
- Non-linear analysis methods
- Cross-correlation and coherence analysis
- Advanced entropy measures
- Symbolic dynamics analysis
- Transfer entropy computation
- Signal change detection
- Ensemble-based feature extraction

Examples:
--------
Time-domain HRV analysis:
    >>> from vitalDSP.physiological_features import TimeDomainFeatures
    >>> tdf = TimeDomainFeatures(nn_intervals)
    >>> sdnn = tdf.compute_sdnn()
    >>> rmssd = tdf.compute_rmssd()

Frequency-domain analysis:
    >>> from vitalDSP.physiological_features import FrequencyDomainFeatures
    >>> fdf = FrequencyDomainFeatures(nn_intervals, fs=4)
    >>> psd_result = fdf.compute_psd()

Comprehensive HRV analysis:
    >>> from vitalDSP.physiological_features import HRVFeatures
    >>> hrv = HRVFeatures(nn_intervals, fs=4)
    >>> all_features = hrv.extract_all_features()
"""

from .time_domain import TimeDomainFeatures
from .frequency_domain import FrequencyDomainFeatures
from .hrv_analysis import HRVFeatures
from .beat_to_beat import BeatToBeatAnalysis
from .energy_analysis import EnergyAnalysis
from .envelope_detection import EnvelopeDetection

# from .peak_detection import PeakDetection
from .signal_segmentation import SignalSegmentation
from .trend_analysis import TrendAnalysis
from .waveform import WaveformMorphology
from .nonlinear import NonlinearFeatures
from .cross_correlation import CrossCorrelationFeatures
from .signal_power_analysis import SignalPowerAnalysis

# Advanced Features (Nonlinear Dynamics & Information Theory)
from .advanced_entropy import MultiScaleEntropy
from .symbolic_dynamics import SymbolicDynamics
from .transfer_entropy import TransferEntropy

__all__ = [
    "TimeDomainFeatures",
    "FrequencyDomainFeatures",
    "HRVFeatures",
    "BeatToBeatAnalysis",
    "EnergyAnalysis",
    "EnvelopeDetection",
    "SignalSegmentation",
    "TrendAnalysis",
    "WaveformMorphology",
    "NonlinearFeatures",
    "CrossCorrelationFeatures",
    "SignalPowerAnalysis",
    # Advanced Features
    "MultiScaleEntropy",
    "SymbolicDynamics",
    "TransferEntropy",
]
