"""
vitalDSP Physiological Features Module

Provides feature extraction capabilities for physiological signals
including time domain, frequency domain, and HRV analysis.
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
from .cross_correlation import CrossCorrelation
from .signal_power_analysis import SignalPowerAnalysis

__all__ = [
    'TimeDomainFeatures', 'FrequencyDomainFeatures', 'HRVFeatures',
    'BeatToBeatAnalysis', 'EnergyAnalysis', 'EnvelopeDetection',
    'SignalSegmentation', 'TrendAnalysis', 'WaveformMorphology',
    'NonlinearFeatures', 'CrossCorrelation', 'SignalPowerAnalysis'
]
