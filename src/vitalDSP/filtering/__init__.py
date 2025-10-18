"""
Signal Filtering Module for Physiological Signal Processing

This module provides comprehensive signal filtering capabilities for physiological
signals including ECG, PPG, EEG, and other vital signs. It implements various
filtering techniques, artifact removal methods, and advanced signal processing
algorithms for noise reduction and signal enhancement.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Advanced signal filtering algorithms
- Artifact removal and noise reduction
- Multiple filter types (Butterworth, Chebyshev, Elliptic)
- Bandpass, lowpass, highpass, and notch filtering
- Real-time filtering capabilities
- Adaptive filtering techniques
- Comprehensive signal conditioning

Examples:
--------
Basic signal filtering:
    >>> from vitalDSP.filtering import SignalFiltering
    >>> filter_obj = SignalFiltering(signal, fs=250)
    >>> filtered = filter_obj.bandpass_filter(low=0.5, high=40)

Advanced filtering:
    >>> from vitalDSP.filtering import AdvancedSignalFiltering
    >>> af = AdvancedSignalFiltering(signal)
    >>> kalman_filtered = af.kalman_filter(R=0.1, Q=0.01)

Artifact removal:
    >>> from vitalDSP.filtering import ArtifactRemoval
    >>> ar = ArtifactRemoval(signal, fs=250)
    >>> clean_signal = ar.remove_artifacts()
"""

from .signal_filtering import SignalFiltering, BandpassFilter
from .artifact_removal import ArtifactRemoval
from .advanced_signal_filtering import AdvancedSignalFiltering

__all__ = [
    "SignalFiltering",
    "BandpassFilter",
    "ArtifactRemoval",
    "AdvancedSignalFiltering",
]
