"""
vitalDSP Filtering Module

Provides signal filtering capabilities including bandpass filters,
artifact removal, and advanced signal processing.
"""

from .signal_filtering import SignalFiltering, BandpassFilter
from .artifact_removal import ArtifactRemoval
from .advanced_signal_filtering import AdvancedSignalFiltering

__all__ = ['SignalFiltering', 'BandpassFilter', 'ArtifactRemoval', 'AdvancedSignalFiltering']
