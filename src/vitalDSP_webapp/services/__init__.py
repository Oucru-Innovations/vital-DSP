"""
Services module for vitalDSP webapp.

This module contains service classes that handle integration between
the webapp frontend and the core vitalDSP processing components.
"""

from .progress_tracker import get_progress_tracker, ProgressTracker, track_progress

__all__ = [
    "get_progress_tracker",
    "ProgressTracker",
    "track_progress",
]
