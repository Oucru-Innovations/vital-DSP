"""
Warning configuration for vitalDSP.

This module provides centralized warning management to suppress
known warnings that don't affect functionality.
"""

import warnings
import sys

try:
    from scipy.stats import SmallSampleWarning
except ImportError:
    SmallSampleWarning = UserWarning


def configure_warnings():
    """Configure warning filters for vitalDSP."""
    
    # Suppress numpy matrix deprecation warnings from scipy
    warnings.filterwarnings(
        "ignore", 
        message=".*matrix subclass is not the recommended way.*",
        category=PendingDeprecationWarning
    )
    
    # Suppress all numpy matrixlib warnings
    warnings.filterwarnings(
        "ignore",
        category=PendingDeprecationWarning,
        module="numpy.matrixlib"
    )
    
    # Suppress numpy bool8 deprecation warnings from TensorFlow
    warnings.filterwarnings(
        "ignore",
        message=".*np.bool8 is a deprecated alias.*",
        category=DeprecationWarning,
        module="tensorflow"
    )
    
    # Suppress scipy runtime warnings for edge cases
    warnings.filterwarnings(
        "ignore",
        message=".*invalid value encountered in.*",
        category=RuntimeWarning,
        module="scipy"
    )
    
    # Suppress scipy warnings about empty slices
    warnings.filterwarnings(
        "ignore",
        message=".*Mean of empty slice.*",
        category=RuntimeWarning,
        module="numpy"
    )
    
    # Suppress scipy warnings about nperseg being greater than input length
    warnings.filterwarnings(
        "ignore",
        message=".*nperseg.*is greater than input length.*",
        category=UserWarning,
        module="scipy"
    )
    
    # Suppress scipy warnings about log scaling with no positive values
    warnings.filterwarnings(
        "ignore",
        message=".*Data has no positive values.*",
        category=UserWarning,
        module="scipy"
    )
    
    # Suppress scipy warnings about small samples
    warnings.filterwarnings(
        "ignore",
        message=".*One or more sample arguments is too small.*",
        category=SmallSampleWarning
    )
    
    # Suppress deprecation warnings for beat_to_beat functions
    warnings.filterwarnings(
        "ignore",
        message=".*is deprecated.*",
        category=DeprecationWarning,
        module="vitalDSP.physiological_features.beat_to_beat"
    )


def suppress_test_warnings():
    """Suppress warnings specifically for test environments."""
    if "pytest" in sys.modules:
        # Additional warning suppression for tests
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)


# Auto-configure warnings when module is imported
configure_warnings()
