"""
Backward-compatibility shim. The module has been renamed to
ecg_ppg_synchronization_features (fixing the typo).
"""
import warnings

warnings.warn(
    "ecg_ppg_synchronyzation_features is deprecated due to a typo. "
    "Use ecg_ppg_synchronization_features instead.",
    DeprecationWarning,
    stacklevel=2,
)

from vitalDSP.feature_engineering.ecg_ppg_synchronization_features import *  # noqa: F401,F403
from vitalDSP.feature_engineering.ecg_ppg_synchronization_features import ECGPPGSynchronization  # noqa: F401
