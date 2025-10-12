"""
vitalDSP Utilities Module

Provides utility functions for signal processing including
peak detection, data synthesis, scaling, data loading, and more.
"""

from .peak_detection import PeakDetection
from .synthesize_data import SynthesizeData
from .scaler import StandardScaler
from .normalization import z_score_normalization, min_max_normalization
from .common import find_peaks, argrelextrema
from . import common
from .interpolations import linear_interpolation, spline_interpolation, mean_imputation
from .mother_wavelets import Wavelet
from .loss_functions import LossFunctions
from .convolutional_kernels import ConvolutionKernels
from .attention_weights import AttentionWeights
from .data_loader import (
    DataLoader,
    StreamDataLoader,
    DataFormat,
    SignalType,
    load_signal,
    load_multi_channel,
    load_oucru_csv
)

__all__ = [
    "PeakDetection",
    "SynthesizeData",
    "StandardScaler",
    "z_score_normalization",
    "min_max_normalization",
    "find_peaks",
    "argrelextrema",
    "linear_interpolation",
    "spline_interpolation",
    "mean_imputation",
    "Wavelet",
    "LossFunctions",
    "ConvolutionKernels",
    "AttentionWeights",
    "DataLoader",
    "StreamDataLoader",
    "DataFormat",
    "SignalType",
    "load_signal",
    "load_multi_channel",
    "load_oucru_csv",
]
