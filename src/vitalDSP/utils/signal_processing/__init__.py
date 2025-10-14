"""
Signal Processing Module

This module contains utilities for signal processing, transforms, and features.

Components:
- peak_detection: Peak detection algorithms
- interpolations: Signal interpolation methods
- normalization: Signal normalization utilities
- scaler: Data scaling utilities
- mother_wavelets: Wavelet transform utilities
- convolutional_kernels: Convolutional operation kernels
- loss_functions: Loss functions for machine learning
- attention_weights: Attention mechanism utilities
"""

from .peak_detection import PeakDetection

from .interpolations import (
    linear_interpolation,
    spline_interpolation,
    mean_imputation,
    median_imputation,
)

from .normalization import z_score_normalization, min_max_normalization

from .scaler import StandardScaler

from .mother_wavelets import Wavelet

from .convolutional_kernels import ConvolutionKernels

from .loss_functions import LossFunctions

from .attention_weights import AttentionWeights

__all__ = [
    # Peak Detection
    "PeakDetection",
    # Interpolations
    "linear_interpolation",
    "spline_interpolation",
    "mean_imputation",
    "median_imputation",
    # Normalization
    "z_score_normalization",
    "min_max_normalization",
    # Scaler
    "StandardScaler",
    # Mother Wavelets
    "Wavelet",
    # Convolutional Kernels
    "ConvolutionKernels",
    # Loss Functions
    "LossFunctions",
    # Attention Weights
    "AttentionWeights",
]
