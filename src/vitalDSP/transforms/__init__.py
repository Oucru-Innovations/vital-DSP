"""
Signal Transforms Module for Physiological Signal Processing

This module provides comprehensive signal transformation capabilities for
physiological signals including ECG, PPG, EEG, and other vital signs. It
implements various transform methods for time-frequency analysis, feature
extraction, and signal decomposition.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Fourier Transform (FFT) and Inverse FFT
- Short-Time Fourier Transform (STFT)
- Wavelet Transform (DWT) with multiple mother wavelets
- Discrete Cosine Transform (DCT)
- Hilbert Transform for instantaneous frequency
- Mel-Frequency Cepstral Coefficients (MFCC)
- Principal Component Analysis (PCA) and Independent Component Analysis (ICA)
- Chroma STFT for musical analysis
- Event-Related Potential (ERP) analysis
- Time-frequency representation methods
- Vital signal-specific transformations
- Beat transformation algorithms
- DCT-Wavelet fusion techniques
- Wavelet-FFT fusion methods

Examples:
--------
Basic Fourier Transform:
    >>> from vitalDSP.transforms import FourierTransform
    >>> ft = FourierTransform(signal)
    >>> spectrum = ft.compute_dft()

Wavelet Transform:
    >>> from vitalDSP.transforms import WaveletTransform
    >>> wt = WaveletTransform(signal, wavelet_name="haar")
    >>> coefficients = wt.perform_wavelet_transform()

STFT analysis:
    >>> from vitalDSP.transforms import STFT
    >>> stft = STFT(signal, fs=250)
    >>> spectrogram = stft.compute_stft()
"""

from .fourier_transform import FourierTransform
from .wavelet_transform import WaveletTransform
from .stft import STFT
from .discrete_cosine_transform import DiscreteCosineTransform
from .hilbert_transform import HilbertTransform
from .mfcc import MFCC
from .pca_ica_signal_decomposition import PCASignalDecomposition, ICASignalDecomposition
from .chroma_stft import ChromaSTFT
from .event_related_potential import EventRelatedPotential
from .time_freq_representation import TimeFreqRepresentation
from .vital_transformation import VitalTransformation
from .beats_transformation import RRTransformation
from .dct_wavelet_fusion import DCTWaveletFusion
from .wavelet_fft_fusion import WaveletFFTfusion

__all__ = [
    "FourierTransform",
    "WaveletTransform",
    "STFT",
    "DiscreteCosineTransform",
    "HilbertTransform",
    "MFCC",
    "PCASignalDecomposition",
    "ICASignalDecomposition",
    "ChromaSTFT",
    "EventRelatedPotential",
    "TimeFreqRepresentation",
    "VitalTransformation",
    "RRTransformation",
    "DCTWaveletFusion",
    "WaveletFFTfusion",
]
