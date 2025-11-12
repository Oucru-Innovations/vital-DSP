# src/webapp/api/endpoints.py
"""
VitalDSP API Endpoints - Version 0.2.1

Comprehensive REST API for digital signal processing in healthcare applications.
Provides access to all core VitalDSP functionality including filtering, feature extraction,
respiratory analysis, signal quality assessment, and more.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import psutil
import numpy as np

# Import VitalDSP core modules
try:
    from vitalDSP.filtering.signal_filtering import SignalFiltering
    from vitalDSP.physiological_features.time_domain import TimeDomainFeatures
    from vitalDSP.physiological_features.frequency_domain import FrequencyDomainFeatures
    from vitalDSP.physiological_features.hrv_analysis import HRVFeatures
    from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis
    from vitalDSP.signal_quality_assessment.signal_quality import SignalQuality
    from vitalDSP.transforms.fourier_transform import FourierTransform
    from vitalDSP.transforms.wavelet_transform import WaveletTransform
    from vitalDSP.health_analysis.health_report_generator import HealthReportGenerator
except ImportError as e:
    print(f"Warning: Some VitalDSP modules could not be imported: {e}")

router = APIRouter()

# ==================== Request/Response Models ====================

class SignalData(BaseModel):
    """Input signal data model"""
    data: List[float] = Field(..., description="Signal data points")
    sampling_rate: Optional[float] = Field(250.0, description="Sampling rate in Hz")
    signal_type: Optional[str] = Field("ECG", description="Type of signal (ECG, PPG, etc.)")

class FilterRequest(BaseModel):
    """Request model for signal filtering"""
    data: List[float]
    sampling_rate: float = 250.0
    filter_type: str = Field(..., description="Filter type: butterworth, chebyshev, elliptic, bessel, gaussian, savgol")
    low_cutoff: Optional[float] = None
    high_cutoff: Optional[float] = None
    order: Optional[int] = 5

class FeatureExtractionRequest(BaseModel):
    """Request model for feature extraction"""
    data: List[float]
    sampling_rate: float = 250.0
    feature_type: str = Field(..., description="Feature type: time_domain, frequency_domain, hrv, nonlinear")

class RespiratoryRequest(BaseModel):
    """Request model for respiratory analysis"""
    data: List[float]
    sampling_rate: float = 250.0
    signal_type: str = Field("PPG", description="Signal type: ECG, PPG")
    method: Optional[str] = Field("fft", description="Method: fft, wavelet, peak_detection")

class TransformRequest(BaseModel):
    """Request model for signal transforms"""
    data: List[float]
    sampling_rate: float = 250.0
    transform_type: str = Field(..., description="Transform type: fft, wavelet, stft, hilbert")

class QualityAssessmentRequest(BaseModel):
    """Request model for signal quality assessment"""
    original_data: List[float]
    processed_data: Optional[List[float]] = None
    sampling_rate: float = 250.0

# ==================== Health & System Endpoints ====================

@router.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint for monitoring service status.

    Returns system health metrics including:
    - Service status
    - Version information
    - Memory and disk usage
    - Timestamp
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "vitalDSP Webapp",
        "version": "0.2.1",
        "uptime": "running",
        "memory_usage": f"{psutil.virtual_memory().percent}%",
        "disk_usage": f"{psutil.disk_usage('/').percent}%",
    }

@router.get("/version", tags=["System"])
async def get_version():
    """Get VitalDSP version information"""
    return {
        "version": "0.2.1",
        "release_date": "2025-01-27",
        "api_version": "1.0",
        "python_version": "3.9+",
    }

# ==================== Signal Filtering Endpoints ====================

@router.post("/filter/butterworth", tags=["Filtering"])
async def apply_butterworth_filter(request: FilterRequest):
    """
    Apply Butterworth filter to signal.

    Supports: lowpass, highpass, bandpass, bandstop
    """
    try:
        signal = np.array(request.data)
        filter_obj = SignalFiltering(signal, request.sampling_rate)

        if request.low_cutoff and request.high_cutoff:
            # Bandpass
            filtered = filter_obj.bandpass_filter(
                lowcut=request.low_cutoff,
                highcut=request.high_cutoff,
                order=request.order
            )
        elif request.low_cutoff:
            # Highpass
            filtered = filter_obj.highpass_filter(
                cutoff=request.low_cutoff,
                order=request.order
            )
        elif request.high_cutoff:
            # Lowpass
            filtered = filter_obj.lowpass_filter(
                cutoff=request.high_cutoff,
                order=request.order
            )
        else:
            raise HTTPException(status_code=400, detail="Must specify cutoff frequencies")

        return {
            "filtered_data": filtered.tolist(),
            "filter_type": "butterworth",
            "sampling_rate": request.sampling_rate
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/filter/adaptive", tags=["Filtering"])
async def apply_adaptive_filter(request: SignalData):
    """
    Apply adaptive filtering for noise reduction.

    Uses advanced adaptive algorithms for artifact removal.
    """
    try:
        signal = np.array(request.data)
        filter_obj = SignalFiltering(signal, request.sampling_rate)
        filtered = filter_obj.adaptive_filter()

        return {
            "filtered_data": filtered.tolist(),
            "filter_type": "adaptive",
            "sampling_rate": request.sampling_rate
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Feature Extraction Endpoints ====================

@router.post("/features/time-domain", tags=["Feature Extraction"])
async def extract_time_domain_features(request: FeatureExtractionRequest):
    """
    Extract time-domain features from signal.

    Returns statistical features including:
    - Mean, median, std, variance
    - Min, max, range
    - RMS, skewness, kurtosis
    - Zero-crossing rate
    """
    try:
        signal = np.array(request.data)
        features_obj = TimeDomainFeatures(signal, request.sampling_rate)
        features = features_obj.extract_features()

        return {
            "features": features,
            "feature_type": "time_domain",
            "sampling_rate": request.sampling_rate
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/features/frequency-domain", tags=["Feature Extraction"])
async def extract_frequency_domain_features(request: FeatureExtractionRequest):
    """
    Extract frequency-domain features from signal.

    Returns spectral features including:
    - Dominant frequency
    - Power spectral density
    - Spectral entropy
    - Band powers
    """
    try:
        signal = np.array(request.data)
        features_obj = FrequencyDomainFeatures(signal, request.sampling_rate)
        features = features_obj.extract_features()

        return {
            "features": features,
            "feature_type": "frequency_domain",
            "sampling_rate": request.sampling_rate
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/features/hrv", tags=["Feature Extraction"])
async def extract_hrv_features(request: FeatureExtractionRequest):
    """
    Extract Heart Rate Variability (HRV) features.

    Returns HRV metrics including:
    - Time-domain: SDNN, RMSSD, pNN50
    - Frequency-domain: LF, HF, LF/HF ratio
    - Nonlinear: SD1, SD2, sample entropy
    """
    try:
        signal = np.array(request.data)
        hrv_obj = HRVFeatures(signal, request.sampling_rate)
        features = hrv_obj.extract_features()

        return {
            "features": features,
            "feature_type": "hrv",
            "sampling_rate": request.sampling_rate
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Respiratory Analysis Endpoints ====================

@router.post("/respiratory/estimate-rate", tags=["Respiratory Analysis"])
async def estimate_respiratory_rate(request: RespiratoryRequest):
    """
    Estimate respiratory rate from physiological signal.

    Supports multiple methods:
    - FFT-based estimation
    - Wavelet-based estimation
    - Peak detection
    """
    try:
        signal = np.array(request.data)
        resp_obj = RespiratoryAnalysis(signal, request.sampling_rate)

        if request.method == "fft":
            rr = resp_obj.estimate_rr_fft()
        elif request.method == "wavelet":
            rr = resp_obj.estimate_rr_wavelet()
        elif request.method == "peak_detection":
            rr = resp_obj.estimate_rr_peak_detection()
        else:
            rr = resp_obj.estimate_respiratory_rate()

        return {
            "respiratory_rate": rr,
            "method": request.method,
            "signal_type": request.signal_type,
            "unit": "breaths per minute"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Signal Quality Endpoints ====================

@router.post("/quality/assess", tags=["Signal Quality"])
async def assess_signal_quality(request: QualityAssessmentRequest):
    """
    Assess signal quality with comprehensive metrics.

    Returns quality metrics including:
    - SNR (Signal-to-Noise Ratio)
    - Signal quality index
    - Artifact detection
    - Quality score
    """
    try:
        original = np.array(request.original_data)
        processed = np.array(request.processed_data) if request.processed_data else original

        quality_obj = SignalQuality(original, processed)

        metrics = {
            "snr_db": quality_obj.snr(),
            "quality_index": quality_obj.signal_quality_index(),
            "has_artifacts": quality_obj.detect_artifacts(),
            "quality_score": quality_obj.overall_quality_score()
        }

        return {
            "quality_metrics": metrics,
            "sampling_rate": request.sampling_rate
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Transform Endpoints ====================

@router.post("/transform/fft", tags=["Transforms"])
async def apply_fft(request: TransformRequest):
    """
    Apply Fast Fourier Transform (FFT) to signal.

    Returns frequency domain representation.
    """
    try:
        signal = np.array(request.data)
        fft_obj = FourierTransform(signal, request.sampling_rate)
        frequencies, magnitudes = fft_obj.compute_fft()

        return {
            "frequencies": frequencies.tolist(),
            "magnitudes": magnitudes.tolist(),
            "transform_type": "fft",
            "sampling_rate": request.sampling_rate
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/transform/wavelet", tags=["Transforms"])
async def apply_wavelet(request: TransformRequest):
    """
    Apply Wavelet Transform to signal.

    Returns time-frequency representation.
    """
    try:
        signal = np.array(request.data)
        wavelet_obj = WaveletTransform(signal, request.sampling_rate)
        coefficients = wavelet_obj.decompose()

        return {
            "coefficients": [c.tolist() for c in coefficients],
            "transform_type": "wavelet",
            "sampling_rate": request.sampling_rate
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Health Report Endpoints ====================

@router.post("/health-report/generate", tags=["Health Analysis"])
async def generate_health_report(request: SignalData):
    """
    Generate comprehensive health report from physiological signal.

    Analyzes signal and provides:
    - Heart rate analysis
    - HRV metrics
    - Signal quality assessment
    - Clinical insights
    """
    try:
        signal = np.array(request.data)
        report_gen = HealthReportGenerator(signal, request.sampling_rate)
        report = report_gen.generate_report()

        return {
            "report": report,
            "signal_type": request.signal_type,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Batch Processing Endpoints ====================

@router.post("/batch/process", tags=["Batch Processing"])
async def batch_process_signals(signals: List[SignalData]):
    """
    Process multiple signals in batch.

    Applies filtering and feature extraction to multiple signals efficiently.
    """
    try:
        results = []
        for signal_data in signals:
            signal = np.array(signal_data.data)
            filter_obj = SignalFiltering(signal, signal_data.sampling_rate)
            filtered = filter_obj.bandpass_filter(low=0.5, high=40)

            features_obj = TimeDomainFeatures(filtered, signal_data.sampling_rate)
            features = features_obj.extract_features()

            results.append({
                "signal_type": signal_data.signal_type,
                "filtered_data": filtered.tolist(),
                "features": features
            })

        return {
            "processed_count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
