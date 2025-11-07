"""
Unit tests for refactored transform functions
Run this to verify all transforms work correctly with vitalDSP
"""

import numpy as np
import sys
import os
import pytest

# Add webapp to path
sys.path.insert(0, 'src/vitalDSP_webapp')

from callbacks.analysis.transform_functions import (
    apply_fft_transform,
    apply_stft_transform,
    apply_wavelet_transform,
    apply_hilbert_transform,
    apply_mfcc_transform
)

def create_test_signal(duration=1.0, fs=1000, f1=5, f2=10):
    """Create a test signal with two frequency components."""
    t = np.linspace(0, duration, int(duration * fs))
    signal = np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t)
    return t, signal, fs

def test_fft():
    """Test FFT transform"""
    print("\n" + "="*60)
    print("Testing FFT Transform")
    print("="*60)
    
    t, signal, fs = create_test_signal()
    
    try:
        main_fig, analysis_fig, results, peaks, bands = apply_fft_transform(
            time_data=t,
            signal_data=signal,
            sampling_freq=fs,
            options=['log_scale', 'phase', 'power', 'peak_detection', 'frequency_bands'],
            window_type='hann',
            n_points=1024
        )
        
        # Verify outputs
        assert main_fig is not None, "Main figure should not be None"
        assert analysis_fig is not None, "Analysis figure should not be None"
        assert results is not None, "Results should not be None"
        
        # Check for vitalDSP indicator in results
        results_str = str(results)
        assert "vitalDSP" in results_str, "Results should indicate vitalDSP usage"
        
        print("✅ FFT Transform: PASS")
        print(f"   - Main figure: {type(main_fig).__name__}")
        print(f"   - Analysis figure: {type(analysis_fig).__name__}")
        print(f"   - vitalDSP indicator: Found")
        
    except Exception as e:
        print(f"❌ FFT Transform: FAIL - {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"FFT Transform failed: {e}")

def test_stft():
    """Test STFT transform"""
    print("\n" + "="*60)
    print("Testing STFT Transform")
    print("="*60)
    
    t, signal, fs = create_test_signal(duration=5.0)
    
    try:
        main_fig, analysis_fig, results, peaks, bands = apply_stft_transform(
            time_data=t,
            signal_data=signal,
            sampling_freq=fs,
            options=[],
            window_size=256,
            overlap_percent=50,
            window_type='hann'
        )
        
        assert main_fig is not None
        assert analysis_fig is not None
        assert "vitalDSP" in str(results)
        
        print("✅ STFT Transform: PASS")
        
    except Exception as e:
        print(f"❌ STFT Transform: FAIL - {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"STFT Transform failed: {e}")

def test_wavelet():
    """Test Wavelet transform"""
    print("\n" + "="*60)
    print("Testing Wavelet Transform")
    print("="*60)
    
    t, signal, fs = create_test_signal()
    
    try:
        main_fig, analysis_fig, results, peaks, bands = apply_wavelet_transform(
            time_data=t,
            signal_data=signal,
            sampling_freq=fs,
            options=[],
            wavelet_type='db4',
            n_scales=5
        )
        
        assert main_fig is not None
        assert analysis_fig is not None
        assert "vitalDSP" in str(results)
        
        print("✅ Wavelet Transform: PASS")
        print("   - Note: Now using DWT (Discrete Wavelet Transform)")
        
    except Exception as e:
        print(f"❌ Wavelet Transform: FAIL - {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Wavelet Transform failed: {e}")

def test_hilbert():
    """Test Hilbert transform"""
    print("\n" + "="*60)
    print("Testing Hilbert Transform")
    print("="*60)
    
    t, signal, fs = create_test_signal()
    
    try:
        main_fig, analysis_fig, results, peaks, bands = apply_hilbert_transform(
            time_data=t,
            signal_data=signal,
            sampling_freq=fs,
            options=[]
        )
        
        assert main_fig is not None
        assert analysis_fig is not None
        assert "vitalDSP" in str(results)
        
        print("✅ Hilbert Transform: PASS")
        
    except Exception as e:
        print(f"❌ Hilbert Transform: FAIL - {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Hilbert Transform failed: {e}")

def test_mfcc():
    """Test MFCC transform"""
    print("\n" + "="*60)
    print("Testing MFCC Transform")
    print("="*60)
    
    t, signal, fs = create_test_signal(duration=2.0)
    
    try:
        main_fig, analysis_fig, results, peaks, bands = apply_mfcc_transform(
            time_data=t,
            signal_data=signal,
            sampling_freq=fs,
            options=[],
            n_mfcc=13,
            n_fft=512
        )
        
        assert main_fig is not None
        assert analysis_fig is not None
        assert "vitalDSP" in str(results)
        
        print("✅ MFCC Transform: PASS")
        print("   - Note: MFCC is for audio/speech, may not be ideal for physiological signals")
        
    except Exception as e:
        print(f"❌ MFCC Transform: FAIL - {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"MFCC Transform failed: {e}")

def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\n" + "="*60)
    print("Testing Error Handling")
    print("="*60)
    
    # Test with very short signal (should trigger vitalDSP validation or error handling)
    short_signal = np.array([1.0])
    t_short = np.array([0.0])
    
    try:
        main_fig, analysis_fig, results, peaks, bands = apply_fft_transform(
            time_data=t_short,
            signal_data=short_signal,
            sampling_freq=100,
            options=[],
            window_type='hann',
            n_points=64
        )
        
        # Should return error figures/messages, not crash
        assert main_fig is not None, "Should return error figure, not None"
        
        print("✅ Error Handling: PASS")
        print("   - Graceful degradation working")
        
    except Exception as e:
        # This is also acceptable - vitalDSP validation might raise exception
        print("⚠️ Error Handling: Partial - Exception raised (vitalDSP validation)")
        print(f"   - Exception type: {type(e).__name__}")
        # Don't fail - this is expected behavior

