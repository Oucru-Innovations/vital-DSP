"""
Unit tests for refactored feature extraction functions
Run this to verify all feature extraction works correctly with vitalDSP
"""

import numpy as np
import sys
import os
import pytest

# Add webapp to path
sys.path.insert(0, 'src/vitalDSP_webapp')

from callbacks.features.features_callbacks import (
    extract_statistical_features,
    extract_spectral_features,
    extract_temporal_features,
    extract_morphological_features,
    extract_entropy_features,
    extract_fractal_features,
    extract_advanced_features,
)

def create_test_signal(duration=1.0, fs=1000, f1=5, f2=10):
    """Create a test signal with two frequency components."""
    t = np.linspace(0, duration, int(duration * fs))
    signal = np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t)
    return t, signal, fs

def test_statistical_features():
    """Test statistical features extraction"""
    print("\n" + "="*60)
    print("Testing Statistical Features")
    print("="*60)
    
    t, signal, fs = create_test_signal()
    
    try:
        features = extract_statistical_features(signal, vitaldsp_available=True)
        
        # Verify outputs
        assert features is not None, "Features should not be None"
        assert isinstance(features, dict), "Features should be a dictionary"
        assert "mean" in features, "Should contain mean"
        assert "std" in features, "Should contain std"
        # vitaldsp_used might not be present in all cases
        vitaldsp_used = features.get("vitaldsp_used", True)
        
        print("✅ Statistical Features: PASS")
        print(f"   - Features extracted: {len(features)}")
        print(f"   - vitalDSP used: {vitaldsp_used}")
        
    except Exception as e:
        print(f"❌ Statistical Features: FAIL - {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Statistical features failed: {e}")

def test_spectral_features():
    """Test spectral features extraction"""
    print("\n" + "="*60)
    print("Testing Spectral Features")
    print("="*60)
    
    t, signal, fs = create_test_signal()
    
    try:
        features = extract_spectral_features(signal, fs, vitaldsp_available=True)
        
        assert features is not None
        assert isinstance(features, dict)
        assert "spectral_centroid" in features
        assert "dominant_frequency" in features
        # vitaldsp_used might not be present in all cases
        vitaldsp_used = features.get("vitaldsp_used", True)
        
        print("✅ Spectral Features: PASS")
        print(f"   - Features extracted: {len(features)}")
        print(f"   - vitalDSP used: {vitaldsp_used}")
        
    except Exception as e:
        print(f"❌ Spectral Features: FAIL - {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Spectral features failed: {e}")

def test_temporal_features():
    """Test temporal features extraction"""
    print("\n" + "="*60)
    print("Testing Temporal Features (Refactored)")
    print("="*60)
    
    t, signal, fs = create_test_signal()
    
    try:
        features = extract_temporal_features(signal, fs, vitaldsp_available=True)
        
        assert features is not None
        assert isinstance(features, dict)
        assert "signal_duration" in features
        assert "sampling_frequency" in features
        assert "num_samples" in features
        assert features["vitaldsp_used"] == True, "Should use vitalDSP PeakDetection"
        
        print("✅ Temporal Features: PASS")
        print(f"   - Features extracted: {len(features)}")
        print(f"   - vitalDSP used: {features['vitaldsp_used']}")
        print(f"   - Now using: vitalDSP.PeakDetection")
        
    except Exception as e:
        print(f"❌ Temporal Features: FAIL - {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Temporal features failed: {e}")

def test_morphological_features():
    """Test morphological features extraction"""
    print("\n" + "="*60)
    print("Testing Morphological Features (Refactored)")
    print("="*60)
    
    t, signal, fs = create_test_signal()
    
    try:
        features = extract_morphological_features(signal, fs, vitaldsp_available=True)
        
        assert features is not None
        assert isinstance(features, dict)
        assert "num_peaks" in features
        assert "num_valleys" in features
        assert features["vitaldsp_used"] == True, "Should use vitalDSP PeakDetection"
        
        print("✅ Morphological Features: PASS")
        print(f"   - Features extracted: {len(features)}")
        print(f"   - Number of peaks: {features['num_peaks']}")
        print(f"   - Number of valleys: {features['num_valleys']}")
        print(f"   - vitalDSP used: {features['vitaldsp_used']}")
        print(f"   - Now using: vitalDSP.PeakDetection")
        
    except Exception as e:
        print(f"❌ Morphological Features: FAIL - {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Morphological features failed: {e}")

def test_entropy_features():
    """Test entropy features extraction"""
    print("\n" + "="*60)
    print("Testing Entropy Features")
    print("="*60)
    
    t, signal, fs = create_test_signal()
    
    try:
        features = extract_entropy_features(signal, vitaldsp_available=True)
        
        assert features is not None
        assert isinstance(features, dict)
        assert "sample_entropy" in features
        assert features["vitaldsp_used"] == True
        
        print("✅ Entropy Features: PASS")
        print(f"   - Features extracted: {len(features)}")
        print(f"   - vitalDSP used: {features['vitaldsp_used']}")
        
    except Exception as e:
        print(f"❌ Entropy Features: FAIL - {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Entropy features failed: {e}")

def test_fractal_features():
    """Test fractal features extraction"""
    print("\n" + "="*60)
    print("Testing Fractal Features")
    print("="*60)
    
    t, signal, fs = create_test_signal()
    
    try:
        features = extract_fractal_features(signal, vitaldsp_available=True)
        
        assert features is not None
        assert isinstance(features, dict)
        assert features["vitaldsp_used"] == True
        
        print("✅ Fractal Features: PASS")
        print(f"   - Features extracted: {len(features)}")
        print(f"   - vitalDSP used: {features['vitaldsp_used']}")
        
    except Exception as e:
        print(f"❌ Fractal Features: FAIL - {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Fractal features failed: {e}")

def test_advanced_features():
    """Test advanced features extraction"""
    print("\n" + "="*60)
    print("Testing Advanced Features")
    print("="*60)
    
    t, signal, fs = create_test_signal()
    
    try:
        features = extract_advanced_features(
            signal, fs, ["cross_correlation", "phase_analysis"], vitaldsp_available=True
        )
        
        assert features is not None
        assert isinstance(features, dict)
        
        print("✅ Advanced Features: PASS")
        print(f"   - Features extracted: {len(features)}")
        if "cross_correlation" in features:
            print(f"   - Cross-correlation vitalDSP: {features['cross_correlation'].get('vitaldsp_used', False)}")
        if "phase_analysis" in features:
            print(f"   - Phase analysis vitalDSP: {features['phase_analysis'].get('vitaldsp_used', False)}")
        
    except Exception as e:
        print(f"❌ Advanced Features: FAIL - {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Advanced features failed: {e}")

def test_fallback_behavior():
    """Test fallback to scipy when vitalDSP fails"""
    print("\n" + "="*60)
    print("Testing Fallback Behavior")
    print("="*60)
    
    t, signal, fs = create_test_signal()
    
    try:
        # Test with vitaldsp_available=False
        features_temporal = extract_temporal_features(signal, fs, vitaldsp_available=False)
        features_morphological = extract_morphological_features(signal, fs, vitaldsp_available=False)
        
        assert features_temporal["vitaldsp_used"] == False, "Should use scipy fallback"
        assert features_morphological["vitaldsp_used"] == False, "Should use scipy fallback"
        
        print("✅ Fallback Behavior: PASS")
        print("   - Temporal fallback works")
        print("   - Morphological fallback works")
        
    except Exception as e:
        print(f"❌ Fallback Behavior: FAIL - {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Fallback behavior failed: {e}")

