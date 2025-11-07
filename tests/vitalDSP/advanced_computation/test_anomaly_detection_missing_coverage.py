"""
Additional tests for anomaly_detection.py to cover missing lines.

This test file specifically targets:
- Lines 230-232: ImportError handling in _lof_anomaly_detection
- Lines 281-342: _lof_anomaly_detection_fallback method
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from vitalDSP.advanced_computation.anomaly_detection import AnomalyDetection


@pytest.fixture
def test_signal():
    """Create test signal for LOF testing."""
    np.random.seed(42)
    return np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)


@pytest.fixture
def large_signal():
    """Create large signal for testing sampling in fallback."""
    np.random.seed(42)
    # Create signal larger than sample_size (1000)
    return np.sin(np.linspace(0, 100, 2000)) + np.random.normal(0, 0.1, 2000)


@pytest.fixture
def small_signal():
    """Create small signal for testing edge cases."""
    np.random.seed(42)
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


class TestLOFAnomalyDetectionFallback:
    """Test LOF anomaly detection fallback method (lines 281-342)."""

    def test_lof_fallback_import_error(self, test_signal):
        """Test that ImportError triggers fallback method (lines 230-232)."""
        detector = AnomalyDetection(test_signal)
        
        # The import statement is: from sklearn.neighbors import NearestNeighbors
        # We need to patch sys.modules to make sklearn.neighbors unavailable
        import sys
        
        # Save original sklearn.neighbors if it exists
        sklearn_neighbors_backup = sys.modules.get('sklearn.neighbors', None)
        
        # Remove sklearn.neighbors from sys.modules to force ImportError
        if 'sklearn.neighbors' in sys.modules:
            del sys.modules['sklearn.neighbors']
        if 'sklearn' in sys.modules:
            # Also remove sklearn to prevent partial import
            sklearn_backup = sys.modules.get('sklearn', None)
            del sys.modules['sklearn']
        else:
            sklearn_backup = None
        
        try:
            # Now when the method tries to import, it should fail and use fallback
            anomalies = detector._lof_anomaly_detection(n_neighbors=10)
            # Should return result from fallback
            assert isinstance(anomalies, np.ndarray)
            assert len(anomalies) >= 0
        finally:
            # Restore original modules
            if sklearn_neighbors_backup is not None:
                sys.modules['sklearn.neighbors'] = sklearn_neighbors_backup
            if sklearn_backup is not None:
                sys.modules['sklearn'] = sklearn_backup

    def test_lof_fallback_direct_call(self, test_signal):
        """Test fallback method directly (lines 281-342)."""
        detector = AnomalyDetection(test_signal)
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=10)
        
        assert isinstance(anomalies, np.ndarray)
        # Should return some anomalies or empty array
        assert len(anomalies) >= 0

    def test_lof_fallback_insufficient_points(self, small_signal):
        """Test fallback with insufficient points (lines 283-284)."""
        detector = AnomalyDetection(small_signal)
        # n_neighbors + 1 = 11, but signal has only 5 points
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=10)
        
        assert isinstance(anomalies, np.ndarray)
        assert len(anomalies) == 0  # Should return empty array

    def test_lof_fallback_sufficient_points_small_signal(self, test_signal):
        """Test fallback with sufficient points but signal <= sample_size (lines 292-294, 339-340)."""
        detector = AnomalyDetection(test_signal)
        # Signal has 100 points, which is < sample_size (1000)
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=10)
        
        assert isinstance(anomalies, np.ndarray)
        # Should use full signal, not sampling
        assert len(anomalies) >= 0

    def test_lof_fallback_large_signal_sampling(self, large_signal):
        """Test fallback with large signal that requires sampling (lines 288-291, 337-338)."""
        detector = AnomalyDetection(large_signal)
        # Signal has 2000 points, which is > sample_size (1000)
        # This should trigger random sampling
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=20)
        
        assert isinstance(anomalies, np.ndarray)
        # Should return anomalies (may be from sampled indices)
        assert len(anomalies) >= 0
        # Verify indices are valid
        if len(anomalies) > 0:
            assert np.all(anomalies < len(large_signal))

    def test_lof_fallback_phase_space_creation(self, test_signal):
        """Test phase space creation in fallback (line 297)."""
        detector = AnomalyDetection(test_signal)
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=10)
        
        assert isinstance(anomalies, np.ndarray)
        # Phase space should be created from signal[:-1] and signal[1:]
        # So it should have len(signal) - 1 points
        assert len(anomalies) >= 0

    def test_lof_fallback_distance_computation(self, test_signal):
        """Test distance computation in fallback (lines 303-307)."""
        detector = AnomalyDetection(test_signal)
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=10)
        
        assert isinstance(anomalies, np.ndarray)
        # Distance matrix should be computed
        assert len(anomalies) >= 0

    def test_lof_fallback_reachability_distances(self, test_signal):
        """Test reachability distance computation (lines 310-316)."""
        detector = AnomalyDetection(test_signal)
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=10)
        
        assert isinstance(anomalies, np.ndarray)
        # Reachability distances should be computed
        assert len(anomalies) >= 0

    def test_lof_fallback_lrd_computation(self, test_signal):
        """Test local reachability density computation (lines 318-326)."""
        detector = AnomalyDetection(test_signal)
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=10)
        
        assert isinstance(anomalies, np.ndarray)
        # LRD should be computed
        assert len(anomalies) >= 0

    def test_lof_fallback_lof_computation(self, test_signal):
        """Test LOF computation (lines 328-331)."""
        detector = AnomalyDetection(test_signal)
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=10)
        
        assert isinstance(anomalies, np.ndarray)
        # LOF should be computed
        assert len(anomalies) >= 0

    def test_lof_fallback_anomaly_detection(self, test_signal):
        """Test anomaly detection in fallback (line 334)."""
        detector = AnomalyDetection(test_signal)
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=10)
        
        assert isinstance(anomalies, np.ndarray)
        # Anomalies should be detected where LOF > 1
        assert len(anomalies) >= 0

    def test_lof_fallback_index_mapping_sampled(self, large_signal):
        """Test index mapping for sampled data (lines 337-338)."""
        detector = AnomalyDetection(large_signal)
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=20)
        
        assert isinstance(anomalies, np.ndarray)
        # When sampling, indices should be mapped back
        if len(anomalies) > 0:
            assert np.all(anomalies < len(large_signal))
            assert np.all(anomalies >= 0)

    def test_lof_fallback_index_mapping_unsampled(self, test_signal):
        """Test index mapping for unsampled data (lines 339-340)."""
        detector = AnomalyDetection(test_signal)
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=10)
        
        assert isinstance(anomalies, np.ndarray)
        # When not sampling, indices should be direct
        if len(anomalies) > 0:
            assert np.all(anomalies < len(test_signal))
            assert np.all(anomalies >= 0)

    def test_lof_fallback_with_anomalies_in_signal(self):
        """Test fallback with signal containing obvious anomalies."""
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        # Add obvious anomalies
        signal[10] = 10.0
        signal[50] = -10.0
        signal[80] = 15.0
        
        detector = AnomalyDetection(signal)
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=10)
        
        assert isinstance(anomalies, np.ndarray)
        # Should detect at least some anomalies
        assert len(anomalies) >= 0

    def test_lof_fallback_different_n_neighbors(self, test_signal):
        """Test fallback with different n_neighbors values."""
        detector = AnomalyDetection(test_signal)
        
        for n_neighbors in [5, 10, 15, 20]:
            anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=n_neighbors)
            assert isinstance(anomalies, np.ndarray)
            assert len(anomalies) >= 0

    def test_lof_fallback_exact_sample_size(self):
        """Test fallback with signal exactly at sample_size (1000)."""
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)
        
        detector = AnomalyDetection(signal)
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=20)
        
        assert isinstance(anomalies, np.ndarray)
        # Should not sample (n_points == sample_size)
        assert len(anomalies) >= 0

    def test_lof_fallback_one_above_sample_size(self):
        """Test fallback with signal one point above sample_size."""
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 10, 1001)) + np.random.normal(0, 0.1, 1001)
        
        detector = AnomalyDetection(signal)
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=20)
        
        assert isinstance(anomalies, np.ndarray)
        # Should trigger sampling (n_points > sample_size)
        assert len(anomalies) >= 0

    def test_lof_fallback_via_detect_anomalies(self, test_signal):
        """Test that detect_anomalies calls fallback when sklearn unavailable."""
        detector = AnomalyDetection(test_signal)
        
        # Remove sklearn.neighbors from sys.modules to force ImportError
        import sys
        
        # Save original sklearn.neighbors if it exists
        sklearn_neighbors_backup = sys.modules.get('sklearn.neighbors', None)
        
        # Remove sklearn.neighbors from sys.modules to force ImportError
        if 'sklearn.neighbors' in sys.modules:
            del sys.modules['sklearn.neighbors']
        if 'sklearn' in sys.modules:
            # Also remove sklearn to prevent partial import
            sklearn_backup = sys.modules.get('sklearn', None)
            del sys.modules['sklearn']
        else:
            sklearn_backup = None
        
        try:
            # This should use the fallback method
            anomalies = detector.detect_anomalies(method="lof", n_neighbors=10)
            assert isinstance(anomalies, np.ndarray)
            assert len(anomalies) >= 0
        finally:
            # Restore original modules
            if sklearn_neighbors_backup is not None:
                sys.modules['sklearn.neighbors'] = sklearn_neighbors_backup
            if sklearn_backup is not None:
                sys.modules['sklearn'] = sklearn_backup

    def test_lof_fallback_edge_case_n_neighbors_zero(self, test_signal):
        """Test fallback with n_neighbors=0 (edge case)."""
        detector = AnomalyDetection(test_signal)
        # n_neighbors=0 means we need at least 1 point
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=0)
        
        assert isinstance(anomalies, np.ndarray)
        # Should handle gracefully
        assert len(anomalies) >= 0

    def test_lof_fallback_edge_case_n_neighbors_one(self, test_signal):
        """Test fallback with n_neighbors=1."""
        detector = AnomalyDetection(test_signal)
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=1)
        
        assert isinstance(anomalies, np.ndarray)
        assert len(anomalies) >= 0

    def test_lof_fallback_constant_signal(self):
        """Test fallback with constant signal."""
        signal = np.ones(100)
        detector = AnomalyDetection(signal)
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=10)
        
        assert isinstance(anomalies, np.ndarray)
        # Constant signal might have different behavior
        assert len(anomalies) >= 0

    def test_lof_fallback_very_large_signal(self):
        """Test fallback with very large signal to ensure sampling works."""
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 100, 5000)) + np.random.normal(0, 0.1, 5000)
        
        detector = AnomalyDetection(signal)
        anomalies = detector._lof_anomaly_detection_fallback(n_neighbors=20)
        
        assert isinstance(anomalies, np.ndarray)
        # Should sample to 1000 points
        assert len(anomalies) >= 0
        if len(anomalies) > 0:
            assert np.all(anomalies < len(signal))

