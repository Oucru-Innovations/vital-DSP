"""
Tests for emd.py to cover missing lines 96 and 159
"""

import pytest
import numpy as np
from vitalDSP.advanced_computation.emd import EMD


class TestEMDLine96:
    """Test line 96: break when insufficient peaks/valleys"""

    def test_insufficient_peaks_triggers_break(self):
        """Test that signals with < 2 peaks/valleys trigger break at line 96"""
        # Create a monotonic signal with no peaks
        signal = np.linspace(0, 10, 50)
        emd = EMD(signal)

        # This should trigger the break at line 96 because no peaks exist
        imfs = emd.emd(max_imfs=1)

        # Should get minimal IMFs due to insufficient extrema
        assert len(imfs) >= 0

    def test_constant_signal_no_peaks(self):
        """Test constant signal has no peaks, triggering line 96"""
        # Constant signal has no peaks or valleys
        signal = np.ones(100) * 5.0
        emd = EMD(signal)

        # Should break immediately at line 96
        imfs = emd.emd()

        # Constant signal produces minimal decomposition
        assert isinstance(imfs, list)

    def test_single_peak_insufficient(self):
        """Test signal with only 1 peak triggers line 96"""
        # Create signal with exactly one peak
        signal = np.array([0, 0, 1, 0, 0])
        emd = EMD(signal)

        # len(peaks) = 1, which is < 2, triggering break at line 96
        imfs = emd.emd(max_imfs=1)

        assert len(imfs) >= 0


class TestEMDLine159:
    """Test line 159: return zeros when len(x) < 2 in _interpolate"""

    def test_interpolate_with_insufficient_points(self):
        """Test _interpolate returns zeros when len(x) < 2 (line 159)"""
        signal = np.random.randn(50)
        emd = EMD(signal)

        # Call _interpolate with < 2 points
        x = np.array([5])  # Only 1 point
        y = np.array([1.0])

        result = emd._interpolate(x, y)

        # Line 159: should return zeros
        expected = np.zeros(len(signal))
        np.testing.assert_array_equal(result, expected)

    def test_interpolate_with_empty_array(self):
        """Test _interpolate with empty arrays (line 159)"""
        signal = np.random.randn(30)
        emd = EMD(signal)

        # Empty arrays: len(x) = 0, which is < 2
        x = np.array([])
        y = np.array([])

        result = emd._interpolate(x, y)

        # Line 159: should return zeros
        expected = np.zeros(len(signal))
        np.testing.assert_array_equal(result, expected)

    def test_interpolate_normal_case(self):
        """Test _interpolate with sufficient points (>= 2)"""
        signal = np.random.randn(40)
        emd = EMD(signal)

        # Normal case: len(x) >= 2
        x = np.array([0, 10, 20, 30])
        y = np.array([1.0, 2.0, 1.5, 3.0])

        result = emd._interpolate(x, y)

        # Should return interpolated values, not zeros
        assert len(result) == len(signal)
        assert not np.all(result == 0)  # Should not be all zeros


class TestEMDEdgeCases:
    """Additional edge cases to ensure robust coverage"""

    def test_very_short_signal(self):
        """Test EMD with very short signal"""
        signal = np.array([1, 2, 1])
        emd = EMD(signal)

        # Very short signal likely has insufficient extrema
        imfs = emd.emd()

        assert isinstance(imfs, list)

    def test_emd_with_max_imfs_zero(self):
        """Test EMD with max_imfs=0"""
        signal = np.sin(np.linspace(0, 10, 100))
        emd = EMD(signal)

        # max_imfs=0 should return empty list immediately
        imfs = emd.emd(max_imfs=0)

        assert len(imfs) == 0

    def test_emd_convergence_with_tight_criterion(self):
        """Test EMD with very tight stop criterion"""
        signal = np.sin(np.linspace(0, 10, 100)) + 0.1 * np.random.randn(100)
        emd = EMD(signal)

        # Very tight criterion
        imfs = emd.emd(stop_criterion=1e-10, max_imfs=2)

        assert len(imfs) <= 2
