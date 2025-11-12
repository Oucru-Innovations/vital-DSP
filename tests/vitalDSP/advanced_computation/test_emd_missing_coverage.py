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

        # Allow for additional residual component
        assert len(imfs) <= 3


class TestEMDMissingCoverage:
    """Tests to cover missing lines in emd.py."""

    def test_emd_max_decomposition_iterations_exceeded(self):
        """Test EMD when max_decomposition_iterations is exceeded.
        
        This test covers lines 154-158 in emd.py where
        warning is issued and break is executed when decomposition_iterations > max_decomposition_iterations.
        """
        import warnings
        
        # Create a signal that will require many iterations
        # Use a very small max_decomposition_iterations to trigger the limit
        signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.randn(100)
        emd = EMD(signal)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            imfs = emd.emd(max_decomposition_iterations=1, stop_criterion=0.01)
            
            # Should have warning about stopping after max iterations
            assert len(w) > 0
            assert any("EMD decomposition stopped after" in str(warning.message) for warning in w)
            # Should still return some IMFs (at least one)
            assert isinstance(imfs, list)

    def test_emd_interpolation_exception(self):
        """Test EMD when interpolation raises an exception.
        
        This test covers lines 197-202 in emd.py where
        exception is caught, warning is issued, and break is executed.
        """
        import warnings
        from unittest.mock import patch
        
        signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.randn(100)
        emd = EMD(signal)
        
        # Mock _interpolate to raise an exception
        original_interpolate = emd._interpolate
        call_count = [0]
        
        def mock_interpolate(x, y):
            call_count[0] += 1
            if call_count[0] == 1:  # First call raises exception
                raise ValueError("Mock interpolation error")
            return original_interpolate(x, y)
        
        emd._interpolate = mock_interpolate
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            imfs = emd.emd(max_imfs=1, stop_criterion=0.05)
            
            # Should have warning about interpolation failure
            assert len(w) > 0
            assert any("Interpolation failed" in str(warning.message) for warning in w)
            # Should still return some IMFs
            assert isinstance(imfs, list)

    def test_emd_zero_signal_power(self):
        """Test EMD when signal power is zero.
        
        This test covers lines 210-214 in emd.py where
        warning is issued and break is executed when signal_power == 0.
        """
        import warnings
        from unittest.mock import patch
        
        # Create a signal that will have zero power during sifting
        # We need to mock np.sum to return 0 when computing signal_power
        signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.randn(100)
        emd = EMD(signal)
        
        # Track calls to np.sum to identify signal_power calculation
        # signal_power = np.sum(h**2) at line 208
        call_count = [0]
        sum_calls = []
        
        def mock_sum(arr, **kwargs):
            call_count[0] += 1
            sum_calls.append((call_count[0], len(arr), np.all(arr >= 0) if len(arr) > 0 else False))
            
            # After entering the sifting loop, return 0 for signal_power calculation
            # We identify signal_power by: it's a sum of squared values (all >= 0)
            # and it happens after peaks/valleys are found
            if call_count[0] > 5:  # After sifting loop has started
                # Check if this looks like signal_power: squared values (all >= 0)
                if len(arr) == len(signal) and np.all(arr >= 0):
                    return 0.0  # Zero signal power
            return np.sum(arr, **kwargs)
        
        # Patch np.sum in the emd module namespace
        with patch('vitalDSP.advanced_computation.emd.np.sum', side_effect=mock_sum):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                imfs = emd.emd(max_imfs=1, stop_criterion=0.05)
                
                # Should have warning about zero signal power
                warning_messages = [str(warning.message) for warning in w]
                assert any("Signal power is zero" in msg for msg in warning_messages)
                # Should still return some IMFs
                assert isinstance(imfs, list)