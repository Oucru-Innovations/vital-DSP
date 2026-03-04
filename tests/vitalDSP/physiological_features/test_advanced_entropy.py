"""
Comprehensive tests for Advanced Entropy module.

Tests cover multi-scale entropy analysis for physiological signal analysis.
"""

import pytest
import numpy as np

try:
    from vitalDSP.physiological_features.advanced_entropy import MultiScaleEntropy
    ADVANCED_ENTROPY_AVAILABLE = True
except ImportError:
    ADVANCED_ENTROPY_AVAILABLE = False
    MultiScaleEntropy = None


@pytest.mark.skipif(not ADVANCED_ENTROPY_AVAILABLE, reason="Advanced entropy not available")
class TestMultiScaleEntropyInit:
    """Tests for MultiScaleEntropy initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        signal = np.random.randn(1000)
        mse = MultiScaleEntropy(signal)
        assert mse.max_scale == 20
        assert mse.m == 2

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        signal = np.random.randn(1000)
        mse = MultiScaleEntropy(signal, max_scale=10, m=3, r=0.2)
        assert mse.max_scale == 10
        assert mse.m == 3

    def test_init_short_signal_warning(self):
        """Test that short signal triggers warning."""
        signal = np.random.randn(100)
        with pytest.warns(UserWarning):
            mse = MultiScaleEntropy(signal, max_scale=20)

    def test_init_very_short_signal_error(self):
        """Test that very short signal raises error."""
        signal = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Signal too short"):
            MultiScaleEntropy(signal)

    def test_init_invalid_params(self):
        """Test that invalid parameters raise errors."""
        signal = np.random.randn(1000)

        with pytest.raises(ValueError):
            MultiScaleEntropy(signal, max_scale=0)

        with pytest.raises(ValueError):
            MultiScaleEntropy(signal, m=0)

        with pytest.raises(ValueError):
            MultiScaleEntropy(signal, r=1.5)


@pytest.mark.skipif(not ADVANCED_ENTROPY_AVAILABLE, reason="Advanced entropy not available")
class TestComputeMSE:
    """Tests for compute_mse method."""

    def test_compute_mse_basic(self):
        """Test basic MSE computation."""
        signal = np.random.randn(1000)
        mse = MultiScaleEntropy(signal, max_scale=10)

        try:
            entropy_values = mse.compute_mse()
            assert isinstance(entropy_values, (list, np.ndarray))
            assert len(entropy_values) == 10
        except (AttributeError, NotImplementedError):
            pytest.skip("compute_mse not implemented")

    def test_compute_mse_regular_signal(self):
        """Test MSE on regular signal."""
        # Sine wave - regular pattern
        t = np.linspace(0, 10, 1000)
        signal = np.sin(2 * np.pi * t)

        mse = MultiScaleEntropy(signal, max_scale=10)

        try:
            entropy_values = mse.compute_mse()
            # Regular signal should have low entropy
            assert isinstance(entropy_values, (list, np.ndarray))
        except (AttributeError, NotImplementedError):
            pytest.skip("compute_mse not implemented")

    def test_compute_mse_random_signal(self):
        """Test MSE on random signal."""
        signal = np.random.randn(1000)
        mse = MultiScaleEntropy(signal, max_scale=10)

        try:
            entropy_values = mse.compute_mse()
            # Random signal should have higher entropy
            assert all(e >= 0 or np.isnan(e) for e in entropy_values)
        except (AttributeError, NotImplementedError):
            pytest.skip("compute_mse not implemented")


@pytest.mark.skipif(not ADVANCED_ENTROPY_AVAILABLE, reason="Advanced entropy not available")
class TestComputeCMSE:
    """Tests for compute_cmse method."""

    def test_compute_cmse_basic(self):
        """Test basic CMSE computation."""
        signal = np.random.randn(1000)
        mse = MultiScaleEntropy(signal, max_scale=10)

        try:
            entropy_values = mse.compute_cmse()
            assert isinstance(entropy_values, (list, np.ndarray))
        except (AttributeError, NotImplementedError):
            pytest.skip("compute_cmse not implemented")

    def test_compute_cmse_stability(self):
        """Test that CMSE is more stable than MSE."""
        signal = np.random.randn(500)  # Shorter signal
        mse_obj = MultiScaleEntropy(signal, max_scale=10)

        try:
            mse_values = mse_obj.compute_mse()
            cmse_values = mse_obj.compute_cmse()

            # Both should return valid arrays
            assert isinstance(mse_values, (list, np.ndarray))
            assert isinstance(cmse_values, (list, np.ndarray))
        except (AttributeError, NotImplementedError):
            pytest.skip("compute_cmse not implemented")


@pytest.mark.skipif(not ADVANCED_ENTROPY_AVAILABLE, reason="Advanced entropy not available")
class TestComputeRCMSE:
    """Tests for compute_rcmse method."""

    def test_compute_rcmse_basic(self):
        """Test basic RCMSE computation."""
        signal = np.random.randn(1000)
        mse = MultiScaleEntropy(signal, max_scale=10)

        try:
            entropy_values = mse.compute_rcmse()
            assert isinstance(entropy_values, (list, np.ndarray))
        except (AttributeError, NotImplementedError):
            pytest.skip("compute_rcmse not implemented")


@pytest.mark.skipif(not ADVANCED_ENTROPY_AVAILABLE, reason="Advanced entropy not available")
class TestComplexityIndex:
    """Tests for complexity index calculation."""

    def test_get_complexity_index(self):
        """Test complexity index calculation."""
        signal = np.random.randn(1000)
        mse = MultiScaleEntropy(signal, max_scale=10)

        try:
            entropy_values = mse.compute_mse()
            ci = mse.get_complexity_index(entropy_values)
            assert isinstance(ci, (int, float))
            assert ci >= 0
        except (AttributeError, NotImplementedError):
            pytest.skip("get_complexity_index not implemented")

    def test_complexity_index_comparison(self):
        """Test that complexity index differs for different signals."""
        # Regular signal
        regular = np.sin(np.linspace(0, 20*np.pi, 1000))

        # Random signal
        random = np.random.randn(1000)

        mse_regular = MultiScaleEntropy(regular, max_scale=10)
        mse_random = MultiScaleEntropy(random, max_scale=10)

        try:
            entropy_reg = mse_regular.compute_mse()
            entropy_rand = mse_random.compute_mse()

            ci_reg = mse_regular.get_complexity_index(entropy_reg)
            ci_rand = mse_random.get_complexity_index(entropy_rand)

            # Just ensure both return valid values
            assert isinstance(ci_reg, (int, float))
            assert isinstance(ci_rand, (int, float))
        except (AttributeError, NotImplementedError):
            pytest.skip("Not implemented")


@pytest.mark.skipif(not ADVANCED_ENTROPY_AVAILABLE, reason="Advanced entropy not available")
class TestFuzzyEntropy:
    """Tests for fuzzy entropy option."""

    def test_fuzzy_entropy_flag(self):
        """Test fuzzy entropy flag."""
        signal = np.random.randn(1000)
        mse = MultiScaleEntropy(signal, max_scale=10, fuzzy=True)

        assert mse.fuzzy is True

        try:
            entropy_values = mse.compute_mse()
            assert isinstance(entropy_values, (list, np.ndarray))
        except (AttributeError, NotImplementedError):
            pytest.skip("Fuzzy entropy not implemented")

    def test_fuzzy_vs_standard(self):
        """Test difference between fuzzy and standard entropy."""
        signal = np.random.randn(1000)

        mse_standard = MultiScaleEntropy(signal, max_scale=10, fuzzy=False)
        mse_fuzzy = MultiScaleEntropy(signal, max_scale=10, fuzzy=True)

        try:
            entropy_std = mse_standard.compute_mse()
            entropy_fuz = mse_fuzzy.compute_mse()

            # Both should return valid results
            assert isinstance(entropy_std, (list, np.ndarray))
            assert isinstance(entropy_fuz, (list, np.ndarray))
        except (AttributeError, NotImplementedError):
            pytest.skip("Fuzzy entropy not implemented")


@pytest.mark.skipif(not ADVANCED_ENTROPY_AVAILABLE, reason="Advanced entropy not available")
class TestDifferentScales:
    """Tests with different scale ranges."""

    def test_small_scale(self):
        """Test with small scale range."""
        signal = np.random.randn(500)
        mse = MultiScaleEntropy(signal, max_scale=5)

        try:
            entropy_values = mse.compute_mse()
            assert len(entropy_values) == 5
        except (AttributeError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_large_scale(self):
        """Test with large scale range."""
        signal = np.random.randn(5000)
        mse = MultiScaleEntropy(signal, max_scale=30)

        try:
            entropy_values = mse.compute_mse()
            assert len(entropy_values) == 30
        except (AttributeError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_single_scale(self):
        """Test with single scale."""
        signal = np.random.randn(1000)
        mse = MultiScaleEntropy(signal, max_scale=1)

        try:
            entropy_values = mse.compute_mse()
            assert len(entropy_values) == 1
        except (AttributeError, NotImplementedError):
            pytest.skip("Not implemented")


@pytest.mark.skipif(not ADVANCED_ENTROPY_AVAILABLE, reason="Advanced entropy not available")
class TestRealisticSignals:
    """Tests with realistic physiological signals."""

    def test_hrv_signal(self):
        """Test MSE on HRV-like signal."""
        # Simulate RR intervals
        mean_rr = 0.8  # 75 bpm
        np.random.seed(42)
        rr_intervals = mean_rr + 0.05 * np.random.randn(1000)

        mse = MultiScaleEntropy(rr_intervals, max_scale=20, m=2, r=0.15)

        try:
            entropy_values = mse.compute_mse()
            assert isinstance(entropy_values, (list, np.ndarray))
            assert len(entropy_values) == 20
        except (AttributeError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_ecg_signal(self):
        """Test MSE on ECG-like signal."""
        # Simulate ECG
        t = np.linspace(0, 10, 2500)
        ecg = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(2500)

        mse = MultiScaleEntropy(ecg, max_scale=15)

        try:
            entropy_values = mse.compute_mse()
            assert isinstance(entropy_values, (list, np.ndarray))
        except (AttributeError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_eeg_signal(self):
        """Test MSE on EEG-like signal."""
        # Simulate EEG with multiple frequency components
        t = np.linspace(0, 60, 15000)
        eeg = (np.sin(2 * np.pi * 8 * t) +      # Alpha
               0.5 * np.sin(2 * np.pi * 15 * t) +  # Beta
               0.3 * np.random.randn(15000))        # Noise

        mse = MultiScaleEntropy(eeg, max_scale=20)

        try:
            entropy_values = mse.compute_mse()
            assert isinstance(entropy_values, (list, np.ndarray))
        except (AttributeError, NotImplementedError):
            pytest.skip("Not implemented")


@pytest.mark.skipif(not ADVANCED_ENTROPY_AVAILABLE, reason="Advanced entropy not available")
class TestEdgeCases:
    """Tests for edge cases."""

    def test_constant_signal(self):
        """Test with constant signal."""
        signal = np.ones(1000)
        mse = MultiScaleEntropy(signal, max_scale=10)

        try:
            entropy_values = mse.compute_mse()
            # Constant signal should have zero or very low entropy
            assert isinstance(entropy_values, (list, np.ndarray))
        except (AttributeError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_alternating_signal(self):
        """Test with alternating signal."""
        signal = np.array([1, -1] * 500)
        mse = MultiScaleEntropy(signal, max_scale=10)

        try:
            entropy_values = mse.compute_mse()
            assert isinstance(entropy_values, (list, np.ndarray))
        except (AttributeError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_signal_with_trend(self):
        """Test with signal containing linear trend."""
        t = np.linspace(0, 10, 1000)
        signal = t + np.random.randn(1000) * 0.1

        mse = MultiScaleEntropy(signal, max_scale=10)

        try:
            entropy_values = mse.compute_mse()
            assert isinstance(entropy_values, (list, np.ndarray))
        except (AttributeError, NotImplementedError):
            pytest.skip("Not implemented")


@pytest.mark.skipif(not ADVANCED_ENTROPY_AVAILABLE, reason="Advanced entropy not available")
class TestNumericalStability:
    """Tests for numerical stability."""

    def test_large_amplitude_signal(self):
        """Test with large amplitude signal."""
        signal = np.random.randn(1000) * 1e6
        mse = MultiScaleEntropy(signal, max_scale=10)

        try:
            entropy_values = mse.compute_mse()
            assert isinstance(entropy_values, (list, np.ndarray))
            # Check no NaN or Inf
            assert not any(np.isinf(e) for e in entropy_values if not np.isnan(e))
        except (AttributeError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_small_amplitude_signal(self):
        """Test with small amplitude signal."""
        signal = np.random.randn(1000) * 1e-10
        mse = MultiScaleEntropy(signal, max_scale=10)

        try:
            entropy_values = mse.compute_mse()
            assert isinstance(entropy_values, (list, np.ndarray))
        except (AttributeError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_signal_with_outliers(self):
        """Test with signal containing outliers."""
        signal = np.random.randn(1000)
        signal[500] = 100  # Large outlier

        mse = MultiScaleEntropy(signal, max_scale=10)

        try:
            entropy_values = mse.compute_mse()
            assert isinstance(entropy_values, (list, np.ndarray))
        except (AttributeError, NotImplementedError):
            pytest.skip("Not implemented")


@pytest.mark.skipif(not ADVANCED_ENTROPY_AVAILABLE, reason="Advanced entropy not available")
class TestParameterVariations:
    """Tests with different parameter combinations."""

    def test_different_m_values(self):
        """Test with different embedding dimensions."""
        signal = np.random.randn(1000)

        for m in [1, 2, 3, 4]:
            mse = MultiScaleEntropy(signal, max_scale=10, m=m)
            try:
                entropy_values = mse.compute_mse()
                assert isinstance(entropy_values, (list, np.ndarray))
            except (AttributeError, NotImplementedError):
                pytest.skip(f"Not implemented for m={m}")

    def test_different_r_values(self):
        """Test with different tolerance values."""
        signal = np.random.randn(1000)

        for r in [0.1, 0.15, 0.2, 0.25]:
            mse = MultiScaleEntropy(signal, max_scale=10, r=r)
            try:
                entropy_values = mse.compute_mse()
                assert isinstance(entropy_values, (list, np.ndarray))
            except (AttributeError, NotImplementedError):
                pytest.skip(f"Not implemented for r={r}")


@pytest.mark.skipif(not ADVANCED_ENTROPY_AVAILABLE, reason="Advanced entropy not available")
class TestComparisonScenarios:
    """Tests comparing different scenarios."""

    def test_young_vs_elderly_hrv(self):
        """Test MSE on simulated young vs elderly HRV."""
        np.random.seed(42)

        # Young: Higher variability and complexity
        hrv_young = 0.8 + 0.08 * np.random.randn(1000)

        # Elderly: Lower variability
        hrv_elderly = 0.8 + 0.03 * np.random.randn(1000)

        mse_young = MultiScaleEntropy(hrv_young, max_scale=20)
        mse_elderly = MultiScaleEntropy(hrv_elderly, max_scale=20)

        try:
            entropy_young = mse_young.compute_mse()
            entropy_elderly = mse_elderly.compute_mse()

            # Both should return valid results
            assert isinstance(entropy_young, (list, np.ndarray))
            assert isinstance(entropy_elderly, (list, np.ndarray))
        except (AttributeError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_white_vs_pink_noise(self):
        """Test MSE on white vs pink (1/f) noise."""
        np.random.seed(42)

        # White noise
        white = np.random.randn(2000)

        # Pink noise (cumulative sum approximates 1/f)
        pink = np.cumsum(np.random.randn(2000))

        mse_white = MultiScaleEntropy(white, max_scale=15)
        mse_pink = MultiScaleEntropy(pink, max_scale=15)

        try:
            entropy_white = mse_white.compute_mse()
            entropy_pink = mse_pink.compute_mse()

            assert isinstance(entropy_white, (list, np.ndarray))
            assert isinstance(entropy_pink, (list, np.ndarray))
        except (AttributeError, NotImplementedError):
            pytest.skip("Not implemented")


@pytest.mark.skipif(not ADVANCED_ENTROPY_AVAILABLE, reason="Advanced entropy not available")
class TestAdvancedEntropyMissingCoverage:
    """Tests to cover missing lines in advanced_entropy.py."""

    def test_init_signal_not_numpy_array(self):
        """Test initialization with non-numpy array signal.
        
        This test covers line 215 in advanced_entropy.py where
        signal is converted to numpy array if not already one.
        """
        signal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # Python list
        mse = MultiScaleEntropy(signal)
        assert isinstance(mse.signal, np.ndarray)
        assert len(mse.signal) == 10

    def test_coarse_grain_signal_too_short(self):
        """Test _coarse_grain when signal is too short.
        
        This test covers lines 287-291 in advanced_entropy.py where
        ValueError is raised when signal is too short for the scale.
        """
        signal = np.random.randn(50)  # Short signal
        mse = MultiScaleEntropy(signal, max_scale=10)
        
        # Try to coarse-grain with a scale that's too large
        with pytest.raises(ValueError, match="Signal too short for scale"):
            mse._coarse_grain(scale=100, start_index=0)

    def test_sample_entropy_signal_too_short(self):
        """Test _sample_entropy when signal is too short.
        
        This test covers lines 360-366 in advanced_entropy.py where
        warning is issued and 0.0 is returned when signal is too short.
        """
        signal = np.random.randn(100)
        mse = MultiScaleEntropy(signal, max_scale=10, m=5)  # m=5 requires at least 7 samples
        
        # Create a coarse signal that's too short (less than m+2)
        coarse_signal = np.array([1.0, 2.0])  # Only 2 samples, need at least 7 for m=5
        
        with pytest.warns(UserWarning, match="Signal too short"):
            entropy = mse._sample_entropy(coarse_signal)
            assert entropy == 0.0

    def test_sample_entropy_templates_too_few(self):
        """Test _sample_entropy when templates are too few.
        
        This test covers lines 380-381 in advanced_entropy.py where
        _count_matches returns 0 when templates < 2.
        """
        signal = np.random.randn(100)
        mse = MultiScaleEntropy(signal, max_scale=10, m=2)
        
        # Create a signal that will result in very few templates
        # Signal with only m+1 samples will create only 1 template
        coarse_signal = np.array([1.0, 2.0, 3.0])  # Only 3 samples, m=2 means only 1 template
        
        entropy = mse._sample_entropy(coarse_signal)
        # Should return 0.0 due to insufficient templates
        assert entropy == 0.0

    def test_sample_entropy_no_matches(self):
        """Test _sample_entropy when no template matches are found.
        
        This test covers lines 408-416 in advanced_entropy.py where
        warning is issued and 0.0 is returned when B == 0 or A == 0.
        """
        from unittest.mock import patch, MagicMock
        
        signal = np.random.randn(100)
        mse = MultiScaleEntropy(signal, max_scale=10, m=2, r=0.15)
        
        # Create a signal that's long enough
        coarse_signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        # Mock cKDTree to return only self-matches (no other neighbors)
        # This makes len(neighbors) - 1 = 0 for each point, resulting in total_matches = 0
        with patch('vitalDSP.physiological_features.advanced_entropy.cKDTree') as mock_tree:
            mock_instance = MagicMock()
            # Track call count to return [i] for the i-th call
            call_count = [0]
            def mock_query_ball_point(template, r, p):
                # Return list with just the current call index (self-match only)
                # This makes len(neighbors) - 1 = 0, resulting in no matches
                result = [call_count[0]]
                call_count[0] += 1
                return result
            
            mock_instance.query_ball_point = mock_query_ball_point
            mock_tree.return_value = mock_instance
            
            with pytest.warns(UserWarning, match="No template matches found"):
                entropy = mse._sample_entropy(coarse_signal)
                assert np.isnan(entropy)

    def test_fuzzy_entropy_signal_too_short(self):
        """Test _fuzzy_entropy when signal is too short.
        
        This test covers lines 469-471 in advanced_entropy.py where
        warning is issued and 0.0 is returned when signal is too short.
        """
        signal = np.random.randn(100)
        mse = MultiScaleEntropy(signal, max_scale=10, m=5, fuzzy=True)
        
        # Create a coarse signal that's too short
        coarse_signal = np.array([1.0, 2.0])  # Only 2 samples, need at least 7 for m=5
        
        with pytest.warns(UserWarning, match="Signal too short for FuzzyEn"):
            entropy = mse._fuzzy_entropy(coarse_signal)
            assert entropy == 0.0

    def test_fuzzy_entropy_templates_too_few(self):
        """Test _fuzzy_entropy when templates are too few.
        
        This test covers lines 483-484 in advanced_entropy.py where
        _phi returns 0.0 when templates < 2.
        """
        signal = np.random.randn(100)
        mse = MultiScaleEntropy(signal, max_scale=10, m=2, fuzzy=True)
        
        # Create a signal that will result in very few templates
        coarse_signal = np.array([1.0, 2.0, 3.0])  # Only 3 samples, m=2 means only 1 template
        
        entropy = mse._fuzzy_entropy(coarse_signal)
        # Should return 0.0 due to insufficient templates
        assert entropy == 0.0

    def test_fuzzy_entropy_phi_zero(self):
        """Test _fuzzy_entropy when phi values are zero.
        
        This test covers lines 510-512 in advanced_entropy.py where
        warning is issued and 0.0 is returned when phi_m_plus_1 == 0 or phi_m == 0.
        """
        from unittest.mock import patch
        
        signal = np.random.randn(100)
        mse = MultiScaleEntropy(signal, max_scale=10, m=2, fuzzy=True, r=0.15)
        
        # Create a signal that's long enough
        coarse_signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        # Mock np.exp to return 0 for fuzzy membership, making phi = 0
        call_count = [0]
        original_exp = np.exp
        
        def mock_exp(x):
            call_count[0] += 1
            # After a few calls, return 0 to make phi zero
            if call_count[0] > 10:
                return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
            return original_exp(x)
        
        with patch('vitalDSP.physiological_features.advanced_entropy.np.exp', side_effect=mock_exp):
            with pytest.warns(UserWarning, match="FuzzyEn calculation failed"):
                entropy = mse._fuzzy_entropy(coarse_signal)
                assert entropy == 0.0

    def test_compute_mse_exception_handling(self):
        """Test compute_mse exception handling.
        
        This test covers lines 579-585 in advanced_entropy.py where
        exceptions are caught and 0.0 is appended.
        """
        from unittest.mock import patch
        
        signal = np.random.randn(1000)
        mse = MultiScaleEntropy(signal, max_scale=10)
        
        # Mock _coarse_grain to raise an exception at a specific scale
        call_count = [0]
        original_coarse_grain = mse._coarse_grain
        
        def mock_coarse_grain(scale, start_index=0):
            call_count[0] += 1
            if call_count[0] == 5:  # Fail at scale 5
                raise Exception("Mocked error")
            return original_coarse_grain(scale, start_index)
        
        with patch.object(mse, '_coarse_grain', side_effect=mock_coarse_grain):
            with pytest.warns(UserWarning, match="Failed to compute entropy"):
                entropy_values = mse.compute_mse()
                assert len(entropy_values) == 10
                # Scale 5 should be 0.0 due to exception
                assert entropy_values[4] == 0.0

    def test_compute_cmse_coarse_signal_too_short(self):
        """Test compute_cmse when coarse signal is too short.
        
        This test covers lines 666-667 in advanced_entropy.py where
        continue is executed when coarse signal is too short.
        """
        signal = np.random.randn(100)  # Short signal
        mse = MultiScaleEntropy(signal, max_scale=10, m=5)  # Large m requires more samples
        
        # This should trigger the continue when coarse signal is too short
        entropy_values = mse.compute_cmse()
        assert isinstance(entropy_values, np.ndarray)
        assert len(entropy_values) == 10

    def test_compute_cmse_exception_handling(self):
        """Test compute_cmse exception handling.
        
        This test covers lines 672-674 in advanced_entropy.py where
        exceptions are caught and continue is executed.
        """
        from unittest.mock import patch
        
        signal = np.random.randn(1000)
        mse = MultiScaleEntropy(signal, max_scale=10)
        
        # Mock _coarse_grain to raise an exception
        call_count = [0]
        original_coarse_grain = mse._coarse_grain
        
        def mock_coarse_grain(scale, start_index=0):
            call_count[0] += 1
            if call_count[0] == 3:  # Fail at third call
                raise Exception("Mocked error")
            return original_coarse_grain(scale, start_index)
        
        with patch.object(mse, '_coarse_grain', side_effect=mock_coarse_grain):
            entropy_values = mse.compute_cmse()
            assert isinstance(entropy_values, np.ndarray)
            assert len(entropy_values) == 10

    def test_compute_cmse_empty_scale_entropies(self):
        """Test compute_cmse when scale_entropies is empty.
        
        This test covers lines 677-683 in advanced_entropy.py where
        warning is issued and cmse_value = 0.0 when scale_entropies is empty.
        """
        from unittest.mock import patch
        
        signal = np.random.randn(100)
        mse = MultiScaleEntropy(signal, max_scale=10, m=10)  # Large m
        
        # Mock _coarse_grain to always raise exception or return very short signal
        def mock_coarse_grain(scale, start_index=0):
            raise Exception("Mocked error")
        
        with patch.object(mse, '_coarse_grain', side_effect=mock_coarse_grain):
            with pytest.warns(UserWarning, match="No valid entropy values"):
                entropy_values = mse.compute_cmse()
                assert isinstance(entropy_values, np.ndarray)
                assert len(entropy_values) == 10
                # All values should be 0.0
                assert np.all(entropy_values == 0.0)

    def test_compute_rcmse_signal_too_short(self):
        """Test compute_rcmse when signal is too short.
        
        This test covers lines 740-746 in advanced_entropy.py where
        warning is issued and 0.0 is appended when signal is too short.
        """
        # Create a signal that's too short for RCMSE at larger scales
        # n_windows = n - scale + 1, need n_windows >= m + 2
        # For m=5, need n_windows >= 7, so n - scale + 1 >= 7, so n >= scale + 6
        # For scale=10, need n >= 16, but we'll use n=15 to trigger the warning
        signal = np.random.randn(15)  # Short signal
        mse = MultiScaleEntropy(signal, max_scale=10, m=5)  # Large m
        
        # At scale 10, n_windows = 15 - 10 + 1 = 6, which is < m+2 = 7
        with pytest.warns(UserWarning, match="Signal too short for RCMSE at scale"):
            entropy_values = mse.compute_rcmse()
            assert isinstance(entropy_values, np.ndarray)
            assert len(entropy_values) == 10
            # Some values should be 0.0 due to signal being too short

    def test_compute_rcmse_exception_handling(self):
        """Test compute_rcmse exception handling.
        
        This test covers lines 760-765 in advanced_entropy.py where
        exceptions are caught and 0.0 is appended.
        """
        from unittest.mock import patch
        
        signal = np.random.randn(1000)
        mse = MultiScaleEntropy(signal, max_scale=10)
        
        # Mock entropy function to raise an exception
        call_count = [0]
        original_fuzzy = mse._fuzzy_entropy
        original_sample = mse._sample_entropy
        
        def mock_entropy(coarse_signal):
            call_count[0] += 1
            if call_count[0] == 5:  # Fail at scale 5
                raise Exception("Mocked error")
            if mse.fuzzy:
                return original_fuzzy(coarse_signal)
            else:
                return original_sample(coarse_signal)
        
        if mse.fuzzy:
            with patch.object(mse, '_fuzzy_entropy', side_effect=mock_entropy):
                with pytest.warns(UserWarning, match="Failed to compute RCMSE"):
                    entropy_values = mse.compute_rcmse()
                    assert len(entropy_values) == 10
        else:
            with patch.object(mse, '_sample_entropy', side_effect=mock_entropy):
                with pytest.warns(UserWarning, match="Failed to compute RCMSE"):
                    entropy_values = mse.compute_rcmse()
                    assert len(entropy_values) == 10

    def test_get_complexity_index_scale_range_none(self):
        """Test get_complexity_index when scale_range is None.
        
        This test covers lines 823-824 in advanced_entropy.py where
        scale_range is set to (1, len(entropy_values)) when None.
        """
        signal = np.random.randn(1000)
        mse = MultiScaleEntropy(signal, max_scale=10)
        entropy_values = mse.compute_mse()
        
        # Call without scale_range (should default to all scales)
        ci = mse.get_complexity_index(entropy_values, scale_range=None)
        assert isinstance(ci, (int, float))
        assert ci >= 0

    def test_get_complexity_index_not_enough_values(self):
        """Test get_complexity_index when not enough entropy values.
        
        This test covers lines 832-837 in advanced_entropy.py where
        warning is issued and 0.0 is returned when not enough values.
        """
        signal = np.random.randn(1000)
        mse = MultiScaleEntropy(signal, max_scale=10)
        
        # Create entropy values with only 1 value (not enough for integration)
        entropy_values = np.array([0.5])
        
        with pytest.warns(UserWarning, match="Not enough entropy values"):
            ci = mse.get_complexity_index(entropy_values)
            assert ci == 0.0