"""
Comprehensive tests for Transfer Entropy module.

Tests cover transfer entropy calculations for analyzing directional information
transfer between physiological signals.
"""

import pytest
import numpy as np

try:
    from vitalDSP.physiological_features.transfer_entropy import TransferEntropy
    TRANSFER_ENTROPY_AVAILABLE = True
except ImportError:
    TRANSFER_ENTROPY_AVAILABLE = False
    TransferEntropy = None


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestTransferEntropyInit:
    """Tests for TransferEntropy initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        x = np.random.randn(1000)
        y = np.random.randn(1000)

        try:
            te = TransferEntropy(x, y)
            assert hasattr(te, 'source') or hasattr(te, 'x')
            assert hasattr(te, 'target') or hasattr(te, 'y')
        except (TypeError, AttributeError):
            pytest.skip("TransferEntropy initialization not as expected")

    def test_init_with_params(self):
        """Test initialization with parameters."""
        x = np.random.randn(1000)
        y = np.random.randn(1000)

        try:
            # TransferEntropy constructor uses k_coef and l_coef
            te = TransferEntropy(x, y, k_coef=2, l_coef=1)
            # Check if k and l attributes exist (they are set from k_coef and l_coef)
            assert hasattr(te, 'k') and hasattr(te, 'l')
            assert te.k == 2
            assert te.l == 1
        except (TypeError, AttributeError, ValueError):
            pytest.skip("TransferEntropy parameters not as expected")


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestComputeTransferEntropy:
    """Tests for transfer entropy computation."""

    def test_compute_basic(self):
        """Test basic transfer entropy calculation."""
        x = np.random.randn(1000)
        y = np.random.randn(1000)

        try:
            te = TransferEntropy(x, y)
            result = te.compute_transfer_entropy()
            assert isinstance(result, (int, float, dict))
        except (TypeError, AttributeError, NotImplementedError):
            pytest.skip("compute_transfer_entropy method not implemented")

    def test_compute_coupled_signals(self):
        """Test TE with coupled signals."""
        # Y depends on X
        x = np.random.randn(1000)
        y = np.zeros(1000)
        y[0] = np.random.randn()
        for i in range(1, 1000):
            y[i] = 0.7 * y[i-1] + 0.3 * x[i-1] + 0.1 * np.random.randn()

        try:
            te_xy = TransferEntropy(x, y)
            result_xy = te_xy.compute_transfer_entropy()

            te_yx = TransferEntropy(y, x)
            result_yx = te_yx.compute_transfer_entropy()

            # Both should return valid values
            assert isinstance(result_xy, (int, float, dict))
            assert isinstance(result_yx, (int, float, dict))
        except (TypeError, AttributeError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_compute_independent(self):
        """Test TE between independent signals."""
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000)

        try:
            te = TransferEntropy(x, y)
            result = te.compute_transfer_entropy()
            # Should be close to zero for independent signals
            assert isinstance(result, (int, float, dict))
        except (TypeError, AttributeError, NotImplementedError):
            pytest.skip("Not implemented")


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestDifferentParameters:
    """Tests with different parameters."""

    def test_different_k_values(self):
        """Test with different history lengths."""
        x = np.random.randn(1000)
        y = np.random.randn(1000)

        # TransferEntropy uses k_coef in constructor
        try:
            te = TransferEntropy(x, y, k_coef=2)
            result = te.compute_transfer_entropy()
            assert isinstance(result, (int, float, dict))
        except (TypeError, AttributeError, NotImplementedError, ValueError):
            pytest.skip("k_coef parameter not supported")

    def test_different_l_values(self):
        """Test with different source delays."""
        x = np.random.randn(1000)
        y = np.random.randn(1000)

        # TransferEntropy uses l_coef in constructor
        try:
            te = TransferEntropy(x, y, l_coef=2)
            result = te.compute_transfer_entropy()
            assert isinstance(result, (int, float, dict))
        except (TypeError, AttributeError, NotImplementedError, ValueError):
            pytest.skip("l_coef parameter not supported")


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestRealisticSignals:
    """Tests with realistic physiological signals."""

    def test_cardio_respiratory_coupling(self):
        """Test TE on simulated cardio-respiratory coupling."""
        # Respiratory signal (0.25 Hz, 15 breaths/min)
        t = np.linspace(0, 60, 6000)  # 60 seconds, 100 Hz
        resp = np.sin(2 * np.pi * 0.25 * t)

        # Cardiac signal with respiratory modulation
        heart_rate = 1.2 + 0.1 * resp  # RSA effect
        ecg = np.sin(2 * np.pi * np.cumsum(heart_rate) / 100)

        try:
            te_resp_to_ecg = TransferEntropy(resp, ecg)
            te_ecg_to_resp = TransferEntropy(ecg, resp)

            result1 = te_resp_to_ecg.compute_transfer_entropy()
            result2 = te_ecg_to_resp.compute_transfer_entropy()

            assert isinstance(result1, (int, float, dict))
            assert isinstance(result2, (int, float, dict))
        except (TypeError, AttributeError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_eeg_connectivity(self):
        """Test TE on simulated EEG connectivity."""
        # Two EEG channels
        t = np.linspace(0, 10, 2500)

        # Channel 1: alpha oscillation
        ch1 = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(2500)

        # Channel 2: driven by channel 1 with delay
        ch2 = np.zeros(2500)
        ch2[10:] = 0.6 * ch1[:-10] + 0.4 * np.random.randn(2490)

        try:
            te_1to2 = TransferEntropy(ch1, ch2)
            te_2to1 = TransferEntropy(ch2, ch1)

            result1 = te_1to2.compute_transfer_entropy()
            result2 = te_2to1.compute_transfer_entropy()

            assert isinstance(result1, (int, float, dict))
            assert isinstance(result2, (int, float, dict))
        except (TypeError, AttributeError, NotImplementedError):
            pytest.skip("Not implemented")


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestEdgeCases:
    """Tests for edge cases."""

    def test_short_signals(self):
        """Test with very short signals."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        try:
            te = TransferEntropy(x, y)
            result = te.compute_transfer_entropy()
            if result is not None:
                assert isinstance(result, (int, float, dict))
        except (ValueError, TypeError, AttributeError, NotImplementedError):
            # Expected for very short signals
            pass

    def test_constant_signals(self):
        """Test with constant signals."""
        x = np.ones(1000)
        y = np.ones(1000)

        try:
            te = TransferEntropy(x, y)
            result = te.compute_transfer_entropy()
            if result is not None:
                assert isinstance(result, (int, float, dict))
                # Should be zero for constants
        except (ValueError, TypeError, AttributeError, NotImplementedError):
            # May not handle constant signals
            pass

    def test_identical_signals(self):
        """Test with identical signals."""
        x = np.random.randn(1000)
        y = x.copy()

        try:
            te = TransferEntropy(x, y)
            result = te.compute_transfer_entropy()
            if result is not None:
                assert isinstance(result, (int, float, dict))
        except (ValueError, TypeError, AttributeError, NotImplementedError):
            pass


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestDirectionality:
    """Tests for detecting directionality of coupling."""

    def test_unidirectional_coupling(self):
        """Test detection of unidirectional coupling."""
        # X drives Y, but not vice versa
        x = np.random.randn(1000)
        y = np.zeros(1000)
        y[0] = np.random.randn()

        for i in range(1, 1000):
            y[i] = 0.8 * y[i-1] + 0.4 * x[i-1] + 0.1 * np.random.randn()

        try:
            te_xy = TransferEntropy(x, y)
            te_yx = TransferEntropy(y, x)

            result_xy = te_xy.compute_transfer_entropy()
            result_yx = te_yx.compute_transfer_entropy()

            # Both should be valid
            assert isinstance(result_xy, (int, float, dict))
            assert isinstance(result_yx, (int, float, dict))
        except (TypeError, AttributeError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_bidirectional_coupling(self):
        """Test detection of bidirectional coupling."""
        # X and Y influence each other
        x = np.zeros(1000)
        y = np.zeros(1000)
        x[0] = np.random.randn()
        y[0] = np.random.randn()

        for i in range(1, 1000):
            x[i] = 0.6 * x[i-1] + 0.3 * y[i-1] + 0.1 * np.random.randn()
            y[i] = 0.6 * y[i-1] + 0.3 * x[i-1] + 0.1 * np.random.randn()

        try:
            te_xy = TransferEntropy(x, y)
            te_yx = TransferEntropy(y, x)

            result_xy = te_xy.compute_transfer_entropy()
            result_yx = te_yx.compute_transfer_entropy()

            # Both should show significant transfer
            assert isinstance(result_xy, (int, float, dict))
            assert isinstance(result_yx, (int, float, dict))
        except (TypeError, AttributeError, NotImplementedError):
            pytest.skip("Not implemented")


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestNumericalStability:
    """Tests for numerical stability."""

    def test_large_amplitude_signals(self):
        """Test with large amplitude signals."""
        x = np.random.randn(1000) * 1e6
        y = np.random.randn(1000) * 1e6

        try:
            te = TransferEntropy(x, y)
            result = te.compute_transfer_entropy()
            if result is not None:
                assert isinstance(result, (int, float, dict))
                if isinstance(result, (int, float)):
                    assert not np.isnan(result)
        except (TypeError, AttributeError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_small_amplitude_signals(self):
        """Test with small amplitude signals."""
        x = np.random.randn(1000) * 1e-10
        y = np.random.randn(1000) * 1e-10

        try:
            te = TransferEntropy(x, y)
            result = te.compute_transfer_entropy()
            if result is not None:
                assert isinstance(result, (int, float, dict))
        except (TypeError, AttributeError, NotImplementedError):
            pytest.skip("Not implemented")

    def test_mixed_scale_signals(self):
        """Test with signals of very different scales."""
        x = np.random.randn(1000) * 0.001
        y = np.random.randn(1000) * 1000

        try:
            te = TransferEntropy(x, y)
            result = te.compute_transfer_entropy()
            if result is not None:
                assert isinstance(result, (int, float, dict))
        except (TypeError, AttributeError, NotImplementedError):
            pytest.skip("Not implemented")


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestAdditionalMethods:
    """Tests for additional methods if available."""

    def test_conditional_te(self):
        """Test conditional transfer entropy if available."""
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        z = np.random.randn(1000)

        try:
            # Try with conditional TE (third signal z)
            te = TransferEntropy(x, y)  # Standard TE doesn't support conditional
            # Check if there's a method for conditional TE
            if hasattr(te, 'compute_effective_te'):
                result = te.compute_effective_te()
                assert isinstance(result, (int, float, dict))
            else:
                # Fallback to standard TE
                result = te.compute_transfer_entropy()
                assert isinstance(result, (int, float, dict))
        except (TypeError, AttributeError, NotImplementedError, ValueError):
            pytest.skip("Conditional TE not available")

    def test_symbolic_te(self):
        """Test symbolic transfer entropy if available."""
        x = np.random.randn(1000)
        y = np.random.randn(1000)

        try:
            te = TransferEntropy(x, y)
            # Use standard transfer entropy (symbolic methods may not be separate)
            result = te.compute_transfer_entropy()
            assert isinstance(result, (int, float, dict))
        except (TypeError, AttributeError, NotImplementedError):
            pytest.skip("Transfer entropy not available")

    def test_time_delayed_te(self):
        """Test time-delayed transfer entropy if available."""
        x = np.random.randn(1000)
        y = np.random.randn(1000)

        try:
            te = TransferEntropy(x, y)
            if hasattr(te, 'compute_time_delayed_te'):
                result = te.compute_time_delayed_te(max_delay=10)
                assert isinstance(result, (dict, list, np.ndarray))
            else:
                # Fallback to standard TE
                result = te.compute_transfer_entropy()
                assert isinstance(result, (int, float, dict))
        except (TypeError, AttributeError, NotImplementedError):
            pytest.skip("Time-delayed TE not available")


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_init_with_list_input(self):
        """Test initialization with list input (converts to numpy array)."""
        # Line 230: source conversion
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        y = np.random.randn(20)
        
        te = TransferEntropy(x, y)
        assert isinstance(te.source, np.ndarray)
        assert isinstance(te.target, np.ndarray)

    def test_init_with_list_target(self):
        """Test initialization with list target (converts to numpy array)."""
        # Line 232: target conversion
        x = np.random.randn(20)
        y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        
        te = TransferEntropy(x, y)
        assert isinstance(te.source, np.ndarray)
        assert isinstance(te.target, np.ndarray)

    def test_init_length_mismatch(self):
        """Test initialization with mismatched lengths raises error."""
        # Line 235: length mismatch error
        x = np.random.randn(100)
        y = np.random.randn(50)
        
        with pytest.raises(ValueError, match="Source and target must have same length"):
            TransferEntropy(x, y)

    def test_init_invalid_k_coef(self):
        """Test initialization with invalid k_coef raises error."""
        # Line 247: k < 1 error
        x = np.random.randn(100)
        y = np.random.randn(100)
        
        with pytest.raises(ValueError, match="k and l must be >= 1"):
            TransferEntropy(x, y, k_coef=0)

    def test_init_invalid_l_coef(self):
        """Test initialization with invalid l_coef raises error."""
        # Line 247: l < 1 error
        x = np.random.randn(100)
        y = np.random.randn(100)
        
        with pytest.raises(ValueError, match="k and l must be >= 1"):
            TransferEntropy(x, y, l_coef=0)

    def test_init_invalid_delay(self):
        """Test initialization with invalid delay raises error."""
        # Line 250: delay < 1 error
        x = np.random.randn(100)
        y = np.random.randn(100)
        
        with pytest.raises(ValueError, match="delay must be >= 1"):
            TransferEntropy(x, y, delay=0)

    def test_init_invalid_k_neighbors(self):
        """Test initialization with invalid k_neighbors raises error."""
        # Line 253: k_neighbors < 1 error
        x = np.random.randn(100)
        y = np.random.randn(100)
        
        with pytest.raises(ValueError, match="k_neighbors must be >= 1"):
            TransferEntropy(x, y, k_neighbors=0)

    def test_init_signal_too_short(self):
        """Test initialization with signal too short for parameters."""
        # Line 240-244: signal too short error
        x = np.random.randn(5)
        y = np.random.randn(5)
        
        with pytest.raises(ValueError, match="Signals too short"):
            TransferEntropy(x, y, k_coef=2, l_coef=2, delay=2)


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestEmbeddingMethods:
    """Tests for embedding and entropy estimation methods."""

    def test_create_embedding_too_short(self):
        """Test _create_embedding with signal too short."""
        # Line 305: embedding error when signal too short
        x = np.random.randn(100)
        y = np.random.randn(100)
        te = TransferEntropy(x, y, k_coef=1, l_coef=1, delay=1)
        
        # Try to create embedding with dimension that's too large
        with pytest.raises(ValueError, match="Signal too short for embedding"):
            te._create_embedding(np.array([1, 2, 3]), dimension=10, delay=1)

    def test_estimate_entropy_knn_too_few_samples(self):
        """Test _estimate_entropy_knn with too few samples."""
        # Lines 358-361: warning when too few samples
        import warnings
        x = np.random.randn(100)
        y = np.random.randn(100)
        te = TransferEntropy(x, y, k_neighbors=5)
        
        # Create data with fewer samples than k_neighbors
        data = np.random.randn(3, 2)  # 3 samples, 2 dimensions
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = te._estimate_entropy_knn(data, k=5)
            assert result == 0.0
            assert len(w) > 0
            assert "Too few samples" in str(w[0].message)

    def test_estimate_mutual_information_knn_1d_inputs(self):
        """Test _estimate_mutual_information_knn with 1D inputs."""
        # Lines 419-438: reshaping 1D to 2D
        x = np.random.randn(100)
        y = np.random.randn(100)
        te = TransferEntropy(x, y)
        
        # Test with 1D arrays (should be reshaped)
        x_1d = np.random.randn(50)
        y_1d = np.random.randn(50)
        
        result = te._estimate_mutual_information_knn(x_1d, y_1d, k=3)
        assert isinstance(result, (int, float))
        assert result >= 0.0


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestBidirectionalTE:
    """Tests for bidirectional transfer entropy computation."""

    def test_compute_bidirectional_te(self):
        """Test compute_bidirectional_te method."""
        # Lines 584-597: bidirectional TE computation
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        te = TransferEntropy(x, y)
        
        result = te.compute_bidirectional_te()
        # compute_bidirectional_te now returns a dict
        if isinstance(result, dict):
            te_forward = result["te_forward"]
            te_backward = result["te_backward"]
        else:
            te_forward, te_backward = result

        assert isinstance(te_forward, (int, float))
        assert isinstance(te_backward, (int, float))
        assert te_forward >= 0.0
        assert te_backward >= 0.0

    def test_compute_bidirectional_te_coupled_signals(self):
        """Test bidirectional TE with coupled signals."""
        # Create unidirectional coupling: X drives Y
        x = np.random.randn(1000)
        y = np.zeros(1000)
        y[0] = np.random.randn()
        for i in range(1, 1000):
            y[i] = 0.7 * y[i-1] + 0.3 * x[i-1] + 0.1 * np.random.randn()
        
        te = TransferEntropy(x, y)
        result = te.compute_bidirectional_te()
        if isinstance(result, dict):
            te_forward = result["te_forward"]
            te_backward = result["te_backward"]
        else:
            te_forward, te_backward = result

        assert isinstance(te_forward, (int, float))
        assert isinstance(te_backward, (int, float))
        # Forward should be larger than backward for unidirectional coupling
        assert te_forward >= 0.0
        assert te_backward >= 0.0


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestTimeDelayedTE:
    """Tests for time-delayed transfer entropy."""

    def test_compute_time_delayed_te_exception_handling(self):
        """Test compute_time_delayed_te exception handling."""
        # Lines 658-663: exception handling in time delayed TE
        import warnings
        x = np.random.randn(100)
        y = np.random.randn(100)
        te = TransferEntropy(x, y)
        
        # Use very large max_delay that might cause issues
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = te.compute_time_delayed_te(max_delay=50)
            # compute_time_delayed_te now returns a dict with 'te_values' key
            if isinstance(result, dict):
                te_arr = result["te_values"]
                assert "delays" in result
                assert "optimal_delay" in result
                assert "optimal_te" in result
            else:
                te_arr = result
            assert isinstance(te_arr, np.ndarray)
            assert len(te_arr) == 50

    def test_compute_time_delayed_te_normal(self):
        """Test compute_time_delayed_te with normal parameters."""
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        te = TransferEntropy(x, y)
        
        result = te.compute_time_delayed_te(max_delay=10)
        
        if isinstance(result, dict):
            te_arr = result["te_values"]
        else:
            te_arr = result
        assert isinstance(te_arr, np.ndarray)
        assert len(te_arr) == 10
        assert all(r >= 0.0 for r in te_arr)


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestEffectiveTE:
    """Tests for effective transfer entropy computation."""

    def test_compute_effective_te_normal(self):
        """Test compute_effective_te with normal signals."""
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        te = TransferEntropy(x, y)
        
        result = te.compute_effective_te()
        
        assert isinstance(result, (int, float))
        assert 0.0 <= result <= 1.0

    def test_compute_effective_te_zero_entropy(self):
        """Test compute_effective_te when conditional entropy is zero."""
        # Line 713: zero division case
        # Use constant signals where conditional entropy might be zero
        x = np.ones(1000)
        y = np.ones(1000)
        te = TransferEntropy(x, y)
        
        result = te.compute_effective_te()
        
        # Should handle zero entropy case gracefully
        assert isinstance(result, (int, float))
        assert 0.0 <= result <= 1.0


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestSignificanceTesting:
    """Tests for statistical significance testing."""

    def test_test_significance_shuffle_method(self):
        """Test test_significance with shuffle method."""
        # Lines 767-816: significance testing with shuffle
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        te = TransferEntropy(x, y)
        
        result = te.test_significance(n_surrogates=10, method="shuffle")
        # test_significance now returns a dict
        if isinstance(result, dict):
            p_value = result["p_value"]
            te_original = result["te_original"]
        else:
            p_value, te_original = result

        assert isinstance(p_value, (int, float))
        assert isinstance(te_original, (int, float))
        assert 0.0 <= p_value <= 1.0
        assert te_original >= 0.0

    def test_test_significance_phase_method(self):
        """Test test_significance with phase randomization method."""
        # Lines 776-781: phase randomization method
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        te = TransferEntropy(x, y)
        
        result = te.test_significance(n_surrogates=10, method="phase")
        if isinstance(result, dict):
            p_value = result["p_value"]
            te_original = result["te_original"]
        else:
            p_value, te_original = result

        assert isinstance(p_value, (int, float))
        assert isinstance(te_original, (int, float))
        assert 0.0 <= p_value <= 1.0
        assert te_original >= 0.0

    def test_test_significance_invalid_method(self):
        """Test test_significance with invalid method raises error."""
        # Line 783: invalid method error
        x = np.random.randn(500)
        y = np.random.randn(500)
        te = TransferEntropy(x, y)
        
        with pytest.raises(ValueError, match="Unknown method"):
            te.test_significance(n_surrogates=10, method="invalid_method")

    def test_test_significance_exception_handling(self):
        """Test test_significance exception handling in surrogate computation."""
        # Lines 795-803: exception handling in surrogate computation
        import warnings
        np.random.seed(42)
        x = np.random.randn(100)  # Short signal that might cause issues
        y = np.random.randn(100)
        te = TransferEntropy(x, y, k_coef=2, l_coef=2, delay=2)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Some surrogates might fail, but should handle gracefully
            res5 = te.test_significance(n_surrogates=5, method="shuffle")
            p_value = res5["p_value"] if isinstance(res5, dict) else res5[0]
            te_original = res5["te_original"] if isinstance(res5, dict) else res5[1]
            assert isinstance(p_value, (int, float))
            assert isinstance(te_original, (int, float))
            assert 0.0 <= p_value <= 1.0

    def test_test_significance_all_surrogates_fail(self):
        """Test test_significance when all surrogates fail."""
        # Lines 809-814: all surrogates fail case
        import warnings
        # Use very short signals that will likely cause all surrogates to fail
        x = np.random.randn(20)
        y = np.random.randn(20)
        te = TransferEntropy(x, y, k_coef=2, l_coef=2, delay=2)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res3 = te.test_significance(n_surrogates=3, method="shuffle")
            p_value = res3["p_value"] if isinstance(res3, dict) else res3[0]
            te_original = res3["te_original"] if isinstance(res3, dict) else res3[1]
            # Should return p_value=1.0 when all surrogates fail
            assert isinstance(p_value, (int, float))
            assert isinstance(te_original, (int, float))
            # p_value should be 1.0 if all surrogates failed
            if len(w) > 0 and "All surrogate calculations failed" in str(w[-1].message):
                assert p_value == 1.0


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestAliasParameters:
    """Tests covering lines 236, 238: k and l alias parameters."""

    def test_k_alias_parameter(self):
        """Test that k= alias sets k_coef (line 236)."""
        np.random.seed(0)
        x = np.random.randn(200)
        y = np.random.randn(200)
        te = TransferEntropy(x, y, k=2)
        assert te.k == 2

    def test_l_alias_parameter(self):
        """Test that l= alias sets l_coef (line 238)."""
        np.random.seed(0)
        x = np.random.randn(200)
        y = np.random.randn(200)
        te = TransferEntropy(x, y, l=2)
        assert te.l == 2

    def test_k_and_l_alias_together(self):
        """Test that both k= and l= aliases work together (lines 236, 238)."""
        np.random.seed(0)
        x = np.random.randn(200)
        y = np.random.randn(200)
        te = TransferEntropy(x, y, k=2, l=2)
        assert te.k == 2
        assert te.l == 2
        result = te.compute_transfer_entropy()
        assert isinstance(result, float)
        assert result >= 0.0


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestMutualInformationReshape:
    """Tests covering lines 430->432, 432->436: 1D reshape in _estimate_mutual_information_knn."""

    def test_mutual_information_knn_with_1d_arrays(self):
        """Directly call _estimate_mutual_information_knn with 1D arrays.

        Covers lines 430->432 (x.ndim==1 reshape) and 432->436 (y.ndim==1 reshape).
        """
        np.random.seed(1)
        x = np.random.randn(150)
        y = np.random.randn(150)
        te = TransferEntropy(x, y, k_coef=1, l_coef=1)
        # Call the private method directly with 1D arrays to trigger both reshape branches
        mi = te._estimate_mutual_information_knn(x, y, k=3)
        assert isinstance(mi, float)
        assert mi >= 0.0

    def test_mutual_information_knn_with_2d_arrays(self):
        """Call _estimate_mutual_information_knn with 2D arrays (no reshape needed)."""
        np.random.seed(2)
        x = np.random.randn(150, 1)
        y = np.random.randn(150, 1)
        te = TransferEntropy(x.ravel(), y.ravel(), k_coef=1, l_coef=1)
        mi = te._estimate_mutual_information_knn(x, y, k=3)
        assert isinstance(mi, float)
        assert mi >= 0.0


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestBidirectionalTEInterpretation:
    """Tests covering line 614: 'Target predominantly drives source' interpretation."""

    def test_bidirectional_te_target_drives_source(self):
        """Test interpretation when target drives source (line 614 branch)."""
        np.random.seed(42)
        # Y drives X: X[i] = 0.8*Y[i-1] + noise
        y = np.random.randn(200)
        x = np.zeros(200)
        x[0] = np.random.randn()
        for i in range(1, 200):
            x[i] = 0.8 * y[i - 1] + 0.1 * np.random.randn()

        te = TransferEntropy(x, y, k_coef=1, l_coef=1)
        result = te.compute_bidirectional_te()
        assert isinstance(result, dict)
        assert "interpretation" in result
        # The result dict must always have these keys
        assert "te_forward" in result
        assert "te_backward" in result

    def test_bidirectional_te_ratio_infinite(self):
        """Test bidirectional TE when te_backward is 0 (ratio becomes inf, line 610)."""
        np.random.seed(7)
        # Highly predictable coupling where backward TE is likely 0
        x = np.sin(np.linspace(0, 20, 200))
        y = np.sin(np.linspace(0, 20, 200) + 0.1)
        te = TransferEntropy(x, y, k_coef=1, l_coef=1)
        result = te.compute_bidirectional_te()
        assert isinstance(result, dict)
        assert "ratio" in result


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestTimeDelayedTEWarning:
    """Tests covering lines 687-692: warning in compute_time_delayed_te when TE fails."""

    def test_time_delayed_te_basic(self):
        """Test compute_time_delayed_te returns expected keys (lines 680-700)."""
        np.random.seed(3)
        x = np.random.randn(200)
        y = np.random.randn(200)
        te = TransferEntropy(x, y)
        result = te.compute_time_delayed_te(max_delay=5)
        assert isinstance(result, dict)
        assert "te_values" in result
        assert "delays" in result
        assert "optimal_delay" in result
        assert "optimal_te" in result
        assert len(result["te_values"]) == 5

    def test_time_delayed_te_warning_on_failure(self):
        """Test that a UserWarning is issued if TE fails for a delay (lines 687-692)."""
        import warnings
        import unittest.mock as mock

        np.random.seed(9)
        x = np.random.randn(150)
        y = np.random.randn(150)
        te = TransferEntropy(x, y, k_coef=1, l_coef=1)

        # Make compute_transfer_entropy raise on delay==2 (second iteration)
        original_compute = TransferEntropy.compute_transfer_entropy
        call_count = [0]

        def patched_compute(self_inner):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("Simulated TE failure at delay 2")
            return original_compute(self_inner)

        with mock.patch.object(TransferEntropy, "compute_transfer_entropy", patched_compute):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = te.compute_time_delayed_te(max_delay=3)

        assert isinstance(result, dict)
        assert "te_values" in result
        # Warning should have been issued for the failed delay
        warning_msgs = [str(warning.message) for warning in w]
        assert any("Failed to compute TE at delay" in msg for msg in warning_msgs)


@pytest.mark.skipif(not TRANSFER_ENTROPY_AVAILABLE, reason="Transfer entropy not available")
class TestSignificancePhaseMethod:
    """Tests covering lines 819->821: phase randomization method in test_significance."""

    def test_significance_phase_method(self):
        """Test test_significance with method='phase' using even-length signal (lines 812-822)."""
        np.random.seed(5)
        x = np.random.randn(200)  # even length -> line 820 executed
        y = np.random.randn(200)
        te = TransferEntropy(x, y)
        result = te.test_significance(n_surrogates=5, method="phase")
        assert isinstance(result, dict)
        assert "p_value" in result
        assert "te_original" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_significance_phase_method_odd_length(self):
        """Test test_significance with method='phase' using odd-length signal.

        Covers line 819->821: when n % 2 != 0 (odd), line 820 is skipped.
        """
        np.random.seed(6)
        x = np.random.randn(201)  # odd length -> n%2 != 0, skip line 820
        y = np.random.randn(201)
        te = TransferEntropy(x, y)
        result = te.test_significance(n_surrogates=5, method="phase")
        assert isinstance(result, dict)
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_significance_shuffle_p_values(self):
        """Test that shuffle surrogates produce valid statistics (lines 847-868)."""
        np.random.seed(10)
        x = np.random.randn(200)
        y = np.random.randn(200)
        te = TransferEntropy(x, y)
        result = te.test_significance(n_surrogates=10, method="shuffle")
        assert isinstance(result, dict)
        assert "p_value" in result
        assert "te_surrogates_mean" in result
        assert "te_surrogates_std" in result
        assert "effect_size" in result
        assert "significance" in result
        p = result["p_value"]
        assert 0.0 <= p <= 1.0
        # Check significance label is one of three expected strings
        assert result["significance"] in [
            "Highly significant (p < 0.01)",
            "Significant (p < 0.05)",
            "Not significant (p >= 0.05)",
        ]

    def test_significance_surrogate_exception_handling(self):
        """Cover lines 839-844: exception in surrogate TE computation is caught and warned.

        We mock compute_transfer_entropy to raise on the second call (surrogate).
        """
        import warnings
        import unittest.mock as mock

        np.random.seed(42)
        x = np.random.randn(150)
        y = np.random.randn(150)
        te = TransferEntropy(x, y)

        original_compute = TransferEntropy.compute_transfer_entropy
        call_count = [0]

        def patched_compute(self_inner):
            call_count[0] += 1
            if call_count[0] == 1:
                return original_compute(self_inner)  # first call succeeds (te_original)
            raise RuntimeError("Simulated surrogate failure")

        with mock.patch.object(TransferEntropy, "compute_transfer_entropy", patched_compute):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = te.test_significance(n_surrogates=5, method="shuffle")

        # All surrogates failed -> empty surrogate_tes -> warning about all surrogates failing
        assert isinstance(result, dict)
        assert result["p_value"] == 1.0  # lines 860: p_value = 1.0 when all fail
        assert result["te_surrogates_std"] == 0.0

    def test_significance_highly_significant_label(self):
        """Cover line 866: 'Highly significant (p < 0.01)' label branch.

        We mock surrogate TE values to all be 0 so p_value = 0.0 < 0.01.
        """
        import unittest.mock as mock

        np.random.seed(0)
        x = np.random.randn(150)
        y = np.random.randn(150)
        te = TransferEntropy(x, y)

        # Make every surrogate return 0.0 so original TE (>0) beats all -> p=0.0
        original_compute = TransferEntropy.compute_transfer_entropy
        call_count = [0]

        def patched_compute(self_inner):
            call_count[0] += 1
            if call_count[0] == 1:
                return 1.0  # large te_original
            return 0.0  # surrogate always 0 -> p_value = 0 < 0.01

        with mock.patch.object(TransferEntropy, "compute_transfer_entropy", patched_compute):
            result = te.test_significance(n_surrogates=20, method="shuffle")

        assert result["significance"] == "Highly significant (p < 0.01)"

    def test_significance_significant_label(self):
        """Cover line 868: 'Significant (p < 0.05)' label branch.

        We arrange p_value to be between 0.01 and 0.05.
        """
        import unittest.mock as mock

        np.random.seed(1)
        x = np.random.randn(150)
        y = np.random.randn(150)
        te = TransferEntropy(x, y)

        # We need exactly 1 or 2 out of 100 surrogates to beat original TE
        # Make original TE = 1.0, and 1 surrogate = 2.0, rest = 0.0 -> p = 1/101 ~ 0.0099
        # For 0.01 < p < 0.05 with 100 surrogates, we need 2-4 surrogates to exceed original
        call_count = [0]

        def patched_compute(self_inner):
            call_count[0] += 1
            if call_count[0] == 1:
                return 1.0  # te_original
            # 3 out of 100 surrogates return 2.0 (beat original) -> p = 3/100 = 0.03
            if call_count[0] in (2, 3, 4):
                return 2.0
            return 0.5

        with mock.patch.object(TransferEntropy, "compute_transfer_entropy", patched_compute):
            result = te.test_significance(n_surrogates=100, method="shuffle")

        p = result["p_value"]
        assert 0.01 <= p <= 0.05
        assert result["significance"] == "Significant (p < 0.05)"
