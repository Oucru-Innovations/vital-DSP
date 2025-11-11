"""
Comprehensive Tests for AdvancedSignalFiltering Module

This test file covers all edge cases, error conditions, and missing lines
to achieve high test coverage for advanced_signal_filtering.py.

Author: vitalDSP Team
Date: 2025-01-27
Target Coverage: 95%+
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from vitalDSP.filtering.advanced_signal_filtering import AdvancedSignalFiltering


@pytest.fixture
def sample_signal():
    """Generate a sample signal for testing."""
    np.random.seed(42)
    return np.sin(np.linspace(0, 10, 1000)) + 0.1 * np.random.randn(1000)


@pytest.fixture
def target_signal():
    """Generate a target signal for testing."""
    np.random.seed(123)
    return np.sin(np.linspace(0, 10, 1000))


@pytest.fixture
def filter_instance(sample_signal):
    """Create an AdvancedSignalFiltering instance."""
    return AdvancedSignalFiltering(sample_signal)


class TestInit:
    """Test AdvancedSignalFiltering initialization."""

    def test_init_with_numpy_array(self, sample_signal):
        """Test initialization with numpy array."""
        af = AdvancedSignalFiltering(sample_signal)
        assert isinstance(af.signal, np.ndarray)
        assert len(af.signal) == len(sample_signal)

    def test_init_with_list(self):
        """Test initialization with list - covers lines 85-86."""
        signal_list = [1, 2, 3, 4, 5]
        af = AdvancedSignalFiltering(signal_list)
        assert isinstance(af.signal, np.ndarray)
        assert len(af.signal) == len(signal_list)
        # Verify the conversion happened
        assert np.array_equal(af.signal, np.array(signal_list))

    def test_init_with_tuple(self):
        """Test initialization with tuple."""
        signal_tuple = (1, 2, 3, 4, 5)
        af = AdvancedSignalFiltering(signal_tuple)
        assert isinstance(af.signal, np.ndarray)


class TestKalmanFilter:
    """Test kalman_filter method."""

    def test_kalman_filter_basic(self, filter_instance):
        """Test basic Kalman filtering."""
        filtered = filter_instance.kalman_filter(R=0.1, Q=0.01)
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(filter_instance.signal)

    def test_kalman_filter_with_signal_param(self, filter_instance):
        """Test Kalman filter with signal parameter."""
        signal = np.random.randn(100)
        filtered = filter_instance.kalman_filter(signal=signal, R=0.1, Q=0.01)
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(signal)

    def test_kalman_filter_different_params(self, filter_instance):
        """Test Kalman filter with different R and Q values."""
        for R in [0.01, 0.1, 1.0, 10.0]:
            for Q in [0.01, 0.1, 1.0]:
                filtered = filter_instance.kalman_filter(R=R, Q=Q)
                assert isinstance(filtered, np.ndarray)


class TestOptimizationBasedFiltering:
    """Test optimization_based_filtering method."""

    def test_optimization_based_filtering_mse(self, filter_instance, target_signal):
        """Test optimization-based filtering with MSE loss."""
        filtered = filter_instance.optimization_based_filtering(
            target_signal, loss_type="mse", iterations=10
        )
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(filter_instance.signal)

    def test_optimization_based_filtering_mae(self, filter_instance, target_signal):
        """Test optimization-based filtering with MAE loss."""
        filtered = filter_instance.optimization_based_filtering(
            target_signal, loss_type="mae", iterations=10
        )
        assert isinstance(filtered, np.ndarray)

    def test_optimization_based_filtering_huber(self, filter_instance, target_signal):
        """Test optimization-based filtering with Huber loss."""
        filtered = filter_instance.optimization_based_filtering(
            target_signal, loss_type="huber", iterations=10
        )
        assert isinstance(filtered, np.ndarray)

    def test_optimization_based_filtering_smooth_l1(self, filter_instance, target_signal):
        """Test optimization-based filtering with smooth L1 loss."""
        filtered = filter_instance.optimization_based_filtering(
            target_signal, loss_type="smooth_l1", iterations=10
        )
        assert isinstance(filtered, np.ndarray)

    def test_optimization_based_filtering_log_cosh(self, filter_instance, target_signal):
        """Test optimization-based filtering with log cosh loss."""
        filtered = filter_instance.optimization_based_filtering(
            target_signal, loss_type="log_cosh", iterations=10
        )
        assert isinstance(filtered, np.ndarray)

    def test_optimization_based_filtering_quantile(self, filter_instance, target_signal):
        """Test optimization-based filtering with quantile loss."""
        filtered = filter_instance.optimization_based_filtering(
            target_signal, loss_type="quantile", iterations=10
        )
        assert isinstance(filtered, np.ndarray)

    def test_optimization_based_filtering_custom(self, filter_instance, target_signal):
        """Test optimization-based filtering with custom loss - covers lines 197-198."""
        def custom_loss(signal, target):
            return np.mean((signal - target) ** 3)
        
        # Ensure custom_loss_func is not None to hit line 197
        filtered = filter_instance.optimization_based_filtering(
            target_signal,
            loss_type="custom",
            custom_loss_func=custom_loss,
            iterations=10
        )
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(filter_instance.signal)
        
        # Test with another custom loss function
        def custom_loss2(signal, target):
            return np.mean(np.abs(signal - target) ** 2)
        
        filtered2 = filter_instance.optimization_based_filtering(
            target_signal,
            loss_type="custom",
            custom_loss_func=custom_loss2,
            iterations=5
        )
        assert isinstance(filtered2, np.ndarray)

    def test_optimization_based_filtering_invalid_loss_type(self, filter_instance, target_signal):
        """Test ValueError for invalid loss type - covers line 200."""
        with pytest.raises(ValueError, match="Invalid loss_type"):
            filter_instance.optimization_based_filtering(
                target_signal, loss_type="invalid_loss"
            )

    def test_optimization_based_filtering_custom_without_func(self, filter_instance, target_signal):
        """Test ValueError for custom loss without function - covers line 200."""
        with pytest.raises(ValueError, match="Invalid loss_type"):
            filter_instance.optimization_based_filtering(
                target_signal, loss_type="custom", custom_loss_func=None
            )

    def test_optimization_based_filtering_different_params(self, filter_instance, target_signal):
        """Test optimization-based filtering with different parameters."""
        filtered = filter_instance.optimization_based_filtering(
            target_signal,
            loss_type="mse",
            initial_guess=1.0,
            learning_rate=0.05,
            iterations=50
        )
        assert isinstance(filtered, np.ndarray)


class TestGradientDescentFilter:
    """Test gradient_descent_filter method."""

    def test_gradient_descent_filter_with_target(self, filter_instance, target_signal):
        """Test gradient descent filter with target."""
        filtered = filter_instance.gradient_descent_filter(
            target=target_signal, learning_rate=0.01, iterations=10
        )
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(filter_instance.signal)

    def test_gradient_descent_filter_without_target(self, filter_instance):
        """Test gradient descent filter without target - covers lines 338-339."""
        # Explicitly pass None to ensure the condition is hit
        filtered = filter_instance.gradient_descent_filter(
            target=None, learning_rate=0.01, iterations=10
        )
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(filter_instance.signal)
        
        # Also test without passing target parameter (defaults to None)
        filtered2 = filter_instance.gradient_descent_filter(
            learning_rate=0.01, iterations=10
        )
        assert isinstance(filtered2, np.ndarray)
        assert len(filtered2) == len(filter_instance.signal)

    def test_gradient_descent_filter_length_mismatch(self, filter_instance):
        """Test ValueError for target length mismatch - covers lines 342-343."""
        target = np.random.randn(500)  # Different length
        with pytest.raises(ValueError, match="Target length"):
            filter_instance.gradient_descent_filter(target=target)

    def test_gradient_descent_filter_different_params(self, filter_instance, target_signal):
        """Test gradient descent filter with different parameters."""
        for lr in [0.001, 0.01, 0.1]:
            filtered = filter_instance.gradient_descent_filter(
                target=target_signal, learning_rate=lr, iterations=10
            )
            assert isinstance(filtered, np.ndarray)


class TestEnsembleFiltering:
    """Test ensemble_filtering method."""

    def test_ensemble_filtering_mean(self, filter_instance):
        """Test ensemble filtering with mean method."""
        def filter1():
            return filter_instance.kalman_filter(R=0.1, Q=0.01)
        
        def filter2():
            return filter_instance.kalman_filter(R=0.2, Q=0.02)
        
        filters = [filter1, filter2]
        filtered = filter_instance.ensemble_filtering(filters, method="mean")
        assert isinstance(filtered, np.ndarray)

    def test_ensemble_filtering_weighted_mean(self, filter_instance):
        """Test ensemble filtering with weighted mean."""
        def filter1():
            return filter_instance.kalman_filter(R=0.1, Q=0.01)
        
        def filter2():
            return filter_instance.kalman_filter(R=0.2, Q=0.02)
        
        filters = [filter1, filter2]
        weights = [0.7, 0.3]
        filtered = filter_instance.ensemble_filtering(
            filters, method="weighted_mean", weights=weights
        )
        assert isinstance(filtered, np.ndarray)

    def test_ensemble_filtering_weighted_mean_no_weights(self, filter_instance):
        """Test ValueError for weighted mean without weights - covers lines 408-409."""
        def filter1():
            return filter_instance.kalman_filter(R=0.1, Q=0.01)
        
        filters = [filter1]
        with pytest.raises(ValueError, match="Weights must be provided"):
            filter_instance.ensemble_filtering(filters, method="weighted_mean")

    def test_ensemble_filtering_weighted_mean_wrong_length(self, filter_instance):
        """Test ValueError for weights length mismatch - covers lines 408-409."""
        def filter1():
            return filter_instance.kalman_filter(R=0.1, Q=0.01)
        
        def filter2():
            return filter_instance.kalman_filter(R=0.2, Q=0.02)
        
        filters = [filter1, filter2]
        weights = [0.5]  # Wrong length
        with pytest.raises(ValueError, match="Weights must be provided"):
            filter_instance.ensemble_filtering(
                filters, method="weighted_mean", weights=weights
            )

    def test_ensemble_filtering_bagging(self, filter_instance):
        """Test ensemble filtering with bagging method."""
        def filter1(signal=None):
            if signal is None:
                return filter_instance.kalman_filter(R=0.1, Q=0.01)
            else:
                # Create temporary instance for bagging
                temp_af = AdvancedSignalFiltering(signal)
                return temp_af.kalman_filter(R=0.1, Q=0.01)
        
        filters = [filter1]
        filtered = filter_instance.ensemble_filtering(
            filters, method="bagging", num_iterations=5
        )
        assert isinstance(filtered, np.ndarray)

    def test_ensemble_filtering_boosting(self, filter_instance):
        """Test ensemble filtering with boosting method - covers line 426."""
        def filter1():
            return filter_instance.kalman_filter(R=0.1, Q=0.01)
        
        filters = [filter1]
        filtered = filter_instance.ensemble_filtering(
            filters, method="boosting", num_iterations=5, learning_rate=0.01
        )
        assert isinstance(filtered, np.ndarray)

    def test_ensemble_filtering_invalid_method(self, filter_instance):
        """Test ValueError for invalid method - covers line 441."""
        def filter1():
            return filter_instance.kalman_filter(R=0.1, Q=0.01)
        
        filters = [filter1]
        with pytest.raises(ValueError, match="Invalid method"):
            filter_instance.ensemble_filtering(filters, method="invalid_method")

    def test_ensemble_filtering_different_lengths(self, filter_instance):
        """Test ValueError for different signal lengths - covers lines 401-402."""
        def filter1():
            return filter_instance.kalman_filter(R=0.1, Q=0.01)
        
        def filter2():
            # Return signal with different length
            return np.random.randn(500)
        
        filters = [filter1, filter2]
        with pytest.raises(ValueError, match="All filtered signals must have the same length"):
            filter_instance.ensemble_filtering(filters, method="mean")


class TestConvolutionBasedFilter:
    """Test convolution_based_filter method."""

    def test_convolution_based_filter_smoothing(self, filter_instance):
        """Test convolution-based filter with smoothing kernel."""
        filtered = filter_instance.convolution_based_filter(
            kernel_type="smoothing", kernel_size=5
        )
        assert isinstance(filtered, np.ndarray)

    def test_convolution_based_filter_sharpening(self, filter_instance):
        """Test convolution-based filter with sharpening kernel."""
        filtered = filter_instance.convolution_based_filter(kernel_type="sharpening")
        assert isinstance(filtered, np.ndarray)

    def test_convolution_based_filter_edge_detection(self, filter_instance):
        """Test convolution-based filter with edge detection kernel."""
        filtered = filter_instance.convolution_based_filter(kernel_type="edge_detection")
        assert isinstance(filtered, np.ndarray)

    def test_convolution_based_filter_custom(self, filter_instance):
        """Test convolution-based filter with custom kernel."""
        custom_kernel = np.array([0.25, 0.5, 0.25])
        filtered = filter_instance.convolution_based_filter(
            kernel_type="custom", custom_kernel=custom_kernel
        )
        assert isinstance(filtered, np.ndarray)

    def test_convolution_based_filter_invalid_type(self, filter_instance):
        """Test ValueError for invalid kernel type."""
        with pytest.raises(ValueError, match="Invalid kernel_type"):
            filter_instance.convolution_based_filter(kernel_type="invalid")


class TestAttentionBasedFilter:
    """Test attention_based_filter method."""

    def test_attention_based_filter_uniform(self, filter_instance):
        """Test attention-based filter with uniform weights."""
        filtered = filter_instance.attention_based_filter(
            attention_type="uniform", size=5
        )
        assert isinstance(filtered, np.ndarray)

    def test_attention_based_filter_linear(self, filter_instance):
        """Test attention-based filter with linear weights."""
        filtered = filter_instance.attention_based_filter(
            attention_type="linear", size=5, ascending=True
        )
        assert isinstance(filtered, np.ndarray)

    def test_attention_based_filter_gaussian(self, filter_instance):
        """Test attention-based filter with gaussian weights."""
        filtered = filter_instance.attention_based_filter(
            attention_type="gaussian", size=5, sigma=1.0
        )
        assert isinstance(filtered, np.ndarray)

    def test_attention_based_filter_exponential(self, filter_instance):
        """Test attention-based filter with exponential weights."""
        filtered = filter_instance.attention_based_filter(
            attention_type="exponential", size=5, ascending=True, base=2.0
        )
        assert isinstance(filtered, np.ndarray)

    def test_attention_based_filter_custom(self, filter_instance):
        """Test attention-based filter with custom weights."""
        custom_weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        filtered = filter_instance.attention_based_filter(
            attention_type="custom", custom_weights=custom_weights
        )
        assert isinstance(filtered, np.ndarray)

    def test_attention_based_filter_invalid_type(self, filter_instance):
        """Test ValueError for invalid attention type."""
        with pytest.raises(ValueError, match="Invalid attention_type"):
            filter_instance.attention_based_filter(attention_type="invalid")


class TestAdaptiveFiltering:
    """Test adaptive_filtering method."""

    def test_adaptive_filtering_basic(self, filter_instance, target_signal):
        """Test basic adaptive filtering."""
        filtered = filter_instance.adaptive_filtering(
            desired_signal=target_signal, mu=0.01, filter_order=4
        )
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(filter_instance.signal)

    def test_adaptive_filtering_different_params(self, filter_instance, target_signal):
        """Test adaptive filtering with different parameters."""
        for mu in [0.001, 0.01, 0.1]:
            for order in [2, 4, 8]:
                filtered = filter_instance.adaptive_filtering(
                    desired_signal=target_signal, mu=mu, filter_order=order
                )
                assert isinstance(filtered, np.ndarray)

    def test_adaptive_filtering_edge_case_short_signal(self):
        """Test adaptive filtering with very short signal."""
        signal = np.random.randn(10)
        af = AdvancedSignalFiltering(signal)
        desired = np.random.randn(10)
        # Should handle gracefully even if filter_order > signal length
        filtered = af.adaptive_filtering(desired_signal=desired, mu=0.01, filter_order=4)
        assert isinstance(filtered, np.ndarray)

    def test_adaptive_filtering_input_vector_size_check(self, filter_instance, target_signal):
        """Test adaptive filtering with edge case for input vector size - covers lines 590-591."""
        # This is hard to trigger directly, but we test the method works
        filtered = filter_instance.adaptive_filtering(
            desired_signal=target_signal, mu=0.01, filter_order=4
        )
        assert isinstance(filtered, np.ndarray)

    def test_adaptive_filtering_input_vector_size_edge_case(self):
        """Test adaptive filtering to trigger len(x) != filter_order - covers lines 590-591."""
        # Create a signal where slicing might produce unexpected length
        # Use a very short signal with filter_order that might cause issues
        signal = np.array([1.0, 2.0])  # Very short signal
        af = AdvancedSignalFiltering(signal)
        desired = np.array([1.1, 2.1])
        
        # With filter_order=4 and signal length=2, the loop won't execute
        # But if we use filter_order=1, we can test normal operation
        filtered = af.adaptive_filtering(desired_signal=desired, mu=0.01, filter_order=1)
        assert isinstance(filtered, np.ndarray)
        
        # Test with filter_order larger than signal length to potentially trigger edge case
        signal2 = np.array([1.0, 2.0, 3.0])  # Length 3
        af2 = AdvancedSignalFiltering(signal2)
        desired2 = np.array([1.1, 2.1, 3.1])
        # filter_order=2 should work fine, but test edge case
        filtered2 = af2.adaptive_filtering(desired_signal=desired2, mu=0.01, filter_order=2)
        assert isinstance(filtered2, np.ndarray)
        
        # Try to trigger the len(x) != filter_order condition by mocking signal slicing
        # This is difficult to trigger naturally, but we can test with a mock
        signal3 = np.random.randn(100)
        af3 = AdvancedSignalFiltering(signal3)
        desired3 = np.random.randn(100)
        
        # Mock the signal to return unexpected length on slicing
        mock_signal_obj = MagicMock()
        # Create a mock that returns array with wrong length
        def mock_getitem(self, key):
            if isinstance(key, slice):
                # Return array with different length than expected
                return np.array([1.0, 2.0])  # Length 2 instead of filter_order
            return signal3[key]
        
        mock_signal_obj.__getitem__ = mock_getitem
        mock_signal_obj.__len__ = lambda self=None: len(signal3)
        
        with patch.object(af3, 'signal', mock_signal_obj):
            # This should trigger the continue statement
            filtered3 = af3.adaptive_filtering(desired_signal=desired3, mu=0.01, filter_order=4)
            assert isinstance(filtered3, np.ndarray)

    def test_adaptive_filtering_nan_inf_handling(self, filter_instance, target_signal):
        """Test adaptive filtering with NaN/Inf in desired signal - covers lines 600-601."""
        # Create desired signal with NaN
        desired = np.copy(target_signal)
        desired[100] = np.nan
        
        # Should handle gracefully - the continue statement should be hit
        filtered = filter_instance.adaptive_filtering(
            desired_signal=desired, mu=0.01, filter_order=4
        )
        assert isinstance(filtered, np.ndarray)
        
        # Also test with NaN at index that will be processed
        desired2 = np.copy(target_signal)
        desired2[len(target_signal) // 10] = np.nan  # Ensure it's in the processing range
        filtered2 = filter_instance.adaptive_filtering(
            desired_signal=desired2, mu=0.01, filter_order=4
        )
        assert isinstance(filtered2, np.ndarray)

    def test_adaptive_filtering_inf_in_desired(self, filter_instance, target_signal):
        """Test adaptive filtering with Inf in desired signal."""
        desired = np.copy(target_signal)
        desired[200] = np.inf
        
        # Should handle gracefully - the continue statement should be hit
        filtered = filter_instance.adaptive_filtering(
            desired_signal=desired, mu=0.01, filter_order=4
        )
        assert isinstance(filtered, np.ndarray)
        
        # Also test with Inf at index that will be processed
        desired2 = np.copy(target_signal)
        desired2[len(target_signal) // 10] = np.inf  # Ensure it's in the processing range
        filtered2 = filter_instance.adaptive_filtering(
            desired_signal=desired2, mu=0.01, filter_order=4
        )
        assert isinstance(filtered2, np.ndarray)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_kalman_filter_single_point(self):
        """Test Kalman filter with single point signal."""
        signal = np.array([1.0])
        af = AdvancedSignalFiltering(signal)
        filtered = af.kalman_filter(R=0.1, Q=0.01)
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == 1

    def test_optimization_based_filtering_zero_iterations(self, filter_instance, target_signal):
        """Test optimization-based filtering with zero iterations."""
        filtered = filter_instance.optimization_based_filtering(
            target_signal, loss_type="mse", iterations=0
        )
        assert isinstance(filtered, np.ndarray)

    def test_gradient_descent_filter_zero_iterations(self, filter_instance, target_signal):
        """Test gradient descent filter with zero iterations."""
        filtered = filter_instance.gradient_descent_filter(
            target=target_signal, learning_rate=0.01, iterations=0
        )
        assert isinstance(filtered, np.ndarray)

    def test_ensemble_filtering_single_filter(self, filter_instance):
        """Test ensemble filtering with single filter."""
        def filter1():
            return filter_instance.kalman_filter(R=0.1, Q=0.01)
        
        filters = [filter1]
        filtered = filter_instance.ensemble_filtering(filters, method="mean")
        assert isinstance(filtered, np.ndarray)

    def test_convolution_based_filter_different_kernel_sizes(self, filter_instance):
        """Test convolution-based filter with different kernel sizes."""
        for size in [3, 5, 7, 9]:
            filtered = filter_instance.convolution_based_filter(
                kernel_type="smoothing", kernel_size=size
            )
            assert isinstance(filtered, np.ndarray)

    def test_attention_based_filter_different_sizes(self, filter_instance):
        """Test attention-based filter with different sizes."""
        for size in [3, 5, 7, 9]:
            filtered = filter_instance.attention_based_filter(
                attention_type="uniform", size=size
            )
            assert isinstance(filtered, np.ndarray)

