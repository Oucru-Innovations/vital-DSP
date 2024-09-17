import pytest
import numpy as np
from vitalDSP.filtering.advanced_signal_filtering import AdvancedSignalFiltering


@pytest.fixture
def signal():
    return np.array([1, 2, 3, 2, 1, 2, 3, 4, 3, 2])


@pytest.fixture
def target_signal():
    return np.array([1, 1.5, 2.5, 3, 2.5, 2, 2.5, 3.5, 3, 2.5])


def test_kalman_filter(signal):
    af = AdvancedSignalFiltering(signal)
    filtered_signal = af.kalman_filter(R=0.1, Q=0.01)
    expected_signal = np.array([1.0, 1.75, 2.5, 2.12, 1.56, 1.94, 2.7, 3.5, 3.12, 2.56])
    np.testing.assert_almost_equal(np.real(filtered_signal), expected_signal, decimal=0)


def test_optimization_based_filtering(signal, target_signal):
    af = AdvancedSignalFiltering(signal)
    filtered_signal_mse = af.optimization_based_filtering(
        target_signal,
        loss_type="mse",
        initial_guess=2,
        learning_rate=0.01,
        iterations=100,
    )
    filtered_signal_mae = af.optimization_based_filtering(
        target_signal,
        loss_type="mae",
        initial_guess=2,
        learning_rate=0.01,
        iterations=100,
    )
    filtered_signal_huber = af.optimization_based_filtering(
        target_signal,
        loss_type="huber",
        initial_guess=2,
        learning_rate=0.01,
        iterations=100,
    )
    filtered_signal_smooth_l1 = af.optimization_based_filtering(
        target_signal,
        loss_type="smooth_l1",
        initial_guess=2,
        learning_rate=0.01,
        iterations=100,
    )
    filtered_signal_log_cosh = af.optimization_based_filtering(
        target_signal,
        loss_type="log_cosh",
        initial_guess=2,
        learning_rate=0.01,
        iterations=100,
    )
    filtered_signal_quantile = af.optimization_based_filtering(
        target_signal,
        loss_type="quantile",
        initial_guess=2,
        learning_rate=0.01,
        iterations=100,
    )
    assert len(filtered_signal_mse) == len(signal)
    assert len(filtered_signal_mae) == len(signal)
    assert len(filtered_signal_huber) == len(signal)
    assert len(filtered_signal_smooth_l1) == len(signal)
    assert len(filtered_signal_quantile) == len(signal)
    assert len(filtered_signal_smooth_l1) == len(signal)
    assert np.all(np.isfinite(filtered_signal_mse))


def test_gradient_descent_filter(signal, target_signal):
    af = AdvancedSignalFiltering(signal)
    filtered_signal = af.gradient_descent_filter(
        target_signal, learning_rate=0.1, iterations=50
    )
    assert len(filtered_signal) == len(signal)
    assert np.all(np.isfinite(filtered_signal))


def test_ensemble_filtering_mean(signal):
    af = AdvancedSignalFiltering(signal)
    filters = [af.kalman_filter, af.kalman_filter]  # Same filter applied twice
    filtered_signal = af.ensemble_filtering(filters, method="mean")
    assert len(filtered_signal) == len(signal)
    assert np.all(np.isfinite(np.real(filtered_signal)))


def test_ensemble_filtering_weighted_mean(signal):
    af = AdvancedSignalFiltering(signal)
    filters = [af.kalman_filter, af.kalman_filter]  # Same filter applied twice
    weights = [0.7, 0.3]
    filtered_signal = af.ensemble_filtering(
        filters, method="weighted_mean", weights=weights
    )
    filtered_signal_bagging = af.ensemble_filtering(
        filters, method="bagging", num_iterations=5
    )
    filtered_signal_boosting = af.ensemble_filtering(
        filters, method="boosting", weights=weights
    )
    assert len(filtered_signal_bagging) == len(signal)
    assert len(filtered_signal_boosting) == len(signal)
    assert len(filtered_signal) == len(signal)
    assert np.all(np.isfinite(filtered_signal))


def test_convolution_based_filter_smoothing(signal):
    af = AdvancedSignalFiltering(signal)
    filtered_signal_smoothing = af.convolution_based_filter(
        kernel_type="smoothing", kernel_size=3
    )
    filtered_signal_sharpening = af.convolution_based_filter(
        kernel_type="sharpening", kernel_size=3
    )
    filtered_signal_edge_detection = af.convolution_based_filter(
        kernel_type="edge_detection", kernel_size=3
    )
    assert len(filtered_signal_smoothing) == len(signal)
    assert len(filtered_signal_sharpening) == len(signal)
    assert len(filtered_signal_edge_detection) == len(signal)
    assert np.all(np.isfinite(filtered_signal_smoothing))
    filtered_signal_custom = af.convolution_based_filter(
        kernel_type="custom", custom_kernel=[0.2, 0.3, 0.3, 0.2]
    )
    assert len(filtered_signal_custom) == len(signal)
    with pytest.raises(
        ValueError,
        match="Invalid kernel_type. Must be 'smoothing', 'sharpening', 'edge_detection', or 'custom' with a custom_kernel provided.",
    ):
        af.convolution_based_filter(kernel_type="invalid")


def test_attention_based_filter_gaussian(signal):
    af = AdvancedSignalFiltering(signal)
    filtered_signal = af.attention_based_filter(
        attention_type="gaussian", size=5, sigma=1.0
    )
    assert len(filtered_signal) == len(signal)
    filtered_signal = af.attention_based_filter(
        attention_type="linear", size=5, sigma=1.0
    )
    assert len(filtered_signal) == len(signal)
    filtered_signal = af.attention_based_filter(
        attention_type="exponential", size=5, sigma=1.0
    )
    assert len(filtered_signal) == len(signal)
    assert np.all(np.isfinite(filtered_signal))
    filtered_signal = af.attention_based_filter(
        attention_type="custom", custom_weights=[0.2, 0.3, 0.3, 0.2]
    )
    assert len(filtered_signal) == len(signal)
    with pytest.raises(
        ValueError,
        match="Invalid attention_type. Must be 'uniform', 'linear', 'gaussian', 'exponential', or 'custom' with custom_weights provided.",
    ):
        af.attention_based_filter(attention_type="invalid_method")


def test_adaptive_filtering(signal, target_signal):
    af = AdvancedSignalFiltering(signal)
    filtered_signal = af.adaptive_filtering(target_signal, mu=0.01, filter_order=4)
    assert len(filtered_signal) == len(signal)
    assert np.all(np.isfinite(filtered_signal))
