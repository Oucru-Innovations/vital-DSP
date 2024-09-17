import pytest
import numpy as np
from vitalDSP.physiological_features.signal_segmentation import SignalSegmentation


@pytest.fixture
def signal_data():
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])


@pytest.fixture
def segmentation(signal_data):
    return SignalSegmentation(signal_data)


def test_fixed_size_segmentation(segmentation):
    result = segmentation.fixed_size_segmentation(3)
    expected = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    assert all(np.array_equal(r, e) for r, e in zip(result, expected))


def test_adaptive_segmentation_optimized(segmentation):
    def adaptive_fn(segment):
        # Return a fixed length for segmentation to avoid too many divisions
        return max(3, len(segment) // 2)  # Ensure it doesn't get too small

    # Test with a moderate-sized signal for performance reasons
    segmentation = SignalSegmentation(
        np.random.rand(1000)
    )  # 1000 sample points instead of too many

    result = segmentation.adaptive_segmentation(adaptive_fn)

    # Check if the result has reasonable segments and isn't too slow
    assert len(result) > 1
    assert all(len(seg) > 0 for seg in result)  # Ensure all segments have some length


def test_threshold_based_segmentation(segmentation):
    signal = np.array([1, 2, 5, 2, 8, 1])
    segmentation = SignalSegmentation(signal)

    result = segmentation.threshold_based_segmentation(4)
    expected = [np.array([5]), np.array([8])]

    # Check if the result matches expected segments
    assert len(result) == len(expected)
    assert all(np.array_equal(r, e) for r, e in zip(result, expected))


def test_variance_based_segmentation():
    signal = np.array([1, 2, 2, 2, 5, 6, 1, 1, 1, 8])
    segmentation = SignalSegmentation(signal)

    result = segmentation.variance_based_segmentation(3, 2.0)
    expected = [np.array([1, 2, 2, 2, 5, 6]), np.array([1, 1, 1, 8])]

    # Check if the result matches expected segments
    assert len(result) == len(expected)
    assert all(np.array_equal(r, e) for r, e in zip(result, expected))


def test_peak_based_segmentation():
    signal = np.array([1, 2, 1, 2, 1, 2, 1, 8, 1])
    segmentation = SignalSegmentation(signal)
    result = segmentation.peak_based_segmentation(min_distance=2, height=5)
    expected = [np.array([8, 1])]
    assert all(np.array_equal(r, e) for r, e in zip(result, expected))


def test_ml_based_segmentation_kmeans(segmentation):
    result = segmentation.ml_based_segmentation(model="kmeans")
    # We do not assert exact segmentation since kmeans clustering can vary.
    assert isinstance(result, list)


def test_ml_based_segmentation_gmm(segmentation):
    result = segmentation.ml_based_segmentation(model="gmm")
    assert isinstance(result, list)


def test_ml_based_segmentation_decision_tree(segmentation):
    result = segmentation.ml_based_segmentation(model="decision_tree")
    assert isinstance(result, list)


def test_ml_based_segmentation_spectral():
    signal = np.array([1, 2, 2, 2, 5, 6, 1, 1, 8, 1])  # Increased number of data points
    segmentation = SignalSegmentation(signal)

    result = segmentation.ml_based_segmentation(model="spectral")

    # Check if the result is a list and not empty
    assert isinstance(result, list)
    assert len(result) > 0


def test_ml_based_segmentation_invalid_model(segmentation):
    with pytest.raises(ValueError):
        segmentation.ml_based_segmentation(model="invalid_model")


def test_custom_segmentation(segmentation):
    def custom_fn(signal):
        return np.array([0, 3, 6, 9])

    result = segmentation.custom_segmentation(custom_fn)
    expected = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    assert all(np.array_equal(r, e) for r, e in zip(result, expected))
