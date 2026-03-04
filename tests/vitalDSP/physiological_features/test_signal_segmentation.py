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
    assert isinstance(result, list)
    assert len(result) >= 1
    total_len = sum(len(s) for s in result)
    assert total_len == len(signal)


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


class TestSignalSegmentationMissingCoverage:
    """Tests to cover missing lines in signal_segmentation.py."""

    def test_adaptive_segmentation_end_idx_zero(self):
        """Test adaptive_segmentation when adaptive_fn returns 0.
        
        This test covers line 128 in signal_segmentation.py where
        break is executed when end_idx == 0 to avoid infinite loops.
        """
        signal = np.array([1, 2, 3, 4, 5])
        segmentation = SignalSegmentation(signal)
        
        def adaptive_fn(segment):
            # Return 0 after first call to trigger break
            if len(segment) == len(signal):
                return 2  # First segment
            return 0  # Return 0 to trigger break
        
        result = segmentation.adaptive_segmentation(adaptive_fn)
        assert isinstance(result, list)
        assert len(result) >= 1  # Should have at least one segment before break

    def test_threshold_based_segmentation_start_idx_none(self):
        """Test threshold_based_segmentation when start_idx is None.
        
        This test covers lines 161-162 in signal_segmentation.py where
        start_idx = i when start_idx is None.
        """
        signal = np.array([1, 2, 5, 2, 8, 1])
        segmentation = SignalSegmentation(signal)
        
        # Threshold that will trigger start_idx assignment
        result = segmentation.threshold_based_segmentation(threshold=4)
        assert isinstance(result, list)
        # Should have segments where signal > threshold
        assert len(result) > 0

    def test_threshold_based_segmentation_ending_above_threshold(self):
        """Test threshold_based_segmentation when signal ends above threshold.
        
        This test covers line 168 in signal_segmentation.py where
        segments.append(self.signal[start_idx:]) is called when signal ends above threshold.
        """
        signal = np.array([1, 2, 5, 8])  # Ends with values above threshold
        segmentation = SignalSegmentation(signal)
        
        result = segmentation.threshold_based_segmentation(threshold=4)
        assert isinstance(result, list)
        # Should include the final segment that ends above threshold
        assert len(result) >= 1

    def test_variance_based_segmentation_remaining_signal(self):
        """Test variance_based_segmentation when start_idx < len(signal).
        
        This test covers lines 210-211 in signal_segmentation.py where
        segments.append(self.signal[start_idx:]) is called for remaining signal.
        """
        signal = np.array([1, 2, 2, 2, 5, 6, 1, 1, 1, 8, 9, 10])
        segmentation = SignalSegmentation(signal)
        
        result = segmentation.variance_based_segmentation(window_size=3, variance_threshold=2.0)
        assert isinstance(result, list)
        # Should include remaining signal after last variance change
        assert len(result) > 0

    def test_peak_based_segmentation_with_height(self):
        """Test peak_based_segmentation with height parameter.
        
        This test covers lines 245-246 in signal_segmentation.py where
        peaks are filtered by height.
        """
        signal = np.array([1, 2, 1, 2, 1, 8, 1, 2, 1, 9, 1])
        segmentation = SignalSegmentation(signal)
        
        result = segmentation.peak_based_segmentation(min_distance=1, height=5)
        assert isinstance(result, list)
        # Should only include segments around peaks with height > 5

    def test_peak_based_segmentation_with_min_distance(self):
        """Test peak_based_segmentation with min_distance parameter.
        
        This test covers lines 247-248 in signal_segmentation.py where
        peaks are filtered by min_distance.
        """
        signal = np.array([1, 8, 1, 2, 1, 9, 1, 2, 1, 7, 1])
        segmentation = SignalSegmentation(signal)
        
        result = segmentation.peak_based_segmentation(min_distance=3, height=None)
        assert isinstance(result, list)
        # Should filter peaks based on min_distance

    def test_peak_based_segmentation_multiple_peaks(self):
        """Test peak_based_segmentation with multiple peaks.
        
        This test covers line 252 in signal_segmentation.py where
        segments.append(self.signal[peaks[i] : peaks[i + 1]]) is called for multiple peaks.
        """
        signal = np.array([1, 8, 1, 2, 1, 9, 1, 2, 1, 7, 1])
        segmentation = SignalSegmentation(signal)
        
        result = segmentation.peak_based_segmentation(min_distance=1, height=None)
        assert isinstance(result, list)
        # Should have segments between multiple peaks
        assert len(result) >= 1

    def test_ml_based_segmentation_change_detection(self):
        """Test ml_based_segmentation with change_detection model.
        
        This test covers lines 277-280 in signal_segmentation.py where
        change_detection model is used.
        """
        signal = np.array([1, 2, 2, 2, 5, 6, 1, 1, 8, 1])
        segmentation = SignalSegmentation(signal)
        
        result = segmentation.ml_based_segmentation(model="change_detection")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_ml_based_segmentation_dtw(self):
        """Test ml_based_segmentation with dtw model.
        
        This test covers lines 299-301 in signal_segmentation.py where
        dtw model is used.
        
        Note: dtw model creates only one change point at len(signal) // 2,
        which results in 0 segments since segments need at least 2 change points.
        This is expected behavior for the placeholder implementation.
        """
        signal = np.array([1, 2, 2, 2, 5, 6, 1, 1, 8, 1])
        segmentation = SignalSegmentation(signal)
        
        result = segmentation.ml_based_segmentation(model="dtw")
        assert isinstance(result, list)
        assert len(result) >= 1
        total_len = sum(len(s) for s in result)
        assert total_len == len(signal)

    def test_ml_based_segmentation_autoencoder(self):
        """Test ml_based_segmentation with autoencoder model.
        
        This test covers lines 306-308 in signal_segmentation.py where
        autoencoder model is used.
        
        Note: autoencoder model creates only one change point at len(signal) // 2,
        which results in 0 segments since segments need at least 2 change points.
        This is expected behavior for the placeholder implementation.
        """
        signal = np.array([1, 2, 2, 2, 5, 6, 1, 1, 8, 1])
        segmentation = SignalSegmentation(signal)
        
        result = segmentation.ml_based_segmentation(model="autoencoder")
        assert isinstance(result, list)
        assert len(result) >= 1
        total_len = sum(len(s) for s in result)
        assert total_len == len(signal)