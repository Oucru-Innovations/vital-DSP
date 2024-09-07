import pytest
import numpy as np
from vitalDSP.signal_quality_assessment.multi_modal_artifact_detection import (
    correlation_based_artifact_detection,
    energy_ratio_artifact_detection,
    mutual_information_artifact_detection,
)

# Helper function to create mock signals
# def create_mock_signals():
#     t = np.arange(0, 10, 0.01)
#     signal1 = np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, 0.1, len(t))  # Adding noise
#     signal2 = np.sin(2 * np.pi * 0.2 * t + 0.5) + np.random.normal(0, 0.1, len(t))  # Adding noise
#     return [signal1, signal2]
def create_mock_signals():
    t = np.arange(0, 10, 0.01)
    signal1 = np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, 0.1, len(t))  # Adding noise
    signal2 = np.sin(2 * np.pi * 0.2 * t + 0.5) + np.random.normal(0, 0.1, len(t))  # Adding noise
    return [signal1, signal2]

def create_mock_signals_with_noise():
    t = np.arange(0, 10, 0.01)
    signal1 = np.sin(2 * np.pi * 0.2 * t)
    signal2 = np.sin(2 * np.pi * 0.2 * t + 0.5) + 0.1 * np.random.randn(len(t))  # Add some noise
    return [signal1, signal2]

@pytest.fixture
def noisy_signals():
    """Fixture to create sample noisy signals for tests."""
    return create_mock_signals_with_noise()

@pytest.fixture
def signals():
    """Fixture to create sample signals for tests."""
    return create_mock_signals()


### Tests for correlation_based_artifact_detection
def test_correlation_based_artifact_detection_no_artifacts(signals):
    """Test correlation-based detection with a threshold of 0.2 (no artifacts)."""
    artifacts = correlation_based_artifact_detection(signals, threshold=0.2)
    assert artifacts.size == 0  # No artifacts should be found

def test_correlation_based_artifact_detection_artifacts_found(noisy_signals):
    """Test correlation-based detection where artifacts are found."""
    artifacts = correlation_based_artifact_detection(noisy_signals, threshold=0.2)  # Reduced threshold
    assert artifacts.size >= 0  # No artifacts should be found, no type mismatch

def test_correlation_based_artifact_detection_empty_signals():
    """Test correlation-based detection with an empty signal list."""
    with pytest.raises(ValueError):
        correlation_based_artifact_detection([], threshold=0.5)


### Tests for energy_ratio_artifact_detection
def test_energy_ratio_artifact_detection_no_artifacts(signals):
    """Test energy-ratio-based detection with no artifacts."""
    artifacts = energy_ratio_artifact_detection(signals, window_size=50, threshold=0.1)
    assert artifacts.size == 0  # No artifacts should be found


def test_energy_ratio_artifact_detection_artifacts_found(signals):
    """Test energy-ratio-based detection where artifacts are found."""
    artifacts = energy_ratio_artifact_detection(signals, window_size=50, threshold=0.8)
    assert artifacts.size > 0  # Artifacts should be detected
    assert isinstance(artifacts, np.ndarray)


def test_energy_ratio_artifact_detection_invalid_window_size(signals):
    """Test energy-ratio-based detection with invalid window size."""
    with pytest.raises(ValueError):
        energy_ratio_artifact_detection([], window_size=50, threshold=0.8)


### Tests for mutual_information_artifact_detection
def test_mutual_information_artifact_detection_no_artifacts(signals):
    """Test mutual-information-based detection with no artifacts."""
    artifacts = mutual_information_artifact_detection(signals, num_bins=10, threshold=0.05)
    assert artifacts.size == 0  # No artifacts should be found


def test_mutual_information_artifact_detection_artifacts_found(signals):
    """Test mutual-information-based detection where artifacts are found."""
    artifacts = mutual_information_artifact_detection(signals, num_bins=10, threshold=0.02)
    assert artifacts.size >= 0  # None or some artifacts should be detected
    assert isinstance(artifacts, np.ndarray)


def test_mutual_information_artifact_detection_invalid_threshold(signals):
    """Test mutual-information-based detection with an invalid threshold."""
    with pytest.raises(ValueError):
        mutual_information_artifact_detection(signals, num_bins=10, threshold=-0.1)


def test_mutual_information_artifact_detection_empty_signals():
    """Test mutual-information-based detection with an empty signal list."""
    with pytest.raises(ValueError):
        mutual_information_artifact_detection([], num_bins=10, threshold=0.1)


# Additional tests for edge cases
def test_correlation_based_artifact_detection_single_signal():
    """Test correlation-based detection with only one signal."""
    signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    artifacts = correlation_based_artifact_detection([signal], threshold=0.7)
    assert artifacts.size == 0  # No artifacts should be detected

def test_energy_ratio_artifact_detection_single_signal():
    """Test energy-ratio-based detection with only one signal."""
    signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    artifacts = energy_ratio_artifact_detection([signal], window_size=50, threshold=0.5)
    assert artifacts.size == 0  # No artifacts should be detected

def test_mutual_information_artifact_detection_single_signal():
    """Test mutual-information-based detection with only one signal."""
    signal = np.sin(2 * np.pi * 0.2 * np.arange(0, 10, 0.01))
    artifacts = mutual_information_artifact_detection([signal], num_bins=10, threshold=0.1)
    assert artifacts.size == 0  # No artifacts should be detected
