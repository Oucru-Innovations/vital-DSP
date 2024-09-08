import pytest
import numpy as np
from vitalDSP.advanced_computation.emd import EMD

# Test fixture for generating a sample signal
@pytest.fixture
def test_signal():
    np.random.seed(42)
    return np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)

@pytest.fixture
def emd_instance(test_signal):
    return EMD(test_signal)

def test_init_invalid_signal_type():
    # Test that a ValueError is raised when signal is not a numpy array
    with pytest.raises(ValueError, match="The input signal must be a numpy.ndarray."):
        EMD(signal=[1, 2, 3, 4])  # Non-numpy array input

def test_init_invalid_signal_dimension():
    # Test that a ValueError is raised when signal is not 1D
    with pytest.raises(ValueError, match="The input signal must be one-dimensional."):
        EMD(signal=np.array([[1, 2], [3, 4]]))  # 2D array input

def test_emd_basic_functionality(emd_instance):
    # Test the basic functionality of the EMD method
    imfs = emd_instance.emd(max_imfs=2, stop_criterion=0.05)
    assert isinstance(imfs, list)
    assert len(imfs) <= 2  # Ensure no more than 2 IMFs are returned
    assert all(isinstance(imf, np.ndarray) for imf in imfs)

def test_emd_full_decomposition(emd_instance):
    # Test full decomposition without limiting the number of IMFs
    imfs = emd_instance.emd(stop_criterion=0.05)
    assert isinstance(imfs, list)
    assert len(imfs) > 0  # Ensure that some IMFs are returned
    final_residual = emd_instance.signal - sum(imfs)
    assert np.all(np.abs(final_residual) < 0.05)  # Ensure residual is small


def test_find_peaks(emd_instance):
    # Test the _find_peaks method
    signal = np.array([0, 1, 0, 1, 0, 1, 0])  # Simple signal with alternating peaks
    peaks = emd_instance._find_peaks(signal)
    assert isinstance(peaks, np.ndarray)
    assert np.array_equal(peaks, np.array([1, 3, 5]))  # Peaks at indices 1, 3, 5

def test_interpolate(emd_instance):
    # Test the _interpolate method
    x = np.array([0, 50, 99])  # Sample x coordinates (indices)
    y = np.array([1, -1, 1])   # Sample y coordinates (values)
    interpolated = emd_instance._interpolate(x, y)
    assert isinstance(interpolated, np.ndarray)
    assert len(interpolated) == len(emd_instance.signal)  # Ensure correct length
    assert interpolated[0] == 1  # First point should match y[0]
    assert interpolated[50] == -1  # Middle point should match y[1]
    assert interpolated[99] == 1  # Last point should match y[2]

def test_emd_residual_signal(emd_instance):
    # Test that the residual signal (after IMF extraction) is below the stop criterion
    imfs = emd_instance.emd(stop_criterion=0.1)
    final_residual = emd_instance.signal - sum(imfs)
    assert np.all(np.abs(final_residual) < 0.1)  # Check if final residual is small

def test_emd_stop_criterion_reached(emd_instance):
    # Test that the stop criterion is reached for each IMF
    imfs = emd_instance.emd(max_imfs=1, stop_criterion=0.01)
    assert len(imfs) == 1  # Ensure only 1 IMF was returned
