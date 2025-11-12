import pytest
import numpy as np
from vitalDSP.transforms.stft import STFT


@pytest.fixture
def sample_signal():
    """Create a sample signal for testing"""
    np.random.seed(42)
    return np.sin(np.linspace(0, 10, 1000))


def test_init_default_params(sample_signal):
    """Test initialization with default parameters"""
    stft = STFT(sample_signal)
    assert np.array_equal(stft.signal, sample_signal)
    assert stft.window_size == 256
    assert stft.hop_size == 128
    assert stft.n_fft == 512


def test_init_custom_params(sample_signal):
    """Test initialization with custom parameters"""
    stft = STFT(sample_signal, window_size=100, hop_size=50, n_fft=128)
    assert stft.window_size == 100
    assert stft.hop_size == 50
    assert stft.n_fft == 128


def test_validate_parameters_window_size_zero(sample_signal):
    """Test ValueError when window_size is zero.
    
    This test covers lines 84-86 in stft.py where
    ValueError is raised for zero or negative parameters.
    """
    with pytest.raises(ValueError, match="Window size, hop size, and n_fft must be positive integers"):
        STFT(sample_signal, window_size=0, hop_size=128, n_fft=512)


def test_validate_parameters_window_size_negative(sample_signal):
    """Test ValueError when window_size is negative."""
    with pytest.raises(ValueError, match="Window size, hop size, and n_fft must be positive integers"):
        STFT(sample_signal, window_size=-10, hop_size=128, n_fft=512)


def test_validate_parameters_hop_size_zero(sample_signal):
    """Test ValueError when hop_size is zero."""
    with pytest.raises(ValueError, match="Window size, hop size, and n_fft must be positive integers"):
        STFT(sample_signal, window_size=256, hop_size=0, n_fft=512)


def test_validate_parameters_hop_size_negative(sample_signal):
    """Test ValueError when hop_size is negative."""
    with pytest.raises(ValueError, match="Window size, hop size, and n_fft must be positive integers"):
        STFT(sample_signal, window_size=256, hop_size=-10, n_fft=512)


def test_validate_parameters_n_fft_zero(sample_signal):
    """Test ValueError when n_fft is zero."""
    with pytest.raises(ValueError, match="Window size, hop size, and n_fft must be positive integers"):
        STFT(sample_signal, window_size=256, hop_size=128, n_fft=0)


def test_validate_parameters_n_fft_negative(sample_signal):
    """Test ValueError when n_fft is negative."""
    with pytest.raises(ValueError, match="Window size, hop size, and n_fft must be positive integers"):
        STFT(sample_signal, window_size=256, hop_size=128, n_fft=-10)


def test_validate_parameters_all_zero(sample_signal):
    """Test ValueError when all parameters are zero."""
    with pytest.raises(ValueError, match="Window size, hop size, and n_fft must be positive integers"):
        STFT(sample_signal, window_size=0, hop_size=0, n_fft=0)


def test_validate_parameters_window_size_too_large(sample_signal):
    """Test ValueError when window_size is larger than signal length.
    
    This test covers line 88 in stft.py where
    ValueError is raised when window_size > len(signal).
    """
    signal = np.sin(np.linspace(0, 10, 100))  # Short signal
    with pytest.raises(ValueError, match="Window size cannot be larger than the signal length"):
        STFT(signal, window_size=200, hop_size=50, n_fft=128)


def test_validate_parameters_window_size_equal_to_signal_length(sample_signal):
    """Test that window_size equal to signal length is allowed."""
    signal = np.sin(np.linspace(0, 10, 100))
    stft = STFT(signal, window_size=100, hop_size=50, n_fft=128)
    assert stft.window_size == 100


def test_compute_stft(sample_signal):
    """Test compute_stft method"""
    stft = STFT(sample_signal, window_size=100, hop_size=50, n_fft=128)
    stft_result = stft.compute_stft()
    
    assert isinstance(stft_result, np.ndarray)
    assert stft_result.shape[0] > 0  # Frequency bins
    assert stft_result.shape[1] > 0  # Time frames
    assert np.iscomplexobj(stft_result)  # Should be complex-valued


def test_compute_stft_small_signal():
    """Test compute_stft with a small signal"""
    signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))
    stft = STFT(signal, window_size=32, hop_size=16, n_fft=64)
    stft_result = stft.compute_stft()
    
    assert isinstance(stft_result, np.ndarray)
    assert stft_result.shape[0] == 33  # n_fft // 2 + 1 = 64 // 2 + 1 = 33
    assert stft_result.shape[1] > 0


def test_compute_stft_window_padding():
    """Test that windowed signal is padded when shorter than n_fft"""
    signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))
    stft = STFT(signal, window_size=32, hop_size=16, n_fft=128)  # n_fft > window_size
    stft_result = stft.compute_stft()
    
    assert isinstance(stft_result, np.ndarray)
    assert stft_result.shape[0] == 65  # n_fft // 2 + 1 = 128 // 2 + 1 = 65


def test_compute_stft_different_parameters(sample_signal):
    """Test compute_stft with different parameter combinations"""
    # Test with larger window
    stft1 = STFT(sample_signal, window_size=512, hop_size=256, n_fft=1024)
    result1 = stft1.compute_stft()
    assert isinstance(result1, np.ndarray)
    
    # Test with smaller hop (more overlap)
    stft2 = STFT(sample_signal, window_size=256, hop_size=64, n_fft=512)
    result2 = stft2.compute_stft()
    assert isinstance(result2, np.ndarray)
    assert result2.shape[1] > result1.shape[1]  # More time frames with smaller hop


def test_compute_stft_single_window():
    """Test compute_stft when signal fits in a single window"""
    signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 50))
    stft = STFT(signal, window_size=50, hop_size=50, n_fft=64)
    stft_result = stft.compute_stft()
    
    assert isinstance(stft_result, np.ndarray)
    assert stft_result.shape[1] == 1  # Single window

