import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from vitalDSP.transforms.wavelet_fft_fusion import WaveletFFTfusion


@pytest.fixture
def sample_signal():
    """Create a sample signal for testing"""
    np.random.seed(42)
    return np.sin(np.linspace(0, 10, 1000))


def test_init_default_params(sample_signal):
    """Test initialization with default parameters"""
    fusion = WaveletFFTfusion(sample_signal)
    assert np.array_equal(fusion.signal, sample_signal)
    assert fusion.wavelet_type == "db"
    assert fusion.order == 4


def test_init_custom_params(sample_signal):
    """Test initialization with custom parameters"""
    fusion = WaveletFFTfusion(sample_signal, wavelet_type="haar", order=3)
    assert fusion.wavelet_type == "haar"
    assert fusion.order == 3


def test_compute_fusion(sample_signal):
    """Test the fusion computation"""
    fusion = WaveletFFTfusion(sample_signal, wavelet_type="db", order=4)
    fusion_result = fusion.compute_fusion()

    assert isinstance(fusion_result, np.ndarray)
    assert len(fusion_result) > 0


def test_compute_fusion_different_wavelets(sample_signal):
    """Test fusion with different wavelet types"""
    wavelets = ["db", "haar", "sym"]

    for wavelet in wavelets:
        fusion = WaveletFFTfusion(sample_signal, wavelet_type=wavelet, order=4)
        fusion_result = fusion.compute_fusion()
        assert isinstance(fusion_result, np.ndarray)


def test_compute_fusion_fft_longer(sample_signal):
    """Test fusion when FFT coefficients are longer than wavelet coefficients"""
    # Create a short signal that results in fewer wavelet coefficients
    short_signal = sample_signal[:100]
    fusion = WaveletFFTfusion(short_signal, wavelet_type="db", order=6)
    fusion_result = fusion.compute_fusion()

    assert isinstance(fusion_result, np.ndarray)
    assert len(fusion_result) > 0


def test_compute_fusion_wavelet_longer():
    """Test fusion when wavelet coefficients are longer than FFT coefficients"""
    # Create a signal and parameters where wavelet coefficients might be longer
    np.random.seed(42)
    signal = np.sin(np.linspace(0, 10, 50))
    fusion = WaveletFFTfusion(signal, wavelet_type="db", order=2)
    fusion_result = fusion.compute_fusion()

    assert isinstance(fusion_result, np.ndarray)
    assert len(fusion_result) > 0


def test_fusion_result_complex(sample_signal):
    """Test that fusion result contains complex numbers (from FFT)"""
    fusion = WaveletFFTfusion(sample_signal, wavelet_type="db", order=4)
    fusion_result = fusion.compute_fusion()

    # Fusion result should be complex since FFT produces complex coefficients
    assert np.iscomplexobj(fusion_result)


def test_different_orders(sample_signal):
    """Test fusion with different wavelet orders"""
    for order in [2, 3, 4, 5]:
        fusion = WaveletFFTfusion(sample_signal, wavelet_type="db", order=order)
        fusion_result = fusion.compute_fusion()
        assert isinstance(fusion_result, np.ndarray)
        assert len(fusion_result) > 0


def test_compute_fusion_wavelet_coeffs_longer_than_fft():
    """Test fusion when wavelet coefficients list length is longer than FFT length.
    
    This test covers lines 103-104 in wavelet_fft_fusion.py where
    len(fft_coeffs) < len(wavelet_coeffs).
    """
    # Create a short signal (FFT will have length 10)
    signal = np.sin(np.linspace(0, 10, 10))
    
    # Mock WaveletTransform to return a list with more elements than signal length
    # This simulates the edge case where wavelet decomposition returns more
    # coefficient arrays than the FFT has elements
    mock_wavelet_coeffs = [np.array([1.0, 2.0]) for _ in range(15)]  # 15 elements > 10
    
    fusion = WaveletFFTfusion(signal, wavelet_type="db", order=4)
    
    # Patch the WaveletTransform class and its perform_wavelet_transform method
    with patch('vitalDSP.transforms.wavelet_fft_fusion.WaveletTransform') as mock_wt_class:
        mock_wt_instance = MagicMock()
        mock_wt_instance.perform_wavelet_transform.return_value = mock_wavelet_coeffs
        mock_wt_class.return_value = mock_wt_instance
        
        fusion_result = fusion.compute_fusion()
        
        # Verify that wavelet_coeffs was truncated to match fft_coeffs length
        assert isinstance(fusion_result, np.ndarray)
        assert len(fusion_result) == len(signal)  # Should match FFT length (10)
        
        # Verify that perform_wavelet_transform was called
        mock_wt_instance.perform_wavelet_transform.assert_called_once_with(level=4)
