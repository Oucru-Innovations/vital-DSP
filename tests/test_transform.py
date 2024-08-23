import numpy as np
import pytest
from transforms.chroma_stft import ChromaSTFT
from transforms.dct_wavelet_fusion import DCTWaveletFusion
from transforms.discrete_cosine_transform import DiscreteCosineTransform
from transforms.fourier_transform import FourierTransform
from transforms.hilbert_transform import HilbertTransform
from transforms.event_related_potential import EventRelatedPotential
from transforms.mfcc import MFCC
from transforms.pca_ica_signal_decomposition import PCASignalDecomposition
from transforms.stft import STFT
from transforms.wavelet_transform import WaveletTransform
from transforms.wavelet_fft_fusion import WaveletFFTfusion
from transforms.time_freq_representation import TimeFreqRepresentation

@pytest.fixture
def sample_signal():
    """Fixture for creating a sample signal."""
    return np.sin(np.linspace(0, 2 * np.pi, 100)) + np.random.normal(0, 0.1, 100)

def test_fourier_transform(sample_signal):
    transformer = FourierTransform(sample_signal)
    transformed_signal = transformer.compute_dft()
    inversed_signal = transformer.compute_idft(transformed_signal)
    assert len(transformed_signal) == len(sample_signal), "Fourier transform length mismatch"
    assert len(inversed_signal) == len(sample_signal), "Fourier transform length mismatch"

def test_dct(sample_signal):
    transformer = DiscreteCosineTransform(sample_signal)
    transformed_signal = transformer.compute_dct()
    inversed_signal = transformer.compute_idct(transformed_signal)
    assert len(transformed_signal) == (len(sample_signal)), "DCT transform length mismatch"
    assert len(inversed_signal) == len(sample_signal), "inverse DCT transform length mismatch"

def test_hilbert_transform(sample_signal):
    transformer = HilbertTransform(sample_signal)
    transformed_signal = transformer.compute_hilbert()
    assert len(transformed_signal) == len(sample_signal), "Hilbert transform length mismatch"
    assert np.all(np.iscomplex(transformed_signal)), "Hilbert transform should return complex values"

def test_wavelet_transform():
    signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    wavelet_transform = WaveletTransform(signal, wavelet_name='haar')
    coeffs = wavelet_transform.perform_wavelet_transform(level=3)
    
    assert isinstance(coeffs, list)
    assert len(coeffs) == 4  # 3 levels + final approximation
    
    for c in coeffs:
        assert isinstance(c, np.ndarray)
    
    reconstructed_signal = wavelet_transform.perform_inverse_wavelet_transform(coeffs)
    
    assert len(reconstructed_signal) == len(signal)
    np.testing.assert_almost_equal(reconstructed_signal, signal, decimal=5)

# def test_wavelet_transform():
#     # Example signal: a simple sine wave with noise
#     signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    
#     # Initialize the WaveletTransform with the 'haar' wavelet
#     wavelet_transform = WaveletTransform(signal, wavelet_name='haar')
    
#     # Perform the wavelet transform
#     coeffs = wavelet_transform.perform_wavelet_transform(level=3)
    
#     # Check that the coefficients are a list and not empty
#     assert isinstance(coeffs, list), "Coefficients should be a list"
#     assert len(coeffs) == 4, "There should be 4 sets of coefficients (3 levels + final approximation)"
    
#     # Check that each set of coefficients is a numpy array
#     for c in coeffs:
#         assert isinstance(c, np.ndarray), "Each coefficient should be a numpy array"
    
#     # Perform the inverse transform to reconstruct the signal
#     reconstructed_signal = wavelet_transform.perform_inverse_wavelet_transform(coeffs)
    
#     # Check that the reconstructed signal has the same length as the original
#     assert len(reconstructed_signal) == len(signal), "Reconstructed signal should have the same length as the original"
    
#     # Check that the reconstructed signal is close to the original signal
#     np.testing.assert_almost_equal(reconstructed_signal, signal, decimal=5, err_msg="Reconstructed signal should closely match the original signal")

def test_stft(sample_signal):
    transformer = STFT(sample_signal, window_size=50, hop_size=25)
    transformed_signal = transformer.compute_stft()
    assert transformed_signal.shape[0] > 0, "STFT should produce non-empty output"

def test_mfcc(sample_signal):
    transformer = MFCC(sample_signal, num_coefficients=13)
    mfccs = transformer.compute_mfcc()
    assert mfccs.shape[0] > 0, "MFCC transform should return correct number of coefficients"

def test_chroma_stft(sample_signal):
    chroma = ChromaSTFT.compute_chroma_stft(sample_signal, n_chroma=12)
    assert chroma.shape[0] == 12, "Chroma STFT should return correct number of chroma bands"

def test_event_related_potential(sample_signal):
    erp = EventRelatedPotential.compute_erp(sample_signal, event_indices=[20, 40, 60], pre_event=5, post_event=5)
    assert erp.shape[0] == 3, "ERP should return correct number of events"

def test_time_freq_representation(sample_signal):
    tf_representation = TimeFreqRepresentation.compute_tfr(sample_signal)
    assert tf_representation.shape[0] > 0, "Time-frequency representation should produce non-empty output"

def test_wavelet_fft_fusion(sample_signal):
    fused_signal = WaveletFFTfusion.compute_fusion(sample_signal)
    assert len(fused_signal) == len(sample_signal), "Wavelet-FFT fusion length mismatch"

def test_dct_wavelet_fusion(sample_signal):
    fused_signal = DCTWaveletFusion.compute_fusion(sample_signal)
    assert len(fused_signal) == len(sample_signal), "DCT-Wavelet fusion length mismatch"

def test_pca_ica_signal_decomposition(sample_signal):
    decomposed_signal = PCASignalDecomposition.compute_pca(sample_signal, n_components=2)
    assert decomposed_signal.shape[0] == 2, "PCA/ICA decomposition should return correct number of components"

if __name__ == "__main__":
    pytest.main()
