import numpy as np
import pytest
from transforms.signal_transforms import SignalTransforms

@pytest.fixture
def sample_signal():
    """Fixture for creating a sample signal."""
    return np.sin(np.linspace(0, 2 * np.pi, 100)) + np.random.normal(0, 0.1, 100)

def test_fourier_transform(sample_signal):
    transformed_signal = SignalTransforms.fourier_transform(sample_signal)
    assert len(transformed_signal) == len(sample_signal), "Fourier transform length mismatch"

def test_dct(sample_signal):
    transformed_signal = SignalTransforms.dct(sample_signal)
    assert len(transformed_signal) == len(sample_signal), "DCT transform length mismatch"

def test_hilbert_transform(sample_signal):
    transformed_signal = SignalTransforms.hilbert_transform(sample_signal)
    assert len(transformed_signal) == len(sample_signal), "Hilbert transform length mismatch"
    assert np.all(np.iscomplex(transformed_signal)), "Hilbert transform should return complex values"

def test_wavelet_transform(sample_signal):
    transformed_signal = SignalTransforms.wavelet_transform(sample_signal, wavelet_name='db4')
    assert len(transformed_signal) == len(sample_signal), "Wavelet transform length mismatch"

def test_stft(sample_signal):
    transformed_signal, _ = SignalTransforms.stft(sample_signal, n_fft=50, hop_length=25)
    assert transformed_signal.shape[0] > 0, "STFT should produce non-empty output"

def test_mfcc(sample_signal):
    mfccs = SignalTransforms.mfcc(sample_signal, n_mfcc=13)
    assert mfccs.shape[0] == 13, "MFCC transform should return correct number of coefficients"

def test_chroma_stft(sample_signal):
    chroma = SignalTransforms.chroma_stft(sample_signal, n_chroma=12)
    assert chroma.shape[0] == 12, "Chroma STFT should return correct number of chroma bands"

def test_event_related_potential(sample_signal):
    erp = SignalTransforms.event_related_potential(sample_signal, event_indices=[20, 40, 60], pre_event=5, post_event=5)
    assert erp.shape[0] == 3, "ERP should return correct number of events"

def test_time_freq_representation(sample_signal):
    tf_representation = SignalTransforms.time_freq_representation(sample_signal)
    assert tf_representation.shape[0] > 0, "Time-frequency representation should produce non-empty output"

def test_wavelet_fft_fusion(sample_signal):
    fused_signal = SignalTransforms.wavelet_fft_fusion(sample_signal)
    assert len(fused_signal) == len(sample_signal), "Wavelet-FFT fusion length mismatch"

def test_dct_wavelet_fusion(sample_signal):
    fused_signal = SignalTransforms.dct_wavelet_fusion(sample_signal)
    assert len(fused_signal) == len(sample_signal), "DCT-Wavelet fusion length mismatch"

def test_pca_ica_signal_decomposition(sample_signal):
    decomposed_signal = SignalTransforms.pca_ica_signal_decomposition(sample_signal, n_components=2)
    assert decomposed_signal.shape[0] == 2, "PCA/ICA decomposition should return correct number of components"

if __name__ == "__main__":
    pytest.main()
