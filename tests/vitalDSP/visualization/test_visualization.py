import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from vitalDSP.filtering.signal_filtering import SignalFiltering  # Assuming the filtering functions are implemented
from vitalDSP.visualization.filtering_visualization import FilteringVisualization  # Assuming this is the location of the class
import pytest
from vitalDSP.transforms.pca_ica_signal_decomposition import PCASignalDecomposition, ICASignalDecomposition
from vitalDSP.transforms.dct_wavelet_fusion import DCTWaveletFusion
from vitalDSP.transforms.wavelet_fft_fusion import WaveletFFTfusion
from vitalDSP.transforms.stft import STFT
from vitalDSP.transforms.mfcc import MFCC
from vitalDSP.transforms.chroma_stft import ChromaSTFT
from vitalDSP.transforms.event_related_potential import EventRelatedPotential
from vitalDSP.transforms.time_freq_representation import TimeFreqRepresentation
from vitalDSP.visualization.transform_visualization import SignalDecompositionVisualization, \
    DCTWaveletFusionVisualization, STFTVisualization, WaveletFFTfusionVisualization, \
    MFCCVisualization, ChromaSTFTVisualization, ERPVisualization, TFRVisualization

@pytest.fixture
def mock_signal():
    """Fixture for creating a mock signal."""
    return np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)

@pytest.fixture
def mock_visualization(mock_signal):
    """Fixture for initializing FilteringVisualization with a mock signal."""
    return FilteringVisualization(mock_signal)

@pytest.fixture
def mock_filtering(mock_visualization):
    """Fixture for mocking the filtering methods."""
    mock_visualization.filtering = MagicMock(SignalFiltering)
    
    # Return a valid numpy array for each mock filter
    filtered_signal = np.sin(np.linspace(0, 10, 100))  # Example filtered signal
    
    mock_visualization.filtering.moving_average.return_value = filtered_signal
    mock_visualization.filtering.gaussian.return_value = filtered_signal
    mock_visualization.filtering.butterworth.return_value = filtered_signal
    mock_visualization.filtering.median.return_value = filtered_signal
    
    return mock_visualization

@patch("plotly.graph_objs.Figure.show")
def test_visualize_moving_average(mock_show, mock_filtering):
    """Test visualize_moving_average method."""
    mock_filtering.visualize_moving_average(window_size=5)
    mock_filtering.filtering.moving_average.assert_called_once_with(5)
    assert mock_show.called

@patch("plotly.graph_objs.Figure.show")
def test_visualize_gaussian_filter(mock_show, mock_filtering):
    """Test visualize_gaussian_filter method."""
    mock_filtering.visualize_gaussian_filter(sigma=1.0)
    mock_filtering.filtering.gaussian.assert_called_once_with(1.0)
    assert mock_show.called

@patch("plotly.graph_objs.Figure.show")
def test_visualize_butterworth_filter(mock_show, mock_filtering):
    """Test visualize_butterworth_filter method."""
    mock_filtering.visualize_butterworth_filter(cutoff=0.3, order=2, fs=100)
    mock_filtering.filtering.butterworth.assert_called_once_with(0.3, 2, 100)
    assert mock_show.called

@patch("plotly.graph_objs.Figure.show")
def test_visualize_median_filter(mock_show, mock_filtering):
    """Test visualize_median_filter method."""
    mock_filtering.visualize_median_filter(kernel_size=5)
    mock_filtering.filtering.median.assert_called_once_with(5)
    assert mock_show.called

@patch("plotly.graph_objs.Figure.show")
def test_visualize_all_filters(mock_show, mock_filtering):
    """Test visualize_all_filters method."""
    mock_filtering.visualize_all_filters(
        window_size=5, sigma=1.0, cutoff=0.5, order=2, fs=1000, kernel_size=5
    )
    
    mock_filtering.filtering.moving_average.assert_called_once_with(5)
    mock_filtering.filtering.gaussian.assert_called_once_with(1.0)
    mock_filtering.filtering.butterworth.assert_called_once_with(0.5, 2, 1000)
    mock_filtering.filtering.median.assert_called_once_with(5)
    
    assert mock_show.called

@patch("plotly.graph_objs.Figure.show")
def test_plot_signal(mock_show, mock_filtering):
    """Test internal _plot_signal method."""
    filtered_signal = np.sin(np.linspace(0, 10, 100))  # Mock filtered signal
    mock_filtering._plot_signal(filtered_signal, title="Test Title")
    assert mock_show.called

#==========================================================================================
#                               Transform Visualization
#==========================================================================================
@patch("plotly.graph_objs.Figure.show")
def test_signal_decomposition_pca(mock_show):
    signals = np.random.rand(100, 5)  # Shape: (100 samples, 5 signals)
    viz = SignalDecompositionVisualization(signals)
    viz.plot_pca()
    mock_show.assert_called_once()  # Ensure that plot was generated

@patch("plotly.graph_objs.Figure.show")
def test_signal_decomposition_ica(mock_show):
    signals = np.random.rand(100, 5)  # Shape: (100 samples, 5 signals)
    viz = SignalDecompositionVisualization(signals)
    viz.plot_ica()
    mock_show.assert_called_once()  # Ensure that plot was generated

@patch("plotly.graph_objs.Figure.show")
def test_dct_wavelet_fusion(mock_show):
    signal = np.sin(np.linspace(0, 10, 1000))
    viz = DCTWaveletFusionVisualization(signal)
    viz.plot_fusion()
    mock_show.assert_called_once()  # Ensure that plot was generated
    viz.compare_original_fusion()
    assert isinstance(viz.fusion, DCTWaveletFusion)  # Check DCTWaveletFusion object creation

@patch("plotly.graph_objs.Figure.show")
def test_wavelet_fft_fusion(mock_show):
    signal = np.sin(np.linspace(0, 10, 1000))
    viz = WaveletFFTfusionVisualization(signal)
    viz.plot_fusion()
    mock_show.assert_called_once()  # Ensure that plot was generated
    viz.compare_original_fusion()
    assert isinstance(viz.fusion, WaveletFFTfusion)  # Check WaveletFFTfusion object creation

@patch("plotly.graph_objs.Figure.show")
def test_stft_visualization(mock_show):
    signal = np.sin(np.linspace(0, 10, 1000))
    viz = STFTVisualization(signal)
    viz.plot_stft()
    mock_show.assert_called_once()  # Ensure that plot was generated
    viz.compare_original_stft()
    assert isinstance(viz.stft, STFT)  # Check STFT object creation

@patch("plotly.graph_objs.Figure.show")
def test_mfcc_visualization(mock_show):
    signal = np.sin(np.linspace(0, 10, 1000))
    mfcc_viz = MFCCVisualization(signal)
    mfcc_viz.plot_mfcc()
    mock_show.assert_called_once()  # Ensure that plot was generated
    mfcc_viz.compare_original_mfcc()
    assert isinstance(mfcc_viz.mfcc, MFCC)  # Check MFCC object creation

@patch("plotly.graph_objs.Figure.show")
def test_chroma_stft_visualization(mock_show):
    signal = np.sin(np.linspace(0, 10, 1000))
    chroma_stft_viz = ChromaSTFTVisualization(signal,n_fft=128)
    chroma_stft_viz.plot_chroma_stft()
    mock_show.assert_called_once()  # Ensure that plot was generated
    chroma_stft_viz.compare_original_chroma_stft()
    assert isinstance(chroma_stft_viz.chroma_stft, ChromaSTFT)  # Check ChromaSTFT object creation

@patch("plotly.graph_objs.Figure.show")
def test_erp_visualization(mock_show):
    signal = np.sin(np.linspace(0, 10, 1000))
    stimulus_times = np.array([100, 300, 500])
    erp_viz = ERPVisualization(signal, stimulus_times)
    erp_viz.plot_erp()
    mock_show.assert_called_once()  # Ensure that plot was generated
    erp_viz.compare_original_erp()
    assert isinstance(erp_viz.erp, EventRelatedPotential)  # Check ERP object creation
    
    
@patch("plotly.graph_objs.Figure.show")
def test_time_freq_representation(mock_show):
    signal = np.sin(np.linspace(0, 10, 1000))
    tfr_viz = TFRVisualization(signal, method="stft")
    tfr_viz.plot_tfr()
    mock_show.assert_called_once()  # Ensure that plot was generated
    tfr_viz.compare_original_tfr()
    assert isinstance(tfr_viz.tfr, TimeFreqRepresentation)  # Check TimeFreqRepresentation object creation