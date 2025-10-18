import pytest
import numpy as np
from vitalDSP.utils.config_utilities.common import pearsonr, coherence, grangercausalitytests
from vitalDSP.physiological_features.cross_signal_analysis import CrossSignalAnalysis


@pytest.fixture
def test_signals():
    np.random.seed(42)
    signal1 = np.sin(np.linspace(0, 10, 100))
    signal2 = np.cos(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    return signal1, signal2


@pytest.fixture
def analysis(test_signals):
    signal1, signal2 = test_signals
    return CrossSignalAnalysis(signal1, signal2, fs=100)


def test_compute_correlation(analysis):
    correlation = analysis.compute_correlation()
    assert isinstance(correlation, float)
    assert -1 <= correlation <= 1


def test_compute_cross_correlation(analysis):
    lags, cross_corr = analysis.compute_cross_correlation(max_lag=10)
    assert isinstance(lags, np.ndarray)
    assert isinstance(cross_corr, np.ndarray)
    assert len(lags) == len(cross_corr)
    assert len(lags) == 21  # max_lag * 2 + 1


def test_compute_coherence(analysis):
    freqs, coh = analysis.compute_coherence(nperseg=256)
    assert isinstance(freqs, np.ndarray)
    assert isinstance(coh, np.ndarray)
    assert freqs.shape == coh.shape


def test_compute_phase_synchronization(analysis):
    psi = analysis.compute_phase_synchronization()
    assert isinstance(psi, float)
    assert 0 <= psi <= 1


def test_compute_mutual_information(analysis):
    mutual_info = analysis.compute_mutual_information(bins=10)
    assert isinstance(mutual_info, float)
    assert mutual_info >= 0


def test_compute_granger_causality(analysis):
    gc_result = analysis.compute_granger_causality(max_lag=10)
    assert isinstance(gc_result, dict)
    assert "signal1->signal2" in gc_result
    assert "signal2->signal1" in gc_result
    assert isinstance(gc_result["signal1->signal2"], float)
    assert isinstance(gc_result["signal2->signal1"], float)
