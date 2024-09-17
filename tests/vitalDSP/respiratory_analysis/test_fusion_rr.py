import numpy as np
import pytest
from vitalDSP.respiratory_analysis.fusion.ppg_ecg_fusion import ppg_ecg_fusion
from vitalDSP.respiratory_analysis.fusion.multimodal_analysis import multimodal_analysis
from vitalDSP.respiratory_analysis.fusion.respiratory_cardiac_fusion import (
    respiratory_cardiac_fusion,
)


def generate_multimodal_signals(sampling_rate, duration, noise_level=0.1):
    t = np.arange(0, duration, 1 / sampling_rate)
    signal1 = np.sin(2 * np.pi * 0.25 * t) + noise_level * np.random.normal(size=len(t))
    signal2 = np.sin(2 * np.pi * 0.27 * t) + noise_level * np.random.normal(size=len(t))
    return signal1, signal2


def test_ppg_ecg_fusion():
    ppg_signal, ecg_signal = generate_multimodal_signals(sampling_rate=100, duration=60)
    rr_fusion = ppg_ecg_fusion(ppg_signal, ecg_signal, sampling_rate=100)
    assert 12 <= rr_fusion <= 20


def test_multimodal_analysis():
    signals = generate_multimodal_signals(sampling_rate=100, duration=60)
    rr_multimodal = multimodal_analysis(list(signals), sampling_rate=100)
    assert 12 <= rr_multimodal


def test_respiratory_cardiac_fusion():
    resp_signal, cardiac_signal = generate_multimodal_signals(
        sampling_rate=100, duration=60
    )
    rr_fusion = respiratory_cardiac_fusion(
        resp_signal, cardiac_signal, sampling_rate=100
    )
    assert 12 <= rr_fusion <= 40


if __name__ == "__main__":
    pytest.main()
