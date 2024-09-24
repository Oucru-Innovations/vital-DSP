import pytest
import numpy as np
from vitalDSP.feature_engineering.ecg_ppg_synchronyzation_features import (
    ECGPPGSynchronization,
)


@pytest.fixture
def generate_ecg_ppg_signal():
    # Generate a simulated ECG and PPG signal for testing purposes
    ecg_signal = np.sin(np.linspace(0, 10 * np.pi, 1000))  # Simulated ECG signal
    ppg_signal = np.sin(np.linspace(0, 10 * np.pi, 1000))  # Simulated PPG signal
    return (
        ecg_signal,
        ppg_signal,
        1000,
        100,
    )  # Return signal with ECG fs=1000 Hz, PPG fs=100 Hz


def test_ptt_computation(generate_ecg_ppg_signal):
    ecg_signal, ppg_signal, ecg_fs, ppg_fs = generate_ecg_ppg_signal
    sync_analyzer = ECGPPGSynchronization(ecg_signal, ppg_signal, ecg_fs, ppg_fs)

    # Simulate R-peaks and systolic peaks for testing
    r_peaks = np.array([100, 300, 500, 700])
    systolic_peaks = np.array([110, 310, 510, 710])

    ptt = sync_analyzer.compute_ptt(r_peaks, systolic_peaks)

    assert isinstance(ptt, float), "PTT should return a float value"
    assert ptt > 0, "PTT should be a positive value"


def test_pat_computation(generate_ecg_ppg_signal):
    ecg_signal, ppg_signal, ecg_fs, ppg_fs = generate_ecg_ppg_signal
    sync_analyzer = ECGPPGSynchronization(ecg_signal, ppg_signal, ecg_fs, ppg_fs)

    # Simulate R-peaks and systolic peaks for testing
    r_peaks = np.array([100, 300, 500, 700])
    systolic_peaks = np.array([120, 320, 520, 720])

    pat = sync_analyzer.compute_pat(r_peaks, systolic_peaks)

    assert isinstance(pat, float), "PAT should return a float value"
    assert pat > 0, "PAT should be a positive value"


def test_emd_computation(generate_ecg_ppg_signal):
    ecg_signal, ppg_signal, ecg_fs, ppg_fs = generate_ecg_ppg_signal
    sync_analyzer = ECGPPGSynchronization(ecg_signal, ppg_signal, ecg_fs, ppg_fs)

    # Simulate R-peaks and systolic peaks for testing
    r_peaks = np.array([100, 300, 500, 700])
    systolic_peaks = np.array([130, 330, 530, 730])

    emd = sync_analyzer.compute_emd(r_peaks, systolic_peaks)

    assert isinstance(emd, float), "EMD should return a float value"
    assert emd > 0, "EMD should be a positive value"


def test_hr_pr_sync(generate_ecg_ppg_signal):
    ecg_signal, ppg_signal, ecg_fs, ppg_fs = generate_ecg_ppg_signal
    sync_analyzer = ECGPPGSynchronization(ecg_signal, ppg_signal, ecg_fs, ppg_fs)

    # Simulate R-peaks and systolic peaks for testing
    r_peaks = np.array([100, 300, 500, 700])
    systolic_peaks = np.array([110, 310, 510, 710])

    hr_pr_sync = sync_analyzer.compute_hr_pr_sync(r_peaks, systolic_peaks)

    assert isinstance(
        hr_pr_sync, float
    ), "HR-PR synchronization should return a float value"
    assert hr_pr_sync > 0, "HR-PR synchronization should be a positive value"


def test_rsa_computation(generate_ecg_ppg_signal):
    ecg_signal, ppg_signal, ecg_fs, ppg_fs = generate_ecg_ppg_signal
    sync_analyzer = ECGPPGSynchronization(ecg_signal, ppg_signal, ecg_fs, ppg_fs)

    # Simulate R-peaks and systolic peaks for testing
    r_peaks = np.array([100, 300, 500, 700])
    systolic_peaks = np.array([110, 310, 510, 710])

    rsa = sync_analyzer.compute_rsa(r_peaks, systolic_peaks)

    assert isinstance(rsa, float), "RSA should return a float value"
    assert rsa > 0, "RSA should be a positive value"


# Edge case tests


def test_empty_signal():
    """
    Test empty signals for ECG and PPG.
    """
    ecg_signal = np.array([])  # Empty ECG signal
    ppg_signal = np.array([])  # Empty PPG signal
    ecg_fs = 1000
    ppg_fs = 100

    with pytest.raises(
        ValueError, match="Both ECG and PPG signals must have non-zero length."
    ):
        ECGPPGSynchronization(ecg_signal, ppg_signal, ecg_fs, ppg_fs)


def test_invalid_signal_type():
    """
    Test invalid (non-numeric) data for ECG and PPG.
    """
    ecg_signal = "invalid_signal"  # Non-numeric input
    ppg_signal = "invalid_signal"
    ecg_fs = 1000
    ppg_fs = 100

    with pytest.raises(
        TypeError, match="Both ECG and PPG signals must be numpy arrays."
    ):
        ECGPPGSynchronization(ecg_signal, ppg_signal, ecg_fs, ppg_fs)


def test_no_peaks_detected(generate_ecg_ppg_signal):
    """
    Test handling of signals with no detectable peaks.
    """
    ecg_signal, ppg_signal, ecg_fs, ppg_fs = generate_ecg_ppg_signal

    # Use a longer flatline ECG signal to avoid the "ECG signal is too short" error
    ecg_signal = np.ones(
        2000
    )  # Flatline ECG signal longer than 2 seconds (assuming 1000 Hz)
    ppg_signal = np.ones(2000)  # Flatline PPG signal

    sync_analyzer = ECGPPGSynchronization(ecg_signal, ppg_signal, ecg_fs, ppg_fs)

    # Now it should raise "No R-peaks detected in the ECG signal"
    with pytest.raises(ValueError, match="No R-peaks detected in the ECG signal"):
        sync_analyzer.detect_r_peaks()

    with pytest.raises(
        ValueError, match="No systolic peaks detected in the PPG signal."
    ):
        sync_analyzer.detect_systolic_peaks()


def test_single_value_signal():
    """
    Test handling of single-point signal which is insufficient to compute features.
    """
    ecg_signal = np.array([1])  # Single-point ECG signal
    ppg_signal = np.array([1])  # Single-point PPG signal
    ecg_fs = 1000
    ppg_fs = 100

    sync_analyzer = ECGPPGSynchronization(ecg_signal, ppg_signal, ecg_fs, ppg_fs)

    with pytest.raises(ValueError, match="ECG signal is too short to detect peaks."):
        sync_analyzer.detect_r_peaks()

    with pytest.raises(ValueError, match="PPG signal is too short to detect peaks."):
        sync_analyzer.detect_systolic_peaks()


def test_infinite_values_signal():
    """
    Test handling of signals with infinite values.
    """
    ecg_signal = np.array([np.inf] * 1000)  # Signal with infinite values for ECG
    ppg_signal = np.array([np.inf] * 1000)  # Signal with infinite values for PPG
    ecg_fs = 1000
    ppg_fs = 100

    with pytest.raises(ValueError, match="ECG or PPG signal contains invalid values"):
        sync_analyzer = ECGPPGSynchronization(ecg_signal, ppg_signal, ecg_fs, ppg_fs)
