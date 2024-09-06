import pytest
import numpy as np
from vitalDSP.signal_quality_assessment.signal_quality import SignalQuality  # Replace 'your_module' with the actual module name

def test_signalquality_initialization():
    # Test initializing with numpy arrays
    signal = np.array([1, 2, 3, 4, 5])
    processed_signal = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    sq = SignalQuality(signal, processed_signal)
    assert np.array_equal(sq.original_signal, signal)
    assert np.array_equal(sq.processed_signal, processed_signal)

    # Test initializing with lists
    signal_list = [1, 2, 3, 4, 5]
    processed_signal_list = [1.1, 2.1, 3.1, 4.1, 5.1]
    sq = SignalQuality(signal_list, processed_signal_list)
    assert np.array_equal(sq.original_signal, np.array(signal_list))
    assert np.array_equal(sq.processed_signal, np.array(processed_signal_list))

    # Test initializing without processed signal
    sq = SignalQuality(signal)
    assert sq.processed_signal is None

def test_snr():
    signal = np.array([1, 2, 3, 4, 5])
    noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    processed_signal = signal + noise
    sq = SignalQuality(signal, processed_signal)

    snr_value = sq.snr()
    
    # Adjust the expected value based on correct calculation
    assert snr_value == pytest.approx(30.4139, 0.001)

    # Test error when processed_signal is not provided
    sq_no_processed = SignalQuality(signal)
    with pytest.raises(ValueError, match="Processed signal is required to compute SNR."):
        sq_no_processed.snr()

def test_psnr():
    signal = np.array([1, 2, 3, 4, 5])
    noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    processed_signal = signal + noise
    sq = SignalQuality(signal, processed_signal)

    psnr_value = sq.psnr()

    # Adjust the expected value based on correct calculation
    assert psnr_value == pytest.approx(33.9794, 0.001)

    # Test error when processed_signal is not provided
    sq_no_processed = SignalQuality(signal)
    with pytest.raises(ValueError, match="Processed signal is required to compute PSNR."):
        sq_no_processed.psnr()

def test_mse():
    signal = np.array([1, 2, 3, 4, 5])
    noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    processed_signal = signal + noise
    sq = SignalQuality(signal, processed_signal)

    mse_value = sq.mse()
    assert mse_value == pytest.approx(0.01, 0.001)

    # Test error when processed_signal is not provided
    sq_no_processed = SignalQuality(signal)
    with pytest.raises(ValueError, match="Processed signal is required to compute MSE."):
        sq_no_processed.mse()

def test_snr_of_noise():
    signal = np.array([1, 2, 3, 4, 5])
    noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    sq = SignalQuality(signal)

    snr_value = sq.snr_of_noise(noise)

    # Adjust the expected value based on correct calculation
    assert snr_value == pytest.approx(30.4139, 0.001)

    # Test snr_of_noise with a list instead of numpy array
    noise_list = [0.1, 0.1, 0.1, 0.1, 0.1]
    snr_value_list = sq.snr_of_noise(noise_list)
    assert snr_value_list == pytest.approx(30.4139, 0.001)
