import numpy as np
import pytest
from vitalDSP.preprocess.preprocess_operations import (
    preprocess_signal,
    estimate_baseline,
    respiratory_filtering
)
from vitalDSP.preprocess.noise_reduction import (
    wavelet_denoising,
    savgol_denoising,
    median_denoising,
    gaussian_denoising,
    moving_average_denoising,
)


@pytest.fixture
def noisy_signal():
    return np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.2, 100)


def test_wavelet_denoising(noisy_signal):
    denoised_signal = wavelet_denoising(noisy_signal, wavelet_name="haar", level=2)
    assert len(denoised_signal) == len(noisy_signal)
    # assert np.var(denoised_signal) < np.var(noisy_signal)


def test_savgol_denoising(noisy_signal):
    denoised_signal = savgol_denoising(noisy_signal, window_length=5, polyorder=2)
    assert len(denoised_signal) == len(noisy_signal)
    assert np.var(denoised_signal) < np.var(noisy_signal)


def test_median_denoising(noisy_signal):
    denoised_signal = median_denoising(noisy_signal, kernel_size=3)
    assert len(denoised_signal) == len(noisy_signal)
    # assert np.var(denoised_signal) < np.var(noisy_signal)


def test_gaussian_denoising(noisy_signal):
    denoised_signal = gaussian_denoising(noisy_signal, sigma=1.5)
    assert len(denoised_signal) == len(noisy_signal)
    assert np.var(denoised_signal) < np.var(noisy_signal)


def test_moving_average_denoising(noisy_signal):
    denoised_signal = moving_average_denoising(noisy_signal, window_size=5)
    assert len(denoised_signal) == len(noisy_signal)
    # assert np.var(denoised_signal) < np.var(noisy_signal)


@pytest.fixture
def test_signal():
    return np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.2, 100)


def test_preprocess_bandpass_wavelet(test_signal):
    sampling_rate = 1000
    preprocessed_signal = preprocess_signal(
        test_signal,
        sampling_rate,
        filter_type="bandpass",
        noise_reduction_method="wavelet",
    )
    assert len(preprocessed_signal) == len(test_signal)
    # assert np.var(preprocessed_signal) < np.var(test_signal)


def test_preprocess_butterworth_median(test_signal):
    sampling_rate = 1000
    preprocessed_signal = preprocess_signal(
        test_signal,
        sampling_rate,
        filter_type="butterworth",
        noise_reduction_method="median",
    )
    assert len(preprocessed_signal) == len(test_signal)
    assert np.var(preprocessed_signal) < np.var(test_signal)


def test_preprocess_chebyshev_gaussian(test_signal):
    sampling_rate = 1000
    preprocessed_signal = preprocess_signal(
        test_signal,
        sampling_rate,
        filter_type="chebyshev",
        noise_reduction_method="gaussian",
    )
    assert len(preprocessed_signal) == len(test_signal)
    assert np.var(preprocessed_signal) < np.var(test_signal)


def test_preprocess_invalid_filter(test_signal):
    sampling_rate = 1000
    with pytest.raises(
        ValueError,
        match="Unsupported filter type. Choose from 'bandpass', 'butterworth', 'chebyshev', or 'elliptic'.",
    ):
        preprocess_signal(
            test_signal,
            sampling_rate,
            filter_type="invalid",
            noise_reduction_method="gaussian",
        )


def test_preprocess_invalid_noise_reduction(test_signal):
    sampling_rate = 1000
    with pytest.raises(
        ValueError,
        match="Unsupported noise reduction method. Choose from 'wavelet', 'savgol', 'median', 'gaussian', or 'moving_average'.",
    ):
        preprocess_signal(
            test_signal,
            sampling_rate,
            filter_type="bandpass",
            noise_reduction_method="invalid",
        )


def test_preprocess_elliptic_savgol(test_signal):
    sampling_rate = 1000
    preprocessed_signal = preprocess_signal(
        test_signal,
        sampling_rate,
        filter_type="elliptic",
        noise_reduction_method="savgol",
    )
    assert len(preprocessed_signal) == len(test_signal)
    assert np.var(preprocessed_signal) < np.var(test_signal)


def test_preprocess_ignore_ignore(test_signal):
    sampling_rate = 1000
    preprocessed_signal = preprocess_signal(
        test_signal,
        sampling_rate,
        filter_type="ignore",
        noise_reduction_method="ignore",
    )
    assert len(preprocessed_signal) == len(test_signal)
    assert np.var(preprocessed_signal) == np.var(test_signal)


def test_preprocess_bandpass_moving_average(test_signal):
    sampling_rate = 1000
    preprocessed_signal = preprocess_signal(
        test_signal,
        sampling_rate,
        filter_type="bandpass",
        noise_reduction_method="moving_average",
    )
    assert len(preprocessed_signal) == len(test_signal)
    
    signal_baseline = estimate_baseline(preprocessed_signal,
                                        sampling_rate,
                                        method="low_pass")
    assert len(signal_baseline) == len(test_signal)
    signal_baseline = estimate_baseline(preprocessed_signal,
                                        sampling_rate,
                                        method="polynomial_fit")
    assert len(signal_baseline) == len(test_signal)
    signal_baseline = estimate_baseline(preprocessed_signal,
                                        sampling_rate,
                                        method="median_filter")
    assert len(signal_baseline) == len(test_signal)
    with pytest.raises(
        ValueError,
        match="Unsupported baseline estimation method: invalid",
    ):
        estimate_baseline(preprocessed_signal,
                        sampling_rate,
                        method="invalid")
    # assert np.var(preprocessed_signal) < np.var(test_signal)


class TestRespiratoryFilteringMissingCoverage:
    """Tests to cover missing lines in respiratory_filtering function."""

    def test_respiratory_filtering_negative_sampling_rate(self):
        """Test respiratory_filtering with negative sampling rate.
        
        This test covers line 328 in preprocess_operations.py where
        ValueError is raised for sampling_rate <= 0.
        """
        signal = np.sin(np.linspace(0, 10, 100))
        
        with pytest.raises(ValueError, match="Sampling rate must be positive"):
            respiratory_filtering(signal, sampling_rate=-1)
        
        with pytest.raises(ValueError, match="Sampling rate must be positive"):
            respiratory_filtering(signal, sampling_rate=0)

    def test_respiratory_filtering_lowcut_greater_than_highcut(self):
        """Test respiratory_filtering with lowcut >= highcut.
        
        This test covers line 331 in preprocess_operations.py where
        ValueError is raised for lowcut >= highcut.
        """
        signal = np.sin(np.linspace(0, 10, 100))
        
        with pytest.raises(ValueError, match="lowcut.*must be less than highcut"):
            respiratory_filtering(signal, sampling_rate=100, lowcut=0.5, highcut=0.3)
        
        with pytest.raises(ValueError, match="lowcut.*must be less than highcut"):
            respiratory_filtering(signal, sampling_rate=100, lowcut=0.5, highcut=0.5)

    def test_respiratory_filtering_highcut_above_nyquist(self):
        """Test respiratory_filtering with highcut >= Nyquist frequency.
        
        This test covers lines 334-336 in preprocess_operations.py where
        ValueError is raised for highcut >= sampling_rate / 2.
        """
        signal = np.sin(np.linspace(0, 10, 100))
        sampling_rate = 100
        
        with pytest.raises(ValueError, match="highcut.*must be less than Nyquist frequency"):
            respiratory_filtering(signal, sampling_rate=sampling_rate, highcut=sampling_rate / 2)
        
        with pytest.raises(ValueError, match="highcut.*must be less than Nyquist frequency"):
            respiratory_filtering(signal, sampling_rate=sampling_rate, highcut=sampling_rate / 2 + 1)

    def test_respiratory_filtering_normalized_frequencies_too_close(self):
        """Test respiratory_filtering when normalized frequencies are too close.
        
        This test covers lines 347-352 in preprocess_operations.py where
        low_norm >= high_norm triggers frequency adjustment.
        """
        signal = np.sin(np.linspace(0, 10, 1000))
        sampling_rate = 1000
        
        # Use frequencies that will result in low_norm >= high_norm after normalization
        # Very close frequencies near Nyquist
        lowcut = 490  # Close to Nyquist (500)
        highcut = 495  # Very close to lowcut
        
        # This should trigger the adjustment logic
        filtered = respiratory_filtering(signal, sampling_rate=sampling_rate, lowcut=lowcut, highcut=highcut)
        assert len(filtered) == len(signal)
        assert not np.any(np.isnan(filtered))
        assert not np.any(np.isinf(filtered))

    def test_respiratory_filtering_nan_inf_values(self):
        """Test respiratory_filtering when filter produces NaN or Inf values.
        
        This test covers lines 360-362 in preprocess_operations.py where
        ValueError is raised for NaN or Inf values.
        """
        from unittest.mock import patch
        
        signal = np.sin(np.linspace(0, 10, 100))
        
        # Mock filtfilt to return NaN values
        def mock_filtfilt(b, a, x):
            result = np.zeros_like(x)
            result[0] = np.nan
            return result
        
        with patch('vitalDSP.preprocess.preprocess_operations.filtfilt', side_effect=mock_filtfilt):
            # Should catch the NaN and return normalized signal (fallback)
            filtered = respiratory_filtering(signal, sampling_rate=100)
            assert len(filtered) == len(signal)
            # Should return normalized signal (fallback)
            assert not np.any(np.isnan(filtered))
            assert not np.any(np.isinf(filtered))

    def test_respiratory_filtering_unstable_filter_recursion(self):
        """Test respiratory_filtering when filter is unstable and order > 1.
        
        This test covers lines 366-369 in preprocess_operations.py where
        respiratory_filtering is called recursively with order - 1.
        """
        from unittest.mock import patch
        
        signal = np.sin(np.linspace(0, 10, 100))
        
        # Mock filtfilt to return very large values (unstable filter)
        call_count = [0]
        def mock_filtfilt(b, a, x):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call returns unstable values
                return np.ones_like(x) * 1e7
            else:
                # Subsequent calls return stable values
                return np.sin(np.linspace(0, 10, len(x)))
        
        with patch('vitalDSP.preprocess.preprocess_operations.filtfilt', side_effect=mock_filtfilt):
            filtered = respiratory_filtering(signal, sampling_rate=100, order=4)
            assert len(filtered) == len(signal)
            # Should have been called recursively with lower order
            assert call_count[0] > 1

    def test_respiratory_filtering_unstable_filter_order_one(self):
        """Test respiratory_filtering when filter is unstable and order == 1.
        
        This test covers line 372 in preprocess_operations.py where
        normalized signal is returned as last resort.
        """
        from unittest.mock import patch
        
        signal = np.sin(np.linspace(0, 10, 100))
        
        # Mock filtfilt to return very large values (unstable filter)
        def mock_filtfilt(b, a, x):
            return np.ones_like(x) * 1e7
        
        with patch('vitalDSP.preprocess.preprocess_operations.filtfilt', side_effect=mock_filtfilt):
            filtered = respiratory_filtering(signal, sampling_rate=100, order=1)
            assert len(filtered) == len(signal)
            # Should return normalized signal
            assert np.std(filtered) > 0  # Normalized signal should have std > 0
            assert abs(np.mean(filtered)) < 1e-10  # Normalized signal should have mean ~ 0

    def test_respiratory_filtering_exception_fallback(self):
        """Test respiratory_filtering exception handling fallback.
        
        This test covers lines 376-378 in preprocess_operations.py where
        normalized signal is returned if filtering fails.
        """
        from unittest.mock import patch
        
        signal = np.sin(np.linspace(0, 10, 100))
        
        # Mock butter to raise an exception
        with patch('vitalDSP.preprocess.preprocess_operations.butter', side_effect=Exception("Mocked error")):
            filtered = respiratory_filtering(signal, sampling_rate=100)
            assert len(filtered) == len(signal)
            # Should return normalized signal (fallback)
            assert not np.any(np.isnan(filtered))
            assert not np.any(np.isinf(filtered))
            assert np.std(filtered) > 0  # Normalized signal should have std > 0