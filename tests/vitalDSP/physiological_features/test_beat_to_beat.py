import pytest
import numpy as np
import warnings
from vitalDSP.physiological_features.beat_to_beat import BeatToBeatAnalysis

# Filter out deprecation warnings for deprecated functions
warnings.filterwarnings("ignore", category=DeprecationWarning, module="vitalDSP.physiological_features.beat_to_beat")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tests.vitalDSP.physiological_features.test_beat_to_beat")

# Test data setup
SAMPLE_PEAKS = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
SAMPLE_FREQ = 1000


@pytest.fixture
def test_signal():
    # Generating a sample ECG-like signal with some random noise
    np.random.seed(42)
    signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    return signal


@pytest.fixture
def test_r_peaks():
    # Example R-peaks detected from an ECG or PPG signal (ensure variation in peaks)
    return np.array([10, 30, 52, 71, 98])


@pytest.fixture
def btb_analysis(test_signal, test_r_peaks):
    # Create an instance of BeatToBeatAnalysis
    return BeatToBeatAnalysis(
        signal=test_signal, r_peaks=test_r_peaks, fs=1000, signal_type="ECG"
    )


def test_compute_rr_intervals(btb_analysis):
    # Test computation of R-R intervals without correction
    rr_intervals = btb_analysis.compute_rr_intervals()
    assert isinstance(rr_intervals, np.ndarray)
    assert rr_intervals.shape[0] == len(btb_analysis.r_peaks) - 1
    assert np.all(rr_intervals > 0)


def test_compute_rr_intervals_with_interpolation(btb_analysis):
    # Test R-R interval computation with interpolation correction
    rr_intervals = btb_analysis.compute_rr_intervals(correction_method="interpolation")
    assert isinstance(rr_intervals, np.ndarray)
    assert rr_intervals.shape[0] == len(btb_analysis.r_peaks) - 1


def test_compute_rr_intervals_with_resampling(btb_analysis):
    # Test R-R interval computation with resampling correction
    rr_intervals = btb_analysis.compute_rr_intervals(correction_method="resampling")
    assert isinstance(rr_intervals, np.ndarray)
    assert rr_intervals.shape[0] > 0


def test_compute_rr_intervals_with_adaptive_threshold(btb_analysis):
    # Test R-R interval computation with adaptive threshold correction
    rr_intervals = btb_analysis.compute_rr_intervals(
        correction_method="adaptive_threshold", threshold=100
    )
    assert isinstance(rr_intervals, np.ndarray)
    assert rr_intervals.shape[0] > 0


def test_compute_mean_rr(btb_analysis):
    # Test computation of mean R-R interval
    mean_rr = btb_analysis.compute_mean_rr()
    assert isinstance(mean_rr, float)
    assert mean_rr > 0


def test_compute_sdnn(btb_analysis):
    # Test computation of SDNN (standard deviation of R-R intervals)
    rr_intervals = btb_analysis.compute_rr_intervals()
    sdnn = btb_analysis.compute_sdnn()
    assert isinstance(sdnn, float)
    assert sdnn > 0  # SDNN should be positive if there's variability in R-R intervals
    assert (
        np.std(rr_intervals) == sdnn
    )  # Ensure the computed SDNN matches the standard deviation of R-R intervals


def test_compute_rmssd(btb_analysis):
    # Test computation of RMSSD (Root Mean Square of Successive Differences)
    rmssd = btb_analysis.compute_rmssd()
    assert isinstance(rmssd, float)
    assert rmssd > 0  # RMSSD should be positive for signals with variability


def test_compute_pnn50(btb_analysis):
    # Test computation of pNN50 (percentage of R-R intervals differing by more than 50 ms)
    pnn50 = btb_analysis.compute_pnn50()
    assert isinstance(pnn50, float)
    assert 0 <= pnn50 <= 100


def test_compute_hr(btb_analysis):
    # Test computation of heart rate from R-R intervals
    hr = btb_analysis.compute_hr()
    assert isinstance(hr, np.ndarray)
    assert hr.shape[0] == len(btb_analysis.r_peaks) - 1
    assert np.all(hr > 0)


def test_detect_arrhythmias(btb_analysis):
    # Test arrhythmia detection
    arrhythmias = btb_analysis.detect_arrhythmias(threshold=150)
    assert isinstance(arrhythmias, np.ndarray)
    assert arrhythmias.shape[0] >= 0  # Could have 0 or more arrhythmic beats


class TestBeatToBeatMissingCoverage:
    """Tests to cover missing lines in beat_to_beat.py."""

    def test_init_auto_detect_peaks_ecg(self, test_signal):
        """Test initialization with auto-detection of peaks for ECG signal.
        
        This test covers lines 62-67 in beat_to_beat.py where
        r_peaks is None and signal_type is 'ECG'.
        """
        btb = BeatToBeatAnalysis(signal=test_signal, r_peaks=None, fs=1000, signal_type="ECG")
        assert hasattr(btb, 'r_peaks')
        assert isinstance(btb.r_peaks, np.ndarray)
        assert len(btb.r_peaks) > 0

    def test_init_auto_detect_peaks_ppg(self, test_signal):
        """Test initialization with auto-detection of peaks for PPG signal.
        
        This test covers lines 62-65 and 68-69 in beat_to_beat.py where
        r_peaks is None and signal_type is 'PPG'.
        """
        btb = BeatToBeatAnalysis(signal=test_signal, r_peaks=None, fs=1000, signal_type="PPG")
        assert hasattr(btb, 'r_peaks')
        assert isinstance(btb.r_peaks, np.ndarray)
        assert len(btb.r_peaks) > 0

    def test_init_unsupported_signal_type(self, test_signal):
        """Test initialization with unsupported signal type.
        
        This test covers lines 62-65 and 70-71 in beat_to_beat.py where
        ValueError is raised for unsupported signal_type.
        """
        with pytest.raises(ValueError, match="Unsupported signal_type"):
            BeatToBeatAnalysis(signal=test_signal, r_peaks=None, fs=1000, signal_type="EEG")

    def test_correct_by_interpolation_with_outliers(self):
        """Test _correct_by_interpolation when outliers are present.
        
        This test covers line 123 in beat_to_beat.py where
        np.any(outliers) is True.
        """
        # Create RR intervals with clear outliers
        rr_intervals = np.array([800, 850, 2000, 900, 850, 100, 900, 850])  # 2000 and 100 are outliers
        
        btb = BeatToBeatAnalysis(
            signal=np.random.randn(1000),
            r_peaks=np.array([100, 200, 300, 400, 500, 600, 700, 800, 900]),
            fs=1000
        )
        
        corrected = btb._correct_by_interpolation(rr_intervals)
        assert len(corrected) == len(rr_intervals)
        assert not np.any(np.isnan(corrected))
        assert not np.any(np.isinf(corrected))

    def test_correct_by_resampling_fs_zero(self):
        """Test _correct_by_resampling when fs <= 0.
        
        This test covers lines 148-150 in beat_to_beat.py where
        fs <= 0 returns original intervals.
        """
        rr_intervals = np.array([800, 850, 900, 850, 900])
        
        btb = BeatToBeatAnalysis(
            signal=np.random.randn(100),
            r_peaks=np.array([10, 20, 30, 40, 50]),
            fs=0  # Invalid fs
        )
        
        corrected = btb._correct_by_resampling(rr_intervals)
        assert np.array_equal(corrected, rr_intervals)

    def test_correct_by_resampling_fs_negative(self):
        """Test _correct_by_resampling when fs < 0.
        
        This test covers lines 148-150 in beat_to_beat.py where
        fs <= 0 returns original intervals.
        """
        rr_intervals = np.array([800, 850, 900, 850, 900])
        
        btb = BeatToBeatAnalysis(
            signal=np.random.randn(100),
            r_peaks=np.array([10, 20, 30, 40, 50]),
            fs=-1  # Invalid fs
        )
        
        corrected = btb._correct_by_resampling(rr_intervals)
        assert np.array_equal(corrected, rr_intervals)

    def test_correct_by_resampling_target_length_zero(self):
        """Test _correct_by_resampling when target_length <= 0.
        
        This test covers lines 153-155 in beat_to_beat.py where
        target_length <= 0 returns original intervals.
        """
        rr_intervals = np.array([800, 850, 900])
        
        # Use parameters that result in target_length <= 0
        # target_length = num_points * new_rate // self.fs
        # If new_rate is very small compared to fs, target_length could be 0
        btb = BeatToBeatAnalysis(
            signal=np.random.randn(100),
            r_peaks=np.array([10, 20, 30]),
            fs=1000  # High fs
        )
        
        # Use very small new_rate so target_length = 3 * 1 // 1000 = 0
        corrected = btb._correct_by_resampling(rr_intervals, new_rate=1)
        assert np.array_equal(corrected, rr_intervals)

    def test_correct_by_resampling_exception_handling(self):
        """Test _correct_by_resampling exception handling.
        
        This test covers lines 157-162 in beat_to_beat.py where
        resample raises an exception and original intervals are returned.
        """
        from unittest.mock import patch
        
        rr_intervals = np.array([800, 850, 900, 850, 900])
        
        btb = BeatToBeatAnalysis(
            signal=np.random.randn(100),
            r_peaks=np.array([10, 20, 30, 40, 50]),
            fs=1000
        )
        
        # Mock resample to raise ValueError
        with patch('vitalDSP.physiological_features.beat_to_beat.resample', side_effect=ValueError("Mocked error")):
            corrected = btb._correct_by_resampling(rr_intervals)
            assert np.array_equal(corrected, rr_intervals)
        
        # Mock resample to raise ZeroDivisionError
        with patch('vitalDSP.physiological_features.beat_to_beat.resample', side_effect=ZeroDivisionError("Mocked error")):
            corrected = btb._correct_by_resampling(rr_intervals)
            assert np.array_equal(corrected, rr_intervals)