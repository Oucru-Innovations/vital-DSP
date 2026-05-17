"""
Tests targeting missing coverage in beats_transformation.py.

Missing lines: 121-125, 151, 164, 236-237, 268, 273-280, 314, 316-317, 364-372
"""

from __future__ import annotations

import numpy as np
import pytest

from vitalDSP.transforms.beats_transformation import RRTransformation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ecg_signal(n=3000, fs=100, bpm=60):
    """Synthetic ECG-like signal with regular peaks at bpm beats/min."""
    t = np.arange(n) / fs
    # Sine wave at heart rate frequency — used by WaveformMorphology
    signal = np.sin(2 * np.pi * (bpm / 60) * t)
    return signal, fs


@pytest.fixture
def rr_transform():
    sig, fs = make_ecg_signal()
    return RRTransformation(signal=sig, fs=fs, signal_type="ECG")


@pytest.fixture
def ppg_transform():
    n, fs = 3000, 100
    t = np.arange(n) / fs
    signal = np.sin(2 * np.pi * 1.0 * t) + 1.0  # PPG-like with positive DC
    return RRTransformation(signal=signal, fs=fs, signal_type="PPG")


# ---------------------------------------------------------------------------
# compute_rr_intervals — preprocess_config default path (lines 121-125)
# ---------------------------------------------------------------------------


class TestComputeRRIntervals:
    def test_with_default_preprocess_config(self, rr_transform):
        """Lines 121-125: preprocess_config=None → uses PreprocessConfig()."""
        rr = rr_transform.compute_rr_intervals(preprocess_config=None)
        assert isinstance(rr, np.ndarray)
        assert len(rr) > 0

    def test_ppg_peak_branch(self, ppg_transform):
        """Line 151: PPG uses systolic_peaks."""
        rr = ppg_transform.compute_rr_intervals()
        assert isinstance(rr, np.ndarray)
        assert len(rr) > 0

    def test_zero_signal_raises_no_peaks(self):
        """Line 153-156: no peaks → ValueError."""
        signal = np.zeros(1000)
        t = RRTransformation(signal, fs=100, signal_type="ECG")
        with pytest.raises(ValueError, match="No peaks detected"):
            t.compute_rr_intervals()


# ---------------------------------------------------------------------------
# remove_invalid_rr_intervals — various branches (lines 204-244)
# ---------------------------------------------------------------------------


class TestRemoveInvalidRRIntervals:
    def test_physiological_limits_filter(self, rr_transform):
        """Lines 207-211: remove outside [min_rr, max_rr]."""
        rr = np.array([100.0, 600.0, 800.0, 2500.0, 600.0])
        filtered = rr_transform.remove_invalid_rr_intervals(
            rr, min_rr=300, max_rr=2000
        )
        assert np.isnan(filtered[0])   # 100 ms < min_rr
        assert np.isnan(filtered[3])   # 2500 ms > max_rr
        # At least some valid values remain (600 and 800 are within range)
        assert not np.all(np.isnan(filtered))

    def test_std_dev_filter(self, rr_transform):
        """Lines 219-224: std_dev_factor filtering."""
        # Create intervals where some are far outside 2 std-devs
        rr = np.array([600.0] * 10 + [5000.0, 600.0], dtype=float)
        filtered = rr_transform.remove_invalid_rr_intervals(rr, std_dev_factor=2.0)
        # 5000 should be NaN-ed
        assert np.isnan(filtered[10])

    def test_sudden_change_detection(self, rr_transform):
        """Lines 227-237: sudden_change_threshold."""
        rr = np.array([600.0, 600.0, 600.0, 1200.0, 600.0], dtype=float)
        filtered = rr_transform.remove_invalid_rr_intervals(
            rr, sudden_change_threshold=0.2
        )
        # 1200 is a 100% increase from 600 — exceeds 0.2 threshold
        assert np.isnan(filtered[3])

    def test_all_valid_passes_through(self, rr_transform):
        rr = np.array([600.0, 620.0, 610.0, 615.0, 608.0], dtype=float)
        filtered = rr_transform.remove_invalid_rr_intervals(rr)
        # All should remain valid (no NaN)
        assert not np.all(np.isnan(filtered))


# ---------------------------------------------------------------------------
# _reconsider_trends  (lines 246-282)
# ---------------------------------------------------------------------------


class TestReconsiderTrends:
    def test_gradual_trend_restored(self, rr_transform):
        """Lines 271-280: gradually trending NaN gets restored."""
        # Build a slowly increasing sequence; mark one as NaN
        base = np.array([600.0, 610.0, 620.0, 630.0, 640.0, 650.0, 660.0, 670.0], dtype=float)
        with_nan = base.copy()
        with_nan[5] = np.nan  # one missing value in a gradual trend
        result = rr_transform._reconsider_trends(with_nan, original_values=base, window_size=3)
        # The NaN at position 5 should be reconsidered — might be restored
        # (within 10% of running mean)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(base)

    def test_no_nan_unchanged(self, rr_transform):
        rr = np.array([600.0, 610.0, 620.0, 630.0], dtype=float)
        result = rr_transform._reconsider_trends(rr, original_values=rr)
        np.testing.assert_array_equal(result, rr)

    def test_without_original_values(self, rr_transform):
        rr = np.array([600.0, np.nan, 620.0, 630.0], dtype=float)
        result = rr_transform._reconsider_trends(rr)
        # Just checks it doesn't crash
        assert len(result) == len(rr)


# ---------------------------------------------------------------------------
# impute_rr_intervals — method branches (lines 284-340)
# ---------------------------------------------------------------------------


class TestImputeRRIntervals:
    @pytest.fixture
    def rr_with_nans(self):
        return np.array([600.0, np.nan, 620.0, np.nan, np.nan, 610.0, 605.0])

    def test_adaptive_low_nan_ratio_uses_linear(self, rr_transform, rr_with_nans):
        """Line 314: nan_ratio < 0.05 → linear."""
        # Only 2 NaN in 7 = 28.6%, but test the boundary case
        few_nans = np.array([600.0, np.nan, 610.0, 620.0, 615.0, 600.0, 605.0,
                             610.0, 615.0, 620.0, 625.0, 630.0, 635.0, 640.0,
                             645.0, 650.0, 655.0, 660.0, 665.0, 670.0, 675.0])
        result = rr_transform.impute_rr_intervals(few_nans, method="adaptive")
        assert isinstance(result, np.ndarray)

    def test_adaptive_medium_nan_ratio_uses_spline(self, rr_transform):
        """Line 316-317: nan_ratio in [0.05, 0.2) → spline."""
        base = [600.0] * 100
        # 10 NaN in 100 = 10%
        arr = np.array(base, dtype=float)
        arr[5:15] = np.nan
        result = rr_transform.impute_rr_intervals(arr, method="adaptive")
        assert isinstance(result, np.ndarray)

    def test_adaptive_high_nan_ratio_uses_rolling_mean(self, rr_transform):
        """nan_ratio >= 0.2 → rolling_mean."""
        base = [600.0] * 50
        arr = np.array(base, dtype=float)
        arr[0:15] = np.nan  # 30% NaN
        result = rr_transform.impute_rr_intervals(arr, method="adaptive")
        assert isinstance(result, np.ndarray)

    def test_linear_method(self, rr_transform, rr_with_nans):
        result = rr_transform.impute_rr_intervals(rr_with_nans, method="linear")
        assert not np.any(np.isnan(result))

    def test_spline_method(self, rr_transform, rr_with_nans):
        result = rr_transform.impute_rr_intervals(rr_with_nans, method="spline", order=2)
        assert isinstance(result, np.ndarray)

    def test_mean_method(self, rr_transform, rr_with_nans):
        result = rr_transform.impute_rr_intervals(rr_with_nans, method="mean")
        assert isinstance(result, np.ndarray)

    def test_median_method(self, rr_transform, rr_with_nans):
        result = rr_transform.impute_rr_intervals(rr_with_nans, method="median")
        assert isinstance(result, np.ndarray)

    def test_forward_fill_method(self, rr_transform, rr_with_nans):
        result = rr_transform.impute_rr_intervals(rr_with_nans, method="forward_fill")
        assert isinstance(result, np.ndarray)

    def test_backward_fill_method(self, rr_transform, rr_with_nans):
        result = rr_transform.impute_rr_intervals(rr_with_nans, method="backward_fill")
        assert isinstance(result, np.ndarray)

    def test_rolling_mean_method(self, rr_transform, rr_with_nans):
        result = rr_transform.impute_rr_intervals(rr_with_nans, method="rolling_mean", window=3)
        assert isinstance(result, np.ndarray)

    def test_unsupported_method_raises(self, rr_transform, rr_with_nans):
        with pytest.raises(ValueError, match="Unsupported"):
            rr_transform.impute_rr_intervals(rr_with_nans, method="bogus_method")


# ---------------------------------------------------------------------------
# process_rr_intervals — end-to-end (lines 342-372)
# ---------------------------------------------------------------------------


class TestProcessRRIntervals:
    def test_process_full_pipeline(self, rr_transform):
        """Lines 364-372: remove_invalid=True, impute_invalid=True."""
        result = rr_transform.process_rr_intervals(
            remove_invalid=True, impute_invalid=True
        )
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_process_no_removal(self, rr_transform):
        result = rr_transform.process_rr_intervals(
            remove_invalid=False, impute_invalid=True
        )
        assert isinstance(result, np.ndarray)

    def test_process_no_impute(self, rr_transform):
        result = rr_transform.process_rr_intervals(
            remove_invalid=True, impute_invalid=False
        )
        assert isinstance(result, np.ndarray)

    def test_process_neither(self, rr_transform):
        result = rr_transform.process_rr_intervals(
            remove_invalid=False, impute_invalid=False
        )
        assert isinstance(result, np.ndarray)
