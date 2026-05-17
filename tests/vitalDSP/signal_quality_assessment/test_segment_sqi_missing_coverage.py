"""
Tests targeting missing coverage in segment_sqi.py.

Missing lines: 86, 92, 99, 102, 107, 121, 124, 136, 150, 153, 163, 166, 249
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from vitalDSP.signal_quality_assessment.segment_sqi import (
    AVAILABLE_SQIS,
    _entropy_sqi,
    _kurtosis_sqi,
    _peak_to_peak_amplitude_sqi,
    _perfusion_sqi,
    _segment_indices,
    _skewness_sqi,
    _snr_sqi,
    _zero_crossing_sqi,
    compute_segment_sqis,
)


# ---------------------------------------------------------------------------
# _kurtosis_sqi  (lines 84-87) — branch: segment.size < 3 → NaN
# ---------------------------------------------------------------------------


class TestKurtosisSqi:
    def test_too_short_returns_nan(self):
        assert math.isnan(_kurtosis_sqi(np.array([1.0])))       # line 86
        assert math.isnan(_kurtosis_sqi(np.array([1.0, 2.0])))  # line 86

    def test_normal_returns_finite(self):
        arr = np.array([1.0, 2.0, 3.0])
        val = _kurtosis_sqi(arr)
        assert math.isfinite(val)


# ---------------------------------------------------------------------------
# _skewness_sqi  (lines 90-93) — branch: segment.size < 3 → NaN
# ---------------------------------------------------------------------------


class TestSkewnessSqi:
    def test_too_short_returns_nan(self):
        assert math.isnan(_skewness_sqi(np.array([])))          # line 92
        assert math.isnan(_skewness_sqi(np.array([5.0])))       # line 92

    def test_normal_returns_finite(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        val = _skewness_sqi(arr)
        assert math.isfinite(val)


# ---------------------------------------------------------------------------
# _entropy_sqi  (lines 96-110) — branches for size<4, all-nan, zero total
# ---------------------------------------------------------------------------


class TestEntropySqi:
    def test_too_short_size_lt_4(self):
        assert math.isnan(_entropy_sqi(np.array([1.0, 2.0, 3.0])))  # line 99

    def test_all_nan_returns_nan(self):
        arr = np.array([np.nan, np.nan, np.nan, np.nan])
        assert math.isnan(_entropy_sqi(arr))  # line 102

    def test_normal_returns_positive(self):
        arr = np.random.default_rng(0).standard_normal(200)
        val = _entropy_sqi(arr)
        assert math.isfinite(val)
        assert val > 0

    def test_single_value_histogram_returns_nan_or_zero(self):
        # All identical → only one non-zero bin → -p*log(p) = 0 nats
        arr = np.full(20, 3.14)
        val = _entropy_sqi(arr)
        # Either 0 or computed; must be finite (total > 0)
        assert math.isfinite(val)


# ---------------------------------------------------------------------------
# _snr_sqi  (lines 113-130) — branches for size<4, finite<4, mad~0
# ---------------------------------------------------------------------------


class TestSnrSqi:
    def test_too_short_returns_nan(self):
        assert math.isnan(_snr_sqi(np.array([1.0, 2.0, 3.0])))  # line 121

    def test_all_nan_returns_nan(self):
        arr = np.array([np.nan, np.nan, np.nan, np.nan])
        assert math.isnan(_snr_sqi(arr))  # line 124 branch

    def test_flat_signal_nan_due_to_mad_zero(self):
        arr = np.full(100, 5.0)
        val = _snr_sqi(arr)
        assert math.isnan(val)  # line 129 branch (mad < 1e-12)

    def test_normal_signal_finite(self):
        t = np.arange(200) / 200.0
        arr = np.sin(2 * np.pi * 1.0 * t) + 0.01 * np.random.default_rng(1).standard_normal(200)
        val = _snr_sqi(arr)
        assert math.isfinite(val)


# ---------------------------------------------------------------------------
# _zero_crossing_sqi  (lines 133-139) — branch for size<2
# ---------------------------------------------------------------------------


class TestZeroCrossingSqi:
    def test_too_short_returns_nan(self):
        assert math.isnan(_zero_crossing_sqi(np.array([1.0])))  # line 136

    def test_alternating_signal(self):
        arr = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        val = _zero_crossing_sqi(arr)
        assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# _perfusion_sqi  (lines 142-158) — branches for size<2, finite<2, dc~0
# ---------------------------------------------------------------------------


class TestPerfusionSqi:
    def test_too_short_returns_nan(self):
        assert math.isnan(_perfusion_sqi(np.array([1.0])))  # line 150

    def test_all_nan_returns_nan(self):
        arr = np.array([np.nan, np.nan])
        assert math.isnan(_perfusion_sqi(arr))  # line 153

    def test_zero_dc_returns_nan(self):
        # Mean-zero signal → dc ≈ 0
        arr = np.array([1.0, -1.0, 1.0, -1.0])
        val = _perfusion_sqi(arr)
        assert math.isnan(val)  # line 157 branch

    def test_valid_ppg_like(self):
        arr = np.sin(np.linspace(0, 2 * np.pi, 200)) + 5.0  # DC ~ 5
        val = _perfusion_sqi(arr)
        assert math.isfinite(val)


# ---------------------------------------------------------------------------
# _peak_to_peak_amplitude_sqi  (lines 161-167) — branches for size<2, finite<2
# ---------------------------------------------------------------------------


class TestPeakToPeakAmplitudeSqi:
    def test_too_short_returns_nan(self):
        assert math.isnan(_peak_to_peak_amplitude_sqi(np.array([])))  # line 163

    def test_all_nan_returns_nan(self):
        arr = np.array([np.nan, np.nan])
        assert math.isnan(_peak_to_peak_amplitude_sqi(arr))  # line 166

    def test_valid(self):
        arr = np.array([1.0, 5.0, 2.0, 4.0])
        val = _peak_to_peak_amplitude_sqi(arr)
        assert val == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# _segment_indices  (lines 186-199) — branch for too-short signal
# ---------------------------------------------------------------------------


class TestSegmentIndices:
    def test_short_signal_returns_empty(self):
        result = _segment_indices(n_samples=50, window_samples=100, step_samples=50)
        assert result == []

    def test_exact_fit(self):
        result = _segment_indices(n_samples=100, window_samples=100, step_samples=100)
        assert result == [(0, 100)]

    def test_multiple_windows(self):
        result = _segment_indices(n_samples=300, window_samples=100, step_samples=100)
        assert result == [(0, 100), (100, 200), (200, 300)]

    def test_overlap(self):
        result = _segment_indices(n_samples=200, window_samples=100, step_samples=50)
        assert result == [(0, 100), (50, 150), (100, 200)]


# ---------------------------------------------------------------------------
# compute_segment_sqis — partial sqi_names path  (line 249 area)
# ---------------------------------------------------------------------------


class TestComputeSegmentSqisExtra:
    def test_partial_sqi_names(self):
        sig = np.sin(np.linspace(0, 60, 6000)) + 1.0
        df, ms = compute_segment_sqis(
            sig,
            sampling_freq=100,
            segment_seconds=10,
            sqi_names=["kurtosis_sqi", "entropy_sqi"],
        )
        assert list(df.columns) == ["kurtosis_sqi", "entropy_sqi"]
        assert len(df) == 6

    def test_unknown_sqi_logged_and_ignored(self, caplog):
        import logging

        sig = np.zeros(1000)
        with caplog.at_level(logging.WARNING):
            df, _ = compute_segment_sqis(
                sig,
                sampling_freq=100,
                segment_seconds=2,
                sqi_names=["kurtosis_sqi", "nonexistent_sqi"],
            )
        assert "kurtosis_sqi" in df.columns
        assert "nonexistent_sqi" not in df.columns

    def test_all_available_sqis(self):
        sig = np.sin(np.linspace(0, 20, 2000)) + 2.0
        df, ms = compute_segment_sqis(sig, sampling_freq=100, segment_seconds=5)
        for col in df.columns:
            assert col in AVAILABLE_SQIS
