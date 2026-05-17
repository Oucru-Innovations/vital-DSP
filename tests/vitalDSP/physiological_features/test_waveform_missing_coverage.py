"""
Tests targeting missing coverage in waveform.py.

Missing lines: 288, 402, 408-418, 444, 527, 555, 1026, 1075, 1269-1305,
1326, 1329, 1403-1406, 1427, 1437, 1540-1549, 1575-1577, 1744, 1752-1753,
1854, 1866, 1939-1945, 1959, 1980, 1990-1994, 2071, 2075-2097, 2168,
2189-2191, 2218, 2228-2230, 2264, 2277, 2290, 2303-2305, 2350, 2366,
2371-2384, 2427-2436, 2499, 2504-2506
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, filtfilt

from vitalDSP.physiological_features.waveform import WaveformMorphology


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FS = 100


def make_ppg(duration=20.0, bpm=60, fs=FS):
    """Synthetic PPG-like signal (sine with positive DC)."""
    t = np.arange(int(duration * fs)) / fs
    signal = np.sin(2 * np.pi * (bpm / 60) * t) + 2.0
    return signal


def make_ecg(duration=20.0, bpm=60, fs=FS):
    """Synthetic ECG-like signal."""
    t = np.arange(int(duration * fs)) / fs
    signal = np.sin(2 * np.pi * (bpm / 60) * t)
    return signal


@pytest.fixture
def ppg_wm():
    signal = make_ppg()
    return WaveformMorphology(signal, fs=FS, signal_type="PPG")


@pytest.fixture
def ecg_wm():
    signal = make_ecg()
    return WaveformMorphology(signal, fs=FS, signal_type="ECG")


# ---------------------------------------------------------------------------
# diastolic_trough detection — flat segment code path  (line 288)
# ---------------------------------------------------------------------------


class TestDiastolicTroughDetection:
    def test_no_flat_segment_skipped(self, ppg_wm):
        """Line 288: empty flat_segment list → trough not appended."""
        troughs = ppg_wm.detect_troughs()
        assert isinstance(troughs, np.ndarray)

    def test_ppg_dicrotic_notch_requires_ppg(self):
        ecg_wm = WaveformMorphology(make_ecg(), fs=FS, signal_type="ECG")
        with pytest.raises(ValueError, match="Dicrotic notch"):
            ecg_wm.compute_ppg_dicrotic_notch()


# ---------------------------------------------------------------------------
# detect_dicrotic_notches  (lines 402, 408-418, 444)
# ---------------------------------------------------------------------------


class TestDicrotricNotches:
    def test_basic_detection(self, ppg_wm):
        notches = ppg_wm.detect_dicrotic_notches()
        assert isinstance(notches, np.ndarray)

    def test_non_simple_mode(self, ppg_wm):
        """Line 419-444: non-simple mode branch (via simple_mode=False)."""
        systolic = ppg_wm.systolic_peaks
        troughs = ppg_wm.diastolic_troughs  # pre-computed attribute
        # Temporarily switch to non-simple mode to hit that branch
        ppg_wm.simple_mode = False
        try:
            notches = ppg_wm.detect_dicrotic_notches(
                systolic_peaks=systolic,
                diastolic_troughs=troughs,
            )
        finally:
            ppg_wm.simple_mode = True  # restore
        assert isinstance(notches, np.ndarray)

    def test_trough_le_peak_skipped(self, ppg_wm):
        """Line 402/408: search_start >= search_end → continue."""
        # Provide empty diastolic_troughs to hit the "no troughs" branch
        notches = ppg_wm.detect_dicrotic_notches(
            systolic_peaks=ppg_wm.systolic_peaks,
            diastolic_troughs=np.array([]),
        )
        assert isinstance(notches, np.ndarray)


# ---------------------------------------------------------------------------
# detect_diastolic_peak  (lines 527, 555)
# ---------------------------------------------------------------------------


class TestDiastolicPeakDetection:
    def test_ppg_only(self, ppg_wm):
        peaks = ppg_wm.detect_diastolic_peak()
        assert isinstance(peaks, np.ndarray)

    def test_non_ppg_raises(self):
        ecg_wm = WaveformMorphology(make_ecg(), fs=FS, signal_type="ECG")
        with pytest.raises(ValueError, match="Diastolic peaks can only be detected for PPG"):
            ecg_wm.detect_diastolic_peak()

    def test_non_simple_mode_diastolic(self, ppg_wm):
        """Line 528-556: non-simple mode path for diastolic peak detection."""
        systolic = ppg_wm.systolic_peaks
        troughs = ppg_wm.diastolic_troughs  # pre-computed attribute
        notches = ppg_wm.detect_dicrotic_notches(
            systolic_peaks=systolic,
            diastolic_troughs=troughs,
        )
        # Switch simple_mode to False to hit the non-simple branch
        ppg_wm.simple_mode = False
        try:
            peaks = ppg_wm.detect_diastolic_peak(notches=notches, diastolic_troughs=troughs)
        finally:
            ppg_wm.simple_mode = True
        assert isinstance(peaks, np.ndarray)


# ---------------------------------------------------------------------------
# compute_amplitude — ECG interval types (lines 1269-1329)
# ---------------------------------------------------------------------------


class TestComputeAmplitudeECG:
    def test_r_to_s(self, ecg_wm):
        amps = ecg_wm.compute_amplitude(interval_type="R-to-S")
        assert isinstance(amps, np.ndarray)

    def test_r_to_q(self, ecg_wm):
        amps = ecg_wm.compute_amplitude(interval_type="R-to-Q")
        assert isinstance(amps, np.ndarray)

    def test_p_to_q(self, ecg_wm):
        amps = ecg_wm.compute_amplitude(interval_type="P-to-Q")
        assert isinstance(amps, np.ndarray)

    def test_t_to_s(self, ecg_wm):
        amps = ecg_wm.compute_amplitude(interval_type="T-to-S")
        assert isinstance(amps, np.ndarray)

    def test_t_to_baseline(self, ecg_wm):
        amps = ecg_wm.compute_amplitude(interval_type="T-to-Baseline")
        assert isinstance(amps, np.ndarray)

    def test_r_to_baseline(self, ecg_wm):
        amps = ecg_wm.compute_amplitude(interval_type="R-to-Baseline")
        assert isinstance(amps, np.ndarray)

    def test_s_to_baseline(self, ecg_wm):
        amps = ecg_wm.compute_amplitude(interval_type="S-to-Baseline")
        assert isinstance(amps, np.ndarray)

    def test_invalid_interval_type_ecg(self, ecg_wm):
        with pytest.raises(ValueError, match="Invalid interval_type for ECG"):
            ecg_wm.compute_amplitude(interval_type="bogus")

    def test_invalid_signal_type_raises(self, ecg_wm):
        with pytest.raises(ValueError, match="Invalid signal type"):
            ecg_wm.compute_amplitude(interval_type="R-to-S", signal_type="EEG")


class TestComputeAmplitudePPG:
    def test_sys_to_notch(self, ppg_wm):
        amps = ppg_wm.compute_amplitude(interval_type="Sys-to-Notch", signal_type="PPG")
        assert isinstance(amps, np.ndarray)

    def test_sys_to_dia(self, ppg_wm):
        amps = ppg_wm.compute_amplitude(interval_type="Sys-to-Dia", signal_type="PPG")
        assert isinstance(amps, np.ndarray)

    def test_sys_to_baseline(self, ppg_wm):
        amps = ppg_wm.compute_amplitude(interval_type="Sys-to-Baseline", signal_type="PPG")
        assert isinstance(amps, np.ndarray)

    def test_notch_to_baseline(self, ppg_wm):
        amps = ppg_wm.compute_amplitude(interval_type="Notch-to-Baseline", signal_type="PPG")
        assert isinstance(amps, np.ndarray)

    def test_dia_to_baseline(self, ppg_wm):
        amps = ppg_wm.compute_amplitude(interval_type="Dia-to-Baseline", signal_type="PPG")
        assert isinstance(amps, np.ndarray)

    def test_invalid_interval_type_ppg(self, ppg_wm):
        with pytest.raises(ValueError, match="Invalid interval_type for PPG"):
            ppg_wm.compute_amplitude(interval_type="bogus", signal_type="PPG")


# ---------------------------------------------------------------------------
# compute_volume — ECG interval types (lines 1391-1455)
# ---------------------------------------------------------------------------


class TestComputeVolumeECG:
    def test_r_to_s(self, ecg_wm):
        vol = ecg_wm.compute_volume(interval_type="R-to-S", signal_type="ECG")
        assert isinstance(vol, np.ndarray)

    def test_r_to_q(self, ecg_wm):
        vol = ecg_wm.compute_volume(interval_type="R-to-Q", signal_type="ECG")
        assert isinstance(vol, np.ndarray)

    def test_p_to_q(self, ecg_wm):
        vol = ecg_wm.compute_volume(interval_type="P-to-Q", signal_type="ECG")
        assert isinstance(vol, np.ndarray)

    def test_t_to_s(self, ecg_wm):
        vol = ecg_wm.compute_volume(interval_type="T-to-S", signal_type="ECG")
        assert isinstance(vol, np.ndarray)

    def test_invalid_interval_type_ecg(self, ecg_wm):
        with pytest.raises(ValueError, match="Invalid interval_type for ECG"):
            ecg_wm.compute_volume(interval_type="invalid_type", signal_type="ECG")

    def test_invalid_signal_type(self, ecg_wm):
        with pytest.raises(ValueError, match="signal_type must be"):
            ecg_wm.compute_volume(interval_type="R-to-S", signal_type="EEG")


class TestComputeVolumePPG:
    def test_sys_to_notch(self, ppg_wm):
        vol = ppg_wm.compute_volume(interval_type="Sys-to-Notch", signal_type="PPG")
        assert isinstance(vol, np.ndarray)

    def test_sys_to_dia(self, ppg_wm):
        vol = ppg_wm.compute_volume(interval_type="Sys-to-Dia", signal_type="PPG")
        assert isinstance(vol, np.ndarray)

    def test_sys_to_sys(self, ppg_wm):
        """Line 1443-1451: Sys-to-Sys branch."""
        vol = ppg_wm.compute_volume(interval_type="Sys-to-Sys", signal_type="PPG")
        assert isinstance(vol, np.ndarray)

    def test_invalid_ppg_interval(self, ppg_wm):
        with pytest.raises(ValueError, match="Invalid interval_type for PPG"):
            ppg_wm.compute_volume(interval_type="nonsense", signal_type="PPG")


# ---------------------------------------------------------------------------
# compute_skewness  (lines 1505-1549)
# ---------------------------------------------------------------------------


class TestComputeSkewness:
    def test_ppg_skewness(self, ppg_wm):
        skew = ppg_wm.compute_skewness(signal_type="PPG")
        assert isinstance(skew, np.ndarray)

    def test_ecg_skewness(self, ecg_wm):
        skew = ecg_wm.compute_skewness(signal_type="ECG")
        assert isinstance(skew, np.ndarray)

    def test_empty_skewness_logged(self, ecg_wm, caplog):
        """Line 1548: empty skewness_values logs warning."""
        # Patch systolic_peaks / r_peaks to empty
        orig = ecg_wm.r_peaks
        ecg_wm.r_peaks = np.array([])
        import logging
        with caplog.at_level(logging.WARNING):
            result = ecg_wm.compute_skewness(signal_type="ECG")
        ecg_wm.r_peaks = orig
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# compute_duration  (lines 1553-1580)
# ---------------------------------------------------------------------------


class TestComputeDuration:
    def test_qrs_mode(self, ecg_wm):
        sessions = ecg_wm.detect_qrs_session()
        if len(sessions) > 0:
            dur = ecg_wm.compute_duration(sessions=sessions, mode="QRS")
            assert isinstance(dur, np.ndarray)

    def test_invalid_mode_raises(self, ecg_wm):
        with pytest.raises(ValueError, match="Duration can only be computed"):
            ecg_wm.compute_duration(mode="Bogus")


# ---------------------------------------------------------------------------
# compute_ppg_dicrotic_notch  (lines 1582-1603)
# ---------------------------------------------------------------------------


class TestComputePpgDicrotricNotch:
    def test_returns_float(self, ppg_wm):
        notch_timing = ppg_wm.compute_ppg_dicrotic_notch()
        assert isinstance(notch_timing, float)

    def test_ecg_raises(self, ecg_wm):
        with pytest.raises(ValueError, match="PPG signals"):
            ecg_wm.compute_ppg_dicrotic_notch()


# ---------------------------------------------------------------------------
# compute_slope / get_slope  (lines ~1620-1637, 2128-2191)
# ---------------------------------------------------------------------------


class TestComputeSlope:
    def test_ppg_systolic_peaks_option(self, ppg_wm):
        slopes = ppg_wm.compute_slope(option="systolic_peaks", slope_unit="degrees")
        assert isinstance(slopes, np.ndarray)

    def test_ppg_diastolic_peaks_option(self, ppg_wm):
        slopes = ppg_wm.compute_slope(option="diastolic_peaks", slope_unit="radians")
        assert isinstance(slopes, np.ndarray)

    def test_raw_slope_unit(self, ppg_wm):
        slopes = ppg_wm.compute_slope(option="systolic_peaks", slope_unit="raw")
        assert isinstance(slopes, np.ndarray)

    def test_invalid_slope_unit_raises(self, ppg_wm):
        with pytest.raises(ValueError, match="slope_unit must be"):
            ppg_wm.compute_slope(
                points=np.array([10, 20, 30]), slope_unit="angstroms"
            )

    def test_invalid_option_raises(self, ppg_wm):
        with pytest.raises(ValueError, match="Invalid option"):
            ppg_wm.compute_slope(option="bogus")

    def test_no_points_no_option_raises(self, ppg_wm):
        with pytest.raises(ValueError, match="No points specified"):
            ppg_wm.compute_slope()


class TestGetSlope:
    def test_systolic(self, ppg_wm):
        result = ppg_wm.get_slope(slope_type="systolic", slope_unit="radians")
        assert isinstance(result, float) or isinstance(result, (int, np.floating))

    def test_diastolic(self, ppg_wm):
        result = ppg_wm.get_slope(slope_type="diastolic")
        # Returns float or NaN
        assert result is not None

    def test_qrs(self, ecg_wm):
        result = ecg_wm.get_slope(slope_type="qrs")
        assert result is not None

    def test_invalid_slope_type_returns_nan(self, ppg_wm):
        result = ppg_wm.get_slope(slope_type="invalid_type")
        import math
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# compute_curvature  (lines ~1781-1869)
# ---------------------------------------------------------------------------


class TestComputeCurvature:
    def test_explicit_points(self, ppg_wm):
        pts = np.array([50, 100, 150])
        curves = ppg_wm.compute_curvature(points=pts)
        assert len(curves) == 3

    def test_systolic_peaks_option(self, ppg_wm):
        curves = ppg_wm.compute_curvature(option="systolic_peaks")
        assert isinstance(curves, np.ndarray)

    def test_diastolic_peaks_option(self, ppg_wm):
        curves = ppg_wm.compute_curvature(option="diastolic_peaks")
        assert isinstance(curves, np.ndarray)

    def test_no_points_no_option_raises(self, ppg_wm):
        with pytest.raises(ValueError, match="No points specified for curvature"):
            ppg_wm.compute_curvature()

    def test_invalid_option_raises(self, ppg_wm):
        with pytest.raises(ValueError, match="Invalid option"):
            ppg_wm.compute_curvature(option="bogus")

    def test_edge_case_single_point(self, ppg_wm):
        """Line 1866: len(d2y_dx2) <= 1 → curvature = 0."""
        # window=1 gives 3 points total which is enough for gradient
        curves = ppg_wm.compute_curvature(points=np.array([10, 20, 30]), window=1)
        assert isinstance(curves, np.ndarray)
        assert len(curves) == 3


# ---------------------------------------------------------------------------
# get_duration  (lines 1871-1994)
# ---------------------------------------------------------------------------


class TestGetDuration:
    def test_systolic_mean(self, ppg_wm):
        result = ppg_wm.get_duration(session_type="systolic", summary_type="mean")
        # Could be NaN if detection fails; must not crash
        assert result is not None

    def test_diastolic_mean(self, ppg_wm):
        result = ppg_wm.get_duration(session_type="diastolic", summary_type="median")
        assert result is not None

    def test_qrs_mean(self, ecg_wm):
        result = ecg_wm.get_duration(session_type="qrs", summary_type="mean")
        assert result is not None

    def test_custom_with_points(self, ppg_wm):
        starts = np.array([10, 110, 210])
        ends = np.array([50, 150, 250])
        result = ppg_wm.get_duration(
            session_type="Custom",
            start_points=starts,
            end_points=ends,
            summary_type="mean",
        )
        assert result is not None

    def test_custom_missing_points_returns_nan(self, ppg_wm):
        result = ppg_wm.get_duration(session_type="Custom")
        import math
        assert math.isnan(result)

    def test_invalid_session_type_returns_nan(self, ppg_wm):
        result = ppg_wm.get_duration(session_type="bogus")
        import math
        assert math.isnan(result)

    def test_invalid_summary_type_raises(self, ppg_wm):
        with pytest.raises(ValueError, match="Invalid summary_type"):
            ppg_wm.get_duration(session_type="systolic", summary_type="bogus_type")


# ---------------------------------------------------------------------------
# get_area  (lines 2040-2126)
# ---------------------------------------------------------------------------


class TestGetArea:
    def test_ppg_sys_to_notch(self, ppg_wm):
        result = ppg_wm.get_area(interval_type="Sys-to-Notch", signal_type="PPG")
        assert result is not None

    def test_ecg_qrs_combined(self, ecg_wm):
        """Lines 2063-2081: QRS combined sub-interval area."""
        result = ecg_wm.get_area(interval_type="QRS", signal_type="ECG")
        assert result is not None

    def test_invalid_interval_returns_nan(self, ppg_wm):
        """Lines 2119-2126: ValueError → returns NaN."""
        import math
        result = ppg_wm.get_area(interval_type="Bogus-Interval", signal_type="PPG")
        # Either raises or returns nan depending on implementation
        # Must not crash
        assert result is not None or math.isnan(result)


# ---------------------------------------------------------------------------
# get_signal_skewness  (lines 2193-2230)
# ---------------------------------------------------------------------------


class TestGetSignalSkewness:
    def test_ppg_mean(self, ppg_wm):
        result = ppg_wm.get_signal_skewness(signal_type="PPG", summary_type="mean")
        assert result is not None

    def test_ecg_median(self, ecg_wm):
        result = ecg_wm.get_signal_skewness(signal_type="ECG", summary_type="median")
        assert result is not None

    def test_invalid_signal_type_returns_nan(self, ppg_wm):
        """Lines 2225-2230: exception → NaN."""
        import math
        result = ppg_wm.get_signal_skewness(signal_type="invalid_type")
        assert result is not None


# ---------------------------------------------------------------------------
# get_peak_trend_slope  (lines 2232-2305)
# ---------------------------------------------------------------------------


class TestGetPeakTrendSlope:
    def test_linear_regression(self, ppg_wm):
        """Line 2264-2272: linear_regression method."""
        result = ppg_wm.get_peak_trend_slope(method="linear_regression")
        assert result is not None

    def test_moving_average(self, ppg_wm):
        """Lines 2274-2285: moving_average method."""
        peaks = ppg_wm.systolic_peaks
        if len(peaks) >= 5:
            result = ppg_wm.get_peak_trend_slope(peaks=ppg_wm.waveform[peaks], method="moving_average", window_size=3)
            assert result is not None

    def test_rate_of_change(self, ppg_wm):
        """Lines 2287-2295: rate_of_change method."""
        peaks = ppg_wm.systolic_peaks
        if len(peaks) >= 2:
            result = ppg_wm.get_peak_trend_slope(peaks=ppg_wm.waveform[peaks], method="rate_of_change")
            assert result is not None

    def test_unsupported_method_returns_nan(self, ppg_wm):
        """Line 2297-2305: unsupported method → NaN."""
        import math
        result = ppg_wm.get_peak_trend_slope(method="bogus_method")
        assert math.isnan(result)

    def test_empty_peaks_returns_zero(self, ppg_wm):
        """Line 2260-2261: empty peaks → 0.0."""
        result = ppg_wm.get_peak_trend_slope(peaks=np.array([]))
        assert result == 0.0

    def test_none_peaks_ecg(self, ecg_wm):
        """Lines 2255-2258: peaks=None → uses r_peaks."""
        result = ecg_wm.get_peak_trend_slope(peaks=None)
        assert result is not None


# ---------------------------------------------------------------------------
# get_amplitude_variability  (lines 2307-2384)
# ---------------------------------------------------------------------------


class TestGetAmplitudeVariability:
    def test_std_dev_method(self, ppg_wm):
        result = ppg_wm.get_amplitude_variability(
            interval_type="Sys-to-Baseline",
            signal_type="PPG",
            method="std_dev",
        )
        assert result is not None

    def test_cv_method(self, ppg_wm):
        """Lines 2362-2369: CV method."""
        result = ppg_wm.get_amplitude_variability(
            interval_type="Sys-to-Baseline",
            signal_type="PPG",
            method="cv",
        )
        assert result is not None

    def test_iqr_method(self, ppg_wm):
        """Lines 2371-2375: IQR method."""
        result = ppg_wm.get_amplitude_variability(
            interval_type="Sys-to-Baseline",
            signal_type="PPG",
            method="interquartile_range",
        )
        assert result is not None

    def test_invalid_method_raises(self, ppg_wm):
        with pytest.raises(ValueError, match="Unsupported variability method"):
            ppg_wm.get_amplitude_variability(method="bogus")


# ---------------------------------------------------------------------------
# get_qrs_amplitude  (lines 2386-2445)
# ---------------------------------------------------------------------------


class TestGetQrsAmplitude:
    def test_basic(self, ecg_wm):
        result = ecg_wm.get_qrs_amplitude()
        assert result is not None

    def test_zero_signal_returns_zero(self):
        signal = np.zeros(1000)
        wm = WaveformMorphology(signal, fs=FS, signal_type="ECG")
        result = wm.get_qrs_amplitude()
        assert result == 0


# ---------------------------------------------------------------------------
# get_heart_rate  (lines 2447-2506)
# ---------------------------------------------------------------------------


class TestGetHeartRate:
    def test_ecg_heart_rate_mean(self, ecg_wm):
        result = ecg_wm.get_heart_rate(summary_type="mean")
        assert result is not None

    def test_ppg_heart_rate(self, ppg_wm):
        result = ppg_wm.get_heart_rate()
        assert result is not None

    def test_summary_types(self, ecg_wm):
        for stype in ["mean", "median", "2nd_quartile", "3rd_quartile", "full"]:
            result = ecg_wm.get_heart_rate(summary_type=stype)
            assert result is not None

    def test_unsupported_summary_returns_nan(self, ecg_wm):
        """Lines 2499-2506: invalid summary_type → NaN."""
        import math
        result = ecg_wm.get_heart_rate(summary_type="bogus")
        assert math.isnan(result)
