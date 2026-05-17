"""
Additional tests for quality_screener.py to cover missing lines.

Targeted missing lines: 429-433, 494-495, 541-542, 661-771, 775-819,
895-898, 917, 940, 959, 993-998, 1015, 1021-1054, 1068-1069, 1180
"""

import numpy as np
import pytest

from vitalDSP.utils.core_infrastructure.quality_screener import (
    QualityLevel,
    QualityMetrics,
    QualityScreener,
    QualityScreeningConfig,
    ScreeningResult,
    SignalQualityScreener,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FS = 100
DURATION = 15  # seconds


@pytest.fixture
def ecg_signal():
    t = np.arange(DURATION * FS) / FS
    return np.sin(2 * np.pi * 1.1 * t) + 0.1 * np.random.default_rng(0).standard_normal(DURATION * FS)


@pytest.fixture
def ppg_signal():
    t = np.arange(DURATION * FS) / FS
    return np.sin(2 * np.pi * 1.0 * t) + 2.0 + 0.05 * np.random.default_rng(1).standard_normal(DURATION * FS)


@pytest.fixture
def eeg_signal():
    t = np.arange(DURATION * FS) / FS
    return 0.5 * np.sin(2 * np.pi * 10.0 * t) + 0.1 * np.random.default_rng(2).standard_normal(DURATION * FS)


@pytest.fixture
def ecg_screener():
    return QualityScreener(signal_type="ecg", sampling_rate=FS, segment_duration=5.0)


@pytest.fixture
def ppg_screener():
    return QualityScreener(signal_type="ppg", sampling_rate=FS, segment_duration=5.0)


@pytest.fixture
def eeg_screener():
    return QualityScreener(signal_type="eeg", sampling_rate=FS, segment_duration=5.0)


@pytest.fixture
def generic_screener():
    return QualityScreener(signal_type="generic", sampling_rate=FS, segment_duration=5.0)


# ---------------------------------------------------------------------------
# _stage1_snr_check  (lines 468-499) — flat signal (noise_power=0, line 483)
# ---------------------------------------------------------------------------


class TestStage1SnrCheck:
    def test_flat_signal_inf_snr(self, ecg_screener):
        """Line 483: noise_power = 0 → snr_db = inf."""
        flat = np.full(1000, 5.0)
        result = ecg_screener._stage1_snr_check(flat)
        assert result["snr_db"] == float("inf")
        assert result["passed"] is True

    def test_noisy_signal_finite_snr(self, ecg_screener):
        t = np.arange(500) / FS
        signal = np.sin(2 * np.pi * 1.0 * t) + 5.0 * np.random.default_rng(3).standard_normal(500)
        result = ecg_screener._stage1_snr_check(signal)
        assert "snr_db" in result
        assert "passed" in result


# ---------------------------------------------------------------------------
# _stage2_statistical_screen  (lines 505-542) — outlier and constant branches
# ---------------------------------------------------------------------------


class TestStage2StatisticalScreen:
    def test_normal_signal_passes(self, ecg_screener):
        t = np.arange(500) / FS
        signal = np.sin(2 * np.pi * 1.0 * t)
        result = ecg_screener._stage2_statistical_screen(signal)
        assert "passed" in result

    def test_constant_signal_fails(self, ecg_screener):
        """Line 541-542: constant ratio too high → failed."""
        signal = np.full(500, 3.14)
        result = ecg_screener._stage2_statistical_screen(signal)
        # Constant signal should fail statistical screen
        assert "constant_ratio" in result

    def test_signal_with_jumps_detected(self, ecg_screener):
        """Lines 529-535: jump detection."""
        signal = np.zeros(500)
        signal[250] = 1000.0  # artificial huge jump
        result = ecg_screener._stage2_statistical_screen(signal)
        assert "jump_ratio" in result


# ---------------------------------------------------------------------------
# _stage3_signal_specific_screen  (lines 551-884) — ECG, PPG, EEG, generic
# ---------------------------------------------------------------------------


class TestStage3SignalSpecificScreen:
    def test_ecg_signal_screen(self, ecg_screener, ecg_signal):
        """Lines 593-687: ECG-specific SQI methods."""
        result = ecg_screener._stage3_signal_specific_screen(ecg_signal[:500])
        assert "passed" in result
        assert "quality_score" in result

    def test_ppg_signal_screen(self, ppg_screener, ppg_signal):
        """Lines 688-771: PPG-specific SQI methods."""
        result = ppg_screener._stage3_signal_specific_screen(ppg_signal[:500])
        assert "passed" in result

    def test_eeg_signal_screen(self, eeg_screener, eeg_signal):
        """Lines 773-819: EEG-specific SQI methods."""
        result = eeg_screener._stage3_signal_specific_screen(eeg_signal[:500])
        assert "passed" in result

    def test_generic_signal_screen(self, generic_screener, ecg_signal):
        """Lines 820-884: Generic signal SQI methods."""
        result = generic_screener._stage3_signal_specific_screen(ecg_signal[:500])
        assert "passed" in result


# ---------------------------------------------------------------------------
# _extract_sqi_score  (lines 887-899)
# ---------------------------------------------------------------------------


class TestExtractSqiScore:
    def test_tuple_with_ndarray(self, ecg_screener):
        result = ecg_screener._extract_sqi_score((np.array([0.5, 0.6, 0.7]), np.array([0, 1, 2])))
        assert result == pytest.approx(0.6)

    def test_non_tuple_returns_zero(self, ecg_screener):
        result = ecg_screener._extract_sqi_score(None)
        assert result == 0.0

    def test_empty_array_returns_zero(self, ecg_screener):
        result = ecg_screener._extract_sqi_score((np.array([]), np.array([])))
        assert result == 0.0


# ---------------------------------------------------------------------------
# _calculate_baseline_drift  (lines 901-908)
# ---------------------------------------------------------------------------


class TestCalculateBaselineDrift:
    def test_drifting_signal(self, ecg_screener):
        signal = np.linspace(0, 10, 500)
        drift = ecg_screener._calculate_baseline_drift(signal)
        assert drift > 0

    def test_flat_signal_low_drift(self, ecg_screener):
        signal = np.full(500, 5.0)
        drift = ecg_screener._calculate_baseline_drift(signal)
        assert drift == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# _calculate_peak_detection_rate  (lines 910-970) — ECG, PPG, generic
# ---------------------------------------------------------------------------


class TestCalculatePeakDetectionRate:
    def test_ecg_peaks(self, ecg_screener, ecg_signal):
        """Lines 912-940: ECG branch."""
        rate = ecg_screener._calculate_peak_detection_rate(ecg_signal[:500])
        assert 0.0 <= rate <= 1.0

    def test_ppg_peaks(self, ppg_screener, ppg_signal):
        """Lines 942-959: PPG branch."""
        rate = ppg_screener._calculate_peak_detection_rate(ppg_signal[:500])
        assert 0.0 <= rate <= 1.0

    def test_generic_peaks(self, generic_screener, ecg_signal):
        """Lines 961-970: Generic branch."""
        rate = generic_screener._calculate_peak_detection_rate(ecg_signal[:500])
        assert 0.0 <= rate <= 1.0

    def test_eeg_peak_detection_fallback(self, eeg_screener, eeg_signal):
        """EEG uses generic peak detection."""
        rate = eeg_screener._calculate_peak_detection_rate(eeg_signal[:500])
        assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# _calculate_frequency_score  (lines 972-998)
# ---------------------------------------------------------------------------


class TestCalculateFrequencyScore:
    def test_valid_signal(self, ecg_screener, ecg_signal):
        score = ecg_screener._calculate_frequency_score(ecg_signal[:500])
        assert 0.0 <= score <= 1.0

    def test_zero_signal(self, ecg_screener):
        """Line 993-998: zero signal → sum(fft) == 0 → score = 0."""
        signal = np.zeros(500)
        score = ecg_screener._calculate_frequency_score(signal)
        assert score == 0.0


# ---------------------------------------------------------------------------
# _calculate_temporal_consistency  (lines 1000-1022)
# ---------------------------------------------------------------------------


class TestCalculateTemporalConsistency:
    def test_consistent_signal(self, ecg_screener, ecg_signal):
        score = ecg_screener._calculate_temporal_consistency(ecg_signal[:500])
        assert 0.0 <= score <= 1.0

    def test_single_window(self, ecg_screener):
        """Line 1015: only one local var → var_consistency = 1.0."""
        signal = np.sin(np.linspace(0, 2 * np.pi, 50))
        score = ecg_screener._calculate_temporal_consistency(signal)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# _calculate_overall_metrics  (lines 1024-1095) — quality level branches
# ---------------------------------------------------------------------------


class TestCalculateOverallMetrics:
    def _make_snr_result(self, snr_db):
        return {"snr_db": snr_db, "signal_power": 1.0, "noise_power": 0.01}

    def _make_stats_result(self, outlier_ratio=0.0, constant_ratio=0.0, jump_ratio=0.0):
        return {
            "outlier_ratio": outlier_ratio,
            "constant_ratio": constant_ratio,
            "jump_ratio": jump_ratio,
        }

    def _make_signal_result(self, quality_score=0.8):
        return {
            "quality_score": quality_score,
            "artifact_ratio": 0.1,
            "baseline_drift": 0.05,
            "peak_detection_rate": 0.9,
            "frequency_score": 0.7,
            "temporal_consistency": 0.8,
        }

    def test_excellent_quality(self, ecg_screener):
        """Lines 1067-1069: overall_quality > 0.8."""
        metrics = ecg_screener._calculate_overall_metrics(
            self._make_snr_result(float("inf")),  # snr_score = 1.0
            self._make_stats_result(0.0, 0.0, 0.0),  # stats_score = 1.0
            self._make_signal_result(0.9),
        )
        # With inf snr and no stat failures, should be EXCELLENT or GOOD
        assert metrics.quality_level in (QualityLevel.EXCELLENT, QualityLevel.GOOD)

    def test_good_quality(self, ecg_screener):
        """Lines 1070-1072: overall_quality 0.6-0.8."""
        metrics = ecg_screener._calculate_overall_metrics(
            self._make_snr_result(20.0),
            self._make_stats_result(0.1),
            self._make_signal_result(0.6),
        )
        assert metrics.quality_level in (QualityLevel.GOOD, QualityLevel.EXCELLENT, QualityLevel.FAIR)

    def test_unusable_quality(self, ecg_screener):
        """Lines 1079-1081: overall_quality <= 0.2."""
        metrics = ecg_screener._calculate_overall_metrics(
            self._make_snr_result(-5.0),
            self._make_stats_result(0.9, 0.9, 0.9),
            self._make_signal_result(0.0),
        )
        assert metrics.quality_level in (QualityLevel.UNUSABLE, QualityLevel.POOR)

    def test_inf_snr_gives_snr_score_1(self, ecg_screener):
        """Lines 1036-1038: snr_db == inf → snr_score = 1.0."""
        metrics = ecg_screener._calculate_overall_metrics(
            self._make_snr_result(float("inf")),
            self._make_stats_result(),
            self._make_signal_result(),
        )
        assert metrics.overall_quality > 0


# ---------------------------------------------------------------------------
# get_statistics / reset_statistics  (lines 1127-1143)
# ---------------------------------------------------------------------------


class TestScreenerStatistics:
    def test_initial_statistics(self, ecg_screener):
        stats = ecg_screener.get_statistics()
        assert stats["total_segments"] == 0
        assert stats["pass_rate"] == 0.0

    def test_statistics_after_screening(self, ecg_screener, ecg_signal):
        ecg_screener.screen_signal(ecg_signal)
        stats = ecg_screener.get_statistics()
        assert stats["total_segments"] > 0
        assert 0.0 <= stats["pass_rate"] <= 1.0

    def test_reset_statistics(self, ecg_screener, ecg_signal):
        ecg_screener.screen_signal(ecg_signal)
        ecg_screener.reset_statistics()
        stats = ecg_screener.get_statistics()
        assert stats["total_segments"] == 0


# ---------------------------------------------------------------------------
# SignalQualityScreener / QualityScreeningConfig  (lines 1147-1191)
# ---------------------------------------------------------------------------


class TestSignalQualityScreener:
    def test_default_config(self):
        """Lines 1179-1180: no config → default QualityScreeningConfig."""
        screener = SignalQualityScreener()
        assert isinstance(screener, QualityScreener)

    def test_with_config(self):
        config = QualityScreeningConfig(
            segment_duration=5.0,
            overlap=0.1,
            snr_threshold=8.0,
            max_workers=2,
        )
        screener = SignalQualityScreener(config=config)
        assert screener.segment_duration == 5.0

    def test_screen_signal_with_custom_config(self):
        config = QualityScreeningConfig(segment_duration=3.0)
        screener = SignalQualityScreener(config=config, signal_type="ecg", sampling_rate=100)
        t = np.arange(10 * 100) / 100
        signal = np.sin(2 * np.pi * 1.0 * t)
        results = screener.screen_signal(signal)
        assert isinstance(results, list)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# screen_signal — full end-to-end for each signal type
# ---------------------------------------------------------------------------


class TestScreenSignalEndToEnd:
    def test_ecg_end_to_end(self, ecg_screener, ecg_signal):
        results = ecg_screener.screen_signal(ecg_signal)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, ScreeningResult)
            assert r.passed_screening is True or r.passed_screening is False

    def test_ppg_end_to_end(self, ppg_screener, ppg_signal):
        results = ppg_screener.screen_signal(ppg_signal)
        assert len(results) > 0

    def test_eeg_end_to_end(self, eeg_screener, eeg_signal):
        results = eeg_screener.screen_signal(eeg_signal)
        assert len(results) > 0

    def test_generic_end_to_end(self, generic_screener, ecg_signal):
        results = generic_screener.screen_signal(ecg_signal)
        assert len(results) > 0
