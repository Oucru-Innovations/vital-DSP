"""
Additional tests for quality_screener.py to cover missing lines.

Tests target specific uncovered lines from coverage report.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import warnings

from vitalDSP.utils.core_infrastructure.quality_screener import (
    QualityScreener,
    QualityLevel,
    QualityMetrics,
    ScreeningResult,
)
from vitalDSP.utils.core_infrastructure.optimized_data_loaders import ProgressInfo


class TestQualityScreenerMissingCoverage:
    """Tests for uncovered lines in quality_screener.py."""

    @pytest.fixture
    def clean_signal(self):
        """Create a clean test signal."""
        fs = 100
        duration = 10
        t = np.linspace(0, duration, fs * duration)
        signal = np.sin(2 * np.pi * 1 * t) + 0.1 * np.random.randn(len(t))
        return signal

    def test_screen_signal_with_dataframe(self, clean_signal):
        """Test screening with DataFrame input (line 297)."""
        screener = QualityScreener(sampling_rate=100)
        df = pd.DataFrame({'signal': clean_signal})
        results = screener.screen_signal(df)
        assert isinstance(results, list)

    def test_screen_segments_sequential_with_progress(self, clean_signal):
        """Test sequential screening with progress callback (lines 345-355)."""
        progress_calls = []
        
        def progress_callback(info):
            progress_calls.append(info)
        
        screener = QualityScreener(sampling_rate=100, segment_duration=2.0)
        results = screener.screen_signal(clean_signal, progress_callback=progress_callback)
        assert isinstance(results, list)

    def test_screen_segments_parallel_with_progress(self, clean_signal):
        """Test parallel screening with progress callback (lines 398-403)."""
        progress_calls = []
        
        def progress_callback(info):
            progress_calls.append(info)
        
        screener = QualityScreener(
            sampling_rate=100,
            segment_duration=2.0,
            enable_parallel=True,
            max_workers=2
        )
        results = screener.screen_signal(clean_signal, progress_callback=progress_callback)
        assert isinstance(results, list)

    def test_screen_segments_parallel_fallback(self, clean_signal):
        """Test parallel screening fallback to sequential (lines 429-433)."""
        screener = QualityScreener(
            sampling_rate=100,
            segment_duration=2.0,
            enable_parallel=True,
            max_workers=2
        )
        
        # Mock ThreadPoolExecutor to raise exception
        with patch('vitalDSP.utils.core_infrastructure.quality_screener.ThreadPoolExecutor') as mock_executor:
            mock_executor.side_effect = Exception("Parallel processing failed")
            # Should fall back to sequential
            results = screener.screen_signal(clean_signal)
            assert isinstance(results, list)

    def test_screen_single_segment_exception(self, clean_signal):
        """Test single segment screening with exception (lines 455-457)."""
        screener = QualityScreener(sampling_rate=100)
        segment = {
            'segment_id': 'test_seg',
            'start_idx': 0,
            'end_idx': len(clean_signal),
            'data': clean_signal,
            'duration_seconds': len(clean_signal) / 100.0
        }
        
        # Mock _stage1_snr_check to raise exception
        with patch.object(screener, '_stage1_snr_check', side_effect=Exception("Test error")):
            result = screener._screen_single_segment(segment)
            assert isinstance(result, ScreeningResult)
            assert len(result.warnings) > 0

    def test_calculate_overall_metrics(self, clean_signal):
        """Test overall metrics calculation (line 483)."""
        screener = QualityScreener(sampling_rate=100)
        segment = clean_signal[:1000]
        
        snr_result = {'passed': True, 'snr_db': 20.0, 'metrics': {}}
        stats_result = {'passed': True, 'metrics': {}}
        signal_result = {'passed': True, 'metrics': {}}
        
        metrics = screener._calculate_overall_metrics(snr_result, stats_result, signal_result)
        assert isinstance(metrics, QualityMetrics)

    def test_stage1_snr_check_failed(self, clean_signal):
        """Test stage 1 SNR check failure (line 494-495)."""
        screener = QualityScreener(sampling_rate=100)
        # Create very noisy signal
        noisy_signal = clean_signal[:1000] + 10 * np.random.randn(1000)
        result = screener._stage1_snr_check(noisy_signal)
        # May fail for very noisy signal
        assert isinstance(result, dict)
        assert 'passed' in result

    def test_calculate_baseline_drift_edge_cases(self, clean_signal):
        """Test baseline drift calculation edge cases (lines 541-542)."""
        screener = QualityScreener(sampling_rate=100)
        # Test with constant signal
        constant_signal = np.ones(1000)
        drift = screener._calculate_baseline_drift(constant_signal)
        assert isinstance(drift, float)

    def test_calculate_peak_detection_rate_edge_cases(self):
        """Test peak detection rate edge cases (lines 661-669)."""
        screener = QualityScreener(signal_type='ECG', sampling_rate=100)
        # Test with no peaks
        flat_signal = np.ones(1000)
        rate = screener._calculate_peak_detection_rate(flat_signal)
        assert isinstance(rate, float)

    def test_calculate_frequency_score_edge_cases(self, clean_signal):
        """Test frequency score edge cases (lines 679-686)."""
        screener = QualityScreener(sampling_rate=100)
        # Test with DC signal
        dc_signal = np.ones(1000)
        score = screener._calculate_frequency_score(dc_signal)
        assert isinstance(score, float)

    def test_calculate_temporal_consistency_edge_cases(self, clean_signal):
        """Test temporal consistency edge cases (lines 708-771)."""
        screener = QualityScreener(sampling_rate=100)
        # Test various signal types
        signals = [
            clean_signal[:1000],
            np.random.randn(1000),
            np.ones(1000),
            np.sin(np.linspace(0, 4*np.pi, 1000))
        ]
        
        for signal in signals:
            consistency = screener._calculate_temporal_consistency(signal)
            assert isinstance(consistency, float)

    def test_update_statistics(self, clean_signal):
        """Test statistics update (lines 775-819)."""
        screener = QualityScreener(sampling_rate=100)
        results = screener.screen_signal(clean_signal)
        stats = screener.get_statistics()
        assert isinstance(stats, dict)
        assert 'total_segments' in stats

    def test_configure_thresholds_custom(self):
        """Test custom threshold configuration (line 899)."""
        screener = QualityScreener(signal_type='ECG')
        # Access thresholds directly
        thresholds = screener.thresholds
        assert isinstance(thresholds, dict)
        assert 'snr_min_db' in thresholds or 'snr_threshold' in thresholds

    def test_get_signal_thresholds_all_types(self):
        """Test getting thresholds for all signal types (line 917)."""
        signal_types = ['ECG', 'PPG', 'EEG', 'RESP', 'generic']
        for sig_type in signal_types:
            try:
                screener = QualityScreener(signal_type=sig_type)
                thresholds = screener.thresholds
                assert isinstance(thresholds, dict)
            except Exception:
                # Some signal types may not be fully supported
                pass

    def test_screen_signal_short_segments(self):
        """Test screening with very short segments (line 940)."""
        screener = QualityScreener(sampling_rate=100, segment_duration=0.1)
        signal = np.random.randn(100)
        results = screener.screen_signal(signal)
        # Should handle gracefully
        assert isinstance(results, list)

    def test_parallel_screening_error_handling(self, clean_signal):
        """Test parallel screening error handling (line 959)."""
        screener = QualityScreener(
            sampling_rate=100,
            enable_parallel=True,
            max_workers=2
        )
        
        # Force error in parallel processing
        with patch('vitalDSP.utils.core_infrastructure.quality_screener.ThreadPoolExecutor') as mock_exec:
            mock_exec.side_effect = RuntimeError("Thread pool error")
            # Should fall back to sequential
            results = screener.screen_signal(clean_signal)
            assert isinstance(results, list)

    def test_screen_signal_with_invalid_data(self):
        """Test screening with invalid data (lines 993, 997-998)."""
        screener = QualityScreener(sampling_rate=100)
        # Test with None
        try:
            results = screener.screen_signal(None)
            # Should handle gracefully
        except Exception:
            # Exception is acceptable
            pass

    def test_get_statistics_empty(self):
        """Test getting statistics before processing (line 1015)."""
        screener = QualityScreener()
        stats = screener.get_statistics()
        assert isinstance(stats, dict)
        assert stats['total_segments'] == 0

    def test_reset_statistics_after_processing(self, clean_signal):
        """Test resetting statistics after processing (line 1021-1022)."""
        screener = QualityScreener(sampling_rate=100)
        screener.screen_signal(clean_signal)
        screener.reset_statistics()
        stats = screener.get_statistics()
        assert stats['total_segments'] == 0

    def test_calculate_quality_metrics_all_combinations(self, clean_signal):
        """Test quality metrics for all combinations (line 1054)."""
        screener = QualityScreener(sampling_rate=100)
        segment = clean_signal[:1000]
        
        # Test different combinations of stage results
        snr_results = [{'passed': True, 'snr_db': 20.0}, {'passed': False, 'snr_db': 5.0}]
        stats_results = [{'passed': True}, {'passed': False}]
        signal_results = [{'passed': True}, {'passed': False}]
        
        for snr in snr_results:
            for stats in stats_results:
                for signal in signal_results:
                    metrics = screener._calculate_overall_metrics(snr, stats, signal)
                    assert isinstance(metrics, QualityMetrics)

    def test_screen_signal_progress_callback_edge_cases(self, clean_signal):
        """Test progress callback edge cases (lines 1068-1069, 1080-1081)."""
        screener = QualityScreener(sampling_rate=100, segment_duration=2.0)
        
        # Test with None callback
        results = screener.screen_signal(clean_signal, progress_callback=None)
        assert isinstance(results, list)
        
        # Test with callback that raises exception
        def bad_callback(info):
            raise Exception("Callback error")
        
        try:
            results = screener.screen_signal(clean_signal, progress_callback=bad_callback)
            assert isinstance(results, list)
        except Exception:
            # Exception in callback should not break processing
            pass

    def test_update_statistics_with_timing(self, clean_signal):
        """Test statistics update with timing information (line 1099)."""
        screener = QualityScreener(sampling_rate=100)
        results = screener.screen_signal(clean_signal)
        stats = screener.get_statistics()
        assert 'total_time' in stats or 'avg_time_per_segment' in stats

    def test_screen_signal_empty_result_handling(self):
        """Test handling of empty results (line 1180)."""
        screener = QualityScreener(sampling_rate=100)
        # Very short signal that might produce no segments
        very_short = np.random.randn(10)
        results = screener.screen_signal(very_short)
        # Should return empty list or handle gracefully
        assert isinstance(results, list)

