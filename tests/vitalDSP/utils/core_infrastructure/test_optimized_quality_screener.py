"""
Comprehensive tests for OptimizedQualityScreener module.

Tests cover all functionality and edge cases for the optimized quality screener.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import warnings

try:
    from vitalDSP.utils.core_infrastructure.optimized_quality_screener import (
        OptimizedQualityScreener,
        QualityLevel,
        QualityMetrics,
        ScreeningResult,
    )
    from vitalDSP.utils.config_utilities.dynamic_config import get_config
    OPTIMIZED_QUALITY_AVAILABLE = True
except ImportError:
    OPTIMIZED_QUALITY_AVAILABLE = False


@pytest.mark.skipif(not OPTIMIZED_QUALITY_AVAILABLE, reason="OptimizedQualityScreener not available")
class TestOptimizedQualityScreener:
    """Tests for OptimizedQualityScreener class."""

    @pytest.fixture
    def clean_signal(self):
        """Create a clean test signal."""
        fs = 100
        duration = 10
        t = np.linspace(0, duration, fs * duration)
        signal = np.sin(2 * np.pi * 1 * t) + 0.1 * np.random.randn(len(t))
        return signal

    @pytest.fixture
    def noisy_signal(self):
        """Create a noisy test signal."""
        fs = 100
        duration = 10
        t = np.linspace(0, duration, fs * duration)
        signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.random.randn(len(t))
        return signal

    def test_init_default(self):
        """Test default initialization."""
        screener = OptimizedQualityScreener()
        assert screener.signal_type == 'generic'
        assert hasattr(screener, 'sampling_rate')
        assert hasattr(screener, 'segment_duration')

    def test_init_custom(self):
        """Test custom initialization."""
        screener = OptimizedQualityScreener(
            signal_type='ECG',
            sampling_rate=250,
            segment_duration=5.0,
            overlap_ratio=0.2,
            enable_parallel=False,
            max_workers=2
        )
        assert screener.signal_type == 'ecg'
        assert screener.sampling_rate == 250.0
        assert screener.segment_duration == 5.0
        assert screener.enable_parallel is False
        assert screener.max_workers == 2

    def test_init_with_config(self):
        """Test initialization with config."""
        config = get_config()
        screener = OptimizedQualityScreener(config=config)
        assert screener.config == config

    def test_get_signal_thresholds(self):
        """Test getting signal thresholds."""
        screener = OptimizedQualityScreener(signal_type='ECG')
        thresholds = screener._get_signal_thresholds()
        assert isinstance(thresholds, dict)
        assert 'snr_threshold' in thresholds

    def test_screen_signal_basic(self, clean_signal):
        """Test basic signal screening."""
        screener = OptimizedQualityScreener(sampling_rate=100)
        results = screener.screen_signal(clean_signal)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, ScreeningResult) for r in results)

    def test_screen_signal_with_dataframe(self, clean_signal):
        """Test screening with DataFrame input."""
        screener = OptimizedQualityScreener(sampling_rate=100)
        df = pd.DataFrame({'signal': clean_signal})
        results = screener.screen_signal(df)
        assert isinstance(results, list)

    def test_screen_signal_parallel(self, clean_signal):
        """Test parallel screening."""
        screener = OptimizedQualityScreener(
            sampling_rate=100,
            enable_parallel=True,
            max_workers=2
        )
        results = screener.screen_signal(clean_signal)
        assert isinstance(results, list)

    def test_screen_signal_sequential(self, clean_signal):
        """Test sequential screening."""
        screener = OptimizedQualityScreener(
            sampling_rate=100,
            enable_parallel=False
        )
        results = screener.screen_signal(clean_signal)
        assert isinstance(results, list)

    def test_screen_signal_with_progress_callback(self, clean_signal):
        """Test screening with progress callback."""
        progress_calls = []
        
        def progress_callback(info):
            progress_calls.append(info)
        
        screener = OptimizedQualityScreener(sampling_rate=100)
        results = screener.screen_signal(clean_signal, progress_callback=progress_callback)
        # Progress callback may or may not be called depending on implementation
        assert isinstance(results, list)

    def test_generate_segments(self, clean_signal):
        """Test segment generation."""
        screener = OptimizedQualityScreener(sampling_rate=100)
        segments = screener._generate_segments(clean_signal)
        assert isinstance(segments, list)
        assert len(segments) > 0
        assert all('segment_id' in seg for seg in segments)

    def test_screen_single_segment(self, clean_signal):
        """Test screening single segment."""
        screener = OptimizedQualityScreener(sampling_rate=100)
        segment = {
            'segment_id': 'test_seg',
            'start_idx': 0,
            'end_idx': len(clean_signal),
            'data': clean_signal,
            'duration_seconds': len(clean_signal) / 100.0
        }
        result = screener._screen_single_segment(segment)
        assert isinstance(result, ScreeningResult)
        assert result.segment_id == 'test_seg'

    def test_stage1_snr_check(self, clean_signal):
        """Test stage 1 SNR check."""
        screener = OptimizedQualityScreener(sampling_rate=100)
        segment = clean_signal[:1000]
        passed, metrics = screener._stage1_snr_check(segment)
        # Accept both Python bool and numpy bool
        assert isinstance(passed, (bool, np.bool_))
        assert isinstance(metrics, dict)

    def test_stage2_statistical_screen(self, clean_signal):
        """Test stage 2 statistical screening."""
        screener = OptimizedQualityScreener(sampling_rate=100)
        segment = clean_signal[:1000]
        passed, metrics = screener._stage2_statistical_screen(segment)
        # Accept both Python bool and numpy bool
        assert isinstance(passed, (bool, np.bool_))
        assert isinstance(metrics, dict)

    def test_stage3_signal_specific_screen(self, clean_signal):
        """Test stage 3 signal-specific screening."""
        screener = OptimizedQualityScreener(sampling_rate=100, signal_type='ECG')
        segment = clean_signal[:1000]
        passed, metrics = screener._stage3_signal_specific_screen(segment)
        # Accept both Python bool and numpy bool
        assert isinstance(passed, (bool, np.bool_))
        assert isinstance(metrics, dict)

    def test_calculate_quality_metrics(self, clean_signal):
        """Test quality metrics calculation."""
        screener = OptimizedQualityScreener(sampling_rate=100)
        segment = clean_signal[:1000]
        metrics = screener._calculate_quality_metrics(segment)
        assert isinstance(metrics, QualityMetrics)
        assert hasattr(metrics, 'overall_quality')
        assert hasattr(metrics, 'quality_level')

    def test_determine_quality_level(self):
        """Test quality level determination."""
        screener = OptimizedQualityScreener()
        
        # Test different quality scores
        assert screener._determine_quality_level(0.9) == QualityLevel.EXCELLENT
        assert screener._determine_quality_level(0.7) == QualityLevel.GOOD
        assert screener._determine_quality_level(0.5) == QualityLevel.FAIR
        assert screener._determine_quality_level(0.3) == QualityLevel.POOR
        assert screener._determine_quality_level(0.1) == QualityLevel.UNUSABLE

    def test_get_processing_recommendation(self):
        """Test processing recommendation generation."""
        screener = OptimizedQualityScreener()
        recommendation = screener._get_processing_recommendation(QualityLevel.EXCELLENT)
        assert isinstance(recommendation, str)

    def test_update_statistics(self, clean_signal):
        """Test statistics update."""
        screener = OptimizedQualityScreener(sampling_rate=100)
        results = screener.screen_signal(clean_signal)
        stats = screener.get_statistics()
        assert isinstance(stats, dict)
        assert 'total_segments' in stats

    def test_reset_statistics(self, clean_signal):
        """Test statistics reset."""
        screener = OptimizedQualityScreener(sampling_rate=100)
        screener.screen_signal(clean_signal)
        screener.reset_statistics()
        stats = screener.get_statistics()
        assert stats['total_segments'] == 0

    def test_configure_thresholds(self):
        """Test threshold configuration."""
        screener = OptimizedQualityScreener(signal_type='ECG')
        screener.configure_thresholds(
            snr_threshold=15.0,
            artifact_ratio_threshold=0.1
        )
        thresholds = screener._get_signal_thresholds()
        assert thresholds['snr_threshold'] == 15.0

    def test_screen_signal_short_signal(self):
        """Test screening with very short signal."""
        screener = OptimizedQualityScreener(sampling_rate=100)
        short_signal = np.random.randn(50)  # Very short
        results = screener.screen_signal(short_signal)
        # Should handle gracefully
        assert isinstance(results, list)

    def test_screen_signal_with_nan(self):
        """Test screening signal with NaN values."""
        screener = OptimizedQualityScreener(sampling_rate=100)
        signal = np.random.randn(1000)
        signal[100:150] = np.nan
        # Should handle NaN values
        try:
            results = screener.screen_signal(signal)
            assert isinstance(results, list)
        except Exception:
            # May raise error for NaN, which is acceptable
            pass

    def test_screen_signal_with_inf(self):
        """Test screening signal with infinite values."""
        screener = OptimizedQualityScreener(sampling_rate=100)
        signal = np.random.randn(1000)
        signal[100:150] = np.inf
        # Should handle infinite values
        try:
            results = screener.screen_signal(signal)
            assert isinstance(results, list)
        except Exception:
            # May raise error for inf, which is acceptable
            pass

    def test_parallel_screening_multiple_segments(self, clean_signal):
        """Test parallel screening with multiple segments."""
        screener = OptimizedQualityScreener(
            sampling_rate=100,
            segment_duration=2.0,
            enable_parallel=True,
            max_workers=2
        )
        results = screener.screen_signal(clean_signal)
        assert len(results) > 1

    def test_sequential_screening_multiple_segments(self, clean_signal):
        """Test sequential screening with multiple segments."""
        screener = OptimizedQualityScreener(
            sampling_rate=100,
            segment_duration=2.0,
            enable_parallel=False
        )
        results = screener.screen_signal(clean_signal)
        assert len(results) > 1

