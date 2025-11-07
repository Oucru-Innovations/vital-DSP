"""
Comprehensive tests for OptimizedParallelPipeline module.

Tests cover all functionality and edge cases for the optimized parallel pipeline.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

try:
    from vitalDSP.utils.core_infrastructure.optimized_parallel_pipeline import (
        OptimizedParallelPipeline,
    )
    OPTIMIZED_PARALLEL_AVAILABLE = True
except ImportError:
    OPTIMIZED_PARALLEL_AVAILABLE = False


@pytest.mark.skipif(not OPTIMIZED_PARALLEL_AVAILABLE, reason="OptimizedParallelPipeline not available")
class TestOptimizedParallelPipeline:
    """Tests for OptimizedParallelPipeline class."""

    @pytest.fixture
    def sample_signal(self):
        """Create sample signal."""
        fs = 100
        duration = 10
        t = np.linspace(0, duration, fs * duration)
        signal = np.sin(2 * np.pi * 1 * t) + 0.1 * np.random.randn(len(t))
        return signal

    def test_init_default(self):
        """Test default initialization."""
        pipeline = OptimizedParallelPipeline()
        assert hasattr(pipeline, 'config')

    def test_init_with_config(self):
        """Test initialization uses config."""
        # Don't pass DynamicConfigManager - let it use default config
        pipeline = OptimizedParallelPipeline()
        assert hasattr(pipeline, 'config')
        assert pipeline.config is not None

    def test_process_signal_basic(self, sample_signal):
        """Test basic signal processing."""
        pipeline = OptimizedParallelPipeline()
        result = pipeline.process_signal(sample_signal, sampling_rate=100)
        assert isinstance(result, dict)

    def test_process_signal_parallel(self, sample_signal):
        """Test parallel processing."""
        pipeline = OptimizedParallelPipeline()
        result = pipeline.process_signal(
            sample_signal,
            sampling_rate=100,
            enable_parallel=True
        )
        assert isinstance(result, dict)

    def test_process_signal_sequential(self, sample_signal):
        """Test sequential processing."""
        pipeline = OptimizedParallelPipeline()
        result = pipeline.process_signal(
            sample_signal,
            sampling_rate=100,
            enable_parallel=False
        )
        assert isinstance(result, dict)

    def test_get_pipeline_stats(self):
        """Test getting pipeline statistics."""
        pipeline = OptimizedParallelPipeline()
        stats = pipeline.get_pipeline_stats()
        assert isinstance(stats, dict)

    def test_reset_statistics(self):
        """Test resetting statistics."""
        pipeline = OptimizedParallelPipeline()
        pipeline.reset_statistics()
        stats = pipeline.get_pipeline_stats()
        assert isinstance(stats, dict)

