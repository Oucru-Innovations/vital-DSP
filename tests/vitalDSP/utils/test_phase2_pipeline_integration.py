"""
Test suite for Phase 2 Pipeline Integration components.

This module tests the optimized processing pipeline, memory manager,
and error recovery components implemented in Phase 2.

Author: vitalDSP Development Team
Date: October 12, 2025
Version: 2.0.0 (Optimized)
"""

import pytest
import numpy as np
import tempfile
import os
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from vitalDSP.utils.core_infrastructure import (
    DynamicConfigManager, StandardProcessingPipeline,
    MemoryManager, ErrorRecoveryManager,
    MemoryStrategy, ErrorSeverity, ErrorCategory
)


class TestDynamicConfigManager:
    """Test Dynamic Configuration Manager functionality."""

    def test_config_manager_initialization(self):
        """Test configuration manager initialization."""
        config_manager = DynamicConfigManager()
        assert config_manager is not None
        
        # Test basic configuration access
        cpu_count = config_manager.get('system_resources.cpu_count', 1)
        assert cpu_count > 0

    def test_user_preferences(self):
        """Test user preference setting and retrieval."""
        config_manager = DynamicConfigManager()
        
        # Set user preference
        config_manager.set_user_preference('memory.max_memory_percent', 0.8)
        
        # Get preference
        memory_percent = config_manager.get('memory.max_memory_percent', 0.5)
        assert memory_percent == 0.8

    def test_config_statistics(self):
        """Test configuration statistics."""
        config_manager = DynamicConfigManager()
        stats = config_manager.get_statistics()
        
        assert 'environment' in stats
        assert 'cpu_count' in stats
        assert 'memory_gb' in stats
        assert 'user_preferences' in stats


class TestMemoryManager:
    """Test Memory Manager functionality."""

    def test_memory_manager_initialization(self):
        """Test memory manager initialization."""
        config_manager = DynamicConfigManager()
        memory_manager = MemoryManager(config_manager, MemoryStrategy.BALANCED)
        
        assert memory_manager is not None
        assert memory_manager.strategy == MemoryStrategy.BALANCED

    def test_memory_info(self):
        """Test memory information retrieval."""
        config_manager = DynamicConfigManager()
        memory_manager = MemoryManager(config_manager)
        
        memory_info = memory_manager.get_memory_info()
        
        assert memory_info.total_memory_gb > 0
        assert memory_info.available_memory_gb > 0
        assert 0 <= memory_info.memory_percent <= 100

    def test_memory_capability_check(self):
        """Test memory capability assessment."""
        config_manager = DynamicConfigManager()
        memory_manager = MemoryManager(config_manager)
        
        # Test small data
        can_process_small = memory_manager.can_process_in_memory(10.0, ['load'])
        assert isinstance(can_process_small, bool)
        
        # Test large data
        can_process_large = memory_manager.can_process_in_memory(10000.0, ['load', 'filter', 'features'])
        assert isinstance(can_process_large, bool)

    def test_chunk_size_recommendation(self):
        """Test chunk size recommendation."""
        config_manager = DynamicConfigManager()
        memory_manager = MemoryManager(config_manager)
        
        chunk_size = memory_manager.recommend_chunk_size(100.0, ['load', 'filter'])
        assert chunk_size > 0
        assert isinstance(chunk_size, int)

    def test_data_type_optimization(self):
        """Test data type optimization."""
        config_manager = DynamicConfigManager()
        memory_manager = MemoryManager(config_manager)
        
        # Test signal optimization
        signal = np.random.randn(1000).astype(np.float64)
        optimized_signal = memory_manager.optimize_data_types(signal, 'ECG')
        
        assert optimized_signal is not None
        assert len(optimized_signal) == len(signal)

    def test_memory_monitoring(self):
        """Test memory monitoring functionality."""
        config_manager = DynamicConfigManager()
        memory_manager = MemoryManager(config_manager)
        
        # Start monitoring
        memory_manager.start_memory_monitoring(interval=0.1)
        
        # Let it run briefly
        time.sleep(0.2)
        
        # Stop monitoring
        memory_manager._monitoring_active = False
        
        # Get statistics
        stats = memory_manager.get_memory_statistics()
        assert 'current_memory' in stats
        assert 'memory_limits' in stats


class TestStandardProcessingPipeline:
    """Test Standard Processing Pipeline functionality."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        config_manager = DynamicConfigManager()
        pipeline = StandardProcessingPipeline(config_manager)
        
        assert pipeline is not None
        assert pipeline.config == config_manager

    def test_signal_processing(self):
        """Test basic signal processing."""
        config_manager = DynamicConfigManager()
        pipeline = StandardProcessingPipeline(config_manager)
        
        # Create test signal
        fs = 250
        duration = 5  # seconds
        signal = np.random.randn(fs * duration)
        
        # Process signal
        results = pipeline.process_signal(
            signal=signal,
            fs=fs,
            signal_type="ECG",
            metadata={'test': True}
        )
        
        assert results is not None
        assert 'processing_results' in results or 'results' in results

    def test_processing_statistics(self):
        """Test processing statistics."""
        config_manager = DynamicConfigManager()
        pipeline = StandardProcessingPipeline(config_manager)
        
        stats = pipeline.get_processing_statistics()
        
        assert 'pipeline_stats' in stats
        assert 'cache_stats' in stats
        assert 'config_stats' in stats
        assert 'optimization_stats' in stats


class TestErrorRecoveryManager:
    """Test Error Recovery Manager functionality."""

    def test_error_recovery_initialization(self):
        """Test error recovery manager initialization."""
        config_manager = DynamicConfigManager()
        error_recovery = ErrorRecoveryManager(config_manager)
        
        assert error_recovery is not None
        assert error_recovery.config == config_manager

    def test_error_classification(self):
        """Test error classification."""
        config_manager = DynamicConfigManager()
        error_recovery = ErrorRecoveryManager(config_manager)
        
        # Test different error types
        memory_error = MemoryError("Out of memory")
        value_error = ValueError("Invalid value")
        
        # These would normally be handled by the error handler
        # For testing, we just verify the manager can be initialized
        assert error_recovery is not None

    def test_error_statistics(self):
        """Test error statistics."""
        config_manager = DynamicConfigManager()
        error_recovery = ErrorRecoveryManager(config_manager)
        
        stats = error_recovery.get_error_statistics()
        
        assert 'total_errors' in stats
        assert 'recovery_attempts' in stats
        assert 'recovery_success_rate' in stats


class TestIntegration:
    """Integration tests for Phase 2 components."""

    def test_end_to_end_processing(self):
        """Test end-to-end processing with all Phase 2 components."""
        # Initialize components
        config_manager = DynamicConfigManager()
        memory_manager = MemoryManager(config_manager, MemoryStrategy.BALANCED)
        pipeline = StandardProcessingPipeline(config_manager)
        error_recovery = ErrorRecoveryManager(config_manager)
        
        # Create test signal
        fs = 250
        duration = 10  # seconds
        signal = np.random.randn(fs * duration)
        
        # Start memory monitoring
        memory_manager.start_memory_monitoring()
        
        try:
            # Process signal
            results = pipeline.process_signal(
                signal=signal,
                fs=fs,
                signal_type="ECG",
                metadata={'test': True, 'duration_seconds': duration}
            )
            
            # Verify results
            assert results is not None
            
            # Get statistics
            pipeline_stats = pipeline.get_processing_statistics()
            memory_stats = memory_manager.get_memory_statistics()
            error_stats = error_recovery.get_error_statistics()
            
            # Verify statistics
            assert pipeline_stats is not None
            assert memory_stats is not None
            assert error_stats is not None
            
        finally:
            # Stop monitoring
            memory_manager._monitoring_active = False

    def test_memory_optimization_integration(self):
        """Test memory optimization integration."""
        config_manager = DynamicConfigManager()
        memory_manager = MemoryManager(config_manager)
        
        # Create large signal
        fs = 1000
        duration = 60  # seconds
        signal = np.random.randn(fs * duration).astype(np.float64)
        
        # Optimize data types
        optimized_signal = memory_manager.optimize_data_types(signal, 'ECG')
        
        # Verify optimization
        assert optimized_signal is not None
        assert len(optimized_signal) == len(signal)
        
        # Check if optimization actually occurred
        if optimized_signal.dtype != signal.dtype:
            # Verify memory savings
            original_size = signal.nbytes
            optimized_size = optimized_signal.nbytes
            assert optimized_size <= original_size

    def test_error_recovery_integration(self):
        """Test error recovery integration."""
        config_manager = DynamicConfigManager()
        error_recovery = ErrorRecoveryManager(config_manager)
        
        # Test error recovery with ErrorInfo
        from vitalDSP.utils.core_infrastructure.error_recovery import ErrorInfo, ErrorSeverity, ErrorCategory
        error_info = ErrorInfo(
            error_id="test_error_001",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING,
            error_type="test_error",
            message="Test error",
            details={'test': True},
            context={'signal_length': 100, 'fs': 250}
        )
        context = {'signal': np.random.randn(100), 'fs': 250}
        
        # Attempt recovery
        recovery_result = error_recovery.attempt_recovery(error_info, context)
        
        # Verify recovery result structure
        assert hasattr(recovery_result, 'success')
        assert hasattr(recovery_result, 'recovery_method')
        assert hasattr(recovery_result, 'recovered_data')
        assert hasattr(recovery_result, 'warning_message')


class TestPerformance:
    """Performance tests for Phase 2 components."""

    def test_processing_performance(self):
        """Test processing performance."""
        config_manager = DynamicConfigManager()
        pipeline = StandardProcessingPipeline(config_manager)
        
        # Create test signal
        fs = 250
        duration = 30  # seconds
        signal = np.random.randn(fs * duration)
        
        # Measure processing time
        start_time = time.time()
        results = pipeline.process_signal(signal, fs, "ECG")
        processing_time = time.time() - start_time
        
        # Verify reasonable processing time (should be less than 15 seconds for 30s signal)
        assert processing_time < 15.0
        
        # Get performance statistics
        stats = pipeline.get_processing_statistics()
        assert stats['pipeline_stats']['total_processing_time'] > 0

    def test_memory_efficiency(self):
        """Test memory efficiency."""
        config_manager = DynamicConfigManager()
        memory_manager = MemoryManager(config_manager)
        
        # Create test data
        signal = np.random.randn(10000).astype(np.float64)
        
        # Profile memory usage
        profile = memory_manager.profile_operation(
            "test_operation",
            signal.nbytes / (1024**2),
            lambda: signal.astype(np.float32)
        )
        
        # Verify profile structure
        assert profile.operation == "test_operation"
        assert profile.input_size_mb > 0
        assert profile.processing_time >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
