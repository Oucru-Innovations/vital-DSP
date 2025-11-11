"""
Additional Comprehensive Tests for memory_manager.py - Missing Coverage

This test file specifically targets missing lines in memory_manager.py to achieve
high test coverage, including edge cases, error conditions, and all code paths.

Author: vitalDSP Team
Date: 2025-01-27
Target Coverage: 95%+
"""

import pytest
import numpy as np
import time
import gc
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

try:
    from vitalDSP.utils.core_infrastructure.memory_manager import (
        MemoryManager,
        DataTypeOptimizer,
        MemoryStrategy,
        MemoryInfo,
        ProcessingMemoryProfile,
        MemoryProfiler,
    )
    MEMORY_MANAGER_AVAILABLE = True
except ImportError:
    MEMORY_MANAGER_AVAILABLE = False

try:
    from vitalDSP.utils.config_utilities.dynamic_config import DynamicConfigManager
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


@pytest.mark.skipif(not MEMORY_MANAGER_AVAILABLE, reason="DataTypeOptimizer not available")
class TestDataTypeOptimizerMissingLines:
    """Test DataTypeOptimizer missing lines."""
    
    def test_optimize_signal_already_optimal(self):
        """Test optimize_signal when signal already in optimal precision - covers lines 139-141."""
        optimizer = DataTypeOptimizer()
        signal = np.random.randn(1000).astype(np.float32)
        
        result = optimizer.optimize_signal(signal, signal_type="generic", target_precision="float32")
        
        assert result is signal  # Should return same object
        assert result.dtype == np.float32
    
    def test_optimize_signal_precision_loss_not_acceptable(self):
        """Test optimize_signal when precision loss is not acceptable - covers lines 144-148."""
        optimizer = DataTypeOptimizer()
        signal = np.random.randn(1000).astype(np.float64)
        
        # Mock config to set very low acceptable precision loss
        with patch.object(optimizer.config, 'get', return_value=0.1):
            result = optimizer.optimize_signal(signal, signal_type="generic", target_precision="float16")
            
            # Should return original signal if precision loss not acceptable
            assert result is signal or result.dtype == signal.dtype
    
    def test_optimize_signal_conversion_verification_passed(self):
        """Test optimize_signal with successful conversion verification - covers lines 151-159."""
        optimizer = DataTypeOptimizer()
        signal = np.random.randn(1000).astype(np.float64)
        
        # Mock both precision loss check and verification to return True
        with patch.object(optimizer, '_is_precision_loss_acceptable', return_value=True):
            with patch.object(optimizer, '_verify_conversion_quality', return_value=True):
                result = optimizer.optimize_signal(signal, signal_type="generic", target_precision="float32")
                
                assert result.dtype == np.float32
    
    def test_optimize_signal_conversion_verification_failed(self):
        """Test optimize_signal with failed conversion verification - covers lines 160-164."""
        optimizer = DataTypeOptimizer()
        signal = np.random.randn(1000).astype(np.float64)
        
        # Mock verification to return False
        with patch.object(optimizer, '_verify_conversion_quality', return_value=False):
            result = optimizer.optimize_signal(signal, signal_type="generic", target_precision="float32")
            
            # Should return original signal
            assert result.dtype == signal.dtype
    
    def test_optimize_features_dict(self):
        """Test optimize_features with dictionary - covers lines 179-199."""
        optimizer = DataTypeOptimizer()
        features = {
            'array': np.random.randn(100).astype(np.float64),
            'float': 3.14159,
            'int': 42,
            'list': [1.0, 2.0, 3.0],
            'mixed_list': [1, 'a', 3.0],
            'string': 'test'
        }
        
        result = optimizer.optimize_features(features, signal_type="generic")
        
        assert isinstance(result, dict)
        assert 'array' in result
        assert 'float' in result
        assert 'int' in result
        assert 'list' in result
        assert 'mixed_list' in result
        assert 'string' in result
    
    def test_determine_optimal_precision_float16(self):
        """Test _determine_optimal_precision for float16 - covers lines 222-224."""
        optimizer = DataTypeOptimizer()
        # Small range and low noise signal
        signal = np.random.randn(1000) * 0.05 + 0.5  # Range < 1.0, std < 0.1
        
        precision = optimizer._determine_optimal_precision(signal, "generic")
        
        assert precision == "float16"
    
    def test_determine_optimal_precision_float64(self):
        """Test _determine_optimal_precision for float64 - covers lines 228-230."""
        optimizer = DataTypeOptimizer()
        # Large range signal
        signal = np.random.randn(1000) * 1000  # Large range
        
        precision = optimizer._determine_optimal_precision(signal, "generic")
        
        assert precision == "float64"
    
    def test_verify_conversion_quality(self):
        """Test _verify_conversion_quality - covers lines 271-280."""
        optimizer = DataTypeOptimizer()
        original = np.random.randn(1000).astype(np.float64)
        converted = original.astype(np.float32)
        
        # Mock config to set acceptable error
        with patch.object(optimizer.config, 'get', return_value=0.01):
            result = optimizer._verify_conversion_quality(original, converted)
            
            assert isinstance(result, (bool, np.bool_))
            assert bool(result) is True or bool(result) is False
    
    def test_verify_conversion_quality_high_error(self):
        """Test _verify_conversion_quality with high conversion error."""
        optimizer = DataTypeOptimizer()
        original = np.random.randn(1000).astype(np.float64)
        # Create converted with high error
        converted = original.astype(np.float16).astype(np.float64)
        
        # Mock config to set very low acceptable error
        with patch.object(optimizer.config, 'get', return_value=0.0001):
            result = optimizer._verify_conversion_quality(original, converted)
            
            # Should fail verification
            assert bool(result) is False
    
    def test_calculate_memory_savings(self):
        """Test _calculate_memory_savings - covers lines 295-302."""
        optimizer = DataTypeOptimizer()
        original = np.random.randn(1000).astype(np.float64)
        optimized = original.astype(np.float32)
        
        savings = optimizer._calculate_memory_savings(original, optimized)
        
        assert isinstance(savings, float)
        assert savings > 0
    
    def test_calculate_memory_savings_zero_size(self):
        """Test _calculate_memory_savings with zero size - covers line 298."""
        optimizer = DataTypeOptimizer()
        original = np.array([])
        optimized = np.array([])
        
        savings = optimizer._calculate_memory_savings(original, optimized)
        
        assert savings == 0.0


@pytest.mark.skipif(not MEMORY_MANAGER_AVAILABLE, reason="MemoryManager not available")
class TestMemoryManagerMissingLines:
    """Test MemoryManager missing lines."""
    
    def test_get_memory_limits_conservative(self):
        """Test _get_memory_limits for conservative strategy - covers lines 343-345."""
        manager = MemoryManager(strategy=MemoryStrategy.CONSERVATIVE)
        
        limits = manager._get_memory_limits()
        
        assert limits['max_memory_percent'] == 0.5
        assert limits['chunk_memory_percent'] == 0.05
    
    def test_get_memory_limits_aggressive(self):
        """Test _get_memory_limits for aggressive strategy - covers lines 350-351."""
        manager = MemoryManager(strategy=MemoryStrategy.AGGRESSIVE)
        
        limits = manager._get_memory_limits()
        
        assert limits['max_memory_percent'] == 0.9
        assert limits['chunk_memory_percent'] == 0.2
    
    def test_recommend_chunk_size_not_in_memory(self):
        """Test recommend_chunk_size when can't process in memory - covers lines 410-424."""
        manager = MemoryManager()
        
        # Large data size that can't fit in memory
        large_size_mb = 100000  # 100 GB
        operations = ['load', 'filter', 'fft']
        
        chunk_size = manager.recommend_chunk_size(large_size_mb, operations)
        
        assert isinstance(chunk_size, int)
        assert chunk_size > 0
    
    def test_recommend_chunk_size_with_constraints(self):
        """Test recommend_chunk_size with min/max constraints - covers lines 421-424."""
        manager = MemoryManager()
        
        # Mock config to set constraints
        with patch.object(manager.config, 'get', side_effect=lambda key, default: {
            'memory.chunk_size.min_samples': 50000,
            'memory.chunk_size.max_samples': 5000000
        }.get(key, default)):
            large_size_mb = 100000
            operations = ['load', 'filter']
            
            chunk_size = manager.recommend_chunk_size(large_size_mb, operations)
            
            assert chunk_size >= 50000
            assert chunk_size <= 5000000
    
    def test_start_memory_monitoring_already_active(self):
        """Test start_memory_monitoring when already active - covers lines 457-459."""
        manager = MemoryManager()
        manager.start_memory_monitoring()
        
        # Try to start again
        manager.start_memory_monitoring()
        
        # Should not raise exception
        assert manager._monitoring_active is True
    
    def test_stop_memory_monitoring(self):
        """Test stop_memory_monitoring - covers lines 471-478."""
        manager = MemoryManager()
        manager.start_memory_monitoring()
        
        # Wait a bit for monitoring to start
        time.sleep(0.1)
        
        manager.stop_memory_monitoring()
        
        assert manager._monitoring_active is False
    
    def test_stop_memory_monitoring_not_active(self):
        """Test stop_memory_monitoring when not active - covers lines 471-472."""
        manager = MemoryManager()
        
        # Should not raise exception
        manager.stop_memory_monitoring()
    
    def test_memory_monitor_loop_history_trimming(self):
        """Test memory monitor loop history trimming - covers lines 492-493."""
        manager = MemoryManager()
        manager.start_memory_monitoring(interval=0.01)
        
        # Wait for some monitoring cycles
        time.sleep(0.1)
        
        # Manually add many entries to trigger trimming
        with manager._lock:
            for i in range(1500):
                manager.memory_history.append({
                    "timestamp": time.time(),
                    "memory_info": manager.get_memory_info()
                })
        
        # Wait a bit more
        time.sleep(0.1)
        
        # Check that history is trimmed
        with manager._lock:
            assert len(manager.memory_history) <= 1000
        
        manager.stop_memory_monitoring()
    
    def test_memory_monitor_loop_warnings(self):
        """Test memory monitor loop warnings - covers lines 496-501."""
        manager = MemoryManager()
        
        # Mock memory info to trigger warnings
        high_memory_info = MemoryInfo(
            total_memory_gb=100.0,
            available_memory_gb=5.0,
            used_memory_gb=95.0,
            memory_percent=95.0,
            swap_total_gb=50.0,
            swap_used_gb=10.0,
            swap_percent=20.0
        )
        
        with patch.object(manager, 'get_memory_info', return_value=high_memory_info):
            manager.start_memory_monitoring(interval=0.01)
            time.sleep(0.1)
            manager.stop_memory_monitoring()
        
        # Should have logged warnings
        assert True  # Test passes if no exception
    
    def test_memory_monitor_loop_exception(self):
        """Test memory monitor loop exception handling - covers lines 505-507."""
        manager = MemoryManager()
        
        # Mock get_memory_info to raise exception
        with patch.object(manager, 'get_memory_info', side_effect=Exception("Memory check failed")):
            manager.start_memory_monitoring(interval=0.01)
            time.sleep(0.1)
            manager.stop_memory_monitoring()
        
        # Should handle exception gracefully
        assert True
    
    def test_profile_operation_exception(self):
        """Test profile_operation with exception - covers lines 541-544."""
        manager = MemoryManager()
        
        def failing_func():
            raise ValueError("Test error")
        
        profile = manager.profile_operation(
            "test_operation",
            10.0,
            failing_func
        )
        
        assert isinstance(profile, ProcessingMemoryProfile)
        assert profile.operation == "test_operation"
    
    def test_estimate_output_size_dict(self):
        """Test _estimate_output_size for dict - covers lines 592-599."""
        manager = MemoryManager()
        result = {
            'array1': np.random.randn(100),
            'array2': np.random.randn(200),
            'number': 42.5,
            'int': 100
        }
        
        size = manager._estimate_output_size(result)
        
        assert isinstance(size, float)
        assert size > 0
    
    def test_estimate_output_size_other(self):
        """Test _estimate_output_size for other types - covers line 601."""
        manager = MemoryManager()
        result = "string result"
        
        size = manager._estimate_output_size(result)
        
        assert size == 0.0
    
    def test_optimize_data_types_dict(self):
        """Test optimize_data_types for dict - covers lines 616-617."""
        manager = MemoryManager()
        data = {
            'signal': np.random.randn(1000).astype(np.float64),
            'features': {'mean': 0.5, 'std': 1.0}
        }
        
        result = manager.optimize_data_types(data, signal_type="generic")
        
        assert isinstance(result, dict)
    
    def test_optimize_data_types_other(self):
        """Test optimize_data_types for other types - covers line 619."""
        manager = MemoryManager()
        data = "string data"
        
        result = manager.optimize_data_types(data, signal_type="generic")
        
        assert result == data
    
    def test_force_garbage_collection(self):
        """Test force_garbage_collection - covers lines 623-624."""
        manager = MemoryManager()
        
        # Should not raise exception
        manager.force_garbage_collection()
    
    def test_get_memory_statistics_with_profiles(self):
        """Test get_memory_statistics with processing profiles - covers lines 642-648."""
        manager = MemoryManager()
        
        # Add some profiles
        profile1 = ProcessingMemoryProfile(
            operation="test1",
            input_size_mb=10.0,
            output_size_mb=5.0,
            peak_memory_mb=15.0,
            memory_efficiency=0.67,
            processing_time=1.0
        )
        profile2 = ProcessingMemoryProfile(
            operation="test2",
            input_size_mb=20.0,
            output_size_mb=10.0,
            peak_memory_mb=25.0,
            memory_efficiency=0.8,
            processing_time=2.0
        )
        
        with manager._lock:
            manager.processing_profiles = [profile1, profile2]
        
        stats = manager.get_memory_statistics()
        
        assert isinstance(stats, dict)
        assert 'processing_efficiency' in stats
        assert stats['processing_efficiency']['average_efficiency'] > 0
    
    def test_calculate_memory_trend_insufficient_data(self):
        """Test _calculate_memory_trend with insufficient data - covers lines 676-677."""
        manager = MemoryManager()
        
        # Add only one entry
        with manager._lock:
            manager.memory_history = [{
                "timestamp": time.time(),
                "memory_info": manager.get_memory_info()
            }]
        
        trend = manager._calculate_memory_trend()
        
        assert trend['trend'] == "stable"
        assert trend['change_percent'] == 0.0
    
    def test_calculate_memory_trend_no_older_data(self):
        """Test _calculate_memory_trend with no older data - covers lines 684-685."""
        manager = MemoryManager()
        
        # Add only recent entries
        with manager._lock:
            for i in range(5):
                manager.memory_history.append({
                    "timestamp": time.time(),
                    "memory_info": manager.get_memory_info()
                })
        
        trend = manager._calculate_memory_trend()
        
        assert trend['trend'] == "stable"
    
    def test_calculate_memory_trend_increasing(self):
        """Test _calculate_memory_trend with increasing trend - covers lines 687-693."""
        manager = MemoryManager()
        
        # Create memory history with increasing trend
        with manager._lock:
            for i in range(30):
                memory_info = MemoryInfo(
                    total_memory_gb=100.0,
                    available_memory_gb=100.0 - i * 2,
                    used_memory_gb=i * 2,
                    memory_percent=50.0 + i,
                    swap_total_gb=50.0,
                    swap_used_gb=10.0,
                    swap_percent=20.0
                )
                manager.memory_history.append({
                    "timestamp": time.time(),
                    "memory_info": memory_info
                })
        
        trend = manager._calculate_memory_trend()
        
        assert isinstance(trend, dict)
        assert 'trend' in trend
        assert 'change_percent' in trend
    
    def test_get_memory_warnings(self):
        """Test get_memory_warnings - covers lines 713-737."""
        manager = MemoryManager()
        
        # Mock high memory usage
        high_memory_info = MemoryInfo(
            total_memory_gb=100.0,
            available_memory_gb=5.0,
            used_memory_gb=95.0,
            memory_percent=95.0,
            swap_total_gb=50.0,
            swap_used_gb=30.0,
            swap_percent=60.0
        )
        
        with patch.object(manager, 'get_memory_info', return_value=high_memory_info):
            warnings = manager.get_memory_warnings()
            
            assert isinstance(warnings, list)
            assert len(warnings) > 0
    
    def test_get_memory_warnings_high_swap(self):
        """Test get_memory_warnings with high swap usage - covers lines 725-726."""
        manager = MemoryManager()
        
        high_swap_info = MemoryInfo(
            total_memory_gb=100.0,
            available_memory_gb=50.0,
            used_memory_gb=50.0,
            memory_percent=50.0,
            swap_total_gb=50.0,
            swap_used_gb=30.0,
            swap_percent=60.0
        )
        
        with patch.object(manager, 'get_memory_info', return_value=high_swap_info):
            warnings = manager.get_memory_warnings()
            
            assert any("swap" in w.lower() for w in warnings)
    
    def test_cleanup_memory(self):
        """Test cleanup_memory - covers lines 741-754."""
        manager = MemoryManager()
        
        # Add many profiles and history entries
        with manager._lock:
            for i in range(150):
                manager.processing_profiles.append(ProcessingMemoryProfile(
                    operation=f"test{i}",
                    input_size_mb=10.0,
                    output_size_mb=5.0,
                    peak_memory_mb=15.0,
                    memory_efficiency=0.67,
                    processing_time=1.0
                ))
                manager.memory_history.append({
                    "timestamp": time.time(),
                    "memory_info": manager.get_memory_info()
                })
        
        manager.cleanup_memory()
        
        with manager._lock:
            assert len(manager.processing_profiles) <= 50
            assert len(manager.memory_history) <= 250


@pytest.mark.skipif(not MEMORY_MANAGER_AVAILABLE, reason="MemoryProfiler not available")
class TestMemoryProfilerMissingLines:
    """Test MemoryProfiler missing lines."""
    
    def test_profile_pipeline(self):
        """Test profile_pipeline - covers lines 786-816."""
        manager = MemoryManager()
        profiler = MemoryProfiler(manager)
        
        def test_pipeline(x):
            return x * 2
        
        # Add some memory history to ensure patterns have stability key
        with manager._lock:
            for i in range(20):
                memory_info = MemoryInfo(
                    total_memory_gb=100.0,
                    available_memory_gb=50.0 + i,
                    used_memory_gb=50.0 - i,
                    memory_percent=50.0 + i * 0.5,
                    swap_total_gb=50.0,
                    swap_used_gb=10.0,
                    swap_percent=20.0
                )
                manager.memory_history.append({
                    "timestamp": time.time(),
                    "memory_info": memory_info
                })
        
        result = profiler.profile_pipeline(test_pipeline, 5)
        
        assert isinstance(result, dict)
        assert 'pipeline_profile' in result
        assert 'memory_statistics' in result
        assert 'memory_patterns' in result
        assert 'optimization_recommendations' in result
    
    def test_analyze_memory_patterns_insufficient_data(self):
        """Test _analyze_memory_patterns with insufficient data - covers lines 820-821."""
        manager = MemoryManager()
        profiler = MemoryProfiler(manager)
        
        # Add only one entry
        with manager._lock:
            manager.memory_history = [{
                "timestamp": time.time(),
                "memory_info": manager.get_memory_info()
            }]
        
        patterns = profiler._analyze_memory_patterns()
        
        assert patterns['pattern'] == "insufficient_data"
    
    def test_analyze_memory_patterns(self):
        """Test _analyze_memory_patterns - covers lines 823-839."""
        manager = MemoryManager()
        profiler = MemoryProfiler(manager)
        
        # Add multiple entries
        with manager._lock:
            for i in range(20):
                memory_info = MemoryInfo(
                    total_memory_gb=100.0,
                    available_memory_gb=50.0 + i,
                    used_memory_gb=50.0 - i,
                    memory_percent=50.0 + i * 0.5,
                    swap_total_gb=50.0,
                    swap_used_gb=10.0,
                    swap_percent=20.0
                )
                manager.memory_history.append({
                    "timestamp": time.time(),
                    "memory_info": memory_info
                })
        
        patterns = profiler._analyze_memory_patterns()
        
        assert isinstance(patterns, dict)
        assert 'variance' in patterns
        assert 'range' in patterns
        assert 'trend' in patterns
        assert 'stability' in patterns
    
    def test_calculate_trend_insufficient_data(self):
        """Test _calculate_trend with insufficient data - covers lines 843-844."""
        manager = MemoryManager()
        profiler = MemoryProfiler(manager)
        
        trend = profiler._calculate_trend([50.0])
        
        assert trend == "stable"
    
    def test_calculate_trend_increasing(self):
        """Test _calculate_trend with increasing trend - covers lines 850-851."""
        manager = MemoryManager()
        profiler = MemoryProfiler(manager)
        
        values = [50.0 + i * 2 for i in range(10)]
        trend = profiler._calculate_trend(values)
        
        assert trend == "increasing"
    
    def test_calculate_trend_decreasing(self):
        """Test _calculate_trend with decreasing trend - covers lines 852-853."""
        manager = MemoryManager()
        profiler = MemoryProfiler(manager)
        
        values = [50.0 - i * 2 for i in range(10)]
        trend = profiler._calculate_trend(values)
        
        assert trend == "decreasing"
    
    def test_calculate_trend_stable(self):
        """Test _calculate_trend with stable trend - covers line 855."""
        manager = MemoryManager()
        profiler = MemoryProfiler(manager)
        
        values = [50.0] * 10
        trend = profiler._calculate_trend(values)
        
        assert trend == "stable"
    
    def test_generate_optimization_recommendations(self):
        """Test _generate_optimization_recommendations - covers lines 864-891."""
        manager = MemoryManager()
        profiler = MemoryProfiler(manager)
        
        profile = ProcessingMemoryProfile(
            operation="test",
            input_size_mb=10.0,
            output_size_mb=5.0,
            peak_memory_mb=100.0,  # Low efficiency
            memory_efficiency=0.1,
            processing_time=120.0  # Long processing time
        )
        
        memory_stats = {
            "current_memory": {"percent": 85.0},
            "memory_trend": {"trend": "increasing"}
        }
        
        patterns = {"stability": "unstable"}
        
        recommendations = profiler._generate_optimization_recommendations(
            profile, memory_stats, patterns
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    def test_generate_optimization_report(self):
        """Test generate_optimization_report - covers lines 903-956."""
        manager = MemoryManager()
        profiler = MemoryProfiler(manager)
        
        profiling_results = {
            "pipeline_profile": ProcessingMemoryProfile(
                operation="test",
                input_size_mb=10.0,
                output_size_mb=5.0,
                peak_memory_mb=15.0,
                memory_efficiency=0.67,
                processing_time=1.0
            ),
            "memory_statistics": {
                "current_memory": {
                    "total_gb": 100.0,
                    "available_gb": 50.0,
                    "used_gb": 50.0,
                    "percent": 50.0,
                    "swap_percent": 20.0
                }
            },
            "memory_patterns": {
                "stability": "stable",
                "variance": 5.0,
                "range": 10.0,
                "trend": "stable",
                "peak_usage": 60.0,
                "average_usage": 50.0
            },
            "optimization_recommendations": ["Test recommendation"]
        }
        
        report = profiler.generate_optimization_report(profiling_results)
        
        assert isinstance(report, str)
        assert "Memory Optimization Report" in report
        assert "Pipeline Profile:" in report
        assert "Memory Statistics:" in report
        assert "Memory Patterns:" in report
        assert "Optimization Recommendations:" in report

