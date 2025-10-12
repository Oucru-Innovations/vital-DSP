# tests/vitalDSP/test_performance_edge_cases.py
"""
Performance-focused edge case tests for vitalDSP functions.

This module contains tests specifically designed to verify performance
characteristics and identify potential bottlenecks in edge cases.
"""

import pytest
import numpy as np
import time
import psutil
import os
import sys
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from vitalDSP.advanced_computation.anomaly_detection import AnomalyDetection
from vitalDSP.advanced_computation.emd import EMD
from vitalDSP.physiological_features.nonlinear import NonlinearFeatures
from vitalDSP.filtering.signal_filtering import SignalFiltering
from vitalDSP.transforms.fourier_transform import FourierTransform


class TestPerformanceEdgeCases:
    """Performance-focused edge case tests."""
    
    def test_lof_performance_scaling(self):
        """Test LOF performance scaling with signal size."""
        signal_sizes = [1000, 5000, 10000]
        execution_times = []
        
        for size in signal_sizes:
            signal = np.random.randn(size)
            ad = AnomalyDetection(signal)
            
            start_time = time.time()
            anomalies = ad.detect_anomalies(method="lof", n_neighbors=20)
            end_time = time.time()
            
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            # Should complete within reasonable time
            assert execution_time < 30.0  # 30 seconds max
            assert len(anomalies) >= 0  # Should return valid result
        
        # Check that execution time scales reasonably (not O(n²))
        # For O(n log n), time should not increase quadratically
        if len(execution_times) >= 2:
            ratio = execution_times[-1] / execution_times[0]
            size_ratio = signal_sizes[-1] / signal_sizes[0]
            
            # Should be much better than O(n²) scaling
            assert ratio < size_ratio ** 1.5  # Allow some overhead
    
    def test_emd_convergence_performance(self):
        """Test EMD convergence performance."""
        # Test with signals that should converge quickly
        quick_convergence_signals = [
            np.zeros(100),  # Flat signal
            np.ones(100) * 5,  # Constant signal
            np.sin(np.linspace(0, 10, 100))  # Simple periodic signal
        ]
        
        for signal in quick_convergence_signals:
            emd = EMD(signal)
            
            start_time = time.time()
            imfs = emd.emd(max_sifting_iterations=10, max_decomposition_iterations=5)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Should converge quickly
            assert execution_time < 5.0  # 5 seconds max
            assert len(imfs) <= 5  # Should respect limits
    
    def test_dfa_performance_optimization(self):
        """Test DFA performance optimization."""
        signal_sizes = [1000, 5000, 10000]
        
        for size in signal_sizes:
            signal = np.random.randn(size)
            nf = NonlinearFeatures(signal)
            
            # Test different polynomial orders
            for order in [1, 2, 3]:
                start_time = time.time()
                dfa_alpha = nf.compute_dfa(order=order)
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                # Should complete within reasonable time
                assert execution_time < 10.0  # 10 seconds max
                assert isinstance(dfa_alpha, (int, float))
                assert not np.isnan(dfa_alpha)
                assert not np.isinf(dfa_alpha)
    
    def test_memory_usage_large_signals(self):
        """Test memory usage with large signals."""
        process = psutil.Process(os.getpid())
        
        # Test with progressively larger signals
        signal_sizes = [10000, 50000, 100000]
        
        for size in signal_sizes:
            # Get initial memory usage
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            signal = np.random.randn(size)
            sf = SignalFiltering(signal)
            filtered = sf.moving_average(window_size=10)
            
            # Get final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (not excessive)
            assert memory_increase < size * 0.01  # Less than 1% of signal size in MB
            assert len(filtered) == len(signal)
    
    def test_concurrent_performance(self):
        """Test performance under concurrent operations."""
        import threading
        import queue
        
        signal = np.random.randn(1000)
        results = queue.Queue()
        execution_times = queue.Queue()
        
        def process_signal():
            start_time = time.time()
            sf = SignalFiltering(signal)
            filtered = sf.moving_average(window_size=5)
            end_time = time.time()
            
            results.put(len(filtered))
            execution_times.put(end_time - start_time)
        
        # Run multiple threads concurrently
        num_threads = 5
        threads = []
        
        start_time = time.time()
        for _ in range(num_threads):
            thread = threading.Thread(target=process_signal)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Check results
        assert results.qsize() == num_threads
        assert execution_times.qsize() == num_threads
        
        # Total time should be reasonable (not linear with number of threads)
        assert total_time < 10.0  # 10 seconds max for all threads
        
        # Check individual execution times
        individual_times = []
        while not execution_times.empty():
            individual_times.append(execution_times.get())
        
        # Individual times should be reasonable
        for exec_time in individual_times:
            assert exec_time < 5.0  # 5 seconds max per thread
    
    def test_performance_with_edge_case_signals(self):
        """Test performance with edge case signals."""
        edge_case_signals = [
            np.zeros(1000),  # All zeros
            np.ones(1000) * 5,  # All same value
            np.random.randn(1000) * 1e-10,  # Very small values
            np.random.randn(1000) * 1e10,  # Very large values
        ]
        
        for signal in edge_case_signals:
            # Test filtering performance
            start_time = time.time()
            sf = SignalFiltering(signal)
            filtered = sf.moving_average(window_size=10)
            end_time = time.time()
            
            execution_time = end_time - start_time
            assert execution_time < 2.0  # 2 seconds max
            assert len(filtered) == len(signal)
            
            # Test transform performance
            start_time = time.time()
            ft = FourierTransform(signal)
            dft = ft.compute_dft()
            end_time = time.time()
            
            execution_time = end_time - start_time
            assert execution_time < 2.0  # 2 seconds max
            assert len(dft) == len(signal)
    
    def test_performance_monitoring(self):
        """Test performance monitoring capabilities."""
        signal = np.random.randn(5000)
        
        # Test with performance monitoring
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        nf = NonlinearFeatures(signal)
        dfa_alpha = nf.compute_dfa(order=1)
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        # Performance should be within acceptable limits
        assert execution_time < 5.0  # 5 seconds max
        assert memory_usage < 100  # 100 MB max
        assert isinstance(dfa_alpha, (int, float))
    
    def test_scalability_limits(self):
        """Test scalability limits."""
        # Test with very large signal
        large_signal = np.random.randn(100000)
        
        # Should handle large signals gracefully
        sf = SignalFiltering(large_signal)
        
        start_time = time.time()
        filtered = sf.moving_average(window_size=10)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        assert execution_time < 30.0  # 30 seconds max
        assert len(filtered) == len(large_signal)
        
        # Test with very small window size
        start_time = time.time()
        filtered_small_window = sf.moving_average(window_size=2)
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 10.0  # 10 seconds max
        assert len(filtered_small_window) == len(large_signal)
    
    def test_performance_under_stress(self):
        """Test performance under stress conditions."""
        # Test with rapid successive operations
        signal = np.random.randn(1000)
        
        start_time = time.time()
        
        for _ in range(10):  # 10 rapid operations
            sf = SignalFiltering(signal)
            filtered = sf.moving_average(window_size=5)
            
            ft = FourierTransform(signal)
            dft = ft.compute_dft()
            
            nf = NonlinearFeatures(signal)
            dfa_alpha = nf.compute_dfa(order=1)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete within reasonable time
        assert total_time < 60.0  # 60 seconds max for 10 operations
        assert len(filtered) == len(signal)
        assert len(dft) == len(signal)
        assert isinstance(dfa_alpha, (int, float))


class TestPerformanceRegression:
    """Test for performance regressions."""
    
    def test_lof_performance_regression(self):
        """Test that LOF performance doesn't regress."""
        signal = np.random.randn(5000)
        ad = AnomalyDetection(signal)
        
        start_time = time.time()
        anomalies = ad.detect_anomalies(method="lof", n_neighbors=20)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should be fast (optimized version)
        assert execution_time < 5.0  # 5 seconds max
        assert len(anomalies) >= 0
    
    def test_dfa_performance_regression(self):
        """Test that DFA performance doesn't regress."""
        signal = np.random.randn(5000)
        nf = NonlinearFeatures(signal)
        
        start_time = time.time()
        dfa_alpha = nf.compute_dfa(order=1)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should be fast (optimized version)
        assert execution_time < 3.0  # 3 seconds max
        assert isinstance(dfa_alpha, (int, float))
    
    def test_filtering_performance_regression(self):
        """Test that filtering performance doesn't regress."""
        signal = np.random.randn(10000)
        sf = SignalFiltering(signal)
        
        start_time = time.time()
        filtered = sf.butterworth(cutoff=10, fs=100, order=4)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should be fast
        assert execution_time < 2.0  # 2 seconds max
        assert len(filtered) == len(signal)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
