"""
Comprehensive Benchmark Suite for vitalDSP Phase 1-3 Large Data Processing

This benchmark suite tests all major components of the Phase 1-3 implementation:
- Data loading (standard, chunked, memory-mapped, OUCRU)
- Processing pipeline (8-stage pipeline with multi-path processing)
- Quality screening (3-stage screening with parallel processing)
- Integration performance (end-to-end workflows)

Author: vitalDSP Team
Date: October 17, 2025
Phase: 4 (Optimization & Testing)
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import json
import time
import psutil
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime


@pytest.fixture
def benchmark():
    """Mock benchmark fixture for when pytest-benchmark is not available."""
    class MockBenchmark:
        def __call__(self, func, *args, **kwargs):
            # Just run the function and return the result
            return func(*args, **kwargs)
        
        def timer(self, func, *args, **kwargs):
            # Simple timer that just runs the function
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            # Store timing info in the result if it's a BenchmarkMetrics object
            if hasattr(result, 'execution_time'):
                result.execution_time = end_time - start_time
            return result
    
    return MockBenchmark()

from vitalDSP.utils.data_processing.data_loader import DataLoader, DataFormat
from vitalDSP.utils.core_infrastructure.processing_pipeline import (
    StandardProcessingPipeline,
    OptimizedStandardProcessingPipeline,
)
from vitalDSP.utils.core_infrastructure.quality_screener import (
    SignalQualityScreener,
    QualityScreeningConfig,
)


class BenchmarkMetrics:
    """Container for benchmark metrics."""

    def __init__(self):
        self.execution_time: float = 0.0
        self.peak_memory_mb: float = 0.0
        self.cpu_percent: float = 0.0
        self.throughput_samples_per_sec: float = 0.0
        self.memory_efficiency: float = 0.0  # output_size / peak_memory

    def to_dict(self) -> Dict[str, Any]:
        return {
            'execution_time_sec': self.execution_time,
            'peak_memory_mb': self.peak_memory_mb,
            'cpu_percent': self.cpu_percent,
            'throughput_samples_per_sec': self.throughput_samples_per_sec,
            'memory_efficiency': self.memory_efficiency,
        }


class BenchmarkRunner:
    """Utility class for running benchmarks with consistent metrics."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def run_benchmark(
        self, func, *args, n_samples: int = None, **kwargs
    ) -> BenchmarkMetrics:
        """
        Run a benchmark function and collect metrics.

        Args:
            func: Function to benchmark
            *args: Positional arguments to func
            n_samples: Number of samples processed (for throughput calculation)
            **kwargs: Keyword arguments to func

        Returns:
            BenchmarkMetrics with collected performance data
        """
        metrics = BenchmarkMetrics()

        # Get initial memory
        initial_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        peak_memory = initial_memory

        # Run benchmark
        start_time = time.time()
        start_cpu = self.process.cpu_percent()

        result = func(*args, **kwargs)

        end_time = time.time()
        end_cpu = self.process.cpu_percent()

        # Calculate metrics
        metrics.execution_time = end_time - start_time
        final_memory = self.process.memory_info().rss / (1024 * 1024)
        metrics.peak_memory_mb = max(peak_memory, final_memory) - initial_memory
        metrics.cpu_percent = (start_cpu + end_cpu) / 2

        if n_samples:
            metrics.throughput_samples_per_sec = n_samples / metrics.execution_time

        # Calculate memory efficiency if result size available
        if hasattr(result, 'nbytes'):
            output_size_mb = result.nbytes / (1024 * 1024)
            if metrics.peak_memory_mb > 0:
                metrics.memory_efficiency = output_size_mb / metrics.peak_memory_mb

        return metrics, result


# ============================================================================
# BENCHMARK 1: Data Loading Performance
# ============================================================================


class TestDataLoadingBenchmarks:
    """Benchmark different data loading strategies."""

    @pytest.fixture
    def test_files(self, tmp_path):
        """Create test files of various sizes."""
        files = {}

        # Small file: 10MB (1 minute @ 250Hz)
        files['small'] = self._create_csv_file(
            tmp_path / 'small.csv', duration_sec=60, fs=250
        )

        # Medium file: 50MB (5 minutes @ 250Hz)
        files['medium'] = self._create_csv_file(
            tmp_path / 'medium.csv', duration_sec=300, fs=250
        )

        # Large file: 100MB (10 minutes @ 250Hz)
        files['large'] = self._create_csv_file(
            tmp_path / 'large.csv', duration_sec=600, fs=250
        )

        # OUCRU small: 10MB
        files['oucru_small'] = self._create_oucru_file(
            tmp_path / 'oucru_small.csv', duration_sec=60, fs=250
        )

        # OUCRU medium: 50MB
        files['oucru_medium'] = self._create_oucru_file(
            tmp_path / 'oucru_medium.csv', duration_sec=300, fs=250
        )

        return files

    def _create_csv_file(self, path: Path, duration_sec: int, fs: int) -> Path:
        """Create standard CSV test file."""
        n_samples = duration_sec * fs
        timestamps = pd.date_range(
            start='2024-01-01', periods=n_samples, freq=f'{1000000//fs}us'
        )
        signal = np.random.randn(n_samples)

        df = pd.DataFrame({'timestamp': timestamps, 'signal': signal})
        df.to_csv(path, index=False)
        return path

    def _create_oucru_file(self, path: Path, duration_sec: int, fs: int) -> Path:
        """Create OUCRU CSV test file."""
        test_data = []
        for i in range(duration_sec):
            signal = list(np.random.randn(fs))
            test_data.append({
                'timestamp': f'2024-01-01 00:00:{i:02d}',
                'signal': json.dumps(signal),
                'sampling_rate': fs
            })

        df = pd.DataFrame(test_data)
        df.to_csv(path, index=False)
        return path

    def test_benchmark_standard_csv_loading(self, test_files, benchmark):
        """Benchmark standard CSV loading."""
        runner = BenchmarkRunner()

        def load_csv():
            loader = DataLoader(test_files['medium'], format=DataFormat.CSV)
            return loader.load()

        metrics, data = runner.run_benchmark(load_csv, n_samples=300 * 250)

        print(f"\n=== Standard CSV Loading (50MB) ===")
        print(f"Time: {metrics.execution_time:.3f}s")
        print(f"Memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"Throughput: {metrics.throughput_samples_per_sec:.0f} samples/sec")

    def test_benchmark_oucru_csv_loading_small(self, test_files, benchmark):
        """Benchmark OUCRU CSV loading (small file, standard loader)."""
        runner = BenchmarkRunner()

        def load_oucru():
            loader = DataLoader(test_files['oucru_small'], format=DataFormat.OUCRU_CSV)
            return loader.load()

        metrics, data = runner.run_benchmark(load_oucru, n_samples=60 * 250)

        print(f"\n=== OUCRU CSV Loading - Small (10MB, Standard Loader) ===")
        print(f"Time: {metrics.execution_time:.3f}s")
        print(f"Memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"Throughput: {metrics.throughput_samples_per_sec:.0f} samples/sec")

    def test_benchmark_oucru_csv_loading_medium(self, test_files, benchmark):
        """Benchmark OUCRU CSV loading (medium file, streaming loader)."""
        runner = BenchmarkRunner()

        def load_oucru():
            loader = DataLoader(test_files['oucru_medium'], format=DataFormat.OUCRU_CSV)
            # Force streaming with chunk_size
            return loader.load(chunk_size=5000)

        metrics, data = runner.run_benchmark(load_oucru, n_samples=300 * 250)

        print(f"\n=== OUCRU CSV Loading - Medium (50MB, Streaming Loader) ===")
        print(f"Time: {metrics.execution_time:.3f}s")
        print(f"Memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"Throughput: {metrics.throughput_samples_per_sec:.0f} samples/sec")


# ============================================================================
# BENCHMARK 2: Processing Pipeline Performance
# ============================================================================


class TestProcessingPipelineBenchmarks:
    """Benchmark processing pipeline performance."""

    @pytest.fixture
    def test_signals(self):
        """Create test signals of various lengths."""
        fs = 250
        signals = {
            'short': np.random.randn(1 * 60 * fs),  # 1 minute
            'medium': np.random.randn(5 * 60 * fs),  # 5 minutes
            'long': np.random.randn(15 * 60 * fs),  # 15 minutes
        }
        return signals, fs

    def test_benchmark_standard_pipeline_short(self, test_signals, benchmark):
        """Benchmark standard pipeline on short signal."""
        signals, fs = test_signals
        runner = BenchmarkRunner()

        def run_pipeline():
            pipeline = StandardProcessingPipeline()
            return pipeline.process_signal(
                signal=signals['short'],
                fs=fs,
                signal_type='ppg'
            )

        metrics, result = runner.run_benchmark(
            run_pipeline, n_samples=len(signals['short'])
        )

        print(f"\n=== Standard Pipeline - Short (1 min) ===")
        print(f"Time: {metrics.execution_time:.3f}s")
        print(f"Memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"Throughput: {metrics.throughput_samples_per_sec:.0f} samples/sec")
        print(f"Success: {result.get('success', 'Unknown')}")

    def test_benchmark_standard_pipeline_medium(self, test_signals, benchmark):
        """Benchmark standard pipeline on medium signal."""
        signals, fs = test_signals
        runner = BenchmarkRunner()

        def run_pipeline():
            pipeline = StandardProcessingPipeline()
            return pipeline.process_signal(
                signal=signals['medium'],
                fs=fs,
                signal_type='ppg'
            )

        metrics, result = runner.run_benchmark(
            run_pipeline, n_samples=len(signals['medium'])
        )

        print(f"\n=== Standard Pipeline - Medium (5 min) ===")
        print(f"Time: {metrics.execution_time:.3f}s")
        print(f"Memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"Throughput: {metrics.throughput_samples_per_sec:.0f} samples/sec")

    def test_benchmark_optimized_pipeline_medium(self, test_signals, benchmark):
        """Benchmark optimized pipeline on medium signal."""
        signals, fs = test_signals
        runner = BenchmarkRunner()

        def run_pipeline():
            pipeline = OptimizedStandardProcessingPipeline()
            return pipeline.process_signal(
                signal=signals['medium'],
                fs=fs,
                signal_type='ppg'
            )

        metrics, result = runner.run_benchmark(
            run_pipeline, n_samples=len(signals['medium'])
        )

        print(f"\n=== Optimized Pipeline - Medium (5 min) ===")
        print(f"Time: {metrics.execution_time:.3f}s")
        print(f"Memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"Throughput: {metrics.throughput_samples_per_sec:.0f} samples/sec")

    def test_benchmark_pipeline_stage3_multipath(self, test_signals, benchmark):
        """Benchmark Stage 3 multi-path processing specifically."""
        signals, fs = test_signals
        runner = BenchmarkRunner()

        def run_stage3():
            pipeline = StandardProcessingPipeline()
            context = {
                'signal': signals['medium'],
                'fs': fs,
                'signal_type': 'ppg',
                'original_signal': signals['medium'].copy(),
            }
            return pipeline._stage_parallel_processing(context)

        metrics, result = runner.run_benchmark(
            run_stage3, n_samples=len(signals['medium'])
        )

        print(f"\n=== Pipeline Stage 3 (Multi-Path Processing) ===")
        print(f"Time: {metrics.execution_time:.3f}s")
        print(f"Memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"Paths processed: 4 (raw, filtered, preprocessed, full)")


# ============================================================================
# BENCHMARK 3: Quality Screening Performance
# ============================================================================


class TestQualityScreeningBenchmarks:
    """Benchmark quality screening performance."""

    @pytest.fixture
    def test_signals(self):
        """Create test signals for quality screening."""
        fs = 250
        # Create signals with varying quality
        signals = {
            'good': np.sin(2 * np.pi * 1.2 * np.arange(5 * 60 * fs) / fs) + np.random.randn(5 * 60 * fs) * 0.1,
            'noisy': np.sin(2 * np.pi * 1.2 * np.arange(5 * 60 * fs) / fs) + np.random.randn(5 * 60 * fs) * 2.0,
            'mixed': np.concatenate([
                np.sin(2 * np.pi * 1.2 * np.arange(2.5 * 60 * fs) / fs) + np.random.randn(int(2.5 * 60 * fs)) * 0.1,
                np.random.randn(int(2.5 * 60 * fs)) * 2.0
            ])
        }
        return signals, fs

    def test_benchmark_quality_screening_sequential(self, test_signals, benchmark):
        """Benchmark quality screening in sequential mode."""
        signals, fs = test_signals
        runner = BenchmarkRunner()

        def run_screening():
            config = QualityScreeningConfig(
                segment_duration=10.0,
                overlap=0.5,
                max_workers=1  # Sequential
            )
            screener = SignalQualityScreener(config)
            return screener.screen_signal(signals['mixed'])

        metrics, result = runner.run_benchmark(
            run_screening, n_samples=len(signals['mixed'])
        )

        print(f"\n=== Quality Screening - Sequential ===")
        print(f"Time: {metrics.execution_time:.3f}s")
        print(f"Memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"Segments: {len(result)}")
        # Calculate pass rate from results
        passed_count = sum(1 for r in result if r.passed_screening)
        pass_rate = passed_count / len(result) if result else 0.0
        print(f"Pass rate: {pass_rate:.1%}")

    def test_benchmark_quality_screening_parallel(self, test_signals, benchmark):
        """Benchmark quality screening in parallel mode."""
        signals, fs = test_signals
        runner = BenchmarkRunner()

        def run_screening():
            config = QualityScreeningConfig(
                segment_duration=10.0,
                overlap=0.5,
                max_workers=4  # Parallel
            )
            screener = SignalQualityScreener(config)
            return screener.screen_signal(signals['mixed'])

        metrics, result = runner.run_benchmark(
            run_screening, n_samples=len(signals['mixed'])
        )

        print(f"\n=== Quality Screening - Parallel (4 workers) ===")
        print(f"Time: {metrics.execution_time:.3f}s")
        print(f"Memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"Segments: {len(result)}")
        # Calculate pass rate from results
        passed_count = sum(1 for r in result if r.passed_screening)
        pass_rate = passed_count / len(result) if result else 0.0
        print(f"Pass rate: {pass_rate:.1%}")


# ============================================================================
# BENCHMARK 4: End-to-End Integration
# ============================================================================


class TestEndToEndBenchmarks:
    """Benchmark complete end-to-end workflows."""

    def test_benchmark_complete_workflow_small(self, tmp_path, benchmark):
        """Benchmark complete workflow on small file."""
        runner = BenchmarkRunner()

        # Create test file
        duration_sec = 60
        fs = 250
        n_samples = duration_sec * fs
        test_data = []
        for i in range(duration_sec):
            signal = list(np.random.randn(fs))
            test_data.append({
                'timestamp': f'2024-01-01 00:00:{i:02d}',
                'signal': json.dumps(signal),
                'sampling_rate': fs
            })

        test_file = tmp_path / 'test.csv'
        df = pd.DataFrame(test_data)
        df.to_csv(test_file, index=False)

        def run_complete_workflow():
            # Load
            loader = DataLoader(test_file, format=DataFormat.OUCRU_CSV)
            data = loader.load()
            signal = data['signal'].values

            # Screen
            config = QualityScreeningConfig(segment_duration=10.0, max_workers=2)
            screener = SignalQualityScreener(config)
            screening_result = screener.screen_signal(signal)

            # Process
            pipeline = StandardProcessingPipeline()
            processing_result = pipeline.process_signal(signal, fs, 'ppg')

            return data, screening_result, processing_result

        metrics, results = runner.run_benchmark(run_complete_workflow, n_samples=n_samples)

        print(f"\n=== Complete Workflow - Small (1 min) ===")
        print(f"Time: {metrics.execution_time:.3f}s")
        print(f"Memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"Throughput: {metrics.throughput_samples_per_sec:.0f} samples/sec")
        print(f"Quality pass rate: {len(results[1])} segments processed")
        print(f"Processing success: {results[2].get('success', 'Unknown')}")

    def test_benchmark_complete_workflow_medium(self, tmp_path, benchmark):
        """Benchmark complete workflow on medium file."""
        runner = BenchmarkRunner()

        # Create test file
        duration_sec = 300
        fs = 250
        n_samples = duration_sec * fs
        test_data = []
        for i in range(duration_sec):
            signal = list(np.random.randn(fs))
            test_data.append({
                'timestamp': f'2024-01-01 00:{i//60:02d}:{i%60:02d}',
                'signal': json.dumps(signal),
                'sampling_rate': fs
            })

        test_file = tmp_path / 'test.csv'
        df = pd.DataFrame(test_data)
        df.to_csv(test_file, index=False)

        def run_complete_workflow():
            # Load with streaming
            loader = DataLoader(test_file, format=DataFormat.OUCRU_CSV)
            data = loader.load(chunk_size=5000)
            signal = data['signal'].values

            # Screen with parallel
            config = QualityScreeningConfig(segment_duration=10.0, max_workers=4)
            screener = SignalQualityScreener(config)
            screening_result = screener.screen_signal(signal)

            # Process with optimized pipeline
            pipeline = OptimizedStandardProcessingPipeline()
            processing_result = pipeline.process_signal(signal, fs, 'ppg')

            return data, screening_result, processing_result

        metrics, results = runner.run_benchmark(run_complete_workflow, n_samples=n_samples)

        print(f"\n=== Complete Workflow - Medium (5 min) ===")
        print(f"Time: {metrics.execution_time:.3f}s")
        print(f"Memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"Throughput: {metrics.throughput_samples_per_sec:.0f} samples/sec")


# ============================================================================
# BENCHMARK REPORT GENERATION
# ============================================================================


class TestBenchmarkReport:
    """Generate comprehensive benchmark report."""

    def test_generate_benchmark_report(self, tmp_path):
        """Generate comprehensive benchmark report."""
        report = []
        report.append("# vitalDSP Phase 1-3 Benchmark Report")
        report.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**System:** {psutil.cpu_count()} cores, {psutil.virtual_memory().total / (1024**3):.1f}GB RAM")
        report.append("\n---\n")

        report.append("## Executive Summary")
        report.append("\nThis report contains comprehensive benchmarks of all Phase 1-3 components:")
        report.append("- Data loading (CSV, OUCRU standard, OUCRU streaming)")
        report.append("- Processing pipeline (8-stage, multi-path)")
        report.append("- Quality screening (sequential, parallel)")
        report.append("- End-to-end integration workflows")
        report.append("\n---\n")

        # Save report
        report_path = tmp_path / 'benchmark_report.md'
        report_path.write_text('\n'.join(report))

        print(f"\n=== Benchmark report generated: {report_path} ===")


if __name__ == '__main__':
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--benchmark-only',
        '--benchmark-autosave'
    ])
