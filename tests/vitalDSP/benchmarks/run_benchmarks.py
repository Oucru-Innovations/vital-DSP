"""
Standalone Benchmark Runner for vitalDSP Phase 1-3

Run comprehensive benchmarks without requiring pytest or special arguments.
This script can be executed directly with Python.

Usage:
    python run_benchmarks.py
    python run_benchmarks.py --quick      # Run quick benchmarks only
    python run_benchmarks.py --full       # Run all benchmarks including slow ones

Author: vitalDSP Team
Date: October 17, 2025
Phase: 4 (Optimization & Testing)
"""

import sys
import argparse
import numpy as np
import pandas as pd
import tempfile
import json
import time
import psutil
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

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
        self.throughput_samples_per_sec: float = 0.0

    def __str__(self):
        return (
            f"Time: {self.execution_time:.3f}s, "
            f"Memory: {self.peak_memory_mb:.1f}MB, "
            f"Throughput: {self.throughput_samples_per_sec:,.0f} samples/sec"
        )


class BenchmarkRunner:
    """Utility class for running benchmarks."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.results = []

    def run_benchmark(self, func, name: str, n_samples: int = None) -> BenchmarkMetrics:
        """Run a benchmark and collect metrics."""
        metrics = BenchmarkMetrics()

        # Get initial memory
        initial_memory = self.process.memory_info().rss / (1024 * 1024)

        # Run benchmark
        print(f"\n  Running: {name}...", end=" ", flush=True)
        start_time = time.time()

        try:
            result = func()
            metrics.execution_time = time.time() - start_time

            # Calculate metrics
            final_memory = self.process.memory_info().rss / (1024 * 1024)
            metrics.peak_memory_mb = max(final_memory - initial_memory, 0)

            if n_samples:
                metrics.throughput_samples_per_sec = n_samples / metrics.execution_time

            print(f"OK {metrics.execution_time:.3f}s")
            return metrics, result

        except Exception as e:
            print(f"FAILED: {e}")
            return None, None


def create_test_oucru_file(filepath: Path, duration_sec: int, fs: int) -> Path:
    """Create OUCRU CSV test file."""
    test_data = []
    for i in range(duration_sec):
        signal = list(np.random.randn(fs))
        test_data.append({
            'timestamp': f'2024-01-01 00:{i//60:02d}:{i%60:02d}',
            'signal': json.dumps(signal),
            'sampling_rate': fs
        })

    df = pd.DataFrame(test_data)
    df.to_csv(filepath, index=False)
    return filepath


def benchmark_data_loading(runner: BenchmarkRunner, test_dir: Path):
    """Benchmark data loading."""
    print("\n" + "="*70)
    print("BENCHMARK 1: Data Loading")
    print("="*70)

    # Create test files
    small_file = test_dir / 'small.csv'
    medium_file = test_dir / 'medium.csv'

    create_test_oucru_file(small_file, duration_sec=60, fs=250)  # ~5MB
    create_test_oucru_file(medium_file, duration_sec=180, fs=250)  # ~15MB

    # Test 1: Small file
    def load_small():
        loader = DataLoader(small_file, format=DataFormat.OUCRU_CSV)
        return loader.load(time_column='timestamp')

    metrics, _ = runner.run_benchmark(load_small, "Small file (1 min @ 250Hz)", n_samples=60*250)

    # Test 2: Medium file
    def load_medium():
        loader = DataLoader(medium_file, format=DataFormat.OUCRU_CSV)
        return loader.load(time_column='timestamp')

    metrics, _ = runner.run_benchmark(load_medium, "Medium file (3 min @ 250Hz)", n_samples=180*250)

    # Test 3: Medium file with streaming
    def load_streaming():
        loader = DataLoader(medium_file, format=DataFormat.OUCRU_CSV)
        return loader.load(time_column='timestamp', chunk_size=2000)

    metrics, _ = runner.run_benchmark(load_streaming, "Medium file (streaming)", n_samples=180*250)


def benchmark_processing(runner: BenchmarkRunner):
    """Benchmark processing pipeline."""
    print("\n" + "="*70)
    print("BENCHMARK 2: Processing Pipeline")
    print("="*70)

    fs = 250
    signals = {
        'short': np.random.randn(1 * 60 * fs),  # 1 minute
        'medium': np.random.randn(3 * 60 * fs),  # 3 minutes
    }

    # Test 1: Standard pipeline - short
    def process_standard_short():
        pipeline = StandardProcessingPipeline()
        return pipeline.process_signal(signals['short'], fs, 'ppg')

    metrics, _ = runner.run_benchmark(
        process_standard_short,
        "Standard Pipeline (1 min)",
        n_samples=len(signals['short'])
    )

    # Test 2: Standard pipeline - medium
    def process_standard_medium():
        pipeline = StandardProcessingPipeline()
        return pipeline.process_signal(signals['medium'], fs, 'ppg')

    metrics, _ = runner.run_benchmark(
        process_standard_medium,
        "Standard Pipeline (3 min)",
        n_samples=len(signals['medium'])
    )

    # Test 3: Optimized pipeline - medium
    def process_optimized_medium():
        pipeline = OptimizedStandardProcessingPipeline()
        return pipeline.process_signal(signals['medium'], fs, 'ppg')

    metrics, _ = runner.run_benchmark(
        process_optimized_medium,
        "Optimized Pipeline (3 min)",
        n_samples=len(signals['medium'])
    )


def benchmark_quality_screening(runner: BenchmarkRunner):
    """Benchmark quality screening."""
    print("\n" + "="*70)
    print("BENCHMARK 3: Quality Screening")
    print("="*70)

    fs = 250
    signal = np.random.randn(3 * 60 * fs)  # 3 minutes

    # Test 1: Sequential screening
    def screen_sequential():
        config = QualityScreeningConfig(
            segment_duration=10.0,
            max_workers=1
        )
        screener = SignalQualityScreener(config)
        screener.sampling_rate = fs
        screener.signal_type = 'ppg'
        return screener.screen_signal(signal)

    metrics, _ = runner.run_benchmark(
        screen_sequential,
        "Sequential (1 worker)",
        n_samples=len(signal)
    )

    # Test 2: Parallel screening
    def screen_parallel():
        config = QualityScreeningConfig(
            segment_duration=10.0,
            max_workers=4
        )
        screener = SignalQualityScreener(config)
        screener.sampling_rate = fs
        screener.signal_type = 'ppg'
        return screener.screen_signal(signal)

    metrics, _ = runner.run_benchmark(
        screen_parallel,
        "Parallel (4 workers)",
        n_samples=len(signal)
    )


def benchmark_end_to_end(runner: BenchmarkRunner, test_dir: Path):
    """Benchmark end-to-end workflow."""
    print("\n" + "="*70)
    print("BENCHMARK 4: End-to-End Workflow")
    print("="*70)

    # Create test file
    test_file = test_dir / 'workflow_test.csv'
    create_test_oucru_file(test_file, duration_sec=120, fs=250)  # 2 minutes

    def complete_workflow():
        # Load
        loader = DataLoader(test_file, format=DataFormat.OUCRU_CSV)
        data = loader.load(time_column='timestamp')
        signal = data['signal'].values
        fs = loader.sampling_rate

        # Screen
        config = QualityScreeningConfig(max_workers=2)
        screener = SignalQualityScreener(config)
        screener.sampling_rate = fs
        screener.signal_type = 'ppg'
        quality_result = screener.screen_signal(signal)

        # Process
        pipeline = OptimizedStandardProcessingPipeline()
        processing_result = pipeline.process_signal(signal, fs, 'ppg')

        return data, quality_result, processing_result

    metrics, _ = runner.run_benchmark(
        complete_workflow,
        "Complete Workflow (Load → Screen → Process)",
        n_samples=120*250
    )


def generate_report(runner: BenchmarkRunner):
    """Generate benchmark report."""
    print("\n" + "="*70)
    print("BENCHMARK REPORT")
    print("="*70)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System: {psutil.cpu_count()} cores, {psutil.virtual_memory().total / (1024**3):.1f}GB RAM")
    print("\nAll benchmarks completed successfully!")
    print("\nNote: Detailed metrics were printed during execution above.")


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description='Run vitalDSP Phase 1-3 benchmarks')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmarks only')
    parser.add_argument('--full', action='store_true', help='Run all benchmarks including slow ones')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("vitalDSP Phase 1-3 Benchmark Suite")
    print("="*70)
    print(f"Mode: {'Quick' if args.quick else 'Full' if args.full else 'Standard'}")

    runner = BenchmarkRunner()

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)

        try:
            # Run benchmarks
            benchmark_data_loading(runner, test_dir)
            benchmark_processing(runner)
            benchmark_quality_screening(runner)

            if not args.quick:
                benchmark_end_to_end(runner, test_dir)

            # Generate report
            generate_report(runner)

        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user.")
            return 1
        except Exception as e:
            print(f"\n\nBenchmark failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
