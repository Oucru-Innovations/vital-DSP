"""
Simple Phase 1-3 Benchmark Test

A simplified version to test that the basic imports and APIs work.
"""

import numpy as np
import time
from vitalDSP.utils.core_infrastructure.processing_pipeline import (
    StandardProcessingPipeline,
    OptimizedStandardProcessingPipeline
)
from vitalDSP.utils.core_infrastructure.quality_screener import (
    SignalQualityScreener,
    QualityScreeningConfig
)

print("Phase 1-3 Simple Benchmark Test")
print("=" * 60)

# Test 1: Pipeline Import and Basic Execution
print("\n[Test 1] Processing Pipeline")
try:
    signal = np.random.randn(5000)  # 20 seconds @ 250 Hz
    fs = 250.0

    pipeline = StandardProcessingPipeline()
    start = time.time()
    result = pipeline.process_signal(signal, fs, 'ppg')
    elapsed = time.time() - start

    print(f"  Status: {result.get('success', False)}")
    print(f"  Time: {elapsed:.3f}s")
    print("  PASSED")
except Exception as e:
    print(f"  FAILED: {e}")

# Test 2: Optimized Pipeline
print("\n[Test 2] Optimized Processing Pipeline")
try:
    signal = np.random.randn(5000)
    fs = 250.0

    pipeline = OptimizedStandardProcessingPipeline()
    start = time.time()
    result = pipeline.process_signal(signal, fs, 'ppg')
    elapsed = time.time() - start

    print(f"  Status: {result.get('success', False)}")
    print(f"  Time: {elapsed:.3f}s")
    print("  PASSED")
except Exception as e:
    print(f"  FAILED: {e}")

# Test 3: Quality Screener with Config
print("\n[Test 3] Quality Screener with Config")
try:
    signal = np.random.randn(10000)  # 40 seconds @ 250 Hz
    fs = 250.0

    config = QualityScreeningConfig(
        segment_duration=10.0,
        max_workers=2
    )
    screener = SignalQualityScreener(config)
    screener.sampling_rate = fs
    screener.signal_type = 'ppg'

    start = time.time()
    results = screener.screen_signal(signal)
    elapsed = time.time() - start

    passed = sum(1 for r in results if r.passed_screening)
    print(f"  Segments: {len(results)}")
    print(f"  Passed: {passed}")
    print(f"  Pass rate: {passed/len(results)*100:.1f}%")
    print(f"  Time: {elapsed:.3f}s")
    print("  PASSED")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Parallel Quality Screening
print("\n[Test 4] Parallel Quality Screening")
try:
    signal = np.random.randn(10000)
    fs = 250.0

    config = QualityScreeningConfig(
        segment_duration=10.0,
        max_workers=4  # Parallel
    )
    screener = SignalQualityScreener(config)
    screener.sampling_rate = fs
    screener.signal_type = 'ppg'

    start = time.time()
    results = screener.screen_signal(signal)
    elapsed = time.time() - start

    passed = sum(1 for r in results if r.passed_screening)
    print(f"  Segments: {len(results)}")
    print(f"  Passed: {passed}")
    print(f"  Time: {elapsed:.3f}s")
    print("  PASSED")
except Exception as e:
    print(f"  FAILED: {e}")

print("\n" + "=" * 60)
print("Simple benchmark tests complete!")
