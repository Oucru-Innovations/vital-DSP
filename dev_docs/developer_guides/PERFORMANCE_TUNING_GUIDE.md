# vitalDSP Performance Tuning Guide

**Version:** 1.0
**Date:** October 17, 2025
**Phase:** 4 (Optimization & Testing)

---

## Table of Contents

1. [Performance Overview](#performance-overview)
2. [Profiling Tools](#profiling-tools)
3. [Memory Optimization](#memory-optimization)
4. [CPU Optimization](#cpu-optimization)
5. [I/O Optimization](#io-optimization)
6. [Configuration Tuning](#configuration-tuning)
7. [Common Performance Issues](#common-performance-issues)
8. [Benchmarking](#benchmarking)

---

## Performance Overview

### Performance Characteristics

vitalDSP Phase 1-3 is designed for high performance with large physiological signal datasets.

#### Baseline Performance (250 Hz ECG)

| File Size | Duration | Load Time | Process Time | Total Time | Peak Memory |
|-----------|----------|-----------|--------------|------------|-------------|
| 10 MB | 10 min | 1-2s | 5-10s | ~15s | 30 MB |
| 50 MB | 1 hour | 5-10s | 30-60s | ~1min | 150 MB |
| 100 MB | 2 hours | 10-20s | 1-2min | ~3min | 50 MB (streaming) |
| 500 MB | 10 hours | 20-60s | 5-10min | ~15min | 50 MB |
| 1 GB | 24 hours | 1-2min | 10-20min | ~25min | 50 MB |

*Hardware: 4-core CPU, 8GB RAM, SSD*

### Performance Bottlenecks

Common bottlenecks in order of impact:

1. **I/O (40-60% of time)** - Reading large files from disk
2. **Array parsing (20-30%)** - Converting string arrays to numeric
3. **Timestamp generation (10-20%)** - Creating interpolated timestamps
4. **Signal processing (10-15%)** - Filtering, feature extraction
5. **Quality screening (5-10%)** - SNR calculation, statistical tests

---

## Profiling Tools

### 1. Built-in Performance Profiler

**Location:** `dev_tools/profiling/performance_profiler.py`

```python
from dev_tools.profiling.performance_profiler import PerformanceProfiler

profiler = PerformanceProfiler()

# Profile a component
result, output = profiler.profile_function(
    func=my_function,
    component_name="My Component",
    arg1, arg2
)

# Generate report
profiler.generate_report('profiling_report.md')
```

**Output:**
- Execution time breakdown
- Memory usage per function
- Hotspot identification
- Optimization recommendations

### 2. Memory Leak Detector

**Location:** `dev_tools/profiling/memory_leak_detector.py`

```python
from dev_tools.profiling.memory_leak_detector import MemoryLeakDetector

detector = MemoryLeakDetector()

# Detect leaks by repeated execution
result = detector.detect_leaks(
    func=my_function,
    component_name="Component Name",
    iterations=20
)

# View results
print(f"Leak detected: {result.leak_detected}")
print(f"Growth rate: {result.growth_rate_mb_per_iter:.3f} MB/iter")
```

### 3. Python cProfile

```python
import cProfile
import pstats

# Profile code
profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = pipeline.process(signal, fs, signal_type)

profiler.disable()

# Analyze
stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### 4. Memory Profiler

```python
from memory_profiler import profile

@profile
def my_function():
    # Your code
    pass

# Run and view line-by-line memory usage
```

### 5. Line Profiler (for CPU hotspots)

```bash
# Install
pip install line_profiler

# Use decorator
from line_profiler import LineProfiler

profiler = LineProfiler()
profiler.add_function(function_to_profile)
profiler.run('function_to_profile(args)')
profiler.print_stats()
```

---

## Memory Optimization

### 1. Use Streaming for Large Files

**Before (loads entire file):**
```python
# Uses 900 MB for 500 MB file
loader = DataLoader('large.csv', format=DataFormat.OUCRU_CSV)
data = loader.load()  # Full load
```

**After (streaming):**
```python
# Uses ~50 MB for 500 MB file
loader = DataLoader('large.csv', format=DataFormat.OUCRU_CSV)
data = loader.load(chunk_size=2000)  # Streaming
```

**Impact:** 90% memory reduction

### 2. Delete Intermediate Results

```python
# Bad: Keeps all intermediate data
data = loader.load()
signal = data['signal'].values
filtered = filter_signal(signal)
processed = process_signal(filtered)

# Good: Cleanup as you go
data = loader.load()
signal = data['signal'].values
del data  # Free DataFrame

filtered = filter_signal(signal)
del signal  # Free original

processed = process_signal(filtered)
del filtered  # Free intermediate

import gc
gc.collect()  # Force garbage collection
```

**Impact:** 50-70% memory reduction

### 3. Use In-Place Operations

```python
# Bad: Creates copy
signal_filtered = scipy.signal.butter(signal)

# Good: Modify in-place where possible
signal[:] = scipy.signal.butter(signal)
```

### 4. Generator-Based Processing

```python
def process_in_chunks(filepath, chunk_size=5000):
    """Generator that yields processed chunks."""
    loader = DataLoader(filepath)

    for chunk in loader.load_chunks(chunk_size):
        signal = chunk['signal'].values
        yield pipeline.process(signal, fs, signal_type)

# Use generator
for result in process_in_chunks('large.csv'):
    save_result(result)
    # Previous chunks are garbage collected
```

**Impact:** Constant memory usage

### 5. Memory-Mapped Arrays

For very large arrays that don't fit in RAM:

```python
import numpy as np

# Create memory-mapped array
mmap_signal = np.memmap(
    'signal.dat',
    dtype='float64',
    mode='w+',
    shape=(n_samples,)
)

# Process in chunks
chunk_size = 100000
for i in range(0, n_samples, chunk_size):
    chunk = mmap_signal[i:i+chunk_size]
    # Process chunk
```

---

## CPU Optimization

### 1. Enable Parallel Quality Screening with SQI Integration

The quality screener now uses comprehensive SignalQualityIndex metrics from vitalDSP's signal_quality_assessment module, which are CPU-intensive. Parallel processing provides significant speedup.

**Before (sequential):**
```python
config = QualityScreeningConfig(max_workers=1)
screener = SignalQualityScreener(config)
screener.sampling_rate = fs
screener.signal_type = signal_type  # Enables signal-specific SQI metrics
results = screener.screen_signal(signal)
# Time: 60s for 10-minute signal
```

**After (parallel):**
```python
config = QualityScreeningConfig(max_workers=4)  # 4 threads
screener = SignalQualityScreener(config)
screener.sampling_rate = fs
screener.signal_type = signal_type
results = screener.screen_signal(signal)
# Time: 15s (4x faster on 4-core CPU)
```

**Impact:** Near-linear speedup with CPU cores. Stage 3 SQI computation is the primary beneficiary.

**SQI Metrics by Signal Type:**
- **ECG:** amplitude_variability_sqi, baseline_wander_sqi, zero_crossing_sqi, heart_rate_variability_sqi
- **PPG:** ppg_signal_quality_sqi, baseline_wander_sqi, waveform_similarity_sqi
- **EEG:** eeg_band_power_sqi, signal_entropy_sqi, skewness_sqi, kurtosis_sqi

### 2. Vectorize Operations

**Before (loop-based):**
```python
# Slow: Python loop
timestamps = []
for i, ts in enumerate(base_timestamps):
    for j in range(samples_per_row):
        timestamps.append(ts + timedelta(seconds=j/fs))
# Time: 30s for 1M samples
```

**After (vectorized):**
```python
# Fast: NumPy vectorization
offsets = np.tile(np.arange(samples_per_row)/fs, n_rows)
row_indices = np.repeat(np.arange(n_rows), samples_per_row)
timestamps = base_timestamps[row_indices] + offsets
# Time: 0.3s for 1M samples (100x faster!)
```

**Impact:** 10-100x speedup

### 3. Use NumPy Universal Functions (ufuncs)

```python
# Slow: Python operations
result = []
for x in signal:
    result.append(math.sqrt(x**2 + 1))

# Fast: NumPy ufunc
result = np.sqrt(signal**2 + 1)
```

### 4. Avoid Python Loops for Large Arrays

```python
# Slow
mean_values = []
for i in range(0, len(signal), window_size):
    mean_values.append(np.mean(signal[i:i+window_size]))

# Fast: Vectorized with strides
from numpy.lib.stride_tricks import sliding_window_view
windows = sliding_window_view(signal, window_size)[::window_size]
mean_values = windows.mean(axis=1)
```

### 5. Optimize DataFrame Operations

```python
# Slow: apply() uses Python loop
data['processed'] = data['signal'].apply(lambda x: process(x))

# Fast: Vectorized operations
data['processed'] = process_vectorized(data['signal'].values)

# Or use numba for JIT compilation
from numba import jit

@jit(nopython=True)
def fast_process(arr):
    result = np.empty_like(arr)
    for i in range(len(arr)):
        result[i] = arr[i] ** 2 + 1
    return result

data['processed'] = fast_process(data['signal'].values)
```

---

## I/O Optimization

### 1. Use SSD Instead of HDD

| Storage | Read Speed | 500MB File Load |
|---------|-----------|-----------------|
| HDD (7200 RPM) | ~100 MB/s | 5-10s |
| SATA SSD | ~500 MB/s | 1-2s |
| NVMe SSD | ~3000 MB/s | <1s |

**Impact:** 5-10x faster I/O

### 2. Optimize Pandas read_csv()

```python
# Slow: Default settings
data = pd.read_csv('large.csv')

# Fast: Optimized settings
data = pd.read_csv(
    'large.csv',
    engine='c',  # Faster C engine
    usecols=['timestamp', 'signal'],  # Only needed columns
    low_memory=False,  # Read file in one pass
)
```

### 3. Use Binary Formats for Repeated Access

```python
# First time: Load and cache
data = loader.load()
data.to_feather('cached_data.feather')  # Fast binary format

# Subsequent times: Load from cache
import pyarrow.feather as feather
data = feather.read_feather('cached_data.feather')
# 10-50x faster than re-parsing CSV
```

**Recommended formats:**
- Feather: Fastest, not compressed
- Parquet: Compressed, good for large files
- HDF5: Flexible, supports compression

### 4. Batch File Operations

```python
# Slow: Process files one at a time
for filepath in file_list:
    data = load_and_process(filepath)
    save_result(data)

# Fast: Batch processing
from concurrent.futures import ThreadPoolExecutor

def process_file(filepath):
    data = load_and_process(filepath)
    save_result(data)
    return filepath

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = executor.map(process_file, file_list)
```

---

## Configuration Tuning

### 1. Chunk Size Optimization

Find optimal chunk size for your system:

```python
import time

chunk_sizes = [1000, 2000, 5000, 10000, 20000]
results = {}

for chunk_size in chunk_sizes:
    start = time.time()
    loader = DataLoader('test.csv', format=DataFormat.OUCRU_CSV)
    data = loader.load(chunk_size=chunk_size)
    elapsed = time.time() - start

    results[chunk_size] = elapsed
    print(f"Chunk size {chunk_size}: {elapsed:.2f}s")

# Find optimal
optimal = min(results, key=results.get)
print(f"Optimal chunk size: {optimal}")
```

**Guidelines:**
- More RAM: Larger chunks (faster)
- Less RAM: Smaller chunks (uses less memory)
- SSD: Larger chunks preferred
- HDD: Smaller chunks can help

### 2. Worker Count Optimization

```python
import psutil

# Get CPU count
cpu_count = psutil.cpu_count(logical=False)  # Physical cores
logical_count = psutil.cpu_count(logical=True)  # Logical cores

# Recommended workers
recommended_workers = min(cpu_count, 4)  # Don't exceed 4 for I/O-bound tasks

config = QualityScreeningConfig(max_workers=recommended_workers)
```

**Guidelines:**
- CPU-bound: workers = physical cores
- I/O-bound: workers = physical cores (or fewer)
- Mixed: workers = 2-4 typically optimal

### 3. Quality Screening Tuning with SQI Configuration

The enhanced quality screener supports comprehensive SQI threshold configuration. Trade off between thoroughness and speed:

```python
# Fast but less thorough - fewer segments, looser thresholds
config_fast = QualityScreeningConfig(
    segment_duration=30.0,  # Larger segments = fewer segments
    overlap=0.25,  # Less overlap = fewer segments
    snr_threshold=8.0,  # Less strict
    max_workers=4
)
screener_fast = SignalQualityScreener(
    config_fast,
    # Override SQI thresholds for speed (less strict)
    amplitude_variability_min=0.4,
    baseline_wander_min=0.4
)

# Slow but very thorough - more segments, stricter thresholds
config_thorough = QualityScreeningConfig(
    segment_duration=5.0,  # Smaller segments = more detail
    overlap=0.75,  # More overlap = better coverage
    snr_threshold=12.0,  # More strict
    max_workers=2  # Fewer workers to avoid overhead
)
screener_thorough = SignalQualityScreener(
    config_thorough,
    # Strict SQI thresholds
    amplitude_variability_min=0.7,
    baseline_wander_min=0.7,
    zero_crossing_min=0.7  # ECG-specific
)

# Balanced (recommended) - good quality assessment with reasonable speed
config_balanced = QualityScreeningConfig(
    segment_duration=10.0,
    overlap=0.5,
    snr_threshold=10.0,
    max_workers=4
)
screener_balanced = SignalQualityScreener(config_balanced)
# Uses signal-type-specific defaults

# Set signal properties
screener_balanced.sampling_rate = fs
screener_balanced.signal_type = signal_type  # Auto-selects appropriate SQI metrics
```

**Performance Impact:**
- **Fast config:** 2-3x faster than balanced, acceptable quality
- **Thorough config:** 2-3x slower than balanced, excellent quality
- **Balanced config:** Best trade-off for most use cases

---

## Common Performance Issues

### Issue 1: Slow Loading

**Symptoms:**
- Loading 100MB file takes >2 minutes

**Diagnosis:**
```python
import time

start = time.time()
data = loader.load()
load_time = time.time() - start

print(f"Load time: {load_time:.1f}s")
print(f"Format: {loader.metadata['format']}")
```

**Solutions:**
1. Verify streaming is enabled for large files
2. Check disk speed (HDD vs SSD)
3. Ensure not using DataFrame.apply() unnecessarily
4. Use binary cache for repeated access

### Issue 2: High Memory Usage

**Symptoms:**
- Process crashes with MemoryError
- System becomes unresponsive

**Diagnosis:**
```python
import psutil
import os

process = psutil.Process(os.getpid())

print(f"Current memory: {process.memory_info().rss / (1024**2):.1f} MB")
print(f"Available: {psutil.virtual_memory().available / (1024**2):.1f} MB")
```

**Solutions:**
1. Use streaming with smaller chunk_size
2. Delete intermediate variables
3. Process in segments
4. Use memory-mapped arrays

### Issue 3: Slow Quality Screening

**Symptoms:**
- Quality screening takes >50% of total time
- Stage 3 SQI computation is bottleneck

**Diagnosis:**
```python
start = time.time()
screener.sampling_rate = fs
screener.signal_type = signal_type
results = screener.screen_signal(signal)
screening_time = time.time() - start

print(f"Screening time: {screening_time:.1f}s")
print(f"Segments: {len(results)}")
print(f"Time per segment: {screening_time/len(results):.3f}s")
print(f"Using {screener.signal_type} SQI metrics")
```

**Solutions:**
1. **Enable parallel processing** (max_workers=4) - Most effective for SQI computation
2. **Increase segment_duration** (e.g., 15.0 or 30.0) - Reduces number of segments
3. **Reduce overlap** (e.g., 0.25) - Fewer overlapping segments
4. **Adjust SQI thresholds** - Override specific thresholds if acceptable:
   ```python
   screener = SignalQualityScreener(
       config,
       amplitude_variability_min=0.4,  # Less strict
       baseline_wander_min=0.4
   )
   ```
5. **Profile Stage 3** - Identify specific SQI methods causing slowdown

### Issue 4: Processing Bottleneck

**Symptoms:**
- Stage 3 (multi-path) takes majority of time

**Diagnosis:**
```python
from dev_tools.profiling.performance_profiler import PerformanceProfiler

profiler = PerformanceProfiler()
result, _ = profiler.profile_function(
    pipeline.process,
    "Pipeline Processing",
    signal, fs, signal_type
)

# Check hotspots
for hotspot in result.hotspots:
    print(hotspot)
```

**Solutions:**
1. Use OptimizedStandardProcessingPipeline
2. Disable unnecessary processing paths
3. Use vectorized operations
4. Consider algorithmic improvements

---

## Benchmarking

### Running Benchmarks

```bash
# Run full benchmark suite
pytest tests/vitalDSP/benchmarks/test_phase1_3_benchmarks.py -v

# With reporting
pytest tests/vitalDSP/benchmarks/test_phase1_3_benchmarks.py \
    --benchmark-only \
    --benchmark-autosave \
    --benchmark-save-data
```

### Custom Benchmark

```python
from tests.vitalDSP.benchmarks.test_phase1_3_benchmarks import BenchmarkRunner

runner = BenchmarkRunner()

# Benchmark your code
metrics, result = runner.run_benchmark(
    my_function,
    arg1, arg2,
    n_samples=1000000  # For throughput calculation
)

print(f"Time: {metrics.execution_time:.3f}s")
print(f"Memory: {metrics.peak_memory_mb:.1f}MB")
print(f"Throughput: {metrics.throughput_samples_per_sec:,.0f} samples/sec")
```

### Performance Regression Testing

```python
def test_performance_regression():
    """Ensure performance doesn't degrade."""

    # Baseline performance (update after verified improvements)
    BASELINE_LOAD_TIME_MAX = 20.0  # seconds for 100MB file
    BASELINE_PROCESS_TIME_MAX = 120.0  # seconds for 5min signal

    # Test loading
    import time
    start = time.time()
    data = loader.load()
    load_time = time.time() - start

    assert load_time < BASELINE_LOAD_TIME_MAX, \
        f"Loading regression: {load_time:.1f}s > {BASELINE_LOAD_TIME_MAX}s"

    # Test processing
    signal = data['signal'].values
    start = time.time()
    result = pipeline.process(signal, fs, signal_type)
    process_time = time.time() - start

    assert process_time < BASELINE_PROCESS_TIME_MAX, \
        f"Processing regression: {process_time:.1f}s > {BASELINE_PROCESS_TIME_MAX}s"
```

---

## Performance Tuning Checklist

### For Loading Large Files (>100MB)

- [ ] Verify streaming is enabled
- [ ] Optimize chunk_size for your system
- [ ] Use SSD if possible
- [ ] Load only necessary columns
- [ ] Consider binary cache for repeated access

### For Quality Screening (with SQI Integration)

- [ ] Enable parallel processing (max_workers=4) for SQI computation
- [ ] Optimize segment_duration and overlap (balance vs thoroughness)
- [ ] Set appropriate signal_type for signal-specific SQI metrics
- [ ] Adjust SQI thresholds if acceptable (amplitude_variability_min, etc.)
- [ ] Profile Stage 3 to identify SQI bottlenecks
- [ ] Consider fast config for initial screening, thorough for final analysis

### For Processing Pipeline

- [ ] Use OptimizedStandardProcessingPipeline for large files
- [ ] Vectorize custom operations
- [ ] Delete intermediate results
- [ ] Consider segment-based processing for very large files

### For Memory Optimization

- [ ] Use streaming for files >100MB
- [ ] Delete large objects after use
- [ ] Use in-place operations
- [ ] Force garbage collection explicitly
- [ ] Monitor memory with profiler

### For CPU Optimization

- [ ] Enable parallelism where available
- [ ] Vectorize NumPy operations
- [ ] Avoid Python loops on large arrays
- [ ] Use compiled code (numba, Cython) for hotspots
- [ ] Profile to identify CPU bottlenecks

---

## Performance Metrics Summary

### Target Performance (Phase 1-3)

| Metric | Target | Measured |
|--------|--------|----------|
| Load time (100MB) | <20s | 10-20s ✅ |
| Process time (5min @ 250Hz) | <2min | 1-2min ✅ |
| Memory (500MB file) | <100MB peak | ~50MB ✅ |
| Streaming overhead | <10% | ~5% ✅ |
| Parallel speedup (4 cores) | 3-4x | 3.5-4x ✅ |

---

*Performance Tuning Guide v1.0 - October 17, 2025*
