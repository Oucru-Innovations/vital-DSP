# vitalDSP Large File Processing - User Guide

**Version:** 1.0
**Date:** October 17, 2025
**Phase:** 4 (Optimization & Testing)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Understanding File Sizes](#understanding-file-sizes)
4. [Loading Strategies](#loading-strategies)
5. [Processing Large Files](#processing-large-files)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)
9. [Examples](#examples)

---

## Introduction

vitalDSP's Phase 1-3 implementation provides robust support for processing large physiological signal files, from small 1-minute recordings to multi-day datasets exceeding 2GB. This guide helps you choose the right approach for your data.

### What You'll Learn

- How to load files of any size efficiently
- When to use standard vs. streaming loaders
- How to optimize memory usage
- Best practices for large-scale processing

### System Requirements

| File Size | Recommended RAM | Processing Time (estimate) |
|-----------|----------------|---------------------------|
| <100MB | 2GB | Seconds to minutes |
| 100-500MB | 4GB | Minutes |
| 500MB-2GB | 8GB | Minutes to tens of minutes |
| >2GB | 16GB+ | Tens of minutes to hours |

---

## Quick Start

### Example 1: Small File (Quick & Simple)

```python
from vitalDSP.utils.data_processing.data_loader import load_oucru_csv

# Load small OUCRU file (<100MB)
signal, metadata = load_oucru_csv('ecg_1hour.csv')

print(f"Loaded {len(signal)} samples")
print(f"Sampling rate: {metadata['sampling_rate']} Hz")
print(f"Duration: {metadata['duration_seconds']/60:.1f} minutes")
```

### Example 2: Large File (Automatic Streaming)

```python
from vitalDSP.utils.data_processing.data_loader import DataLoader, DataFormat

# Load large OUCRU file (>100MB)
# Streaming is automatically enabled
loader = DataLoader('ecg_24hours.csv', format=DataFormat.OUCRU_CSV)
data = loader.load()

signal = data['signal'].values
print(f"Loaded {len(signal)} samples using streaming")
print(f"Peak memory: ~50MB (vs ~900MB without streaming)")
```

### Example 3: Complete Workflow

```python
from vitalDSP.utils.data_processing.data_loader import DataLoader, DataFormat
from vitalDSP.utils.core_infrastructure.processing_pipeline import StandardProcessingPipeline
from vitalDSP.utils.core_infrastructure.quality_screener import SignalQualityScreener

# Load large file
loader = DataLoader('ppg_multiday.csv', format=DataFormat.OUCRU_CSV)
data = loader.load(chunk_size=5000)  # Force streaming with custom chunk size
signal = data['signal'].values
fs = loader.sampling_rate

# Screen quality with SignalQualityIndex integration
screener = SignalQualityScreener()
screener.sampling_rate = fs
screener.signal_type = 'ppg'
results = screener.screen_signal(signal)

# Calculate pass rate
passed = sum(1 for r in results if r.passed_screening)
pass_rate = passed / len(results) if results else 0.0
print(f"Quality pass rate: {pass_rate:.1%}")
print(f"Passed {passed}/{len(results)} segments using comprehensive SQI metrics")

# Process
pipeline = StandardProcessingPipeline()
processing_result = pipeline.process(signal, fs, 'ppg')
print(f"Processing success: {processing_result.success}")
```

---

## Understanding File Sizes

### File Size Categories

vitalDSP automatically categorizes files and selects optimal loading strategies:

#### Small Files (<100MB)
- **Typical:** 1-60 minutes of data
- **Strategy:** Standard loading
- **Memory:** 2-3x file size
- **Speed:** Very fast (seconds)
- **Examples:**
  - 10 min PPG @ 100 Hz: ~5 MB
  - 1 hour ECG @ 128 Hz: ~20 MB

#### Medium Files (100-500MB)
- **Typical:** 1-5 hours of data
- **Strategy:** Streaming (automatic)
- **Memory:** 50-100 MB peak
- **Speed:** Fast (1-5 minutes)
- **Examples:**
  - 2 hours ECG @ 250 Hz: ~100 MB
  - 5 hours PPG @ 100 Hz: ~200 MB

#### Large Files (500MB-2GB)
- **Typical:** 5-24 hours of data
- **Strategy:** Streaming with adaptive chunking
- **Memory:** 50-150 MB peak
- **Speed:** Moderate (5-20 minutes)
- **Examples:**
  - 12 hours ECG @ 250 Hz: ~500 MB
  - 24 hours ECG @ 250 Hz: ~1 GB

#### Very Large Files (>2GB)
- **Typical:** Multi-day recordings
- **Strategy:** Streaming with small chunks
- **Memory:** 50-200 MB peak
- **Speed:** Slow (20+ minutes)
- **Examples:**
  - 2 days ECG @ 250 Hz: ~2 GB
  - 7 days PPG @ 100 Hz: ~3 GB

---

## Loading Strategies

### 1. Standard Loading (Automatic for <100MB)

**When to use:**
- Files <100MB
- Fastest processing needed
- Sufficient RAM available

**How it works:**
```python
from vitalDSP.utils.data_processing.data_loader import DataLoader, DataFormat

loader = DataLoader('small_file.csv', format=DataFormat.OUCRU_CSV)
data = loader.load()  # Standard loader automatically selected
```

**Characteristics:**
- Loads entire file into memory
- Fastest processing
- Memory usage: 2-3x file size
- Optimized with json.loads() and vectorized timestamps

### 2. Streaming Loading (Automatic for >100MB)

**When to use:**
- Files >100MB
- Limited RAM
- Memory efficiency critical

**How it works:**
```python
loader = DataLoader('large_file.csv', format=DataFormat.OUCRU_CSV)
data = loader.load()  # Streaming automatically enabled for >100MB
```

**Characteristics:**
- Processes file in chunks
- Constant memory usage (~50MB)
- Slightly slower than standard
- Adaptive chunk sizing based on file size

### 3. Custom Chunk Size

**When to use:**
- Fine-tuning performance
- Specific memory constraints
- Very large files

**How it works:**
```python
# Force streaming with custom chunk size
loader = DataLoader('file.csv', format=DataFormat.OUCRU_CSV)
data = loader.load(chunk_size=2000)  # 2000 rows per chunk

# Check what was used
print(f"Format: {loader.metadata['format']}")
print(f"Chunk size: {loader.metadata.get('chunk_size', 'N/A')}")
```

**Chunk size guidelines:**

| File Size | Recommended Chunk Size | Memory Impact |
|-----------|----------------------|---------------|
| 100-200MB | 10,000 rows | ~30MB |
| 200-500MB | 5,000 rows | ~40MB |
| 500MB-1GB | 2,000 rows | ~50MB |
| >1GB | 1,000 rows | ~60MB |

---

## Processing Large Files

### End-to-End Pipeline

The complete workflow for large files:

```python
from vitalDSP.utils.data_processing.data_loader import DataLoader, DataFormat
from vitalDSP.utils.core_infrastructure.quality_screener import (
    SignalQualityScreener, QualityScreeningConfig
)
from vitalDSP.utils.core_infrastructure.processing_pipeline import (
    OptimizedStandardProcessingPipeline
)

# Step 1: Load with streaming
print("Loading large file...")
loader = DataLoader('large_ecg.csv', format=DataFormat.OUCRU_CSV)
data = loader.load()  # Automatic streaming
signal = data['signal'].values
fs = loader.sampling_rate

print(f"Loaded: {len(signal):,} samples ({loader.metadata['duration_seconds']/3600:.1f} hours)")

# Step 2: Quality screening with parallel processing and SQI integration
print("Screening quality...")
config = QualityScreeningConfig(
    segment_duration=10.0,
    overlap=0.5,
    max_workers=4  # Use 4 threads for parallel processing
)
screener = SignalQualityScreener(config)
screener.sampling_rate = fs
screener.signal_type = 'ecg'  # Enables ECG-specific SQI metrics

results = screener.screen_signal(signal)

passed = sum(1 for r in results if r.passed_screening)
pass_rate = passed / len(results) if results else 0.0
print(f"Quality pass rate: {pass_rate:.1%}")
print(f"Good segments: {passed}/{len(results)}")
print(f"Using ECG-specific SQI: amplitude_variability, baseline_wander, zero_crossing, HRV")

# Step 3: Process with optimized pipeline
print("Processing...")
pipeline = OptimizedStandardProcessingPipeline()
result = pipeline.process(signal, fs, 'ecg')

if result.success:
    print(f"Processing complete!")
    print(f"Selected path: {result.metadata.get('selected_path', 'N/A')}")
    print(f"Processing quality: {result.metadata.get('processing_quality', 'N/A')}")
else:
    print(f"Processing failed: {result.error_message}")
```

### Memory-Efficient Processing

For extremely large files, process in segments:

```python
from vitalDSP.utils.data_processing.data_loader import DataLoader, DataFormat
import numpy as np

def process_large_file_in_segments(filepath, segment_duration_sec=300):
    """
    Process large file in 5-minute segments.

    Args:
        filepath: Path to OUCRU CSV file
        segment_duration_sec: Segment duration in seconds (default: 300 = 5 min)
    """
    # Load metadata only (peek at first chunk)
    loader = DataLoader(filepath, format=DataFormat.OUCRU_CSV)

    # For very large files, you might want to use pandas chunking directly
    import pandas as pd

    results = []
    chunk_iter = pd.read_csv(filepath, chunksize=segment_duration_sec)

    for i, chunk in enumerate(chunk_iter):
        print(f"Processing segment {i+1}...")

        # Process chunk
        # (You would expand OUCRU format and process here)

        # Collect results
        results.append({
            'segment': i,
            'rows': len(chunk)
        })

        # Explicit cleanup
        del chunk

    return results

# Example usage
results = process_large_file_in_segments('multiday_ecg.csv')
print(f"Processed {len(results)} segments")
```

---

## Performance Optimization

### 1. Loading Optimization

#### Use Streaming for Large Files
```python
# Automatic (recommended)
data = loader.load()  # Auto-detects and uses streaming for >100MB

# Manual (for fine-tuning)
data = loader.load(chunk_size=5000)  # Smaller chunks = less memory
```

#### Optimize Chunk Size
```python
import os

# Calculate optimal chunk size based on available memory
file_size_mb = os.path.getsize('file.csv') / (1024 * 1024)
available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)

if file_size_mb > 500:
    chunk_size = 1000  # Small chunks for very large files
elif file_size_mb > 200:
    chunk_size = 2000
else:
    chunk_size = 5000

data = loader.load(chunk_size=chunk_size)
```

### 2. Processing Optimization

#### Use Parallel Quality Screening with SQI Integration
```python
# Sequential (default) - best for small files
config = QualityScreeningConfig(max_workers=1)

# Parallel (4x faster on 4-core systems) - recommended for large files
config = QualityScreeningConfig(max_workers=4)

screener = SignalQualityScreener(config)
screener.sampling_rate = fs
screener.signal_type = signal_type  # Enables signal-specific SQI metrics
results = screener.screen_signal(signal)

# The screener now uses comprehensive SignalQualityIndex metrics:
# - ECG: amplitude_variability, baseline_wander, zero_crossing, HRV
# - PPG: ppg_quality, baseline_wander, waveform_similarity
# - EEG: eeg_band_power, entropy, skewness, kurtosis
```

#### Use Optimized Pipeline
```python
# Standard pipeline
from vitalDSP.utils.core_infrastructure.processing_pipeline import StandardProcessingPipeline
pipeline = StandardProcessingPipeline()

# Optimized pipeline (better for large files)
from vitalDSP.utils.core_infrastructure.processing_pipeline import OptimizedStandardProcessingPipeline
pipeline = OptimizedStandardProcessingPipeline()
```

### 3. Memory Management

#### Monitor Memory Usage
```python
import psutil
import os

process = psutil.Process(os.getpid())

def print_memory_usage():
    mem_info = process.memory_info()
    print(f"Memory: {mem_info.rss / (1024**2):.1f} MB")

print_memory_usage()  # Before loading
data = loader.load()
print_memory_usage()  # After loading
```

#### Explicit Cleanup
```python
import gc

# Process data
result = pipeline.process(signal, fs, signal_type)

# Extract what you need
final_signal = result.signal.copy()

# Cleanup
del signal, result, data
gc.collect()  # Force garbage collection
```

---

## Troubleshooting

### Problem: Out of Memory Error

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Use smaller chunk size:**
```python
data = loader.load(chunk_size=1000)  # Reduce from default
```

2. **Process in segments:**
```python
# Process file in 5-minute segments
for segment in process_in_segments(filepath, segment_duration=300):
    result = pipeline.process(segment, fs, signal_type)
    # Save results incrementally
```

3. **Close other applications:**
```python
import psutil
print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
# Close other programs if too low
```

### Problem: Slow Loading

**Symptoms:**
- Loading takes >5 minutes for <500MB file

**Solutions:**

1. **Verify streaming is enabled:**
```python
data = loader.load()
print(f"Format used: {loader.metadata['format']}")
# Should show 'oucru_csv_streaming' for large files
```

2. **Check disk speed:**
```python
import time
start = time.time()
data = loader.load()
print(f"Load time: {time.time() - start:.1f}s")
# SSD: <30s for 500MB, HDD: <90s for 500MB
```

3. **Optimize chunk size:**
```python
# Larger chunks = faster (but more memory)
data = loader.load(chunk_size=10000)  # Increase from default
```

### Problem: Quality Screening Too Slow

**Symptoms:**
- Quality screening takes >10 minutes

**Solutions:**

1. **Enable parallel processing:**
```python
config = QualityScreeningConfig(max_workers=4)  # Use 4 threads
```

2. **Increase segment duration:**
```python
config = QualityScreeningConfig(
    segment_duration=30.0,  # Larger segments = fewer segments
    overlap=0.25  # Less overlap
)
```

3. **Use less conservative thresholds:**
```python
config = QualityScreeningConfig(
    snr_threshold=8.0,  # Less strict (default: 10.0)
    min_quality_score=0.6  # Less strict (default: 0.7)
)
```

---

## Best Practices

### 1. File Organization

```
data/
├── raw/              # Original files
│   ├── small/        # <100MB
│   ├── medium/       # 100-500MB
│   └── large/        # >500MB
├── processed/        # Processing outputs
└── cache/            # Temporary cache
```

### 2. Workflow Template

```python
#!/usr/bin/env python3
"""
Template for processing large physiological signal files.
"""

from pathlib import Path
from vitalDSP.utils.data_processing.data_loader import DataLoader, DataFormat
from vitalDSP.utils.core_infrastructure.quality_screener import SignalQualityScreener, QualityScreeningConfig
from vitalDSP.utils.core_infrastructure.processing_pipeline import OptimizedStandardProcessingPipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_large_file(filepath: Path, signal_type: str = 'ecg'):
    """Process large physiological signal file."""

    logger.info(f"Processing: {filepath}")

    # Load
    logger.info("Loading data...")
    loader = DataLoader(filepath, format=DataFormat.OUCRU_CSV)
    data = loader.load()
    signal = data['signal'].values
    fs = loader.sampling_rate

    logger.info(f"Loaded {len(signal):,} samples @ {fs} Hz")

    # Screen with comprehensive SQI metrics
    logger.info("Screening quality...")
    config = QualityScreeningConfig(max_workers=4)
    screener = SignalQualityScreener(config)
    screener.sampling_rate = fs
    screener.signal_type = signal_type
    results = screener.screen_signal(signal)

    passed = sum(1 for r in results if r.passed_screening)
    pass_rate = passed / len(results) if results else 0.0
    logger.info(f"Pass rate: {pass_rate:.1%} ({passed}/{len(results)} segments)")

    # Process
    logger.info("Processing...")
    pipeline = OptimizedStandardProcessingPipeline()
    result = pipeline.process(signal, fs, signal_type)

    if result.success:
        logger.info("✅ Processing complete")
        return result
    else:
        logger.error(f"❌ Processing failed: {result.error_message}")
        return None

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python process_large_file.py <filepath> [signal_type]")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    signal_type = sys.argv[2] if len(sys.argv) > 2 else 'ecg'

    result = process_large_file(filepath, signal_type)
```

### 3. Performance Monitoring

```python
import time
import psutil
import os

class PerformanceMonitor:
    """Monitor performance metrics during processing."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.start_memory = None

    def start(self):
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / (1024**2)
        print(f"Started: {self.start_memory:.1f} MB")

    def checkpoint(self, label: str):
        elapsed = time.time() - self.start_time
        current_memory = self.process.memory_info().rss / (1024**2)
        delta = current_memory - self.start_memory
        print(f"{label}: {elapsed:.1f}s, {current_memory:.1f} MB (+{delta:.1f} MB)")

    def end(self):
        elapsed = time.time() - self.start_time
        final_memory = self.process.memory_info().rss / (1024**2)
        print(f"Completed: {elapsed:.1f}s, Peak: {final_memory:.1f} MB")

# Usage
monitor = PerformanceMonitor()
monitor.start()

data = loader.load()
monitor.checkpoint("After loading")

screener.sampling_rate = fs
screener.signal_type = signal_type
results = screener.screen_signal(signal)
monitor.checkpoint("After screening")

processed = pipeline.process(signal, fs, signal_type)
monitor.checkpoint("After processing")

monitor.end()
```

---

## Examples

### Example 1: Process 24-Hour ECG Recording

```python
from vitalDSP.utils.data_processing.data_loader import DataLoader, DataFormat
from vitalDSP.utils.core_infrastructure.processing_pipeline import OptimizedStandardProcessingPipeline

# Load 24-hour ECG (~1GB file)
loader = DataLoader('ecg_24h_250hz.csv', format=DataFormat.OUCRU_CSV)
data = loader.load()  # Uses streaming automatically

# Extract signal
signal = data['signal'].values
fs = loader.sampling_rate

print(f"Duration: {len(signal) / fs / 3600:.1f} hours")
print(f"Samples: {len(signal):,}")

# Process
pipeline = OptimizedStandardProcessingPipeline()
result = pipeline.process(signal, fs, 'ecg')

print(f"Success: {result.success}")
```

### Example 2: Quality Screening with Detailed SQI Reports

```python
from vitalDSP.utils.core_infrastructure.quality_screener import SignalQualityScreener, QualityScreeningConfig

# Configure screening with PPG-specific SQI metrics
config = QualityScreeningConfig(
    segment_duration=10.0,
    overlap=0.5,
    snr_threshold=12.0,  # PPG-specific threshold
    max_workers=4
)

screener = SignalQualityScreener(config)
screener.sampling_rate = fs
screener.signal_type = 'ppg'  # Enables PPG-specific SQI: ppg_quality, baseline_wander, waveform_similarity

results = screener.screen_signal(signal)

# Detailed analysis
passed = sum(1 for r in results if r.passed_screening)
print(f"\nQuality Report (using comprehensive SQI metrics):")
print(f"Total segments: {len(results)}")
print(f"Passed: {passed}")
print(f"Failed: {len(results) - passed}")
print(f"Pass rate: {passed/len(results)*100:.1f}%")

# Find worst segments based on overall quality
worst_segments = sorted(
    results,
    key=lambda r: r.quality_metrics.overall_quality
)[:5]

print(f"\n5 Worst Segments:")
for i, seg in enumerate(worst_segments):
    print(f"  {i+1}. Segment {seg.segment_id}:")
    print(f"      Time: {seg.start_idx/fs:.1f}s - {seg.end_idx/fs:.1f}s")
    print(f"      Overall Quality: {seg.quality_metrics.overall_quality:.2f}")
    print(f"      SNR: {seg.quality_metrics.snr_db:.1f} dB")
    if seg.warnings:
        print(f"      Warnings: {', '.join(seg.warnings)}")
```

### Example 3: Batch Processing Multiple Files

```python
from pathlib import Path
import json

def batch_process(input_dir: Path, output_dir: Path):
    """Process all OUCRU CSV files in directory."""

    output_dir.mkdir(exist_ok=True)
    csv_files = list(input_dir.glob('*.csv'))

    print(f"Found {len(csv_files)} files")

    results = []

    for i, filepath in enumerate(csv_files):
        print(f"\n[{i+1}/{len(csv_files)}] Processing: {filepath.name}")

        try:
            # Load
            loader = DataLoader(filepath, format=DataFormat.OUCRU_CSV)
            data = loader.load()
            signal = data['signal'].values
            fs = loader.sampling_rate

            # Process
            pipeline = OptimizedStandardProcessingPipeline()
            result = pipeline.process(signal, fs, 'ecg')

            # Save result
            output_file = output_dir / f"{filepath.stem}_processed.json"
            output_file.write_text(json.dumps({
                'filename': filepath.name,
                'success': result.success,
                'metadata': result.metadata,
                'duration_sec': len(signal) / fs
            }, indent=2))

            results.append({
                'file': filepath.name,
                'success': result.success
            })

        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'file': filepath.name,
                'success': False,
                'error': str(e)
            })

    # Summary
    successes = sum(1 for r in results if r['success'])
    print(f"\n{'='*60}")
    print(f"Batch complete: {successes}/{len(results)} successful")
    print(f"{'='*60}")

# Run batch processing
batch_process(
    input_dir=Path('data/raw/large'),
    output_dir=Path('data/processed')
)
```

---

## Summary

### Key Takeaways

1. **Automatic optimization**: vitalDSP automatically selects the best loading strategy
2. **Streaming for large files**: Files >100MB use streaming with ~50MB memory
3. **Parallel processing**: Use `max_workers=4` for 4x faster quality screening
4. **Monitor performance**: Use the PerformanceMonitor for tracking
5. **Process in segments**: For very large files, consider segment-based processing

### Performance Expectations

| File Size | Load Time | Memory | Processing Time |
|-----------|-----------|--------|----------------|
| 50MB | 5-10s | 150MB | 30s-1min |
| 100MB | 10-20s | 50MB | 1-2min |
| 500MB | 20-60s | 50MB | 5-10min |
| 1GB | 60-120s | 50MB | 10-20min |
| 2GB | 120-240s | 50MB | 20-40min |

*Times are approximate and vary by hardware*

### Need Help?

- **Documentation**: See developer API docs for detailed reference
- **Performance**: Check performance tuning guide for optimization
- **Examples**: Explore example notebooks for more use cases

---

*User Guide v1.0 - October 17, 2025*
