# vitalDSP Phase 1-3 API Reference

**Version:** 1.0
**Date:** October 17, 2025
**Phase:** 4 (Optimization & Testing)

---

## Table of Contents

1. [Data Loading API](#data-loading-api)
2. [Processing Pipeline API](#processing-pipeline-api)
3. [Quality Screening API](#quality-screening-api)
4. [Integration APIs](#integration-apis)
5. [Configuration](#configuration)
6. [Error Handling](#error-handling)
7. [Advanced Usage](#advanced-usage)

---

## Data Loading API

### DataLoader Class

**Location:** `src/vitalDSP/utils/data_processing/data_loader.py`

The main class for loading physiological signal data with automatic format detection and optimization.

#### Constructor

```python
DataLoader(
    file_path: Union[str, Path],
    format: DataFormat = DataFormat.AUTO,
    sampling_rate: Optional[float] = None,
    **kwargs
)
```

**Parameters:**
- `file_path`: Path to data file
- `format`: Data format (AUTO, CSV, OUCRU_CSV, EXCEL, etc.)
- `sampling_rate`: Override sampling rate (Hz)
- `**kwargs`: Format-specific parameters

**Example:**
```python
from vitalDSP.utils.data_processing.data_loader import DataLoader, DataFormat

loader = DataLoader('data.csv', format=DataFormat.OUCRU_CSV, sampling_rate=250)
```

#### Methods

##### `load()`

```python
def load(
    self,
    columns: Optional[List[str]] = None,
    chunk_size: Optional[int] = None,
    **kwargs
) -> pd.DataFrame
```

Load data with automatic optimization.

**Parameters:**
- `columns`: Specific columns to load (optional)
- `chunk_size`: Force chunked loading with specified size
- `**kwargs`: Format-specific loading parameters

**Returns:**
- DataFrame with loaded data

**Behavior:**
- Files <100MB: Standard loading
- Files >100MB: Automatic streaming
- `chunk_size` specified: Force streaming

**Example:**
```python
# Automatic optimization
data = loader.load()

# Force streaming with custom chunk size
data = loader.load(chunk_size=5000)

# Load specific columns
data = loader.load(columns=['timestamp', 'signal'])
```

##### `metadata` Property

```python
@property
def metadata(self) -> Dict[str, Any]
```

Access loading metadata after calling `load()`.

**Returns:**
- Dictionary with metadata:
  - `format`: Format used ('oucru_csv' or 'oucru_csv_streaming')
  - `n_rows`: Number of rows in original file
  - `n_samples`: Total samples after expansion
  - `sampling_rate`: Detected/specified sampling rate
  - `duration_seconds`: Total duration
  - `chunk_size`: Chunk size (if streaming)
  - `start_time`, `end_time`: Time range

**Example:**
```python
data = loader.load()
print(f"Format: {loader.metadata['format']}")
print(f"Duration: {loader.metadata['duration_seconds']:.1f}s")
print(f"Sampling rate: {loader.metadata['sampling_rate']} Hz")
```

### Convenience Functions

#### `load_oucru_csv()`

**Location:** `src/vitalDSP/utils/data_processing/data_loader.py`

```python
def load_oucru_csv(
    file_path: Union[str, Path],
    signal_type_hint: Optional[str] = None,
    sampling_rate: Optional[float] = None,
    time_column: str = "timestamp",
    signal_column: str = "signal",
    sampling_rate_column: Optional[str] = "sampling_rate",
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]
```

Simplified OUCRU CSV loading.

**Parameters:**
- `file_path`: Path to OUCRU CSV file
- `signal_type_hint`: Signal type ('ppg', 'ecg', etc.) for default sampling rate
- `sampling_rate`: Override sampling rate
- `time_column`: Timestamp column name (default: 'timestamp')
- `signal_column`: Signal array column name (default: 'signal')
- `sampling_rate_column`: Sampling rate column name (optional)

**Returns:**
- Tuple of (signal array, metadata dict)

**Example:**
```python
from vitalDSP.utils.data_processing.data_loader import load_oucru_csv

# Simple load
signal, metadata = load_oucru_csv('ecg_data.csv')

# With signal type hint
signal, metadata = load_oucru_csv(
    'ppg_data.csv',
    signal_type_hint='ppg'  # Uses 100 Hz default
)

# Custom column names
signal, metadata = load_oucru_csv(
    'custom.csv',
    time_column='time',
    signal_column='values'
)
```

---

## Processing Pipeline API

### StandardProcessingPipeline Class

**Location:** `src/vitalDSP/utils/core_infrastructure/processing_pipeline.py`

8-stage processing pipeline with multi-path processing.

#### Constructor

```python
StandardProcessingPipeline(config: Optional[Dict[str, Any]] = None)
```

**Parameters:**
- `config`: Pipeline configuration (optional, uses defaults if not provided)

**Example:**
```python
from vitalDSP.utils.core_infrastructure.processing_pipeline import StandardProcessingPipeline

pipeline = StandardProcessingPipeline()
```

#### Methods

##### `process()`

```python
def process(
    self,
    signal: np.ndarray,
    fs: float,
    signal_type: str,
    **kwargs
) -> ProcessingResult
```

Process signal through 8-stage pipeline.

**Parameters:**
- `signal`: Input signal array
- `fs`: Sampling rate (Hz)
- `signal_type`: Signal type ('ppg', 'ecg', 'eeg', etc.)
- `**kwargs`: Additional processing parameters

**Returns:**
- `ProcessingResult` object with:
  - `success`: Boolean indicating success
  - `signal`: Processed signal
  - `metadata`: Processing metadata
  - `quality_metrics`: Quality assessment
  - `error_message`: Error details (if failed)

**Example:**
```python
result = pipeline.process(
    signal=signal_data,
    fs=250,
    signal_type='ecg'
)

if result.success:
    processed_signal = result.signal
    print(f"Quality: {result.metadata['processing_quality']}")
else:
    print(f"Error: {result.error_message}")
```

### OptimizedStandardProcessingPipeline Class

**Location:** `src/vitalDSP/utils/core_infrastructure/processing_pipeline.py`

Optimized version with performance enhancements for large files.

```python
from vitalDSP.utils.core_infrastructure.processing_pipeline import OptimizedStandardProcessingPipeline

# Same API as StandardProcessingPipeline
pipeline = OptimizedStandardProcessingPipeline()
result = pipeline.process(signal, fs, 'ppg')
```

**Differences from Standard:**
- Optimized Stage 3 (multi-path processing)
- Reduced memory allocations
- Better for files >5 minutes

### ProcessingResult Class

#### Attributes

```python
@dataclass
class ProcessingResult:
    success: bool  # Processing succeeded
    signal: np.ndarray  # Processed signal
    metadata: Dict[str, Any]  # Processing metadata
    quality_metrics: Dict[str, float]  # Quality scores
    error_message: Optional[str]  # Error if failed
```

#### Metadata Contents

```python
{
    'pipeline_version': str,  # Pipeline version
    'signal_type': str,  # Input signal type
    'sampling_rate': float,  # Sampling rate
    'selected_path': str,  # Best processing path ('raw', 'filtered', 'preprocessed', 'full')
    'processing_quality': str,  # Quality level ('excellent', 'good', 'acceptable', 'poor')
    'stages_completed': List[str],  # Completed stages
    'processing_time': float,  # Total processing time (seconds)
    'distortion_level': float,  # Distortion introduced
}
```

---

## Quality Screening API

### SignalQualityScreener Class

**Location:** `src/vitalDSP/utils/core_infrastructure/quality_screener.py`

3-stage quality screening with **deep SignalQualityIndex integration** and parallel processing support.

#### Overview

The QualityScreener implements a hierarchical 3-stage approach:

1. **Stage 1: Quick SNR Check** - Fast signal-to-noise ratio estimation
2. **Stage 2: Statistical Screen** - Statistical anomaly detection (outliers, jumps, constant values)
3. **Stage 3: Signal-Specific Screen** - Comprehensive SQI assessment using vitalDSP's `SignalQualityIndex` module with signal-type-specific metrics

**Stage 3 SignalQualityIndex Integration:**
- **ECG signals:** amplitude_variability_sqi, baseline_wander_sqi, zero_crossing_sqi, heart_rate_variability_sqi
- **PPG signals:** ppg_signal_quality_sqi, baseline_wander_sqi, waveform_similarity_sqi
- **EEG signals:** eeg_band_power_sqi, signal_entropy_sqi, skewness_sqi, kurtosis_sqi
- **Generic signals:** energy_sqi, amplitude_variability_sqi, baseline_wander_sqi

All SQI methods are automatically configured based on `segment_duration`, `overlap`, and signal-type-specific thresholds.

#### Constructor

```python
SignalQualityScreener(config: Optional[QualityScreeningConfig] = None)
```

**Parameters:**
- `config`: Screening configuration (optional, uses signal-type-specific defaults)

**Example:**
```python
from vitalDSP.utils.core_infrastructure.quality_screener import (
    SignalQualityScreener,
    QualityScreeningConfig
)

# Use defaults (will adapt to signal_type when set)
screener = SignalQualityScreener()
screener.sampling_rate = 250
screener.signal_type = 'ecg'

# Or with custom config
config = QualityScreeningConfig(
    segment_duration=10.0,
    overlap=0.5,
    snr_threshold=15.0,  # Higher threshold for ECG
    max_workers=4
)
screener = SignalQualityScreener(config)
```

#### Methods

##### `screen_signal()`

```python
def screen_signal(
    self,
    signal: np.ndarray,
    progress_callback: Optional[Callable] = None
) -> List[ScreeningResult]
```

Screen signal quality using 3-stage approach with comprehensive SQI assessment.

**Parameters:**
- `signal`: Input signal array (1D numpy array)
- `progress_callback`: Optional callback for progress updates

**Requirements:**
- Must set `screener.sampling_rate` before calling
- Must set `screener.signal_type` before calling (for signal-specific SQI metrics)

**Returns:**
- List of `ScreeningResult` objects, one per segment

**Example:**
```python
# Setup screener
screener = SignalQualityScreener()
screener.sampling_rate = 250
screener.signal_type = 'ecg'

# Screen signal
results = screener.screen_signal(signal)

# Analyze results
passed = sum(1 for r in results if r.passed_screening)
print(f"Pass rate: {passed/len(results)*100:.1f}%")
print(f"Total segments: {len(results)}")

# Examine SQI scores for first segment
first_result = results[0]
sqi_scores = first_result.quality_metrics  # Access SQI scores
print(f"Overall quality: {first_result.quality_metrics.overall_quality:.2f}")
```

### QualityScreeningConfig Class

Configuration for quality screening with SignalQualityIndex integration.

```python
@dataclass
class QualityScreeningConfig:
    # Segmentation
    segment_duration: float = 10.0  # Segment length (seconds)
    overlap: float = 0.5  # Overlap fraction (0.0-1.0)

    # Stage 1: SNR thresholds
    snr_threshold: float = 10.0  # SNR threshold (dB)

    # Stage 3: Overall quality
    min_quality_score: float = 0.7  # Minimum overall quality score (0.0-1.0)

    # Processing
    max_workers: int = 2  # Parallel workers (1 = sequential)

    # Strictness
    conservative: bool = True  # Conservative thresholds
```

**Additional Threshold Configuration:**

The screener also supports detailed SQI thresholds via constructor kwargs. These are automatically adjusted based on signal_type:

```python
# Common SQI thresholds (0-1 normalized scores)
amplitude_variability_min: float = 0.5
baseline_wander_min: float = 0.5
zero_crossing_min: float = 0.5  # ECG-specific
waveform_similarity_min: float = 0.5  # PPG-specific
signal_entropy_min: float = 0.5  # EEG-specific
skewness_min: float = 0.5  # EEG-specific
kurtosis_min: float = 0.5  # EEG-specific
energy_min: float = 0.5  # Generic signals
hrv_min: float = 0.5  # ECG HRV
ppg_quality_min: float = 0.5  # PPG-specific
eeg_band_power_min: float = 0.5  # EEG-specific
```

**Signal-Type-Specific Defaults:**

The screener automatically adjusts thresholds based on `signal_type`:

- **ECG:** Higher SNR threshold (15 dB), stricter amplitude variability (0.6), baseline wander (0.6), zero crossing (0.6)
- **PPG:** Moderate SNR (12 dB), stricter PPG quality (0.6), baseline wander (0.6), waveform similarity (0.5)
- **EEG:** Lower SNR (8 dB), moderate entropy (0.5), lenient skewness/kurtosis (0.4)
- **Generic:** Balanced thresholds across common metrics

**Example:**
```python
from vitalDSP.utils.core_infrastructure.quality_screener import (
    SignalQualityScreener,
    QualityScreeningConfig
)

# Basic configuration
config = QualityScreeningConfig(
    segment_duration=15.0,  # Longer segments
    overlap=0.25,  # Less overlap
    max_workers=4,  # 4 parallel threads
    snr_threshold=8.0,  # Less strict
    conservative=False  # Allow more segments to pass
)

screener = SignalQualityScreener(config)
screener.signal_type = 'ecg'  # Uses ECG-specific SQI methods and thresholds
screener.sampling_rate = 250

# Advanced: Override specific SQI thresholds
config = QualityScreeningConfig(
    segment_duration=10.0,
    snr_threshold=15.0
)

screener = SignalQualityScreener(
    config,
    # Override ECG-specific thresholds
    amplitude_variability_min=0.7,  # Stricter
    baseline_wander_min=0.7,  # Stricter
    hrv_min=0.6  # Stricter HRV requirement
)
screener.signal_type = 'ecg'
screener.sampling_rate = 250
```

### ScreeningResult Class

#### Attributes

```python
@dataclass
class ScreeningResult:
    overall_pass_rate: float  # Fraction of segments that passed
    segment_results: List[SegmentScreeningResult]  # Per-segment results
    metadata: Dict[str, Any]  # Screening metadata
```

#### SegmentScreeningResult

```python
@dataclass
class SegmentScreeningResult:
    segment_index: int  # Segment number
    start_time: float  # Start time (seconds)
    end_time: float  # End time (seconds)
    passed: bool  # Passed all stages
    quality_score: float  # Overall quality (0.0-1.0)
    snr_db: float  # Signal-to-noise ratio
    stage_results: Dict[str, bool]  # Results per stage
    failure_reasons: List[str]  # Why it failed (if applicable)
```

**Example:**
```python
# Analyze worst segments
worst_segments = sorted(
    result.segment_results,
    key=lambda s: s.quality_score
)[:5]

for seg in worst_segments:
    print(f"Segment {seg.segment_index}:")
    print(f"  Time: {seg.start_time:.1f}s - {seg.end_time:.1f}s")
    print(f"  Quality: {seg.quality_score:.2f}")
    print(f"  SNR: {seg.snr_db:.1f} dB")
    if not seg.passed:
        print(f"  Failures: {', '.join(seg.failure_reasons)}")
```

---

## Integration APIs

### Webapp Integration

#### EnhancedDataService

**Location:** `src/vitalDSP_webapp/services/data/enhanced_data_service.py`

```python
from vitalDSP_webapp.services.data.enhanced_data_service import (
    EnhancedDataService,
    LoadingStrategy,
    FileSizeWarning
)

# Create service
service = EnhancedDataService()

# Analyze file before loading
analysis = service.analyze_file('large_file.csv')
print(f"Warning level: {analysis.warning_level}")
print(f"Recommended strategy: {analysis.recommended_strategy}")
print(f"Estimated time: {analysis.estimated_load_time_seconds:.1f}s")

# Load with recommended strategy
data = service.load_data(
    'large_file.csv',
    strategy=analysis.recommended_strategy
)
```

#### HeavyDataFilteringService

**Location:** `src/vitalDSP_webapp/services/filtering/heavy_data_filtering_service.py`

```python
from vitalDSP_webapp.services.filtering.heavy_data_filtering_service import HeavyDataFilteringService

service = HeavyDataFilteringService()

# Process with automatic pipeline selection
result = service.process_signal(
    signal=signal_data,
    fs=250,
    signal_type='ecg',
    use_optimized=True  # Use OptimizedStandardProcessingPipeline
)
```

---

## Configuration

### Environment Variables

vitalDSP respects the following environment variables:

```bash
# Logging
export VITALDSP_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# Performance
export VITALDSP_MAX_WORKERS=4  # Default parallel workers
export VITALDSP_CHUNK_SIZE=5000  # Default chunk size for streaming

# Memory
export VITALDSP_MEMORY_LIMIT_GB=8  # Memory limit
export VITALDSP_STREAMING_THRESHOLD_MB=100  # Auto-streaming threshold
```

### Configuration Files

#### pipeline_config.yaml

```yaml
# Processing pipeline configuration
pipeline:
  version: "1.0"
  conservative_processing: true
  enable_multi_path: true

  stages:
    validation:
      enabled: true
      min_samples: 100

    quality_check:
      enabled: true
      snr_threshold: 10.0

    parallel_processing:
      paths: ["raw", "filtered", "preprocessed", "full"]
      auto_select: true
```

#### quality_config.yaml

```yaml
# Quality screening configuration
quality_screening:
  segmentation:
    duration_sec: 10.0
    overlap: 0.5

  thresholds:
    snr_db: 10.0
    quality_score: 0.7

  processing:
    max_workers: 2
    conservative: true
```

---

## Error Handling

### Exception Hierarchy

```python
vitalDSP.exceptions.VitalDSPException
├── DataLoadingError
│   ├── FileNotFoundError
│   ├── FormatError
│   └── CorruptedDataError
├── ProcessingError
│   ├── InvalidSignalError
│   ├── ProcessingFailedError
│   └── QualityCheckFailedError
└── ConfigurationError
    ├── InvalidConfigError
    └── MissingParameterError
```

### Error Handling Examples

```python
from vitalDSP.utils.data_processing.data_loader import DataLoader, DataFormat
from vitalDSP.exceptions import DataLoadingError, ProcessingError

try:
    # Load data
    loader = DataLoader('data.csv', format=DataFormat.OUCRU_CSV)
    data = loader.load()
    signal = data['signal'].values

except FileNotFoundError:
    print("Error: File not found")
except DataLoadingError as e:
    print(f"Error loading data: {e}")
    # Fallback logic
except Exception as e:
    print(f"Unexpected error: {e}")
    raise

try:
    # Process
    pipeline = StandardProcessingPipeline()
    result = pipeline.process(signal, fs, 'ecg')

    if not result.success:
        raise ProcessingError(result.error_message)

except ProcessingError as e:
    print(f"Processing failed: {e}")
    # Try alternative pipeline
    pipeline = OptimizedStandardProcessingPipeline()
    result = pipeline.process(signal, fs, 'ecg')
```

### Graceful Degradation

```python
def robust_processing(filepath, signal_type='ecg'):
    """Process with multiple fallback strategies."""

    # Try optimized pipeline first
    try:
        pipeline = OptimizedStandardProcessingPipeline()
        result = pipeline.process(signal, fs, signal_type)
        if result.success:
            return result
    except Exception as e:
        print(f"Optimized pipeline failed: {e}")

    # Fallback to standard pipeline
    try:
        pipeline = StandardProcessingPipeline()
        result = pipeline.process(signal, fs, signal_type)
        if result.success:
            return result
    except Exception as e:
        print(f"Standard pipeline failed: {e}")

    # Final fallback: minimal processing
    try:
        # Implement minimal processing
        return minimal_process(signal, fs)
    except Exception as e:
        print(f"All processing failed: {e}")
        return None
```

---

## Advanced Usage

### Custom Pipeline Stages

```python
from vitalDSP.utils.core_infrastructure.processing_pipeline import StandardProcessingPipeline

class CustomPipeline(StandardProcessingPipeline):
    """Custom pipeline with additional stages."""

    def _stage_custom_processing(self, context):
        """Custom processing stage."""
        signal = context['signal']
        fs = context['fs']

        # Your custom processing
        processed_signal = my_custom_algorithm(signal, fs)

        return ProcessingResult(
            success=True,
            signal=processed_signal,
            metadata={'custom_stage': 'completed'},
            quality_metrics={}
        )

    def process(self, signal, fs, signal_type, **kwargs):
        """Override to include custom stage."""
        # Run standard stages
        result = super().process(signal, fs, signal_type, **kwargs)

        if result.success:
            # Add custom stage
            context = {'signal': result.signal, 'fs': fs}
            custom_result = self._stage_custom_processing(context)
            return custom_result

        return result
```

### Streaming Processing

```python
def stream_process_large_file(filepath, output_path, chunk_duration_sec=300):
    """
    Process very large file by streaming and saving results incrementally.

    Args:
        filepath: Input OUCRU CSV
        output_path: Output file
        chunk_duration_sec: Process in chunks of this duration
    """
    import pandas as pd

    # Setup
    pipeline = OptimizedStandardProcessingPipeline()
    chunk_iter = pd.read_csv(filepath, chunksize=chunk_duration_sec)

    results = []

    for i, chunk in enumerate(chunk_iter):
        print(f"Processing chunk {i+1}...")

        # Expand OUCRU format (implement based on your needs)
        signal = expand_oucru_chunk(chunk)
        fs = detect_sampling_rate(chunk)

        # Process chunk
        result = pipeline.process(signal, fs, 'ecg')

        if result.success:
            # Save chunk result
            save_chunk_result(result, output_path, chunk_index=i)
            results.append({'chunk': i, 'success': True})
        else:
            results.append({'chunk': i, 'success': False, 'error': result.error_message})

    return results
```

### Performance Monitoring

```python
import cProfile
import pstats

def profile_pipeline(signal, fs, signal_type='ecg'):
    """Profile pipeline performance."""

    profiler = cProfile.Profile()
    profiler.enable()

    pipeline = StandardProcessingPipeline()
    result = pipeline.process(signal, fs, signal_type)

    profiler.disable()

    # Print stats
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

    return result
```

---

## API Summary

### Quick Reference

| Component | Main Class | Key Method | Returns |
|-----------|-----------|------------|---------|
| Data Loading | `DataLoader` | `load()` | `pd.DataFrame` |
| Processing | `StandardProcessingPipeline` | `process()` | `ProcessingResult` |
| Quality | `SignalQualityScreener` | `screen_signal()` | `ScreeningResult` |

### Import Cheat Sheet

```python
# Data loading
from vitalDSP.utils.data_processing.data_loader import (
    DataLoader,
    DataFormat,
    load_oucru_csv
)

# Processing
from vitalDSP.utils.core_infrastructure.processing_pipeline import (
    StandardProcessingPipeline,
    OptimizedStandardProcessingPipeline,
    ProcessingResult
)

# Quality screening
from vitalDSP.utils.core_infrastructure.quality_screener import (
    SignalQualityScreener,
    QualityScreeningConfig,
    ScreeningResult
)

# Webapp integration
from vitalDSP_webapp.services.data.enhanced_data_service import (
    EnhancedDataService,
    LoadingStrategy
)
```

---

*API Reference v1.0 - October 17, 2025*
