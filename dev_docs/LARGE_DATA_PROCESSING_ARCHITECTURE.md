# Large Data Processing Architecture - vitalDSP

## Executive Summary

This document outlines a comprehensive solution for handling large physiological signal datasets in vitalDSP, including efficient loading, quality-aware processing pipelines, and optimized webapp integration.

**Document Date**: October 12, 2025
**Status**: Design Document (Not Yet Implemented)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Current Limitations](#current-limitations)
3. [Proposed Architecture](#proposed-architecture)
4. [Data Processing Pipeline](#data-processing-pipeline)
5. [Addressing Critical Processing Concerns](#addressing-critical-processing-concerns)
6. [Parallel Processing Framework](#parallel-processing-framework)
7. [Memory Management Strategies](#memory-management-strategies)
8. [Quality-Aware Processing](#quality-aware-processing)
9. [Webapp Integration](#webapp-integration)
10. [Adaptive Configuration & Parameter Management](#adaptive-configuration--parameter-management)
11. [Implementation Roadmap](#implementation-roadmap)
12. [Performance Targets](#performance-targets)

---

## Problem Statement

### Challenges with Large Physiological Datasets

1. **Memory Constraints**: Loading multi-hour recordings (>1GB) into memory
2. **Processing Speed**: Real-time or near-real-time processing requirements
3. **Quality Issues**: Noisy segments waste processing resources
4. **Data Format Variety**: Multiple formats with different characteristics
5. **Webapp Responsiveness**: Browser memory limits, UI freezing
6. **Feature Extraction**: Computing features on massive datasets
7. **Storage Efficiency**: Saving processed results

### Typical Dataset Sizes

| Signal Type | Duration | Sampling Rate | File Size | Memory Footprint |
|-------------|----------|---------------|-----------|------------------|
| PPG | 1 hour | 100 Hz | ~14 MB | ~29 MB (float64) |
| ECG | 24 hours | 250 Hz | ~830 MB | ~1.6 GB |
| EEG | 8 hours | 256 Hz | ~280 MB | ~560 MB |
| Multi-channel | 12 hours | 500 Hz | ~2.1 GB | ~4.3 GB |

---

## Current Limitations

### 1. Data Loading

**Current Approach**:
```python
# Loads entire file into memory
loader = DataLoader('large_file.csv')
data, metadata = loader.load()  # May fail with MemoryError
```

**Issues**:
- No chunking support
- No lazy loading
- No progress indication
- All-or-nothing approach

### 2. Processing Pipeline

**Current Approach**:
```python
# Sequential, in-memory processing
signal = load_signal('data.csv')
filtered = filter_signal(signal)
features = extract_features(filtered)
```

**Issues**:
- Multiple copies of data in memory
- No intermediate caching
- Cannot resume interrupted processing
- No parallel processing

### 3. Quality Assessment

**Current Approach**:
```python
# Quality check after loading/processing
quality = assess_quality(signal)
```

**Issues**:
- Quality check happens too late
- Poor quality segments still processed
- Wasted computation on bad data

---

## Proposed Architecture

### 1. Hierarchical Data Loading

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Loading Layer                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Lazy Loader  │  │ Chunk Loader │  │ Stream Loader│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  Features:                                                   │
│  - Memory mapping for large files                           │
│  - Progressive loading with generators                      │
│  - Adaptive chunk sizes based on available memory          │
│  - Parallel chunk processing                                │
│  - Built-in caching mechanisms                              │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Strategy**:

#### a. Memory-Mapped Loading (for very large files)
```python
class MemoryMappedLoader:
    """
    Uses numpy.memmap to access large files without loading into RAM.
    Ideal for: Files > 2GB
    """
    def __init__(self, file_path, mode='r'):
        self.mmap = np.memmap(file_path, dtype='float64', mode=mode)

    def get_segment(self, start, end):
        """Access specific segment without loading entire file"""
        return self.mmap[start:end]
```

#### b. Chunked Loading (for medium files)
```python
class ChunkedDataLoader:
    """
    Loads data in configurable chunks.
    Ideal for: Files 100MB - 2GB
    """
    def __init__(self, file_path, chunk_size='auto'):
        self.chunk_size = self._determine_chunk_size(chunk_size)

    def _determine_chunk_size(self, size):
        """Adaptively determine chunk size based on:
        - Available system memory
        - File size
        - CPU cores available
        """
        if size == 'auto':
            available_mem = psutil.virtual_memory().available
            # Use 10% of available memory per chunk
            return int(available_mem * 0.1 / 8)  # 8 bytes per float64
        return size

    def load_chunks(self):
        """Generator yielding data chunks"""
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            yield chunk
```

#### c. Streaming Loader (for real-time or continuous data)
```python
class StreamingDataLoader:
    """
    Processes data as it arrives.
    Ideal for: Real-time monitoring, live data feeds
    """
    def __init__(self, source, buffer_size=1000):
        self.buffer = collections.deque(maxlen=buffer_size)

    def add_sample(self, sample):
        """Add single sample to buffer"""
        self.buffer.append(sample)

    def get_window(self, window_size):
        """Get sliding window of recent data"""
        return list(self.buffer)[-window_size:]
```

---

### 2. Standard Processing Pipeline (REVISED - Conservative Approach)

**Key Philosophy**: Preserve raw data, apply processing conservatively, allow user validation at each step.

```
┌─────────────────────────────────────────────────────────────┐
│              VitalDSP Conservative Processing Pipeline       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────┐
            │   1. DATA INGESTION       │
            │   - Format detection      │
            │   - Metadata extraction   │
            │   - Size estimation       │
            │   - SAVE: Raw data copy   │
            └───────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────┐
            │   2. QUALITY SCREENING    │
            │   (NON-DESTRUCTIVE)       │
            │   - Quality scores only   │
            │   - Flag suspicious       │
            │   - SAVE: Quality report  │
            │   - NO data rejection     │
            └───────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────┐
            │   3. PARALLEL PROCESSING  │
            │   Path A: Raw → Features  │
            │   Path B: Preprocessed →  │
            │           Features        │
            │   Path C: Filtered →      │
            │           Features        │
            │   SAVE: All 3 versions    │
            └───────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────┐
            │   4. QUALITY VALIDATION   │
            │   - Compare all paths     │
            │   - Detect distortions    │
            │   - Quality improvement?  │
            │   - SAVE: Comparison      │
            └───────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────┐
            │   5. SEGMENTATION         │
            │   - Multiple strategies:  │
            │     * Fixed windows       │
            │     * Adaptive (quality)  │
            │     * Event-based         │
            │   - SAVE: All segments    │
            │   - Preserve timestamps   │
            └───────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────┐
            │   6. FEATURE EXTRACTION   │
            │   - Per-segment features  │
            │   - Global features       │
            │   - Quality-tagged        │
            │   - SAVE: Feature DB      │
            └───────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────┐
            │   7. INTELLIGENT OUTPUT   │
            │   Options:                │
            │   A) Best segments only   │
            │   B) Whole signal         │
            │   C) Time-range selection │
            │   D) Quality-filtered     │
            │   - User configurable     │
            └───────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────┐
            │   8. OUTPUT PACKAGE       │
            │   - Selected data         │
            │   - Features (all paths)  │
            │   - Quality report        │
            │   - Processing log        │
            │   - Recommendations       │
            └───────────────────────────┘
```

**Critical Design Principles**:

1. **Non-Destructive Quality Screening**: Quality scores are advisory, not rejecting
2. **Parallel Path Processing**: Compare raw vs preprocessed vs filtered results
3. **User Validation**: User can see and choose which processing path works best
4. **Flexible Output**: Multiple output strategies for different use cases
5. **Full Traceability**: Every decision logged, reversible

---

### 3. Quality-Aware Processing Engine

**Key Innovation**: Integrate quality assessment at every stage to avoid wasting resources on poor-quality segments.

```python
class QualityAwareProcessor:
    """
    Processes data with continuous quality monitoring.
    Only high-quality segments proceed to expensive operations.
    """

    def __init__(self, quality_thresholds):
        self.thresholds = {
            'snr_min': 10.0,      # Minimum SNR (dB)
            'artifact_max': 5.0,   # Max artifact percentage
            'completeness_min': 95.0  # Min data completeness (%)
        }
        self.thresholds.update(quality_thresholds)

    def process_chunk(self, chunk):
        """Process a data chunk with quality gating"""

        # Stage 1: Quick quality screen (lightweight)
        quick_quality = self.quick_quality_check(chunk)

        if not quick_quality['acceptable']:
            return {
                'status': 'rejected',
                'reason': quick_quality['reason'],
                'data': None
            }

        # Stage 2: Preprocessing (if quality acceptable)
        preprocessed = self.preprocess(chunk)

        # Stage 3: Detailed quality assessment
        detailed_quality = self.detailed_quality_check(preprocessed)

        if not detailed_quality['acceptable']:
            return {
                'status': 'rejected_detailed',
                'reason': detailed_quality['reason'],
                'data': None,
                'quick_quality': quick_quality
            }

        # Stage 4: Feature extraction (only on high-quality data)
        features = self.extract_features(preprocessed)

        return {
            'status': 'success',
            'data': preprocessed,
            'features': features,
            'quality': detailed_quality
        }

    def quick_quality_check(self, chunk):
        """
        Lightweight quality check (< 1% of processing time)
        Checks:
        - Basic statistics (mean, std)
        - NaN/Inf presence
        - Amplitude range
        """
        has_nan = np.isnan(chunk).any()
        has_inf = np.isinf(chunk).any()

        if has_nan or has_inf:
            return {
                'acceptable': False,
                'reason': 'Contains NaN or Inf values'
            }

        # Check if signal is flat (std too low)
        if np.std(chunk) < 0.01:
            return {
                'acceptable': False,
                'reason': 'Signal appears flat or constant'
            }

        return {'acceptable': True}

    def detailed_quality_check(self, chunk):
        """
        Comprehensive quality assessment (5-10% of processing time)
        """
        from vitalDSP.signal_quality_assessment.signal_quality_index import SignalQualityIndex

        sqi = SignalQualityIndex(chunk)

        # Multiple quality metrics
        metrics = {
            'snr': sqi.snr_sqi(),
            'baseline': sqi.baseline_wander_sqi(),
            'completeness': self._check_completeness(chunk)
        }

        # Apply thresholds
        if metrics['snr'] < self.thresholds['snr_min']:
            return {
                'acceptable': False,
                'reason': f"SNR too low: {metrics['snr']:.2f} dB"
            }

        return {
            'acceptable': True,
            'metrics': metrics
        }
```

---

### 4. Addressing Critical Processing Concerns

#### Concern 1: False Positives in Quality Screening

**Problem**: Quick SNR checks and artifact scans may incorrectly reject good data (false positives), leading to data loss.

**Solution: Conservative, Multi-Level Quality Assessment with No Rejection**

```python
class ConservativeQualityAssessor:
    """
    Quality assessment that SCORES but NEVER REJECTS data.
    All quality metrics are advisory only.
    """

    def __init__(self):
        self.quality_levels = {
            'excellent': (0.9, 1.0),
            'good': (0.7, 0.9),
            'fair': (0.5, 0.7),
            'poor': (0.3, 0.5),
            'very_poor': (0.0, 0.3)
        }

    def assess_segment(self, segment, fs):
        """
        Assess quality but NEVER reject.
        Returns quality score and confidence level.
        """
        # Multiple independent quality checks
        checks = {
            'snr': self._snr_check(segment),
            'baseline': self._baseline_check(segment),
            'completeness': self._completeness_check(segment),
            'entropy': self._entropy_check(segment),
            'stationarity': self._stationarity_check(segment)
        }

        # Aggregate with confidence weighting
        quality_score = self._aggregate_scores(checks)
        confidence = self._compute_confidence(checks)

        return {
            'quality_score': quality_score,  # 0.0 - 1.0
            'confidence': confidence,         # 0.0 - 1.0
            'level': self._get_quality_level(quality_score),
            'individual_checks': checks,
            'recommendation': self._get_recommendation(quality_score),
            'false_positive_risk': self._estimate_false_positive_risk(checks)
        }

    def _estimate_false_positive_risk(self, checks):
        """
        Estimate likelihood this is a false positive.
        High disagreement between checks = high false positive risk.
        """
        scores = [c['score'] for c in checks.values()]
        disagreement = np.std(scores)

        if disagreement > 0.3:
            return {
                'risk_level': 'high',
                'recommendation': 'Keep data - quality checks disagree',
                'reason': f'Standard deviation of quality checks: {disagreement:.2f}'
            }
        elif disagreement > 0.15:
            return {
                'risk_level': 'medium',
                'recommendation': 'Process with caution',
                'reason': 'Moderate disagreement between quality metrics'
            }
        else:
            return {
                'risk_level': 'low',
                'recommendation': 'Quality assessment reliable',
                'reason': 'Quality metrics agree'
            }

    def _get_recommendation(self, score):
        """
        Advisory recommendations based on score.
        User makes final decision.
        """
        if score > 0.9:
            return "Excellent quality - safe for all analyses"
        elif score > 0.7:
            return "Good quality - suitable for most analyses"
        elif score > 0.5:
            return "Fair quality - usable with preprocessing"
        elif score > 0.3:
            return "Poor quality - consider visual inspection"
        else:
            return "Very poor quality - manual review recommended"
```

**Key Principles**:
1. **Never Reject**: All quality scores are advisory only
2. **Multiple Metrics**: Use 5+ independent quality checks to reduce false positive risk
3. **Confidence Scoring**: Report confidence level alongside quality score
4. **Disagreement Detection**: Flag segments where quality metrics disagree (high false positive risk)
5. **User Decision**: User always has final say on data inclusion

**Practical Implementation**:
```python
# Process ALL segments, tag with quality
results = []
for segment in segments:
    quality = assessor.assess_segment(segment, fs)

    # Process segment regardless of quality
    features_raw = extract_features(segment)

    # Attach quality metadata
    results.append({
        'segment': segment,
        'features': features_raw,
        'quality': quality,
        'timestamp': segment_timestamp,
        'processed': True  # Always processed, never skipped
    })

# User can filter afterwards based on quality + false positive risk
reliable_results = [r for r in results
                    if r['quality']['false_positive_risk']['risk_level'] == 'low']
```

---

#### Concern 2: Preprocessing and Filtering May Distort Data

**Problem**: Preprocessing (detrending, normalization) and filtering may distort data worse than the original noise, leading to worse feature extraction results.

**Solution: Parallel Path Processing with Distortion Detection**

```python
class ParallelPathProcessor:
    """
    Process data through MULTIPLE paths in parallel:
    - Path A: Raw data (no preprocessing)
    - Path B: Preprocessing only (detrending, normalization)
    - Path C: Filtering only (bandpass, notch)
    - Path D: Full pipeline (preprocessing + filtering)

    Compare results and detect distortion.
    """

    def __init__(self):
        self.distortion_detector = DistortionDetector()

    def process_all_paths(self, segment, fs, signal_type):
        """
        Process segment through all paths simultaneously.
        Return ALL results for comparison.
        """
        results = {}

        # PATH A: Raw (baseline)
        results['raw'] = {
            'signal': segment.copy(),
            'features': self.extract_features(segment, fs),
            'quality': self.assess_quality(segment, fs),
            'processing': 'none'
        }

        # PATH B: Preprocessing only
        preprocessed = self.preprocess(segment, fs)
        results['preprocessed'] = {
            'signal': preprocessed,
            'features': self.extract_features(preprocessed, fs),
            'quality': self.assess_quality(preprocessed, fs),
            'processing': 'preprocessing',
            'distortion': self.distortion_detector.compare(segment, preprocessed)
        }

        # PATH C: Filtering only
        filtered = self.filter_signal(segment, fs, signal_type)
        results['filtered'] = {
            'signal': filtered,
            'features': self.extract_features(filtered, fs),
            'quality': self.assess_quality(filtered, fs),
            'processing': 'filtering',
            'distortion': self.distortion_detector.compare(segment, filtered)
        }

        # PATH D: Full pipeline
        full_processed = self.filter_signal(preprocessed, fs, signal_type)
        results['full'] = {
            'signal': full_processed,
            'features': self.extract_features(full_processed, fs),
            'quality': self.assess_quality(full_processed, fs),
            'processing': 'preprocessing+filtering',
            'distortion': self.distortion_detector.compare(segment, full_processed)
        }

        # ANALYSIS: Compare all paths
        comparison = self.compare_all_paths(results)

        return {
            'paths': results,
            'comparison': comparison,
            'recommendation': comparison['best_path'],
            'distortion_report': comparison['distortion_summary']
        }

    def compare_all_paths(self, results):
        """
        Compare processing paths to find best one.
        """
        comparison = {
            'quality_improvement': {},
            'feature_stability': {},
            'distortion_levels': {},
            'best_path': None,
            'distortion_summary': {}
        }

        # Compare each path to raw baseline
        raw_quality = results['raw']['quality']['quality_score']
        raw_features = results['raw']['features']

        for path_name, path_result in results.items():
            if path_name == 'raw':
                continue

            # Quality improvement?
            quality_delta = (path_result['quality']['quality_score'] - raw_quality)
            comparison['quality_improvement'][path_name] = quality_delta

            # Feature stability (do features make sense?)
            feature_delta = self._compare_features(raw_features, path_result['features'])
            comparison['feature_stability'][path_name] = feature_delta

            # Distortion level
            if 'distortion' in path_result:
                comparison['distortion_levels'][path_name] = path_result['distortion']['severity']

        # Determine best path
        comparison['best_path'] = self._select_best_path(comparison)

        # Distortion warnings
        comparison['distortion_summary'] = self._summarize_distortions(comparison)

        return comparison

    def _select_best_path(self, comparison):
        """
        Select best processing path based on:
        1. Quality improvement
        2. Low distortion
        3. Feature stability
        """
        scores = {}

        for path in comparison['quality_improvement'].keys():
            # Score = quality improvement - distortion penalty
            quality_score = comparison['quality_improvement'][path]
            distortion_penalty = comparison['distortion_levels'][path] * 0.5
            feature_score = comparison['feature_stability'][path]

            scores[path] = quality_score - distortion_penalty + feature_score

        # Best path has highest score
        best_path = max(scores.items(), key=lambda x: x[1])

        return {
            'path': best_path[0],
            'score': best_path[1],
            'reason': self._explain_selection(best_path[0], comparison),
            'alternatives': sorted(scores.items(), key=lambda x: x[1], reverse=True)
        }

    def _explain_selection(self, path, comparison):
        """Generate human-readable explanation."""
        quality_delta = comparison['quality_improvement'][path]
        distortion = comparison['distortion_levels'][path]

        if path == 'raw':
            return "Raw data had best quality - no preprocessing needed"
        elif distortion < 0.1 and quality_delta > 0.2:
            return f"{path} improved quality by {quality_delta:.1%} with minimal distortion"
        elif distortion > 0.3:
            return f"{path} has high distortion ({distortion:.1%}) - use with caution"
        else:
            return f"{path} provided best balance of quality and distortion"


class DistortionDetector:
    """
    Detect if preprocessing/filtering distorted the signal.
    """

    def compare(self, original, processed):
        """
        Compare original vs processed signal.
        Detect various types of distortion.
        """
        distortions = {
            'peak_distortion': self._check_peak_distortion(original, processed),
            'morphology_distortion': self._check_morphology(original, processed),
            'timing_distortion': self._check_timing_shift(original, processed),
            'amplitude_distortion': self._check_amplitude_change(original, processed),
            'frequency_distortion': self._check_frequency_content(original, processed)
        }

        # Overall distortion severity
        severity = np.mean([d['severity'] for d in distortions.values()])

        return {
            'severity': severity,  # 0.0 - 1.0
            'level': 'low' if severity < 0.2 else 'medium' if severity < 0.5 else 'high',
            'details': distortions,
            'recommendation': self._distortion_recommendation(severity, distortions)
        }

    def _check_peak_distortion(self, original, processed):
        """Check if peaks were preserved."""
        from scipy.signal import find_peaks

        orig_peaks, _ = find_peaks(original)
        proc_peaks, _ = find_peaks(processed)

        # Compare peak count and locations
        peak_count_ratio = len(proc_peaks) / max(len(orig_peaks), 1)

        if abs(peak_count_ratio - 1.0) > 0.2:
            return {
                'severity': abs(peak_count_ratio - 1.0),
                'warning': f'Peak count changed by {(peak_count_ratio-1)*100:.1f}%'
            }
        return {'severity': 0.0}

    def _check_morphology(self, original, processed):
        """Check if signal shape/morphology preserved."""
        # Normalize both signals
        orig_norm = (original - np.mean(original)) / np.std(original)
        proc_norm = (processed - np.mean(processed)) / np.std(processed)

        # Cross-correlation
        correlation = np.corrcoef(orig_norm, proc_norm)[0, 1]

        # High correlation = low morphology distortion
        distortion = 1.0 - abs(correlation)

        return {
            'severity': distortion,
            'correlation': correlation,
            'warning': 'Morphology significantly changed' if distortion > 0.3 else None
        }

    def _distortion_recommendation(self, severity, distortions):
        """Recommend action based on distortion level."""
        if severity < 0.1:
            return "Minimal distortion - safe to use processed signal"
        elif severity < 0.3:
            return "Moderate distortion - compare features with raw signal"
        else:
            warnings = [d['warning'] for d in distortions.values() if d.get('warning')]
            return f"HIGH DISTORTION - Consider using raw signal. Issues: {', '.join(warnings)}"
```

**Key Features**:
1. **Parallel Processing**: Process raw + 3 variations simultaneously
2. **Distortion Detection**: Quantify how much processing changed the signal
3. **Automatic Selection**: Recommend best path based on quality vs distortion tradeoff
4. **Full Transparency**: User sees all 4 results and can choose manually
5. **Morphology Preservation**: Check that signal shape isn't destroyed

---

#### Concern 3: Processing Granularity - Segment vs Whole Signal

**Problem**: Should we process by segments or whole signal? How do we handle outputs?

**Solution: Flexible Multi-Mode Processing Engine**

```python
class FlexibleProcessingEngine:
    """
    Support multiple processing modes:
    1. Whole signal processing (for short recordings)
    2. Segment-based processing (for long recordings)
    3. Hybrid processing (segment + global)
    """

    def __init__(self, signal, fs, signal_type):
        self.signal = signal
        self.fs = fs
        self.signal_type = signal_type
        self.duration = len(signal) / fs

        # Automatically determine best mode
        self.processing_mode = self._determine_processing_mode()

    def _determine_processing_mode(self):
        """
        Automatically select processing mode based on:
        - Signal duration
        - Available memory
        - Signal type (ECG/PPG/EEG)
        """
        duration_minutes = self.duration / 60

        if duration_minutes < 5:
            return {
                'mode': 'whole_signal',
                'reason': 'Short recording - process as single unit',
                'segment_size': None
            }
        elif duration_minutes < 60:
            return {
                'mode': 'segment_with_overlap',
                'reason': 'Medium recording - process in overlapping segments',
                'segment_size': self._optimal_segment_size(),
                'overlap': 0.2  # 20% overlap
            }
        else:
            return {
                'mode': 'hybrid',
                'reason': 'Long recording - segment processing + global features',
                'segment_size': self._optimal_segment_size(),
                'overlap': 0.1  # 10% overlap for long recordings
            }

    def _optimal_segment_size(self):
        """
        Determine optimal segment size based on signal type.

        Principles:
        - Segments must be long enough for meaningful analysis
        - Must contain multiple cycles of physiological signal
        - Not so long that quality varies significantly within segment
        """
        if self.signal_type == 'ECG':
            # ECG: 30-60 seconds typical (contains 30-100 heartbeats at normal HR)
            return 30 * self.fs
        elif self.signal_type == 'PPG':
            # PPG: 30-60 seconds
            return 30 * self.fs
        elif self.signal_type == 'EEG':
            # EEG: 10-30 seconds (contains multiple brain wave cycles)
            return 20 * self.fs
        elif self.signal_type == 'RESP':
            # Respiratory: 60-120 seconds (contains 5-20 breath cycles)
            return 60 * self.fs
        else:
            # Default: 30 seconds
            return 30 * self.fs

    def process(self):
        """
        Execute processing based on selected mode.
        """
        mode = self.processing_mode['mode']

        if mode == 'whole_signal':
            return self._process_whole_signal()
        elif mode == 'segment_with_overlap':
            return self._process_segments()
        elif mode == 'hybrid':
            return self._process_hybrid()

    def _process_whole_signal(self):
        """Process entire signal as one unit."""
        # Run all processing paths
        paths_processor = ParallelPathProcessor()
        all_paths = paths_processor.process_all_paths(
            self.signal, self.fs, self.signal_type
        )

        return {
            'mode': 'whole_signal',
            'duration': self.duration,
            'all_processing_paths': all_paths,
            'recommended_signal': all_paths['recommendation']['path'],
            'output_signal': all_paths['paths'][all_paths['recommendation']['path']]['signal'],
            'features': all_paths['paths'][all_paths['recommendation']['path']]['features'],
            'quality': all_paths['paths'][all_paths['recommendation']['path']]['quality']
        }

    def _process_segments(self):
        """Process in segments with overlap."""
        segment_size = self.processing_mode['segment_size']
        overlap_ratio = self.processing_mode['overlap']
        overlap_size = int(segment_size * overlap_ratio)
        hop_size = segment_size - overlap_size

        segment_results = []
        paths_processor = ParallelPathProcessor()

        # Process each segment
        for start_idx in range(0, len(self.signal) - segment_size + 1, hop_size):
            end_idx = start_idx + segment_size
            segment = self.signal[start_idx:end_idx]

            # Timestamp for this segment
            timestamp_start = start_idx / self.fs
            timestamp_end = end_idx / self.fs

            # Process all paths for this segment
            segment_result = paths_processor.process_all_paths(
                segment, self.fs, self.signal_type
            )

            segment_results.append({
                'segment_id': len(segment_results),
                'time_range': (timestamp_start, timestamp_end),
                'sample_range': (start_idx, end_idx),
                'duration': segment_size / self.fs,
                'all_paths': segment_result,
                'quality': segment_result['paths']['raw']['quality'],
                'recommended_path': segment_result['recommendation']['path']
            })

        # Aggregate segment results
        aggregated = self._aggregate_segments(segment_results)

        return {
            'mode': 'segment_with_overlap',
            'total_duration': self.duration,
            'segment_count': len(segment_results),
            'segment_duration': segment_size / self.fs,
            'overlap': overlap_ratio,
            'segments': segment_results,
            'aggregated_features': aggregated['features'],
            'quality_distribution': aggregated['quality_distribution'],
            'best_segments': aggregated['best_segments'],
            'poor_segments': aggregated['poor_segments'],
            'output_options': self._generate_output_options(segment_results)
        }

    def _aggregate_segments(self, segment_results):
        """
        Aggregate results from all segments.
        Quality-weighted aggregation.
        """
        # Extract features and quality from all segments
        all_features = []
        all_qualities = []

        for seg in segment_results:
            path = seg['recommended_path']
            features = seg['all_paths']['paths'][path]['features']
            quality = seg['quality']['quality_score']

            all_features.append(features)
            all_qualities.append(quality)

        # Quality-weighted average
        weights = np.array(all_qualities)
        weights = weights / np.sum(weights)  # Normalize

        aggregated_features = {}
        for key in all_features[0].keys():
            values = [f[key] for f in all_features]
            aggregated_features[key] = np.average(values, weights=weights)

        # Identify best and worst segments
        sorted_segments = sorted(enumerate(all_qualities), key=lambda x: x[1], reverse=True)

        return {
            'features': aggregated_features,
            'quality_distribution': {
                'mean': np.mean(all_qualities),
                'std': np.std(all_qualities),
                'min': np.min(all_qualities),
                'max': np.max(all_qualities)
            },
            'best_segments': [segment_results[i] for i, _ in sorted_segments[:5]],  # Top 5
            'poor_segments': [segment_results[i] for i, q in sorted_segments if q < 0.5]
        }

    def _generate_output_options(self, segment_results):
        """
        Generate multiple output options for user to choose from.
        """
        return {
            'option_1_all_segments': {
                'description': 'Export all segments with timestamps',
                'output_type': 'segmented_signal',
                'segment_count': len(segment_results),
                'includes_quality': True
            },
            'option_2_best_quality': {
                'description': 'Export only high-quality segments (quality > 0.7)',
                'output_type': 'filtered_segments',
                'segment_count': sum(1 for seg in segment_results
                                    if seg['quality']['quality_score'] > 0.7)
            },
            'option_3_concatenated': {
                'description': 'Concatenate best segments into continuous signal',
                'output_type': 'continuous_signal',
                'note': 'Timestamps preserved, gaps where poor quality removed'
            },
            'option_4_features_only': {
                'description': 'Export aggregated features (per-segment + global)',
                'output_type': 'features_database',
                'format': 'CSV/JSON with timestamps'
            },
            'option_5_time_range': {
                'description': 'Export specific time range (user selects)',
                'output_type': 'time_range_selection',
                'interactive': True
            },
            'option_6_whole_signal': {
                'description': 'Export entire signal (all segments merged)',
                'output_type': 'complete_signal',
                'file_size_estimate': self._estimate_file_size(len(segment_results))
            }
        }

    def _process_hybrid(self):
        """
        Hybrid mode: Process segments + extract global features.
        Best for very long recordings.
        """
        # First process in segments
        segment_results = self._process_segments()

        # Then extract global features from entire signal
        global_features = self._extract_global_features(self.signal)

        return {
            'mode': 'hybrid',
            'segment_results': segment_results,
            'global_features': global_features,
            'recommendation': {
                'use_segment_features_for': ['HRV', 'local patterns', 'event detection'],
                'use_global_features_for': ['overall trends', 'long-term variability', 'circadian patterns']
            }
        }

    def _extract_global_features(self, signal):
        """
        Extract features that require the entire signal.
        Examples: long-term trends, circadian patterns, overall statistics.
        """
        return {
            'global_mean': np.mean(signal),
            'global_std': np.std(signal),
            'overall_trend': self._compute_trend(signal),
            'long_term_variability': self._compute_long_term_variability(signal),
            'total_duration': len(signal) / self.fs,
            'data_completeness': self._compute_completeness(signal)
        }
```

**Key Features**:
1. **Automatic Mode Selection**: Chooses whole/segment/hybrid based on duration
2. **Signal-Type Aware**: Optimal segment sizes for ECG/PPG/EEG/RESP
3. **Overlapping Segments**: Prevents edge effects
4. **Multiple Output Options**: 6 different output formats for user to choose
5. **Quality-Based Filtering**: User can export only good segments
6. **Timestamp Preservation**: All segments tagged with time ranges
7. **Hybrid Mode**: Combines segment analysis with global features for long recordings

---

### 5. Parallel Processing Framework

```python
class ParallelPipeline:
    """
    Distributes processing across multiple CPU cores.
    Uses multiprocessing for CPU-bound tasks, threading for I/O.
    """

    def __init__(self, n_workers='auto'):
        if n_workers == 'auto':
            self.n_workers = os.cpu_count() - 1  # Leave one core free
        else:
            self.n_workers = n_workers

    def process_large_file(self, file_path, pipeline_func):
        """
        Process large file in parallel chunks

        Architecture:
        1. Main thread: Read chunks from disk
        2. Worker pool: Process chunks in parallel
        3. Writer thread: Aggregate and save results
        """

        # Setup queues
        input_queue = mp.Queue(maxsize=self.n_workers * 2)
        output_queue = mp.Queue()

        # Start worker processes
        workers = []
        for _ in range(self.n_workers):
            p = mp.Process(
                target=self._worker,
                args=(input_queue, output_queue, pipeline_func)
            )
            p.start()
            workers.append(p)

        # Start reader thread
        reader = threading.Thread(
            target=self._reader,
            args=(file_path, input_queue)
        )
        reader.start()

        # Start writer thread
        results = []
        writer = threading.Thread(
            target=self._writer,
            args=(output_queue, results)
        )
        writer.start()

        # Wait for completion
        reader.join()
        for w in workers:
            w.join()
        writer.join()

        return results

    def _worker(self, input_q, output_q, func):
        """Worker process that applies function to chunks"""
        while True:
            chunk = input_q.get()
            if chunk is None:  # Poison pill
                break

            result = func(chunk)
            output_q.put(result)

    def _reader(self, file_path, queue):
        """Read chunks and feed to workers"""
        loader = ChunkedDataLoader(file_path)
        for chunk in loader.load_chunks():
            queue.put(chunk)

        # Send poison pills
        for _ in range(self.n_workers):
            queue.put(None)

    def _writer(self, queue, results):
        """Collect results from workers"""
        processed_count = 0
        while True:
            result = queue.get()
            if result is None:
                break

            results.append(result)
            processed_count += 1
```

---

### 5. Caching and Intermediate Storage

```python
class ProcessingCache:
    """
    Intelligent caching system for intermediate results.
    Prevents re-computation of expensive operations.
    """

    def __init__(self, cache_dir='~/.vitaldsp/cache'):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, data, operation, params):
        """Generate unique cache key"""
        data_hash = hashlib.md5(data.tobytes()).hexdigest()
        params_hash = hashlib.md5(str(params).encode()).hexdigest()
        return f"{operation}_{data_hash}_{params_hash}"

    def get(self, key):
        """Retrieve cached result"""
        cache_file = self.cache_dir / f"{key}.npz"
        if cache_file.exists():
            return np.load(cache_file)
        return None

    def set(self, key, result):
        """Cache result"""
        cache_file = self.cache_dir / f"{key}.npz"
        np.savez_compressed(cache_file, **result)

    def cached_operation(self, operation_func):
        """Decorator for cacheable operations"""
        def wrapper(data, **params):
            key = self.get_cache_key(data, operation_func.__name__, params)

            # Check cache
            cached = self.get(key)
            if cached is not None:
                return cached

            # Compute and cache
            result = operation_func(data, **params)
            self.set(key, result)
            return result

        return wrapper
```

---

### 6. Progressive Processing with Checkpoints

```python
class CheckpointedPipeline:
    """
    Save processing state at checkpoints to allow resumption.
    Critical for very long processing jobs.
    """

    def __init__(self, checkpoint_dir='~/.vitaldsp/checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def process_with_checkpoints(self, data_loader, pipeline_stages):
        """
        Process data through multiple stages with checkpointing.

        Stages:
        1. Load → Checkpoint
        2. Quality screen → Checkpoint
        3. Filter → Checkpoint
        4. Extract features → Checkpoint
        5. Aggregate → Final result
        """

        session_id = self._create_session_id()

        for stage_idx, stage in enumerate(pipeline_stages):
            checkpoint_file = self._get_checkpoint_file(session_id, stage_idx)

            # Try to load from checkpoint
            if checkpoint_file.exists():
                print(f"Resuming from checkpoint: Stage {stage_idx}")
                data = self._load_checkpoint(checkpoint_file)
            else:
                # Process stage
                print(f"Processing stage {stage_idx}: {stage.__name__}")
                data = stage(data)

                # Save checkpoint
                self._save_checkpoint(checkpoint_file, data)

        return data

    def _create_session_id(self):
        """Create unique session identifier"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"session_{timestamp}"

    def _get_checkpoint_file(self, session_id, stage_idx):
        """Get checkpoint file path"""
        return self.checkpoint_dir / f"{session_id}_stage_{stage_idx}.pkl"

    def _save_checkpoint(self, file_path, data):
        """Save checkpoint"""
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def _load_checkpoint(self, file_path):
        """Load checkpoint"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
```

---

## Memory Management Strategies

### 1. Adaptive Memory Allocation

```python
class MemoryManager:
    """
    Monitors and manages memory usage during processing.
    Automatically adjusts chunk sizes and processing parameters.
    """

    def __init__(self, max_memory_percent=80):
        self.max_memory = max_memory_percent

    def get_available_memory(self):
        """Get available system memory"""
        mem = psutil.virtual_memory()
        return mem.available

    def estimate_memory_requirement(self, file_size, operations):
        """
        Estimate memory needed for processing pipeline.

        Factors:
        - File size
        - Number of operations
        - Each operation's memory multiplier
        """
        multipliers = {
            'load': 2.0,           # Loading into memory
            'filter': 1.5,         # Filtering creates temp arrays
            'fft': 3.0,            # FFT needs extra workspace
            'features': 0.5,       # Feature extraction minimal
            'quality': 1.0         # Quality assessment moderate
        }

        total_multiplier = sum(multipliers.get(op, 1.0) for op in operations)
        return file_size * total_multiplier

    def can_process_in_memory(self, file_size, operations):
        """Check if entire dataset can fit in memory"""
        required = self.estimate_memory_requirement(file_size, operations)
        available = self.get_available_memory()

        return required < available * (self.max_memory / 100)

    def recommend_chunk_size(self, file_size, operations):
        """Recommend optimal chunk size"""
        if self.can_process_in_memory(file_size, operations):
            return file_size  # Process all at once

        available = self.get_available_memory()
        multiplier = sum(multipliers.get(op, 1.0) for op in operations)

        # Use 50% of available memory per chunk
        chunk_size = int((available * 0.5) / multiplier)
        return chunk_size
```

### 2. Data Type Optimization

```python
class DataTypeOptimizer:
    """
    Optimize data types to reduce memory footprint.
    Convert float64 to float32 where precision loss is acceptable.
    """

    @staticmethod
    def optimize_signal(signal, target_precision='float32'):
        """
        Convert signal to lower precision.

        Trade-offs:
        - float64: 8 bytes, full precision
        - float32: 4 bytes, sufficient for most signals
        - float16: 2 bytes, lossy but OK for visualization
        """
        if signal.dtype == np.float64 and target_precision == 'float32':
            return signal.astype(np.float32)
        return signal

    @staticmethod
    def compress_features(features):
        """
        Compress feature dictionary.
        Store only necessary precision for each feature.
        """
        compressed = {}
        for key, value in features.items():
            if isinstance(value, float):
                # Most features don't need float64 precision
                compressed[key] = np.float32(value)
            elif isinstance(value, np.ndarray):
                compressed[key] = value.astype(np.float32)
            else:
                compressed[key] = value
        return compressed
```

---

## Quality-Aware Processing

### Quality Metrics for Screening

```python
class QualityScreener:
    """
    Multi-stage quality screening:
    1. Quick screen (< 1 ms per segment)
    2. Medium screen (< 10 ms per segment)
    3. Full assessment (< 100 ms per segment)
    """

    def quick_screen(self, segment):
        """
        Lightning-fast quality check.
        Decision: process or discard immediately.

        Checks:
        - NaN/Inf presence (instant)
        - Amplitude range (instant)
        - Zero variance (instant)
        """
        # Check for invalid values
        if np.isnan(segment).any() or np.isinf(segment).any():
            return {'pass': False, 'reason': 'invalid_values'}

        # Check variance
        if np.var(segment) < 1e-10:
            return {'pass': False, 'reason': 'flat_signal'}

        # Check amplitude range
        amplitude_range = np.ptp(segment)
        if amplitude_range < 0.01:
            return {'pass': False, 'reason': 'low_amplitude'}

        return {'pass': True}

    def medium_screen(self, segment, fs):
        """
        Medium-speed quality check.
        Decision: worth full processing?

        Checks:
        - Approximate SNR
        - Clipping detection
        - Artifact density
        """
        # Estimate SNR using simple methods
        signal_power = np.mean(segment ** 2)
        noise_power = np.var(np.diff(segment))  # High-freq content
        snr_estimate = 10 * np.log10(signal_power / noise_power)

        if snr_estimate < 5.0:  # Very noisy
            return {'pass': False, 'reason': 'low_snr', 'snr': snr_estimate}

        # Check for clipping (saturated signal)
        max_val = np.max(np.abs(segment))
        clipped_points = np.sum(np.abs(segment) > 0.95 * max_val)
        clip_ratio = clipped_points / len(segment)

        if clip_ratio > 0.05:  # More than 5% clipped
            return {'pass': False, 'reason': 'clipping', 'clip_ratio': clip_ratio}

        return {'pass': True, 'snr': snr_estimate}

    def full_assessment(self, segment, fs):
        """
        Complete quality assessment using vitalDSP modules.
        Only called on segments that passed quick + medium screens.
        """
        from vitalDSP.signal_quality_assessment import SignalQualityIndex

        sqi = SignalQualityIndex(segment, fs)

        metrics = {
            'snr': sqi.snr_sqi(),
            'baseline': sqi.baseline_wander_sqi(),
            'entropy': sqi.signal_entropy_sqi(),
            'energy': sqi.energy_sqi()
        }

        # Aggregate quality score (0-1 scale)
        quality_score = np.mean(list(metrics.values()))

        return {
            'pass': quality_score > 0.6,
            'score': quality_score,
            'metrics': metrics
        }
```

### Quality-Weighted Aggregation

```python
def quality_weighted_features(segments_with_quality):
    """
    Aggregate features from multiple segments,
    weighted by their quality scores.

    High-quality segments contribute more to final features.
    """
    weighted_features = {}
    total_weight = 0

    for segment_data in segments_with_quality:
        features = segment_data['features']
        quality = segment_data['quality_score']

        # Weight is quality score
        weight = quality
        total_weight += weight

        # Weighted accumulation
        for key, value in features.items():
            if key not in weighted_features:
                weighted_features[key] = 0
            weighted_features[key] += value * weight

    # Normalize by total weight
    for key in weighted_features:
        weighted_features[key] /= total_weight

    return weighted_features
```

---

## Webapp Integration

### 1. Asynchronous Processing

```python
class WebappAsyncProcessor:
    """
    Non-blocking processing for webapp.
    Uses Celery or similar task queue.
    """

    def __init__(self):
        self.task_queue = TaskQueue()
        self.results_cache = ResultsCache()

    def submit_processing_job(self, file_path, pipeline_config):
        """
        Submit processing job, return immediately with job ID.
        User can check status and retrieve results later.
        """
        job_id = str(uuid.uuid4())

        # Create job metadata
        job_meta = {
            'id': job_id,
            'status': 'queued',
            'progress': 0,
            'created_at': datetime.now(),
            'file_path': file_path,
            'config': pipeline_config
        }

        # Submit to task queue
        self.task_queue.enqueue(
            process_large_file,
            args=(file_path, pipeline_config, job_id)
        )

        return job_id

    def get_job_status(self, job_id):
        """Check job status and progress"""
        return self.results_cache.get_status(job_id)

    def get_job_results(self, job_id):
        """Retrieve completed job results"""
        return self.results_cache.get_results(job_id)
```

### 2. Progressive Results Display

```python
class ProgressiveResultsHandler:
    """
    Stream results to webapp as they become available.
    Show partial results instead of waiting for completion.
    """

    def __init__(self, websocket_connection):
        self.ws = websocket_connection

    def on_chunk_processed(self, chunk_results):
        """Called after each chunk is processed"""
        # Send partial results to webapp
        self.ws.send({
            'type': 'partial_results',
            'data': chunk_results,
            'progress': self.get_progress()
        })

    def on_processing_complete(self, final_results):
        """Called when all processing complete"""
        self.ws.send({
            'type': 'complete',
            'data': final_results
        })
```

### 3. Adaptive Downsampling for Visualization

```python
class AdaptiveVisualizer:
    """
    Intelligently downsample large datasets for browser display.
    Preserve important features while reducing data points.
    """

    def prepare_for_display(self, signal, max_points=10000):
        """
        Reduce signal to max_points while preserving shape.

        Strategy:
        - If signal < max_points: return as-is
        - If signal > max_points: adaptive downsampling
        """
        if len(signal) <= max_points:
            return signal

        # Calculate decimation factor
        decimate_factor = len(signal) // max_points

        # Use LTTB (Largest Triangle Three Buckets) algorithm
        # Preserves peaks and important features
        return self.lttb_downsample(signal, max_points)

    def lttb_downsample(self, data, threshold):
        """
        Largest Triangle Three Buckets downsampling.
        Preserves visual appearance better than simple decimation.
        """
        # Implementation of LTTB algorithm
        # (Preserves peaks, valleys, and trends)
        sampled = [data[0]]  # Always include first point

        bucket_size = (len(data) - 2) / (threshold - 2)

        a = 0
        for i in range(threshold - 2):
            avg_x = int((i + 1) * bucket_size) + 1
            avg_y = np.mean(data[int(i * bucket_size):int((i + 1) * bucket_size)])

            # Find point in current bucket with largest triangle area
            max_area = -1
            max_area_point = 0

            for j in range(int(i * bucket_size), int((i + 1) * bucket_size)):
                area = abs((data[a] - avg_y) * (j - avg_x) -
                          (a - avg_x) * (data[j] - avg_y))
                if area > max_area:
                    max_area = area
                    max_area_point = j

            sampled.append(data[max_area_point])
            a = max_area_point

        sampled.append(data[-1])  # Always include last point
        return np.array(sampled)
```

---

## Adaptive Configuration & Parameter Management

### Philosophy: Zero Hardcoded Values

**Critical Design Principle**: ALL thresholds, limits, and parameters must be:
1. **User-Configurable**: Via settings file, webapp UI, or API
2. **Adaptively Learned**: System learns optimal values from signal characteristics
3. **Context-Aware**: Different defaults for different signal types and use cases
4. **Override-Friendly**: Users can always override automated choices
5. **Traceable**: Every parameter change logged with reasoning

### Three-Tier Configuration System

```
┌─────────────────────────────────────────────────────────────┐
│            Configuration Hierarchy (3-Tier System)           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  TIER 1: Factory Defaults (Built-in)                        │
│  ┌────────────────────────────────────────────────────┐     │
│  │ - Signal-type specific defaults (ECG/PPG/EEG/RESP) │     │
│  │ - Conservative, safe values                        │     │
│  │ - Research-backed thresholds                       │     │
│  └────────────────────────────────────────────────────┘     │
│                            │                                 │
│                            ▼                                 │
│  TIER 2: User Preferences (Persistent)                      │
│  ┌────────────────────────────────────────────────────┐     │
│  │ - User-saved profiles                              │     │
│  │ - Webapp configurations                            │     │
│  │ - Per-project settings                             │     │
│  │ - Overrides factory defaults                       │     │
│  └────────────────────────────────────────────────────┘     │
│                            │                                 │
│                            ▼                                 │
│  TIER 3: Adaptive Runtime (Dynamic)                         │
│  ┌────────────────────────────────────────────────────┐     │
│  │ - Signal-analyzed parameters                       │     │
│  │ - Performance-optimized values                     │     │
│  │ - Context-specific adjustments                     │     │
│  │ - Machine learning recommendations                 │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### Core Configuration Classes

#### 1. Central Configuration Manager

```python
class ConfigurationManager:
    """
    Central configuration management system.
    Manages 3-tier configuration hierarchy with validation and persistence.
    """

    def __init__(self, config_file=None):
        self.factory_defaults = self._load_factory_defaults()
        self.user_preferences = self._load_user_preferences(config_file)
        self.runtime_config = {}
        self.config_history = []  # Track all changes

    def _load_factory_defaults(self):
        """
        Tier 1: Factory defaults for all parameters.
        NEVER hardcode in classes - always load from config.
        """
        return {
            # Quality Assessment Thresholds
            'quality_thresholds': {
                'ecg': {
                    'snr_min': 10.0,
                    'snr_recommended': 15.0,
                    'artifact_max_percent': 5.0,
                    'completeness_min_percent': 95.0,
                    'baseline_wander_max': 0.3,
                    'quality_score_min': 0.6,
                    'quality_score_good': 0.8,
                    'quality_score_excellent': 0.9
                },
                'ppg': {
                    'snr_min': 8.0,
                    'snr_recommended': 12.0,
                    'artifact_max_percent': 8.0,
                    'completeness_min_percent': 90.0,
                    'baseline_wander_max': 0.4,
                    'quality_score_min': 0.5,
                    'quality_score_good': 0.7,
                    'quality_score_excellent': 0.85
                },
                'eeg': {
                    'snr_min': 5.0,
                    'snr_recommended': 10.0,
                    'artifact_max_percent': 10.0,
                    'completeness_min_percent': 85.0,
                    'baseline_wander_max': 0.5,
                    'quality_score_min': 0.4,
                    'quality_score_good': 0.6,
                    'quality_score_excellent': 0.8
                },
                'resp': {
                    'snr_min': 6.0,
                    'snr_recommended': 10.0,
                    'artifact_max_percent': 12.0,
                    'completeness_min_percent': 80.0,
                    'baseline_wander_max': 0.6,
                    'quality_score_min': 0.4,
                    'quality_score_good': 0.6,
                    'quality_score_excellent': 0.75
                }
            },

            # Segmentation Parameters
            'segmentation': {
                'ecg': {
                    'segment_duration_sec': 30,
                    'segment_duration_min_sec': 10,
                    'segment_duration_max_sec': 120,
                    'overlap_ratio': 0.2,
                    'min_cycles_per_segment': 10,  # Minimum heartbeats
                    'quality_based_boundaries': True
                },
                'ppg': {
                    'segment_duration_sec': 30,
                    'segment_duration_min_sec': 10,
                    'segment_duration_max_sec': 120,
                    'overlap_ratio': 0.2,
                    'min_cycles_per_segment': 10,
                    'quality_based_boundaries': True
                },
                'eeg': {
                    'segment_duration_sec': 20,
                    'segment_duration_min_sec': 5,
                    'segment_duration_max_sec': 60,
                    'overlap_ratio': 0.25,
                    'min_cycles_per_segment': 5,  # Minimum brain wave cycles
                    'quality_based_boundaries': True
                },
                'resp': {
                    'segment_duration_sec': 60,
                    'segment_duration_min_sec': 20,
                    'segment_duration_max_sec': 300,
                    'overlap_ratio': 0.15,
                    'min_cycles_per_segment': 5,  # Minimum breaths
                    'quality_based_boundaries': True
                }
            },

            # Processing Mode Selection
            'processing_modes': {
                'short_duration_threshold_min': 5,
                'medium_duration_threshold_min': 60,
                'long_duration_threshold_min': 1440,  # 24 hours
                'whole_signal_memory_limit_mb': 500,
                'chunk_processing_memory_limit_mb': 2000
            },

            # Distortion Detection Thresholds
            'distortion_thresholds': {
                'peak_count_tolerance': 0.2,  # 20% change acceptable
                'morphology_correlation_min': 0.7,
                'timing_shift_max_samples': 5,
                'amplitude_change_max_percent': 30.0,
                'frequency_coherence_min': 0.6,
                'distortion_low_threshold': 0.2,
                'distortion_medium_threshold': 0.5
            },

            # Memory Management
            'memory': {
                'max_memory_percent': 80,
                'chunk_memory_percent': 10,
                'processing_memory_percent': 50,
                'cache_memory_percent': 20,
                'file_size_thresholds_mb': {
                    'small': 100,
                    'medium': 2000,
                    'large': 10000
                },
                'loading_strategies': {
                    'small': 'standard',
                    'medium': 'chunked',
                    'large': 'memory_mapped'
                }
            },

            # SQI Metrics Selection
            'sqi_metrics': {
                'ecg': ['snr', 'baseline_wander', 'signal_to_noise', 'kurtosis', 'skewness'],
                'ppg': ['snr', 'baseline_wander', 'perfusion', 'signal_quality'],
                'eeg': ['snr', 'entropy', 'spectral_edge', 'artifact_ratio'],
                'resp': ['snr', 'baseline_wander', 'regularity', 'amplitude_variation'],
                'quick_check_metrics': ['snr'],  # Fast screening
                'full_check_metrics': 'all'  # Comprehensive assessment
            },

            # Visualization Parameters
            'visualization': {
                'max_points_browser': 10000,
                'max_points_high_res': 50000,
                'downsample_algorithm': 'lttb',  # Largest Triangle Three Buckets
                'preserve_peaks': True,
                'preserve_valleys': True
            },

            # False Positive Mitigation
            'false_positive_mitigation': {
                'confidence_thresholds': {
                    'high': 0.9,
                    'medium': 0.75,
                    'low': 0.5
                },
                'disagreement_thresholds': {
                    'high_risk': 0.3,
                    'medium_risk': 0.15,
                    'low_risk': 0.05
                },
                'min_metrics_agreement': 3,  # Out of 5 metrics
                'manual_review_confidence_threshold': 0.6
            },

            # Parallel Processing
            'parallel_processing': {
                'n_workers': 'auto',  # Will use os.cpu_count() - 1
                'worker_memory_limit_mb': 1000,
                'queue_size_multiplier': 2,
                'timeout_seconds': 300
            },

            # Caching
            'caching': {
                'cache_enabled': True,
                'cache_dir': '~/.vitaldsp/cache',
                'max_cache_size_gb': 10,
                'cache_ttl_hours': 24,
                'cache_compression': True
            }
        }

    def _load_user_preferences(self, config_file):
        """
        Tier 2: Load user-saved preferences from file.
        """
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def get(self, key_path, signal_type=None, context=None):
        """
        Get configuration value with tier priority:
        Tier 3 (runtime) > Tier 2 (user) > Tier 1 (factory)

        Args:
            key_path: Dot-notation path (e.g., 'quality_thresholds.ecg.snr_min')
            signal_type: Optional signal type for context-specific config
            context: Optional dict with additional context

        Returns:
            Configuration value
        """
        # First check runtime config (Tier 3)
        value = self._get_from_dict(self.runtime_config, key_path)
        if value is not None:
            self._log_config_access(key_path, value, 'runtime')
            return value

        # Then check user preferences (Tier 2)
        value = self._get_from_dict(self.user_preferences, key_path)
        if value is not None:
            self._log_config_access(key_path, value, 'user')
            return value

        # Finally fall back to factory defaults (Tier 1)
        value = self._get_from_dict(self.factory_defaults, key_path)
        if value is not None:
            self._log_config_access(key_path, value, 'factory')
            return value

        raise KeyError(f"Configuration key not found: {key_path}")

    def get_signal_config(self, signal_type, category):
        """
        Get all configuration for specific signal type and category.

        Example:
            config.get_signal_config('ecg', 'quality_thresholds')
            # Returns all ECG quality thresholds
        """
        key_path = f"{category}.{signal_type}"
        return self.get(key_path)

    def set_runtime(self, key_path, value, reason=None):
        """
        Set runtime configuration (Tier 3).
        Used for adaptive parameter adjustments.
        """
        self._set_in_dict(self.runtime_config, key_path, value)
        self._log_config_change(key_path, value, 'runtime', reason)

    def set_user_preference(self, key_path, value, save=True):
        """
        Set user preference (Tier 2).
        Optionally save to persistent storage.
        """
        self._set_in_dict(self.user_preferences, key_path, value)
        self._log_config_change(key_path, value, 'user', 'user_override')

        if save:
            self._save_user_preferences()

    def _get_from_dict(self, d, key_path):
        """Navigate nested dict using dot notation."""
        keys = key_path.split('.')
        value = d
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def _set_in_dict(self, d, key_path, value):
        """Set value in nested dict using dot notation."""
        keys = key_path.split('.')
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def _log_config_access(self, key, value, tier):
        """Log configuration access for debugging."""
        logger.debug(f"Config access: {key} = {value} (tier: {tier})")

    def _log_config_change(self, key, value, tier, reason):
        """Log configuration changes for traceability."""
        self.config_history.append({
            'timestamp': datetime.now(),
            'key': key,
            'value': value,
            'tier': tier,
            'reason': reason
        })
        logger.info(f"Config changed: {key} = {value} (tier: {tier}, reason: {reason})")

    def _save_user_preferences(self):
        """Persist user preferences to file."""
        config_file = Path('~/.vitaldsp/user_config.yaml').expanduser()
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(self.user_preferences, f)

    def export_current_config(self, output_file):
        """Export complete merged configuration for reproducibility."""
        merged_config = {}
        # Merge all tiers (factory < user < runtime)
        self._deep_merge(merged_config, self.factory_defaults)
        self._deep_merge(merged_config, self.user_preferences)
        self._deep_merge(merged_config, self.runtime_config)

        with open(output_file, 'w') as f:
            yaml.dump({
                'merged_configuration': merged_config,
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'config_history': self.config_history
                }
            }, f)

    def _deep_merge(self, target, source):
        """Deep merge source dict into target dict."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
```

---

#### 2. Adaptive Parameter Optimizer

```python
class AdaptiveParameterOptimizer:
    """
    Automatically learn and adjust parameters based on signal characteristics.
    Tier 3 (Runtime) configuration intelligence.
    """

    def __init__(self, config_manager):
        self.config = config_manager
        self.signal_history = []  # Track processed signals
        self.performance_metrics = {}

    def analyze_and_optimize(self, signal, fs, signal_type, metadata=None):
        """
        Analyze signal characteristics and recommend optimal parameters.

        Returns:
            dict: Optimized parameters for this specific signal
        """
        analysis = self._analyze_signal_characteristics(signal, fs, signal_type)
        optimized_params = {}

        # Optimize quality thresholds based on signal characteristics
        optimized_params['quality_thresholds'] = self._optimize_quality_thresholds(
            signal, analysis, signal_type
        )

        # Optimize segmentation parameters
        optimized_params['segmentation'] = self._optimize_segmentation_params(
            signal, fs, analysis, signal_type
        )

        # Optimize processing mode
        optimized_params['processing_mode'] = self._optimize_processing_mode(
            signal, fs, analysis
        )

        # Update runtime configuration
        for category, params in optimized_params.items():
            for key, value in params.items():
                key_path = f"{category}.{signal_type}.{key}"
                self.config.set_runtime(
                    key_path,
                    value,
                    reason=f"adaptive_optimization_based_on_{analysis['optimization_reason']}"
                )

        return optimized_params

    def _analyze_signal_characteristics(self, signal, fs, signal_type):
        """
        Comprehensive signal analysis for parameter optimization.
        """
        return {
            # Basic statistics
            'mean': np.mean(signal),
            'std': np.std(signal),
            'snr_estimate': self._estimate_snr(signal),
            'duration_minutes': len(signal) / fs / 60,

            # Quality indicators
            'noise_level': self._estimate_noise_level(signal),
            'artifact_density': self._estimate_artifact_density(signal),
            'baseline_stability': self._estimate_baseline_stability(signal),

            # Cyclicity (for periodic signals)
            'dominant_frequency': self._estimate_dominant_frequency(signal, fs),
            'frequency_stability': self._estimate_frequency_stability(signal, fs),

            # Data completeness
            'missing_ratio': np.sum(np.isnan(signal)) / len(signal),
            'clipping_ratio': self._estimate_clipping_ratio(signal),

            # Complexity
            'entropy': self._calculate_entropy(signal),
            'stationarity': self._assess_stationarity(signal),

            # Optimization reasoning
            'optimization_reason': self._determine_optimization_reason(signal, signal_type)
        }

    def _optimize_quality_thresholds(self, signal, analysis, signal_type):
        """
        Adaptively set quality thresholds based on signal characteristics.
        """
        factory_defaults = self.config.get_signal_config(signal_type, 'quality_thresholds')
        optimized = factory_defaults.copy()

        # If signal has high baseline noise, relax SNR threshold
        if analysis['noise_level'] > 0.3:
            optimized['snr_min'] = max(
                factory_defaults['snr_min'] - 2.0,  # Relax by 2 dB
                5.0  # But never below 5 dB
            )
            logger.info(f"Relaxed SNR threshold to {optimized['snr_min']} due to high baseline noise")

        # If signal shows many artifacts, adjust artifact threshold
        if analysis['artifact_density'] > 0.15:
            optimized['artifact_max_percent'] = min(
                factory_defaults['artifact_max_percent'] + 5.0,
                20.0  # But never above 20%
            )
            logger.info(f"Increased artifact tolerance to {optimized['artifact_max_percent']}%")

        # If signal has gaps, adjust completeness requirement
        if analysis['missing_ratio'] > 0.05:
            optimized['completeness_min_percent'] = max(
                factory_defaults['completeness_min_percent'] - 10.0,
                75.0  # But never below 75%
            )
            logger.info(f"Adjusted completeness requirement to {optimized['completeness_min_percent']}%")

        return optimized

    def _optimize_segmentation_params(self, signal, fs, analysis, signal_type):
        """
        Adaptively set segmentation parameters based on signal characteristics.
        """
        factory_defaults = self.config.get_signal_config(signal_type, 'segmentation')
        optimized = factory_defaults.copy()

        # Adjust segment duration based on frequency stability
        if analysis['frequency_stability'] < 0.7:  # Unstable frequency
            # Use shorter segments for unstable signals
            optimized['segment_duration_sec'] = max(
                factory_defaults['segment_duration_sec'] * 0.5,
                factory_defaults['segment_duration_min_sec']
            )
            logger.info(f"Reduced segment duration to {optimized['segment_duration_sec']}s due to frequency instability")

        elif analysis['frequency_stability'] > 0.9:  # Very stable
            # Use longer segments for stable signals
            optimized['segment_duration_sec'] = min(
                factory_defaults['segment_duration_sec'] * 1.5,
                factory_defaults['segment_duration_max_sec']
            )
            logger.info(f"Increased segment duration to {optimized['segment_duration_sec']}s due to high stability")

        # Adjust overlap based on quality variation
        if analysis['stationarity'] < 0.6:  # Non-stationary signal
            # Increase overlap for non-stationary signals
            optimized['overlap_ratio'] = min(
                factory_defaults['overlap_ratio'] + 0.1,
                0.5  # Maximum 50% overlap
            )
            logger.info(f"Increased overlap to {optimized['overlap_ratio']*100}% for non-stationary signal")

        return optimized

    def _optimize_processing_mode(self, signal, fs, analysis):
        """
        Determine optimal processing mode based on signal characteristics.
        """
        duration_min = analysis['duration_minutes']
        available_memory = psutil.virtual_memory().available / (1024**2)  # MB
        signal_size_mb = signal.nbytes / (1024**2)

        # Calculate processing overhead estimate
        processing_overhead = 3.0  # Typical multiplier for processing
        required_memory_mb = signal_size_mb * processing_overhead

        if required_memory_mb > available_memory * 0.8:
            mode = 'chunked'
            reason = f'insufficient_memory_available_{available_memory:.0f}MB_needed_{required_memory_mb:.0f}MB'
        elif duration_min < 5:
            mode = 'whole_signal'
            reason = 'short_duration_efficient_whole_signal_processing'
        elif duration_min < 60:
            mode = 'segment_with_overlap'
            reason = 'medium_duration_optimal_for_segmentation'
        else:
            mode = 'hybrid'
            reason = 'long_duration_hybrid_approach_recommended'

        logger.info(f"Selected processing mode: {mode} (reason: {reason})")

        return {
            'mode': mode,
            'reason': reason,
            'estimated_memory_mb': required_memory_mb
        }

    def _estimate_snr(self, signal):
        """Quick SNR estimation."""
        signal_power = np.var(signal)
        noise_power = np.var(np.diff(signal)) / 2  # Approximation
        return 10 * np.log10(signal_power / max(noise_power, 1e-10))

    def _estimate_noise_level(self, signal):
        """Estimate noise level (0-1 scale)."""
        high_freq_energy = np.var(np.diff(signal))
        total_energy = np.var(signal)
        return min(high_freq_energy / max(total_energy, 1e-10), 1.0)

    def _estimate_artifact_density(self, signal):
        """Estimate proportion of signal affected by artifacts."""
        # Simple heuristic: count points beyond 3 standard deviations
        threshold = 3 * np.std(signal)
        outliers = np.sum(np.abs(signal - np.mean(signal)) > threshold)
        return outliers / len(signal)

    def _estimate_baseline_stability(self, signal, window_size=1000):
        """Assess baseline stability using moving average."""
        if len(signal) < window_size:
            return 1.0

        baseline = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
        baseline_variation = np.std(baseline)
        signal_range = np.ptp(signal)

        return 1.0 - min(baseline_variation / max(signal_range, 1e-10), 1.0)

    def _estimate_dominant_frequency(self, signal, fs):
        """Estimate dominant frequency using FFT."""
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1/fs)
        dominant_idx = np.argmax(np.abs(fft[1:]))  # Skip DC component
        return freqs[dominant_idx + 1]

    def _estimate_frequency_stability(self, signal, fs, window_size=10000):
        """Assess frequency stability over time."""
        if len(signal) < window_size * 2:
            return 0.5

        freqs = []
        for i in range(0, len(signal) - window_size, window_size // 2):
            segment = signal[i:i+window_size]
            freqs.append(self._estimate_dominant_frequency(segment, fs))

        freq_std = np.std(freqs)
        freq_mean = np.mean(freqs)

        return 1.0 - min(freq_std / max(freq_mean, 1e-10), 1.0)

    def _estimate_clipping_ratio(self, signal):
        """Estimate proportion of clipped samples."""
        max_val = np.max(np.abs(signal))
        clipped = np.sum(np.abs(signal) > 0.95 * max_val)
        return clipped / len(signal)

    def _calculate_entropy(self, signal, bins=50):
        """Calculate signal entropy."""
        hist, _ = np.histogram(signal, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zeros
        return -np.sum(hist * np.log2(hist))

    def _assess_stationarity(self, signal, n_windows=10):
        """Assess signal stationarity using window-based statistics."""
        if len(signal) < n_windows * 100:
            return 0.5

        window_size = len(signal) // n_windows
        stats = []

        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            window = signal[start:end]
            stats.append({
                'mean': np.mean(window),
                'std': np.std(window)
            })

        # Check consistency of statistics across windows
        means = [s['mean'] for s in stats]
        stds = [s['std'] for s in stats]

        mean_stability = 1.0 - (np.std(means) / max(np.mean(np.abs(means)), 1e-10))
        std_stability = 1.0 - (np.std(stds) / max(np.mean(stds), 1e-10))

        return (mean_stability + std_stability) / 2

    def _determine_optimization_reason(self, signal, signal_type):
        """Generate human-readable optimization reasoning."""
        reasons = []

        if np.std(signal) < 0.1:
            reasons.append('low_signal_variation')

        noise_level = self._estimate_noise_level(signal)
        if noise_level > 0.3:
            reasons.append('high_noise_level')

        artifact_density = self._estimate_artifact_density(signal)
        if artifact_density > 0.1:
            reasons.append('high_artifact_density')

        baseline_stability = self._estimate_baseline_stability(signal)
        if baseline_stability < 0.7:
            reasons.append('unstable_baseline')

        return '_'.join(reasons) if reasons else 'signal_characteristics'
```

---

#### 3. Webapp Configuration UI Integration

```python
class WebappConfigurationUI:
    """
    Webapp-friendly configuration interface.
    Provides user-friendly controls for all parameters.
    """

    def __init__(self, config_manager):
        self.config = config_manager

    def get_configuration_form(self, signal_type):
        """
        Generate Dash components for configuration UI.
        Returns interactive form for webapp.
        """
        return dbc.Card([
            dbc.CardHeader(f"{signal_type.upper()} Configuration Settings"),
            dbc.CardBody([
                # Quality Thresholds Section
                self._create_quality_section(signal_type),
                html.Hr(),

                # Segmentation Parameters Section
                self._create_segmentation_section(signal_type),
                html.Hr(),

                # SQI Metrics Selection
                self._create_sqi_metrics_section(signal_type),
                html.Hr(),

                # Processing Options
                self._create_processing_options_section(),
                html.Hr(),

                # Buttons
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Reset to Factory Defaults",
                                  id=f"btn-reset-{signal_type}",
                                  color="secondary", outline=True),
                    ], width=4),
                    dbc.Col([
                        dbc.Button("Apply Adaptive Optimization",
                                  id=f"btn-optimize-{signal_type}",
                                  color="info", outline=True),
                    ], width=4),
                    dbc.Col([
                        dbc.Button("Save Configuration",
                                  id=f"btn-save-{signal_type}",
                                  color="primary"),
                    ], width=4),
                ])
            ])
        ])

    def _create_quality_section(self, signal_type):
        """Create quality thresholds configuration UI."""
        thresholds = self.config.get_signal_config(signal_type, 'quality_thresholds')

        return html.Div([
            html.H5("Quality Assessment Thresholds"),
            dbc.Row([
                dbc.Col([
                    html.Label("Minimum SNR (dB)"),
                    dbc.Input(
                        id=f"input-snr-min-{signal_type}",
                        type="number",
                        value=thresholds['snr_min'],
                        min=0, max=30, step=0.5
                    ),
                    dbc.FormText("Lower = more permissive, Higher = more strict")
                ], width=6),

                dbc.Col([
                    html.Label("Maximum Artifact (%)"),
                    dbc.Input(
                        id=f"input-artifact-max-{signal_type}",
                        type="number",
                        value=thresholds['artifact_max_percent'],
                        min=0, max=50, step=1
                    ),
                    dbc.FormText("Maximum acceptable artifact percentage")
                ], width=6),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    html.Label("Minimum Completeness (%)"),
                    dbc.Input(
                        id=f"input-completeness-min-{signal_type}",
                        type="number",
                        value=thresholds['completeness_min_percent'],
                        min=50, max=100, step=5
                    ),
                    dbc.FormText("Minimum data completeness required")
                ], width=6),

                dbc.Col([
                    html.Label("Quality Score Threshold"),
                    dbc.Input(
                        id=f"input-quality-min-{signal_type}",
                        type="number",
                        value=thresholds['quality_score_min'],
                        min=0, max=1, step=0.05
                    ),
                    dbc.FormText("Minimum overall quality score (0-1)")
                ], width=6),
            ])
        ])

    def _create_segmentation_section(self, signal_type):
        """Create segmentation parameters configuration UI."""
        seg_params = self.config.get_signal_config(signal_type, 'segmentation')

        return html.Div([
            html.H5("Segmentation Parameters"),
            dbc.Row([
                dbc.Col([
                    html.Label("Segment Duration (seconds)"),
                    dbc.Input(
                        id=f"input-segment-duration-{signal_type}",
                        type="number",
                        value=seg_params['segment_duration_sec'],
                        min=seg_params['segment_duration_min_sec'],
                        max=seg_params['segment_duration_max_sec'],
                        step=5
                    ),
                    dbc.FormText(f"Range: {seg_params['segment_duration_min_sec']}-{seg_params['segment_duration_max_sec']} seconds")
                ], width=6),

                dbc.Col([
                    html.Label("Overlap Ratio"),
                    dbc.Input(
                        id=f"input-overlap-ratio-{signal_type}",
                        type="number",
                        value=seg_params['overlap_ratio'],
                        min=0, max=0.5, step=0.05
                    ),
                    dbc.FormText("Overlap between segments (0-0.5)")
                ], width=6),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Checklist(
                        id=f"check-quality-boundaries-{signal_type}",
                        options=[
                            {"label": " Use quality-based segment boundaries",
                             "value": "quality_based"}
                        ],
                        value=["quality_based"] if seg_params['quality_based_boundaries'] else [],
                        switch=True
                    ),
                    dbc.FormText("Adjust boundaries based on quality transitions")
                ], width=12),
            ])
        ])

    def _create_sqi_metrics_section(self, signal_type):
        """Create SQI metrics selection UI."""
        available_metrics = self.config.get(f"sqi_metrics.{signal_type}")

        return html.Div([
            html.H5("Signal Quality Metrics"),
            dbc.Row([
                dbc.Col([
                    html.Label("Select metrics for quality assessment:"),
                    dbc.Checklist(
                        id=f"checklist-sqi-metrics-{signal_type}",
                        options=[
                            {"label": f" {metric.replace('_', ' ').title()}", "value": metric}
                            for metric in available_metrics
                        ],
                        value=available_metrics,  # All selected by default
                        inline=False
                    ),
                    dbc.FormText("More metrics = more accurate but slower")
                ], width=12),
            ])
        ])

    def _create_processing_options_section(self):
        """Create general processing options UI."""
        return html.Div([
            html.H5("Processing Options"),
            dbc.Row([
                dbc.Col([
                    html.Label("Processing Mode"),
                    dbc.RadioItems(
                        id="radio-processing-mode",
                        options=[
                            {"label": " Auto-select (Recommended)", "value": "auto"},
                            {"label": " Whole Signal", "value": "whole_signal"},
                            {"label": " Segment-based", "value": "segment"},
                            {"label": " Hybrid", "value": "hybrid"},
                        ],
                        value="auto",
                        inline=False
                    ),
                ], width=6),

                dbc.Col([
                    html.Label("Distortion Tolerance"),
                    dbc.RadioItems(
                        id="radio-distortion-tolerance",
                        options=[
                            {"label": " Conservative (Reject high distortion)", "value": "conservative"},
                            {"label": " Moderate", "value": "moderate"},
                            {"label": " Permissive (Allow more distortion)", "value": "permissive"},
                        ],
                        value="moderate",
                        inline=False
                    ),
                ], width=6),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Checklist(
                        id="check-adaptive-optimization",
                        options=[
                            {"label": " Enable adaptive parameter optimization",
                             "value": "adaptive"}
                        ],
                        value=["adaptive"],
                        switch=True
                    ),
                    dbc.FormText("Automatically adjust parameters based on signal characteristics")
                ], width=12),
            ])
        ])

    def create_configuration_callbacks(self, app, signal_type):
        """
        Register Dash callbacks for configuration UI.
        """
        @app.callback(
            Output(f"config-store-{signal_type}", "data"),
            [Input(f"btn-save-{signal_type}", "n_clicks")],
            [State(f"input-snr-min-{signal_type}", "value"),
             State(f"input-artifact-max-{signal_type}", "value"),
             State(f"input-completeness-min-{signal_type}", "value"),
             State(f"input-quality-min-{signal_type}", "value"),
             State(f"input-segment-duration-{signal_type}", "value"),
             State(f"input-overlap-ratio-{signal_type}", "value"),
             State(f"checklist-sqi-metrics-{signal_type}", "value")],
            prevent_initial_call=True
        )
        def save_configuration(n_clicks, snr_min, artifact_max, completeness_min,
                             quality_min, segment_duration, overlap_ratio, sqi_metrics):
            """Save user configuration to Tier 2 (user preferences)."""
            if n_clicks:
                # Update quality thresholds
                self.config.set_user_preference(
                    f"quality_thresholds.{signal_type}.snr_min",
                    snr_min,
                    save=False
                )
                self.config.set_user_preference(
                    f"quality_thresholds.{signal_type}.artifact_max_percent",
                    artifact_max,
                    save=False
                )
                self.config.set_user_preference(
                    f"quality_thresholds.{signal_type}.completeness_min_percent",
                    completeness_min,
                    save=False
                )
                self.config.set_user_preference(
                    f"quality_thresholds.{signal_type}.quality_score_min",
                    quality_min,
                    save=False
                )

                # Update segmentation parameters
                self.config.set_user_preference(
                    f"segmentation.{signal_type}.segment_duration_sec",
                    segment_duration,
                    save=False
                )
                self.config.set_user_preference(
                    f"segmentation.{signal_type}.overlap_ratio",
                    overlap_ratio,
                    save=False
                )

                # Update SQI metrics
                self.config.set_user_preference(
                    f"sqi_metrics.{signal_type}",
                    sqi_metrics,
                    save=True  # Save all at once
                )

                return {
                    'status': 'success',
                    'message': f'Configuration saved for {signal_type.upper()}',
                    'timestamp': datetime.now().isoformat()
                }

            raise PreventUpdate
```

---

### Configuration Usage Examples

#### Example 1: Basic Usage with Defaults

```python
# Initialize with factory defaults
config = ConfigurationManager()

# Get quality thresholds for ECG
ecg_thresholds = config.get_signal_config('ecg', 'quality_thresholds')
print(f"ECG SNR minimum: {ecg_thresholds['snr_min']} dB")

# Use in quality assessment
assessor = ConservativeQualityAssessor()
quality = assessor.assess_segment(
    segment,
    fs,
    snr_threshold=config.get('quality_thresholds.ecg.snr_min')
)
```

#### Example 2: User Override via Webapp

```python
# User adjusts SNR threshold via webapp
config.set_user_preference(
    'quality_thresholds.ecg.snr_min',
    12.0,  # User wants stricter quality
    save=True  # Persist to file
)

# Next processing uses new threshold
ecg_thresholds = config.get_signal_config('ecg', 'quality_thresholds')
print(f"ECG SNR minimum (user override): {ecg_thresholds['snr_min']} dB")
```

#### Example 3: Adaptive Optimization

```python
# Analyze signal and optimize parameters
optimizer = AdaptiveParameterOptimizer(config)
optimized_params = optimizer.analyze_and_optimize(signal, fs, 'ecg')

# Runtime config now has adaptive values
snr_threshold = config.get('quality_thresholds.ecg.snr_min')
print(f"Adaptive SNR threshold: {snr_threshold} dB")

# Configuration history shows why it changed
for change in config.config_history:
    print(f"{change['timestamp']}: {change['key']} = {change['value']} ({change['reason']})")
```

#### Example 4: Export Configuration for Reproducibility

```python
# Process signal with current configuration
result = process_signal(signal, fs, config)

# Export complete configuration used
config.export_current_config('processing_config_20251012.yaml')

# Later: Load exact same configuration
config_reproduced = ConfigurationManager('processing_config_20251012.yaml')
result_reproduced = process_signal(signal, fs, config_reproduced)
# Results will be identical
```

#### Example 5: Signal-Type Profiles

```python
# Get recommended configuration for different signal types
for signal_type in ['ecg', 'ppg', 'eeg', 'resp']:
    thresholds = config.get_signal_config(signal_type, 'quality_thresholds')
    segmentation = config.get_signal_config(signal_type, 'segmentation')

    print(f"\n{signal_type.upper()} Configuration:")
    print(f"  SNR threshold: {thresholds['snr_min']} dB")
    print(f"  Segment duration: {segmentation['segment_duration_sec']}s")
    print(f"  Overlap: {segmentation['overlap_ratio']*100}%")
```

---

### Benefits of Configuration System

1. **Zero Hardcoded Values**: All thresholds loaded from configuration
2. **User Control**: Webapp UI allows easy customization
3. **Adaptive Intelligence**: System learns optimal parameters
4. **Full Traceability**: Every configuration change logged
5. **Reproducibility**: Export exact configuration used for any processing
6. **Context-Aware**: Different defaults for different signal types
7. **Safe Overrides**: User can override any parameter at any time
8. **Validation**: Configuration manager validates all values
9. **Persistence**: User preferences saved across sessions
10. **Performance**: Configuration access is fast (O(1) dict lookup)

---

### Configuration File Example (YAML)

```yaml
# ~/.vitaldsp/user_config.yaml

# User's custom overrides (Tier 2)
quality_thresholds:
  ecg:
    snr_min: 12.0  # User wants stricter quality
    quality_score_min: 0.7  # Higher threshold

  ppg:
    artifact_max_percent: 10.0  # More permissive

segmentation:
  ecg:
    segment_duration_sec: 45  # Longer segments preferred
    overlap_ratio: 0.25  # More overlap

sqi_metrics:
  ecg:
    - snr
    - baseline_wander
    - signal_to_noise
    # User disabled kurtosis and skewness for speed

processing_modes:
  prefer_mode: 'segment_with_overlap'  # User preference

distortion_thresholds:
  peak_count_tolerance: 0.15  # More conservative

visualization:
  max_points_browser: 15000  # User has powerful machine

# User metadata
user_profile:
  name: "Research Lab A"
  created_at: "2025-10-12T10:30:00"
  last_modified: "2025-10-12T14:45:00"
```

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (2-3 weeks)

**Week 1: Data Loading**
- [ ] Implement `ChunkedDataLoader` with adaptive sizing
- [ ] Implement `MemoryMappedLoader` for large files
- [ ] Add progress callbacks and cancellation support
- [ ] Unit tests for all loaders

**Week 2: Quality Screening**
- [ ] Implement `QualityScreener` with 3-stage screening
- [ ] Integrate with existing quality assessment modules
- [ ] Benchmark screening performance
- [ ] Quality-aware processing tests

**Week 3: Parallel Processing**
- [ ] Implement `ParallelPipeline` with multiprocessing
- [ ] Add worker pool management
- [ ] Implement result aggregation
- [ ] Performance benchmarks

### Phase 2: Pipeline Integration (2-3 weeks)

**Week 4: Processing Pipeline**
- [ ] Build standard 8-stage pipeline
- [ ] Implement checkpointing system
- [ ] Add caching layer
- [ ] Integration tests

**Week 5: Memory Management**
- [ ] Implement `MemoryManager`
- [ ] Add data type optimization
- [ ] Memory profiling tools
- [ ] Optimization benchmarks

**Week 6: Error Handling**
- [ ] Robust error recovery
- [ ] Partial result preservation
- [ ] User-friendly error messages
- [ ] Error logging and reporting

### Phase 3: Webapp Integration (2 weeks)

**Week 7: Async Processing**
- [ ] Set up task queue (Celery/RQ)
- [ ] Implement job management
- [ ] Progress tracking system
- [ ] Status API endpoints

**Week 8: UI/UX**
- [ ] Progressive results display
- [ ] Adaptive visualization
- [ ] File size warnings
- [ ] Processing time estimates

### Phase 4: Optimization & Testing (1-2 weeks)

**Week 9: Performance**
- [ ] Comprehensive benchmarks
- [ ] Profile bottlenecks
- [ ] Optimize hot paths
- [ ] Memory leak detection

**Week 10: Documentation**
- [ ] User guide for large files
- [ ] Developer API documentation
- [ ] Performance tuning guide
- [ ] Example notebooks

---

## Performance Targets

| Operation | Current | Target | Strategy |
|-----------|---------|--------|----------|
| Load 1GB file | 30s | 5s | Memory mapping + chunking |
| Quality screen | N/A | <1s per hour | 3-stage screening |
| Filter 24h ECG | 60s | 10s | Parallel processing |
| Extract features | 120s | 15s | Parallel + caching |
| Full pipeline | 300s | 30s | All optimizations |

---

## Conclusion

This architecture provides:

1. **Scalability**: Handle files from KB to GB seamlessly
2. **Efficiency**: Process only high-quality data
3. **Responsiveness**: Non-blocking webapp operation
4. **Reliability**: Checkpoints and error recovery
5. **Performance**: Parallel processing and caching
6. **User Experience**: Progress indication and partial results

**Next Steps**: Review and approve design, then proceed with Phase 1 implementation.
