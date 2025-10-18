# Pipeline Page User Guide

## Overview

The **Processing Pipeline** page provides a comprehensive interface for visualizing and monitoring the vitalDSP 8-stage signal processing pipeline. This page allows you to configure, execute, and track multi-stage processing of physiological signals with quality screening, parallel processing paths, and intelligent output recommendations.

## Accessing the Pipeline Page

1. **From the Sidebar**: Click on **"Processing Pipeline"** under the "Pipeline" section (icon: project-diagram)
2. **Direct URL**: Navigate to `/pipeline` in your browser

## Page Layout

The Pipeline page is organized into a **3-panel layout**:

### Left Panel - Pipeline Configuration
Configure pipeline settings before execution:

#### Signal Type Selection
- **ECG**: Electrocardiogram signals
- **PPG**: Photoplethysmogram signals
- **EEG**: Electroencephalogram signals
- **Respiratory**: Respiratory signals
- **Generic**: Generic physiological signals

#### Processing Paths
Choose which processing paths to execute (can select multiple):
- **RAW**: No filtering applied (baseline)
- **FILTERED**: Bandpass filtering only
- **PREPROCESSED**: Filtered + artifact removal (recommended)

#### Quality Screening Settings
- **Enable Quality Screening**: Toggle quality screening on/off
- **Quality Threshold**: Minimum quality score (0.0 - 1.0, default: 0.5)

#### Segmentation Settings
- **Window Size**: Segment duration in seconds (5-300s, default: 30s)
- **Overlap Ratio**: Segment overlap (0.0 - 0.9, default: 0.5)

#### Feature Extraction
Select feature domains to extract:
- **Time Domain**: Mean, std, RMS, etc.
- **Frequency Domain**: Spectral features, dominant frequency
- **Nonlinear**: Entropy, complexity measures

### Right Panel - Visualization and Results

#### 1. Pipeline Progress Indicator
Visual flowchart showing the 8 pipeline stages:
1. **Data Ingestion**: Loading and validating input data
2. **Quality Screening**: 3-stage quality assessment (SNR, Statistical, Signal-specific)
3. **Parallel Processing**: Multi-path signal processing
4. **Quality Validation**: Cross-path quality comparison
5. **Segmentation**: Dividing signal into windows
6. **Feature Extraction**: Extracting features from segments
7. **Intelligent Output**: Generating recommendations
8. **Output Package**: Packaging results for export

#### 2. Overall Progress Bar
Shows percentage completion of pipeline execution.

#### 3. Stage Details Panel
Displays detailed information about the current stage:
- Stage name and description
- Stage-specific metrics
- Processing status

#### 4. Processing Paths Comparison
Interactive plot comparing different processing paths:
- Overlays RAW, FILTERED, and PREPROCESSED signals
- Only shows after Stage 3 (Parallel Processing) completes
- Helps visualize the effect of each processing step

#### 5. Quality Screening Results
Shows results from the 3-stage quality screening:
- **Stage 1 - SNR Assessment**: Signal-to-noise ratio in dB
- **Stage 2 - Statistical Screen**: Outlier ratio, jump ratio
- **Stage 3 - Signal-Specific**: Baseline wander, amplitude variability

#### 6. Feature Extraction Summary
Lists all extracted features with values and units:
- Time-domain features (mean, std, RMS)
- Frequency-domain features (spectral centroid, dominant frequency)
- Nonlinear features (if selected)

#### 7. Intelligent Output Recommendations
AI-generated recommendations including:
- Best processing path (highest quality score)
- Confidence level
- Specific recommendations for downstream analysis

## How to Use the Pipeline

### Step 1: Upload Data
Before using the pipeline, ensure you have uploaded signal data via the **Upload Data** page.

### Step 2: Configure Pipeline
1. Select your **signal type** (e.g., ECG, PPG)
2. Choose **processing paths** to compare (recommended: FILTERED + PREPROCESSED)
3. Enable **quality screening** if you want automatic quality checks
4. Set **quality threshold** (signals below threshold will be flagged)
5. Configure **segmentation** parameters (window size, overlap)
6. Select **feature types** to extract

### Step 3: Run Pipeline
1. Click the **"Run Pipeline"** button
2. Watch the progress indicator advance through stages
3. Monitor the overall progress bar
4. Review stage details as each stage completes

### Step 4: Review Results
As the pipeline progresses, review:
- **Stage Details**: Current stage metrics and status
- **Paths Comparison**: Visual comparison of processing paths (after Stage 3)
- **Quality Results**: Quality screening outcomes (after Stage 2)
- **Features Summary**: Extracted features (after Stage 6)
- **Recommendations**: Intelligent suggestions (after Stage 7)

### Step 5: Control Execution
- **Stop Button**: Pause pipeline at current stage
- **Reset Button**: Clear results and return to initial state

### Step 6: Export Results (After Completion)
Once pipeline completes (100% progress):
- **Export Results**: Download processing results as JSON/CSV
- **Generate Report**: Create PDF report with visualizations and recommendations

## Pipeline Control Buttons

### Run Pipeline (Primary Button)
- **Icon**: Play icon
- **Color**: Blue
- **Function**: Starts pipeline execution
- **Enabled**: Only when data is uploaded and pipeline is not running

### Stop (Danger Button)
- **Icon**: Stop icon
- **Color**: Red
- **Function**: Stops pipeline at current stage
- **Enabled**: Only while pipeline is running

### Reset (Secondary Button)
- **Icon**: Redo icon
- **Color**: Gray
- **Function**: Clears all results and resets to initial state
- **Enabled**: Always available

### Export Results (Success Button)
- **Icon**: Download icon
- **Color**: Green
- **Function**: Downloads pipeline results
- **Enabled**: Only after pipeline completes successfully

### Generate Report (Info Button)
- **Icon**: PDF icon
- **Color**: Blue
- **Function**: Creates comprehensive PDF report
- **Enabled**: Only after pipeline completes successfully

## Understanding the 8 Pipeline Stages

### Stage 1: Data Ingestion
**Purpose**: Load and validate input data
**Outputs**:
- Format detection (CSV, JSON, etc.)
- Sampling rate extraction
- Signal duration calculation
- Metadata extraction

### Stage 2: Quality Screening
**Purpose**: Assess signal quality before processing
**3-Stage Screening**:
1. **SNR Assessment**: Measures signal-to-noise ratio
2. **Statistical Screen**: Detects outliers, jumps, constant segments
3. **Signal-Specific**: ECG/PPG-specific quality metrics

**Outputs**: Overall quality score (0.0 - 1.0)

### Stage 3: Parallel Processing
**Purpose**: Process signal through multiple paths
**Paths**:
- **RAW**: Original signal (no processing)
- **FILTERED**: Butterworth bandpass filter
- **PREPROCESSED**: Filtered + artifact removal

**Outputs**: 3 processed signal versions for comparison

### Stage 4: Quality Validation
**Purpose**: Compare quality across processing paths
**Outputs**:
- Quality scores for each path
- Best path recommendation
- Confidence level

### Stage 5: Segmentation
**Purpose**: Divide signal into analysis windows
**Outputs**:
- Signal segments with overlap
- Valid vs. invalid segment counts
- Segment metadata

### Stage 6: Feature Extraction
**Purpose**: Extract features from each segment
**Feature Types**:
- **Time Domain**: Mean, std, RMS, skewness, kurtosis
- **Frequency Domain**: Spectral centroid, dominant frequency, spectral entropy
- **Nonlinear**: Sample entropy, approximate entropy, DFA

**Outputs**: Feature matrix (segments × features)

### Stage 7: Intelligent Output
**Purpose**: Analyze results and generate recommendations
**Outputs**:
- Best processing path
- Confidence scores
- Actionable recommendations

### Stage 8: Output Package
**Purpose**: Package all results for export
**Outputs**:
- Consolidated results JSON
- Processing metadata
- Quality reports
- Feature matrices

## Interpreting Results

### Quality Scores
- **0.0 - 0.3**: Poor quality (not suitable for analysis)
- **0.3 - 0.5**: Fair quality (use with caution)
- **0.5 - 0.7**: Good quality (suitable for basic analysis)
- **0.7 - 0.9**: Very good quality (suitable for advanced analysis)
- **0.9 - 1.0**: Excellent quality (ideal for all analyses)

### Processing Path Selection
The pipeline automatically recommends the best path based on:
1. **Quality Score**: Higher is better
2. **SNR**: Signal-to-noise ratio
3. **Artifact Removal**: Reduction in noise/artifacts
4. **Feature Stability**: Consistency across segments

**Typical Recommendations**:
- **RAW**: Only when signal is already clean (quality > 0.9)
- **FILTERED**: When signal has moderate noise
- **PREPROCESSED**: When signal has significant artifacts (most common)

## Current Mode: Simulation

The pipeline currently runs in **simulation mode** to demonstrate the interface and workflow. In simulation mode:
- Pipeline advances through stages automatically (0.5s per stage)
- Sample data and metrics are displayed
- All UI features are functional
- No real signal processing occurs

**Note**: Real pipeline integration is being optimized and will be enabled in a future update.

## Troubleshooting

### Pipeline Won't Start
- **Issue**: "No data uploaded" error
- **Solution**: Upload data via the Upload Data page first

### Pipeline Stuck at Stage
- **Issue**: Progress bar not advancing
- **Solution**: Click "Stop" then "Reset", try again with different settings

### Export/Report Buttons Disabled
- **Issue**: Buttons remain grayed out
- **Solution**: Wait for pipeline to reach 100% completion (Stage 8)

### Quality Score Too Low
- **Issue**: Signal quality below threshold
- **Solution**:
  - Lower quality threshold setting
  - Try PREPROCESSED path for better artifact removal
  - Check input data quality

## Best Practices

1. **Always Enable Quality Screening**: Helps identify poor-quality signals early
2. **Use Multiple Paths**: Compare FILTERED and PREPROCESSED to see processing effects
3. **Choose Appropriate Window Size**:
   - ECG/PPG: 30-60 seconds
   - EEG: 60-120 seconds
   - Respiratory: 60-300 seconds
4. **Set Overlap**: Use 50% overlap for better temporal resolution
5. **Select Relevant Features**: Only extract features you need to improve performance
6. **Review Stage Details**: Check each stage's metrics to understand processing

## Future Enhancements

Upcoming features (Phase 3 & 4):
- Real pipeline integration with vitalDSP core
- Background task monitoring page (`/tasks`)
- Persistent pipeline state across sessions
- Batch file processing
- Pipeline templates and presets
- Configuration export/import
- Advanced PDF report generation
- Estimated time remaining (ETA)

## Related Pages

- **Upload Data** (`/upload`): Upload signal files before running pipeline
- **Background Tasks** (`/tasks`): Monitor long-running pipeline executions (coming soon)
- **Settings** (`/settings`): Configure global pipeline defaults

## Support

For issues, questions, or feature requests, please contact the vitalDSP development team or file an issue on the project repository.

---

**Last Updated**: 2025-10-18
**Version**: 1.0 (Simulation Mode)
**Status**: Ready for Use
