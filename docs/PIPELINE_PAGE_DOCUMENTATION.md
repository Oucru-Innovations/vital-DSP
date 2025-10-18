# vitalDSP Processing Pipeline Documentation

## Overview

The vitalDSP Processing Pipeline page provides a comprehensive interface for visualizing, configuring, and monitoring the 8-stage signal processing pipeline. This page allows users to understand the complete signal processing workflow and monitor the execution of each stage in real-time.

## Features

### 🎯 **8-Stage Processing Pipeline**
The pipeline implements a conservative processing approach with the following stages:

1. **Data Ingestion** - Loading and validating input data
2. **Quality Screening** - Three-stage quality assessment
3. **Parallel Processing** - Multiple processing paths
4. **Quality Validation** - Comparing processing paths
5. **Segmentation** - Dividing signal into windows
6. **Feature Extraction** - Extracting time/frequency features
7. **Intelligent Output** - Generating recommendations
8. **Output Package** - Packaging results for export

### 🔧 **Configuration Panel**
- **Signal Type Selection**: Choose from ECG, PPG, EEG, Respiratory, or Generic
- **Processing Paths**: Select RAW, FILTERED, or PREPROCESSED paths
- **Quality Screening**: Enable/disable quality assessment with configurable thresholds
- **Segmentation**: Configure window size and overlap ratio
- **Feature Extraction**: Select time-domain, frequency-domain, and nonlinear features

### 📊 **Real-time Monitoring**
- **Progress Tracking**: Visual progress bar with stage-by-stage updates
- **Stage Details**: Detailed information about each processing stage
- **Quality Results**: Three-stage quality screening results
- **Path Comparison**: Visual comparison of different processing paths
- **Feature Summary**: Extracted features with values and units
- **Recommendations**: Intelligent processing recommendations

## User Interface

### Layout Structure
The pipeline page uses a 3-panel layout:

1. **Left Panel - Configuration**
   - Pipeline settings and parameters
   - Signal type and processing path selection
   - Quality screening configuration
   - Segmentation and feature extraction options

2. **Center Panel - Pipeline Visualization**
   - 8-stage pipeline flow diagram
   - Real-time progress tracking
   - Stage details and metrics
   - Control buttons (Run, Stop, Reset)

3. **Right Panel - Results & Monitoring**
   - Quality screening results
   - Processing path comparison charts
   - Feature extraction summary
   - Intelligent recommendations

### Control Buttons
- **Run Pipeline**: Start the processing pipeline
- **Stop Pipeline**: Halt execution at current stage
- **Reset Pipeline**: Reset to initial state
- **Export Results**: Download processing results
- **Generate Report**: Create comprehensive report

## Usage Guide

### Step 1: Upload Data
Before using the pipeline, ensure you have uploaded signal data through the Upload page.

### Step 2: Configure Pipeline
1. **Select Signal Type**: Choose the appropriate signal type (ECG, PPG, etc.)
2. **Choose Processing Paths**: Select which processing paths to use:
   - **RAW**: No filtering applied
   - **FILTERED**: Basic bandpass filtering
   - **PREPROCESSED**: Filtering + artifact removal
3. **Configure Quality Screening**: Enable quality assessment and set thresholds
4. **Set Segmentation Parameters**: Configure window size and overlap
5. **Select Features**: Choose which features to extract

### Step 3: Run Pipeline
1. Click **"Run Pipeline"** to start processing
2. Monitor progress in real-time through the progress bar
3. View stage details and metrics as they become available
4. Observe quality screening results and path comparisons

### Step 4: Review Results
1. **Quality Results**: Review three-stage quality screening outcomes
2. **Path Comparison**: Compare different processing paths visually
3. **Feature Summary**: Examine extracted features and their values
4. **Recommendations**: Review intelligent processing recommendations

### Step 5: Export & Report
1. **Export Results**: Download processing results as JSON/CSV
2. **Generate Report**: Create a comprehensive PDF report

## Pipeline Stages Details

### Stage 1: Data Ingestion
- **Purpose**: Load and validate input data
- **Process**: Detect format, extract metadata, validate structure
- **Output**: Validated signal data with metadata
- **Metrics**: Format detected, rows loaded, sampling rate, duration

### Stage 2: Quality Screening
- **Purpose**: Three-stage quality assessment
- **Process**: 
  - Stage 1: SNR assessment
  - Stage 2: Statistical anomaly detection
  - Stage 3: Signal-specific quality checks
- **Output**: Quality scores and recommendations
- **Metrics**: SNR, outlier ratio, jump ratio, overall quality

### Stage 3: Parallel Processing
- **Purpose**: Process signal through multiple paths
- **Process**: Apply different processing strategies simultaneously
- **Output**: Multiple processed signal versions
- **Metrics**: Active paths, filters applied, artifacts removed

### Stage 4: Quality Validation
- **Purpose**: Compare processing paths and select best
- **Process**: Analyze quality metrics across all paths
- **Output**: Best path selection with confidence scores
- **Metrics**: Quality scores for each path, best path selection

### Stage 5: Segmentation
- **Purpose**: Divide signal into overlapping windows
- **Process**: Create segments with configurable overlap
- **Output**: Signal segments for detailed analysis
- **Metrics**: Window size, overlap ratio, total segments, valid segments

### Stage 6: Feature Extraction
- **Purpose**: Extract comprehensive features from segments
- **Process**: Calculate time-domain, frequency-domain, and nonlinear features
- **Output**: Feature matrix with values and units
- **Metrics**: Number of features by type, total features extracted

### Stage 7: Intelligent Output
- **Purpose**: Generate intelligent processing recommendations
- **Process**: Analyze results and provide actionable insights
- **Output**: Recommendations and confidence scores
- **Metrics**: Best path, confidence level, number of recommendations

### Stage 8: Output Package
- **Purpose**: Package all results for export
- **Process**: Compile results, metadata, and recommendations
- **Output**: Complete processing package
- **Metrics**: Processing time, output size, completion status

## Quality Screening Details

### Three-Stage Quality Assessment

#### Stage 1: SNR Assessment
- **Purpose**: Evaluate signal-to-noise ratio
- **Method**: Fast SNR estimation
- **Threshold**: Configurable (default: 10 dB)
- **Output**: SNR value and pass/fail status

#### Stage 2: Statistical Screen
- **Purpose**: Detect statistical anomalies
- **Method**: Outlier detection and jump analysis
- **Metrics**: Outlier ratio, jump ratio
- **Output**: Statistical quality score

#### Stage 3: Signal-Specific Screen
- **Purpose**: Domain-specific quality assessment
- **Method**: Signal-type-specific quality metrics
- **Metrics**: Baseline wander, amplitude variability, zero-crossing rate
- **Output**: Signal-specific quality score

### Quality Metrics
- **Overall Quality Score**: Combined score from all stages (0-1)
- **Pass/Fail Status**: Binary decision based on thresholds
- **Recommendations**: Processing recommendations based on quality

## Processing Paths

### RAW Path
- **Description**: No processing applied
- **Use Case**: Baseline comparison
- **Quality**: Original signal quality
- **Distortion**: None

### FILTERED Path
- **Description**: Basic bandpass filtering
- **Use Case**: Noise reduction
- **Quality**: Improved SNR
- **Distortion**: Minimal filtering artifacts

### PREPROCESSED Path
- **Description**: Filtering + artifact removal
- **Use Case**: Maximum quality
- **Quality**: Highest quality
- **Distortion**: Some signal modification

## Feature Extraction

### Time-Domain Features
- Mean, Standard Deviation, RMS
- Peak-to-peak amplitude
- Zero-crossing rate
- Signal energy

### Frequency-Domain Features
- Spectral centroid
- Dominant frequency
- Band power ratios
- Spectral entropy

### Nonlinear Features
- Sample entropy
- Detrended fluctuation analysis
- Multiscale entropy
- Fractal dimension

## Troubleshooting

### Common Issues

#### No Data Available
- **Problem**: "Error: No data uploaded"
- **Solution**: Upload signal data through the Upload page first

#### Pipeline Stops Unexpectedly
- **Problem**: Pipeline stops at a specific stage
- **Solution**: Check quality thresholds and signal quality

#### Poor Quality Results
- **Problem**: Low quality scores
- **Solution**: Adjust quality thresholds or check input signal

#### Missing Features
- **Problem**: Some features not extracted
- **Solution**: Ensure sufficient signal length and quality

### Performance Tips

1. **Signal Length**: Use signals longer than 30 seconds for best results
2. **Quality Thresholds**: Adjust thresholds based on signal type
3. **Processing Paths**: Select only necessary paths to improve performance
4. **Segmentation**: Use appropriate window sizes for your analysis

## API Integration

The pipeline page integrates with the vitalDSP backend API:

- **Pipeline Execution**: `/api/pipeline/execute`
- **Progress Monitoring**: `/api/pipeline/progress`
- **Results Export**: `/api/pipeline/export`
- **Report Generation**: `/api/pipeline/report`

## Technical Specifications

### Requirements
- **Signal Length**: Minimum 10 seconds, recommended 30+ seconds
- **Sampling Rate**: 100-1000 Hz (signal-type dependent)
- **Data Format**: CSV, JSON, or NumPy arrays
- **Browser**: Modern browser with JavaScript enabled

### Performance
- **Processing Time**: 2-10 seconds (depending on signal length)
- **Memory Usage**: 50-200 MB (depending on signal size)
- **Concurrent Users**: Supports multiple simultaneous users

### Limitations
- **Maximum Signal Length**: 1 hour (3600 seconds)
- **Maximum Sampling Rate**: 1000 Hz
- **File Size Limit**: 100 MB per upload

## Future Enhancements

### Planned Features
- **Real-time Processing**: Live signal processing
- **Custom Pipelines**: User-defined processing stages
- **Batch Processing**: Multiple signal processing
- **Advanced Visualization**: 3D pipeline visualization
- **Machine Learning**: Intelligent parameter optimization

### Integration Plans
- **Cloud Processing**: Cloud-based pipeline execution
- **Database Integration**: Direct database connectivity
- **API Extensions**: RESTful API for external integration
- **Mobile Support**: Mobile-optimized interface

## Support

For technical support or feature requests:
- **Documentation**: Check this guide and inline help
- **Issues**: Report bugs through the issue tracker
- **Community**: Join the vitalDSP community forum
- **Email**: Contact the development team

---

*Last Updated: October 17, 2025*
*Version: 1.0.0*
