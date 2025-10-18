# vitalDSP Pipeline Page - Quick Reference

## 🚀 Quick Start

1. **Upload Data** → Go to Upload page and upload your signal file
2. **Navigate to Pipeline** → Click the pipeline icon (📊) in the sidebar
3. **Configure Settings** → Select signal type and processing paths
4. **Run Pipeline** → Click "Run Pipeline" button
5. **Monitor Progress** → Watch real-time progress and stage details
6. **Export Results** → Download results or generate report

## 🎛️ Configuration Options

### Signal Types
- **ECG** - Electrocardiogram signals
- **PPG** - Photoplethysmography signals  
- **EEG** - Electroencephalography signals
- **Respiratory** - Breathing signals
- **Generic** - General purpose signals

### Processing Paths
- **RAW** - No processing (baseline)
- **FILTERED** - Bandpass filtering only
- **PREPROCESSED** - Filtering + artifact removal

### Quality Settings
- **Enable Quality Screening** - Toggle quality assessment
- **Quality Threshold** - Set minimum quality score (0.0-1.0)

### Segmentation
- **Window Size** - Segment duration in seconds (5-300)
- **Overlap Ratio** - Overlap between segments (0.0-0.9)

### Features
- **Time Domain** - Statistical features
- **Frequency Domain** - Spectral features
- **Nonlinear** - Complexity features

## 📊 Pipeline Stages

| Stage | Name | Purpose | Duration |
|-------|------|---------|----------|
| 1 | Data Ingestion | Load & validate data | ~0.1s |
| 2 | Quality Screening | 3-stage quality check | ~0.2s |
| 3 | Parallel Processing | Multiple processing paths | ~0.5s |
| 4 | Quality Validation | Compare paths | ~0.1s |
| 5 | Segmentation | Create overlapping windows | ~0.1s |
| 6 | Feature Extraction | Extract features | ~1.0s |
| 7 | Intelligent Output | Generate recommendations | ~0.1s |
| 8 | Output Package | Package results | ~0.1s |

## 🔍 Quality Screening

### Stage 1: SNR Assessment
- **Purpose**: Signal-to-noise ratio evaluation
- **Threshold**: Default 10 dB
- **Output**: SNR value + pass/fail

### Stage 2: Statistical Screen  
- **Purpose**: Detect outliers and jumps
- **Metrics**: Outlier ratio, jump ratio
- **Output**: Statistical quality score

### Stage 3: Signal-Specific Screen
- **Purpose**: Domain-specific quality checks
- **Metrics**: Baseline wander, amplitude variability
- **Output**: Signal-specific quality score

## 🎯 Control Buttons

| Button | Function | When Available |
|--------|----------|----------------|
| **Run Pipeline** | Start processing | Always (when data uploaded) |
| **Stop Pipeline** | Halt execution | During processing |
| **Reset Pipeline** | Reset to start | Always |
| **Export Results** | Download results | After completion |
| **Generate Report** | Create PDF report | After completion |

## 📈 Monitoring Panels

### Left Panel - Configuration
- Signal type selection
- Processing path options
- Quality screening settings
- Segmentation parameters
- Feature extraction options

### Center Panel - Pipeline Flow
- 8-stage pipeline diagram
- Real-time progress bar
- Current stage details
- Control buttons

### Right Panel - Results
- Quality screening results
- Processing path comparison
- Feature extraction summary
- Intelligent recommendations

## ⚡ Performance Tips

### Signal Requirements
- **Minimum Length**: 10 seconds
- **Recommended Length**: 30+ seconds
- **Sampling Rate**: 100-1000 Hz
- **File Size**: <100 MB

### Optimization
- Select only necessary processing paths
- Use appropriate window sizes
- Adjust quality thresholds for your signal type
- Enable only required feature types

## 🚨 Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| "No data uploaded" | No signal data | Upload data first |
| Pipeline stops early | Poor signal quality | Check quality thresholds |
| Low quality scores | Signal too noisy | Adjust thresholds or preprocess |
| Missing features | Insufficient signal length | Use longer signals (>30s) |

### Error Messages

- **"Error: No data uploaded"** → Upload signal data first
- **"Quality screening failed"** → Check signal quality and thresholds
- **"Feature extraction failed"** → Ensure sufficient signal length
- **"Pipeline execution failed"** → Check signal format and parameters

## 📋 Best Practices

### Before Running Pipeline
1. ✅ Upload clean, high-quality signal data
2. ✅ Select appropriate signal type
3. ✅ Choose relevant processing paths
4. ✅ Set realistic quality thresholds
5. ✅ Configure appropriate window sizes

### During Processing
1. 👀 Monitor progress and stage details
2. 📊 Check quality screening results
3. 🔍 Review processing path comparisons
4. ⚠️ Watch for warnings or errors

### After Processing
1. 📈 Review all results and recommendations
2. 💾 Export results for further analysis
3. 📄 Generate comprehensive report
4. 🔄 Adjust parameters if needed for re-processing

## 🔗 Related Pages

- **Upload Page** - Upload signal data
- **Quality Page** - Detailed quality analysis
- **Features Page** - Advanced feature extraction
- **Settings Page** - Global configuration

## 📞 Support

- **Documentation**: Full documentation available
- **Inline Help**: Hover over elements for tooltips
- **Error Messages**: Check console for detailed errors
- **Community**: Join vitalDSP community forum

---

*Quick Reference v1.0 - October 17, 2025*
