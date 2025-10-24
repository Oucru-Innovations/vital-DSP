# VitalDSP User Guide

**Last Updated**: October 23, 2025  
**Version**: 2.0.0  
**Status**: ✅ Production Ready  

---

## 🚀 Getting Started

### **Quick Start**
1. **Install VitalDSP**: `pip install vitalDSP`
2. **Start Web Application**: `python -m vitalDSP_webapp.app`
3. **Open Browser**: Navigate to `http://localhost:8000`
4. **Upload Data**: Upload your physiological signal data
5. **Run Analysis**: Select analysis type and run

### **System Requirements**
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 1GB free space
- **Browser**: Chrome, Firefox, Safari, or Edge

---

## 📊 Data Upload and Management

### **Supported File Formats**
- **CSV**: Comma-separated values
- **Excel**: .xlsx and .xls files
- **JSON**: JavaScript Object Notation
- **Text**: Tab-delimited text files

### **Data Requirements**
- **Signal Data**: Time series data with timestamps
- **Sampling Rate**: Automatically detected or manually specified
- **Signal Types**: ECG, PPG, EMG, EEG, and other physiological signals
- **File Size**: Up to 100MB per file (larger files use memory mapping)

### **Upload Process**
1. **Navigate** to the Upload page
2. **Select File** using the file picker
3. **Configure Parameters**:
   - Sampling frequency
   - Column mapping (time, signal columns)
   - Signal type (ECG, PPG, etc.)
4. **Upload** - Progress bar shows upload status
5. **Preview** - Verify data looks correct

---

## 🔬 Analysis Pages

### **1. Time Domain Analysis**
**Purpose**: Comprehensive time-domain feature extraction

**Features**:
- Heart Rate Variability (HRV) metrics
- Statistical features (mean, std, skewness, kurtosis)
- Waveform morphology analysis
- Beat-to-beat analysis
- Trend analysis

**Usage**:
1. Select time range using slider or input fields
2. Choose analysis parameters
3. Click "Update Analysis"
4. View results in interactive plots

### **2. Frequency Domain Analysis**
**Purpose**: Spectral analysis and frequency-domain features

**Features**:
- FFT (Fast Fourier Transform)
- STFT (Short-Time Fourier Transform)
- Wavelet Transform
- Power Spectral Density
- Frequency band analysis

**Usage**:
1. Select analysis type (FFT, STFT, Wavelet)
2. Configure parameters (window size, overlap, etc.)
3. Click "Update Analysis"
4. View frequency spectrum and features

### **3. Signal Filtering**
**Purpose**: Advanced signal filtering and artifact removal

**Features**:
- Traditional filters (Butterworth, Chebyshev, Elliptic)
- Advanced filters (Kalman, adaptive)
- Artifact removal (baseline wander, motion artifacts)
- Neural network filtering
- Ensemble filtering

**Usage**:
1. Select filter type and parameters
2. Configure frequency cutoffs
3. Click "Apply Filter"
4. Compare original vs filtered signals

### **4. Respiratory Analysis**
**Purpose**: Respiratory rate estimation and analysis

**Features**:
- Multiple RR estimation methods
- Respiratory quality assessment
- Respiratory feature extraction
- Ensemble consensus estimation

**Usage**:
1. Enable preprocessing if needed
2. Select RR estimation methods
3. Configure frequency bands (0.1-0.5 Hz)
4. Click "Analyze"
5. View RR estimates and quality metrics

### **5. Quality Assessment**
**Purpose**: Signal quality evaluation and assessment

**Features**:
- Signal Quality Index (SQI)
- Artifact detection
- Quality-based filtering
- Quality reporting

**Usage**:
1. Select quality assessment methods
2. Configure quality thresholds
3. Click "Analyze"
4. View quality metrics and reports

### **6. Advanced Analysis**
**Purpose**: Complex signal processing and advanced features

**Features**:
- Nonlinear analysis
- Entropy analysis
- Symbolic dynamics
- Transfer entropy
- Cross-signal analysis

**Usage**:
1. Select analysis type
2. Configure parameters
3. Click "Analyze"
4. View advanced features

### **7. Pipeline Processing**
**Purpose**: Multi-stage signal processing pipeline

**Features**:
- 8-stage processing pipeline
- Configurable parameters for each stage
- Real-time progress tracking
- Comprehensive reporting

**Usage**:
1. Configure pipeline parameters
2. Select processing stages
3. Click "Run Pipeline"
4. Monitor progress and view results

### **8. Feature Extraction**
**Purpose**: Comprehensive feature engineering

**Features**:
- Time-domain features
- Frequency-domain features
- Nonlinear features
- Physiological features
- ML-ready feature vectors

**Usage**:
1. Select feature types
2. Configure extraction parameters
3. Click "Extract Features"
4. View feature matrix and statistics

---

## ⚙️ Configuration and Settings

### **Run Modes**

#### **Normal Mode** (Default)
- **Logging Level**: INFO and above
- **Performance**: Optimized for production
- **Log Files**: `webapp.log`
- **Auto-reload**: Disabled
- **Use Case**: Production, monitoring, general use

**Start Command**:
```bash
python src/vitalDSP_webapp/run_webapp_debug.py
```

#### **Debug Mode**
- **Logging Level**: DEBUG and above (all logs)
- **Performance**: Detailed logging may impact performance
- **Log Files**: `webapp_debug.log`
- **Auto-reload**: Enabled (code changes auto-restart server)
- **Use Case**: Development, troubleshooting, detailed analysis

**Start Command**:
```bash
python src/vitalDSP_webapp/run_webapp_debug.py --debug
```

#### **Custom Configuration**
```bash
# Custom port
python src/vitalDSP_webapp/run_webapp_debug.py --port 8080

# Custom host
python src/vitalDSP_webapp/run_webapp_debug.py --host 127.0.0.1

# Debug mode on custom port
python src/vitalDSP_webapp/run_webapp_debug.py --debug --port 8080
```

### **Environment Variables**
```bash
# Set debug mode via environment
export DEBUG=true
python src/vitalDSP_webapp/run_webapp_debug.py

# Set custom port
export PORT=8080
python src/vitalDSP_webapp/run_webapp_debug.py

# Set custom host
export HOST=127.0.0.1
python src/vitalDSP_webapp/run_webapp_debug.py
```

---

## 📈 Performance Optimization

### **Plot Data Limiting**
- **Maximum Duration**: 5 minutes per plot
- **Maximum Points**: 10,000 points per plot
- **Smart Downsampling**: Preserves peaks and valleys
- **Automatic Limiting**: Applied to all analysis pages

### **Data Loading**
- **Chunked Loading**: Large files loaded in chunks
- **Memory Mapping**: Efficient handling of large files
- **Progressive Loading**: Background loading with progress
- **Lazy Loading**: Load data only when needed

### **Performance Tips**
1. **Use Appropriate Time Ranges**: Smaller ranges = faster processing
2. **Enable Plot Limiting**: Prevents browser freezing
3. **Use Debug Mode Sparingly**: Debug mode impacts performance
4. **Monitor Memory Usage**: Large datasets require more memory

---

## 🔧 Troubleshooting

### **Common Issues**

#### **Page Loads Slowly**
- **Cause**: Large datasets or callback loops
- **Solution**: Use plot data limiting, check for callback loops
- **Status**: ✅ Fixed in current version

#### **Browser Freezes**
- **Cause**: Plotting too many data points
- **Solution**: Plot data limiting automatically applied
- **Status**: ✅ Fixed in current version

#### **Analysis Takes Too Long**
- **Cause**: Large time ranges or complex analysis
- **Solution**: Reduce time range, use simpler analysis
- **Status**: ✅ Optimized in current version

#### **Memory Issues**
- **Cause**: Large files loaded into memory
- **Solution**: Use memory mapping for large files
- **Status**: ✅ Optimized in current version

### **Debug Mode Usage**
When troubleshooting issues:

1. **Enable Debug Mode**:
   ```bash
   python src/vitalDSP_webapp/run_webapp_debug.py --debug
   ```

2. **Check Log Files**:
   ```bash
   tail -f webapp_debug.log
   ```

3. **Monitor Performance**:
   - Check CPU usage
   - Monitor memory usage
   - Watch for error messages

4. **Common Debug Commands**:
   ```bash
   # Monitor logs in real-time
   tail -f webapp_debug.log
   
   # Search for specific patterns
   grep "ERROR" webapp_debug.log
   grep "Data stored" webapp_debug.log
   ```

---

## 📊 Export and Results

### **Export Formats**
- **PNG**: High-quality images
- **SVG**: Scalable vector graphics
- **CSV**: Data export
- **JSON**: Structured data export
- **PDF**: Comprehensive reports

### **Export Options**
1. **Individual Plots**: Export specific plots
2. **Complete Analysis**: Export all results
3. **Feature Matrix**: Export feature data
4. **Quality Reports**: Export quality assessments
5. **Pipeline Results**: Export pipeline outputs

### **Results Interpretation**
- **Quality Metrics**: Higher values = better quality
- **Feature Values**: Normalized and standardized
- **Statistical Significance**: P-values and confidence intervals
- **Visual Indicators**: Color-coded results

---

## 🎯 Best Practices

### **Data Preparation**
1. **Clean Data**: Remove obvious artifacts before upload
2. **Consistent Sampling**: Use consistent sampling rates
3. **Proper Formatting**: Ensure proper column headers
4. **Metadata**: Include relevant metadata (patient info, etc.)

### **Analysis Workflow**
1. **Start Simple**: Begin with basic analysis
2. **Check Quality**: Always assess signal quality first
3. **Iterate**: Refine parameters based on results
4. **Validate**: Cross-check results with multiple methods
5. **Document**: Keep track of analysis parameters

### **Performance Optimization**
1. **Use Appropriate Ranges**: Smaller time ranges = faster processing
2. **Enable Plot Limiting**: Prevents browser issues
3. **Monitor Resources**: Watch CPU and memory usage
4. **Batch Processing**: Process multiple files efficiently

---

## 📚 Additional Resources

### **Documentation**
- [API Documentation](../docs/api/)
- [Architecture Guide](../dev_docs/architecture/)
- [Performance Guide](../dev_docs/performance_optimization/)
- [Developer Guide](../dev_docs/developer_guides/)

### **Examples**
- [Sample Data](../sample_data/)
- [Notebook Examples](../examples/notebooks/)
- [Code Examples](../examples/)

### **Support**
- **Issues**: Report issues on GitHub
- **Questions**: Ask questions in discussions
- **Contributions**: Contribute improvements
- **Documentation**: Help improve documentation

---

## ✅ Quick Reference

### **Start Commands**
```bash
# Normal mode
python src/vitalDSP_webapp/run_webapp_debug.py

# Debug mode
python src/vitalDSP_webapp/run_webapp_debug.py --debug

# Custom port
python src/vitalDSP_webapp/run_webapp_debug.py --port 8080
```

### **Key URLs**
- **Web Application**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### **Important Files**
- **Main App**: `src/vitalDSP_webapp/app.py`
- **Configuration**: `src/vitalDSP_webapp/config/`
- **Logs**: `webapp.log` (normal) or `webapp_debug.log` (debug)

---

**Status**: ✅ **Production Ready**  
**Last Updated**: October 23, 2025  
**Version**: 2.0.0
