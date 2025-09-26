VitalDSP Web Application
========================

The VitalDSP web application provides an intuitive, interactive interface for signal processing and analysis. Built with Dash and Plotly, it offers real-time visualization and analysis capabilities for physiological signals.

Overview
========

The web application is designed to make VitalDSP accessible to users who prefer a graphical interface over programming. It provides:

* **Interactive Signal Upload**: Support for CSV, Excel, and other data formats
* **Real-time Processing**: Live signal filtering and analysis
* **Comprehensive Visualization**: Interactive plots with zoom, pan, and export capabilities
* **Feature Extraction**: Automated extraction of physiological features
* **Health Reports**: Generation of comprehensive analysis reports

Installation and Setup
======================

To run the web application locally:

.. code-block:: bash

   # Install the package
   pip install vital-DSP
   
   # Run the web application
   python -m vitalDSP_webapp.run_webapp

The application will be available at `http://localhost:8050` by default.

Main Features
=============

The VitalDSP web application provides a comprehensive, user-friendly interface for physiological signal analysis. The application is organized into several specialized screens, each designed for specific aspects of signal processing and analysis.

Application Screens Overview
----------------------------

The web application consists of the following main screens:

1. **Upload Screen**: Data upload and initial configuration
2. **Filtering Screen**: Signal preprocessing and filtering
3. **Time Domain Analysis**: Temporal signal analysis and visualization
4. **Frequency Domain Analysis**: Spectral analysis and frequency features
5. **Physiological Analysis**: Comprehensive physiological feature extraction
6. **Respiratory Analysis**: Respiratory rate estimation and breathing pattern analysis
7. **Advanced Features**: Machine learning and advanced signal processing

Signal Upload and Management
----------------------------

The upload screen is the entry point for all signal processing workflows. It provides comprehensive data management capabilities:

**Supported Data Formats**
    * **CSV Files**: Comma-separated values with time and signal columns
    * **Excel Files**: Multiple sheets and data ranges
    * **JSON Files**: Structured data with metadata
    * **Real-time Data**: Live streaming from connected devices

**Automatic Signal Type Detection**
    The application automatically detects signal types based on:
    * Column names (e.g., "ecg", "ppg", "pleth")
    * Signal characteristics and frequency content
    * Data patterns and morphology

**Column Mapping**
    Intelligent column detection and mapping:
    * **Time Column**: Automatically identifies time/timestamp columns
    * **Signal Column**: Detects signal data columns (ECG, PPG, etc.)
    * **Multi-Channel Support**: Handles RED/IR channels for pulse oximetry
    * **Custom Mapping**: Manual override for complex data structures

**Data Validation**
    Comprehensive data validation including:
    * Format verification and error checking
    * Sampling frequency validation
    * Data quality assessment
    * Missing value detection and handling

.. code-block:: python

   # Example: Uploading signal data
   from vitalDSP_webapp.services.data_service import DataService
   
   data_service = DataService()
   signal_data = data_service.load_signal_data('path/to/signal.csv')

Signal Processing
-----------------

The web interface provides comprehensive signal processing capabilities with a streamlined workflow:

* **Preprocessing**: Detrending, normalization, and noise reduction
* **Multi-Type Filtering**: Traditional, advanced, artifact removal, neural, and ensemble filters
* **Dynamic Filter Application**: Automatic filter application when time windows change
* **Quality Assessment**: Enhanced signal quality metrics and validation
* **Critical Points Detection**: PPG/ECG-specific peak and notch detection

**New Workflow**: The filtering screen handles all signal processing, while analysis screens use the pre-filtered data for consistent results.

.. code-block:: python

   # Example: Multi-type filtering through web interface
   from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import apply_traditional_filter
   
   # Traditional filtering
   filtered_signal = apply_traditional_filter(
       signal_data=signal,
       sampling_freq=100,
       filter_family='butter',
       filter_response='bandpass',
       low_freq=0.5,
       high_freq=5.0,
       filter_order=4
   )
   
   # Advanced filtering
   from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import apply_advanced_filter
   
   filtered_signal = apply_advanced_filter(
       signal_data=signal,
       advanced_method='kalman',
       noise_level=0.1,
       iterations=100,
       learning_rate=0.01
   )

Physiological Analysis
----------------------

Comprehensive physiological feature extraction and analysis:

* **Heart Rate Variability (HRV)**: Time-domain, frequency-domain, and nonlinear features
* **Respiratory Analysis**: Multi-modal respiratory rate estimation
* **Morphological Features**: Waveform analysis and feature extraction
* **Quality Metrics**: Signal quality assessment and validation

.. code-block:: python

   # Example: HRV analysis
   from vitalDSP_webapp.callbacks.features.physiological_callbacks import analyze_hrv
   
   hrv_results = analyze_hrv(
       rr_intervals=rr_data,
       analysis_type='comprehensive',
       time_domain=True,
       frequency_domain=True,
       nonlinear=True
   )

Interactive Visualization
========================

The web application provides rich, interactive visualizations with enhanced features:

* **Time Series Plots**: Interactive signal visualization with zoom and pan
* **Critical Points Visualization**: PPG/ECG-specific peak and notch detection and display
* **Signal Comparison**: Side-by-side comparison of original vs filtered signals
* **Frequency Domain**: FFT and power spectral density plots
* **Feature Plots**: Scatter plots, histograms, and correlation matrices
* **3D Visualizations**: Multi-dimensional data exploration

**Enhanced Features**:
* **Dynamic Signal Type Detection**: Automatic adaptation to PPG, ECG, or other signal types
* **Real-time Critical Points**: Live detection and display of physiological features
* **Filter Information Display**: Clear presentation of applied filter parameters
* **Quality Metrics Visualization**: Interactive display of signal quality indicators

Detailed Screen Descriptions
============================

Filtering Screen
----------------

The filtering screen is the central hub for signal preprocessing and enhancement. It provides comprehensive filtering capabilities with automatic signal type detection and intelligent defaults.

**Key Features**
    * **Multi-Type Filtering**: Traditional, advanced, artifact removal, neural network, and ensemble filtering
    * **Automatic Signal Type Detection**: Automatically detects ECG, PPG, or other signal types
    * **Smart Defaults**: ECG signals default to Advanced Filters with convolution method
    * **Real-Time Preview**: Live preview of filtered results
    * **Quality Metrics**: Built-in signal quality assessment

**Filter Types Available**
    * **Traditional Filters**: Butterworth, Chebyshev, Elliptic, and Bessel filters
    * **Advanced Filters**: Kalman, adaptive, and machine learning-based filtering
    * **Artifact Removal**: Motion artifacts, powerline interference, and baseline wander
    * **Neural Network Filters**: Deep learning-based artifact removal
    * **Ensemble Methods**: Combination of multiple filtering approaches

**Usage Instructions**
    1. Upload your signal data on the Upload screen
    2. Navigate to the Filtering screen
    3. The signal type will be automatically detected and appropriate defaults set
    4. Select your desired filter type and configure parameters
    5. Apply filtering and review results
    6. Filtered data is automatically available for analysis screens

**Clinical Applications**
    * **ECG Processing**: Removal of powerline interference, muscle artifacts, and baseline wander
    * **PPG Enhancement**: Filtering of motion artifacts and ambient light interference
    * **Real-Time Monitoring**: Live filtering for clinical monitoring applications

Physiological Analysis Screen
-----------------------------

The physiological analysis screen provides comprehensive feature extraction and analysis for physiological signals, with automatic signal type detection and clinical interpretation.

**Key Features**
    * **Comprehensive HRV Analysis**: 50+ heart rate variability metrics
    * **Morphological Analysis**: Waveform shape and structure analysis
    * **Quality Assessment**: Signal quality metrics and validation
    * **Clinical Interpretation**: Built-in clinical significance assessment
    * **Multi-Signal Support**: ECG, PPG, and other physiological signals

**Analysis Categories**
    * **HRV Analysis**: Time-domain, frequency-domain, and nonlinear HRV metrics
    * **Morphological Features**: Peak detection, duration analysis, and waveform characteristics
    * **Quality Metrics**: Signal-to-noise ratio, stability, and artifact detection
    * **Advanced Features**: Cross-signal analysis and complexity measures

**Clinical Applications**
    * **Cardiovascular Health**: Assessment of heart function and vascular compliance
    * **Stress and Infection Detection**: Early identification of physiological stress
    * **Disease Progression**: Monitoring of chronic conditions and treatment response
    * **Sleep and Respiratory Health**: Analysis of breathing patterns and sleep quality

**Usage Instructions**
    1. Ensure signal data is uploaded and filtered
    2. Navigate to the Physiological Analysis screen
    3. Signal type is automatically detected and appropriate analysis configured
    4. Select analysis categories and parameters
    5. Review comprehensive analysis results with clinical interpretation

Frequency Domain Analysis Screen
--------------------------------

The frequency domain analysis screen provides spectral analysis and frequency-based feature extraction with support for both original and filtered signals.

**Key Features**
    * **Spectral Analysis**: FFT, PSD, and spectrogram analysis
    * **Frequency Features**: Power spectral density and frequency domain metrics
    * **Multi-Signal Support**: Analysis of original or filtered signals
    * **Interactive Visualization**: Zoom, pan, and export capabilities
    * **Clinical Interpretation**: Frequency-based health indicators

**Analysis Types**
    * **FFT Analysis**: Fast Fourier Transform with configurable parameters
    * **Power Spectral Density**: Welch's method and other PSD techniques
    * **Spectrogram**: Time-frequency analysis using STFT
    * **Wavelet Analysis**: Continuous and discrete wavelet transforms

**Usage Instructions**
    1. Upload and optionally filter your signal data
    2. Navigate to the Frequency Domain Analysis screen
    3. Select signal source (original or filtered)
    4. Configure analysis parameters (window type, overlap, etc.)
    5. Review spectral analysis results and frequency features

Respiratory Analysis Screen
---------------------------

The respiratory analysis screen specializes in respiratory rate estimation and breathing pattern analysis using multiple estimation methods.

**Key Features**
    * **Multi-Modal Estimation**: Peak detection, FFT-based, and ensemble methods
    * **Breathing Pattern Analysis**: Detection of apnea, hypopnea, and irregular patterns
    * **Real-Time Processing**: Live respiratory rate monitoring
    * **Clinical Validation**: Methods validated on clinical datasets
    * **Signal-Specific Optimization**: Optimized for ECG, PPG, and respiratory signals

**Estimation Methods**
    * **Peak Detection**: Time-domain peak detection for respiratory cycles
    * **FFT-Based**: Frequency domain analysis of respiratory patterns
    * **Ensemble Methods**: Combination of multiple estimation approaches
    * **Machine Learning**: Advanced algorithms for complex breathing patterns

**Clinical Applications**
    * **Sleep Apnea Detection**: Identification of breathing irregularities during sleep
    * **ICU Monitoring**: Real-time respiratory rate monitoring in critical care
    * **COVID-19 Assessment**: Respiratory pattern analysis for infection monitoring
    * **Chronic Disease Management**: Long-term respiratory health monitoring

**Usage Instructions**
    1. Upload signal data (ECG, PPG, or respiratory signals)
    2. Navigate to the Respiratory Analysis screen
    3. Signal type is automatically detected
    4. Select estimation methods and configure parameters
    5. Review respiratory rate estimates and breathing pattern analysis

Updated Workflow
================

The web application follows an improved workflow that separates filtering from analysis:

Signal Upload and Processing
----------------------------

1. **Upload Data**: Upload signal data with automatic format detection
2. **Configure Parameters**: Set sampling frequency, signal type, and other parameters
3. **Apply Filtering**: Use the dedicated filtering screen for signal processing
4. **Analyze Results**: Use analysis screens with pre-filtered data

Key Improvements
----------------

* **Separation of Concerns**: Filtering and analysis are handled in separate screens
* **Consistent Results**: Same filtered data used across all analysis screens
* **Dynamic Filtering**: Automatic filter application when time windows change
* **Multi-Filter Support**: Support for all filter types (traditional, advanced, neural, etc.)
* **Enhanced UI**: Cleaner interface with better organization and defaults
* **Robust Error Handling**: Graceful fallbacks and comprehensive error management

Export and Reporting
====================

Generate comprehensive reports and export results:

* **PDF Reports**: Professional analysis reports with visualizations
* **Data Export**: CSV, Excel, and JSON format exports
* **Image Export**: High-resolution plots and figures
* **Custom Reports**: Configurable report templates

API Integration
===============

The web application exposes RESTful APIs for integration with other systems:

.. code-block:: python

   # Example: API endpoint usage
   import requests
   
   # Upload signal data
   response = requests.post(
       'http://localhost:8050/api/upload',
       files={'file': open('signal.csv', 'rb')}
   )
   
   # Process signal
   response = requests.post(
       'http://localhost:8050/api/process',
       json={
           'signal_id': 'signal_123',
           'operations': ['filter', 'detrend', 'normalize']
       }
   )

Data Management
===============

The web application includes enhanced data management capabilities:

Global Data Storage
-------------------

The data service now supports global storage of filtered data for use across multiple screens:

.. code-block:: python

   # Example: Storing and retrieving filtered data
   from vitalDSP_webapp.services.data.data_service import DataService
   
   data_service = DataService()
   
   # Store filtered data after processing
   filter_info = {
       "filter_type": "traditional",
       "parameters": {
           "filter_family": "butter",
           "filter_response": "bandpass",
           "low_freq": 0.5,
           "high_freq": 5.0,
           "filter_order": 4
       },
       "detrending_applied": True,
       "timestamp": "2024-01-01T12:00:00"
   }
   
   data_service.store_filtered_data(
       data_id="signal_123",
       filtered_signal=filtered_data,
       filter_info=filter_info
   )
   
   # Retrieve filtered data in analysis screens
   filtered_data = data_service.get_filtered_data("signal_123")
   filter_info = data_service.get_filter_info("signal_123")
   
   # Check if filtered data is available
   has_filtered = data_service.has_filtered_data("signal_123")

Signal Type Detection
--------------------

Automatic signal type detection and appropriate critical points detection:

.. code-block:: python

   # Example: Dynamic signal type handling
   from vitalDSP.physiological_features.waveform import WaveformMorphology
   
   # PPG signal analysis
   wm_ppg = WaveformMorphology(signal_data, fs=100, signal_type="PPG")
   systolic_peaks = wm_ppg.systolic_peaks
   dicrotic_notches = wm_ppg.dicrotic_notches
   
   # ECG signal analysis
   wm_ecg = WaveformMorphology(signal_data, fs=100, signal_type="ECG")
   r_peaks = wm_ecg.r_peaks
   p_waves = wm_ecg.p_peaks

Configuration
=============

The web application can be configured through environment variables or configuration files:

.. code-block:: python

   # Example configuration
   from vitalDSP_webapp.config.settings import Settings
   
   settings = Settings()
   settings.DEBUG = True
   settings.HOST = '0.0.0.0'
   settings.PORT = 8050
   settings.MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

Advanced Features
=================

Custom Analysis Pipelines
-------------------------

Create and save custom analysis pipelines:

.. code-block:: python

   # Example: Custom pipeline
   from vitalDSP_webapp.services.analysis_service import AnalysisService
   
   pipeline = AnalysisService.create_pipeline([
       'preprocess',
       'filter_bandpass',
       'extract_features',
       'generate_report'
   ])
   
   results = pipeline.execute(signal_data)

Batch Processing
----------------

Process multiple signals in batch:

.. code-block:: python

   # Example: Batch processing
   from vitalDSP_webapp.services.batch_service import BatchProcessor
   
   processor = BatchProcessor()
   results = processor.process_directory(
       input_dir='signals/',
       output_dir='results/',
       pipeline_config='config.json'
   )

Troubleshooting
===============

Common Issues and Solutions
---------------------------

**Q: The web application won't start**
A: Check that all dependencies are installed and ports are available. Ensure no other application is using port 8050.

**Q: Signal upload fails**
A: Ensure the file format is supported and data is properly formatted. Check that the signal column is correctly identified.

**Q: Filtered data not available in analysis screens**
A: Make sure to apply filtering in the filtering screen first. The analysis screens use pre-filtered data from the filtering screen.

**Q: Critical points not detected correctly**
A: Verify the signal type is set correctly (PPG/ECG/Other) and the sampling frequency is accurate. Different signal types use different detection algorithms.

**Q: Time window changes cause errors**
A: The system automatically applies stored filter parameters to new time windows. If errors persist, check that the filter parameters are valid.

**Q: Processing is slow**
A: Consider reducing signal length or using more efficient processing options. The new workflow reduces redundant processing.

**Q: Visualizations don't display**
A: Check browser compatibility and ensure JavaScript is enabled. Clear browser cache if plots appear corrupted.

**Q: Filter information shows incorrect parameters**
A: The system now displays filter parameters in a readable format. If parameters appear wrong, re-apply filtering in the filtering screen.

**Q: Signal quality metrics show unrealistic values**
A: The system now includes robust calculations for signal stability and SNR. If values seem incorrect, check the signal data quality and preprocessing steps.

Support and Documentation
=========================

For additional support and documentation:

* **GitHub Issues**: Report bugs and request features
* **Documentation**: Comprehensive API and user guides
* **Community Forum**: Connect with other users and developers
* **Email Support**: Direct support for enterprise users

.. note::
   The web application is continuously updated with new features and improvements. Check the changelog for the latest updates.
