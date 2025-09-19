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

Signal Upload and Management
----------------------------

The web application supports various data formats for signal upload:

* **CSV Files**: Comma-separated values with time and signal columns
* **Excel Files**: Multiple sheets and data ranges
* **JSON Files**: Structured data with metadata
* **Real-time Data**: Live streaming from connected devices

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

Updated Workflow
================

The web application now follows an improved workflow that separates filtering from analysis:

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
   wm_ppg = WaveformMorphology(signal_data, fs=100, signal_type="ppg")
   systolic_peaks = wm_ppg.systolic_peaks
   dicrotic_notches = wm_ppg.dicrotic_notches
   
   # ECG signal analysis
   wm_ecg = WaveformMorphology(signal_data, fs=100, signal_type="ecg")
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
