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

The web interface provides access to all major signal processing capabilities:

* **Preprocessing**: Detrending, normalization, and noise reduction
* **Filtering**: Bandpass, lowpass, highpass, and custom filters
* **Artifact Removal**: Advanced algorithms for noise and artifact detection
* **Quality Assessment**: Signal quality metrics and validation

.. code-block:: python

   # Example: Signal filtering through web interface
   from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import apply_filter
   
   filtered_signal = apply_filter(
       signal_data=signal,
       filter_type='bandpass',
       low_cut=0.5,
       high_cut=40.0,
       sampling_rate=1000
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

The web application provides rich, interactive visualizations:

* **Time Series Plots**: Interactive signal visualization with zoom and pan
* **Frequency Domain**: FFT and power spectral density plots
* **Feature Plots**: Scatter plots, histograms, and correlation matrices
* **3D Visualizations**: Multi-dimensional data exploration

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
A: Check that all dependencies are installed and ports are available.

**Q: Signal upload fails**
A: Ensure the file format is supported and data is properly formatted.

**Q: Processing is slow**
A: Consider reducing signal length or using more efficient processing options.

**Q: Visualizations don't display**
A: Check browser compatibility and ensure JavaScript is enabled.

Support and Documentation
=========================

For additional support and documentation:

* **GitHub Issues**: Report bugs and request features
* **Documentation**: Comprehensive API and user guides
* **Community Forum**: Connect with other users and developers
* **Email Support**: Direct support for enterprise users

.. note::
   The web application is continuously updated with new features and improvements. Check the changelog for the latest updates.
