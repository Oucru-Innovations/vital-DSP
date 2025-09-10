Getting Started with VitalDSP
=============================

Welcome to VitalDSP, a comprehensive digital signal processing (DSP) library designed for healthcare and biomedical applications. This guide will walk you through the initial steps to get the library up and running and explore its core functionalities.

What is VitalDSP?
=================

VitalDSP is a powerful Python library specifically designed for processing and analyzing physiological signals such as:

* **ECG (Electrocardiogram)**: Heart rhythm and electrical activity analysis
* **PPG (Photoplethysmogram)**: Blood volume changes and pulse analysis
* **Respiratory Signals**: Breathing patterns and respiratory rate estimation
* **Other Vital Signs**: Blood pressure, temperature, and more

The library provides both a comprehensive Python API and an intuitive web application for signal processing, making it accessible to researchers, clinicians, and developers.

Installation
============

Prerequisites
~~~~~~~~~~~~~

Before installing VitalDSP, ensure you have Python 3.8 or higher installed on your system.

.. code-block:: bash

   python --version  # Should be 3.8 or higher

Installation Methods
~~~~~~~~~~~~~~~~~~~~

**Option 1: Install from PyPI (Recommended)**

.. code-block:: bash

   pip install vital-DSP

**Option 2: Install from Source**

.. code-block:: bash

   git clone https://github.com/Oucru-Innovations/vital-DSP.git
   cd vital-DSP
   pip install -e .

**Option 3: Install with Development Dependencies**

.. code-block:: bash

   pip install vital-DSP[dev]

Verify Installation
~~~~~~~~~~~~~~~~~~~

Test your installation by importing the library:

.. code-block:: python

   import vitalDSP
   print(f"VitalDSP version: {vitalDSP.__version__}")

Quick Start
===========

Basic Signal Processing
~~~~~~~~~~~~~~~~~~~~~~~

Here's a simple example to get you started with signal processing:

.. code-block:: python

   import numpy as np
   import vitalDSP
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.physiological_features.time_domain import TimeDomainFeatures

   # Create a sample signal (replace with your actual data)
   fs = 1000  # Sampling frequency
   t = np.linspace(0, 10, fs * 10)  # 10 seconds of data
   signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.random.randn(len(t))

   # Initialize signal filtering
   sf = SignalFiltering(signal, fs)
   
   # Apply bandpass filter
   filtered_signal = sf.bandpass_filter(low_cut=0.5, high_cut=40.0)
   
   # Extract time domain features
   tdf = TimeDomainFeatures(filtered_signal, fs)
   features = tdf.extract_features()
   
   print(f"Extracted features: {features}")

Web Application
~~~~~~~~~~~~~~~

Launch the interactive web application:

.. code-block:: python

   from vitalDSP_webapp.run_webapp import run_webapp
   
   # Start the web application
   run_webapp(debug=True, port=8050)

Then open your browser and navigate to `http://localhost:8050`.

Core Modules Overview
=====================

Signal Filtering
~~~~~~~~~~~~~~~~

The filtering module provides various signal processing techniques:

.. code-block:: python

   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.filtering.artifact_removal import ArtifactRemoval
   
   # Basic filtering
   sf = SignalFiltering(signal, sampling_rate)
   filtered = sf.bandpass_filter(low_cut=0.5, high_cut=40.0)
   
   # Artifact removal
   ar = ArtifactRemoval(signal)
   clean_signal = ar.median_filter_removal(kernel_size=3)

Physiological Features
~~~~~~~~~~~~~~~~~~~~~~

Extract meaningful features from physiological signals:

.. code-block:: python

   from vitalDSP.physiological_features.time_domain import TimeDomainFeatures
   from vitalDSP.physiological_features.hrv_analysis import HRVFeatures
   
   # Time domain features
   tdf = TimeDomainFeatures(signal, sampling_rate)
   time_features = tdf.extract_features()
   
   # HRV analysis
   hrv = HRVFeatures(rr_intervals)
   hrv_features = hrv.analyze_hrv()

Respiratory Analysis
~~~~~~~~~~~~~~~~~~~~

Analyze respiratory patterns and estimate respiratory rate:

.. code-block:: python

   from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis
   
   resp_analysis = RespiratoryAnalysis(signal, sampling_rate)
   respiratory_rate = resp_analysis.estimate_respiratory_rate()
   features = resp_analysis.extract_respiratory_features()

Advanced Computation
~~~~~~~~~~~~~~~~~~~~

Use machine learning and advanced algorithms:

.. code-block:: python

   from vitalDSP.advanced_computation.neural_network_filtering import NeuralNetworkFiltering
   from vitalDSP.advanced_computation.anomaly_detection import AnomalyDetection
   
   # Neural network filtering
   nn_filter = NeuralNetworkFiltering()
   filtered_signal = nn_filter.filter(signal)
   
   # Anomaly detection
   anomaly_detector = AnomalyDetection()
   anomalies = anomaly_detector.detect_anomalies(signal)

Data Synthesis
~~~~~~~~~~~~~~

Generate synthetic physiological signals for testing and development:

.. code-block:: python

   from vitalDSP.utils.synthesize_data import SynthesizeData
   
   synthesizer = SynthesizeData()
   
   # Generate synthetic ECG
   ecg_signal = synthesizer.generate_synthetic_ecg(
       duration=10,
       sampling_rate=1000,
       heart_rate=72
   )
   
   # Generate synthetic PPG
   ppg_signal = synthesizer.generate_synthetic_ppg(
       duration=10,
       sampling_rate=1000,
       heart_rate=72
   )

Working with Real Data
======================

Loading Data
~~~~~~~~~~~~

VitalDSP supports various data formats:

.. code-block:: python

   import pandas as pd
   from vitalDSP_webapp.services.data_service import DataService
   
   # Load CSV data
   data = pd.read_csv('ecg_data.csv')
   signal = data['ecg'].values
   time = data['time'].values
   
   # Or use the data service
   data_service = DataService()
   signal_data = data_service.load_signal_data('ecg_data.csv')

Preprocessing
~~~~~~~~~~~~~

Clean and preprocess your signals:

.. code-block:: python

   from vitalDSP.preprocess.preprocess_operations import PreprocessConfig
   from vitalDSP.preprocess.noise_reduction import NoiseReduction
   
   # Configure preprocessing
   config = PreprocessConfig(
       detrend=True,
       normalize=True,
       remove_baseline=True
   )
   
   # Apply preprocessing
   nr = NoiseReduction(signal, sampling_rate)
   clean_signal = nr.apply_preprocessing(config)

Feature Extraction Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a complete analysis pipeline:

.. code-block:: python

   def analyze_physiological_signal(signal, sampling_rate):
       """Complete analysis pipeline for physiological signals."""
       
       # 1. Preprocessing
       sf = SignalFiltering(signal, sampling_rate)
       filtered = sf.bandpass_filter(low_cut=0.5, high_cut=40.0)
       
       # 2. Feature extraction
       tdf = TimeDomainFeatures(filtered, sampling_rate)
       time_features = tdf.extract_features()
       
       fdf = FrequencyDomainFeatures(filtered, sampling_rate)
       freq_features = fdf.extract_features()
       
       # 3. Quality assessment
       from vitalDSP.signal_quality_assessment.signal_quality import SignalQuality
       sq = SignalQuality(filtered, sampling_rate)
       quality_metrics = sq.assess_quality()
       
       return {
           'time_features': time_features,
           'frequency_features': freq_features,
           'quality_metrics': quality_metrics
       }

Web Application Features
========================

Interactive Analysis
~~~~~~~~~~~~~~~~~~~~

The web application provides an intuitive interface for:

* **Signal Upload**: Drag-and-drop file upload
* **Real-time Processing**: Live signal analysis
* **Interactive Plots**: Zoom, pan, and export visualizations
* **Feature Extraction**: Automated feature computation
* **Report Generation**: Comprehensive analysis reports

API Integration
~~~~~~~~~~~~~~~

Integrate VitalDSP with your existing applications:

.. code-block:: python

   import requests
   
   # Upload signal data
   with open('signal.csv', 'rb') as f:
       response = requests.post(
           'http://localhost:8050/api/upload',
           files={'file': f}
       )
   
   # Process signal
   response = requests.post(
       'http://localhost:8050/api/process',
       json={
           'signal_id': 'signal_123',
           'operations': ['filter', 'extract_features']
       }
   )

Explore More with Jupyter Notebooks
===================================

For more detailed examples and practical tutorials, explore the Jupyter Notebooks provided with the library:

.. toctree::
   :maxdepth: 2
   :caption: Jupyter Notebooks:

   sample_notebooks

The notebooks cover:

* **Signal Filtering**: Various filtering techniques and their applications
* **Feature Engineering**: Advanced feature extraction methods
* **Health Report Analysis**: Automated report generation
* **Artifact Removal**: Noise reduction and signal cleaning
* **Data Synthesis**: Generating synthetic signals for testing

Best Practices
==============

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

* Use appropriate sampling rates for your analysis
* Consider signal length vs. processing time trade-offs
* Utilize batch processing for multiple signals
* Cache frequently used computations

Data Quality
~~~~~~~~~~~~

* Always assess signal quality before analysis
* Apply appropriate preprocessing steps
* Validate results against known standards
* Document your processing pipeline

Error Handling
~~~~~~~~~~~~~~

* Use try-catch blocks for robust error handling
* Validate input data before processing
* Log important processing steps
* Provide meaningful error messages

Support and Community
=====================

Getting Help
~~~~~~~~~~~~

* **Documentation**: Comprehensive guides and API reference
* **GitHub Issues**: Report bugs and request features
* **Community Forum**: Connect with other users
* **Email Support**: Direct support for enterprise users

Contributing
~~~~~~~~~~~~

We welcome contributions! Check out our GitHub repository:

`VitalDSP GitHub <https://github.com/Oucru-Innovations/vital-DSP>`_

Ways to contribute:

* Report bugs and issues
* Suggest new features
* Submit pull requests
* Improve documentation
* Share use cases and examples

Next Steps
==========

Now that you have VitalDSP installed and running, explore:

1. **Core Modules**: Dive deeper into filtering, feature extraction, and analysis
2. **Web Application**: Use the interactive interface for signal processing
3. **Jupyter Notebooks**: Follow along with detailed tutorials
4. **API Reference**: Explore the complete function and class documentation
5. **Advanced Features**: Try machine learning and advanced computation modules

Happy analyzing with VitalDSP! 🫀📊

