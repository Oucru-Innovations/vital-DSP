API Reference
=============

This section provides comprehensive documentation for all VitalDSP modules, classes, and functions.

Core Library
============

Filtering Module
----------------

Signal Filtering
~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.filtering.signal_filtering
   :members:
   :undoc-members:
   :show-inheritance:

Artifact Removal
~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.filtering.artifact_removal
   :members:
   :undoc-members:
   :show-inheritance:

Advanced Signal Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.filtering.advanced_signal_filtering
   :members:
   :undoc-members:
   :show-inheritance:

Physiological Features Module
=============================

Time Domain Features
~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.physiological_features.time_domain
   :members:
   :undoc-members:
   :show-inheritance:

Frequency Domain Features
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.physiological_features.frequency_domain
   :members:
   :undoc-members:
   :show-inheritance:

HRV Analysis
~~~~~~~~~~~~

.. automodule:: vitalDSP.physiological_features.hrv_analysis
   :members:
   :undoc-members:
   :show-inheritance:

Beat-to-Beat Analysis
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.physiological_features.beat_to_beat
   :members:
   :undoc-members:
   :show-inheritance:

Nonlinear Features
~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.physiological_features.nonlinear
   :members:
   :undoc-members:
   :show-inheritance:

Waveform Morphology
~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.physiological_features.waveform
   :members:
   :undoc-members:
   :show-inheritance:

Respiratory Analysis Module
===========================

Respiratory Analysis
~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.respiratory_analysis.respiratory_analysis
   :members:
   :undoc-members:
   :show-inheritance:

FFT-Based RR Estimation
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.respiratory_analysis.estimate_rr.fft_based_rr
   :members:
   :undoc-members:
   :show-inheritance:

Peak Detection RR Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.respiratory_analysis.estimate_rr.peak_detection_rr
   :members:
   :undoc-members:
   :show-inheritance:

Sleep Apnea Detection
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.respiratory_analysis.sleep_apnea_detection.amplitude_threshold
   :members:
   :undoc-members:
   :show-inheritance:

Transforms Module
=================

Fourier Transform
~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.transforms.fourier_transform
   :members:
   :undoc-members:
   :show-inheritance:

Wavelet Transform
~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.transforms.wavelet_transform
   :members:
   :undoc-members:
   :show-inheritance:

Discrete Cosine Transform
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.transforms.discrete_cosine_transform
   :members:
   :undoc-members:
   :show-inheritance:

Hilbert Transform
~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.transforms.hilbert_transform
   :members:
   :undoc-members:
   :show-inheritance:

Advanced Computation Module
===========================

Anomaly Detection
~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.advanced_computation.anomaly_detection
   :members:
   :undoc-members:
   :show-inheritance:

Bayesian Analysis
~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.advanced_computation.bayesian_analysis
   :members:
   :undoc-members:
   :show-inheritance:

Neural Network Filtering
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.advanced_computation.neural_network_filtering
   :members:
   :undoc-members:
   :show-inheritance:

Reinforcement Learning Filter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.advanced_computation.reinforcement_learning_filter
   :members:
   :undoc-members:
   :show-inheritance:

EMD (Empirical Mode Decomposition)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.advanced_computation.emd
   :members:
   :undoc-members:
   :show-inheritance:

Feature Engineering Module
===========================

ECG Autonomic Features
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.feature_engineering.ecg_autonomic_features
   :members:
   :undoc-members:
   :show-inheritance:

PPG Autonomic Features
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.feature_engineering.ppg_autonomic_features
   :members:
   :undoc-members:
   :show-inheritance:

Morphology Features
~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.feature_engineering.morphology_features
   :members:
   :undoc-members:
   :show-inheritance:

Signal Quality Assessment Module
================================

Signal Quality
~~~~~~~~~~~~~~

.. automodule:: vitalDSP.signal_quality_assessment.signal_quality
   :members:
   :undoc-members:
   :show-inheritance:

Signal Quality Index
~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.signal_quality_assessment.signal_quality_index
   :members:
   :undoc-members:
   :show-inheritance:

SNR Computation
~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.signal_quality_assessment.snr_computation
   :members:
   :undoc-members:
   :show-inheritance:

Utils Module
============

Peak Detection
~~~~~~~~~~~~~~

.. automodule:: vitalDSP.utils.peak_detection
   :members:
   :undoc-members:
   :show-inheritance:

Data Synthesis
~~~~~~~~~~~~~~

.. automodule:: vitalDSP.utils.synthesize_data
   :members:
   :undoc-members:
   :show-inheritance:

Standard Scaler
~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.utils.scaler
   :members:
   :undoc-members:
   :show-inheritance:

Normalization
~~~~~~~~~~~~~

.. automodule:: vitalDSP.utils.normalization
   :members:
   :undoc-members:
   :show-inheritance:

Interpolations
~~~~~~~~~~~~~~

.. automodule:: vitalDSP.utils.interpolations
   :members:
   :undoc-members:
   :show-inheritance:

Health Analysis Module
======================

Health Report Generator
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.health_analysis.health_report_generator
   :members:
   :undoc-members:
   :show-inheritance:

Health Report Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.health_analysis.health_report_visualization
   :members:
   :undoc-members:
   :show-inheritance:

Interpretation Engine
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.health_analysis.interpretation_engine
   :members:
   :undoc-members:
   :show-inheritance:

Web Application API
===================

Data Service
~~~~~~~~~~~~

.. automodule:: vitalDSP_webapp.services.data.data_service
   :members:
   :undoc-members:
   :show-inheritance:

Settings Service
~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP_webapp.services.settings_service
   :members:
   :undoc-members:
   :show-inheritance:

API Endpoints
~~~~~~~~~~~~~

.. automodule:: vitalDSP_webapp.api.endpoints
   :members:
   :undoc-members:
   :show-inheritance:

Web Application Callbacks
=========================

Core Callbacks
~~~~~~~~~~~~~~

.. automodule:: vitalDSP_webapp.callbacks.core.app_callbacks
   :members:
   :undoc-members:
   :show-inheritance:

Upload Callbacks
~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP_webapp.callbacks.core.upload_callbacks
   :members:
   :undoc-members:
   :show-inheritance:

Page Routing Callbacks
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP_webapp.callbacks.core.page_routing_callbacks
   :members:
   :undoc-members:
   :show-inheritance:

Analysis Callbacks
~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP_webapp.callbacks.analysis.vitaldsp_callbacks
   :members:
   :undoc-members:
   :show-inheritance:

Signal Filtering Callbacks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks
   :members:
   :undoc-members:
   :show-inheritance:

Respiratory Analysis Callbacks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP_webapp.callbacks.analysis.respiratory_callbacks
   :members:
   :undoc-members:
   :show-inheritance:

Features Callbacks
~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP_webapp.callbacks.features.features_callbacks
   :members:
   :undoc-members:
   :show-inheritance:

Physiological Callbacks
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP_webapp.callbacks.features.physiological_callbacks
   :members:
   :undoc-members:
   :show-inheritance:

Respiratory Callbacks
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP_webapp.callbacks.features.respiratory_callbacks
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
=================

Common Utilities
~~~~~~~~~~~~~~~~

.. automodule:: vitalDSP.utils.common
   :members:
   :undoc-members:
   :show-inheritance:

Error Handling
~~~~~~~~~~~~~~

.. automodule:: vitalDSP_webapp.utils.error_handler
   :members:
   :undoc-members:
   :show-inheritance:

Data Processor
~~~~~~~~~~~~~~

.. automodule:: vitalDSP_webapp.utils.data_processor
   :members:
   :undoc-members:
   :show-inheritance:

Settings Utils
~~~~~~~~~~~~~~

.. automodule:: vitalDSP_webapp.utils.settings_utils
   :members:
   :undoc-members:
   :show-inheritance:
