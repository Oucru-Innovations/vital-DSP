Feature Engineering
====================

This section covers the comprehensive feature extraction techniques provided by the VitalDSP library for analyzing physiological signals such as ECG and PPG. These methods help to derive meaningful features that describe various morphological, autonomic, and synchronization characteristics of the signals.

Overview
========

The feature engineering module provides specialized tools for:

* **Morphological Features**: Waveform shape and structure analysis
* **Autonomic Features**: Heart rate variability and autonomic nervous system indicators
* **Synchronization Features**: Multi-signal correlation and timing analysis
* **Light Source Features**: PPG-specific features for different light wavelengths
* **Volume Features**: Blood volume and pressure-related characteristics

Morphological Features
======================

Morphology Features
~~~~~~~~~~~~~~~~~~~

Techniques to extract morphological features from physiological waveforms, including the detection of peaks, troughs, and various segments in ECG and PPG signals.

.. automodule:: vitalDSP.feature_engineering.morphology_features
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Autonomic Features
==================

ECG Autonomic Features
~~~~~~~~~~~~~~~~~~~~~~~

Extract autonomic nervous system indicators from ECG signals, including heart rate variability and autonomic balance measures.

.. automodule:: vitalDSP.feature_engineering.ecg_autonomic_features
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

PPG Autonomic Features
~~~~~~~~~~~~~~~~~~~~~~~

Extract autonomic nervous system indicators from PPG signals, focusing on pulse rate variability and vascular tone measures.

.. automodule:: vitalDSP.feature_engineering.ppg_autonomic_features
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Synchronization Features
========================

ECG-PPG Synchronization Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze the synchronization and timing relationships between ECG and PPG signals for comprehensive cardiovascular assessment.

.. automodule:: vitalDSP.feature_engineering.ecg_ppg_synchronyzation_features
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Light Source Features
=====================

PPG Light Features
~~~~~~~~~~~~~~~~~~

Extract features specific to different light wavelengths in PPG signals, useful for multi-wavelength PPG analysis.

.. automodule:: vitalDSP.feature_engineering.ppg_light_features
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Usage Examples
==============

Basic Feature Extraction
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from vitalDSP.feature_engineering.morphology_features import MorphologyFeatures
   from vitalDSP.feature_engineering.ecg_autonomic_features import ECGAutonomicFeatures
   
   # Extract morphological features
   morph = MorphologyFeatures(ecg_signal, sampling_rate)
   morph_features = morph.extract_features()
   
   # Extract autonomic features
   autonomic = ECGAutonomicFeatures(ecg_signal, sampling_rate)
   autonomic_features = autonomic.extract_autonomic_features()

Multi-Signal Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from vitalDSP.feature_engineering.ecg_ppg_synchronyzation_features import ECGPPSynchronizationFeatures
   
   # Analyze ECG-PPG synchronization
   sync = ECGPPSynchronizationFeatures(ecg_signal, ppg_signal, sampling_rate)
   sync_features = sync.extract_synchronization_features()

PPG Light Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from vitalDSP.feature_engineering.ppg_light_features import PPGLightFeatures
   
   # Analyze multi-wavelength PPG
   light = PPGLightFeatures(red_ppg, ir_ppg, sampling_rate)
   light_features = light.extract_light_features()

