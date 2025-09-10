Physiological Features
====================

This section covers the comprehensive physiological feature extraction capabilities provided by the VitalDSP library. These methods focus on analyzing physiological signals to extract meaningful features for healthcare applications, including time-domain, frequency-domain, and nonlinear analysis techniques.

Overview
========

The physiological features module provides a comprehensive suite of tools for analyzing physiological signals such as ECG, PPG, and other vital signs. The module is organized into several categories:

* **Time Domain Analysis**: Statistical and temporal features
* **Frequency Domain Analysis**: Spectral and power features  
* **HRV Analysis**: Heart rate variability metrics
* **Nonlinear Analysis**: Complexity and entropy measures
* **Morphological Analysis**: Waveform shape and structure
* **Cross-Signal Analysis**: Multi-signal relationships

Time Domain Features
====================

Time domain analysis focuses on statistical and temporal characteristics of physiological signals.

Time Domain Features
~~~~~~~~~~~~~~~~~~~~

Statistical and temporal feature extraction from physiological signals.

.. automodule:: vitalDSP.physiological_features.time_domain
   :members:
   :undoc-members:
   :private-members:
   :exclude-members: __dict__, __weakref__, __module__, __annotations__
   :noindex:

Beat-to-Beat Analysis
~~~~~~~~~~~~~~~~~~~~~

Advanced technique to analyze beat-to-beat intervals and heart rate variability (HRV) analysis in ECG and PPG signals.

.. automodule:: vitalDSP.physiological_features.beat_to_beat
   :members:
   :undoc-members:
   :private-members:
   :exclude-members: __dict__, __weakref__, __module__, __annotations__
   :noindex:

Frequency Domain Features
=========================

Spectral analysis and frequency domain feature extraction.

Frequency Domain Features
~~~~~~~~~~~~~~~~~~~~~~~~~

Spectral analysis and frequency domain feature extraction from physiological signals.

.. automodule:: vitalDSP.physiological_features.frequency_domain
   :members:
   :undoc-members:
   :private-members:
   :exclude-members: __dict__, __weakref__, __module__, __annotations__
   :noindex:

HRV Analysis
============

Comprehensive heart rate variability analysis including time-domain, frequency-domain, and nonlinear metrics.

HRV Features
~~~~~~~~~~~~

Comprehensive heart rate variability analysis including time-domain, frequency-domain, and nonlinear metrics.

.. automodule:: vitalDSP.physiological_features.hrv_analysis
   :members:
   :undoc-members:
   :private-members:
   :exclude-members: __dict__, __weakref__, __module__, __annotations__
   :noindex:

Nonlinear Analysis
==================

Nonlinear dynamics and complexity measures for physiological signals.

Nonlinear Features
~~~~~~~~~~~~~~~~~~

Nonlinear dynamics and complexity measures for physiological signal analysis.

.. automodule:: vitalDSP.physiological_features.nonlinear
   :members:
   :undoc-members:
   :private-members:
   :exclude-members: __dict__, __weakref__, __module__, __annotations__
   :noindex:

Morphological Analysis
======================

Waveform morphology and shape analysis for physiological signals.

Waveform Morphology
~~~~~~~~~~~~~~~~~~~

Waveform morphology and shape analysis for physiological signals.

.. automodule:: vitalDSP.physiological_features.waveform
   :members:
   :undoc-members:
   :private-members:
   :exclude-members: __dict__, __weakref__, __module__, __annotations__
   :noindex:

Cross-Signal Analysis
=====================

Multi-signal analysis and relationship detection.

Cross Correlation Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Techniques for analyzing the cross correlation between two physiological signals.

.. automodule:: vitalDSP.physiological_features.cross_correlation
   :members:
   :undoc-members:
   :private-members:
   :exclude-members: __dict__, __weakref__, __module__, __annotations__
   :noindex:

Cross-Signal Analysis
~~~~~~~~~~~~~~~~~~~~~

Techniques for analyzing the relationships between multiple signals over time, useful for studying interactions and dependencies.

.. automodule:: vitalDSP.physiological_features.cross_signal_analysis
   :members:
   :undoc-members:
   :private-members:
   :exclude-members: __dict__, __weakref__, __module__, __annotations__
   :noindex:

Coherence Analysis
~~~~~~~~~~~~~~~~~~

Techniques for analyzing the coherence between two physiological signals.

.. automodule:: vitalDSP.physiological_features.coherence_analysis
   :members:
   :undoc-members:
   :private-members:
   :exclude-members: __dict__, __weakref__, __module__, __annotations__
   :noindex:

Signal Processing Features
==========================

Signal segmentation, power analysis, and energy features.

Signal Segmentation
~~~~~~~~~~~~~~~~~~~

Techniques for segmenting physiological signals into meaningful intervals.

.. automodule:: vitalDSP.physiological_features.signal_segmentation
   :members:
   :undoc-members:
   :private-members:
   :exclude-members: __dict__, __weakref__, __module__, __annotations__
   :noindex:

Signal Power Analysis
~~~~~~~~~~~~~~~~~~~~~

Power analysis and energy features for physiological signals.

.. automodule:: vitalDSP.physiological_features.signal_power_analysis
   :members:
   :undoc-members:
   :private-members:
   :exclude-members: __dict__, __weakref__, __module__, __annotations__
   :noindex:

Energy Analysis
~~~~~~~~~~~~~~~

Energy analysis and power spectral features for physiological signals.

.. automodule:: vitalDSP.physiological_features.energy_analysis
   :members:
   :undoc-members:
   :private-members:
   :exclude-members: __dict__, __weakref__, __module__, __annotations__
   :noindex:

Envelope Detection
~~~~~~~~~~~~~~~~~~

Envelope detection and amplitude modulation analysis.

.. automodule:: vitalDSP.physiological_features.envelope_detection
   :members:
   :undoc-members:
   :private-members:
   :exclude-members: __dict__, __weakref__, __module__, __annotations__
   :noindex:

Trend Analysis
~~~~~~~~~~~~~~

Trend analysis and long-term variability measures.

.. automodule:: vitalDSP.physiological_features.trend_analysis
   :members:
   :undoc-members:
   :private-members:
   :exclude-members: __dict__, __weakref__, __module__, __annotations__
   :noindex:

Signal Change Detection
~~~~~~~~~~~~~~~~~~~~~~~

Change point detection and signal transition analysis.

.. automodule:: vitalDSP.physiological_features.signal_change_detection
   :members:
   :undoc-members:
   :private-members:
   :exclude-members: __dict__, __weakref__, __module__, __annotations__
   :noindex:

Ensemble-Based Feature Extraction
---------------------------------
Extract features from signals using ensemble methods, which aggregate multiple models or analyses to improve robustness and accuracy.

.. automodule:: vitalDSP.physiological_features.ensemble_based_feature_extraction
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Energy Analysis
---------------------
Techniques for analyzing the energy-related features from physiological signals (ECG, PPG, EEG).

.. automodule:: vitalDSP.physiological_features.energy_analysis
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Envelope Detection
------------------
Methods for detecting the envelope of a signal, which captures the signal’s amplitude variations over time, often used in audio and biomedical signal processing.

.. automodule:: vitalDSP.physiological_features.envelope_detection
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Frequency Domain
---------------------
Techniques for analyzing the frequency domain features from physiological signals (ECG, PPG, EEG).

.. automodule:: vitalDSP.physiological_features.frequency_domain
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Nonlinear Analysis
---------------------
Techniques for analyzing the nonlinear features from physiological signals (ECG, PPG, EEG).

.. automodule:: vitalDSP.physiological_features.nonlinear
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Signal Change Detection
-----------------------
Detect abrupt changes in signals, which may signify transitions between different physiological states or responses.

.. automodule:: vitalDSP.physiological_features.signal_change_detection
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Signal Power Analysis
---------------------
Analyze the power of a signal over time, which can provide insights into the energy distribution and overall activity within the signal.

.. automodule:: vitalDSP.physiological_features.signal_power_analysis
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Signal Segmentation
-------------------
Segment signals into meaningful parts for further analysis, a crucial step in processing long-duration biomedical signals.

.. automodule:: vitalDSP.physiological_features.signal_segmentation
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Time Domain Analysis
---------------------
Techniques for analyzing the time domain features from physiological signals (ECG, PPG, EEG).

.. automodule:: vitalDSP.physiological_features.time_domain
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Trend Analysis
--------------
Analyze trends within signals over time, identifying underlying patterns and long-term behaviors that may be of clinical significance.

.. automodule:: vitalDSP.physiological_features.trend_analysis
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Waveform Analysis
---------------------
Techniques for analyzing the waveform features from physiological signals (ECG, PPG, EEG).

.. automodule:: vitalDSP.physiological_features.waveform
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex: