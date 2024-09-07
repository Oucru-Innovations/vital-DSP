Signal Quality Assessment
==========================

This section covers the signal quality assessment techniques provided by the VitalDSP library. These methods are designed to evaluate and improve the quality of biomedical signals by estimating signal-to-noise ratios, detecting and removing artifacts, and separating sources.

Adaptive SNR Estimation
-----------------------
Methods for adaptively estimating the Signal-to-Noise Ratio (SNR) of signals, allowing for real-time adjustments to improve signal quality.

.. automodule:: vitalDSP.signal_quality_assessment.adaptive_snr_estimation
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Artifact Detection and Removal
------------------------------
Techniques for detecting and removing artifacts from signals, crucial for maintaining the integrity of biomedical signal analysis.

.. automodule:: vitalDSP.signal_quality_assessment.artifact_detection_removal
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Blind Source Separation
-----------------------
Methods for separating mixed signals into their original, independent sources without prior knowledge of the mixing process, often used in EEG and ECG analysis.

.. automodule:: vitalDSP.signal_quality_assessment.blind_source_separation
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Multi-Modal Artifact Detection
------------------------------
Techniques for detecting artifacts across multiple modalities, ensuring that combined signals are free from noise and other distortions.

.. automodule:: vitalDSP.signal_quality_assessment.multi_modal_artifact_detection
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

SNR Computation
---------------
Functions for computing the Signal-to-Noise Ratio (SNR) of signals, which is a critical metric in assessing the quality of biomedical signals.

.. automodule:: vitalDSP.signal_quality_assessment.snr_computation
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Signal Quality Index
--------------------
This module provides methods to calculate various signal quality indices (SQI), which help in assessing the reliability and usability of the recorded signals.

.. automodule:: vitalDSP.signal_quality_assessment.signal_quality_index
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Signal Quality
--------------
Comprehensive tools for assessing and enhancing the overall quality of signals. This includes both objective measures and methods for improving signal integrity.

.. automodule:: vitalDSP.signal_quality_assessment.signal_quality
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:
