Feature Engineering
====================

This section covers the comprehensive feature extraction techniques provided by the VitalDSP library for analyzing physiological signals such as ECG and PPG. These methods help to derive meaningful features that describe various morphological, autonomic, and synchronization characteristics of the signals, with detailed clinical interpretations and applications.

Overview
========

The feature engineering module provides specialized tools for extracting clinically relevant features from physiological signals. These features are designed to provide insights into cardiovascular health, autonomic nervous system function, and disease progression.

**Key Capabilities**
    * **Morphological Features**: Waveform shape and structure analysis with clinical significance
    * **Autonomic Features**: Heart rate variability and autonomic nervous system indicators
    * **Synchronization Features**: Multi-signal correlation and timing analysis
    * **Light Source Features**: PPG-specific features for different light wavelengths
    * **Volume Features**: Blood volume and pressure-related characteristics
    * **Clinical Interpretation**: Built-in clinical significance assessment and health indicators

**Clinical Applications**
    * **Cardiovascular Health Assessment**: Evaluation of heart function and vascular compliance
    * **Stress and Infection Detection**: Early identification of physiological stress and infection severity
    * **Disease Progression Monitoring**: Tracking of chronic conditions and treatment response
    * **Sleep and Respiratory Health**: Analysis of breathing patterns and sleep quality
    * **Mental Health Assessment**: Evaluation of stress, anxiety, and cognitive load

**Feature Categories**
    * **Time-Domain Features**: Statistical and temporal characteristics
    * **Frequency-Domain Features**: Spectral and power characteristics
    * **Nonlinear Features**: Complexity and entropy measures
    * **Morphological Features**: Waveform shape and structure
    * **Cross-Signal Features**: Multi-signal relationships and synchronization

Clinical Feature Interpretation
================================

The following sections provide detailed clinical interpretation guidelines for extracted features, based on extensive research and clinical validation. These guidelines help healthcare professionals understand the clinical significance of extracted features.

ECG Morphological Features
~~~~~~~~~~~~~~~~~~~~~~~~~~

ECG morphological features provide crucial information about cardiac health and disease progression:

**P-Wave Features**
    * **P-Wave Duration**: 
        - Normal Range: 80-110 ms
        - Clinical Significance: Prolonged duration suggests atrial dilation, often associated with heart failure or infections affecting the heart
        - Changes in amplitude may indicate pericarditis (inflammation of the pericardium)

**PR Interval Features**
    * **PR Interval Duration**:
        - Normal Range: 120-200 ms
        - Clinical Significance: Prolonged PR interval may suggest electrolyte imbalances or autonomic dysfunction, often seen in sepsis
        - Correlates with atrioventricular (AV) nodal conduction

**QRS Complex Features**
    * **QRS Duration**:
        - Normal Range: 80-120 ms
        - Clinical Significance: Widened QRS complexes suggest conduction delays, often caused by myocardial ischemia, bundle branch blocks, or ventricular hypertrophy
        - Correlated with ventricular conduction and conditions like bundle branch blocks

**ST Segment Features**
    * **ST Segment Duration**:
        - Normal Range: 80-120 ms
        - Clinical Significance: ST elevation can indicate myocarditis, pericarditis, or acute myocardial infarction
        - ST depression suggests ischemia, which can occur during sepsis, shock, or cardiac complications

**QT Interval Features**
    * **QT Interval Duration**:
        - Normal Range: 350-450 ms (corrected for heart rate)
        - Clinical Significance: Prolonged QT interval indicates risk of life-threatening arrhythmias such as torsades de pointes
        - Can be triggered by electrolyte imbalances, medications, or infection-induced stress

PPG Morphological Features
~~~~~~~~~~~~~~~~~~~~~~~~~~

PPG morphological features provide insights into cardiovascular and respiratory health:

**Systolic and Diastolic Features**
    * **Systolic Duration**:
        - Normal Ratio: 0.6-0.8 (Systolic:Diastolic)
        - Clinical Significance: Longer systolic durations indicate reduced arterial compliance
        - Alterations may reflect arterial stiffness, hypertension, or atherosclerosis

**Amplitude Features**
    * **Systolic Amplitude**:
        - Clinical Significance: Decrease in systolic amplitude suggests poor perfusion
        - Patients with systemic infections (sepsis) may show reduced systolic amplitude due to decreased cardiac output

**Pulse Wave Features**
    * **Pulse Wave Transit Time (PWTT)**:
        - Normal Range: 100-300 ms
        - Clinical Significance: Shorter PWTT indicates increased arterial stiffness
        - Related to hypertension, atherosclerosis, or cardiovascular stress aggravated by infection

**Respiratory Features**
    * **Respiratory Sinus Arrhythmia (RSA)**:
        - Normal Range: 5-20% variation during respiration
        - Clinical Significance: Reduced RSA indicates poor autonomic control, often associated with stress or chronic disease
        - Patients with respiratory infections may exhibit reduced RSA

Heart Rate Variability Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HRV features provide insights into autonomic nervous system function and health status:

**Time-Domain HRV Features**
    * **SDNN (Standard Deviation of NN Intervals)**:
        - Normal Range: 20-50 ms (1 min), 50-150 ms (5 min)
        - Clinical Significance: Decreasing SDNN indicates reduced HRV, reflecting stress, infection, or autonomic dysfunction
        - Low SDNN is associated with increased mortality in sepsis, cardiac dysfunction, and ARDS

    * **RMSSD (Root Mean Square of Successive Differences)**:
        - Normal Range: 20-50 ms (1 min), 30-60 ms (5 min)
        - Clinical Significance: Lower RMSSD suggests parasympathetic withdrawal, common in infections and sepsis
        - Indicates parasympathetic dysfunction and increased sympathetic dominance

    * **pNN50 (Proportion of NN Intervals differing by more than 50 ms)**:
        - Normal Range: 10-40% (1 min), 15-45% (5 min)
        - Clinical Significance: Decrease indicates early autonomic nervous system imbalance
        - Common in chronic diseases or infections

**Frequency-Domain HRV Features**
    * **LF Power (Low Frequency)**:
        - Normal Range: 300-1200 ms²
        - Clinical Significance: Increased LF power can indicate elevated stress or infection levels
        - In sepsis or systemic infections, sympathetic activation may increase LF power

    * **HF Power (High Frequency)**:
        - Normal Range: 200-1000 ms²
        - Clinical Significance: Reduced HF power suggests stress, fatigue, or infection
        - In chronic or acute illness, HF power may drop due to reduced parasympathetic influence

    * **LF/HF Ratio**:
        - Normal Range: 0.5-2.0
        - Clinical Significance: Higher ratio indicates sympathetic dominance (stress, acute infection)
        - In infectious diseases or sepsis, higher LF/HF ratio indicates autonomic imbalance

Morphological Features
======================

Morphology Features
~~~~~~~~~~~~~~~~~~~

Techniques to extract morphological features from physiological waveforms, including the detection of peaks, troughs, and various segments in ECG and PPG signals with clinical interpretation.

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

