Physiological Features
====================

This section covers the comprehensive physiological feature extraction capabilities provided by the VitalDSP library. These methods focus on analyzing physiological signals to extract meaningful features for healthcare applications, including time-domain, frequency-domain, and nonlinear analysis techniques with detailed clinical interpretations.

Overview
========

The physiological features module provides a comprehensive suite of tools for analyzing physiological signals such as ECG, PPG, and other vital signs. The module is organized into several categories and includes clinical interpretation guidelines based on extensive research and clinical validation:

* **Time Domain Analysis**: Statistical and temporal features with clinical significance
* **Frequency Domain Analysis**: Spectral and power features with autonomic nervous system insights
* **HRV Analysis**: Heart rate variability metrics for stress, infection, and cardiovascular health assessment
* **Nonlinear Analysis**: Complexity and entropy measures for autonomic function evaluation
* **Morphological Analysis**: Waveform shape and structure analysis for disease detection
* **Cross-Signal Analysis**: Multi-signal relationships for comprehensive health assessment

Clinical Applications
~~~~~~~~~~~~~~~~~~~~~

The physiological features extracted by VitalDSP are designed to provide insights into:

* **Cardiovascular Health**: Assessment of heart function, blood pressure, and vascular compliance
* **Autonomic Nervous System**: Evaluation of sympathetic and parasympathetic balance
* **Stress and Infection Detection**: Early identification of physiological stress and infection severity
* **Disease Progression**: Monitoring of chronic conditions and treatment response
* **Sleep and Respiratory Health**: Analysis of breathing patterns and sleep quality
* **Mental Health**: Assessment of stress, anxiety, and cognitive load

Key Features
~~~~~~~~~~~~

* **Comprehensive HRV Analysis**: 50+ heart rate variability metrics with clinical interpretation
* **Signal-Specific Processing**: Optimized algorithms for ECG, PPG, EEG, and respiratory signals
* **Real-Time Processing**: Optimized for live monitoring and clinical applications
* **Clinical Validation**: Features validated on clinical datasets and real-world applications
* **Automated Interpretation**: Built-in clinical significance assessment and health indicators

Clinical Interpretation Guidelines
===================================

The following sections provide detailed clinical interpretation guidelines for physiological features, based on extensive research and clinical validation. These guidelines help healthcare professionals understand the clinical significance of extracted features.

ECG Signal Analysis
~~~~~~~~~~~~~~~~~~~

ECG (electrocardiogram) signals provide detailed insights into cardiac function, disease progression, and infection severity. Key features include:

**Heart Rate Variability (HRV) from ECG**
    ECG is the gold standard for measuring HRV, providing accurate assessment of autonomic nervous system balance:

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

**ECG Morphology Features**
    ECG morphology provides crucial information about cardiac health and stress:

    * **P-Wave Analysis**:
        - Normal Range: 80-110 ms duration
        - Clinical Significance: Prolonged P-wave duration suggests atrial dilation, often associated with heart failure or infections affecting the heart
        - Changes in P-wave amplitude may indicate pericarditis (inflammation of the pericardium)

    * **PR Interval**:
        - Normal Range: 120-200 ms
        - Clinical Significance: Prolonged PR interval may suggest electrolyte imbalances or autonomic dysfunction, often seen in sepsis

    * **QRS Complex**:
        - Normal Range: 80-120 ms duration
        - Clinical Significance: Widened QRS complexes suggest conduction delays, often caused by myocardial ischemia, bundle branch blocks, or ventricular hypertrophy

    * **ST Segment**:
        - Normal Range: 80-120 ms duration
        - Clinical Significance: ST elevation can indicate myocarditis, pericarditis, or acute myocardial infarction
        - ST depression suggests ischemia, which can occur during sepsis, shock, or cardiac complications

    * **QT Interval**:
        - Normal Range: 350-450 ms (corrected for heart rate)
        - Clinical Significance: Prolonged QT interval indicates risk of life-threatening arrhythmias such as torsades de pointes
        - Can be triggered by electrolyte imbalances, medications, or infection-induced stress

**Arrhythmias and Abnormal Rhythms**
    Certain arrhythmias can predict disease progression:

    * **Atrial Fibrillation (AFib)**: Irregular atrial contractions, often seen in patients with sepsis, heart failure, or systemic inflammation
    * **Ventricular Tachycardia (VTach)**: Fast, abnormal ventricular rhythms, suggesting cardiac decompensation
    * **Bradycardia**: Can occur in critically ill patients, particularly those in septic shock or with autonomic dysfunction

PPG Signal Analysis
~~~~~~~~~~~~~~~~~~~

PPG (photoplethysmography) signals provide insights into cardiovascular health, autonomic function, and stress levels:

**Heart Rate Variability from PPG**
    PPG-derived HRV features can reveal autonomic nervous system balance:

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

**PPG Morphology Features**
    PPG waveform shape provides insights into cardiovascular and respiratory health:

    * **Systolic and Diastolic Durations**:
        - Normal Ratio: 0.6-0.8 (Systolic:Diastolic)
        - Clinical Significance: Longer systolic durations indicate reduced arterial compliance
        - Alterations may reflect arterial stiffness, hypertension, or atherosclerosis

    * **Systolic Amplitude and Variability**:
        - Clinical Significance: Decrease in systolic amplitude suggests poor perfusion
        - Patients with systemic infections (sepsis) may show reduced systolic amplitude due to decreased cardiac output

    * **Pulse Wave Transit Time (PWTT)**:
        - Normal Range: 100-300 ms
        - Clinical Significance: Shorter PWTT indicates increased arterial stiffness
        - Related to hypertension, atherosclerosis, or cardiovascular stress aggravated by infection

**Respiratory and Autonomic Features**
    PPG signals can reveal respiratory patterns important for assessing respiratory distress:

    * **Respiratory Sinus Arrhythmia (RSA)**:
        - Normal Range: 5-20% variation during respiration
        - Clinical Significance: Reduced RSA indicates poor autonomic control, often associated with stress or chronic disease
        - Patients with respiratory infections may exhibit reduced RSA

    * **Respiratory Rate Variability (RRV)**:
        - Normal Range: 0.1-0.3 Hz
        - Clinical Significance: Increased RRV is often seen in patients with respiratory infections, pneumonia, or lung conditions

**Infection and Sepsis Detection**
    Early detection of sepsis or infection is critical:

    * **Sepsis Indicators**: Low HRV (low SDNN or pNN50) combined with high LF/HF ratio is often associated with sepsis
    * **Cytokine Storm**: In severe infections like COVID-19, cytokine storm can result in acute drop in HRV due to overwhelming stress

Time Domain Features
====================

Time domain analysis focuses on statistical and temporal characteristics of physiological signals with clinical significance.

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

Advanced Nonlinear Features
============================

State-of-the-art nonlinear dynamics and information-theoretic methods for advanced physiological signal analysis.

Multi-Scale Entropy Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-scale entropy (MSE) analysis quantifies signal complexity across multiple temporal scales, providing insights into physiological regulation and system health.

**Key Features:**
    * Standard MSE, Composite MSE (CMSE), and Refined Composite MSE (RCMSE)
    * KD-tree accelerated sample entropy calculation (O(N log N))
    * Complexity Index for single-value assessment
    * Fuzzy entropy option for noisy signals

**Clinical Applications:**
    * Cardiac arrhythmia detection and classification
    * Aging assessment and cardiovascular health evaluation
    * Autonomic nervous system function assessment
    * Disease progression monitoring (heart failure, diabetes)

**Usage Example:**

.. code-block:: python

    from vitalDSP.physiological_features.advanced_entropy import MultiScaleEntropy

    # Analyze RR interval complexity
    mse = MultiScaleEntropy(rr_intervals, max_scale=20, m=2, r=0.15)
    entropy_curve = mse.compute_rcmse()
    complexity_index = mse.get_complexity_index(entropy_curve, scale_range=(1, 15))

    # Interpret results
    if complexity_index > 30:
        print("Healthy complexity profile")
    elif complexity_index > 15:
        print("Reduced complexity - monitoring recommended")
    else:
        print("Severely reduced complexity - clinical attention needed")

.. automodule:: vitalDSP.physiological_features.advanced_entropy
   :members:
   :undoc-members:
   :private-members:
   :exclude-members: __dict__, __weakref__, __module__, __annotations__
   :noindex:

Symbolic Dynamics Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symbolic dynamics transforms continuous physiological signals into discrete symbol sequences for pattern analysis and complexity quantification.

**Key Features:**
    * Multiple symbolization methods: 0V (HRV-specific), Quantile, SAX, Threshold
    * Shannon entropy, Renyi entropy, Permutation entropy
    * Word distribution and forbidden words analysis
    * Transition matrix computation for Markov analysis

**Clinical Applications:**
    * Heart rate variability pattern classification
    * Autonomic regulation assessment
    * Arrhythmia detection (atrial fibrillation screening)
    * Sleep stage classification from EEG

**Usage Example:**

.. code-block:: python

    from vitalDSP.physiological_features.symbolic_dynamics import SymbolicDynamics

    # HRV symbolic analysis with 0V method
    sd = SymbolicDynamics(rr_intervals, n_symbols=4, word_length=3, method='0V')

    # Compute multiple metrics
    shannon = sd.compute_shannon_entropy()
    forbidden = sd.detect_forbidden_words()
    perm_ent = sd.compute_permutation_entropy(order=3)

    print(f"Shannon Entropy: {shannon['normalized_entropy']:.3f}")
    print(f"Forbidden Words: {forbidden['forbidden_percentage']:.1f}%")
    print(f"Interpretation: {forbidden['interpretation']}")

.. automodule:: vitalDSP.physiological_features.symbolic_dynamics
   :members:
   :undoc-members:
   :private-members:
   :exclude-members: __dict__, __weakref__, __module__, __annotations__
   :noindex:

Transfer Entropy Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Transfer entropy quantifies directional information flow between coupled physiological signals, revealing causal relationships and coupling dynamics.

**Key Features:**
    * KNN-based transfer entropy estimation (Kraskov-Stögbauer-Grassberger method)
    * Bidirectional coupling analysis
    * Time-delayed TE for temporal dynamics assessment
    * Statistical significance testing with surrogate data

**Clinical Applications:**
    * Cardio-respiratory coupling analysis
    * Brain-heart interaction assessment
    * Autonomic nervous system evaluation
    * Multi-organ system dynamics monitoring

**Usage Example:**

.. code-block:: python

    from vitalDSP.physiological_features.transfer_entropy import TransferEntropy

    # Analyze respiration → heart rate coupling
    te = TransferEntropy(respiration, heart_rate, k=2, l=2, delay=1, k_neighbors=3)

    # Bidirectional analysis
    bidirectional = te.compute_bidirectional_te()
    print(f"Respiration → HR: {bidirectional['te_forward']:.3f}")
    print(f"HR → Respiration: {bidirectional['te_backward']:.3f}")
    print(f"Coupling type: {bidirectional['interpretation']}")

    # Statistical significance
    significance = te.test_significance(n_surrogates=1000)
    print(f"p-value: {significance['p_value']:.4f} {significance['significance']}")

    # Find optimal coupling delay
    delayed = te.compute_time_delayed_te(max_delay=10)
    print(f"Optimal delay: {delayed['optimal_delay']} seconds")

.. automodule:: vitalDSP.physiological_features.transfer_entropy
   :members:
   :undoc-members:
   :private-members:
   :exclude-members: __dict__, __weakref__, __module__, __annotations__
   :noindex:

Advanced Features Guide
~~~~~~~~~~~~~~~~~~~~~~~~

For comprehensive documentation including detailed mathematical formulations, code explanations, clinical validation results, and performance optimization guidelines, see the `Advanced Features Guide <../../ADVANCED_FEATURES_GUIDE.md>`_.

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