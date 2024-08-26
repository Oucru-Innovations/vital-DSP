Respiratory Analysis
====================

This section covers the respiratory analysis techniques provided by the VitalDSP library. These methods are designed to estimate respiratory rates, perform multimodal analysis, preprocess signals, and detect sleep apnea.

Estimation of Respiratory Rate (RR)
-----------------------------------
Techniques for estimating respiratory rate using various methods including time-domain, frequency-domain, peak detection, and FFT-based approaches.

### FFT-Based Respiratory Rate Estimation
Estimate respiratory rate using FFT-based methods, which analyze the frequency components of the signal.

.. automodule:: vitalDSP.respiratory_analysis.estimate_rr.fft_based_rr
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

### Frequency Domain Respiratory Rate Estimation
Estimate respiratory rate in the frequency domain, leveraging spectral analysis techniques.

.. automodule:: vitalDSP.respiratory_analysis.estimate_rr.frequency_domain_rr
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

### Peak Detection Respiratory Rate Estimation
Estimate respiratory rate using peak detection algorithms, focusing on identifying inhalation and exhalation peaks in the signal.

.. automodule:: vitalDSP.respiratory_analysis.estimate_rr.peak_detection_rr
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

### Time Domain Respiratory Rate Estimation
Estimate respiratory rate using time-domain methods, which analyze the signal’s temporal characteristics.

.. automodule:: vitalDSP.respiratory_analysis.estimate_rr.time_domain_rr
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Multimodal Fusion
-----------------
Techniques for combining different signals and modalities to enhance the accuracy and reliability of respiratory analysis.

### Multimodal Analysis
Perform comprehensive analysis by fusing multiple data sources for a more robust respiratory rate estimation.

.. automodule:: vitalDSP.respiratory_analysis.fusion.multimodal_analysis
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

### PPG and ECG Fusion
Fuse Photoplethysmogram (PPG) and Electrocardiogram (ECG) signals to improve respiratory rate estimation and other analyses.

.. automodule:: vitalDSP.respiratory_analysis.fusion.ppg_ecg_fusion
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

### Respiratory and Cardiac Signal Fusion
Combine respiratory and cardiac signals to enhance analysis and gain insights into the interactions between respiratory and cardiac activities.

.. automodule:: vitalDSP.respiratory_analysis.fusion.respiratory_cardiac_fusion
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Preprocessing
-------------
Techniques for preprocessing respiratory signals before analysis, including noise reduction and general preprocessing methods.

### Noise Reduction
Apply noise reduction techniques to respiratory signals to improve the quality of analysis.

.. automodule:: vitalDSP.respiratory_analysis.preprocess.noise_reduction
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

### Signal Preprocessing
Preprocess respiratory signals to prepare them for subsequent analysis, including normalization and filtering.

.. automodule:: vitalDSP.respiratory_analysis.preprocess.preprocess
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Sleep Apnea Detection
---------------------
Methods for detecting sleep apnea events by analyzing respiratory signals for patterns indicative of pauses in breathing.

### Amplitude Threshold Sleep Apnea Detection
Detect sleep apnea events based on amplitude thresholds in respiratory signals.

.. automodule:: vitalDSP.respiratory_analysis.sleep_apnea_detection.amplitude_threshold
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

### Pause Detection for Sleep Apnea
Identify pauses in respiratory signals that may indicate sleep apnea events, critical for early diagnosis and intervention.

.. automodule:: vitalDSP.respiratory_analysis.sleep_apnea_detection.pause_detection
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:
