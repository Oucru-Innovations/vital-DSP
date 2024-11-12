Utils
=====

This section covers various utility functions and modules provided by the VitalDSP library. These utilities are essential for performing a wide range of operations, from data normalization to the synthesis of signals, and supporting the core signal processing tasks.

Overview
========

The utils module provides comprehensive utility functions for:

* **Peak Detection**: Advanced algorithms for identifying peaks in physiological signals
* **Data Synthesis**: Generation of synthetic physiological signals for testing and development
* **Normalization**: Various normalization and scaling techniques
* **Interpolation**: Data interpolation and gap filling methods
* **Wavelets**: Mother wavelet functions for wavelet transforms
* **Machine Learning**: Loss functions, attention weights, and convolutional kernels

Peak Detection
==============

Peak Detection
~~~~~~~~~~~~~~

Identify peaks in the signal, which are often indicative of significant events or features, such as heartbeats or respiratory cycles.

.. automodule:: vitalDSP.utils.peak_detection
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Data Synthesis
==============

Synthesize Data
~~~~~~~~~~~~~~~

Methods for generating synthetic data, which can be used for training models, testing algorithms, or simulating signal processing scenarios.

.. automodule:: vitalDSP.utils.synthesize_data
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Normalization and Scaling
=========================

Normalization
~~~~~~~~~~~~~

Techniques and functions for normalizing data, ensuring that signals have consistent scales, which is essential for accurate analysis and model training.

.. automodule:: vitalDSP.utils.normalization
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Scaler
~~~~~~

Functions for scaling data, which can include methods for both normalization and standardization to prepare signals for analysis or machine learning.

.. automodule:: vitalDSP.utils.scaler
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Data Interpolation
==================

Interpolations
~~~~~~~~~~~~~~

Functions for interpolating and filling gaps in data, essential for handling missing values and irregular sampling.

.. automodule:: vitalDSP.utils.interpolations
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Wavelet Functions
=================

Mother Wavelets
~~~~~~~~~~~~~~~

Functions related to mother wavelets, which are used as the basis for wavelet transforms, crucial for time-frequency analysis.

.. automodule:: vitalDSP.utils.mother_wavelets
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Machine Learning Utilities
==========================

Attention Weights
~~~~~~~~~~~~~~~~~

Functions and methods related to calculating and applying attention weights, commonly used in machine learning models to emphasize important parts of the data.

.. automodule:: vitalDSP.utils.attention_weights
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Loss Functions
~~~~~~~~~~~~~~~

Various loss functions used in training machine learning models, particularly in the context of signal processing tasks.

.. automodule:: vitalDSP.utils.loss_functions
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Convolutional Kernels
~~~~~~~~~~~~~~~~~~~~~

Functions for creating and applying convolutional kernels, which are essential in convolutional neural networks and other filtering tasks.

.. automodule:: vitalDSP.utils.convolutional_kernels
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Common Utilities
================

Common Utilities
~~~~~~~~~~~~~~~~

A collection of common utility functions that provide general-purpose support across various signal processing tasks.

.. automodule:: vitalDSP.utils.common
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Usage Examples
==============

Peak Detection
~~~~~~~~~~~~~~

.. code-block:: python

   from vitalDSP.utils.peak_detection import PeakDetection
   
   # Initialize peak detector
   detector = PeakDetection(signal, sampling_rate)
   
   # Detect peaks
   peaks = detector.detect_peaks(method='savgol')
   
   # Detect specific signal types
   ecg_peaks = detector.detect_ecg_peaks()
   ppg_peaks = detector.detect_ppg_peaks()

Data Synthesis
~~~~~~~~~~~~~~

.. code-block:: python

   from vitalDSP.utils.synthesize_data import SynthesizeData
   
   synthesizer = SynthesizeData()
   
   # Generate synthetic ECG
   ecg = synthesizer.generate_synthetic_ecg(
       duration=10,
       sampling_rate=1000,
       heart_rate=72
   )
   
   # Generate synthetic PPG
   ppg = synthesizer.generate_synthetic_ppg(
       duration=10,
       sampling_rate=1000,
       heart_rate=72
   )

Normalization
~~~~~~~~~~~~~

.. code-block:: python

   from vitalDSP.utils.normalization import z_score_normalization, min_max_normalization
   from vitalDSP.utils.scaler import StandardScaler
   
   # Z-score normalization
   normalized_signal = z_score_normalization(signal)
   
   # Min-max normalization
   scaled_signal = min_max_normalization(signal)
   
   # Using scaler class
   scaler = StandardScaler()
   fitted_signal = scaler.fit_transform(signal)

Interpolation
~~~~~~~~~~~~~

.. code-block:: python

   from vitalDSP.utils.interpolations import linear_interpolation, spline_interpolation
   
   # Linear interpolation
   interpolated = linear_interpolation(signal, missing_indices)
   
   # Spline interpolation
   smooth_interpolated = spline_interpolation(signal, missing_indices)

Convolutional Kernels
---------------------
Functions for creating and applying convolutional kernels, which are essential in convolutional neural networks and other filtering tasks.

.. automodule:: vitalDSP.utils.convolutional_kernels
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:


Peak Detection
--------------
Identify peaks in the signal, which are often indicative of significant events or features, such as heartbeats or respiratory cycles.

.. automodule:: vitalDSP.utils.peak_detection
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:


Loss Functions
--------------
Various loss functions used in training machine learning models, particularly in the context of signal processing tasks.

.. automodule:: vitalDSP.utils.loss_functions
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Mother Wavelets
---------------
Functions related to mother wavelets, which are used as the basis for wavelet transforms, crucial for time-frequency analysis.

.. automodule:: vitalDSP.utils.mother_wavelets
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Normalization
-------------
Techniques and functions for normalizing data, ensuring that signals have consistent scales, which is essential for accurate analysis and model training.

.. automodule:: vitalDSP.utils.normalization
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Scaler
------
Functions for scaling data, which can include methods for both normalization and standardization to prepare signals for analysis or machine learning.

.. automodule:: vitalDSP.utils.scaler
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Synthesize Data
---------------
Methods for generating synthetic data, which can be used for training models, testing algorithms, or simulating signal processing scenarios.

.. automodule:: vitalDSP.utils.synthesize_data
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:
