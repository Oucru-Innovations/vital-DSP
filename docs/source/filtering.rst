Filtering
=======================

This section provides documentation for the filtering techniques used in the VitalDSP library. Each submodule offers different functionalities for processing and improving the quality of signals, particularly in healthcare and biomedical applications.

Signal Filtering
----------------
This module provides basic filtering techniques such as Butterworth, Chebyshev, and moving average filters, which are essential for preprocessing signals by removing noise and other unwanted components.

.. automodule:: vitalDSP.filtering.signal_filtering
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Artifact Removal
----------------
This module focuses on identifying and removing artifacts from signals, ensuring cleaner and more accurate data for analysis. Methods include adaptive thresholding, wavelet-based removal, and iterative techniques.

.. automodule:: vitalDSP.filtering.artifact_removal
    :members:
    :undoc-members:
    :private-members:
    :exclude-members: __dict__, __weakref__, __module__, __annotations__
    :noindex:

Advanced Signal Filtering
-------------------------
Advanced filtering techniques such as adaptive filtering, Kalman filtering, and more complex algorithms that are suited for dynamic and non-linear systems.

.. automodule:: vitalDSP.filtering.advanced_signal_filtering
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
