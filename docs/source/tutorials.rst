Tutorials
=========

This section provides comprehensive, step-by-step tutorials to help you master VitalDSP. Each tutorial is designed to build upon previous knowledge and provide practical, hands-on experience with the library.

Tutorial Overview
=================

Our tutorials are organized by complexity and use case:

**Beginner Tutorials**
   * Basic signal processing and filtering
   * Introduction to physiological feature extraction
   * Getting started with the web application

**Intermediate Tutorials**
   * Advanced filtering techniques
   * Heart rate variability analysis
   * Respiratory signal analysis

**Advanced Tutorials**
   * Machine learning integration
   * Custom analysis pipelines
   * Performance optimization

**Specialized Tutorials**
   * Clinical research applications
   * Wearable device integration
   * Real-time monitoring systems

Tutorial 1: Basic Signal Processing
====================================

Learn the fundamentals of signal processing with VitalDSP.

**Prerequisites:**
* Python 3.8 or higher
* Basic understanding of signal processing concepts
* VitalDSP installed (see :ref:`getting_started`)

**Learning Objectives:**
* Load and visualize physiological signals
* Apply basic filtering techniques
* Extract fundamental features
* Assess signal quality

**Step 1: Installation and Setup**

.. code-block:: python

   # Install VitalDSP
   pip install vital-DSP
   
   # Import required modules
   import numpy as np
   import matplotlib.pyplot as plt
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.feature_engineering.morphology_features import PhysiologicalFeatureExtractor
   from vitalDSP.signal_quality_assessment.signal_quality_index import SignalQualityIndex

**Step 2: Load Sample Data**

.. code-block:: python

   # Generate sample ECG signal
   from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal
   
   # Parameters
   fs = 1000  # Sampling frequency (Hz)
   duration = 10  # Duration (seconds)
   heart_rate = 72  # Heart rate (BPM)
   
   # Generate synthetic ECG
   ecg_signal = generate_ecg_signal(
       sfecg=fs,
       duration=duration,
       hrmean=heart_rate,
       Anoise=0.1  # Add some noise
   )
   
   # Create time vector
   time = np.linspace(0, duration, len(ecg_signal))

**Step 3: Signal Filtering**

.. code-block:: python

   # Initialize signal filtering
   sf = SignalFiltering(ecg_signal, fs)
   
   # Apply bandpass filter (0.5-40 Hz for ECG)
   filtered_signal = sf.bandpass_filter(
       low_cut=0.5,
       high_cut=40.0,
       filter_order=4
   )
   
   # Visualize results
   plt.figure(figsize=(12, 8))
   
   plt.subplot(2, 1, 1)
   plt.plot(time, ecg_signal)
   plt.title('Original ECG Signal')
   plt.xlabel('Time (s)')
   plt.ylabel('Amplitude')
   
   plt.subplot(2, 1, 2)
   plt.plot(time, filtered_signal)
   plt.title('Filtered ECG Signal')
   plt.xlabel('Time (s)')
   plt.ylabel('Amplitude')
   
   plt.tight_layout()
   plt.show()

**Step 4: Feature Extraction**

.. code-block:: python

   # Extract physiological features
   extractor = PhysiologicalFeatureExtractor(filtered_signal, fs=fs)
   features = extractor.extract_features(signal_type="ECG")
   
   # Display key features
   print("Physiological Features:")
   print(f"QRS Duration: {features.get('qrs_duration', 'N/A'):.4f}")
   print(f"Heart Rate: {features.get('heart_rate', 'N/A'):.2f} BPM")
   print(f"QRS Amplitude: {features.get('qrs_amplitude', 'N/A'):.4f}")
   print(f"Signal Skewness: {features.get('signal_skewness', 'N/A'):.4f}")

**Step 5: Signal Quality Assessment**

.. code-block:: python

   # Assess signal quality
   sqi = SignalQualityIndex(filtered_signal)
   
   # Calculate various quality indices
   amplitude_sqi, _, _ = sqi.amplitude_variability_sqi(
       window_size=fs*5,  # 5-second windows
       step_size=fs*1,    # 1-second steps
       threshold=2
   )
   
   print(f"Signal Quality Index: {np.mean(amplitude_sqi):.4f}")

**Exercise: Try It Yourself**

1. Modify the heart rate and noise level in the synthetic signal generation
2. Experiment with different filter parameters
3. Extract additional features and compare results
4. Assess how signal quality changes with different noise levels

Tutorial 2: Heart Rate Variability Analysis
============================================

Learn to perform comprehensive HRV analysis using VitalDSP.

**Prerequisites:**
* Completion of Tutorial 1
* Understanding of heart rate variability concepts
* Basic knowledge of frequency domain analysis

**Learning Objectives:**
* Extract R-peaks from ECG signals
* Calculate RR intervals
* Perform time-domain HRV analysis
* Perform frequency-domain HRV analysis
* Interpret HRV results clinically

**Step 1: R-Peak Detection**

.. code-block:: python

   from vitalDSP.physiological_features.waveform import WaveformMorphology
   
   # Initialize waveform morphology for ECG
   wm = WaveformMorphology(filtered_signal, fs=fs, signal_type="ECG")
   
   # Detect R-peaks
   r_peaks = wm.r_peaks
   r_peak_times = r_peaks / fs  # Convert to seconds
   
   print(f"Detected {len(r_peaks)} R-peaks")
   print(f"Average heart rate: {60 * fs / np.mean(np.diff(r_peaks)):.1f} BPM")

**Step 2: RR Interval Calculation**

.. code-block:: python

   # Calculate RR intervals
   rr_intervals = np.diff(r_peaks) / fs * 1000  # Convert to milliseconds
   
   # Remove outliers (RR intervals outside physiological range)
   valid_rr = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
   
   print(f"Valid RR intervals: {len(valid_rr)}")
   print(f"RR interval range: {valid_rr.min():.1f} - {valid_rr.max():.1f} ms")

**Step 3: Time-Domain HRV Analysis**

.. code-block:: python

   from vitalDSP.physiological_features.hrv_analysis import HRVFeatures
   
   # Initialize HRV analysis
   hrv = HRVFeatures(valid_rr)
   
   # Extract time-domain features
   time_domain_features = hrv.time_domain_features()
   
   print("Time-Domain HRV Features:")
   print(f"SDNN: {time_domain_features['sdnn']:.2f} ms")
   print(f"RMSSD: {time_domain_features['rmssd']:.2f} ms")
   print(f"pNN50: {time_domain_features['pnn50']:.2f} %")

**Step 4: Frequency-Domain HRV Analysis**

.. code-block:: python

   # Extract frequency-domain features
   freq_domain_features = hrv.frequency_domain_features()
   
   print("Frequency-Domain HRV Features:")
   print(f"LF Power: {freq_domain_features['lf_power']:.2f} ms²")
   print(f"HF Power: {freq_domain_features['hf_power']:.2f} ms²")
   print(f"LF/HF Ratio: {freq_domain_features['lf_hf_ratio']:.2f}")

**Step 5: Comprehensive HRV Analysis**

.. code-block:: python

   # Perform comprehensive HRV analysis
   comprehensive_hrv = hrv.compute_all_features()
   
   # Display all features
   print("Comprehensive HRV Analysis:")
   for feature, value in comprehensive_hrv.items():
       if isinstance(value, (int, float)):
           print(f"{feature}: {value:.4f}")

**Exercise: Clinical Interpretation**

1. Compare HRV values with clinical norms
2. Analyze how different signal quality affects HRV metrics
3. Investigate the relationship between heart rate and HRV
4. Create visualizations of HRV trends over time

Tutorial 3: Respiratory Signal Analysis
========================================

Learn to analyze respiratory signals and estimate respiratory rate.

**Prerequisites:**
* Completion of Tutorial 1
* Understanding of respiratory physiology
* Basic knowledge of signal processing

**Learning Objectives:**
* Load and preprocess respiratory signals
* Apply respiratory-specific filtering
* Estimate respiratory rate using multiple methods
* Analyze breathing patterns
* Detect respiratory events

**Step 1: Load Respiratory Data**

.. code-block:: python

   from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis
   
   # Generate sample respiratory signal
   from vitalDSP.utils.data_processing.synthesize_data import generate_respiratory_signal
   
   # Parameters
   fs = 100  # Sampling frequency (Hz)
   duration = 60  # Duration (seconds)
   resp_rate = 16  # Respiratory rate (breaths per minute)
   
   # Generate synthetic respiratory signal
   resp_signal = generate_respiratory_signal(
       sf=fs,
       duration=duration,
       resp_rate=resp_rate,
       noise_level=0.05
   )
   
   # Create time vector
   time = np.linspace(0, duration, len(resp_signal))

**Step 2: Respiratory Signal Preprocessing**

.. code-block:: python

   # Initialize respiratory analysis
   resp_analysis = RespiratoryAnalysis(resp_signal, fs)
   
   # Apply respiratory-specific filtering
   filtered_resp = resp_analysis.preprocess_signal(
       detrend=True,
       normalize=True,
       filter_type='bandpass',
       low_freq=0.1,  # 0.1 Hz (6 breaths/min)
       high_freq=0.5   # 0.5 Hz (30 breaths/min)
   )

**Step 3: Respiratory Rate Estimation**

.. code-block:: python

   # Estimate respiratory rate using multiple methods
   resp_rate = resp_analysis.compute_respiratory_rate()
   
   print("Respiratory Rate Estimate:")
   print(f"Respiratory Rate: {resp_rate:.1f} breaths/min")
   print(f"True Rate: {resp_rate:.1f} breaths/min")

**Step 4: Breathing Pattern Analysis**

.. code-block:: python

   # Analyze breathing patterns
   # RespiratoryAnalysis only provides respiratory rate computation
   
   print("Respiratory Analysis Complete")

**Step 5: Respiratory Event Detection**

.. code-block:: python

   # Detect respiratory events
   events = resp_analysis.detect_respiratory_events(
       apnea_threshold=0.1,  # 10% reduction in amplitude
       hypopnea_threshold=0.3  # 30% reduction in amplitude
   )
   
   print(f"Detected {len(events['apneas'])} apnea events")
   print(f"Detected {len(events['hypopneas'])} hypopnea events")

**Exercise: Advanced Analysis**

1. Compare different respiratory rate estimation methods
2. Analyze the effect of noise on respiratory rate estimation
3. Implement custom respiratory event detection algorithms
4. Create visualizations of breathing patterns and events

Tutorial 4: Web Application Usage
==================================

Learn to use the VitalDSP web application for interactive signal analysis.

**Prerequisites:**
* VitalDSP installed
* Basic understanding of web interfaces
* Sample physiological data files

**Learning Objectives:**
* Launch the web application
* Upload and configure signal data
* Apply filtering and preprocessing
* Perform interactive analysis
* Generate reports and export results

**Step 1: Launch the Web Application**

.. code-block:: python

   from vitalDSP_webapp.run_webapp import run_webapp
   
   # Start the web application
   run_webapp(
       debug=True,
       port=8050,
       host='localhost'
   )

**Step 2: Data Upload and Configuration**

1. Open your browser and navigate to `http://localhost:8050`
2. Click on the "Upload" tab
3. Drag and drop your signal data file (CSV, Excel, or JSON)
4. Configure the data parameters:
   * Select the time column
   * Select the signal column
   * Set the sampling frequency
   * Choose the signal type (ECG, PPG, etc.)

**Step 3: Signal Filtering**

1. Navigate to the "Filtering" tab
2. The signal type will be automatically detected
3. Choose your filtering method:
   * **Traditional Filters**: Butterworth, Chebyshev, etc.
   * **Advanced Filters**: Kalman, adaptive filtering
   * **Artifact Removal**: Motion artifacts, baseline wander
   * **Neural Network**: Deep learning-based filtering
4. Configure filter parameters
5. Apply filtering and review results

**Step 4: Interactive Analysis**

1. Navigate to analysis screens:
   * **Time Domain Analysis**: Statistical and morphological features
   * **Frequency Domain Analysis**: Spectral analysis and frequency features
   * **Physiological Analysis**: HRV and comprehensive feature extraction
   * **Respiratory Analysis**: Respiratory rate estimation and pattern analysis

2. Use interactive features:
   * Zoom and pan on plots
   * Adjust time windows
   * Export visualizations
   * Download processed data

**Step 5: Report Generation**

1. Navigate to the "Health Report" section
2. Configure report parameters
3. Generate comprehensive analysis report
4. Export results in various formats:
   * PDF reports
   * CSV data exports
   * High-resolution images

**Exercise: Complete Workflow**

1. Upload a real physiological signal file
2. Apply appropriate filtering
3. Perform comprehensive analysis
4. Generate a health report
5. Export results for further analysis

Tutorial 5: Machine Learning Integration
========================================

Learn to integrate machine learning algorithms with VitalDSP for advanced signal analysis.

**Prerequisites:**
* Completion of previous tutorials
* Basic understanding of machine learning
* Familiarity with scikit-learn

**Learning Objectives:**
* Use neural network filtering
* Implement anomaly detection
* Apply Bayesian optimization
* Create ensemble methods
* Evaluate model performance

**Step 1: Neural Network Filtering**

.. code-block:: python

   from vitalDSP.advanced_computation.neural_network_filtering import NeuralNetworkFiltering
   
   # Initialize neural network filter
   nn_filter = NeuralNetworkFiltering(
       model_type='autoencoder',
       hidden_layers=[64, 32, 16],
       epochs=100,
       learning_rate=0.001
   )
   
   # Train the model (if needed)
   nn_filter.train(filtered_signal)
   
   # Apply neural network filtering
   nn_filtered_signal = nn_filter.filter(filtered_signal)

**Step 2: Anomaly Detection**

.. code-block:: python

   from vitalDSP.advanced_computation.anomaly_detection import AnomalyDetection
   
   # Initialize anomaly detector
   anomaly_detector = AnomalyDetection(
       method='isolation_forest',
       contamination=0.1
   )
   
   # Detect anomalies
   anomalies = anomaly_detector.detect_anomalies(filtered_signal)
   
   print(f"Detected {np.sum(anomalies)} anomalous samples")

**Step 3: Bayesian Optimization**

.. code-block:: python

   from vitalDSP.advanced_computation.bayesian_optimization import BayesianOptimization
   
   # Define objective function
   def objective_function(params):
       # Apply filtering with given parameters
       sf = SignalFiltering(filtered_signal, fs)
       filtered = sf.bandpass_filter(
           low_cut=params['low_cut'],
           high_cut=params['high_cut'],
           filter_order=int(params['filter_order'])
       )
       
       # Calculate signal quality metric
       sqi = SignalQualityIndex(filtered)
       quality, _, _ = sqi.amplitude_variability_sqi(
           window_size=fs*5,
           step_size=fs*1,
           threshold=2
       )
       
       return np.mean(quality)
   
   # Initialize Bayesian optimization
   bo = BayesianOptimization(
       objective_function,
       {'low_cut': (0.1, 2.0), 'high_cut': (20.0, 50.0), 'filter_order': (2, 8)}
   )
   
   # Optimize parameters
   bo.optimize(n_iter=20)
   
   print(f"Best parameters: {bo.max['params']}")
   print(f"Best score: {bo.max['target']:.4f}")

**Step 4: Ensemble Methods**

.. code-block:: python

   from vitalDSP.advanced_computation.ensemble_methods import EnsembleFiltering
   
   # Initialize ensemble filter
   ensemble = EnsembleFiltering(
       methods=['butterworth', 'kalman', 'neural_network'],
       weights=[0.4, 0.3, 0.3]
   )
   
   # Apply ensemble filtering
   ensemble_filtered = ensemble.filter(filtered_signal)

**Exercise: Advanced Applications**

1. Compare different machine learning approaches
2. Optimize hyperparameters for your specific use case
3. Implement custom ensemble methods
4. Evaluate performance on different signal types

Best Practices
==============

**Performance Optimization**
* Use appropriate sampling rates for your analysis
* Consider signal length vs. processing time trade-offs
* Utilize batch processing for multiple signals
* Cache frequently used computations

**Data Quality**
* Always assess signal quality before analysis
* Apply appropriate preprocessing steps
* Validate results against known standards
* Document your processing pipeline

**Error Handling**
* Use try-catch blocks for robust error handling
* Validate input data before processing
* Log important processing steps
* Provide meaningful error messages

**Clinical Applications**
* Understand the clinical significance of your analysis
* Validate results against clinical standards
* Consider patient safety and data privacy
* Document methodology for reproducibility

Troubleshooting Common Issues
==============================

**Installation Issues**
* Ensure Python 3.8+ is installed
* Check all dependencies are properly installed
* Verify virtual environment setup

**Signal Processing Issues**
* Validate input signal format and parameters
* Check sampling frequency accuracy
* Verify signal quality before processing

**Web Application Issues**
* Ensure port 8050 is available
* Check browser compatibility
* Clear browser cache if visualizations don't display

**Performance Issues**
* Reduce signal length for faster processing
* Use appropriate filter parameters
* Consider using more efficient algorithms

Next Steps
==========

After completing these tutorials, you should be able to:

1. **Process physiological signals** using various filtering techniques
2. **Extract meaningful features** from ECG, PPG, and respiratory signals
3. **Perform comprehensive analysis** including HRV and respiratory analysis
4. **Use the web application** for interactive signal processing
5. **Integrate machine learning** for advanced signal analysis

Continue exploring the documentation to learn about:
* Advanced signal processing techniques
* Custom analysis pipelines
* Performance optimization
* Clinical applications
* Contributing to the project

Happy learning with VitalDSP! 🫀📊
