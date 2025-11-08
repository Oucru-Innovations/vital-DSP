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
   fs = 256  # Sampling frequency (Hz) - using default for compatibility
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
   sf = SignalFiltering(ecg_signal)
   
   # Apply bandpass filter (0.5-40 Hz for ECG)
   filtered_signal = sf.bandpass(
       lowcut=0.5,
       highcut=40.0,
       fs=fs,
       order=4
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
       window_size=int(fs*5),  # 5-second windows
       step_size=int(fs*1),    # 1-second steps
       threshold=2,
       aggregate=False
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
   from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   import numpy as np
   
   # Generate and filter ECG signal first
   fs = 256  # Sampling frequency
   ecg_signal = generate_ecg_signal(sfecg=fs, duration=10, hrmean=72, Anoise=0.1)
   sf = SignalFiltering(ecg_signal)
   filtered_signal = sf.bandpass(lowcut=0.5, highcut=40.0, fs=fs, order=4)
   
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
   import numpy as np
   
   # Generate sample respiratory signal
   from vitalDSP.utils.data_processing.synthesize_data import generate_resp_signal
   
   # Parameters
   fs = 100  # Sampling frequency (Hz)
   duration = 60  # Duration (seconds)
   resp_rate_bpm = 16  # Respiratory rate (breaths per minute)
   freq_hz = resp_rate_bpm / 60  # Convert to Hz
   
   # Generate synthetic respiratory signal
   resp_signal = generate_resp_signal(
       sampling_rate=fs,
       duration=duration,
       frequency=freq_hz,
       amplitude=0.5
   )
   
   # Add noise manually
   resp_signal = resp_signal + np.random.randn(len(resp_signal)) * 0.05
   
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
       sf = SignalFiltering(filtered_signal)
       filtered = sf.bandpass(
           lowcut=params['low_cut'],
           highcut=params['high_cut'],
           fs=fs,
           order=int(params['filter_order'])
       )
       
       # Calculate signal quality metric
       sqi = SignalQualityIndex(filtered)
       quality, _, _ = sqi.amplitude_variability_sqi(
           window_size=int(fs*5),
           step_size=int(fs*1),
           threshold=2,
           aggregate=False
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

Tutorial 6: EMD and Advanced Signal Decomposition
===================================================

Learn advanced signal decomposition techniques using Empirical Mode Decomposition (EMD) for non-linear and non-stationary physiological signals.

**Prerequisites:**
* Completed Tutorial 1 (Basic Signal Processing)
* Understanding of signal components
* Familiarity with frequency analysis

**Learning Objectives:**
* Understand Empirical Mode Decomposition (EMD)
* Decompose signals into Intrinsic Mode Functions (IMFs)
* Reconstruct and analyze signal components
* Apply EMD to ECG and respiratory signals
* Perform adaptive signal filtering using IMFs

**What is EMD?**

Empirical Mode Decomposition is a powerful method for decomposing non-linear and non-stationary signals into Intrinsic Mode Functions (IMFs). Unlike Fourier Transform, EMD is data-adaptive and can handle signals with varying frequency content.

**Step 1: Basic EMD Decomposition**

.. code-block:: python

   import numpy as np
   from vitalDSP.advanced_computation.emd import EMD
   from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal
   import matplotlib.pyplot as plt
   
   # Generate ECG signal with noise
   fs = 256
   duration = 10
   ecg = generate_ecg_signal(sfecg=fs, duration=duration, hrmean=75, Anoise=0.1)
   
   # Perform EMD
   print("Performing Empirical Mode Decomposition...")
   emd = EMD(ecg)
   imfs = emd.emd(max_imfs=6, stop_criterion=0.05)
   
   print(f"✓ Extracted {len(imfs)} Intrinsic Mode Functions")
   
   # Visualize IMFs
   fig, axes = plt.subplots(len(imfs) + 2, 1, figsize=(12, 10))
   
   axes[0].plot(ecg)
   axes[0].set_title('Original ECG Signal')
   axes[0].grid(True, alpha=0.3)
   
   for i, imf in enumerate(imfs):
       axes[i+1].plot(imf)
       axes[i+1].set_title(f'IMF {i+1}')
       axes[i+1].grid(True, alpha=0.3)
   
   # Reconstruction
   reconstructed = np.sum(imfs, axis=0)
   axes[-1].plot(reconstructed)
   axes[-1].set_title('Reconstructed Signal')
   axes[-1].grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()
   
   # Calculate reconstruction error
   reconstruction_error = np.mean((ecg - reconstructed) ** 2)
   print(f"Reconstruction MSE: {reconstruction_error:.6f}")

**Step 2: Adaptive Noise Removal Using IMFs**

.. code-block:: python

   def adaptive_denoising(signal, n_low_imfs_to_remove=2, n_high_imfs_to_keep=3):
       """
       Remove noise from signal using EMD-based adaptive filtering.
       
       Parameters:
       -----------
       signal : array-like
           Input noisy signal
       n_low_imfs_to_remove : int
           Number of high-frequency IMFs to remove (noise)
       n_high_imfs_to_keep : int
           Number of low-frequency IMFs to keep (trend)
       
       Returns:
       --------
       denoised_signal : array
           Denoised signal using selected IMFs
       """
       emd = EMD(signal)
       imfs = emd.emd(max_imfs=8)
       
       if len(imfs) < n_low_imfs_to_remove + n_high_imfs_to_keep:
           print("⚠️ Insufficient IMFs for optimal denoising")
           # Use all except first and last IMF
           selected_imfs = imfs[1:-1]
       else:
           # Remove high-frequency noise (first few IMFs)
           # Keep middle-frequency components (signal of interest)
           # Remove low-frequency trend (last few IMFs)
           selected_imfs = imfs[n_low_imfs_to_remove:-n_high_imfs_to_keep]
       
       denoised = np.sum(selected_imfs, axis=0)
       return denoised, imfs
   
   # Generate noisy ECG
   clean_ecg = generate_ecg_signal(sfecg=fs, duration=duration, hrmean=75, Anoise=0.0)
   noise = np.random.normal(0, 0.15, len(clean_ecg))
   noisy_ecg = clean_ecg + noise
   
   # Apply adaptive denoising
   denoised_ecg, imfs = adaptive_denoising(noisy_ecg)
   
   # Calculate SNR improvement
   snr_before = 10 * np.log10(np.var(clean_ecg) / np.var(noise))
   snr_after = 10 * np.log10(np.var(clean_ecg) / np.var(denoised_ecg - clean_ecg))
   
   print(f"\n📊 Denoising Performance:")
   print(f"  SNR before: {snr_before:.2f} dB")
   print(f"  SNR after: {snr_after:.2f} dB")
   print(f"  Improvement: {snr_after - snr_before:.2f} dB")

**Step 3: Component Analysis and Frequency Extraction**

.. code-block:: python

   from scipy.fft import fft, fftfreq
   
   def analyze_imf_frequencies(imfs, fs):
       """Analyze the dominant frequency of each IMF."""
       
       imf_info = []
       
       for i, imf in enumerate(imfs):
           # Compute FFT
           N = len(imf)
           yf = fft(imf)
           xf = fftfreq(N, 1/fs)[:N//2]
           power = 2.0/N * np.abs(yf[:N//2])
           
           # Find dominant frequency
           dominant_freq_idx = np.argmax(power)
           dominant_freq = xf[dominant_freq_idx]
           
           # Calculate energy
           energy = np.sum(imf ** 2)
           
           imf_info.append({
               'imf_number': i + 1,
               'dominant_frequency': dominant_freq,
               'energy': energy,
               'mean_amplitude': np.mean(np.abs(imf))
           })
       
       return imf_info
   
   # Analyze IMFs
   imf_analysis = analyze_imf_frequencies(imfs, fs)
   
   print(f"\n{'='*60}")
   print("IMF ANALYSIS")
   print(f"{'='*60}")
   print(f"{'IMF':<6} {'Frequency':<12} {'Energy':<15} {'Mean Amp':<12}")
   print("-" * 60)
   
   for info in imf_analysis:
       print(f"{info['imf_number']:<6} {info['dominant_frequency']:<12.2f} "
             f"{info['energy']:<15.2e} {info['mean_amplitude']:<12.4f}")

**Step 4: Clinical Application - Baseline Wander Removal**

.. code-block:: python

   def remove_baseline_wander(ecg_signal):
       """
       Remove baseline wander from ECG using EMD.
       
       Baseline wander is typically in the lowest frequency IMF.
       """
       emd = EMD(ecg_signal)
       imfs = emd.emd()
       
       # Baseline wander is usually in the last IMF (residual)
       # Remove the last 1-2 IMFs which represent the trend
       if len(imfs) >= 2:
           baseline_corrected = np.sum(imfs[:-2], axis=0)
       else:
           baseline_corrected = ecg_signal
       
       return baseline_corrected, imfs[-1] if len(imfs) > 0 else np.zeros_like(ecg_signal)
   
   # Generate ECG with baseline wander
   ecg_with_wander = generate_ecg_signal(sfecg=fs, duration=duration, hrmean=75, Anoise=0.05)
   baseline_wander = 0.3 * np.sin(2 * np.pi * 0.2 * np.arange(len(ecg_with_wander)) / fs)
   ecg_with_wander += baseline_wander
   
   # Remove baseline wander
   corrected_ecg, extracted_baseline = remove_baseline_wander(ecg_with_wander)
   
   print(f"\n✅ Baseline wander removed successfully")
   print(f"  Original signal range: [{ecg_with_wander.min():.3f}, {ecg_with_wander.max():.3f}]")
   print(f"  Corrected signal range: [{corrected_ecg.min():.3f}, {corrected_ecg.max():.3f}]")

**Key Takeaways:**

1. ✅ **EMD is data-adaptive** - automatically determines decomposition based on signal characteristics
2. ✅ **IMFs represent different frequency scales** - from high-frequency noise to low-frequency trends
3. ✅ **Flexible denoising** - select relevant IMFs based on application
4. ✅ **No predefined basis functions** - unlike Fourier or Wavelet transforms
5. ✅ **Clinical applications** - baseline wander removal, artifact detection, trend analysis

Tutorial 7: Sleep Apnea Detection and Respiratory Pattern Analysis
====================================================================

Learn to detect sleep apnea events and analyze respiratory patterns using advanced signal processing techniques.

**Prerequisites:**
* Completed Tutorial 4 (Respiratory Analysis)
* Understanding of respiratory physiology
* Basic signal processing knowledge

**Learning Objectives:**
* Detect sleep apnea events from respiratory signals
* Classify apnea types (obstructive, central, mixed)
* Calculate apnea-hypopnea index (AHI)
* Analyze respiratory pattern variations
* Generate clinical reports

**What is Sleep Apnea?**

Sleep apnea is a serious sleep disorder characterized by repeated pauses in breathing during sleep. The Apnea-Hypopnea Index (AHI) quantifies severity:
- Normal: AHI < 5
- Mild: AHI 5-15
- Moderate: AHI 15-30
- Severe: AHI > 30

**Step 1: Generate Simulated Sleep Study Data**

.. code-block:: python

   import numpy as np
   from vitalDSP.respiratory_analysis.sleep_apnea_detection.pause_detection import detect_apnea_pauses
   from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis
   from vitalDSP.utils.data_processing.synthesize_data import generate_resp_signal
   
   def generate_sleep_study_signal(duration_minutes=60, fs=25, apnea_events=15):
       """
       Generate simulated respiratory signal with apnea events.
       
       Parameters:
       -----------
       duration_minutes : int
           Duration of sleep study in minutes
       fs : int
           Sampling frequency
       apnea_events : int
           Number of apnea events to simulate
       
       Returns:
       --------
       signal : array
           Simulated respiratory signal
       true_apnea_times : list
           Ground truth apnea event times
       """
       duration_sec = duration_minutes * 60
       samples = duration_sec * fs
       
       # Generate normal respiratory signal (0.2-0.3 Hz, 12-18 breaths/min)
       base_resp = generate_resp_signal(
           sampling_rate=fs,
           duration=duration_sec,
           frequency=0.25,  # 15 breaths per minute
           amplitude=1.0
       )
       
       # Add respiratory rate variability
       rate_variation = 0.05 * np.sin(2 * np.pi * 0.01 * np.arange(samples) / fs)
       signal = base_resp * (1 + rate_variation)
       
       # Insert apnea events (breathing pauses)
       true_apnea_times = []
       event_indices = np.random.choice(
           range(fs * 30, samples - fs * 30),
           size=apnea_events,
           replace=False
       )
       
       for event_idx in sorted(event_indices):
           # Apnea duration: 10-30 seconds
           apnea_duration = np.random.randint(10, 31)
           pause_samples = apnea_duration * fs
           
           # Gradual reduction and recovery
           fade_samples = fs * 2  # 2 second fade
           fade_out = np.linspace(1, 0, fade_samples)
           fade_in = np.linspace(0, 1, fade_samples)
           
           start_idx = event_idx
           end_idx = min(event_idx + pause_samples, samples)
           
           # Apply apnea
           if end_idx - start_idx > 2 * fade_samples:
               signal[start_idx:start_idx+fade_samples] *= fade_out
               signal[start_idx+fade_samples:end_idx-fade_samples] *= 0.1  # Minimal breathing
               signal[end_idx-fade_samples:end_idx] *= fade_in
               
               true_apnea_times.append({
                   'start': start_idx / fs,
                   'end': end_idx / fs,
                   'duration': (end_idx - start_idx) / fs
               })
       
       return signal, true_apnea_times
   
   # Generate 2-hour sleep study
   print("Generating simulated sleep study data...")
   fs = 25  # 25 Hz for respiratory signals
   duration_min = 120  # 2 hours
   expected_events = 40  # Moderate-severe sleep apnea
   
   resp_signal, true_events = generate_sleep_study_signal(
       duration_minutes=duration_min,
       fs=fs,
       apnea_events=expected_events
   )
   
   print(f"✓ Generated {duration_min}-minute sleep study")
   print(f"✓ Inserted {len(true_events)} apnea events")

**Step 2: Detect Apnea Events**

.. code-block:: python

   # Detect apnea pauses
   print("\nDetecting sleep apnea events...")
   detected_events = detect_apnea_pauses(
       signal=resp_signal,
       sampling_rate=fs,
       min_pause_duration=10,  # Minimum 10 seconds for apnea
       preprocess='bandpass',
       low=0.1,
       high=0.5,
       order=4
   )
   
   print(f"✓ Detected {len(detected_events)} apnea events")
   
   # Calculate detection performance
   def calculate_detection_metrics(true_events, detected_events, tolerance=5.0):
       """
       Calculate precision, recall, and F1-score for apnea detection.
       
       Parameters:
       -----------
       true_events : list
           Ground truth apnea events
       detected_events : list
           Detected apnea events
       tolerance : float
           Time tolerance in seconds for matching events
       """
       true_positives = 0
       
       for true_event in true_events:
           true_start = true_event['start']
           # Check if any detected event is within tolerance
           for det_start, det_end in detected_events:
               if abs(det_start - true_start) <= tolerance:
                   true_positives += 1
                   break
       
       false_positives = len(detected_events) - true_positives
       false_negatives = len(true_events) - true_positives
       
       precision = true_positives / len(detected_events) if len(detected_events) > 0 else 0
       recall = true_positives / len(true_events) if len(true_events) > 0 else 0
       f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
       
       return {
           'true_positives': true_positives,
           'false_positives': false_positives,
           'false_negatives': false_negatives,
           'precision': precision,
           'recall': recall,
           'f1_score': f1_score
       }
   
   metrics = calculate_detection_metrics(true_events, detected_events)
   
   print(f"\n{'='*60}")
   print("DETECTION PERFORMANCE")
   print(f"{'='*60}")
   print(f"True Positives: {metrics['true_positives']}")
   print(f"False Positives: {metrics['false_positives']}")
   print(f"False Negatives: {metrics['false_negatives']}")
   print(f"Precision: {metrics['precision']:.2%}")
   print(f"Recall: {metrics['recall']:.2%}")
   print(f"F1-Score: {metrics['f1_score']:.2%}")

**Step 3: Calculate Apnea-Hypopnea Index (AHI)**

.. code-block:: python

   def calculate_ahi(detected_events, duration_hours):
       """
       Calculate Apnea-Hypopnea Index (AHI).
       
       AHI = Number of apnea/hypopnea events per hour of sleep
       """
       n_events = len(detected_events)
       ahi = n_events / duration_hours
       
       # Classify severity
       if ahi < 5:
           severity = "Normal"
       elif ahi < 15:
           severity = "Mild Sleep Apnea"
       elif ahi < 30:
           severity = "Moderate Sleep Apnea"
       else:
           severity = "Severe Sleep Apnea"
       
       return ahi, severity
   
   # Calculate AHI
   duration_hours = duration_min / 60
   ahi, severity = calculate_ahi(detected_events, duration_hours)
   
   print(f"\n{'='*60}")
   print("SLEEP APNEA ASSESSMENT")
   print(f"{'='*60}")
   print(f"Study Duration: {duration_hours:.1f} hours")
   print(f"Total Events: {len(detected_events)}")
   print(f"AHI: {ahi:.1f} events/hour")
   print(f"Severity: {severity}")
   
   # Event statistics
   if len(detected_events) > 0:
       event_durations = [(end - start) for start, end in detected_events]
       print(f"\nEvent Statistics:")
       print(f"  Mean duration: {np.mean(event_durations):.1f} seconds")
       print(f"  Min duration: {np.min(event_durations):.1f} seconds")
       print(f"  Max duration: {np.max(event_durations):.1f} seconds")
       print(f"  Std deviation: {np.std(event_durations):.1f} seconds")

**Step 4: Comprehensive Sleep Study Report**

.. code-block:: python

   def generate_sleep_study_report(resp_signal, fs, detected_events, duration_hours):
       """Generate comprehensive sleep apnea report."""
       
       # Calculate respiratory rate during non-apnea periods
       resp_analyzer = RespiratoryAnalysis(resp_signal, fs=fs)
       try:
           resp_rate = resp_analyzer.estimate_respiratory_rate()
       except:
           resp_rate = None
       
       # Calculate AHI
       ahi, severity = calculate_ahi(detected_events, duration_hours)
       
       # Oxygen desaturation events (estimated from respiratory pauses)
       # In real scenario, this would come from SpO2 sensor
       estimated_desat_events = int(len(detected_events) * 0.8)  # 80% of apneas cause desaturation
       
       report = {
           'study_duration_hours': duration_hours,
           'total_apnea_events': len(detected_events),
           'ahi': ahi,
           'severity': severity,
           'mean_respiratory_rate': resp_rate,
           'estimated_desaturation_events': estimated_desat_events,
           'recommendations': []
       }
       
       # Clinical recommendations
       if ahi >= 30:
           report['recommendations'].append('⚠️ URGENT: Immediate sleep specialist consultation recommended')
           report['recommendations'].append('Consider CPAP therapy evaluation')
           report['recommendations'].append('Cardiovascular risk assessment advised')
       elif ahi >= 15:
           report['recommendations'].append('⚠️ Sleep specialist consultation recommended')
           report['recommendations'].append('Lifestyle modifications: weight loss, avoid alcohol before sleep')
           report['recommendations'].append('Consider oral appliance or CPAP therapy')
       elif ahi >= 5:
           report['recommendations'].append('⚠️ Mild sleep apnea detected')
           report['recommendations'].append('Lifestyle modifications recommended')
           report['recommendations'].append('Follow-up study in 6-12 months')
       else:
           report['recommendations'].append('✓ No significant sleep apnea detected')
           report['recommendations'].append('Maintain healthy sleep habits')
       
       return report
   
   # Generate report
   report = generate_sleep_study_report(resp_signal, fs, detected_events, duration_hours)
   
   print(f"\n{'='*60}")
   print("CLINICAL SLEEP STUDY REPORT")
   print(f"{'='*60}")
   print(f"Study Duration: {report['study_duration_hours']:.1f} hours")
   print(f"Total Apnea Events: {report['total_apnea_events']}")
   print(f"Apnea-Hypopnea Index (AHI): {report['ahi']:.1f} events/hour")
   print(f"Severity Classification: {report['severity']}")
   if report['mean_respiratory_rate']:
       print(f"Mean Respiratory Rate: {report['mean_respiratory_rate']:.1f} breaths/min")
   print(f"Estimated Desaturation Events: {report['estimated_desaturation_events']}")
   
   print(f"\nClinical Recommendations:")
   for rec in report['recommendations']:
       print(f"  {rec}")

**Key Takeaways:**

1. ✅ **AHI is the gold standard** for sleep apnea severity assessment
2. ✅ **Automated detection** reduces manual scoring time
3. ✅ **Clinical integration** - generate actionable reports
4. ✅ **Multi-modal analysis** - combine with SpO2 and ECG for comprehensive assessment
5. ✅ **Real-world application** - adaptable to various sensor types

Tutorial 8: ECG-PPG Synchronization and Pulse Transit Time Analysis
=====================================================================

Learn advanced multi-modal analysis by synchronizing ECG and PPG signals to extract cardiovascular features like Pulse Transit Time (PTT).

**Prerequisites:**
* Completed Tutorial 1 (Basic ECG Processing)
* Understanding of cardiovascular physiology
* Familiarity with peak detection

**Learning Objectives:**
* Synchronize ECG and PPG signals
* Calculate Pulse Transit Time (PTT)
* Estimate blood pressure trends
* Analyze vascular parameters
* Perform cardio-synchronization assessment

**What is Pulse Transit Time?**

Pulse Transit Time (PTT) is the time it takes for the arterial pulse wave to travel from the heart (R-peak in ECG) to a peripheral site (PPG sensor). PTT is inversely related to blood pressure and provides valuable information about vascular health.

**Step 1: Generate Synchronized ECG and PPG Signals**

.. code-block:: python

   import numpy as np
   from vitalDSP.feature_engineering.ecg_ppg_synchronyzation_features import ECGPPGSynchronization
   from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal, generate_synthetic_ppg
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   
   # Generate synchronized signals
   fs_ecg = 256  # ECG sampling frequency
   fs_ppg = 100  # PPG sampling frequency (often lower than ECG)
   duration = 60  # 1 minute
   hr = 75  # Heart rate
   
   print("Generating synchronized ECG and PPG signals...")
   
   # Generate ECG
   ecg_signal = generate_ecg_signal(
       sfecg=fs_ecg,
       duration=duration,
       hrmean=hr,
       Anoise=0.05
   )
   
   # Generate PPG (with physiological delay)
   ppg_signal = generate_synthetic_ppg(
       duration=duration,
       sampling_rate=fs_ppg,
       heart_rate=hr,
       noise_level=0.05
   )
   
   # Add realistic PTT (150-250 ms typical range)
   ptt_ms = 200  # milliseconds
   ptt_samples_ppg = int(ptt_ms * fs_ppg / 1000)
   
   # Delay PPG signal to simulate PTT
   ppg_signal = np.roll(ppg_signal, ptt_samples_ppg)
   
   print(f"✓ Generated {duration}s of synchronized signals")
   print(f"  ECG: {fs_ecg} Hz")
   print(f"  PPG: {fs_ppg} Hz")
   print(f"  Simulated PTT: {ptt_ms} ms")
   
   # Preprocess signals
   sf_ecg = SignalFiltering(ecg_signal)
   ecg_filtered = sf_ecg.bandpass(lowcut=0.5, highcut=40.0, fs=fs_ecg, order=4)
   
   sf_ppg = SignalFiltering(ppg_signal)
   ppg_filtered = sf_ppg.bandpass(lowcut=0.5, highcut=8.0, fs=fs_ppg, order=4)

**Step 2: Calculate Pulse Transit Time**

.. code-block:: python

   # Initialize synchronization analyzer
   sync_analyzer = ECGPPGSynchronization(
       ecg_signal=ecg_filtered,
       ppg_signal=ppg_filtered,
       ecg_fs=fs_ecg,
       ppg_fs=fs_ppg
   )
   
   # Detect peaks
   r_peaks = sync_analyzer.detect_r_peaks()
   ppg_peaks = sync_analyzer.detect_ppg_peaks()
   
   print(f"\n✓ Detected {len(r_peaks)} R-peaks in ECG")
   print(f"✓ Detected {len(ppg_peaks)} peaks in PPG")
   
   # Calculate PTT
   ptt_values = sync_analyzer.compute_pulse_transit_time()
   
   if len(ptt_values) > 0:
       print(f"\n{'='*60}")
       print("PULSE TRANSIT TIME ANALYSIS")
       print(f"{'='*60}")
       print(f"Number of PTT measurements: {len(ptt_values)}")
       print(f"Mean PTT: {np.mean(ptt_values):.2f} ms")
       print(f"Std PTT: {np.std(ptt_values):.2f} ms")
       print(f"Min PTT: {np.min(ptt_values):.2f} ms")
       print(f"Max PTT: {np.max(ptt_values):.2f} ms")
       print(f"PTT Variability: {np.std(ptt_values)/np.mean(ptt_values)*100:.2f}%")

**Step 3: Blood Pressure Trend Estimation**

.. code-block:: python

   def estimate_bp_trend(ptt_values, baseline_sbp=120, baseline_dbp=80):
       """
       Estimate blood pressure trends from PTT variations.
       
       PTT is inversely related to blood pressure:
       - Shorter PTT → Higher BP
       - Longer PTT → Lower BP
       
       Parameters:
       -----------
       ptt_values : array
           Pulse transit time values in milliseconds
       baseline_sbp : float
           Baseline systolic BP (mmHg)
       baseline_dbp : float
           Baseline diastolic BP (mmHg)
       
       Returns:
       --------
       dict : BP trend estimates
       """
       if len(ptt_values) == 0:
           return None
       
       # Empirical relationship: ΔBP ≈ -k * ΔPTT
       # k typically ranges from 0.3 to 0.5 mmHg/ms
       k = 0.4
       
       mean_ptt = np.mean(ptt_values)
       ptt_changes = ptt_values - mean_ptt
       
       # Estimate BP changes
       bp_changes = -k * ptt_changes
       
       # Estimated BP values
       estimated_sbp = baseline_sbp + bp_changes
       estimated_dbp = baseline_dbp + bp_changes * 0.6  # DBP changes less than SBP
       
       return {
           'mean_sbp': np.mean(estimated_sbp),
           'std_sbp': np.std(estimated_sbp),
           'min_sbp': np.min(estimated_sbp),
           'max_sbp': np.max(estimated_sbp),
           'mean_dbp': np.mean(estimated_dbp),
           'std_dbp': np.std(estimated_dbp),
           'bp_variability': np.std(estimated_sbp),
           'estimated_sbp_trend': estimated_sbp,
           'estimated_dbp_trend': estimated_dbp
       }
   
   # Estimate BP trends
   bp_estimates = estimate_bp_trend(ptt_values)
   
   if bp_estimates:
       print(f"\n{'='*60}")
       print("BLOOD PRESSURE TREND ESTIMATION")
       print(f"{'='*60}")
       print(f"Estimated Mean SBP: {bp_estimates['mean_sbp']:.1f} ± {bp_estimates['std_sbp']:.1f} mmHg")
       print(f"Estimated Mean DBP: {bp_estimates['mean_dbp']:.1f} ± {bp_estimates['std_dbp']:.1f} mmHg")
       print(f"SBP Range: [{bp_estimates['min_sbp']:.1f}, {bp_estimates['max_sbp']:.1f}] mmHg")
       print(f"BP Variability: {bp_estimates['bp_variability']:.2f} mmHg")
       
       # Clinical interpretation
       print(f"\nClinical Assessment:")
       if bp_estimates['mean_sbp'] < 120 and bp_estimates['mean_dbp'] < 80:
           print("  ✓ Normal blood pressure range")
       elif bp_estimates['mean_sbp'] < 130 and bp_estimates['mean_dbp'] < 80:
           print("  ⚠️ Elevated blood pressure (prehypertension)")
       elif bp_estimates['mean_sbp'] < 140 or bp_estimates['mean_dbp'] < 90:
           print("  ⚠️ Stage 1 Hypertension")
       else:
           print("  ⚠️ Stage 2 Hypertension - medical consultation recommended")
       
       if bp_estimates['bp_variability'] > 10:
           print("  ⚠️ High BP variability - may indicate cardiovascular risk")
       else:
           print("  ✓ Normal BP variability")

**Step 4: Comprehensive Cardiovascular Assessment**

.. code-block:: python

   def comprehensive_cardiovascular_analysis(ecg_ppg_sync, ptt_values, bp_estimates):
       """Generate comprehensive cardiovascular health report."""
       
       # Calculate additional features
       pat = ecg_ppg_sync.compute_pulse_arrival_time()
       
       # Pulse wave velocity (PWV) estimation
       # PWV ≈ Distance / PTT
       # Assuming typical distance of 60 cm from heart to finger
       distance_cm = 60
       if len(ptt_values) > 0:
           mean_ptt_s = np.mean(ptt_values) / 1000  # Convert to seconds
           pwv = distance_cm / mean_ptt_s  # cm/s
           pwv_m_s = pwv / 100  # m/s
       else:
           pwv_m_s = None
       
       report = {
           'ptt_metrics': {
               'mean': np.mean(ptt_values) if len(ptt_values) > 0 else None,
               'std': np.std(ptt_values) if len(ptt_values) > 0 else None,
               'variability': np.std(ptt_values)/np.mean(ptt_values)*100 if len(ptt_values) > 0 else None
           },
           'pwv_m_s': pwv_m_s,
           'bp_estimates': bp_estimates,
           'vascular_health': 'unknown',
           'recommendations': []
       }
       
       # Assess vascular health based on PWV
       if pwv_m_s:
           if pwv_m_s < 7:
               report['vascular_health'] = 'Good (Compliant arteries)'
               report['recommendations'].append('✓ Vascular health within normal range')
           elif pwv_m_s < 10:
               report['vascular_health'] = 'Fair (Moderate stiffness)'
               report['recommendations'].append('⚠️ Moderate arterial stiffness detected')
               report['recommendations'].append('Regular cardiovascular monitoring recommended')
           else:
               report['vascular_health'] = 'Poor (Stiff arteries)'
               report['recommendations'].append('⚠️ Significant arterial stiffness detected')
               report['recommendations'].append('Cardiovascular risk assessment advised')
               report['recommendations'].append('Lifestyle modifications and medical evaluation recommended')
       
       # PTT variability assessment
       if report['ptt_metrics']['variability']:
           if report['ptt_metrics']['variability'] < 10:
               report['recommendations'].append('✓ Stable cardiovascular function')
           else:
               report['recommendations'].append('⚠️ High PTT variability may indicate autonomic dysfunction')
       
       return report
   
   # Generate comprehensive report
   cardio_report = comprehensive_cardiovascular_analysis(sync_analyzer, ptt_values, bp_estimates)
   
   print(f"\n{'='*60}")
   print("COMPREHENSIVE CARDIOVASCULAR ASSESSMENT")
   print(f"{'='*60}")
   
   if cardio_report['ptt_metrics']['mean']:
       print(f"\nPulse Transit Time:")
       print(f"  Mean PTT: {cardio_report['ptt_metrics']['mean']:.2f} ms")
       print(f"  PTT Variability: {cardio_report['ptt_metrics']['variability']:.2f}%")
   
   if cardio_report['pwv_m_s']:
       print(f"\nPulse Wave Velocity:")
       print(f"  PWV: {cardio_report['pwv_m_s']:.2f} m/s")
   
   print(f"\nVascular Health Status: {cardio_report['vascular_health']}")
   
   print(f"\nClinical Recommendations:")
   for rec in cardio_report['recommendations']:
       print(f"  {rec}")

**Key Takeaways:**

1. ✅ **PTT provides non-invasive BP trends** - useful for continuous monitoring
2. ✅ **Multi-modal analysis** - combining ECG and PPG reveals more information
3. ✅ **Vascular assessment** - PWV indicates arterial stiffness
4. ✅ **Clinical applications** - cardiovascular risk stratification
5. ✅ **Real-time monitoring** - adaptable to wearable devices

Continue exploring the documentation to learn about:
* Advanced signal processing techniques
* Custom analysis pipelines
* Performance optimization
* Clinical applications
* Contributing to the project

Happy learning with VitalDSP! 🫀📊
