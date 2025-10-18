Getting Started with VitalDSP
=============================

Welcome to VitalDSP, a comprehensive digital signal processing (DSP) library designed for healthcare and biomedical applications. This guide will walk you through the initial steps to get the library up and running and explore its core functionalities.

What is VitalDSP?
=================

VitalDSP is a powerful Python library specifically designed for processing and analyzing physiological signals such as:

* **ECG (Electrocardiogram)**: Heart rhythm and electrical activity analysis
* **PPG (Photoplethysmogram)**: Blood volume changes and pulse analysis
* **Respiratory Signals**: Breathing patterns and respiratory rate estimation
* **Other Vital Signs**: Blood pressure, temperature, and more

The library provides both a comprehensive Python API and an intuitive web application for signal processing, making it accessible to researchers, clinicians, and developers.

**🚀 Quick Start**: Try the live demo immediately at `https://vital-dsp.onrender.com <https://vital-dsp.onrender.com>`_ without any installation!

Installation
============

Prerequisites
~~~~~~~~~~~~~

Before installing VitalDSP, ensure you have Python 3.8 or higher installed on your system.

.. code-block:: bash

   python --version  # Should be 3.8 or higher

Installation Methods
~~~~~~~~~~~~~~~~~~~~

**Option 1: Install from PyPI (Recommended)**

.. code-block:: bash

   pip install vital-DSP

**Option 2: Install from Source (Recommended for Development)**

.. code-block:: bash

   git clone https://github.com/Oucru-Innovations/vital-DSP.git
   cd vital-DSP
   pip install -e .

**Option 3: Install with Development Dependencies**

.. code-block:: bash

   pip install vital-DSP[dev]

Verify Installation
~~~~~~~~~~~~~~~~~~~

Test your installation by importing the library:

.. code-block:: python

   import vitalDSP
   print(f"VitalDSP version: {vitalDSP.__version__}")

Quick Start
===========

Comprehensive Signal Analysis Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a quick example demonstrating VitalDSP's capabilities for physiological signal analysis:

.. code-block:: python

   import numpy as np
   import vitalDSP
   from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal, generate_synthetic_ppg
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.physiological_features.waveform import WaveformMorphology
   from vitalDSP.physiological_features.time_domain import TimeDomainFeatures
   from vitalDSP.physiological_features.frequency_domain import FrequencyDomainFeatures
   from vitalDSP.physiological_features.hrv_analysis import HRVFeatures
   import plotly.graph_objects as go
   from plotly.subplots import make_subplots

   # Generate synthetic ECG signal using VitalDSP
   fs = 128  # Sampling frequency
   duration = 10  # seconds
   hr_mean = 75  # beats per minute
   noise_amplitude = 0.1

   print("Generating synthetic ECG signal using VitalDSP...")
   ecg_signal = generate_ecg_signal(
       sfecg=fs, 
       N=duration, 
       Anoise=noise_amplitude, 
       hrmean=hr_mean
   )

   print(f"Generated ECG signal: {len(ecg_signal)} samples at {fs} Hz")

   # Generate synthetic PPG signal for comparison
   print("\\nGenerating synthetic PPG signal using VitalDSP...")
   ppg_time, ppg_signal = generate_synthetic_ppg(
       duration=duration,
       sampling_rate=fs,
       heart_rate=hr_mean,
       noise_level=noise_amplitude
   )

   print(f"Generated PPG signal: {len(ppg_signal)} samples at {fs} Hz")

   # Initialize signal filtering
   sf = SignalFiltering(ecg_signal)

   # Apply bandpass filter to remove noise and artifacts
   filtered_signal = sf.bandpass(lowcut=0.3, highcut=20.0, fs=fs)
   print("Applied bandpass filter (0.3-20 Hz)")

   # Extract comprehensive features using WaveformMorphology
   print("Extracting comprehensive features using WaveformMorphology...")
   wm = WaveformMorphology(filtered_signal, fs=fs, signal_type="ECG")

   # Extract all available features
   try:
       # Get basic waveform features
       r_peaks = wm.r_peaks
       q_valleys = wm.detect_q_valley()
       s_valleys = wm.detect_s_valley()
       p_peaks = wm.detect_p_peak()
       t_peaks = wm.detect_t_peak()
       
       print(f"Detected {len(r_peaks)} R-peaks")
       print(f"Detected {len(q_valleys)} Q-valleys")
       print(f"Detected {len(s_valleys)} S-valleys")
       print(f"Detected {len(p_peaks)} P-peaks")
       print(f"Detected {len(t_peaks)} T-peaks")
       
       # Calculate heart rate
       if len(r_peaks) > 1:
           rr_intervals = np.diff(r_peaks) / fs * 1000  # Convert to milliseconds
           heart_rate = 60000 / np.mean(rr_intervals)  # BPM
           print(f"Average heart rate: {heart_rate:.1f} BPM")
           
           # Extract HRV features using TimeDomainFeatures
           print("Extracting HRV features...")
           hrv_features = TimeDomainFeatures(rr_intervals)
           
           hrv_results = {
               'SDNN': hrv_features.compute_sdnn(),
               'RMSSD': hrv_features.compute_rmssd(),
               'NN50': hrv_features.compute_nn50(),
               'pNN50': hrv_features.compute_pnn50(),
               'Mean_NN': hrv_features.compute_mean_nn(),
               'Median_NN': hrv_features.compute_median_nn(),
               'IQR_NN': hrv_features.compute_iqr_nn(),
               'CVNN': hrv_features.compute_cvnn(),
               'HRV_Triangular_Index': hrv_features.compute_hrv_triangular_index(),
               'TINN': hrv_features.compute_tinn()
           }
           
           print("HRV Features:")
           for feature, value in hrv_results.items():
               print(f"  {feature}: {value:.4f}")
       
       # Extract morphological features
       print("Extracting morphological features...")
       try:
           qrs_amplitude = wm.get_qrs_amplitude()
           print(f"QRS Amplitude: {qrs_amplitude}")
       except:
           print("Could not extract QRS amplitude")
       
   except Exception as e:
       print(f"Error extracting features: {e}")

   print("\\n" + "="*50)
   print("FEATURE EXTRACTION COMPLETED!")
   print("="*50)

   # Create comprehensive visualizations
   print("\\nCreating visualizations...")

   # Create time axis for plotting
   time_axis = np.arange(len(ecg_signal)) / fs

   # Create subplots for ECG and PPG
   fig = make_subplots(
       rows=1, cols=1,
       subplot_titles=('ECG Signal with Critical Points (Q, R, S, T, P)',),
       vertical_spacing=0.1
   )

   # Plot ECG signal
   fig.add_trace(
       go.Scatter(
           x=time_axis, 
           y=filtered_signal, 
           mode='lines', 
           name='ECG Signal',
           line=dict(color='blue', width=1)
       ),
       row=1, col=1
   )

   # Add critical ECG points if they exist
   if 'r_peaks' in locals() and len(r_peaks) > 0:
       fig.add_trace(
           go.Scatter(
               x=r_peaks/fs, 
               y=filtered_signal[r_peaks], 
               mode='markers', 
               name='R Peaks',
               marker=dict(color='red', size=8, symbol='circle'),
               text=[f'R{i+1}' for i in range(len(r_peaks))],
               textposition='top center'
           ),
           row=1, col=1
       )

   if 'q_valleys' in locals() and len(q_valleys) > 0:
       fig.add_trace(
           go.Scatter(
               x=q_valleys/fs, 
               y=filtered_signal[q_valleys], 
               mode='markers', 
               name='Q Valleys',
               marker=dict(color='green', size=6, symbol='triangle-down'),
               text=[f'Q{i+1}' for i in range(len(q_valleys))],
               textposition='bottom center'
           ),
           row=1, col=1
       )

   if 's_valleys' in locals() and len(s_valleys) > 0:
       fig.add_trace(
           go.Scatter(
               x=s_valleys/fs, 
               y=filtered_signal[s_valleys], 
               mode='markers', 
               name='S Valleys',
               marker=dict(color='orange', size=6, symbol='triangle-down'),
               text=[f'S{i+1}' for i in range(len(s_valleys))],
               textposition='bottom center'
           ),
           row=1, col=1
       )

   if 'p_peaks' in locals() and len(p_peaks) > 0:
       fig.add_trace(
           go.Scatter(
               x=p_peaks/fs, 
               y=filtered_signal[p_peaks], 
               mode='markers', 
               name='P Peaks',
               marker=dict(color='purple', size=6, symbol='triangle-up'),
               text=[f'P{i+1}' for i in range(len(p_peaks))],
               textposition='top center'
           ),
           row=1, col=1
       )

   if 't_peaks' in locals() and len(t_peaks) > 0:
       fig.add_trace(
           go.Scatter(
               x=t_peaks/fs, 
               y=filtered_signal[t_peaks], 
               mode='markers', 
               name='T Peaks',
               marker=dict(color='brown', size=6, symbol='triangle-up'),
               text=[f'T{i+1}' for i in range(len(t_peaks))],
               textposition='top center'
           ),
           row=1, col=1
       )

   # Update layout
   fig.update_layout(
       height=600,
       title_text="VitalDSP Signal Analysis: ECG with Critical Points",
       title_x=0.5,
       showlegend=True
   )

   # Update x and y axis labels
   fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
   fig.update_yaxes(title_text="Amplitude", row=1, col=1)

   # Show the plot
   fig.show()

   # Create a summary statistics plot
   print("\\nCreating summary statistics visualization...")

   # Prepare data for summary plot
   if 'hrv_results' in locals():
       hrv_names = list(hrv_results.keys())
       hrv_values = list(hrv_results.values())
       
       # Create bar chart for HRV features
       fig_hrv = go.Figure(data=[
           go.Bar(
               x=hrv_names,
               y=hrv_values,
               marker_color='lightblue',
               text=[f'{v:.3f}' for v in hrv_values],
               textposition='auto'
           )
       ])
       
       fig_hrv.update_layout(
           title="Heart Rate Variability (HRV) Features",
           xaxis_title="HRV Metrics",
           yaxis_title="Values",
           height=500
       )
       
       fig_hrv.show()

   print("\\n" + "="*60)
   print("VISUALIZATION COMPLETED!")
   print("="*60)
   print("Summary:")
   print(f"- ECG Signal: {len(ecg_signal)} samples at {fs} Hz")
   if 'r_peaks' in locals():
       print(f"- Detected {len(r_peaks)} R-peaks")
       print(f"- Heart Rate: {heart_rate:.1f} BPM")
   print("="*60)

Web Application
~~~~~~~~~~~~~~~

**Try the Live Demo**

We have deployed a free example of the VitalDSP web application on Render. You can try it immediately without any installation:

`🚀 Live Demo: https://vital-dsp-1.onrender.com <https://vital-dsp-1.onrender.com>`_

The live demo includes all the core features:
- Signal upload and processing
- Interactive visualizations
- Feature extraction
- Real-time analysis
- Report generation

**Local Installation**

To run the web application locally:

.. code-block:: python

   from vitalDSP_webapp.run_webapp import run_webapp
   
   # Start the web application
   run_webapp(debug=True, port=8050)

Then open your browser and navigate to `http://localhost:8050`.

Core Modules Overview
=====================

Data Synthesis
~~~~~~~~~~~~~~

Generate synthetic physiological signals for testing and development:

.. code-block:: python

   import numpy as np
   import vitalDSP
   from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal, generate_synthetic_ppg

   # Generate synthetic signals
   fs = 128  # Sampling frequency
   duration = 10  # seconds
   hr_mean = 75  # beats per minute
   noise_amplitude = 0.1

   # Generate ECG signal
   ecg_signal = generate_ecg_signal(
       sfecg=fs, 
       N=duration, 
       Anoise=noise_amplitude, 
       hrmean=hr_mean
   )

   # Generate PPG signal
   ppg_time, ppg_signal = generate_synthetic_ppg(
       duration=duration,
       sampling_rate=fs,
       heart_rate=hr_mean,
       noise_level=noise_amplitude
   )

Signal Filtering and Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The filtering module provides comprehensive signal processing techniques with visualization:

.. code-block:: python
   import vitalDSP
   
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.filtering.artifact_removal import ArtifactRemoval
   import plotly.graph_objects as go
   from plotly.subplots import make_subplots


   # Apply filtering to both signals
   # ECG filtering
   sf_ecg = SignalFiltering(ecg_signal)
   filtered_ecg = sf_ecg.bandpass(lowcut=0.3, highcut=20.0, fs=fs)

   # PPG filtering
   sf_ppg = SignalFiltering(ppg_signal)
   filtered_ppg = sf_ppg.bandpass(lowcut=0.5, highcut=8.0, fs=fs)

   # Artifact removal
   ar = ArtifactRemoval(ecg_signal)
   clean_ecg = ar.median_filter_removal(kernel_size=3)

   # Create comprehensive visualization
   ecg_time = np.arange(len(ecg_signal)) / fs
   ppg_time_axis = np.arange(len(ppg_signal)) / fs

   fig = make_subplots(
       rows=2, cols=2,
       subplot_titles=(
           'ECG Signal: Original vs Filtered', 
           'PPG Signal: Original vs Filtered',
           'ECG Signal: Original vs Artifact Removed',
           'Signal Quality Comparison'
       ),
       vertical_spacing=0.12,
       horizontal_spacing=0.1
   )

   # Plot ECG Original vs Filtered
   fig.add_trace(
       go.Scatter(x=ecg_time, y=ecg_signal, mode='lines', 
                 name='ECG Original', line=dict(color='blue', width=1), opacity=0.7),
       row=1, col=1
   )
   fig.add_trace(
       go.Scatter(x=ecg_time, y=filtered_ecg, mode='lines', 
                 name='ECG Filtered', line=dict(color='red', width=1.5)),
       row=1, col=1
   )

   # Plot PPG Original vs Filtered
   fig.add_trace(
       go.Scatter(x=ppg_time_axis, y=ppg_signal, mode='lines', 
                 name='PPG Original', line=dict(color='green', width=1), opacity=0.7),
       row=1, col=2
   )
   fig.add_trace(
       go.Scatter(x=ppg_time_axis, y=filtered_ppg, mode='lines', 
                 name='PPG Filtered', line=dict(color='orange', width=1.5)),
       row=1, col=2
   )

   # Plot ECG Original vs Artifact Removed
   fig.add_trace(
       go.Scatter(x=ecg_time, y=ecg_signal, mode='lines', 
                 name='ECG Original', line=dict(color='blue', width=1), 
                 opacity=0.7, showlegend=False),
       row=2, col=1
   )
   fig.add_trace(
       go.Scatter(x=ecg_time, y=clean_ecg, mode='lines', 
                 name='ECG Artifact Removed', line=dict(color='purple', width=1.5)),
       row=2, col=1
   )

   # Calculate and plot quality metrics
   ecg_snr_original = 20 * np.log10(np.std(ecg_signal) / (np.std(ecg_signal - np.mean(ecg_signal)) + 1e-10))
   ecg_snr_filtered = 20 * np.log10(np.std(filtered_ecg) / (np.std(filtered_ecg - np.mean(filtered_ecg)) + 1e-10))
   ppg_snr_original = 20 * np.log10(np.std(ppg_signal) / (np.std(ppg_signal - np.mean(ppg_signal)) + 1e-10))
   ppg_snr_filtered = 20 * np.log10(np.std(filtered_ppg) / (np.std(filtered_ppg - np.mean(filtered_ppg)) + 1e-10))

   ecg_rms_original = np.sqrt(np.mean(ecg_signal**2))
   ecg_rms_filtered = np.sqrt(np.mean(filtered_ecg**2))
   ppg_rms_original = np.sqrt(np.mean(ppg_signal**2))
   ppg_rms_filtered = np.sqrt(np.mean(filtered_ppg**2))

   quality_metrics = ['ECG SNR', 'PPG SNR', 'ECG RMS', 'PPG RMS']
   original_values = [ecg_snr_original, ppg_snr_original, ecg_rms_original, ppg_rms_original]
   filtered_values = [ecg_snr_filtered, ppg_snr_filtered, ecg_rms_filtered, ppg_rms_filtered]

   fig.add_trace(
       go.Bar(x=quality_metrics, y=original_values, name='Original',
             marker_color='lightblue', opacity=0.7),
       row=2, col=2
   )
   fig.add_trace(
       go.Bar(x=quality_metrics, y=filtered_values, name='Filtered',
             marker_color='lightcoral', opacity=0.7),
       row=2, col=2
   )

   # Update layout and show
   fig.update_layout(height=800, title_text="VitalDSP Signal Processing: Original vs Filtered Signals")
   fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
   fig.update_yaxes(title_text="Amplitude", row=1, col=1)
   fig.update_xaxes(title_text="Time (seconds)", row=1, col=2)
   fig.update_yaxes(title_text="Amplitude", row=1, col=2)
   fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
   fig.update_yaxes(title_text="Amplitude", row=2, col=1)
   fig.update_xaxes(title_text="Quality Metrics", row=2, col=2)
   fig.update_yaxes(title_text="Values", row=2, col=2)

   fig.show()

   # Print quality analysis
   print("Signal Quality Analysis:")
   print(f"ECG SNR Improvement: {ecg_snr_filtered - ecg_snr_original:.2f} dB")
   print(f"PPG SNR Improvement: {ppg_snr_filtered - ppg_snr_original:.2f} dB")
   print(f"ECG Noise Reduction: {((ecg_rms_original - np.sqrt(np.mean(clean_ecg**2))) / ecg_rms_original * 100):.1f}%")

Physiological Features
~~~~~~~~~~~~~~~~~~~~~~

Extract meaningful features from physiological signals:

.. code-block:: python

   # Step 1: Import the feature extraction classes
   from vitalDSP.feature_engineering.morphology_features import PhysiologicalFeatureExtractor
   from vitalDSP.physiological_features.hrv_analysis import HRVFeatures

   # Step 2: Extract physiological features from ECG signal
   print("1. Extracting physiological features...")
   extractor = PhysiologicalFeatureExtractor(filtered_ecg, fs=fs)
   features = extractor.extract_features(signal_type="ECG")
   print(f"   ✓ Extracted {len(features) if features else 0} features")

   # Step 3: Show some example features
   if features:
      print("2. Example features:")
      for key, value in features.items():
            print(f"   • {key}: {value}")

   # Step 4: Extract HRV features
   print("3. Extracting HRV features...")

   hrv = HRVFeatures(signals=filtered_ecg, signal_type="ECG")
   hrv_features = hrv.compute_all_features()

   print(f"   ✓ Extracted {len(hrv_features) if hrv_features else 0} HRV features")

   if hrv_features:
      print("   Example HRV features:")
      for key, value in hrv_features.items():
         print(f"   • {key}: {value}")


   print()
   print("=== Summary ===")
   print("• Use PhysiologicalFeatureExtractor for general features")
   print("• Use HRVFeatures for heart rate variability analysis")
   print("• Both classes work with ECG signals and return dictionaries")

Respiratory Analysis
~~~~~~~~~~~~~~~~~~~~

Analyze respiratory patterns and estimate respiratory rate:

.. code-block:: python

   from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis
   
   sf_ecg = SignalFiltering(ecg_signal)
   resp_low = 0.1
   resp_high = 0.5
   filtered_ecg_resp = sf_ecg.bandpass(lowcut=resp_low, highcut=resp_high, fs=fs)
   print(f"Filtered ECG signal: {len(filtered_ecg_resp)} samples at {60*resp_low}-{60*resp_high} bpm")

   resp_analysis = RespiratoryAnalysis(filtered_ecg_resp, fs)
   respiratory_rate = resp_analysis.compute_respiratory_rate(
      method="fft_based",
      correction_method=None,
      min_breath_duration=1.5,
      max_breath_duration=5,
      preprocess_config=None
   )
   print(respiratory_rate)

Advanced Computation
~~~~~~~~~~~~~~~~~~~~

Use machine learning and advanced algorithms:

.. code-block:: python

    from vitalDSP.advanced_computation.emd import EMD
    from vitalDSP.advanced_computation.pitch_shift import PitchShift
    from vitalDSP.advanced_computation.anomaly_detection import AnomalyDetection
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Perform EMD decomposition
    emd = EMD(ecg_signal)
    imfs = emd.emd()
    print(f"Number of IMFs: {len(imfs)}")

    # Simple plot of all IMFs
    plt.figure(figsize=(12, 8))

    # Plot original signal
    plt.subplot(len(imfs) + 1, 1, 1)
    plt.plot(ecg_signal, 'k-', linewidth=1.5)
    plt.title('Original ECG Signal', fontweight='bold')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    # Plot each IMF
    for i, imf in enumerate(imfs):
        plt.subplot(len(imfs) + 1, 1, i + 2)
        plt.plot(imf, linewidth=1.2, color=sns.color_palette("husl", len(imfs))[i])
        plt.title(f'IMF {i + 1}', fontweight='bold')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Simple heatmap
    plt.figure(figsize=(12, 6))
    imf_matrix = np.array(imfs)
    sns.heatmap(imf_matrix, 
                yticklabels=[f'IMF {i+1}' for i in range(len(imfs))],
                xticklabels=False,
                cmap='RdBu_r',
                center=0)
    plt.title('IMFs Heatmap', fontweight='bold')
    plt.ylabel('Intrinsic Mode Functions')
    plt.tight_layout()
    plt.show()


.. code-block:: python
    # Pitch Shift filtering
    pitch_shift_filter = PitchShift(ecg_signal,sampling_rate=fs)
    filtered_signal = pitch_shift_filter.shift_pitch(semitones=2)


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(filtered_signal)), y=ecg_signal, mode='lines', name='Original Signal'))
    fig.add_trace(go.Scatter(x=np.arange(len(filtered_signal)), y=filtered_signal, mode='lines', name='Pitch Shifted Signal'))
    fig.show()

    # Anomaly detection
    anomaly_detector = AnomalyDetection(ecg_signal)
    anomalies = anomaly_detector.detect_anomalies(method="z_score")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.arange(len(ecg_signal)), x=anomalies, mode='lines', name='Anomlies Lines'))
    fig.show()


Working with Real Data
======================

Loading Data
~~~~~~~~~~~~

VitalDSP supports various data formats:

.. code-block:: python

   import pandas as pd
   from vitalDSP_webapp.services.data_service import DataService
   
   # Load CSV data
   ecg_data = pd.read_csv('sample_data/ECG_short.csv')
   signal = ecg_data['ECG'].values
   

Preprocessing
~~~~~~~~~~~~~

Clean and preprocess your signals:

.. code-block:: python

   from vitalDSP.preprocess.preprocess_operations import PreprocessConfig, preprocess_signal

   # Configure preprocessing
   config = PreprocessConfig(
       filter_type="bandpass",
       noise_reduction_method="wavelet",
       lowcut=0.1,
       highcut=4.5,
       order=4,
       wavelet_name="haar",
       level=1,
       respiratory_mode=False
   )

   # Apply preprocessing
   clean_signal = preprocess_signal(
       signal=ecg_signal,
       sampling_rate=fs,
       filter_type=config.filter_type,
       lowcut=config.lowcut,
       highcut=config.highcut,
       order=config.order,
       noise_reduction_method=config.noise_reduction_method,
       wavelet_name=config.wavelet_name,
       level=config.level,
       window_length=config.window_length,
       polyorder=config.polyorder,
       kernel_size=config.kernel_size,
       sigma=config.sigma,
       respiratory_mode=config.respiratory_mode,
       repreprocess=config.repreprocess
   )

   fig = go.Figure()
   fig.add_trace(go.Scatter(x=np.arange(len(ecg_signal)), y=ecg_signal, mode='lines', name='Original Signal'))
   fig.add_trace(go.Scatter(x=np.arange(len(clean_signal)), y=clean_signal, mode='lines', name='Preprocessed Signal'))
   fig.show()

Feature Extraction Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a simple analysis pipeline:

.. code-block:: python

    # 1. Preprocessing
    from vitalDSP.preprocess.preprocess_operations import PreprocessConfig, preprocess_signal

    # Configure preprocessing
    config = PreprocessConfig(
        filter_type="bandpass",
        noise_reduction_method="wavelet",
        lowcut=0.1,
        highcut=4.5,
        order=4
    )

    # Apply preprocessing
    clean_signal = preprocess_signal(
        signal=ecg_signal,
        sampling_rate=fs,
        filter_type=config.filter_type,
        lowcut=config.lowcut,
        highcut=config.highcut,
        order=config.order,
        noise_reduction_method=config.noise_reduction_method
    )

    print("✅ Preprocessing completed!")

    # 2. Feature extraction
    from vitalDSP.feature_engineering.morphology_features import PhysiologicalFeatureExtractor

    extractor = PhysiologicalFeatureExtractor(clean_signal, fs=fs)
    features = extractor.extract_features(signal_type="ECG")

    print("✅ Features extracted!")
    print(f"Features: {features}")

    # 3. Quality assessment
    from vitalDSP.signal_quality_assessment.signal_quality import SignalQuality

    sq = SignalQuality(ecg_signal, clean_signal)
    quality_metrics = {
        'snr': sq.snr(),
        'psnr': sq.psnr(),
        'mse': sq.mse()
    }

    print("✅ Quality assessment completed!")
    print(f"Quality metrics: {quality_metrics}")

Web Application Features
========================

Interactive Analysis
~~~~~~~~~~~~~~~~~~~~

The web application provides an intuitive interface for:

* **Signal Upload**: Drag-and-drop file upload
* **Real-time Processing**: Live signal analysis
* **Interactive Plots**: Zoom, pan, and export visualizations
* **Feature Extraction**: Automated feature computation
* **Report Generation**: Comprehensive analysis reports

API Integration
~~~~~~~~~~~~~~~

Integrate VitalDSP with your existing applications:

.. code-block:: python

   import requests
   
   # Upload signal data
   with open('signal.csv', 'rb') as f:
       response = requests.post(
           'http://localhost:8050/api/upload',
           files={'file': f}
       )
   
   # Process signal
   response = requests.post(
       'http://localhost:8050/api/process',
       json={
           'signal_id': 'signal_123',
           'operations': ['filter', 'extract_features']
       }
   )

Explore More with Jupyter Notebooks
===================================

For more detailed examples and practical tutorials, explore the Jupyter Notebooks provided with the library:

.. toctree::
   :maxdepth: 2
   :caption: Jupyter Notebooks:

   sample_notebooks

The notebooks cover:

* **Signal Filtering**: Various filtering techniques and their applications
* **Feature Engineering**: Advanced feature extraction methods
* **Health Report Analysis**: Automated report generation
* **Artifact Removal**: Noise reduction and signal cleaning
* **Data Synthesis**: Generating synthetic signals for testing

Best Practices
==============

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

* Use appropriate sampling rates for your analysis
* Consider signal length vs. processing time trade-offs
* Utilize batch processing for multiple signals
* Cache frequently used computations

Data Quality
~~~~~~~~~~~~

* Always assess signal quality before analysis
* Apply appropriate preprocessing steps
* Validate results against known standards
* Document your processing pipeline

Error Handling
~~~~~~~~~~~~~~

* Use try-catch blocks for robust error handling
* Validate input data before processing
* Log important processing steps
* Provide meaningful error messages

Support and Community
=====================

Getting Help
~~~~~~~~~~~~

* **Documentation**: Comprehensive guides and API reference
* **GitHub Issues**: Report bugs and request features
* **Community Forum**: Connect with other users
* **Email Support**: Direct support for enterprise users

Contributing
~~~~~~~~~~~~

We welcome contributions! Check out our GitHub repository:

`VitalDSP GitHub <https://github.com/Oucru-Innovations/vital-DSP>`_

Ways to contribute:

* Report bugs and issues
* Suggest new features
* Submit pull requests
* Improve documentation
* Share use cases and examples

Next Steps
==========

Now that you have VitalDSP installed and running, explore:

1. **Live Demo**: Try the web application immediately at `https://vital-dsp.onrender.com <https://vital-dsp.onrender.com>`_
2. **Core Modules**: Dive deeper into filtering, feature extraction, and analysis
3. **Web Application**: Use the interactive interface for signal processing
4. **Jupyter Notebooks**: Follow along with detailed tutorials
5. **API Reference**: Explore the complete function and class documentation
6. **Advanced Features**: Try machine learning and advanced computation modules

Happy analyzing with VitalDSP! 🫀📊

