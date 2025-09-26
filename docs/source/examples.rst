Examples
=========

This section provides practical examples and real-world use cases for VitalDSP. Each example demonstrates specific functionality and can be used as a starting point for your own projects.

Example Categories
==================

**Clinical Research Examples**
   * ECG analysis for cardiovascular research
   * PPG analysis for hemodynamic studies
   * Respiratory analysis for sleep studies

**Healthcare Monitoring Examples**
   * Real-time vital signs monitoring
   * Continuous patient monitoring
   * Telemedicine applications

**Wearable Device Examples**
   * Fitness tracker integration
   * Smartwatch signal processing
   * Mobile health applications

**Medical Device Examples**
   * Patient monitoring systems
   * Diagnostic equipment integration
   * Clinical decision support systems

Example 1: ECG Analysis for Clinical Research
==============================================

This example demonstrates comprehensive ECG analysis for cardiovascular research applications.

**Use Case:** Analyzing ECG signals from clinical trials to assess cardiovascular health and detect abnormalities.

**Key Features:**
* R-peak detection and RR interval analysis
* Heart rate variability (HRV) analysis
* Morphological feature extraction
* Signal quality assessment
* Clinical interpretation

**Implementation:**

.. code-block:: python

   import numpy as np
   import pandas as pd
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.physiological_features.waveform import WaveformMorphology
   from vitalDSP.physiological_features.hrv_analysis import HRVFeatures
   from vitalDSP.signal_quality_assessment.signal_quality_index import SignalQualityIndex
   
   def analyze_ecg_clinical_research(ecg_signal, fs, patient_id=None):
       """
       Comprehensive ECG analysis for clinical research.
       
       Parameters:
       -----------
       ecg_signal : array-like
           ECG signal data
       fs : float
           Sampling frequency (Hz)
       patient_id : str, optional
           Patient identifier
       
       Returns:
       --------
       dict : Comprehensive analysis results
       """
       
       results = {
           'patient_id': patient_id,
           'sampling_frequency': fs,
           'signal_length': len(ecg_signal),
           'analysis_timestamp': pd.Timestamp.now()
       }
       
       # 1. Signal Preprocessing
       sf = SignalFiltering(ecg_signal, fs)
       filtered_ecg = sf.bandpass_filter(low_cut=0.5, high_cut=40.0)
       
       # 2. Signal Quality Assessment
       sqi = SignalQualityIndex(filtered_ecg)
       quality_metrics = {}
       
       # Amplitude variability SQI
       amp_sqi, _, _ = sqi.amplitude_variability_sqi(
           window_size=fs*5, step_size=fs*1, threshold=2
       )
       quality_metrics['amplitude_variability'] = np.mean(amp_sqi)
       
       # Baseline wander SQI
       baseline_sqi, _, _ = sqi.baseline_wander_sqi(
           window_size=fs*5, step_size=fs*1, threshold=2
       )
       quality_metrics['baseline_wander'] = np.mean(baseline_sqi)
       
       # Signal-to-noise ratio
       snr_sqi, _, _ = sqi.snr_sqi(
           window_size=fs*5, step_size=fs*1, threshold=-1
       )
       quality_metrics['signal_to_noise_ratio'] = np.mean(snr_sqi)
       
       results['quality_metrics'] = quality_metrics
       
       # 3. R-Peak Detection and RR Interval Analysis
       wm = WaveformMorphology(filtered_ecg, fs=fs, signal_type="ECG")
       r_peaks = wm.r_peaks
       
       if len(r_peaks) > 10:  # Ensure sufficient data
           # Calculate RR intervals
           rr_intervals = np.diff(r_peaks) / fs * 1000  # Convert to ms
           
           # Remove outliers
           valid_rr = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
           
           if len(valid_rr) > 5:
               # Heart rate calculation
               heart_rate = 60 * fs / np.mean(np.diff(r_peaks))
               
               # HRV Analysis
               hrv = HRVFeatures(valid_rr)
               hrv_features = hrv.compute_all_features()
               
               results['heart_rate'] = heart_rate
               results['rr_intervals'] = {
                   'count': len(valid_rr),
                   'mean': np.mean(valid_rr),
                   'std': np.std(valid_rr),
                   'min': np.min(valid_rr),
                   'max': np.max(valid_rr)
               }
               results['hrv_features'] = hrv_features
               
               # Morphological features
               morphological_features = {
                   'r_peak_count': len(r_peaks),
                   'qrs_duration': wm.qrs_duration if hasattr(wm, 'qrs_duration') else None,
                   'qt_interval': wm.qt_interval if hasattr(wm, 'qt_interval') else None
               }
               results['morphological_features'] = morphological_features
               
               # Clinical interpretation
               clinical_interpretation = interpret_ecg_results(hrv_features, heart_rate)
               results['clinical_interpretation'] = clinical_interpretation
       
       return results
   
   def interpret_ecg_results(hrv_features, heart_rate):
       """Provide clinical interpretation of ECG analysis results."""
       
       interpretation = {
           'heart_rate_status': 'normal' if 60 <= heart_rate <= 100 else 'abnormal',
           'hrv_status': 'normal',
           'risk_assessment': 'low',
           'recommendations': []
       }
       
       # Heart rate interpretation
       if heart_rate < 60:
           interpretation['heart_rate_status'] = 'bradycardia'
           interpretation['recommendations'].append('Consider bradycardia evaluation')
       elif heart_rate > 100:
           interpretation['heart_rate_status'] = 'tachycardia'
           interpretation['recommendations'].append('Consider tachycardia evaluation')
       
       # HRV interpretation
       if 'sdnn' in hrv_features:
           sdnn = hrv_features['sdnn']
           if sdnn < 30:
               interpretation['hrv_status'] = 'reduced'
               interpretation['risk_assessment'] = 'elevated'
               interpretation['recommendations'].append('Reduced HRV may indicate stress or cardiovascular risk')
           elif sdnn > 50:
               interpretation['hrv_status'] = 'good'
               interpretation['risk_assessment'] = 'low'
       
       return interpretation

**Usage Example:**

.. code-block:: python

   # Load ECG data (replace with your data loading method)
   ecg_data = np.load('ecg_data.npy')  # Your ECG signal
   fs = 1000  # Sampling frequency
   
   # Perform analysis
   results = analyze_ecg_clinical_research(ecg_data, fs, patient_id="P001")
   
   # Display results
   print(f"Patient ID: {results['patient_id']}")
   print(f"Heart Rate: {results['heart_rate']:.1f} BPM")
   print(f"HRV Status: {results['clinical_interpretation']['hrv_status']}")
   print(f"Risk Assessment: {results['clinical_interpretation']['risk_assessment']}")

Example 2: PPG Analysis for Hemodynamic Studies
=================================================

This example demonstrates PPG signal analysis for hemodynamic studies and cardiovascular assessment.

**Use Case:** Analyzing PPG signals to assess blood volume changes, pulse wave characteristics, and cardiovascular health.

**Key Features:**
* Systolic and diastolic peak detection
* Pulse wave analysis
* Hemodynamic parameter estimation
* Signal quality assessment
* Clinical interpretation

**Implementation:**

.. code-block:: python

   def analyze_ppg_hemodynamic(ppg_signal, fs, patient_id=None):
       """
       Comprehensive PPG analysis for hemodynamic studies.
       
       Parameters:
       -----------
       ppg_signal : array-like
           PPG signal data
       fs : float
           Sampling frequency (Hz)
       patient_id : str, optional
           Patient identifier
       
       Returns:
       --------
       dict : Comprehensive PPG analysis results
       """
       
       results = {
           'patient_id': patient_id,
           'sampling_frequency': fs,
           'signal_length': len(ppg_signal),
           'analysis_timestamp': pd.Timestamp.now()
       }
       
       # 1. Signal Preprocessing
       sf = SignalFiltering(ppg_signal, fs)
       filtered_ppg = sf.bandpass_filter(low_cut=0.5, high_cut=8.0)  # PPG-specific range
       
       # 2. Signal Quality Assessment
       sqi = SignalQualityIndex(filtered_ppg)
       quality_metrics = {}
       
       # Amplitude variability SQI
       amp_sqi, _, _ = sqi.amplitude_variability_sqi(
           window_size=fs*5, step_size=fs*1, threshold=2
       )
       quality_metrics['amplitude_variability'] = np.mean(amp_sqi)
       
       # Signal-to-noise ratio
       snr_sqi, _, _ = sqi.snr_sqi(
           window_size=fs*5, step_size=fs*1, threshold=-1
       )
       quality_metrics['signal_to_noise_ratio'] = np.mean(snr_sqi)
       
       results['quality_metrics'] = quality_metrics
       
       # 3. PPG-Specific Analysis
       wm = WaveformMorphology(filtered_ppg, fs=fs, signal_type="PPG")
       
       # Detect systolic peaks
       systolic_peaks = wm.systolic_peaks
       
       if len(systolic_peaks) > 5:
           # Calculate pulse rate
           pulse_rate = 60 * fs / np.mean(np.diff(systolic_peaks))
           
           # Pulse interval analysis
           pulse_intervals = np.diff(systolic_peaks) / fs * 1000  # Convert to ms
           valid_intervals = pulse_intervals[(pulse_intervals > 400) & (pulse_intervals < 2000)]
           
           if len(valid_intervals) > 3:
               # Pulse rate variability (similar to HRV)
               prv_features = {
                   'mean_pulse_interval': np.mean(valid_intervals),
                   'std_pulse_interval': np.std(valid_intervals),
                   'pulse_rate_variability': np.std(valid_intervals) / np.mean(valid_intervals) * 100
               }
               
               # Hemodynamic parameters
               hemodynamic_params = calculate_hemodynamic_parameters(filtered_ppg, systolic_peaks, fs)
               
               results['pulse_rate'] = pulse_rate
               results['pulse_intervals'] = {
                   'count': len(valid_intervals),
                   'mean': np.mean(valid_intervals),
                   'std': np.std(valid_intervals)
               }
               results['prv_features'] = prv_features
               results['hemodynamic_parameters'] = hemodynamic_params
               
               # Clinical interpretation
               clinical_interpretation = interpret_ppg_results(prv_features, pulse_rate, hemodynamic_params)
               results['clinical_interpretation'] = clinical_interpretation
       
       return results
   
   def calculate_hemodynamic_parameters(ppg_signal, systolic_peaks, fs):
       """Calculate hemodynamic parameters from PPG signal."""
       
       params = {}
       
       if len(systolic_peaks) > 2:
           # Pulse amplitude
           pulse_amplitudes = []
           for i in range(len(systolic_peaks) - 1):
               start = systolic_peaks[i]
               end = systolic_peaks[i + 1]
               pulse_segment = ppg_signal[start:end]
               amplitude = np.max(pulse_segment) - np.min(pulse_segment)
               pulse_amplitudes.append(amplitude)
           
           params['mean_pulse_amplitude'] = np.mean(pulse_amplitudes)
           params['pulse_amplitude_variability'] = np.std(pulse_amplitudes) / np.mean(pulse_amplitudes) * 100
           
           # Pulse wave analysis
           params['pulse_wave_analysis'] = analyze_pulse_wave_morphology(ppg_signal, systolic_peaks, fs)
       
       return params
   
   def analyze_pulse_wave_morphology(ppg_signal, systolic_peaks, fs):
       """Analyze pulse wave morphology."""
       
       morphology = {}
       
       if len(systolic_peaks) > 1:
           # Calculate average pulse wave
           pulse_waves = []
           for i in range(len(systolic_peaks) - 1):
               start = systolic_peaks[i]
               end = systolic_peaks[i + 1]
               pulse_wave = ppg_signal[start:end]
               if len(pulse_wave) > 10:  # Ensure sufficient data
                   pulse_waves.append(pulse_wave)
           
           if pulse_waves:
               # Normalize pulse waves to same length
               min_length = min(len(wave) for wave in pulse_waves)
               normalized_waves = [wave[:min_length] for wave in pulse_waves]
               
               # Calculate average pulse wave
               average_pulse_wave = np.mean(normalized_waves, axis=0)
               
               # Analyze morphology
               morphology['pulse_width'] = calculate_pulse_width(average_pulse_wave)
               morphology['pulse_slope'] = calculate_pulse_slope(average_pulse_wave)
               morphology['pulse_area'] = np.trapz(average_pulse_wave)
       
       return morphology
   
   def interpret_ppg_results(prv_features, pulse_rate, hemodynamic_params):
       """Provide clinical interpretation of PPG analysis results."""
       
       interpretation = {
           'pulse_rate_status': 'normal' if 60 <= pulse_rate <= 100 else 'abnormal',
           'hemodynamic_status': 'normal',
           'risk_assessment': 'low',
           'recommendations': []
       }
       
       # Pulse rate interpretation
       if pulse_rate < 60:
           interpretation['pulse_rate_status'] = 'bradycardia'
           interpretation['recommendations'].append('Consider bradycardia evaluation')
       elif pulse_rate > 100:
           interpretation['pulse_rate_status'] = 'tachycardia'
           interpretation['recommendations'].append('Consider tachycardia evaluation')
       
       # Hemodynamic interpretation
       if 'mean_pulse_amplitude' in hemodynamic_params:
           pulse_amplitude = hemodynamic_params['mean_pulse_amplitude']
           if pulse_amplitude < 0.1:  # Threshold depends on your signal scaling
               interpretation['hemodynamic_status'] = 'reduced'
               interpretation['risk_assessment'] = 'elevated'
               interpretation['recommendations'].append('Reduced pulse amplitude may indicate poor perfusion')
       
       return interpretation

Example 3: Real-Time Vital Signs Monitoring
=============================================

This example demonstrates real-time vital signs monitoring using VitalDSP.

**Use Case:** Continuous monitoring of vital signs in clinical settings or remote patient monitoring applications.

**Key Features:**
* Real-time signal processing
* Continuous vital signs calculation
* Alert generation for abnormal values
* Data logging and storage
* Web interface for monitoring

**Implementation:**

.. code-block:: python

   import time
   import threading
   from collections import deque
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.physiological_features.waveform import WaveformMorphology
   from vitalDSP.signal_quality_assessment.signal_quality_index import SignalQualityIndex
   
   class RealTimeVitalSignsMonitor:
       """Real-time vital signs monitoring system."""
       
       def __init__(self, fs=1000, window_size=10, update_interval=1):
           """
           Initialize real-time monitor.
           
           Parameters:
           -----------
           fs : float
               Sampling frequency (Hz)
           window_size : int
               Window size in seconds
           update_interval : float
               Update interval in seconds
           """
           
           self.fs = fs
           self.window_size = window_size
           self.update_interval = update_interval
           self.window_samples = fs * window_size
           
           # Data buffers
           self.signal_buffer = deque(maxlen=self.window_samples)
           self.vital_signs_history = deque(maxlen=100)  # Keep last 100 measurements
           
           # Current vital signs
           self.current_vital_signs = {
               'heart_rate': None,
               'pulse_rate': None,
               'signal_quality': None,
               'timestamp': None
           }
           
           # Alert thresholds
           self.alert_thresholds = {
               'heart_rate_low': 50,
               'heart_rate_high': 120,
               'signal_quality_low': 0.5
           }
           
           # Monitoring state
           self.is_monitoring = False
           self.monitor_thread = None
           
       def add_signal_data(self, signal_chunk):
           """Add new signal data to the buffer."""
           self.signal_buffer.extend(signal_chunk)
       
       def start_monitoring(self):
           """Start real-time monitoring."""
           if not self.is_monitoring:
               self.is_monitoring = True
               self.monitor_thread = threading.Thread(target=self._monitoring_loop)
               self.monitor_thread.daemon = True
               self.monitor_thread.start()
               print("Real-time monitoring started")
       
       def stop_monitoring(self):
           """Stop real-time monitoring."""
           self.is_monitoring = False
           if self.monitor_thread:
               self.monitor_thread.join()
               print("Real-time monitoring stopped")
       
       def _monitoring_loop(self):
           """Main monitoring loop."""
           while self.is_monitoring:
               if len(self.signal_buffer) >= self.window_samples:
                   # Get current window
                   current_signal = np.array(list(self.signal_buffer))
                   
                   # Process signal
                   vital_signs = self._process_signal_window(current_signal)
                   
                   # Update current vital signs
                   self.current_vital_signs.update(vital_signs)
                   self.current_vital_signs['timestamp'] = time.time()
                   
                   # Add to history
                   self.vital_signs_history.append(vital_signs.copy())
                   
                   # Check for alerts
                   self._check_alerts(vital_signs)
                   
                   # Log results
                   self._log_vital_signs(vital_signs)
               
               time.sleep(self.update_interval)
       
       def _process_signal_window(self, signal):
           """Process a window of signal data."""
           vital_signs = {}
           
           try:
               # Signal preprocessing
               sf = SignalFiltering(signal, self.fs)
               filtered_signal = sf.bandpass_filter(low_cut=0.5, high_cut=40.0)
               
               # Signal quality assessment
               sqi = SignalQualityIndex(filtered_signal)
               quality_sqi, _, _ = sqi.amplitude_variability_sqi(
                   window_size=self.fs*5, step_size=self.fs*1, threshold=2
               )
               vital_signs['signal_quality'] = np.mean(quality_sqi)
               
               # Detect peaks (assuming ECG signal)
               wm = WaveformMorphology(filtered_signal, fs=self.fs, signal_type="ECG")
               r_peaks = wm.r_peaks
               
               if len(r_peaks) > 2:
                   # Calculate heart rate
                   rr_intervals = np.diff(r_peaks) / self.fs
                   valid_rr = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]
                   
                   if len(valid_rr) > 1:
                       heart_rate = 60 / np.mean(valid_rr)
                       vital_signs['heart_rate'] = heart_rate
                       vital_signs['pulse_rate'] = heart_rate  # Assuming same for this example
               
           except Exception as e:
               print(f"Error processing signal: {e}")
               vital_signs['error'] = str(e)
           
           return vital_signs
       
       def _check_alerts(self, vital_signs):
           """Check for alert conditions."""
           alerts = []
           
           if 'heart_rate' in vital_signs:
               hr = vital_signs['heart_rate']
               if hr < self.alert_thresholds['heart_rate_low']:
                   alerts.append(f"Low heart rate: {hr:.1f} BPM")
               elif hr > self.alert_thresholds['heart_rate_high']:
                   alerts.append(f"High heart rate: {hr:.1f} BPM")
           
           if 'signal_quality' in vital_signs:
               sq = vital_signs['signal_quality']
               if sq < self.alert_thresholds['signal_quality_low']:
                   alerts.append(f"Poor signal quality: {sq:.2f}")
           
           if alerts:
               self._handle_alerts(alerts)
       
       def _handle_alerts(self, alerts):
           """Handle alert conditions."""
           timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
           for alert in alerts:
               print(f"[ALERT {timestamp}] {alert}")
               # Here you could send notifications, log to database, etc.
       
       def _log_vital_signs(self, vital_signs):
           """Log vital signs data."""
           timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
           log_entry = f"[{timestamp}] "
           
           if 'heart_rate' in vital_signs:
               log_entry += f"HR: {vital_signs['heart_rate']:.1f} BPM "
           
           if 'signal_quality' in vital_signs:
               log_entry += f"SQ: {vital_signs['signal_quality']:.2f}"
           
           print(log_entry)
       
       def get_current_vital_signs(self):
           """Get current vital signs."""
           return self.current_vital_signs.copy()
       
       def get_vital_signs_history(self):
           """Get vital signs history."""
           return list(self.vital_signs_history)
       
       def get_statistics(self):
           """Get statistics from vital signs history."""
           if not self.vital_signs_history:
               return {}
           
           stats = {}
           
           # Heart rate statistics
           heart_rates = [vs.get('heart_rate') for vs in self.vital_signs_history if vs.get('heart_rate')]
           if heart_rates:
               stats['heart_rate'] = {
                   'mean': np.mean(heart_rates),
                   'std': np.std(heart_rates),
                   'min': np.min(heart_rates),
                   'max': np.max(heart_rates)
               }
           
           # Signal quality statistics
           signal_qualities = [vs.get('signal_quality') for vs in self.vital_signs_history if vs.get('signal_quality')]
           if signal_qualities:
               stats['signal_quality'] = {
                   'mean': np.mean(signal_qualities),
                   'std': np.std(signal_qualities),
                   'min': np.min(signal_qualities),
                   'max': np.max(signal_qualities)
               }
           
           return stats

**Usage Example:**

.. code-block:: python

   # Initialize real-time monitor
   monitor = RealTimeVitalSignsMonitor(fs=1000, window_size=10, update_interval=1)
   
   # Start monitoring
   monitor.start_monitoring()
   
   # Simulate real-time data (replace with your data source)
   for i in range(100):
       # Generate sample data (replace with your data acquisition)
       sample_data = np.random.randn(100) + np.sin(2 * np.pi * 1.2 * np.arange(100) / 1000)
       monitor.add_signal_data(sample_data)
       time.sleep(0.1)
   
   # Get current vital signs
   current_vitals = monitor.get_current_vital_signs()
   print(f"Current heart rate: {current_vitals.get('heart_rate', 'N/A')} BPM")
   
   # Get statistics
   stats = monitor.get_statistics()
   print(f"Average heart rate: {stats.get('heart_rate', {}).get('mean', 'N/A')} BPM")
   
   # Stop monitoring
   monitor.stop_monitoring()

Example 4: Wearable Device Integration
========================================

This example demonstrates how to integrate VitalDSP with wearable devices for health monitoring.

**Use Case:** Processing data from fitness trackers, smartwatches, or other wearable health devices.

**Key Features:**
* Data format conversion and preprocessing
* Real-time processing for wearable constraints
* Battery-efficient algorithms
* Mobile-friendly analysis
* Cloud integration

**Implementation:**

.. code-block:: python

   import json
   import requests
   from datetime import datetime, timedelta
   
   class WearableDeviceIntegration:
       """Integration with wearable devices for health monitoring."""
       
       def __init__(self, device_type="fitness_tracker", cloud_endpoint=None):
           """
           Initialize wearable device integration.
           
           Parameters:
           -----------
           device_type : str
               Type of wearable device
           cloud_endpoint : str, optional
               Cloud endpoint for data synchronization
           """
           
           self.device_type = device_type
           self.cloud_endpoint = cloud_endpoint
           self.device_config = self._get_device_config(device_type)
           
           # Data processing parameters optimized for wearables
           self.processing_params = {
               'fs': self.device_config['sampling_rate'],
               'window_size': 30,  # 30-second windows
               'update_interval': 5,  # 5-second updates
               'battery_optimized': True
           }
           
           # Local data storage
           self.local_data = []
           self.processed_data = []
           
       def _get_device_config(self, device_type):
           """Get device-specific configuration."""
           
           configs = {
               'fitness_tracker': {
                   'sampling_rate': 100,
                   'signal_types': ['ppg', 'accelerometer'],
                   'battery_life': '7_days',
                   'data_format': 'json'
               },
               'smartwatch': {
                   'sampling_rate': 200,
                   'signal_types': ['ppg', 'ecg', 'accelerometer'],
                   'battery_life': '2_days',
                   'data_format': 'json'
               },
               'chest_strap': {
                   'sampling_rate': 1000,
                   'signal_types': ['ecg', 'accelerometer'],
                   'battery_life': '24_hours',
                   'data_format': 'csv'
               }
           }
           
           return configs.get(device_type, configs['fitness_tracker'])
       
       def process_wearable_data(self, raw_data, data_type="ppg"):
           """
           Process data from wearable device.
           
           Parameters:
           -----------
           raw_data : dict or array
               Raw data from wearable device
           data_type : str
               Type of signal data
           
           Returns:
           --------
           dict : Processed analysis results
           """
           
           # Convert raw data to standard format
           processed_signal = self._convert_wearable_data(raw_data, data_type)
           
           if processed_signal is None:
               return {'error': 'Failed to convert data'}
           
           # Apply battery-optimized processing
           results = self._battery_optimized_processing(processed_signal, data_type)
           
           # Add metadata
           results['device_type'] = self.device_type
           results['data_type'] = data_type
           results['timestamp'] = datetime.now().isoformat()
           results['processing_params'] = self.processing_params
           
           # Store locally
           self.processed_data.append(results)
           
           # Sync to cloud if enabled
           if self.cloud_endpoint:
               self._sync_to_cloud(results)
           
           return results
       
       def _convert_wearable_data(self, raw_data, data_type):
           """Convert wearable data to standard format."""
           
           try:
               if self.device_config['data_format'] == 'json':
                   if isinstance(raw_data, str):
                       data = json.loads(raw_data)
                   else:
                       data = raw_data
                   
                   # Extract signal data based on device type
                   if self.device_type == 'fitness_tracker':
                       signal = data.get('ppg_data', [])
                   elif self.device_type == 'smartwatch':
                       signal = data.get('heart_rate_data', [])
                   elif self.device_type == 'chest_strap':
                       signal = data.get('ecg_data', [])
                   else:
                       signal = data.get('signal_data', [])
               
               elif self.device_config['data_format'] == 'csv':
                   # Handle CSV data
                   signal = raw_data if isinstance(raw_data, list) else raw_data.tolist()
               
               else:
                   signal = raw_data
               
               # Convert to numpy array
               signal_array = np.array(signal)
               
               # Validate signal
               if len(signal_array) < 10:
                   return None
               
               return signal_array
           
           except Exception as e:
               print(f"Error converting wearable data: {e}")
               return None
       
       def _battery_optimized_processing(self, signal, data_type):
           """Apply battery-optimized signal processing."""
           
           results = {}
           
           try:
               fs = self.processing_params['fs']
               
               # Lightweight preprocessing
               sf = SignalFiltering(signal, fs)
               
               # Use simpler filters for battery optimization
               if data_type == 'ppg':
                   filtered_signal = sf.bandpass_filter(low_cut=0.5, high_cut=8.0, filter_order=2)
               elif data_type == 'ecg':
                   filtered_signal = sf.bandpass_filter(low_cut=0.5, high_cut=40.0, filter_order=2)
               else:
                   filtered_signal = sf.bandpass_filter(low_cut=0.1, high_cut=20.0, filter_order=2)
               
               # Basic feature extraction
               if data_type in ['ppg', 'ecg']:
                   wm = WaveformMorphology(filtered_signal, fs=fs, signal_type=data_type)
                   
                   if data_type == 'ppg':
                       peaks = wm.systolic_peaks
                   else:
                       peaks = wm.r_peaks
                   
                   if len(peaks) > 2:
                       # Calculate heart rate
                       intervals = np.diff(peaks) / fs
                       valid_intervals = intervals[(intervals > 0.3) & (intervals < 2.0)]
                       
                       if len(valid_intervals) > 1:
                           heart_rate = 60 / np.mean(valid_intervals)
                           results['heart_rate'] = heart_rate
                           results['heart_rate_variability'] = np.std(valid_intervals) / np.mean(valid_intervals) * 100
               
               # Basic signal quality
               sqi = SignalQualityIndex(filtered_signal)
               quality_sqi, _, _ = sqi.amplitude_variability_sqi(
                   window_size=fs*5, step_size=fs*1, threshold=2
               )
               results['signal_quality'] = np.mean(quality_sqi)
               
               # Activity level (if accelerometer data available)
               if data_type == 'accelerometer':
                   results['activity_level'] = self._calculate_activity_level(signal)
           
           except Exception as e:
               print(f"Error in battery-optimized processing: {e}")
               results['error'] = str(e)
           
           return results
       
       def _calculate_activity_level(self, accelerometer_data):
           """Calculate activity level from accelerometer data."""
           
           if len(accelerometer_data) < 3:
               return 'unknown'
           
           # Calculate magnitude of acceleration
           if accelerometer_data.ndim > 1:
               magnitude = np.sqrt(np.sum(accelerometer_data**2, axis=1))
           else:
               magnitude = np.abs(accelerometer_data)
           
           # Calculate activity level
           mean_magnitude = np.mean(magnitude)
           
           if mean_magnitude < 1.0:
               return 'sedentary'
           elif mean_magnitude < 2.0:
               return 'light'
           elif mean_magnitude < 3.0:
               return 'moderate'
           else:
               return 'vigorous'
       
       def _sync_to_cloud(self, data):
           """Sync processed data to cloud endpoint."""
           
           try:
               if self.cloud_endpoint:
                   response = requests.post(
                       self.cloud_endpoint,
                       json=data,
                       timeout=10
                   )
                   
                   if response.status_code == 200:
                       print("Data synced to cloud successfully")
                   else:
                       print(f"Cloud sync failed: {response.status_code}")
           
           except Exception as e:
               print(f"Cloud sync error: {e}")
       
       def get_health_summary(self, hours=24):
           """Get health summary for specified time period."""
           
           cutoff_time = datetime.now() - timedelta(hours=hours)
           
           # Filter data for specified time period
           recent_data = [
               data for data in self.processed_data
               if datetime.fromisoformat(data['timestamp']) > cutoff_time
           ]
           
           if not recent_data:
               return {'message': 'No data available for specified period'}
           
           # Calculate summary statistics
           summary = {
               'period_hours': hours,
               'data_points': len(recent_data),
               'heart_rate_stats': self._calculate_heart_rate_stats(recent_data),
               'activity_summary': self._calculate_activity_summary(recent_data),
               'signal_quality_stats': self._calculate_signal_quality_stats(recent_data)
           }
           
           return summary
       
       def _calculate_heart_rate_stats(self, data):
           """Calculate heart rate statistics."""
           
           heart_rates = [d.get('heart_rate') for d in data if d.get('heart_rate')]
           
           if not heart_rates:
               return None
           
           return {
               'mean': np.mean(heart_rates),
               'std': np.std(heart_rates),
               'min': np.min(heart_rates),
               'max': np.max(heart_rates),
               'count': len(heart_rates)
           }
       
       def _calculate_activity_summary(self, data):
           """Calculate activity summary."""
           
           activities = [d.get('activity_level') for d in data if d.get('activity_level')]
           
           if not activities:
               return None
           
           activity_counts = {}
           for activity in activities:
               activity_counts[activity] = activity_counts.get(activity, 0) + 1
           
           return activity_counts
       
       def _calculate_signal_quality_stats(self, data):
           """Calculate signal quality statistics."""
           
           qualities = [d.get('signal_quality') for d in data if d.get('signal_quality')]
           
           if not qualities:
               return None
           
           return {
               'mean': np.mean(qualities),
               'std': np.std(qualities),
               'min': np.min(qualities),
               'max': np.max(qualities)
           }

**Usage Example:**

.. code-block:: python

   # Initialize wearable integration
   wearable = WearableDeviceIntegration(
       device_type="fitness_tracker",
       cloud_endpoint="https://api.example.com/vital-signs"
   )
   
   # Simulate wearable data
   sample_data = {
       'ppg_data': np.random.randn(1000) + np.sin(2 * np.pi * 1.2 * np.arange(1000) / 100),
       'timestamp': datetime.now().isoformat(),
       'device_id': 'tracker_001'
   }
   
   # Process wearable data
   results = wearable.process_wearable_data(sample_data, data_type="ppg")
   
   print(f"Heart rate: {results.get('heart_rate', 'N/A')} BPM")
   print(f"Signal quality: {results.get('signal_quality', 'N/A')}")
   
   # Get health summary
   summary = wearable.get_health_summary(hours=24)
   print(f"Health summary: {summary}")

Best Practices for Examples
============================

**Code Organization**
* Use clear, descriptive function and variable names
* Include comprehensive docstrings
* Implement proper error handling
* Follow PEP 8 style guidelines

**Performance Considerations**
* Optimize for your specific use case
* Consider memory usage for large datasets
* Use appropriate sampling rates
* Implement efficient data structures

**Clinical Applications**
* Validate results against clinical standards
* Consider patient safety and data privacy
* Document methodology for reproducibility
* Include appropriate disclaimers

**Integration Guidelines**
* Design for your target platform
* Consider real-time constraints
* Implement proper data validation
* Include comprehensive logging

**Testing and Validation**
* Test with various signal types and qualities
* Validate against known datasets
* Include edge case handling
* Document test procedures

These examples provide a solid foundation for implementing VitalDSP in various real-world scenarios. Adapt them to your specific needs and requirements.

For more advanced examples and use cases, explore the Jupyter notebooks in the :ref:`sample_notebooks` section.
