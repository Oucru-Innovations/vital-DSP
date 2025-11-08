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
   sf = SignalFiltering(ecg_signal)
   filtered_ecg = sf.bandpass(lowcut=0.5, highcut=40.0, fs=fs, order=4)
       
       # 2. Signal Quality Assessment
       sqi = SignalQualityIndex(filtered_ecg)
       quality_metrics = {}
       
   # Amplitude variability SQI
   amp_sqi, _, _ = sqi.amplitude_variability_sqi(
       window_size=int(fs*5), step_size=int(fs*1), threshold=2, aggregate=False
   )
   quality_metrics['amplitude_variability'] = np.mean(amp_sqi)
   
   # Baseline wander SQI
   baseline_sqi, _, _ = sqi.baseline_wander_sqi(
       window_size=int(fs*5), step_size=int(fs*1), threshold=2, aggregate=False
   )
   quality_metrics['baseline_wander'] = np.mean(baseline_sqi)
   
   # Signal-to-noise ratio
   snr_sqi, _, _ = sqi.snr_sqi(
       window_size=int(fs*5), step_size=int(fs*1), threshold=-1, aggregate=False
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
   sf = SignalFiltering(ppg_signal)
   filtered_ppg = sf.bandpass(lowcut=0.5, highcut=8.0, fs=fs, order=4)  # PPG-specific range
       
       # 2. Signal Quality Assessment
       sqi = SignalQualityIndex(filtered_ppg)
       quality_metrics = {}
       
   # Amplitude variability SQI
   amp_sqi, _, _ = sqi.amplitude_variability_sqi(
       window_size=int(fs*5), step_size=int(fs*1), threshold=2, aggregate=False
   )
   quality_metrics['amplitude_variability'] = np.mean(amp_sqi)
   
   # Signal-to-noise ratio
   snr_sqi, _, _ = sqi.snr_sqi(
       window_size=int(fs*5), step_size=int(fs*1), threshold=-1, aggregate=False
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
           sf = SignalFiltering(signal)
           filtered_signal = sf.bandpass(lowcut=0.5, highcut=40.0, fs=self.fs, order=4)
           
           # Signal quality assessment
           sqi = SignalQualityIndex(filtered_signal)
           quality_sqi, _, _ = sqi.amplitude_variability_sqi(
               window_size=int(self.fs*5), step_size=int(self.fs*1), threshold=2, aggregate=False
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
               sf = SignalFiltering(signal)
               
               # Use simpler filters for battery optimization
               if data_type == 'ppg':
                   filtered_signal = sf.bandpass(lowcut=0.5, highcut=8.0, fs=fs, order=2)
               elif data_type == 'ecg':
                   filtered_signal = sf.bandpass(lowcut=0.5, highcut=40.0, fs=fs, order=2)
               else:
                   filtered_signal = sf.bandpass(lowcut=0.1, highcut=20.0, fs=fs, order=2)
               
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
               window_size=int(fs*5), step_size=int(fs*1), threshold=2, aggregate=False
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

Example 5: Advanced Multi-Scale Entropy Analysis
==================================================

This example demonstrates advanced nonlinear analysis techniques for assessing physiological signal complexity.

**Use Case:** Quantifying cardiac health and autonomic nervous system function through multi-scale entropy analysis.

**Key Features:**
* Multi-scale entropy (MSE) computation
* Composite multi-scale entropy (CMSE)
* Refined composite multi-scale entropy (RCMSE)
* Clinical interpretation of entropy metrics
* Complexity-loss aging assessment

**Implementation:**

.. code-block:: python

   import numpy as np
   from vitalDSP.physiological_features.advanced_entropy import MultiScaleEntropy
   from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal
   from vitalDSP.physiological_features.waveform import WaveformMorphology
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   
   def analyze_cardiac_complexity(ecg_signal, fs, patient_id=None):
       """
       Comprehensive cardiac complexity analysis using multi-scale entropy.
       
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
       dict : Comprehensive complexity analysis results
       """
       
       results = {
           'patient_id': patient_id,
           'sampling_frequency': fs,
           'analysis': 'Multi-Scale Entropy'
       }
       
       # 1. Preprocess ECG signal
       sf = SignalFiltering(ecg_signal)
       filtered_ecg = sf.bandpass(lowcut=0.5, highcut=40.0, fs=fs, order=4)
       
       # 2. Extract RR intervals
       wm = WaveformMorphology(filtered_ecg, fs=fs, signal_type="ECG")
       r_peaks = wm.r_peaks
       
       if len(r_peaks) > 50:
           # Calculate RR intervals in milliseconds
           rr_intervals = np.diff(r_peaks) / fs * 1000
           
           # Remove outliers
           valid_rr = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
           
           if len(valid_rr) > 50:
               # 3. Multi-Scale Entropy Analysis
               mse_analyzer = MultiScaleEntropy(
                   signal=valid_rr,
                   m=2,           # Embedding dimension
                   r_multiplier=0.15,  # Tolerance multiplier
                   max_scale=20   # Maximum scale
               )
               
               # Standard MSE
               mse_values, complexity_index = mse_analyzer.compute_mse()
               results['mse_values'] = mse_values.tolist()
               results['complexity_index'] = float(complexity_index)
               
               # Composite MSE (more robust)
               cmse_values, cmse_ci = mse_analyzer.compute_cmse()
               results['cmse_values'] = cmse_values.tolist()
               results['cmse_complexity_index'] = float(cmse_ci)
               
               # Refined Composite MSE (best for short signals)
               rcmse_values, rcmse_ci = mse_analyzer.compute_rcmse()
               results['rcmse_values'] = rcmse_values.tolist()
               results['rcmse_complexity_index'] = float(rcmse_ci)
               
               # 4. Clinical Interpretation
               interpretation = interpret_complexity_metrics(
                   complexity_index,
                   cmse_ci,
                   rcmse_ci,
                   len(valid_rr)
               )
               results['clinical_interpretation'] = interpretation
       
       return results
   
   def interpret_complexity_metrics(mse_ci, cmse_ci, rcmse_ci, signal_length):
       """Provide clinical interpretation of complexity metrics."""
       
       interpretation = {
           'complexity_level': 'normal',
           'autonomic_status': 'balanced',
           'cardiac_health': 'healthy',
           'recommendations': []
       }
       
       # Complexity Index thresholds (clinically validated)
       if cmse_ci < 30:
           interpretation['complexity_level'] = 'reduced'
           interpretation['autonomic_status'] = 'impaired'
           interpretation['cardiac_health'] = 'at_risk'
           interpretation['recommendations'].append(
               'Reduced complexity suggests autonomic dysfunction or aging'
           )
           interpretation['recommendations'].append(
               'Consider comprehensive cardiac evaluation'
           )
       elif cmse_ci > 60:
           interpretation['complexity_level'] = 'high'
           interpretation['autonomic_status'] = 'healthy'
           interpretation['cardiac_health'] = 'excellent'
           interpretation['recommendations'].append(
               'High complexity indicates good cardiovascular health'
           )
       else:
           interpretation['complexity_level'] = 'normal'
           interpretation['autonomic_status'] = 'balanced'
           interpretation['recommendations'].append(
               'Complexity within normal range for age'
           )
       
       # Signal quality check
       if signal_length < 100:
           interpretation['recommendations'].append(
               'Signal length is short - consider longer recording for robust assessment'
           )
       
       return interpretation

**Usage Example:**

.. code-block:: python

   # Generate or load ECG data
   fs = 256
   ecg_data = generate_ecg_signal(sfecg=fs, duration=300, hrmean=72, Anoise=0.05)
   
   # Perform complexity analysis
   results = analyze_cardiac_complexity(ecg_data, fs, patient_id="P001")
   
   # Display results
   print(f"Patient ID: {results['patient_id']}")
   print(f"Complexity Index (CMSE): {results['cmse_complexity_index']:.2f}")
   print(f"Cardiac Health: {results['clinical_interpretation']['cardiac_health']}")
   print(f"Autonomic Status: {results['clinical_interpretation']['autonomic_status']}")
   
   # Plot MSE curves
   import matplotlib.pyplot as plt
   
   scales = range(1, len(results['mse_values']) + 1)
   plt.figure(figsize=(12, 6))
   plt.plot(scales, results['mse_values'], 'b-', label='MSE', linewidth=2)
   plt.plot(scales, results['cmse_values'], 'r-', label='CMSE', linewidth=2)
   plt.plot(scales, results['rcmse_values'], 'g-', label='RCMSE', linewidth=2)
   plt.xlabel('Scale Factor')
   plt.ylabel('Sample Entropy')
   plt.title('Multi-Scale Entropy Analysis')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Example 6: Comprehensive Health Monitoring System
===================================================

This example demonstrates a complete health monitoring pipeline with automated report generation.

**Use Case:** Continuous health monitoring with automated assessment and alerting for clinical decision support.

**Key Features:**
* Multi-modal signal processing (ECG, PPG, Respiratory)
* Comprehensive feature extraction
* Automated health report generation
* Clinical decision support
* Trend analysis and alerts

**Implementation:**

.. code-block:: python

   from vitalDSP.health_analysis.health_report_generator import HealthReportGenerator
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.physiological_features.hrv_analysis import HRVFeatures
   from vitalDSP.physiological_features.waveform import WaveformMorphology
   from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis
   import pandas as pd
   import numpy as np
   
   class ComprehensiveHealthMonitor:
       """Complete health monitoring system with automated reporting."""
       
       def __init__(self, patient_id, fs=256):
           """
           Initialize health monitoring system.
           
           Parameters:
           -----------
           patient_id : str
               Patient identifier
           fs : float
               Sampling frequency (Hz)
           """
           self.patient_id = patient_id
           self.fs = fs
           self.monitoring_history = []
           
       def process_vital_signs(self, ecg_signal, ppg_signal=None, resp_signal=None):
           """
           Process multiple vital signs and generate comprehensive assessment.
           
           Parameters:
           -----------
           ecg_signal : array-like
               ECG signal data
           ppg_signal : array-like, optional
               PPG signal data
           resp_signal : array-like, optional
               Respiratory signal data
           
           Returns:
           --------
           dict : Comprehensive health assessment
           """
           
           assessment = {
               'patient_id': self.patient_id,
               'timestamp': pd.Timestamp.now(),
               'vital_signs': {}
           }
           
           # Process ECG signal
           if ecg_signal is not None:
               ecg_results = self._process_ecg(ecg_signal)
               assessment['vital_signs']['ecg'] = ecg_results
           
           # Process PPG signal
           if ppg_signal is not None:
               ppg_results = self._process_ppg(ppg_signal)
               assessment['vital_signs']['ppg'] = ppg_results
           
           # Process Respiratory signal
           if resp_signal is not None:
               resp_results = self._process_respiratory(resp_signal)
               assessment['vital_signs']['respiratory'] = resp_results
           
           # Generate health score
           health_score = self._calculate_health_score(assessment['vital_signs'])
           assessment['health_score'] = health_score
           
           # Clinical recommendations
           assessment['recommendations'] = self._generate_recommendations(
               assessment['vital_signs'],
               health_score
           )
           
           # Store in history
           self.monitoring_history.append(assessment)
           
           return assessment
       
       def _process_ecg(self, ecg_signal):
           """Process ECG signal and extract features."""
           
           results = {}
           
           # Filter ECG
           sf = SignalFiltering(ecg_signal)
           filtered_ecg = sf.bandpass(lowcut=0.5, highcut=40.0, fs=self.fs, order=4)
           
           # Detect R-peaks
           wm = WaveformMorphology(filtered_ecg, fs=self.fs, signal_type="ECG")
           r_peaks = wm.r_peaks
           
           if len(r_peaks) > 10:
               # RR intervals
               rr_intervals = np.diff(r_peaks) / self.fs * 1000
               valid_rr = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
               
               if len(valid_rr) > 5:
                   # Heart rate
                   heart_rate = 60000 / np.mean(valid_rr)
                   results['heart_rate'] = float(heart_rate)
                   results['heart_rate_status'] = self._assess_heart_rate(heart_rate)
                   
                   # HRV analysis
                   hrv = HRVFeatures(valid_rr)
                   hrv_features = hrv.compute_all_features()
                   results['hrv'] = hrv_features
                   results['hrv_status'] = self._assess_hrv(hrv_features)
           
           return results
       
       def _process_ppg(self, ppg_signal):
           """Process PPG signal and extract features."""
           
           results = {}
           
           # Filter PPG
           sf = SignalFiltering(ppg_signal)
           filtered_ppg = sf.bandpass(lowcut=0.5, highcut=8.0, fs=self.fs, order=4)
           
           # Detect peaks
           wm = WaveformMorphology(filtered_ppg, fs=self.fs, signal_type="PPG")
           systolic_peaks = wm.systolic_peaks
           
           if len(systolic_peaks) > 5:
               # Pulse rate
               pulse_intervals = np.diff(systolic_peaks) / self.fs * 1000
               valid_intervals = pulse_intervals[(pulse_intervals > 400) & (pulse_intervals < 2000)]
               
               if len(valid_intervals) > 3:
                   pulse_rate = 60000 / np.mean(valid_intervals)
                   results['pulse_rate'] = float(pulse_rate)
                   results['pulse_rate_status'] = self._assess_heart_rate(pulse_rate)
                   
                   # Pulse rate variability
                   prv = np.std(valid_intervals)
                   results['pulse_rate_variability'] = float(prv)
           
           return results
       
       def _process_respiratory(self, resp_signal):
           """Process respiratory signal."""
           
           results = {}
           
           try:
               # Respiratory analysis
               resp_analyzer = RespiratoryAnalysis(resp_signal, self.fs)
               
               # Preprocess
               filtered_resp = resp_analyzer.preprocess_signal(
                   detrend=True,
                   normalize=True,
                   filter_type='bandpass',
                   low_freq=0.1,
                   high_freq=0.5
               )
               
               # Estimate rate
               resp_rate = resp_analyzer.compute_respiratory_rate()
               results['respiratory_rate'] = float(resp_rate)
               results['respiratory_status'] = self._assess_respiratory_rate(resp_rate)
           
           except Exception as e:
               results['error'] = str(e)
           
           return results
       
       def _assess_heart_rate(self, hr):
           """Assess heart rate status."""
           if hr < 60:
               return 'bradycardia'
           elif hr > 100:
               return 'tachycardia'
           else:
               return 'normal'
       
       def _assess_hrv(self, hrv_features):
           """Assess HRV status."""
           if 'sdnn' in hrv_features:
               sdnn = hrv_features['sdnn']
               if sdnn < 30:
                   return 'reduced'
               elif sdnn > 50:
                   return 'good'
               else:
                   return 'normal'
           return 'unknown'
       
       def _assess_respiratory_rate(self, rr):
           """Assess respiratory rate status."""
           if rr < 12:
               return 'bradypnea'
           elif rr > 20:
               return 'tachypnea'
           else:
               return 'normal'
       
       def _calculate_health_score(self, vital_signs):
           """Calculate overall health score (0-100)."""
           
           score = 100
           
           # ECG assessment
           if 'ecg' in vital_signs:
               ecg = vital_signs['ecg']
               if ecg.get('heart_rate_status') != 'normal':
                   score -= 15
               if ecg.get('hrv_status') == 'reduced':
                   score -= 20
               elif ecg.get('hrv_status') == 'good':
                   score += 5
           
           # PPG assessment
           if 'ppg' in vital_signs:
               ppg = vital_signs['ppg']
               if ppg.get('pulse_rate_status') != 'normal':
                   score -= 10
           
           # Respiratory assessment
           if 'respiratory' in vital_signs:
               resp = vital_signs['respiratory']
               if resp.get('respiratory_status') != 'normal':
                   score -= 15
           
           return max(0, min(100, score))
       
       def _generate_recommendations(self, vital_signs, health_score):
           """Generate clinical recommendations."""
           
           recommendations = []
           
           if health_score < 60:
               recommendations.append('⚠️ URGENT: Immediate clinical evaluation recommended')
           elif health_score < 75:
               recommendations.append('⚠️ Clinical review recommended within 24 hours')
           
           # Specific recommendations
           if 'ecg' in vital_signs:
               ecg = vital_signs['ecg']
               if ecg.get('heart_rate_status') == 'bradycardia':
                   recommendations.append('• Evaluate for bradycardia causes')
               elif ecg.get('heart_rate_status') == 'tachycardia':
                   recommendations.append('• Assess for tachycardia etiology')
               
               if ecg.get('hrv_status') == 'reduced':
                   recommendations.append('• Reduced HRV - consider autonomic assessment')
           
           if 'respiratory' in vital_signs:
               resp = vital_signs['respiratory']
               if resp.get('respiratory_status') != 'normal':
                   recommendations.append('• Abnormal respiratory rate - assess pulmonary function')
           
           if not recommendations:
               recommendations.append('✓ All vital signs within normal ranges')
               recommendations.append('✓ Continue regular monitoring')
           
           return recommendations
       
       def generate_health_report(self, assessment, output_path='health_report.html'):
           """Generate comprehensive health report."""
           
           # Prepare feature data for report generator
           feature_data = {}
           
           if 'ecg' in assessment['vital_signs']:
               ecg = assessment['vital_signs']['ecg']
               if 'hrv' in ecg:
                   for key, value in ecg['hrv'].items():
                       if isinstance(value, (int, float, np.number)):
                           feature_data[key] = [float(value)]
           
           if feature_data:
               # Generate report
               report_gen = HealthReportGenerator(
                   feature_data=feature_data,
                   sampling_rate=self.fs
               )
               
               report_path = report_gen.generate_report(
                   patient_id=self.patient_id,
                   output_path=output_path
               )
               
               return report_path
           
           return None

**Usage Example:**

.. code-block:: python

   # Initialize monitoring system
   monitor = ComprehensiveHealthMonitor(patient_id="P001", fs=256)
   
   # Generate or load vital signs
   ecg_signal = generate_ecg_signal(sfecg=256, duration=60, hrmean=75, Anoise=0.05)
   
   # Process vital signs
   assessment = monitor.process_vital_signs(ecg_signal=ecg_signal)
   
   # Display results
   print(f"Patient ID: {assessment['patient_id']}")
   print(f"Health Score: {assessment['health_score']}/100")
   print(f"Timestamp: {assessment['timestamp']}")
   
   print("\nVital Signs:")
   if 'ecg' in assessment['vital_signs']:
       ecg = assessment['vital_signs']['ecg']
       print(f"  Heart Rate: {ecg.get('heart_rate', 'N/A'):.1f} BPM ({ecg.get('heart_rate_status', 'unknown')})")
       print(f"  HRV Status: {ecg.get('hrv_status', 'unknown')}")
   
   print("\nRecommendations:")
   for rec in assessment['recommendations']:
       print(f"  {rec}")
   
   # Generate HTML report
   report_path = monitor.generate_health_report(assessment)
   if report_path:
       print(f"\n📄 Health report generated: {report_path}")

Example 7: Cross-Signal Synchronization Analysis
==================================================

This example demonstrates advanced analysis of coupling between multiple physiological signals.

**Use Case:** Analyzing cardio-respiratory coupling and autonomic nervous system coordination.

**Key Features:**
* Transfer entropy analysis
* Phase synchronization
* Cross-correlation analysis
* Directional coupling assessment
* Clinical interpretation of coupling metrics

**Implementation:**

.. code-block:: python

   from vitalDSP.physiological_features.transfer_entropy import TransferEntropy
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.physiological_features.waveform import WaveformMorphology
   import numpy as np
   from scipy import signal as sp_signal
   
   def analyze_cardiorespiratory_coupling(ecg_signal, resp_signal, fs):
       """
       Analyze coupling between cardiac and respiratory signals.
       
       Parameters:
       -----------
       ecg_signal : array-like
           ECG signal data
       resp_signal : array-like
           Respiratory signal data
       fs : float
           Sampling frequency (Hz)
       
       Returns:
       --------
       dict : Coupling analysis results
       """
       
       results = {
           'sampling_frequency': fs,
           'analysis_type': 'cardio-respiratory_coupling'
       }
       
       # 1. Preprocess signals
       sf_ecg = SignalFiltering(ecg_signal)
       filtered_ecg = sf_ecg.bandpass(lowcut=0.5, highcut=40.0, fs=fs, order=4)
       
       sf_resp = SignalFiltering(resp_signal)
       filtered_resp = sf_resp.bandpass(lowcut=0.1, highcut=0.5, fs=fs, order=4)
       
       # 2. Extract RR intervals
       wm = WaveformMorphology(filtered_ecg, fs=fs, signal_type="ECG")
       r_peaks = wm.r_peaks
       
       if len(r_peaks) > 50:
           rr_intervals = np.diff(r_peaks) / fs * 1000
           valid_rr = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
           
           if len(valid_rr) > 50 and len(filtered_resp) > 100:
               # Resample respiratory signal to match RR intervals
               resp_resampled = sp_signal.resample(
                   filtered_resp,
                   len(valid_rr)
               )
               
               # 3. Transfer Entropy Analysis
               te_analyzer = TransferEntropy(
                   source=resp_resampled[:len(valid_rr)],
                   target=valid_rr,
                   k=1,  # History length
                   delay=1  # Time delay
               )
               
               # Respiratory -> Cardiac influence
               te_resp_to_cardiac = te_analyzer.compute_transfer_entropy()
               results['te_resp_to_cardiac'] = float(te_resp_to_cardiac)
               
               # Cardiac -> Respiratory influence
               te_analyzer_reverse = TransferEntropy(
                   source=valid_rr,
                   target=resp_resampled[:len(valid_rr)],
                   k=1,
                   delay=1
               )
               te_cardiac_to_resp = te_analyzer_reverse.compute_transfer_entropy()
               results['te_cardiac_to_resp'] = float(te_cardiac_to_resp)
               
               # Net directionality
               net_directionality = te_resp_to_cardiac - te_cardiac_to_resp
               results['net_directionality'] = float(net_directionality)
               
               # 4. Cross-correlation analysis
               cross_corr = np.correlate(
                   valid_rr - np.mean(valid_rr),
                   resp_resampled[:len(valid_rr)] - np.mean(resp_resampled[:len(valid_rr)]),
                   mode='full'
               )
               cross_corr = cross_corr / (len(valid_rr) * np.std(valid_rr) * np.std(resp_resampled[:len(valid_rr)]))
               
               max_corr_idx = np.argmax(np.abs(cross_corr))
               max_correlation = cross_corr[max_corr_idx]
               lag = max_corr_idx - len(valid_rr) + 1
               
               results['max_correlation'] = float(max_correlation)
               results['optimal_lag'] = int(lag)
               
               # 5. Clinical interpretation
               interpretation = interpret_coupling_metrics(
                   te_resp_to_cardiac,
                   te_cardiac_to_resp,
                   net_directionality,
                   max_correlation
               )
               results['clinical_interpretation'] = interpretation
       
       return results
   
   def interpret_coupling_metrics(te_r_to_c, te_c_to_r, net_dir, max_corr):
       """Provide clinical interpretation of coupling metrics."""
       
       interpretation = {
           'coupling_strength': 'normal',
           'dominant_direction': 'balanced',
           'autonomic_status': 'healthy',
           'recommendations': []
       }
       
       # Assess coupling strength
       avg_te = (te_r_to_c + te_c_to_r) / 2
       if avg_te < 0.05:
           interpretation['coupling_strength'] = 'weak'
           interpretation['autonomic_status'] = 'impaired'
           interpretation['recommendations'].append(
               'Weak cardio-respiratory coupling suggests autonomic dysfunction'
           )
       elif avg_te > 0.2:
           interpretation['coupling_strength'] = 'strong'
           interpretation['autonomic_status'] = 'healthy'
           interpretation['recommendations'].append(
               'Strong coupling indicates good autonomic function'
           )
       
       # Assess directionality
       if abs(net_dir) > 0.1:
           if net_dir > 0:
               interpretation['dominant_direction'] = 'respiratory_to_cardiac'
               interpretation['recommendations'].append(
                   'Respiratory drive dominant - typical of healthy state'
               )
           else:
               interpretation['dominant_direction'] = 'cardiac_to_respiratory'
               interpretation['recommendations'].append(
                   'Cardiac drive dominant - may indicate stress or pathology'
               )
       else:
           interpretation['dominant_direction'] = 'balanced'
           interpretation['recommendations'].append(
               'Balanced bidirectional coupling'
           )
       
       # Assess correlation
       if abs(max_corr) > 0.5:
           interpretation['recommendations'].append(
               f'High correlation ({max_corr:.2f}) indicates strong synchronization'
           )
       elif abs(max_corr) < 0.2:
           interpretation['recommendations'].append(
               f'Low correlation ({max_corr:.2f}) suggests reduced synchronization'
           )
       
       return interpretation

**Usage Example:**

.. code-block:: python

   from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal, generate_resp_signal
   
   # Generate signals
   fs = 256
   duration = 300
   
   ecg_signal = generate_ecg_signal(sfecg=fs, duration=duration, hrmean=72, Anoise=0.05)
   resp_signal = generate_resp_signal(sampling_rate=fs, duration=duration, frequency=0.25, amplitude=0.5)
   
   # Analyze coupling
   results = analyze_cardiorespiratory_coupling(ecg_signal, resp_signal, fs)
   
   # Display results
   print("Cardio-Respiratory Coupling Analysis")
   print("=" * 50)
   print(f"Transfer Entropy (Resp → Cardiac): {results['te_resp_to_cardiac']:.4f}")
   print(f"Transfer Entropy (Cardiac → Resp): {results['te_cardiac_to_resp']:.4f}")
   print(f"Net Directionality: {results['net_directionality']:.4f}")
   print(f"Maximum Correlation: {results['max_correlation']:.3f}")
   print(f"Optimal Lag: {results['optimal_lag']} samples")
   
   print("\nClinical Interpretation:")
   interp = results['clinical_interpretation']
   print(f"Coupling Strength: {interp['coupling_strength']}")
   print(f"Dominant Direction: {interp['dominant_direction']}")
   print(f"Autonomic Status: {interp['autonomic_status']}")
   
   print("\nRecommendations:")
   for rec in interp['recommendations']:
       print(f"  • {rec}")

Example 8: Comprehensive Physiological Feature Extraction
===========================================================

This example demonstrates comprehensive feature extraction across all domains: time, frequency, nonlinear, and energy analysis.

**Use Case:** Complete physiological characterization for research and clinical assessment.

**Key Features:**
* Time-domain feature extraction
* Frequency-domain analysis (PSD, spectral bands)
* Nonlinear dynamics (Poincaré, entropy, fractal dimension)
* Beat-to-beat variability analysis
* Energy distribution analysis
* Automated feature interpretation

**Implementation:**

.. code-block:: python

   import numpy as np
   from vitalDSP.physiological_features.time_domain import TimeDomainFeatures
   from vitalDSP.physiological_features.frequency_domain import FrequencyDomainFeatures
   from vitalDSP.physiological_features.nonlinear import NonlinearFeatures
   from vitalDSP.physiological_features.beat_to_beat import BeatToBeatAnalysis
   from vitalDSP.physiological_features.energy_analysis import EnergyAnalysis
   from vitalDSP.physiological_features.waveform import WaveformMorphology
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal
   import pandas as pd
   
   def extract_comprehensive_features(ecg_signal, fs, patient_id=None):
       """
       Extract comprehensive physiological features from ECG signal.
       
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
       dict : Complete feature set with clinical interpretation
       """
       
       results = {
           'patient_id': patient_id,
           'sampling_frequency': fs,
           'features': {},
           'interpretation': {}
       }
       
       # 1. Preprocess signal
       sf = SignalFiltering(ecg_signal)
       filtered_ecg = sf.bandpass(lowcut=0.5, highcut=40.0, fs=fs, order=4)
       
       # 2. Extract R-peaks and RR intervals
       wm = WaveformMorphology(filtered_ecg, fs=fs, signal_type="ECG")
       r_peaks = wm.r_peaks
       
       if len(r_peaks) < 10:
           results['error'] = 'Insufficient R-peaks detected'
           return results
       
       # Calculate RR intervals in milliseconds
       rr_intervals = np.diff(r_peaks) / fs * 1000
       valid_rr = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
       
       if len(valid_rr) < 5:
           results['error'] = 'Insufficient valid RR intervals'
           return results
       
       # 3. TIME DOMAIN FEATURES
       print("Extracting time-domain features...")
       td_features = TimeDomainFeatures(valid_rr)
       
       # Basic statistics
       results['features']['mean_rr'] = td_features.mean()
       results['features']['std_rr'] = td_features.std()
       results['features']['min_rr'] = td_features.min()
       results['features']['max_rr'] = td_features.max()
       results['features']['range_rr'] = td_features.max() - td_features.min()
       
       # HRV time-domain metrics
       results['features']['sdnn'] = td_features.sdnn()
       results['features']['rmssd'] = td_features.rmssd()
       results['features']['sdsd'] = td_features.sdsd()
       results['features']['nn50'] = td_features.nn50()
       results['features']['pnn50'] = td_features.pnn50()
       results['features']['nn20'] = td_features.nn20()
       results['features']['pnn20'] = td_features.pnn20()
       
       # Heart rate
       results['features']['mean_hr'] = 60000 / results['features']['mean_rr']
       
       # 4. FREQUENCY DOMAIN FEATURES
       print("Extracting frequency-domain features...")
       fd_features = FrequencyDomainFeatures(valid_rr, fs=4.0)  # Typical RR sampling ~4Hz
       
       # Compute PSD
       psd_results = fd_features.compute_psd(method='welch')
       results['features']['vlf_power'] = psd_results['vlf_power']
       results['features']['lf_power'] = psd_results['lf_power']
       results['features']['hf_power'] = psd_results['hf_power']
       results['features']['total_power'] = psd_results['total_power']
       results['features']['lf_hf_ratio'] = psd_results['lf_hf_ratio']
       
       # Normalized powers
       if results['features']['total_power'] > 0:
           results['features']['lf_nu'] = (results['features']['lf_power'] / 
                                           (results['features']['lf_power'] + results['features']['hf_power']) * 100)
           results['features']['hf_nu'] = (results['features']['hf_power'] / 
                                           (results['features']['lf_power'] + results['features']['hf_power']) * 100)
       
       # Peak frequencies
       results['features']['lf_peak'] = psd_results.get('lf_peak_frequency', 0)
       results['features']['hf_peak'] = psd_results.get('hf_peak_frequency', 0)
       
       # 5. NONLINEAR FEATURES
       print("Extracting nonlinear features...")
       nl_features = NonlinearFeatures(valid_rr)
       
       # Poincaré analysis
       poincare = nl_features.compute_poincare_features()
       results['features']['sd1'] = poincare['sd1']
       results['features']['sd2'] = poincare['sd2']
       results['features']['sd_ratio'] = poincare['sd_ratio']
       
       # Entropy measures
       results['features']['sample_entropy'] = nl_features.sample_entropy(m=2, r=0.2)
       results['features']['approximate_entropy'] = nl_features.approximate_entropy(m=2, r=0.2)
       
       # Detrended Fluctuation Analysis
       results['features']['dfa_alpha1'] = nl_features.dfa(scale_min=4, scale_max=16)
       results['features']['dfa_alpha2'] = nl_features.dfa(scale_min=16, scale_max=64)
       
       # 6. BEAT-TO-BEAT ANALYSIS
       print("Extracting beat-to-beat features...")
       bb_analysis = BeatToBeatAnalysis(filtered_ecg, fs=fs, r_peaks=r_peaks)
       
       # Beat-to-beat variability
       bb_features = bb_analysis.compute_variability_features()
       results['features']['beat_variability'] = bb_features.get('variability_index', 0)
       results['features']['successive_difference'] = bb_features.get('successive_difference', 0)
       
       # 7. ENERGY ANALYSIS
       print("Extracting energy features...")
       energy_analysis = EnergyAnalysis(filtered_ecg, fs=fs)
       
       # Signal energy distribution
       energy_features = energy_analysis.compute_energy_features()
       results['features']['total_energy'] = energy_features.get('total_energy', 0)
       results['features']['energy_mean'] = energy_features.get('mean_energy', 0)
       results['features']['energy_std'] = energy_features.get('std_energy', 0)
       
       # Spectral energy
       spectral_energy = energy_analysis.compute_spectral_energy()
       results['features']['spectral_energy'] = spectral_energy.get('total_spectral_energy', 0)
       
       # 8. CLINICAL INTERPRETATION
       results['interpretation'] = interpret_comprehensive_features(results['features'])
       
       return results
   
   def interpret_comprehensive_features(features):
       """Provide comprehensive clinical interpretation."""
       
       interpretation = {
           'hrv_status': 'normal',
           'autonomic_balance': 'balanced',
           'cardiac_health': 'healthy',
           'risk_level': 'low',
           'recommendations': []
       }
       
       # Time-domain assessment
       sdnn = features.get('sdnn', 0)
       rmssd = features.get('rmssd', 0)
       pnn50 = features.get('pnn50', 0)
       
       if sdnn < 50:
           interpretation['hrv_status'] = 'reduced'
           interpretation['cardiac_health'] = 'at_risk'
           interpretation['risk_level'] = 'moderate'
           interpretation['recommendations'].append(
               'Reduced SDNN (<50ms) suggests decreased HRV and increased cardiac risk'
           )
       elif sdnn < 100:
           interpretation['hrv_status'] = 'fair'
           interpretation['recommendations'].append(
               'SDNN within fair range (50-100ms) - regular monitoring recommended'
           )
       else:
           interpretation['hrv_status'] = 'good'
           interpretation['recommendations'].append(
               'Good SDNN (>100ms) indicates healthy HRV'
           )
       
       # Frequency-domain assessment
       lf_hf_ratio = features.get('lf_hf_ratio', 1.0)
       
       if lf_hf_ratio < 0.5:
           interpretation['autonomic_balance'] = 'parasympathetic_dominant'
           interpretation['recommendations'].append(
               'Low LF/HF ratio (<0.5) suggests parasympathetic dominance'
           )
       elif lf_hf_ratio > 2.0:
           interpretation['autonomic_balance'] = 'sympathetic_dominant'
           interpretation['recommendations'].append(
               'High LF/HF ratio (>2.0) suggests sympathetic dominance or stress'
           )
       else:
           interpretation['autonomic_balance'] = 'balanced'
           interpretation['recommendations'].append(
               'LF/HF ratio within normal range indicates balanced autonomic function'
           )
       
       # Nonlinear assessment
       sd1 = features.get('sd1', 0)
       sd2 = features.get('sd2', 0)
       sample_entropy = features.get('sample_entropy', 0)
       
       if sd1 < 10:
           interpretation['recommendations'].append(
               'Low SD1 (<10ms) indicates reduced short-term variability'
           )
       
       if sample_entropy < 1.0:
           interpretation['recommendations'].append(
               'Low sample entropy (<1.0) suggests reduced signal complexity'
           )
       elif sample_entropy > 2.0:
           interpretation['recommendations'].append(
               'High sample entropy (>2.0) indicates good signal complexity'
           )
       
       # Overall risk assessment
       risk_factors = 0
       if sdnn < 50:
           risk_factors += 2
       if rmssd < 20:
           risk_factors += 1
       if lf_hf_ratio > 2.5:
           risk_factors += 1
       if sample_entropy < 1.0:
           risk_factors += 1
       
       if risk_factors >= 3:
           interpretation['risk_level'] = 'high'
           interpretation['cardiac_health'] = 'concerning'
           interpretation['recommendations'].append(
               '⚠️ Multiple risk factors detected - comprehensive cardiac evaluation recommended'
           )
       elif risk_factors >= 1:
           interpretation['risk_level'] = 'moderate'
           interpretation['cardiac_health'] = 'fair'
           interpretation['recommendations'].append(
               '⚠️ Some risk factors present - regular monitoring advised'
           )
       else:
           interpretation['risk_level'] = 'low'
           interpretation['cardiac_health'] = 'healthy'
           interpretation['recommendations'].append(
               '✓ All metrics within healthy ranges'
           )
       
       return interpretation

**Usage Example:**

.. code-block:: python

   # Generate or load ECG data
   fs = 256
   duration = 300  # 5 minutes
   ecg_data = generate_ecg_signal(sfecg=fs, duration=duration, hrmean=75, Anoise=0.05)
   
   # Extract comprehensive features
   results = extract_comprehensive_features(ecg_data, fs, patient_id="P001")
   
   # Display results
   print(f"Patient ID: {results['patient_id']}")
   print(f"\n{'='*60}")
   print("TIME DOMAIN FEATURES")
   print(f"{'='*60}")
   print(f"Mean RR: {results['features']['mean_rr']:.2f} ms")
   print(f"SDNN: {results['features']['sdnn']:.2f} ms")
   print(f"RMSSD: {results['features']['rmssd']:.2f} ms")
   print(f"pNN50: {results['features']['pnn50']:.2f} %")
   print(f"Heart Rate: {results['features']['mean_hr']:.1f} BPM")
   
   print(f"\n{'='*60}")
   print("FREQUENCY DOMAIN FEATURES")
   print(f"{'='*60}")
   print(f"VLF Power: {results['features']['vlf_power']:.2f} ms²")
   print(f"LF Power: {results['features']['lf_power']:.2f} ms²")
   print(f"HF Power: {results['features']['hf_power']:.2f} ms²")
   print(f"LF/HF Ratio: {results['features']['lf_hf_ratio']:.2f}")
   print(f"Total Power: {results['features']['total_power']:.2f} ms²")
   
   print(f"\n{'='*60}")
   print("NONLINEAR FEATURES")
   print(f"{'='*60}")
   print(f"SD1: {results['features']['sd1']:.2f} ms")
   print(f"SD2: {results['features']['sd2']:.2f} ms")
   print(f"SD1/SD2 Ratio: {results['features']['sd_ratio']:.3f}")
   print(f"Sample Entropy: {results['features']['sample_entropy']:.3f}")
   print(f"Approximate Entropy: {results['features']['approximate_entropy']:.3f}")
   
   print(f"\n{'='*60}")
   print("CLINICAL INTERPRETATION")
   print(f"{'='*60}")
   interp = results['interpretation']
   print(f"HRV Status: {interp['hrv_status']}")
   print(f"Autonomic Balance: {interp['autonomic_balance']}")
   print(f"Cardiac Health: {interp['cardiac_health']}")
   print(f"Risk Level: {interp['risk_level']}")
   
   print(f"\nRecommendations:")
   for rec in interp['recommendations']:
       print(f"  {rec}")
   
   # Export to DataFrame for further analysis
   features_df = pd.DataFrame([results['features']])
   features_df.to_csv(f"features_{results['patient_id']}.csv", index=False)
   print(f"\n✅ Features exported to features_{results['patient_id']}.csv")

Example 9: Machine Learning for Physiological Signal Analysis
===============================================================

This example demonstrates machine learning applications including anomaly detection, signal denoising with autoencoders, and pattern classification.

**Use Case:** Automated signal quality assessment, artifact detection, and arrhythmia classification using deep learning.

**Key Features:**
* Autoencoder-based signal denoising
* Anomaly detection for artifact identification
* CNN1D for pattern classification
* Feature extraction for ML
* Model training and evaluation
* Real-time inference pipeline

**Implementation:**

.. code-block:: python

   import numpy as np
   from vitalDSP.ml_models.deep_models import (
       StandardAutoencoder, ConvolutionalAutoencoder, CNN1D
   )
   from vitalDSP.advanced_computation.anomaly_detection import AnomalyDetection
   from vitalDSP.ml_models.feature_extractor import FeatureExtractor
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   
   def ml_signal_analysis_pipeline(clean_signals, noisy_signals, fs, anomaly_threshold=2.5):
       """
       Complete ML pipeline for physiological signal analysis.
       
       Parameters:
       -----------
       clean_signals : list of arrays
           Clean reference signals for training
       noisy_signals : list of arrays
           Noisy signals to denoise
       fs : float
           Sampling frequency
       anomaly_threshold : float
           Z-score threshold for anomaly detection
       
       Returns:
       --------
       dict : ML analysis results including denoised signals and detected anomalies
       """
       
       results = {
           'denoising': {},
           'anomaly_detection': {},
           'classification': {},
           'model_performance': {}
       }
       
       # 1. SIGNAL DENOISING WITH AUTOENCODER
       print("="*60)
       print("1. AUTOENCODER-BASED SIGNAL DENOISING")
       print("="*60)
       
       # Prepare data for autoencoder
       signal_length = min([len(s) for s in clean_signals])
       X_clean = np.array([s[:signal_length] for s in clean_signals])
       X_noisy = np.array([s[:signal_length] for s in noisy_signals])
       
       # Normalize
       scaler = StandardScaler()
       X_clean_norm = scaler.fit_transform(X_clean)
       X_noisy_norm = scaler.transform(X_noisy)
       
       # Reshape for autoencoder (samples, timesteps, features)
       X_clean_reshaped = X_clean_norm.reshape(X_clean_norm.shape[0], X_clean_norm.shape[1], 1)
       X_noisy_reshaped = X_noisy_norm.reshape(X_noisy_norm.shape[0], X_noisy_norm.shape[1], 1)
       
       # Train Convolutional Autoencoder
       print("Training Convolutional Autoencoder...")
       autoencoder = ConvolutionalAutoencoder(
           input_shape=(signal_length, 1),
           latent_dim=32
       )
       
       history = autoencoder.fit(
           X_noisy_reshaped,
           X_clean_reshaped,
           epochs=50,
           batch_size=32,
           validation_split=0.2,
           verbose=0
       )
       
       # Denoise signals
       X_denoised = autoencoder.predict(X_noisy_reshaped)
       X_denoised = scaler.inverse_transform(X_denoised.reshape(X_denoised.shape[0], -1))
       
       # Calculate reconstruction error
       reconstruction_error = np.mean((X_clean - X_denoised) ** 2, axis=1)
       
       results['denoising']['clean_signals'] = X_clean
       results['denoising']['noisy_signals'] = X_noisy
       results['denoising']['denoised_signals'] = X_denoised
       results['denoising']['reconstruction_error'] = reconstruction_error
       results['denoising']['mean_error'] = np.mean(reconstruction_error)
       results['denoising']['std_error'] = np.std(reconstruction_error)
       
       print(f"✓ Denoising complete")
       print(f"  Mean Reconstruction Error: {results['denoising']['mean_error']:.4f}")
       print(f"  Std Reconstruction Error: {results['denoising']['std_error']:.4f}")
       
       # 2. ANOMALY DETECTION
       print(f"\n{'='*60}")
       print("2. ANOMALY DETECTION")
       print("="*60)
       
       anomaly_results = []
       for i, signal in enumerate(noisy_signals):
           detector = AnomalyDetection(signal)
           
           # Z-score based detection
           anomalies_zscore = detector.detect_anomalies(
               method='z_score',
               threshold=anomaly_threshold
           )
           
           # Statistical detection
           anomalies_stat = detector.detect_anomalies(
               method='statistical',
               window_size=100
           )
           
           # Isolation Forest
           try:
               anomalies_isolation = detector.detect_anomalies(
                   method='isolation_forest',
                   contamination=0.1
               )
           except:
               anomalies_isolation = []
           
           anomaly_info = {
               'signal_index': i,
               'z_score_anomalies': len(anomalies_zscore),
               'statistical_anomalies': len(anomalies_stat),
               'isolation_forest_anomalies': len(anomalies_isolation),
               'z_score_locations': anomalies_zscore,
               'signal_quality': 'good' if len(anomalies_zscore) < 10 else 'poor'
           }
           anomaly_results.append(anomaly_info)
       
       results['anomaly_detection']['per_signal'] = anomaly_results
       results['anomaly_detection']['total_signals'] = len(noisy_signals)
       results['anomaly_detection']['good_quality'] = sum(1 for a in anomaly_results if a['signal_quality'] == 'good')
       results['anomaly_detection']['poor_quality'] = sum(1 for a in anomaly_results if a['signal_quality'] == 'poor')
       
       print(f"✓ Anomaly detection complete")
       print(f"  Total signals analyzed: {results['anomaly_detection']['total_signals']}")
       print(f"  Good quality signals: {results['anomaly_detection']['good_quality']}")
       print(f"  Poor quality signals: {results['anomaly_detection']['poor_quality']}")
       
       # 3. FEATURE EXTRACTION FOR ML
       print(f"\n{'='*60}")
       print("3. FEATURE EXTRACTION FOR MACHINE LEARNING")
       print("="*60)
       
       extractor = FeatureExtractor()
       
       features_list = []
       for signal in X_denoised:
           # Extract comprehensive features
           features = extractor.extract_features(
               signal,
               fs=fs,
               feature_types=['statistical', 'temporal', 'spectral']
           )
           features_list.append(features)
       
       features_array = np.array(features_list)
       
       results['classification']['features'] = features_array
       results['classification']['n_features'] = features_array.shape[1]
       
       print(f"✓ Feature extraction complete")
       print(f"  Number of features extracted: {features_array.shape[1]}")
       print(f"  Feature dimensions: {features_array.shape}")
       
       # 4. CNN1D CLASSIFICATION (Example with synthetic labels)
       print(f"\n{'='*60}")
       print("4. CNN1D PATTERN CLASSIFICATION")
       print("="*60)
       
       # Create synthetic labels for demonstration (normal vs. abnormal)
       labels = np.array([0 if e < np.median(reconstruction_error) else 1 
                         for e in reconstruction_error])
       
       # Split data
       X_train, X_test, y_train, y_test = train_test_split(
           X_clean_reshaped, labels, test_size=0.2, random_state=42, stratify=labels
       )
       
       # Train CNN1D classifier
       print("Training CNN1D classifier...")
       cnn = CNN1D(
           input_shape=(signal_length, 1),
           num_classes=2,
           num_filters=[32, 64, 128],
           kernel_size=3,
           pool_size=2
       )
       
       cnn_history = cnn.fit(
           X_train, y_train,
           epochs=30,
           batch_size=32,
           validation_split=0.2,
           verbose=0
       )
       
       # Evaluate
       test_loss, test_accuracy = cnn.evaluate(X_test, y_test)
       
       results['classification']['test_accuracy'] = test_accuracy
       results['classification']['test_loss'] = test_loss
       results['classification']['n_train'] = len(X_train)
       results['classification']['n_test'] = len(X_test)
       
       print(f"✓ Classification complete")
       print(f"  Test Accuracy: {test_accuracy:.2%}")
       print(f"  Test Loss: {test_loss:.4f}")
       print(f"  Training samples: {len(X_train)}")
       print(f"  Test samples: {len(X_test)}")
       
       return results
   
   def evaluate_ml_pipeline(results):
       """Evaluate and display ML pipeline results."""
       
       print(f"\n{'='*60}")
       print("ML PIPELINE PERFORMANCE SUMMARY")
       print("="*60)
       
       # Denoising performance
       print(f"\n📊 Denoising Performance:")
       print(f"  Mean Reconstruction Error: {results['denoising']['mean_error']:.4f}")
       print(f"  Std Reconstruction Error: {results['denoising']['std_error']:.4f}")
       
       # Calculate SNR improvement
       original_noise = np.mean(np.var(results['denoising']['noisy_signals'] - 
                                       results['denoising']['clean_signals'], axis=1))
       residual_noise = np.mean(np.var(results['denoising']['denoised_signals'] - 
                                       results['denoising']['clean_signals'], axis=1))
       snr_improvement = 10 * np.log10(original_noise / (residual_noise + 1e-10))
       
       print(f"  SNR Improvement: {snr_improvement:.2f} dB")
       
       # Anomaly detection
       print(f"\n🔍 Anomaly Detection:")
       print(f"  Total signals: {results['anomaly_detection']['total_signals']}")
       print(f"  Good quality: {results['anomaly_detection']['good_quality']} "
             f"({results['anomaly_detection']['good_quality']/results['anomaly_detection']['total_signals']*100:.1f}%)")
       print(f"  Poor quality: {results['anomaly_detection']['poor_quality']} "
             f"({results['anomaly_detection']['poor_quality']/results['anomaly_detection']['total_signals']*100:.1f}%)")
       
       # Classification
       print(f"\n🎯 Classification Performance:")
       print(f"  Test Accuracy: {results['classification']['test_accuracy']:.2%}")
       print(f"  Number of features: {results['classification']['n_features']}")
       
       # Overall assessment
       print(f"\n{'='*60}")
       print("OVERALL ASSESSMENT")
       print("="*60)
       
       quality_score = (
           (snr_improvement / 10) * 30 +  # Denoising (30 points)
           (results['anomaly_detection']['good_quality'] / 
            results['anomaly_detection']['total_signals']) * 30 +  # Quality (30 points)
           results['classification']['test_accuracy'] * 40  # Classification (40 points)
       )
       
       print(f"Quality Score: {quality_score:.1f}/100")
       
       if quality_score >= 80:
           print("✅ Excellent - ML pipeline performing optimally")
       elif quality_score >= 60:
           print("⚠️ Good - ML pipeline acceptable, some improvements possible")
       else:
           print("❌ Fair - ML pipeline needs optimization")

**Usage Example:**

.. code-block:: python

   # Generate training data
   n_signals = 100
   fs = 256
   signal_length = 2560  # 10 seconds
   
   print("Generating training data...")
   clean_signals = []
   noisy_signals = []
   
   for i in range(n_signals):
       # Generate clean ECG
       clean = generate_ecg_signal(sfecg=fs, duration=10, hrmean=70+np.random.randn()*5, Anoise=0.0)
       clean_signals.append(clean[:signal_length])
       
       # Add noise
       noise = np.random.normal(0, 0.1, signal_length)
       noisy = clean[:signal_length] + noise
       noisy_signals.append(noisy)
   
   # Run ML pipeline
   results = ml_signal_analysis_pipeline(clean_signals, noisy_signals, fs=fs)
   
   # Evaluate results
   evaluate_ml_pipeline(results)
   
   # Save models for deployment
   print("\n💾 Models ready for deployment")
   print("  - Autoencoder trained for real-time denoising")
   print("  - Anomaly detector calibrated for quality assessment")
   print("  - CNN1D classifier ready for pattern recognition")

These examples provide a solid foundation for implementing VitalDSP in various real-world scenarios. Adapt them to your specific needs and requirements.

For more advanced examples and use cases, explore the Jupyter notebooks in the :ref:`sample_notebooks` section.
