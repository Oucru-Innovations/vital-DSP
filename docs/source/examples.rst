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
   * Multi-signal synchronization

**Advanced Analysis**
   * Multi-scale entropy for cardiac complexity
   * Comprehensive feature extraction (time, frequency, nonlinear)
   * Machine learning pipeline for signal classification

Example 1: ECG Analysis for Clinical Research
==============================================

Comprehensive ECG analysis including bandpass filtering, signal quality assessment, waveform morphology, and HRV feature extraction.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.physiological_features.waveform import WaveformMorphology
   from vitalDSP.physiological_features.hrv_analysis import HRVFeatures
   from vitalDSP.signal_quality_assessment.signal_quality_index import SignalQualityIndex
   from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal

   def analyze_ecg_clinical_research(ecg_signal, fs, patient_id=None):
       """
       Comprehensive ECG analysis for clinical research.

       Parameters
       ----------
       ecg_signal : numpy.ndarray
           Raw ECG signal.
       fs : float
           Sampling frequency in Hz.
       patient_id : str, optional
           Patient identifier for logging.

       Returns
       -------
       dict : Analysis results including quality metrics, heart rate, and HRV.
       """
       results = {
           'patient_id': patient_id,
           'sampling_frequency': fs,
           'signal_length': len(ecg_signal),
       }

       # 1. Bandpass filter (0.5–40 Hz for ECG)
       sf = SignalFiltering(ecg_signal)
       filtered_ecg = sf.bandpass(lowcut=0.5, highcut=40.0, fs=fs, order=4)

       # 2. Signal Quality Assessment
       sqi = SignalQualityIndex(filtered_ecg)
       results['snr_sqi']      = sqi.snr_sqi(window_size=int(fs * 5), step_size=int(fs), aggregate=True)
       results['baseline_sqi'] = sqi.baseline_wander_sqi(window_size=int(fs * 5), step_size=int(fs), aggregate=True)

       # 3. Waveform Morphology — detects R-peaks, computes heart rate
       wm = WaveformMorphology(filtered_ecg, fs=fs, signal_type='ECG')
       heart_rate = wm.get_heart_rate(summary_type='mean')
       results['heart_rate'] = heart_rate
       results['heart_rate_status'] = (
           'bradycardia' if heart_rate < 60 else
           'tachycardia' if heart_rate > 100 else 'normal'
       )

       # 4. HRV analysis from RR intervals
       #    BeatToBeatAnalysis gives us cleaned RR intervals
       from vitalDSP.physiological_features.beat_to_beat import BeatToBeatAnalysis
       bb = BeatToBeatAnalysis(filtered_ecg, fs=fs, signal_type='ECG')
       rr_intervals = bb.compute_rr_intervals()       # ms, outlier-corrected

       if len(rr_intervals) > 10:
           hrv = HRVFeatures(signals=filtered_ecg, nn_intervals=rr_intervals, fs=fs)
           hrv_features = hrv.compute_all_features()
           results['hrv'] = hrv_features

           sdnn = hrv_features.get('sdnn', 0)
           results['hrv_status'] = (
               'reduced' if sdnn < 30 else
               'good'    if sdnn > 50 else 'normal'
           )

       return results

   # --- Usage ---
   fs = 256
   ecg_data = generate_ecg_signal(sfecg=fs, duration=60, hrmean=72, Anoise=0.05)
   results = analyze_ecg_clinical_research(ecg_data, fs, patient_id='P001')

   print(f"Patient   : {results['patient_id']}")
   print(f"Heart rate: {results['heart_rate']:.1f} BPM ({results['heart_rate_status']})")
   print(f"SNR SQI   : {results['snr_sqi']:.3f}")
   if 'hrv_status' in results:
       print(f"HRV status: {results['hrv_status']}")

Example 2: PPG Analysis for Hemodynamic Studies
=================================================

Analyzes PPG signals for pulse rate, pulse amplitude, and signal quality — using real data from the vitalDSP sample loader.

.. code-block:: python

   import numpy as np
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.physiological_features.waveform import WaveformMorphology
   from vitalDSP.physiological_features.beat_to_beat import BeatToBeatAnalysis
   from vitalDSP.signal_quality_assessment.signal_quality_index import SignalQualityIndex
   from vitalDSP.notebooks import load_sample_ppg

   def analyze_ppg_hemodynamic(ppg_signal, fs, patient_id=None):
       """
       PPG analysis for hemodynamic characterization.

       Parameters
       ----------
       ppg_signal : numpy.ndarray
           Raw PPG signal.
       fs : float
           Sampling frequency in Hz.

       Returns
       -------
       dict : Pulse rate, variability, and signal quality metrics.
       """
       results = {'patient_id': patient_id, 'sampling_frequency': fs}

       # 1. Bandpass filter (0.5–8 Hz for PPG)
       sf = SignalFiltering(ppg_signal)
       filtered_ppg = sf.bandpass(lowcut=0.5, highcut=8.0, fs=fs, order=4)

       # 2. Signal quality
       sqi = SignalQualityIndex(filtered_ppg)
       results['snr_sqi'] = sqi.snr_sqi(
           window_size=int(fs * 5), step_size=int(fs), aggregate=True
       )

       # 3. Pulse rate via WaveformMorphology
       wm = WaveformMorphology(filtered_ppg, fs=fs, signal_type='PPG')
       pulse_rate = wm.get_heart_rate(summary_type='mean')
       results['pulse_rate'] = pulse_rate
       results['pulse_rate_status'] = (
           'bradycardia' if pulse_rate < 60 else
           'tachycardia' if pulse_rate > 100 else 'normal'
       )

       # 4. Beat-to-beat pulse interval variability
       bb = BeatToBeatAnalysis(filtered_ppg, fs=fs, signal_type='PPG')
       pi_intervals = bb.compute_rr_intervals()

       if len(pi_intervals) > 5:
           results['mean_pulse_interval'] = float(np.mean(pi_intervals))
           results['std_pulse_interval']  = float(np.std(pi_intervals))
           results['pulse_rate_variability'] = float(
               np.std(pi_intervals) / np.mean(pi_intervals) * 100
           )

       # 5. Pulse amplitude variability
       results['amplitude_sqi'] = sqi.amplitude_variability_sqi(
           window_size=int(fs * 5), step_size=int(fs), aggregate=True
       )

       return results

   # --- Usage ---
   ppg_col, _ = load_sample_ppg()
   ppg_col = np.array(ppg_col)
   fs = 128

   results = analyze_ppg_hemodynamic(ppg_col, fs=fs, patient_id='P001')
   print(f"Pulse rate   : {results['pulse_rate']:.1f} BPM ({results['pulse_rate_status']})")
   print(f"SNR SQI      : {results['snr_sqi']:.3f}")
   print(f"Amplitude SQI: {results['amplitude_sqi']:.3f}")
   if 'pulse_rate_variability' in results:
       print(f"PRV (CV)     : {results['pulse_rate_variability']:.2f} %")

Example 3: Real-Time Vital Signs Monitoring
=============================================

A sliding-window monitoring system that continuously processes incoming signal chunks and logs vital signs.

.. code-block:: python

   import time
   import threading
   import numpy as np
   from collections import deque
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.physiological_features.waveform import WaveformMorphology
   from vitalDSP.signal_quality_assessment.signal_quality_index import SignalQualityIndex

   class RealTimeVitalSignsMonitor:
       """Sliding-window real-time vital signs monitor."""

       def __init__(self, fs=128, window_size=10, update_interval=1):
           self.fs = fs
           self.window_samples = fs * window_size
           self.update_interval = update_interval
           self.signal_buffer = deque(maxlen=self.window_samples)
           self.history = deque(maxlen=100)
           self.current = {}
           self.is_running = False
           self.thresholds = {'hr_low': 50, 'hr_high': 120, 'sqi_low': 0.3}

       def add_data(self, chunk):
           """Push a new signal chunk into the sliding buffer."""
           self.signal_buffer.extend(chunk)

       def start(self):
           self.is_running = True
           t = threading.Thread(target=self._loop, daemon=True)
           t.start()
           print("Monitoring started.")

       def stop(self):
           self.is_running = False
           print("Monitoring stopped.")

       def _loop(self):
           while self.is_running:
               if len(self.signal_buffer) >= self.window_samples:
                   window = np.array(list(self.signal_buffer))
                   vitals = self._process(window)
                   self.current = vitals
                   self.history.append(vitals)
                   self._check_alerts(vitals)
                   self._log(vitals)
               time.sleep(self.update_interval)

       def _process(self, signal):
           vitals = {}
           try:
               sf = SignalFiltering(signal)
               filtered = sf.bandpass(lowcut=0.5, highcut=40.0, fs=self.fs, order=4)

               sqi = SignalQualityIndex(filtered)
               vitals['sqi'] = sqi.snr_sqi(
                   window_size=int(self.fs * 5), step_size=int(self.fs), aggregate=True
               )

               wm = WaveformMorphology(filtered, fs=self.fs, signal_type='ECG')
               vitals['heart_rate'] = wm.get_heart_rate(summary_type='mean')
           except Exception as e:
               vitals['error'] = str(e)
           return vitals

       def _check_alerts(self, vitals):
           hr = vitals.get('heart_rate', None)
           if hr is not None:
               if hr < self.thresholds['hr_low']:
                   print(f"[ALERT] Low HR: {hr:.1f} BPM")
               elif hr > self.thresholds['hr_high']:
                   print(f"[ALERT] High HR: {hr:.1f} BPM")
           if vitals.get('sqi', 1.0) < self.thresholds['sqi_low']:
               print(f"[ALERT] Poor signal quality: SQI={vitals['sqi']:.2f}")

       def _log(self, vitals):
           hr  = vitals.get('heart_rate', float('nan'))
           sqi = vitals.get('sqi', float('nan'))
           print(f"HR={hr:.1f} BPM  SQI={sqi:.3f}")

       def get_stats(self):
           hrs = [v['heart_rate'] for v in self.history if 'heart_rate' in v]
           if not hrs:
               return {}
           return {'mean_hr': np.mean(hrs), 'std_hr': np.std(hrs),
                   'min_hr': np.min(hrs), 'max_hr': np.max(hrs)}

   # --- Usage ---
   from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal

   fs = 128
   monitor = RealTimeVitalSignsMonitor(fs=fs, window_size=10, update_interval=1)
   monitor.start()

   ecg_stream = generate_ecg_signal(sfecg=fs, duration=15, hrmean=72, Anoise=0.05)
   chunk_size = fs  # 1 second per push

   for start in range(0, len(ecg_stream), chunk_size):
       monitor.add_data(ecg_stream[start:start + chunk_size])
       time.sleep(0.1)  # simulate real-time

   monitor.stop()
   stats = monitor.get_stats()
   print(f"Session mean HR: {stats.get('mean_hr', float('nan')):.1f} BPM")

Example 4: Wearable Device Integration
========================================

Processes data from wearable devices (fitness tracker, smartwatch, chest strap) with device-specific filter settings.

.. code-block:: python

   import numpy as np
   from datetime import datetime
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.physiological_features.waveform import WaveformMorphology
   from vitalDSP.signal_quality_assessment.signal_quality_index import SignalQualityIndex

   DEVICE_CONFIGS = {
       'fitness_tracker': {'fs': 100, 'signal': 'PPG', 'lowcut': 0.5, 'highcut': 8.0},
       'smartwatch':      {'fs': 200, 'signal': 'PPG', 'lowcut': 0.5, 'highcut': 8.0},
       'chest_strap':     {'fs': 256, 'signal': 'ECG', 'lowcut': 0.5, 'highcut': 40.0},
   }

   def process_wearable_data(raw_signal, device_type='fitness_tracker'):
       """
       Process a wearable signal chunk and return vital signs.

       Parameters
       ----------
       raw_signal : numpy.ndarray
           Raw signal from the wearable sensor.
       device_type : str
           One of 'fitness_tracker', 'smartwatch', 'chest_strap'.

       Returns
       -------
       dict : Heart rate, signal quality, device metadata, timestamp.
       """
       cfg = DEVICE_CONFIGS.get(device_type, DEVICE_CONFIGS['fitness_tracker'])
       fs  = cfg['fs']

       sf = SignalFiltering(raw_signal)
       filtered = sf.bandpass(lowcut=cfg['lowcut'], highcut=cfg['highcut'], fs=fs, order=2)

       result = {
           'device_type': device_type,
           'signal_type': cfg['signal'],
           'timestamp': datetime.now().isoformat(),
       }

       # Signal quality
       sqi = SignalQualityIndex(filtered)
       result['sqi'] = sqi.snr_sqi(
           window_size=min(int(fs * 5), len(filtered) - 1),
           step_size=int(fs),
           aggregate=True
       )

       # Heart / pulse rate
       wm = WaveformMorphology(filtered, fs=fs, signal_type=cfg['signal'])
       result['heart_rate'] = wm.get_heart_rate(summary_type='mean')

       return result

   # --- Usage ---
   np.random.seed(42)

   for device in ['fitness_tracker', 'smartwatch', 'chest_strap']:
       cfg = DEVICE_CONFIGS[device]
       t = np.arange(2000) / cfg['fs']
       sim_signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))

       r = process_wearable_data(sim_signal, device_type=device)
       print(f"{device}: HR={r['heart_rate']:.1f} BPM  SQI={r['sqi']:.3f}")

Example 5: Advanced Multi-Scale Entropy Analysis
==================================================

Quantifies cardiac signal complexity using Multi-Scale Entropy (MSE), Composite MSE, and Refined Composite MSE.

.. code-block:: python

   import numpy as np
   from vitalDSP.physiological_features.advanced_entropy import MultiScaleEntropy
   from vitalDSP.physiological_features.beat_to_beat import BeatToBeatAnalysis
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal

   def analyze_cardiac_complexity(ecg_signal, fs, patient_id=None):
       """
       Multi-scale entropy analysis of cardiac dynamics.

       Parameters
       ----------
       ecg_signal : numpy.ndarray
           ECG signal.
       fs : float
           Sampling frequency in Hz.

       Returns
       -------
       dict : MSE, CMSE, RCMSE values and complexity indices.
       """
       results = {'patient_id': patient_id, 'fs': fs}

       # Preprocess
       sf = SignalFiltering(ecg_signal)
       filtered = sf.bandpass(lowcut=0.5, highcut=40.0, fs=fs, order=4)

       # Extract RR intervals
       bb = BeatToBeatAnalysis(filtered, fs=fs, signal_type='ECG')
       rr = bb.compute_rr_intervals()

       if len(rr) < 50:
           results['error'] = f'Only {len(rr)} RR intervals — need ≥50 for MSE.'
           return results

       # Multi-Scale Entropy
       mse_analyzer = MultiScaleEntropy(signal=rr, max_scale=20, m=2, r=0.15)

       mse_values   = mse_analyzer.compute_mse()
       cmse_values  = mse_analyzer.compute_cmse()
       rcmse_values = mse_analyzer.compute_rcmse()

       results['mse']               = mse_values.tolist()
       results['cmse']              = cmse_values.tolist()
       results['rcmse']             = rcmse_values.tolist()
       results['complexity_index']  = float(mse_analyzer.get_complexity_index(mse_values))
       results['cmse_ci']           = float(mse_analyzer.get_complexity_index(cmse_values))
       results['rcmse_ci']          = float(mse_analyzer.get_complexity_index(rcmse_values))

       # Clinical interpretation
       cmse_ci = results['cmse_ci']
       results['interpretation'] = (
           'reduced complexity — possible autonomic dysfunction' if cmse_ci < 30 else
           'excellent complexity — healthy autonomic function'   if cmse_ci > 60 else
           'normal complexity'
       )

       return results

   # --- Usage ---
   fs = 256
   ecg_data = generate_ecg_signal(sfecg=fs, duration=120, hrmean=72, Anoise=0.03)
   results = analyze_cardiac_complexity(ecg_data, fs=fs, patient_id='P001')

   if 'error' not in results:
       print(f"Patient          : {results['patient_id']}")
       print(f"MSE CI           : {results['complexity_index']:.2f}")
       print(f"CMSE CI          : {results['cmse_ci']:.2f}")
       print(f"RCMSE CI         : {results['rcmse_ci']:.2f}")
       print(f"Interpretation   : {results['interpretation']}")
   else:
       print(results['error'])

Example 6: Comprehensive Health Monitoring System
===================================================

Multi-modal vital signs assessment (ECG + PPG + respiratory) with automated health scoring and clinical recommendations.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.physiological_features.waveform import WaveformMorphology
   from vitalDSP.physiological_features.hrv_analysis import HRVFeatures
   from vitalDSP.physiological_features.beat_to_beat import BeatToBeatAnalysis
   from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis
   from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal

   class ComprehensiveHealthMonitor:
       """Multi-modal health monitoring with automated assessment."""

       def __init__(self, patient_id, fs=256):
           self.patient_id = patient_id
           self.fs = fs

       def process_vital_signs(self, ecg_signal, ppg_signal=None, resp_signal=None):
           assessment = {
               'patient_id': self.patient_id,
               'timestamp': pd.Timestamp.now().isoformat(),
               'vital_signs': {},
           }

           if ecg_signal is not None:
               assessment['vital_signs']['ecg'] = self._process_ecg(ecg_signal)
           if ppg_signal is not None:
               assessment['vital_signs']['ppg'] = self._process_ppg(ppg_signal)
           if resp_signal is not None:
               assessment['vital_signs']['respiratory'] = self._process_resp(resp_signal)

           assessment['health_score']    = self._health_score(assessment['vital_signs'])
           assessment['recommendations'] = self._recommendations(
               assessment['vital_signs'], assessment['health_score']
           )
           return assessment

       def _process_ecg(self, signal):
           sf = SignalFiltering(signal)
           filtered = sf.bandpass(lowcut=0.5, highcut=40.0, fs=self.fs, order=4)

           wm = WaveformMorphology(filtered, fs=self.fs, signal_type='ECG')
           hr = wm.get_heart_rate(summary_type='mean')

           bb  = BeatToBeatAnalysis(filtered, fs=self.fs, signal_type='ECG')
           rr  = bb.compute_rr_intervals()
           res = {'heart_rate': float(hr),
                  'heart_rate_status': self._hr_status(hr)}

           if len(rr) > 10:
               hrv = HRVFeatures(signals=filtered, nn_intervals=rr, fs=self.fs)
               hrv_feats = hrv.compute_all_features()
               res['hrv'] = hrv_feats
               sdnn = hrv_feats.get('sdnn', 0)
               res['hrv_status'] = ('reduced' if sdnn < 30 else
                                    'good'    if sdnn > 50 else 'normal')
           return res

       def _process_ppg(self, signal):
           sf = SignalFiltering(signal)
           filtered = sf.bandpass(lowcut=0.5, highcut=8.0, fs=self.fs, order=4)

           wm = WaveformMorphology(filtered, fs=self.fs, signal_type='PPG')
           pr = wm.get_heart_rate(summary_type='mean')

           bb = BeatToBeatAnalysis(filtered, fs=self.fs, signal_type='PPG')
           pi = bb.compute_rr_intervals()
           res = {'pulse_rate': float(pr), 'pulse_rate_status': self._hr_status(pr)}
           if len(pi) > 3:
               res['prv'] = float(np.std(pi))
           return res

       def _process_resp(self, signal):
           try:
               ra = RespiratoryAnalysis(signal, fs=self.fs)
               rr = ra.compute_respiratory_rate(method='fft')
               return {
                   'respiratory_rate': float(rr),
                   'status': ('bradypnea' if rr < 12 else
                              'tachypnea' if rr > 20 else 'normal'),
               }
           except Exception as e:
               return {'error': str(e)}

       def _hr_status(self, hr):
           return 'bradycardia' if hr < 60 else 'tachycardia' if hr > 100 else 'normal'

       def _health_score(self, vitals):
           score = 100
           ecg = vitals.get('ecg', {})
           if ecg.get('heart_rate_status') != 'normal':
               score -= 15
           if ecg.get('hrv_status') == 'reduced':
               score -= 20
           if vitals.get('ppg', {}).get('pulse_rate_status') != 'normal':
               score -= 10
           if vitals.get('respiratory', {}).get('status') not in ('normal', None):
               score -= 15
           return max(0, min(100, score))

       def _recommendations(self, vitals, score):
           recs = []
           if score < 60:
               recs.append('URGENT: immediate clinical evaluation recommended')
           elif score < 75:
               recs.append('Clinical review recommended within 24 hours')
           ecg = vitals.get('ecg', {})
           if ecg.get('hrv_status') == 'reduced':
               recs.append('Reduced HRV — consider autonomic assessment')
           if ecg.get('heart_rate_status') == 'bradycardia':
               recs.append('Bradycardia detected — evaluate underlying cause')
           if not recs:
               recs.append('All vital signs within normal ranges')
           return recs

   # --- Usage ---
   fs = 256
   ecg = generate_ecg_signal(sfecg=fs, duration=60, hrmean=75, Anoise=0.05)

   monitor = ComprehensiveHealthMonitor(patient_id='P001', fs=fs)
   assessment = monitor.process_vital_signs(ecg_signal=ecg)

   print(f"Patient      : {assessment['patient_id']}")
   print(f"Health Score : {assessment['health_score']}/100")
   ecg_vs = assessment['vital_signs']['ecg']
   print(f"Heart Rate   : {ecg_vs['heart_rate']:.1f} BPM ({ecg_vs['heart_rate_status']})")
   if 'hrv_status' in ecg_vs:
       print(f"HRV Status   : {ecg_vs['hrv_status']}")
   print("Recommendations:")
   for r in assessment['recommendations']:
       print(f"  • {r}")

Example 7: Cross-Signal Synchronization Analysis
==================================================

Analyzes directional coupling between cardiac and respiratory signals using Transfer Entropy and cross-correlation.

.. code-block:: python

   import numpy as np
   from scipy import signal as sp_signal
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.physiological_features.beat_to_beat import BeatToBeatAnalysis
   from vitalDSP.physiological_features.transfer_entropy import TransferEntropy
   from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal

   def analyze_cardiorespiratory_coupling(ecg_signal, resp_signal, fs):
       """
       Directional coupling between cardiac and respiratory signals.

       Parameters
       ----------
       ecg_signal : numpy.ndarray
           ECG signal.
       resp_signal : numpy.ndarray
           Respiratory signal (same length, same fs).
       fs : float
           Sampling frequency.

       Returns
       -------
       dict : Transfer entropy, cross-correlation, and clinical interpretation.
       """
       results = {'fs': fs}

       # Filter ECG
       sf_ecg = SignalFiltering(ecg_signal)
       filtered_ecg = sf_ecg.bandpass(lowcut=0.5, highcut=40.0, fs=fs, order=4)

       # Filter respiratory signal to the respiratory band
       sf_resp = SignalFiltering(resp_signal)
       filtered_resp = sf_resp.bandpass(lowcut=0.1, highcut=0.5, fs=fs, order=4)

       # Extract RR intervals
       bb = BeatToBeatAnalysis(filtered_ecg, fs=fs, signal_type='ECG')
       rr = bb.compute_rr_intervals()

       if len(rr) < 50:
           return {'error': f'Need ≥50 RR intervals, got {len(rr)}'}

       # Resample respiratory signal to length of RR series
       resp_resampled = sp_signal.resample(filtered_resp, len(rr))

       # Transfer Entropy: respiratory → cardiac
       te_r2c = TransferEntropy(
           source=resp_resampled, target=rr,
           k_coef=1, l_coef=1, delay=1
       ).compute_transfer_entropy()

       # Transfer Entropy: cardiac → respiratory
       te_c2r = TransferEntropy(
           source=rr, target=resp_resampled,
           k_coef=1, l_coef=1, delay=1
       ).compute_transfer_entropy()

       results['te_resp_to_cardiac'] = float(te_r2c)
       results['te_cardiac_to_resp'] = float(te_c2r)
       results['net_directionality']  = float(te_r2c - te_c2r)

       # Normalized cross-correlation
       x = rr - np.mean(rr)
       y = resp_resampled - np.mean(resp_resampled)
       denom = len(rr) * (np.std(x) * np.std(y) + 1e-9)
       xcorr = np.correlate(x, y, mode='full') / denom
       idx = np.argmax(np.abs(xcorr))
       results['max_correlation'] = float(xcorr[idx])
       results['optimal_lag']     = int(idx - len(rr) + 1)

       # Interpretation
       avg_te = (te_r2c + te_c2r) / 2
       results['coupling_strength'] = ('weak' if avg_te < 0.05 else
                                       'strong' if avg_te > 0.2 else 'moderate')
       if abs(results['net_directionality']) > 0.1:
           results['dominant_direction'] = (
               'respiratory→cardiac' if results['net_directionality'] > 0
               else 'cardiac→respiratory'
           )
       else:
           results['dominant_direction'] = 'balanced'

       return results

   # --- Usage ---
   fs = 256
   ecg  = generate_ecg_signal(sfecg=fs, duration=120, hrmean=72, Anoise=0.03)
   t    = np.arange(len(ecg)) / fs
   resp = 0.5 * np.sin(2 * np.pi * 0.25 * t) + 0.05 * np.random.randn(len(t))

   results = analyze_cardiorespiratory_coupling(ecg, resp, fs)

   if 'error' not in results:
       print(f"TE (Resp→Cardiac): {results['te_resp_to_cardiac']:.4f}")
       print(f"TE (Cardiac→Resp): {results['te_cardiac_to_resp']:.4f}")
       print(f"Net directionality: {results['net_directionality']:.4f}")
       print(f"Coupling strength : {results['coupling_strength']}")
       print(f"Dominant direction: {results['dominant_direction']}")
       print(f"Max cross-corr    : {results['max_correlation']:.3f} (lag={results['optimal_lag']})")

Example 8: Comprehensive Physiological Feature Extraction
===========================================================

Extracts time-domain, frequency-domain, nonlinear, and energy features from a single ECG recording.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from vitalDSP.filtering.signal_filtering import SignalFiltering
   from vitalDSP.physiological_features.time_domain import TimeDomainFeatures
   from vitalDSP.physiological_features.frequency_domain import FrequencyDomainFeatures
   from vitalDSP.physiological_features.nonlinear import NonlinearFeatures
   from vitalDSP.physiological_features.beat_to_beat import BeatToBeatAnalysis
   from vitalDSP.physiological_features.energy_analysis import EnergyAnalysis
   from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal

   def extract_comprehensive_features(ecg_signal, fs, patient_id=None):
       """
       Full-spectrum feature extraction: time, frequency, nonlinear, energy.

       Parameters
       ----------
       ecg_signal : numpy.ndarray
           ECG signal.
       fs : float
           Sampling frequency in Hz.

       Returns
       -------
       dict : Feature dictionary with clinical interpretation.
       """
       results = {'patient_id': patient_id, 'features': {}, 'interpretation': {}}

       # Preprocessing
       sf = SignalFiltering(ecg_signal)
       filtered = sf.bandpass(lowcut=0.5, highcut=40.0, fs=fs, order=4)

       # RR intervals
       bb = BeatToBeatAnalysis(filtered, fs=fs, signal_type='ECG')
       rr = bb.compute_rr_intervals()

       if len(rr) < 5:
           results['error'] = 'Insufficient RR intervals'
           return results

       # 1. TIME DOMAIN
       td = TimeDomainFeatures(rr)
       results['features'].update({
           'mean_rr'  : td.compute_mean_nn(),
           'sdnn'     : td.compute_sdnn(),
           'rmssd'    : td.compute_rmssd(),
           'sdsd'     : td.compute_sdsd(),
           'nn50'     : td.compute_nn50(),
           'pnn50'    : td.compute_pnn50(),
           'pnn20'    : td.compute_pnn20(),
           'mean_hr'  : 60000 / max(td.compute_mean_nn(), 1),
       })

       # 2. FREQUENCY DOMAIN  (nn_intervals as input, fs=4 Hz = typical RR series rate)
       fd = FrequencyDomainFeatures(rr, fs=4)
       psd = fd.compute_psd()
       lf  = psd['lf_power']
       hf  = psd['hf_power']
       results['features'].update({
           'vlf_power'   : psd['vlf_power'],
           'lf_power'    : lf,
           'hf_power'    : hf,
           'total_power' : psd['total_power'],
           'lf_hf_ratio' : psd['lf_hf_ratio'],
           'lf_nu'       : lf / (lf + hf + 1e-9) * 100,
           'hf_nu'       : hf / (lf + hf + 1e-9) * 100,
       })

       # 3. NONLINEAR
       nl = NonlinearFeatures(rr)
       poincare = nl.compute_poincare_features()
       results['features'].update({
           'sd1'               : poincare['sd1'],
           'sd2'               : poincare['sd2'],
           'sd_ratio'          : poincare['sd_ratio'],
           'sample_entropy'    : nl.compute_sample_entropy(m=2, r=0.2),
           'approx_entropy'    : nl.compute_approximate_entropy(m=2, r=0.2),
           'dfa_alpha'         : nl.compute_dfa(),
       })

       # 4. ENERGY (on the filtered ECG signal)
       ea = EnergyAnalysis(filtered, fs=fs)
       results['features'].update({
           'total_energy'    : ea.compute_total_energy(),
           'band_energy_lf'  : ea.compute_band_energy(low_freq=0.5, high_freq=5.0),
           'spectral_energy' : ea.compute_spectral_energy(),
       })

       # 5. INTERPRETATION
       feats = results['features']
       sdnn     = feats.get('sdnn', 0)
       lf_hf    = feats.get('lf_hf_ratio', 1.0)
       samp_ent = feats.get('sample_entropy', 1.0)

       interp = {}
       interp['hrv_status']       = ('reduced' if sdnn < 50 else 'good' if sdnn > 100 else 'normal')
       interp['autonomic_balance'] = ('parasympathetic' if lf_hf < 0.5 else
                                      'sympathetic'     if lf_hf > 2.0 else 'balanced')
       risk = sum([sdnn < 50, feats.get('rmssd', 50) < 20,
                   lf_hf > 2.5, samp_ent < 1.0])
       interp['risk_level']   = ('high' if risk >= 3 else 'moderate' if risk >= 1 else 'low')
       interp['cardiac_health'] = ('concerning' if risk >= 3 else 'fair' if risk >= 1 else 'healthy')
       results['interpretation'] = interp

       return results

   # --- Usage ---
   fs  = 256
   ecg = generate_ecg_signal(sfecg=fs, duration=120, hrmean=75, Anoise=0.05)
   res = extract_comprehensive_features(ecg, fs=fs, patient_id='P001')

   if 'error' not in res:
       f = res['features']
       print(f"=== Time Domain ===")
       print(f"Mean RR : {f['mean_rr']:.1f} ms    Mean HR : {f['mean_hr']:.1f} BPM")
       print(f"SDNN    : {f['sdnn']:.1f} ms    RMSSD   : {f['rmssd']:.1f} ms")
       print(f"pNN50   : {f['pnn50']:.2f} %")
       print(f"\n=== Frequency Domain ===")
       print(f"LF      : {f['lf_power']:.2f}    HF  : {f['hf_power']:.2f}")
       print(f"LF/HF   : {f['lf_hf_ratio']:.2f}")
       print(f"\n=== Nonlinear ===")
       print(f"SD1     : {f['sd1']:.2f}  SD2     : {f['sd2']:.2f}")
       print(f"SampEn  : {f['sample_entropy']:.3f}  DFA α: {f['dfa_alpha']:.3f}")
       i = res['interpretation']
       print(f"\n=== Interpretation ===")
       print(f"HRV     : {i['hrv_status']}    ANS   : {i['autonomic_balance']}")
       print(f"Risk    : {i['risk_level']}    Health: {i['cardiac_health']}")
   else:
       print(res['error'])

   # Export features to CSV
   features_df = pd.DataFrame([res['features']])
   features_df.to_csv(f"features_{res['patient_id']}.csv", index=False)

Example 9: Machine Learning for Physiological Signal Analysis
===============================================================

Anomaly detection, feature extraction with ``FeatureExtractor``, and CNN1D classification on synthetic ECG signals.

.. code-block:: python

   import numpy as np
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   from vitalDSP.advanced_computation.anomaly_detection import AnomalyDetection
   from vitalDSP.ml_models.feature_extractor import FeatureExtractor
   from vitalDSP.ml_models.deep_models import CNN1D

   def ml_signal_analysis_pipeline(clean_signals, noisy_signals, fs, anomaly_threshold=2.5):
       """
       ML pipeline: anomaly detection → feature extraction → CNN1D classification.

       Parameters
       ----------
       clean_signals : list of numpy.ndarray
           Reference (normal) signals for training.
       noisy_signals : list of numpy.ndarray
           Signals to screen for anomalies and classify.
       fs : float
           Sampling frequency.
       anomaly_threshold : float
           Z-score threshold for anomaly flagging.

       Returns
       -------
       dict : Anomaly counts, extracted features, and CNN1D training history.
       """
       results = {'anomaly': {}, 'classification': {}}
       sig_len = min(len(s) for s in clean_signals)

       # 1. ANOMALY DETECTION
       anomaly_info = []
       for i, sig in enumerate(noisy_signals):
           det = AnomalyDetection(sig)
           z_anomalies  = det.detect_anomalies(method='z_score',  threshold=anomaly_threshold)
           iqr_anomalies = det.detect_anomalies(method='iqr')
           anomaly_info.append({
               'signal_index'       : i,
               'z_score_anomalies'  : len(z_anomalies),
               'iqr_anomalies'      : len(iqr_anomalies),
               'quality'            : 'good' if len(z_anomalies) < 10 else 'poor',
           })

       results['anomaly']['per_signal'] = anomaly_info
       results['anomaly']['good']       = sum(1 for a in anomaly_info if a['quality'] == 'good')
       results['anomaly']['poor']       = sum(1 for a in anomaly_info if a['quality'] == 'poor')
       print(f"Good-quality signals: {results['anomaly']['good']}/{len(noisy_signals)}")

       # 2. FEATURE EXTRACTION  (sklearn-compatible transformer)
       X_clean = np.array([s[:sig_len] for s in clean_signals])
       extractor = FeatureExtractor(signal_type='ecg', sampling_rate=float(fs))
       X_features = extractor.fit_transform(X_clean)
       results['classification']['n_features'] = X_features.shape[1]
       print(f"Features extracted : {X_features.shape[1]}")

       # 3. CNN1D CLASSIFICATION
       scaler = StandardScaler()
       X_norm = scaler.fit_transform(X_clean)
       X_3d   = X_norm.reshape(X_norm.shape[0], sig_len, 1)

       # Synthetic binary labels: high-noise vs low-noise
       noise_std = np.array([np.std(n[:sig_len] - c[:sig_len])
                             for n, c in zip(noisy_signals, clean_signals)])
       labels = (noise_std > np.median(noise_std)).astype(int)

       X_train, X_test, y_train, y_test = train_test_split(
           X_3d, labels, test_size=0.2, random_state=42
       )

       cnn = CNN1D(
           input_shape=(sig_len, 1),
           n_classes=2,
           n_filters=[32, 64],
           kernel_sizes=[3],
           pool_sizes=[2],
       )
       history = cnn.train(X_train, y_train, X_val=X_test, y_val=y_test,
                           epochs=10, batch_size=16)
       results['classification']['history']  = history
       results['classification']['n_train']  = len(X_train)
       results['classification']['n_test']   = len(X_test)
       print(f"CNN1D trained: {len(X_train)} train / {len(X_test)} test samples")

       return results

   # --- Usage ---
   np.random.seed(42)
   n_signals = 30
   fs = 128
   sig_len = 640  # 5 seconds

   clean  = [np.sin(2 * np.pi * 1.2 * np.arange(sig_len) / fs) +
             0.05 * np.random.randn(sig_len) for _ in range(n_signals)]
   noisy  = [c + np.random.normal(0, np.random.uniform(0.05, 0.3), sig_len)
             for c in clean]

   results = ml_signal_analysis_pipeline(clean, noisy, fs=fs)
   print(f"Features: {results['classification']['n_features']}")
   print(f"Anomaly good/poor: {results['anomaly']['good']}/{results['anomaly']['poor']}")

These examples provide a solid foundation for implementing VitalDSP in real-world scenarios. Refer to the :ref:`sample_notebooks` section for interactive Jupyter notebook versions.
