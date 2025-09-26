Troubleshooting
===============

This section provides solutions to common issues encountered when using VitalDSP. If you don't find your issue here, please check the GitHub issues page or contact our support team.

Installation Issues
====================

**Q: Installation fails with "No module named 'vitalDSP'"**

A: This usually indicates that VitalDSP wasn't installed correctly. Try the following solutions:

1. **Verify Python version:**
   .. code-block:: bash
   
      python --version  # Should be 3.8 or higher

2. **Reinstall VitalDSP:**
   .. code-block:: bash
   
      pip uninstall vital-DSP
      pip install vital-DSP

3. **Install with verbose output:**
   .. code-block:: bash
   
      pip install vital-DSP -v

4. **Check installation:**
   .. code-block:: python
   
      import vitalDSP
      print(f"VitalDSP version: {vitalDSP.__version__}")

**Q: ImportError: No module named 'scipy' or other dependencies**

A: VitalDSP requires several scientific Python packages. Install them:

.. code-block:: bash

   pip install scipy numpy pandas matplotlib plotly scikit-learn

**Q: Installation fails on Windows with Microsoft Visual C++ errors**

A: This is a common issue with packages that include C extensions. Try:

1. **Install Microsoft Visual C++ Build Tools:**
   Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

2. **Use conda instead of pip:**
   .. code-block:: bash
   
      conda install -c conda-forge vital-DSP

3. **Install pre-compiled wheels:**
   .. code-block:: bash
   
      pip install --only-binary=all vital-DSP

**Q: Permission denied errors during installation**

A: This usually happens when trying to install to system Python. Try:

1. **Use virtual environment:**
   .. code-block:: bash
   
      python -m venv vitaldsp_env
      source vitaldsp_env/bin/activate  # On Windows: vitaldsp_env\Scripts\activate
      pip install vital-DSP

2. **Install for user only:**
   .. code-block:: bash
   
      pip install --user vital-DSP

3. **Use sudo (Linux/Mac):**
   .. code-block:: bash
   
      sudo pip install vital-DSP

Signal Processing Issues
========================

**Q: Signal filtering produces unexpected results**

A: Check the following:

1. **Verify sampling frequency:**
   .. code-block:: python
   
      print(f"Sampling frequency: {fs} Hz")
      print(f"Signal length: {len(signal)} samples")
      print(f"Duration: {len(signal)/fs:.2f} seconds")

2. **Check filter parameters:**
   .. code-block:: python
   
      # For ECG signals
      filtered = sf.bandpass_filter(low_cut=0.5, high_cut=40.0)
      
      # For PPG signals
      filtered = sf.bandpass_filter(low_cut=0.5, high_cut=8.0)
      
      # For respiratory signals
      filtered = sf.bandpass_filter(low_cut=0.1, high_cut=0.5)

3. **Validate input signal:**
   .. code-block:: python
   
      print(f"Signal range: {np.min(signal):.4f} to {np.max(signal):.4f}")
      print(f"Signal mean: {np.mean(signal):.4f}")
      print(f"Signal std: {np.std(signal):.4f}")

**Q: R-peak detection fails or produces incorrect results**

A: This is often due to signal quality or parameter issues:

1. **Check signal quality:**
   .. code-block:: python
   
      from vitalDSP.signal_quality_assessment.signal_quality_index import SignalQualityIndex
      sqi = SignalQualityIndex(signal)
      quality, _, _ = sqi.amplitude_variability_sqi(window_size=fs*5, step_size=fs*1, threshold=2)
      print(f"Signal quality: {np.mean(quality):.4f}")

2. **Adjust detection parameters:**
   .. code-block:: python
   
      from vitalDSP.physiological_features.waveform import WaveformMorphology
      wm = WaveformMorphology(signal, fs=fs, signal_type="ecg")
      
      # Check detected peaks
      r_peaks = wm.r_peaks
      print(f"Detected {len(r_peaks)} R-peaks")
      
      # Visualize results
      import matplotlib.pyplot as plt
      plt.plot(signal)
      plt.plot(r_peaks, signal[r_peaks], 'ro')
      plt.show()

3. **Preprocess signal before detection:**
   .. code-block:: python
   
      # Apply filtering first
      sf = SignalFiltering(signal, fs)
      filtered_signal = sf.bandpass_filter(low_cut=0.5, high_cut=40.0)
      
      # Then detect peaks
      wm = WaveformMorphology(filtered_signal, fs=fs, signal_type="ecg")

**Q: HRV analysis produces unrealistic values**

A: HRV analysis requires high-quality RR intervals:

1. **Validate RR intervals:**
   .. code-block:: python
   
      rr_intervals = np.diff(r_peaks) / fs * 1000  # Convert to ms
      print(f"RR interval range: {np.min(rr_intervals):.1f} - {np.max(rr_intervals):.1f} ms")
      
      # Remove outliers
      valid_rr = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]
      print(f"Valid RR intervals: {len(valid_rr)} out of {len(rr_intervals)}")

2. **Check signal length:**
   .. code-block:: python
   
      duration = len(signal) / fs
      print(f"Signal duration: {duration:.1f} seconds")
      # HRV analysis typically requires at least 2-5 minutes of data

3. **Use appropriate analysis parameters:**
   .. code-block:: python
   
      from vitalDSP.physiological_features.hrv_analysis import HRVFeatures
      hrv = HRVFeatures(valid_rr)
      
      # For short-term analysis (2-5 minutes)
      hrv_features = hrv.analyze_hrv()
      
      # For long-term analysis (24 hours)
      hrv_features = hrv.analyze_hrv(long_term=True)

**Q: Respiratory rate estimation is inaccurate**

A: Respiratory rate estimation depends on signal quality and method selection:

1. **Check signal preprocessing:**
   .. code-block:: python
   
      from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis
      resp_analysis = RespiratoryAnalysis(signal, fs)
      
      # Apply respiratory-specific filtering
      filtered_resp = resp_analysis.preprocess_signal(
          detrend=True,
          normalize=True,
          filter_type='bandpass',
          low_freq=0.1,  # 6 breaths/min
          high_freq=0.5   # 30 breaths/min
      )

2. **Try multiple estimation methods:**
   .. code-block:: python
   
      # Peak detection method
      resp_rate_peak = resp_analysis.estimate_respiratory_rate_peak_detection()
      
      # FFT method
      resp_rate_fft = resp_analysis.estimate_respiratory_rate_fft()
      
      # Ensemble method
      resp_rate_ensemble = resp_analysis.estimate_respiratory_rate_ensemble()
      
      print(f"Peak detection: {resp_rate_peak:.1f} breaths/min")
      print(f"FFT method: {resp_rate_fft:.1f} breaths/min")
      print(f"Ensemble method: {resp_rate_ensemble:.1f} breaths/min")

3. **Validate against known respiratory rate:**
   .. code-block:: python
   
      # Check if estimated rate is physiologically reasonable
      if 8 <= resp_rate_ensemble <= 30:
           print("Respiratory rate is within normal range")
      else:
           print("Respiratory rate may be inaccurate")

Web Application Issues
=======================

**Q: Web application won't start**

A: Check the following:

1. **Verify port availability:**
   .. code-block:: bash
   
      # Check if port 8050 is available
      netstat -an | grep 8050
      
      # Or try a different port
      python -m vitalDSP_webapp.run_webapp --port 8051

2. **Check dependencies:**
   .. code-block:: bash
   
      pip install dash plotly dash-bootstrap-components

3. **Check Python version:**
   .. code-block:: bash
   
      python --version  # Should be 3.8 or higher

4. **Run with debug mode:**
   .. code-block:: python
   
      from vitalDSP_webapp.run_webapp import run_webapp
      run_webapp(debug=True, port=8050)

**Q: Signal upload fails in web application**

A: This is usually due to data format issues:

1. **Check file format:**
   - Supported formats: CSV, Excel (.xlsx), JSON
   - Ensure file is not corrupted
   - Check file size (should be < 16MB)

2. **Verify data structure:**
   .. code-block:: python
   
      import pandas as pd
      
      # Check CSV format
      df = pd.read_csv('your_file.csv')
      print(df.head())
      print(df.columns)
      
      # Ensure time and signal columns exist
      if 'time' not in df.columns:
           print("Time column not found")
      if 'signal' not in df.columns:
           print("Signal column not found")

3. **Check data quality:**
   .. code-block:: python
   
      # Check for missing values
      print(df.isnull().sum())
      
      # Check data types
      print(df.dtypes)
      
      # Check data range
      print(df.describe())

**Q: Visualizations don't display in web application**

A: This is usually a browser or JavaScript issue:

1. **Check browser compatibility:**
   - Use Chrome, Firefox, or Safari
   - Ensure JavaScript is enabled
   - Clear browser cache

2. **Check console for errors:**
   - Open browser developer tools (F12)
   - Check Console tab for JavaScript errors
   - Check Network tab for failed requests

3. **Try different browser:**
   - Test in incognito/private mode
   - Try different browser
   - Check if ad blockers are interfering

**Q: Filtered data not available in analysis screens**

A: This is due to the new workflow where filtering and analysis are separated:

1. **Apply filtering first:**
   - Navigate to Filtering screen
   - Select and configure filter
   - Apply filtering
   - Verify filtered data is available

2. **Check filter status:**
   - Look for filter information display
   - Verify filter parameters are correct
   - Check if filter was applied successfully

3. **Reapply filtering if needed:**
   - Go back to Filtering screen
   - Reconfigure filter parameters
   - Apply filtering again

Performance Issues
==================

**Q: Signal processing is slow**

A: Optimize performance with these techniques:

1. **Reduce signal length:**
   .. code-block:: python
   
      # Process shorter segments
      segment_length = fs * 60  # 1 minute segments
      for i in range(0, len(signal), segment_length):
           segment = signal[i:i+segment_length]
           # Process segment
           results = process_signal(segment, fs)

2. **Use appropriate sampling rates:**
   .. code-block:: python
   
      # For ECG analysis, 250-500 Hz is usually sufficient
      # For PPG analysis, 100-200 Hz is usually sufficient
      # For respiratory analysis, 50-100 Hz is usually sufficient

3. **Optimize filter parameters:**
   .. code-block:: python
   
      # Use lower filter orders for faster processing
      filtered = sf.bandpass_filter(low_cut=0.5, high_cut=40.0, filter_order=2)
      
      # Use simpler filters when possible
      filtered = sf.lowpass_filter(cutoff=40.0, filter_order=2)

4. **Use batch processing:**
   .. code-block:: python
   
      # Process multiple signals in batch
      results = []
      for signal in signal_list:
           result = process_signal(signal, fs)
           results.append(result)

**Q: Memory usage is high**

A: Reduce memory usage with these techniques:

1. **Process signals in chunks:**
   .. code-block:: python
   
      chunk_size = 10000  # Process 10k samples at a time
      for i in range(0, len(signal), chunk_size):
           chunk = signal[i:i+chunk_size]
           # Process chunk
           result = process_signal(chunk, fs)

2. **Use efficient data types:**
   .. code-block:: python
   
      # Use float32 instead of float64 when possible
      signal = signal.astype(np.float32)
      
      # Use appropriate data types
      rr_intervals = rr_intervals.astype(np.float32)

3. **Clear unused variables:**
   .. code-block:: python
   
      # Clear large variables when done
      del large_signal
      del intermediate_results
      
      # Force garbage collection
      import gc
      gc.collect()

**Q: Real-time processing is not fast enough**

A: Optimize for real-time performance:

1. **Use efficient algorithms:**
   .. code-block:: python
   
      # Use simpler filters
      filtered = sf.bandpass_filter(low_cut=0.5, high_cut=40.0, filter_order=2)
      
      # Use faster peak detection
      wm = WaveformMorphology(signal, fs=fs, signal_type="ecg")
      r_peaks = wm.r_peaks

2. **Reduce processing frequency:**
   .. code-block:: python
   
      # Process every 5 seconds instead of every second
      update_interval = 5
      time.sleep(update_interval)

3. **Use threading for parallel processing:**
   .. code-block:: python
   
      import threading
      
      def process_signal_async(signal, fs):
           # Process signal in background thread
           result = process_signal(signal, fs)
           return result
      
      # Start processing in background
      thread = threading.Thread(target=process_signal_async, args=(signal, fs))
      thread.start()

Data Quality Issues
====================

**Q: Signal quality is poor**

A: Improve signal quality with these techniques:

1. **Check signal preprocessing:**
   .. code-block:: python
   
      # Apply detrending
      from scipy import signal
      detrended = signal.detrend(signal)
      
      # Apply normalization
      normalized = (signal - np.mean(signal)) / np.std(signal)

2. **Apply appropriate filtering:**
   .. code-block:: python
   
      # Remove high-frequency noise
      filtered = sf.lowpass_filter(cutoff=40.0)
      
      # Remove low-frequency drift
      filtered = sf.highpass_filter(cutoff=0.5)

3. **Check for artifacts:**
   .. code-block:: python
   
      from vitalDSP.filtering.artifact_removal import ArtifactRemoval
      ar = ArtifactRemoval(signal)
      
      # Remove motion artifacts
      clean_signal = ar.motion_artifact_removal()
      
      # Remove powerline interference
      clean_signal = ar.powerline_interference_removal(freq=50)  # or 60 Hz

**Q: Missing values in signal data**

A: Handle missing values appropriately:

1. **Check for missing values:**
   .. code-block:: python
   
      missing_count = np.isnan(signal).sum()
      print(f"Missing values: {missing_count}")
      
      if missing_count > 0:
           print("Signal contains missing values")

2. **Interpolate missing values:**
   .. code-block:: python
   
      from scipy import interpolate
      
      # Create mask for valid values
      valid_mask = ~np.isnan(signal)
      valid_indices = np.where(valid_mask)[0]
      valid_values = signal[valid_mask]
      
      # Interpolate missing values
      if len(valid_indices) > 1:
           f = interpolate.interp1d(valid_indices, valid_values, kind='linear')
           all_indices = np.arange(len(signal))
           interpolated = f(all_indices)
           signal = interpolated

3. **Remove segments with too many missing values:**
   .. code-block:: python
   
      # Remove segments with >50% missing values
      window_size = fs * 5  # 5-second windows
      for i in range(0, len(signal), window_size):
           window = signal[i:i+window_size]
           missing_ratio = np.isnan(window).sum() / len(window)
           
           if missing_ratio > 0.5:
               # Mark window as invalid
               signal[i:i+window_size] = np.nan

**Q: Signal amplitude is too low or too high**

A: Adjust signal amplitude:

1. **Check signal range:**
   .. code-block:: python
   
      print(f"Signal range: {np.min(signal):.4f} to {np.max(signal):.4f}")
      print(f"Signal mean: {np.mean(signal):.4f}")
      print(f"Signal std: {np.std(signal):.4f}")

2. **Normalize signal:**
   .. code-block:: python
   
      # Z-score normalization
      normalized = (signal - np.mean(signal)) / np.std(signal)
      
      # Min-max normalization
      normalized = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
      
      # Scale to specific range
      target_min, target_max = -1, 1
      scaled = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
      scaled = scaled * (target_max - target_min) + target_min

3. **Check for clipping:**
   .. code-block:: python
   
      # Check for signal clipping
      clipped_count = np.sum(np.abs(signal) >= 0.99 * np.max(np.abs(signal)))
      if clipped_count > len(signal) * 0.01:  # More than 1% clipped
           print("Signal may be clipped")

Machine Learning Issues
========================

**Q: Neural network filtering fails to train**

A: Check the following:

1. **Verify data quality:**
   .. code-block:: python
   
      # Ensure signal is properly preprocessed
      if np.any(np.isnan(signal)):
           print("Signal contains NaN values")
           signal = np.nan_to_num(signal)
      
      if np.any(np.isinf(signal)):
           print("Signal contains infinite values")
           signal = np.nan_to_num(signal)

2. **Check data size:**
   .. code-block:: python
   
      print(f"Signal length: {len(signal)}")
      # Neural networks typically require at least 1000 samples
      if len(signal) < 1000:
           print("Signal may be too short for neural network training")

3. **Adjust training parameters:**
   .. code-block:: python
   
      from vitalDSP.advanced_computation.neural_network_filtering import NeuralNetworkFiltering
      
      nn_filter = NeuralNetworkFiltering(
           model_type='autoencoder',
           hidden_layers=[32, 16, 8],  # Smaller network
           epochs=50,  # Fewer epochs
           learning_rate=0.01,  # Higher learning rate
           batch_size=32
      )

**Q: Anomaly detection produces too many false positives**

A: Adjust detection parameters:

1. **Tune contamination parameter:**
   .. code-block:: python
   
      from vitalDSP.advanced_computation.anomaly_detection import AnomalyDetection
      
      # Lower contamination for fewer anomalies
      anomaly_detector = AnomalyDetection(
           method='isolation_forest',
           contamination=0.05  # 5% instead of 10%
      )

2. **Use different detection methods:**
   .. code-block:: python
   
      # Try different methods
      methods = ['isolation_forest', 'one_class_svm', 'local_outlier_factor']
      
      for method in methods:
           detector = AnomalyDetection(method=method)
           anomalies = detector.detect_anomalies(signal)
           print(f"{method}: {np.sum(anomalies)} anomalies")

3. **Preprocess signal before detection:**
   .. code-block:: python
   
      # Apply filtering first
      sf = SignalFiltering(signal, fs)
      filtered_signal = sf.bandpass_filter(low_cut=0.5, high_cut=40.0)
      
      # Then detect anomalies
      anomaly_detector = AnomalyDetection()
      anomalies = anomaly_detector.detect_anomalies(filtered_signal)

**Q: Bayesian optimization is slow**

A: Optimize performance:

1. **Reduce search space:**
   .. code-block:: python
   
      # Narrow parameter ranges
      param_bounds = {
           'low_cut': (0.5, 2.0),  # Narrower range
           'high_cut': (20.0, 40.0),  # Narrower range
           'filter_order': (2, 4)  # Fewer options
      }

2. **Reduce iterations:**
   .. code-block:: python
   
      from vitalDSP.advanced_computation.bayesian_optimization import BayesianOptimization
      
      bo = BayesianOptimization(objective_function, param_bounds)
      bo.optimize(n_iter=10)  # Fewer iterations

3. **Use faster objective function:**
   .. code-block:: python
   
      def fast_objective_function(params):
           # Use simpler calculations
           sf = SignalFiltering(signal, fs)
           filtered = sf.bandpass_filter(
               low_cut=params['low_cut'],
               high_cut=params['high_cut'],
               filter_order=int(params['filter_order'])
           )
           
           # Use faster quality metric
           return np.std(filtered)  # Simple metric

Deployment Issues
==================

**Q: Application fails to start in production**

A: Check the following:

1. **Verify environment:**
   .. code-block:: bash
   
      # Check Python version
      python --version
      
      # Check installed packages
      pip list | grep vital-DSP
      
      # Check environment variables
      echo $PYTHONPATH

2. **Check dependencies:**
   .. code-block:: bash
   
      # Install all dependencies
      pip install -r requirements.txt
      
      # Check for missing packages
      python -c "import vitalDSP; print('VitalDSP imported successfully')"

3. **Check file permissions:**
   .. code-block:: bash
   
      # Ensure proper permissions
      chmod +x run_webapp.py
      
      # Check file ownership
      ls -la run_webapp.py

**Q: Performance is poor in production**

A: Optimize for production:

1. **Use production settings:**
   .. code-block:: python
   
      from vitalDSP_webapp.run_webapp import run_webapp
      
      run_webapp(
           debug=False,  # Disable debug mode
           host='0.0.0.0',  # Allow external connections
           port=8050,
           threaded=True  # Enable threading
      )

2. **Use production WSGI server:**
   .. code-block:: bash
   
      # Install gunicorn
      pip install gunicorn
      
      # Run with gunicorn
      gunicorn --bind 0.0.0.0:8050 vitalDSP_webapp.app:app

3. **Optimize resource usage:**
   .. code-block:: python
   
      # Set memory limits
      import resource
      resource.setrlimit(resource.RLIMIT_AS, (2**30, 2**30))  # 1GB limit
      
      # Use efficient data structures
      import gc
      gc.set_threshold(1000, 10, 10)  # Optimize garbage collection

Getting Help
============

If you're still experiencing issues after trying these solutions:

1. **Check the GitHub Issues:**
   - Search for similar issues: https://github.com/Oucru-Innovations/vital-DSP/issues
   - Create a new issue with detailed information

2. **Contact Support:**
   - Email: support@vitaldsp.com
   - Include error messages, system information, and code examples

3. **Community Forum:**
   - Join our community forum for discussions and help
   - Share your experiences and solutions

4. **Documentation:**
   - Check the API reference for detailed function documentation
   - Review the tutorials for step-by-step guidance
   - Explore the examples for practical implementations

**When reporting issues, please include:**
- Python version
- Operating system
- VitalDSP version
- Complete error message
- Code that reproduces the issue
- Expected vs. actual behavior

This information helps us provide faster and more accurate support.
