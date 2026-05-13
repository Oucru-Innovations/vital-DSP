Tutorials
=========

This section provides step-by-step tutorials for working with vitalDSP. Each
tutorial uses verified, working APIs and runnable code examples.

**Prerequisites:** Python 3.8+, vitalDSP installed (see :ref:`getting_started`),
numpy and scipy available.

.. contents:: Tutorials
   :local:
   :depth: 1


Tutorial 1: Basic Signal Processing and Filtering
==================================================

This tutorial covers synthetic signal generation, common filtering techniques,
and artifact removal.

Generating a Synthetic ECG Signal
----------------------------------

.. code-block:: python

    import numpy as np
    from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal

    # Generate a 10-second ECG at 256 Hz with mild noise
    ecg = generate_ecg_signal(sfecg=256, duration=10, hrmean=70, Anoise=0.01)
    print(f"Signal length: {len(ecg)} samples")
    print(f"Duration: {len(ecg) / 256:.1f} s at 256 Hz")

``generate_ecg_signal`` uses the McSharry dynamical model and returns a 1-D
numpy array. Key parameters:

* ``sfecg`` -- output sampling frequency in Hz (must divide ``sfint``, default 512)
* ``duration`` -- approximate duration in seconds
* ``hrmean`` -- mean heart rate in beats per minute
* ``Anoise`` -- amplitude of additive uniform noise

Bandpass and Moving-Average Filtering
--------------------------------------

.. code-block:: python

    from vitalDSP.filtering.signal_filtering import SignalFiltering

    sf = SignalFiltering(ecg)

    # Bandpass filter: keep 0.5–40 Hz (typical ECG band)
    filtered = sf.bandpass(lowcut=0.5, highcut=40, fs=256, order=4)

    # Moving average for baseline trend estimation
    smoothed = sf.moving_average(window_size=11)

``SignalFiltering`` wraps common filters. The ``bandpass`` method uses a
Butterworth filter by default. ``moving_average`` supports ``'edge'``,
``'reflect'``, and ``'constant'`` padding modes and an ``iterations``
parameter for repeated smoothing passes.

Other available filter methods on ``SignalFiltering``:

* ``sf.gaussian(sigma=1.0)`` -- Gaussian smoothing
* ``sf.butterworth(cutoff, fs, order, btype)`` -- direct Butterworth
* ``SignalFiltering.savgol_filter(signal, window_length, polyorder)`` -- Savitzky-Golay (static)

Artifact Removal
-----------------

``ArtifactRemoval`` provides baseline correction and spike suppression. It
takes only the signal on construction (no ``fs`` argument in ``__init__``);
sampling frequency is passed to individual methods where needed.

.. code-block:: python

    from vitalDSP.filtering.artifact_removal import ArtifactRemoval

    ar = ArtifactRemoval(ecg)

    # High-pass filter to remove baseline wander below 0.5 Hz
    corrected = ar.baseline_correction(cutoff=0.5, fs=256)

    # Median filter to suppress spike artifacts
    despiked = ar.median_filter_removal(kernel_size=5)

    # Wavelet denoising using Daubechies-4 at two decomposition levels
    denoised = ar.wavelet_denoising(wavelet_type='db', level=2, order=4)

    print(f"Corrected: {corrected.shape}, Denoised: {denoised.shape}")

Additional ``ArtifactRemoval`` methods:

* ``mean_subtraction()`` -- removes DC offset
* ``wavelet_denoising(wavelet_type, level, order)`` -- supports ``'haar'``, ``'db'``, ``'sym'``, ``'coif'``, ``'custom'``

Plotting the Results
---------------------

.. code-block:: python

    import matplotlib.pyplot as plt

    t = np.arange(len(ecg)) / 256.0

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, ecg, lw=0.8, label="Raw ECG")
    axes[1].plot(t, filtered, lw=0.8, color="steelblue", label="Bandpass 0.5–40 Hz")
    axes[2].plot(t, denoised, lw=0.8, color="darkgreen", label="Wavelet denoised")
    for ax in axes:
        ax.legend(loc="upper right")
        ax.set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


Tutorial 2: Heart Rate Variability Analysis
============================================

This tutorial detects R-peaks from a synthetic ECG, derives RR intervals, and
computes time-domain and frequency-domain HRV features.

R-Peak Detection with WaveformMorphology
-----------------------------------------

.. code-block:: python

    import numpy as np
    from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal
    from vitalDSP.physiological_features.waveform import WaveformMorphology

    # Use a longer signal so more beats are available for HRV
    ecg = generate_ecg_signal(sfecg=256, duration=60, hrmean=70, Anoise=0.005)
    fs = 256

    wm = WaveformMorphology(ecg, fs=fs, signal_type="ECG")
    r_peaks = wm.r_peaks          # array of sample indices
    print(f"Detected {len(r_peaks)} R-peaks")

``WaveformMorphology`` constructor parameters:

* ``waveform`` -- 1-D numpy array
* ``fs`` -- sampling frequency in Hz (default 256)
* ``signal_type`` -- ``'ECG'``, ``'PPG'``, or ``'EEG'``
* ``peak_config`` -- optional dict to override detection thresholds

``wm.r_peaks`` is computed during construction. Additional ECG morphology
methods: ``wm.detect_q_valley()``, ``wm.detect_s_valley()``.

Computing RR Intervals and HRV Features
-----------------------------------------

.. code-block:: python

    from vitalDSP.physiological_features.hrv_analysis import HRVFeatures

    # Convert R-peak sample indices to RR intervals in milliseconds
    rr_intervals = np.diff(r_peaks) / fs * 1000.0
    print(f"RR intervals: n={len(rr_intervals)}, mean={rr_intervals.mean():.1f} ms")

    # Initialise HRVFeatures with pre-computed intervals
    hrv = HRVFeatures(ecg, nn_intervals=rr_intervals, fs=fs, signal_type="ECG")
    features = hrv.compute_all_features()

    # Time-domain features
    print(f"SDNN:   {features['sdnn']:.2f} ms")
    print(f"RMSSD:  {features['rmssd']:.2f} ms")
    print(f"pNN50:  {features['pnn50']:.4f}")
    print(f"Mean NN:{features['mean_nn']:.1f} ms")

``HRVFeatures`` constructor:

* ``signals`` -- raw signal array (may be ``None`` when ``nn_intervals`` is provided)
* ``nn_intervals`` -- pre-computed RR/NN intervals in milliseconds
* ``fs`` -- sampling frequency
* ``signal_type`` -- ``'ECG'`` or ``'PPG'``

``compute_all_features()`` returns a dict with time-domain keys (``sdnn``,
``rmssd``, ``nn50``, ``pnn50``, ``mean_nn``, ``median_nn``, ``cvnn``,
``sdsd``, etc.), frequency-domain keys (``lf``, ``hf``, ``lf_hf_ratio``,
``vlf``), and nonlinear keys (``sd1``, ``sd2``).

Poincare Plot
--------------

.. code-block:: python

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(rr_intervals[:-1], rr_intervals[1:], s=10, alpha=0.6)
    ax.set_xlabel("RR(n) [ms]")
    ax.set_ylabel("RR(n+1) [ms]")
    ax.set_title("Poincare Plot")
    plt.tight_layout()
    plt.show()


Tutorial 3: Respiratory Signal Analysis
========================================

This tutorial generates a respiratory signal and estimates the respiratory rate
using multiple methods provided by ``RespiratoryAnalysis``.

Generating a Respiratory Signal
---------------------------------

.. code-block:: python

    import numpy as np
    from vitalDSP.utils.data_processing.synthesize_data import generate_resp_signal

    fs = 256
    duration = 60   # seconds

    # Sinusoidal respiratory signal at 0.25 Hz (~15 breaths/min)
    resp = generate_resp_signal(
        sampling_rate=fs,
        duration=duration,
        frequency=0.25,   # Hz
        amplitude=1.0,
    )
    print(f"Respiratory signal: {len(resp)} samples, {duration} s")

``generate_resp_signal`` returns a single numpy array. For a more realistic
signal, add cardiac contamination:

.. code-block:: python

    from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal

    ecg = generate_ecg_signal(sfecg=fs, duration=duration, hrmean=70, Anoise=0.0)
    # Respiratory modulation embedded in PPG/ECG amplitude
    resp_with_cardiac = resp + 0.05 * ecg[:len(resp)]

Estimating Respiratory Rate
-----------------------------

``RespiratoryAnalysis`` supports several estimation methods. It returns either a
``float`` (breaths per minute) or a dict depending on the method.

.. code-block:: python

    from vitalDSP.respiratory_analysis.respiratory_analysis import RespiratoryAnalysis

    ra = RespiratoryAnalysis(resp, fs=fs)

    # FFT-based estimate: returns a float (breaths/min)
    rr_fft = ra.compute_respiratory_rate(method="fft_based")
    print(f"FFT-based respiratory rate: {rr_fft:.1f} breaths/min")

    # Peak-counting estimate
    rr_count = ra.compute_respiratory_rate(method="counting")
    print(f"Counting-based respiratory rate: {rr_count:.1f} breaths/min")

Available ``method`` values: ``'fft_based'``, ``'counting'``, ``'peaks'``,
``'zero_crossing'``, ``'time_domain'``, ``'frequency_domain'``.

Preprocessing Before Estimation
---------------------------------

.. code-block:: python

    from vitalDSP.preprocess.preprocess_operations import PreprocessConfig

    config = PreprocessConfig(
        filter_type="bandpass",
        lowcut=0.1,
        highcut=2.0,
        noise_reduction_method="wavelet",
    )

    ra_noisy = RespiratoryAnalysis(resp_with_cardiac, fs=fs)
    rr_preprocessed = ra_noisy.compute_respiratory_rate(
        method="fft_based",
        preprocess_config=config,
    )
    print(f"Preprocessed estimate: {rr_preprocessed:.1f} breaths/min")


Tutorial 4: Signal Quality Assessment
=======================================

``SignalQualityIndex`` evaluates signal quality over sliding windows and
returns either an aggregated scalar (``aggregate=True``, the default) or a
detailed tuple.

Computing SQI Metrics
-----------------------

.. code-block:: python

    import numpy as np
    from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal
    from vitalDSP.signal_quality_assessment.signal_quality_index import SignalQualityIndex

    ecg = generate_ecg_signal(sfecg=256, duration=10, hrmean=70, Anoise=0.01)
    sqi = SignalQualityIndex(ecg)

    window_size = 256   # 1-second window at 256 Hz
    step_size   = 128   # 50 % overlap

    amp_sqi  = sqi.amplitude_variability_sqi(window_size, step_size)
    bw_sqi   = sqi.baseline_wander_sqi(window_size, step_size)
    zc_sqi   = sqi.zero_crossing_sqi(window_size, step_size)
    snr_sqi  = sqi.snr_sqi(window_size, step_size)
    energy   = sqi.energy_sqi(window_size, step_size)
    p2p      = sqi.peak_to_peak_amplitude_sqi(window_size, step_size)

    print(f"Amplitude variability SQI : {amp_sqi:.4f}")
    print(f"Baseline wander SQI       : {bw_sqi:.4f}")
    print(f"Zero-crossing SQI         : {zc_sqi:.4f}")
    print(f"SNR SQI                   : {snr_sqi:.4f}")
    print(f"Energy SQI                : {energy:.4f}")
    print(f"Peak-to-peak SQI          : {p2p:.4f}")

All SQI methods share the same signature pattern::

    sqi_value = sqi.<method>(window_size, step_size,
                             threshold=None,
                             threshold_type='below',
                             scale='zscore',
                             aggregate=True)

Segment-level Detail
---------------------

Pass ``aggregate=False`` to get per-segment values and normal/abnormal
segment lists:

.. code-block:: python

    sqi_vals, normal_segs, abnormal_segs = sqi.amplitude_variability_sqi(
        window_size, step_size,
        threshold=0.5,
        threshold_type="below",
        aggregate=False,
    )
    print(f"Segments evaluated : {len(sqi_vals)}")
    print(f"Normal segments    : {len(normal_segs)}")
    print(f"Abnormal segments  : {len(abnormal_segs)}")
    print(f"First SQI values   : {sqi_vals[:5].round(4)}")

``normal_segs`` and ``abnormal_segs`` are lists of ``(start, end)`` index
tuples that can be used to extract the corresponding signal windows.

Available SQI methods:

* ``amplitude_variability_sqi``
* ``baseline_wander_sqi``
* ``zero_crossing_sqi``
* ``waveform_similarity_sqi``
* ``signal_entropy_sqi``
* ``skewness_sqi``
* ``kurtosis_sqi``
* ``peak_to_peak_amplitude_sqi``
* ``snr_sqi``
* ``energy_sqi``
* ``heart_rate_variability_sqi``
* ``ppg_signal_quality_sqi``
* ``eeg_band_power_sqi``
* ``respiratory_signal_quality_sqi``


Tutorial 5: Signal Decomposition with EMD and Wavelets
=======================================================

This tutorial demonstrates two decomposition approaches: Empirical Mode
Decomposition (EMD) and the Discrete Wavelet Transform (DWT).

Empirical Mode Decomposition
------------------------------

EMD is a data-driven method that decomposes a signal into Intrinsic Mode
Functions (IMFs) without assuming a fixed basis.

.. code-block:: python

    import numpy as np
    from vitalDSP.advanced_computation.emd import EMD

    # Composite signal: 1 Hz + 5 Hz sinusoids with noise
    t = np.linspace(0, 5, 500)
    signal = (np.sin(2 * np.pi * 1.0 * t)
              + 0.5 * np.sin(2 * np.pi * 5.0 * t)
              + 0.1 * np.random.randn(500))

    emd = EMD(signal)
    imfs = emd.emd(max_imfs=4)

    print(f"Number of IMFs extracted: {len(imfs)}")
    for i, imf in enumerate(imfs):
        print(f"  IMF {i + 1}: length={len(imf)}, "
              f"energy={np.sum(imf**2):.3f}")

    # Perfect reconstruction: sum of all IMFs equals the original signal
    reconstructed = np.sum(imfs, axis=0)
    error = np.mean((signal - reconstructed) ** 2)
    print(f"Mean squared reconstruction error: {error:.2e}")

``EMD.emd()`` parameters:

* ``max_imfs`` -- maximum number of IMFs to extract (``None`` = all)
* ``stop_criterion`` -- sifting stop threshold (default ``0.05``)
* ``max_sifting_iterations`` -- per-IMF sifting limit (default ``20``)
* ``max_decomposition_iterations`` -- outer loop limit (default ``10``)

Discrete Wavelet Transform
----------------------------

``WaveletTransform`` implements an undecimated (same-length) DWT using
convolution-based filter banks.

.. code-block:: python

    from vitalDSP.transforms.wavelet_transform import WaveletTransform

    wt = WaveletTransform(signal, wavelet_name="db4")

    # Three-level decomposition
    coeffs = wt.perform_wavelet_transform(level=3)

    # coeffs[0..level-2] are detail arrays, coeffs[-1] is the approximation
    print(f"Coefficients returned: {len(coeffs)} arrays")
    for i, c in enumerate(coeffs[:-1]):
        print(f"  Detail level {i + 1}: length={len(c)}")
    print(f"  Approximation: length={len(coeffs[-1])}")

    # Reconstruct the signal from coefficients
    reconstructed_wt = wt.perform_inverse_wavelet_transform(coeffs)
    print(f"Reconstructed signal length: {len(reconstructed_wt)}")

Supported ``wavelet_name`` values include ``'haar'``, ``'db1'`` through
``'db8'``, ``'sym1'`` through ``'sym8'``, and ``'coif1'`` through
``'coif5'``.

Denoising via Wavelet Coefficient Thresholding
-----------------------------------------------

A common use of the DWT is noise reduction by zeroing small detail
coefficients before reconstruction.

.. code-block:: python

    noisy = signal + 0.5 * np.random.randn(len(signal))

    wt_noisy = WaveletTransform(noisy, wavelet_name="db4")
    coeffs_noisy = wt_noisy.perform_wavelet_transform(level=3)

    # Soft-threshold each detail level
    threshold = 0.3
    thresholded = list(coeffs_noisy)
    for i in range(len(thresholded) - 1):   # skip the approximation
        d = thresholded[i]
        thresholded[i] = np.sign(d) * np.maximum(np.abs(d) - threshold, 0)

    denoised_wt = wt_noisy.perform_inverse_wavelet_transform(thresholded)
    print(f"Denoised signal length: {len(denoised_wt)}")

For a fully automated wavelet-denoising workflow with built-in threshold
selection, see ``ArtifactRemoval.wavelet_denoising()`` demonstrated in
Tutorial 1.

Symbolic Dynamics (Bonus)
--------------------------

Symbolic dynamics converts a continuous time series into a discrete symbol
sequence and measures its complexity.

.. code-block:: python

    from vitalDSP.physiological_features.symbolic_dynamics import SymbolicDynamics

    # RR intervals in milliseconds
    rr_ms = np.array([800, 820, 810, 790, 830, 815, 795, 825, 810, 800,
                       815, 790, 835, 810, 820, 800, 810, 825, 795, 815])

    sd = SymbolicDynamics(rr_ms, n_symbols=4, word_length=3)
    symbols = sd.symbolize()
    print(f"Symbol sequence: {symbols}")

    shannon_result = sd.compute_shannon_entropy()
    print(f"Shannon entropy       : {shannon_result['entropy']:.4f}")
    print(f"Normalised entropy    : {shannon_result['normalized_entropy']:.4f}")

    forbidden_result = sd.detect_forbidden_words()
    print(f"Forbidden word count  : {forbidden_result['n_forbidden']}")
    print(f"Forbidden percentage  : {forbidden_result['forbidden_percentage']:.1f} %")

``SymbolicDynamics`` constructor parameters:

* ``signal`` -- 1-D numpy array (typically RR intervals)
* ``n_symbols`` -- number of discrete symbols (default ``4``)
* ``word_length`` -- pattern length for word analysis (default ``3``)
* ``method`` -- symbolisation scheme: ``'0V'`` (default), ``'quantile'``,
  ``'SAX'``, or ``'threshold'``

Available analysis methods: ``symbolize()``, ``compute_shannon_entropy()``,
``compute_word_distribution()``, ``detect_forbidden_words()``,
``compute_transition_matrix()``, ``compute_renyi_entropy(alpha)``,
``compute_permutation_entropy()``.
