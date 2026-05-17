# Changelog

All notable changes to vitalDSP are documented in this file.

---

## [0.2.5] - 2026-05-17

This release delivers a significant redesign and cleanup of the **vitalDSP web application**, improving accuracy of respiratory rate estimation, consolidating the page structure, and polishing the UI.

### Web Application — New Features and Improvements

#### Respiratory Rate Analysis

- Fixed double-preprocessing bug: the already-bandpass-filtered signal was being re-preprocessed internally by `RespiratoryAnalysis`, causing inconsistent results across methods. Resolved by passing `PreprocessConfig(filter_type="ignore", noise_reduction_method="ignore")` to skip internal preprocessing.
- Fixed counting and zero-crossing methods returning inflated rates (~2×) caused by residual cardiac AM ripple. A 1.25 s moving-average is now applied before peak/zero-crossing detection.
- Fixed peak interval method returning no result (`—`) by calling `compute_respiratory_rate()` directly with physiologically correct duration bounds (1.25–10 s from the 0.1–0.8 Hz band), bypassing the ensemble's hardcoded 0.5–6 s limits.
- All 6 methods (counting, FFT-based, frequency domain, time domain, peaks, zero-crossing) now run simultaneously and agree within ±1 bpm on clean signals.
- Enhanced per-method plots: breath-cycle shading, inter-peak interval annotations, valid/rejected cycle colour coding, normal adult range (12–20 bpm) bands on spectral plots, autocorrelation lag shading, inspiration/expiration phase shading on zero-crossing plot.
- Added Methods Comparison & Ensemble panel at the top of the page showing consensus, mean, std dev, confidence, and quality rating.
- Removed Agreement (±2 bpm) card from the summary panel.

#### Sidebar and Navigation

- Restructured sidebar into two categories: **Preprocessing** (Upload Data, Filtering) and **Analysis** (Time Domain, Frequency Domain, Respiratory Rate).
- Added Introduction page link at the top of the sidebar (maps to `/preview`).
- Category headers styled as small uppercase labels with letter-spacing for readability.
- Removed Pipeline and Background Tasks pages (and all associated callbacks).

#### Theme

- Sidebar now fully responds to the light/dark theme toggle: light mode uses the existing dark-navy gradient; dark mode switches to a deep blue-black gradient (`#111827 → #1f2937`).
- All sidebar nav links, hover states, and section headers use CSS variables that update with the theme.
- Fixed header/sidebar gap caused by a `60 px` inline header height mismatching the `4 rem` (64 px) CSS variable.

### Web Application — Removed Pages and Callbacks

Removed unused pages and their associated callbacks, tests, and export functions:

- Pages: Advanced, Physiological Analysis, Pipeline, Quality, Background Tasks, Transforms
- Analysis callbacks: `advanced_callbacks`, `enhanced_filtering_callbacks`, `health_report_callbacks`, `pipeline_callbacks`, `quality_callbacks`, `settings_callbacks`, `tasks_callbacks`, `transform_callbacks`
- Feature callbacks: `features_callbacks`, `physiological_callbacks`, `features/respiratory_callbacks`
- 35 orphaned test files that imported the deleted modules

### Documentation

- Rewrote `docs/source/webapp.rst` to match the current application structure: updated navigation table, signal flow diagram, per-screen descriptions, theme section, and troubleshooting entries.

---

## [0.2.2] - 2026-03-04

This release is a focused quality and correctness release for the `vitalDSP` core library. It resolves **38 bugs and algorithm errors** identified in a two-agent independent code review, covering crashes, wrong numerical results, silent failures, and API inconsistencies across all core modules. No new features are introduced; all changes are fixes and robustness improvements.

### Breaking Changes

- **`respiratory_analysis.py`** — `compute_respiratory_rate()` now returns a `dict` (`{"respiratory_rate": float, "method": str}`) instead of a plain `float`. Update callers: `rr = result["respiratory_rate"]`. The class docstring had already documented dict-style access; the implementation now matches.
- **`ecg_ppg_synchronization_features.py`** — `compute_emd()` renamed to `compute_ppg_rise_time()`. The original label "electromechanical delay" was physiologically incorrect; the method computes PPG systolic rise time (time from PPG foot to systolic peak). A deprecated backward-compatible alias `compute_emd()` is retained with a `DeprecationWarning`.

---

### Bug Fixes — Crashes and Division by Zero

- **`signal_filtering.py`** — Removed `np.asmatrix` (deprecated NumPy 1.25, removed in 2.x) from the custom Savitzky-Golay implementation. Replaced with `np.array` / `np.linalg.pinv(b)[0]`. Previously raised `AttributeError` on NumPy >= 2.0.
- **`ppg_light_features.py`** — Fixed `ZeroDivisionError` in SpO2 calculation when the IR AC component is zero (flat IR segment). The segment is now skipped.
- **`nonlinear.py`** — Fixed division by zero in recurrence quantification analysis (Det / Laminarity) when `np.sum(recurrences) == 0`.
- **`common.py`** — Fixed division by zero in `pearsonr()` for constant signals (`std = 0`). Returns `0.0`.
- **`signal_quality_index.py`** — Fixed `baseline_wander_sqi()` returning NaN when `moving_avg_window >= len(segment)` (`np.convolve mode='valid'` returns empty). Returns `1.0` in that edge case.
- **`signal_quality_index.py`** — Fixed division by zero in waveform template similarity when segment or reference waveform is all-zeros.
- **`beats_transformation.py`** — Fixed silent NaN propagation in `remove_invalid_rr_intervals()` when all RR intervals fail physiological bounds. Now returns early.
- **`time_domain.py`** — Fixed `compute_cvnn()` returning `NaN` for a single NN interval (sample std with `ddof=1` is undefined for N=1). Returns `0.0`.

---

### Bug Fixes — Wrong Numerical Results

- **`non_linear_analysis.py`** — Fixed Lyapunov exponent calculation. The previous implementation measured 1D amplitude drift from the current sample, giving positive "chaos" values for any monotone signal. Replaced with the Rosenstein algorithm: phase-space embedding (dim=2, tau=1), Theiler window exclusion, nearest-neighbor divergence tracking, and linear regression slope as the LLE estimate.
- **`common.py`** — Fixed Granger causality (`grangercausalitytests()`). The previous implementation compared two regressions with different response variables, which is mathematically meaningless. Rebuilt with the correct restricted AR model (`y ~ lagged_y`) and unrestricted ARX model (`y ~ lagged_y + lagged_x`); F-statistic from `(SSR_r - SSR_u) / lag / (SSR_u / df)`.
- **`common.py`** — Fixed DTW windowed distance (`dtw_distance_windowed()`). The first row and column of the DP matrix were left as `inf`, blocking any optimal alignment starting at `(0, j > 0)`. Now correctly initialized within the Sakoe-Chiba band.
- **`advanced_entropy.py`** — Fixed Refined Composite Multiscale Entropy (RCMSE). Previously concatenated all coarse-grained series into one flat array before computing entropy, violating temporal ordering. Now correctly pools raw match counts `A_k` and `B_k` across all coarse-grained series per Wu et al. (2014): `SampEn = -ln(sum(A_k) / sum(B_k))`.
- **`nonlinear.py`** — Fixed `compute_sample_entropy()` returning `0` for degenerate cases (no template matches), which falsely implies perfect regularity. Now returns `NaN` when `phi_m == 0` and `Inf` when `phi_m1 == 0`.
- **`signal_quality.py`** — Fixed PSNR using `np.max(signal)` instead of `np.max(np.abs(signal))`. For zero-centered or negative signals this could be zero or negative. Added `inf` guard for all-zero reference signals.
- **`ppg_autonomic_features.py`** — Fixed Higuchi Fractal Dimension normalization: `(N - m) / k` is now computed with integer floor division per the original Higuchi (1988) algorithm.
- **`anomaly_detection.py`** — Fixed LOF fallback phase-space construction: random sample indices are now sorted before extracting the sub-signal, ensuring `(x_t, x_{t+1})` delay embedding uses temporally adjacent points. LOF threshold tightened from 1.3 to 1.5 (standard recommendation for physiological signals to reduce false positives from normal HRV).
- **`frequency_domain.py`** — Fixed Welch PSD for short NN interval series: `nperseg` is now `max(4, N // 4)` instead of `N`, enabling variance reduction through segment averaging.
- **`signal_quality_index.py`** — Fixed entropy SQI normalization: now divides by `log2(n_bins)` (= `log2(10)`) instead of `log2(len(segment))`, giving consistent SQI values across different window sizes.
- **`waveform.py`** — Fixed off-by-one in `detect_q_session()`: `p_peaks` was not trimmed alongside `q_valleys[1:]` and `r_peaks[1:]`, shifting all Q-session beat boundaries by one beat.

---

### Bug Fixes — Silent Failures and Missing Warnings

- **`signal_filtering.py`** — Fixed silent ripple clamping in Chebyshev I and Elliptic filters: `ripple < 1 dB` was silently replaced with `0.5 dB`. Now emits `UserWarning`.
- **`preprocess_operations.py`** — Fixed silent `lowcut` parameter being ignored for lowpass-only filter types (`'butterworth'`, `'chebyshev'`, `'elliptic'`). Now emits `UserWarning`.
- **`preprocess_operations.py`** — Fixed incorrect fallback for failed Butterworth filter: z-score normalization was applied (changes amplitude statistics, not frequencies). Now returns the unfiltered original signal with a `UserWarning`.
- **`preprocess_operations.py`** — `estimate_baseline()` now raises `ValueError` for `fs <= 0` or `window_size <= 0`.
- **`respiratory_analysis.py`** — Fixed spurious zero-crossings in the breath counting method: samples exactly equal to `0.0` produced `±1` sign differences instead of `±2`, creating false breath events. Now uses `np.sign(signal + 1e-10)`.
- **`respiratory_analysis.py`** — Fixed negative breath intervals from linear extrapolation at signal boundaries. Interpolated outlier intervals are clipped to `[0.5, 6.0]` seconds (physiological respiratory range).
- **`frequency_domain_rr.py`** — Added `UserWarning` when the input signal is too short for the target frequency resolution and `nperseg` is silently clamped.
- **`wavelet_transform.py`** — Continuous wavelets (Morlet, Mexican Hat) now raise `ValueError` instead of silently falling back to a trivial `[1, -1]` filter producing meaningless results.

---

### Bug Fixes — Filtering and Signal Processing

- **`adaptive_snr_estimation.py`** — Fixed inconsistent EMA alpha convention: `avg_mean` used `alpha * old + (1-alpha) * new` (slow adaptation for large alpha) while `noise_estimate` used the opposite convention. Both now use `alpha * old + (1 - alpha) * new`. Docstring updated.
- **`artifact_removal.py`** — Fixed asymmetric padding in median filter for even kernel sizes: `(k//2, k//2)` corrected to `(k//2, k - 1 - k//2)`.
- **`artifact_removal.py`** — Fixed wavelet denoising: `wavelet.db(order)` returns a `(low_pass, high_pass)` tuple; previously the full tuple was incorrectly used as the low-pass filter, producing a nonsensical 2-element filter.
- **`artifact_removal.py`** — Fixed wavelet reconstruction boundary mismatch for odd-length signals.
- **`advanced_signal_filtering.py`** — Fixed boosting ensemble: the loop previously filtered `self.signal` on every iteration (making all iterations identical). Now correctly filters the residual signal at each step.
- **`noise_reduction.py`** — Fixed Donoho universal threshold: sigma was estimated separately per wavelet level (over-thresholding at coarser scales). Now estimated once from the finest-scale detail coefficients: `sigma = median(|d_0|) / 0.6745`.

---

### Robustness Improvements

- **`nonlinear.py`** — Added epsilon guard in `compute_lyapunov_exponent()`: returns `0.0` immediately for near-constant signals where `epsilon < 1e-10`.
- **`waveform.py`** — Added fallback in ECG session boundary backward search: if the loop exits at signal origin, uses `max(0, p - 0.1 * fs)` as the minimum start index.
- **`vital_transformation.py`** — Dynamic filter order reduction now clamped to `max(1, filter_order)` to prevent `filtfilt` crash on very short signals.
- **`fourier_transform.py`** — Signal validator now rejects NaN and Inf inputs (`allow_nan=False, allow_inf=False`).
- **`signal_quality_index.py`** — Added length validation between `segment` and `reference_waveform` in template similarity; mismatched lengths now raise `ValueError`.
- **`wavelet_transform.py`** — Implementation documented as Undecimated/Stationary Wavelet Transform (USWT/SWT) throughout all relevant docstrings.
- **`signal_filtering.py`** — Removed stale commented-out Nyquist normalization code.

---

### Test Suite

- All fixes verified: **4007 tests passing, 43 skipped, 0 failures** (Python 3.13, NumPy < 2.0).
- Updated tests in `test_anomaly_detection.py`, `test_non_linear_analysis_comprehensive.py`, `test_ecg_ppg_synchronization_features.py`, `test_others.py`, `test_common.py`, and `test_time_domain.py` to reflect corrected APIs and behavior.

---

## [0.2.1] - 2026-01-15

See [GitHub release notes for v0.2.1](https://github.com/Oucru-Innovations/vital-DSP/releases/tag/v0.2.1).
