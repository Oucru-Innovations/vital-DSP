VitalDSP Web Application
========================

The VitalDSP web application provides an interactive, browser-based interface for physiological signal processing and analysis. Built with Dash and Plotly, it lets you upload, preprocess, and analyse ECG or PPG signals without writing any code.

Overview
========

The application is organised around a left sidebar with two categories of pages:

**Preprocessing**
  Data upload and signal filtering — the necessary first steps before running any analysis.

**Analysis**
  Three dedicated analysis screens that operate on the preprocessed signal.

A theme toggle button in the sidebar switches between light and dark modes; the selection is persisted across page navigations.

Installation and Setup
======================

.. code-block:: bash

   # Install the package with webapp extras
   pip install vital-DSP[webapp]

   # Start the server
   python -m vitalDSP_webapp.run_webapp

The application is then available at ``http://localhost:8050``.

Navigation
==========

The sidebar is always visible and contains:

+-------------------+------------------------------------------+
| Link              | Destination                              |
+===================+==========================================+
| Introduction      | Welcome page with getting-started guide  |
+-------------------+------------------------------------------+
| Upload Data       | Data upload and column configuration     |
+-------------------+------------------------------------------+
| Filtering         | Signal preprocessing and filtering       |
+-------------------+------------------------------------------+
| Time Domain       | HRV and time-domain feature analysis     |
+-------------------+------------------------------------------+
| Frequency Domain  | Spectral analysis and PSD features       |
+-------------------+------------------------------------------+
| Respiratory Rate  | Multi-method respiratory rate estimation |
+-------------------+------------------------------------------+

Recommended workflow::

  Upload Data → Filtering → (Time Domain | Frequency Domain | Respiratory Rate)

The filtered signal produced on the Filtering screen is automatically passed to all three analysis screens, ensuring consistent results.

Upload Data
===========

The upload screen is the entry point for all analysis workflows.

**Supported Formats**
  * CSV (comma or semicolon separated)
  * Excel (``.xlsx``, ``.xls``)
  * JSON

**Column Configuration**
  After uploading, map your file's columns to:

  * **Time column** — timestamp or sample-index column (optional; the app infers sample index if absent)
  * **Signal column** — the ECG or PPG channel to analyse
  * **Sampling frequency** — in Hz (required for all downstream analysis)

**Signal Preview**
  A scrollable preview plot lets you verify the data looks correct before proceeding.

Filtering (Preprocessing)
=========================

The filtering screen handles all signal conditioning. The cleaned signal is stored globally and forwarded to every analysis screen.

Controls
--------

* **Filter Type** — select the filtering strategy:

  * *Traditional* — Butterworth, Chebyshev, Elliptic, or Bessel filters with configurable passband, stopband, and order
  * *Advanced* — Kalman, adaptive (LMS/RLS), and convolution-based methods
  * *Artifact Removal* — motion artefact suppression, powerline notch, and baseline wander removal
  * *Ensemble* — combines multiple filter outputs for robust conditioning

* **Signal Type** — ECG or PPG; changes the default filter presets automatically
* **Start Position / Window** — navigate to any segment of the recording
* **Apply Filter** button — runs the selected filter and updates the preview

Output
------

After applying a filter the screen shows:

* Original vs filtered signal overlay
* Filter quality metrics (SNR improvement, signal stability)
* Applied filter parameters displayed for reference

Time Domain Analysis
====================

Computes heart-rate variability (HRV) and morphological features from the filtered signal.

**Analysis includes**

* Statistical time-domain HRV metrics: SDNN, RMSSD, pNN50, mean RR, and more
* Peak detection with configurable prominence and distance thresholds
* Tachogram (RR interval series) plot
* Poincaré plot for nonlinear HRV visualisation

**Controls**

* Start Position slider and Window dropdown (30 s – 10 min) to select the analysis segment
* Navigation buttons (⏮ ◀ ▶ ⏭) for quick ±5 % / ±10 % jumps
* Minimum / maximum peak distance inputs

**Export**
  Results can be exported as CSV or JSON from the export panel.

Frequency Domain Analysis
=========================

Provides spectral decomposition of the filtered signal.

**Analysis includes**

* FFT magnitude spectrum
* Welch power spectral density (PSD)
* Spectrogram (short-time Fourier transform)
* Frequency-band power ratios (VLF, LF, HF, LF/HF)

**Controls**

Same Start Position / Window / Navigate controls as the Time Domain screen, plus:

* FFT window function selector (Hann, Hamming, Blackman, …)
* Overlap percentage
* PSD resolution / nperseg setting

**Export**
  Frequency-domain features exportable as CSV or JSON.

Respiratory Rate Analysis
=========================

Estimates respiratory rate (RR) from a PPG or ECG signal using six independent methods running simultaneously. All methods receive the same bandpass-filtered (0.1–0.8 Hz) signal, and an ensemble consensus is computed.

Methods
-------

+-----------------------------------------+------------+-----------------------------------------------------------+
| Method                                  | Category   | Principle                                                 |
+=========================================+============+===========================================================+
| Counting (Peak Detection RR)            | Time       | Counts peaks; computes RR from inter-peak intervals       |
+-----------------------------------------+------------+-----------------------------------------------------------+
| Peak Interval Detection                 | Time       | Detects breath cycles via peak-to-peak intervals          |
+-----------------------------------------+------------+-----------------------------------------------------------+
| Zero-Crossing Detection                 | Time       | Counts positive-going zero crossings as breath markers    |
+-----------------------------------------+------------+-----------------------------------------------------------+
| Time Domain (Autocorrelation)           | Time       | Finds dominant breath period from autocorrelation lag     |
+-----------------------------------------+------------+-----------------------------------------------------------+
| FFT-Based RR                            | Frequency  | Identifies dominant frequency via FFT magnitude spectrum  |
+-----------------------------------------+------------+-----------------------------------------------------------+
| Frequency Domain RR (Welch PSD)         | Frequency  | Estimates dominant respiratory frequency from Welch PSD   |
+-----------------------------------------+------------+-----------------------------------------------------------+

Methods Comparison & Ensemble panel
------------------------------------

Shown at the top of the screen, this panel displays:

* **Consensus (median)** — the ensemble respiratory rate in bpm
* **Mean** — arithmetic mean across valid method estimates
* **Std Dev** — inter-method standard deviation (lower = higher agreement)
* **Confidence** — 0–1 score based on method agreement
* **Quality** — HIGH / MEDIUM / LOW / FAILED rating
* A per-method table with individual estimates and colour-coded domain badges

Per-Method Plots
----------------

Six individual plots are shown below the ensemble panel, one per method:

* **Counting / Peak Interval** — smoothed respiratory signal with detected peaks, cycle shading, and inter-peak interval annotations (green = valid, red = rejected)
* **Zero-Crossing** — signal with inspiration/expiration phase shading and breath-start markers
* **Autocorrelation** — lag-domain plot with the valid respiratory range shaded, dominant period annotated in bpm and seconds
* **FFT-Based / Welch PSD** — spectrum with normal adult range (12–20 bpm) highlighted, dominant peak marked, half-power bandwidth shaded

.. note::

   Time-domain peak and zero-crossing methods operate on a lightly smoothed version of the bandpassed signal (1.25 s moving average) to suppress residual cardiac AM ripple. This is applied only for detection — the displayed raw bandpass trace is shown in the background for reference.

Controls
--------

* **Start Position** slider — select the window start as a percentage of total recording
* **Window** dropdown — 30 s, 1 min, 2 min, 5 min, 10 min
* **Navigate** buttons (⏮ ◀ ▶ ⏭) — ±5 % / ±10 % jumps
* **Min / Max Breath Duration** — user-facing hint inputs (the analysis internally uses the band-derived physiological limits of 1.25–10 s)
* **Run Analysis** button — triggers all six methods and refreshes all plots

Signal Flow
===========

The following diagram summarises how data flows through the application::

   ┌──────────────┐
   │  Upload Data │  CSV / Excel / JSON → column mapping → sampling frequency
   └──────┬───────┘
          │ raw signal
          ▼
   ┌──────────────┐
   │  Filtering   │  bandpass / advanced / artifact removal → filtered signal stored globally
   └──────┬───────┘
          │ filtered signal (shared)
          ├──────────────────┬──────────────────────┐
          ▼                  ▼                      ▼
   ┌─────────────┐  ┌────────────────┐  ┌────────────────────┐
   │ Time Domain │  │Frequency Domain│  │ Respiratory Rate   │
   │   (HRV)     │  │  (Spectral)    │  │  (6 methods +      │
   │             │  │                │  │   ensemble)        │
   └─────────────┘  └────────────────┘  └────────────────────┘

Theme
=====

The sidebar theme toggle button switches the entire application between **Light** and **Dark** mode:

* **Light** — white content area, dark navy sidebar (``#2c3e50 → #34495e``)
* **Dark** — dark grey content area (``#1a1a1a``), deep blue-black sidebar (``#111827 → #1f2937``)

The selected theme is applied immediately via a client-side callback and persisted in a browser store across page navigations.

Export
======

The Time Domain, Frequency Domain, and Respiratory Rate screens each provide export buttons:

* **CSV export** — tabular feature values
* **JSON export** — full structured results including metadata

Troubleshooting
===============

**Application won't start**
  Verify that port 8050 is free and all dependencies are installed (``pip install vital-DSP[webapp]``).

**Signal upload fails**
  Confirm the file is a supported format (CSV / Excel / JSON) and that the delimiter is a comma or semicolon. Check that the signal column contains numeric data.

**Analysis screens show "No data — upload data first"**
  The analysis screens require data to be uploaded on the Upload screen. If data was uploaded but the message persists, reload the page and re-upload.

**Respiratory rate results are inconsistent**
  Ensure a filter has been applied on the Filtering screen before running respiratory analysis. Passing a raw, unfiltered signal yields unreliable results because the respiratory band (0.1–0.8 Hz) will be dominated by noise.

**Peak Interval Detection shows "—"**
  This typically means all detected inter-peak intervals fall outside the physiological range (1.25–10 s). Try a longer window (2 min or more) or verify the signal quality on the Filtering screen.

**Plots appear blank after theme switch**
  Click **Run Analysis** again to regenerate the plots in the new theme.

**Gap between header and sidebar**
  Ensure the browser zoom level is 100 %. At non-standard zoom levels sub-pixel rounding can produce a thin line.
