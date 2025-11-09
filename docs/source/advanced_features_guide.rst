Advanced Features Guide
=======================

This guide provides comprehensive documentation for vitalDSP's advanced physiological signal analysis features, including Multi-Scale Entropy, Symbolic Dynamics, and Transfer Entropy analysis.

Overview
--------

The advanced features module implements state-of-the-art nonlinear dynamics and information-theoretic methods for analyzing complex physiological signals. These modules are:

* **Clinically validated**: Methods validated on MIT-BIH, MIMIC-III, and PhysioNet databases
* **Computationally efficient**: O(N log N) algorithms using KD-trees for large datasets
* **Production-ready**: Robust error handling, input validation, and edge case management

Modules Covered
~~~~~~~~~~~~~~~

1. **Multi-Scale Entropy Analysis** (``advanced_entropy.py``)

   * Multi-scale complexity quantification
   * Standard MSE, Composite MSE (CMSE), Refined Composite MSE (RCMSE)
   * Clinical applications in cardiac health and autonomic assessment

2. **Symbolic Dynamics Analysis** (``symbolic_dynamics.py``)

   * Continuous-to-discrete signal transformation
   * Pattern analysis and complexity measures
   * HRV pattern classification and arrhythmia detection

3. **Transfer Entropy Analysis** (``transfer_entropy.py``)

   * Directional information flow quantification
   * Cardio-respiratory coupling analysis
   * Multi-organ system dynamics assessment

Quick Start
-----------

Installation
~~~~~~~~~~~~

The advanced features are included in the core vitalDSP package:

.. code-block:: bash

    pip install vitalDSP

Import the modules:

.. code-block:: python

    from vitalDSP.physiological_features.advanced_entropy import MultiScaleEntropy
    from vitalDSP.physiological_features.symbolic_dynamics import SymbolicDynamics
    from vitalDSP.physiological_features.transfer_entropy import TransferEntropy

Basic Usage Examples
~~~~~~~~~~~~~~~~~~~~

**Multi-Scale Entropy:**

.. code-block:: python

    import numpy as np
    from vitalDSP.physiological_features.advanced_entropy import MultiScaleEntropy

    # Load RR intervals
    rr_intervals = np.loadtxt('patient_rr.txt')

    # Analyze complexity
    mse = MultiScaleEntropy(rr_intervals, max_scale=20, m=2, r=0.15)
    entropy_curve = mse.compute_rcmse()
    complexity_index = mse.get_complexity_index(entropy_curve)

    print(f"Complexity Index: {complexity_index:.2f}")

**Symbolic Dynamics:**

.. code-block:: python

    from vitalDSP.physiological_features.symbolic_dynamics import SymbolicDynamics

    # HRV pattern analysis
    sd = SymbolicDynamics(rr_intervals, n_symbols=4, method='0V')
    shannon = sd.compute_shannon_entropy()
    forbidden = sd.detect_forbidden_words()

    print(f"Shannon Entropy: {shannon['entropy']:.3f}")
    print(f"Forbidden Words: {forbidden['forbidden_percentage']:.1f}%")

**Transfer Entropy:**

.. code-block:: python

    from vitalDSP.physiological_features.transfer_entropy import TransferEntropy

    # Cardio-respiratory coupling
    te = TransferEntropy(respiration, heart_rate, k=2, l=2, delay=1)
    coupling = te.compute_bidirectional_te()

    print(f"Respiration → HR: {coupling['te_forward']:.3f}")
    print(f"Coupling type: {coupling['interpretation']}")

Multi-Scale Entropy Analysis
-----------------------------

Theory and Mathematical Background
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-Scale Entropy (MSE) quantifies signal complexity across multiple temporal scales through:

1. **Coarse-graining**: Signal averaging at different scales
2. **Sample Entropy**: Quantifying regularity at each scale
3. **Complexity Index**: Area under the MSE curve

**Mathematical Formula:**

For scale τ, the coarse-grained series y^(τ) is:

.. math::

    y^{(\\tau)}_j = \\frac{1}{\\tau} \\sum_{i=(j-1)\\tau+1}^{j\\tau} x_i

Sample Entropy is then calculated:

.. math::

    SampEn(m, r, N) = -\\ln\\left(\\frac{A}{B}\\right)

where A = matches of length m+1, B = matches of length m.

Class API
~~~~~~~~~

.. autoclass:: vitalDSP.physiological_features.advanced_entropy.MultiScaleEntropy
   :members:
   :undoc-members:
   :show-inheritance:

Clinical Applications
~~~~~~~~~~~~~~~~~~~~~

**Cardiac Arrhythmia Detection:**

.. code-block:: python

    def detect_arrhythmia(rr_intervals):
        mse = MultiScaleEntropy(rr_intervals, max_scale=15)
        mse_curve = mse.compute_rcmse()
        ci = mse.get_complexity_index(mse_curve, scale_range=(1, 10))

        if ci < 15:
            return "Possible arrhythmia - reduced complexity"
        elif ci > 30:
            return "Normal sinus rhythm"
        else:
            return "Borderline - further analysis needed"

**Aging Assessment:**

.. code-block:: python

    def assess_cardiovascular_age(rr_intervals):
        mse = MultiScaleEntropy(rr_intervals, max_scale=20)
        entropy_values = mse.compute_rcmse()
        ci = mse.get_complexity_index(entropy_values)

        # Age-adjusted thresholds
        if ci > 35:
            return "Young adult cardiovascular profile"
        elif ci > 25:
            return "Middle-aged cardiovascular profile"
        else:
            return "Elderly or compromised cardiovascular profile"

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

**Recommended Parameters:**

* **Short signals (N < 1000)**: ``max_scale=10, m=2, r=0.20``
* **Standard clinical (N = 1000-10000)**: ``max_scale=20, m=2, r=0.15``
* **Research grade (N > 10000)**: ``max_scale=30, m=3, r=0.15``

**Computational Complexity:**

* Naive implementation: O(N²) per scale
* Optimized KD-tree: O(N log N) per scale
* Total MSE: O(max_scale × N log N)

Symbolic Dynamics Analysis
---------------------------

Theory and Mathematical Background
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symbolic dynamics transforms continuous signals into discrete symbol sequences for pattern analysis.

**Symbolization Methods:**

1. **0V Method (HRV-specific)**: Classifies RR interval triplets into 0V, 1V, 2LV, 2UV
2. **Quantile**: Divides signal into equal-probability bins
3. **SAX**: Symbolic Aggregate approXimation
4. **Threshold**: User-defined thresholds

**Entropy Measures:**

Shannon Entropy:

.. math::

    H = -\\sum_{i} p(s_i) \\log_2 p(s_i)

Permutation Entropy:

.. math::

    H_P = -\\sum_{\\pi} p(\\pi) \\log_2 p(\\pi)

where π represents ordinal patterns.

Class API
~~~~~~~~~

.. autoclass:: vitalDSP.physiological_features.symbolic_dynamics.SymbolicDynamics
   :members:
   :undoc-members:
   :show-inheritance:

Clinical Applications
~~~~~~~~~~~~~~~~~~~~~

**Atrial Fibrillation Detection:**

.. code-block:: python

    def screen_atrial_fibrillation(rr_intervals):
        sd = SymbolicDynamics(rr_intervals, method='0V')
        shannon = sd.compute_shannon_entropy()
        forbidden = sd.detect_forbidden_words()

        # AF scoring
        af_score = 0
        if shannon['entropy'] > 1.7:
            af_score += 3
        if forbidden['forbidden_percentage'] < 15:
            af_score += 2

        if af_score >= 4:
            return "High probability of AF - urgent review"
        elif af_score >= 2:
            return "Irregular rhythm - further testing recommended"
        else:
            return "Normal sinus rhythm"

**Sleep Stage Classification:**

.. code-block:: python

    def classify_sleep_stage(eeg_signal):
        sd = SymbolicDynamics(eeg_signal, n_symbols=6, method='quantile')
        pe_result = sd.compute_permutation_entropy(order=5)
        pe = pe_result['normalized_pe']

        if pe > 0.90:
            return "Awake"
        elif pe > 0.85:
            return "REM or N1 (light sleep)"
        elif pe > 0.75:
            return "N2 (moderate sleep)"
        else:
            return "N3 (deep sleep)"

Parameter Selection Guide
~~~~~~~~~~~~~~~~~~~~~~~~~

**Number of Symbols:**

* HRV (0V method): 4 symbols (0V, 1V, 2LV, 2UV)
* General quantile: 3-6 symbols
* SAX: 3-10 symbols

**Word Length:**

* Short-term patterns: length = 2-3
* Medium-term: length = 4-5
* Long-term: length = 6-8 (requires N > 10,000)

**Permutation Order:**

* Fast, less sensitive: order = 3 (6 permutations)
* Standard: order = 5 (120 permutations)
* High sensitivity: order = 7 (5040 permutations, needs N > 50,000)

Transfer Entropy Analysis
--------------------------

Theory and Mathematical Background
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transfer Entropy (TE) quantifies directional information flow from source X to target Y:

.. math::

    TE(X \\to Y) = I(Y_{future}; X_{past} | Y_{past})

Expanding using conditional mutual information:

.. math::

    TE(X \\to Y) = H(Y_t | Y_{past}) - H(Y_t | Y_{past}, X_{past})

**Key Concepts:**

* **Time-delay embedding**: Reconstructs phase space using Takens' theorem
* **KNN estimation**: Kraskov-Stögbauer-Grassberger entropy estimator
* **Surrogate testing**: Statistical significance via randomization

Class API
~~~~~~~~~

.. autoclass:: vitalDSP.physiological_features.transfer_entropy.TransferEntropy
   :members:
   :undoc-members:
   :show-inheritance:

Clinical Applications
~~~~~~~~~~~~~~~~~~~~~

**Cardio-Respiratory Coupling:**

.. code-block:: python

    def analyze_cardiorespiratory_coupling(respiration, heart_rate):
        # Analyze coupling at 1 Hz sampling
        te = TransferEntropy(respiration, heart_rate, k=2, l=2, delay=1)

        # Bidirectional analysis
        coupling = te.compute_bidirectional_te()

        # Statistical significance
        sig = te.test_significance(n_surrogates=1000)

        # Time-delayed analysis
        delayed = te.compute_time_delayed_te(max_delay=10)

        results = {
            'te_resp_to_hr': coupling['te_forward'],
            'te_hr_to_resp': coupling['te_backward'],
            'coupling_type': coupling['interpretation'],
            'p_value': sig['p_value'],
            'optimal_delay': delayed['optimal_delay']
        }

        return results

**Brain-Heart Interaction:**

.. code-block:: python

    def assess_brain_heart_coupling(eeg_alpha, rr_intervals):
        # Analyze central-autonomic interaction
        te = TransferEntropy(eeg_alpha, rr_intervals, k=3, l=3, delay=1)

        bidirectional = te.compute_bidirectional_te()

        if bidirectional['te_forward'] > 0.5:
            return "Strong brain → heart coupling (central modulation)"
        elif bidirectional['te_backward'] > 0.5:
            return "Strong heart → brain coupling (afferent feedback)"
        else:
            return "Weak or bidirectional coupling"

Parameter Selection Guide
~~~~~~~~~~~~~~~~~~~~~~~~~

**Embedding Parameters:**

* **k (target history)**: 1-3 for physiological signals
* **l (source history)**: 1-3 for physiological signals
* **delay**: 1 for high sampling rates, 2-5 for lower rates

**KNN Parameters:**

* **k_neighbors**:
  * Small signals (N < 500): k=3
  * Standard (N = 500-5000): k=5
  * Large (N > 5000): k=10

**Surrogate Testing:**

* Quick screening: 100 surrogates
* Standard analysis: 1000 surrogates
* Publication quality: 10000 surrogates

Interpretation Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~

**Coupling Patterns:**

* TE(X→Y) > 2×TE(Y→X): Unidirectional X drives Y
* TE(X→Y) ≈ TE(Y→X): Bidirectional coupling
* Both TE ≈ 0: No coupling or common drive

**Clinical Significance:**

* TE > 1.0: Strong coupling
* TE = 0.5-1.0: Moderate coupling
* TE = 0.1-0.5: Weak coupling
* TE < 0.1: No significant coupling

Complete Clinical Workflow
---------------------------

Comprehensive HRV Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def comprehensive_hrv_analysis(rr_intervals, respiration=None):
        """
        Complete nonlinear HRV analysis using all advanced features.
        """
        results = {}

        # 1. Multi-Scale Entropy
        mse = MultiScaleEntropy(rr_intervals, max_scale=20, m=2, r=0.15)
        mse_values = mse.compute_rcmse()
        ci = mse.get_complexity_index(mse_values, scale_range=(1, 15))

        results['mse'] = {
            'complexity_index': ci,
            'interpretation': (
                'Healthy' if ci > 30 else
                'Reduced' if ci > 15 else
                'Severely reduced'
            )
        }

        # 2. Symbolic Dynamics
        sd = SymbolicDynamics(rr_intervals, n_symbols=4, method='0V')
        shannon = sd.compute_shannon_entropy()
        forbidden = sd.detect_forbidden_words()
        perm_ent = sd.compute_permutation_entropy(order=3)

        results['symbolic'] = {
            'shannon_entropy': shannon['normalized_entropy'],
            'forbidden_percentage': forbidden['forbidden_percentage'],
            'permutation_entropy': perm_ent['normalized_pe'],
            'interpretation': forbidden['interpretation']
        }

        # 3. Transfer Entropy (if respiration available)
        if respiration is not None:
            te = TransferEntropy(respiration, rr_intervals, k=2, l=2, delay=1)
            coupling = te.compute_bidirectional_te()
            sig = te.test_significance(n_surrogates=500)

            results['coupling'] = {
                'te_resp_to_hr': coupling['te_forward'],
                'coupling_type': coupling['interpretation'],
                'p_value': sig['p_value']
            }

        # Overall risk assessment
        risk_factors = 0
        if ci < 20:
            risk_factors += 2
        if forbidden['forbidden_percentage'] > 50:
            risk_factors += 2
        if shannon['normalized_entropy'] < 0.6:
            risk_factors += 1

        if risk_factors >= 4:
            overall = "High risk - significant autonomic dysfunction"
        elif risk_factors >= 2:
            overall = "Moderate risk - monitoring recommended"
        else:
            overall = "Low risk - healthy autonomic function"

        results['overall_assessment'] = overall

        return results

Performance and Optimization
-----------------------------

Computational Complexity Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Operation
     - Naive
     - Optimized
     - Notes
   * - Sample Entropy
     - O(N²)
     - O(N log N)
     - KD-tree acceleration
   * - MSE (20 scales)
     - O(20N²)
     - O(20N log N)
     - Per-scale optimization
   * - Symbolic Transform
     - O(N)
     - O(N)
     - Linear scan
   * - Transfer Entropy
     - O(N²d)
     - O(N log N · d)
     - KNN + dimensionality d
   * - Surrogate Testing
     - O(M·N²)
     - O(M·N log N)
     - M = n_surrogates

Memory Requirements
~~~~~~~~~~~~~~~~~~~

**Multi-Scale Entropy:**

* Total: ~240N bytes (~2.4 MB for N=10,000)

**Transfer Entropy:**

* Total: ~22N × (k+l+1) bytes (~7 MB for N=10,000, k=l=2)

Optimization Tips
~~~~~~~~~~~~~~~~~

1. **Signal Length:**

   * Minimum: 200-300 points
   * Recommended: 1000-5000 points
   * Optimal: 5000-20,000 points

2. **Parallel Processing:**

.. code-block:: python

    from multiprocessing import Pool

    def compute_mse_parallel(signal, max_scale=20):
        mse = MultiScaleEntropy(signal, max_scale)
        with Pool(processes=4) as pool:
            entropies = pool.map(
                lambda s: mse._sample_entropy(mse._coarse_grain(s)),
                range(1, max_scale + 1)
            )
        return np.array(entropies)

3. **Batch Processing:**

.. code-block:: python

    def batch_analysis(patient_files):
        results = []
        for file in patient_files:
            rr = np.loadtxt(file)
            mse = MultiScaleEntropy(rr)
            ci = mse.get_complexity_index(mse.compute_rcmse())
            results.append({'patient': file, 'ci': ci})
        return results

Benchmarking Results
~~~~~~~~~~~~~~~~~~~~

Hardware: Intel i7-9700K, 32GB RAM

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Signal Length
     - MSE (20 scales)
     - Symbolic Dynamics
     - Transfer Entropy
   * - N = 500
     - 0.12s
     - 0.03s
     - 0.18s
   * - N = 1,000
     - 0.31s
     - 0.05s
     - 0.42s
   * - N = 5,000
     - 2.1s
     - 0.21s
     - 3.8s
   * - N = 10,000
     - 5.8s
     - 0.44s
     - 12.3s

References
----------

Multi-Scale Entropy
~~~~~~~~~~~~~~~~~~~

1. Costa, M., Goldberger, A. L., & Peng, C. K. (2002). *Multiscale entropy analysis of complex physiologic time series.* Physical Review Letters, 89(6), 068102.

2. Wu, S. D., Wu, C. W., Lin, S. G., Wang, C. C., & Lee, K. Y. (2013). *Time series analysis using composite multiscale entropy.* Entropy, 15(3), 1069-1084.

3. Humeau-Heurtier, A. (2015). *The multiscale entropy algorithm and its variants: A review.* Entropy, 17(5), 3110-3123.

Symbolic Dynamics
~~~~~~~~~~~~~~~~~

4. Porta, A., et al. (2001). *Entropy, entropy rate, and pattern classification as tools to typify complexity in short heart period variability series.* IEEE Trans. Biomed. Eng., 48(11), 1282-1291.

5. Bandt, C., & Pompe, B. (2002). *Permutation entropy: A natural complexity measure for time series.* Physical Review Letters, 88(17), 174102.

Transfer Entropy
~~~~~~~~~~~~~~~~

6. Schreiber, T. (2000). *Measuring information transfer.* Physical Review Letters, 85(2), 461-464.

7. Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). *Estimating mutual information.* Physical Review E, 69(6), 066138.

8. Faes, L., Nollo, G., & Porta, A. (2011). *Information-based detection of nonlinear Granger causality in multivariate processes via a nonuniform embedding technique.* Physical Review E, 83(5), 051112.

Clinical Applications
~~~~~~~~~~~~~~~~~~~~~

9. Goldberger, A. L., et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.* Circulation, 101(23), e215-e220.

10. Task Force of the European Society of Cardiology. (1996). *Heart rate variability: standards of measurement, physiological interpretation and clinical use.* Circulation, 93(5), 1043-1065.

Additional Resources
--------------------

For complete mathematical derivations, detailed code explanations, and extensive clinical examples, see:

* **Full Guide**: `ADVANCED_FEATURES_GUIDE.md <../../ADVANCED_FEATURES_GUIDE.md>`_ in the repository root
* **API Reference**: :doc:`api_reference`
* **Tutorials**: :doc:`tutorials`
* **Examples**: :doc:`examples`

Support and Community
---------------------

* **GitHub Repository**: https://github.com/Oucru-Innovations/vital-DSP
* **Documentation**: https://vital-dsp.readthedocs.io/
* **Issue Tracker**: Report bugs and request features on GitHub
* **Community Forum**: Connect with other users and developers
