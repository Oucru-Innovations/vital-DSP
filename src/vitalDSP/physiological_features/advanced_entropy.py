"""
Advanced Entropy Analysis Module
=================================

This module provides advanced entropy-based complexity measures for physiological
signal analysis, including:

1. Multi-Scale Entropy (MSE) - Costa et al. (2002)
2. Composite Multi-Scale Entropy (CMSE) - Wu et al. (2013)
3. Refined Composite Multi-Scale Entropy (RCMSE) - Wu et al. (2014)
4. Multi-Scale Sample Entropy (MSSE)
5. Multi-Scale Fuzzy Entropy (MFE)

These methods analyze signal complexity across multiple time scales, providing
insights into the multi-scale structure of physiological signals.

Clinical Applications:
---------------------
- Cardiac arrhythmia detection and classification
- Aging assessment and cardiovascular health
- Autonomic nervous system function evaluation
- Disease progression monitoring (heart failure, diabetes)
- Sleep stage classification
- Seizure prediction and epilepsy monitoring

Mathematical Background:
-----------------------
Multi-scale entropy extends traditional entropy measures by analyzing the signal
at multiple temporal scales through a coarse-graining procedure. This reveals
complexity at different time scales, which is crucial for understanding
physiological regulation mechanisms.

References:
----------
1. Costa, M., Goldberger, A. L., & Peng, C. K. (2002). Multiscale entropy analysis
   of complex physiologic time series. Physical review letters, 89(6), 068102.

2. Wu, S. D., Wu, C. W., Lin, S. G., Wang, C. C., & Lee, K. Y. (2013). Time series
   analysis using composite multiscale entropy. Entropy, 15(3), 1069-1084.

3. Wu, S. D., Wu, C. W., Lin, S. G., Lee, K. Y., & Peng, C. K. (2014). Analysis of
   complex time series using refined composite multiscale entropy. Physics Letters A,
   378(20), 1369-1374.

4. Ahmed, M. U., & Mandic, D. P. (2011). Multivariate multiscale entropy: A tool for
   complexity analysis of multichannel data. Physical Review E, 84(6), 061918.

Author: Claude (Sonnet 4.5)
Date: October 10, 2025
Version: 1.0
"""

"""
Physiological Features Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations
- SciPy integration for advanced signal processing
- Interactive visualization capabilities

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.physiological_features.advanced_entropy import AdvancedEntropy
    >>> signal = np.random.randn(1000)
    >>> processor = AdvancedEntropy(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""


import numpy as np
from scipy.spatial import cKDTree
from scipy.special import gamma
import warnings
from typing import Tuple, List, Optional, Union


class MultiScaleEntropy:
    """
    Multi-Scale Entropy (MSE) analysis for physiological signals.

    MSE quantifies the complexity of a signal across multiple temporal scales
    through coarse-graining followed by entropy calculation at each scale.

    The method reveals how signal complexity changes with scale, providing
    insights into the multi-scale regulatory mechanisms of physiological systems.

    Parameters
    ----------
    signal : numpy.ndarray
        Input time series signal (1D array)
    max_scale : int, optional
        Maximum scale factor for coarse-graining (default: 20)
        Recommended: 20 for HRV analysis, 10-15 for shorter signals
    m : int, optional
        Embedding dimension (pattern length) for entropy calculation (default: 2)
        Typically m=2 for physiological signals
    r : float, optional
        Tolerance for pattern matching (default: 0.15)
        Expressed as fraction of signal standard deviation
        Recommended: 0.15-0.25 for physiological signals
    fuzzy : bool, optional
        Use fuzzy membership functions instead of binary matching (default: False)
        Fuzzy entropy is more stable for short signals

    Attributes
    ----------
    signal : numpy.ndarray
        Original input signal
    max_scale : int
        Maximum scale for analysis
    m : int
        Embedding dimension
    r : float
        Tolerance (absolute value)
    fuzzy : bool
        Whether to use fuzzy entropy

    Methods
    -------
    compute_mse()
        Compute Multi-Scale Entropy across all scales
    compute_cmse()
        Compute Composite Multi-Scale Entropy (improved stability)
    compute_rcmse()
        Compute Refined Composite Multi-Scale Entropy (best stability)
    get_complexity_index()
        Calculate complexity index (area under MSE curve)

    Examples
    --------
    >>> # Analyze heart rate variability
    >>> import numpy as np
    >>> from vitalDSP.physiological_features.advanced_entropy import MultiScaleEntropy
    >>>
    >>> # Generate synthetic HRV signal (RR intervals in seconds)
    >>> np.random.seed(42)
    >>> rr_intervals = 1.0 + 0.05 * np.random.randn(1000)  # 60 BPM baseline
    >>>
    >>> # Compute MSE
    >>> mse = MultiScaleEntropy(rr_intervals, max_scale=20, m=2, r=0.15)
    >>> entropy_values = mse.compute_mse()
    >>>
    >>> # Get complexity index
    >>> ci = mse.get_complexity_index(entropy_values)
    >>> print(f"Complexity Index: {ci:.4f}")
    >>>
    >>> # Compare young vs elderly (example)
    >>> # Young: Higher complexity at multiple scales
    >>> # Elderly: Reduced complexity, flatter MSE curve

    Notes
    -----
    **Interpretation Guidelines:**

    - **Healthy/Young:** MSE values remain high or increase at larger scales
      indicating rich multi-scale complexity

    - **Disease/Aging:** MSE values decrease more rapidly with scale,
      indicating loss of complexity and adaptive capacity

    - **Scale-Specific Information:**
        - Scales 1-4: Short-term dynamics (seconds to minutes)
        - Scales 5-10: Mid-term dynamics (minutes to tens of minutes)
        - Scales 10-20: Long-term dynamics (tens of minutes to hours)

    **Signal Length Requirements:**
    - Minimum: 100 * scale samples for reliable estimation
    - Recommended: 500-1000+ samples for max_scale=20
    - Shorter signals: Use smaller max_scale or CMSE/RCMSE variants

    **Parameter Selection:**
    - m=2: Standard for most physiological signals
    - m=3: For signals requiring more detailed patterns
    - r=0.15: Conservative choice (good specificity)
    - r=0.20-0.25: More lenient (better for noisy signals)
    """

    def __init__(
        self,
        signal: np.ndarray,
        max_scale: int = 20,
        m: int = 2,
        r: float = 0.15,
        fuzzy: bool = False,
    ):
        """
        Initialize Multi-Scale Entropy analyzer.

        Parameters
        ----------
        signal : numpy.ndarray
            Input time series (1D)
        max_scale : int
            Maximum coarse-graining scale
        m : int
            Embedding dimension
        r : float
            Tolerance (fraction of std)
        fuzzy : bool
            Use fuzzy entropy
        """
        # Input validation
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)

        if len(signal) < 10:
            raise ValueError(
                f"Signal too short ({len(signal)} samples). Minimum: 10 samples."
            )

        if max_scale < 1:
            raise ValueError(f"max_scale must be >= 1, got {max_scale}")

        if m < 1 or m > 10:
            raise ValueError(f"Embedding dimension m must be 1-10, got {m}")

        if r < 0 or r > 1:
            raise ValueError(f"Tolerance r must be 0-1 (fraction of std), got {r}")

        # Store parameters
        self.signal = signal
        self.max_scale = max_scale
        self.m = m
        self.r = r * np.std(signal)  # Convert to absolute tolerance
        self.fuzzy = fuzzy

        # Warn if signal might be too short
        min_recommended_length = 100 * max_scale
        if len(signal) < min_recommended_length:
            warnings.warn(
                f"Signal length ({len(signal)}) is less than recommended "
                f"({min_recommended_length} for scale {max_scale}). "
                f"Consider reducing max_scale or using CMSE/RCMSE for better stability.",
                UserWarning,
            )

    def _coarse_grain(self, scale: int, start_index: int = 0) -> np.ndarray:
        """
        Perform coarse-graining operation on the signal.

        Coarse-graining averages consecutive non-overlapping windows of length
        'scale' to create a new time series at the specified temporal scale.

        Parameters
        ----------
        scale : int
            Scale factor (window size for averaging)
        start_index : int, optional
            Starting index for coarse-graining (default: 0)
            Used in composite methods to create multiple coarse-grained series

        Returns
        -------
        coarse_signal : numpy.ndarray
            Coarse-grained time series

        Mathematical Definition:
        -----------------------
        For scale τ, the coarse-grained series y^(τ) is:

        y^(τ)_j = (1/τ) * Σ(i=(j-1)τ+1 to jτ) x_i

        where j = 1, 2, ..., N/τ

        Examples:
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> coarse = _coarse_grain(signal, scale=2)
        >>> # Result: [1.5, 3.5, 5.5, 7.5, 9.5]
        """
        n = len(self.signal)

        # Calculate number of complete windows
        n_windows = (n - start_index) // scale

        if n_windows < 1:
            raise ValueError(
                f"Signal too short for scale {scale} with start_index {start_index}. "
                f"Need at least {scale + start_index} samples, got {n}."
            )

        # Extract relevant portion of signal
        end_index = start_index + n_windows * scale
        signal_portion = self.signal[start_index:end_index]

        # Reshape and average
        # Shape: (n_windows, scale) -> average over axis 1
        coarse_signal = signal_portion.reshape(n_windows, scale).mean(axis=1)

        return coarse_signal

    def _sample_entropy(self, coarse_signal: np.ndarray) -> float:
        """
        Compute Sample Entropy for a given signal.

        Sample Entropy (SampEn) is a modification of Approximate Entropy that
        is more consistent and less biased. It measures the negative natural
        logarithm of the conditional probability that two sequences similar for
        m points remain similar at m+1 points.

        Parameters
        ----------
        coarse_signal : numpy.ndarray
            Coarse-grained time series

        Returns
        -------
        sample_entropy : float
            Sample entropy value
            Returns 0 if calculation fails (e.g., signal too short)

        Mathematical Definition:
        -----------------------
        SampEn(m, r, N) = -ln(A/B)

        where:
        - A = number of template matches of length m+1
        - B = number of template matches of length m
        - r = tolerance for matching
        - N = signal length

        Algorithm Steps:
        ---------------
        1. Form all possible patterns of length m and m+1
        2. For each pattern, count matches within tolerance r
        3. Compute ratio of matches: A/B
        4. Return -ln(A/B)

        Computational Complexity:
        ------------------------
        O(N²) for naive implementation
        O(N log N) with spatial data structures (KD-tree)

        Implementation Notes:
        --------------------
        This implementation uses scipy's cKDTree for efficient nearest neighbor
        search, achieving O(N log N) complexity instead of O(N²).

        References:
        ----------
        Richman, J. S., & Moorman, J. R. (2000). Physiological time-series
        analysis using approximate entropy and sample entropy. American Journal
        of Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
        """
        N = len(coarse_signal)

        # Minimum length check
        # Need at least m+2 points to form templates
        if N < self.m + 2:
            warnings.warn(
                f"Signal too short ({N} samples) for SampEn with m={self.m}. "
                f"Returning 0.",
                UserWarning,
            )
            return 0.0

        # Helper function to count template matches using KD-tree
        def _count_matches(m_current: int) -> int:
            """
            Count template matches of length m_current within tolerance r.

            Uses KD-tree for efficient nearest neighbor search.
            """
            # Create templates (delay vectors)
            templates = np.array(
                [coarse_signal[i : i + m_current] for i in range(N - m_current + 1)]
            )

            if len(templates) < 2:
                return 0

            # Build KD-tree for efficient search
            # Note: Chebyshev distance (L∞) is used for SampEn
            tree = cKDTree(templates)

            # Count matches within radius r
            # We exclude self-matches by checking distance > 0
            total_matches = 0

            for i, template in enumerate(templates):
                # Query neighbors within distance r
                # We use p=np.inf for Chebyshev distance (maximum norm)
                neighbors = tree.query_ball_point(template, r=self.r, p=np.inf)

                # Subtract 1 to exclude self-match
                # (point is always within r of itself)
                matches = len(neighbors) - 1
                total_matches += matches

            return total_matches

        # Count matches for length m and m+1
        B = _count_matches(self.m)  # Matches of length m
        A = _count_matches(self.m + 1)  # Matches of length m+1

        # Calculate SampEn
        if B == 0 or A == 0:
            # No matches found - signal is very irregular
            # Return maximum entropy (conventional choice)
            warnings.warn(
                "No template matches found. Signal may be too short or too irregular. "
                "Returning 0.",
                UserWarning,
            )
            return 0.0

        # Sample Entropy = -ln(A/B)
        sampen = -np.log(A / B)

        # Ensure non-negative (numerical precision issues)
        sampen = max(0.0, sampen)

        return sampen

    def _fuzzy_entropy(self, coarse_signal: np.ndarray) -> float:
        """
        Compute Fuzzy Entropy for a given signal.

        Fuzzy Entropy (FuzzyEn) uses fuzzy membership functions instead of
        binary matching, making it more stable for short and noisy signals.

        Parameters
        ----------
        coarse_signal : numpy.ndarray
            Coarse-grained time series

        Returns
        -------
        fuzzy_entropy : float
            Fuzzy entropy value

        Mathematical Definition:
        -----------------------
        FuzzyEn uses an exponential membership function:

        μ(d) = exp(-(d/r)^n)

        where:
        - d = distance between patterns
        - r = tolerance
        - n = gradient parameter (typically n=2)

        Advantages over SampEn:
        ----------------------
        1. More stable for short signals
        2. Continuous similarity measure
        3. Better statistical properties
        4. Less sensitive to parameter choices

        References:
        ----------
        Chen, W., Wang, Z., Xie, H., & Yu, W. (2007). Characterization of
        surface EMG signal based on fuzzy entropy. IEEE Transactions on neural
        systems and rehabilitation engineering, 15(2), 266-272.
        """
        N = len(coarse_signal)

        if N < self.m + 2:
            warnings.warn("Signal too short for FuzzyEn. Returning 0.", UserWarning)
            return 0.0

        # Gradient parameter for fuzzy function
        n = 2

        def _phi(m_current: int) -> float:
            """Compute phi function for fuzzy entropy."""
            # Create templates
            templates = np.array(
                [coarse_signal[i : i + m_current] for i in range(N - m_current + 1)]
            )

            if len(templates) < 2:
                return 0.0

            # Calculate all pairwise distances
            similarities = 0.0
            n_patterns = len(templates)

            for i in range(n_patterns):
                for j in range(n_patterns):
                    if i != j:
                        # Maximum absolute difference (Chebyshev distance)
                        d = np.max(np.abs(templates[i] - templates[j]))

                        # Fuzzy membership function
                        similarity = np.exp(-((d / self.r) ** n))
                        similarities += similarity

            # Average similarity
            phi = similarities / (n_patterns * (n_patterns - 1))

            return phi

        # Compute phi for m and m+1
        phi_m = _phi(self.m)
        phi_m_plus_1 = _phi(self.m + 1)

        # Fuzzy Entropy
        if phi_m_plus_1 == 0 or phi_m == 0:
            warnings.warn("FuzzyEn calculation failed. Returning 0.", UserWarning)
            return 0.0

        fuzzy_en = np.log(phi_m) - np.log(phi_m_plus_1)

        return max(0.0, fuzzy_en)

    def compute_mse(self) -> np.ndarray:
        """
        Compute Multi-Scale Entropy (MSE) across all scales.

        This is the standard MSE algorithm that computes entropy at each
        coarse-grained scale from 1 to max_scale.

        Returns
        -------
        mse_values : numpy.ndarray
            Array of entropy values for each scale (length: max_scale)
            Index i corresponds to scale i+1

        Algorithm:
        ---------
        For each scale τ = 1, 2, ..., max_scale:
            1. Coarse-grain signal at scale τ
            2. Compute Sample Entropy (or Fuzzy Entropy) of coarse-grained signal
            3. Store entropy value for scale τ

        Time Complexity:
        ---------------
        O(max_scale * N log N) where N is signal length

        Examples:
        --------
        >>> mse = MultiScaleEntropy(signal, max_scale=20)
        >>> entropy_values = mse.compute_mse()
        >>>
        >>> # Plot MSE curve
        >>> import matplotlib.pyplot as plt
        >>> scales = np.arange(1, 21)
        >>> plt.plot(scales, entropy_values, 'o-')
        >>> plt.xlabel('Scale Factor')
        >>> plt.ylabel('Sample Entropy')
        >>> plt.title('Multi-Scale Entropy')
        >>> plt.grid(True)
        >>> plt.show()

        Clinical Interpretation:
        -----------------------
        - **Healthy/Young:** MSE stays elevated or increases at larger scales
        - **Disease/Aging:** MSE decreases rapidly with scale
        - **Heart Failure:** Marked decrease in entropy at all scales
        - **Atrial Fibrillation:** Very high entropy at small scales, rapid decrease
        """
        mse_values = []

        # Select entropy calculation method
        entropy_func = self._fuzzy_entropy if self.fuzzy else self._sample_entropy

        for scale in range(1, self.max_scale + 1):
            try:
                # Coarse-grain signal at current scale
                coarse_signal = self._coarse_grain(scale)

                # Compute entropy
                entropy = entropy_func(coarse_signal)

                mse_values.append(entropy)

            except Exception as e:
                warnings.warn(
                    f"Failed to compute entropy at scale {scale}: {str(e)}. "
                    f"Using 0.",
                    UserWarning,
                )
                mse_values.append(0.0)

        return np.array(mse_values)

    def compute_cmse(self) -> np.ndarray:
        """
        Compute Composite Multi-Scale Entropy (CMSE).

        CMSE improves upon standard MSE by averaging entropy values across
        multiple coarse-grained series with different starting points. This
        reduces variance and provides more stable estimates, especially for
        shorter signals.

        Returns
        -------
        cmse_values : numpy.ndarray
            Array of composite entropy values for each scale

        Algorithm:
        ---------
        For each scale τ = 1, 2, ..., max_scale:
            1. Create τ different coarse-grained series starting at indices 0, 1, ..., τ-1
            2. Compute entropy for each coarse-grained series
            3. Average the τ entropy values

        Advantages over Standard MSE:
        -----------------------------
        1. **Reduced Variance:** Averaging reduces statistical fluctuations
        2. **Better Stability:** More reliable for short signals
        3. **Improved Discrimination:** Better separates different signal classes
        4. **Consistent Results:** Less sensitive to signal length

        Time Complexity:
        ---------------
        O(max_scale² * N log N)
        Note: ~τ times slower than MSE due to multiple coarse-grainings

        Examples:
        --------
        >>> mse = MultiScaleEntropy(signal, max_scale=15)
        >>> cmse_values = mse.compute_cmse()
        >>>
        >>> # Compare with standard MSE
        >>> mse_values = mse.compute_mse()
        >>>
        >>> import matplotlib.pyplot as plt
        >>> scales = np.arange(1, 16)
        >>> plt.plot(scales, mse_values, 'o-', label='MSE')
        >>> plt.plot(scales, cmse_values, 's-', label='CMSE')
        >>> plt.xlabel('Scale')
        >>> plt.ylabel('Entropy')
        >>> plt.legend()
        >>> plt.grid(True)

        References:
        ----------
        Wu, S. D., Wu, C. W., Lin, S. G., Wang, C. C., & Lee, K. Y. (2013).
        Time series analysis using composite multiscale entropy. Entropy,
        15(3), 1069-1084.

        Notes:
        -----
        CMSE is particularly recommended when:
        - Signal length < 1000 samples
        - max_scale > 10
        - Comparing signals of different lengths
        - High precision is required
        """
        cmse_values = []

        entropy_func = self._fuzzy_entropy if self.fuzzy else self._sample_entropy

        for scale in range(1, self.max_scale + 1):
            scale_entropies = []

            # Create multiple coarse-grained series with different starting points
            for start_idx in range(scale):
                try:
                    coarse_signal = self._coarse_grain(scale, start_index=start_idx)

                    # Skip if coarse-grained signal is too short
                    if len(coarse_signal) < self.m + 2:
                        continue

                    entropy = entropy_func(coarse_signal)
                    scale_entropies.append(entropy)

                except Exception as e:
                    # Skip this starting point if it fails
                    continue

            # Average entropy across all starting points
            if scale_entropies:
                cmse_value = np.mean(scale_entropies)
            else:
                warnings.warn(
                    f"No valid entropy values at scale {scale}. Using 0.", UserWarning
                )
                cmse_value = 0.0

            cmse_values.append(cmse_value)

        return np.array(cmse_values)

    def compute_rcmse(self) -> np.ndarray:
        """
        Compute Refined Composite Multi-Scale Entropy (RCMSE).

        RCMSE further refines CMSE by using a modified coarse-graining procedure
        that preserves more information from the original signal.

        Returns
        -------
        rcmse_values : numpy.ndarray
            Array of refined composite entropy values

        Refined Coarse-Graining:
        -----------------------
        Instead of non-overlapping windows, RCMSE uses overlapping windows:

        y^(τ)_j = (1/τ) * Σ(i=j to j+τ-1) x_i

        This preserves more temporal structure and reduces information loss.

        Advantages over CMSE:
        --------------------
        1. **Better Information Preservation:** Overlapping windows retain more details
        2. **Smoother Curves:** Less jagged MSE curves
        3. **Improved Sensitivity:** Better detects subtle changes
        4. **Best Stability:** Superior performance on short signals

        When to Use RCMSE:
        -----------------
        - Short signals (< 500 samples)
        - Need maximum stability
        - Require smooth, interpretable curves
        - Comparing very different conditions

        References:
        ----------
        Wu, S. D., Wu, C. W., Lin, S. G., Lee, K. Y., & Peng, C. K. (2014).
        Analysis of complex time series using refined composite multiscale
        entropy. Physics Letters A, 378(20), 1369-1374.
        """
        rcmse_values = []

        entropy_func = self._fuzzy_entropy if self.fuzzy else self._sample_entropy

        for scale in range(1, self.max_scale + 1):
            scale_entropies = []

            # Refined coarse-graining with overlapping windows
            n = len(self.signal)
            n_windows = n - scale + 1

            if n_windows < self.m + 2:
                warnings.warn(
                    f"Signal too short for RCMSE at scale {scale}. Using 0.",
                    UserWarning,
                )
                rcmse_values.append(0.0)
                continue

            # Create overlapping coarse-grained series
            coarse_signals = []
            for start_idx in range(n_windows):
                window = self.signal[start_idx : start_idx + scale]
                coarse_value = np.mean(window)
                coarse_signals.append(coarse_value)

            coarse_signal = np.array(coarse_signals)

            try:
                entropy = entropy_func(coarse_signal)
                rcmse_values.append(entropy)
            except Exception as e:
                warnings.warn(
                    f"Failed to compute RCMSE at scale {scale}: {str(e)}. Using 0.",
                    UserWarning,
                )
                rcmse_values.append(0.0)

        return np.array(rcmse_values)

    def get_complexity_index(
        self, entropy_values: np.ndarray, scale_range: Optional[Tuple[int, int]] = None
    ) -> float:
        """
        Calculate Complexity Index (CI) as area under the MSE curve.

        The complexity index summarizes the overall complexity across scales
        into a single scalar value. Higher CI indicates more complex, healthy
        physiological regulation.

        Parameters
        ----------
        entropy_values : numpy.ndarray
            MSE/CMSE/RCMSE values
        scale_range : tuple of int, optional
            (start_scale, end_scale) for integration (default: all scales)
            Useful for focusing on specific temporal scales

        Returns
        -------
        complexity_index : float
            Area under the entropy curve (using trapezoidal integration)

        Formula:
        -------
        CI = Σ(i=1 to max_scale-1) [(Entropy_i + Entropy_(i+1)) / 2]

        Clinical Interpretation:
        -----------------------
        - **High CI:** Complex, adaptive physiological regulation (healthy)
        - **Low CI:** Simple, less adaptive regulation (disease, aging)
        - **Very Low CI:** Pathological simplification (severe disease)

        Examples:
        --------
        >>> mse = MultiScaleEntropy(signal)
        >>> entropy = mse.compute_mse()
        >>>
        >>> # Overall complexity
        >>> ci_total = mse.get_complexity_index(entropy)
        >>>
        >>> # Short-term complexity (scales 1-5)
        >>> ci_short = mse.get_complexity_index(entropy, scale_range=(1, 5))
        >>>
        >>> # Long-term complexity (scales 10-20)
        >>> ci_long = mse.get_complexity_index(entropy, scale_range=(10, 20))

        Notes:
        -----
        Different scale ranges provide insights into different regulatory mechanisms:
        - Scales 1-5: Intrinsic cardiac dynamics
        - Scales 5-10: Sympathovagal balance
        - Scales 10-20: Long-term regulatory mechanisms
        """
        if scale_range is None:
            scale_range = (1, len(entropy_values))

        start_idx = scale_range[0] - 1  # Convert to 0-indexed
        end_idx = scale_range[1]

        # Extract relevant portion
        entropy_subset = entropy_values[start_idx:end_idx]

        if len(entropy_subset) < 2:
            warnings.warn(
                "Not enough entropy values for complexity index. Returning 0.",
                UserWarning,
            )
            return 0.0

        # Trapezoidal integration
        complexity_index = np.trapz(entropy_subset)

        return complexity_index


# Export main class
__all__ = ["MultiScaleEntropy"]
