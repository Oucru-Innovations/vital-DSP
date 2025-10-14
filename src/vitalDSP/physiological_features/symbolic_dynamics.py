"""
Symbolic Dynamics Analysis Module
==================================

This module provides symbolic dynamics methods for analyzing physiological signals
by transforming continuous signals into discrete symbol sequences for pattern analysis.

Implemented Methods:
-------------------
1. Symbolic Transformation (0V, 1V, 2LV, 2UV patterns)
2. Shannon Entropy of Symbol Distribution
3. Word Distribution Analysis
4. Forbidden Words Detection
5. Pattern Transition Analysis
6. Renyi Entropy
7. Permutation Entropy

Clinical Applications:
---------------------
- Cardiac autonomic function assessment
- Arrhythmia detection and classification
- Sleep stage classification
- Fetal heart rate monitoring
- Blood pressure variability analysis
- Seizure prediction

Mathematical Background:
-----------------------
Symbolic dynamics transforms a continuous-valued time series into a sequence of
discrete symbols based on pattern recognition. This approach reduces noise sensitivity
and reveals underlying regulatory patterns.

The transformation captures important dynamical features while being robust to:
- Measurement noise
- Non-stationarity
- Missing data
- Computational complexity

References:
----------
1. Voss, A., Schulz, S., Schroeder, R., Baumert, M., & Caminal, P. (2009).
   Methods derived from nonlinear dynamics for analysing heart rate variability.
   Philosophical Transactions of the Royal Society A, 367(1887), 277-296.

2. Porta, A., Guzzetti, S., Montano, N., Furlan, R., Pagani, M., Malliani, A.,
   & Cerutti, S. (2001). Entropy, entropy rate, and pattern classification as
   tools to typify complexity in short heart period variability series.
   IEEE Transactions on Biomedical Engineering, 48(11), 1282-1291.

3. Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity
   measure for time series. Physical review letters, 88(17), 174102.

Author: Claude (Sonnet 4.5)
Date: October 10, 2025
Version: 1.0
"""

import numpy as np
from collections import Counter
from itertools import permutations, product
from typing import List, Dict, Tuple, Optional
import warnings


class SymbolicDynamics:
    """
    Symbolic Dynamics Analysis for physiological signals.

    Transforms continuous time series into symbolic sequences and analyzes
    the distribution and patterns of symbols.

    Parameters
    ----------
    signal : numpy.ndarray
        Input time series signal (1D array)
    n_symbols : int, optional
        Number of symbols to use (default: 4)
        Common choices: 3, 4, 6
    word_length : int, optional
        Length of words to analyze (default: 3)
        Typical range: 2-5
    method : str, optional
        Symbolization method (default: '0V')
        Options: '0V' (variations), 'quantile', 'SAX', 'threshold'

    Attributes
    ----------
    signal : numpy.ndarray
        Original signal
    n_symbols : int
        Number of symbols
    word_length : int
        Word length for pattern analysis
    method : str
        Symbolization method
    symbols : numpy.ndarray
        Symbolic sequence

    Methods
    -------
    symbolize()
        Transform signal to symbol sequence
    compute_shannon_entropy()
        Shannon entropy of symbol distribution
    compute_word_distribution()
        Distribution of words
    detect_forbidden_words()
        Find patterns that never occur
    compute_transition_matrix()
        Symbol transition probabilities
    compute_renyi_entropy(alpha)
        Generalized Renyi entropy
    compute_permutation_entropy()
        Permutation entropy

    Examples
    --------
    >>> # Analyze heart rate variability
    >>> from vitalDSP.physiological_features.symbolic_dynamics import SymbolicDynamics
    >>> import numpy as np
    >>>
    >>> # RR intervals (seconds)
    >>> rr = np.array([1.0, 0.95, 1.02, 0.98, 1.01, 0.96, ...])
    >>>
    >>> # Create symbolic representation
    >>> sd = SymbolicDynamics(rr, n_symbols=4, word_length=3)
    >>> symbols = sd.symbolize()
    >>>
    >>> # Compute Shannon entropy
    >>> h = sd.compute_shannon_entropy()
    >>> print(f"Shannon Entropy: {h:.4f}")
    >>>
    >>> # Analyze word distribution
    >>> word_dist = sd.compute_word_distribution()
    >>>
    >>> # Find forbidden words (never occurring patterns)
    >>> forbidden = sd.detect_forbidden_words()
    >>> print(f"Forbidden words: {len(forbidden)}")

    Notes
    -----
    **Symbol Interpretation (0V method):**

    - **0V (no variation):** Three consecutive values are approximately equal
      Represents stable regulation

    - **1V (one variation):** Two values equal, one different
      Represents small perturbations

    - **2LV (two variations, low first):** Low-High-Low or similar
      Represents oscillatory pattern with deceleration

    - **2UV (two variations, high first):** High-Low-High or similar
      Represents oscillatory pattern with acceleration

    **Clinical Interpretation:**

    - **Healthy:** Balanced distribution of symbols, few forbidden words
    - **Disease:** Skewed distribution, many forbidden words
    - **Atrial Fibrillation:** Very high entropy, nearly uniform distribution
    - **Heart Failure:** Low entropy, many forbidden words

    **Parameter Recommendations:**

    - **n_symbols:** 4-6 for HRV analysis
    - **word_length:** 3 for balance of detail and statistics
    - **method:** '0V' for HRV, 'quantile' for general signals
    """

    def __init__(
        self,
        signal: np.ndarray,
        n_symbols: int = 4,
        word_length: int = 3,
        method: str = "0V",
    ):
        """
        Initialize Symbolic Dynamics analyzer.

        Parameters
        ----------
        signal : numpy.ndarray
            Input time series
        n_symbols : int
            Number of symbols (3-10)
        word_length : int
            Word length (2-5)
        method : str
            Symbolization method
        """
        # Input validation
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)

        if len(signal) < word_length:
            raise ValueError(
                f"Signal too short ({len(signal)} samples). "
                f"Need at least {word_length} samples."
            )

        if n_symbols < 2 or n_symbols > 26:
            raise ValueError(f"n_symbols must be 2-26, got {n_symbols}")

        if word_length < 2 or word_length > 10:
            raise ValueError(f"word_length must be 2-10, got {word_length}")

        valid_methods = ["0V", "quantile", "SAX", "threshold"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {method}")

        self.signal = signal
        self.n_symbols = n_symbols
        self.word_length = word_length
        self.method = method
        self.symbols = None

    def symbolize(self) -> np.ndarray:
        """
        Transform continuous signal to symbolic sequence.

        Returns
        -------
        symbols : numpy.ndarray
            Array of symbol indices (integers 0 to n_symbols-1)

        Methods:
        -------
        **1. 0V Method (Variations):**
        Classifies triplets based on pattern variations:
        - 0V: all approximately equal (|a-b|<δ, |b-c|<δ, |a-c|<δ)
        - 1V: two equal, one different
        - 2LV: two variations with low-high-low pattern
        - 2UV: two variations with high-low-high pattern

        **2. Quantile Method:**
        Divides signal into n_symbols quantiles

        **3. SAX (Symbolic Aggregate approXimation):**
        Uses Gaussian quantiles for symbolization

        **4. Threshold Method:**
        Simple thresholding based on percentiles

        Examples:
        --------
        >>> sd = SymbolicDynamics(signal, n_symbols=4, method='0V')
        >>> symbols = sd.symbolize()
        >>>
        >>> # Convert to letter representation
        >>> letters = ''.join([chr(65+s) for s in symbols])  # A, B, C, D...
        >>> print(f"Symbolic sequence: {letters[:50]}...")
        """
        if self.method == "0V":
            symbols = self._symbolize_0v()
        elif self.method == "quantile":
            symbols = self._symbolize_quantile()
        elif self.method == "SAX":
            symbols = self._symbolize_sax()
        elif self.method == "threshold":
            symbols = self._symbolize_threshold()

        self.symbols = symbols
        return symbols

    def _symbolize_0v(self) -> np.ndarray:
        """
        0V symbolization method for HRV analysis.

        Classifies consecutive triplets into 4 categories:
        0V, 1V, 2LV, 2UV

        Returns:
        -------
        symbols : numpy.ndarray
            Symbol sequence (0-3)
        """
        # Calculate tolerance (typically 0.5% of signal range)
        delta = 0.005 * (np.max(self.signal) - np.min(self.signal))

        symbols = []

        for i in range(len(self.signal) - 2):
            a, b, c = self.signal[i : i + 3]

            # Calculate differences
            diff_ab = abs(a - b)
            diff_bc = abs(b - c)
            diff_ac = abs(a - c)

            # Count variations
            variations = sum([diff_ab > delta, diff_bc > delta, diff_ac > delta])

            if variations == 0:
                # 0V: no variation (all approximately equal)
                symbol = 0
            elif variations == 1 or variations == 2:
                # 1V: one variation (two equal, one different)
                symbol = 1
            else:
                # 2 variations: determine if Low-High-Low or High-Low-High
                if (a < b and b > c) or (a > b and b < c):
                    # 2LV: two variations, oscillatory with extremum in middle
                    if a < c:
                        symbol = 2  # Low-High-Low
                    else:
                        symbol = 3  # High-Low-High (2UV)
                else:
                    # Other patterns
                    symbol = 1

            symbols.append(symbol)

        return np.array(symbols)

    def _symbolize_quantile(self) -> np.ndarray:
        """
        Quantile-based symbolization.

        Divides signal range into equal-frequency bins.
        """
        # Calculate quantile thresholds
        quantiles = np.linspace(0, 100, self.n_symbols + 1)
        thresholds = np.percentile(self.signal, quantiles[1:-1])

        # Assign symbols based on quantile
        symbols = np.digitize(self.signal, thresholds)

        return symbols

    def _symbolize_sax(self) -> np.ndarray:
        """
        SAX (Symbolic Aggregate approXimation) symbolization.

        Uses Gaussian quantiles assuming normalized signal.
        """
        # Normalize signal (z-score)
        normalized = (self.signal - np.mean(self.signal)) / np.std(self.signal)

        # Calculate Gaussian quantile breakpoints
        from scipy import stats

        quantiles = np.linspace(0, 1, self.n_symbols + 1)[1:-1]
        thresholds = stats.norm.ppf(quantiles)

        # Assign symbols
        symbols = np.digitize(normalized, thresholds)

        return symbols

    def _symbolize_threshold(self) -> np.ndarray:
        """Simple threshold-based symbolization."""
        thresholds = np.linspace(
            np.min(self.signal), np.max(self.signal), self.n_symbols + 1
        )[1:-1]
        symbols = np.digitize(self.signal, thresholds)
        return symbols

    def compute_shannon_entropy(self) -> float:
        """
        Compute Shannon entropy of symbol distribution.

        Shannon entropy quantifies the average information content or
        unpredictability of the symbol sequence.

        Returns
        -------
        entropy : float
            Shannon entropy in bits (log base 2)

        Formula:
        -------
        H = -Σ p(i) * log2(p(i))

        where p(i) is the probability of symbol i.

        Interpretation:
        --------------
        - **0:** Completely predictable (only one symbol appears)
        - **log2(n_symbols):** Maximum entropy (uniform distribution)
        - **Between:** Degree of predictability/complexity

        Clinical Significance:
        ---------------------
        - **Low H:** Regular, predictable rhythm (may indicate reduced adaptability)
        - **High H:** Variable, unpredictable rhythm (healthy variability)
        - **Very High H:** Chaotic, random (e.g., atrial fibrillation)

        Examples:
        --------
        >>> sd = SymbolicDynamics(signal)
        >>> sd.symbolize()
        >>> h = sd.compute_shannon_entropy()
        >>>
        >>> # Normalize by maximum possible entropy
        >>> h_max = np.log2(sd.n_symbols)
        >>> h_norm = h / h_max
        >>> print(f"Normalized entropy: {h_norm:.4f}")
        """
        if self.symbols is None:
            self.symbolize()

        # Count symbol frequencies
        symbol_counts = Counter(self.symbols)
        total_symbols = len(self.symbols)

        # Calculate probabilities
        probabilities = np.array(
            [count / total_symbols for count in symbol_counts.values()]
        )

        # Shannon entropy: -Σ p*log2(p)
        # Handle log(0) by removing zero probabilities
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log2(probabilities))

        return entropy

    def compute_word_distribution(self) -> Dict[str, float]:
        """
        Compute distribution of words (symbol patterns).

        Returns
        -------
        word_dist : dict
            Dictionary mapping words to their probabilities
            Keys: words (strings of symbols)
            Values: probabilities (0-1)

        Examples:
        --------
        >>> sd = SymbolicDynamics(signal, word_length=3)
        >>> sd.symbolize()
        >>> word_dist = sd.compute_word_distribution()
        >>>
        >>> # Most common words
        >>> sorted_words = sorted(word_dist.items(), key=lambda x: x[1], reverse=True)
        >>> print("Top 5 most common words:")
        >>> for word, prob in sorted_words[:5]:
        ...     print(f"{word}: {prob:.4f}")
        """
        if self.symbols is None:
            self.symbolize()

        # Extract words
        words = []
        for i in range(len(self.symbols) - self.word_length + 1):
            word = tuple(self.symbols[i : i + self.word_length])
            words.append(word)

        # Count word frequencies
        word_counts = Counter(words)
        total_words = len(words)

        # Convert to probabilities
        word_dist = {
            "".join(str(s) for s in word): count / total_words
            for word, count in word_counts.items()
        }

        return word_dist

    def detect_forbidden_words(self) -> List[str]:
        """
        Detect forbidden words (patterns that never occur).

        Returns
        -------
        forbidden_words : list of str
            List of words that never appear in the sequence

        Significance:
        ------------
        Forbidden words indicate deterministic constraints or regulatory
        mechanisms that prevent certain patterns from occurring.

        - **Many forbidden words:** Strong regulatory constraints (often pathological)
        - **Few forbidden words:** Flexible regulation (typically healthy)
        - **No forbidden words:** Complete randomness (e.g., atrial fibrillation)

        Examples:
        --------
        >>> sd = SymbolicDynamics(signal, n_symbols=4, word_length=3)
        >>> sd.symbolize()
        >>> forbidden = sd.detect_forbidden_words()
        >>>
        >>> total_possible = sd.n_symbols ** sd.word_length
        >>> forbidden_ratio = len(forbidden) / total_possible
        >>> print(f"Forbidden word ratio: {forbidden_ratio:.2%}")
        """
        if self.symbols is None:
            self.symbolize()

        # Get observed words
        word_dist = self.compute_word_distribution()
        observed_words = set(word_dist.keys())

        # Generate all possible words
        all_possible_words = set()
        for perm in product(range(self.n_symbols), repeat=self.word_length):
            word = "".join(str(s) for s in perm)
            all_possible_words.add(word)

        # Find forbidden words
        forbidden_words = list(all_possible_words - observed_words)

        return forbidden_words

    def compute_transition_matrix(self) -> np.ndarray:
        """
        Compute symbol transition probability matrix.

        Returns
        -------
        transition_matrix : numpy.ndarray
            Matrix of transition probabilities (n_symbols x n_symbols)
            Element [i,j] = P(next symbol is j | current symbol is i)

        Examples:
        --------
        >>> sd = SymbolicDynamics(signal, n_symbols=4)
        >>> sd.symbolize()
        >>> trans = sd.compute_transition_matrix()
        >>>
        >>> # Visualize transition matrix
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(trans, cmap='hot', interpolation='nearest')
        >>> plt.colorbar(label='Transition Probability')
        >>> plt.xlabel('Next Symbol')
        >>> plt.ylabel('Current Symbol')
        >>> plt.title('Symbol Transition Matrix')
        """
        if self.symbols is None:
            self.symbolize()

        # Initialize transition count matrix
        trans_counts = np.zeros((self.n_symbols, self.n_symbols))

        # Count transitions
        for i in range(len(self.symbols) - 1):
            current_symbol = self.symbols[i]
            next_symbol = self.symbols[i + 1]
            trans_counts[current_symbol, next_symbol] += 1

        # Convert to probabilities
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero

        transition_matrix = trans_counts / row_sums

        return transition_matrix

    def compute_renyi_entropy(self, alpha: float = 2.0) -> float:
        """
        Compute Renyi entropy (generalized entropy measure).

        Parameters
        ----------
        alpha : float
            Order parameter
            - alpha=0: Hartley entropy (log of number of distinct symbols)
            - alpha=1: Shannon entropy (limit as alpha→1)
            - alpha=2: Collision entropy
            - alpha=∞: Min-entropy

        Returns
        -------
        renyi_entropy : float
            Renyi entropy value

        Formula:
        -------
        H_α = (1/(1-α)) * log2(Σ p_i^α)

        where p_i are symbol probabilities.

        Clinical Use:
        ------------
        Different alpha values emphasize different aspects:
        - α < 1: Emphasizes rare events
        - α > 1: Emphasizes common events
        - α = 2: Good balance, computationally efficient
        """
        if self.symbols is None:
            self.symbolize()

        # Count symbol frequencies
        symbol_counts = Counter(self.symbols)
        total_symbols = len(self.symbols)

        # Calculate probabilities
        probabilities = np.array(
            [count / total_symbols for count in symbol_counts.values()]
        )

        # Handle special cases
        if alpha == 1.0:
            # Shannon entropy
            return self.compute_shannon_entropy()

        if alpha == 0.0:
            # Hartley entropy
            return np.log2(len(probabilities))

        if np.isinf(alpha):
            # Min-entropy
            return -np.log2(np.max(probabilities))

        # General Renyi entropy
        renyi = (1 / (1 - alpha)) * np.log2(np.sum(probabilities**alpha))

        return renyi

    def compute_permutation_entropy(self, order: int = 3) -> float:
        """
        Compute Permutation Entropy.

        Permutation entropy analyzes the order relationships between consecutive
        values, making it robust to noise and monotonic transformations.

        Parameters
        ----------
        order : int
            Order of permutation patterns (default: 3)
            Typical range: 3-7

        Returns
        -------
        perm_entropy : float
            Permutation entropy value

        Algorithm:
        ---------
        1. Extract overlapping windows of length 'order'
        2. Determine ranking permutation for each window
        3. Count frequency of each permutation pattern
        4. Calculate Shannon entropy of permutation distribution

        Advantages:
        ----------
        - Robust to noise
        - Fast computation
        - Conceptually simple
        - Good for nonlinear signals

        References:
        ----------
        Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity
        measure for time series. Physical review letters, 88(17), 174102.

        Examples:
        --------
        >>> sd = SymbolicDynamics(signal)
        >>> pe = sd.compute_permutation_entropy(order=3)
        >>> print(f"Permutation Entropy: {pe:.4f}")
        """
        signal = self.signal
        n = len(signal)

        if n < order:
            raise ValueError(f"Signal length ({n}) < order ({order})")

        # Extract permutation patterns
        patterns = []

        for i in range(n - order + 1):
            # Get window
            window = signal[i : i + order]

            # Determine permutation (argsort gives ranking)
            perm = tuple(np.argsort(window))
            patterns.append(perm)

        # Count pattern frequencies
        pattern_counts = Counter(patterns)
        total_patterns = len(patterns)

        # Calculate probabilities
        probabilities = np.array(
            [count / total_patterns for count in pattern_counts.values()]
        )

        # Shannon entropy of permutations
        perm_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

        # Normalize by maximum possible entropy
        max_entropy = np.log2(np.math.factorial(order))
        normalized_pe = perm_entropy / max_entropy if max_entropy > 0 else 0

        return normalized_pe


# Export main class
__all__ = ["SymbolicDynamics"]
