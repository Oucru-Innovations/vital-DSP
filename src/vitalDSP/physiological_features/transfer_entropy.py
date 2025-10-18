"""
Transfer Entropy Module for Coupling Analysis
==============================================

This module provides transfer entropy methods for analyzing directional information
flow and coupling between physiological signals.

Implemented Methods:
-------------------
1. Transfer Entropy (TE)
2. Conditional Transfer Entropy
3. Time-Delayed Transfer Entropy
4. Normalized Transfer Entropy
5. Effective Transfer Entropy

Clinical Applications:
---------------------
- Cardio-respiratory coupling analysis
- Brain-heart interaction
- Autonomic nervous system assessment
- Multi-organ system dynamics
- Baroreflex sensitivity
- Neurovascular coupling

Mathematical Background:
-----------------------
Transfer entropy quantifies the directed (causal) information flow from one
time series to another. Unlike correlation, it captures nonlinear relationships
and distinguishes the direction of influence.

TE from X to Y measures how much uncertainty about the future of Y is reduced
by knowing the past of X, given the past of Y.

References:
----------
1. Schreiber, T. (2000). Measuring information transfer. Physical review letters,
   85(2), 461.

2. Faes, L., Nollo, G., & Porta, A. (2011). Information-based detection of
   nonlinear Granger causality in multivariate processes via a nonuniform
   embedding technique. Physical Review E, 83(5), 051112.

3. Barnett, L., Barrett, A. B., & Seth, A. K. (2009). Granger causality and
   transfer entropy are equivalent for Gaussian variables. Physical review
   letters, 103(23), 238701.

4. Vakorin, V. A., Krakovska, O. A., & McIntosh, A. R. (2009). Confounding
   effects of indirect connections on causality estimation. Journal of
   neuroscience methods, 184(1), 152-160.

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
    >>> from vitalDSP.physiological_features.transfer_entropy import TransferEntropy
    >>> signal = np.random.randn(1000)
    >>> processor = TransferEntropy(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""



import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional, List
import warnings


class TransferEntropy:
    """
    Transfer Entropy analysis for directional coupling between signals.

    Transfer Entropy (TE) quantifies the directional information flow from
    a source signal to a target signal, revealing causal relationships.

    Parameters
    ----------
    source : numpy.ndarray
        Source time series (potential driver)
    target : numpy.ndarray
        Target time series (potentially driven)
    k : int, optional
        History length (embedding dimension) for target (default: 1)
    l : int, optional
        History length for source (default: 1)
    delay : int, optional
        Time delay for embedding (default: 1)
    n_bins : int, optional
        Number of bins for histogram estimation (default: None, uses KNN)
    k_neighbors : int, optional
        Number of nearest neighbors for KNN estimation (default: 3)

    Attributes
    ----------
    source : numpy.ndarray
        Source signal
    target : numpy.ndarray
        Target signal
    k : int
        Target history length
    l : int
        Source history length
    delay : int
        Embedding delay

    Methods
    -------
    compute_transfer_entropy()
        Compute TE from source to target
    compute_bidirectional_te()
        Compute TE in both directions
    compute_time_delayed_te(max_delay)
        TE across multiple time delays
    compute_effective_te()
        Normalized effective TE
    test_significance(n_surrogates)
        Statistical significance testing

    Examples
    --------
    >>> # Analyze cardio-respiratory coupling
    >>> from vitalDSP.physiological_features.transfer_entropy import TransferEntropy
    >>> import numpy as np
    >>>
    >>> # Heart rate (BPM) and respiration rate
    >>> heart_rate = np.array([...])  # Time series of HR
    >>> resp_rate = np.array([...])   # Time series of respiration
    >>>
    >>> # Compute transfer entropy
    >>> te = TransferEntropy(resp_rate, heart_rate, k=1, l=1)
    >>>
    >>> # Respiratory influence on heart rate
    >>> te_resp_to_hr = te.compute_transfer_entropy()
    >>> print(f"TE(Resp → HR): {te_resp_to_hr:.4f}")
    >>>
    >>> # Bidirectional coupling
    >>> te_forward, te_backward = te.compute_bidirectional_te()
    >>> print(f"TE(Resp → HR): {te_forward:.4f}")
    >>> print(f"TE(HR → Resp): {te_backward:.4f}")
    >>>
    >>> # Net directional influence
    >>> net_te = te_forward - te_backward
    >>> if net_te > 0:
    ...     print("Respiration drives heart rate")
    >>> else:
    ...     print("Heart rate drives respiration")

    Notes
    -----
    **Interpretation:**

    - **TE > 0:** Information flows from source to target
    - **TE ≈ 0:** No directional coupling detected
    - **TE < 0:** Should not occur (implementation error)

    **Comparison with Bidirectional TE:**

    - If TE(X→Y) > TE(Y→X): X predominantly drives Y
    - If TE(X→Y) ≈ TE(Y→X): Bidirectional coupling or common drive
    - Significance testing required to confirm non-zero values

    **Parameter Guidelines:**

    - **k, l:** Start with 1, increase if signals have memory
    - **delay:** Typically 1 for high sampling rate, larger for slower dynamics
    - **k_neighbors:** 3-5 for most applications

    **Computational Considerations:**

    - Uses KNN estimation (Kraskov method) for continuous signals
    - Time complexity: O(N log N) with KD-trees
    - Requires signals of same length
    - Stationary signals recommended
    """

    def __init__(
        self,
        source: np.ndarray,
        target: np.ndarray,
        k_coef: int = 1,
        l_coef: int = 1,
        delay: int = 1,
        n_bins: Optional[int] = None,
        k_neighbors: int = 3,
    ):
        """
        Initialize Transfer Entropy analyzer.

        Parameters
        ----------
        source : numpy.ndarray
            Source signal
        target : numpy.ndarray
            Target signal
        k_coef : int
            Target history
        l_coef : int
            Source history
        delay : int
            Time delay
        n_bins : int, optional
            Binning (None for KNN)
        k_neighbors : int
            Neighbors for KNN
        """
        # Input validation
        if not isinstance(source, np.ndarray):
            source = np.array(source)
        if not isinstance(target, np.ndarray):
            target = np.array(target)

        if len(source) != len(target):
            raise ValueError(
                f"Source and target must have same length. "
                f"Got {len(source)} and {len(target)}."
            )

        if len(source) < max(k_coef, l_coef) * delay + 10:
            raise ValueError(
                f"Signals too short ({len(source)} samples) for "
                f"k={k_coef}, l={l_coef}, delay={delay}."
            )

        if k_coef < 1 or l_coef < 1:
            raise ValueError(f"k and l must be >= 1. Got k={k_coef}, l={l_coef}")

        if delay < 1:
            raise ValueError(f"delay must be >= 1. Got {delay}")

        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1. Got {k_neighbors}")

        self.source = source
        self.target = target
        self.k = k_coef
        self.l = l_coef
        self.delay = delay
        self.n_bins = n_bins
        self.k_neighbors = k_neighbors

    def _create_embedding(
        self, signal: np.ndarray, dimension: int, delay: int
    ) -> np.ndarray:
        """
        Create time-delay embedding (phase space reconstruction).

        Parameters
        ----------
        signal : numpy.ndarray
            Input signal
        dimension : int
            Embedding dimension
        delay : int
            Time delay

        Returns
        -------
        embedded : numpy.ndarray
            Embedded vectors (n_vectors x dimension)

        Mathematical Background:
        -----------------------
        Time-delay embedding reconstructs the state space from a scalar time series:

        X(t) = [x(t), x(t-τ), x(t-2τ), ..., x(t-(d-1)τ)]

        where:
        - d = dimension
        - τ = delay
        - t = time index

        This is based on Takens' embedding theorem.

        References:
        ----------
        Takens, F. (1981). Detecting strange attractors in turbulence.
        In Dynamical systems and turbulence, Warwick 1980 (pp. 366-381).
        """
        n = len(signal)
        n_vectors = n - (dimension - 1) * delay

        if n_vectors < 1:
            raise ValueError(
                f"Signal too short for embedding. Need at least "
                f"{(dimension - 1) * delay + 1} samples."
            )

        # Create embedded vectors
        embedded = np.zeros((n_vectors, dimension))

        for i in range(dimension):
            start_idx = i * delay
            end_idx = start_idx + n_vectors
            embedded[:, i] = signal[start_idx:end_idx]

        return embedded

    def _estimate_entropy_knn(self, data: np.ndarray, k: int) -> float:
        """
        Estimate entropy using k-nearest neighbors (Kraskov method).

        Parameters
        ----------
        data : numpy.ndarray
            Data points (n_samples x n_dimensions)
        k : int
            Number of nearest neighbors

        Returns
        -------
        entropy : float
            Estimated entropy in nats (natural log)

        Algorithm:
        ---------
        Uses Kraskov-Stögbauer-Grassberger (KSG) estimator:

        H(X) = ψ(N) - ψ(k) + ψ(k_i)

        where:
        - ψ = digamma function
        - N = number of samples
        - k = number of neighbors
        - k_i = number of points within k-th neighbor distance

        References:
        ----------
        Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating
        mutual information. Physical review E, 69(6), 066138.
        """
        from scipy.special import digamma

        n_samples, n_dims = data.shape

        if n_samples < k + 1:
            warnings.warn(
                f"Too few samples ({n_samples}) for k={k}. Returning 0.", UserWarning
            )
            return 0.0

        # Build KD-tree
        tree = cKDTree(data)

        # Find k-nearest neighbors for each point
        distances, _ = tree.query(data, k=k + 1)  # +1 to exclude self

        # Distance to k-th neighbor
        epsilon = distances[:, -1]

        # Add small constant to avoid log(0)
        epsilon = np.maximum(epsilon, 1e-10)

        # Estimate entropy using KSG estimator
        # H(X) ≈ -ψ(k) + ψ(N) + log(c_d) + (d/N) * Σ log(ε_i)
        # where c_d is the volume of d-dimensional unit ball

        # Volume of d-dimensional unit ball
        from scipy.special import gamma

        c_d = (np.pi ** (n_dims / 2)) / gamma(n_dims / 2 + 1)

        # Average log distance
        avg_log_dist = np.mean(np.log(epsilon))

        # KSG entropy estimate
        entropy = -digamma(k) + digamma(n_samples) + np.log(c_d) + n_dims * avg_log_dist

        return entropy

    def _estimate_mutual_information_knn(
        self, x: np.ndarray, y: np.ndarray, k: int
    ) -> float:
        """
        Estimate mutual information using KNN.

        Parameters
        ----------
        x : numpy.ndarray
            First variable (n_samples x dim_x)
        y : numpy.ndarray
            Second variable (n_samples x dim_y)
        k : int
            Number of neighbors

        Returns
        -------
        mi : float
            Mutual information estimate

        Formula:
        -------
        I(X;Y) = H(X) + H(Y) - H(X,Y)

        where H is entropy and (X,Y) is joint distribution.
        """
        # Reshape if 1D
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Joint distribution
        xy = np.hstack([x, y])

        # Estimate entropies
        h_x = self._estimate_entropy_knn(x, k)
        h_y = self._estimate_entropy_knn(y, k)
        h_xy = self._estimate_entropy_knn(xy, k)

        # Mutual information
        mi = h_x + h_y - h_xy

        # Ensure non-negative (numerical precision)
        mi = max(0.0, mi)

        return mi

    def compute_transfer_entropy(self) -> float:
        """
        Compute transfer entropy from source to target.

        Returns
        -------
        te : float
            Transfer entropy value in nats

        Formula:
        -------
        TE(X→Y) = I(Y_future; X_past | Y_past)

        More formally:
        TE(X→Y) = H(Y_t | Y_past) - H(Y_t | Y_past, X_past)

        where:
        - Y_t = target at time t
        - Y_past = k past values of target
        - X_past = l past values of source

        Algorithm Steps:
        ---------------
        1. Create embeddings for target history (k values)
        2. Create embeddings for source history (l values)
        3. Extract future target values
        4. Compute conditional mutual information
        5. Return TE estimate

        Examples:
        --------
        >>> te_analyzer = TransferEntropy(x, y, k=1, l=1)
        >>> te_value = te_analyzer.compute_transfer_entropy()
        >>>
        >>> # Convert nats to bits
        >>> te_bits = te_value / np.log(2)
        >>> print(f"TE: {te_bits:.4f} bits")

        Clinical Interpretation:
        -----------------------
        - **Cardio-respiratory:**
            - Healthy: Moderate bidirectional coupling
            - Sleep apnea: Reduced respiratory → cardiac TE
            - Heart failure: Altered coupling patterns

        - **Brain-heart:**
            - Mental stress: Increased brain → heart TE
            - Relaxation: Reduced directional coupling

        Notes:
        -----
        - Returns value in nats (natural logarithm base)
        - Convert to bits by dividing by ln(2)
        - Significance should be tested with surrogate data
        """
        # Create embeddings
        # Target history: Y(t-delay), Y(t-2*delay), ..., Y(t-k*delay)
        target_past = self._create_embedding(self.target[:-1], self.k, self.delay)

        # Source history: X(t-delay), X(t-2*delay), ..., X(t-l*delay)
        source_past = self._create_embedding(self.source[:-1], self.l, self.delay)

        # Align to same time points
        max_lookback = max(self.k, self.l) * self.delay
        target_past = target_past[max_lookback - self.k * self.delay :]
        source_past = source_past[max_lookback - self.l * self.delay :]

        # Future target: Y(t)
        target_future = self.target[max_lookback:].reshape(-1, 1)

        # Ensure same length
        min_len = min(len(target_past), len(source_past), len(target_future))
        target_past = target_past[:min_len]
        source_past = source_past[:min_len]
        target_future = target_future[:min_len]

        # Compute Transfer Entropy
        # TE = I(Y_future; X_past | Y_past)
        #    = I(Y_future, X_past; Y_past) - I(X_past; Y_past)

        # Method 1: Using conditional MI
        # TE = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

        # Joint of target past and future
        y_past_future = np.hstack([target_past, target_future])

        # Joint of target past, source past, and target future
        y_x_past_future = np.hstack([target_past, source_past, target_future])

        # Estimate entropies
        h_y_future_given_y_past = self._estimate_entropy_knn(
            y_past_future, self.k_neighbors
        ) - self._estimate_entropy_knn(target_past, self.k_neighbors)

        h_y_future_given_y_x_past = self._estimate_entropy_knn(
            y_x_past_future, self.k_neighbors
        ) - self._estimate_entropy_knn(
            np.hstack([target_past, source_past]), self.k_neighbors
        )

        # Transfer Entropy
        te = h_y_future_given_y_past - h_y_future_given_y_x_past

        # Ensure non-negative
        te = max(0.0, te)

        return te

    def compute_bidirectional_te(self) -> Tuple[float, float]:
        """
        Compute transfer entropy in both directions.

        Returns
        -------
        te_forward : float
            TE from source to target
        te_backward : float
            TE from target to source

        Examples:
        --------
        >>> te = TransferEntropy(resp, hr)
        >>> te_resp_hr, te_hr_resp = te.compute_bidirectional_te()
        >>>
        >>> # Net directional coupling
        >>> net_coupling = te_resp_hr - te_hr_resp
        >>> dominant_direction = "Resp → HR" if net_coupling > 0 else "HR → Resp"
        >>> print(f"Dominant direction: {dominant_direction}")
        >>> print(f"Coupling asymmetry: {abs(net_coupling):.4f}")

        Interpretation:
        --------------
        Comparing bidirectional TE reveals:

        1. **Dominant Direction:**
           - TE(X→Y) >> TE(Y→X): X drives Y
           - TE(X→Y) << TE(Y→X): Y drives X
           - TE(X→Y) ≈ TE(Y→X): Bidirectional or common drive

        2. **Coupling Strength:**
           - Sum = TE(X→Y) + TE(Y→X): Total coupling
           - Difference = |TE(X→Y) - TE(Y→X)|: Directional asymmetry
        """
        # Forward: source → target
        te_forward = self.compute_transfer_entropy()

        # Backward: target → source (swap signals)
        te_backward_analyzer = TransferEntropy(
            self.target,
            self.source,
            k=self.l,
            l=self.k,  # Swap k and l
            delay=self.delay,
            k_neighbors=self.k_neighbors,
        )
        te_backward = te_backward_analyzer.compute_transfer_entropy()

        return te_forward, te_backward

    def compute_time_delayed_te(self, max_delay: int = 10) -> np.ndarray:
        """
        Compute transfer entropy across multiple time delays.

        Parameters
        ----------
        max_delay : int
            Maximum time delay to test

        Returns
        -------
        te_values : numpy.ndarray
            TE values for each delay (length: max_delay)

        Purpose:
        -------
        Different physiological processes operate at different time scales.
        Time-delayed TE reveals the temporal dynamics of coupling.

        Examples:
        --------
        >>> te = TransferEntropy(source, target)
        >>> te_delays = te.compute_time_delayed_te(max_delay=20)
        >>>
        >>> # Find optimal delay
        >>> optimal_delay = np.argmax(te_delays) + 1
        >>> print(f"Peak coupling at delay: {optimal_delay}")
        >>>
        >>> # Plot delay profile
        >>> import matplotlib.pyplot as plt
        >>> delays = np.arange(1, 21)
        >>> plt.plot(delays, te_delays, 'o-')
        >>> plt.xlabel('Time Delay')
        >>> plt.ylabel('Transfer Entropy')
        >>> plt.title('TE vs Time Delay')
        >>> plt.grid(True)

        Clinical Significance:
        ---------------------
        - **Short delays (1-3):** Immediate physiological responses
        - **Medium delays (5-10):** Regulatory mechanisms
        - **Long delays (>10):** Slow adaptive processes
        """
        te_values = []

        for delay_val in range(1, max_delay + 1):
            # Create TE analyzer with this delay
            te_analyzer = TransferEntropy(
                self.source,
                self.target,
                k=self.k,
                l=self.l,
                delay=delay_val,
                k_neighbors=self.k_neighbors,
            )

            try:
                te = te_analyzer.compute_transfer_entropy()
                te_values.append(te)
            except Exception as e:
                warnings.warn(
                    f"Failed to compute TE at delay {delay_val}: {str(e)}. Using 0.",
                    UserWarning,
                )
                te_values.append(0.0)

        return np.array(te_values)

    def compute_effective_te(self) -> float:
        """
        Compute normalized effective transfer entropy.

        Returns
        -------
        effective_te : float
            Normalized TE in range [0, 1]

        Formula:
        -------
        Effective TE = TE / H(target_future | target_past)

        Normalization provides:
        - Scale-independent measure
        - Interpretability as fraction of uncertainty reduced
        - Easier comparison across different signal pairs

        Examples:
        --------
        >>> te_analyzer = TransferEntropy(x, y)
        >>> eff_te = te_analyzer.compute_effective_te()
        >>> print(f"Effective TE: {eff_te:.2%}")
        """
        # Compute TE
        te = self.compute_transfer_entropy()

        # Compute H(Y_future | Y_past) for normalization
        target_past = self._create_embedding(self.target[:-1], self.k, self.delay)
        max_lookback = self.k * self.delay
        target_future = self.target[max_lookback:].reshape(-1, 1)

        min_len = min(len(target_past), len(target_future))
        target_past = target_past[:min_len]
        target_future = target_future[:min_len]

        y_past_future = np.hstack([target_past, target_future])

        h_y_future_given_y_past = self._estimate_entropy_knn(
            y_past_future, self.k_neighbors
        ) - self._estimate_entropy_knn(target_past, self.k_neighbors)

        # Normalize
        if h_y_future_given_y_past > 0:
            effective_te = te / h_y_future_given_y_past
        else:
            effective_te = 0.0

        # Clip to [0, 1]
        effective_te = np.clip(effective_te, 0.0, 1.0)

        return effective_te

    def test_significance(
        self, n_surrogates: int = 100, method: str = "shuffle"
    ) -> Tuple[float, float]:
        """
        Test statistical significance of transfer entropy.

        Parameters
        ----------
        n_surrogates : int
            Number of surrogate datasets
        method : str
            Surrogate generation method
            - 'shuffle': Random permutation (destroys temporal structure)
            - 'phase': Phase randomization (preserves power spectrum)

        Returns
        -------
        p_value : float
            Statistical significance (0-1)
        te_original : float
            Original TE value

        Algorithm:
        ---------
        1. Compute TE for original data
        2. Generate n_surrogates by shuffling source signal
        3. Compute TE for each surrogate
        4. p-value = fraction of surrogates with TE >= original TE

        Examples:
        --------
        >>> te = TransferEntropy(x, y)
        >>> p_value, te_value = te.test_significance(n_surrogates=1000)
        >>>
        >>> if p_value < 0.05:
        ...     print(f"Significant coupling (p={p_value:.4f})")
        >>> else:
        ...     print(f"No significant coupling (p={p_value:.4f})")

        Notes:
        -----
        - p < 0.05: Significant coupling
        - p < 0.01: Highly significant
        - More surrogates = more reliable p-value
        - Computationally expensive for large n_surrogates
        """
        # Compute original TE
        te_original = self.compute_transfer_entropy()

        # Generate surrogates and compute TE
        surrogate_tes = []

        for _ in range(n_surrogates):
            if method == "shuffle":
                # Shuffle source signal
                surrogate_source = np.random.permutation(self.source)
            elif method == "phase":
                # Phase randomization (preserves power spectrum)
                fft = np.fft.fft(self.source)
                phases = np.random.uniform(0, 2 * np.pi, len(fft))
                fft_randomized = np.abs(fft) * np.exp(1j * phases)
                surrogate_source = np.real(np.fft.ifft(fft_randomized))
            else:
                raise ValueError(f"Unknown method: {method}")

            # Compute TE for surrogate
            te_surrogate_analyzer = TransferEntropy(
                surrogate_source,
                self.target,
                k=self.k,
                l=self.l,
                delay=self.delay,
                k_neighbors=self.k_neighbors,
            )

            try:
                te_surrogate = te_surrogate_analyzer.compute_transfer_entropy()
                surrogate_tes.append(te_surrogate)
            except Exception as e:
                warnings.warn(
                    f"Failed to compute TE for surrogate: {str(e)}. Using 0.",
                    UserWarning,
                )
                continue

        # Compute p-value
        if surrogate_tes:
            surrogate_tes = np.array(surrogate_tes)
            p_value = np.mean(surrogate_tes >= te_original)
        else:
            warnings.warn(
                "All surrogate calculations failed. Cannot compute p-value.",
                UserWarning,
            )
            p_value = 1.0

        return p_value, te_original


# Export main class
__all__ = ["TransferEntropy"]
