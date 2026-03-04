"""
Advanced Computation Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations
- Interactive visualization capabilities
- Comprehensive signal analysis

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.advanced_computation.non_linear_analysis import NonLinearAnalysis
    >>> signal = np.random.randn(1000)
    >>> processor = NonLinearAnalysis(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""

import numpy as np
import matplotlib.pyplot as plt


class NonlinearAnalysis:
    """
    Nonlinear Analysis for examining chaotic signals, such as Heart Rate Variability (HRV).

    This class provides methods for analyzing the chaotic behavior of signals using techniques
    like the estimation of Lyapunov exponents, generation of Poincaré plots, and calculation of
    correlation dimensions.

    Methods
    -------
    lyapunov_exponent(max_iter=1000, epsilon=1e-8)
        Estimates the largest Lyapunov exponent to assess chaos in the signal.
    poincare_plot()
        Generates a Poincaré plot to visualize the dynamics of the signal.
    correlation_dimension(radius=0.1)
        Estimates the correlation dimension of the signal.

    Example Usage
    -------------
    >>> signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
    >>> nonlinear_analysis = NonlinearAnalysis(signal)
    >>> lyapunov = nonlinear_analysis.lyapunov_exponent()
    >>> print("Lyapunov Exponent:", lyapunov)

    >>> nonlinear_analysis.poincare_plot()

    >>> correlation_dim = nonlinear_analysis.correlation_dimension()
    >>> print("Correlation Dimension:", correlation_dim)
    """

    def __init__(self, signal):
        """
        Initialize the NonlinearAnalysis class with the provided signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be analyzed for nonlinear dynamics.
        """
        self.signal = signal

    def lyapunov_exponent(self, max_iter=1000, epsilon=1e-8, normalize=True):
        """
        Estimate the largest Lyapunov exponent of the signal.

        The Lyapunov exponent is a measure of the rate of separation of infinitesimally
        close trajectories in a chaotic system. A positive Lyapunov exponent indicates chaos.

        **IMPORTANT:** For reliable results with physiological signals (ECG, PPG), signal
        normalization is highly recommended to avoid numerical instabilities.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations to compute the exponent (default is 1000).
        epsilon : float, optional
            Small perturbation used to calculate divergence, to avoid division by zero (default is 1e-8).
        normalize : bool, optional
            If True, automatically normalize the signal before computation (default is True).
            Normalization prevents divide-by-zero errors with signals containing flat segments.

        Returns
        -------
        float
            The estimated largest Lyapunov exponent.

        Raises
        ------
        ValueError
            If signal is too short for the specified max_iter.
        Warning
            If computation results in NaN or inf values.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
        >>> nonlinear_analysis = NonlinearAnalysis(signal)
        >>> lyapunov = nonlinear_analysis.lyapunov_exponent()
        >>> print("Lyapunov Exponent:", lyapunov)

        Notes
        -----
        Uses the Rosenstein et al. (1993) algorithm: delay-embeds the signal into a
        dim-dimensional phase space, finds the nearest neighbor for each reference point
        (with Theiler window to exclude temporal neighbors), tracks trajectory divergence,
        and fits the slope of the mean log-divergence curve.
        """
        import warnings

        n = len(self.signal)

        # Validate signal length
        if n <= max_iter:
            raise ValueError(
                f"Signal length ({n}) must be greater than max_iter ({max_iter}). "
                f"Either reduce max_iter or provide a longer signal."
            )

        # Normalize signal if requested (recommended for physiological signals)
        signal = self.signal.copy()
        if normalize:
            signal_mean = np.mean(signal)
            signal_std = np.std(signal)
            if signal_std > epsilon:
                signal = (signal - signal_mean) / signal_std
            else:
                warnings.warn(
                    "Signal has zero or very low variance. Lyapunov exponent may not be meaningful."
                )

        # Phase-space embedding parameters (Rosenstein algorithm)
        tau = 1   # Time delay
        dim = 2   # Embedding dimension
        min_sep = max(1, int(0.1 * n))  # Theiler window (minimum temporal separation)

        # Create delay-embedded phase space
        n_emb = n - (dim - 1) * tau
        if n_emb < max_iter + min_sep + 1:
            return np.nan

        embedded = np.array([signal[i:i + dim * tau:tau] for i in range(n_emb)])

        all_log_ratios = []
        skipped_points = 0

        for i in range(n_emb - max_iter):
            # Find nearest neighbor with Theiler exclusion
            best_dist = np.inf
            nn_idx = -1
            for j in range(n_emb):
                if abs(i - j) <= min_sep:
                    continue
                d = np.linalg.norm(embedded[i] - embedded[j])
                if d < best_dist and d > 0:
                    best_dist = d
                    nn_idx = j

            if nn_idx == -1 or best_dist < epsilon:
                skipped_points += 1
                continue

            # Track divergence of nearest-neighbor trajectory
            log_ratio = []
            for t in range(1, max_iter):
                if i + t >= n_emb or nn_idx + t >= n_emb:
                    break
                d_t = np.linalg.norm(embedded[i + t] - embedded[nn_idx + t])
                if d_t > 0:
                    log_ratio.append(np.log(d_t / best_dist))

            if len(log_ratio) > 0 and np.all(np.isfinite(log_ratio)):
                all_log_ratios.append(log_ratio)

        if len(all_log_ratios) == 0:
            warnings.warn(
                "No valid distance calculations obtained. Signal may have too many "
                "repeated values or be unsuitable for Lyapunov exponent estimation. "
                "Try normalizing the signal or using a different analysis method."
            )
            return np.nan

        if skipped_points > (n_emb - max_iter) * 0.5:
            warnings.warn(
                f"Skipped {skipped_points}/{n_emb - max_iter} points due to near-zero distances. "
                f"Results may be unreliable. Consider signal preprocessing."
            )

        # Pad shorter arrays with NaN for consistent shape
        max_len = max(len(r) for r in all_log_ratios)
        padded = np.full((len(all_log_ratios), max_len), np.nan)
        for i, r in enumerate(all_log_ratios):
            padded[i, :len(r)] = r

        avg_log_div = np.nanmean(padded, axis=0)
        time_indices = np.arange(1, len(avg_log_div) + 1)
        valid = ~np.isnan(avg_log_div) & ~np.isinf(avg_log_div)
        if np.sum(valid) < 2:
            return 0.0
        coeffs = np.polyfit(time_indices[valid], avg_log_div[valid], 1)
        lyapunov = coeffs[0]

        if not np.isfinite(lyapunov):
            warnings.warn(
                f"Lyapunov exponent is {lyapunov}. This indicates numerical issues. "
                f"Try signal normalization or different epsilon value."
            )

        return lyapunov

    def poincare_plot(self):
        """
        Generate a Poincaré plot to visualize the dynamics of the signal.

        The Poincaré plot is a scatter plot of the signal against its delayed version.
        It is often used to visualize periodic and chaotic dynamics in the signal.

        Returns
        -------
        matplotlib.figure.Figure
            The generated Poincaré plot.

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
        >>> nonlinear_analysis = NonlinearAnalysis(signal)
        >>> nonlinear_analysis.poincare_plot()
        """
        x = self.signal[:-1]
        y = self.signal[1:]

        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_xlabel(r"$x_{n}$")
        ax.set_ylabel(r"$x_{n+1}$")
        ax.set_title("Poincaré Plot")
        return fig

    def correlation_dimension(self, radius=0.1, normalize=True):
        """
        Estimate the correlation dimension of the signal using the Grassberger-Procaccia method.

        The correlation dimension is a measure of the fractal dimension of the attractor
        in the phase space of the signal. It provides insight into the complexity of the signal.

        **IMPORTANT:** Signal normalization is highly recommended for physiological signals
        to ensure the radius parameter is appropriate for the signal's scale.

        Parameters
        ----------
        radius : float, optional
            Radius within which points are considered neighbors (default is 0.1).
            For normalized signals, typical values are 0.1-2.0.
            For raw signals, choose radius based on signal scale.
        normalize : bool, optional
            If True, automatically normalize the signal before computation (default is True).
            Normalization ensures radius is appropriate regardless of signal amplitude.

        Returns
        -------
        float
            The estimated correlation dimension.

        Raises
        ------
        ValueError
            If radius is 1.0 (causes divide-by-zero), or if no point pairs found within radius.
        Warning
            If correlation dimension is negative (suggests inappropriate radius selection).

        Examples
        --------
        >>> signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.normal(size=100)
        >>> nonlinear_analysis = NonlinearAnalysis(signal)
        >>> correlation_dim = nonlinear_analysis.correlation_dimension(radius=0.5)
        >>> print("Correlation Dimension:", correlation_dim)

        Notes
        -----
        For physiological signals (ECG, PPG):
        1. Always normalize the signal (normalize=True, default)
        2. Use radius in range [0.1, 2.0] for normalized signals
        3. Avoid radius = 1.0 exactly (causes log(1.0) = 0)
        4. If negative result, try different radius or check signal quality

        The correlation dimension should typically be between 0 and the embedding dimension.
        Values outside this range suggest inappropriate parameters or unsuitable signal.
        """
        # Validate radius
        if np.abs(radius - 1.0) < 1e-10:
            raise ValueError(
                "Radius cannot be 1.0 because log(1.0) = 0, causing division by zero. "
                "Use radius != 1.0 (e.g., 0.9 or 1.1)."
            )

        if radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")

        n = len(self.signal)

        signal = self.signal.copy()
        if normalize:
            signal_mean = np.mean(signal)
            signal_std = np.std(signal)
            if signal_std > 1e-10:
                signal = (signal - signal_mean) / signal_std
            else:
                import warnings

                warnings.warn(
                    "Signal has zero or very low variance. "
                    "Correlation dimension may not be meaningful."
                )

        # Time-delay embedding
        tau = 1
        m = 2
        if n <= (m - 1) * tau:
            return 0.0
        embedded = np.array([signal[i:i + (m-1)*tau + 1:tau] for i in range(n - (m-1)*tau)])
        n_emb = len(embedded)

        # Compute C(r) at multiple radii and use slope in log-log space
        sig_std = np.std(signal)
        if sig_std < 1e-10:
            sig_std = 1.0
        radii = np.logspace(np.log10(0.1 * sig_std), np.log10(2 * sig_std), 10)
        log_C = []
        log_r = []
        for r in radii:
            count = 0
            for i in range(n_emb):
                for j in range(i + 1, n_emb):
                    if np.linalg.norm(embedded[i] - embedded[j]) < r:
                        count += 1
            C = 2.0 * count / (n_emb * (n_emb - 1)) if n_emb > 1 else 0
            if C > 0:
                log_C.append(np.log(C))
                log_r.append(np.log(r))

        if len(log_C) < 2:
            raise ValueError(
                f"No point pairs found within radius range. "
                f"Signal characteristics: mean={np.mean(signal):.4f}, std={np.std(signal):.4f}, "
                f"range=[{np.min(signal):.4f}, {np.max(signal):.4f}]. "
                f"Try: (1) Increasing radius, (2) Enabling normalization (normalize=True), "
                f"or (3) Using a longer signal segment."
            )

        slope = np.polyfit(log_r, log_C, 1)[0]

        if not np.isfinite(slope):
            import warnings

            warnings.warn(
                f"Correlation dimension is {slope}. This indicates numerical issues."
            )

        return slope
