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
        For ECG and PPG signals, normalization is essential due to flat segments and
        baseline wander that can cause divide-by-zero errors.
        """
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
                import warnings
                warnings.warn(
                    "Signal has zero or very low variance. Lyapunov exponent may not be meaningful."
                )

        distances = []
        skipped_points = 0

        for i in range(n - max_iter):
            dist = np.abs(signal[i + 1 : i + max_iter] - signal[i])

            # Critical fix: Check if dist[0] is too small BEFORE division
            if dist[0] < epsilon:
                # Skip this point to avoid divide-by-zero
                skipped_points += 1
                continue

            # Clamp other small values
            dist[dist < epsilon] = epsilon

            # Now safe to divide by dist[0]
            log_ratio = np.log(dist / dist[0])

            # Filter out any inf/nan values that may still occur
            log_ratio = log_ratio[np.isfinite(log_ratio)]

            if len(log_ratio) > 0:
                distances.append(log_ratio)

        # Check if we have enough valid data points
        if len(distances) == 0:
            import warnings
            warnings.warn(
                "No valid distance calculations obtained. Signal may have too many "
                "repeated values or be unsuitable for Lyapunov exponent estimation. "
                "Try normalizing the signal or using a different analysis method."
            )
            return np.nan

        if skipped_points > (n - max_iter) * 0.5:
            import warnings
            warnings.warn(
                f"Skipped {skipped_points}/{n-max_iter} points due to near-zero distances. "
                f"Results may be unreliable. Consider signal preprocessing."
            )

        # Flatten list of arrays and compute mean
        all_distances = np.concatenate(distances)
        lyapunov = np.mean(all_distances) / max_iter

        # Validate result
        if not np.isfinite(lyapunov):
            import warnings
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

        plt.scatter(x, y)
        plt.xlabel(r"$x_{n}$")
        plt.ylabel(r"$x_{n+1}$")
        plt.title("Poincaré Plot")
        plt.show()

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

        # Normalize signal if requested (recommended for physiological signals)
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

        # Count point pairs within radius
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(signal[i] - signal[j]) < radius:
                    count += 1

        # Validate we found some pairs
        if count == 0:
            raise ValueError(
                f"No point pairs found within radius {radius}. "
                f"Signal characteristics: mean={np.mean(signal):.4f}, std={np.std(signal):.4f}, "
                f"range=[{np.min(signal):.4f}, {np.max(signal):.4f}]. "
                f"Try: (1) Increasing radius, (2) Enabling normalization (normalize=True), "
                f"or (3) Using a longer signal segment."
            )

        # Compute correlation dimension
        log_count = np.log(count)
        log_radius = np.log(radius)

        # Additional safety check (should not occur given earlier validation)
        if np.abs(log_radius) < 1e-10:
            raise ValueError(
                f"log(radius) = {log_radius} is too close to zero. "
                f"This should not occur with radius != 1.0"
            )

        correlation_dim = log_count / log_radius

        # Validate result
        if correlation_dim < 0:
            import warnings
            warnings.warn(
                f"Negative correlation dimension ({correlation_dim:.4f}) suggests "
                f"inappropriate radius selection or unsuitable signal. "
                f"This occurs when log(count)/log(radius) < 0. "
                f"Recommendations: "
                f"(1) For radius < 1: if few pairs found, increase radius; "
                f"(2) For radius > 1: if many pairs found, decrease radius; "
                f"(3) Try radius in range [0.1, 2.0] for normalized signals; "
                f"(4) Ensure signal is normalized (normalize=True)."
            )

        if not np.isfinite(correlation_dim):
            import warnings
            warnings.warn(
                f"Correlation dimension is {correlation_dim}. This indicates numerical issues."
            )

        return correlation_dim
