"""
Advanced Computation Module for Physiological Signal Processing

This module provides Bayesian inference and optimization capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.1

Key Features:
- Gaussian Process modeling for signal prediction
- Bayesian Optimization for hyperparameter tuning
- Expected Improvement acquisition function
- Numerical stability improvements
- Graceful error handling

Classes:
--------
GaussianProcess
    Implements Gaussian Process regression for probabilistic modeling.
BayesianOptimization
    Implements Bayesian Optimization using Gaussian Processes.

Examples:
---------
**Example 1: Gaussian Process Prediction**

    >>> import numpy as np
    >>> from vitalDSP.advanced_computation.bayesian_analysis import GaussianProcess
    >>>
    >>> # Create GP model
    >>> gp = GaussianProcess(length_scale=1.0, noise=1e-5)
    >>>
    >>> # Training data
    >>> X_train = np.array([[0.1], [0.4], [0.7]])
    >>> y_train = np.sin(3 * X_train.flatten())
    >>> gp.update(X_train, y_train)
    >>>
    >>> # Predict at new points
    >>> X_new = np.array([[0.2], [0.5]])
    >>> mean, variance = gp.predict(X_new)
    >>> print(f"Predicted mean: {mean}")
    >>> print(f"Predicted variance: {variance}")

**Example 2: Bayesian Optimization for Hyperparameter Tuning**

    >>> from vitalDSP.advanced_computation.bayesian_analysis import BayesianOptimization
    >>>
    >>> # Define objective function to maximize
    >>> def filter_performance(cutoff_freq):
    ...     # Simulate filter performance metric
    ...     cutoff = np.atleast_1d(cutoff_freq)[0]
    ...     return -(cutoff - 0.5)**2 + 1.0  # Peak at 0.5
    >>>
    >>> # Optimize cutoff frequency in range [0, 1]
    >>> optimizer = BayesianOptimization(
    ...     func=filter_performance,
    ...     bounds=(0, 1),
    ...     length_scale=0.1,
    ...     noise=1e-5
    ... )
    >>>
    >>> # Run optimization
    >>> best_cutoff, best_performance = optimizer.optimize(n_iter=20, random_seed=42)
    >>> print(f"Best cutoff frequency: {best_cutoff[0]:.4f}")
    >>> print(f"Best performance: {best_performance:.4f}")

**Example 3: ECG Filter Parameter Optimization**

    >>> from vitalDSP.filtering.signal_filtering import SignalFiltering
    >>>
    >>> # Simulated ECG signal
    >>> ecg_signal = np.sin(2*np.pi*1.2*np.linspace(0, 10, 1000)) + 0.1*np.random.randn(1000)
    >>>
    >>> # Define objective: minimize noise while preserving signal
    >>> def filter_quality(params):
    ...     cutoff = np.atleast_1d(params)[0]
    ...     sf = SignalFiltering(ecg_signal)
    ...     filtered = sf.butterworth_lowpass(cutoff_freq=cutoff, fs=100, order=4)
    ...     # Quality metric: SNR-like measure
    ...     signal_power = np.var(filtered)
    ...     noise_power = np.var(ecg_signal - filtered)
    ...     return signal_power / (noise_power + 1e-10)
    >>>
    >>> # Optimize filter cutoff
    >>> optimizer = BayesianOptimization(filter_quality, bounds=(0.1, 10))
    >>> best_cutoff, best_snr = optimizer.optimize(n_iter=15, random_seed=42)
    >>> print(f"Optimal cutoff: {best_cutoff[0]:.2f} Hz, SNR: {best_snr:.2f}")

Notes:
------
- **Numerical Stability**: The default noise parameter (1e-5) provides good numerical
  stability. Avoid using values smaller than 1e-7 as they can cause singular matrix errors.
- **Initialization**: BayesianOptimization starts with 3 random samples to bootstrap
  the Gaussian Process before beginning optimization.
- **Fallback**: If numerical issues occur during optimization, the algorithm falls
  back to random sampling to ensure it always returns a result.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf


class GaussianProcess:
    """
    A class implementing Gaussian Process (GP) for Bayesian Optimization.

    Gaussian Process models are used to predict the mean and variance of an unknown function based on observed data. This is particularly useful in Bayesian Optimization, where the GP helps in selecting the next sample point to evaluate.

    Methods
    -------
    predict(X)
        Predict the mean and variance of the objective function at new points X.
    update(X_train, y_train)
        Update the GP model with new observations.

    Example Usage
    -------------
    >>> gp = GaussianProcess(length_scale=1.0, noise=1e-10)
    >>> X_train = np.array([[0.1], [0.4], [0.7]])
    >>> y_train = np.sin(3 * X_train) - X_train ** 2 + 0.7 * X_train
    >>> gp.update(X_train, y_train)
    >>> X_new = np.array([[0.2], [0.5]])
    >>> mean, variance = gp.predict(X_new)
    >>> print("Predicted Mean:", mean)
    >>> print("Predicted Variance:", variance)
    """

    def __init__(self, length_scale=1.0, noise=1e-10):
        """
        Initialize the Gaussian Process model.

        Parameters
        ----------
        length_scale : float, optional
            The length scale parameter for the RBF kernel. Controls the smoothness of the function (default is 1.0).
        noise : float, optional
            The noise level in the observations. Used to regularize the covariance matrix (default is 1e-10).
        """
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K = None

    def _rbf_kernel(self, X1, X2):
        """
        Radial Basis Function (RBF) kernel, also known as the Gaussian kernel.

        Parameters
        ----------
        X1 : numpy.ndarray
            The first set of input points.
        X2 : numpy.ndarray
            The second set of input points.

        Returns
        -------
        numpy.ndarray
            The kernel matrix computed between X1 and X2.
        """
        sqdist = (
            np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        )
        return np.exp(-0.5 / self.length_scale**2 * sqdist)

    def predict(self, X):
        """
        Predict the mean and variance of the objective function at new points.

        Parameters
        ----------
        X : numpy.ndarray
            The points at which to predict the mean and variance.

        Returns
        -------
        tuple
            The predicted mean and variance at the input points X.

        Raises
        ------
        ValueError
            If the GP model has not been updated with any training data.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "The GP model has not been updated with any training data."
            )
        K_s = self._rbf_kernel(self.X_train, X)
        K_ss = self._rbf_kernel(X, X)
        K_inv = np.linalg.inv(self.K + self.noise**2 * np.eye(len(self.X_train)))

        mu_s = K_s.T.dot(K_inv).dot(self.y_train)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu_s.ravel(), np.diag(cov_s)

    def update(self, X_train, y_train):
        """
        Update the GP model with new observations.

        Parameters
        ----------
        X_train : numpy.ndarray
            The observed input points.
        y_train : numpy.ndarray
            The observed output values corresponding to X_train.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.K = self._rbf_kernel(X_train, X_train)


class BayesianOptimization:
    """
    A class implementing Bayesian Optimization for parameter tuning.

    Bayesian Optimization is an efficient method for finding the minimum or maximum of an objective function, especially when the function is expensive to evaluate. It uses a Gaussian Process model to predict the function's behavior and an acquisition function to determine the next point to sample.

    Methods
    -------
    optimize(n_iter=10, random_seed=None)
        Perform Bayesian optimization to find the best parameters.
    acquisition(X, xi=0.01)
        Compute the Expected Improvement (EI) at new points.
    propose_location(n_restarts=10)
        Propose the next sampling location based on the acquisition function.

    Example Usage
    -------------
    >>> def objective_function(x):
    >>>     return -np.sin(3 * x) - x ** 2 + 0.7 * x
    >>>
    >>> bayesian_optimizer = BayesianOptimization(objective_function, bounds=(0, 2))
    >>> best_x, best_y = bayesian_optimizer.optimize(n_iter=10)
    >>> print("Best X:", best_x)
    >>> print("Best Y:", best_y)
    """

    def __init__(self, func, bounds, length_scale=1.0, noise=1e-5):
        """
        Initialize BayesianOptimization with the objective function and bounds.

        Parameters
        ----------
        func : callable
            The objective function to be optimized.
        bounds : tuple
            The bounds within which to search for the optimal parameters.
        length_scale : float, optional
            Length scale parameter for the Gaussian Process kernel (default is 1.0).
        noise : float, optional
            Noise level in the observations for numerical stability (default is 1e-5).
            Larger values (1e-5 to 1e-3) provide better numerical stability.
        """
        self.func = func
        self.bounds = bounds
        self.gp = GaussianProcess(length_scale, noise)
        self.X_samples = []
        self.Y_samples = []

    def acquisition(self, X, xi=0.01):
        """
        Expected Improvement (EI) acquisition function.

        The EI acquisition function is used to balance exploration and exploitation in Bayesian Optimization. It selects points that have the potential to improve upon the best observed value.

        Parameters
        ----------
        X : numpy.ndarray
            The points at which to evaluate the acquisition function.
        xi : float, optional
            Exploration-exploitation trade-off parameter (default is 0.01).

        Returns
        -------
        numpy.ndarray
            The EI values at the provided points.
        """
        mu, sigma = self.gp.predict(X)
        mu_sample = np.max(self.Y_samples)
        imp = mu - mu_sample - xi

        # Avoid divide by zero
        sigma_safe = np.where(sigma == 0.0, 1e-10, sigma)
        Z = imp / sigma_safe

        ei = imp * self._cdf(Z) + sigma * self._pdf(Z)
        ei[sigma == 0.0] = 0.0

        return ei

    def _cdf(self, x):
        """
        Cumulative Distribution Function (CDF) for the standard normal distribution.

        Parameters
        ----------
        x : numpy.ndarray
            The input array.

        Returns
        -------
        numpy.ndarray
            The CDF values.
        """
        return 0.5 * (1 + erf(x / np.sqrt(2)))

    def _pdf(self, x):
        """
        Probability Density Function (PDF) for the standard normal distribution.

        Parameters
        ----------
        x : numpy.ndarray
            The input array.

        Returns
        -------
        numpy.ndarray
            The PDF values.
        """
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def propose_location(self, n_restarts=10):
        """
        Propose the next sampling location by optimizing the acquisition function.

        Parameters
        ----------
        n_restarts : int, optional
            Number of restarts for the optimization to avoid local optima (default is 10).

        Returns
        -------
        numpy.ndarray
            The proposed next sampling point.
        """
        best_value = None
        best_x = None

        for _ in range(n_restarts):
            x_init = np.random.uniform(
                self.bounds[0], self.bounds[1], size=(1,)
            ).flatten()  # Flatten x_init
            res = minimize(
                lambda x: -self.acquisition(np.atleast_2d(x)),
                x_init,
                bounds=[self.bounds],
            )
            if best_value is None or res.fun < best_value:
                best_value = res.fun
                best_x = res.x

        return best_x

    def optimize(self, n_iter=10, random_seed=None):
        """
        Perform Bayesian optimization to find the best parameters.

        Parameters
        ----------
        n_iter : int, optional
            Number of iterations for the optimization (default is 10).
        random_seed : int or None, optional
            Random seed for reproducibility (default is None).

        Returns
        -------
        tuple
            The best parameters and the corresponding function value.

        Notes
        -----
        The optimization process starts with a random sample to initialize the Gaussian Process,
        then iteratively selects new points using the acquisition function to balance exploration
        and exploitation.

        Examples
        --------
        >>> def objective(x):
        ...     return -np.sin(3 * x) - x ** 2 + 0.7 * x
        >>> optimizer = BayesianOptimization(objective, bounds=(0, 2))
        >>> best_x, best_y = optimizer.optimize(n_iter=10, random_seed=42)
        >>> print(f"Best X: {best_x}, Best Y: {best_y}")
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize with 2-3 random samples to bootstrap the GP
        # This prevents singular matrix issues
        # Normalize Y_samples to ensure all values are scalars
        # This handles cases where Y_samples might contain arrays from test setup
        normalized_Y_samples = []
        for y in self.Y_samples:
            if isinstance(y, (list, np.ndarray)):
                y = float(np.atleast_1d(y).flatten()[0])
            normalized_Y_samples.append(float(y))
        self.Y_samples = normalized_Y_samples

        if len(self.X_samples) == 0:
            n_init = 3  # Start with 3 random samples for stability
            for _ in range(n_init):
                X_init = np.random.uniform(
                    self.bounds[0], self.bounds[1], size=(1,)
                ).reshape(-1, 1)
                Y_init = self.func(X_init)
                self.X_samples.append(X_init)
                self.Y_samples.append(float(np.atleast_1d(Y_init).flatten()[0]))

        for _ in range(n_iter):
            try:
                # Convert samples to proper arrays for GP update
                X_array = np.vstack(self.X_samples)
                # Ensure all Y_samples are scalars before converting to array
                Y_scalars = []
                for y in self.Y_samples:
                    if isinstance(y, (list, np.ndarray)):
                        y = float(np.atleast_1d(y).flatten()[0])
                    Y_scalars.append(float(y))
                Y_array = np.array(Y_scalars).flatten()

                # Update GP with current samples
                self.gp.update(X_array, Y_array)

                # Propose next location using acquisition function
                X_next = self.propose_location()
                Y_next = self.func(X_next)

                # Append new sample to the lists
                self.X_samples.append(X_next.reshape(-1, 1))
                self.Y_samples.append(float(np.atleast_1d(Y_next).flatten()[0]))

            except np.linalg.LinAlgError:
                # If numerical issues occur, fall back to random sampling
                X_random = np.random.uniform(
                    self.bounds[0], self.bounds[1], size=(1,)
                ).reshape(-1, 1)
                Y_random = self.func(X_random)
                self.X_samples.append(X_random)
                self.Y_samples.append(float(np.atleast_1d(Y_random).flatten()[0]))

        # Find and return the best sample
        best_index = np.argmax(self.Y_samples)
        best_x = self.X_samples[best_index]
        best_y = self.Y_samples[best_index]

        # Ensure best_y is a numpy scalar (not Python float) for consistency with test expectations
        best_y = np.float64(best_y)

        return best_x.flatten(), best_y
