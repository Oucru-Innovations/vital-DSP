"""
Utility Functions Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.signal_processing.loss_functions import LossFunctions
    >>> signal = np.random.randn(1000)
    >>> processor = LossFunctions(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""


import numpy as np


class LossFunctions:
    """
    A class that provides a collection of common loss functions for signal processing.

    This class includes various loss functions commonly used in regression tasks and signal processing. These loss functions measure the discrepancy between predicted and actual values, helping to optimize models during training.

    Methods
    -------
    mse(signal, target)
        Compute the Mean Square Error (MSE) loss.
    mae(signal, target)
        Compute the Mean Absolute Error (MAE) loss.
    huber(signal, target, delta=1.0)
        Compute the Huber loss, combining MSE and MAE.
    smooth_l1(signal, target, beta=1.0)
        Compute the Smooth L1 loss, similar to Huber loss with a different formulation.
    log_cosh(signal, target)
        Compute the Log-Cosh loss, a smooth approximation to the absolute error.
    quantile(signal, target, quantile=0.5)
        Compute the Quantile loss, useful for quantile regression.
    custom_loss(loss_func)
        Use a custom loss function provided by the user.
    """

    def mse(self, signal, target):
        """
        Compute the Mean Square Error (MSE) between the signal and target.

        The MSE loss is a commonly used loss function for regression tasks, penalizing larger errors more heavily by squaring the difference between the predicted and actual values.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal or predictions.
        target : numpy.ndarray
            The target signal or true values.

        Returns
        -------
        float
            The computed MSE value.

        Examples
        --------
        >>> lf = LossFunctions()
        >>> signal = np.array([1, 2, 3])
        >>> target = np.array([1.5, 2.5, 3.5])
        >>> print(lf.mse(signal, target))
        0.25
        """
        return np.mean((signal - target) ** 2)

    def mae(self, signal, target):
        """
        Compute the Mean Absolute Error (MAE) between the signal and target.

        The MAE loss is a robust measure of the average magnitude of errors in a set of predictions, without considering their direction.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal or predictions.
        target : numpy.ndarray
            The target signal or true values.

        Returns
        -------
        float
            The computed MAE value.

        Examples
        --------
        >>> lf = LossFunctions()
        >>> signal = np.array([1, 2, 3])
        >>> target = np.array([1.5, 2.5, 3.5])
        >>> print(lf.mae(signal, target))
        0.5
        """
        return np.mean(np.abs(signal - target))

    def huber(self, signal, target, delta=1.0):
        """
        Compute the Huber loss between the signal and target.

        Huber loss is a combination of MSE and MAE, being quadratic for small errors and linear for large errors. It is less sensitive to outliers compared to MSE.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal or predictions.
        target : numpy.ndarray
            The target signal or true values.
        delta : float, optional
            The threshold at which to switch from MSE to MAE (default is 1.0).

        Returns
        -------
        float
            The computed Huber loss value.

        Examples
        --------
        >>> lf = LossFunctions()
        >>> signal = np.array([1, 2, 3])
        >>> target = np.array([1.5, 2.5, 3.5])
        >>> print(lf.huber(signal, target, delta=1.0))
        0.125
        """
        error = signal - target
        is_small_error = np.abs(error) <= delta
        squared_loss = 0.5 * error**2
        linear_loss = delta * (np.abs(error) - 0.5 * delta)
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))

    def smooth_l1(self, signal, target, beta=1.0):
        """
        Compute the Smooth L1 loss between the signal and target.

        Smooth L1 loss, also known as the Huber loss in object detection, is less sensitive to outliers than MSE and provides a smooth transition between L1 and L2 losses.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal or predictions.
        target : numpy.ndarray
            The target signal or true values.
        beta : float, optional
            The transition point between quadratic and linear loss (default is 1.0).

        Returns
        -------
        float
            The computed Smooth L1 loss value.

        Examples
        --------
        >>> lf = LossFunctions()
        >>> signal = np.array([1, 2, 3])
        >>> target = np.array([1.5, 2.5, 3.5])
        >>> print(lf.smooth_l1(signal, target, beta=1.0))
        0.125
        """
        error = np.abs(signal - target)
        loss = np.where(error < beta, 0.5 * (error**2) / beta, error - 0.5 * beta)
        return np.mean(loss)

    def log_cosh(self, signal, target):
        """
        Compute the Log-Cosh loss between the signal and target.

        Log-Cosh loss is the logarithm of the hyperbolic cosine of the prediction error. It behaves similarly to MAE but is smoother near zero, which makes it less sensitive to outliers than MAE.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal or predictions.
        target : numpy.ndarray
            The target signal or true values.

        Returns
        -------
        float
            The computed Log-Cosh loss value.

        Examples
        --------
        >>> lf = LossFunctions()
        >>> signal = np.array([1, 2, 3])
        >>> target = np.array([1.5, 2.5, 3.5])
        >>> print(lf.log_cosh(signal, target))
        0.120114
        """
        return np.mean(np.log(np.cosh(signal - target)))

    def quantile(self, signal, target, quantile=0.5):
        """
        Compute the Quantile loss between the signal and target.

        Quantile loss is used in quantile regression to predict the quantiles of the target variable. The loss is asymmetric and assigns different penalties to overestimates and underestimates.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal or predictions.
        target : numpy.ndarray
            The target signal or true values.
        quantile : float, optional
            The quantile to predict (0 < quantile < 1) (default is 0.5).

        Returns
        -------
        float
            The computed Quantile loss value.

        Examples
        --------
        >>> lf = LossFunctions()
        >>> signal = np.array([1, 2, 3])
        >>> target = np.array([1.5, 2.5, 3.5])
        >>> print(lf.quantile(signal, target, quantile=0.5))
        0.25
        """
        error = target - signal
        return np.mean(np.maximum(quantile * error, (quantile - 1) * error))

    def custom_loss(self, loss_func):
        """
        Use a custom loss function provided by the user.

        This method allows the user to define their own loss function, which can be passed as a callable to this method.

        Parameters
        ----------
        loss_func : callable
            The custom loss function provided by the user.

        Returns
        -------
        callable
            The custom loss function itself, which can be applied to a signal and target.

        Examples
        --------
        >>> def custom_lf(signal, target): return np.sum((signal - target) ** 2)
        >>> lf = LossFunctions()
        >>> loss = lf.custom_loss(custom_lf)
        >>> signal = np.array([1, 2, 3])
        >>> target = np.array([1.5, 2.5, 3.5])
        >>> print(loss(signal, target))
        0.75
        """
        return loss_func
