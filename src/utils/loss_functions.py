import numpy as np

class LossFunctions:
    """
    A class that provides a collection of common loss functions for signal processing.

    Methods:
    - mse: Mean Square Error loss.
    - mae: Mean Absolute Error loss.
    - huber: Huber loss, a combination of MSE and MAE.
    - smooth_l1: Smooth L1 loss, similar to Huber but with a different formulation.
    - log_cosh: Log-Cosh loss, a smooth approximation to the absolute error.
    - quantile: Quantile loss, useful for predicting quantiles.
    - custom_loss: Accepts a custom loss function provided by the user.
    """

    def mse(self, signal, target):
        """
        Compute the Mean Square Error (MSE) between the signal and target.

        Parameters:
        signal (numpy.ndarray): The input signal.
        target (numpy.ndarray): The target signal.

        Returns:
        float: The computed MSE value.

        Example:
        >>> lf = LossFunctions()
        >>> signal = np.array([1, 2, 3])
        >>> target = np.array([1.5, 2.5, 3.5])
        >>> print(lf.mse(signal, target))
        """
        return np.mean((signal - target) ** 2)

    def mae(self, signal, target):
        """
        Compute the Mean Absolute Error (MAE) between the signal and target.

        Parameters:
        signal (numpy.ndarray): The input signal.
        target (numpy.ndarray): The target signal.

        Returns:
        float: The computed MAE value.

        Example:
        >>> lf = LossFunctions()
        >>> signal = np.array([1, 2, 3])
        >>> target = np.array([1.5, 2.5, 3.5])
        >>> print(lf.mae(signal, target))
        """
        return np.mean(np.abs(signal - target))

    def huber(self, signal, target, delta=1.0):
        """
        Compute the Huber loss between the signal and target.

        Huber loss is less sensitive to outliers than MSE but behaves similarly to MSE for small errors.

        Parameters:
        signal (numpy.ndarray): The input signal.
        target (numpy.ndarray): The target signal.
        delta (float): The threshold where the loss transitions from MSE to MAE. Default is 1.0.

        Returns:
        float: The computed Huber loss value.

        Example:
        >>> lf = LossFunctions()
        >>> signal = np.array([1, 2, 3])
        >>> target = np.array([1.5, 2.5, 3.5])
        >>> print(lf.huber(signal, target, delta=1.0))
        """
        error = signal - target
        is_small_error = np.abs(error) <= delta
        squared_loss = 0.5 * error ** 2
        linear_loss = delta * (np.abs(error) - 0.5 * delta)
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))

    def smooth_l1(self, signal, target, beta=1.0):
        """
        Compute the Smooth L1 loss between the signal and target.

        Similar to Huber loss but with a different transition formula.

        Parameters:
        signal (numpy.ndarray): The input signal.
        target (numpy.ndarray): The target signal.
        beta (float): Transition point from quadratic to linear. Default is 1.0.

        Returns:
        float: The computed Smooth L1 loss value.

        Example:
        >>> lf = LossFunctions()
        >>> signal = np.array([1, 2, 3])
        >>> target = np.array([1.5, 2.5, 3.5])
        >>> print(lf.smooth_l1(signal, target, beta=1.0))
        """
        error = np.abs(signal - target)
        loss = np.where(error < beta, 0.5 * (error ** 2) / beta, error - 0.5 * beta)
        return np.mean(loss)

    def log_cosh(self, signal, target):
        """
        Compute the Log-Cosh loss between the signal and target.

        Log-Cosh is a smooth approximation to the absolute error and is less sensitive to outliers.

        Parameters:
        signal (numpy.ndarray): The input signal.
        target (numpy.ndarray): The target signal.

        Returns:
        float: The computed Log-Cosh loss value.

        Example:
        >>> lf = LossFunctions()
        >>> signal = np.array([1, 2, 3])
        >>> target = np.array([1.5, 2.5, 3.5])
        >>> print(lf.log_cosh(signal, target))
        """
        return np.mean(np.log(np.cosh(signal - target)))

    def quantile(self, signal, target, quantile=0.5):
        """
        Compute the Quantile loss between the signal and target.

        Quantile loss is useful for quantile regression, where the goal is to predict a particular quantile.

        Parameters:
        signal (numpy.ndarray): The input signal.
        target (numpy.ndarray): The target signal.
        quantile (float): The quantile to predict (0 < quantile < 1).

        Returns:
        float: The computed Quantile loss value.

        Example:
        >>> lf = LossFunctions()
        >>> signal = np.array([1, 2, 3])
        >>> target = np.array([1.5, 2.5, 3.5])
        >>> print(lf.quantile(signal, target, quantile=0.5))
        """
        error = target - signal
        return np.mean(np.maximum(quantile * error, (quantile - 1) * error))

    def custom_loss(self, loss_func):
        """
        Use a custom loss function provided by the user.

        Parameters:
        loss_func (callable): The custom loss function.

        Returns:
        callable: The custom loss function itself.

        Example:
        >>> def custom_lf(signal, target): return np.sum((signal - target) ** 2)
        >>> lf = LossFunctions()
        >>> loss = lf.custom_loss(custom_lf)
        >>> signal = np.array([1, 2, 3])
        >>> target = np.array([1.5, 2.5, 3.5])
        >>> print(loss(signal, target))
        """
        return loss_func
