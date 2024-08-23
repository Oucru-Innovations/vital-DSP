import numpy as np
from utils.loss_functions import LossFunctions
from utils.convolutional_kernels import ConvolutionKernels
from utils.attention_weights import AttentionWeights

class AdvancedSignalFiltering:
    """
    A class for applying advanced filtering techniques to signals.

    This class implements a variety of advanced filtering methods, including Kalman filters,
    optimization-based filtering, gradient descent filtering, ensemble filtering, convolution-based
    filtering, and attention-based filtering. These methods are designed to enhance signal processing
    in complex scenarios such as real-time filtering, adaptive filtering, and filtering in noisy environments.

    Methods
    -------
    kalman_filter(R=1, Q=1)
        Applies a Kalman filter for real-time signal estimation and noise reduction.
    optimization_based_filtering(target, loss_type="mse", custom_loss_func=None, initial_guess=0, learning_rate=0.01, iterations=100)
        Applies filtering based on a custom or predefined loss function using optimization techniques.
    gradient_descent_filter(target, learning_rate=0.01, iterations=100)
        Uses gradient descent for adaptive filtering to reduce the difference between the signal and a target.
    ensemble_filtering(filters, method="mean", weights=None, num_iterations=10, learning_rate=0.01)
        Combines multiple filters using ensemble methods such as mean, weighted mean, bagging, or boosting.
    convolution_based_filter(kernel_type="smoothing", custom_kernel=None, kernel_size=3)
        Applies convolutional filtering using predefined or custom kernels.
    attention_based_filter(attention_type="uniform", custom_weights=None, size=5, **kwargs)
        Uses attention mechanisms for dynamic filtering based on various attention schemes.
    """

    def __init__(self, signal):
        """
        Initialize the AdvancedSignalFiltering class with the signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input signal to be filtered.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> af = AdvancedSignalFiltering(signal)
        """
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)
        self.signal = signal

    def kalman_filter(self, R=1, Q=1):
        """
        Apply a Kalman filter to the signal.

        The Kalman filter provides real-time signal estimation and noise reduction, making it suitable
        for scenarios where signal measurements are noisy and need continuous estimation.

        Parameters
        ----------
        R : float
            Measurement noise covariance. A lower value assumes more trust in the measurements.
        Q : float
            Process noise covariance. A lower value assumes less uncertainty in the model.

        Returns
        -------
        filtered_signal : numpy.ndarray
            The filtered signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 2, 1, 2, 3, 4, 3, 2])
        >>> af = AdvancedSignalFiltering(signal)
        >>> filtered_signal = af.kalman_filter(R=0.1, Q=0.01)
        >>> print(filtered_signal)
        [1.  1.75 2.5  2.12 1.56 1.94 2.7  3.5  3.12 2.56]
        """
        n = len(self.signal)
        xhat = np.zeros(n)  # Estimated state
        P = np.zeros(n)  # Estimated covariance
        xhat[0] = self.signal[0]
        P[0] = 1.0

        for k in range(1, n):
            # Prediction update
            xhatminus = xhat[k - 1]
            Pminus = P[k - 1] + Q

            # Measurement update
            K = Pminus / (Pminus + R)
            xhat[k] = xhatminus + K * (self.signal[k] - xhatminus)
            P[k] = (1 - K) * Pminus

        return xhat

    def optimization_based_filtering(
        self,
        target,
        loss_type="mse",
        custom_loss_func=None,
        initial_guess=0,
        learning_rate=0.01,
        iterations=100,
    ):
        """
        Apply an optimization-based filter using a custom or predefined loss function.

        This method uses optimization techniques to filter the signal by minimizing a specified loss
        function. It can be customized to use different types of loss functions, such as mean squared error (MSE),
        mean absolute error (MAE), Huber loss, and more, depending on the specific needs of the signal processing task.

        Parameters
        ----------
        target : numpy.ndarray
            The target signal to compare against. The filter will attempt to make the signal closer to this target.
        loss_type : str
            The type of loss function to use ('mse', 'mae', 'huber', 'smooth_l1', 'log_cosh', 'quantile', or 'custom').
        custom_loss_func : callable, optional
            A custom loss function provided by the user if 'custom' is selected for loss_type.
        initial_guess : float
            Initial guess for the filter parameter, used as a starting point in optimization.
        learning_rate : float
            Learning rate for the optimization, determining the step size of each iteration.
        iterations : int
            Number of iterations for the optimization process.

        Returns
        -------
        filtered_signal : numpy.ndarray
            The filtered signal after optimization.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 2, 1, 2, 3, 4, 3, 2])
        >>> target = np.array([1, 1.5, 2.5, 3, 2.5, 2, 2.5, 3.5, 3, 2.5])
        >>> af = AdvancedSignalFiltering(signal)
        >>> filtered_signal = af.optimization_based_filtering(target, loss_type='mse', initial_guess=2, learning_rate=0.01, iterations=100)
        >>> print(filtered_signal)
        """
        lf = LossFunctions()

        if loss_type == "mse":
            loss_func = lf.mse
        elif loss_type == "mae":
            loss_func = lf.mae
        elif loss_type == "huber":
            loss_func = lambda signal, target: lf.huber(signal, target, delta=1.0)
        elif loss_type == "smooth_l1":
            loss_func = lambda signal, target: lf.smooth_l1(signal, target, beta=1.0)
        elif loss_type == "log_cosh":
            loss_func = lf.log_cosh
        elif loss_type == "quantile":
            loss_func = lambda signal, target: lf.quantile(signal, target, quantile=0.5)
        elif loss_type == "custom" and custom_loss_func is not None:
            loss_func = lf.custom_loss(custom_loss_func)
        else:
            raise ValueError(
                "Invalid loss_type. Must be 'mse', 'mae', 'huber', 'smooth_l1', 'log_cosh', 'quantile', or 'custom' with a custom_loss_func provided."
            )

        x = initial_guess
        for i in range(iterations):
            grad = (
                loss_func(self.signal - x, target + 1e-5)
                - loss_func(self.signal - x, target)
            ) / 1e-5
            x -= learning_rate * grad

        return self.signal - x

    def gradient_descent_filter(self, target, learning_rate=0.01, iterations=100):
        """
        Apply a gradient descent filter to adaptively filter the signal towards a target.

        This method uses gradient descent to iteratively minimize the difference between the signal and a
        target signal. It is useful in scenarios where the signal needs to be progressively adjusted to match
        a desired outcome.

        Parameters
        ----------
        target : numpy.ndarray
            The target signal to adapt to.
        learning_rate : float
            Learning rate for the gradient descent, controlling how much the signal is adjusted at each step.
        iterations : int
            Number of iterations for the gradient descent process.

        Returns
        -------
        filtered_signal : numpy.ndarray
            The filtered signal after applying gradient descent.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> target = np.array([1, 2, 2.5, 3, 3.5])
        >>> af = AdvancedSignalFiltering(signal)
        >>> filtered_signal = af.gradient_descent_filter(target, learning_rate=0.1, iterations=50)
        >>> print(filtered_signal)
        """
        if target is None:
            target = np.zeros_like(target, dtype=np.float64)

        signal = self.signal.astype(np.float64)
        filtered_signal = signal.copy()

        for i in range(iterations):
            grad = np.sign(filtered_signal - target)
            filtered_signal -= learning_rate * grad
        return filtered_signal

    def ensemble_filtering(
        self,
        filters,
        method="mean",
        weights=None,
        num_iterations=10,
        learning_rate=0.01,
    ):
        """
        Apply ensemble filtering by combining the results of multiple filters using various ensemble techniques.

        This method aggregates the results of different filtering techniques to achieve a more robust
        and reliable filtered signal. Ensemble methods like mean, weighted mean, bagging, and boosting
        are implemented to leverage the strengths of multiple filters.

        Parameters
        ----------
        filters : list of callable
            A list of filtering functions to apply. Each function should return a filtered signal.
        method : str
            The ensemble method to use ('mean', 'weighted_mean', 'bagging', 'boosting').
        weights : list of float, optional
            Weights for weighted mean if 'weighted_mean' is chosen. Should match the number of filters.
        num_iterations : int, optional
            Number of iterations for bagging/boosting. Default is 10.
        learning_rate : float, optional
            Learning rate for boosting. Default is 0.01.

        Returns
        -------
        filtered_signal : numpy.ndarray
            The ensemble filtered signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 6])
        >>> af = AdvancedSignalFiltering(signal)
        >>> filters = [af.kalman_filter, af.kalman_filter]  # Example with the same filter twice
        >>> filtered_signal = af.ensemble_filtering(filters, method='mean')
        >>> print(filtered_signal)
        """
        if method == "mean":
            filtered_signals = np.array([f() for f in filters])
            return np.mean(filtered_signals, axis=0)

        elif method == "weighted_mean":
            if weights is None or len(weights) != len(filters):
                raise ValueError(
                    "Weights must be provided and match the number of filters for weighted_mean."
                )
            filtered_signals = np.array([f() for f in filters])
            return np.average(filtered_signals, axis=0, weights=weights)

        elif method == "bagging":
            aggregated_signal = np.zeros_like(self.signal)
            for _ in range(num_iterations):
                sampled_indices = np.random.choice(
                    len(self.signal), size=len(self.signal), replace=True
                )
                sampled_signal = self.signal[sampled_indices]
                for filter_func in filters:
                    aggregated_signal += filter_func(sampled_signal)
            return aggregated_signal / (len(filters) * num_iterations)

        elif method == "boosting":
            boosted_signal = np.zeros_like(self.signal)
            residual = self.signal.copy()
            for _ in range(num_iterations):
                for filter_func in filters:
                    prediction = filter_func()
                    residual -= learning_rate * prediction
                    boosted_signal += prediction
            return boosted_signal / num_iterations

        else:
            raise ValueError(
                "Invalid method. Must be 'mean', 'weighted_mean', 'bagging', or 'boosting'."
            )

    def convolution_based_filter(
        self, kernel_type="smoothing", custom_kernel=None, kernel_size=3
    ):
        """
        Apply a convolution-based filter to the signal using predefined or custom kernels.

        Convolution-based filtering is a powerful technique that applies a kernel to the signal to
        achieve various effects such as smoothing, sharpening, and edge detection. This method allows
        for the use of standard convolutional kernels or custom kernels provided by the user.

        Parameters
        ----------
        kernel_type : str
            The type of kernel to use ('smoothing', 'sharpening', 'edge_detection', 'custom').
        custom_kernel : numpy.ndarray, optional
            The custom kernel to use if 'custom' is selected. This must be provided by the user.
        kernel_size : int, optional
            The size of the kernel for predefined kernels. Default is 3.

        Returns
        -------
        filtered_signal : numpy.ndarray
            The filtered signal after convolution.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 6])
        >>> af = AdvancedSignalFiltering(signal)
        >>> filtered_signal = af.convolution_based_filter(kernel_type='sharpening')
        >>> print(filtered_signal)
        """
        ck = ConvolutionKernels()

        if kernel_type == "smoothing":
            kernel = ck.smoothing(size=kernel_size)
        elif kernel_type == "sharpening":
            kernel = ck.sharpening()
        elif kernel_type == "edge_detection":
            kernel = ck.edge_detection()
        elif kernel_type == "custom" and custom_kernel is not None:
            kernel = custom_kernel
        else:
            raise ValueError(
                "Invalid kernel_type. Must be 'smoothing', 'sharpening', 'edge_detection', or 'custom' with a custom_kernel provided."
            )

        return np.convolve(self.signal, kernel, mode="same")

    def attention_based_filter(
        self, attention_type="uniform", custom_weights=None, size=5, **kwargs
    ):
        """
        Apply an attention-based filter to the signal using predefined or custom attention weights.

        Attention-based filtering uses an attention mechanism to dynamically adjust the influence of
        different parts of the signal. This method supports various types of attention mechanisms,
        including uniform, linear, gaussian, and exponential, as well as custom-defined attention weights.

        Parameters
        ----------
        attention_type : str
            The type of attention weights to use ('uniform', 'linear', 'gaussian', 'exponential', 'custom').
        custom_weights : numpy.ndarray, optional
            The custom attention weights to use if 'custom' is selected.
        size : int, optional
            The size of the attention window for predefined attention weights. Default is 5.
        **kwargs : dict, optional
            Additional parameters for specific attention types (e.g., 'ascending' for linear, 'sigma' for gaussian, 'base' for exponential).

        Returns
        -------
        filtered_signal : numpy.ndarray
            The filtered signal after applying the attention-based filter.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 6])
        >>> af = AdvancedSignalFiltering(signal)
        >>> filtered_signal = af.attention_based_filter(attention_type='gaussian', size=5, sigma=1.0)
        >>> print(filtered_signal)
        """
        aw = AttentionWeights()

        if attention_type == "uniform":
            weights = aw.uniform(size)
        elif attention_type == "linear":
            weights = aw.linear(size, ascending=kwargs.get("ascending", True))
        elif attention_type == "gaussian":
            weights = aw.gaussian(size, sigma=kwargs.get("sigma", 1.0))
        elif attention_type == "exponential":
            weights = aw.exponential(
                size,
                ascending=kwargs.get("ascending", True),
                base=kwargs.get("base", 2.0),
            )
        elif attention_type == "custom" and custom_weights is not None:
            weights = custom_weights
        else:
            raise ValueError(
                "Invalid attention_type. Must be 'uniform', 'linear', 'gaussian', 'exponential', or 'custom' with custom_weights provided."
            )

        return np.convolve(self.signal, weights, mode="same")
