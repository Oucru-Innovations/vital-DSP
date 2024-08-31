import numpy as np


class TrendAnalysis:
    """
    A comprehensive class to track and analyze long-term trends in physiological signals.

    This class provides various methods to compute trends, moving averages, momentum,
    and decompose the signal into its components for in-depth trend analysis.

    Methods
    -------
    compute_moving_average : function
        Computes the moving average of the signal.
    compute_weighted_moving_average : function
        Computes the weighted moving average of the signal.
    compute_exponential_smoothing : function
        Computes exponential smoothing of the signal.
    compute_linear_trend : function
        Computes the linear trend using least squares regression.
    compute_polynomial_trend : function
        Computes a polynomial trend.
    compute_momentum : function
        Computes the momentum of the signal.
    compute_seasonal_decomposition : function
        Decomposes the signal into trend, seasonal, and residual components.
    compute_trend_strength : function
        Calculates the strength of the trend.
    detect_trend_reversal : function
        Detects trend reversal points in the signal.
    """

    def __init__(self, signal):
        """
        Initialize the TrendAnalysis class with the signal.

        Parameters
        ----------
        signal : numpy.ndarray
            The input physiological signal to be analyzed.
        """
        self.signal = signal

    def compute_moving_average(self, window_size):
        """
        Compute the moving average of the signal.

        This method smooths the signal by averaging data points within a moving window.

        Parameters
        ----------
        window_size : int
            The size of the moving average window.

        Returns
        -------
        moving_avg : numpy.ndarray
            The moving average of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 6, 7])
        >>> ta = TrendAnalysis(signal)
        >>> ta.compute_moving_average(3)
        array([2., 3., 4., 5., 6.])
        """
        cumsum = np.cumsum(np.insert(self.signal, 0, 0))
        moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
        return moving_avg

    def compute_weighted_moving_average(self, window_size, weights=None):
        """
        Compute the weighted moving average of the signal.

        This method smooths the signal by averaging data points within a moving window,
        with each point weighted by the given weights.

        Parameters
        ----------
        window_size : int
            The size of the moving average window.
        weights : numpy.ndarray or None, optional
            Weights for the moving average. If None, uniform weights are used.

        Returns
        -------
        weighted_avg : numpy.ndarray
            The weighted moving average of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 6, 7])
        >>> ta = TrendAnalysis(signal)
        >>> ta.compute_weighted_moving_average(3, weights=[0.1, 0.3, 0.6])
        array([2.2, 3.2, 4.2, 5.2, 6.2])
        """
        if weights is None:
            weights = np.ones(window_size) / window_size
        else:
            weights = np.array(weights)
            weights /= np.sum(weights)

        weighted_avg = np.convolve(self.signal, weights, mode="valid")
        return weighted_avg

    def compute_exponential_smoothing(self, alpha):
        """
        Compute exponential smoothing of the signal.

        Exponential smoothing applies a decay factor to smooth the signal, giving more weight
        to recent observations while maintaining a smooth trend.

        Parameters
        ----------
        alpha : float
            Smoothing factor (0 < alpha <= 1).

        Returns
        -------
        smoothed_signal : numpy.ndarray
            The exponentially smoothed signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 6, 7])
        >>> ta = TrendAnalysis(signal)
        >>> ta.compute_exponential_smoothing(0.3)
        array([1., 1.3, 1.81, 2.467, 3.227, 4.059, 4.941])
        """
        smoothed_signal = np.zeros_like(self.signal)
        smoothed_signal[0] = self.signal[0]
        for t in range(1, len(self.signal)):
            smoothed_signal[t] = (
                alpha * self.signal[t] + (1 - alpha) * smoothed_signal[t - 1]
            )
        return smoothed_signal

    def compute_linear_trend(self):
        """
        Compute the linear trend of the signal using least squares regression.

        This method fits a straight line to the signal data, representing the overall direction
        of the signal over time.

        Returns
        -------
        linear_trend : numpy.ndarray
            The linear trend of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 5, 8, 13, 21])
        >>> ta = TrendAnalysis(signal)
        >>> ta.compute_linear_trend()
        array([ 0.6       ,  2.45714286,  4.31428571,  6.17142857,  8.02857143,
                9.88571429, 11.74285714])
        """
        x = np.arange(len(self.signal))
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, self.signal, rcond=None)[0]
        linear_trend = m * x + c
        return linear_trend

    def compute_polynomial_trend(self, degree):
        """
        Compute a polynomial trend of the signal.

        This method fits a polynomial of the specified degree to the signal data, providing
        a more flexible trend line that can account for curvature in the data.

        Parameters
        ----------
        degree : int
            The degree of the polynomial to fit.

        Returns
        -------
        polynomial_trend : numpy.ndarray
            The polynomial trend of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 5, 8, 13, 21])
        >>> ta = TrendAnalysis(signal)
        >>> ta.compute_polynomial_trend(degree=2)
        array([ 1.05714286,  1.84285714,  3.05714286,  4.7       ,  6.77142857,
                9.27142857, 12.2       ])
        """
        x = np.arange(len(self.signal))
        coeffs = np.polyfit(x, self.signal, degree)
        polynomial_trend = np.polyval(coeffs, x)
        return polynomial_trend

    def compute_momentum(self, window_size):
        """
        Compute the momentum of the signal.

        Momentum measures the rate of change of the signal over a specified window size,
        indicating the strength and direction of the trend.

        Parameters
        ----------
        window_size : int
            The size of the window to calculate momentum.

        Returns
        -------
        momentum : numpy.ndarray
            The momentum of the signal.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 5, 8, 13, 21])
        >>> ta = TrendAnalysis(signal)
        >>> ta.compute_momentum(window_size=2)
        array([ 0,  0,  1,  2,  3,  5,  8])
        """
        momentum = np.zeros_like(self.signal)
        momentum[window_size:] = self.signal[window_size:] - self.signal[:-window_size]
        return momentum

    def compute_seasonal_decomposition(self, period):
        """
        Decompose the signal into trend, seasonal, and residual components using seasonal decomposition.

        This method separates the signal into its underlying trend, seasonal variations,
        and random noise components, providing a clearer view of each aspect of the signal.

        Parameters
        ----------
        period : int
            The period of the seasonal component.

        Returns
        -------
        decomposition : dict
            A dictionary containing 'trend', 'seasonal', and 'residual' components.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 4, 5, 6, 7])
        >>> ta = TrendAnalysis(signal)
        >>> decomposition = ta.compute_seasonal_decomposition(period=2)
        >>> decomposition['trend']
        array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        >>> decomposition['seasonal']
        array([ 0. ,  0. , -0.5,  0.5, -0.5,  0.5])
        >>> decomposition['residual']
        array([-0.5,  0.5,  0. ,  0. ,  0. ,  0. ])
        """
        trend = self.compute_moving_average(period)
        detrended = self.signal[: len(trend)] - trend
        seasonal = np.array([np.mean(detrended[i::period]) for i in range(period)])
        seasonal = np.tile(seasonal, len(detrended) // period + 1)[: len(detrended)]
        residual = detrended - seasonal
        return {"trend": trend, "seasonal": seasonal, "residual": residual}

    def compute_trend_strength(self):
        """
        Calculate the strength of the trend in the signal.

        Trend strength is determined by comparing the variance of the detrended signal
        to the variance of the original signal, with a value close to 1 indicating a strong trend.

        Returns
        -------
        trend_strength : float
            The strength of the trend (between 0 and 1).

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 5, 8, 13, 21])
        >>> ta = TrendAnalysis(signal)
        >>> ta.compute_trend_strength()
        0.9571428571428573
        """
        linear_trend = self.compute_linear_trend()
        detrended_signal = self.signal - linear_trend
        trend_strength = 1 - (np.var(detrended_signal) / np.var(self.signal))
        return trend_strength

    def detect_trend_reversal(self, window_size):
        """
        Detect trend reversal points in the signal.

        This method identifies points in the signal where the trend changes direction,
        which can be critical in understanding shifts in physiological signals.

        Parameters
        ----------
        window_size : int
            The window size to evaluate trend reversals.

        Returns
        -------
        reversals : numpy.ndarray
            Indices of the detected trend reversal points.

        Examples
        --------
        >>> signal = np.array([1, 2, 3, 2, 1, 2, 3])
        >>> ta = TrendAnalysis(signal)
        >>> ta.detect_trend_reversal(window_size=2)
        array([2, 4])
        """
        trend = self.compute_moving_average(window_size)
        diff_trend = np.diff(trend)
        reversals = np.where(np.diff(np.sign(diff_trend)))[0] + 1
        return reversals
