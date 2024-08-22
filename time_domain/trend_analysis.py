import numpy as np

class TrendAnalysis:
    """
    A comprehensive class to track and analyze long-term trends in physiological signals.

    Methods:
    - compute_moving_average: Computes the moving average of the signal.
    - compute_weighted_moving_average: Computes the weighted moving average of the signal.
    - compute_exponential_smoothing: Computes exponential smoothing of the signal.
    - compute_linear_trend: Computes the linear trend using least squares regression.
    - compute_polynomial_trend: Computes a polynomial trend.
    - compute_momentum: Computes the momentum of the signal.
    - compute_seasonal_decomposition: Decomposes the signal into trend, seasonal, and residual components.
    - compute_trend_strength: Calculates the strength of the trend.
    - detect_trend_reversal: Detects trend reversal points in the signal.
    """

    def __init__(self, signal):
        """
        Initialize the TrendAnalysis class with the signal.

        Parameters:
        signal (numpy.ndarray): The input physiological signal.
        """
        self.signal = signal

    def compute_moving_average(self, window_size):
        """
        Compute the moving average of the signal.

        Parameters:
        window_size (int): The size of the moving average window.

        Returns:
        numpy.ndarray: The moving average of the signal.
        """
        cumsum = np.cumsum(np.insert(self.signal, 0, 0))
        moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
        return moving_avg

    def compute_weighted_moving_average(self, window_size, weights=None):
        """
        Compute the weighted moving average of the signal.

        Parameters:
        window_size (int): The size of the moving average window.
        weights (numpy.ndarray or None): Weights for the moving average. If None, uniform weights are used.

        Returns:
        numpy.ndarray: The weighted moving average of the signal.
        """
        if weights is None:
            weights = np.ones(window_size) / window_size
        else:
            weights = np.array(weights)
            weights /= np.sum(weights)

        weighted_avg = np.convolve(self.signal, weights, mode='valid')
        return weighted_avg

    def compute_exponential_smoothing(self, alpha):
        """
        Compute exponential smoothing of the signal.

        Parameters:
        alpha (float): Smoothing factor (0 < alpha <= 1).

        Returns:
        numpy.ndarray: The exponentially smoothed signal.
        """
        smoothed_signal = np.zeros_like(self.signal)
        smoothed_signal[0] = self.signal[0]
        for t in range(1, len(self.signal)):
            smoothed_signal[t] = alpha * self.signal[t] + (1 - alpha) * smoothed_signal[t-1]
        return smoothed_signal

    def compute_linear_trend(self):
        """
        Compute the linear trend of the signal using least squares regression.

        Returns:
        numpy.ndarray: The linear trend of the signal.
        """
        x = np.arange(len(self.signal))
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, self.signal, rcond=None)[0]
        linear_trend = m * x + c
        return linear_trend

    def compute_polynomial_trend(self, degree):
        """
        Compute a polynomial trend of the signal.

        Parameters:
        degree (int): The degree of the polynomial to fit.

        Returns:
        numpy.ndarray: The polynomial trend of the signal.
        """
        x = np.arange(len(self.signal))
        coeffs = np.polyfit(x, self.signal, degree)
        polynomial_trend = np.polyval(coeffs, x)
        return polynomial_trend

    def compute_momentum(self, window_size):
        """
        Compute the momentum of the signal.

        Parameters:
        window_size (int): The size of the window to calculate momentum.

        Returns:
        numpy.ndarray: The momentum of the signal.
        """
        momentum = np.zeros_like(self.signal)
        momentum[window_size:] = self.signal[window_size:] - self.signal[:-window_size]
        return momentum

    def compute_seasonal_decomposition(self, period):
        """
        Decompose the signal into trend, seasonal, and residual components using seasonal decomposition.

        Parameters:
        period (int): The period of the seasonal component.

        Returns:
        dict: A dictionary containing 'trend', 'seasonal', and 'residual' components.
        """
        trend = self.compute_moving_average(period)
        detrended = self.signal[:len(trend)] - trend
        seasonal = np.array([np.mean(detrended[i::period]) for i in range(period)])
        seasonal = np.tile(seasonal, len(detrended) // period + 1)[:len(detrended)]
        residual = detrended - seasonal
        return {'trend': trend, 'seasonal': seasonal, 'residual': residual}

    def compute_trend_strength(self):
        """
        Calculate the strength of the trend in the signal.

        Returns:
        float: The strength of the trend (between 0 and 1).
        """
        linear_trend = self.compute_linear_trend()
        detrended_signal = self.signal - linear_trend
        trend_strength = 1 - (np.var(detrended_signal) / np.var(self.signal))
        return trend_strength

    def detect_trend_reversal(self, window_size):
        """
        Detect trend reversal points in the signal.

        Parameters:
        window_size (int): The window size to evaluate trend reversals.

        Returns:
        numpy.ndarray: Indices of the detected trend reversal points.
        """
        trend = self.compute_moving_average(window_size)
        diff_trend = np.diff(trend)
        reversals = np.where(np.diff(np.sign(diff_trend)))[0] + 1
        return reversals
