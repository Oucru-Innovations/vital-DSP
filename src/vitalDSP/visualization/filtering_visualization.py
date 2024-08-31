import plotly.graph_objs as go
from vitalDSP.filtering.signal_filtering import (
    SignalFiltering,
)  # Assuming SignalFiltering class has been implemented


class FilteringVisualization:
    """
    A class to visualize the results of different filtering methods using Plotly.

    Methods:
    - visualize_moving_average: Visualizes the moving average filter.
    - visualize_gaussian_filter: Visualizes the Gaussian filter.
    - visualize_butterworth_filter: Visualizes the Butterworth filter.
    - visualize_median_filter: Visualizes the median filter.
    - visualize_all_filters: Visualizes all the filters in a single plot for comparison.
    """

    def __init__(self, signal):
        """
        Initialize the FilteringVisualization class with the signal.

        Parameters:
        signal (numpy.ndarray): The input signal to be filtered and visualized.
        """
        self.signal = signal
        self.filtering = SignalFiltering(signal)

    def visualize_moving_average(self, window_size=3):
        """
        Visualize the moving average filter on the signal.

        Parameters:
        window_size (int): The size of the moving average window.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> fv = FilteringVisualization(signal)
        >>> fv.visualize_moving_average(window_size=5)
        """
        filtered_signal = self.filtering.moving_average(window_size)
        self._plot_signal(
            filtered_signal, title=f"Moving Average Filter (Window Size: {window_size})"
        )

    def visualize_gaussian_filter(self, sigma=1.0):
        """
        Visualize the Gaussian filter on the signal.

        Parameters:
        sigma (float): The standard deviation for the Gaussian filter.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> fv = FilteringVisualization(signal)
        >>> fv.visualize_gaussian_filter(sigma=1.0)
        """
        filtered_signal = self.filtering.gaussian(sigma)
        self._plot_signal(filtered_signal, title=f"Gaussian Filter (Sigma: {sigma})")

    def visualize_butterworth_filter(self, cutoff=0.5, order=2, fs=1000):
        """
        Visualize the Butterworth filter on the signal.

        Parameters:
        cutoff (float): The cutoff frequency for the Butterworth filter.
        order (int): The order of the Butterworth filter.
        fs (float): The sampling frequency of the signal.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> fv = FilteringVisualization(signal)
        >>> fv.visualize_butterworth_filter(cutoff=0.3, order=2, fs=100)
        """
        filtered_signal = self.filtering.butterworth(cutoff, order, fs)
        self._plot_signal(
            filtered_signal,
            title=f"Butterworth Filter (Cutoff: {cutoff} Hz, Order: {order})",
        )

    def visualize_median_filter(self, kernel_size=3):
        """
        Visualize the median filter on the signal.

        Parameters:
        kernel_size (int): The size of the median filter kernel.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> fv = FilteringVisualization(signal)
        >>> fv.visualize_median_filter(kernel_size=5)
        """
        filtered_signal = self.filtering.median(kernel_size)
        self._plot_signal(
            filtered_signal, title=f"Median Filter (Kernel Size: {kernel_size})"
        )

    def visualize_all_filters(
        self, window_size=3, sigma=1.0, cutoff=0.5, order=2, fs=1000, kernel_size=3
    ):
        """
        Visualize all implemented filters in a single plot for comparison.

        Parameters:
        window_size (int): The size of the moving average window.
        sigma (float): The standard deviation for the Gaussian filter.
        cutoff (float): The cutoff frequency for the Butterworth filter.
        order (int): The order of the Butterworth filter.
        fs (float): The sampling frequency of the signal.
        kernel_size (int): The size of the median filter kernel.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> fv = FilteringVisualization(signal)
        >>> fv.visualize_all_filters(window_size=5, sigma=1.0, cutoff=0.3, order=2, fs=100, kernel_size=5)
        """
        moving_avg_signal = self.filtering.moving_average(window_size)
        gaussian_signal = self.filtering.gaussian(sigma)
        butterworth_signal = self.filtering.butterworth(cutoff, order, fs)
        median_signal = self.filtering.median(kernel_size)

        traces = [
            go.Scatter(y=self.signal, mode="lines", name="Original Signal"),
            go.Scatter(
                y=moving_avg_signal,
                mode="lines",
                name=f"Moving Average (Window Size: {window_size})",
            ),
            go.Scatter(
                y=gaussian_signal,
                mode="lines",
                name=f"Gaussian Filter (Sigma: {sigma})",
            ),
            go.Scatter(
                y=butterworth_signal,
                mode="lines",
                name=f"Butterworth Filter (Cutoff: {cutoff} Hz, Order: {order})",
            ),
            go.Scatter(
                y=median_signal,
                mode="lines",
                name=f"Median Filter (Kernel Size: {kernel_size})",
            ),
        ]

        layout = go.Layout(
            title="Comparison of Different Filtering Methods",
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )

        fig = go.Figure(data=traces, layout=layout)
        fig.show()

    def _plot_signal(self, filtered_signal, title="Filtered Signal"):
        """
        Private method to plot the original and filtered signals.

        Parameters:
        filtered_signal (numpy.ndarray): The filtered signal.
        title (str): The title of the plot.
        """
        trace1 = go.Scatter(y=self.signal, mode="lines", name="Original Signal")
        trace2 = go.Scatter(y=filtered_signal, mode="lines", name="Filtered Signal")

        layout = go.Layout(
            title=title,
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )

        fig = go.Figure(data=[trace1, trace2], layout=layout)
        fig.show()
