import plotly.graph_objs as go
from vitalDSP.filtering.artifact_removal import (
    ArtifactRemoval,
)  # Assuming ArtifactRemoval class has been implemented


class ArtifactRemovalVisualization:
    """
    A class to visualize the results of different artifact removal methods using Plotly.

    Methods:
    - visualize_artifact_removal: Visualizes a specific artifact removal method.
    - visualize_all_removals: Visualizes all artifact removal methods in a single plot for comparison.
    """

    def __init__(self, signal):
        """
        Initialize the ArtifactRemovalVisualization class with the signal.

        Parameters:
        signal (numpy.ndarray): The input signal to be cleaned and visualized.
        """
        self.signal = signal
        self.artifact_removal = ArtifactRemoval(signal)

    def visualize_artifact_removal(self, method="mean_subtraction", **kwargs):
        """
        Visualize the effect of a specific artifact removal method on the signal.

        Parameters:
        method (str): The artifact removal method to visualize.
        kwargs: Additional parameters required by the specific method.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> ar_viz = ArtifactRemovalVisualization(signal)
        >>> ar_viz.visualize_artifact_removal(method='mean_subtraction')
        """
        if method == "mean_subtraction":
            cleaned_signal = self.artifact_removal.mean_subtraction()
        elif method == "baseline_correction":
            cleaned_signal = self.artifact_removal.baseline_correction(**kwargs)
        elif method == "median_filter_removal":
            cleaned_signal = self.artifact_removal.median_filter_removal(**kwargs)
        elif method == "wavelet_denoising":
            cleaned_signal = self.artifact_removal.wavelet_denoising(**kwargs)
        elif method == "adaptive_filtering":
            cleaned_signal = self.artifact_removal.adaptive_filtering(**kwargs)
        elif method == "notch_filter":
            cleaned_signal = self.artifact_removal.notch_filter(**kwargs)
        elif method == "pca_artifact_removal":
            cleaned_signal = self.artifact_removal.pca_artifact_removal(**kwargs)
        elif method == "ica_artifact_removal":
            cleaned_signal = self.artifact_removal.ica_artifact_removal(**kwargs)
        else:
            raise ValueError("Invalid artifact removal method specified.")

        self._plot_signal(
            cleaned_signal,
            title=f"Artifact Removal - {method.replace('_', ' ').title()}",
        )

    def visualize_all_removals(self, **kwargs):
        """
        Visualize all artifact removal methods in a single plot for comparison.

        kwargs: Additional parameters required by the specific methods.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> ar_viz = ArtifactRemovalVisualization(signal)
        >>> ar_viz.visualize_all_removals()
        """
        mean_sub_signal = self.artifact_removal.mean_subtraction()
        baseline_corr_signal = self.artifact_removal.baseline_correction(**kwargs)
        median_filt_signal = self.artifact_removal.median_filter_removal(**kwargs)
        wavelet_denoise_signal = self.artifact_removal.wavelet_denoising(**kwargs)
        adaptive_filt_signal = self.artifact_removal.adaptive_filtering(**kwargs)
        notch_filt_signal = self.artifact_removal.notch_filter(**kwargs)
        pca_signal = self.artifact_removal.pca_artifact_removal(**kwargs)
        ica_signal = self.artifact_removal.ica_artifact_removal(**kwargs)

        traces = [
            go.Scatter(y=self.signal, mode="lines", name="Original Signal"),
            go.Scatter(y=mean_sub_signal, mode="lines", name="Mean Subtraction"),
            go.Scatter(
                y=baseline_corr_signal, mode="lines", name="Baseline Correction"
            ),
            go.Scatter(
                y=median_filt_signal, mode="lines", name="Median Filter Removal"
            ),
            go.Scatter(
                y=wavelet_denoise_signal, mode="lines", name="Wavelet Denoising"
            ),
            go.Scatter(y=adaptive_filt_signal, mode="lines", name="Adaptive Filtering"),
            go.Scatter(y=notch_filt_signal, mode="lines", name="Notch Filter"),
            go.Scatter(y=pca_signal, mode="lines", name="PCA Artifact Removal"),
            go.Scatter(y=ica_signal, mode="lines", name="ICA Artifact Removal"),
        ]

        layout = go.Layout(
            title="Comparison of Artifact Removal Methods",
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )

        fig = go.Figure(data=traces, layout=layout)
        fig.show()

    def _plot_signal(self, cleaned_signal, title="Artifact Removed Signal"):
        """
        Private method to plot the original and cleaned signals.

        Parameters:
        cleaned_signal (numpy.ndarray): The cleaned signal.
        title (str): The title of the plot.
        """
        trace1 = go.Scatter(y=self.signal, mode="lines", name="Original Signal")
        trace2 = go.Scatter(y=cleaned_signal, mode="lines", name="Cleaned Signal")

        layout = go.Layout(
            title=title,
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )

        fig = go.Figure(data=[trace1, trace2], layout=layout)
        fig.show()
