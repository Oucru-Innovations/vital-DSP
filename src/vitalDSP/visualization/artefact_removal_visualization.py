import plotly.graph_objs as go
from vitalDSP.filtering.artifact_removal import ArtifactRemoval
import numpy as np


class ArtifactRemovalVisualization:
    def __init__(self, signal, reference_signal=None):
        """
        Initialize the ArtifactRemovalVisualization class with the signal and optional reference signal.

        Parameters:
        signal (numpy.ndarray): The input signal to be cleaned and visualized.
        reference_signal (numpy.ndarray, optional): An optional reference signal for artifact removal methods.
        """
        self.signal = signal
        self.reference_signal = reference_signal
        self.artifact_removal = ArtifactRemoval(signal)

    def _set_reference_signal(self):
        """Set the reference signal in ArtifactRemoval if applicable."""
        if hasattr(self.artifact_removal, "set_reference_signal"):
            self.artifact_removal.set_reference_signal(self.reference_signal)

    def visualize_artifact_removal(self, method="mean_subtraction"):
        """
        Visualize the effect of a specific artifact removal method on the signal.

        Parameters:
        method (str): The artifact removal method to visualize.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> reference_signal = np.sin(np.linspace(0, 10, 100))
        >>> ar_viz = ArtifactRemovalVisualization(signal, reference_signal)
        >>> ar_viz.visualize_artifact_removal(method='mean_subtraction')
        """
        self._set_reference_signal()

        if method == "mean_subtraction":
            cleaned_signal = self.artifact_removal.mean_subtraction()
        elif method == "baseline_correction":
            cleaned_signal = self.artifact_removal.baseline_correction()
        elif method == "median_filter_removal":
            cleaned_signal = self.artifact_removal.median_filter_removal()
        elif method == "wavelet_denoising":
            cleaned_signal = self.artifact_removal.wavelet_denoising()
            cleaned_signal = np.real(cleaned_signal)  # Take the real part
        elif method == "adaptive_filtering":
            cleaned_signal = self.artifact_removal.adaptive_filtering(
                self.reference_signal
            )
        elif method == "notch_filter":
            cleaned_signal = self.artifact_removal.notch_filter()
        elif method == "pca_artifact_removal":
            cleaned_signal = self.artifact_removal.pca_artifact_removal()
        elif method == "ica_artifact_removal":
            cleaned_signal = self.artifact_removal.ica_artifact_removal()
        else:
            raise ValueError("Invalid artifact removal method specified.")

        self._plot_signal(
            cleaned_signal,
            title=f"Artifact Removal - {method.replace('_', ' ').title()}",
        )

    def visualize_all_removals(self):
        """
        Visualize all artifact removal methods in a single plot for comparison.

        Example Usage:
        >>> signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        >>> reference_signal = np.sin(np.linspace(0, 10, 100))
        >>> ar_viz = ArtifactRemovalVisualization(signal, reference_signal)
        >>> ar_viz.visualize_all_removals()
        """
        self._set_reference_signal()

        mean_sub_signal = self.artifact_removal.mean_subtraction()
        baseline_corr_signal = self.artifact_removal.baseline_correction()
        median_filt_signal = self.artifact_removal.median_filter_removal()
        wavelet_denoise_signal = np.real(
            self.artifact_removal.wavelet_denoising()
        )  # Real part
        adaptive_filt_signal = self.artifact_removal.adaptive_filtering(
            self.reference_signal
        )  # Pass reference_signal
        notch_filt_signal = self.artifact_removal.notch_filter()
        pca_signal = self.artifact_removal.pca_artifact_removal(num_components=1)
        ica_signal = self.artifact_removal.ica_artifact_removal(num_components=1)

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
        trace1 = go.Scatter(y=self.signal, mode="lines", name="Original Signal")
        trace2 = go.Scatter(y=cleaned_signal, mode="lines", name="Cleaned Signal")

        layout = go.Layout(
            title=title,
            xaxis=dict(title="Sample Index"),
            yaxis=dict(title="Amplitude"),
        )

        fig = go.Figure(data=[trace1, trace2], layout=layout)
        fig.show()
