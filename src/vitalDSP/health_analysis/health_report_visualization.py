import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import spectrogram, periodogram, welch
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr
from collections import Counter
from scipy.interpolate import make_interp_spline
from vitalDSP.utils.common import find_peaks
import logging


class HealthReportVisualizer:
    """
    A class responsible for creating visualizations of health feature data, including line plots and heatmaps.

    The class takes feature data and creates visualizations, such as normal distributions for ranges and heatmaps, and stores them as images.
    """

    def __init__(self, config, segment_duration="1_min"):
        """
        Initializes the Visualization class by loading the feature configuration.

        Args:
            config (dict): Configuration data that includes normal ranges and interpretations for features.

        Example Usage:
            >>> visualization = Visualization(config)
        """
        if not isinstance(config, dict):
            raise TypeError("Config should be a dictionary.")
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.segment_duration = segment_duration

    def _fetch_and_validate_normal_range(self, feature, value):
        """
        Fetches the normal range for a given feature, handles NaN, Inf values, and validates the feature.

        Args:
            feature (str): The feature name.
            value (float): The current value for the feature.

        Returns:
            tuple: (normal_min, normal_max, feature_names) where feature_names are ["Min Range", "Max Range", "Current Value"].

        Raises:
            ValueError: If the feature has NaN or invalid values, or normal range is not found.
        """
        normal_range = self._get_normal_range_for_feature(feature)
        if not normal_range:
            raise ValueError(f"Normal range for feature '{feature}' not found.")

        normal_min, normal_max = normal_range

        # Initialize valid normal range
        valid_values = []

        # Collect valid values, avoiding NaN and Inf
        if not np.isnan(normal_min) and not np.isinf(normal_min):
            valid_values.append(normal_min)
        if not np.isnan(normal_max) and not np.isinf(normal_max):
            valid_values.append(normal_max)
        if not np.isnan(value) and not np.isinf(value):
            valid_values.append(value)

        # Check if we have valid values to compute the normal range
        if not valid_values:
            raise ValueError(f"All values for feature '{feature}' are NaN or Inf.")
        print(valid_values)
        # Compute new normal_min and normal_max based on valid values
        normal_min = min(valid_values)
        normal_max = max(valid_values)

        if len(valid_values) == 1:
            value = valid_values[0]
            return (value-2*value), (value+2*value)
        return normal_min, normal_max

    def create_visualizations(self, feature_data, output_dir="visualizations"):
        """
        Creates visualizations for the provided feature data and saves them as image files.

        Args:
            feature_data (dict): Dictionary containing feature values.
            output_dir (str): Directory where the visualizations will be saved.

        Returns:
            dict: Dictionary with feature names as keys and paths to the saved visualizations as values.

        Example Usage:
            >>> visualizations = visualization.create_visualizations(feature_data)
            >>> print(visualizations)
        """
        if output_dir is None:
            output_dir = "visualizations"
            os.makedirs(output_dir, exist_ok=True)

        visualization_paths = {}

        for feature, values in feature_data.items():
            if not isinstance(values, list):
                values = [values]  # Handle list (multiple segments) for each feature
            visualization_paths[feature] = {
                "heatmap": os.path.normpath(
                    self._create_heatmap_plot(feature, values, output_dir)
                ),
                "bell_plot": os.path.normpath(
                    self._create_bell_shape_plot(feature, values, output_dir)
                ),
                "radar_plot": os.path.normpath(
                    self._create_radar_plot(feature, values, output_dir)
                ),
                "violin_plot": os.path.normpath(
                    self._create_violin_plot(feature, values, output_dir)
                ),
                "line_with_rolling_stats": os.path.normpath(
                    self._create_plot_line_with_rolling_stats(
                        feature, values, output_dir
                    )
                ),
                "lag_plot": os.path.normpath(
                    self._create_plot_lag(feature, values, output_dir)
                ),
                "plot_periodogram": os.path.normpath(
                    self._create_plot_periodogram(feature, values, output_dir)
                ),
                "plot_spectrogram": os.path.normpath(
                    self._create_plot_spectrogram(feature, values, output_dir)
                ),
                "plot_spectral_density": os.path.normpath(
                    self._create_spectral_density_plot(feature, values, output_dir)
                ),
                "plot_box_swarm": os.path.normpath(
                    self._create_box_swarm_plot(feature, values, output_dir)
                ),
            }
        return visualization_paths

    def _create_bell_shape_plot(self, feature, values, output_dir):
        """
        Creates a bell shape plot with an overlaid histogram for better understanding.
        Caps outliers at 1.5 times the normal range.

        Args:
            feature (str): The name of the feature.
            values (list): List of values for the feature.
            output_dir (str): Directory where the plot will be saved.

        Returns:
            str: Path to the saved bell shape plot image.
        """
        try:
            # Fetch normal range
            normal_min, normal_max = self._fetch_and_validate_normal_range(
                feature, values[0]
            )

            # Outlier thresholds
            outlier_min = normal_min - 1.5 * (normal_max - normal_min)
            outlier_max = normal_max + 1.5 * (normal_max - normal_min)

            # Cap the values to fit the normal range
            capped_values = np.clip(values, outlier_min, outlier_max)

            # Calculate statistics
            mean_value = (
                normal_min + normal_max
            ) / 2  # Center bell curve between normal min and max
            stddev_value = (
                normal_max - normal_min
            ) / 4  # Standard deviation should reflect the range

            # Create the figure
            plt.figure(figsize=(10, 6))

            # Histogram
            sns.histplot(
                capped_values,
                bins=15,
                kde=False,
                color="#a4c6f7",
                edgecolor="black",
                alpha=0.7,
                label="Histogram",
            )

            # Generate x values for the bell curve
            x = np.linspace(
                normal_min - 2 * stddev_value, normal_max + 2 * stddev_value, 200
            )

            # Bell curve calculation
            bell_curve = (1 / (stddev_value * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x - mean_value) / stddev_value) ** 2
            )

            # Plot the bell-shaped curve
            plt.plot(
                x,
                bell_curve * max(np.histogram(capped_values, bins=15)[0]),
                color="#007bff",
                linestyle="--",
                label="Bell Curve",
            )

            # Mark vertical lines for the normal range, median, and mean
            plt.axvline(
                normal_min,
                color="#f39c12",
                linestyle="--",
                label=f"Normal Min: {normal_min:.2f}",
            )
            plt.axvline(
                normal_max,
                color="#e74c3c",
                linestyle="--",
                label=f"Normal Max: {normal_max:.2f}",
            )
            plt.axvline(
                np.median(capped_values),
                color="#2ecc71",
                linestyle="-",
                label=f"Median: {np.median(capped_values):.2f}",
            )
            plt.axvline(
                mean_value,
                color="#8e44ad",
                linestyle="--",
                label=f"Mean: {mean_value:.2f}",
            )

            # Highlight outliers
            outliers = [
                value for value in values if value < outlier_min or value > outlier_max
            ]
            for outlier in outliers:
                plt.scatter(
                    outlier,
                    0,
                    color="#e74c3c",
                    s=100,
                    zorder=5,
                    edgecolor="black",
                    label="Outlier" if outlier == outliers[0] else "",
                )

            # Add annotation for key statistics
            plt.annotate(
                f"Mean: {mean_value:.2f}\nStd Dev: {stddev_value:.2f}",
                xy=(mean_value, bell_curve.max() / 2),
                xytext=(mean_value + 1, bell_curve.max() / 1.5),
                arrowprops=dict(facecolor="black", shrink=0.05),
                fontsize=12,
            )

            # Calculate and annotate percentage of values within the normal range
            in_range = [v for v in values if normal_min <= v <= normal_max]
            percentage_in_range = (len(in_range) / len(values)) * 100
            plt.annotate(
                f"{percentage_in_range:.1f}% within normal range",
                xy=(normal_min, bell_curve.max() / 1.5),
                fontsize=12,
                color="blue",
            )

            # Add title and labels
            plt.title(f"{feature} Bell Shape Plot with Histogram and KDE", fontsize=16)
            plt.xlabel(f"{feature} Values", fontsize=14)
            plt.ylabel("Frequency", fontsize=14)

            # Display legend
            plt.legend(loc="upper right", fontsize=12)

            # Save the plot
            filepath = os.path.join(output_dir, f"{feature}_bell_plot.png")
            plt.savefig(filepath, bbox_inches="tight", dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error generating heatmap for {feature}: {e}")
            return f"Error generating plot for {feature}"
        return filepath

    def _create_spectral_density_plot(
        self,
        feature,
        values,
        output_dir,
        sampling_rate=1,
        nfft=16,
        nperseg=8,
        highlight_freqs=None,
        segment_overlap=20,
        peak_threshold=None,
    ):
        """
        Creates an enhanced spectral density plot with smoother curves, confidence intervals,
        and highlighted regions of interest (ROI) in the frequency domain.

        Args:
            feature (str): Name of the feature being analyzed.
            values (np.array): Time-series data for the feature.
            output_dir (str): Directory to save the plot.
            sampling_rate (int): Sampling rate of the time-series data.
            nfft (int): Number of data points used in each block for the FFT.
            highlight_freqs (list of tuples): List of (min_freq, max_freq) ranges to highlight specific frequencies.

        Returns:
            str: Path to the saved spectral density plot image.
        """
        try:
            # Calculate the Power Spectral Density (PSD) using Welch's method
            freqs, psd = welch(values, fs=sampling_rate, nfft=nfft, nperseg=nperseg)
            freqs = freqs * (60 / segment_overlap)

            # Create a smooth curve using interpolation for the PSD
            freqs_smooth = np.linspace(freqs.min(), freqs.max(), 500)
            spline = make_interp_spline(freqs, psd, k=3)
            psd_smooth = spline(freqs_smooth)

            # Create the plot
            plt.figure(figsize=(10, 6))

            # Plot the smoothed PSD curve
            plt.plot(
                freqs_smooth,
                10 * np.log10(psd_smooth),
                color="blue",
                label="Spectral Density (dB)",
                linewidth=2,
            )

            # Confidence interval shading (using an arbitrary ±10% range for illustration)
            lower_bound = 10 * np.log10(psd_smooth * 0.9)
            upper_bound = 10 * np.log10(psd_smooth * 1.1)
            plt.fill_between(
                freqs_smooth,
                lower_bound,
                upper_bound,
                color="blue",
                alpha=0.2,
                label="Confidence Interval",
            )

            # Highlight specific frequency ranges if provided
            if highlight_freqs:
                for min_freq, max_freq in highlight_freqs:
                    plt.axvspan(
                        min_freq,
                        max_freq,
                        color="orange",
                        alpha=0.3,
                        label=f"ROI: {min_freq}-{max_freq} Hz",
                    )

            # Detect peaks in the PSD and highlight them
            if peak_threshold:
                peaks, _ = find_peaks(10 * np.log10(psd), height=peak_threshold)
                plt.scatter(
                    freqs[peaks],
                    10 * np.log10(psd[peaks]),
                    color="red",
                    zorder=5,
                    s=80,
                    edgecolors="black",
                    label="Peaks",
                )

            # Mark the dominant frequency
            dominant_freq = freqs[np.argmax(psd)]
            plt.axvline(
                x=dominant_freq,
                color="green",
                linestyle="--",
                label=f"Dominant Frequency: {dominant_freq:.2f} Hz",
            )

            # Plot trend line
            z = np.polyfit(freqs_smooth, 10 * np.log10(psd_smooth), 1)
            p = np.poly1d(z)
            plt.plot(
                freqs_smooth,
                p(freqs_smooth),
                color="red",
                linestyle="--",
                label="Trend Line",
                alpha=0.7,
            )

            # Add grid, labels, title, and legend
            plt.grid(True, which="both", linestyle="--", alpha=0.5)
            plt.title(
                f"Spectral Density Plot of {feature} (with Peaks & Trend)", fontsize=16
            )
            plt.xlabel("Frequency (Hz)", fontsize=14)
            plt.ylabel("Power/Frequency (dB/Hz)", fontsize=14)

            # Optionally use logarithmic scale for the frequency axis
            plt.xscale("log")

            # Move the legend outside the plot
            plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

            # Save the plot
            filepath = os.path.join(
                output_dir, f"{feature}_enhanced_spectral_density_plot.png"
            )
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()
        except Exception as e:
            self.logger.error(
                f"Error generating spectral density plot for {feature}: {e}"
            )
            return f"Error generating plot for {feature}"
        return filepath

    def _create_plot_spectrogram(
        self,
        feature,
        values,
        output_dir,
        sampling_rate=1,
        nfft=16,
        nperseg=4,
        noverlap=2,
        threshold=0.55,
        seg_overlap=20,
    ):
        """
        Plots a spectrogram for a given time-series data.

        Args:
            time_series (np.array): Time-series data for which to compute the spectrogram.
            sampling_rate (int): The sampling rate of the time-series data.
            output_dir (str): Directory where the spectrogram image will be saved.
            nfft (int): Number of data points used in each block for FFT. Higher values give better frequency resolution.
            noverlap (int): Number of points to overlap between segments.

        Returns:
            str: File path to the saved spectrogram plot image.
        """
        try:
            # Convert values to numpy array
            values = np.array(values)

            # Compute the spectrogram
            frequencies, times, Sxx = spectrogram(
                values, fs=sampling_rate, nfft=nfft, nperseg=nperseg, noverlap=noverlap
            )

            # Convert frequencies from Hz to CPM (Cycles Per Minute) and times to minutes
            frequencies_cpm = frequencies * 60 / seg_overlap
            times_minutes = times / (60 / seg_overlap)

            # Automatically detect ROI based on the power in the spectrogram
            roi_time, roi_freq = self.auto_detect_roi(
                Sxx, times_minutes, frequencies_cpm, threshold
            )

            # Start plotting
            plt.figure(figsize=(10, 6))

            # Plot the spectrogram with a perceptual color map (plasma, magma, or inferno for better intensity contrast)
            plt.pcolormesh(
                times_minutes,
                frequencies_cpm,
                10 * np.log10(Sxx),
                shading="gouraud",
                cmap="magma",
            )
            plt.title(f"Spectrogram of {feature} Over Time", fontsize=14)
            plt.ylabel("Frequency (Cycles Per Minute)", fontsize=12)
            plt.xlabel("Time (Minutes)", fontsize=12)
            plt.colorbar(label="Power (dB)")

            # Highlight areas above a threshold with contour
            plt.contour(
                times_minutes,
                frequencies_cpm,
                10 * np.log10(Sxx),
                levels=[10 * np.log10(threshold)],
                colors="white",
                linewidths=1,
            )

            # Add rectangles for the Region(s) of Interest (ROI)
            roi_rect = Rectangle(
                (roi_time[0], roi_freq[0]),  # Bottom-left corner (x, y)
                roi_time[1] - roi_time[0],  # Width (end_time - start_time)
                roi_freq[1] - roi_freq[0],  # Height (end_freq - start_freq)
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
            )
            plt.gca().add_patch(roi_rect)

            # Annotate the ROI in the plot
            plt.text(
                roi_time[0],
                roi_freq[1],
                "R.o.I",
                color="red",
                fontsize=12,
                verticalalignment="bottom",
                horizontalalignment="left",
            )

            # Save the plot as an image file
            filepath = os.path.join(output_dir, f"{feature}_enhanced_spectrogram.png")
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()
        except Exception as e:
            self.logger.error(f"Error generating spectrogram for {feature}: {e}")
            return f"Error generating plot for {feature}"
        return filepath

    def _create_plot_periodogram(
        self,
        feature,
        values,
        output_dir,
        sampling_rate=1,
        nfft=16,
        threshold=0.55,
        seg_overlap=20,
    ):
        """
        Plots a spectrogram for a given time-series data.

        Args:
            time_series (np.array): Time-series data for which to compute the spectrogram.
            sampling_rate (int): The sampling rate of the time-series data.
            output_dir (str): Directory where the spectrogram image will be saved.
            nfft (int): Number of data points used in each block for FFT. Higher values give better frequency resolution.
            noverlap (int): Number of points to overlap between segments.

        Returns:
            str: File path to the saved spectrogram plot image.
        """
        try:
            # Ensure values are converted to a numpy array for FFT computation
            values = np.array(values)

            # Compute the periodogram
            frequencies, Pxx = periodogram(values, fs=sampling_rate, nfft=nfft)

            # Convert frequencies from Hz to CPM (Cycles Per Minute)
            frequencies_cpm = frequencies * 60

            # Create a vibrant plot
            plt.figure(figsize=(12, 8))

            # Use a lively color palette and thicker line for the periodogram
            plt.semilogy(
                frequencies_cpm, Pxx, color="royalblue", linewidth=2.5, alpha=0.8
            )

            # Highlight the threshold line if applicable
            if threshold is not None:
                plt.axhline(
                    threshold,
                    color="crimson",
                    linestyle="--",
                    label="Threshold",
                    linewidth=1.8,
                    alpha=0.7,
                )

            # Add mean and standard deviation lines for additional context
            mean_power = np.mean(Pxx)
            std_power = np.std(Pxx)
            plt.axhline(
                mean_power,
                color="forestgreen",
                linestyle=":",
                label=f"Mean Power: {mean_power:.2f}",
                linewidth=2.0,
            )
            plt.axhline(
                mean_power + std_power,
                color="orange",
                linestyle=":",
                label=f"Mean + 1 Std Dev: {mean_power + std_power:.2f}",
                linewidth=1.5,
            )
            plt.axhline(
                mean_power - std_power,
                color="orange",
                linestyle=":",
                label=f"Mean - 1 Std Dev: {mean_power - std_power:.2f}",
                linewidth=1.5,
            )

            # Add a fill between the standard deviation range for better visualization
            plt.fill_between(
                frequencies_cpm,
                mean_power - std_power,
                mean_power + std_power,
                color="lightgoldenrodyellow",
                alpha=0.4,
                label="1 Std Dev Range",
            )

            # Add labels and title with enhanced styling
            plt.xlabel("Frequency (Cycles Per Minute)", fontsize=14, fontweight="bold")
            plt.ylabel("Power Spectral Density (PSD)", fontsize=14, fontweight="bold")
            plt.title(
                f"Periodogram of {feature}",
                fontsize=16,
                fontweight="bold",
                color="darkslategray",
            )
            plt.legend(loc="upper right", fontsize=12)

            # Save the plot
            filepath = os.path.join(output_dir, f"{feature}_periodogram_plot.png")
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()
        except Exception as e:
            self.logger.error(f"Error generating periodogram for {feature}: {e}")
            return f"Error generating plot for {feature}"
        return filepath

    def _create_plot_line_with_rolling_stats(
        self, feature, values, output_dir, time=None, window=10
    ):
        """
        Plots a line chart with rolling mean and standard deviation over time.
        Args:
            data (list or np.array): Time series data points.
            time (list or np.array): Corresponding time points.
            output_dir (str): Directory to save the plot.
            window (int): Window size for rolling statistics.
        """
        try:
            if time is None:
                time = np.arange(len(values))

            # Create DataFrame for rolling statistics
            df = pd.DataFrame({"data": values, "time": time})
            df["rolling_mean"] = df["data"].rolling(window=window).mean()
            df["rolling_std"] = df["data"].rolling(window=window).std()
            df["rolling_median"] = df["data"].rolling(window=window).median()
            df["quantile_25"] = df["data"].rolling(window=window).quantile(0.25)
            df["quantile_75"] = df["data"].rolling(window=window).quantile(0.75)

            # Create the plot
            plt.figure(figsize=(12, 6))

            # Plot original data
            plt.plot(df["time"], df["data"], label="Data", color="blue", alpha=0.7)

            # Plot rolling mean and standard deviation bands
            plt.plot(
                df["time"],
                df["rolling_mean"],
                label="Rolling Mean",
                color="orange",
                linestyle="--",
                linewidth=2,
            )

            plt.fill_between(
                df["time"],
                df["rolling_mean"] - df["rolling_std"],
                df["rolling_mean"] + df["rolling_std"],
                color="orange",
                alpha=0.3,
                label="Rolling Std Dev",
            )

            # Plot rolling median and quantiles for additional insights
            plt.plot(
                df["time"],
                df["rolling_median"],
                label="Rolling Median",
                color="green",
                linestyle="--",
                linewidth=2,
            )

            plt.fill_between(
                df["time"],
                df["quantile_25"],
                df["quantile_75"],
                color="green",
                alpha=0.2,
                label="Interquartile Range (25-75%)",
            )

            # Highlight the normal range if provided
            normal_min, normal_max = self._fetch_and_validate_normal_range(
                feature, values[0]
            )
            plt.axhline(
                normal_min,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Normal Min: {normal_min}",
            )
            plt.axhline(
                normal_max,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Normal Max: {normal_max}",
            )

            # Highlight important events (peaks/troughs) in the data
            peaks = find_peaks(df["data"])
            troughs = find_peaks(-df["data"])

            plt.scatter(
                df["time"].iloc[peaks],
                df["data"].iloc[peaks],
                color="purple",
                s=100,
                label="Peaks",
                zorder=5,
            )
            plt.scatter(
                df["time"].iloc[troughs],
                df["data"].iloc[troughs],
                color="cyan",
                s=100,
                label="Troughs",
                zorder=5,
            )

            # Add titles, labels, and legends
            plt.title(
                f"Time-Series Data with Enhanced Rolling Statistics ({feature})",
                fontsize=14,
            )
            plt.xlabel("Time", fontsize=12)
            plt.ylabel("Values", fontsize=12)

            plt.legend(loc="upper right", fontsize=10, frameon=True)
            plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.6)

            # Save the plot
            filepath = os.path.join(
                output_dir, f"{feature}_line_with_enhanced_stats.png"
            )
            plt.savefig(filepath, bbox_inches="tight", dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(
                f"Error creating line chart with rolling statistics for {feature}: {e}"
            )
            return f"Error generating plot for {feature}"
        return filepath

    def _create_plot_autocorrelation(self, feature, values, output_dir, lags=5):
        """
        Plots the autocorrelation function (ACF) for the time-series data.
        Args:
            data (list or np.array): Time series data points.
            output_dir (str): Directory to save the plot.
            lags (int): Number of lags to plot.
        """
        try:
            plt.figure(figsize=(10, 6))
            plot_acf(values, lags=lags)
            plt.title("Autocorrelation Plot")
            filepath = os.path.join(output_dir, f"{feature}_autocorrelation_plot.png")
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()
        except Exception as e:
            self.logger.error(f"Error creating autocorrelation plot for {feature}: {e}")
            return f"Error generating plot for {feature}"
        return filepath

    def _create_box_swarm_plot(self, feature, values, output_dir):
        """
        Creates an enhanced box plot combined with a swarm plot for visualizing the feature value compared to the normal range.

        Args:
            feature (str): The name of the feature.
            values (list): The values of the feature for the current segment.
            output_dir (str): Directory where the plot will be saved.

        Returns:
            str: Path to the saved box + swarm plot image.
        """
        try:
            normal_min, normal_max = self._fetch_and_validate_normal_range(
                feature, values[0]
            )

            # Define the outlier threshold
            outlier_min = normal_min - 1.5 * (normal_max - normal_min)
            outlier_max = normal_max + 1.5 * (normal_max - normal_min)

            # Cap the values to fit within the plot and mark outliers
            capped_values = []
            for value in values:
                if value < outlier_min:
                    capped_values.append(outlier_min)
                elif value > outlier_max:
                    capped_values.append(outlier_max)
                else:
                    capped_values.append(value)

            # Set up the figure
            plt.figure(figsize=(10, 6))
            sns.set(style="whitegrid")

            # Create boxplot for overall distribution
            sns.boxplot(capped_values, color="#D0E1F9", linewidth=2, width=0.3)

            # Overlay swarm plot for individual data points with enhanced appearance
            sns.swarmplot(
                capped_values, color="#34495E", size=6, edgecolor="black", alpha=0.8
            )

            # Add shaded normal range
            plt.fill_between(
                [-0.4, 0.4],
                normal_min,
                normal_max,
                color="#A9DFBF",
                alpha=0.25,
                label="Normal Range",
            )

            # Add statistical markers for key values
            mean_value = (normal_max + normal_min) / 2
            plt.axhline(
                mean_value,
                color="#27AE60",
                linestyle="--",
                label=f"Normal Mean: {mean_value:.2f}",
            )
            plt.axhline(
                normal_min,
                color="#E74C3C",
                linestyle="--",
                alpha=0.6,
                label=f"Normal Min: {normal_min:.2f}",
            )
            plt.axhline(
                normal_max,
                color="#E74C3C",
                linestyle="--",
                alpha=0.6,
                label=f"Normal Max: {normal_max:.2f}",
            )

            # Style enhancements for modern appearance
            plt.title(
                f"{feature} Distribution and Data Points", fontsize=16, color="#2C3E50"
            )
            plt.xlabel("")
            plt.ylabel(f"{feature} Values", fontsize=12, color="#2C3E50")
            plt.xticks([])  # No x-ticks since there's only one category

            # Add legend
            plt.legend(
                loc="upper right",
                fontsize=10,
                frameon=True,
                fancybox=True,
                framealpha=0.7,
                shadow=True,
            )

            # Improve layout aesthetics
            plt.grid(visible=False)
            sns.despine(left=True, bottom=True)  # Remove spines for a cleaner look

            # Add labels for mean, min, max (optional but informative)
            plt.text(
                0.35,
                mean_value,
                f"{mean_value:.2f}",
                verticalalignment="center",
                color="#27AE60",
            )
            plt.text(
                0.35,
                normal_min,
                f"{normal_min:.2f}",
                verticalalignment="center",
                color="#E74C3C",
            )
            plt.text(
                0.35,
                normal_max,
                f"{normal_max:.2f}",
                verticalalignment="center",
                color="#E74C3C",
            )

            # Save the plot
            filepath = os.path.join(
                output_dir, f"{feature}_enhanced_box_swarm_plot.png"
            )
            plt.savefig(
                filepath, bbox_inches="tight", dpi=300
            )  # Save with higher resolution for clarity
            plt.close()
        except Exception as e:
            self.logger.error(
                f"Error creating enhanced box + swarm plot for {feature}: {e}"
            )
            return f"Error generating plot for {feature}"
        return filepath

    def _create_plot_lag(self, feature, values, output_dir, lags=3):
        """
        Creates an enhanced lag plot for the time-series data with additional visual cues like trend line,
        density coloring, and correlation coefficient.

        Args:
            feature (str): Name of the feature to label the plot.
            values (list or np.array): Time series data points.
            output_dir (str): Directory to save the plot.
            lags (int): Number of lags to plot.

        Returns:
            str: Path to the saved enhanced lag plot image.
        """
        try:
            # Ensure values are converted to a numpy array for indexing
            values = np.array(values)

            # Ensure that data_lagged and data_original have matching lengths
            if len(values) > lags:
                data_lagged = values[:-lags]
                data_original = values[lags:]

                # Create the enhanced lag plot
                plt.figure(figsize=(10, 6))

                # Use Seaborn's kdeplot to show density distribution
                sns.kdeplot(
                    x=data_lagged,
                    y=data_original,
                    fill=True,
                    cmap="Blues",
                    thresh=0.05,
                    alpha=0.5,
                )

                # Plot scatter points
                plt.scatter(
                    data_lagged,
                    data_original,
                    alpha=0.6,
                    label="Data Points",
                    color="royalblue",
                    s=50,
                )

                # Plot a trend line using a simple linear regression (polyfit)
                coeffs = np.polyfit(data_lagged, data_original, 1)
                trend_line = np.polyval(coeffs, data_lagged)
                plt.plot(
                    data_lagged,
                    trend_line,
                    color="darkorange",
                    label="Trend Line",
                    linewidth=2,
                    linestyle="--",
                )

                # Calculate and display Pearson correlation coefficient
                corr_coeff, _ = pearsonr(data_lagged, data_original)
                plt.text(
                    0.05,
                    0.95,
                    f"Correlation (r) = {corr_coeff:.3f}",
                    transform=plt.gca().transAxes,
                    fontsize=12,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

                # Additional Information: Mean and Median of Original Values
                mean_original = np.mean(data_original)
                median_original = np.median(data_original)
                plt.axhline(
                    mean_original,
                    color="green",
                    linestyle=":",
                    label=f"Mean: {mean_original:.2f}",
                )
                plt.axhline(
                    median_original,
                    color="purple",
                    linestyle=":",
                    label=f"Median: {median_original:.2f}",
                )

                # Labels and title
                plt.title(
                    f"Enhanced Lag Plot of {feature} with Lag = {lags}", fontsize=16
                )
                plt.xlabel(f"{feature} (t)", fontsize=14)
                plt.ylabel(f"{feature} (t+{lags})", fontsize=14)

                # Display the legend
                plt.legend(loc="upper left", fontsize=12)

                # Save the plot
                filepath = os.path.join(output_dir, f"{feature}_enhanced_lag_plot.png")
                plt.savefig(filepath, bbox_inches="tight")
                plt.close()

                return filepath
            else:
                raise ValueError(
                    f"The length of the data ({len(values)}) must be greater than the lag ({lags})."
                )
        except Exception as e:
            self.logger.error(f"Error creating enhanced lag plot for {feature}: {e}")
            return f"Error generating plot for {feature}"

    def _create_plot_seasonal_decomposition(
        self, feature, values, output_dir, freq=1, time=None
    ):
        """
        Decomposes the time-series data into trend, seasonality, and residuals.
        Args:
            data (list or np.array): Time series data points.
            time (list or np.array): Corresponding time points.
            output_dir (str): Directory to save the plot.
            freq (int): Frequency of seasonality in data (e.g., 12 for monthly data).
        """
        try:
            if time is None:
                time = np.arange(len(values))
            df = pd.DataFrame({"data": values}, index=pd.to_datetime(time))
            decomposition = seasonal_decompose(
                df["data"], model="additive", period=freq
            )

            fig = decomposition.plot()
            fig.set_size_inches(10, 8)
            plt.suptitle("Seasonal Decomposition of Time-Series Data", y=1.05)

            filepath = os.path.join(output_dir, f"{feature}_seasonal_decomposition.png")
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()
        except Exception as e:
            self.logger.error(
                f"Error creating seasonal decomposition plot for {feature}: {e}"
            )
            return f"Error generating plot for {feature}"
        return filepath

    def _create_violin_plot(self, feature, values, output_dir):
        """
        Creates an enhanced violin plot for visualizing the feature value compared to the normal range.

        Args:
            feature (str): The name of the feature.
            value (list): The values of the feature for the current segment.
            output_dir (str): Directory where the plot will be saved.

        Returns:
            str: Path to the saved violin plot image.
        """
        try:
            # Fetch normal range
            normal_min, normal_max = self._fetch_and_validate_normal_range(
                feature, values[0]
            )

            # Define the outlier threshold
            outlier_min = normal_min - 1.5 * (normal_max - normal_min)
            outlier_max = normal_max + 1.5 * (normal_max - normal_min)

            # Cap the values to fit within the plot and mark outliers
            capped_values = np.clip(values, outlier_min, outlier_max)

            # Create data for the violin plot (normal distribution)
            # data = np.random.normal(
            #     loc=(normal_max + normal_min) / 2,
            #     scale=(normal_max - normal_min) / 6,
            #     size=1000,
            # )

            plt.figure(figsize=(10, 7))

            # Create the violin plot for the capped values with updated density_norm parameter
            sns.violinplot(
                data=capped_values,
                color="lightgreen",
                inner="quartile",
                linewidth=2,
                density_norm="area",
            )

            # Overlay shaded normal range
            plt.fill_betweenx(
                [normal_min, normal_max],
                0.25,
                0.75,
                color="lightblue",
                alpha=0.5,
                label="Normal Range",
            )

            # Add horizontal lines for the normal range statistics
            mean_value = (normal_max + normal_min) / 2
            plt.axhline(
                mean_value,
                color="green",
                linestyle="--",
                label=f"Normal Range Mean: {mean_value:.2f}",
            )
            plt.axhline(
                normal_min,
                color="blue",
                linestyle="--",
                alpha=0.6,
                label=f"Normal Min: {normal_min:.2f}",
            )
            plt.axhline(
                normal_max,
                color="red",
                linestyle="--",
                alpha=0.6,
                label=f"Normal Max: {normal_max:.2f}",
            )

            # Plot each capped value with dynamic scatter sizes
            value_counts = Counter(capped_values)
            for value, count in value_counts.items():
                color = (
                    "red" if value == outlier_min or value == outlier_max else "blue"
                )
                plt.scatter(
                    [0.5] * count,
                    [value] * count,
                    color=color,
                    zorder=5,
                    s=50 + count * 10,
                    edgecolors="black",
                )

            # Add statistical information
            median_value = np.median(capped_values)
            plt.scatter(
                0.5,
                median_value,
                color="purple",
                zorder=6,
                s=150,
                edgecolors="black",
                label=f"Median: {median_value:.2f}",
            )

            # Titles and labels
            plt.title(f"{feature} Enhanced Violin Plot", fontsize=16)
            plt.ylabel(f"{feature} Values", fontsize=14)
            plt.xticks([])  # Remove x-ticks as there's only one plot axis

            # Add a legend
            plt.legend(loc="upper right", fontsize=12)

            # Save the plot
            filepath = os.path.join(output_dir, f"{feature}_enhanced_violin_plot.png")
            plt.savefig(filepath, bbox_inches="tight", dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error creating enhanced violin plot for {feature}: {e}")
            return f"Error generating plot for {feature}"

        return filepath

    def _create_heatmap_plot(self, feature, values, output_dir):
        """
        Creates a heatmap plot for visualizing the feature values with highlights for the normal range.

        Args:
            feature (str): The name of the feature.
            values (list): The list of values for the current segment.
            output_dir (str): Directory where the plot will be saved.

        Returns:
            str: Path to the saved heatmap plot image.
        """
        try:
            normal_min, normal_max = self._fetch_and_validate_normal_range(
                feature, values[0]
            )

            # Set outlier thresholds
            outlier_min = normal_min - 1.5 * (normal_max - normal_min)
            outlier_max = normal_max + 1.5 * (normal_max - normal_min)

            # Create x-axis values for the heatmap
            x = np.linspace(
                normal_min - 1.5 * (normal_max - normal_min),
                normal_max + 1.5 * (normal_max - normal_min),
                100,
            )
            y = np.exp(
                -((x - np.mean([normal_min, normal_max])) ** 2)
                / (2 * ((normal_max - normal_min) / 2) ** 2)
            )
            heatmap_data = np.outer(y, y)

            plt.figure(figsize=(8, 4))
            sns.heatmap(
                heatmap_data,
                cmap="coolwarm",
                cbar=False,
                xticklabels=False,
                yticklabels=False,
            )

            # Track the number of occurrences for each value
            value_counts = {}
            for value in values:
                capped_value = min(max(value, outlier_min), outlier_max)
                if capped_value in value_counts:
                    value_counts[capped_value] += 1
                else:
                    value_counts[capped_value] = 1

            # Plot scatter points with size based on frequency of the value
            for capped_value, count in value_counts.items():
                value_interp = np.interp(capped_value, x, np.linspace(0, 100, len(x)))
                scatter_size = (
                    100 + (count - 1) * 50
                )  # Increase size for repeated values
                plt.scatter(
                    [value_interp],
                    [50],
                    color="#3498db",
                    s=scatter_size,
                    zorder=5,
                    edgecolor="white",
                )

            # Mark the normal range
            plt.axvline(
                np.interp(normal_min, x, np.linspace(0, 100, len(x))),
                color="blue",
                linestyle="-",
                label="Normal Min",
            )
            plt.axvline(
                np.interp(normal_max, x, np.linspace(0, 100, len(x))),
                color="red",
                linestyle="-",
                label="Normal Max",
            )

            plt.title(f"Feature: {feature}\nNormal Range: [{normal_min}, {normal_max}]")
            plt.legend()

            # Save heatmap
            filepath = os.path.join(output_dir, f"{feature}_heatmap.png")
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()
        except Exception as e:
            self.logger.error(f"Error creating heatmap plot for {feature}: {e}")
            return f"Error generating plot for {feature}"
        return filepath

    def _create_radar_plot(self, feature, values, output_dir):
        """
        Creates an enhanced radar plot that shows how the feature value compares to the normal range.
        Each axis represents different statistics of the signal, and the plot shows how the current feature deviates
        from its normal range.

        Args:
            feature (str): The name of the feature.
            values (list): The list of values for the current segment.
            output_dir (str): Directory where the plot will be saved.

        Returns:
            str: Path to the saved radar plot image.
        """
        try:
            # Fetch normal range
            normal_min, normal_max = self._fetch_and_validate_normal_range(
                feature, values[0]
            )
            feature_names = ["Min Range", "Max Range", "Mid Range", "Mean", "Std Dev"]

            # Handle NaN and Inf cases
            if np.isnan(normal_min) or np.isnan(normal_max) or np.any(np.isnan(values)):
                raise ValueError(f"NaN value encountered in feature '{feature}'.")

            if np.isinf(normal_min):
                normal_min = -10 * abs(np.min(values))
            if np.isinf(normal_max):
                normal_max = 10 * abs(np.max(values))

            # Outlier thresholds
            outlier_min = normal_min - 1.5 * (normal_max - normal_min)
            outlier_max = normal_max + 1.5 * (normal_max - normal_min)

            # Cap the values to fit within the plot and mark outliers
            capped_values = np.clip(values, outlier_min, outlier_max)

            # Statistical metrics for radar plot
            mean_value = np.mean(capped_values)
            std_value = np.std(capped_values)
            # Ensure that all statistical values are finite
            if not np.isfinite(mean_value) or not np.isfinite(std_value):
                mean_value = 0  # Fallback value for plotting purposes
                std_value = 0

            # Prepare the "normal range" triangle data (min, max, and mid-point of normal range)
            normal_values = [
                normal_min,
                normal_max,
                (normal_min + normal_max) / 2,
                mean_value,
                std_value,
            ]
            normal_values += normal_values[:1]  # Complete the loop

            # Prepare the "current value" triangle data: min, max, median, mean, std of capped values
            current_value_triangle = [
                np.min(capped_values),
                np.max(capped_values),
                np.median(capped_values),
                mean_value,
                std_value,
            ]
            current_value_triangle += current_value_triangle[:1]  # Complete the loop

            # Calculate angles for radar plot
            angles = np.linspace(
                0, 2 * np.pi, len(feature_names), endpoint=False
            ).tolist()
            angles += angles[:1]  # Complete the loop for radar chart

            # Start plot
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

            # Configure axes
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            plt.xticks(angles[:-1], feature_names, color="#34495E", size=10)

            # Set radial limits based on min and max of normal and capped data
            y_min = min(normal_min, np.min(capped_values))
            y_max = max(normal_max, np.max(capped_values))
            plt.ylim(y_min, y_max)

            # Plot normal range triangle
            ax.plot(
                angles,
                normal_values,
                linewidth=2,
                linestyle="solid",
                color="#16A085",
                label="Normal Range",
            )
            ax.fill(angles, normal_values, color="#1ABC9C", alpha=0.3)

            # Plot actual values triangle (min, max, median, mean, std)
            ax.plot(
                angles,
                current_value_triangle,
                linewidth=2,
                linestyle="solid",
                color="#2980B9",
                label="Actual Values",
            )
            ax.fill(angles, current_value_triangle, color="#3498DB", alpha=0.3)

            # Add a scatter point for the median value on the third axis
            median_value = np.median(capped_values)
            ax.scatter(
                angles[2],
                median_value,
                color="#E74C3C",
                zorder=5,
                s=150,
                edgecolors="black",
            )

            # Mark only one outlier in the plot
            outlier_detected = False
            for value in values:
                if value < outlier_min or value > outlier_max and np.isfinite(value):
                    if not outlier_detected:  # Only plot the first outlier
                        ax.scatter(
                            angles[2],
                            median_value,
                            color="#E67E22",
                            zorder=5,
                            s=200,
                            edgecolors="black",
                            label="Outlier",
                        )
                        outlier_detected = True

            # Move the legend outside the plot
            plt.legend(
                loc="upper right",
                bbox_to_anchor=(
                    1.3,
                    1.1,
                ),
                frameon=True,
                framealpha=0.7,
                shadow=True,
                fontsize=10,
            )  # Adjust the position

            # Modern gridlines and spines
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
            ax.spines["polar"].set_visible(False)

            # Save the plot
            filepath = os.path.join(output_dir, f"{feature}_radar_plot.png")
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()
        except Exception as e:
            self.logger.error(f"Error creating radar plot for {feature}: {e}")
            return f"Error generating plot for {feature}"
        return filepath

    def _get_normal_range_for_feature(self, feature):
        """
        Retrieves the normal range for a given feature from the loaded configuration.

        Args:
            feature (str): The name of the feature to get the normal range for.

        Returns:
            tuple: The (min, max) normal range values for the feature, or None if not found.

        Example Usage:
            >>> normal_range = visualization._get_normal_range_for_feature("RMSSD")
            >>> print(normal_range)  # Output: (20, 100)
        """
        feature_info = self.config.get(feature, {})
        normal_range = feature_info.get("normal_range", {}).get(
            self.segment_duration, None
        )
        if normal_range is not None:
            # Handle string '-inf' and 'inf' cases
            normal_range = [self._parse_inf_values(val) for val in normal_range]
        return normal_range

    def _parse_inf_values(self, val):
        """
        Parses 'inf' and '-inf' strings and converts them to numpy infinity values.

        Args:
            val (str or float): The value to parse.

        Returns:
            float: Parsed value where 'inf' or '-inf' are converted to np.inf or -np.inf respectively.

        Example Usage:
            >>> value = visualization._parse_inf_values("-inf")
            >>> print(value)  # Output: -inf
        """
        if isinstance(val, str):
            if val.lower() == "inf":
                return np.inf
            elif val.lower() == "-inf":
                return -np.inf
        return val

    def auto_detect_roi(self, Sxx, times, frequencies, threshold=0.4):
        """
        Automatically detects the region of interest (ROI) in the spectrogram based on power.

        Args:
            Sxx (2D array): Spectrogram values.
            times (1D array): Time values corresponding to the spectrogram.
            frequencies (1D array): Frequency values corresponding to the spectrogram.
            threshold (float): A power threshold (percentage of max) for detecting regions of interest.

        Returns:
            roi_time (tuple): Start and end time for the region of interest.
            roi_freq (tuple): Start and end frequency for the region of interest.
        """
        # Normalize the spectrogram power to the range [0, 1]
        normalized_Sxx = Sxx / np.max(Sxx)

        # Find areas where power exceeds the threshold
        mask = normalized_Sxx > threshold

        # Find time and frequency indices that exceed the threshold
        freq_indices, time_indices = np.where(mask)

        # If no ROI is detected, return the full time and frequency range
        if len(time_indices) == 0 or len(freq_indices) == 0:
            return (times[0], times[-1]), (frequencies[0], frequencies[-1])

        # Find the min and max time and frequency indices that exceed the threshold
        min_time_idx = np.min(time_indices)
        max_time_idx = np.max(time_indices)
        min_freq_idx = np.min(freq_indices)
        max_freq_idx = np.max(freq_indices)

        # Determine ROI time based on detected indices
        roi_time_start = times[min_time_idx]
        roi_time_end = times[max_time_idx]

        # Determine ROI frequency based on detected indices
        roi_freq_start = frequencies[min_freq_idx]
        roi_freq_end = frequencies[max_freq_idx]

        return (roi_time_start, roi_time_end), (roi_freq_start, roi_freq_end)
