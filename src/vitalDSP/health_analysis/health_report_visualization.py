"""
Health Analysis Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations
- SciPy integration for advanced signal processing
- Interactive visualization capabilities

Examples:
---------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.health_analysis.health_report_visualization import HealthReportVisualization
    >>> signal = np.random.randn(1000)
    >>> hrv = HealthReportVisualizer(config={'sdnn': {'normal_range': {'1_min': [30,70]}}})
    >>> result = hrv.create_visualizations({'sdnn': [35,40,45]}, output_dir='./out')
    >>> print(f'Processing result: {result}')
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
import pandas as pd
from vitalDSP.utils.config_utilities.common import find_peaks
import logging
import threading

# Set thread-safe matplotlib backend
matplotlib.use("Agg")
# Enable thread safety
matplotlib.rcParams["figure.max_open_warning"] = 0

# Create a global lock for matplotlib operations
matplotlib_lock = threading.Lock()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Known units for common HRV/physiological features
FEATURE_UNITS = {
    "sdnn": "ms",
    "rmssd": "ms",
    "mean_nn": "ms",
    "median_nn": "ms",
    "nn50": "count",
    "pnn50": "%",
    "lf_power": "ms²",
    "hf_power": "ms²",
    "lf_hf_ratio": "ratio",
    "heart_rate": "bpm",
    "spo2": "%",
    "perfusion_index": "%",
    "systolic_duration": "ms",
    "diastolic_duration": "ms",
}


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

    def _thread_safe_matplotlib_operation(self, func, *args, **kwargs):
        """
        Execute matplotlib operations in a thread-safe manner.

        Args:
            func: The matplotlib function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function execution
        """
        with matplotlib_lock:
            try:
                # Clear any existing figures to prevent conflicts
                plt.clf()
                plt.close("all")

                # Execute the function
                result = func(*args, **kwargs)

                # Ensure proper cleanup
                plt.clf()
                return result
            except Exception as e:
                # Clean up on error
                plt.clf()
                plt.close("all")
                raise e

    def _normalize_web_path(self, filepath):
        """
        Normalize file path for web usage by ensuring forward slashes.

        Args:
            filepath (str): The file path to normalize.

        Returns:
            str: Web-compatible path with forward slashes.
        """
        if filepath is None:
            return None
        # Convert backslashes to forward slashes for web compatibility
        return filepath.replace("\\", "/")

    def _fetch_and_validate_normal_range(self, feature, value):
        """
        Fetches the normal range for a given feature from config. Returns config bounds
        unchanged — does NOT corrupt them with the current value.

        Args:
            feature (str): The feature name.
            value (float): The current value (used only to validate finiteness).

        Returns:
            tuple: (normal_min, normal_max) from config, with inf substituted when
                   the config bound itself is inf and value provides a fallback.

        Raises:
            ValueError: If the feature is not found or all values are NaN/Inf.
        """
        normal_range = self._get_normal_range_for_feature(feature)
        if not normal_range:
            raise ValueError(f"Normal range for feature '{feature}' not found.")

        config_min, config_max = normal_range

        # Substitute infinite config bounds using value as a fallback only
        if np.isinf(config_min):
            config_min = value * 0.5 if np.isfinite(value) and value != 0 else -1e6
        if np.isinf(config_max):
            config_max = value * 1.5 if np.isfinite(value) and value != 0 else 1e6

        if np.isnan(config_min) or np.isnan(config_max):
            raise ValueError(
                f"Normal range for feature '{feature}' contains NaN after substitution."
            )

        return config_min, config_max

    def _get_plot_bounds(self, normal_min, normal_max, values):
        """
        Returns axis bounds that encompass both the normal range and all finite values.
        Adds 10% padding.
        """
        finite_vals = [v for v in values if np.isfinite(v)]
        lo = min(normal_min, min(finite_vals)) if finite_vals else normal_min
        hi = max(normal_max, max(finite_vals)) if finite_vals else normal_max
        pad = max((hi - lo) * 0.10, 1e-6)
        return lo - pad, hi + pad

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

            charts = {
                "gauge_chart": self._normalize_web_path(
                    self._create_gauge_chart(feature, values, output_dir)
                ),
                "violin_plot": self._normalize_web_path(
                    self._create_violin_plot(feature, values, output_dir)
                ),
                "line_with_rolling_stats": self._normalize_web_path(
                    self._create_plot_line_with_rolling_stats(
                        feature, values, output_dir
                    )
                ),
                "plot_box_swarm": self._normalize_web_path(
                    self._create_box_swarm_plot(feature, values, output_dir)
                ),
            }

            sparkline = self._create_trend_sparkline(feature, values, output_dir)
            if sparkline is not None:
                charts["trend_sparkline"] = self._normalize_web_path(sparkline)

            visualization_paths[feature] = charts

        return visualization_paths

    # ------------------------------------------------------------------
    # Gauge chart (replaces bell_plot)
    # ------------------------------------------------------------------

    def _create_gauge_chart(self, feature, values, output_dir):
        """
        Creates a gauge/speedometer chart showing the most recent value vs the normal range.
        When multiple segments exist, also plots a small trend inset.

        Args:
            feature (str): The name of the feature.
            values (list): Scalar feature values per segment.
            output_dir (str): Directory where the plot will be saved.

        Returns:
            str: Path to the saved gauge chart image.
        """
        try:
            normal_min, normal_max = self._fetch_and_validate_normal_range(
                feature, values[-1]
            )
            current = values[-1]
            if not np.isfinite(current):
                current = normal_min

            mid = (normal_min + normal_max) / 2
            span = normal_max - normal_min if normal_max != normal_min else 1.0

            # Map value to gauge angle: 180° (left/below) → 0° (right/above)
            # normal_min → 135°, normal_max → 45° (semicircle with margins)
            gauge_min_val = normal_min - 0.5 * span
            gauge_max_val = normal_max + 0.5 * span
            gauge_span = gauge_max_val - gauge_min_val

            def val_to_angle(v):
                # 180° at gauge_min_val, 0° at gauge_max_val
                frac = np.clip((v - gauge_min_val) / gauge_span, 0, 1)
                return np.radians(180 - frac * 180)

            # Determine status
            if current < normal_min:
                status = "BELOW"
                status_color = "#e74c3c"
            elif current > normal_max:
                status = "ABOVE"
                status_color = "#e67e22"
            else:
                status = "IN RANGE"
                status_color = "#27ae60"

            has_trend = len(values) >= 3
            fig_w = 10 if has_trend else 7
            fig, ax = plt.subplots(1, 1, figsize=(fig_w, 5))
            ax.set_aspect("equal")
            ax.axis("off")

            # Draw arc zones: below normal (red), normal (green), above normal (orange/red)
            theta = np.linspace(np.pi, 0, 300)
            r_inner, r_outer = 0.55, 0.90

            # Convert zone boundaries to angles
            ang_min = val_to_angle(normal_min)
            ang_max = val_to_angle(normal_max)
            ang_gauge_min = np.pi  # leftmost
            ang_gauge_max = 0.0  # rightmost

            def draw_arc_zone(a_start, a_end, color, alpha=0.85):
                angles = np.linspace(a_start, a_end, 60)
                x_outer = r_outer * np.cos(angles)
                y_outer = r_outer * np.sin(angles)
                x_inner = r_inner * np.cos(angles[::-1])
                y_inner = r_inner * np.sin(angles[::-1])
                xs = np.concatenate([x_outer, x_inner])
                ys = np.concatenate([y_outer, y_inner])
                ax.fill(xs, ys, color=color, alpha=alpha, zorder=2)

            draw_arc_zone(ang_gauge_min, ang_min, "#e74c3c")  # below-normal zone
            draw_arc_zone(ang_min, ang_max, "#27ae60")  # normal zone
            draw_arc_zone(ang_max, ang_gauge_max, "#e67e22")  # above-normal zone

            # Arc border
            arc_theta = np.linspace(np.pi, 0, 200)
            ax.plot(
                r_outer * np.cos(arc_theta),
                r_outer * np.sin(arc_theta),
                "k-",
                lw=1.5,
                zorder=3,
            )
            ax.plot(
                r_inner * np.cos(arc_theta),
                r_inner * np.sin(arc_theta),
                "k-",
                lw=1.0,
                zorder=3,
            )

            # Tick marks at normal_min, mid, normal_max, and current value
            tick_vals = [normal_min, mid, normal_max]
            for tv in tick_vals:
                ta = val_to_angle(tv)
                ax.plot(
                    [r_inner * np.cos(ta), r_outer * np.cos(ta)],
                    [r_inner * np.sin(ta), r_outer * np.sin(ta)],
                    "k-",
                    lw=2,
                    zorder=4,
                )
                ax.text(
                    (r_outer + 0.07) * np.cos(ta),
                    (r_outer + 0.07) * np.sin(ta),
                    f"{tv:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    zorder=5,
                )

            # Needle
            needle_angle = val_to_angle(current)
            needle_len = 0.75
            ax.annotate(
                "",
                xy=(
                    needle_len * np.cos(needle_angle),
                    needle_len * np.sin(needle_angle),
                ),
                xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="-|>", color="#2c3e50", lw=2.5, mutation_scale=20
                ),
                zorder=6,
            )
            ax.add_patch(plt.Circle((0, 0), 0.06, color="#2c3e50", zorder=7))

            # Center annotation
            unit = FEATURE_UNITS.get(feature.lower(), "")
            label_unit = f" {unit}" if unit else ""
            ax.text(
                0,
                -0.20,
                f"{current:.2f}{label_unit}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=status_color,
                zorder=8,
            )
            ax.text(
                0,
                -0.35,
                status,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=status_color,
                zorder=8,
            )
            ax.text(
                0,
                -0.48,
                f"Normal: [{normal_min:.1f} – {normal_max:.1f}]{label_unit}",
                ha="center",
                va="center",
                fontsize=9,
                color="#555",
                zorder=8,
            )

            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.65, 1.1)
            ax.set_title(
                f"{feature} — Current Value vs Normal Range", fontsize=13, pad=12
            )

            # Trend inset when enough segments
            if has_trend:
                ax_inset = fig.add_axes([0.72, 0.15, 0.24, 0.35])
                seg_idx = np.arange(len(values))
                finite_vals = [v if np.isfinite(v) else np.nan for v in values]
                ax_inset.plot(seg_idx, finite_vals, "b-o", ms=4, lw=1.5)
                ax_inset.axhspan(normal_min, normal_max, color="#27ae60", alpha=0.15)
                ax_inset.axhline(normal_min, color="#27ae60", lw=0.8, ls="--")
                ax_inset.axhline(normal_max, color="#27ae60", lw=0.8, ls="--")
                ax_inset.set_xlabel("Segment", fontsize=7)
                ax_inset.set_ylabel("Value", fontsize=7)
                ax_inset.tick_params(labelsize=6)
                ax_inset.set_title("Trend", fontsize=8)

            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{feature}_gauge_chart.png")
            plt.savefig(filepath, bbox_inches="tight", dpi=150)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error generating gauge chart for {feature}: {e}")
            return f"Error generating plot for {feature}"
        return filepath

    # ------------------------------------------------------------------
    # Trend sparkline (replaces lag_plot)
    # ------------------------------------------------------------------

    def _create_trend_sparkline(self, feature, values, output_dir):
        """
        Compact line chart showing how a feature's value changed across segments.
        Returns None for single-segment reports.

        Args:
            feature (str): The name of the feature.
            values (list): Scalar feature values per segment.
            output_dir (str): Directory where the plot will be saved.

        Returns:
            str or None: Path to the saved sparkline image, or None if < 3 segments.
        """
        if len(values) < 3:
            return None
        try:
            normal_min, normal_max = self._fetch_and_validate_normal_range(
                feature, values[0]
            )
            seg_idx = np.arange(len(values))
            finite_mask = np.array([np.isfinite(v) for v in values])
            vals = np.array([v if np.isfinite(v) else np.nan for v in values])

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.fill_between(
                seg_idx,
                normal_min,
                normal_max,
                color="#27ae60",
                alpha=0.12,
                label="Normal range",
            )
            ax.axhline(normal_min, color="#27ae60", lw=0.8, ls="--", alpha=0.7)
            ax.axhline(normal_max, color="#27ae60", lw=0.8, ls="--", alpha=0.7)
            ax.plot(seg_idx, vals, color="#3498db", lw=1.5, zorder=3)

            # Color dots by in/out of range
            in_range = finite_mask & (vals >= normal_min) & (vals <= normal_max)
            out_range = finite_mask & ~in_range
            ax.scatter(
                seg_idx[in_range], vals[in_range], color="#27ae60", s=40, zorder=4
            )
            ax.scatter(
                seg_idx[out_range], vals[out_range], color="#e74c3c", s=40, zorder=4
            )

            unit = FEATURE_UNITS.get(feature.lower(), "")
            ylabel = f"{feature} ({unit})" if unit else feature
            ax.set_xlabel("Segment", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f"{feature} — Trend Across Segments", fontsize=11)
            ax.set_xticks(seg_idx)
            lo, hi = self._get_plot_bounds(normal_min, normal_max, values)
            ax.set_ylim(lo, hi)
            ax.grid(True, ls="--", alpha=0.4)
            plt.tight_layout()

            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{feature}_trend_sparkline.png")
            plt.savefig(filepath, bbox_inches="tight", dpi=150)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error generating trend sparkline for {feature}: {e}")
            return f"Error generating plot for {feature}"
        return filepath

    # ------------------------------------------------------------------
    # Violin plot (fixed)
    # ------------------------------------------------------------------

    def _create_violin_plot(self, feature, values, output_dir):
        """
        Creates an enhanced violin plot for visualizing the feature value compared to the normal range.
        Falls back to box+swarm when len(values) < 4 (KDE requires sufficient data).

        Args:
            feature (str): The name of the feature.
            values (list): The values of the feature for the current segment.
            output_dir (str): Directory where the plot will be saved.

        Returns:
            str: Path to the saved violin plot image.
        """
        if len(values) < 4:
            return self._create_box_swarm_plot(feature, values, output_dir)

        try:
            normal_min, normal_max = self._fetch_and_validate_normal_range(
                feature, values[0]
            )

            outlier_min = normal_min - 1.5 * (normal_max - normal_min)
            outlier_max = normal_max + 1.5 * (normal_max - normal_min)
            capped_values = np.clip(values, outlier_min, outlier_max)

            plt.figure(figsize=(8, 7))

            sns.violinplot(
                data=capped_values,
                color="lightgreen",
                inner="quartile",
                linewidth=2,
                density_norm="area",
            )

            # Normal range shading — violin is centered at x=0, extends ±0.5
            plt.fill_betweenx(
                [normal_min, normal_max],
                -0.5,
                0.5,
                color="lightblue",
                alpha=0.5,
                label="Normal Range",
            )

            mean_value = (normal_max + normal_min) / 2
            plt.axhline(
                mean_value,
                color="green",
                linestyle="--",
                label=f"Normal Mean: {mean_value:.2f}",
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

            # Scatter individual values at x=0 (violin center)
            from collections import Counter

            value_counts = Counter(capped_values)
            for value, count in value_counts.items():
                color = (
                    "red" if value == outlier_min or value == outlier_max else "blue"
                )
                plt.scatter(
                    [0] * count,
                    [value] * count,
                    color=color,
                    zorder=5,
                    s=50 + count * 10,
                    edgecolors="black",
                )

            median_value = np.median(capped_values)
            plt.scatter(
                0,
                median_value,
                color="purple",
                zorder=6,
                s=150,
                edgecolors="black",
                label=f"Median: {median_value:.2f}",
            )

            unit = FEATURE_UNITS.get(feature.lower(), "")
            ylabel = f"{feature} ({unit})" if unit else f"{feature} Values"
            plt.title(f"{feature} Violin Plot", fontsize=14)
            plt.ylabel(ylabel, fontsize=12)
            plt.xticks([])
            plt.legend(loc="upper right", fontsize=10)

            lo, hi = self._get_plot_bounds(normal_min, normal_max, list(capped_values))
            plt.ylim(lo, hi)

            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"{feature}_enhanced_violin_plot.png")
            plt.savefig(filepath, bbox_inches="tight", dpi=150)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error creating violin plot for {feature}: {e}")
            return f"Error generating plot for {feature}"
        return filepath

    # ------------------------------------------------------------------
    # Box + swarm plot (fixed)
    # ------------------------------------------------------------------

    def _create_box_swarm_plot(self, feature, values, output_dir):
        """
        Creates an enhanced box plot combined with a swarm plot for visualizing the feature value
        compared to the normal range.

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

            outlier_min = normal_min - 1.5 * (normal_max - normal_min)
            outlier_max = normal_max + 1.5 * (normal_max - normal_min)
            capped_values = [max(outlier_min, min(outlier_max, v)) for v in values]

            plt.figure(figsize=(8, 6))
            sns.set(style="whitegrid")

            if len(capped_values) == 1:
                # Single value: scatter point against normal range band
                plt.axhspan(
                    normal_min,
                    normal_max,
                    color="#A9DFBF",
                    alpha=0.35,
                    label="Normal Range",
                )
                plt.scatter(
                    [0],
                    capped_values,
                    color="#34495E",
                    s=120,
                    edgecolor="black",
                    zorder=5,
                    label="Value",
                )
                plt.xlim(-0.5, 0.5)
            else:
                # Box + swarm using data= keyword (seaborn ≥0.12)
                sns.boxplot(data=capped_values, color="#D0E1F9", linewidth=2, width=0.3)
                sns.swarmplot(
                    data=capped_values,
                    color="#34495E",
                    size=6,
                    edgecolor="black",
                    alpha=0.8,
                )
                # Normal range shading using axhspan (axis-relative, always correct)
                plt.axhspan(
                    normal_min,
                    normal_max,
                    color="#A9DFBF",
                    alpha=0.25,
                    label="Normal Range",
                )

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

            unit = FEATURE_UNITS.get(feature.lower(), "")
            ylabel = f"{feature} ({unit})" if unit else f"{feature} Values"
            plt.title(
                f"{feature} Distribution and Data Points", fontsize=14, color="#2C3E50"
            )
            plt.xlabel("")
            plt.ylabel(ylabel, fontsize=12, color="#2C3E50")
            plt.xticks([])

            lo, hi = self._get_plot_bounds(normal_min, normal_max, capped_values)
            plt.ylim(lo, hi)

            plt.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.7)
            plt.grid(visible=False)
            sns.despine(left=True, bottom=True)

            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(
                output_dir, f"{feature}_enhanced_box_swarm_plot.png"
            )
            plt.savefig(filepath, bbox_inches="tight", dpi=150)
            plt.close()
        except Exception as e:
            self.logger.error(
                f"Error creating enhanced box + swarm plot for {feature}: {e}"
            )
            return f"Error generating plot for {feature}"
        return filepath

    # ------------------------------------------------------------------
    # Line with rolling stats (fixed)
    # ------------------------------------------------------------------

    def _create_plot_line_with_rolling_stats(
        self, feature, values, output_dir, time=None
    ):
        """
        Plots a line chart with rolling mean and standard deviation over time.
        Window is set dynamically to avoid all-NaN rolling stats on short series.
        Peak detection is only applied when len(values) >= 20.

        Args:
            feature (str): Feature name.
            values (list or np.array): Scalar feature values per segment.
            output_dir (str): Directory to save the plot.
            time (list or np.array, optional): Segment time labels.
        """
        try:
            if time is None:
                time = np.arange(len(values))

            # Dynamic window: at least 2, at most 10, roughly 1/3 of series length
            window = min(10, max(2, len(values) // 3))

            df = pd.DataFrame({"data": values, "time": time})
            df["rolling_mean"] = df["data"].rolling(window=window).mean()
            df["rolling_std"] = df["data"].rolling(window=window).std()

            plt.figure(figsize=(12, 6))
            plt.plot(df["time"], df["data"], label="Data", color="blue", alpha=0.7)
            plt.plot(
                df["time"],
                df["rolling_mean"],
                label=f"Rolling Mean (w={window})",
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
                label="Rolling ±1 SD",
            )

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
            plt.axhspan(normal_min, normal_max, color="#27ae60", alpha=0.08)

            # Peak detection only when enough data to be meaningful
            if len(values) >= 20:
                series = df["data"].dropna()
                peaks = find_peaks(series.values)
                troughs = find_peaks(-series.values)
                plt.scatter(
                    df["time"].iloc[peaks],
                    df["data"].iloc[peaks],
                    color="purple",
                    s=80,
                    label="Peaks",
                    zorder=5,
                )
                plt.scatter(
                    df["time"].iloc[troughs],
                    df["data"].iloc[troughs],
                    color="cyan",
                    s=80,
                    label="Troughs",
                    zorder=5,
                )

            lo, hi = self._get_plot_bounds(normal_min, normal_max, list(values))
            plt.ylim(lo, hi)

            unit = FEATURE_UNITS.get(feature.lower(), "")
            ylabel = f"{feature} ({unit})" if unit else feature
            plt.title(f"{feature} — Time Series with Rolling Statistics", fontsize=13)
            plt.xlabel("Segment", fontsize=11)
            plt.ylabel(ylabel, fontsize=11)
            plt.legend(loc="upper right", fontsize=9, frameon=True)
            plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(
                output_dir, f"{feature}_line_with_enhanced_stats.png"
            )
            plt.savefig(filepath, bbox_inches="tight", dpi=150)
            plt.close()
        except Exception as e:
            self.logger.error(
                f"Error creating line chart with rolling statistics for {feature}: {e}"
            )
            return f"Error generating plot for {feature}"
        return filepath

    # ------------------------------------------------------------------
    # Summary chart (report-level)
    # ------------------------------------------------------------------

    def create_summary_chart(self, segment_values, output_dir="visualizations"):
        """
        Creates a single horizontal bar chart showing ALL features color-coded by range status.
        Called once per report, not per feature.

        Args:
            segment_values (dict): {feature: list_of_values_per_segment}
            output_dir (str): Directory where the chart will be saved.

        Returns:
            str: Path to the saved summary chart image.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            features = []
            normalized_vals = []
            colors = []
            value_labels = []

            for feature, values in segment_values.items():
                if not isinstance(values, list):
                    values = [values]
                current = values[-1]
                if not np.isfinite(current):
                    continue
                try:
                    normal_min, normal_max = self._fetch_and_validate_normal_range(
                        feature, current
                    )
                except ValueError:
                    continue

                span = normal_max - normal_min if normal_max != normal_min else 1.0
                # Normalize: 0% = normal_min, 100% = normal_max
                norm = (current - normal_min) / span * 100

                if current < normal_min:
                    color = "#e74c3c"  # red = below
                elif current > normal_max:
                    color = "#e67e22"  # orange = above
                else:
                    color = "#27ae60"  # green = in range

                features.append(feature)
                normalized_vals.append(np.clip(norm, -50, 150))
                colors.append(color)
                value_labels.append(f"{current:.1f}")

            if not features:
                return None

            n = len(features)
            fig, ax = plt.subplots(figsize=(max(10, n * 0.9 + 2), 5))

            bars = ax.bar(
                features,
                normalized_vals,
                color=colors,
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )

            # Reference lines at 0% (normal_min) and 100% (normal_max)
            ax.axhline(
                0, color="#27ae60", lw=1.5, ls="--", label="Normal min (0%)", zorder=4
            )
            ax.axhline(
                100,
                color="#27ae60",
                lw=1.5,
                ls="--",
                label="Normal max (100%)",
                zorder=4,
            )
            ax.axhspan(0, 100, color="#27ae60", alpha=0.06, zorder=2)

            for bar, label in zip(bars, value_labels):
                ypos = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    ypos + 2,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

            legend_patches = [
                mpatches.Patch(color="#27ae60", label="In range"),
                mpatches.Patch(color="#e74c3c", label="Below normal"),
                mpatches.Patch(color="#e67e22", label="Above normal"),
            ]
            ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

            ax.set_xlabel("Feature", fontsize=11)
            ax.set_ylabel("Normalized value (% of normal range)", fontsize=11)
            ax.set_title("Feature Status Overview — Most Recent Segment", fontsize=13)
            plt.xticks(rotation=30, ha="right", fontsize=9)
            ax.grid(axis="y", ls="--", alpha=0.4, zorder=1)
            plt.tight_layout()

            filepath = os.path.join(output_dir, "summary_chart.png")
            plt.savefig(filepath, bbox_inches="tight", dpi=150)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error creating summary chart: {e}")
            return f"Error generating summary chart"
        return filepath

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

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
