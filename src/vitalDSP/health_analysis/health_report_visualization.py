import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


class HealthReportVisualizer:
    """
    A class responsible for creating visualizations of health feature data, including line plots and heatmaps.

    The class takes feature data and creates visualizations, such as normal distributions for ranges and heatmaps, and stores them as images.
    """

    def __init__(self, config):
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
            # return None
            raise ValueError(f"Normal range for feature '{feature}' not found.")

        normal_min, normal_max = normal_range
        # feature_names = ["Min Range", "Max Range", "Current Value"]

        # Handle NaN and Inf cases
        if np.isnan(normal_min) or np.isnan(normal_max) or np.isnan(value):
            raise ValueError(f"NaN value encountered in feature '{feature}'.")

        if np.isinf(normal_min):
            normal_min = -10 * abs(value)
        if np.isinf(normal_max):
            normal_max = 10 * abs(value)

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
        os.makedirs(output_dir, exist_ok=True)

        visualization_paths = {}
        # Define plot types
        # plot_types = [
        #     self._create_normal_distribution_plot,
        #     self._create_bell_shape_plot,
        #     self._create_radar_plot,
        #     self._create_violin_plot,
        #     # self._create_difference_plot,
        #     # self._create_waterfall_plot,
        #     # self._create_donut_plot
        # ]

        for idx, (feature, value) in enumerate(feature_data.items()):
            if isinstance(
                value, list
            ):  # If the value is a list, treat it as time-series data
                visualization_paths[feature] = self._create_line_plot(
                    feature, value, output_dir
                )
            else:
                # plot_type = plot_types[
                #     idx % len(plot_types)
                # ]  # Rotate between plot types
                # visualization_paths[feature] = plot_type(feature, value, output_dir)
                # visualization_paths[feature] = {
                # 'heatmap': self._create_heatmap_plot(feature, value, output_dir),
                # 'bell_plot': self._create_bell_shape_plot(feature, value, output_dir),
                # 'radar_plot': self._create_radar_plot(feature, value, output_dir),
                # 'violin_plot': self._create_violin_plot(feature, value, output_dir)
                # }
                visualization_paths[feature] = {
                    "heatmap": os.path.normpath(
                        self._create_heatmap_plot(feature, value, output_dir)
                    ),
                    "bell_plot": os.path.normpath(
                        self._create_bell_shape_plot(feature, value, output_dir)
                    ),
                    "radar_plot": os.path.normpath(
                        self._create_radar_plot(feature, value, output_dir)
                    ),
                    "violin_plot": os.path.normpath(
                        self._create_violin_plot(feature, value, output_dir)
                    ),
                }

        return visualization_paths

    def _create_donut_plot(self, feature, value, output_dir):
        """
        Creates a donut plot to show how far the current value is within or outside the normal range.

        Args:
            feature (str): The name of the feature.
            value (float): The value of the feature for the current segment.
            output_dir (str): Directory where the plot will be saved.

        Returns:
            str: Path to the saved donut plot image.
        """
        # normal_range = self._get_normal_range_for_feature(feature)
        # if not normal_range:
        #     raise ValueError(f"Normal range for feature '{feature}' not found.")

        # normal_min, normal_max = normal_range
        normal_min, normal_max = self._fetch_and_validate_normal_range(feature, value)
        normal_span = normal_max - normal_min

        # Check how much the current value falls within the normal range
        if value < normal_min:
            in_range = 0
            out_of_range = normal_min - value
        elif value > normal_max:
            in_range = 1
            out_of_range = value - normal_max
        else:
            in_range = (value - normal_min) / normal_span
            out_of_range = 0

        fig, ax = plt.subplots(figsize=(6, 6))

        # Donut plot parts
        wedges, texts, autotexts = ax.pie(
            [in_range, out_of_range],
            labels=["In Range", "Out of Range"],
            autopct="%1.1f%%",
            colors=["lightgreen", "lightcoral"],
            startangle=90,
            wedgeprops=dict(width=0.3),
            pctdistance=0.85,
        )

        plt.setp(autotexts, size=12, weight="bold", color="white")
        ax.set_title(f"{feature} Donut Plot", fontsize=15)

        # Add a center circle to make it a donut
        center_circle = plt.Circle((0, 0), 0.70, color="white")
        fig.gca().add_artist(center_circle)

        # Save the plot
        filepath = os.path.join(output_dir, f"{feature}_donut_plot.png")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

        return filepath

    def _create_line_plot(self, feature, values, output_dir):
        """
        Creates a line plot for time-series data.

        Args:
            feature (str): Name of the feature being plotted.
            values (list): List of feature values over time.
            output_dir (str): Directory where the plot will be saved.

        Returns:
            str: Path to the saved line plot image.

        Example Usage:
            >>> path = visualization._create_line_plot("SDNN", [30, 45, 50], "output/")
            >>> print(path)
        """
        plt.figure(figsize=(6, 4))
        plt.plot(values, marker="o", color="b")
        plt.title(f"{feature} Over Time")
        plt.xlabel("Time (Segments)")
        plt.ylabel(f"{feature} Value")
        plt.grid(True)

        filepath = os.path.join(output_dir, f"{feature}_line_plot.png")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

        return filepath

    def _create_heatmap_plot(self, feature, value, output_dir):
        """
        Creates a normal distribution plot with the feature value and highlights for normal ranges.

        Args:
            feature (str): Name of the feature being visualized.
            value (float): The value of the feature for the current segment.
            output_dir (str): Directory where the plot will be saved.

        Returns:
            str: Path to the saved normal distribution plot image.

        Example Usage:
            >>> path = visualization._create_normal_distribution_plot("RMSSD", 45, "output/")
            >>> print(path)
        """
        # Load normal range for the feature
        normal_min, normal_max = self._fetch_and_validate_normal_range(feature, value)
        # mean = (normal_max + normal_min) / 2
        # std_dev = (
        #     normal_max - normal_min
        # ) / 4  # Estimate std. deviation assuming 95% of data within range

        # Determine if the value is an outlier (>1.5 times normal range)
        outlier_threshold = normal_max + 1.5 * (normal_max - normal_min)
        is_outlier = value > outlier_threshold

        # x = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, 500)
        # y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

        x = np.linspace(
            normal_min - 1.5 * (normal_max - normal_min),
            normal_max + 1.5 * (normal_max - normal_min),
            100,
        )
        y = np.exp(
            -((x - np.mean([self._get_normal_range_for_feature(feature)])) ** 2)
            / (2 * ((normal_max - normal_min) / 2) ** 2)
        )  # Gaussian-like curve
        heatmap_data = np.outer(y, y)  # To create the heatmap data

        # Quartile positions for the normal range
        # quartile_2 = np.percentile(x, 25)  # Represents the start of the 2nd quartile
        # quartile_3 = np.percentile(x, 75)  # Represents the end of the 3rd quartile

        plt.figure(figsize=(8, 4))
        sns.heatmap(
            heatmap_data,
            cmap="coolwarm",
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )

        # Interpolating values correctly
        current_value_interp = np.interp(value, x, np.linspace(0, 100, len(x)))
        normal_min_interp = np.interp(normal_min, x, np.linspace(0, 100, len(x)))
        normal_max_interp = np.interp(normal_max, x, np.linspace(0, 100, len(x)))

        # Overlay the normal range (2nd and 3rd quartile) and current value as markers
        plt.axvline(
            current_value_interp,
            color="black",
            linestyle="--",
            label=f"Current Value: {value}",
            linewidth=2 if is_outlier else 1,
        )
        plt.axvline(
            normal_min_interp,
            color="blue",
            linestyle="-",
            label=f"Normal Min: {normal_min}",
        )
        plt.axvline(
            normal_max_interp,
            color="red",
            linestyle="-",
            label=f"Normal Max: {normal_max}",
        )

        # Highlight the 2nd and 3rd quartiles with shaded areas
        # plt.fill_betweenx(np.arange(100), np.interp(quartile_2, x, np.linspace(0, 100, len(x))), np.interp(quartile_3, x, np.linspace(0, 100, len(x))), color='lightgreen', alpha=0.3, label="2nd & 3rd Quartile")

        # If outlier, add an annotation and warning area
        if is_outlier:
            plt.axvline(
                np.interp(outlier_threshold, x, np.linspace(0, 100, len(x))),
                color="orange",
                linestyle="--",
                label=f"Outlier Threshold: {outlier_threshold}",
                linewidth=2,
            )
            plt.text(
                current_value_interp,
                20,
                "Outlier!",
                color="red",
                fontsize=12,
                bbox=dict(facecolor="yellow", alpha=0.5),
            )

        # Add title and legend
        plt.title(
            f"Feature: {feature}\nCurrent Value: {value}, Normal Range: [{normal_min}, {normal_max}], Outlier Threshold: {outlier_threshold}"
        )
        plt.legend()

        # Save heatmap
        filepath = os.path.join(output_dir, f"{feature}_heatmap.png")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

        return filepath

    def _create_waterfall_plot(self, feature, value, output_dir):
        """
        Creates a waterfall plot for showing the incremental deviation of the feature value from the normal range.

        Args:
            feature (str): The name of the feature.
            value (float): The value of the feature for the current segment.
            output_dir (str): Directory where the plot will be saved.

        Returns:
            str: Path to the saved waterfall plot image.
        """
        normal_min, normal_max = self._fetch_and_validate_normal_range(feature, value)
        diff_to_min = value - normal_min
        diff_to_max = normal_max - value

        plt.figure(figsize=(6, 4))

        # Waterfall plot: Incremental steps showing deviation
        plt.bar(
            [0], [diff_to_min], color="green", label=f"Above Min by {diff_to_min:.2f}"
        )
        plt.bar(
            [0],
            [diff_to_max],
            bottom=[diff_to_min],
            color="lightcoral",
            label=f"Below Max by {diff_to_max:.2f}",
        )

        plt.axhline(0, color="black", linestyle="--")

        plt.title(f"{feature} Waterfall Plot", fontsize=15)
        plt.ylabel(f"{feature} Deviation")
        plt.xticks([])

        # Add legend
        plt.legend(loc="upper left")

        # Save the plot
        filepath = os.path.join(output_dir, f"{feature}_waterfall_plot.png")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

        return filepath

    def _create_difference_plot(self, feature, value, output_dir):
        """
        Creates a difference plot to show how far the feature value is from the normal range.

        Args:
            feature (str): The name of the feature.
            value (float): The value of the feature for the current segment.
            output_dir (str): Directory where the plot will be saved.

        Returns:
            str: Path to the saved difference plot image.
        """
        normal_min, normal_max = self._fetch_and_validate_normal_range(feature, value)
        mean_value = (normal_max + normal_min) / 2

        plt.figure(figsize=(6, 4))

        # Difference between the current value and normal range
        difference = value - mean_value
        plt.barh([feature], [difference], color="purple", alpha=0.6)

        # Draw line at 0 (no difference from normal range)
        plt.axvline(0, color="black", linestyle="--")

        # Annotate current value
        plt.text(
            difference,
            feature,
            f"{value}",
            va="center",
            ha="left" if difference > 0 else "right",
            fontsize=12,
        )

        plt.title(f"{feature} Difference Plot", fontsize=15)
        plt.xlabel("Deviation from Normal Range")
        plt.grid(True)

        # Save the plot
        filepath = os.path.join(output_dir, f"{feature}_difference_plot.png")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

        return filepath

    def _create_violin_plot(self, feature, value, output_dir):
        """
        Creates an enhanced violin plot for visualizing the feature value compared to the normal range.

        Args:
            feature (str): The name of the feature.
            value (float): The value of the feature for the current segment.
            output_dir (str): Directory where the plot will be saved.

        Returns:
            str: Path to the saved violin plot image.
        """
        normal_min, normal_max = self._fetch_and_validate_normal_range(feature, value)

        # Create data for the violin plot
        data = np.random.normal(
            loc=(normal_max + normal_min) / 2,
            scale=(normal_max - normal_min) / 6,
            size=1000,
        )

        plt.figure(figsize=(8, 6))

        # Create violin plot for normal distribution
        sns.violinplot(data=data, color="lightgreen", inner="quartile")

        # Overlay shaded normal range
        plt.fill_between(
            [0, 1],
            normal_min,
            normal_max,
            color="lightblue",
            alpha=0.3,
            label="Normal Range",
        )

        # Add a horizontal line for the mean of the normal range
        mean_value = (normal_max + normal_min) / 2
        plt.axhline(
            mean_value,
            color="green",
            linestyle="--",
            label=f"Normal Range Mean: {mean_value:.2f}",
        )

        # Plot the current value as a scatter point (only one blue dot)
        plt.scatter(
            0,
            value,
            color="blue",
            zorder=5,
            label=f"Current Value: {value}",
            s=150,
            edgecolors="black",
        )

        plt.title(f"{feature} Violin Plot", fontsize=15)
        plt.xlabel(f"{feature} Value")

        # Add legend
        plt.legend(loc="upper right")

        # Save the plot
        filepath = os.path.join(output_dir, f"{feature}_violin_plot.png")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

        return filepath

    def _create_bell_shape_plot(self, feature, value, output_dir):
        """
        Creates a bell shape plot showing the feature's value with error bars to indicate deviation from the normal range.

        Args:
            feature (str): The name of the feature.
            value (float): The value of the feature for the current segment.
            output_dir (str): Directory where the plot will be saved.

        Returns:
            str: Path to the saved bar plot image.
        """
        normal_min, normal_max = self._fetch_and_validate_normal_range(feature, value)
        quartile_2 = (normal_min + normal_max) * 0.25
        quartile_3 = (normal_min + normal_max) * 0.75

        # Generate x values for the bell curve and a corresponding Gaussian curve
        x = np.linspace(
            normal_min - 1.5 * (normal_max - normal_min),
            normal_max + 1.5 * (normal_max - normal_min),
            100,
        )
        mean = (normal_min + normal_max) / 2
        stddev = (
            normal_max - normal_min
        ) / 4  # Approximate the standard deviation based on the range
        y = np.exp(-((x - mean) ** 2) / (2 * stddev**2))

        # Plot the bell-shaped distribution curve
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, label="Normal Distribution", color="blue")
        plt.fill_between(x, y, color="lightblue", alpha=0.3)

        # Mark the quartiles and current value
        plt.axvline(
            value, color="black", linestyle="--", label=f"Current Value: {value}"
        )
        plt.axvline(quartile_2, color="green", linestyle=":", label="2nd Quartile")
        plt.axvline(quartile_3, color="orange", linestyle=":", label="3rd Quartile")

        # Title and legend
        plt.title(
            f"Feature: {feature}\nCurrent Value: {value}, Normal Range: [{normal_min}, {normal_max}]"
        )
        plt.legend()

        # Save the plot
        filepath = os.path.join(output_dir, f"{feature}_bell_plot.png")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

        return filepath

    def _create_radar_plot(self, feature, value, output_dir):
        """
        Creates an enhanced radar plot that shows how the feature value compares to the normal range.
        Each axis represents a different feature, and the plot shows how the current feature deviates
        from its normal range.

        Args:
            feature (str): The name of the feature.
            value (float): The value of the feature for the current segment.
            output_dir (str): Directory where the plot will be saved.

        Returns:
            str: Path to the saved radar plot image.
        """
        normal_min, normal_max = self._fetch_and_validate_normal_range(feature, value)
        feature_names = ["Min Range", "Max Range", "Current Value"]

        # Handle NaN and Inf cases
        if np.isnan(normal_min) or np.isnan(normal_max) or np.isnan(value):
            raise ValueError(f"NaN value encountered in feature '{feature}'.")

        if np.isinf(normal_min):
            normal_min = -10 * abs(value)
        if np.isinf(normal_max):
            normal_max = 10 * abs(value)

        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop for radar chart

        normal_values = [normal_min, normal_max, value]
        normal_values += normal_values[:1]  # Complete the loop

        current_value_triangle = [value, value, value]
        current_value_triangle += current_value_triangle[:1]  # Complete the loop

        # Start plot
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        # Draw one axe per feature and add labels
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], feature_names)

        # Draw ylabels
        ax.set_rlabel_position(0)
        y_min = min(normal_min, 0)
        y_max = max(normal_max, value)

        plt.ylim(y_min, y_max)

        # Plot normal range data
        ax.plot(
            angles,
            normal_values,
            linewidth=2,
            linestyle="solid",
            color="green",
            label="Normal Range",
        )
        ax.fill(angles, normal_values, color="green", alpha=0.3)

        # Plot current value as a second triangle
        ax.plot(
            angles,
            current_value_triangle,
            linewidth=2,
            linestyle="solid",
            color="blue",
            label=f"Current Value: {value:.2f}",
        )
        ax.fill(angles, current_value_triangle, color="blue", alpha=0.3)

        # Add scatter point for clarity
        ax.scatter(angles[2], value, color="red", zorder=5, s=150, edgecolors="black")

        # Add a legend
        plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))

        # Save the plot
        filepath = os.path.join(output_dir, f"{feature}_radar_plot.png")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

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
        normal_range = feature_info.get("normal_range", {}).get("1_min", None)
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
