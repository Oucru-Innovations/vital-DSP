from vitalDSP.health_analysis.interpretation_engine import InterpretationEngine
from vitalDSP.health_analysis.health_report_visualization import HealthReportVisualizer
from vitalDSP.health_analysis.html_template import (
    render_report,
    process_interpretations,
)
import logging

# from vitalDSP.health_analysis.file_io import FileIO
import numpy as np
import multiprocessing
from functools import lru_cache


class HealthReportGenerator:
    """
    A class to generate a health report based on feature data, including interpretations, visualizations, and contradictions/correlations.

    This class handles the process of interpreting feature data (e.g., heart rate variability, ECG/PPG data),
    generating visualizations for features, and rendering the final HTML report.

    Attributes:
        feature_data (dict): Dictionary containing feature names as keys and their corresponding values.
        segment_duration (str): Duration of the segment, either "1_min" or "5_min". Default is "1 min".
        interpreter (InterpretationEngine): Instance of InterpretationEngine to interpret feature data.
        visualizer (HealthReportVisualizer): Instance of HealthReportVisualizer to create visualizations.
    """

    def __init__(
        self, feature_data, segment_duration="1_min", feature_config_path=None
    ):
        """
        Initializes the HealthReportGenerator with the provided feature data and segment duration.

        Args:
            feature_data (dict): Dictionary containing feature names as keys and their corresponding values as values.
                                Example: {"nn50": 35, "rmssd": 55, "sdnn": 70}
            segment_duration (str): The duration of the analyzed segment, either '1 min' or '5 min'. Default is '1 min'.
            feature_config_path (str, optional): Path to a custom feature YAML configuration file. If not provided, the default config will be used.

        Example Usage:
            >>> feature_data = {"nn50": 45, "rmssd": 70, "sdnn": 120}
            >>> generator = HealthReportGenerator(feature_data, segment_duration="5 min", feature_config_path="path/to/config.yml")
            >>> report_html = generator.generate()
        """
        self.feature_data = feature_data
        self.segment_duration = segment_duration
        self.interpreter = InterpretationEngine(feature_config_path)
        self.visualizer = HealthReportVisualizer(
            self.interpreter.config, segment_duration
        )
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def downsample(values, factor=10):
        """
        Downsample large datasets by selecting every nth value.
        Args:
            values (list or np.array): The dataset to downsample.
            factor (int): The downsampling factor.
        Returns:
            np.array: The downsampled dataset.
        """
        return values[::factor]

    @staticmethod
    def batch_visualization(visualizer, feature_data, output_dir, processes=4):
        """
        Generate visualizations in parallel batches with limited processes.
        Args:
            visualizer (HealthReportVisualizer): The visualizer instance.
            feature_data (dict): Dictionary of features and their values.
            output_dir (str): Directory to save the visualizations.
            processes (int): Number of processes for parallel visualization.
        Returns:
            dict: Paths to the generated visualizations.
        """
        def generate_visualizations_batch(args):
            feature, values = args
            return feature, visualizer.create_visualizations({feature: values}, output_dir)

        with multiprocessing.Pool(processes) as pool:
            results = pool.map(
                generate_visualizations_batch, feature_data.items()
            )

        return {feature: paths for feature, paths in results}

    def generate(self, filter_status="all", output_dir=None):
        """
        Generates the complete health report by interpreting the features, generating visualizations, and rendering an HTML report.

        The report will include the description of each feature, its interpretation (in-range, above range, or below range),
        any detected contradictions, correlations, and visualizations for each feature.

        Returns:
            str: HTML content of the generated health report.

        Example Usage:
            >>> feature_data = {"nn50": 45, "rmssd": 70, "sdnn": 120}
            >>> generator = HealthReportGenerator(feature_data, segment_duration="5 min")
            >>> report_html = generator.generate()
            >>> with open('health_report.html', 'w') as file:
            >>>     file.write(report_html)
        """
        segment_values = {}

        try:
            # Step 1: Interpret each feature with the loaded configuration
            for feature_name, values in self.feature_data.items():
                try:
                    # Convert the values to a NumPy array
                    values = np.array(values)

                    # Drop NaN or Inf values
                    values = values[np.isfinite(values)]

                    # If all values are NaN/Inf and nothing remains, log an error and continue
                    if len(values) == 0:
                        raise ValueError(
                            f"All values for feature '{feature_name}' are NaN or Inf"
                        )
                    # Downsample data to reduce memory overhead for large datasets
                    if len(values) > 1000:
                        self.logger.warning(logging.UserWarning,
                            f"Downsampling feature '{feature_name}' to reduce memory usage"
                        )
                        values = self.downsample(values)
                    
                    # Calculate the mean of the feature's values
                    mean_value = np.mean(values)

                    # Interpret based on the mean value
                    interpretation = self.interpreter.interpret_feature(
                        feature_name, mean_value, self.segment_duration
                    )
                    range_status = self.interpreter.get_range_status(
                        feature_name, mean_value, self.segment_duration
                    )

                    # Skip features not matching the filter status
                    if filter_status != "all" and range_status != filter_status:
                        continue

                    # Store aggregated information
                    median_value = np.median(values)
                    stddev_value = np.std(values)
                    segment_values[feature_name] = {
                        "description": interpretation["description"],
                        "value": values.tolist(),  # Store the list of valid values
                        "median": median_value,
                        "stddev": stddev_value,
                        "interpretation": interpretation["interpretation"],
                        "normal_range": interpretation["normal_range"],
                        "contradiction": interpretation.get("contradiction", None),
                        "correlation": interpretation.get("correlation", None),
                        "range_status": range_status,
                    }
                except Exception as e:
                    # Log the error but continue processing other features
                    self.logger.error(f"Error processing {feature_name}: {e}")
                    continue

            # Step 1.5: Process contradictions and correlations
            segment_values = process_interpretations(segment_values)

        except Exception as e:
            # Log the error if step 1 fails
            self.logger.error(f"Error in processing feature interpretations: {e}")

        # Step 2: Generate visualizations for all features
        # visualizations = self.visualizer.create_visualizations(
        #     self.feature_data, output_dir=output_dir
        # )
        try:
            visualizations = self.batch_visualization(
                self.visualizer, self.feature_data, output_dir, processes=4
            )
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            visualizations = {}

        try:
            # Step 3: Render the report
            report_html = render_report(segment_values, visualizations)
        except Exception as e:
            # Log the error if report rendering fails
            self.logger.error(f"Error rendering report: {e}")
            report_html = "<h1>Error generating report</h1>"

        return report_html

    def _generate_feature_report(self, feature_name, value):
        """
        Generates the interpretation report for an individual feature including its description, normal range,
        interpretation (based on the range), contradictions, and correlations.

        Args:
            feature_name (str): The name of the feature to generate the report for (e.g., "NN50", "RMSSD").
            value (float): The value of the feature (e.g., 35.0 for NN50).

        Returns:
            dict: A dictionary containing:
            - "description": Description of the feature.
            - "value": The actual feature value.
            - "interpretation": Interpretation of the value based on the normal range.
            - "normal_range": The normal range of the feature.
            - "contradiction": Contradictions with related features.
            - "correlation": Correlations with related features.

        Example Usage:
            >>> feature_name = "NN50"
            >>> value = 45
            >>> generator = HealthReportGenerator(feature_data)
            >>> feature_report = generator._generate_feature_report(feature_name, value)
            >>> print(feature_report)
            {
                "description": "Number of significant changes in heart rate. High NN50 suggests healthy parasympathetic activity.",
                "value": 45,
                "interpretation": "Normal parasympathetic activity. No immediate concern.",
                "normal_range": [10, 50],
                "contradiction": "Low NN50 contradicts high RMSSD, as both should indicate parasympathetic activity.",
                "correlation": "Positively correlated with RMSSD, as both represent short-term heart rate variability."
            }
        """
        # Interpret the feature using the InterpretationEngine
        feature_info = self.interpreter.interpret_feature(
            feature_name, value, self.segment_duration
        )

        # Add the description, interpretation, contradiction, and correlation to the report
        feature_report = {
            "description": feature_info.get("description", "No description available."),
            "value": value,
            "interpretation": feature_info.get(
                "interpretation", "No interpretation available."
            ),
            "normal_range": feature_info.get(
                "normal_range", "No normal range available."
            ),
            "contradiction": feature_info.get(
                "contradiction", "No contradiction found."
            ),
            "correlation": feature_info.get("correlation", "No correlation found."),
        }

        return feature_report
