from vitalDSP.health_analysis.interpretation_engine import InterpretationEngine
from vitalDSP.health_analysis.health_report_visualization import HealthReportVisualizer
from vitalDSP.health_analysis.html_template import render_report

# from vitalDSP.health_analysis.file_io import FileIO
import numpy as np


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
        self, feature_data, segment_duration="1 min", feature_config_path=None
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
        self.visualizer = HealthReportVisualizer(self.interpreter.config)

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
        # report_content = {}
        segment_values = {}  # To store segmented interpretations

        # Step 1: Interpret each feature with the loaded configuration
        for feature_name, values in self.feature_data.items():
            # Calculate the mean of the feature's values
            mean_value = sum(values) / len(values)

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

            median_value = np.median(values)
            stddev_value = np.std(values)
            # Store aggregated information
            segment_values[feature_name] = {
                "description": interpretation["description"],
                "value": values,  # Store the list of values for the feature
                "median": median_value,  # Median value
                "stddev": stddev_value,  # Standard deviation
                "interpretation": interpretation["interpretation"],
                "normal_range": interpretation["normal_range"],
                "contradiction": interpretation.get("contradiction", None),
                "correlation": interpretation.get("correlation", None),
                "range_status": range_status,  # Based on the mean value
            }

        # Step 2: Generate visualizations for all features (pass all segments of data)
        visualizations = self.visualizer.create_visualizations(
            self.feature_data, output_dir=output_dir
        )

        # Step 3: Provide all visualizations for each feature
        selected_visualizations = visualizations

        # Step 4: Render the report
        report_html = render_report(segment_values, selected_visualizations)

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


if __name__ == "__main__":
    # Example feature data for a 1-minute segment
    # Set a random seed for reproducibility
    np.random.seed(42)
    duration = 60
    fs = 100
    # Generate mock feature data
    feature_data = {
        "sdnn": np.random.normal(
            50, 10, duration
        ).tolist(),  # Normally distributed around 50 with std dev of 10
        "rmssd": np.random.normal(
            30, 8, duration
        ).tolist(),  # Normally distributed around 30 with std dev of 8
        "total_power": np.random.normal(
            1600, 300, duration
        ).tolist(),  # Normally distributed around 1600 with std dev of 300
        "lfnu_power": np.random.normal(
            45, 5, duration
        ).tolist(),  # Normally distributed around 45 with std dev of 5
        "hfnu_power": np.random.normal(
            35, 4, duration
        ).tolist(),  # Normally distributed around 35 with std dev of 4
        "fractal_dimension": np.random.normal(
            1.25, 0.05, duration
        ).tolist(),  # Normally around 1.25 with std dev of 0.05
        "dfa": np.random.normal(
            1.2, 0.08, duration
        ).tolist(),  # Normally distributed around 1.2 with std dev of 0.08
        "poincare_sd1": np.random.normal(
            28.5, 3, duration
        ).tolist(),  # Normally around 28.5 with std dev of 3
        "poincare_sd2": np.random.normal(
            45, 4, duration
        ).tolist(),  # Normally around 45 with std dev of 4
        "sample_entropy": np.random.normal(
            0.85, 0.1, duration
        ).tolist(),  # Normally around 0.85 with std dev of 0.1
        "approximate_entropy": np.random.normal(
            0.72, 0.1, duration
        ).tolist(),  # Normally around 0.72 with std dev of 0.1
        "recurrence_rate": np.random.normal(
            0.65, 0.1, duration
        ).tolist(),  # Normally around 0.65 with std dev of 0.1
        "determinism": np.random.normal(
            0.78, 0.1, duration
        ).tolist(),  # Normally around 0.78 with std dev of 0.1
        "laminarity": np.random.normal(
            0.85, 0.05, duration
        ).tolist(),  # Normally around 0.85 with std dev of 0.05
        "systolic_duration": np.random.normal(
            0.35, 0.05, duration
        ).tolist(),  # Normally around 0.35 with std dev of 0.05
        "diastolic_duration": np.random.normal(
            0.5, 0.05, duration
        ).tolist(),  # Normally around 0.5 with std dev of 0.05
        "systolic_area": np.random.normal(
            200, 20, duration
        ).tolist(),  # Normally around 200 with std dev of 20
        "diastolic_area": np.random.normal(
            180, 20, duration
        ).tolist(),  # Normally around 180 with std dev of 20
        "systolic_slope": np.random.normal(
            1.5, 0.1, duration
        ).tolist(),  # Normally around 1.5 with std dev of 0.1
        "diastolic_slope": np.random.normal(
            1.2, 0.1, duration
        ).tolist(),  # Normally around 1.2 with std dev of 0.1
        "signal_skewness": np.random.normal(
            0.1, 0.02, duration
        ).tolist(),  # Normally around 0.1 with std dev of 0.02
        "peak_trend_slope": np.random.normal(
            0.05, 0.01, duration
        ).tolist(),  # Normally around 0.05 with std dev of 0.01
        "systolic_amplitude_variability": np.random.normal(
            5.0, 0.5, duration
        ).tolist(),  # Normally around 5.0 with std dev of 0.5
        "diastolic_amplitude_variability": np.random.normal(
            4.8, 0.5, duration
        ).tolist(),  # Normally around 4.8 with std dev of 0.5
        # "respiratory_rate": np.random.normal(16, 3, duration).tolist(),  # Normally around 16 breaths/min with std dev of 3
        # "pulse_pressure": np.random.normal(40, 5, duration).tolist(),  # Normally around 40 mmHg with std dev of 5
        # "stroke_volume": np.random.normal(70, 10, duration).tolist(),  # Normally around 70 mL with std dev of 10
        # "cardiac_output": np.random.normal(5.5, 0.7, duration).tolist(),  # Normally around 5.5 L/min with std dev of 0.7
        # "qt_interval": np.random.normal(400, 20, duration).tolist(),  # Normally around 400 ms with std dev of 20
        # "qrs_duration": np.random.normal(90, 10, duration).tolist(),  # Normally around 90 ms with std dev of 10
        # "p_wave_duration": np.random.normal(100, 10, duration).tolist(),  # Normally around 100 ms with std dev of 10
        # "pr_interval": np.random.normal(160, 20, duration).tolist(),  # Normally around 160 ms with std dev of 20
    }

    # Example output to check the data format
    # print(feature_data)

    # Initialize the health report generator
    report_generator = HealthReportGenerator(
        feature_data=feature_data, segment_duration="1_min"
    )

    # Generate the report (HTML)
    report_html = report_generator.generate()

    # Write the HTML content to a file
    with open("report.html", "w", encoding="utf-8") as file:
        file.write(report_html)
