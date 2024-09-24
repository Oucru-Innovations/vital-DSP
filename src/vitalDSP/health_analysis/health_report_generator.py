from vitalDSP.health_analysis.interpretation_engine import InterpretationEngine
from vitalDSP.health_analysis.health_report_visualization import HealthReportVisualizer
from vitalDSP.health_analysis.html_template import render_report
# from vitalDSP.health_analysis.file_io import FileIO


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

    def generate(self, filter_status="all"):
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
        report_content = {}

        # Step 1: Interpret each feature with the loaded configuration
        for feature_name, value in self.feature_data.items():
            # Interpret each feature and get its range status
            interpretation = self.interpreter.interpret_feature(
                feature_name, value, self.segment_duration
            )
            range_status = self.interpreter.get_range_status(
                feature_name, value, self.segment_duration
            )

            # Skip features not matching the filter status
            if filter_status != "all" and range_status != filter_status:
                continue

            # Add interpretation details and range status for each feature
            report_content[feature_name] = {
                "description": interpretation["description"],
                "value": value,
                "interpretation": interpretation["interpretation"],
                "normal_range": interpretation["normal_range"],
                "contradiction": interpretation.get("contradiction", None),
                "correlation": interpretation.get("correlation", None),
                "range_status": range_status,
            }

        # Step 2: Generate visualizations for all features
        visualizations = self.visualizer.create_visualizations(self.feature_data)

        # Step 3: Provide all visualizations for each feature
        selected_visualizations = visualizations

        # Step 4: Render the report
        report_html = render_report(report_content, selected_visualizations)

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


# if __name__ == "__main__":
    # # Example feature data for a 1-minute segment
    # feature_data = {
    #     "sdnn": 35.0,
    #     "rmssd": 45.0,
    #     "nn50": 30,
    #     "pnn50": 28.5,
    #     "mean_nn": 750.0,
    #     "median_nn": 745.0,
    #     "iqr_nn": 60.0,
    #     "std_nn": 25.0,
    #     "pnn20": 40.0,
    #     "cvnn": 0.12,
    #     "hrv_triangular_index": 5.5,
    #     "tinn": 40.0,
    #     "sdsd": 35.0,
    #     "lf_power": 800.0,
    #     "hf_power": 600.0,
    #     "lf_hf_ratio": 1.33,
    #     "ulf_power": 200.0,
    #     "vlf_power": 300.0,
    #     "total_power": 1600.0,
    #     "lfnu_power": 45.0,
    #     "hfnu_power": 35.0,
    #     "fractal_dimension": 1.25,
    #     # "lyapunov_exponent": 0.85,
    #     "dfa": 1.2,
    #     "poincare_sd1": 28.5,
    #     "poincare_sd2": 45.0,
    #     "sample_entropy": 0.85,
    #     "approximate_entropy": 0.72,
    #     "recurrence_rate": 0.65,
    #     "determinism": 0.78,
    #     "laminarity": 0.85,
    #     "systolic_duration": 0.35,
    #     "diastolic_duration": 0.5,
    #     "systolic_area": 200.0,
    #     "diastolic_area": 180.0,
    #     "systolic_slope": 1.5,
    #     "diastolic_slope": 1.2,
    #     "signal_skewness": 0.1,
    #     "peak_trend_slope": 0.05,
    #     "systolic_amplitude_variability": 5.0,
    #     "diastolic_amplitude_variability": 4.8,
    # }

    # # Initialize the health report generator
    # report_generator = HealthReportGenerator(
    #     feature_data=feature_data, segment_duration="1_min"
    # )

    # # Generate the report (HTML)
    # report_html = report_generator.generate()

    # # Save or display the report HTML
    # with open("health_analysis_report.html", "w") as report_file:
    #     report_file.write(report_html)

    # print(
    #     "Health report has been generated and saved as 'health_analysis_report.html'."
    # )
