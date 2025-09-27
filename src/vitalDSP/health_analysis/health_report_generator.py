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

# from functools import lru_cache


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

    Examples
    --------
    >>> from vitalDSP.health_analysis.health_report_generator import HealthReportGenerator
    >>>
    >>> # Example 1: Basic health report generation
    >>> feature_data = {"nn50": 45, "rmssd": 70, "sdnn": 120}
    >>> generator = HealthReportGenerator(feature_data, segment_duration="5_min")
    >>> report_html = generator.generate()
    >>> print(f"Report generated: {len(report_html)} characters")
    >>>
    >>> # Example 2: Health report with custom configuration
    >>> generator_custom = HealthReportGenerator(
    ...     feature_data,
    ...     segment_duration="1_min",
    ...     feature_config_path="path/to/custom_config.yml"
    ... )
    >>> report_custom = generator_custom.generate()
    >>>
    >>> # Example 3: Health report with filtering
    >>> report_filtered = generator.generate(filter_status="above_range")
    >>> print("Filtered report for above-range parameters only")
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

        Examples
        --------
        >>> feature_data = {"nn50": 45, "rmssd": 70, "sdnn": 120}
        >>> generator = HealthReportGenerator(feature_data, segment_duration="5_min")
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
            return feature, visualizer.create_visualizations(
                {feature: values}, output_dir
            )

        with multiprocessing.Pool(processes) as pool:
            results = pool.map(generate_visualizations_batch, feature_data.items())

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
                        self.logger.warning(
                            logging.UserWarning,
                            f"Downsampling feature '{feature_name}' to reduce memory usage",
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
            # Step 3: Generate dynamic analysis components
            dynamic_analysis = self._generate_dynamic_analysis(segment_values)

            # Step 4: Render the report with dynamic components
            report_html = render_report(
                segment_values, visualizations, dynamic_analysis=dynamic_analysis
            )
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

    def _generate_dynamic_analysis(self, segment_values):
        """
        Generates dynamic analysis components based on the feature results.

        Args:
            segment_values (dict): Dictionary containing processed feature data.

        Returns:
            dict: Dictionary containing dynamic analysis components including:
                - executive_summary: Overall assessment summary
                - risk_assessment: Risk level and concerns
                - recommendations: Dynamic recommendations based on findings
                - key_insights: Important findings and patterns
                - overall_health_score: Calculated health score
        """
        try:
            # Analyze overall patterns
            total_features = len(segment_values)
            in_range_count = sum(
                1
                for f in segment_values.values()
                if f.get("range_status") == "in_range"
            )
            above_range_count = sum(
                1
                for f in segment_values.values()
                if f.get("range_status") == "above_range"
            )
            below_range_count = sum(
                1
                for f in segment_values.values()
                if f.get("range_status") == "below_range"
            )

            # Calculate health score (0-100)
            health_score = (
                (in_range_count / total_features * 100) if total_features > 0 else 0
            )

            # Generate executive summary
            executive_summary = self._generate_executive_summary(
                total_features,
                in_range_count,
                above_range_count,
                below_range_count,
                health_score,
            )

            # Generate risk assessment
            risk_assessment = self._generate_risk_assessment(
                segment_values, health_score
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                segment_values, risk_assessment
            )

            # Generate key insights
            key_insights = self._generate_key_insights(segment_values)

            # Generate cross-correlations
            cross_correlations = self._generate_cross_correlations(segment_values)

            return {
                "executive_summary": executive_summary,
                "risk_assessment": risk_assessment,
                "recommendations": recommendations,
                "key_insights": key_insights,
                "cross_correlations": cross_correlations,
                "overall_health_score": health_score,
                "statistics": {
                    "total_features": total_features,
                    "in_range": in_range_count,
                    "above_range": above_range_count,
                    "below_range": below_range_count,
                },
            }

        except Exception as e:
            self.logger.error(f"Error generating dynamic analysis: {e}")
            return {
                "executive_summary": "Unable to generate summary due to processing error.",
                "risk_assessment": {"level": "unknown", "concerns": []},
                "recommendations": [
                    "Consult with a healthcare professional for detailed analysis."
                ],
                "key_insights": [],
                "overall_health_score": 0,
                "statistics": {
                    "total_features": 0,
                    "in_range": 0,
                    "above_range": 0,
                    "below_range": 0,
                },
            }

    def _generate_executive_summary(
        self,
        total_features,
        in_range_count,
        above_range_count,
        below_range_count,
        health_score,
    ):
        """Generate dynamic executive summary based on results."""
        if total_features == 0:
            return "No features available for analysis."

        in_range_percentage = (in_range_count / total_features) * 100

        if health_score >= 80:
            status = "excellent"
            summary = f"Your physiological parameters show excellent overall health with {in_range_percentage:.1f}% of features within normal ranges."
        elif health_score >= 60:
            status = "good"
            summary = f"Your physiological parameters indicate good health with {in_range_percentage:.1f}% of features within normal ranges."
        elif health_score >= 40:
            status = "fair"
            summary = f"Your physiological parameters show fair health with {in_range_percentage:.1f}% of features within normal ranges. Some areas may need attention."
        else:
            status = "poor"
            summary = f"Your physiological parameters indicate areas of concern with only {in_range_percentage:.1f}% of features within normal ranges."

        # Add specific concerns
        concerns = []
        if above_range_count > 0:
            concerns.append(f"{above_range_count} parameter(s) above normal range")
        if below_range_count > 0:
            concerns.append(f"{below_range_count} parameter(s) below normal range")

        if concerns:
            summary += f" Areas of concern include: {', '.join(concerns)}."

        return {"status": status, "summary": summary, "health_score": health_score}

    def _generate_risk_assessment(self, segment_values, health_score):
        """Generate dynamic risk assessment based on results."""
        risk_level = "low"
        concerns = []

        # Assess based on health score
        if health_score < 40:
            risk_level = "high"
            concerns.append("Multiple parameters outside normal ranges")
        elif health_score < 60:
            risk_level = "moderate"
            concerns.append("Several parameters outside normal ranges")

        # Check for specific high-risk patterns
        hrv_features = ["sdnn", "rmssd", "nn50", "pnn50"]
        hrv_out_of_range = sum(
            1
            for f in hrv_features
            if f in segment_values
            and segment_values[f].get("range_status") != "in_range"
        )

        if hrv_out_of_range >= 3:
            risk_level = "high" if risk_level == "low" else risk_level
            concerns.append("Multiple heart rate variability parameters abnormal")

        # Check for cardiovascular risk indicators
        if (
            "sdnn" in segment_values
            and segment_values["sdnn"].get("range_status") == "below_range"
        ):
            concerns.append(
                "Low heart rate variability may indicate cardiovascular risk"
            )

        return {
            "level": risk_level,
            "concerns": concerns,
            "recommendation": self._get_risk_recommendation(risk_level),
        }

    def _get_risk_recommendation(self, risk_level):
        """Get recommendation based on risk level."""
        recommendations = {
            "low": "Continue current lifestyle and regular monitoring",
            "moderate": "Consider lifestyle modifications and increased monitoring frequency",
            "high": "Consult healthcare professional immediately for comprehensive evaluation",
        }
        return recommendations.get(
            risk_level, "Consult healthcare professional for evaluation"
        )

    def _generate_recommendations(self, segment_values, risk_assessment):
        """Generate dynamic recommendations based on findings."""
        recommendations = []

        # General recommendations based on risk level
        if risk_assessment["level"] == "high":
            recommendations.append(
                "Schedule immediate consultation with a healthcare professional"
            )
        elif risk_assessment["level"] == "moderate":
            recommendations.append(
                "Consider lifestyle modifications and regular health monitoring"
            )

        # Specific recommendations based on out-of-range features
        for feature_name, feature_data in segment_values.items():
            if feature_data.get("range_status") == "below_range":
                if feature_name in ["sdnn", "rmssd", "nn50"]:
                    recommendations.append(
                        "Consider stress management techniques and regular exercise to improve heart rate variability"
                    )
                elif feature_name in ["heart_rate"]:
                    recommendations.append(
                        "Monitor heart rate patterns and consider cardiovascular assessment"
                    )
            elif feature_data.get("range_status") == "above_range":
                if feature_name in ["heart_rate"]:
                    recommendations.append(
                        "Monitor for signs of tachycardia and consider cardiovascular evaluation"
                    )

        # Add general health recommendations
        if not recommendations:
            recommendations.append("Continue maintaining healthy lifestyle habits")

        recommendations.append(
            "Regular monitoring and follow-up assessments are recommended"
        )

        return recommendations

    def _generate_key_insights(self, segment_values):
        """Generate key insights based on feature analysis."""
        insights = []

        # Analyze patterns
        out_of_range_features = [
            name
            for name, data in segment_values.items()
            if data.get("range_status") != "in_range"
        ]

        if len(out_of_range_features) == 0:
            insights.append("All analyzed parameters are within normal ranges")
        elif len(out_of_range_features) <= 2:
            insights.append(
                f"Most parameters are normal, with {len(out_of_range_features)} requiring attention"
            )
        else:
            insights.append(
                f"Multiple parameters ({len(out_of_range_features)}) require attention"
            )

        # Check for specific patterns
        hrv_features = ["sdnn", "rmssd", "nn50", "pnn50"]
        hrv_status = [
            segment_values.get(f, {}).get("range_status")
            for f in hrv_features
            if f in segment_values
        ]

        if all(status == "in_range" for status in hrv_status):
            insights.append(
                "Heart rate variability parameters indicate healthy autonomic function"
            )
        elif any(status == "below_range" for status in hrv_status):
            insights.append(
                "Some heart rate variability parameters suggest reduced autonomic function"
            )

        return insights

    def _generate_cross_correlations(self, segment_values):
        """Generate cross-feature correlation analysis."""
        try:
            # Use the interpretation engine to analyze cross-correlations
            cross_correlations = self.interpreter._analyze_cross_feature_correlations(
                segment_values, self.segment_duration
            )
            return cross_correlations
        except Exception as e:
            self.logger.error(f"Error generating cross-correlations: {e}")
            return []
