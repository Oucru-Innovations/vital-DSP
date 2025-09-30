from vitalDSP.health_analysis.interpretation_engine import InterpretationEngine
from vitalDSP.health_analysis.health_report_visualization import HealthReportVisualizer
from vitalDSP.health_analysis.html_template import (
    render_report,
)
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import time

# from vitalDSP.health_analysis.file_io import FileIO
import numpy as np
import multiprocessing

# from functools import lru_cache


def process_single_feature_visualization(
    feature_item, config, segment_duration, output_dir
):
    """
    Process a single feature for visualization in a separate process.
    This function must be at module level for multiprocessing to work.
    """
    feature, values = feature_item

    try:
        # Import here to avoid issues with multiprocessing
        from vitalDSP.health_analysis.health_report_visualization import (
            HealthReportVisualizer,
        )

        # Create a new visualizer instance for each process
        process_visualizer = HealthReportVisualizer(config, segment_duration)

        print(f"Processing {feature}...")
        feature_visualizations = process_visualizer.create_visualizations(
            {feature: values}, output_dir
        )
        print(f"✅ Completed {feature}")
        return (feature, feature_visualizations[feature])

    except Exception as e:
        print(f"❌ Error generating visualization for {feature}: {e}")
        return (feature, {})


class HealthReportGenerator:
    """
    A class to generate a health report based on feature data, including interpretations, visualizations, and contradictions/correlations.

    This class handles the process of interpreting feature data (e.g., heart rate variability, ECG/PPG data),
    generating visualizations for features, and rendering the final HTML report. It uses fully concurrent processing:
    - Feature interpretation: Concurrent processing using ThreadPoolExecutor (CPU-bound, thread-safe)
    - Visualization generation: Concurrent processing using ProcessPoolExecutor (matplotlib process-safe)

    Attributes:
        feature_data (dict): Dictionary containing feature names as keys and their corresponding values.
        segment_duration (str): Duration of the segment, either "1_min" or "5_min". Default is "1_min".
        interpreter (InterpretationEngine): Instance of InterpretationEngine to interpret feature data.
        visualizer (HealthReportVisualizer): Instance of HealthReportVisualizer to create visualizations.
        max_workers (int): Maximum number of concurrent workers for processing. Default is CPU count.

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
        self,
        feature_data,
        segment_duration="1_min",
        feature_config_path=None,
        max_workers=None,
    ):
        """
        Initializes the HealthReportGenerator with the provided feature data and segment duration.

        Args:
            feature_data (dict): Dictionary containing feature names as keys and their corresponding values as values.
                                Example: {"nn50": 35, "rmssd": 55, "sdnn": 70}
            segment_duration (str): The duration of the analyzed segment, either '1_min' or '5_min'. Default is '1_min'.
            feature_config_path (str, optional): Path to a custom feature YAML configuration file. If not provided, the default config will be used.
            max_workers (int, optional): Maximum number of concurrent workers for processing. Default is CPU count.

        Examples
        --------
        >>> feature_data = {"nn50": 45, "rmssd": 70, "sdnn": 120}
        >>> generator = HealthReportGenerator(feature_data, segment_duration="5_min")
        >>> report_html = generator.generate()
        >>>
        >>> # With custom concurrency settings
        >>> generator_fast = HealthReportGenerator(feature_data, max_workers=8)
        >>> report_html = generator_fast.generate()
        """
        self.feature_data = feature_data
        self.segment_duration = self._validate_segment_duration(segment_duration)
        self.interpreter = InterpretationEngine(feature_config_path)
        self.visualizer = HealthReportVisualizer(
            self.interpreter.config, self.segment_duration
        )
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers or multiprocessing.cpu_count()

    def set_concurrency(self, max_workers=None):
        """
        Configure concurrency settings for report generation.

        Args:
            max_workers (int, optional): Maximum number of concurrent workers.
                                       If None, uses CPU count.

        Example:
            >>> generator.set_concurrency(max_workers=4)
            >>> generator.set_concurrency()  # Reset to CPU count
        """
        if max_workers is None:
            self.max_workers = multiprocessing.cpu_count()
        else:
            self.max_workers = max(1, min(max_workers, multiprocessing.cpu_count() * 2))
        print(f"Concurrency set to {self.max_workers} workers")

    def get_performance_info(self):
        """
        Get information about system performance and recommended settings.

        Returns:
            dict: Performance information including CPU count, recommended workers, etc.
        """
        cpu_count = multiprocessing.cpu_count()
        feature_count = len(self.feature_data)

        return {
            "cpu_count": cpu_count,
            "feature_count": feature_count,
            "current_max_workers": self.max_workers,
            "recommended_feature_workers": min(feature_count, cpu_count),
            "recommended_viz_workers": min(cpu_count, 8),
            "estimated_speedup": (
                min(feature_count, cpu_count) if feature_count > 1 else 1
            ),
        }

    def _validate_segment_duration(self, segment_duration):
        """Validate and normalize segment duration format."""
        valid_durations = ["1_min", "5_min"]

        # Handle common variations
        if segment_duration in ["1 min", "1min", "1"]:
            return "1_min"
        elif segment_duration in ["5 min", "5min", "5"]:
            return "5_min"
        elif segment_duration in valid_durations:
            return segment_duration
        else:
            self.logger.warning(
                f"Invalid segment duration '{segment_duration}', defaulting to '1_min'"
            )
            return "1_min"

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

    def _process_feature(self, feature_name, values, filter_status):
        """
        Process a single feature for interpretation.

        Args:
            feature_name (str): Name of the feature
            values: Feature values
            filter_status (str): Filter status for feature selection

        Returns:
            tuple: (feature_name, processed_data) or (feature_name, None) if error
        """
        try:
            # Handle both single values and arrays
            if not isinstance(values, (list, tuple, np.ndarray)):
                values = [values]

            # Convert the values to a NumPy array
            values = np.array(values)

            # Drop NaN or Inf values
            values = values[np.isfinite(values)]

            # If all values are NaN/Inf and nothing remains, raise an error
            if len(values) == 0:
                raise ValueError(
                    f"All values for feature '{feature_name}' are NaN or Inf"
                )

            # Downsample data to reduce memory overhead for large datasets
            if len(values) > 1000:
                self.logger.warning(
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
                return (feature_name, None)

            # Store aggregated information
            median_value = np.median(values)
            stddev_value = np.std(values)

            processed_data = {
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

            return (feature_name, processed_data)

        except Exception as e:
            # Log the error but continue processing other features
            self.logger.error(f"Error processing {feature_name}: {e}")
            return (feature_name, None)

    @staticmethod
    def batch_visualization(visualizer, feature_data, output_dir, processes=4):
        """
        Generate visualizations in parallel using ProcessPoolExecutor to avoid matplotlib threading issues.
        Args:
            visualizer (HealthReportVisualizer): The visualizer instance.
            feature_data (dict): Dictionary of features and their values.
            output_dir (str): Directory to save the visualizations.
            processes (int): Number of processes for parallel visualization.
        Returns:
            dict: Paths to the generated visualizations.
        """
        # Generate visualizations for all features using processes to avoid matplotlib threading issues
        filtered_data = feature_data

        print(
            f"Generating visualizations for {len(filtered_data)} features using {processes} processes (matplotlib process-safe)..."
        )

        visualizations = {}

        # Use ProcessPoolExecutor to avoid matplotlib threading issues
        with ProcessPoolExecutor(max_workers=processes) as executor:
            # Submit all tasks using the module-level function
            future_to_feature = {
                executor.submit(
                    process_single_feature_visualization,
                    item,
                    visualizer.config,
                    visualizer.segment_duration,
                    output_dir,
                ): item[0]
                for item in filtered_data.items()
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_feature):
                feature_name = future_to_feature[future]
                try:
                    feature, feature_plots = future.result()
                    visualizations[feature] = feature_plots
                except Exception as e:
                    print(f"❌ Error processing {feature_name}: {e}")
                    visualizations[feature_name] = {}

        # Normalize all paths for web compatibility
        normalized_visualizations = {}
        for feature, plots in visualizations.items():
            normalized_visualizations[feature] = {}
            for plot_type, path in plots.items():
                if path:
                    # Convert backslashes to forward slashes for web compatibility
                    web_path = path.replace("\\", "/")
                    # Keep relative paths for file:// protocol compatibility
                    normalized_visualizations[feature][plot_type] = web_path
                else:
                    normalized_visualizations[feature][plot_type] = path

        print(
            f"✅ Completed visualization generation for {len(normalized_visualizations)} features"
        )
        return normalized_visualizations

    def generate(self, filter_status="all", output_dir=None):
        """
        Generates the complete health report by interpreting the features, generating visualizations, and rendering an HTML report.

        The report will include the description of each feature, its interpretation (in-range, above range, or below range),
        any detected contradictions, correlations, and visualizations for each feature.

        Returns:
            str: HTML content of the generated health report.

        Example Usage:
            >>> feature_data = {"nn50": 45, "rmssd": 70, "sdnn": 120}
            >>> generator = HealthReportGenerator(feature_data, segment_duration="5_min")
            >>> report_html = generator.generate()
            >>> with open('health_report.html', 'w') as file:
            >>>     file.write(report_html)
        """
        segment_values = {}

        try:
            # Step 1: Interpret each feature concurrently
            print(f"Processing {len(self.feature_data)} features concurrently...")
            start_time = time.time()

            # Determine optimal number of workers (CPU bound tasks)
            max_workers = min(len(self.feature_data), self.max_workers)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all feature processing tasks
                future_to_feature = {
                    executor.submit(
                        self._process_feature, feature_name, values, filter_status
                    ): feature_name
                    for feature_name, values in self.feature_data.items()
                }

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_feature):
                    feature_name = future_to_feature[future]
                    try:
                        processed_feature, processed_data = future.result()
                        if processed_data is not None:
                            segment_values[processed_feature] = processed_data
                    except Exception as e:
                        self.logger.error(f"Error processing {feature_name}: {e}")
                        continue

            processing_time = time.time() - start_time
            print(f"✅ Feature processing completed in {processing_time:.2f} seconds")

            # Step 1.5: Process contradictions and correlations (now handled dynamically)
            # segment_values = process_interpretations(segment_values)  # No longer needed

        except Exception as e:
            # Log the error if step 1 fails
            self.logger.error(f"Error in processing feature interpretations: {e}")

        # Step 2: Generate visualizations for all features (optional)
        visualizations = {}
        optimal_processes = 0  # Initialize to avoid UnboundLocalError
        if output_dir:
            try:
                print(
                    "Generating visualizations concurrently using processes (matplotlib process-safe)... This may take a moment."
                )
                viz_start_time = time.time()

                # Use optimal number of processes for visualization (CPU-bound)
                optimal_processes = min(
                    self.max_workers, 8
                )  # Cap at 8 for CPU-bound tasks

                visualizations = self.batch_visualization(
                    self.visualizer,
                    self.feature_data,
                    output_dir,
                    processes=optimal_processes,
                )

                viz_time = time.time() - viz_start_time
                print(
                    f"✅ Generated visualizations for {len(visualizations)} features in {viz_time:.2f} seconds"
                )
            except Exception as e:
                self.logger.error(f"Error generating visualizations: {e}")
                print("Skipping visualizations due to error.")
                visualizations = {}
        else:
            print("No output directory specified, skipping visualizations.")

        try:
            # Step 3: Generate dynamic analysis components
            if not segment_values:
                self.logger.warning("No segment values available for dynamic analysis")
                dynamic_analysis = {}
            else:
                dynamic_analysis = self._generate_dynamic_analysis(segment_values)

            # Step 4: Render the report with dynamic components
            report_html = render_report(
                segment_values, visualizations, dynamic_analysis=dynamic_analysis
            )
        except Exception as e:
            # Log the error if report rendering fails
            self.logger.error(f"Error rendering report: {e}")
            report_html = "<h1>Error generating report</h1>"

        # Print total timing summary
        total_time = time.time() - start_time
        print("\n🚀 Report generation completed!")
        print(f"📊 Total time: {total_time:.2f} seconds")
        print(f"⚡ Features processed: {len(segment_values)}")
        print(f"📈 Visualizations generated: {len(visualizations)}")
        print(f"🔧 Feature processing: Concurrent ({max_workers} workers)")
        if optimal_processes > 0:
            print(
                f"🎨 Visualization generation: Concurrent ({optimal_processes} processes)"
            )
        else:
            print("🎨 Visualization generation: Skipped (no output directory)")

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
            # Extract raw feature values for correlation analysis
            feature_data = {}
            for feature_name, feature_data_dict in segment_values.items():
                if isinstance(feature_data_dict, dict) and "value" in feature_data_dict:
                    # Extract mean value from the list of values
                    values = feature_data_dict["value"]
                    if isinstance(values, list) and len(values) > 0:
                        feature_data[feature_name] = np.mean(values)
                    else:
                        feature_data[feature_name] = values
                else:
                    # If it's already a raw value
                    feature_data[feature_name] = feature_data_dict

            # Use the interpretation engine to analyze cross-correlations
            cross_correlations = self.interpreter._analyze_cross_feature_correlations(
                feature_data, self.segment_duration
            )
            return cross_correlations
        except Exception as e:
            self.logger.error(f"Error generating cross-correlations: {e}")
            return []
