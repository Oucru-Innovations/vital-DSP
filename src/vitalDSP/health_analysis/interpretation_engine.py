import numpy as np
import yaml
import importlib.resources as pkg_resources


class InterpretationEngine:
    """
    A class responsible for interpreting feature values based on a YAML configuration file.

    This class loads a YAML file that contains the normal ranges, interpretation, contradictions, and correlations for different features.
    It then provides methods to interpret feature values, check for contradictions, and identify correlations.

    Attributes:
        config (dict): The loaded YAML configuration data for feature interpretation.

    Examples
    --------
    >>> from vitalDSP.health_analysis.interpretation_engine import InterpretationEngine
    >>>
    >>> # Example 1: Using default configuration
    >>> engine = InterpretationEngine()
    >>> result = engine.interpret_feature("sdnn", 45.0, "1_min")
    >>> print(f"SDNN interpretation: {result['interpretation']}")
    >>>
    >>> # Example 2: Using custom configuration file
    >>> engine_custom = InterpretationEngine("path/to/custom_config.yml")
    >>> result_custom = engine_custom.interpret_feature("rmssd", 25.0, "5_min")
    >>> print(f"RMSSD range status: {engine_custom.get_range_status('rmssd', 25.0, '5_min')}")
    >>>
    >>> # Example 3: Multiple feature interpretation
    >>> feature_data = {"sdnn": 45.0, "rmssd": 25.0, "nn50": 15}
    >>> multi_result = engine.interpret_multiple_features(feature_data, "1_min")
    >>> print(f"Overall assessment: {multi_result['overall_assessment']}")
    """

    def __init__(self, yaml_file=None):
        """
        Initializes the InterpretationEngine by loading the feature configuration YAML file.

        If no custom YAML file path is provided, it loads the default configuration file from the package.

        Args:
            yaml_file (str, optional): Path to a custom YAML configuration file. If not provided, it loads the default feature_config.yml.

        Examples
        --------
        >>> engine = InterpretationEngine("path/to/custom_feature_config.yml")
        >>> engine = InterpretationEngine()  # Will load the default feature_config.yml from the package
        """
        if yaml_file is None:
            self.config = self.load_feature_config()
        else:
            self.config = self._load_yaml(yaml_file)

    def _load_yaml(self, filepath):
        """
        Loads the YAML file from the provided file path.

        Args:
            filepath (str): Path to the YAML file to load.

        Returns:
            dict: Parsed YAML data as a dictionary.

        Raises:
            FileNotFoundError: If the YAML file is not found at the given path.
            Exception: If there is an error in parsing the YAML file.

        Example Usage:
            >>> yaml_data = engine._load_yaml("path/to/config.yml")
        """
        try:
            with open(filepath, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file not found: {filepath}")
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing YAML file: {e}")

    def load_feature_config(self):
        """
        Loads the default feature_config.yml file from the package resources.

        Returns:
            dict: Parsed YAML content from the feature_config.yml file.

        Raises:
            FileNotFoundError: If the feature_config.yml cannot be loaded.

        Example Usage:
            >>> config = engine.load_feature_config()
        """
        try:
            with pkg_resources.open_text(
                "vitalDSP.health_analysis", "feature_config.yml"
            ) as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise FileNotFoundError(f"Error loading feature_config.yml: {str(e)}")

    def get_range_status(self, feature_name, value, segment_duration="1 min"):
        """
        Determines whether the given feature value is within, above, or below the normal range.

        Args:
            feature_name (str): The name of the feature to check.
            value (float): The value of the feature to check.
            segment_duration (str): Segment duration, either "1 min" or "5 min".

        Returns:
            str: 'in_range', 'above_range', or 'below_range' based on the value's relation to the normal range.
        """
        if feature_name not in self.config:
            return "unknown"

        feature_info = self.config[feature_name]
        normal_range = feature_info.get("normal_range", {}).get(segment_duration, None)

        if not normal_range:
            return "unknown"

        # Convert 'inf' or '-inf' to np.inf and -np.inf
        normal_min, normal_max = normal_range
        if isinstance(normal_min, str) and normal_min == "-inf":
            normal_min = -np.inf
        if isinstance(normal_max, str) and normal_max == "inf":
            normal_max = np.inf

        # Check the value against the normal range
        if value < normal_min:
            return "below_range"
        elif value > normal_max:
            return "above_range"
        else:
            return "in_range"

    def interpret_feature(self, feature_name, value, segment_duration="1 min"):
        """
        Interprets the value of a given feature based on its configuration in the YAML file.

        This method compares the feature value with its normal range and provides interpretation,
        which could indicate whether the value is in-range, below-range, or above-range.

        Args:
            feature_name (str): The name of the feature to interpret (e.g., "NN50", "RMSSD").
            value (float): The value of the feature to be interpreted.
            segment_duration (str): The duration of the segment, either '1 min' or '5 min'. Default is '1 min'.

        Returns:
            dict: A dictionary containing:
                  - "description": The description of the feature.
                  - "normal_range": The normal range of the feature for the given segment duration.
                  - "interpretation": The interpretation of the feature value (whether in-range, below-range, or above-range).
                  - "contradiction": Contradiction explanation with other related features.
                  - "correlation": Correlation explanation with other related features.

        Example Usage:
            >>> engine = InterpretationEngine()
            >>> result = engine.interpret_feature("NN50", 45, "5 min")
            >>> print(result)
            {
                "description": "Number of significant changes in heart rate. High NN50 suggests healthy parasympathetic activity.",
                "normal_range": [40, 150],
                "interpretation": "Normal parasympathetic activity. No immediate concern.",
                "contradiction": "Low NN50 contradicts high RMSSD, as both should indicate parasympathetic activity.",
                "correlation": "Positively correlated with RMSSD, as both represent short-term heart rate variability."
            }
        """
        if feature_name not in self.config:
            return {
                "description": f"Feature '{feature_name}' not found in configuration."
            }

        feature_info = self.config[feature_name]

        # Get normal range for the specific segment duration
        normal_range = feature_info.get("normal_range", {}).get(segment_duration, None)
        if not normal_range:
            return {"normal_range": f"Normal range for '{feature_name}' not available."}

        # Parse and handle 'inf' and '-inf' strings in the normal range
        normal_range = [self._parse_inf_values(val) for val in normal_range]

        # Dynamic interpretation based on value and context
        interpretation = self._generate_dynamic_interpretation(
            feature_name, value, normal_range, feature_info, segment_duration
        )

        return {
            "description": self._generate_dynamic_description(
                feature_name, value, normal_range, feature_info, segment_duration
            ),
            "normal_range": normal_range,
            "interpretation": interpretation,
            "contradiction": self._get_feature_contradiction(feature_name),
            "correlation": self._generate_dynamic_correlation(
                feature_name, value, normal_range, feature_info, segment_duration
            ),
        }

    def _parse_inf_values(self, val):
        """
        Parses the 'inf' and '-inf' strings and returns numpy infinity values.

        Args:
            val (str or float): The value to parse.

        Returns:
            float: Parsed value where 'inf' or '-inf' strings are converted to np.inf or -np.inf respectively.

        Example Usage:
            >>> value = engine._parse_inf_values("-inf")
            >>> print(value)
            -inf
        """
        if isinstance(val, str):
            if val.lower() == "inf":
                return np.inf
            elif val.lower() == "-inf":
                return -np.inf
        return val

    def _get_feature_contradiction(self, feature_name):
        """
        Retrieves the contradiction information for a given feature from the YAML configuration.

        Args:
            feature_name (str): The name of the feature.

        Returns:
            str: The contradiction information for the feature.

        Example Usage:
            >>> contradiction = engine._get_feature_contradiction("NN50")
            >>> print(contradiction)
            "Low NN50 contradicts high RMSSD, as both should indicate parasympathetic activity."
        """
        feature_info = self.config.get(feature_name, {})
        contradiction_info = feature_info.get(
            "contradiction", "No contradiction found."
        )

        # Handle both string and dictionary formats for contradiction
        if isinstance(contradiction_info, dict):
            # If it's a dictionary, get the first contradiction description
            contradiction_descriptions = list(contradiction_info.values())
            return (
                contradiction_descriptions[0]
                if contradiction_descriptions
                else "No contradiction found."
            )
        else:
            return contradiction_info

    def _get_feature_correlation(self, feature_name):
        """
        Retrieves the correlation information for a given feature from the YAML configuration.

        Args:
            feature_name (str): The name of the feature.

        Returns:
            str: The correlation information for the feature.

        Example Usage:
            >>> correlation = engine._get_feature_correlation("NN50")
            >>> print(correlation)
            "Positively correlated with RMSSD, as both represent short-term heart rate variability."
        """
        feature_info = self.config.get(feature_name, {})
        correlation_info = feature_info.get("correlation", "No correlation found.")

        # Handle both string and dictionary formats for correlation
        if isinstance(correlation_info, dict):
            # If it's a dictionary, get the first correlation description
            correlation_descriptions = list(correlation_info.values())
            return (
                correlation_descriptions[0]
                if correlation_descriptions
                else "No correlation found."
            )
        else:
            return correlation_info

    def _generate_dynamic_correlation(
        self, feature_name, value, normal_range, feature_info, segment_duration
    ):
        """Generate dynamic correlation information based on feature value and context."""
        base_correlation = feature_info.get("correlation", "No correlation found.")

        # Handle both string and dictionary formats for base correlation
        if isinstance(base_correlation, dict):
            # If it's a dictionary, get the first correlation description
            correlation_descriptions = list(base_correlation.values())
            base_correlation_text = (
                correlation_descriptions[0]
                if correlation_descriptions
                else "No correlation found."
            )
        else:
            base_correlation_text = base_correlation

        # Get correlation thresholds and related features
        correlation_thresholds = feature_info.get("thresholds", {}).get(
            "correlation", {}
        )

        # Generate dynamic correlation context based on feature type and value
        dynamic_context = self._get_correlation_context(
            feature_name, value, normal_range, correlation_thresholds
        )

        # Combine base correlation with dynamic context
        if dynamic_context:
            return f"{base_correlation_text} {dynamic_context}"
        else:
            return base_correlation_text

    def _get_correlation_context(
        self, feature_name, value, normal_range, correlation_thresholds
    ):
        """Generate contextual correlation information based on feature value."""
        min_val, max_val = normal_range
        range_span = max_val - min_val

        # Calculate how the value relates to the normal range
        if value < min_val:
            range_position = "below_normal"
            deviation = (min_val - value) / range_span if range_span > 0 else 1
        elif value > max_val:
            range_position = "above_normal"
            deviation = (value - max_val) / range_span if range_span > 0 else 1
        else:
            range_position = "within_normal"
            center = (min_val + max_val) / 2
            distance_from_center = (
                abs(value - center) / (range_span / 2) if range_span > 0 else 0
            )
            if distance_from_center < 0.2:
                range_position = "optimal"
            elif distance_from_center < 0.5:
                range_position = "good"
            else:
                range_position = "acceptable"

        # Generate correlation context based on feature type and range position
        correlation_contexts = {
            "sdnn": {
                "below_normal": f"Your SDNN of {value:.1f} ms is below normal, which typically correlates with reduced overall heart rate variability. This may indicate that other HRV parameters like RMSSD and NN50 are also likely to be reduced, suggesting a pattern of decreased autonomic function.",
                "above_normal": f"Your SDNN of {value:.1f} ms is above normal, which typically correlates with increased overall heart rate variability. This may indicate that other HRV parameters like RMSSD and NN50 are also likely to be elevated, suggesting enhanced autonomic function or potential measurement artifacts.",
                "optimal": f"Your SDNN of {value:.1f} ms is in the optimal range, which typically correlates well with other HRV parameters. This suggests a balanced autonomic nervous system where RMSSD, NN50, and other variability measures should also be within healthy ranges.",
                "good": f"Your SDNN of {value:.1f} ms is in the good range, which typically correlates positively with other HRV parameters. This suggests that RMSSD, NN50, and other variability measures should also be within acceptable ranges.",
                "acceptable": f"Your SDNN of {value:.1f} ms is within normal limits, which typically correlates moderately with other HRV parameters. This suggests that other variability measures should also be within normal ranges.",
            },
            "rmssd": {
                "below_normal": f"Your RMSSD of {value:.1f} ms is below normal, which typically correlates with reduced parasympathetic activity. This may indicate that other short-term variability measures like NN50 are also likely to be reduced, suggesting decreased vagal tone.",
                "above_normal": f"Your RMSSD of {value:.1f} ms is above normal, which typically correlates with increased parasympathetic activity. This may indicate that other short-term variability measures like NN50 are also likely to be elevated, suggesting enhanced vagal tone or potential measurement artifacts.",
                "optimal": f"Your RMSSD of {value:.1f} ms is in the optimal range, which typically correlates strongly with other short-term HRV parameters. This suggests that NN50 and other parasympathetic indicators should also be within healthy ranges.",
                "good": f"Your RMSSD of {value:.1f} ms is in the good range, which typically correlates well with other short-term HRV parameters. This suggests that NN50 and other parasympathetic indicators should also be within acceptable ranges.",
                "acceptable": f"Your RMSSD of {value:.1f} ms is within normal limits, which typically correlates moderately with other short-term HRV parameters. This suggests that other parasympathetic indicators should also be within normal ranges.",
            },
            "nn50": {
                "below_normal": f"Your NN50 of {value:.0f} is below normal, which typically correlates with reduced short-term heart rate variability. This may indicate that other variability measures like RMSSD and SDNN are also likely to be reduced, suggesting decreased autonomic function.",
                "above_normal": f"Your NN50 of {value:.0f} is above normal, which typically correlates with increased short-term heart rate variability. This may indicate that other variability measures like RMSSD and SDNN are also likely to be elevated, suggesting enhanced autonomic function or potential measurement artifacts.",
                "optimal": f"Your NN50 of {value:.0f} is in the optimal range, which typically correlates strongly with other HRV parameters. This suggests that RMSSD, SDNN, and other variability measures should also be within healthy ranges.",
                "good": f"Your NN50 of {value:.0f} is in the good range, which typically correlates well with other HRV parameters. This suggests that RMSSD, SDNN, and other variability measures should also be within acceptable ranges.",
                "acceptable": f"Your NN50 of {value:.0f} is within normal limits, which typically correlates moderately with other HRV parameters. This suggests that other variability measures should also be within normal ranges.",
            },
            "pnn50": {
                "below_normal": f"Your pNN50 of {value:.1f}% is below normal, which typically correlates with reduced short-term heart rate variability. This may indicate that other variability measures like RMSSD and NN50 are also likely to be reduced, suggesting decreased parasympathetic activity.",
                "above_normal": f"Your pNN50 of {value:.1f}% is above normal, which typically correlates with increased short-term heart rate variability. This may indicate that other variability measures like RMSSD and NN50 are also likely to be elevated, suggesting enhanced parasympathetic activity or potential measurement artifacts.",
                "optimal": f"Your pNN50 of {value:.1f}% is in the optimal range, which typically correlates strongly with other short-term HRV parameters. This suggests that RMSSD, NN50, and other parasympathetic indicators should also be within healthy ranges.",
                "good": f"Your pNN50 of {value:.1f}% is in the good range, which typically correlates well with other short-term HRV parameters. This suggests that RMSSD, NN50, and other parasympathetic indicators should also be within acceptable ranges.",
                "acceptable": f"Your pNN50 of {value:.1f}% is within normal limits, which typically correlates moderately with other short-term HRV parameters. This suggests that other parasympathetic indicators should also be within normal ranges.",
            },
        }

        # Get the appropriate correlation context
        feature_contexts = correlation_contexts.get(feature_name.lower(), {})
        return feature_contexts.get(range_position, "")

    def _analyze_cross_feature_correlations(self, feature_data, segment_duration):
        """Analyze correlations between different features based on their values."""
        correlations = []

        # Define feature groups for correlation analysis
        hrv_features = ["sdnn", "rmssd", "nn50", "pnn50"]
        available_hrv_features = [f for f in hrv_features if f in feature_data]

        if len(available_hrv_features) >= 2:
            # Analyze HRV feature correlations
            hrv_correlations = self._analyze_hrv_correlations(
                available_hrv_features, feature_data, segment_duration
            )
            correlations.extend(hrv_correlations)

        # Analyze other feature correlations
        other_correlations = self._analyze_other_correlations(
            feature_data, segment_duration
        )
        correlations.extend(other_correlations)

        return correlations

    def _analyze_hrv_correlations(self, hrv_features, feature_data, segment_duration):
        """Analyze correlations between HRV features."""
        correlations = []

        # Check SDNN-RMSSD correlation
        if "sdnn" in hrv_features and "rmssd" in hrv_features:
            sdnn_value = (
                feature_data["sdnn"].get("value", 0)
                if isinstance(feature_data["sdnn"], dict)
                else feature_data["sdnn"]
            )
            rmssd_value = (
                feature_data["rmssd"].get("value", 0)
                if isinstance(feature_data["rmssd"], dict)
                else feature_data["rmssd"]
            )
            correlation_analysis = self._analyze_sdnn_rmssd_correlation(
                sdnn_value, rmssd_value, segment_duration
            )
            if correlation_analysis:
                correlations.append(correlation_analysis)

        # Check RMSSD-NN50 correlation
        if "rmssd" in hrv_features and "nn50" in hrv_features:
            rmssd_value = (
                feature_data["rmssd"].get("value", 0)
                if isinstance(feature_data["rmssd"], dict)
                else feature_data["rmssd"]
            )
            nn50_value = (
                feature_data["nn50"].get("value", 0)
                if isinstance(feature_data["nn50"], dict)
                else feature_data["nn50"]
            )
            correlation_analysis = self._analyze_rmssd_nn50_correlation(
                rmssd_value, nn50_value, segment_duration
            )
            if correlation_analysis:
                correlations.append(correlation_analysis)

        # Check SDNN-NN50 correlation
        if "sdnn" in hrv_features and "nn50" in hrv_features:
            sdnn_value = (
                feature_data["sdnn"].get("value", 0)
                if isinstance(feature_data["sdnn"], dict)
                else feature_data["sdnn"]
            )
            nn50_value = (
                feature_data["nn50"].get("value", 0)
                if isinstance(feature_data["nn50"], dict)
                else feature_data["nn50"]
            )
            correlation_analysis = self._analyze_sdnn_nn50_correlation(
                sdnn_value, nn50_value, segment_duration
            )
            if correlation_analysis:
                correlations.append(correlation_analysis)

        return correlations

    def _analyze_sdnn_rmssd_correlation(
        self, sdnn_value, rmssd_value, segment_duration
    ):
        """Analyze correlation between SDNN and RMSSD."""
        # Get normal ranges
        sdnn_range = (
            self.config.get("sdnn", {})
            .get("normal_range", {})
            .get(segment_duration, [20, 50])
        )
        rmssd_range = (
            self.config.get("rmssd", {})
            .get("normal_range", {})
            .get(segment_duration, [10, 30])
        )

        # Calculate relative positions within normal ranges
        sdnn_position = self._get_relative_position(sdnn_value, sdnn_range)
        rmssd_position = self._get_relative_position(rmssd_value, rmssd_range)

        # Analyze correlation strength
        position_diff = abs(sdnn_position - rmssd_position)

        if position_diff < 0.2:
            return {
                "features": ["SDNN", "RMSSD"],
                "type": "strong_positive",
                "description": f"SDNN ({sdnn_value:.1f} ms) and RMSSD ({rmssd_value:.1f} ms) show strong positive correlation, both indicating similar levels of heart rate variability. This suggests consistent autonomic function across different time scales.",
                "strength": "strong",
            }
        elif position_diff < 0.5:
            return {
                "features": ["SDNN", "RMSSD"],
                "type": "moderate_positive",
                "description": f"SDNN ({sdnn_value:.1f} ms) and RMSSD ({rmssd_value:.1f} ms) show moderate positive correlation, indicating generally consistent heart rate variability patterns.",
                "strength": "moderate",
            }
        else:
            return {
                "features": ["SDNN", "RMSSD"],
                "type": "weak_correlation",
                "description": f"SDNN ({sdnn_value:.1f} ms) and RMSSD ({rmssd_value:.1f} ms) show weak correlation, suggesting different patterns of heart rate variability. This may indicate mixed autonomic function or measurement inconsistencies.",
                "strength": "weak",
            }

    def _analyze_rmssd_nn50_correlation(
        self, rmssd_value, nn50_value, segment_duration
    ):
        """Analyze correlation between RMSSD and NN50."""
        rmssd_range = (
            self.config.get("rmssd", {})
            .get("normal_range", {})
            .get(segment_duration, [10, 30])
        )
        nn50_range = (
            self.config.get("nn50", {})
            .get("normal_range", {})
            .get(segment_duration, [10, 50])
        )

        rmssd_position = self._get_relative_position(rmssd_value, rmssd_range)
        nn50_position = self._get_relative_position(nn50_value, nn50_range)

        position_diff = abs(rmssd_position - nn50_position)

        if position_diff < 0.2:
            return {
                "features": ["RMSSD", "NN50"],
                "type": "strong_positive",
                "description": f"RMSSD ({rmssd_value:.1f} ms) and NN50 ({nn50_value:.0f}) show strong positive correlation, both indicating similar levels of short-term heart rate variability. This suggests consistent parasympathetic activity.",
                "strength": "strong",
            }
        elif position_diff < 0.5:
            return {
                "features": ["RMSSD", "NN50"],
                "type": "moderate_positive",
                "description": f"RMSSD ({rmssd_value:.1f} ms) and NN50 ({nn50_value:.0f}) show moderate positive correlation, indicating generally consistent short-term variability patterns.",
                "strength": "moderate",
            }
        else:
            return {
                "features": ["RMSSD", "NN50"],
                "type": "weak_correlation",
                "description": f"RMSSD ({rmssd_value:.1f} ms) and NN50 ({nn50_value:.0f}) show weak correlation, suggesting different patterns of short-term heart rate variability. This may indicate mixed parasympathetic function.",
                "strength": "weak",
            }

    def _analyze_sdnn_nn50_correlation(self, sdnn_value, nn50_value, segment_duration):
        """Analyze correlation between SDNN and NN50."""
        sdnn_range = (
            self.config.get("sdnn", {})
            .get("normal_range", {})
            .get(segment_duration, [20, 50])
        )
        nn50_range = (
            self.config.get("nn50", {})
            .get("normal_range", {})
            .get(segment_duration, [10, 50])
        )

        sdnn_position = self._get_relative_position(sdnn_value, sdnn_range)
        nn50_position = self._get_relative_position(nn50_value, nn50_range)

        position_diff = abs(sdnn_position - nn50_position)

        if position_diff < 0.2:
            return {
                "features": ["SDNN", "NN50"],
                "type": "strong_positive",
                "description": f"SDNN ({sdnn_value:.1f} ms) and NN50 ({nn50_value:.0f}) show strong positive correlation, both indicating similar levels of heart rate variability. This suggests consistent overall autonomic function.",
                "strength": "strong",
            }
        elif position_diff < 0.5:
            return {
                "features": ["SDNN", "NN50"],
                "type": "moderate_positive",
                "description": f"SDNN ({sdnn_value:.1f} ms) and NN50 ({nn50_value:.0f}) show moderate positive correlation, indicating generally consistent variability patterns.",
                "strength": "moderate",
            }
        else:
            return {
                "features": ["SDNN", "NN50"],
                "type": "weak_correlation",
                "description": f"SDNN ({sdnn_value:.1f} ms) and NN50 ({nn50_value:.0f}) show weak correlation, suggesting different patterns of heart rate variability. This may indicate mixed autonomic function.",
                "strength": "weak",
            }

    def _analyze_other_correlations(self, feature_data, segment_duration):
        """Analyze correlations between other features."""
        correlations = []

        # Add other feature correlation analyses here as needed
        # For example, heart rate vs HRV features, etc.

        return correlations

    def _get_relative_position(self, value, normal_range):
        """Get the relative position of a value within its normal range (0-1 scale)."""
        min_val, max_val = normal_range
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)

    def interpret_multiple_features(self, feature_data, segment_duration="1 min"):
        """
        Interprets multiple features together to provide context-aware analysis.

        Args:
            feature_data (dict): Dictionary containing feature names and their values.
            segment_duration (str): The duration of the segment, either '1 min' or '5 min'.

        Returns:
            dict: Dictionary containing cross-feature analysis and patterns.
        """
        try:
            interpretations = {}
            patterns = []
            warnings = []

            # Interpret each feature individually
            for feature_name, value in feature_data.items():
                interpretations[feature_name] = self.interpret_feature(
                    feature_name, value, segment_duration
                )

            # Analyze patterns across features
            patterns = self._analyze_cross_feature_patterns(
                feature_data, segment_duration
            )

            # Analyze cross-feature correlations
            cross_correlations = self._analyze_cross_feature_correlations(
                feature_data, segment_duration
            )

            # Generate warnings based on combinations
            warnings = self._generate_cross_feature_warnings(
                feature_data, segment_duration
            )

            return {
                "individual_interpretations": interpretations,
                "cross_feature_patterns": patterns,
                "cross_correlations": cross_correlations,
                "warnings": warnings,
                "overall_assessment": self._generate_overall_assessment(
                    interpretations
                ),
            }

        except Exception as e:
            return {
                "individual_interpretations": {},
                "cross_feature_patterns": [],
                "warnings": [f"Error in cross-feature analysis: {str(e)}"],
                "overall_assessment": "Unable to complete cross-feature analysis",
            }

    def _analyze_cross_feature_patterns(self, feature_data, segment_duration):
        """Analyze patterns across multiple features."""
        patterns = []

        # Check for autonomic dysfunction patterns
        hrv_features = ["sdnn", "rmssd", "nn50", "pnn50"]
        hrv_values = {f: feature_data.get(f) for f in hrv_features if f in feature_data}

        if len(hrv_values) >= 2:
            out_of_range_hrv = sum(
                1
                for f, v in hrv_values.items()
                if self.get_range_status(f, v, segment_duration) != "in_range"
            )

            if out_of_range_hrv >= 2:
                patterns.append(
                    {
                        "type": "autonomic_dysfunction",
                        "description": "Multiple heart rate variability parameters are abnormal, suggesting potential autonomic dysfunction",
                        "severity": "moderate" if out_of_range_hrv == 2 else "high",
                        "affected_features": [
                            f
                            for f, v in hrv_values.items()
                            if self.get_range_status(f, v, segment_duration)
                            != "in_range"
                        ],
                    }
                )

        # Check for cardiovascular risk patterns
        if "sdnn" in feature_data and "mean_nn" in feature_data:
            sdnn_status = self.get_range_status(
                "sdnn", feature_data["sdnn"], segment_duration
            )
            mean_nn_status = self.get_range_status(
                "mean_nn", feature_data["mean_nn"], segment_duration
            )

            # Low SDNN (below range) + Low mean_nn (below range = high heart rate) = cardiovascular risk
            if sdnn_status == "below_range" and mean_nn_status == "below_range":
                patterns.append(
                    {
                        "type": "cardiovascular_risk",
                        "description": "Low heart rate variability combined with elevated heart rate may indicate cardiovascular stress",
                        "severity": "high",
                        "affected_features": ["sdnn", "mean_nn"],
                    }
                )

        return patterns

    def _generate_cross_feature_warnings(self, feature_data, segment_duration):
        """Generate warnings based on feature combinations."""
        warnings = []

        # Check for contradictory patterns
        if "sdnn" in feature_data and "rmssd" in feature_data:
            sdnn_status = self.get_range_status(
                "sdnn", feature_data["sdnn"], segment_duration
            )
            rmssd_status = self.get_range_status(
                "rmssd", feature_data["rmssd"], segment_duration
            )

            if sdnn_status != rmssd_status:
                warnings.append(
                    f"Contradictory HRV patterns: SDNN is {sdnn_status} while RMSSD is {rmssd_status}"
                )

        # Check for extreme values
        for feature_name, value in feature_data.items():
            if feature_name in self.config:
                normal_range = (
                    self.config[feature_name]
                    .get("normal_range", {})
                    .get(segment_duration, [])
                )
                if normal_range and len(normal_range) == 2:
                    min_val, max_val = normal_range
                    if value < min_val * 0.5:  # Extremely low
                        warnings.append(
                            f"Extremely low {feature_name} value may indicate measurement error or critical condition"
                        )
                    elif value > max_val * 2:  # Extremely high
                        warnings.append(
                            f"Extremely high {feature_name} value may indicate measurement error or critical condition"
                        )

        return warnings

    def _generate_overall_assessment(self, interpretations):
        """Generate overall assessment based on individual interpretations."""
        if not interpretations:
            return "No features available for assessment"

        in_range_count = sum(
            1 for i in interpretations.values() if i.get("range_status") == "in_range"
        )
        total_count = len(interpretations)

        if in_range_count == total_count:
            return "All parameters are within normal ranges - excellent overall health"
        elif in_range_count >= total_count * 0.8:
            return "Most parameters are normal - good overall health with minor areas of concern"
        elif in_range_count >= total_count * 0.6:
            return "Mixed results - some parameters require attention"
        else:
            return "Multiple parameters are abnormal - comprehensive evaluation recommended"

    def _generate_dynamic_interpretation(
        self, feature_name, value, normal_range, feature_info, segment_duration
    ):
        """Generate dynamic interpretation based on value, context, and severity."""
        min_val, max_val = normal_range
        range_span = max_val - min_val

        # Calculate how far from normal range
        if value < min_val:
            deviation = (min_val - value) / range_span if range_span > 0 else 1
            severity = self._get_deviation_severity(deviation)
            base_interpretation = feature_info.get("interpretation", {}).get(
                "below_range", "Value is below normal range."
            )
            context = self._get_contextual_interpretation(
                feature_name, value, "below", severity, segment_duration
            )
        elif value > max_val:
            deviation = (value - max_val) / range_span if range_span > 0 else 1
            severity = self._get_deviation_severity(deviation)
            base_interpretation = feature_info.get("interpretation", {}).get(
                "above_range", "Value is above normal range."
            )
            context = self._get_contextual_interpretation(
                feature_name, value, "above", severity, segment_duration
            )
        else:
            # Within normal range - check if it's optimal, good, or just acceptable
            center = (min_val + max_val) / 2
            distance_from_center = (
                abs(value - center) / (range_span / 2) if range_span > 0 else 0
            )
            severity = (
                "optimal"
                if distance_from_center < 0.2
                else "good" if distance_from_center < 0.5 else "acceptable"
            )
            base_interpretation = feature_info.get("interpretation", {}).get(
                "in_range", "Value is within normal range."
            )
            context = self._get_contextual_interpretation(
                feature_name, value, "normal", severity, segment_duration
            )

        # Combine base interpretation with dynamic context
        return f"{base_interpretation} {context}"

    def _get_deviation_severity(self, deviation):
        """Determine severity based on deviation from normal range."""
        if deviation >= 2.0:
            return "critical"
        elif deviation >= 1.0:
            return "severe"
        elif deviation >= 0.5:
            return "moderate"
        else:
            return "mild"

    def _get_contextual_interpretation(
        self, feature_name, value, range_status, severity, segment_duration
    ):
        """Generate contextual interpretation based on feature type and severity."""
        context_templates = {
            "sdnn": {
                "below": {
                    "critical": f"Your SDNN of {value:.1f} ms indicates critically low heart rate variability, suggesting severe autonomic dysfunction. This may indicate advanced cardiovascular disease, heart failure, or severe stress. Immediate medical evaluation is strongly recommended.",
                    "severe": f"Your SDNN of {value:.1f} ms shows severely reduced heart rate variability, indicating significant autonomic dysfunction. This suggests increased cardiovascular risk and may be associated with chronic stress or underlying heart conditions. Medical consultation is advised.",
                    "moderate": f"Your SDNN of {value:.1f} ms indicates moderately reduced heart rate variability. This may suggest increased sympathetic activity or early signs of autonomic dysfunction. Consider stress management and lifestyle modifications.",
                    "mild": f"Your SDNN of {value:.1f} ms is slightly below optimal, indicating mild reduction in heart rate variability. This could be due to temporary stress or minor autonomic changes. Monitor and consider relaxation techniques.",
                },
                "above": {
                    "critical": f"Your SDNN of {value:.1f} ms is critically high, indicating extreme heart rate variability. This may suggest arrhythmias, vagal overactivity, or measurement artifacts. Immediate medical evaluation is recommended.",
                    "severe": f"Your SDNN of {value:.1f} ms shows severely elevated heart rate variability. This may indicate arrhythmias, vagal overactivity, or compensatory mechanisms. Medical consultation is advised.",
                    "moderate": f"Your SDNN of {value:.1f} ms is moderately elevated, suggesting increased heart rate variability. This could indicate good recovery, athletic conditioning, or potential arrhythmias. Monitor closely.",
                    "mild": f"Your SDNN of {value:.1f} ms is slightly above optimal, indicating increased heart rate variability. This may suggest good cardiovascular fitness or minor autonomic changes.",
                },
                "normal": {
                    "optimal": f"Your SDNN of {value:.1f} ms is in the optimal range, indicating excellent heart rate variability and balanced autonomic function. This suggests good cardiovascular health and stress resilience.",
                    "good": f"Your SDNN of {value:.1f} ms is in the good range, indicating healthy heart rate variability and well-functioning autonomic nervous system. This suggests good cardiovascular health.",
                    "acceptable": f"Your SDNN of {value:.1f} ms is within normal limits, indicating adequate heart rate variability. While not optimal, this suggests functional autonomic regulation.",
                },
            },
            "rmssd": {
                "below": {
                    "critical": f"Your RMSSD of {value:.1f} ms indicates critically low parasympathetic activity, suggesting severe autonomic dysfunction. This may indicate advanced cardiovascular disease or severe stress. Immediate medical evaluation is strongly recommended.",
                    "severe": f"Your RMSSD of {value:.1f} ms shows severely reduced parasympathetic activity, indicating significant autonomic dysfunction. This suggests increased cardiovascular risk. Medical consultation is advised.",
                    "moderate": f"Your RMSSD of {value:.1f} ms indicates moderately reduced parasympathetic activity. This may suggest increased sympathetic dominance or early autonomic dysfunction. Consider stress management techniques.",
                    "mild": f"Your RMSSD of {value:.1f} ms is slightly below optimal, indicating mild reduction in parasympathetic activity. This could be due to temporary stress. Consider relaxation techniques.",
                },
                "above": {
                    "critical": f"Your RMSSD of {value:.1f} ms is critically high, indicating extreme parasympathetic activity. This may suggest vagal overactivity or measurement artifacts. Immediate medical evaluation is recommended.",
                    "severe": f"Your RMSSD of {value:.1f} ms shows severely elevated parasympathetic activity. This may indicate vagal overactivity or compensatory mechanisms. Medical consultation is advised.",
                    "moderate": f"Your RMSSD of {value:.1f} ms is moderately elevated, suggesting increased parasympathetic activity. This could indicate good recovery or athletic conditioning.",
                    "mild": f"Your RMSSD of {value:.1f} ms is slightly above optimal, indicating increased parasympathetic activity. This may suggest good cardiovascular fitness.",
                },
                "normal": {
                    "optimal": f"Your RMSSD of {value:.1f} ms is in the optimal range, indicating excellent parasympathetic function and healthy short-term heart rate variability. This suggests good stress resilience and cardiovascular health.",
                    "good": f"Your RMSSD of {value:.1f} ms is in the good range, indicating healthy parasympathetic activity and well-functioning autonomic nervous system. This suggests good cardiovascular health.",
                    "acceptable": f"Your RMSSD of {value:.1f} ms is within normal limits, indicating adequate parasympathetic function. While not optimal, this suggests functional autonomic regulation.",
                },
            },
            "nn50": {
                "below": {
                    "critical": f"Your NN50 of {value:.0f} indicates critically low heart rate variability, suggesting severe autonomic dysfunction. This may indicate advanced cardiovascular disease. Immediate medical evaluation is strongly recommended.",
                    "severe": f"Your NN50 of {value:.0f} shows severely reduced heart rate variability, indicating significant autonomic dysfunction. Medical consultation is advised.",
                    "moderate": f"Your NN50 of {value:.0f} indicates moderately reduced heart rate variability. Consider stress management and lifestyle modifications.",
                    "mild": f"Your NN50 of {value:.0f} is slightly below optimal, indicating mild reduction in heart rate variability. Consider relaxation techniques.",
                },
                "above": {
                    "critical": f"Your NN50 of {value:.0f} is critically high, indicating extreme heart rate variability. This may suggest arrhythmias or measurement artifacts. Immediate medical evaluation is recommended.",
                    "severe": f"Your NN50 of {value:.0f} shows severely elevated heart rate variability. This may indicate arrhythmias or compensatory mechanisms. Medical consultation is advised.",
                    "moderate": f"Your NN50 of {value:.0f} is moderately elevated, suggesting increased heart rate variability. This could indicate good recovery or potential arrhythmias.",
                    "mild": f"Your NN50 of {value:.0f} is slightly above optimal, indicating increased heart rate variability. This may suggest good cardiovascular fitness.",
                },
                "normal": {
                    "optimal": f"Your NN50 of {value:.0f} is in the optimal range, indicating excellent heart rate variability and balanced autonomic function. This suggests good cardiovascular health.",
                    "good": f"Your NN50 of {value:.0f} is in the good range, indicating healthy heart rate variability and well-functioning autonomic nervous system.",
                    "acceptable": f"Your NN50 of {value:.0f} is within normal limits, indicating adequate heart rate variability and functional autonomic regulation.",
                },
            },
        }

        # Get the appropriate context template
        feature_contexts = context_templates.get(feature_name.lower(), {})
        range_contexts = feature_contexts.get(range_status, {})
        return range_contexts.get(
            severity,
            f"Your {feature_name} value of {value:.1f} is {range_status} the normal range.",
        )

    def _generate_dynamic_description(
        self, feature_name, value, normal_range, feature_info, segment_duration
    ):
        """Generate dynamic description based on the feature value and context."""
        base_description = feature_info.get("description", "No description available.")

        # Add contextual information based on the value
        min_val, max_val = normal_range
        range_span = max_val - min_val

        if value < min_val:
            deviation = (min_val - value) / range_span if range_span > 0 else 1
            if deviation >= 1.0:
                context = f" Your current value of {value:.1f} is significantly below the normal range of {min_val}-{max_val}, indicating potential health concerns."
            else:
                context = f" Your current value of {value:.1f} is below the normal range of {min_val}-{max_val}."
        elif value > max_val:
            deviation = (value - max_val) / range_span if range_span > 0 else 1
            if deviation >= 1.0:
                context = f" Your current value of {value:.1f} is significantly above the normal range of {min_val}-{max_val}, which may require attention."
            else:
                context = f" Your current value of {value:.1f} is above the normal range of {min_val}-{max_val}."
        else:
            center = (min_val + max_val) / 2
            distance_from_center = (
                abs(value - center) / (range_span / 2) if range_span > 0 else 0
            )
            if distance_from_center < 0.2:
                context = f" Your current value of {value:.1f} is in the optimal range of {min_val}-{max_val}, indicating excellent health."
            elif distance_from_center < 0.5:
                context = f" Your current value of {value:.1f} is in the good range of {min_val}-{max_val}, indicating healthy function."
            else:
                context = f" Your current value of {value:.1f} is within the normal range of {min_val}-{max_val}."

        return base_description + context
