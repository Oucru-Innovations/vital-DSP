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

        # Interpretation based on value
        if value < normal_range[0]:
            interpretation = feature_info.get("interpretation", {}).get(
                "below_range", "Value is below the normal range."
            )
        elif value > normal_range[1]:
            interpretation = feature_info.get("interpretation", {}).get(
                "above_range", "Value is above the normal range."
            )
        else:
            interpretation = feature_info.get("interpretation", {}).get(
                "in_range", "Value is within the normal range."
            )

        return {
            "description": feature_info.get("description", "No description available."),
            "normal_range": normal_range,
            "interpretation": interpretation,
            "contradiction": self._get_feature_contradiction(feature_name),
            "correlation": self._get_feature_correlation(feature_name),
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
        return correlation_info

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
                interpretations[feature_name] = self.interpret_feature(feature_name, value, segment_duration)
            
            # Analyze patterns across features
            patterns = self._analyze_cross_feature_patterns(feature_data, segment_duration)
            
            # Generate warnings based on combinations
            warnings = self._generate_cross_feature_warnings(feature_data, segment_duration)
            
            return {
                'individual_interpretations': interpretations,
                'cross_feature_patterns': patterns,
                'warnings': warnings,
                'overall_assessment': self._generate_overall_assessment(interpretations)
            }
            
        except Exception as e:
            return {
                'individual_interpretations': {},
                'cross_feature_patterns': [],
                'warnings': [f"Error in cross-feature analysis: {str(e)}"],
                'overall_assessment': "Unable to complete cross-feature analysis"
            }

    def _analyze_cross_feature_patterns(self, feature_data, segment_duration):
        """Analyze patterns across multiple features."""
        patterns = []
        
        # Check for autonomic dysfunction patterns
        hrv_features = ['sdnn', 'rmssd', 'nn50', 'pnn50']
        hrv_values = {f: feature_data.get(f) for f in hrv_features if f in feature_data}
        
        if len(hrv_values) >= 2:
            out_of_range_hrv = sum(1 for f, v in hrv_values.items() 
                                 if self.get_range_status(f, v, segment_duration) != 'in_range')
            
            if out_of_range_hrv >= 2:
                patterns.append({
                    'type': 'autonomic_dysfunction',
                    'description': 'Multiple heart rate variability parameters are abnormal, suggesting potential autonomic dysfunction',
                    'severity': 'moderate' if out_of_range_hrv == 2 else 'high',
                    'affected_features': [f for f, v in hrv_values.items() 
                                       if self.get_range_status(f, v, segment_duration) != 'in_range']
                })
        
        # Check for cardiovascular risk patterns
        if 'sdnn' in feature_data and 'heart_rate' in feature_data:
            sdnn_status = self.get_range_status('sdnn', feature_data['sdnn'], segment_duration)
            hr_status = self.get_range_status('heart_rate', feature_data['heart_rate'], segment_duration)
            
            if sdnn_status == 'below_range' and hr_status == 'above_range':
                patterns.append({
                    'type': 'cardiovascular_risk',
                    'description': 'Low heart rate variability combined with elevated heart rate may indicate cardiovascular stress',
                    'severity': 'high',
                    'affected_features': ['sdnn', 'heart_rate']
                })
        
        return patterns

    def _generate_cross_feature_warnings(self, feature_data, segment_duration):
        """Generate warnings based on feature combinations."""
        warnings = []
        
        # Check for contradictory patterns
        if 'sdnn' in feature_data and 'rmssd' in feature_data:
            sdnn_status = self.get_range_status('sdnn', feature_data['sdnn'], segment_duration)
            rmssd_status = self.get_range_status('rmssd', feature_data['rmssd'], segment_duration)
            
            if sdnn_status != rmssd_status:
                warnings.append(f"Contradictory HRV patterns: SDNN is {sdnn_status} while RMSSD is {rmssd_status}")
        
        # Check for extreme values
        for feature_name, value in feature_data.items():
            if feature_name in self.config:
                normal_range = self.config[feature_name].get('normal_range', {}).get(segment_duration, [])
                if normal_range and len(normal_range) == 2:
                    min_val, max_val = normal_range
                    if value < min_val * 0.5:  # Extremely low
                        warnings.append(f"Extremely low {feature_name} value may indicate measurement error or critical condition")
                    elif value > max_val * 2:  # Extremely high
                        warnings.append(f"Extremely high {feature_name} value may indicate measurement error or critical condition")
        
        return warnings

    def _generate_overall_assessment(self, interpretations):
        """Generate overall assessment based on individual interpretations."""
        if not interpretations:
            return "No features available for assessment"
        
        in_range_count = sum(1 for i in interpretations.values() 
                           if i.get('range_status') == 'in_range')
        total_count = len(interpretations)
        
        if in_range_count == total_count:
            return "All parameters are within normal ranges - excellent overall health"
        elif in_range_count >= total_count * 0.8:
            return "Most parameters are normal - good overall health with minor areas of concern"
        elif in_range_count >= total_count * 0.6:
            return "Mixed results - some parameters require attention"
        else:
            return "Multiple parameters are abnormal - comprehensive evaluation recommended"