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

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.health_analysis.file_io import FileIo
    >>> signal = np.random.randn(1000)
    >>> processor = FileIo(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""


import yaml
import os


class FileIO:
    """
    A utility class for handling file input/output operations.
    """

    @staticmethod
    def load_yaml(file_path):
        """
        Loads a YAML file from the given file path.

        Args:
            file_path (str): Path to the YAML file.

        Returns:
            dict: Parsed YAML file contents as a Python dictionary.

        Raises:
            FileNotFoundError: If the file is not found at the given path.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file not found at path: {file_path}")

        try:
            with open(file_path, "r") as file:
                return yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    @staticmethod
    def save_report_html(report_html, output_path):
        """
        Saves a generated HTML report to the specified output path.

        Args:
            report_html (str): HTML content of the report.
            output_path (str): Path where the HTML report should be saved.

        Returns:
            bool: True if the report was successfully saved, False otherwise.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(report_html)
            return True
        except Exception as e:
            raise IOError(f"Failed to save report: {e}")

    @staticmethod
    def ensure_directory_exists(directory_path):
        """
        Ensures that the specified directory exists. Creates it if it does not exist.

        Args:
            directory_path (str): The path to the directory.
        """
        os.makedirs(directory_path, exist_ok=True)
