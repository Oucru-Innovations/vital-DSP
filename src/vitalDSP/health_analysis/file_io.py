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
            with open(output_path, "w") as file:
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
