import pytest
import os
import yaml
from unittest import mock
from vitalDSP.health_analysis.file_io import FileIO


@pytest.fixture
def yaml_content():
    return {
        "feature": {
            "normal_range": [10, 50],
            "interpretation": "This is a test interpretation.",
        }
    }


@pytest.fixture
def yaml_file(tmp_path, yaml_content):
    file_path = tmp_path / "test_config.yml"
    with open(file_path, "w") as f:
        yaml.dump(yaml_content, f)
    return file_path


@pytest.fixture
def html_content():
    return "<html><body>This is a test report</body></html>"


def test_load_yaml_success(yaml_file, yaml_content):
    # Test successful loading of a YAML file
    loaded_content = FileIO.load_yaml(yaml_file)
    assert (
        loaded_content == yaml_content
    ), "Loaded YAML content should match expected content."


def test_load_yaml_file_not_found():
    # Test loading a non-existent YAML file
    with pytest.raises(FileNotFoundError):
        FileIO.load_yaml("non_existent_file.yml")


def test_load_yaml_parsing_error(tmp_path):
    # Test loading a malformed YAML file
    bad_yaml_file = tmp_path / "bad_config.yml"
    with open(bad_yaml_file, "w") as f:
        f.write("bad: [unclosed list")

    with pytest.raises(ValueError):
        FileIO.load_yaml(bad_yaml_file)


def test_save_report_html_success(tmp_path, html_content):
    # Test successful saving of an HTML report
    output_path = tmp_path / "report.html"
    result = FileIO.save_report_html(html_content, output_path)

    assert result is True, "HTML report should be saved successfully."
    assert os.path.exists(output_path), "HTML file should exist at the specified path."

    with open(output_path, "r") as f:
        saved_content = f.read()
        assert (
            saved_content == html_content
        ), "Saved HTML content should match the expected content."


def test_save_report_html_failure(monkeypatch, html_content):
    # Simulate an error when saving the HTML file
    def mock_open(*args, **kwargs):
        raise IOError("Mocked file system error")

    monkeypatch.setattr("builtins.open", mock_open)

    with pytest.raises(IOError):
        FileIO.save_report_html(html_content, "invalid_path/report.html")


def test_ensure_directory_exists(tmp_path):
    # Test ensuring a directory exists
    directory = tmp_path / "new_directory"

    assert not os.path.exists(directory), "Directory should not exist initially."

    FileIO.ensure_directory_exists(directory)

    assert os.path.exists(directory), "Directory should be created if it doesn't exist."


def test_ensure_directory_exists_already_exists(tmp_path):
    # Test ensuring a directory exists when it already exists
    directory = tmp_path / "existing_directory"
    os.makedirs(directory)

    # Ensure no error is raised if the directory already exists
    FileIO.ensure_directory_exists(directory)

    assert os.path.exists(
        directory
    ), "Directory should still exist after calling ensure_directory_exists."
