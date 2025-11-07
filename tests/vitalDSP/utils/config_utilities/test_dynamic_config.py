"""
Comprehensive tests for Dynamic Configuration System.

Tests cover configuration loading, saving, environment optimization,
and all configuration classes.
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

try:
    from vitalDSP.utils.config_utilities.dynamic_config import (
        DynamicConfig,
        Environment,
        SystemResources,
        DataLoaderConfig,
        QualityScreenerConfig,
        ParallelPipelineConfig,
        DynamicConfigManager,
        get_config,
        set_config,
        reset_config,
        YAML_AVAILABLE,
    )
    DYNAMIC_CONFIG_AVAILABLE = True
except ImportError:
    DYNAMIC_CONFIG_AVAILABLE = False
    DynamicConfig = None


@pytest.mark.skipif(
    not DYNAMIC_CONFIG_AVAILABLE, reason="Dynamic config not available"
)
class TestYAMLImport:
    """Test YAML import handling."""

    def test_yaml_import_handling(self):
        """Test that YAML import is handled gracefully."""
        # Lines 20-21: YAML import exception handling
        from vitalDSP.utils.config_utilities import dynamic_config

        # YAML_AVAILABLE should be a boolean
        assert isinstance(dynamic_config.YAML_AVAILABLE, bool)


@pytest.mark.skipif(
    not DYNAMIC_CONFIG_AVAILABLE, reason="Dynamic config not available"
)
class TestEnvironmentOptimization:
    """Test environment-based optimization."""

    def test_production_optimization(self):
        """Test production environment optimization."""
        # Lines 299-304: Production optimizations
        config = DynamicConfig(environment=Environment.PRODUCTION)

        assert config.parallel_pipeline.max_workers_cap <= 8
        assert config.data_loader.memory_usage_ratio == 0.05
        assert config.quality_screener.parallel_processing["max_workers_factor"] == 0.5
        assert config.debug_mode is False

    def test_development_optimization(self):
        """Test development environment optimization."""
        # Lines 306-312: Development optimizations
        config = DynamicConfig(environment=Environment.DEVELOPMENT)

        assert config.parallel_pipeline.max_workers_cap <= 4
        assert config.data_loader.memory_usage_ratio == 0.15
        assert config.debug_mode is True

    def test_testing_optimization(self):
        """Test testing environment optimization."""
        # Lines 314-318: Testing optimizations
        config = DynamicConfig(environment=Environment.TESTING)

        assert config.parallel_pipeline.max_workers_cap == 2
        assert config.data_loader.memory_usage_ratio == 0.20
        assert config.quality_screener.parallel_processing["enable_by_default"] is False


@pytest.mark.skipif(
    not DYNAMIC_CONFIG_AVAILABLE, reason="Dynamic config not available"
)
class TestConfigurationValidation:
    """Test configuration validation."""

    def test_memory_limit_validation_min(self):
        """Test memory limit validation - minimum."""
        # Lines 332-334: Memory limit minimum validation
        config = DynamicConfig()
        # Force low memory to trigger minimum validation
        config.system_resources.available_memory_gb = 0.01  # Very low memory

        # Re-validate
        config._validate_configuration()

        assert (
            config.parallel_pipeline.memory_limit_mb
            >= config.parallel_pipeline.min_memory_limit_mb
        )

    def test_memory_limit_validation_max(self):
        """Test memory limit validation - maximum."""
        # Lines 340-342: Memory limit maximum validation
        config = DynamicConfig()
        # Force high memory to trigger maximum validation
        config.system_resources.available_memory_gb = 1000.0  # Very high memory

        # Re-validate
        config._validate_configuration()

        assert (
            config.parallel_pipeline.memory_limit_mb
            <= config.parallel_pipeline.max_memory_limit_mb
        )

    def test_chunk_size_validation_swap(self):
        """Test chunk size validation when min > max."""
        # Lines 346-349: Chunk size swap validation
        config = DynamicConfig()
        # Set invalid chunk sizes
        config.data_loader.min_chunk_size = 100000
        config.data_loader.max_chunk_size = 1000

        # Re-validate
        config._validate_configuration()

        assert config.data_loader.min_chunk_size <= config.data_loader.max_chunk_size

    def test_quality_threshold_validation_error(self):
        """Test quality threshold validation with invalid value."""
        # Lines 358-360: Quality threshold validation error
        config = DynamicConfig()
        # Set invalid threshold (outside 0-1 range, not SNR)
        config.quality_screener.quality_thresholds["generic"]["artifact_max_ratio"] = 1.5

        with pytest.raises(ValueError, match="Invalid threshold"):
            config._validate_configuration()


@pytest.mark.skipif(
    not DYNAMIC_CONFIG_AVAILABLE, reason="Dynamic config not available"
)
class TestOptimalChunkSize:
    """Test optimal chunk size calculation."""

    def test_get_optimal_chunk_size_with_sampling_rate(self):
        """Test optimal chunk size with sampling rate alignment."""
        # Lines 391-401: Sampling rate alignment
        config = DynamicConfig()
        chunk_size = config.get_optimal_chunk_size(file_size_mb=100.0, sampling_rate=100.0)

        assert isinstance(chunk_size, int)
        assert chunk_size >= config.data_loader.min_chunk_size
        assert chunk_size <= config.data_loader.max_chunk_size

    def test_get_optimal_chunk_size_zero_sampling_rate(self):
        """Test optimal chunk size with zero sampling rate."""
        # Line 391: sampling_rate > 0 check
        config = DynamicConfig()
        chunk_size = config.get_optimal_chunk_size(file_size_mb=100.0, sampling_rate=0.0)

        assert isinstance(chunk_size, int)
        assert chunk_size >= config.data_loader.min_chunk_size


@pytest.mark.skipif(
    not DYNAMIC_CONFIG_AVAILABLE, reason="Dynamic config not available"
)
class TestOptimalWorkerCount:
    """Test optimal worker count calculation."""

    def test_get_optimal_worker_count_memory_based(self):
        """Test worker count calculation based on memory."""
        # Lines 406-426: Worker count calculation
        config = DynamicConfig()
        workers = config.get_optimal_worker_count(task_count=100, data_size_mb=500.0)

        assert isinstance(workers, int)
        assert workers >= config.parallel_pipeline.min_workers

    def test_get_optimal_worker_count_task_based(self):
        """Test worker count calculation based on task count."""
        # Line 420: Task-based worker calculation
        config = DynamicConfig()
        workers = config.get_optimal_worker_count(task_count=5, data_size_mb=1000.0)

        assert isinstance(workers, int)
        assert workers <= 5  # Should be limited by task count

    def test_get_optimal_worker_count_zero_memory(self):
        """Test worker count with zero memory per worker."""
        # Lines 414-416: Zero memory per worker handling
        config = DynamicConfig()
        # Force zero memory per worker by patching the property using PropertyMock
        from unittest.mock import PropertyMock
        with patch.object(
            type(config.system_resources), "memory_per_worker_mb", new_callable=PropertyMock, return_value=0
        ):
            workers = config.get_optimal_worker_count(task_count=100, data_size_mb=500.0)
            assert isinstance(workers, int)
            assert workers >= config.parallel_pipeline.min_workers


@pytest.mark.skipif(
    not DYNAMIC_CONFIG_AVAILABLE, reason="Dynamic config not available"
)
class TestSaveConfig:
    """Test configuration saving."""

    def test_save_config_json(self):
        """Test saving configuration to JSON file."""
        # Lines 430-500: Save config to JSON
        config = DynamicConfig()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            config.save_config(temp_path)

            # Verify file was created and contains valid JSON
            assert os.path.exists(temp_path)
            with open(temp_path, "r") as f:
                saved_config = json.load(f)
                assert "environment" in saved_config
                assert "system_resources" in saved_config
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_config_yaml(self):
        """Test saving configuration to YAML file."""
        # Lines 494-497: Save config to YAML
        if not YAML_AVAILABLE:
            pytest.skip("YAML not available")

        config = DynamicConfig()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            config.save_config(temp_path)

            # Verify file was created
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_config_yaml_extension(self):
        """Test saving configuration to .yml file."""
        # Line 494: .yml extension handling
        if not YAML_AVAILABLE:
            pytest.skip("YAML not available")

        config = DynamicConfig()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            temp_path = f.name

        try:
            config.save_config(temp_path)
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


@pytest.mark.skipif(
    not DYNAMIC_CONFIG_AVAILABLE, reason="Dynamic config not available"
)
class TestLoadConfig:
    """Test configuration loading."""

    def test_load_config_json(self):
        """Test loading configuration from JSON file."""
        # Lines 507-617: Load config from JSON
        config = DynamicConfig()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
            json.dump(
                {
                    "environment": "development",
                    "system_resources": {
                        "cpu_count": 4,
                        "memory_gb": 8.0,
                        "available_memory_gb": 4.0,
                        "disk_space_gb": 100.0,
                    },
                    "data_loader": {},
                    "quality_screener": {},
                    "parallel_pipeline": {},
                    "global_settings": {"debug_mode": True, "log_level": "DEBUG"},
                },
                f,
            )

        try:
            loaded_config = DynamicConfig.load_config(temp_path)

            assert loaded_config.environment == Environment.DEVELOPMENT
            assert loaded_config.debug_mode is True
            assert loaded_config.log_level == "DEBUG"
            assert loaded_config.config_file_path == temp_path
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_config_yaml(self):
        """Test loading configuration from YAML file."""
        # Lines 509-513: Load config from YAML
        if not YAML_AVAILABLE:
            pytest.skip("YAML not available")

        import yaml

        config = DynamicConfig()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name
            yaml.dump(
                {
                    "environment": "testing",
                    "system_resources": {
                        "cpu_count": 2,
                        "memory_gb": 4.0,
                        "available_memory_gb": 2.0,
                        "disk_space_gb": 50.0,
                    },
                },
                f,
            )

        try:
            loaded_config = DynamicConfig.load_config(temp_path)
            assert loaded_config.environment == Environment.TESTING
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_config_with_environment_override(self):
        """Test loading config with environment override."""
        # Lines 522-525: Environment override
        config = DynamicConfig()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
            json.dump({"environment": "development"}, f)

        try:
            loaded_config = DynamicConfig.load_config(
                temp_path, environment=Environment.PRODUCTION
            )
            assert loaded_config.environment == Environment.PRODUCTION
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_config_partial_data(self):
        """Test loading config with partial data."""
        # Lines 528-541: Partial system resources
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
            json.dump({"system_resources": {"cpu_count": 8}}, f)

        try:
            loaded_config = DynamicConfig.load_config(temp_path)
            assert loaded_config.system_resources.cpu_count == 8
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


@pytest.mark.skipif(
    not DYNAMIC_CONFIG_AVAILABLE, reason="Dynamic config not available"
)
class TestFromEnvironment:
    """Test configuration from environment variables."""

    def test_from_environment_debug(self):
        """Test from_environment with debug env var."""
        # Lines 641-641: Debug mode from env
        with patch.dict(os.environ, {"VITALDSP_DEBUG": "true"}):
            config = DynamicConfig.from_environment()
            assert config.debug_mode is True

    def test_from_environment_log_level(self):
        """Test from_environment with log level env var."""
        # Lines 643-645: Log level from env
        with patch.dict(os.environ, {"VITALDSP_LOG_LEVEL": "DEBUG"}):
            config = DynamicConfig.from_environment()
            assert config.log_level == "DEBUG"

    def test_from_environment_max_workers(self):
        """Test from_environment with max workers env var."""
        # Lines 646-647: Max workers from env
        with patch.dict(os.environ, {"VITALDSP_MAX_WORKERS": "8"}, clear=False):
            # Force reload by importing the module again or ensure env is read fresh
            config = DynamicConfig.from_environment()
            # The environment variable should override the default DEVELOPMENT max_workers_cap
            assert config.parallel_pipeline.max_workers_cap == 8, \
                f"Expected 8, got {config.parallel_pipeline.max_workers_cap}. " \
                f"CPU count: {config.system_resources.cpu_count}, " \
                f"Env var: {os.getenv('VITALDSP_MAX_WORKERS')}"

    def test_from_environment_memory_limit(self):
        """Test from_environment with memory limit env var."""
        # Lines 649-652: Memory limit from env
        with patch.dict(os.environ, {"VITALDSP_MEMORY_LIMIT_MB": "512"}):
            config = DynamicConfig.from_environment()
            assert config.parallel_pipeline.max_memory_limit_mb == 512

    def test_from_environment_chunk_size(self):
        """Test from_environment with chunk size env var."""
        # Lines 654-657: Chunk size from env
        with patch.dict(os.environ, {"VITALDSP_CHUNK_SIZE": "5000"}):
            config = DynamicConfig.from_environment()
            assert config.parallel_pipeline.default_chunk_size == 5000

    def test_from_environment_quality_threshold(self):
        """Test from_environment with quality threshold env var."""
        # Lines 659-662: Quality threshold from env
        with patch.dict(os.environ, {"VITALDSP_QUALITY_THRESHOLD": "0.6"}):
            config = DynamicConfig.from_environment()
            assert config.parallel_pipeline.default_quality_threshold == 0.6


@pytest.mark.skipif(
    not DYNAMIC_CONFIG_AVAILABLE, reason="Dynamic config not available"
)
class TestGlobalConfigFunctions:
    """Test global configuration functions."""

    def test_get_config(self):
        """Test get_config function."""
        # Lines 674-679: Get config
        reset_config()
        config = get_config()
        assert isinstance(config, DynamicConfig)

    def test_set_config(self):
        """Test set_config function."""
        # Lines 682-685: Set config
        reset_config()
        new_config = DynamicConfig(environment=Environment.PRODUCTION)
        set_config(new_config)
        assert get_config().environment == Environment.PRODUCTION

    def test_reset_config(self):
        """Test reset_config function."""
        # Lines 688-691: Reset config
        config = get_config()
        reset_config()
        new_config = get_config()
        # Should create a new instance
        assert new_config is not None


@pytest.mark.skipif(
    not DYNAMIC_CONFIG_AVAILABLE, reason="Dynamic config not available"
)
class TestDynamicConfigManager:
    """Test DynamicConfigManager class."""

    def test_dynamic_config_manager_init(self):
        """Test DynamicConfigManager initialization."""
        # Lines 718-722: Manager initialization
        manager = DynamicConfigManager()
        assert hasattr(manager, "_config")

    def test_dynamic_config_manager_init_with_file(self):
        """Test DynamicConfigManager initialization with config file."""
        # Line 722: Manager init with file
        config = DynamicConfig()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
            config.save_config(temp_path)

        try:
            manager = DynamicConfigManager(config_file=temp_path)
            assert manager._config.config_file_path == temp_path
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_dynamic_config_manager_get(self):
        """Test DynamicConfigManager get method."""
        # Lines 724-743: Manager get method
        manager = DynamicConfigManager()
        # Test getting nested attribute
        cpu_count = manager.get("system_resources.cpu_count")
        assert isinstance(cpu_count, int)

    def test_dynamic_config_manager_get_dict(self):
        """Test DynamicConfigManager get with dictionary access."""
        # Lines 737-738: Dictionary access
        manager = DynamicConfigManager()
        # Test getting from dict-like structure
        result = manager.get("quality_screener.quality_thresholds.generic.snr_min_db")
        assert isinstance(result, (int, float))

    def test_dynamic_config_manager_get_default(self):
        """Test DynamicConfigManager get with default value."""
        # Line 740: Default value
        manager = DynamicConfigManager()
        result = manager.get("nonexistent.key", default="default_value")
        assert result == "default_value"

    def test_dynamic_config_manager_get_exception(self):
        """Test DynamicConfigManager get with exception handling."""
        # Lines 742-743: Exception handling
        manager = DynamicConfigManager()
        # Should return default on exception
        result = manager.get("invalid..path", default="default")
        assert result == "default"

    def test_dynamic_config_manager_set_user_preference(self):
        """Test DynamicConfigManager set_user_preference."""
        # Lines 745-750: Set user preference
        manager = DynamicConfigManager()
        manager.set_user_preference("test_key", "test_value")
        result = manager.get("test_key")
        assert result == "test_value"

    def test_dynamic_config_manager_get_statistics(self):
        """Test DynamicConfigManager get_statistics."""
        # Lines 752-759: Get statistics
        manager = DynamicConfigManager()
        stats = manager.get_statistics()
        assert "environment" in stats
        assert "cpu_count" in stats
        assert "memory_gb" in stats

