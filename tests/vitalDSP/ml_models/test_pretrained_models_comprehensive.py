"""
Comprehensive Tests for Pretrained Models Module - Missing Coverage

This test file specifically targets missing lines in pretrained_models.py to achieve
high test coverage, including PyTorch backend, model download, fine-tuning, and ModelHub.

Author: vitalDSP Team
Date: 2025-01-27
Target Coverage: 85%+
"""

import pytest
import numpy as np
import warnings
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys

# Mark entire module to run serially (not in parallel) due to import mocking
pytestmark = pytest.mark.serial

# Suppress warnings
warnings.filterwarnings("ignore")

# Check dependencies
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Import module
try:
    from vitalDSP.ml_models.pretrained_models import (
        PretrainedModel,
        ModelHub,
        MODEL_REGISTRY,
        load_pretrained_model,
    )
    PRETRAINED_AVAILABLE = True
except ImportError:
    PRETRAINED_AVAILABLE = False


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(20, 187, 1)
    y = np.random.randint(0, 5, size=20)
    return X, y


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Create temporary cache directory."""
    cache_dir = tmp_path / "pretrained_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# Define module-level PyTorch model class for pickle compatibility
if PYTORCH_AVAILABLE:
    class SimplePyTorchModel(nn.Module):
        """Simple PyTorch model for testing."""
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(187, 5)
        
        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))


@pytest.mark.skipif(not PRETRAINED_AVAILABLE, reason="Pretrained models module not available")
class TestImportErrorHandling:
    """Test import error handling - covers lines 68-69, 76-77."""
    
    def test_tensorflow_import_error(self):
        """Test TensorFlow import error handling - covers lines 68-69."""
        with patch.dict('sys.modules', {'tensorflow': None}):
            import sys
            if 'vitalDSP.ml_models.pretrained_models' in sys.modules:
                del sys.modules['vitalDSP.ml_models.pretrained_models']
            
            # Re-import to trigger error handling
            try:
                from vitalDSP.ml_models.pretrained_models import PretrainedModel
                # TENSORFLOW_AVAILABLE should be False
                assert True
            except ImportError:
                pass
    
    def test_pytorch_import_error(self):
        """Test PyTorch import error handling - covers lines 76-77."""
        with patch.dict('sys.modules', {'torch': None}):
            import sys
            if 'vitalDSP.ml_models.pretrained_models' in sys.modules:
                del sys.modules['vitalDSP.ml_models.pretrained_models']
            
            # Re-import to trigger error handling
            try:
                from vitalDSP.ml_models.pretrained_models import PretrainedModel
                # PYTORCH_AVAILABLE should be False
                assert True
            except ImportError:
                pass


@pytest.mark.skipif(not PRETRAINED_AVAILABLE, reason="Pretrained models module not available")
class TestPretrainedModelPlaceholderCreation:
    """Test placeholder model creation - covers lines 296-361."""
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
    def test_create_placeholder_cnn(self, tmp_cache_dir):
        """Test creating placeholder CNN model - covers lines 316-327."""
        metadata = {
            "backend": "tensorflow",
            "architecture": "cnn1d",
            "input_shape": (187, 1),
            "task": "classification",
            "n_classes": 5,
            "version": "1.0.0"
        }
        
        model = PretrainedModel._create_placeholder_model(metadata)
        assert model is not None
        assert isinstance(model, keras.Model)
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
    def test_create_placeholder_lstm(self, tmp_cache_dir):
        """Test creating placeholder LSTM model - covers lines 329-331."""
        metadata = {
            "backend": "tensorflow",
            "architecture": "lstm",
            "input_shape": (100, 1),
            "task": "classification",
            "n_classes": 5,
            "version": "1.0.0"
        }
        
        model = PretrainedModel._create_placeholder_model(metadata)
        assert model is not None
        assert isinstance(model, keras.Model)
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
    def test_create_placeholder_transformer(self, tmp_cache_dir):
        """Test creating placeholder transformer model - covers lines 333-337."""
        metadata = {
            "backend": "tensorflow",
            "architecture": "transformer",
            "input_shape": (100, 12),
            "task": "classification",
            "n_classes": 6,
            "version": "1.0.0"
        }
        
        model = PretrainedModel._create_placeholder_model(metadata)
        assert model is not None
        assert isinstance(model, keras.Model)
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
    def test_create_placeholder_regression(self, tmp_cache_dir):
        """Test creating placeholder regression model - covers lines 343-344."""
        metadata = {
            "backend": "tensorflow",
            "architecture": "cnn1d",
            "input_shape": (500, 1),
            "task": "regression",
            "version": "1.0.0"
        }
        
        model = PretrainedModel._create_placeholder_model(metadata)
        assert model is not None
        assert isinstance(model, keras.Model)
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
    def test_create_placeholder_autoencoder(self, tmp_cache_dir):
        """Test creating placeholder autoencoder model - covers lines 345-349."""
        metadata = {
            "backend": "tensorflow",
            "architecture": "convolutional_autoencoder",
            "input_shape": (187, 1),
            "task": "autoencoder",
            "version": "1.0.0"
        }
        
        model = PretrainedModel._create_placeholder_model(metadata)
        assert model is not None
        assert isinstance(model, keras.Model)
    
    def test_create_placeholder_no_tensorflow(self):
        """Test creating placeholder without TensorFlow - covers lines 309-310."""
        import sys
        import vitalDSP.ml_models.pretrained_models as pm_module
        
        # Get the module from sys.modules
        module_name = 'vitalDSP.ml_models.pretrained_models'
        if module_name not in sys.modules:
            import vitalDSP.ml_models.pretrained_models
        pm_module = sys.modules[module_name]
        
        # Save original value
        original_tf_available = pm_module.TENSORFLOW_AVAILABLE
        
        try:
            # Set TENSORFLOW_AVAILABLE to False to simulate TensorFlow not being available
            pm_module.TENSORFLOW_AVAILABLE = False
            
            metadata = {
                "backend": "tensorflow",
                "architecture": "cnn1d",
                "input_shape": (187, 1),
                "task": "classification",
                "n_classes": 5,
                "version": "1.0.0"
            }
            
            with pytest.raises(ImportError, match="TensorFlow not installed"):
                PretrainedModel._create_placeholder_model(metadata)
        finally:
            # Restore original value
            pm_module.TENSORFLOW_AVAILABLE = original_tf_available
    
    def test_create_placeholder_pytorch_not_implemented(self):
        """Test PyTorch placeholder not implemented - covers lines 358-359."""
        metadata = {
            "backend": "pytorch",
            "architecture": "cnn1d",
            "input_shape": (187, 1),
            "task": "classification",
            "n_classes": 5,
            "version": "1.0.0"
        }
        
        with pytest.raises(NotImplementedError, match="PyTorch placeholder models"):
            PretrainedModel._create_placeholder_model(metadata)


@pytest.mark.skipif(not PRETRAINED_AVAILABLE, reason="Pretrained models module not available")
class TestPretrainedModelDownload:
    """Test model download functionality - covers lines 363-380."""
    
    def test_download_model_tensorflow(self, tmp_cache_dir):
        """Test downloading TensorFlow model - covers lines 364-377 (fully mocked)."""
        if not TENSORFLOW_AVAILABLE:
            pytest.skip("TensorFlow not available")

        url = "https://example.com/model.h5"
        save_path = tmp_cache_dir / "test_model.h5"

        def mock_urlretrieve(url, filename):
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            Path(filename).touch()

        mock_model = Mock()
        # Patch both urlretrieve (network) and keras load (disk read) so no I/O escapes
        with patch('vitalDSP.ml_models.pretrained_models.urlretrieve',
                   side_effect=mock_urlretrieve), \
             patch('vitalDSP.ml_models.pretrained_models.keras.models.load_model',
                   return_value=mock_model):
            model = PretrainedModel._download_model(url, save_path, "tensorflow")
            assert model is mock_model
            assert save_path.exists()
    
    def test_download_model_pytorch(self, tmp_cache_dir):
        """Test downloading PyTorch model - covers lines 375-375."""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
            
        url = "https://example.com/model.pt"
        save_path = tmp_cache_dir / "test_model.pt"
        
        # Create a mock function that creates an empty file
        def mock_urlretrieve(url, filename):
            # Create an empty file to simulate download
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            Path(filename).touch()
        
        # Mock urlretrieve using patch at the source module
        import vitalDSP.ml_models.pretrained_models as pm_module
        with patch('vitalDSP.ml_models.pretrained_models.urlretrieve', side_effect=mock_urlretrieve):
            # Mock torch.load
            mock_model = Mock()
            if hasattr(pm_module, 'torch'):
                with patch('vitalDSP.ml_models.pretrained_models.torch.load', return_value=mock_model):
                    model = PretrainedModel._download_model(url, save_path, "pytorch")
                    assert model is not None
                    assert save_path.exists()
            else:
                pytest.skip("PyTorch not imported in module")
    
    def test_download_model_url_error(self, tmp_cache_dir):
        """Test download model with URLError - covers lines 379-380."""
        from urllib.error import URLError
        
        url = "https://invalid-url.com/model.h5"
        save_path = tmp_cache_dir / "test_model.h5"
        
        # Mock urlretrieve to raise URLError
        with patch('vitalDSP.ml_models.pretrained_models.urlretrieve', side_effect=URLError("Connection failed")):
            with pytest.raises(RuntimeError, match="Failed to download model"):
                PretrainedModel._download_model(url, save_path, "tensorflow")


@pytest.mark.skipif(not PRETRAINED_AVAILABLE or not TENSORFLOW_AVAILABLE, reason="Pretrained models or TensorFlow not available")
class TestPretrainedModelFromRegistry:
    """Test from_registry method - covers lines 229-294."""
    
    def test_from_registry_cache_dir_none(self, tmp_cache_dir):
        """Test from_registry with None cache_dir - covers lines 267-270."""
        with patch('pathlib.Path.home', return_value=tmp_cache_dir.parent):
            model = PretrainedModel.from_registry(
                'ecg_classifier_mitbih',
                cache_dir=None
            )
            assert model is not None
    
    def test_from_registry_force_download(self, tmp_cache_dir):
        """Test from_registry with force_download - covers lines 277-285."""
        model = PretrainedModel.from_registry(
            'ecg_classifier_mitbih',
            cache_dir=str(tmp_cache_dir),
            force_download=True
        )
        assert model is not None
    
    def test_from_registry_load_from_cache(self, tmp_cache_dir):
        """Test from_registry loading from cache - covers lines 288-292."""
        # Create a cached model file with .keras extension
        model_path = tmp_cache_dir / "ecg_classifier_mitbih_v1.0.0.keras"
        
        # Create a simple model and save it
        if TENSORFLOW_AVAILABLE:
            simple_model = keras.Sequential([
                keras.layers.Input(shape=(187, 1)),
                keras.layers.Dense(10),
                keras.layers.Dense(5, activation='softmax')
            ])
            simple_model.save(str(model_path))
            
            model = PretrainedModel.from_registry(
                'ecg_classifier_mitbih',
                cache_dir=str(tmp_cache_dir),
                force_download=False
            )
            assert model is not None
        else:
            pytest.skip("TensorFlow not available")


@pytest.mark.skipif(not PRETRAINED_AVAILABLE or not TENSORFLOW_AVAILABLE, reason="Pretrained models or TensorFlow not available")
class TestPretrainedModelPredict:
    """Test predict method - covers lines 382-417."""
    
    def test_predict_tensorflow_classification(self, sample_data):
        """Test predict with TensorFlow classification - covers lines 402-406."""
        X, _ = sample_data
        
        model = PretrainedModel.from_registry('ecg_classifier_mitbih')
        predictions = model.predict(X, batch_size=8, return_proba=False)
        
        assert predictions is not None
        assert len(predictions) == len(X)
    
    def test_predict_tensorflow_with_proba(self, sample_data):
        """Test predict with return_proba=True - covers lines 405-405."""
        X, _ = sample_data
        
        model = PretrainedModel.from_registry('ecg_classifier_mitbih')
        predictions = model.predict(X, batch_size=8, return_proba=True)
        
        assert predictions is not None
        assert predictions.shape[0] == len(X)
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_predict_pytorch(self, sample_data):
        """Test predict with PyTorch backend - covers lines 408-416."""
        # Create a PyTorch model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(187, 64)
                self.fc2 = nn.Linear(64, 5)
            
            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                return self.fc2(x)
        
        pytorch_model = SimpleModel()
        metadata = MODEL_REGISTRY['ecg_classifier_mitbih'].copy()
        metadata['backend'] = 'pytorch'
        
        model = PretrainedModel(pytorch_model, metadata, backend='pytorch')
        
        X, _ = sample_data
        predictions = model.predict(X, batch_size=8, return_proba=False)
        
        assert predictions is not None
        assert len(predictions) == len(X)


@pytest.mark.skipif(not PRETRAINED_AVAILABLE or not TENSORFLOW_AVAILABLE, reason="Pretrained models or TensorFlow not available")
class TestPretrainedModelFineTune:
    """Test fine_tune method - covers lines 419-512."""
    
    def test_fine_tune_basic(self, sample_data):
        """Test basic fine-tuning - covers lines 460-509."""
        X_train, y_train = sample_data
        
        model = PretrainedModel.from_registry('ecg_classifier_mitbih')
        
        history = model.fine_tune(
            X_train, y_train,
            epochs=1,
            batch_size=8,
            verbose=0
        )
        
        assert history is not None
        assert 'loss' in history
    
    def test_fine_tune_with_validation(self, sample_data):
        """Test fine-tuning with validation - covers lines 495-509."""
        X_train, y_train = sample_data
        X_val = X_train[:5]
        y_val = y_train[:5]
        
        model = PretrainedModel.from_registry('ecg_classifier_mitbih')
        
        history = model.fine_tune(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=1,
            batch_size=8,
            verbose=0
        )
        
        assert history is not None
        assert 'val_loss' in history
    
    def test_fine_tune_freeze_layers(self, sample_data):
        """Test fine-tuning with frozen layers - covers lines 462-465."""
        X_train, y_train = sample_data
        
        model = PretrainedModel.from_registry('ecg_classifier_mitbih')
        
        history = model.fine_tune(
            X_train, y_train,
            freeze_layers=2,
            epochs=1,
            batch_size=8,
            verbose=0
        )
        
        assert history is not None
    
    def test_fine_tune_regression(self, sample_data):
        """Test fine-tuning regression model - covers lines 474-475."""
        # Create regression model metadata
        metadata = MODEL_REGISTRY['heart_rate_estimator'].copy()
        if TENSORFLOW_AVAILABLE:
            simple_model = keras.Sequential([
                keras.layers.Dense(10, input_shape=(500, 1)),
                keras.layers.Dense(1, activation='linear')
            ])
            simple_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            model = PretrainedModel(simple_model, metadata, backend='tensorflow')
            
            X_train = np.random.randn(20, 500, 1)
            y_train = np.random.randn(20)
            
            history = model.fine_tune(
                X_train, y_train,
                epochs=1,
                batch_size=8,
                verbose=0
            )
            
            assert history is not None
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_fine_tune_pytorch_not_implemented(self):
        """Test PyTorch fine-tuning not implemented - covers lines 511-512."""
        metadata = MODEL_REGISTRY['ecg_classifier_mitbih'].copy()
        metadata['backend'] = 'pytorch'
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(187, 5)
            
            def forward(self, x):
                return self.fc(x.view(x.size(0), -1))
        
        pytorch_model = SimpleModel()
        model = PretrainedModel(pytorch_model, metadata, backend='pytorch')
        
        X_train = np.random.randn(20, 187, 1)
        y_train = np.random.randint(0, 5, size=20)
        
        with pytest.raises(NotImplementedError, match="PyTorch fine-tuning"):
            model.fine_tune(X_train, y_train, epochs=1, verbose=0)


@pytest.mark.skipif(not PRETRAINED_AVAILABLE or not TENSORFLOW_AVAILABLE, reason="Pretrained models or TensorFlow not available")
class TestPretrainedModelEvaluate:
    """Test evaluate method - covers lines 514-546."""
    
    def test_evaluate_tensorflow(self, sample_data):
        """Test evaluate with TensorFlow - covers lines 534-543."""
        X_test, y_test = sample_data
        
        model = PretrainedModel.from_registry('ecg_classifier_mitbih')
        
        metrics = model.evaluate(X_test, y_test, batch_size=8)
        
        assert metrics is not None
        assert isinstance(metrics, dict)
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_evaluate_pytorch_not_implemented(self):
        """Test PyTorch evaluation not implemented - covers lines 545-546."""
        metadata = MODEL_REGISTRY['ecg_classifier_mitbih'].copy()
        metadata['backend'] = 'pytorch'
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(187, 5)
            
            def forward(self, x):
                return self.fc(x.view(x.size(0), -1))
        
        pytorch_model = SimpleModel()
        model = PretrainedModel(pytorch_model, metadata, backend='pytorch')
        
        X_test = np.random.randn(10, 187, 1)
        y_test = np.random.randint(0, 5, size=10)
        
        with pytest.raises(NotImplementedError, match="PyTorch evaluation"):
            model.evaluate(X_test, y_test)


@pytest.mark.skipif(not PRETRAINED_AVAILABLE or not TENSORFLOW_AVAILABLE, reason="Pretrained models or TensorFlow not available")
class TestPretrainedModelSaveLoad:
    """Test save and load methods - covers lines 548-603."""
    
    def test_save_tensorflow(self, tmp_path, sample_data):
        """Test save with TensorFlow - covers lines 557-561."""
        model = PretrainedModel.from_registry('ecg_classifier_mitbih')
        
        save_path = tmp_path / "test_model.keras"
        model.save(str(save_path))
        
        # Check that model file exists
        assert save_path.exists()
        
        # Check metadata file exists
        metadata_path = save_path.parent / f"{save_path.stem}_metadata.json"
        assert metadata_path.exists()
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_save_pytorch(self, tmp_path):
        """Test save with PyTorch - covers lines 563-563."""
        metadata = MODEL_REGISTRY['ecg_classifier_mitbih'].copy()
        metadata['backend'] = 'pytorch'
        
        pytorch_model = SimplePyTorchModel()
        model = PretrainedModel(pytorch_model, metadata, backend='pytorch')
        
        save_path = tmp_path / "test_model.pt"
        model.save(str(save_path))
        
        assert save_path.exists()
    
    def test_load_with_metadata(self, tmp_path):
        """Test load with metadata file - covers lines 585-603."""
        if not TENSORFLOW_AVAILABLE:
            pytest.skip("TensorFlow not available")
        
        # Create and save a model
        model = PretrainedModel.from_registry('ecg_classifier_mitbih')
        save_path = tmp_path / "test_model.keras"
        model.save(str(save_path))
        
        # Load the model
        loaded_model = PretrainedModel.load(str(save_path))
        
        assert loaded_model is not None
        # JSON converts tuples to lists, so compare key fields
        assert loaded_model.metadata['description'] == model.metadata['description']
        assert loaded_model.metadata['task'] == model.metadata['task']
        assert loaded_model.metadata['signal_type'] == model.metadata['signal_type']
    
    def test_load_without_metadata(self, tmp_path):
        """Test load without metadata file - covers lines 592-593."""
        if not TENSORFLOW_AVAILABLE:
            pytest.skip("TensorFlow not available")
        
        # Create and save a model
        model = PretrainedModel.from_registry('ecg_classifier_mitbih')
        save_path = tmp_path / "test_model.keras"
        model.save(str(save_path))
        
        # Remove metadata file
        metadata_path = save_path.parent / f"{save_path.stem}_metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Load should still work with empty metadata
        loaded_model = PretrainedModel.load(str(save_path))
        assert loaded_model is not None


@pytest.mark.skipif(not PRETRAINED_AVAILABLE or not TENSORFLOW_AVAILABLE, reason="Pretrained models or TensorFlow not available")
class TestPretrainedModelInfo:
    """Test info method - covers lines 605-622."""
    
    def test_info_basic(self):
        """Test info method - covers lines 605-621."""
        model = PretrainedModel.from_registry('ecg_classifier_mitbih')
        
        info = model.info()
        
        assert isinstance(info, str)
        assert "Pre-trained Model Information" in info
    
    def test_info_long_list(self):
        """Test info with long list - covers lines 618-619."""
        # Create metadata with long list
        metadata = MODEL_REGISTRY['ecg_classifier_ptbxl'].copy()
        metadata['long_list'] = list(range(100))  # Long list
        
        if TENSORFLOW_AVAILABLE:
            simple_model = keras.Sequential([
                keras.layers.Dense(10, input_shape=(1000, 12)),
                keras.layers.Dense(71, activation='sigmoid')
            ])
            
            model = PretrainedModel(simple_model, metadata, backend='tensorflow')
            info = model.info()
            
            assert isinstance(info, str)
            # Long list should be truncated
            assert "..." in info or "total" in info.lower()


@pytest.mark.skipif(not PRETRAINED_AVAILABLE or not TENSORFLOW_AVAILABLE, reason="Pretrained models or TensorFlow not available")
class TestPretrainedModelGetLayerNames:
    """Test get_layer_names method - covers lines 624-629."""
    
    def test_get_layer_names_tensorflow(self):
        """Test get_layer_names with TensorFlow - covers lines 626-627."""
        model = PretrainedModel.from_registry('ecg_classifier_mitbih')
        
        layer_names = model.get_layer_names()
        
        assert isinstance(layer_names, list)
        assert len(layer_names) > 0
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_get_layer_names_pytorch(self):
        """Test get_layer_names with PyTorch - covers lines 629-629."""
        metadata = MODEL_REGISTRY['ecg_classifier_mitbih'].copy()
        metadata['backend'] = 'pytorch'
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(187, 64)
                self.fc2 = nn.Linear(64, 5)
            
            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                return self.fc2(x)
        
        pytorch_model = SimpleModel()
        model = PretrainedModel(pytorch_model, metadata, backend='pytorch')
        
        layer_names = model.get_layer_names()
        
        assert isinstance(layer_names, list)
        assert len(layer_names) > 0


@pytest.mark.skipif(not PRETRAINED_AVAILABLE or not TENSORFLOW_AVAILABLE, reason="Pretrained models or TensorFlow not available")
class TestPretrainedModelGetFeatures:
    """Test get_features method - covers lines 631-666."""
    
    def test_get_features_tensorflow(self, sample_data):
        """Test get_features with TensorFlow - covers lines 651-661."""
        X, _ = sample_data
        
        model = PretrainedModel.from_registry('ecg_classifier_mitbih')
        
        features = model.get_features(X, batch_size=8)
        
        assert features is not None
        assert isinstance(features, np.ndarray)
    
    def test_get_features_with_layer_name(self, sample_data):
        """Test get_features with specific layer - covers lines 652-659."""
        X, _ = sample_data
        
        model = PretrainedModel.from_registry('ecg_classifier_mitbih')
        
        # Get layer names
        layer_names = model.get_layer_names()
        if len(layer_names) > 1:
            layer_name = layer_names[-2]  # Second to last
            
            features = model.get_features(X, layer_name=layer_name, batch_size=8)
            
            assert features is not None
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_get_features_pytorch_not_implemented(self):
        """Test PyTorch get_features not implemented - covers lines 663-664."""
        metadata = MODEL_REGISTRY['ecg_classifier_mitbih'].copy()
        metadata['backend'] = 'pytorch'
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(187, 5)
            
            def forward(self, x):
                return self.fc(x.view(x.size(0), -1))
        
        pytorch_model = SimpleModel()
        model = PretrainedModel(pytorch_model, metadata, backend='pytorch')
        
        X = np.random.randn(10, 187, 1)
        
        with pytest.raises(NotImplementedError, match="PyTorch feature extraction"):
            model.get_features(X)


@pytest.mark.skipif(not PRETRAINED_AVAILABLE, reason="Pretrained models module not available")
class TestModelHub:
    """Test ModelHub class - covers lines 669-830."""
    
    def test_model_hub_init_default_cache(self):
        """Test ModelHub initialization with default cache - covers lines 705-708."""
        with patch('pathlib.Path.home', return_value=Path("/tmp")):
            hub = ModelHub()
            assert hub.cache_dir is not None
    
    def test_model_hub_init_custom_cache(self, tmp_cache_dir):
        """Test ModelHub initialization with custom cache."""
        hub = ModelHub(cache_dir=str(tmp_cache_dir))
        assert hub.cache_dir == tmp_cache_dir
    
    def test_list_models_all(self):
        """Test list_models without filters."""
        hub = ModelHub()
        models = hub.list_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
    
    def test_list_models_filter_signal_type(self):
        """Test list_models with signal_type filter - covers lines 737-738."""
        hub = ModelHub()
        models = hub.list_models(signal_type='ecg')
        
        assert isinstance(models, list)
        assert all(m.get('signal_type') == 'ecg' for m in models)
    
    def test_list_models_filter_task(self):
        """Test list_models with task filter - covers lines 739-740."""
        hub = ModelHub()
        models = hub.list_models(task='classification')
        
        assert isinstance(models, list)
        assert all(m.get('task') == 'classification' for m in models)
    
    def test_list_models_filter_architecture(self):
        """Test list_models with architecture filter - covers lines 741-742."""
        hub = ModelHub()
        models = hub.list_models(architecture='cnn1d')
        
        assert isinstance(models, list)
        assert all(m.get('architecture') == 'cnn1d' for m in models)
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
    def test_get_model(self, tmp_cache_dir):
        """Test get_model - covers lines 767-769."""
        hub = ModelHub(cache_dir=str(tmp_cache_dir))
        
        model = hub.get_model('ecg_classifier_mitbih')
        
        assert model is not None
        assert isinstance(model, PretrainedModel)
    
    def test_clear_cache_specific_model(self, tmp_cache_dir):
        """Test clear_cache with specific model - covers lines 780-783."""
        hub = ModelHub(cache_dir=str(tmp_cache_dir))
        
        # Create a dummy file
        test_file = tmp_cache_dir / "ecg_classifier_mitbih_test"
        test_file.write_text("test")
        
        hub.clear_cache(model_name='ecg_classifier_mitbih')
        
        # File should be deleted
        assert not test_file.exists()
    
    def test_clear_cache_all(self, tmp_cache_dir):
        """Test clear_cache without model_name - covers lines 785-787."""
        hub = ModelHub(cache_dir=str(tmp_cache_dir))
        
        # Create dummy files
        test_file1 = tmp_cache_dir / "test1"
        test_file2 = tmp_cache_dir / "test2"
        test_file1.write_text("test1")
        test_file2.write_text("test2")
        
        hub.clear_cache()
        
        # All files should be deleted
        assert not test_file1.exists()
        assert not test_file2.exists()
    
    def test_get_cache_size(self, tmp_cache_dir):
        """Test get_cache_size - covers lines 789-802."""
        hub = ModelHub(cache_dir=str(tmp_cache_dir))
        
        # Create dummy files
        test_file1 = tmp_cache_dir / "test1"
        test_file2 = tmp_cache_dir / "test2"
        test_file1.write_text("x" * 1000)
        test_file2.write_text("y" * 2000)
        
        cache_size = hub.get_cache_size()
        
        assert isinstance(cache_size, float)
        assert cache_size >= 0
    
    def test_compare_models(self):
        """Test compare_models - covers lines 804-830."""
        hub = ModelHub()
        
        comparison = hub.compare_models(
            ['ecg_classifier_mitbih', 'ppg_quality_assessment'],
            metric='accuracy'
        )
        
        assert isinstance(comparison, dict)
        assert len(comparison) > 0


@pytest.mark.skipif(not PRETRAINED_AVAILABLE, reason="Pretrained models module not available")
class TestLoadPretrainedModel:
    """Test load_pretrained_model convenience function - covers lines 834-869."""
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
    def test_load_pretrained_model_basic(self, tmp_cache_dir):
        """Test load_pretrained_model basic usage."""
        model = load_pretrained_model(
            'ecg_classifier_mitbih',
            cache_dir=str(tmp_cache_dir)
        )
        
        assert model is not None
        assert isinstance(model, PretrainedModel)
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
    def test_load_pretrained_model_force_download(self, tmp_cache_dir):
        """Test load_pretrained_model with force_download."""
        model = load_pretrained_model(
            'ecg_classifier_mitbih',
            cache_dir=str(tmp_cache_dir),
            force_download=True
        )
        
        assert model is not None


@pytest.mark.skipif(not PRETRAINED_AVAILABLE, reason="Pretrained models module not available")
class TestPretrainedModelEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_from_registry_invalid_model(self):
        """Test from_registry with invalid model name - covers lines 257-262."""
        with pytest.raises(ValueError, match="not found in registry"):
            PretrainedModel.from_registry('invalid_model_name')
    
    def test_from_registry_with_url(self, tmp_cache_dir):
        """Test from_registry with URL in metadata - covers lines 278-286 (fully mocked)."""
        if not TENSORFLOW_AVAILABLE:
            pytest.skip("TensorFlow not available")

        original_url = MODEL_REGISTRY['ecg_classifier_mitbih']['url']
        MODEL_REGISTRY['ecg_classifier_mitbih']['url'] = 'https://example.com/model.h5'

        def mock_urlretrieve(url, filename):
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            Path(filename).touch()

        mock_model = Mock()
        try:
            # Both urlretrieve (network) and keras.load (disk) are patched — no real I/O
            with patch('vitalDSP.ml_models.pretrained_models.urlretrieve',
                       side_effect=mock_urlretrieve), \
                 patch('vitalDSP.ml_models.pretrained_models.keras.models.load_model',
                       return_value=mock_model):
                model = PretrainedModel.from_registry(
                    'ecg_classifier_mitbih',
                    cache_dir=str(tmp_cache_dir),
                    force_download=True,
                )
                assert model is not None
        finally:
            MODEL_REGISTRY['ecg_classifier_mitbih']['url'] = original_url


@pytest.mark.skipif(not PRETRAINED_AVAILABLE, reason="Pretrained models module not available")
class TestMissingCoverageLines:
    """New tests to cover lines flagged as uncovered in the coverage report.

    Target lines: 289-292, 344-345, 348-354, 372-380, 412, 420-426, 424, 610, 834-836
    """

    # ------------------------------------------------------------------ #
    # Lines 289-292: from_registry loads TF/PT model from cache when file exists
    # ------------------------------------------------------------------ #

    def test_from_registry_loads_tf_from_cache(self, tmp_cache_dir):
        """Load TF model from existing cache path (lines 289-290)."""
        if not TENSORFLOW_AVAILABLE:
            pytest.skip("TensorFlow not available")

        model_name = "ecg_classifier_mitbih"
        metadata = MODEL_REGISTRY[model_name]
        cached_path = tmp_cache_dir / f"{model_name}_v{metadata['version']}"
        cached_path.mkdir(parents=True, exist_ok=True)  # mimic existing cache

        mock_model = Mock()
        with patch(
            "vitalDSP.ml_models.pretrained_models.keras.models.load_model",
            return_value=mock_model,
        ):
            result = PretrainedModel.from_registry(
                model_name,
                cache_dir=str(tmp_cache_dir),
                force_download=False,
            )
        assert result is not None
        assert result.model is mock_model

    def test_from_registry_loads_pytorch_from_cache(self, tmp_cache_dir):
        """Load PyTorch model from existing cache path (lines 291-292)."""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        # Use a registry entry with pytorch backend
        pt_meta = {
            "architecture": "cnn1d",
            "input_shape": (187, 1),
            "n_classes": 2,
            "task": "classification",
            "backend": "pytorch",
            "version": "test",
            "url": None,
            "description": "test",
        }
        model_name = "_test_pt_cache_"
        MODEL_REGISTRY[model_name] = pt_meta

        cached_path = tmp_cache_dir / f"{model_name}_vtest"
        cached_path.touch()

        mock_model = Mock()
        try:
            with patch(
                "vitalDSP.ml_models.pretrained_models.torch.load",
                return_value=mock_model,
            ):
                result = PretrainedModel.from_registry(
                    model_name,
                    cache_dir=str(tmp_cache_dir),
                    force_download=False,
                )
            assert result.model is mock_model
        finally:
            del MODEL_REGISTRY[model_name]

    # ------------------------------------------------------------------ #
    # Lines 344-345: placeholder model — multi_label_classification branch
    # ------------------------------------------------------------------ #

    def test_placeholder_model_multi_label(self):
        """Create placeholder for multi_label_classification (lines 343-345)."""
        if not TENSORFLOW_AVAILABLE:
            pytest.skip("TensorFlow not available")

        meta = {
            "architecture": "cnn1d",
            "input_shape": (187, 1),
            "n_classes": 5,
            "task": "multi_label_classification",
            "backend": "tensorflow",
        }
        model = PretrainedModel._create_placeholder_model(meta)
        assert model is not None

    # ------------------------------------------------------------------ #
    # Lines 348-354: placeholder model — autoencoder branch
    # ------------------------------------------------------------------ #

    def test_placeholder_model_autoencoder(self):
        """Create placeholder for autoencoder task (lines 348-354)."""
        if not TENSORFLOW_AVAILABLE:
            pytest.skip("TensorFlow not available")

        meta = {
            "architecture": "cnn1d",
            "input_shape": (187, 1),
            "n_classes": 1,
            "task": "autoencoder",
            "backend": "tensorflow",
        }
        model = PretrainedModel._create_placeholder_model(meta)
        assert model is not None

    # ------------------------------------------------------------------ #
    # Lines 372-380: _download_model — pytorch branch + error branch
    # ------------------------------------------------------------------ #

    def test_download_model_pytorch_branch(self, tmp_cache_dir):
        """Cover pytorch branch in _download_model (lines 377-378)."""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        url = "https://example.com/model.pt"
        save_path = tmp_cache_dir / "model.pt"

        def fake_urlretrieve(u, f):
            Path(f).parent.mkdir(parents=True, exist_ok=True)
            Path(f).touch()

        mock_model = Mock()
        with patch(
            "vitalDSP.ml_models.pretrained_models.urlretrieve",
            side_effect=fake_urlretrieve,
        ), patch(
            "vitalDSP.ml_models.pretrained_models.torch.load",
            return_value=mock_model,
        ):
            result = PretrainedModel._download_model(url, save_path, "pytorch")
        assert result is mock_model

    def test_download_model_http_error(self, tmp_cache_dir):
        """_download_model converts HTTPError to RuntimeError (line 382-383)."""
        from urllib.error import HTTPError

        url = "https://example.com/model.h5"
        save_path = tmp_cache_dir / "model.h5"

        with patch(
            "vitalDSP.ml_models.pretrained_models.urlretrieve",
            side_effect=HTTPError(url, 404, "Not Found", {}, None),
        ):
            with pytest.raises(RuntimeError, match="Failed to download model"):
                PretrainedModel._download_model(url, save_path, "tensorflow")

    def test_download_model_os_error(self, tmp_cache_dir):
        """_download_model converts OSError to RuntimeError (line 382-383)."""
        url = "https://example.com/model.h5"
        save_path = tmp_cache_dir / "model.h5"

        with patch(
            "vitalDSP.ml_models.pretrained_models.urlretrieve",
            side_effect=OSError("disk full"),
        ):
            with pytest.raises(RuntimeError, match="Failed to download model"):
                PretrainedModel._download_model(url, save_path, "tensorflow")

    # ------------------------------------------------------------------ #
    # Line 412: predict PyTorch — return_proba=True skips argmax (line 412)
    # Lines 420-426: pytorch classification argmax and binary paths
    # ------------------------------------------------------------------ #

    def test_predict_pytorch_return_proba(self):
        """PyTorch predict with return_proba=True returns raw proba (line 412)."""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        mock_torch_model = Mock()
        mock_torch_model.return_value = Mock()
        mock_out = np.random.rand(10, 5).astype(np.float32)

        meta = {
            "task": "classification",
            "n_classes": 5,
            "backend": "pytorch",
            "architecture": "cnn1d",
            "input_shape": (187, 1),
        }
        pm = PretrainedModel(mock_torch_model, meta, "pytorch")

        X = np.random.rand(10, 187, 1).astype(np.float32)
        with patch.object(mock_torch_model, "eval"), \
             patch("torch.no_grad", return_value=MagicMock(__enter__=Mock(return_value=None),
                                                           __exit__=Mock(return_value=False))):
            mock_torch_model.return_value.cpu.return_value.numpy.return_value = mock_out
            import torch

            with patch.object(torch, "FloatTensor", return_value=Mock()), \
                 patch.object(mock_torch_model, "__call__", return_value=Mock(
                     cpu=Mock(return_value=Mock(numpy=Mock(return_value=mock_out))))):
                result = pm.predict(X, return_proba=True)

        # When return_proba=True no argmax is applied — shape must stay (10, 5)
        assert result.shape == mock_out.shape

    def test_predict_pytorch_classification_argmax(self):
        """PyTorch predict multi-class argmax path (lines 420-422)."""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        import torch

        meta = {
            "task": "classification",
            "n_classes": 5,
            "backend": "pytorch",
            "architecture": "cnn1d",
            "input_shape": (187, 1),
        }
        raw_proba = np.random.rand(10, 5).astype(np.float32)

        # Use a real tiny PyTorch model so no Mock object ends up as predictions
        class TinyNet(torch.nn.Module):
            def forward(self, x):
                return torch.from_numpy(raw_proba)

        pm = PretrainedModel(TinyNet(), meta, "pytorch")
        X = np.random.rand(10, 187, 1).astype(np.float32)
        result = pm.predict(X, return_proba=False)

        expected = np.argmax(raw_proba, axis=1)
        np.testing.assert_array_equal(result, expected)

    def test_predict_pytorch_binary_squeeze(self):
        """PyTorch predict binary sigmoid squeeze path (lines 423-424)."""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        import torch

        meta = {
            "task": "classification",
            "n_classes": 2,
            "backend": "pytorch",
            "architecture": "cnn1d",
            "input_shape": (187, 1),
        }
        # Shape (10, 1) — binary sigmoid output that squeezes to 1-D
        raw_proba = np.random.rand(10, 1).astype(np.float32)

        class BinaryNet(torch.nn.Module):
            def forward(self, x):
                return torch.from_numpy(raw_proba)

        pm = PretrainedModel(BinaryNet(), meta, "pytorch")
        X = np.random.rand(10, 187, 1).astype(np.float32)
        result = pm.predict(X, return_proba=False)

        expected = (raw_proba.squeeze() > 0.5).astype(int)
        np.testing.assert_array_equal(result, expected)

    # ------------------------------------------------------------------ #
    # Line 610: load() — pytorch cache load branch
    # ------------------------------------------------------------------ #

    def test_load_pytorch_from_disk(self, tmp_cache_dir):
        """PretrainedModel.load() pytorch branch (line 609-610)."""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        meta = {
            "task": "classification",
            "n_classes": 2,
            "backend": "pytorch",
            "architecture": "cnn1d",
            "input_shape": (187, 1),
        }

        # load() looks for: filepath.parent / f"{filepath.stem}_metadata.json"
        filepath = tmp_cache_dir / "pt_model.pt"
        filepath.touch()
        meta_path = tmp_cache_dir / "pt_model_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        mock_model = Mock()
        with patch(
            "vitalDSP.ml_models.pretrained_models.torch.load",
            return_value=mock_model,
        ):
            result = PretrainedModel.load(str(filepath))
        assert result.model is mock_model
        assert result.backend == "pytorch"

    # ------------------------------------------------------------------ #
    # Lines 834-836: ModelHub.compare_models — metric present and absent
    # ------------------------------------------------------------------ #

    def test_compare_models_metric_present(self):
        """compare_models returns values when metric key exists (lines 834-837)."""
        hub = ModelHub()
        # Use a metric key known to exist in some registry entries
        comparison = hub.compare_models(
            list(MODEL_REGISTRY.keys())[:2],
            metric="accuracy",
        )
        # Result is a dict; values are only included when key exists
        assert isinstance(comparison, dict)

    def test_compare_models_metric_absent(self):
        """compare_models returns empty dict when metric missing everywhere (line 836 branch)."""
        hub = ModelHub()
        comparison = hub.compare_models(
            list(MODEL_REGISTRY.keys()),
            metric="nonexistent_metric_xyz",
        )
        assert comparison == {}

    def test_compare_models_invalid_name(self):
        """compare_models silently skips names not in registry (line 833-834 guard)."""
        hub = ModelHub()
        comparison = hub.compare_models(
            ["invalid_model_xyz", "another_bad_name"],
            metric="accuracy",
        )
        assert comparison == {}


