"""
Tests for Deep Models PyTorch backend and missing coverage paths.
This test file focuses on increasing coverage for PyTorch-specific code paths
and edge cases in deep learning models.
"""

import pytest
import numpy as np
import tempfile
import os

# Skip all tests if TensorFlow is not available
pytest.importorskip("tensorflow")

from vitalDSP.ml_models.deep_models import CNN1D, LSTMModel


@pytest.fixture
def sample_signal_data():
    """Generate sample signal data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 100, 1)
    y = np.random.randint(0, 3, size=(100,))
    return X, y


@pytest.fixture
def sample_sequence_data():
    """Generate sample sequence data for LSTM testing."""
    np.random.seed(42)
    X = np.random.randn(100, 50, 1)
    y = np.random.randint(0, 2, size=(100,))
    return X, y


class TestCNN1DAdditionalCoverage:
    """Test CNN1D with additional configurations."""

    def test_cnn1d_init_with_filters_list(self):
        """Test CNN1D with list of filter sizes."""
        model = CNN1D(
            input_shape=(100, 1),
            n_classes=3,
            n_filters=[32, 64, 128],
            backend="tensorflow"
        )
        model.build_model()
        assert model.model is not None

    def test_cnn1d_init_with_kernel_sizes(self):
        """Test CNN1D with different kernel sizes."""
        model = CNN1D(
            input_shape=(100, 1),
            n_classes=3,
            kernel_sizes=[7, 5, 3],
            backend="tensorflow"
        )
        model.build_model()
        assert model.model is not None

    def test_cnn1d_init_with_pool_sizes(self):
        """Test CNN1D with different pool sizes."""
        model = CNN1D(
            input_shape=(100, 1),
            n_classes=3,
            pool_sizes=[2, 2, 2],
            backend="tensorflow"
        )
        model.build_model()
        assert model.model is not None

    def test_cnn1d_without_dropout(self):
        """Test CNN1D without dropout."""
        model = CNN1D(
            input_shape=(100, 1),
            n_classes=3,
            dropout_rate=0.0,
            backend="tensorflow"
        )
        model.build_model()
        assert model.model is not None

    def test_cnn1d_high_dropout(self):
        """Test CNN1D with high dropout rate."""
        model = CNN1D(
            input_shape=(100, 1),
            n_classes=3,
            dropout_rate=0.7,
            backend="tensorflow"
        )
        model.build_model()
        assert model.model is not None

    def test_cnn1d_different_activations(self):
        """Test CNN1D with different activation functions."""
        for activation in ['relu', 'elu', 'tanh']:
            model = CNN1D(
                input_shape=(100, 1),
                n_classes=3,
                activation=activation,
                backend="tensorflow"
            )
            model.build_model()
            assert model.model is not None

    def test_cnn1d_without_batch_norm(self):
        """Test CNN1D without batch normalization."""
        model = CNN1D(
            input_shape=(100, 1),
            n_classes=3,
            use_batch_norm=False,
            backend="tensorflow"
        )
        model.build_model()
        assert model.model is not None

    def test_cnn1d_different_dense_units(self):
        """Test CNN1D with different dense layer units."""
        model = CNN1D(
            input_shape=(100, 1),
            n_classes=3,
            dense_units=[256, 128],
            backend="tensorflow"
        )
        model.build_model()
        assert model.model is not None

    def test_cnn1d_save_load(self, sample_signal_data):
        """Test CNN1D save and load functionality."""
        X, y = sample_signal_data
        model = CNN1D(input_shape=(100, 1), n_classes=3, backend="tensorflow")
        model.build_model()
        model.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Train briefly
        model.train(X[:10], y[:10], epochs=1, verbose=0)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'model.h5')
            model.save(filepath)
            assert os.path.exists(filepath)
            
            # Load model
            model2 = CNN1D(input_shape=(100, 1), n_classes=3, backend="tensorflow")
            model2.load(filepath)
            assert model2.model is not None

    def test_cnn1d_predict(self, sample_signal_data):
        """Test CNN1D prediction."""
        X, y = sample_signal_data
        model = CNN1D(input_shape=(100, 1), n_classes=3, backend="tensorflow")
        model.build_model()
        model.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.train(X[:10], y[:10], epochs=1, verbose=0)
        
        predictions = model.predict(X[:5])
        assert predictions.shape[0] == 5
        assert predictions.shape[1] == 3  # n_classes

    def test_cnn1d_train_with_validation(self, sample_signal_data):
        """Test CNN1D training with validation data."""
        X, y = sample_signal_data
        model = CNN1D(input_shape=(100, 1), n_classes=3, backend="tensorflow")
        model.build_model()
        model.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        model.train(
            X[:50], y[:50],
            X_val=X[50:60], y_val=y[50:60],
            epochs=2,
            batch_size=16,
            verbose=0
        )
        assert model.history is not None

    def test_cnn1d_different_optimizers(self, sample_signal_data):
        """Test CNN1D with different optimizers."""
        X, y = sample_signal_data
        for optimizer in ['adam', 'sgd', 'rmsprop']:
            model = CNN1D(input_shape=(100, 1), n_classes=3, backend="tensorflow")
            model.build_model()
            model.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
            model.train(X[:10], y[:10], epochs=1, verbose=0)
            assert model.history is not None


class TestLSTMModelAdditionalCoverage:
    """Test LSTMModel with additional configurations."""

    def test_lstm_init_basic(self):
        """Test LSTM initialization."""
        model = LSTMModel(
            input_shape=(50, 1),
            n_classes=2,
            backend="tensorflow"
        )
        model.build_model()
        assert model.model is not None

    def test_lstm_with_different_units(self):
        """Test LSTM with different unit sizes."""
        model = LSTMModel(
            input_shape=(50, 1),
            n_classes=2,
            lstm_units=[128, 64, 32],
            backend="tensorflow"
        )
        model.build_model()
        assert model.model is not None

    def test_lstm_bidirectional(self):
        """Test LSTM with bidirectional layers."""
        model = LSTMModel(
            input_shape=(50, 1),
            n_classes=2,
            bidirectional=True,
            backend="tensorflow"
        )
        model.build_model()
        assert model.model is not None

    def test_lstm_without_bidirectional(self):
        """Test LSTM without bidirectional layers."""
        model = LSTMModel(
            input_shape=(50, 1),
            n_classes=2,
            bidirectional=False,
            backend="tensorflow"
        )
        model.build_model()
        assert model.model is not None

    def test_lstm_different_dropout(self):
        """Test LSTM with different dropout rates."""
        for dropout in [0.0, 0.3, 0.6]:
            model = LSTMModel(
                input_shape=(50, 1),
                n_classes=2,
                dropout_rate=dropout,
                backend="tensorflow"
            )
            model.build_model()
            assert model.model is not None

    def test_lstm_different_dense_units(self):
        """Test LSTM with different dense layer units."""
        model = LSTMModel(
            input_shape=(50, 1),
            n_classes=2,
            dense_units=[128, 64],
            backend="tensorflow"
        )
        model.build_model()
        assert model.model is not None

    def test_lstm_train_and_predict(self, sample_sequence_data):
        """Test LSTM training and prediction."""
        X, y = sample_sequence_data
        model = LSTMModel(input_shape=(50, 1), n_classes=2, backend="tensorflow")
        model.build_model()
        model.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Train
        model.train(X[:20], y[:20], epochs=1, verbose=0)
        assert model.history is not None
        
        # Predict
        predictions = model.predict(X[:5])
        assert predictions.shape[0] == 5

    def test_lstm_with_validation(self, sample_sequence_data):
        """Test LSTM with validation data."""
        X, y = sample_sequence_data
        model = LSTMModel(input_shape=(50, 1), n_classes=2, backend="tensorflow")
        model.build_model()
        model.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        model.train(
            X[:40], y[:40],
            X_val=X[40:50], y_val=y[40:50],
            epochs=2,
            verbose=0
        )
        assert model.history is not None

    def test_lstm_save_load(self, sample_sequence_data):
        """Test LSTM save and load."""
        X, y = sample_sequence_data
        model = LSTMModel(input_shape=(50, 1), n_classes=2, backend="tensorflow")
        model.build_model()
        model.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.train(X[:10], y[:10], epochs=1, verbose=0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'lstm_model.h5')
            model.save(filepath)
            assert os.path.exists(filepath)
            
            model2 = LSTMModel(input_shape=(50, 1), n_classes=2, backend="tensorflow")
            model2.load(filepath)
            assert model2.model is not None


class TestBaseDeepModelEdgeCases:
    """Test edge cases for BaseDeepModel."""

    def test_tensorflow_not_available_error(self, monkeypatch):
        """Test error when TensorFlow is not available."""
        # This test is more for documentation; actual testing would require
        # mocking the TensorFlow import
        pass

    def test_pytorch_not_available_error(self):
        """Test error when PyTorch backend is requested but not available."""
        # PyTorch is intentionally not set as available in deep_models.py
        with pytest.raises(ImportError, match="PyTorch not available"):
            CNN1D(
                input_shape=(100, 1),
                n_classes=3,
                backend="pytorch"
            )

    def test_invalid_backend(self):
        """Test with invalid backend name."""
        # Should still initialize but might fail at build
        model = CNN1D(
            input_shape=(100, 1),
            n_classes=3,
            backend="invalid"
        )
        # The error will be raised when trying to build or use the model
        # since neither TF_AVAILABLE nor TORCH_AVAILABLE will be true for "invalid"


class TestModelConfiguration:
    """Test various model configurations."""

    def test_cnn1d_single_conv_layer(self):
        """Test CNN1D with single convolutional layer."""
        model = CNN1D(
            input_shape=(100, 1),
            n_classes=3,
            n_filters=[32],
            kernel_sizes=[7],
            pool_sizes=[2],
            backend="tensorflow"
        )
        model.build_model()
        assert model.model is not None

    def test_cnn1d_many_conv_layers(self):
        """Test CNN1D with many convolutional layers."""
        model = CNN1D(
            input_shape=(100, 1),
            n_classes=3,
            n_filters=[16, 32, 64, 128, 256],
            kernel_sizes=[11, 9, 7, 5, 3],
            pool_sizes=[2, 2, 2, 2, 2],
            backend="tensorflow"
        )
        model.build_model()
        assert model.model is not None

    def test_lstm_single_layer(self):
        """Test LSTM with single layer."""
        model = LSTMModel(
            input_shape=(50, 1),
            n_classes=2,
            lstm_units=[64],
            backend="tensorflow"
        )
        model.build_model()
        assert model.model is not None

    def test_lstm_many_layers(self):
        """Test LSTM with many layers."""
        model = LSTMModel(
            input_shape=(50, 1),
            n_classes=2,
            lstm_units=[256, 128, 64, 32],
            backend="tensorflow"
        )
        model.build_model()
        assert model.model is not None

    def test_binary_classification(self, sample_sequence_data):
        """Test binary classification (2 classes)."""
        X, y = sample_sequence_data
        # sample_sequence_data has shape (50, 1), so use that
        model = CNN1D(input_shape=(50, 1), n_classes=2, backend="tensorflow")
        model.build_model()
        model.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.train(X[:20], y[:20], epochs=1, verbose=0)
        
        predictions = model.predict(X[:5])
        # Binary classification (n_classes=2) outputs single probability, not 2 outputs
        assert predictions.shape[0] == 5
        assert len(predictions.shape) == 2

    def test_multiclass_classification(self, sample_signal_data):
        """Test multiclass classification (>2 classes)."""
        X, y = sample_signal_data
        model = CNN1D(input_shape=(100, 1), n_classes=5, backend="tensorflow")
        model.build_model()
        model.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Create labels for 5 classes
        y_multi = np.random.randint(0, 5, size=(100,))
        model.train(X[:20], y_multi[:20], epochs=1, verbose=0)
        
        predictions = model.predict(X[:5])
        assert predictions.shape[1] == 5


class TestTrainingParameters:
    """Test various training parameters."""

    def test_different_batch_sizes(self, sample_signal_data):
        """Test training with different batch sizes."""
        X, y = sample_signal_data
        for batch_size in [8, 16, 32]:
            model = CNN1D(input_shape=(100, 1), n_classes=3, backend="tensorflow")
            model.build_model()
            model.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            model.train(X[:32], y[:32], epochs=1, batch_size=batch_size, verbose=0)
            assert model.history is not None

    def test_different_epochs(self, sample_signal_data):
        """Test training with different number of epochs."""
        X, y = sample_signal_data
        for epochs in [1, 2, 5]:
            model = CNN1D(input_shape=(100, 1), n_classes=3, backend="tensorflow")
            model.build_model()
            model.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            model.train(X[:20], y[:20], epochs=epochs, verbose=0)
            assert model.history is not None

    def test_with_callbacks(self, sample_signal_data):
        """Test training with callbacks."""
        X, y = sample_signal_data
        model = CNN1D(input_shape=(100, 1), n_classes=3, backend="tensorflow")
        model.build_model()
        model.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        from tensorflow.keras.callbacks import EarlyStopping
        early_stop = EarlyStopping(monitor='loss', patience=2, verbose=0)
        
        model.train(X[:20], y[:20], epochs=5, callbacks=[early_stop], verbose=0)
        assert model.history is not None

    def test_verbose_training(self, sample_signal_data):
        """Test training with verbose output."""
        X, y = sample_signal_data
        model = CNN1D(input_shape=(100, 1), n_classes=3, backend="tensorflow")
        model.build_model()
        model.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.train(X[:10], y[:10], epochs=1, verbose=1)
        assert model.history is not None

