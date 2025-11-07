"""
Pragmatic ML Models Tests - Focus on What Works

This test file contains tests that target actual working functionality
in the ml_models module, based on verified API signatures.

Author: vitalDSP Team
Date: 2025-11-06
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import warnings

warnings.filterwarnings("ignore")

# Check availability
try:
    from vitalDSP.ml_models.autoencoder import BaseAutoencoder, StandardAutoencoder
    AUTOENCODER_AVAILABLE = True
except ImportError:
    AUTOENCODER_AVAILABLE = False

try:
    from vitalDSP.ml_models.deep_models import BaseDeepModel, CNN1D
    DEEP_MODELS_AVAILABLE = True
except ImportError:
    DEEP_MODELS_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ============================================================================
# AUTOENCODER TESTS - VERIFIED API
# ============================================================================

@pytest.mark.skipif(not AUTOENCODER_AVAILABLE or not TF_AVAILABLE, reason="Requirements not available")
class TestBaseAutoencoderVerified:
    """Tests for BaseAutoencoder with verified API."""

    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        ae = BaseAutoencoder(
            input_shape=(100,),
            latent_dim=16,
            backend='tensorflow',
            random_state=42
        )
        assert ae.input_shape == (100,)
        assert ae.latent_dim == 16
        assert ae.backend == 'tensorflow'
        assert ae.random_state == 42

    def test_init_minimal_params(self):
        """Test initialization with minimal parameters."""
        ae = BaseAutoencoder(input_shape=(50,), latent_dim=8)
        assert ae.input_shape == (50,)
        assert ae.latent_dim == 8

    def test_init_different_input_shapes(self):
        """Test with different input shapes."""
        shapes = [(100,), (200,), (500,), (1000,)]
        for shape in shapes:
            ae = BaseAutoencoder(input_shape=shape, latent_dim=16)
            assert ae.input_shape == shape

    def test_init_different_latent_dims(self):
        """Test with different latent dimensions."""
        latent_dims = [8, 16, 32, 64]
        for dim in latent_dims:
            ae = BaseAutoencoder(input_shape=(100,), latent_dim=dim)
            assert ae.latent_dim == dim

    def test_backend_tensorflow(self):
        """Test TensorFlow backend initialization."""
        ae = BaseAutoencoder(
            input_shape=(100,),
            latent_dim=16,
            backend='tensorflow'
        )
        assert ae.backend == 'tensorflow'

    def test_random_state_reproducibility(self):
        """Test that random state is stored."""
        ae1 = BaseAutoencoder(input_shape=(100,), latent_dim=16, random_state=42)
        ae2 = BaseAutoencoder(input_shape=(100,), latent_dim=16, random_state=42)
        assert ae1.random_state == ae2.random_state


@pytest.mark.skipif(not AUTOENCODER_AVAILABLE or not TF_AVAILABLE, reason="Requirements not available")
class TestStandardAutoencoderVerified:
    """Tests for StandardAutoencoder with verified API."""

    def test_init_default_params(self):
        """Test initialization with mostly default parameters."""
        ae = StandardAutoencoder(
            input_shape=(100,),
            latent_dim=16
        )
        assert ae.input_shape == (100,)
        assert ae.latent_dim == 16

    def test_init_custom_hidden_dims(self):
        """Test with custom hidden dimensions."""
        ae = StandardAutoencoder(
            input_shape=(100,),
            latent_dim=16,
            hidden_dims=[128, 64, 32]
        )
        assert ae.hidden_dims == [128, 64, 32]

    def test_init_activation_functions(self):
        """Test with different activation functions."""
        activations = ['relu', 'tanh', 'sigmoid']
        for act in activations:
            ae = StandardAutoencoder(
                input_shape=(100,),
                latent_dim=16,
                activation=act
            )
            assert ae.activation == act

    def test_init_output_activation(self):
        """Test different output activations."""
        output_acts = ['linear', 'sigmoid', 'tanh']
        for act in output_acts:
            ae = StandardAutoencoder(
                input_shape=(100,),
                latent_dim=16,
                output_activation=act
            )
            assert ae.output_activation == act

    def test_init_with_batch_norm(self):
        """Test with and without batch normalization."""
        ae_with_bn = StandardAutoencoder(
            input_shape=(100,),
            latent_dim=16,
            use_batch_norm=True
        )
        ae_without_bn = StandardAutoencoder(
            input_shape=(100,),
            latent_dim=16,
            use_batch_norm=False
        )
        assert ae_with_bn.use_batch_norm == True
        assert ae_without_bn.use_batch_norm == False

    def test_init_dropout_rate(self):
        """Test with different dropout rates."""
        dropout_rates = [0.0, 0.1, 0.2, 0.5]
        for rate in dropout_rates:
            ae = StandardAutoencoder(
                input_shape=(100,),
                latent_dim=16,
                dropout_rate=rate
            )
            assert ae.dropout_rate == rate

    def test_fit_basic(self):
        """Test basic model fitting."""
        ae = StandardAutoencoder(
            input_shape=(100,),
            latent_dim=16
        )

        X = np.random.randn(50, 100).astype(np.float32)

        # Fit model
        history = ae.fit(X, epochs=1, batch_size=16, verbose=0)
        assert history is not None

    def test_fit_with_validation_split(self):
        """Test fitting with validation split."""
        ae = StandardAutoencoder(
            input_shape=(100,),
            latent_dim=16
        )

        X = np.random.randn(100, 100).astype(np.float32)

        history = ae.fit(
            X,
            epochs=1,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )
        assert history is not None

    def test_encode_after_fit(self):
        """Test encoding after fitting."""
        ae = StandardAutoencoder(
            input_shape=(100,),
            latent_dim=16
        )

        X = np.random.randn(50, 100).astype(np.float32)
        ae.fit(X, epochs=1, batch_size=16, verbose=0)

        # Encode
        encoded = ae.encode(X[:10])
        assert encoded.shape == (10, 16)

    def test_decode_after_fit(self):
        """Test decoding after fitting."""
        ae = StandardAutoencoder(
            input_shape=(100,),
            latent_dim=16
        )

        X = np.random.randn(50, 100).astype(np.float32)
        ae.fit(X, epochs=1, batch_size=16, verbose=0)

        # Encode then decode
        encoded = ae.encode(X[:10])
        decoded = ae.decode(encoded)
        assert decoded.shape == (10, 100)

    def test_reconstruction_error_mse(self):
        """Test reconstruction error with MSE metric."""
        ae = StandardAutoencoder(
            input_shape=(100,),
            latent_dim=16
        )

        X = np.random.randn(50, 100).astype(np.float32)
        ae.fit(X, epochs=1, batch_size=16, verbose=0)

        # Compute reconstruction error
        errors = ae.compute_reconstruction_error(X[:10], metric='mse')
        assert len(errors) == 10
        assert all(e >= 0 for e in errors)

    def test_reconstruction_error_mae(self):
        """Test reconstruction error with MAE metric."""
        ae = StandardAutoencoder(
            input_shape=(100,),
            latent_dim=16
        )

        X = np.random.randn(50, 100).astype(np.float32)
        ae.fit(X, epochs=1, batch_size=16, verbose=0)

        errors = ae.compute_reconstruction_error(X[:10], metric='mae')
        assert len(errors) == 10
        assert all(e >= 0 for e in errors)

    def test_reconstruction_error_rmse(self):
        """Test reconstruction error with RMSE metric."""
        ae = StandardAutoencoder(
            input_shape=(100,),
            latent_dim=16
        )

        X = np.random.randn(50, 100).astype(np.float32)
        ae.fit(X, epochs=1, batch_size=16, verbose=0)

        errors = ae.compute_reconstruction_error(X[:10], metric='rmse')
        assert len(errors) == 10
        assert all(e >= 0 for e in errors)

    def test_latent_space_dimensionality(self):
        """Test that latent space has correct dimensionality."""
        latent_dims = [8, 16, 32]
        X = np.random.randn(50, 100).astype(np.float32)

        for dim in latent_dims:
            ae = StandardAutoencoder(
                input_shape=(100,),
                latent_dim=dim
            )
            ae.fit(X, epochs=1, batch_size=16, verbose=0)
            encoded = ae.encode(X[:5])
            assert encoded.shape[1] == dim

    def test_different_input_sizes(self):
        """Test with different input sizes."""
        input_sizes = [50, 100, 200]

        for size in input_sizes:
            X = np.random.randn(30, size).astype(np.float32)
            ae = StandardAutoencoder(
                input_shape=(size,),
                latent_dim=16
            )
            ae.fit(X, epochs=1, batch_size=16, verbose=0)
            encoded = ae.encode(X[:5])
            assert encoded.shape == (5, 16)

    def test_multiple_hidden_layer_configs(self):
        """Test different hidden layer configurations."""
        hidden_configs = [
            [64, 32],
            [128, 64, 32],
            [256, 128, 64, 32]
        ]

        X = np.random.randn(50, 100).astype(np.float32)

        for config in hidden_configs:
            ae = StandardAutoencoder(
                input_shape=(100,),
                latent_dim=16,
                hidden_dims=config
            )
            ae.fit(X, epochs=1, batch_size=16, verbose=0)
            assert ae.hidden_dims == config


# ============================================================================
# DEEP MODELS TESTS - VERIFIED API
# ============================================================================

@pytest.mark.skipif(not DEEP_MODELS_AVAILABLE or not TF_AVAILABLE, reason="Requirements not available")
class TestCNN1DVerified:
    """Tests for CNN1D with verified API."""

    def test_init_basic(self):
        """Test basic CNN1D initialization."""
        model = CNN1D(
            input_shape=(1000, 1),
            n_classes=5
        )
        assert model.input_shape == (1000, 1)
        assert model.n_classes == 5

    def test_init_with_custom_filters(self):
        """Test with custom filter configurations."""
        model = CNN1D(
            input_shape=(1000, 1),
            n_classes=5,
            n_filters=[16, 32, 64]
        )
        assert model.n_filters == [16, 32, 64]

    def test_init_with_kernel_sizes(self):
        """Test with custom kernel sizes."""
        model = CNN1D(
            input_shape=(1000, 1),
            n_classes=5,
            kernel_sizes=[7, 5, 3]
        )
        assert model.kernel_sizes == [7, 5, 3]

    def test_init_with_pool_sizes(self):
        """Test with custom pool sizes."""
        model = CNN1D(
            input_shape=(1000, 1),
            n_classes=5,
            pool_sizes=[2, 2, 2]
        )
        assert model.pool_sizes == [2, 2, 2]

    def test_init_dropout_rate(self):
        """Test with different dropout rates."""
        dropout_rates = [0.0, 0.3, 0.5]
        for rate in dropout_rates:
            model = CNN1D(
                input_shape=(1000, 1),
                n_classes=5,
                dropout_rate=rate
            )
            assert model.dropout_rate == rate

    def test_init_with_residual(self):
        """Test with residual connections."""
        model = CNN1D(
            input_shape=(1000, 1),
            n_classes=5,
            use_residual=True
        )
        assert model.use_residual == True

    def test_build_model_tensorflow(self):
        """Test building TensorFlow model."""
        model = CNN1D(
            input_shape=(1000, 1),
            n_classes=5,
            backend='tensorflow'
        )
        built_model = model.build_model()
        assert built_model is not None
        assert model.model is not None

    def test_model_architecture_output_shape(self):
        """Test model output shape."""
        model = CNN1D(
            input_shape=(1000, 1),
            n_classes=5
        )
        model.build_model()

        # Create sample input
        X = np.random.randn(1, 1000, 1).astype(np.float32)
        output = model.model.predict(X, verbose=0)

        # For multiclass, output should be (1, 5)
        assert output.shape == (1, 5)

    def test_binary_classification(self):
        """Test binary classification output."""
        model = CNN1D(
            input_shape=(1000, 1),
            n_classes=2
        )
        model.build_model()

        X = np.random.randn(1, 1000, 1).astype(np.float32)
        output = model.model.predict(X, verbose=0)

        # Binary classification uses sigmoid
        assert output.shape == (1, 1)

    def test_different_input_lengths(self):
        """Test with different input sequence lengths."""
        lengths = [500, 1000, 2000]

        for length in lengths:
            model = CNN1D(
                input_shape=(length, 1),
                n_classes=5
            )
            model.build_model()
            assert model.input_shape == (length, 1)

    def test_multiple_input_channels(self):
        """Test with multiple input channels."""
        channels = [1, 2, 3]

        for n_channels in channels:
            model = CNN1D(
                input_shape=(1000, n_channels),
                n_classes=5
            )
            model.build_model()
            assert model.input_shape == (1000, n_channels)


# ============================================================================
# UTILITY AND EDGE CASE TESTS
# ============================================================================

@pytest.mark.skipif(not AUTOENCODER_AVAILABLE or not TF_AVAILABLE, reason="Requirements not available")
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_sample_encode(self):
        """Test encoding single sample."""
        ae = StandardAutoencoder(input_shape=(100,), latent_dim=16)
        X = np.random.randn(50, 100).astype(np.float32)
        ae.fit(X, epochs=1, batch_size=16, verbose=0)

        single = X[:1]
        encoded = ae.encode(single)
        assert encoded.shape == (1, 16)

    def test_large_batch_encode(self):
        """Test encoding large batch."""
        ae = StandardAutoencoder(input_shape=(100,), latent_dim=16)
        X = np.random.randn(200, 100).astype(np.float32)
        ae.fit(X[:150], epochs=1, batch_size=16, verbose=0)

        encoded = ae.encode(X)
        assert encoded.shape == (200, 16)

    def test_reconstruction_preserves_shape(self):
        """Test that reconstruction preserves input shape."""
        ae = StandardAutoencoder(input_shape=(100,), latent_dim=16)
        X = np.random.randn(50, 100).astype(np.float32)
        ae.fit(X, epochs=1, batch_size=16, verbose=0)

        encoded = ae.encode(X[:20])
        decoded = ae.decode(encoded)
        assert decoded.shape == X[:20].shape

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        batch_sizes = [8, 16, 32]
        X = np.random.randn(100, 100).astype(np.float32)

        for batch_size in batch_sizes:
            ae = StandardAutoencoder(input_shape=(100,), latent_dim=16)
            history = ae.fit(X, epochs=1, batch_size=batch_size, verbose=0)
            assert history is not None


