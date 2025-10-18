"""
Autoencoder Models Module for Physiological Signal Processing

This module provides comprehensive autoencoder architectures for physiological
signal analysis including ECG, PPG, EEG, and other vital signs. It implements
various autoencoder types for unsupervised anomaly detection, signal denoising,
dimensionality reduction, and feature learning.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Standard Autoencoder for basic reconstruction
- Variational Autoencoder (VAE) for probabilistic modeling
- Denoising Autoencoder for noise reduction
- Convolutional Autoencoder for spatial feature learning
- LSTM Autoencoder for temporal sequence modeling
- Comprehensive training and evaluation utilities
- Model saving and loading capabilities

Examples:
--------
Basic autoencoder for anomaly detection:
    >>> import numpy as np
    >>> from vitalDSP.ml_models.autoencoder import StandardAutoencoder
    >>> signal_data = np.random.randn(1000, 100)  # 1000 samples, 100 features
    >>> autoencoder = StandardAutoencoder(input_dim=100, encoding_dim=32)
    >>> autoencoder.compile(optimizer='adam', loss='mse')
    >>> autoencoder.fit(signal_data, signal_data, epochs=10)

Variational autoencoder:
    >>> from vitalDSP.ml_models.autoencoder import VariationalAutoencoder
    >>> vae = VariationalAutoencoder(input_dim=100, latent_dim=16)
    >>> vae.compile(optimizer='adam', loss='mse')
    >>> vae.fit(signal_data, signal_data, epochs=10)

Denoising autoencoder:
    >>> from vitalDSP.ml_models.autoencoder import DenoisingAutoencoder
    >>> noisy_data = signal_data + np.random.normal(0, 0.1, signal_data.shape)
    >>> dae = DenoisingAutoencoder(input_dim=100, encoding_dim=32)
    >>> dae.compile(optimizer='adam', loss='mse')
    >>> dae.fit(noisy_data, signal_data, epochs=10)
"""

import numpy as np
from typing import Optional, Union, Tuple, List, Dict, Any, Callable
from pathlib import Path
import warnings

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


class BaseAutoencoder:
    """
    Base class for all autoencoder models.

    Provides common functionality for encoding, decoding, and anomaly detection.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        latent_dim: int = 32,
        backend: str = "tensorflow",
        random_state: Optional[int] = None,
    ):
        """
        Initialize base autoencoder.

        Parameters
        ----------
        input_shape : tuple
            Shape of input signals (length, n_channels) or (length,)
        latent_dim : int, default=32
            Dimensionality of latent space
        backend : str, default='tensorflow'
            Deep learning backend ('tensorflow' or 'pytorch')
        random_state : int, optional
            Random seed for reproducibility
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.backend = backend.lower()
        self.random_state = random_state

        self.encoder = None
        self.decoder = None
        self.model = None
        self.history = None

        # Set random seeds
        if random_state is not None:
            np.random.seed(random_state)
            if self.backend == "tensorflow" and TENSORFLOW_AVAILABLE:
                tf.random.set_seed(random_state)
            elif self.backend == "pytorch" and PYTORCH_AVAILABLE:
                torch.manual_seed(random_state)

        # Validate backend
        if self.backend == "tensorflow" and not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is not installed. Install with: pip install tensorflow"
            )
        elif self.backend == "pytorch" and not PYTORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Install with: pip install torch"
            )

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode signals to latent space.

        Parameters
        ----------
        X : np.ndarray
            Input signals of shape (n_samples, length) or (n_samples, length, n_channels)

        Returns
        -------
        np.ndarray
            Latent representations of shape (n_samples, latent_dim)
        """
        if self.encoder is None:
            raise ValueError("Model not built. Call fit() first.")

        if self.backend == "tensorflow":
            return self.encoder.predict(X, verbose=0)
        else:  # pytorch
            self.encoder.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                latent = self.encoder(X_tensor)
                return latent.cpu().numpy()

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """
        Decode latent representations to signals.

        Parameters
        ----------
        latent : np.ndarray
            Latent representations of shape (n_samples, latent_dim)

        Returns
        -------
        np.ndarray
            Reconstructed signals
        """
        if self.decoder is None:
            raise ValueError("Model not built. Call fit() first.")

        if self.backend == "tensorflow":
            return self.decoder.predict(latent, verbose=0)
        else:  # pytorch
            self.decoder.eval()
            with torch.no_grad():
                latent_tensor = torch.FloatTensor(latent)
                reconstructed = self.decoder(latent_tensor)
                return reconstructed.cpu().numpy()

    def compute_reconstruction_error(
        self, X: np.ndarray, metric: str = "mse"
    ) -> np.ndarray:
        """
        Compute reconstruction error for anomaly detection.

        Parameters
        ----------
        X : np.ndarray
            Input signals
        metric : str, default='mse'
            Error metric ('mse', 'mae', 'rmse')

        Returns
        -------
        np.ndarray
            Reconstruction errors of shape (n_samples,)
        """
        # Get reconstructions
        if self.backend == "tensorflow":
            X_reconstructed = self.model.predict(X, verbose=0)
        else:  # pytorch
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                X_reconstructed = self.model(X_tensor).cpu().numpy()

        # Compute error
        if metric == "mse":
            errors = np.mean((X - X_reconstructed) ** 2, axis=tuple(range(1, X.ndim)))
        elif metric == "mae":
            errors = np.mean(np.abs(X - X_reconstructed), axis=tuple(range(1, X.ndim)))
        elif metric == "rmse":
            errors = np.sqrt(
                np.mean((X - X_reconstructed) ** 2, axis=tuple(range(1, X.ndim)))
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return errors

    def detect_anomalies(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None,
        contamination: float = 0.1,
        metric: str = "mse",
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Detect anomalies using reconstruction error.

        Parameters
        ----------
        X : np.ndarray
            Input signals
        threshold : float, optional
            Anomaly threshold. If None, computed from contamination
        contamination : float, default=0.1
            Expected proportion of anomalies (used if threshold is None)
        metric : str, default='mse'
            Error metric

        Returns
        -------
        anomalies : np.ndarray
            Boolean array indicating anomalies
        scores : np.ndarray
            Anomaly scores (reconstruction errors)
        threshold : float
            Threshold used for detection
        """
        # Compute reconstruction errors
        scores = self.compute_reconstruction_error(X, metric=metric)

        # Determine threshold
        if threshold is None:
            threshold = np.percentile(scores, 100 * (1 - contamination))

        # Detect anomalies
        anomalies = scores > threshold

        return anomalies, scores, threshold

    def save(self, filepath: str):
        """Save model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if self.backend == "tensorflow":
            self.model.save(str(filepath))
        else:  # pytorch
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "encoder_state_dict": self.encoder.state_dict(),
                    "decoder_state_dict": self.decoder.state_dict(),
                    "input_shape": self.input_shape,
                    "latent_dim": self.latent_dim,
                },
                str(filepath),
            )

    def load(self, filepath: str):
        """Load model from file."""
        if self.backend == "tensorflow":
            self.model = keras.models.load_model(filepath)
        else:  # pytorch
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            self.decoder.load_state_dict(checkpoint["decoder_state_dict"])


class StandardAutoencoder(BaseAutoencoder):
    """
    Standard feedforward autoencoder.

    Architecture:
    - Encoder: Input -> Dense layers -> Latent space
    - Decoder: Latent space -> Dense layers -> Output

    Use cases:
    - Dimensionality reduction
    - Feature learning
    - Anomaly detection

    Examples
    --------
    >>> from vitalDSP.ml_models.autoencoder import StandardAutoencoder
    >>> import numpy as np
    >>>
    >>> # Generate sample ECG signals
    >>> X_train = np.random.randn(1000, 500)  # 1000 signals, 500 samples each
    >>> X_test = np.random.randn(100, 500)
    >>>
    >>> # Create and train autoencoder
    >>> ae = StandardAutoencoder(
    ...     input_shape=(500,),
    ...     latent_dim=32,
    ...     hidden_dims=[256, 128, 64],
    ...     activation='relu'
    ... )
    >>> ae.fit(X_train, epochs=50, batch_size=32, validation_split=0.2)
    >>>
    >>> # Detect anomalies
    >>> anomalies, scores, threshold = ae.detect_anomalies(X_test, contamination=0.1)
    >>> print(f"Detected {anomalies.sum()} anomalies")
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        latent_dim: int = 32,
        hidden_dims: List[int] = [256, 128, 64],
        activation: str = "relu",
        output_activation: str = "linear",
        use_batch_norm: bool = True,
        dropout_rate: float = 0.2,
        backend: str = "tensorflow",
        random_state: Optional[int] = None,
    ):
        """
        Initialize standard autoencoder.

        Parameters
        ----------
        input_shape : tuple
            Shape of input signals
        latent_dim : int, default=32
            Dimensionality of latent space
        hidden_dims : list, default=[256, 128, 64]
            Dimensions of hidden layers in encoder (reversed for decoder)
        activation : str, default='relu'
            Activation function for hidden layers
        output_activation : str, default='linear'
            Activation function for output layer
        use_batch_norm : bool, default=True
            Whether to use batch normalization
        dropout_rate : float, default=0.2
            Dropout rate for regularization
        backend : str, default='tensorflow'
            Deep learning backend
        random_state : int, optional
            Random seed
        """
        super().__init__(input_shape, latent_dim, backend, random_state)

        self.hidden_dims = hidden_dims
        self.activation = activation
        self.output_activation = output_activation
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        # Build model
        self._build_model()

    def _build_tensorflow_model(self):
        """Build TensorFlow/Keras model."""
        # Encoder
        encoder_input = keras.Input(shape=self.input_shape)
        x = layers.Flatten()(encoder_input)

        for dim in self.hidden_dims:
            x = layers.Dense(dim)(x)
            if self.use_batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(self.activation)(x)
            x = layers.Dropout(self.dropout_rate)(x)

        latent = layers.Dense(self.latent_dim, name="latent")(x)
        self.encoder = Model(encoder_input, latent, name="encoder")

        # Decoder
        decoder_input = keras.Input(shape=(self.latent_dim,))
        x = decoder_input

        for dim in reversed(self.hidden_dims):
            x = layers.Dense(dim)(x)
            if self.use_batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(self.activation)(x)
            x = layers.Dropout(self.dropout_rate)(x)

        # Output
        output_dim = np.prod(self.input_shape)
        x = layers.Dense(output_dim, activation=self.output_activation)(x)
        decoder_output = layers.Reshape(self.input_shape)(x)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

        # Full autoencoder
        self.model = Model(
            encoder_input, self.decoder(self.encoder(encoder_input)), name="autoencoder"
        )

    def _build_pytorch_model(self):
        """Build PyTorch model."""

        class Encoder(nn.Module):
            def __init__(
                self,
                input_dim,
                hidden_dims,
                latent_dim,
                activation,
                use_batch_norm,
                dropout_rate,
            ):
                super().__init__()
                self.input_dim = input_dim
                layers_list = []

                prev_dim = input_dim
                for dim in hidden_dims:
                    layers_list.append(nn.Linear(prev_dim, dim))
                    if use_batch_norm:
                        layers_list.append(nn.BatchNorm1d(dim))
                    if activation == "relu":
                        layers_list.append(nn.ReLU())
                    elif activation == "tanh":
                        layers_list.append(nn.Tanh())
                    layers_list.append(nn.Dropout(dropout_rate))
                    prev_dim = dim

                layers_list.append(nn.Linear(prev_dim, latent_dim))
                self.network = nn.Sequential(*layers_list)

            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.network(x)

        class Decoder(nn.Module):
            def __init__(
                self,
                latent_dim,
                hidden_dims,
                output_shape,
                activation,
                output_activation,
                use_batch_norm,
                dropout_rate,
            ):
                super().__init__()
                self.output_shape = output_shape
                layers_list = []

                prev_dim = latent_dim
                for dim in reversed(hidden_dims):
                    layers_list.append(nn.Linear(prev_dim, dim))
                    if use_batch_norm:
                        layers_list.append(nn.BatchNorm1d(dim))
                    if activation == "relu":
                        layers_list.append(nn.ReLU())
                    elif activation == "tanh":
                        layers_list.append(nn.Tanh())
                    layers_list.append(nn.Dropout(dropout_rate))
                    prev_dim = dim

                output_dim = np.prod(output_shape)
                layers_list.append(nn.Linear(prev_dim, output_dim))
                if output_activation == "sigmoid":
                    layers_list.append(nn.Sigmoid())
                elif output_activation == "tanh":
                    layers_list.append(nn.Tanh())

                self.network = nn.Sequential(*layers_list)

            def forward(self, x):
                x = self.network(x)
                return x.view(x.size(0), *self.output_shape)

        class Autoencoder(nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, x):
                latent = self.encoder(x)
                reconstructed = self.decoder(latent)
                return reconstructed

        input_dim = np.prod(self.input_shape)
        self.encoder = Encoder(
            input_dim,
            self.hidden_dims,
            self.latent_dim,
            self.activation,
            self.use_batch_norm,
            self.dropout_rate,
        )
        self.decoder = Decoder(
            self.latent_dim,
            self.hidden_dims,
            self.input_shape,
            self.activation,
            self.output_activation,
            self.use_batch_norm,
            self.dropout_rate,
        )
        self.model = Autoencoder(self.encoder, self.decoder)

    def _build_model(self):
        """Build model based on backend."""
        if self.backend == "tensorflow":
            self._build_tensorflow_model()
        else:
            self._build_pytorch_model()

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        callbacks: Optional[List] = None,
        verbose: int = 1,
    ):
        """
        Train the autoencoder.

        Parameters
        ----------
        X : np.ndarray
            Training signals
        y : np.ndarray, optional
            Ignored (for sklearn compatibility)
        epochs : int, default=100
            Number of training epochs
        batch_size : int, default=32
            Batch size
        validation_split : float, default=0.2
            Fraction of data to use for validation
        validation_data : tuple, optional
            Validation data (X_val, X_val)
        callbacks : list, optional
            Training callbacks
        verbose : int, default=1
            Verbosity level

        Returns
        -------
        self
        """
        if self.backend == "tensorflow":
            # Compile model
            self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

            # Default callbacks
            if callbacks is None:
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=10, restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
                    ),
                ]

            # Train
            self.history = self.model.fit(
                X,
                X,  # Input and target are the same
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split if validation_data is None else 0,
                validation_data=(
                    (validation_data[0], validation_data[0])
                    if validation_data is not None
                    else None
                ),
                callbacks=callbacks,
                verbose=verbose,
            )

        else:  # pytorch
            # Training setup
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # Prepare data
            if validation_data is not None:
                X_train, X_val = X, validation_data[0]
            else:
                split_idx = int(len(X) * (1 - validation_split))
                X_train, X_val = X[:split_idx], X[split_idx:]

            train_dataset = TensorDataset(
                torch.FloatTensor(X_train), torch.FloatTensor(X_train)
            )
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )

            val_dataset = TensorDataset(
                torch.FloatTensor(X_val), torch.FloatTensor(X_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # Training loop
            history = {"loss": [], "val_loss": []}
            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)
                history["loss"].append(train_loss)

                # Validation
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                history["val_loss"].append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        self.model.load_state_dict(best_model_state)
                        break

                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}"
                    )

            self.history = history

        return self


class ConvolutionalAutoencoder(BaseAutoencoder):
    """
    Convolutional autoencoder for 1D signals.

    Architecture:
    - Encoder: Conv1D layers with pooling
    - Decoder: Transposed Conv1D layers (upsampling)

    Best for signals with spatial structure and local patterns.

    Examples
    --------
    >>> from vitalDSP.ml_models.autoencoder import ConvolutionalAutoencoder
    >>> import numpy as np
    >>>
    >>> # Generate sample signals
    >>> X_train = np.random.randn(1000, 500, 1)  # 1000 signals, 500 samples, 1 channel
    >>>
    >>> # Create and train
    >>> cae = ConvolutionalAutoencoder(
    ...     input_shape=(500, 1),
    ...     latent_dim=32,
    ...     n_filters=[32, 64, 128],
    ...     kernel_sizes=[7, 5, 3],
    ...     pool_sizes=[2, 2, 2]
    ... )
    >>> cae.fit(X_train, epochs=50)
    >>>
    >>> # Encode signals
    >>> latent_features = cae.encode(X_train[:10])
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        latent_dim: int = 32,
        n_filters: List[int] = [32, 64, 128],
        kernel_sizes: Union[int, List[int]] = 7,
        pool_sizes: Union[int, List[int]] = 2,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout_rate: float = 0.2,
        backend: str = "tensorflow",
        random_state: Optional[int] = None,
    ):
        """
        Initialize convolutional autoencoder.

        Parameters
        ----------
        input_shape : tuple
            Shape of input signals (length, n_channels)
        latent_dim : int, default=32
            Dimensionality of latent space
        n_filters : list, default=[32, 64, 128]
            Number of filters in each conv layer
        kernel_sizes : int or list, default=7
            Kernel sizes for conv layers
        pool_sizes : int or list, default=2
            Pool sizes for max pooling
        activation : str, default='relu'
            Activation function
        use_batch_norm : bool, default=True
            Whether to use batch normalization
        dropout_rate : float, default=0.2
            Dropout rate
        backend : str, default='tensorflow'
            Deep learning backend
        random_state : int, optional
            Random seed
        """
        super().__init__(input_shape, latent_dim, backend, random_state)

        self.n_filters = n_filters
        self.kernel_sizes = (
            [kernel_sizes] * len(n_filters)
            if isinstance(kernel_sizes, int)
            else kernel_sizes
        )
        self.pool_sizes = (
            [pool_sizes] * len(n_filters) if isinstance(pool_sizes, int) else pool_sizes
        )
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        # Build model
        self._build_model()

    def _build_tensorflow_model(self):
        """Build TensorFlow/Keras model."""
        # Encoder
        encoder_input = keras.Input(shape=self.input_shape)
        x = encoder_input

        # Track shapes for decoder
        encoder_shapes = []

        for n_filter, kernel_size, pool_size in zip(
            self.n_filters, self.kernel_sizes, self.pool_sizes
        ):
            encoder_shapes.append(x.shape[1:])
            x = layers.Conv1D(n_filter, kernel_size, padding="same")(x)
            if self.use_batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(self.activation)(x)
            x = layers.MaxPooling1D(pool_size, padding="same")(x)
            x = layers.Dropout(self.dropout_rate)(x)

        # Latent space
        shape_before_flatten = x.shape[1:]
        x = layers.Flatten()(x)
        latent = layers.Dense(self.latent_dim, name="latent")(x)
        self.encoder = Model(encoder_input, latent, name="encoder")

        # Decoder
        decoder_input = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(np.prod(shape_before_flatten))(decoder_input)
        x = layers.Reshape(shape_before_flatten)(x)

        for i, (n_filter, kernel_size, pool_size) in enumerate(
            reversed(list(zip(self.n_filters, self.kernel_sizes, self.pool_sizes)))
        ):
            x = layers.UpSampling1D(pool_size)(x)
            x = layers.Conv1D(n_filter, kernel_size, padding="same")(x)
            if self.use_batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(self.activation)(x)
            x = layers.Dropout(self.dropout_rate)(x)

        # Output layer
        decoder_output = layers.Conv1D(
            self.input_shape[-1], 3, padding="same", activation="linear"
        )(x)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

        # Full autoencoder
        self.model = Model(
            encoder_input,
            self.decoder(self.encoder(encoder_input)),
            name="conv_autoencoder",
        )

    def _build_pytorch_model(self):
        """Build PyTorch model."""
        # Similar to TensorFlow implementation
        # Implementation omitted for brevity - follows same pattern as StandardAutoencoder
        raise NotImplementedError(
            "PyTorch backend for ConvolutionalAutoencoder not yet implemented"
        )

    def _build_model(self):
        """Build model based on backend."""
        if self.backend == "tensorflow":
            self._build_tensorflow_model()
        else:
            self._build_pytorch_model()

    def fit(self, X: np.ndarray, **kwargs):
        """Train the autoencoder. See StandardAutoencoder.fit() for parameters."""
        # Ensure input has channel dimension
        if X.ndim == 2:
            X = X[..., np.newaxis]

        return super(StandardAutoencoder, self).fit(X, **kwargs)


class LSTMAutoencoder(BaseAutoencoder):
    """
    LSTM-based autoencoder for sequential signals.

    Architecture:
    - Encoder: LSTM layers -> Latent representation
    - Decoder: RepeatVector -> LSTM layers -> Output sequence

    Best for signals with temporal dependencies.

    Examples
    --------
    >>> from vitalDSP.ml_models.autoencoder import LSTMAutoencoder
    >>> import numpy as np
    >>>
    >>> # Generate time series data
    >>> X_train = np.random.randn(1000, 100, 1)  # 1000 sequences, 100 timesteps, 1 feature
    >>>
    >>> # Create and train
    >>> lstm_ae = LSTMAutoencoder(
    ...     input_shape=(100, 1),
    ...     latent_dim=32,
    ...     lstm_units=[64, 32]
    ... )
    >>> lstm_ae.fit(X_train, epochs=50)
    >>>
    >>> # Detect anomalies in heartbeat sequences
    >>> anomalies, scores, threshold = lstm_ae.detect_anomalies(X_test)
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        latent_dim: int = 32,
        lstm_units: List[int] = [64, 32],
        use_bidirectional: bool = False,
        dropout_rate: float = 0.2,
        backend: str = "tensorflow",
        random_state: Optional[int] = None,
    ):
        """
        Initialize LSTM autoencoder.

        Parameters
        ----------
        input_shape : tuple
            Shape of input sequences (timesteps, n_features)
        latent_dim : int, default=32
            Dimensionality of latent space
        lstm_units : list, default=[64, 32]
            Number of units in each LSTM layer
        use_bidirectional : bool, default=False
            Whether to use bidirectional LSTMs
        dropout_rate : float, default=0.2
            Dropout rate
        backend : str, default='tensorflow'
            Deep learning backend
        random_state : int, optional
            Random seed
        """
        super().__init__(input_shape, latent_dim, backend, random_state)

        self.lstm_units = lstm_units
        self.use_bidirectional = use_bidirectional
        self.dropout_rate = dropout_rate

        # Build model
        self._build_model()

    def _build_tensorflow_model(self):
        """Build TensorFlow/Keras model."""
        # Encoder
        encoder_input = keras.Input(shape=self.input_shape)
        x = encoder_input

        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1

            lstm_layer = layers.LSTM(
                units, return_sequences=return_sequences, dropout=self.dropout_rate
            )
            if self.use_bidirectional:
                x = layers.Bidirectional(lstm_layer)(x)
            else:
                x = lstm_layer(x)

        latent = layers.Dense(self.latent_dim, name="latent")(x)
        self.encoder = Model(encoder_input, latent, name="encoder")

        # Decoder
        decoder_input = keras.Input(shape=(self.latent_dim,))
        x = layers.RepeatVector(self.input_shape[0])(decoder_input)

        for i, units in enumerate(reversed(self.lstm_units)):
            lstm_layer = layers.LSTM(
                units, return_sequences=True, dropout=self.dropout_rate
            )
            if self.use_bidirectional:
                x = layers.Bidirectional(lstm_layer)(x)
            else:
                x = lstm_layer(x)

        decoder_output = layers.TimeDistributed(layers.Dense(self.input_shape[-1]))(x)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

        # Full autoencoder
        self.model = Model(
            encoder_input,
            self.decoder(self.encoder(encoder_input)),
            name="lstm_autoencoder",
        )

    def _build_pytorch_model(self):
        """Build PyTorch model."""
        raise NotImplementedError(
            "PyTorch backend for LSTMAutoencoder not yet implemented"
        )

    def _build_model(self):
        """Build model based on backend."""
        if self.backend == "tensorflow":
            self._build_tensorflow_model()
        else:
            self._build_pytorch_model()

    def fit(self, X: np.ndarray, **kwargs):
        """Train the autoencoder. See StandardAutoencoder.fit() for parameters."""
        # Ensure input has feature dimension
        if X.ndim == 2:
            X = X[..., np.newaxis]

        # Use parent's fit method from BaseAutoencoder through StandardAutoencoder
        if self.backend == "tensorflow":
            # Compile model
            self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

            # Get training parameters
            epochs = kwargs.get("epochs", 100)
            batch_size = kwargs.get("batch_size", 32)
            validation_split = kwargs.get("validation_split", 0.2)
            validation_data = kwargs.get("validation_data", None)
            verbose = kwargs.get("verbose", 1)
            callbacks = kwargs.get("callbacks", None)

            # Default callbacks
            if callbacks is None:
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=10, restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
                    ),
                ]

            # Train
            self.history = self.model.fit(
                X,
                X,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split if validation_data is None else 0,
                validation_data=(
                    (validation_data[0], validation_data[0])
                    if validation_data is not None
                    else None
                ),
                callbacks=callbacks,
                verbose=verbose,
            )

        return self


class VariationalAutoencoder(BaseAutoencoder):
    """
    Variational Autoencoder (VAE) for probabilistic signal generation.

    Architecture:
    - Encoder: Input -> mu and log_var (latent distribution parameters)
    - Sampling: Reparameterization trick
    - Decoder: Latent sample -> Reconstructed output

    Loss: Reconstruction loss + KL divergence

    Use cases:
    - Generative modeling
    - Signal synthesis
    - Anomaly detection with probability

    Examples
    --------
    >>> from vitalDSP.ml_models.autoencoder import VariationalAutoencoder
    >>> import numpy as np
    >>>
    >>> # Generate sample data
    >>> X_train = np.random.randn(1000, 500)
    >>>
    >>> # Create and train VAE
    >>> vae = VariationalAutoencoder(
    ...     input_shape=(500,),
    ...     latent_dim=32,
    ...     beta=1.0  # KL divergence weight
    ... )
    >>> vae.fit(X_train, epochs=100)
    >>>
    >>> # Generate new signals
    >>> z_samples = np.random.randn(10, 32)
    >>> generated_signals = vae.decode(z_samples)
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        latent_dim: int = 32,
        hidden_dims: List[int] = [256, 128, 64],
        activation: str = "relu",
        beta: float = 1.0,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.2,
        backend: str = "tensorflow",
        random_state: Optional[int] = None,
    ):
        """
        Initialize VAE.

        Parameters
        ----------
        input_shape : tuple
            Shape of input signals
        latent_dim : int, default=32
            Dimensionality of latent space
        hidden_dims : list, default=[256, 128, 64]
            Dimensions of hidden layers
        activation : str, default='relu'
            Activation function
        beta : float, default=1.0
            Weight for KL divergence term (beta-VAE)
        use_batch_norm : bool, default=True
            Whether to use batch normalization
        dropout_rate : float, default=0.2
            Dropout rate
        backend : str, default='tensorflow'
            Deep learning backend
        random_state : int, optional
            Random seed
        """
        super().__init__(input_shape, latent_dim, backend, random_state)

        self.hidden_dims = hidden_dims
        self.activation = activation
        self.beta = beta
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        # Build model
        self._build_model()

    def _build_tensorflow_model(self):
        """Build TensorFlow/Keras model."""
        # Encoder
        encoder_input = keras.Input(shape=self.input_shape)
        x = layers.Flatten()(encoder_input)

        for dim in self.hidden_dims:
            x = layers.Dense(dim)(x)
            if self.use_batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(self.activation)(x)
            x = layers.Dropout(self.dropout_rate)(x)

        # Latent distribution parameters
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)

        # Sampling layer
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

        self.encoder = Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

        # Decoder
        decoder_input = keras.Input(shape=(self.latent_dim,))
        x = decoder_input

        for dim in reversed(self.hidden_dims):
            x = layers.Dense(dim)(x)
            if self.use_batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(self.activation)(x)
            x = layers.Dropout(self.dropout_rate)(x)

        output_dim = np.prod(self.input_shape)
        x = layers.Dense(output_dim, activation="linear")(x)
        decoder_output = layers.Reshape(self.input_shape)(x)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

        # Full VAE
        outputs = self.decoder(self.encoder(encoder_input)[2])
        self.model = Model(encoder_input, outputs, name="vae")

        # Custom loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(encoder_input - outputs),
                axis=list(range(1, len(self.input_shape) + 1)),
            )
        )

        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )

        vae_loss = reconstruction_loss + self.beta * kl_loss
        self.model.add_loss(vae_loss)
        self.model.add_metric(reconstruction_loss, name="reconstruction_loss")
        self.model.add_metric(kl_loss, name="kl_loss")

    def _build_pytorch_model(self):
        """Build PyTorch model."""
        raise NotImplementedError(
            "PyTorch backend for VariationalAutoencoder not yet implemented"
        )

    def _build_model(self):
        """Build model based on backend."""
        if self.backend == "tensorflow":
            self._build_tensorflow_model()
        else:
            self._build_pytorch_model()

    def encode(self, X: np.ndarray, return_distribution: bool = False) -> np.ndarray:
        """
        Encode signals to latent space.

        Parameters
        ----------
        X : np.ndarray
            Input signals
        return_distribution : bool, default=False
            If True, return (z_mean, z_log_var, z). If False, return only z.

        Returns
        -------
        np.ndarray or tuple
            Latent representations
        """
        if self.backend == "tensorflow":
            z_mean, z_log_var, z = self.encoder.predict(X, verbose=0)
            if return_distribution:
                return z_mean, z_log_var, z
            return z
        else:
            raise NotImplementedError()

    def fit(self, X: np.ndarray, **kwargs):
        """Train the VAE. See StandardAutoencoder.fit() for parameters."""
        if self.backend == "tensorflow":
            # Compile model
            self.model.compile(optimizer="adam")

            # Get training parameters
            epochs = kwargs.get("epochs", 100)
            batch_size = kwargs.get("batch_size", 32)
            validation_split = kwargs.get("validation_split", 0.2)
            validation_data = kwargs.get("validation_data", None)
            verbose = kwargs.get("verbose", 1)
            callbacks = kwargs.get("callbacks", None)

            # Default callbacks
            if callbacks is None:
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=15, restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6
                    ),
                ]

            # Train
            self.history = self.model.fit(
                X,
                X,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split if validation_data is None else 0,
                validation_data=(
                    (validation_data[0], validation_data[0])
                    if validation_data is not None
                    else None
                ),
                callbacks=callbacks,
                verbose=verbose,
            )

        return self


class DenoisingAutoencoder(StandardAutoencoder):
    """
    Denoising Autoencoder (DAE) for signal cleaning.

    Trained to reconstruct clean signals from noisy inputs.

    Architecture: Same as StandardAutoencoder
    Training: Add noise to inputs, train to reconstruct clean signals

    Examples
    --------
    >>> from vitalDSP.ml_models.autoencoder import DenoisingAutoencoder
    >>> import numpy as np
    >>>
    >>> # Clean ECG signals
    >>> X_clean = np.random.randn(1000, 500)
    >>>
    >>> # Create and train denoising autoencoder
    >>> dae = DenoisingAutoencoder(
    ...     input_shape=(500,),
    ...     latent_dim=32,
    ...     noise_type='gaussian',
    ...     noise_level=0.1
    ... )
    >>> dae.fit(X_clean, epochs=100)
    >>>
    >>> # Denoise signals
    >>> X_noisy = X_clean + 0.1 * np.random.randn(*X_clean.shape)
    >>> X_denoised = dae.predict(X_noisy)
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        latent_dim: int = 32,
        hidden_dims: List[int] = [256, 128, 64],
        noise_type: str = "gaussian",
        noise_level: float = 0.1,
        **kwargs,
    ):
        """
        Initialize denoising autoencoder.

        Parameters
        ----------
        input_shape : tuple
            Shape of input signals
        latent_dim : int, default=32
            Dimensionality of latent space
        hidden_dims : list, default=[256, 128, 64]
            Dimensions of hidden layers
        noise_type : str, default='gaussian'
            Type of noise ('gaussian', 'uniform', 'salt_pepper')
        noise_level : float, default=0.1
            Noise intensity
        **kwargs
            Additional arguments passed to StandardAutoencoder
        """
        super().__init__(input_shape, latent_dim, hidden_dims, **kwargs)

        self.noise_type = noise_type
        self.noise_level = noise_level

    def _add_noise(self, X: np.ndarray) -> np.ndarray:
        """Add noise to signals."""
        if self.noise_type == "gaussian":
            noise = np.random.normal(0, self.noise_level, X.shape)
            return X + noise
        elif self.noise_type == "uniform":
            noise = np.random.uniform(-self.noise_level, self.noise_level, X.shape)
            return X + noise
        elif self.noise_type == "salt_pepper":
            noisy = X.copy()
            # Salt
            salt_mask = np.random.random(X.shape) < self.noise_level / 2
            noisy[salt_mask] = np.max(X)
            # Pepper
            pepper_mask = np.random.random(X.shape) < self.noise_level / 2
            noisy[pepper_mask] = np.min(X)
            return noisy
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def fit(self, X: np.ndarray, **kwargs):
        """
        Train the denoising autoencoder.

        Parameters
        ----------
        X : np.ndarray
            Clean training signals
        **kwargs
            Additional arguments passed to StandardAutoencoder.fit()

        Returns
        -------
        self
        """
        # Add noise to inputs
        X_noisy = self._add_noise(X)

        if self.backend == "tensorflow":
            # Compile model
            self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

            # Get training parameters
            epochs = kwargs.get("epochs", 100)
            batch_size = kwargs.get("batch_size", 32)
            validation_split = kwargs.get("validation_split", 0.2)
            validation_data = kwargs.get("validation_data", None)
            verbose = kwargs.get("verbose", 1)
            callbacks = kwargs.get("callbacks", None)

            # Prepare validation data
            if validation_data is not None:
                val_noisy = self._add_noise(validation_data[0])
                validation_data = (val_noisy, validation_data[0])

            # Default callbacks
            if callbacks is None:
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=10, restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
                    ),
                ]

            # Train with noisy inputs and clean targets
            self.history = self.model.fit(
                X_noisy,
                X,  # Noisy input, clean target
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split if validation_data is None else 0,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=verbose,
            )

        return self

    def denoise(self, X: np.ndarray) -> np.ndarray:
        """
        Denoise signals.

        Parameters
        ----------
        X : np.ndarray
            Noisy signals

        Returns
        -------
        np.ndarray
            Denoised signals
        """
        if self.backend == "tensorflow":
            return self.model.predict(X, verbose=0)
        else:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                denoised = self.model(X_tensor)
                return denoised.cpu().numpy()


# Convenience functions
def detect_anomalies(
    X: np.ndarray,
    autoencoder_type: str = "standard",
    contamination: float = 0.1,
    **autoencoder_kwargs,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Quick anomaly detection using autoencoders.

    Parameters
    ----------
    X : np.ndarray
        Input signals
    autoencoder_type : str, default='standard'
        Type of autoencoder ('standard', 'conv', 'lstm', 'vae')
    contamination : float, default=0.1
        Expected proportion of anomalies
    **autoencoder_kwargs
        Arguments passed to autoencoder constructor

    Returns
    -------
    anomalies : np.ndarray
        Boolean array indicating anomalies
    scores : np.ndarray
        Anomaly scores
    threshold : float
        Threshold used for detection

    Examples
    --------
    >>> from vitalDSP.ml_models.autoencoder import detect_anomalies
    >>> import numpy as np
    >>>
    >>> # Generate signals with anomalies
    >>> X_normal = np.random.randn(900, 500)
    >>> X_anomaly = np.random.randn(100, 500) * 3  # Larger variance
    >>> X = np.vstack([X_normal, X_anomaly])
    >>>
    >>> # Detect anomalies
    >>> anomalies, scores, threshold = detect_anomalies(
    ...     X,
    ...     autoencoder_type='standard',
    ...     contamination=0.1,
    ...     latent_dim=32
    ... )
    >>> print(f"Detected {anomalies.sum()} anomalies")
    """
    # Select autoencoder class
    if autoencoder_type == "standard":
        ae_class = StandardAutoencoder
    elif autoencoder_type == "conv":
        ae_class = ConvolutionalAutoencoder
    elif autoencoder_type == "lstm":
        ae_class = LSTMAutoencoder
    elif autoencoder_type == "vae":
        ae_class = VariationalAutoencoder
    else:
        raise ValueError(f"Unknown autoencoder type: {autoencoder_type}")

    # Determine input shape
    input_shape = X.shape[1:]

    # Create and train autoencoder
    ae = ae_class(input_shape=input_shape, **autoencoder_kwargs)

    # Split data for training
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]

    # Train
    ae.fit(X_train, epochs=50, verbose=0)

    # Detect anomalies on full dataset
    return ae.detect_anomalies(X, contamination=contamination)


def denoise_signal(
    X_noisy: np.ndarray,
    X_clean: Optional[np.ndarray] = None,
    noise_type: str = "gaussian",
    noise_level: float = 0.1,
    **autoencoder_kwargs,
) -> np.ndarray:
    """
    Denoise signals using denoising autoencoder.

    Parameters
    ----------
    X_noisy : np.ndarray
        Noisy signals to denoise
    X_clean : np.ndarray, optional
        Clean signals for training. If None, use X_noisy as both input and target.
    noise_type : str, default='gaussian'
        Type of noise
    noise_level : float, default=0.1
        Noise intensity
    **autoencoder_kwargs
        Arguments passed to DenoisingAutoencoder

    Returns
    -------
    np.ndarray
        Denoised signals

    Examples
    --------
    >>> from vitalDSP.ml_models.autoencoder import denoise_signal
    >>> import numpy as np
    >>>
    >>> # Generate clean signals
    >>> X_clean = np.random.randn(1000, 500)
    >>>
    >>> # Add noise
    >>> X_noisy = X_clean + 0.2 * np.random.randn(*X_clean.shape)
    >>>
    >>> # Denoise
    >>> X_denoised = denoise_signal(
    ...     X_noisy,
    ...     X_clean=X_clean,
    ...     noise_type='gaussian',
    ...     noise_level=0.1,
    ...     latent_dim=32
    ... )
    """
    input_shape = X_noisy.shape[1:]

    # Create denoising autoencoder
    dae = DenoisingAutoencoder(
        input_shape=input_shape,
        noise_type=noise_type,
        noise_level=noise_level,
        **autoencoder_kwargs,
    )

    # Train
    if X_clean is not None:
        # Use clean signals for training
        dae.fit(X_clean, epochs=100, verbose=0)
    else:
        # Use noisy signals (assumes they have some clean structure)
        dae.fit(X_noisy, epochs=100, verbose=0)

    # Denoise
    return dae.denoise(X_noisy)
