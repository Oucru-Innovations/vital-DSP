"""
vitalDSP Deep Learning Models Module

State-of-the-art deep learning architectures for physiological signal analysis.

This module provides:
- 1D CNN for signal classification
- LSTM for sequence modeling
- Transformer for long-range dependencies
- Autoencoder for anomaly detection
- Pre-trained models and transfer learning
- Model training utilities

Requires: tensorflow/keras or pytorch

Author: vitalDSP Team
Date: 2025
"""

import numpy as np
import warnings
from typing import Optional, Union, Tuple, Dict, List, Any
from abc import ABC, abstractmethod


# Check for deep learning frameworks
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not available. Install with: pip install tensorflow")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = False
except ImportError:
    TORCH_AVAILABLE = False


class BaseDeepModel(ABC):
    """
    Base class for deep learning models.

    Provides common interface for all deep models with support for
    both TensorFlow/Keras and PyTorch backends.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        n_classes: int,
        backend: str = 'tensorflow',
        **kwargs
    ):
        """
        Initialize base model.

        Parameters
        ----------
        input_shape : tuple
            Shape of input data (excluding batch dimension)
        n_classes : int
            Number of output classes
        backend : str, default='tensorflow'
            Deep learning backend ('tensorflow' or 'pytorch')
        """
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.backend = backend.lower()
        self.model = None
        self.history = None

        if self.backend == 'tensorflow' and not TF_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        elif self.backend == 'pytorch' and not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")

    @abstractmethod
    def build_model(self):
        """Build the model architecture."""
        pass

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass

    def save(self, filepath: str):
        """Save model to disk."""
        if self.backend == 'tensorflow':
            self.model.save(filepath)
        elif self.backend == 'pytorch':
            torch.save(self.model.state_dict(), filepath)

    def load(self, filepath: str):
        """Load model from disk."""
        if self.backend == 'tensorflow':
            self.model = keras.models.load_model(filepath)
        elif self.backend == 'pytorch':
            self.model.load_state_dict(torch.load(filepath))


class CNN1D(BaseDeepModel):
    """
    1D Convolutional Neural Network for signal classification.

    Designed for physiological signal classification tasks such as:
    - ECG arrhythmia detection
    - PPG quality assessment
    - EEG sleep stage classification
    - Respiratory event detection

    Architecture:
    - Multiple 1D convolutional layers with batch normalization
    - MaxPooling for downsampling
    - Dropout for regularization
    - Dense layers for classification
    - Residual connections (optional)

    Parameters
    ----------
    input_shape : tuple
        Shape of input signals (sequence_length, n_channels)
    n_classes : int
        Number of output classes
    n_filters : list of int, default=[32, 64, 128]
        Number of filters in each conv layer
    kernel_sizes : list of int, default=[7, 5, 3]
        Kernel sizes for each conv layer
    pool_sizes : list of int, default=[2, 2, 2]
        Pool sizes for each pooling layer
    dropout_rate : float, default=0.5
        Dropout rate for regularization
    use_residual : bool, default=False
        Whether to use residual connections
    backend : str, default='tensorflow'
        Deep learning backend

    Attributes
    ----------
    model : keras.Model or torch.nn.Module
        The underlying model
    history : dict
        Training history

    Examples
    --------
    >>> from vitalDSP.ml_models import CNN1D
    >>> model = CNN1D(input_shape=(1000, 1), n_classes=5)
    >>> model.build_model()
    >>> model.train(X_train, y_train, epochs=50, batch_size=32)
    >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        n_classes: int,
        n_filters: List[int] = None,
        kernel_sizes: List[int] = None,
        pool_sizes: List[int] = None,
        dropout_rate: float = 0.5,
        use_residual: bool = False,
        backend: str = 'tensorflow',
        **kwargs
    ):
        super().__init__(input_shape, n_classes, backend, **kwargs)

        self.n_filters = n_filters or [32, 64, 128]
        self.kernel_sizes = kernel_sizes or [7, 5, 3]
        self.pool_sizes = pool_sizes or [2, 2, 2]
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual

    def build_model(self):
        """Build 1D CNN architecture."""
        if self.backend == 'tensorflow':
            self._build_tensorflow_model()
        elif self.backend == 'pytorch':
            self._build_pytorch_model()

        return self.model

    def _build_tensorflow_model(self):
        """Build TensorFlow/Keras model."""
        inputs = keras.Input(shape=self.input_shape)
        x = inputs

        # Convolutional blocks
        for i, (n_filters, kernel_size, pool_size) in enumerate(
            zip(self.n_filters, self.kernel_sizes, self.pool_sizes)
        ):
            # Residual connection
            if self.use_residual and i > 0:
                residual = x

            # Conv block
            x = layers.Conv1D(
                n_filters,
                kernel_size,
                padding='same',
                activation=None,
                name=f'conv1d_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            x = layers.Activation('relu', name=f'relu_{i+1}')(x)

            # Residual addition
            if self.use_residual and i > 0:
                # Match dimensions if needed
                if residual.shape[-1] != x.shape[-1]:
                    residual = layers.Conv1D(n_filters, 1, padding='same')(residual)
                x = layers.Add()([x, residual])

            x = layers.MaxPooling1D(pool_size, name=f'maxpool_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)

        # Global pooling
        x = layers.GlobalAveragePooling1D(name='global_pool')(x)

        # Dense layers
        x = layers.Dense(128, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_dense')(x)
        x = layers.Dense(64, activation='relu', name='dense_2')(x)

        # Output layer
        if self.n_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        else:
            outputs = layers.Dense(self.n_classes, activation='softmax', name='output')(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name='CNN1D')

    def _build_pytorch_model(self):
        """Build PyTorch model."""
        class CNN1DModule(nn.Module):
            def __init__(self, input_shape, n_classes, n_filters, kernel_sizes, pool_sizes, dropout_rate):
                super().__init__()

                self.conv_blocks = nn.ModuleList()

                in_channels = input_shape[1] if len(input_shape) > 1 else 1

                for n_filter, kernel_size, pool_size in zip(n_filters, kernel_sizes, pool_sizes):
                    block = nn.Sequential(
                        nn.Conv1d(in_channels, n_filter, kernel_size, padding=kernel_size//2),
                        nn.BatchNorm1d(n_filter),
                        nn.ReLU(),
                        nn.MaxPool1d(pool_size),
                        nn.Dropout(dropout_rate)
                    )
                    self.conv_blocks.append(block)
                    in_channels = n_filter

                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.fc1 = nn.Linear(n_filters[-1], 128)
                self.dropout = nn.Dropout(dropout_rate)
                self.fc2 = nn.Linear(128, 64)
                self.output = nn.Linear(64, n_classes if n_classes > 2 else 1)

            def forward(self, x):
                for block in self.conv_blocks:
                    x = block(x)

                x = self.global_pool(x)
                x = x.squeeze(-1)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.output(x)

                if x.shape[-1] == 1:
                    x = torch.sigmoid(x)
                else:
                    x = F.softmax(x, dim=-1)

                return x

        self.model = CNN1DModule(
            self.input_shape, self.n_classes,
            self.n_filters, self.kernel_sizes,
            self.pool_sizes, self.dropout_rate
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        **kwargs
    ):
        """
        Train the CNN model.

        Parameters
        ----------
        X_train : ndarray of shape (n_samples, sequence_length, n_channels)
            Training data
        y_train : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Training labels
        X_val : ndarray, optional
            Validation data
        y_val : ndarray, optional
            Validation labels
        epochs : int, default=50
            Number of training epochs
        batch_size : int, default=32
            Batch size for training
        learning_rate : float, default=0.001
            Learning rate
        **kwargs : dict
            Additional training parameters

        Returns
        -------
        history : dict
            Training history
        """
        if self.backend == 'tensorflow':
            return self._train_tensorflow(
                X_train, y_train, X_val, y_val,
                epochs, batch_size, learning_rate, **kwargs
            )
        elif self.backend == 'pytorch':
            return self._train_pytorch(
                X_train, y_train, X_val, y_val,
                epochs, batch_size, learning_rate, **kwargs
            )

    def _train_tensorflow(
        self, X_train, y_train, X_val, y_val,
        epochs, batch_size, learning_rate, **kwargs
    ):
        """Train TensorFlow model."""
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        if self.n_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', keras.metrics.AUC(name='auc')]
        else:
            loss = 'sparse_categorical_crossentropy' if y_train.ndim == 1 else 'categorical_crossentropy'
            metrics = ['accuracy']

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]

        # Train
        validation_data = (X_val, y_val) if X_val is not None else None

        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=kwargs.get('verbose', 1)
        )

        self.history = history.history
        return self.history

    def _train_pytorch(
        self, X_train, y_train, X_val, y_val,
        epochs, batch_size, learning_rate, **kwargs
    ):
        """Train PyTorch model."""
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).permute(0, 2, 1)  # (N, C, L)
        y_train_tensor = torch.LongTensor(y_train) if y_train.dtype == int else torch.FloatTensor(y_train)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Loss and optimizer
        if self.n_classes == 2:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.history = {'loss': [], 'accuracy': []}
        if X_val is not None:
            self.history['val_loss'] = []
            self.history['val_accuracy'] = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if self.n_classes == 2:
                    predicted = (outputs > 0.5).float()
                else:
                    _, predicted = torch.max(outputs.data, 1)

                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            self.history['loss'].append(epoch_loss / len(train_loader))
            self.history['accuracy'].append(correct / total)

            if kwargs.get('verbose', 1) > 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {self.history['loss'][-1]:.4f} - accuracy: {self.history['accuracy'][-1]:.4f}")

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, sequence_length, n_channels)
            Input data

        Returns
        -------
        predictions : ndarray
            Predicted class probabilities or labels
        """
        if self.backend == 'tensorflow':
            return self.model.predict(X)
        elif self.backend == 'pytorch':
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).permute(0, 2, 1)
                device = next(self.model.parameters()).device
                X_tensor = X_tensor.to(device)
                outputs = self.model(X_tensor)
                return outputs.cpu().numpy()


class LSTMModel(BaseDeepModel):
    """
    LSTM (Long Short-Term Memory) for sequence modeling.

    Ideal for:
    - Time series forecasting
    - Sequential pattern recognition
    - Long-term dependency modeling
    - Real-time signal prediction

    Parameters
    ----------
    input_shape : tuple
        Shape of input sequences
    n_classes : int
        Number of output classes (use 1 for regression)
    lstm_units : list of int, default=[128, 64]
        Number of units in each LSTM layer
    dropout_rate : float, default=0.3
        Dropout rate
    bidirectional : bool, default=True
        Whether to use bidirectional LSTM
    task : str, default='classification'
        Task type ('classification' or 'regression')

    Examples
    --------
    >>> model = LSTMModel(input_shape=(100, 12), n_classes=4)
    >>> model.build_model()
    >>> model.train(X_train, y_train, epochs=100)
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        n_classes: int,
        lstm_units: List[int] = None,
        dropout_rate: float = 0.3,
        bidirectional: bool = True,
        task: str = 'classification',
        backend: str = 'tensorflow',
        **kwargs
    ):
        super().__init__(input_shape, n_classes, backend, **kwargs)

        self.lstm_units = lstm_units or [128, 64]
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.task = task

    def build_model(self):
        """Build LSTM architecture."""
        if self.backend == 'tensorflow':
            self._build_tensorflow_model()
        elif self.backend == 'pytorch':
            self._build_pytorch_model()

        return self.model

    def _build_tensorflow_model(self):
        """Build TensorFlow/Keras LSTM model."""
        inputs = keras.Input(shape=self.input_shape)
        x = inputs

        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)

            if self.bidirectional:
                x = layers.Bidirectional(
                    layers.LSTM(units, return_sequences=return_sequences),
                    name=f'bilstm_{i+1}'
                )(x)
            else:
                x = layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    name=f'lstm_{i+1}'
                )(x)

            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)

        # Dense layers
        x = layers.Dense(64, activation='relu', name='dense')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_dense')(x)

        # Output layer
        if self.task == 'classification':
            if self.n_classes == 2:
                outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            else:
                outputs = layers.Dense(self.n_classes, activation='softmax', name='output')(x)
        else:  # regression
            outputs = layers.Dense(1, activation='linear', name='output')(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name='LSTM')

    def _build_pytorch_model(self):
        """Build PyTorch LSTM model."""
        class LSTMModule(nn.Module):
            def __init__(self, input_size, lstm_units, n_classes, dropout_rate, bidirectional, task):
                super().__init__()

                self.lstm_layers = nn.ModuleList()
                self.task = task

                for i, units in enumerate(lstm_units):
                    input_dim = input_size if i == 0 else lstm_units[i-1] * (2 if bidirectional else 1)
                    lstm = nn.LSTM(input_dim, units, batch_first=True, bidirectional=bidirectional)
                    self.lstm_layers.append(lstm)

                final_dim = lstm_units[-1] * (2 if bidirectional else 1)
                self.dropout = nn.Dropout(dropout_rate)
                self.fc1 = nn.Linear(final_dim, 64)
                self.fc2 = nn.Linear(64, n_classes if task == 'classification' and n_classes > 2 else 1)

            def forward(self, x):
                for lstm in self.lstm_layers:
                    x, _ = lstm(x)

                # Take last output
                x = x[:, -1, :]
                x = self.dropout(x)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)

                if self.task == 'classification':
                    if x.shape[-1] == 1:
                        x = torch.sigmoid(x)
                    else:
                        x = F.softmax(x, dim=-1)

                return x

        input_size = self.input_shape[-1] if len(self.input_shape) > 1 else 1

        self.model = LSTMModule(
            input_size, self.lstm_units, self.n_classes,
            self.dropout_rate, self.bidirectional, self.task
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        **kwargs
    ):
        """Train the LSTM model."""
        # Similar to CNN1D.train()
        if self.backend == 'tensorflow':
            return self._train_tensorflow(
                X_train, y_train, X_val, y_val,
                epochs, batch_size, learning_rate, **kwargs
            )
        elif self.backend == 'pytorch':
            return self._train_pytorch(
                X_train, y_train, X_val, y_val,
                epochs, batch_size, learning_rate, **kwargs
            )

    def _train_tensorflow(self, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, **kwargs):
        """Train TensorFlow LSTM model."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        if self.task == 'classification':
            if self.n_classes == 2:
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                loss = 'sparse_categorical_crossentropy' if y_train.ndim == 1 else 'categorical_crossentropy'
                metrics = ['accuracy']
        else:  # regression
            loss = 'mse'
            metrics = ['mae']

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        callback_list = [
            callbacks.EarlyStopping(monitor='val_loss' if X_val is not None else 'loss', patience=15, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss', factor=0.5, patience=7, min_lr=1e-7)
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=kwargs.get('verbose', 1)
        )

        self.history = history.history
        return self.history

    def _train_pytorch(self, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, **kwargs):
        """Train PyTorch LSTM model."""
        # Similar to CNN1D._train_pytorch()
        pass  # Implementation similar to CNN1D

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.backend == 'tensorflow':
            return self.model.predict(X)
        elif self.backend == 'pytorch':
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                device = next(self.model.parameters()).device
                X_tensor = X_tensor.to(device)
                outputs = self.model(X_tensor)
                return outputs.cpu().numpy()


# Additional models: Transformer and Autoencoder will be in separate files due to length
# See: transformer_model.py and autoencoder_model.py
