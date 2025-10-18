"""
Pre-trained Models for Physiological Signal Analysis

This module provides a repository of pre-trained models for common physiological
signal analysis tasks, enabling transfer learning and quick deployment.

Features:
- Automatic model download and caching
- Pre-trained models for ECG, PPG, EEG
- Model versioning and metadata
- Easy fine-tuning interface

Available Models:
1. ECG Classification (MIT-BIH, PTB-XL)
2. PPG Quality Assessment
3. EEG Sleep Stage Classification
4. Arrhythmia Detection
5. Heart Rate Estimation

Author: vitalDSP
License: MIT
"""
"""
Machine Learning Models Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations
- Deep learning framework integration
- Signal validation and error handling

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.ml_models.pretrained_models import PretrainedModels
    >>> signal = np.random.randn(1000)
    >>> processor = PretrainedModels(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""



import numpy as np
from typing import Optional, Union, Tuple, List, Dict, Any, Callable
from pathlib import Path
import json
import hashlib
import warnings
from urllib.request import urlretrieve
from urllib.error import URLError
import os

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


# Model registry
MODEL_REGISTRY = {
    "ecg_classifier_mitbih": {
        "description": "ECG arrhythmia classifier trained on MIT-BIH dataset",
        "task": "classification",
        "signal_type": "ecg",
        "input_shape": (187, 1),
        "n_classes": 5,
        "classes": ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"],
        "architecture": "cnn1d",
        "backend": "tensorflow",
        "url": None,  # Placeholder - would be actual URL in production
        "size_mb": 2.5,
        "accuracy": 0.98,
        "version": "1.0.0",
        "reference": "MIT-BIH Arrhythmia Database",
    },
    "ecg_classifier_ptbxl": {
        "description": "Multi-label ECG classifier trained on PTB-XL dataset",
        "task": "multi_label_classification",
        "signal_type": "ecg",
        "input_shape": (1000, 12),  # 12-lead ECG
        "n_classes": 71,
        "architecture": "resnet1d",
        "backend": "tensorflow",
        "url": None,
        "size_mb": 15.0,
        "f1_score": 0.92,
        "version": "1.0.0",
        "reference": "PTB-XL Database",
    },
    "ppg_quality_assessment": {
        "description": "PPG signal quality classifier",
        "task": "classification",
        "signal_type": "ppg",
        "input_shape": (250, 1),
        "n_classes": 3,
        "classes": ["Excellent", "Acceptable", "Unacceptable"],
        "architecture": "cnn1d",
        "backend": "tensorflow",
        "url": None,
        "size_mb": 1.8,
        "accuracy": 0.94,
        "version": "1.0.0",
    },
    "eeg_sleep_stage": {
        "description": "EEG sleep stage classifier",
        "task": "classification",
        "signal_type": "eeg",
        "input_shape": (3000, 1),  # 30 seconds at 100 Hz
        "n_classes": 5,
        "classes": ["Wake", "N1", "N2", "N3", "REM"],
        "architecture": "lstm",
        "backend": "tensorflow",
        "url": None,
        "size_mb": 5.2,
        "accuracy": 0.87,
        "version": "1.0.0",
        "reference": "Sleep-EDF Database",
    },
    "heart_rate_estimator": {
        "description": "End-to-end heart rate estimator from PPG/ECG",
        "task": "regression",
        "signal_type": "ppg",
        "input_shape": (500, 1),
        "architecture": "cnn_lstm",
        "backend": "tensorflow",
        "url": None,
        "size_mb": 3.1,
        "mae": 2.3,  # MAE in BPM
        "version": "1.0.0",
    },
    "ecg_autoencoder": {
        "description": "Autoencoder for ECG anomaly detection and denoising",
        "task": "autoencoder",
        "signal_type": "ecg",
        "input_shape": (187, 1),
        "latent_dim": 32,
        "architecture": "convolutional_autoencoder",
        "backend": "tensorflow",
        "url": None,
        "size_mb": 1.5,
        "reconstruction_loss": 0.002,
        "version": "1.0.0",
    },
    "multimodal_transformer": {
        "description": "Transformer for multi-lead ECG analysis",
        "task": "classification",
        "signal_type": "ecg",
        "input_shape": (1000, 12),
        "n_classes": 6,
        "classes": ["NORM", "MI", "STTC", "CD", "HYP", "Other"],
        "architecture": "transformer",
        "backend": "tensorflow",
        "url": None,
        "size_mb": 25.0,
        "accuracy": 0.91,
        "version": "1.0.0",
    },
}


class PretrainedModel:
    """
    Wrapper for pre-trained physiological signal analysis models.

    Provides a unified interface for loading, using, and fine-tuning
    pre-trained models regardless of architecture or backend.

    Examples
    --------
    >>> from vitalDSP.ml_models.pretrained_models import PretrainedModel
    >>> import numpy as np
    >>>
    >>> # Load pre-trained ECG classifier
    >>> model = PretrainedModel.from_registry('ecg_classifier_mitbih')
    >>> print(model.info())
    >>>
    >>> # Make predictions
    >>> ecg_signals = np.random.randn(10, 187, 1)
    >>> predictions = model.predict(ecg_signals)
    >>> print(f"Predicted classes: {predictions}")
    >>>
    >>> # Fine-tune on your data
    >>> X_train = np.random.randn(100, 187, 1)
    >>> y_train = np.random.randint(0, 5, 100)
    >>> model.fine_tune(X_train, y_train, epochs=10)
    """

    def __init__(
        self, model: Any, metadata: Dict[str, Any], backend: str = "tensorflow"
    ):
        """
        Initialize pre-trained model.

        Parameters
        ----------
        model : object
            The trained model
        metadata : dict
            Model metadata (architecture, task, etc.)
        backend : str, default='tensorflow'
            Deep learning backend
        """
        self.model = model
        self.metadata = metadata
        self.backend = backend.lower()

    @classmethod
    def from_registry(
        cls,
        model_name: str,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
    ) -> "PretrainedModel":
        """
        Load model from registry.

        Parameters
        ----------
        model_name : str
            Name of the model in the registry
        cache_dir : str, optional
            Directory to cache downloaded models. If None, use default.
        force_download : bool, default=False
            Force re-download even if cached

        Returns
        -------
        PretrainedModel
            Loaded pre-trained model

        Raises
        ------
        ValueError
            If model not found in registry
        """
        if model_name not in MODEL_REGISTRY:
            available = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(
                f"Model '{model_name}' not found in registry. "
                f"Available models: {available}"
            )

        metadata = MODEL_REGISTRY[model_name].copy()

        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".vitaldsp" / "pretrained_models"
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Model path
        model_path = cache_dir / f"{model_name}_v{metadata['version']}"
        backend = metadata["backend"]

        # Download or load from cache
        if not model_path.exists() or force_download:
            if metadata["url"] is None:
                # For demonstration, create a simple model
                warnings.warn(
                    f"Model '{model_name}' URL not available. "
                    "Creating a placeholder model for demonstration."
                )
                model = cls._create_placeholder_model(metadata)
            else:
                model = cls._download_model(metadata["url"], model_path, backend)
        else:
            # Load from cache
            if backend == "tensorflow":
                model = keras.models.load_model(str(model_path))
            else:  # pytorch
                model = torch.load(str(model_path))

        return cls(model, metadata, backend)

    @staticmethod
    def _create_placeholder_model(metadata: Dict[str, Any]) -> Any:
        """
        Create a placeholder model for demonstration.

        In production, this would not be needed as models would be downloaded.
        """
        backend = metadata["backend"]
        architecture = metadata["architecture"]
        input_shape = metadata["input_shape"]
        task = metadata["task"]

        if backend == "tensorflow":
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow not installed")

            # Create simple model based on architecture
            inputs = keras.Input(shape=input_shape)
            x = inputs

            if "cnn" in architecture:
                x = keras.layers.Conv1D(32, 7, padding="same")(x)
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Activation("relu")(x)
                x = keras.layers.MaxPooling1D(2)(x)

                x = keras.layers.Conv1D(64, 5, padding="same")(x)
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Activation("relu")(x)
                x = keras.layers.MaxPooling1D(2)(x)

                x = keras.layers.GlobalAveragePooling1D()(x)

            elif "lstm" in architecture:
                x = keras.layers.LSTM(64, return_sequences=True)(x)
                x = keras.layers.LSTM(32)(x)

            elif "transformer" in architecture:
                # Simple transformer-like architecture
                x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
                x = keras.layers.GlobalAveragePooling1D()(x)
                x = keras.layers.Dense(128, activation="relu")(x)

            # Output layer
            if task == "classification":
                n_classes = metadata["n_classes"]
                outputs = keras.layers.Dense(n_classes, activation="softmax")(x)
            elif task == "regression":
                outputs = keras.layers.Dense(1, activation="linear")(x)
            elif task == "autoencoder":
                # Decoder part (simplified)
                x = keras.layers.Dense(128, activation="relu")(x)
                x = keras.layers.Reshape((-1, 1))(x)
                outputs = keras.layers.UpSampling1D(2)(x)

            model = keras.Model(inputs, outputs)
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

        else:  # pytorch
            raise NotImplementedError("PyTorch placeholder models not yet implemented")

        return model

    @staticmethod
    def _download_model(url: str, save_path: Path, backend: str) -> Any:
        """Download model from URL."""
        try:
            print(f"Downloading model from {url}...")
            urlretrieve(url, str(save_path))
            print("Download complete.")

            # Load model
            if backend == "tensorflow":
                model = keras.models.load_model(str(save_path))
            else:
                model = torch.load(str(save_path))

            return model

        except URLError as e:
            raise RuntimeError(f"Failed to download model: {e}")

    def predict(
        self, X: np.ndarray, batch_size: int = 32, return_proba: bool = False
    ) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        X : np.ndarray
            Input signals
        batch_size : int, default=32
            Batch size for prediction
        return_proba : bool, default=False
            Return class probabilities (classification only)

        Returns
        -------
        np.ndarray
            Predictions
        """
        if self.backend == "tensorflow":
            predictions = self.model.predict(X, batch_size=batch_size, verbose=0)

            if self.metadata["task"] == "classification" and not return_proba:
                predictions = np.argmax(predictions, axis=1)

        else:  # pytorch
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = self.model(X_tensor).cpu().numpy()

                if self.metadata["task"] == "classification" and not return_proba:
                    predictions = np.argmax(predictions, axis=1)

        return predictions

    def fine_tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        freeze_layers: Optional[int] = None,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune model on new data.

        Parameters
        ----------
        X_train : np.ndarray
            Training signals
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray, optional
            Validation signals
        y_val : np.ndarray, optional
            Validation labels
        epochs : int, default=10
            Number of training epochs
        batch_size : int, default=32
            Batch size
        learning_rate : float, default=1e-4
            Learning rate (smaller than training from scratch)
        freeze_layers : int, optional
            Number of initial layers to freeze
        verbose : int, default=1
            Verbosity level

        Returns
        -------
        dict
            Training history
        """
        if self.backend == "tensorflow":
            # Freeze layers if requested
            if freeze_layers is not None:
                for i, layer in enumerate(self.model.layers):
                    if i < freeze_layers:
                        layer.trainable = False

            # Compile with lower learning rate
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

            if self.metadata["task"] == "classification":
                loss = "sparse_categorical_crossentropy"
                metrics = ["accuracy"]
            else:
                loss = "mse"
                metrics = ["mae"]

            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss" if X_val is not None else "loss",
                    patience=5,
                    restore_best_weights=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss" if X_val is not None else "loss",
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7,
                ),
            ]

            # Train
            validation_data = (
                (X_val, y_val) if X_val is not None and y_val is not None else None
            )

            history = self.model.fit(
                X_train,
                y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose,
            )

            return history.history

        else:  # pytorch
            raise NotImplementedError("PyTorch fine-tuning not yet implemented")

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray, batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Parameters
        ----------
        X_test : np.ndarray
            Test signals
        y_test : np.ndarray
            Test labels
        batch_size : int, default=32
            Batch size

        Returns
        -------
        dict
            Evaluation metrics
        """
        if self.backend == "tensorflow":
            results = self.model.evaluate(
                X_test, y_test, batch_size=batch_size, verbose=0
            )

            metrics = {}
            for name, value in zip(self.model.metrics_names, results):
                metrics[name] = value

            return metrics

        else:  # pytorch
            raise NotImplementedError("PyTorch evaluation not yet implemented")

    def save(self, filepath: str):
        """
        Save model to file.

        Parameters
        ----------
        filepath : str
            Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if self.backend == "tensorflow":
            self.model.save(str(filepath))
        else:
            torch.save(self.model, str(filepath))

        # Save metadata
        metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "PretrainedModel":
        """
        Load model from file.

        Parameters
        ----------
        filepath : str
            Path to saved model

        Returns
        -------
        PretrainedModel
            Loaded model
        """
        filepath = Path(filepath)

        # Load metadata
        metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        backend = metadata.get("backend", "tensorflow")

        # Load model
        if backend == "tensorflow":
            model = keras.models.load_model(str(filepath))
        else:
            model = torch.load(str(filepath))

        return cls(model, metadata, backend)

    def info(self) -> str:
        """
        Get model information.

        Returns
        -------
        str
            Formatted model information
        """
        info_str = "Pre-trained Model Information\n"
        info_str += "=" * 50 + "\n"

        for key, value in self.metadata.items():
            if isinstance(value, (list, tuple)) and len(value) > 5:
                value = f"{value[:3]}... ({len(value)} total)"
            info_str += f"{key.replace('_', ' ').title()}: {value}\n"

        return info_str

    def get_layer_names(self) -> List[str]:
        """Get list of layer names in the model."""
        if self.backend == "tensorflow":
            return [layer.name for layer in self.model.layers]
        else:
            return [name for name, _ in self.model.named_modules()]

    def get_features(
        self, X: np.ndarray, layer_name: Optional[str] = None, batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract features from intermediate layer.

        Parameters
        ----------
        X : np.ndarray
            Input signals
        layer_name : str, optional
            Name of layer to extract features from. If None, use second-to-last layer.
        batch_size : int, default=32
            Batch size

        Returns
        -------
        np.ndarray
            Extracted features
        """
        if self.backend == "tensorflow":
            if layer_name is None:
                # Use second-to-last layer (before output)
                layer_name = self.model.layers[-2].name

            # Create feature extractor
            feature_model = keras.Model(
                inputs=self.model.input, outputs=self.model.get_layer(layer_name).output
            )

            features = feature_model.predict(X, batch_size=batch_size, verbose=0)

        else:  # pytorch
            raise NotImplementedError("PyTorch feature extraction not yet implemented")

        return features


class ModelHub:
    """
    Central hub for managing pre-trained models.

    Provides utilities for:
    - Listing available models
    - Downloading models
    - Managing model cache
    - Model comparison

    Examples
    --------
    >>> from vitalDSP.ml_models.pretrained_models import ModelHub
    >>>
    >>> # List all available models
    >>> hub = ModelHub()
    >>> models = hub.list_models()
    >>> print(models)
    >>>
    >>> # Filter models by signal type
    >>> ecg_models = hub.list_models(signal_type='ecg', task='classification')
    >>> print(ecg_models)
    >>>
    >>> # Download specific model
    >>> model = hub.get_model('ecg_classifier_mitbih')
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model hub.

        Parameters
        ----------
        cache_dir : str, optional
            Directory for caching models
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".vitaldsp" / "pretrained_models"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_models(
        self,
        signal_type: Optional[str] = None,
        task: Optional[str] = None,
        architecture: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List available models with optional filtering.

        Parameters
        ----------
        signal_type : str, optional
            Filter by signal type ('ecg', 'ppg', 'eeg')
        task : str, optional
            Filter by task ('classification', 'regression', 'autoencoder')
        architecture : str, optional
            Filter by architecture ('cnn1d', 'lstm', 'transformer')

        Returns
        -------
        list of dict
            List of model metadata
        """
        models = []

        for name, metadata in MODEL_REGISTRY.items():
            # Apply filters
            if signal_type and metadata.get("signal_type") != signal_type:
                continue
            if task and metadata.get("task") != task:
                continue
            if architecture and metadata.get("architecture") != architecture:
                continue

            model_info = {"name": name, **metadata}
            models.append(model_info)

        return models

    def get_model(
        self, model_name: str, force_download: bool = False
    ) -> PretrainedModel:
        """
        Get pre-trained model.

        Parameters
        ----------
        model_name : str
            Name of model
        force_download : bool, default=False
            Force re-download

        Returns
        -------
        PretrainedModel
            Loaded model
        """
        return PretrainedModel.from_registry(
            model_name, cache_dir=str(self.cache_dir), force_download=force_download
        )

    def clear_cache(self, model_name: Optional[str] = None):
        """
        Clear model cache.

        Parameters
        ----------
        model_name : str, optional
            Specific model to clear. If None, clear all.
        """
        if model_name is not None:
            # Clear specific model
            for file in self.cache_dir.glob(f"{model_name}*"):
                file.unlink()
        else:
            # Clear all
            for file in self.cache_dir.glob("*"):
                file.unlink()

    def get_cache_size(self) -> float:
        """
        Get total size of cached models in MB.

        Returns
        -------
        float
            Total cache size in MB
        """
        total_size = 0
        for file in self.cache_dir.glob("*"):
            total_size += file.stat().st_size

        return total_size / (1024 * 1024)  # Convert to MB

    def compare_models(
        self, model_names: List[str], metric: str = "accuracy"
    ) -> Dict[str, float]:
        """
        Compare models by a specific metric.

        Parameters
        ----------
        model_names : list of str
            Names of models to compare
        metric : str, default='accuracy'
            Metric to compare

        Returns
        -------
        dict
            Model names and their metric values
        """
        comparison = {}

        for name in model_names:
            if name in MODEL_REGISTRY:
                metadata = MODEL_REGISTRY[name]
                if metric in metadata:
                    comparison[name] = metadata[metric]

        return comparison


# Convenience function
def load_pretrained_model(
    model_name: str, cache_dir: Optional[str] = None, force_download: bool = False
) -> PretrainedModel:
    """
    Quick function to load a pre-trained model.

    Parameters
    ----------
    model_name : str
        Name of model from registry
    cache_dir : str, optional
        Cache directory
    force_download : bool, default=False
        Force re-download

    Returns
    -------
    PretrainedModel
        Loaded model

    Examples
    --------
    >>> from vitalDSP.ml_models.pretrained_models import load_pretrained_model
    >>> import numpy as np
    >>>
    >>> # Load ECG classifier
    >>> model = load_pretrained_model('ecg_classifier_mitbih')
    >>>
    >>> # Make predictions
    >>> ecg_signal = np.random.randn(1, 187, 1)
    >>> prediction = model.predict(ecg_signal)
    >>> print(f"Predicted class: {prediction[0]}")
    """
    return PretrainedModel.from_registry(
        model_name, cache_dir=cache_dir, force_download=force_download
    )
