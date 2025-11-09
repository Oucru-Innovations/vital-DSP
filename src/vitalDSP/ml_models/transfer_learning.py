"""
Transfer Learning for Physiological Signal Analysis

This module provides utilities for transfer learning, enabling efficient model
adaptation to new tasks and domains with limited labeled data.

Features:
- Feature extraction (frozen base model)
- Fine-tuning strategies
- Domain adaptation techniques
- Few-shot learning support
- Progressive unfreezing
- Learning rate scheduling

Use Cases:
- Adapt ECG classifier to different populations
- Transfer from large to small datasets
- Cross-domain transfer (e.g., ECG to PPG)
- Personalized model adaptation

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
    >>> from vitalDSP.ml_models.transfer_learning import TransferLearning
    >>> signal = np.random.randn(1000)
    >>> processor = TransferLearning(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""


import numpy as np
from typing import Optional, Union, Tuple, List, Dict, Any, Callable
from pathlib import Path
import warnings

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


class TransferLearningStrategy:
    """
    Base class for transfer learning strategies.

    Implements common functionality for adapting pre-trained models to new tasks.

    Examples
    --------
    >>> from vitalDSP.ml_models.transfer_learning import TransferFeatureExtractor
    >>> from vitalDSP.ml_models.pretrained_models import load_pretrained_model
    >>> import numpy as np
    >>>
    >>> # Load pre-trained model
    >>> base_model = load_pretrained_model('ecg_classifier_mitbih')
    >>>
    >>> # Create feature extractor
    >>> extractor = TransferFeatureExtractor(base_model.model, freeze_base=True)
    >>>
    >>> # Train on new task
    >>> X_train = np.random.randn(100, 187, 1)
    >>> y_train = np.random.randint(0, 3, 100)  # New 3-class task
    >>> extractor.fit(X_train, y_train, n_classes=3, epochs=20)
    """

    def __init__(self, base_model: Any, backend: str = "tensorflow"):
        """
        Initialize transfer learning strategy.

        Parameters
        ----------
        base_model : object
            Pre-trained base model
        backend : str, default='tensorflow'
            Deep learning backend
        """
        self.base_model = base_model
        self.backend = backend.lower()
        self.model = None
        self.history = None

    def freeze_layers(self, n_layers: Optional[int] = None):
        """
        Freeze layers in base model.

        Parameters
        ----------
        n_layers : int, optional
            Number of layers to freeze. If None, freeze all.
        """
        raise NotImplementedError("Subclasses must implement freeze_layers()")

    def unfreeze_layers(self, n_layers: Optional[int] = None):
        """
        Unfreeze layers for fine-tuning.

        Parameters
        ----------
        n_layers : int, optional
            Number of layers to unfreeze from the end. If None, unfreeze all.
        """
        raise NotImplementedError("Subclasses must implement unfreeze_layers()")


class TransferFeatureExtractor(TransferLearningStrategy):
    """
    Feature extraction transfer learning.

    Uses pre-trained model as fixed feature extractor, only training
    new classification/regression head.

    This is the fastest approach and works well when:
    - Source and target tasks are similar
    - Target dataset is small
    - Computational resources are limited

    Examples
    --------
    >>> from vitalDSP.ml_models.transfer_learning import TransferFeatureExtractor
    >>> import numpy as np
    >>> import tensorflow as tf
    >>>
    >>> # Create base model
    >>> base = tf.keras.Sequential([
    ...     tf.keras.layers.Conv1D(32, 7, input_shape=(187, 1)),
    ...     tf.keras.layers.MaxPooling1D(2),
    ...     tf.keras.layers.GlobalAveragePooling1D()
    ... ])
    >>>
    >>> # Create feature extractor
    >>> extractor = TransferFeatureExtractor(base, freeze_base=True)
    >>>
    >>> # Fit to new task
    >>> X = np.random.randn(100, 187, 1)
    >>> y = np.random.randint(0, 3, 100)
    >>> extractor.fit(X, y, n_classes=3, epochs=10)
    """

    def __init__(
        self, base_model: Any, freeze_base: bool = True, backend: str = "tensorflow"
    ):
        """
        Initialize feature extractor.

        Parameters
        ----------
        base_model : object
            Pre-trained base model
        freeze_base : bool, default=True
            Whether to freeze base model
        backend : str, default='tensorflow'
            Deep learning backend
        """
        super().__init__(base_model, backend)
        self.freeze_base = freeze_base

        if freeze_base:
            self.freeze_layers()

    def freeze_layers(self, n_layers: Optional[int] = None):
        """Freeze base model layers."""
        if self.backend == "tensorflow":
            if n_layers is None:
                # Freeze all base model layers
                self.base_model.trainable = False
            else:
                # Freeze first n_layers
                for i, layer in enumerate(self.base_model.layers):
                    if i < n_layers:
                        layer.trainable = False
        else:  # pytorch
            if n_layers is None:
                for param in self.base_model.parameters():
                    param.requires_grad = False
            else:
                for i, (name, param) in enumerate(self.base_model.named_parameters()):
                    if i < n_layers:
                        param.requires_grad = False

    def unfreeze_layers(self, n_layers: Optional[int] = None):
        """Unfreeze base model layers."""
        if self.backend == "tensorflow":
            if n_layers is None:
                self.base_model.trainable = True
            else:
                # Unfreeze last n_layers
                total_layers = len(self.base_model.layers)
                for i, layer in enumerate(self.base_model.layers):
                    if i >= total_layers - n_layers:
                        layer.trainable = True
        else:  # pytorch
            if n_layers is None:
                for param in self.base_model.parameters():
                    param.requires_grad = True
            else:
                params = list(self.base_model.named_parameters())
                for i in range(len(params) - n_layers, len(params)):
                    params[i][1].requires_grad = True

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_classes: Optional[int] = None,
        n_outputs: int = 1,
        task: str = "classification",
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """
        Fit feature extractor to new task.

        Parameters
        ----------
        X_train : np.ndarray
            Training signals
        y_train : np.ndarray
            Training labels
        n_classes : int, optional
            Number of classes (for classification)
        n_outputs : int, default=1
            Number of outputs (for regression)
        task : str, default='classification'
            Task type ('classification' or 'regression')
        X_val : np.ndarray, optional
            Validation signals
        y_val : np.ndarray, optional
            Validation labels
        epochs : int, default=50
            Number of training epochs
        batch_size : int, default=32
            Batch size
        learning_rate : float, default=1e-3
            Learning rate
        verbose : int, default=1
            Verbosity level

        Returns
        -------
        dict
            Training history
        """
        if self.backend == "tensorflow":
            # Build model with new head
            inputs = keras.Input(shape=X_train.shape[1:])
            x = self.base_model(inputs)

            # Ensure we have a flattened feature vector
            if len(x.shape) > 2:
                x = (
                    layers.GlobalAveragePooling1D()(x)
                    if len(x.shape) == 3
                    else layers.Flatten()(x)
                )

            # Add new head
            x = layers.Dense(128, activation="relu")(x)
            x = layers.Dropout(0.3)(x)

            if task == "classification":
                if n_classes is None:
                    n_classes = len(np.unique(y_train))
                outputs = layers.Dense(n_classes, activation="softmax")(x)
                loss = "sparse_categorical_crossentropy"
                metrics = ["accuracy"]
            else:  # regression
                outputs = layers.Dense(n_outputs, activation="linear")(x)
                loss = "mse"
                metrics = ["mae"]

            self.model = keras.Model(inputs, outputs)

            # Compile
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss=loss,
                metrics=metrics,
            )

            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss" if X_val is not None else "loss",
                    patience=10,
                    restore_best_weights=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss" if X_val is not None else "loss",
                    factor=0.5,
                    patience=5,
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

            self.history = history.history
            return self.history

        else:  # pytorch
            raise NotImplementedError("PyTorch backend not yet implemented")

    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.backend == "tensorflow":
            return self.model.predict(X, batch_size=batch_size, verbose=0)
        else:
            raise NotImplementedError("PyTorch backend not yet implemented")


class FineTuner(TransferLearningStrategy):
    """
    Fine-tuning transfer learning.

    Unfreezes part or all of the base model and trains with a small learning rate.

    Best when:
    - Target dataset is moderately sized
    - Target task differs somewhat from source task
    - More accuracy is needed than feature extraction provides

    Strategies:
    1. All-at-once: Unfreeze all layers, train with small LR
    2. Progressive: Gradually unfreeze layers from top to bottom
    3. Discriminative: Use different learning rates for different layers

    Examples
    --------
    >>> from vitalDSP.ml_models.transfer_learning import FineTuner
    >>> import numpy as np
    >>>
    >>> # Assuming we have a base model
    >>> finetuner = FineTuner(base_model, strategy='progressive')
    >>>
    >>> # Fine-tune with progressive unfreezing
    >>> X_train = np.random.randn(500, 187, 1)
    >>> y_train = np.random.randint(0, 4, 500)
    >>> finetuner.fit(X_train, y_train, n_classes=4, epochs=30)
    """

    def __init__(
        self,
        base_model: Any,
        strategy: str = "all_at_once",
        backend: str = "tensorflow",
    ):
        """
        Initialize fine-tuner.

        Parameters
        ----------
        base_model : object
            Pre-trained base model
        strategy : str, default='all_at_once'
            Fine-tuning strategy ('all_at_once', 'progressive', 'discriminative')
        backend : str, default='tensorflow'
            Deep learning backend
        """
        super().__init__(base_model, backend)
        self.strategy = strategy.lower()

        # Initially freeze all layers
        self.freeze_layers()

    def freeze_layers(self, n_layers: Optional[int] = None):
        """Freeze layers."""
        if self.backend == "tensorflow":
            if hasattr(self.base_model, "trainable"):
                if n_layers is None:
                    self.base_model.trainable = False
                else:
                    for i, layer in enumerate(self.base_model.layers):
                        layer.trainable = i >= n_layers
        else:
            raise NotImplementedError("PyTorch backend not yet implemented")

    def unfreeze_layers(self, n_layers: Optional[int] = None):
        """Unfreeze layers."""
        if self.backend == "tensorflow":
            if hasattr(self.base_model, "trainable"):
                if n_layers is None:
                    self.base_model.trainable = True
                else:
                    total = len(self.base_model.layers)
                    for i, layer in enumerate(self.base_model.layers):
                        layer.trainable = i >= total - n_layers
        else:
            raise NotImplementedError("PyTorch backend not yet implemented")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_classes: Optional[int] = None,
        task: str = "classification",
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        base_learning_rate: float = 1e-5,
        head_learning_rate: float = 1e-3,
        progressive_epochs: Optional[List[int]] = None,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune model on new task.

        Parameters
        ----------
        X_train : np.ndarray
            Training signals
        y_train : np.ndarray
            Training labels
        n_classes : int, optional
            Number of classes
        task : str, default='classification'
            Task type
        X_val : np.ndarray, optional
            Validation signals
        y_val : np.ndarray, optional
            Validation labels
        epochs : int, default=50
            Total training epochs
        batch_size : int, default=32
            Batch size
        base_learning_rate : float, default=1e-5
            Learning rate for base model (smaller)
        head_learning_rate : float, default=1e-3
            Learning rate for new head (larger)
        progressive_epochs : list of int, optional
            Epochs at which to unfreeze layers (for progressive strategy)
        verbose : int, default=1
            Verbosity level

        Returns
        -------
        dict
            Training history
        """
        if self.backend != "tensorflow":
            raise NotImplementedError("Only TensorFlow backend currently supported")

        # Build model with new head
        inputs = keras.Input(shape=X_train.shape[1:])
        x = self.base_model(inputs, training=False)  # Start with frozen base

        # Ensure flattened features
        if len(x.shape) > 2:
            x = (
                layers.GlobalAveragePooling1D()(x)
                if len(x.shape) == 3
                else layers.Flatten()(x)
            )

        # New head
        x = layers.Dense(256, activation="relu", name="ft_dense1")(x)
        x = layers.BatchNormalization(name="ft_bn1")(x)
        x = layers.Dropout(0.4, name="ft_dropout1")(x)
        x = layers.Dense(128, activation="relu", name="ft_dense2")(x)
        x = layers.Dropout(0.3, name="ft_dropout2")(x)

        if task == "classification":
            if n_classes is None:
                n_classes = len(np.unique(y_train))
            outputs = layers.Dense(n_classes, activation="softmax", name="ft_output")(x)
            loss = "sparse_categorical_crossentropy"
            metrics = ["accuracy"]
        else:
            outputs = layers.Dense(1, activation="linear", name="ft_output")(x)
            loss = "mse"
            metrics = ["mae"]

        self.model = keras.Model(inputs, outputs)

        # Strategy-specific training
        if self.strategy == "all_at_once":
            return self._fit_all_at_once(
                X_train,
                y_train,
                X_val,
                y_val,
                loss,
                metrics,
                epochs,
                batch_size,
                base_learning_rate,
                head_learning_rate,
                verbose,
            )

        elif self.strategy == "progressive":
            return self._fit_progressive(
                X_train,
                y_train,
                X_val,
                y_val,
                loss,
                metrics,
                epochs,
                batch_size,
                base_learning_rate,
                head_learning_rate,
                progressive_epochs,
                verbose,
            )

        elif self.strategy == "discriminative":
            return self._fit_discriminative(
                X_train,
                y_train,
                X_val,
                y_val,
                loss,
                metrics,
                epochs,
                batch_size,
                base_learning_rate,
                head_learning_rate,
                verbose,
            )

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _fit_all_at_once(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        loss,
        metrics,
        epochs,
        batch_size,
        base_lr,
        head_lr,
        verbose,
    ):
        """Fit with all layers unfrozen at once."""
        # First train only the head
        self.freeze_layers()
        self.model.compile(
            optimizer=keras.optimizers.Adam(head_lr), loss=loss, metrics=metrics
        )

        history_head = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=min(10, epochs // 3),
            batch_size=batch_size,
            verbose=verbose,
        )

        # Then unfreeze all and fine-tune
        self.unfreeze_layers()
        self.model.compile(
            optimizer=keras.optimizers.Adam(base_lr), loss=loss, metrics=metrics
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=15,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=7,
                min_lr=1e-8,
            ),
        ]

        history_finetune = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        # Combine histories
        combined_history = {}
        for key in history_head.history:
            combined_history[key] = (
                history_head.history[key] + history_finetune.history[key]
            )

        self.history = combined_history
        return self.history

    def _fit_progressive(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        loss,
        metrics,
        epochs,
        batch_size,
        base_lr,
        head_lr,
        progressive_epochs,
        verbose,
    ):
        """Fit with progressive layer unfreezing."""
        if progressive_epochs is None:
            # Default: unfreeze in 3 stages
            progressive_epochs = [epochs // 4, epochs // 2, 3 * epochs // 4]

        # Stage 1: Train head only
        self.freeze_layers()
        self.model.compile(
            optimizer=keras.optimizers.Adam(head_lr), loss=loss, metrics=metrics
        )

        history = {"loss": [], "val_loss": []}
        if "accuracy" in metrics:
            history["accuracy"] = []
            history["val_accuracy"] = []

        current_epoch = 0
        n_layers = len(self.base_model.layers)

        for stage, target_epoch in enumerate(progressive_epochs):
            stage_epochs = target_epoch - current_epoch

            if stage > 0:
                # Progressively unfreeze layers
                n_to_unfreeze = min(
                    n_layers // len(progressive_epochs) * stage, n_layers
                )
                self.unfreeze_layers(n_to_unfreeze)

                # Reduce learning rate for base
                lr = base_lr * (0.5 ** (stage - 1))
                self.model.compile(
                    optimizer=keras.optimizers.Adam(lr), loss=loss, metrics=metrics
                )

            if verbose:
                print(f"\nStage {stage + 1}: Training for {stage_epochs} epochs")

            h = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=stage_epochs,
                batch_size=batch_size,
                verbose=verbose,
            )

            # Accumulate history
            for key in h.history:
                history[key].extend(h.history[key])

            current_epoch = target_epoch

        self.history = history
        return self.history

    def _fit_discriminative(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        loss,
        metrics,
        epochs,
        batch_size,
        base_lr,
        head_lr,
        verbose,
    ):
        """Fit with discriminative learning rates."""
        # Unfreeze all layers
        self.unfreeze_layers()

        # Use different LRs for different parts
        # This is simplified - in practice, you'd create custom optimizer
        # with per-layer learning rates

        self.model.compile(
            optimizer=keras.optimizers.Adam(base_lr), loss=loss, metrics=metrics
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=15,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=7,
                min_lr=1e-8,
            ),
        ]

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        self.history = history.history
        return self.history

    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict(X, batch_size=batch_size, verbose=0)


class DomainAdapter:
    """
    Domain adaptation for transfer learning.

    Adapts model from source domain to target domain when distributions differ.

    Techniques:
    - Feature alignment
    - Adversarial training
    - Maximum Mean Discrepancy (MMD)

    Use cases:
    - Transfer from one patient population to another
    - Adapt across different sensors/devices
    - Cross-institution transfer

    Examples
    --------
    >>> from vitalDSP.ml_models.transfer_learning import DomainAdapter
    >>> import numpy as np
    >>>
    >>> # Source and target data from different domains
    >>> X_source = np.random.randn(500, 187, 1)
    >>> y_source = np.random.randint(0, 3, 500)
    >>> X_target = np.random.randn(100, 187, 1) * 1.5  # Different distribution
    >>>
    >>> # Create domain adapter
    >>> adapter = DomainAdapter(base_model, method='mmd')
    >>> adapter.fit(X_source, y_source, X_target, epochs=50)
    """

    def __init__(
        self, base_model: Any, method: str = "mmd", backend: str = "tensorflow"
    ):
        """
        Initialize domain adapter.

        Parameters
        ----------
        base_model : object
            Pre-trained base model
        method : str, default='mmd'
            Domain adaptation method ('mmd', 'coral', 'dann')
        backend : str, default='tensorflow'
            Deep learning backend
        """
        self.base_model = base_model
        self.method = method.lower()
        self.backend = backend.lower()
        self.model = None

    def fit(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target: np.ndarray,
        y_target: Optional[np.ndarray] = None,
        n_classes: Optional[int] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        lambda_da: float = 0.1,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """
        Fit domain adaptation model.

        Parameters
        ----------
        X_source : np.ndarray
            Source domain signals (labeled)
        y_source : np.ndarray
            Source domain labels
        X_target : np.ndarray
            Target domain signals (can be unlabeled)
        y_target : np.ndarray, optional
            Target domain labels (if available)
        n_classes : int, optional
            Number of classes
        epochs : int, default=50
            Training epochs
        batch_size : int, default=32
            Batch size
        learning_rate : float, default=1e-4
            Learning rate
        lambda_da : float, default=0.1
            Weight for domain adaptation loss
        verbose : int, default=1
            Verbosity level

        Returns
        -------
        dict
            Training history
        """
        if self.backend != "tensorflow":
            raise NotImplementedError("Only TensorFlow backend currently supported")

        if n_classes is None:
            n_classes = len(np.unique(y_source))

        if self.method == "mmd":
            return self._fit_mmd(
                X_source,
                y_source,
                X_target,
                y_target,
                n_classes,
                epochs,
                batch_size,
                learning_rate,
                lambda_da,
                verbose,
            )
        else:
            raise NotImplementedError(f"Method '{self.method}' not yet implemented")

    def _fit_mmd(
        self,
        X_source,
        y_source,
        X_target,
        y_target,
        n_classes,
        epochs,
        batch_size,
        learning_rate,
        lambda_da,
        verbose,
    ):
        """Fit using Maximum Mean Discrepancy."""
        # Build model
        inputs = keras.Input(shape=X_source.shape[1:])
        features = self.base_model(inputs)

        if len(features.shape) > 2:
            features = layers.GlobalAveragePooling1D()(features)

        classifier = layers.Dense(n_classes, activation="softmax")(features)

        self.model = keras.Model(inputs, [classifier, features])

        # Custom training loop for MMD
        optimizer = keras.optimizers.Adam(learning_rate)

        @tf.function
        def train_step(x_src, y_src, x_tgt):
            with tf.GradientTape() as tape:
                # Forward pass
                pred_src, feat_src = self.model(x_src, training=True)
                _, feat_tgt = self.model(x_tgt, training=True)

                # Classification loss
                class_loss = tf.reduce_mean(
                    keras.losses.sparse_categorical_crossentropy(y_src, pred_src)
                )

                # MMD loss (simplified)
                mmd_loss = self._compute_mmd(feat_src, feat_tgt)

                # Total loss
                total_loss = class_loss + lambda_da * mmd_loss

            # Backward pass
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            return class_loss, mmd_loss, total_loss

        # Training loop
        history = {"class_loss": [], "mmd_loss": [], "total_loss": []}

        n_batches = len(X_source) // batch_size

        for epoch in range(epochs):
            epoch_class_loss = 0.0
            epoch_mmd_loss = 0.0
            epoch_total_loss = 0.0

            # Shuffle data
            indices_src = np.random.permutation(len(X_source))
            indices_tgt = np.random.permutation(len(X_target))

            for batch in range(n_batches):
                # Get batch data
                batch_idx_src = indices_src[
                    batch * batch_size : (batch + 1) * batch_size
                ]
                batch_idx_tgt = indices_tgt[
                    batch * batch_size : (batch + 1) * batch_size
                ]

                x_src_batch = X_source[batch_idx_src]
                y_src_batch = y_source[batch_idx_src]
                x_tgt_batch = X_target[batch_idx_tgt]

                # Train step
                c_loss, m_loss, t_loss = train_step(
                    x_src_batch, y_src_batch, x_tgt_batch
                )

                epoch_class_loss += c_loss.numpy()
                epoch_mmd_loss += m_loss.numpy()
                epoch_total_loss += t_loss.numpy()

            # Average losses
            epoch_class_loss /= n_batches
            epoch_mmd_loss /= n_batches
            epoch_total_loss /= n_batches

            history["class_loss"].append(epoch_class_loss)
            history["mmd_loss"].append(epoch_mmd_loss)
            history["total_loss"].append(epoch_total_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"class_loss: {epoch_class_loss:.4f} - "
                    f"mmd_loss: {epoch_mmd_loss:.4f} - "
                    f"total_loss: {epoch_total_loss:.4f}"
                )

        return history

    @staticmethod
    def _compute_mmd(features_source, features_target):
        """Compute Maximum Mean Discrepancy between source and target features."""

        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for MMD computation")

        # Gaussian kernel
        def gaussian_kernel(x, y, sigma=1.0):
            x = tf.expand_dims(x, 1)  # (n, 1, d)
            y = tf.expand_dims(y, 0)  # (1, m, d)
            diff = x - y  # (n, m, d)
            dist_sq = tf.reduce_sum(diff**2, axis=2)  # (n, m)
            return tf.exp(-dist_sq / (2 * sigma**2))

        # MMD computation
        K_ss = gaussian_kernel(features_source, features_source)
        K_tt = gaussian_kernel(features_target, features_target)
        K_st = gaussian_kernel(features_source, features_target)

        m = tf.cast(tf.shape(features_source)[0], tf.float32)
        n = tf.cast(tf.shape(features_target)[0], tf.float32)

        mmd = (
            tf.reduce_sum(K_ss) / (m * m)
            + tf.reduce_sum(K_tt) / (n * n)
            - 2 * tf.reduce_sum(K_st) / (m * n)
        )

        return mmd

    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions, _ = self.model.predict(X, batch_size=batch_size, verbose=0)
        return np.argmax(predictions, axis=1)


# Convenience functions
def quick_transfer(
    base_model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str = "feature_extraction",
    n_classes: Optional[int] = None,
    epochs: int = 30,
    **kwargs,
) -> Union[TransferFeatureExtractor, FineTuner]:
    """
    Quick transfer learning with sensible defaults.

    Parameters
    ----------
    base_model : object
        Pre-trained base model
    X_train : np.ndarray
        Training signals
    y_train : np.ndarray
        Training labels
    strategy : str, default='feature_extraction'
        Transfer strategy ('feature_extraction', 'fine_tuning')
    n_classes : int, optional
        Number of classes
    epochs : int, default=30
        Training epochs
    **kwargs
        Additional arguments

    Returns
    -------
    TransferFeatureExtractor or FineTuner
        Trained transfer learning model

    Examples
    --------
    >>> from vitalDSP.ml_models.transfer_learning import quick_transfer
    >>> from vitalDSP.ml_models.pretrained_models import load_pretrained_model
    >>> import numpy as np
    >>>
    >>> # Load pre-trained model
    >>> base = load_pretrained_model('ecg_classifier_mitbih')
    >>>
    >>> # Quick transfer learning
    >>> X = np.random.randn(200, 187, 1)
    >>> y = np.random.randint(0, 3, 200)
    >>> model = quick_transfer(base.model, X, y, strategy='feature_extraction', epochs=20)
    """
    if strategy == "feature_extraction":
        model = TransferFeatureExtractor(base_model, freeze_base=True)
        model.fit(X_train, y_train, n_classes=n_classes, epochs=epochs, **kwargs)
    elif strategy == "fine_tuning":
        model = FineTuner(base_model, strategy="all_at_once")
        model.fit(X_train, y_train, n_classes=n_classes, epochs=epochs, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return model
