"""
Explainable AI (XAI) for Physiological Signal Analysis

This module provides interpretability and explainability tools for machine learning
and deep learning models applied to physiological signals.

Supported Methods:
1. SHAP (SHapley Additive exPlanations)
2. LIME (Local Interpretable Model-agnostic Explanations)
3. GradCAM for 1D signals
4. Attention Visualization
5. Feature Importance Analysis

Author: vitalDSP
License: MIT
"""

import numpy as np
from typing import Optional, Union, Tuple, List, Dict, Any, Callable
from pathlib import Path
import warnings

# Core dependencies
from scipy.ndimage import gaussian_filter1d

# Optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

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

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class BaseExplainer:
    """
    Base class for all explainability methods.

    Provides common functionality for model interpretation and visualization.
    """

    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize base explainer.

        Parameters
        ----------
        model : object
            Trained model to explain
        feature_names : list of str, optional
            Names of features
        class_names : list of str, optional
            Names of classes
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.explanations = {}

    def explain(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Generate explanations for predictions.

        Parameters
        ----------
        X : np.ndarray
            Input samples to explain
        **kwargs
            Additional arguments

        Returns
        -------
        dict
            Explanation results
        """
        raise NotImplementedError("Subclasses must implement explain()")

    def plot(self, explanation: Dict[str, Any], **kwargs):
        """
        Visualize explanations.

        Parameters
        ----------
        explanation : dict
            Explanation to visualize
        **kwargs
            Additional plotting arguments
        """
        raise NotImplementedError("Subclasses must implement plot()")


class SHAPExplainer(BaseExplainer):
    """
    SHAP (SHapley Additive exPlanations) for model interpretation.

    SHAP values represent the contribution of each feature to the prediction,
    based on game-theoretic Shapley values.

    Supports:
    - TreeExplainer (for tree-based models)
    - DeepExplainer (for deep learning models)
    - KernelExplainer (model-agnostic)

    Examples
    --------
    >>> from vitalDSP.ml_models.explainability import SHAPExplainer
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> import numpy as np
    >>>
    >>> # Train a model
    >>> X_train = np.random.randn(1000, 50)
    >>> y_train = np.random.randint(0, 2, 1000)
    >>> model = RandomForestClassifier()
    >>> model.fit(X_train, y_train)
    >>>
    >>> # Create explainer
    >>> explainer = SHAPExplainer(
    ...     model,
    ...     explainer_type='tree',
    ...     feature_names=[f'feature_{i}' for i in range(50)]
    ... )
    >>>
    >>> # Explain predictions
    >>> X_test = np.random.randn(10, 50)
    >>> explanations = explainer.explain(X_test, background_data=X_train)
    >>>
    >>> # Visualize
    >>> explainer.plot_summary(explanations)
    >>> explainer.plot_waterfall(explanations, instance_idx=0)
    """

    def __init__(
        self,
        model: Any,
        explainer_type: str = 'kernel',
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize SHAP explainer.

        Parameters
        ----------
        model : object
            Trained model to explain
        explainer_type : str, default='kernel'
            Type of SHAP explainer ('tree', 'deep', 'kernel', 'linear')
        feature_names : list of str, optional
            Names of features
        class_names : list of str, optional
            Names of classes
        """
        super().__init__(model, feature_names, class_names)

        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")

        self.explainer_type = explainer_type.lower()
        self.explainer = None

    def _create_explainer(self, background_data: Optional[np.ndarray] = None):
        """Create SHAP explainer based on type."""
        if self.explainer_type == 'tree':
            # For tree-based models (RandomForest, XGBoost, LightGBM)
            self.explainer = shap.TreeExplainer(self.model)

        elif self.explainer_type == 'deep':
            # For deep learning models
            if not TENSORFLOW_AVAILABLE and not PYTORCH_AVAILABLE:
                raise ImportError("TensorFlow or PyTorch required for DeepExplainer")

            if background_data is None:
                raise ValueError("background_data required for DeepExplainer")

            self.explainer = shap.DeepExplainer(self.model, background_data)

        elif self.explainer_type == 'kernel':
            # Model-agnostic explainer
            if background_data is None:
                raise ValueError("background_data required for KernelExplainer")

            # Create prediction function
            if hasattr(self.model, 'predict_proba'):
                predict_fn = self.model.predict_proba
            elif hasattr(self.model, 'predict'):
                predict_fn = self.model.predict
            else:
                raise ValueError("Model must have predict or predict_proba method")

            self.explainer = shap.KernelExplainer(predict_fn, background_data)

        elif self.explainer_type == 'linear':
            # For linear models
            self.explainer = shap.LinearExplainer(self.model, background_data)

        else:
            raise ValueError(f"Unknown explainer type: {self.explainer_type}")

    def explain(
        self,
        X: np.ndarray,
        background_data: Optional[np.ndarray] = None,
        nsamples: int = 100
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations.

        Parameters
        ----------
        X : np.ndarray
            Samples to explain
        background_data : np.ndarray, optional
            Background dataset for kernel/deep explainers
        nsamples : int, default=100
            Number of samples for kernel explainer

        Returns
        -------
        dict
            Dictionary containing:
            - 'shap_values': SHAP values for each sample
            - 'base_values': Base values (expected model output)
            - 'data': Original input data
        """
        # Create explainer if not already created
        if self.explainer is None:
            self._create_explainer(background_data)

        # Compute SHAP values
        if self.explainer_type == 'kernel':
            shap_values = self.explainer.shap_values(X, nsamples=nsamples)
        else:
            shap_values = self.explainer.shap_values(X)

        # Get base values
        if hasattr(self.explainer, 'expected_value'):
            base_values = self.explainer.expected_value
        else:
            base_values = None

        explanation = {
            'shap_values': shap_values,
            'base_values': base_values,
            'data': X,
            'feature_names': self.feature_names
        }

        self.explanations['last'] = explanation
        return explanation

    def plot_summary(
        self,
        explanation: Optional[Dict[str, Any]] = None,
        plot_type: str = 'dot',
        max_display: int = 20,
        show: bool = True
    ):
        """
        Create SHAP summary plot.

        Parameters
        ----------
        explanation : dict, optional
            Explanation to plot. If None, use last explanation.
        plot_type : str, default='dot'
            Type of plot ('dot', 'bar', 'violin')
        max_display : int, default=20
            Maximum number of features to display
        show : bool, default=True
            Whether to show the plot
        """
        if explanation is None:
            explanation = self.explanations.get('last')
            if explanation is None:
                raise ValueError("No explanation available. Run explain() first.")

        shap_values = explanation['shap_values']
        data = explanation['data']

        # Create summary plot
        shap.summary_plot(
            shap_values,
            data,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=show
        )

    def plot_waterfall(
        self,
        explanation: Optional[Dict[str, Any]] = None,
        instance_idx: int = 0,
        max_display: int = 20,
        show: bool = True
    ):
        """
        Create SHAP waterfall plot for a single prediction.

        Parameters
        ----------
        explanation : dict, optional
            Explanation to plot
        instance_idx : int, default=0
            Index of instance to explain
        max_display : int, default=20
            Maximum number of features to display
        show : bool, default=True
            Whether to show the plot
        """
        if explanation is None:
            explanation = self.explanations.get('last')
            if explanation is None:
                raise ValueError("No explanation available. Run explain() first.")

        shap_values = explanation['shap_values']
        base_values = explanation['base_values']
        data = explanation['data']

        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if isinstance(base_values, (list, np.ndarray)):
            base_values = base_values[0]

        # Create waterfall plot
        shap_exp = shap.Explanation(
            values=shap_values[instance_idx],
            base_values=base_values,
            data=data[instance_idx],
            feature_names=self.feature_names
        )
        shap.plots.waterfall(shap_exp, max_display=max_display, show=show)

    def plot_force(
        self,
        explanation: Optional[Dict[str, Any]] = None,
        instance_idx: int = 0,
        matplotlib: bool = True
    ):
        """
        Create SHAP force plot.

        Parameters
        ----------
        explanation : dict, optional
            Explanation to plot
        instance_idx : int, default=0
            Index of instance to explain
        matplotlib : bool, default=True
            Whether to use matplotlib backend
        """
        if explanation is None:
            explanation = self.explanations.get('last')
            if explanation is None:
                raise ValueError("No explanation available. Run explain() first.")

        shap_values = explanation['shap_values']
        base_values = explanation['base_values']
        data = explanation['data']

        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if isinstance(base_values, (list, np.ndarray)):
            base_values = base_values[0]

        # Create force plot
        shap.force_plot(
            base_values,
            shap_values[instance_idx],
            data[instance_idx],
            feature_names=self.feature_names,
            matplotlib=matplotlib
        )

    def plot_dependence(
        self,
        feature_idx: Union[int, str],
        explanation: Optional[Dict[str, Any]] = None,
        interaction_idx: Union[int, str, None] = 'auto',
        show: bool = True
    ):
        """
        Create SHAP dependence plot showing feature interaction.

        Parameters
        ----------
        feature_idx : int or str
            Feature to plot
        explanation : dict, optional
            Explanation to plot
        interaction_idx : int, str, or None, default='auto'
            Feature to show interaction with
        show : bool, default=True
            Whether to show the plot
        """
        if explanation is None:
            explanation = self.explanations.get('last')
            if explanation is None:
                raise ValueError("No explanation available. Run explain() first.")

        shap_values = explanation['shap_values']
        data = explanation['data']

        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap.dependence_plot(
            feature_idx,
            shap_values,
            data,
            feature_names=self.feature_names,
            interaction_index=interaction_idx,
            show=show
        )


class LIMEExplainer(BaseExplainer):
    """
    LIME (Local Interpretable Model-agnostic Explanations).

    LIME explains individual predictions by approximating the model locally
    with an interpretable model.

    Examples
    --------
    >>> from vitalDSP.ml_models.explainability import LIMEExplainer
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> import numpy as np
    >>>
    >>> # Train a model
    >>> X_train = np.random.randn(1000, 50)
    >>> y_train = np.random.randint(0, 2, 1000)
    >>> model = RandomForestClassifier()
    >>> model.fit(X_train, y_train)
    >>>
    >>> # Create explainer
    >>> explainer = LIMEExplainer(
    ...     model,
    ...     training_data=X_train,
    ...     feature_names=[f'feature_{i}' for i in range(50)],
    ...     class_names=['Normal', 'Abnormal']
    ... )
    >>>
    >>> # Explain a prediction
    >>> X_test = np.random.randn(1, 50)
    >>> explanation = explainer.explain(X_test[0])
    >>> explainer.plot(explanation)
    """

    def __init__(
        self,
        model: Any,
        training_data: np.ndarray,
        mode: str = 'classification',
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        discretize_continuous: bool = False
    ):
        """
        Initialize LIME explainer.

        Parameters
        ----------
        model : object
            Trained model to explain
        training_data : np.ndarray
            Training data for LIME
        mode : str, default='classification'
            'classification' or 'regression'
        feature_names : list of str, optional
            Names of features
        class_names : list of str, optional
            Names of classes (for classification)
        discretize_continuous : bool, default=False
            Whether to discretize continuous features
        """
        super().__init__(model, feature_names, class_names)

        if not LIME_AVAILABLE:
            raise ImportError("LIME is not installed. Install with: pip install lime")

        self.training_data = training_data
        self.mode = mode.lower()
        self.discretize_continuous = discretize_continuous

        # Create LIME explainer
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data,
            mode=self.mode,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=discretize_continuous
        )

    def explain(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000,
        labels: Optional[Tuple[int, ...]] = None
    ) -> Any:
        """
        Explain a single prediction.

        Parameters
        ----------
        instance : np.ndarray
            Instance to explain (1D array)
        num_features : int, default=10
            Number of features to include in explanation
        num_samples : int, default=5000
            Number of samples for local approximation
        labels : tuple of int, optional
            Labels to explain (for classification)

        Returns
        -------
        lime.explanation.Explanation
            LIME explanation object
        """
        # Get prediction function
        if self.mode == 'classification':
            if hasattr(self.model, 'predict_proba'):
                predict_fn = self.model.predict_proba
            else:
                raise ValueError("Model must have predict_proba for classification")
        else:
            predict_fn = self.model.predict

        # Generate explanation
        explanation = self.lime_explainer.explain_instance(
            instance,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=labels
        )

        return explanation

    def plot(
        self,
        explanation: Any,
        label: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot LIME explanation.

        Parameters
        ----------
        explanation : lime.explanation.Explanation
            Explanation to plot
        label : int, optional
            Label to visualize (for classification)
        figsize : tuple, default=(10, 6)
            Figure size
        """
        if not MATPLOTLIB_AVAILABLE:
            warnings.warn("Matplotlib not available, using default plot")
            explanation.show_in_notebook()
            return

        # Use LIME's built-in visualization
        if self.mode == 'classification' and label is not None:
            fig = explanation.as_pyplot_figure(label=label)
        else:
            fig = explanation.as_pyplot_figure()

        plt.tight_layout()
        plt.show()


class GradCAM1D:
    """
    Gradient-weighted Class Activation Mapping (GradCAM) for 1D signals.

    Visualizes which parts of the signal are important for the prediction
    by computing gradients of the output with respect to feature maps.

    Examples
    --------
    >>> from vitalDSP.ml_models.explainability import GradCAM1D
    >>> import tensorflow as tf
    >>> import numpy as np
    >>>
    >>> # Assume we have a trained 1D CNN model
    >>> model = tf.keras.models.load_model('my_cnn_model.h5')
    >>>
    >>> # Create GradCAM explainer
    >>> gradcam = GradCAM1D(model, layer_name='conv1d_3')
    >>>
    >>> # Generate heatmap for a signal
    >>> signal = np.random.randn(1, 1000, 1)
    >>> heatmap = gradcam.compute_heatmap(signal, class_idx=1)
    >>>
    >>> # Visualize
    >>> gradcam.plot_overlay(signal[0], heatmap)
    """

    def __init__(
        self,
        model: Any,
        layer_name: Optional[str] = None,
        backend: str = 'tensorflow'
    ):
        """
        Initialize GradCAM.

        Parameters
        ----------
        model : object
            Trained neural network model
        layer_name : str, optional
            Name of convolutional layer to visualize. If None, use last conv layer.
        backend : str, default='tensorflow'
            Deep learning backend ('tensorflow' or 'pytorch')
        """
        self.model = model
        self.layer_name = layer_name
        self.backend = backend.lower()

        if self.backend == 'tensorflow':
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow not installed")
            self._setup_tensorflow()
        elif self.backend == 'pytorch':
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch not installed")
            self._setup_pytorch()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _setup_tensorflow(self):
        """Setup for TensorFlow backend."""
        # Find last convolutional layer if not specified
        if self.layer_name is None:
            for layer in reversed(self.model.layers):
                if isinstance(layer, keras.layers.Conv1D):
                    self.layer_name = layer.name
                    break

        if self.layer_name is None:
            raise ValueError("No Conv1D layer found in model")

        # Create gradient model
        self.grad_model = keras.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )

    def _setup_pytorch(self):
        """Setup for PyTorch backend."""
        # Find last convolutional layer if not specified
        if self.layer_name is None:
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, nn.Conv1d):
                    self.layer_name = name
                    break

        if self.layer_name is None:
            raise ValueError("No Conv1d layer found in model")

        # Register hooks for gradients
        self.gradients = None
        self.activations = None

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        # Get target layer
        target_layer = dict(self.model.named_modules())[self.layer_name]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def compute_heatmap(
        self,
        signal: np.ndarray,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute GradCAM heatmap.

        Parameters
        ----------
        signal : np.ndarray
            Input signal of shape (batch, length, channels) or (length, channels)
        class_idx : int, optional
            Target class index. If None, use predicted class.

        Returns
        -------
        np.ndarray
            Heatmap of shape (length,)
        """
        if signal.ndim == 2:
            signal = signal[np.newaxis, :]

        if self.backend == 'tensorflow':
            return self._compute_heatmap_tensorflow(signal, class_idx)
        else:
            return self._compute_heatmap_pytorch(signal, class_idx)

    def _compute_heatmap_tensorflow(
        self,
        signal: np.ndarray,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """Compute heatmap using TensorFlow."""
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(signal)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)

        # Pool gradients across channels
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        for i in range(pooled_grads.shape[-1]):
            conv_outputs = conv_outputs[:, i] * pooled_grads[i]

        # Create heatmap
        heatmap = tf.reduce_mean(conv_outputs, axis=-1)
        heatmap = tf.maximum(heatmap, 0)  # ReLU
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)  # Normalize

        # Resize to input length
        original_length = signal.shape[1]
        heatmap = tf.image.resize(
            heatmap[..., tf.newaxis, tf.newaxis],
            (original_length, 1)
        )
        heatmap = tf.squeeze(heatmap).numpy()

        return heatmap

    def _compute_heatmap_pytorch(
        self,
        signal: np.ndarray,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """Compute heatmap using PyTorch."""
        signal_tensor = torch.FloatTensor(signal).permute(0, 2, 1)  # (batch, channels, length)
        signal_tensor.requires_grad = True

        # Forward pass
        self.model.eval()
        output = self.model(signal_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Get gradients and activations
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()

        # Pool gradients
        pooled_grads = torch.mean(gradients, dim=(0, 2))

        # Weight activations
        for i in range(pooled_grads.shape[0]):
            activations[:, i, :] *= pooled_grads[i]

        # Create heatmap
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap = heatmap / (torch.max(heatmap) + 1e-10)

        # Resize to input length
        original_length = signal.shape[1]
        heatmap = torch.nn.functional.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=original_length,
            mode='linear'
        )
        heatmap = heatmap.squeeze().numpy()

        return heatmap

    def plot_overlay(
        self,
        signal: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: str = 'jet',
        figsize: Tuple[int, int] = (15, 5)
    ):
        """
        Plot signal with GradCAM heatmap overlay.

        Parameters
        ----------
        signal : np.ndarray
            Original signal (1D or 2D with channels)
        heatmap : np.ndarray
            GradCAM heatmap
        alpha : float, default=0.4
            Transparency of heatmap overlay
        colormap : str, default='jet'
            Colormap for heatmap
        figsize : tuple, default=(15, 5)
            Figure size
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for visualization")

        # Flatten signal if multi-channel
        if signal.ndim > 1:
            signal = signal[:, 0]

        fig, ax = plt.subplots(figsize=figsize)

        # Plot signal
        time = np.arange(len(signal))
        ax.plot(time, signal, 'k-', linewidth=1, label='Signal')

        # Create colored overlay
        cmap = plt.cm.get_cmap(colormap)
        colors = cmap(heatmap)

        # Plot heatmap as background
        for i in range(len(time) - 1):
            ax.axvspan(
                time[i], time[i + 1],
                alpha=alpha * heatmap[i],
                color=colors[i]
            )

        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.set_title('GradCAM Visualization')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class AttentionVisualizer:
    """
    Visualize attention weights from transformer models.

    Examples
    --------
    >>> from vitalDSP.ml_models.explainability import AttentionVisualizer
    >>> import numpy as np
    >>>
    >>> # Assume we have attention weights from a transformer
    >>> attention_weights = np.random.rand(8, 100, 100)  # (n_heads, seq_len, seq_len)
    >>>
    >>> # Create visualizer
    >>> viz = AttentionVisualizer()
    >>>
    >>> # Plot attention patterns
    >>> viz.plot_attention_map(attention_weights, head_idx=0)
    >>> viz.plot_attention_rollout(attention_weights)
    """

    def __init__(self):
        """Initialize attention visualizer."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for visualization")

    def plot_attention_map(
        self,
        attention_weights: np.ndarray,
        head_idx: int = 0,
        figsize: Tuple[int, int] = (10, 10),
        title: Optional[str] = None
    ):
        """
        Plot attention map for a specific head.

        Parameters
        ----------
        attention_weights : np.ndarray
            Attention weights of shape (n_heads, seq_len, seq_len)
        head_idx : int, default=0
            Index of attention head to visualize
        figsize : tuple, default=(10, 10)
            Figure size
        title : str, optional
            Plot title
        """
        if attention_weights.ndim == 2:
            attn = attention_weights
        else:
            attn = attention_weights[head_idx]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(attn, cmap='viridis', aspect='auto')

        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        if title is None:
            title = f'Attention Map (Head {head_idx})'
        ax.set_title(title)

        plt.colorbar(im, ax=ax, label='Attention Weight')
        plt.tight_layout()
        plt.show()

    def plot_attention_rollout(
        self,
        attention_weights: np.ndarray,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot attention rollout (average across heads and layers).

        Parameters
        ----------
        attention_weights : np.ndarray
            Attention weights of shape (n_heads, seq_len, seq_len)
        figsize : tuple, default=(12, 6)
            Figure size
        """
        # Average across heads
        if attention_weights.ndim == 3:
            attn_rollout = np.mean(attention_weights, axis=0)
        else:
            attn_rollout = attention_weights

        # Compute attention flow
        seq_len = attn_rollout.shape[0]
        attention_flow = np.sum(attn_rollout, axis=1)  # Sum over keys

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot attention matrix
        im1 = ax1.imshow(attn_rollout, cmap='viridis', aspect='auto')
        ax1.set_xlabel('Key Position')
        ax1.set_ylabel('Query Position')
        ax1.set_title('Attention Rollout')
        plt.colorbar(im1, ax=ax1, label='Attention')

        # Plot attention flow
        ax2.plot(attention_flow, linewidth=2)
        ax2.fill_between(range(seq_len), attention_flow, alpha=0.3)
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Total Attention')
        ax2.set_title('Attention Flow')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_head_comparison(
        self,
        attention_weights: np.ndarray,
        query_idx: int = 0,
        figsize: Tuple[int, int] = (15, 4)
    ):
        """
        Compare attention patterns across different heads.

        Parameters
        ----------
        attention_weights : np.ndarray
            Attention weights of shape (n_heads, seq_len, seq_len)
        query_idx : int, default=0
            Query position to visualize
        figsize : tuple, default=(15, 4)
            Figure size
        """
        n_heads = attention_weights.shape[0]
        seq_len = attention_weights.shape[1]

        fig, axes = plt.subplots(1, n_heads, figsize=figsize)
        if n_heads == 1:
            axes = [axes]

        for head_idx, ax in enumerate(axes):
            attn = attention_weights[head_idx, query_idx, :]
            ax.plot(attn, linewidth=2)
            ax.fill_between(range(seq_len), attn, alpha=0.3)
            ax.set_title(f'Head {head_idx}')
            ax.set_xlabel('Key Position')
            if head_idx == 0:
                ax.set_ylabel('Attention Weight')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Attention Patterns for Query Position {query_idx}')
        plt.tight_layout()
        plt.show()


# Convenience function
def explain_prediction(
    model: Any,
    X: np.ndarray,
    method: str = 'shap',
    background_data: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick explanation of model predictions.

    Parameters
    ----------
    model : object
        Trained model
    X : np.ndarray
        Samples to explain
    method : str, default='shap'
        Explanation method ('shap', 'lime')
    background_data : np.ndarray, optional
        Background data (required for some methods)
    feature_names : list of str, optional
        Feature names
    class_names : list of str, optional
        Class names
    **kwargs
        Additional arguments for the explainer

    Returns
    -------
    dict
        Explanation results

    Examples
    --------
    >>> from vitalDSP.ml_models.explainability import explain_prediction
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> import numpy as np
    >>>
    >>> # Train model
    >>> X_train = np.random.randn(1000, 50)
    >>> y_train = np.random.randint(0, 2, 1000)
    >>> model = RandomForestClassifier()
    >>> model.fit(X_train, y_train)
    >>>
    >>> # Explain test predictions
    >>> X_test = np.random.randn(10, 50)
    >>> explanation = explain_prediction(
    ...     model, X_test,
    ...     method='shap',
    ...     explainer_type='tree'
    ... )
    """
    method = method.lower()

    if method == 'shap':
        explainer = SHAPExplainer(
            model,
            feature_names=feature_names,
            class_names=class_names,
            **kwargs
        )
        return explainer.explain(X, background_data=background_data)

    elif method == 'lime':
        if background_data is None:
            raise ValueError("background_data (training data) required for LIME")

        explainer = LIMEExplainer(
            model,
            training_data=background_data,
            feature_names=feature_names,
            class_names=class_names,
            **kwargs
        )

        # LIME explains one instance at a time
        if X.ndim == 1 or len(X) == 1:
            instance = X if X.ndim == 1 else X[0]
            return explainer.explain(instance)
        else:
            # Explain multiple instances
            explanations = []
            for instance in X:
                exp = explainer.explain(instance)
                explanations.append(exp)
            return {'explanations': explanations}

    else:
        raise ValueError(f"Unknown method: {method}")
