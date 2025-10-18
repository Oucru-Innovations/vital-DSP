"""
Machine Learning Models Module for Physiological Signal Processing

This module provides comprehensive machine learning and deep learning capabilities
for physiological signal analysis including ECG, PPG, EEG, and other vital signs.
It implements state-of-the-art models for classification, regression, anomaly
detection, and feature extraction.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Deep learning models (CNN, LSTM, Transformer, Autoencoder)
- Pre-trained models and transfer learning
- Feature extraction for ML pipelines
- Explainable AI (SHAP/LIME) integration
- Model training utilities and pipelines
- Comprehensive evaluation metrics
- Model optimization and hyperparameter tuning
- Cross-validation and model selection

Examples:
--------
Basic CNN model:
    >>> from vitalDSP.ml_models import CNN1D
    >>> model = CNN1D(input_shape=(1000, 1), num_classes=5)
    >>> model.compile(optimizer='adam', loss='categorical_crossentropy')

Autoencoder for anomaly detection:
    >>> from vitalDSP.ml_models import StandardAutoencoder
    >>> autoencoder = StandardAutoencoder(input_dim=100, encoding_dim=32)
    >>> autoencoder.compile(optimizer='adam', loss='mse')

Feature extraction:
    >>> from vitalDSP.ml_models import FeatureExtractor
    >>> extractor = FeatureExtractor()
    >>> features = extractor.extract_features(signal_data)
"""

# Feature Extraction
from .feature_extractor import FeatureExtractor, FeatureEngineering, extract_features

# Deep Learning Models
from .deep_models import BaseDeepModel, CNN1D, LSTMModel

from .transformer_model import TransformerModel

from .autoencoder import (
    BaseAutoencoder,
    StandardAutoencoder,
    ConvolutionalAutoencoder,
    LSTMAutoencoder,
    VariationalAutoencoder,
    DenoisingAutoencoder,
    detect_anomalies,
    denoise_signal,
)

from .explainability import (
    BaseExplainer,
    SHAPExplainer,
    LIMEExplainer,
    GradCAM1D,
    AttentionVisualizer,
    explain_prediction,
)

from .pretrained_models import (
    PretrainedModel,
    ModelHub,
    load_pretrained_model,
    MODEL_REGISTRY,
)

from .transfer_learning import (
    TransferLearningStrategy,
    FeatureExtractor,
    FineTuner,
    DomainAdapter,
    quick_transfer,
)

# Future imports (to be implemented)
# from .model_utils import SignalAugmentation, ModelEvaluator, ModelComparison
# from .training_pipeline import TrainingPipeline

__all__ = [
    # Feature Extraction
    "FeatureExtractor",
    "FeatureEngineering",
    "extract_features",
    # Deep Models
    "BaseDeepModel",
    "CNN1D",
    "LSTMModel",
    "TransformerModel",
    # Autoencoders
    "BaseAutoencoder",
    "StandardAutoencoder",
    "ConvolutionalAutoencoder",
    "LSTMAutoencoder",
    "VariationalAutoencoder",
    "DenoisingAutoencoder",
    "detect_anomalies",
    "denoise_signal",
    # Explainability
    "BaseExplainer",
    "SHAPExplainer",
    "LIMEExplainer",
    "GradCAM1D",
    "AttentionVisualizer",
    "explain_prediction",
    # Pre-trained Models
    "PretrainedModel",
    "ModelHub",
    "load_pretrained_model",
    "MODEL_REGISTRY",
    # Transfer Learning
    "TransferLearningStrategy",
    "FeatureExtractor",
    "FineTuner",
    "DomainAdapter",
    "quick_transfer",
    # Future exports (commented until implemented)
    # 'SignalAugmentation',
    # 'ModelEvaluator',
    # 'ModelComparison',
    # 'TrainingPipeline',
]

__version__ = "1.0.0"
