"""
vitalDSP Machine Learning and Deep Learning Module

Comprehensive ML/DL integration for physiological signal analysis.

This module provides:
- Feature extraction for ML pipelines
- Deep learning models (CNN, LSTM, Transformer, Autoencoder)
- Pre-trained models and transfer learning
- Explainable AI (SHAP/LIME)
- Training utilities and pipelines

Author: vitalDSP Team
Date: 2025
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
