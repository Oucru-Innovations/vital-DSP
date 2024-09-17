import pytest
import numpy as np
from vitalDSP.physiological_features.ensemble_based_feature_extraction import (
    EnsembleBasedFeatureExtraction,
)


@pytest.fixture
def test_data():
    np.random.seed(42)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    return X, y


@pytest.fixture
def feature_extractor():
    return EnsembleBasedFeatureExtraction(
        n_estimators=5, max_depth=3, min_samples_split=2
    )


def test_random_forest_features(feature_extractor, test_data):
    X, y = test_data
    features = feature_extractor.random_forest_features(X, y)
    assert features.shape == (X.shape[0], feature_extractor.n_estimators)
    assert isinstance(features, np.ndarray)


def test_bagging_features(feature_extractor, test_data):
    X, y = test_data
    bagging_predictions = feature_extractor.bagging_features(X, y)
    assert bagging_predictions.shape == (X.shape[0],)
    assert isinstance(bagging_predictions, np.ndarray)
    assert np.all(np.isfinite(bagging_predictions))  # Ensure no NaNs


def test_boosting_features(feature_extractor, test_data):
    X, y = test_data
    boosting_predictions = feature_extractor.boosting_features(X, y, learning_rate=0.1)
    assert boosting_predictions.shape == (X.shape[0],)
    assert isinstance(boosting_predictions, np.ndarray)
    assert np.all(np.isfinite(boosting_predictions))  # Ensure there are no NaNs or inf


def test_stacking_features_with_default_meta_model(feature_extractor, test_data):
    X, y = test_data
    stacked_features = feature_extractor.stacking_features(X, y)
    assert stacked_features.shape == (X.shape[0],)
    assert isinstance(stacked_features, np.ndarray)


def test_stacking_features_with_custom_meta_model(feature_extractor, test_data):
    X, y = test_data
    meta_model = lambda predictions: np.median(predictions, axis=1)
    stacked_features = feature_extractor.stacking_features(X, y, meta_model=meta_model)
    assert stacked_features.shape == (X.shape[0],)
    assert isinstance(stacked_features, np.ndarray)


def test_build_tree(feature_extractor, test_data):
    X, y = test_data
    tree = feature_extractor._build_tree(X, y, depth=0)
    assert isinstance(tree, tuple) or isinstance(tree, float)


def test_best_split(feature_extractor, test_data):
    X, y = test_data
    best_feature, best_threshold = feature_extractor._best_split(X, y)
    # Ensure that the best split is found and not None
    assert best_feature is not None
    assert best_threshold is not None


def test_predict_tree_leaf_node(feature_extractor):
    X = np.array([[1, 2], [3, 4], [5, 6]])
    tree = 0.5  # Simulate a leaf node
    predictions = feature_extractor._predict_tree(tree, X)
    assert predictions.shape == (X.shape[0],)
    assert np.all(predictions == 0.5)


def test_predict_tree_decision_node(feature_extractor, test_data):
    X, y = test_data
    tree = feature_extractor._build_tree(X, y)
    predictions = feature_extractor._predict_tree(tree, X)
    assert predictions.shape == (X.shape[0],)
    assert isinstance(predictions, np.ndarray)
