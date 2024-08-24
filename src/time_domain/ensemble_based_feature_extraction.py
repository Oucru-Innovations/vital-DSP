import numpy as np
# from collections import Counter


class EnsembleBasedFeatureExtraction:
    """
    A comprehensive class for feature extraction using ensemble methods such as Random Forest, Bagging, and Boosting.

    Methods
    -------
    random_forest_features : function
        Extracts features using a custom Random Forest.
    bagging_features : function
        Extracts features using a Bagging ensemble.
    boosting_features : function
        Extracts features using a Boosting ensemble.
    stacking_features : function
        Extracts features using a Stacking ensemble.
    """
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2):
        """
        Initialize the EnsembleBasedFeatureExtraction class.

        Parameters
        ----------
        n_estimators : int
            The number of estimators in the ensemble.
        max_depth : int or None
            The maximum depth of the trees. If None, the trees will expand until all leaves are pure.
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def random_forest_features(self, X, y):
        """
        Extract features using a custom Random Forest.

        Parameters
        ----------
        X : numpy.ndarray
            The input features with shape (n_samples, n_features).
        y : numpy.ndarray
            The target labels with shape (n_samples,).

        Returns
        -------
        features : numpy.ndarray
            The extracted features from the Random Forest, with shape (n_samples, n_estimators).

        Examples
        --------
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([0, 1, 0])
        >>> extractor = EnsembleBasedFeatureExtraction()
        >>> features = extractor.random_forest_features(X, y)
        >>> print(features)
        """
        forest = [self._build_tree(X, y) for _ in range(self.n_estimators)]
        features = np.array([self._predict_tree(tree, X) for tree in forest]).T
        return features

    def bagging_features(self, X, y):
        """
        Extract features using a Bagging ensemble.

        Parameters
        ----------
        X : numpy.ndarray
            The input features with shape (n_samples, n_features).
        y : numpy.ndarray
            The target labels with shape (n_samples,).

        Returns
        -------
        aggregated_predictions : numpy.ndarray
            The aggregated predictions from the Bagging ensemble, with shape (n_samples,).

        Examples
        --------
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([0, 1, 0])
        >>> extractor = EnsembleBasedFeatureExtraction()
        >>> features = extractor.bagging_features(X, y)
        >>> print(features)
        """
        bagged_models = [
            self._build_tree(X[np.random.choice(len(X), len(X), replace=True)], y)
            for _ in range(self.n_estimators)
        ]
        predictions = np.array(
            [self._predict_tree(tree, X) for tree in bagged_models]
        ).T
        aggregated_predictions = np.mean(predictions, axis=1)
        return aggregated_predictions

    def boosting_features(self, X, y, learning_rate=0.1):
        """
        Extract features using a Boosting ensemble.

        Parameters
        ----------
        X : numpy.ndarray
            The input features with shape (n_samples, n_features).
        y : numpy.ndarray
            The target labels with shape (n_samples,).
        learning_rate : float, optional
            The learning rate for the boosting. Default is 0.1.

        Returns
        -------
        predictions : numpy.ndarray
            The extracted features from the Boosting ensemble, with shape (n_samples,).

        Examples
        --------
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([0, 1, 0])
        >>> extractor = EnsembleBasedFeatureExtraction()
        >>> features = extractor.boosting_features(X, y, learning_rate=0.1)
        >>> print(features)
        """
        predictions = np.zeros(X.shape[0])
        for _ in range(self.n_estimators):
            residual = y - predictions
            tree = self._build_tree(X, residual)
            update = self._predict_tree(tree, X)
            predictions += learning_rate * update
        return predictions

    def stacking_features(self, X, y, meta_model=None):
        """
        Extract features using a Stacking ensemble.

        Parameters
        ----------
        X : numpy.ndarray
            The input features with shape (n_samples, n_features).
        y : numpy.ndarray
            The target labels with shape (n_samples,).
        meta_model : function or None, optional
            The meta-model used to aggregate the base models' predictions. Default is None.

        Returns
        -------
        stacked_features : numpy.ndarray
            The extracted features from the Stacking ensemble, with shape (n_samples,).

        Examples
        --------
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([0, 1, 0])
        >>> extractor = EnsembleBasedFeatureExtraction()
        >>> features = extractor.stacking_features(X, y)
        >>> print(features)
        """
        base_models = [self._build_tree(X, y) for _ in range(self.n_estimators)]
        base_predictions = np.array(
            [self._predict_tree(tree, X) for tree in base_models]
        ).T
        if meta_model is None:
            meta_model = lambda preds: np.mean(preds, axis=1)
        stacked_features = meta_model(base_predictions)
        return stacked_features

    def _build_tree(self, X, y, depth=0):
        if len(y) <= self.min_samples_split or (
            self.max_depth and depth >= self.max_depth
        ):
            return np.mean(y)
        feature, threshold = self._best_split(X, y)
        left_indices = X[:, feature] < threshold
        right_indices = ~left_indices
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return (feature, threshold, left_tree, right_tree)

    def _best_split(self, X, y):
        best_feature, best_threshold, best_score = None, None, float("inf")
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = ~left_indices
                left_score = np.mean((y[left_indices] - np.mean(y[left_indices])) ** 2)
                right_score = np.mean(
                    (y[right_indices] - np.mean(y[right_indices])) ** 2
                )
                score = left_score * np.sum(left_indices) + right_score * np.sum(
                    right_indices
                )
                if score < best_score:
                    best_feature, best_threshold, best_score = feature, threshold, score
        return best_feature, best_threshold

    def _predict_tree(self, tree, X):
        if not isinstance(tree, tuple):
            return np.full(X.shape[0], tree)
        feature, threshold, left_tree, right_tree = tree
        left_indices = X[:, feature] < threshold
        right_indices = ~left_indices
        predictions = np.empty(X.shape[0])
        predictions[left_indices] = self._predict_tree(left_tree, X[left_indices])
        predictions[right_indices] = self._predict_tree(right_tree, X[right_indices])
        return predictions
