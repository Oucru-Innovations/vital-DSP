import numpy as np


class EnsembleBasedFeatureExtraction:
    """
    A comprehensive class for feature extraction using ensemble methods such as Random Forest, Bagging, Boosting, and Stacking.

    This class implements various ensemble techniques to extract features from data, enhancing the representational capacity of the models and improving prediction accuracy.

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
            The number of estimators (trees) in the ensemble. Default is 100.
        max_depth : int or None
            The maximum depth of the trees. If None, the trees will expand until all leaves are pure or contain fewer than min_samples_split samples. Default is None.
        min_samples_split : int
            The minimum number of samples required to split an internal node. Default is 2.

        Notes
        -----
        These parameters control the complexity of the ensemble models. A higher number of estimators generally improves performance but increases computational cost. Limiting tree depth (max_depth) and increasing min_samples_split helps prevent overfitting.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def random_forest_features(self, X, y):
        """
        Extract features using a custom Random Forest.

        This method builds multiple decision trees and aggregates their predictions to form a feature representation for each sample.

        Parameters
        ----------
        X : numpy.ndarray
            The input features with shape (n_samples, n_features).
        y : numpy.ndarray
            The target labels with shape (n_samples,).

        Returns
        -------
        features : numpy.ndarray
            The extracted features from the Random Forest, with shape (n_samples, n_estimators). Each feature corresponds to the prediction of a single tree in the forest.

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

        Bagging (Bootstrap Aggregating) builds multiple trees on different subsets of the data, each generated by random sampling with replacement, and aggregates their predictions.

        Parameters
        ----------
        X : numpy.ndarray
            The input features with shape (n_samples, n_features).
        y : numpy.ndarray
            The target labels with shape (n_samples,).

        Returns
        -------
        aggregated_predictions : numpy.ndarray
            The aggregated predictions from the Bagging ensemble, with shape (n_samples,). This represents the averaged output of all trees.

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

        Boosting builds trees sequentially, each one trying to correct the errors of the previous one. The final prediction is a weighted sum of the predictions from all trees.

        Parameters
        ----------
        X : numpy.ndarray
            The input features with shape (n_samples, n_features).
        y : numpy.ndarray
            The target labels with shape (n_samples,).
        learning_rate : float, optional
            The learning rate for boosting, controlling the contribution of each tree. Default is 0.1.

        Returns
        -------
        predictions : numpy.ndarray
            The extracted features from the Boosting ensemble, with shape (n_samples,). This represents the cumulative prediction after all boosting iterations.

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

        Stacking combines the predictions of multiple base models using a meta-model. The base models are first trained independently, and their predictions are used as inputs to the meta-model.

        Parameters
        ----------
        X : numpy.ndarray
            The input features with shape (n_samples, n_features).
        y : numpy.ndarray
            The target labels with shape (n_samples,).
        meta_model : function or None, optional
            The meta-model used to aggregate the base models' predictions. If None, a simple average is used. Default is None.

        Returns
        -------
        stacked_features : numpy.ndarray
            The extracted features from the Stacking ensemble, with shape (n_samples,). These features are the output of the meta-model applied to the base models' predictions.

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
        """
        Build a decision tree recursively.

        This is a helper method used by the ensemble methods to create decision trees.

        Parameters
        ----------
        X : numpy.ndarray
            The input features for training the tree.
        y : numpy.ndarray
            The target labels for training the tree.
        depth : int
            The current depth of the tree.

        Returns
        -------
        tree : tuple or float
            A tuple representing the decision tree or a float representing a leaf node with the predicted value.

        Notes
        -----
        The tree is built by recursively finding the best split that minimizes the variance of the target values in each node.
        """
        if len(y) <= self.min_samples_split or (
            self.max_depth is not None and depth >= self.max_depth
        ):
            return np.mean(y)

        feature, threshold = self._best_split(X, y)

        if feature is None or threshold is None:  # Safeguard for no valid split
            return np.mean(y)

        left_indices = X[:, feature] < threshold
        right_indices = ~left_indices

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (feature, threshold, left_tree, right_tree)

    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split the data.

        This is a helper method used by the decision tree builder to find the optimal split.

        Parameters
        ----------
        X : numpy.ndarray
            The input features.
        y : numpy.ndarray
            The target labels.

        Returns
        -------
        best_feature : int
            The index of the best feature to split on.
        best_threshold : float
            The value of the best threshold to split on.
        best_score : float
            The score of the best split.

        Notes
        -----
        The best split is determined by evaluating all possible splits and selecting the one that minimizes the weighted variance of the target values in the left and right nodes.
        """
        best_feature, best_threshold, best_score = None, None, float("inf")

        if X.shape[0] <= 1:  # Handle edge case where splitting is not feasible
            return best_feature, best_threshold

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = ~left_indices

                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue  # Skip if one of the groups is empty

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
        """
        Predict the output for a given input using a decision tree.

        This is a helper method used by the ensemble methods to generate predictions.

        Parameters
        ----------
        tree : tuple or float
            The decision tree used for prediction. It can be a tuple representing a decision node or a float representing a leaf node.
        X : numpy.ndarray
            The input features to predict.

        Returns
        -------
        predictions : numpy.ndarray
            The predicted values for the input features.

        Notes
        -----
        If the tree is a leaf node, it returns the same prediction for all inputs. Otherwise, it recursively traverses the tree to make predictions.
        """
        if not isinstance(tree, tuple):
            return np.full(X.shape[0], tree)
        feature, threshold, left_tree, right_tree = tree
        left_indices = X[:, feature] < threshold
        right_indices = ~left_indices
        predictions = np.empty(X.shape[0])
        predictions[left_indices] = self._predict_tree(left_tree, X[left_indices])
        predictions[right_indices] = self._predict_tree(right_tree, X[right_indices])
        return predictions