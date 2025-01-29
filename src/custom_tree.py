import numpy as np


class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.value = None
        self.is_leaf = False


class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def _calculate_mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _find_best_split(self, X, y):
        m, n = X.shape
        if m <= self.min_samples_split:
            return None, None, None

        best_mse = self._calculate_mse(y)
        best_feature = None
        best_threshold = None

        for feature in range(n):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if (
                    np.sum(left_mask) < self.min_samples_leaf
                    or np.sum(right_mask) < self.min_samples_leaf
                ):
                    continue

                left_mse = self._calculate_mse(y[left_mask])
                right_mse = self._calculate_mse(y[right_mask])
                current_mse = (
                    np.sum(left_mask) * left_mse + np.sum(right_mask) * right_mse
                ) / m

                if current_mse < best_mse:
                    best_mse = current_mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_mse

    def _build_tree(self, X, y, depth=0):
        node = Node()

        if (
            depth >= self.max_depth
            or len(np.unique(y)) == 1
            or len(y) <= self.min_samples_split
        ):
            node.is_leaf = True
            node.value = np.mean(y)
            return node

        feature, threshold, mse = self._find_best_split(X, y)

        if feature is None:
            node.is_leaf = True
            node.value = np.mean(y)
            return node

        node.feature = feature
        node.threshold = threshold

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        self.root = self._build_tree(X, y)
        return self

    def _predict_single(self, x, node):
        if node.is_leaf:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        predictions = np.array([self._predict_single(x, self.root) for x in X])
        return predictions
