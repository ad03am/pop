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
        
    def _calculate_mse_vectorized(self, y_true, masks):
        n_splits = masks.shape[0]
        n_samples = len(y_true)
        
        sums = np.dot(masks, y_true)
        counts = masks.sum(axis=1)
        means = sums / counts
        
        means_expanded = means.reshape(-1, 1)
        squared_diff = masks * (y_true - means_expanded) ** 2
        
        return np.sum(squared_diff, axis=1) / n_samples

    def _find_best_split(self, X, y):
        m, n = X.shape
        if m <= self.min_samples_split:
            return None, None, None

        total_mean = np.mean(y)
        best_mse = np.mean((y - total_mean) ** 2)
        best_feature = None
        best_threshold = None

        percentiles = np.linspace(0, 100, min(20, m // 5))
        
        for feature in range(n):
            thresholds = np.percentile(X[:, feature], percentiles)
            if len(thresholds) < 2:
                continue
                
            feature_values = X[:, feature].reshape(-1, 1)
            threshold_values = thresholds.reshape(1, -1)
            left_masks = feature_values <= threshold_values
            
            valid_splits = np.logical_and(
                left_masks.sum(axis=0) >= self.min_samples_leaf,
                (~left_masks).sum(axis=0) >= self.min_samples_leaf
            )
            
            if not np.any(valid_splits):
                continue
                
            valid_thresholds = thresholds[valid_splits]
            valid_left_masks = left_masks[:, valid_splits]
            
            left_mses = np.zeros(len(valid_thresholds))
            right_mses = np.zeros(len(valid_thresholds))
            
            for i, mask in enumerate(valid_left_masks.T):
                left_mean = np.mean(y[mask])
                right_mean = np.mean(y[~mask])
                left_mses[i] = np.sum((y[mask] - left_mean) ** 2)
                right_mses[i] = np.sum((y[~mask] - right_mean) ** 2)
            
            total_mses = (left_mses + right_mses) / m
            
            best_split_idx = np.argmin(total_mses)
            if total_mses[best_split_idx] < best_mse:
                best_mse = total_mses[best_split_idx]
                best_feature = feature
                best_threshold = valid_thresholds[best_split_idx]

        return best_feature, best_threshold, best_mse

    def _build_tree(self, X, y, depth=0):
        node = Node()
        
        if depth >= self.max_depth or len(y) <= self.min_samples_split:
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

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return np.array([self._predict_single(x, self.root) for x in X])
    
    def _predict_single(self, x, node):
        if node.is_leaf:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)