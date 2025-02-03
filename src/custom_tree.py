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
    def __init__(self, max_depth=7, min_samples_split=5, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        
    def _find_best_split_vectorized(self, X, y):
        m, n = X.shape
        if m <= self.min_samples_split:
            return None, None, None

        n_percentiles = min(10, m // 10) 
        percentiles = np.linspace(0, 100, n_percentiles)
        
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        
        feature_masks = []
        thresholds_per_feature = []
        
        for feature in range(n):
            thresholds = np.percentile(X[:, feature], percentiles)
            feature_values = X[:, feature].reshape(-1, 1)
            threshold_values = thresholds.reshape(1, -1)
            masks = feature_values <= threshold_values
            
            valid_splits = np.logical_and(
                masks.sum(axis=0) >= self.min_samples_leaf,
                (~masks).sum(axis=0) >= self.min_samples_leaf
            )
            
            if np.any(valid_splits):
                feature_masks.append(masks[:, valid_splits])
                thresholds_per_feature.append(thresholds[valid_splits])
        
        if not feature_masks:
            return None, None, None
            
        y_squared = y ** 2
        total_sum = np.sum(y)
        total_sum_squared = total_sum ** 2
        total_squared_sum = np.sum(y_squared)
        n_samples = len(y)
        
        for feature_idx, (masks, thresholds) in enumerate(zip(feature_masks, thresholds_per_feature)):
            left_sums = np.dot(masks.T, y)
            left_counts = masks.sum(axis=0)
            
            right_sums = total_sum - left_sums
            right_counts = n_samples - left_counts
            
            left_means = left_sums / left_counts
            right_means = right_sums / right_counts
            
            left_squared_means = left_means ** 2
            right_squared_means = right_means ** 2
            
            mse = (total_squared_sum - 
                  left_counts * left_squared_means - 
                  right_counts * right_squared_means) / n_samples
            
            min_mse_idx = np.argmin(mse)
            if mse[min_mse_idx] < best_mse:
                best_mse = mse[min_mse_idx]
                best_feature = feature_idx
                best_threshold = thresholds[min_mse_idx]
        
        return best_feature, best_threshold, best_mse

    def _build_tree(self, X, y, depth=0):
        node = Node()
        
        if depth >= self.max_depth or len(y) <= self.min_samples_split:
            node.is_leaf = True
            node.value = np.mean(y)
            return node

        feature, threshold, mse = self._find_best_split_vectorized(X, y)
        
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