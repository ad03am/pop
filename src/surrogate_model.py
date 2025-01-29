from sklearn.tree import DecisionTreeRegressor as SDecisionTreeRegressor
from custom_tree import DecisionTreeRegressor as CDecisionTreeRegressor
import numpy as np


class surrogate_model:
    def __init__(self, min_samples=10):
        self.model = CDecisionTreeRegressor()
        self.min_samples = min_samples
        self.history_X = []
        self.history_y = []

    def add_sample(self, x, y):
        self.history_X.append(x)
        self.history_y.append(y)

    def train(self):
        if len(self.history_X) < self.min_samples:
            return False

        X = np.array(self.history_X)
        y = np.array(self.history_y)
        self.model.fit(X, y)
        return True

    def predict(self, X):
        if len(self.history_X) < self.min_samples:
            return None

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        return self.model.predict(X)
