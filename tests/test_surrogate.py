import numpy as np
from src.surrogate_model import surrogate_model
from src.functions import rastrigin


def test_surrogate_model():
    surrogate = surrogate_model()

    for _ in range(10):
        x = np.random.uniform(-5.12, 5.12, 10)
        y = rastrigin(x)
        surrogate.add_sample(x, y)

    assert surrogate.train() is True

    x_test = np.random.uniform(-5.12, 5.12, 10)
    pred = surrogate.predict(x_test)
    assert pred is not None
    assert isinstance(pred, np.ndarray)

    surrogate_small = surrogate_model(min_samples=10)
    assert surrogate_small.predict(x_test) is None
