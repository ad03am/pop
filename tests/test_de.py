from src.main import DifferentialEvolution
from src.functions import rastrigin, rosenbrock


def test_rastrigin():
    dim = 10
    bounds = [(-5.12, 5.12)] * dim
    de = DifferentialEvolution(rastrigin, bounds)
    solution, fitness = de.optimize()
    assert fitness < 100


def test_rosenbrock():
    dim = 10
    bounds = [(-2.048, 2.048)] * dim
    de = DifferentialEvolution(rosenbrock, bounds)
    solution, fitness = de.optimize()
    assert fitness < 100
