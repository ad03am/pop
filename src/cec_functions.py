import numpy as np


class CECFunctions:
    @staticmethod
    def shifted_sphere(x, shift=None):
        if shift is None:
            shift = np.zeros_like(x)
        z = x - shift
        return np.sum(z**2)

    @staticmethod
    def ackley(x, shift=None):
        if shift is None:
            shift = np.zeros_like(x)
        z = x - shift
        n = len(z)
        sum1 = np.sum(z**2)
        sum2 = np.sum(np.cos(2 * np.pi * z))
        term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)
        return term1 + term2 + 20 + np.e

    @staticmethod
    def shifted_rotated_high_conditioned_elliptic(x, shift=None, rotation=None):
        if shift is None:
            shift = np.zeros_like(x)
        if rotation is None:
            rotation = np.eye(len(x))
        z = np.dot(rotation, (x - shift))
        d = len(x)
        powers = 2 * np.arange(d) / (d - 1)
        return np.sum((10**powers) * z**2)

    @staticmethod
    def shifted_rotated_griewank(x, shift=None, rotation=None):
        if shift is None:
            shift = np.zeros_like(x)
        if rotation is None:
            rotation = np.eye(len(x))
        z = np.dot(rotation, (x - shift))
        return (
            1
            + np.sum(z**2) / 4000
            - np.prod(np.cos(z / np.sqrt(np.arange(1, len(x) + 1))))
        )


def generate_rotation_matrix(dim):
    A = np.random.randn(dim, dim)
    Q, R = np.linalg.qr(A)
    return Q


def generate_shift_vector(dim, bounds):
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    return np.random.uniform(lower, upper)
