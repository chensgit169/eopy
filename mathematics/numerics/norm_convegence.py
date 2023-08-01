"""
date: 2023-1-18
"""
import numpy as np
from numpy.linalg import norm


def random_sum_1_vector(dim: int, non_negative: bool = True):
    v = np.random.random(size=dim)
    if not non_negative:
        v -= 0.5
    return v/norm(v)


def random_orthogonal_matrix(dim: int):
    """
    Generate orthogonal matrix randomly by Schmidt procedure.
    """
    u = np.zeros(shape=(dim, dim))
    u0 = np.random.random(size=dim) - 0.5
    u0 /= norm(u0)
    u[0] = u0
    for i in range(1, dim):
        while True:
            ui = np.random.random(size=dim) - 0.5
            ui -= (u @ ui) @ u
            if norm(ui) > 1e-10:
                ui /= norm(ui)
                u[i] = ui
                break
    return u


def contraction_norm():
    d = 10
    min_value = 1
    for _ in range(10000):
        u = random_orthogonal_matrix(dim=d)
        ea = random_sum_1_vector(dim=d)
        eb = random_sum_1_vector(dim=d)
        assert np.allclose(u@u.T, np.eye(d))
        min_value = min(min_value, ea @ eb)
    print(min_value)


if __name__ == '__main__':
   contraction_norm()













