from scipy.special import hermite
from scipy.integrate import quad
import numpy as np
from numpy import inf, exp, pi, ndarray

from math import factorial
from typing import Callable

import matplotlib.pyplot as plt


def derivative(coefficients: ndarray, n: int):
    """
    Compute derivative of Hermite series:
        f(x)=\sum_n c_n * H_n(x)
    from recursion relation:
        dH_{n-1}(x)/dx = H_n(x) - 2*x*H_{n-1}(x)
    f'(x)=\sum c_{n-1} * H_n(x) - 2*x*\sum_n c_n * H_n(x)
    """
