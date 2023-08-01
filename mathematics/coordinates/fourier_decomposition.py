import numpy as np
from numpy import inf, exp, pi, ndarray, sin, cos
from scipy.integrate import quad

from typing import Callable

import matplotlib.pyplot as plt


def discrete_fourier(thetas: ndarray, fs: ndarray, max_order: int):
    """
    Given discrete data of scaler-value function on a circle,
    compute n-th order fourier coefficient

    thetas: evenly spaced in [0, 2*pi)
    """
    d_theta = 2*pi/len(thetas)
    a = np.zeros(shape=(max_order,))
    b = np.zeros(shape=(max_order,))

    a[0] = np.sum(fs*d_theta)/(2*pi)
    for n in range(1, max_order):
        a[n] = np.sum(cos(n*thetas) * fs * d_theta)/pi
        b[n] = np.sum(sin(n*thetas) * fs * d_theta)/pi
    return a, b


def fourier_decomposition(func: Callable, max_order: int, return_error: bool = False):
    a = np.zeros(shape=(max_order,))
    b = np.zeros(shape=(max_order,))
    errors_a = np.zeros(shape=(max_order,))
    errors_b = np.zeros(shape=(max_order,))

    integrand_0 = lambda t: func(t) / (2*pi)
    [a[0], errors_a[0]] = quad(integrand_0, 0, 2*pi)
    for n in range(1, max_order):
        integrand_c = lambda t: cos(n*t) * func(t) / pi
        integrand_s = lambda t: sin(n*t) * func(t) / pi
        [a[n], errors_a[n]] = quad(integrand_c, 0, 2*pi)
        [b[n], errors_b[n]] = quad(integrand_s, 0, 2*pi)
    if return_error:
        return a, b, errors_a, errors_b
    else:
        return a, b


def fourier_combination(ts: ndarray, c_cos: ndarray, c_sin: ndarray):
    ys = np.zeros_like(ts) + c_cos[0]
    for n in range(1, len(c_cos)):
        ys = ys + c_sin[n]*sin(n*ts) + c_cos[n]*cos(n*ts)
    return ys


def demo():
    def func(theta: ndarray):
        c = cos(theta+1)
        s = sin(theta)
        return 12 * exp(c + abs(s**3))

    ts = 2 * pi * np.arange(1000)/1000
    fs = func(ts)

    # a, b = fourier_decomposition(func, max_order=7)
    a, b = discrete_fourier(ts, fs, max_order=7)

    # plt.figure(dpi=200)
    plt.plot(ts, fs, label='$exact$')
    for i in [3, 5, 7]:
        y2 = fourier_combination(ts, a[:i], b[:i])
        plt.plot(ts, y2, label=f'{i}th order')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    demo()







