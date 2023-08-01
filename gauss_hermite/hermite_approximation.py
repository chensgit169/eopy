# hermite_approximation.py
from scipy.special import hermite
from scipy.integrate import quad
import numpy as np
from numpy import inf, exp, pi, ndarray

from math import factorial
from typing import Callable

import matplotlib.pyplot as plt


"""
By virtue of Hermitian polynomials' orthogonality over (-inf, inf) 
with weight function exp(-x**2) to construct orthonormal bases, which
can be utilized to expand or re-build functions which decay quickly
as |x| -> inf.

Practically, the input function may have no closed form, so the coefficients
need to be extracted by fitting methods from discrete data.

This module is in particularly designed for analyzing fields of electron lenses.

Author: Chen Wei, weichen191@mails.ucas.ac.cn
Date: 2023-1-16 23:30
"""


def hermite_norm(n: int):
    return 2 ** n * factorial(n) * pi ** 0.5


def orthonormal_hermite(xs: ndarray, n: int):
    h_n = hermite(n)
    return h_n(xs) * exp(-xs**2/2) / hermite_norm(n)**0.5


def hermite_decomposition(func: Callable, order: int):
    coefficients = np.zeros(shape=(order,))
    errors = np.zeros(shape=(order,))
    for n in range(order):
        integrand = lambda x: orthonormal_hermite(x, n=n) * func(x)
        [coefficients[n], errors[n]] = quad(integrand, -inf, inf)
    return coefficients, errors


def hermite_fitting(xs: ndarray, ys: ndarray, order: int):
    """
    Extract Hermite-coefficients of function in 1 dimension from
    discrete data. Integration with rectangle rule was implemented,
    which may be rather rough and can be replaced by more advanced
    method in the future (still under development).
    """
    coefficients = np.zeros(shape=(order,))
    for n in range(order):
        # xs should be ordered
        dxs = xs[1:] - xs[:-1]
        xs_central = (xs[1:] + xs[:-1])/2
        ys_central = (ys[1:] + ys[:-1])/2
        fs = orthonormal_hermite(xs_central, n)
        coefficients[n] = np.sum(ys_central * fs * dxs)
    return coefficients


def hermite_combination(xs: ndarray, coefficients: ndarray):
    ys = np.zeros_like(xs)
    for i in range(len(coefficients)):
        ys = ys + coefficients[i] * orthonormal_hermite(xs, i)
    return ys


def demo_decomposition():
    def func(xs: ndarray):
        return exp(-xs**4) * (xs-1)**3

    x = np.linspace(-3, 3, 400)
    y1 = func(x)
    cs, es = hermite_decomposition(func, 50)
    plt.figure(dpi=200)
    plt.plot(x, y1, label='$(x-1)^3e^{-x^4}$')

    for i in [5, 10, 50]:
        y2 = hermite_combination(x, cs[:i])
        plt.plot(x, y2, label=f'{i}th order')

    plt.title("Hermite approximation of local-distributed function \n(though decomposing function)")
    plt.xlabel("x")
    plt.ylabel("f(x)", rotation=0)
    plt.legend()
    # plt.show()
    plt.savefig('./figures/Hermite_decomposition_demo.png')


def demo_fitting():
    def func(xs: ndarray):
        # with gaussian noise
        return exp(-xs**4) * (xs-1)**3 + 0.05*np.random.normal(size=len(xs))

    x = np.linspace(-3, 3, 400)
    y1 = func(x)
    cs = hermite_fitting(x, y1, 50)

    plt.figure(dpi=200)
    plt.plot(x, y1, label='$(x-1)^3e^{-x^4}$+noise')
    for i in [5, 10, 50]:
        y2 = hermite_combination(x, cs[:i])
        plt.plot(x, y2, label=f'{i}th order')
    plt.title("Hermite approximation of local-distributed function\n(through fitting data)")
    plt.xlabel("x")
    plt.ylabel("f(x)", rotation=0)
    plt.legend()
    # plt.show()
    plt.savefig('./figures/Hermite_fitting_demo.png')
    return None


def coefficients(max_order: int = 50):
    cs = np.zeros(shape=(max_order, max_order))
    for i in range(max_order):
        cs[i, -(i+1):] = hermite(i).coefficients
        print(cs[i, -(i+1):])
    # np.save(f'hermite_poly_coefficients_first{max_order}', cs)


if __name__ == '__main__':
    coefficients()

