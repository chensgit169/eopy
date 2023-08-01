import numpy as np
from numpy import ndarray, exp, log

from scipy.special import hermite
from math import factorial

import matplotlib.pyplot as plt
from matplotlib import cm


def hermite_harmonic(rs: ndarray, zs: ndarray, approx_order: int, **kwargs):
    """
    Construct a harmonic field from multipole decomposition,
    whose on-axis function is hermite-gaussian.
    (without angular part written)
    """
    n_h = kwargs['n_h']
    m = kwargs['m']
    
    p = hermite(n_h)
    exponents = exp(-zs ** 2 / 2)

    fs = np.zeros(shape=(len(zs), len(rs)))
    for n in range(approx_order):
        c_mn = factorial(m) / ((-4) ** n * factorial(n) * factorial(m + n))
        fz_2n = p(zs) * exponents  # 2n-th order derivative of f0
        r_power = rs ** (2*n + m)
        fs = fs + c_mn * fz_2n[:, None] * r_power
        # taking derivatives twice
        p = p.deriv(2) - 2 * p.deriv(1) * np.poly1d([1, 0]) + p * np.poly1d([1, 0, 0])
    return fs


def demo():
    zs = np.linspace(-8, 8, 200)
    rs = np.logspace(-6, -4, 100)
    n_h = 90
    m = 0
    fs = hermite_harmonic(rs, zs, approx_order=20, n_h=n_h, m=m)
    r, z = np.meshgrid(log(rs), zs)
    plt.figure()
    ax3d = plt.axes(projection='3d')
    surf1 = ax3d.plot_surface(z, r, fs, cmap=cm.coolwarm)
    # surf1._facecolors2d = surf1._facecolor3d
    # surf1._edgecolors2d = surf1._edgecolor3d
    # surf2._facecolors2d = surf2._facecolors
    # surf2._edgecolors2d = surf2._edgecolors
    ax3d.set_title(f'hermite-harmonic field: $h_{n_h}$, m={m}\n($\\theta=0 \degree$ cross-section)')
    ax3d.set_xlabel('z')
    ax3d.set_ylabel('r')
    ax3d.set_zlabel('field value', rotation=90)
    # ax3d.legend()
    # plt.savefig('./figures/demo_dipole_reconstruction_.png', rotation=180)
    plt.show()


if __name__ == '__main__':
    demo()

