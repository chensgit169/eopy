# multipole_decomposition.py
import numpy as np
from numpy import ndarray, sin, cos, exp, sqrt, pi, mean, abs, allclose

from scipy.special import hermite
from math import factorial

from gauss_hermite.hermite_approximation import hermite_fitting, hermite_norm, hermite_combination

import matplotlib.pyplot as plt
from matplotlib import cm

"""
This module is in particularly designed for analyzing fields of electron lenses
in contexture of electron microscopy.

Multipole-decompose scaler harmonic fields, namely fields satisfying laplacian 
equation, in 3 Dimension. Input discrete data of harmonic field, output combination
coefficients of special functions, achieving analyticalization.

The input field data is expected to be given in cylindrical coordinates, 
    (nz, nr, n_theta) = fs_raw.shape
with angular (theta) evenly spaced for each (z, r) and all coordinates ordered
in range:
    z: (-inf, inf)
    r: (0, inf)
    theta: [0, 2*pi)
Thus in practical, a data pre-handling procedure is necessary from, say, finite
element simulation results of instruments.

Note that theoretically for every multipole components it is enough to know the 
distribution on axis, namely r=0. However, raw data of r=0 is actually a superposition
from all components, thus can not be applied.

Author: Chen Wei, weichen191@mails.ucas.ac.cn
Date: 2023-1-25 
"""


def fourier_component(thetas: ndarray, fs: ndarray, n: int):
    """
    Extract n-th order fourier component of field.

    Coordinate convention:
    (nz, nr, n_theta) = fs.shape
    """
    nt = len(thetas)
    if n == 0:
        zero_components = np.sum(fs, axis=2) / nt
        return zero_components
    elif n > 0:
        cos_component = np.tensordot(fs, cos(n * thetas), axes=([2], [0])) * 2 / nt
        sin_component = np.tensordot(fs, sin(n * thetas), axes=([2], [0])) * 2 / nt
        return cos_component, sin_component
    else:
        raise ValueError(f'n should be non-negative integer')


def on_axis_function(zs: ndarray, rs: ndarray, fs: ndarray, n: int,
                     hermite_fit_order: int = 50, poly_fit_deg: int = 15):
    """
    Extract f(z, r=0) of data n-th order component by polynomial-fitting values
    out of axis, fitting by hermite functions. Other methods might be used to
    improve in future development.

    poly_fit_deg is a control parameter for accuracy and efficiency
    """
    nz = fs.shape[0]
    reduced_fs = fs / (rs ** n)
    f0 = np.zeros(nz)
    for i in range(nz):
        poly_coe = np.polyfit(rs, reduced_fs[i], deg=poly_fit_deg)
        f0[i] = poly_coe[-1]
        # optional:
        # f0[i] = np.polyfit(rs, fs[i], deg=poly_fit_deg)[-1-n]
    c_f0 = hermite_fitting(zs, f0, hermite_fit_order)
    return f0, c_f0


def multipole_reconstructed(c_f0: ndarray, m: int):
    """
    Reconstruct callable function from hermitian coefficients of multipole,
    which evaluates field in an analytical way.

    Structure for sin and cos components are the same. Total field should be
    able to be constructed by linear combination.

    m: multi_pole order
    max_n: max r-power series order
    """
    def fs_constructed(zs: ndarray, rs: ndarray, max_n: int):
        p = np.poly1d([0])
        for n, c in enumerate(c_f0):
            p = p + c * hermite(n) / hermite_norm(n) ** 0.5
        exponents = exp(-zs ** 2 / 2)

        # f0 = p(x) * exp(-x**2/2) where p(x) is a polynomial
        fs = np.zeros(shape=(len(zs), len(rs)))
        for n in range(max_n):
            c_mn = factorial(m) / ((-4) ** n * factorial(n) * factorial(m + n))
            fz_2n = p(zs) * exponents  # 2n-th order derivative of f0
            r_power = rs ** (2*n + m)
            fs = fs + c_mn * fz_2n[:, None] * r_power
            # taking derivatives twice
            p = p.deriv(2) - 2 * p.deriv(1) * np.poly1d([1, 0]) + p * np.poly1d([1, 0, 0])
        return fs
    return fs_constructed


def demo():
    def dipole_field(r, theta, z, d=1, z0=0):
        x = r[:, None] * cos(theta)
        y = r[:, None] * sin(theta)
        r2_1 = (x-d)**2 + y**2
        r2_2 = (x+d)**2 + y**2
        z2 = (z-z0)**2
        r1 = sqrt(z2[:, None, None] + r2_1)
        r2 = sqrt(z2[:, None, None] + r2_2)
        return 1e5*(1/r1 - 1/r2)

    # original data
    zs = np.linspace(-8, 8, 200)
    rs = np.logspace(-4, -1, 103)
    ts = np.linspace(0, 2*pi, 104, endpoint=False)  # symmetrically even points suggested
    fs_original = dipole_field(r=rs, theta=ts, z=zs)

    # 0-th order, expectedly 0:
    round_component = fourier_component(ts, fs_original, n=0)
    f0_0, c_f0_0 = on_axis_function(zs, rs, round_component, n=0)
    fs_0 = multipole_reconstructed(c_f0_0, 0)(zs, rs, 10)
    print(f'zero-th order component is 0 : {allclose(fs_0, np.zeros_like(fs_0))}')

    # higher components are expectedly all 0
    cos_2, sin_2 = fourier_component(ts, fs_original, n=2)
    print(f'2nd sin component is 0 : {allclose(sin_2, np.zeros_like(sin_2))}')
    print(f'2nd cos component is 0 : {allclose(cos_2, np.zeros_like(cos_2))}')

    # sin components are expectedly zero
    cos_1, sin_1 = fourier_component(ts, fs_original, n=1)
    print(f'1st sin component is 0 : {allclose(sin_1, np.zeros_like(sin_1))}')

    # check hermite fitting
    f0, c_f0_1 = on_axis_function(zs, rs, cos_1, n=1)
    f0_reproduced = hermite_combination(zs, c_f0_1)
    print(f'average relative error for f0 fitting = {mean(abs(f0/f0_reproduced-1))}')
    # plt.plot(zs, f0)
    # plt.plot(zs, f0_reproduced)
    # plt.yscale('log')
    # plt.show()

    # reconstruct callable field function
    func_dipole = multipole_reconstructed(c_f0_1, 1)
    fs_reconstructed = func_dipole(zs=zs, rs=rs, max_n=10)[:, :, None] * cos(ts)
    print(f'average relative error for field reconstruction = {mean(abs(fs_original/fs_reconstructed - 1))}')

    # cross-section of theta=0
    r, z = np.meshgrid(rs, zs)
    plt.figure()
    ax3d = plt.axes(projection='3d')
    surf1 = ax3d.plot_surface(z, r, func_dipole(zs=zs, rs=rs, max_n=10) * cos(0), cmap=cm.coolwarm, label='original')
    surf2 = ax3d.plot_wireframe(z, r, dipole_field(r=rs, z=zs, theta=np.array(0))[:, :, 0], label='reconstructed')
    surf1._facecolors2d = surf1._facecolor3d
    surf1._edgecolors2d = surf1._edgecolor3d
    surf2._facecolors2d = surf2._facecolors
    surf2._edgecolors2d = surf2._edgecolors
    ax3d.set_title('Reconstruction of dipole field \n($\\theta=0 \degree$ cross-section)')
    ax3d.set_xlabel('z')
    ax3d.set_ylabel('r')
    ax3d.set_zlabel('field value', rotation=0)
    ax3d.legend()
    # plt.savefig('./figures/demo_dipole_reconstruction_.png', rotation=180)
    plt.show()

