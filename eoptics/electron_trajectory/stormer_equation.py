# stormer_equation.py
import numpy as np
from numpy import ndarray, exp, sign
from scipy.constants import m_e, c, e
from scipy.integrate import solve_ivp

from scipy.special import hermite
from math import factorial
from gauss_hermite.hermite_approximation import hermite_fitting, hermite_norm

import matplotlib.pyplot as plt


color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

"""
Stormer equation applies to electron motion in electro-magnetic lens 
with axial symmetry. 

This module reconstruct magnetic flux distribution (and its derivatives) 
from hermite decomposition coefficients of Bz(z, r=0) into analytic forms.
Then integrate stormer equation with to get r-z trajectory.

Author: Chen Wei weichen191@mails.ucas.ac.cn
Date: 2023-1-29
"""


def stormer_equation(r0: float, rp_0: float, zs: ndarray, c_b0: ndarray, **kwargs):
    """
    Applies to special case of magnetic field. By default, angular momentum N = 0,
    which is enough for evaluating focus length when electrons are expected to come
    into lens from outside of fields and parallel with axis.

    It should be straightforward to generalize to case when electric field exists,
    namely g relies on z.

    This function may be improved by Sympy in future development.

    Note: unit of length is mm and for numerical stability, g = \gamma * m_e * v / e
    """
    g2 = kwargs['g2']
    if 'N' in kwargs:
        angular_m = kwargs['N']
    else:
        angular_m = 0
    max_n = kwargs['max_n']

    p0 = np.poly1d([0])
    for i, a in enumerate(c_b0):
        p0 = p0 + a * hermite(i) / hermite_norm(i) ** 0.5

    def force_in_2d(z: float, y: ndarray):
        r, rp = y
        exponent = exp(-z**2 / 2)

        p = p0
        bz = 0  # B_z(z, r=0) = p(x) * exp(-x**2/2), unit: T
        s = 0  # 1/r * \int_0^r r*B_z(r) dr, unit: T * m
        ps_pz = 0  # \partial s/ \partial z, unit: T

        for n in range(max_n):
            c_n = 1 / ((-4)**n * factorial(n)**2)
            fz_2n = p(z) * exponent  # 2n-th order derivative of f0
            p = p.deriv(1) - p * np.poly1d([1, 0])
            fz_2n_p1 = p(z) * exponent  # (2n+1)-th order derivative of f0
            p = p.deriv(1) - p * np.poly1d([1, 0])

            bz += c_n * fz_2n * r ** (2*n)
            s += c_n * fz_2n * abs(r)**(2*n+1) / (2*n+2)
            ps_pz += c_n * fz_2n_p1 * abs(r)**(2*n+1) / (2*n+2)

        mu2 = g2 - s**2
        # print(f"z={z}, r={r}, r'={rp}:")
        # print(f'bz={bz}, s={s}, mu2={mu2}')
        assert mu2 >= 0, f"mu**2 must >=0 : mu2={mu2}, possible problem is that r has gone too large:" \
                         f"\nz={z}, r={r}, r'={rp}, \nbz={bz}, s={s}, mu2={mu2}"
        if angular_m == 0:
            rpp = sign(r)*(1 + rp**2) * s * (sign(r)*rp*ps_pz - bz) / mu2
        else:
            # numerical error needs to be handled carefully when N~0 !
            rpp = (1 + rp**2) * (s + angular_m/(r+1e-500)) * (rp*ps_pz - bz+angular_m/(r**2+1e-500)) / mu2
        force = np.array([rp, rpp])
        # print(f'mu2= {mu2}, force = {force}')
        return force

    solution = solve_ivp(force_in_2d, t_span=[zs[0], zs[-1]], y0=np.array([r0, rp_0]), t_eval=zs)
    zs = solution.t
    trajectory = solution.y
    return zs, trajectory


def demo(show_mag_field: bool = True):
    # magnetic field data
    from eoptics.special_fields.round_magnetic_lens.ei_kareh_model import bz_0
    zs_0 = np.linspace(-4, 4, 1000)
    [d, n, i, s] = [1, 1000, 1, 1]
    bz0 = bz_0(zs=zs_0, S=s, D=s, N=n, I=i)
    c_b0 = hermite_fitting(zs_0, bz0, order=40)
    # bz_reproduced = hermite_combination(zs_0, coefficients=c_b0)
    # plt.plot(zs_0, bz_0)
    # plt.plot(zs_0, bz_reproduced)
    # plt.show()

    # initial velocity of trajectory
    beta = 0.1
    g2 = (m_e*c*1000)**2 / (1/beta**2 - 1) / e**2  # 1000 from m -> mm

    # trajectory computation
    zs = np.linspace(-1, 1.0, 10000)
    plt.figure(dpi=160)
    plt.title(f'Focusing of EI-Kareh Model\n(N={n}, I={i}A, D={d}mm, S/D={s/d}, $v_0$={beta}c, Stormer Equation)')
    for _, r0 in enumerate(np.linspace(0.0, 0.18, 6, endpoint=False)):
        zs, trajectory = stormer_equation(r0=r0, rp_0=0.0, zs=zs, c_b0=c_b0, g2=g2, max_n=20)
        plt.plot(zs, trajectory[0], color=color_list[_])
        plt.plot(zs, -trajectory[0], color=color_list[_])
        print(f'r0={r0} done')

    plt.xlabel('z / mm')
    plt.ylabel('r / mm', size=12)
    if show_mag_field:
        plt.twinx()
        plt.plot(zs, bz_0(zs=zs, S=s, D=s, N=n, I=i), '--', color=color_list[-1])
        plt.ylabel('$B_z(z, r=0)$ / T', size=12, color=color_list[-1])

    plt.tight_layout()
    plt.savefig(f'./figures/EI-Kareh_Model_focusing_{beta}c_{r0}.png')
    # plt.show()
    return None


if __name__ == '__main__':
    demo()
