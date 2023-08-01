from scipy.integrate import solve_ivp
from scipy.constants import c, e, m_e

import numpy as np
from numpy import ndarray, cross, concatenate
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt


"""
Solving Newton's equation of electron under Lorentz force
of given magnetic and electric field by numerical integration
on the given time interval.

        F = -e(E + v × B)
        r'' = -e/m(E + v × B)

A non-relativistic version is implemented, yet it should be straight
forward to generalized:
        d(\gamma v) = -e/m(E + v/c × B)
only more complicated in expression and might need to choose suitable
scales to ensure numerical stability.

S.I. unit is used here. But it is suggested to absorb these constant
into unit of the input fields E and B in future development.

This module should be able to be improved by virtue of sympy.

Author: Chen Wei weichen191@mails.ucas.ac.cn
Date: 2023-1-19
"""


def lorentz_force_integration(r0: ndarray, v0: ndarray, ts: ndarray,
                              magnetic_field: Callable = None, electric_field: Callable = None,
                              **kwargs):
    """
    Unit:
    r -- mm
    v -- c
    t -- mm/c
    B -- T
    e/(1000*m_e*c) = 0.586.6792055096206 T^-1 * mm^-1 = 9.67335876074641e-11 V^-1 * (mm/c)^-1
    """
    def force_in_6d(t: float, y: ndarray):
        x, y, z, vx, vy, vz = y

        force = np.zeros(shape=3)
        if electric_field:
            force += - e/(1000*m_e*c) * c * electric_field(x, y, z, t, **kwargs)
        if magnetic_field:
            force += - e/(1000*m_e*c) * cross([vx, vy, vz], magnetic_field(x, y, z, t, **kwargs))
            print(force)
        fx, fy, fz = force

        return np.array([vx, vy, vz, fx, fy, fz])

    solution = solve_ivp(force_in_6d, t_span=[ts[0], ts[-1]], y0=concatenate((r0, v0)), t_eval=ts)
    ts = solution.t
    trajectory = solution.y
    return ts, trajectory


def demo():
    def demo_magnetic_field(x, y, z, t):
        """
        An artificial magnetic field to demonstrate
        magnetic-focusing, physically this kind of
        field does not exist.
        """
        b0 = 5
        l0 = 1
        [bx, by, bz] = [0, 0, b0 * z / l0]
        return np.array([bx, by, bz])

    # Computation
    r0 = np.array([0, 0, 1])
    v0 = np.array([0.1, 0.2, 0.3])
    ts = np.linspace(0, 8, 1000)
    ts, trajectory = lorentz_force_integration(r0, v0, ts=ts, magnetic_field=demo_magnetic_field)
    x, y, z = trajectory[:3]

    # Plot in 3D real space
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title('Demo: Magnetic field $\\vec B=\\frac{z}{L_0}B_0\\vec e_z$')
    ax.plot(x, y, z, label='electron trajectory')
    ax.plot(x[0], y[0], z[0], '*', label='starting point')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    mpl.rcParams['legend.fontsize'] = 10
    ax.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig('./figures/mag_focusing_demo.png')
    return None


if __name__ == '__main__':
    demo()

