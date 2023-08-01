import scipy.integrate as integrate
from scipy.constants import mu_0, m_e, c, e
import numpy as np
from numpy import ndarray, cos, pi
from typing import Callable


def simple_coil_psi(z: float, r: float, **kwargs):
    i = kwargs['I']
    coil_radius = kwargs['R']

    def psi_integrand(phi: ndarray, rs: float):
        if z == 0 and rs == coil_radius:
            raise ValueError("Can't compute field on the coil!")
        return mu_0 * i / (4 * pi) * rs * coil_radius * (coil_radius - rs * cos(phi)) / (z**2 + rs**2 + coil_radius**2 - 2*rs*coil_radius*cos(phi))**1.5

    return integrate.dblquad(psi_integrand, 0, r, lambda phi: 0, lambda phi: 2*pi)[0]


def simple_coil_psi_z(z: float, r: float, **kwargs):
    i = kwargs['I']
    coil_radius = kwargs['R']

    def psi_integrand(phi: ndarray, rs: float):
        if z == 0 and rs == coil_radius:
            raise ValueError("Can't compute field on the coil!")
        return mu_0 * i / (4 * pi) * (-3 * z) * rs * coil_radius * (coil_radius - rs * cos(phi)) / (z**2 + rs**2 + coil_radius**2 - 2*rs*coil_radius*cos(phi))**2.5

    return integrate.dblquad(psi_integrand, 0, r, lambda phi: 0, lambda phi: 2*pi)[0]


def psi_angular_momentum(z: float, r: float, psi: Callable = simple_coil_psi, **kwargs):
    print('Note: unit is kg * m^2 * s^-1')
    return e * psi(z, r, **kwargs)


if __name__ == '__main__':
    r_coil = 0.05
    current = 100
    print(psi_angular_momentum(0, r_coil/2, I=current, R=r_coil))




