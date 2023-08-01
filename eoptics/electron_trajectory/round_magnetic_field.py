import numpy as np
from numpy import ndarray

from typing import Callable

"""
From the fact that a harmonic field with axial symmetry preserves 
angular momentum and has only the 0-th order multipole component,
dynamic equation for electron in static electromagnetic field can 
be reduced into 1 dimension. It is also enough to reconstruct whole
field from distribution on axis.

This module reconstructs magnetic field from B_z(z, r=0) and computes
trajectory of electron by integrating Stormer's equation. 
"""


def round_mag_reconstruct(c_f0: ndarray, **kwargs):

    def get_psi():
        return None
    return get_psi


def stormer_equation(r: ndarray, rp: ndarray, c_quantities: list, psi: Callable):
    angular_m, energy = c_quantities
    force = None
    return np.array([force, rp])


def lorentz_force_integration(r0: ndarray, rp_0: ndarray, zs: ndarray, c_f0: ndarray, **kwargs):
    def integrand():
        return None
    return None
