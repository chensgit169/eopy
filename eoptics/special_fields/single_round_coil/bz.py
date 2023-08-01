import scipy.integrate as integrate
from scipy.constants import mu_0, m_e, c, e
import numpy as np
from numpy import ndarray, cos, pi
import matplotlib.pyplot as plt


def simple_coil_bz(z: float, r: float, **kwargs):
    i = kwargs['I']
    r_c = kwargs['R']

    def bz_integrand(phi: ndarray):
        if z == 0 and r == r_c:
            raise ValueError("Can't compute field on the coil!")
        return mu_0 * i / (4 * pi) * (r_c - r * cos(phi)) * r_c / (z**2 + r**2 + r_c**2 - 2*r*r_c*cos(phi))**1.5

    return integrate.quad(bz_integrand, 0, 2*pi)[0]


def simple_coil_bz_z(z: float, r: float, **kwargs):
    i = kwargs['I']
    r_c = kwargs['R']

    def bz_integrand(phi: ndarray):
        if z == 0 and r == r_c:
            raise ValueError("Can't compute field on the coil!")
        return mu_0 * i / (4 * pi) * (-3 * z) * (r_c - r * cos(phi)) * r_c / (z**2 + r**2 + r_c**2 - 2*r*r_c*cos(phi))**2.5

    return integrate.quad(bz_integrand, 0, 2*pi)[0]


def bz_along_r(r: float):
    r_coil = 0.05
    current = 100
    zs = np.linspace(0, 0.15, 100)
    bz = np.zeros_like(zs)
    for i in range(zs.size):
        z = zs[i]
        bz[i] = simple_coil_bz(z, r, I=current, R=r_coil)

    plt.figure(figsize=(8, 6))
    if r == 0:
        bz_analytical = mu_0 * current / 2 * r_coil**2 / (r_coil**2 + zs**2)**(3/2)
        plt.plot(100*zs, 1000 * bz, '*-', label='computational')
        plt.plot(100*zs, 1000 * bz_analytical, label='analytical')
        r_line_name = 'axis'
        plt.legend(fontsize=12)
    else:
        plt.plot(100*zs, 1000 * bz)
        r_line_name = f'r={r}'
    plt.title(f'$B_z$ of SRC along ' + r_line_name + f' (R={100*r_coil}cm, I={current}A)', fontsize=14)
    plt.xlabel('$z/cm$', fontsize=12)
    plt.ylabel('$B_z/mT$', fontsize=12)
    plt.savefig('./figures/src_bz_along_'+r_line_name+'.png')


if __name__ == '__main__':
    bz_along_r(0)


