import numpy as np
from paraxial_from_axis import eb_z_over_mc, eb_z_z_over_mc
from numpy import ndarray
from scipy.constants import c
import matplotlib.pyplot as plt
import scipy.integrate as integrate


def coil_assembly_ebz(z: ndarray, n: int = 2000, **kwargs):
    rho = kwargs['rho']
    r_coil = kwargs['R']
    current = kwargs['I']

    d = r_coil / rho
    displacements = d * np.arange(0, n, 1)
    zs = np.array(z).reshape(1, -1) - displacements.reshape(-1, 1)
    eb = eb_z_over_mc(zs, I=current, R=r_coil)
    return np.sum(eb, axis=0)


def coil_assembly_ebz_z(z: ndarray, n: int = 2000, **kwargs):
    rho = kwargs['rho']
    r_coil = kwargs['R']
    current = kwargs['I']

    d = r_coil / rho
    displacements = d * np.arange(0, n, 1)
    zs = np.array(z).reshape(1, -1) - displacements.reshape(-1, 1)

    eb = eb_z_z_over_mc(zs, I=current, R=r_coil)
    return np.sum(eb, axis=0)


def stormer_equation(y: ndarray, z: float, **kwargs):
    g0 = kwargs['g']
    r_coil = kwargs['R']
    current = kwargs['I']
    rho = kwargs['rho']

    r, r_prime = y
    ebz = coil_assembly_ebz(z, rho=rho, I=current, R=r_coil)
    ebz_z = coil_assembly_ebz_z(z, rho=rho, I=current, R=r_coil)

    mu2 = g0**2 - (ebz/2)**2 * r**2
    r_prime2 = (1 + r_prime**2) / mu2 * (ebz/2 * ebz_z/2 * r**2 * r_prime - (ebz/2)**2 * r)
    return [r_prime, r_prime2]


def g_over_mc(r_prime_0: float, z_prime_0: float):
    v_0 = (r_prime_0**2 + z_prime_0**2)**0.5
    gamma_0 = 1 / (1 - (v_0/c)**2)
    g0 = gamma_0 * v_0 / c
    return g0


def trajectory(zs: ndarray, *initial_conditions, **kwargs):
    r_prime_0, z_prime_0 = initial_conditions
    g0 = g_over_mc(r_prime_0, z_prime_0)
    r_coil = kwargs['R']
    current = kwargs['I']
    rho = kwargs['rho']

    def ode_integrand(y, z):
        return stormer_equation(y, z, g=g0, rho=rho, I=current, R=r_coil)
    print(r_prime_0/z_prime_0)
    return integrate.odeint(ode_integrand, [0, r_prime_0/z_prime_0], zs)[:, 0]


if __name__ == '__main__':
    r_coil = 0.04
    current = 0.3
    n = 2000
    rho = 400
    zs = np.linspace(-0.1, 1.5, 1000)
    bz = coil_assembly_ebz(zs, n, rho=rho, I=current, R=r_coil)
    plt.figure(figsize=(8, 6))
    plt.title(f'$B_z$ of {n} coils (I={current}A, R={100*r_coil}cm, L={n/rho * 100 * r_coil}cm)')
    plt.plot(100*zs, 1000*bz)
    plt.xlabel('$z/cm$', fontsize=12)
    plt.ylabel('$B_z/mT$', fontsize=12)
    # plt.savefig('./figures/coil_assembly_bz_first_try.png')
    plt.show()

    r_prime_0s = 1e-5 * np.arange(15, 40, 5)
    plt.figure(figsize=(8, 6))
    for r_prime in r_prime_0s:
        plt.plot(100*zs, 100*trajectory(zs, r_prime*c, 0.2*c, rho=rho, R=r_coil, I=current), label='$v_{r0}$='+f'{r_prime:.2E}c')
    plt.title(f'electron trajectories in z-r plane (R={100 * r_coil}cm, I={current}A, N={n}, '+'$v_{z0}$=' + '0.2c)')
    plt.xlabel('z/cm')
    plt.ylabel('r/cm')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('./figures/trajectory_first_try.png')
    plt.show()
