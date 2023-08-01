import scipy.integrate as integrate
from scipy.constants import mu_0, m_e, c, e
import numpy as np
from numpy import ndarray, cos, pi
import matplotlib.pyplot as plt


def eb_z_over_mc(z: ndarray, **kwargs):
    r_coil = kwargs['R']
    current = kwargs['I']
    return e * mu_0 / (m_e * c) * current / 2 * r_coil**2 / (r_coil**2 + z**2)**(3/2)


def eb_z_z_over_mc(z: ndarray, **kwargs):
    r_coil = kwargs['R']
    current = kwargs['I']
    return e * mu_0 / (m_e * c) * current * (-3 * z) / 2 * r_coil**2 / (r_coil**2 + z**2)**(5/2)


def g_over_mc(r_prime_0: float, z_prime_0: float):
    v_0 = (r_prime_0**2 + z_prime_0**2)**0.5
    gamma_0 = 1 / (1 - (v_0/c)**2)
    g0 = gamma_0 * v_0 / c
    return g0


def stormer_equation(y: ndarray, z: float, **kwargs):
    g0 = kwargs['g']
    r_coil = kwargs['R']
    current = kwargs['I']

    r, r_prime = y
    ebz = eb_z_over_mc(z, I=current, R=r_coil)
    print(ebz)
    ebz_z = eb_z_z_over_mc(z, I=current, R=r_coil)

    mu2 = g0**2 - (ebz/2)**2 * r**2
    r_prime2 = (1 + r_prime**2) / mu2 * (ebz/2 * ebz_z/2 * r**2 * r_prime - (ebz/2)**2 * r)
    return np.array([r_prime, r_prime2])


def trajectory(zs: ndarray, *initial_conditions, **kwargs):
    r_prime_0, z_prime_0 = initial_conditions
    g0 = g_over_mc(r_prime_0, z_prime_0)
    r_coil = kwargs['R']
    current = kwargs['I']

    def ode_integrand(y, z):
        return stormer_equation(y, z, g=g0, I=current, R=r_coil)

    return integrate.odeint(ode_integrand, [0, r_prime_0/z_prime_0], zs)[:, 0]


if __name__ == '__main__':
    # print(stormer_equation([0.1, 0.01/0.2], 0, g=g_over_mc(0.01*c, 0.2*c), I=100, R=0.01))
    zs = np.linspace(-2.0, 2.0, 10000)
    initial_velocity = (0.01*c, 0.2*c)
    r_prime_0s = np.arange(0.015, 0.04, 0.005)
    r_coil = 5
    current = 5000
    plt.figure(figsize=(8, 6))
    for r_prime in r_prime_0s:
        plt.plot(100*zs, 100*trajectory(zs, r_prime, 0.2*c, R=r_coil, I=current), label='$v_{r0}$='+f'{r_prime:.3f}c')
    plt.title(f'electron trajectories in z-r plane (R={100 * r_coil}cm, I={current}A, '+'$v_{z0}$=' + '0.2c)')
    plt.xlabel('z/cm')
    plt.ylabel('r/cm')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./figures/trajectory_I{current}_R{r_coil}.png')
    # plt.show()
