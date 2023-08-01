from scipy.constants import e, m_e, c
from numpy import ndarray, cos, sin, sqrt
import matplotlib.pyplot as plt
import numpy as np


def uniform_field_trajectory(zs: ndarray, r0: float, v_r_0: float, v_z_0: float, v_theta_0: float, b: float):
    k = e * b / (m_e * v_z_0)
    omega = e * b / (2 * m_e)
    k_n = v_z_0 / (r0**2 * (v_theta_0 - omega))

    ek_r = m_e * v_r_0**2 / (2*e)
    ek_t = m_e * r0**2 * v_theta_0**2 / (2*e)
    ek_xy = ek_t + ek_r
    ek_z = m_e * v_z_0**2 / (2*e)

    a = (ek_xy/ek_z - k/k_n) / k**2

    s2 = (ek_xy/ek_z - k/k_n)

    x0 = r0**2 / 2
    x0_p = r0 * v_r_0 / v_z_0
    print(k*r0)
    n_omega = m_e * r0**2 * (v_theta_0 - omega) * omega / e

    return sqrt(2*(a + (x0-a)*cos(k*zs) + x0_p*sin(k*zs)/k))


if __name__ == '__main__':
    z = np.arange(0, 0.2, 0.00001)
    r0 = 0.00001
    v_r0 = 0.001*c
    v_t0 = 0#.001*c/r0
    v_z0 = 0.1*c
    magnitude = 0.01
    r = uniform_field_trajectory(zs=z, r0=r0, v_r_0=v_r0, v_theta_0=v_t0, v_z_0=v_z0, b=0.005)
    plt.figure(dpi=256)
    plt.plot(100*z, 100*r)
    plt.xlabel('$z/cm$')
    plt.ylabel('$r/cm$')
    plt.title(f'uniform magnetic field B={magnitude*1000:.2f}mT')
    plt.text(0, np.min(100*r), f'$v^z_0$={v_z0/c:.3f}c\n$v^r_0$={v_r0/c:.3f}c\n'+'$v^{\\theta}_0$'+f'={v_t0*r0/c:.3f}c\n')
    plt.savefig('./uniform_field.png')


