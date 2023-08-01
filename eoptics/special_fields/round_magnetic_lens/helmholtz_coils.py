import numpy as np
from numpy import ndarray
import scipy.integrate as integrate


def bz(z: ndarray, r: float, coil_radius: float):
    return None



def helmholtz_coil():
    r_coil = 1
    d = r_coil
    rs = [0, 0.5, 0.9, 1.1, 1.5, 2.0]
    zs = np.linspace(-3, 3, 200)
    bz = np.zeros_like(zs)
    for r in rs:
        for i in range(zs.size):
            bz[i] = integrate.quad(lambda phi: bz_integrand(phi, zs[i]-d/2, r, r_coil), 0, 2*pi)[0] + integrate.quad(lambda phi: bz_integrand(phi, zs[i]+d/2, r, r_coil), 0, 2*pi)[0]
        plt.plot(zs, bz, label=f'r={r}')
    # for i in range(zs.size):
    #     bz[i] = integrate.quad(lambda phi: bz_integrand(phi, zs[i]-d/2, r, r_coil), 0, 2*pi)[0] + integrate.quad(lambda phi: bz_integrand(phi, zs[i]+d/2, r, r_coil), 0, 2*pi)[0]
    #     print(zs[i])
    # plt.plot(zs, bz)
    # plt.title(f'magnetic field of Helmholtz coil along r={r}')
    plt.legend()
    plt.title(f'magnetic field of Helmholtz coil parallel with axis')
    plt.xlabel('$z$')
    plt.ylabel('$B_z$')
    plt.savefig('./figures/helmholtz_mag_field.png')

def helmholtz_coil_flux():
    r_coil = 1
    d = r_coil
    rs = list(np.arange(0.1, 1, 0.1))
    zs = np.linspace(0, 6, 200)
    psi = np.zeros_like(zs)
    for r in rs:
        for i in range(zs.size):
            psi[i] = compute_psi(r, zs[i]-d/2, coil_radius=r_coil) + compute_psi(r, zs[i]+d/2, coil_radius=r_coil)
        plt.plot(zs, psi, label=f'r={r}')
    plt.legend()
    plt.title(f'magnetic flux of Helmholtz coils')
    plt.xlabel('$z$')
    plt.ylabel('$\Psi$')
    plt.savefig('./figures/helmholtz_mag_flux2.png')

