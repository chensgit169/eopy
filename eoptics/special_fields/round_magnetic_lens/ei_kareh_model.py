# ei_kareh_model.py
import numpy as np
from numpy import ndarray, pi, sin, cos, exp
from scipy.integrate import quad
from scipy.special import iv
from scipy.constants import mu_0


from scipy.special import hermite
from math import factorial
from gauss_hermite.hermite_approximation import hermite_fitting, hermite_norm

import matplotlib.pyplot as plt
from matplotlib import cm

color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def bz_0_single_point(z: float, **kwargs):
    s = kwargs['S']  # mm
    d = kwargs['D']  # mm
    n = kwargs['N']  # mm
    i = kwargs['I']  # A

    c = 2*mu_0*n*i/(pi*s/1000)

    fake_zero = 1e-100
    fake_inf = 10  # to avoid divergence and save time

    def integrand(x: ndarray):
        return sin(s*x/d) * cos(2*x*z/d) / (x*iv(0, x))
    return c*quad(integrand, fake_zero, fake_inf)[0]  # T


def bz_0(zs: ndarray, **kwargs):
    bz = np.zeros_like(zs)
    for _, z in enumerate(zs):
        bz[_] = bz_0_single_point(z, **kwargs)
    return bz


def field_reconstructed(c_b0: ndarray, max_n: int):
    def bz_func(zs: ndarray, rs: ndarray):
        p = np.poly1d([0])
        for i, a in enumerate(c_b0):
            p = p + a * hermite(i) / hermite_norm(i) ** 0.5

        exponents = exp(-zs ** 2 / 2)
        bz_field = np.zeros(shape=(len(zs), len(rs)))
        for n in range(max_n):
            c_n = 1 / ((-4) ** n * factorial(n) ** 2)
            fz_2n = p(zs) * exponents  # 2n-th order derivative of f0
            bz_field = bz_field + c_n * fz_2n[:, None] * rs**(2 * n)
        return bz_field
    return bz_func


def demo_bz0():
    zs = np.linspace(0, 5, 1000)
    [d, n, i] = [1, 1000, 1]
    plt.figure(dpi=160)
    for s in [0.5, 1, 5]:
        bz0 = bz_0(zs=zs, S=s, D=s, N=n, I=i)
        plt.plot(zs/d, bz0, label=f'S/D={s/d}')

    plt.title(f'Magnetic Field on Axis of EI-Kareh Model\n(N={n}, I={i}A, D={d}mm)')
    plt.xlabel('z / D')
    plt.ylabel('$B_z$ / T')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./figures/EI-Kareh_Model_demo.png')
    # plt.show()
    return None


def demo_bz_field(choice: str = None):
    zs_0 = np.linspace(-4, 4, 1000)
    [d, n, i, s] = [1, 1000, 1, 1]
    bz0 = bz_0(zs=zs_0, S=s, D=s, N=1e4, I=0.1)
    c_b0 = hermite_fitting(zs_0, bz0, order=40)

    bz_func = field_reconstructed(c_b0, max_n=20)

    # 3D plot
    if choice == '3D':
        zs = np.linspace(-3, 3, 1000)
        rs = np.logspace(-6, 0, 100)
        plt.figure(dpi=160)
        ax3d = plt.axes(projection='3d')
        r, z = np.meshgrid(rs/d, zs/d)
        plt.title(f'Reconstructed Magnetic Field of EI-Kareh Model\n(N={n}, I={i}A, D={d}mm, S/D={s/d})')
        ax3d.plot_surface(z, r, bz_func(zs, rs), cmap=cm.coolwarm)
        ax3d.set_xlabel('z/D')
        ax3d.set_ylabel('r/D')
        ax3d.set_zlabel('$B_z$/T', rotation=45)
        # plt.show()
        plt.savefig('./figures/EI-Kareh_Model_bz_logr.png')
    else:
        # distribution along r-direction
        z = 0
        rs = np.linspace(0, 0.1, 100)
        bz = bz_func(zs=np.array([z]), rs=rs)[0]

        fig = plt.figure(dpi=160)
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(rs/d, bz, color=color_list[0])
        ax.set_xlabel('r/D')
        ax.set_ylabel(f'$B_z$ / T', size=12)

        ax2 = ax.twinx()
        db_br = (bz[1:]-bz[:-1])/(rs[1:]-rs[:-1])
        ax2.plot(rs[:-1], db_br, '-', color=color_list[1])
        ax2.set_ylabel('$\\frac{\partial B_z}{\partial r}$ / $Tmm^{-1}$', size=12)

        ax2.spines['left'].set_color(color_list[0])
        ax2.spines['right'].set_color(color_list[1])
        ax.tick_params(axis='y', colors=color_list[0])
        ax2.tick_params(axis='y', colors=color_list[1])

        plt.title(f'$B_z(z={z}, r)$ of EI-Kareh Model\n(N={n}, I={i}A, D={d}mm, S/D={s/d})')
        plt.tight_layout()
        plt.savefig('./figures/EI-Kareh_Model_bz_along_r.png')
        # plt.show()


"incomplete"
# def paraxial_motion():
#     r0 = np.array([0.1, 0.0, -4])
#     v0 = np.array([0.0, 0.0, 0.1])
#     [d, n, i, s] = [1, 1000, 1, 1]
#
#     def mag_field(x_, y_, z_, t):
#         return np.array([0, 0, bz_0_single_point(z=z_, S=s, D=s, N=n, I=i)])
#
#     ts = np.linspace(0, 800, 1000)
#
#     ts, trajectory = lorentz_force_integration(r0, v0, ts=ts, magnetic_field=mag_field)
#     x, y, z = trajectory[:3]
#
#     # Plot in 3D real space
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     ax.set_title('Demo: Magnetic field $\\vec B=\\frac{z}{L_0}B_0\\vec e_z$')
#     ax.plot(x, y, z, label='electron trajectory')
#     ax.plot(x[0], y[0], z[0], '*', label='starting point')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     mpl.rcParams['legend.fontsize'] = 10
#     ax.legend()
#     plt.tight_layout()
#     plt.show()
#     # plt.savefig('./figures/mag_focusing_demo.png')
#     return None





