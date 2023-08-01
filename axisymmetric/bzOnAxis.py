import numpy as np
from matplotlib import cm
from numpy import ndarray, exp
from math import factorial
import matplotlib.pyplot as plt

from gauss_hermite.hermite_approximation import hermite_fitting, \
    hermite_combination, hermite, hermite_norm


class BzOnAxis:
    """
    Reconstruct Bz field from raw data on axis.

    """
    eps = 1e-3

    def __init__(self, z_raw: ndarray, bz0_raw: ndarray,
                 n_h: int = 40, n_r: int = 10):
        self._z_raw = z_raw
        self._bz0_raw = bz0_raw

        self._n_h = n_h
        self._n_r = n_r

        self.c0, self.bz0 = self.reconstruct_bz0(n_h)
        self.bz = self.reconstruct_bz()

        # checking...
        bz00 = self.bz0(z)
        bz01 = self.bz(z, rs=np.array([0])).reshape(-1)
        assert np.linalg.norm(bz00 - bz01) < 1e-4, 'Error too large.'

    def reconstruct_bz0(self, order: int = 40):
        c0 = hermite_fitting(self._z_raw, self._bz0_raw, order)

        def bz0_func(zs: ndarray):
            return hermite_combination(zs, c0)

        return c0, bz0_func

    def reconstruct_bz(self):
        def bz_func(zs: ndarray, rs: ndarray):
            p = np.poly1d([0])
            for i, a in enumerate(self.c0):
                p = p + a * hermite(i) / hermite_norm(i) ** 0.5

            exponents = exp(-zs ** 2 / 2)
            bz_field = np.zeros(shape=(len(zs), len(rs)))
            for n in range(self._n_r):
                c_n = 1 / ((-4) ** n * factorial(n) ** 2)
                fz_2n = p(zs) * exponents  # 2n-th order derivative of f0
                bz_field = bz_field + c_n * fz_2n[:, None] * rs ** (2 * n)
            return bz_field

        return bz_func

    def update_order(self, order: int):
        self._n_h = order
        self.c0, self.bz0 = self.reconstruct_bz0(order)

    def plot_bz0(self, zs: ndarray = None, save: bool = False):
        if zs is None:
            zs = self._z_raw
        plt.title(r'$B_z$ on Axis')
        plt.plot(self._z_raw, self._bz0_raw, '*', label='Raw Data')
        plt.plot(zs, self.bz0(zs), '-', label='Reconstructed')
        plt.ylabel(r'$B_z$ / mT')
        plt.xlabel(r'z / mm')
        plt.legend()

        if save:
            for fm in ['svg', 'png']:
                plt.savefig('./data/bz.' + fm, dpi=160)
        else:
            plt.show()

    def plot_bz(self, zs: ndarray, rs: ndarray):
        r, z = np.meshgrid(rs, zs)
        plt.figure()
        ax3d = plt.axes(projection='3d')
        surf1 = ax3d.plot_surface(z, r, self.bz(zs, rs), cmap=cm.coolwarm,
                                  label='original')

        surf1._facecolors2d = surf1._facecolor3d
        surf1._edgecolors2d = surf1._edgecolor3d

        ax3d.set_title('Reconstruction of $B_z$ field')
        ax3d.set_xlabel('z / mm')
        ax3d.set_ylabel('r / mm')
        ax3d.set_zlabel('$B_z$', rotation=180)
        ax3d.legend()
        # plt.savefig('./figures/demo_dipole_reconstruction_.png', rotation=180)
        plt.show()


if __name__ == '__main__':
    data = np.loadtxt('data/bz.txt', encoding='utf-8', skiprows=9)

    z = data[:, 0] * 1000  # mm
    bz = data[:, 1] / 1000  # mT

    eps = 1e-3
    none_zero = bz > eps
    z0 = z[bz.argmax()]  # 42.5
    z -= z0

    z = z[none_zero]
    bz = bz[none_zero]

    boa = BzOnAxis(z, bz, n_h=40)
    # boa.plot_bz0()
    zs = np.linspace(-10, 10, 200)
    rs = np.logspace(-4, 0.4, 100)
    boa.plot_bz(zs, rs)

