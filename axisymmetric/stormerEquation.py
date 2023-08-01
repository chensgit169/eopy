import numpy as np
from numpy import ndarray, exp, sign
from scipy.integrate import solve_ivp


class StormerEquation:
    def __init__(self):
        self.constants = {'energy': 0, 'momentum': 0}
        self.init_values = {'r0': 0, 'rp0': 0, 'z0': 0}

    def force_in_2d(self, z: float, y: ndarray):
        r, rp = y
        exponent = exp(-z**2 / 2)

        p = p0
        bz = 0  # B_z(z, r=0) = p(x) * exp(-x**2/2), unit: T
        s = 0  # 1/r * \int_0^r r*B_z(r) dr, unit: T * m
        ps_pz = 0  # \partial s/ \partial z, unit: T

        for n in range(max_n):
            c_n = 1 / ((-4)**n * factorial(n)**2)
            fz_2n = p(z) * exponent  # 2n-th order derivative of f0
            p = p.deriv(1) - p * np.poly1d([1, 0])
            fz_2n_p1 = p(z) * exponent  # (2n+1)-th order derivative of f0
            p = p.deriv(1) - p * np.poly1d([1, 0])

            bz += c_n * fz_2n * r ** (2*n)
            s += c_n * fz_2n * abs(r)**(2*n+1) / (2*n+2)
            ps_pz += c_n * fz_2n_p1 * abs(r)**(2*n+1) / (2*n+2)

        mu2 = g2 - s**2
        # print(f"z={z}, r={r}, r'={rp}:")
        # print(f'bz={bz}, s={s}, mu2={mu2}')
        assert mu2 >= 0, f"mu**2 must >=0 : mu2={mu2}, possible problem is that r has gone too large:" \
                         f"\nz={z}, r={r}, r'={rp}, \nbz={bz}, s={s}, mu2={mu2}"
        if angular_m == 0:
            rpp = sign(r)*(1 + rp**2) * s * (sign(r)*rp*ps_pz - bz) / mu2
        else:
            # numerical error needs to be handled carefully when N~0 !
            rpp = (1 + rp**2) * (s + angular_m/(r+1e-500)) * (rp*ps_pz - bz+angular_m/(r**2+1e-500)) / mu2
        force = np.array([rp, rpp])
        # print(f'mu2= {mu2}, force = {force}')
        return force

    def solve(self, zs: ndarray):
        solution = solve_ivp(self.force_in_2d, t_span=[zs[0], zs[-1]], y0=np.array([r0, rp0]), t_eval=zs)
        zs = solution.t
        trajectory = solution.y
        return zs, trajectory

    def plot(self):
        raise NotImplemented
