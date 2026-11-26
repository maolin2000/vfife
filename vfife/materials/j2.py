"""J2 (von Mises) plasticity: isotropic + linear kinematic hardening.

The workhorse for steel, and the one Chaboche is overkill for.  Chaboche with
C1 = C2 = 0 gives the same physics, but it carries 21 state variables and three
Newton iterations over `exp()` to do it; this carries 13 and, for the common
case of a piecewise-linear hardening table, converges in one or two.

Backward-Euler radial return.  Isotropic hardening is a `Curve` -- the same type
the concrete models use -- so `linear`, `power` and tabulated data all arrive
through one interpolator, and the GPU gets one packed buffer whichever it is.
Kinematic hardening is linear (Prager); if you need the Bauschinger rounding of
several backstresses, that is what `chaboche` is for.

The consistency condition, with the shifted stress xi = s - X:

    q_tr - 3G dp - H_kin dp - [sigma_y(a + dp) - sigma_y(a)] = 0

which is exact in one step for a linear segment of the hardening curve, and
Newton otherwise.  The 3G is the piece the original MATLAB Chaboche had as 2G
(see chaboche.py, P2).

State layout (n_state = 13):

    [0:6]   eps_p   plastic strain, ENGINEERING Voigt
    [6]     alpha   equivalent plastic strain
    [7:13]  X       backstress
"""

from __future__ import annotations

import numpy as np

from ..base.tensor import ddot
from ..base.voigt import TENSOR_TO_ENG, dilatational_wave_speed, elastic_matrix
from . import curves as cv
from .base import Material, register

_SQ32 = np.sqrt(1.5)
_SQ23 = np.sqrt(2.0 / 3.0)


@register("j2", verification="constitutive_path",
          output_fields=("stress", "strain", "von_mises", "eps_p", "alpha", "X"))
class J2Plasticity(Material):
    n_state = 13
    state_names = {
        "eps_p": slice(0, 6),
        "alpha": slice(6, 7),
        "X": slice(7, 13),
    }

    def __init__(self, E, nu, rho, hardening=None, yield_strength=None,
                 kinematic=0.0, n_newton=3, **_ignored):
        self.E, self.nu, self.rho = float(E), float(nu), float(rho)
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.D0 = elastic_matrix(self.E, self.nu)
        self.wave_speed = dilatational_wave_speed(self.E, self.nu, self.rho)
        self.H_kin = float(kinematic)
        self.n_newton = int(n_newton)

        if hardening is None:
            if yield_strength is None:
                raise ValueError(
                    "j2 needs either `hardening` or a plain `yield_strength`"
                )
            hardening = {"source": "linear", "yield_strength": yield_strength}
        self.curve = cv.build_hardening(hardening, name="J2 isotropic hardening")
        self.sy0 = self.curve.sigma0

    def describe(self) -> str:
        return (f"    sigma_y0 = {self.sy0 / 1e6:.1f} MPa, "
                f"isotropic = {self.curve.name}, "
                f"kinematic H = {self.H_kin / 1e9:.3g} GPa")

    # ------------------------------------------------------------------ #

    def update(self, strain, state, dt):
        eps_p = state[:, 0:6]
        alpha = state[:, 6]
        X = state[:, 7:13]

        sig = (strain - eps_p) @ self.D0.T
        p = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
        s = sig.copy()
        s[:, 0] -= p
        s[:, 1] -= p
        s[:, 2] -= p

        xi = s - X
        nrm = np.sqrt(np.maximum(ddot(xi, xi), 0.0))
        q_tr = _SQ32 * nrm
        sy, H_iso, _ = self.curve.evaluate(alpha)
        f_tr = q_tr - sy
        plastic = f_tr > 1e-9 * np.maximum(sy, 1.0)
        if not plastic.any():
            return sig

        N = xi / np.maximum(nrm, 1e-30)[:, None]        # unit tensor, N:N = 1
        denom = np.maximum(3.0 * self.G + self.H_kin + H_iso, 1e-12 * self.E)
        dp = np.where(plastic, f_tr / denom, 0.0)
        np.maximum(dp, 0.0, out=dp)

        for _ in range(self.n_newton):
            sy_n, H_n, _ = self.curve.evaluate(alpha + dp)
            R = q_tr - (3.0 * self.G + self.H_kin) * dp - sy_n
            dR = np.minimum(-(3.0 * self.G + self.H_kin) - H_n, -1e-12 * self.E)
            dp = np.maximum(dp - np.where(plastic, R / dR, 0.0), 0.0)

        # the flow rule gives a TENSOR increment; shear doubles into engineering
        deps_t = (_SQ32 * dp)[:, None] * N
        state[:, 0:6] = eps_p + deps_t * TENSOR_TO_ENG
        state[:, 6] = alpha + dp
        state[:, 7:13] = X + (2.0 / 3.0) * self.H_kin * deps_t

        return (strain - state[:, 0:6]) @ self.D0.T

    def output_fields(self, state):
        return {"alpha": state[:, 6].copy()}

    # ------------------------------------------------------------------ #

    P = dict(lam=0, G=1, E=2, H_kin=3, n_newton=4, n_tab=5, sq32=6, tol=7,
             eps_den=8)

    def _prm(self, dtype=np.float64):
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        return np.array([
            lam, self.G, self.E, self.H_kin, float(self.n_newton),
            float(self.curve.ep.size), _SQ32, 1.0e-9,
            max(1.0e-12 * self.E, 1.0),
        ], dtype=dtype)

    def numba_point_fn(self):
        """Scalar twin of update(), inlined into the element kernel."""
        from numba import njit

        from ..kernels.scalar import curve_eval

        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        G = self.G
        E = self.E
        H_kin = self.H_kin
        n_newton = self.n_newton
        c_ep, c_sig, c_dmg, c_slope = (self.curve.ep, self.curve.sig,
                                       self.curve.dmg, self.curve.slope)
        SQ32 = _SQ32
        two_third = 2.0 / 3.0

        @njit(cache=True, fastmath=True, inline="always")
        def point(strain, state, dt, k, stress):
            res = np.empty(3)
            e0 = strain[0] - state[0]
            e1 = strain[1] - state[1]
            e2 = strain[2] - state[2]
            e3 = strain[3] - state[3]
            e4 = strain[4] - state[4]
            e5 = strain[5] - state[5]
            tr = e0 + e1 + e2
            t0 = lam * tr + 2.0 * G * e0
            t1 = lam * tr + 2.0 * G * e1
            t2 = lam * tr + 2.0 * G * e2
            t3 = G * e3
            t4 = G * e4
            t5 = G * e5

            pm = (t0 + t1 + t2) / 3.0
            x0 = t0 - pm - state[7]
            x1 = t1 - pm - state[8]
            x2 = t2 - pm - state[9]
            x3 = t3 - state[10]
            x4 = t4 - state[11]
            x5 = t5 - state[12]
            nn = (x0 * x0 + x1 * x1 + x2 * x2
                  + 2.0 * (x3 * x3 + x4 * x4 + x5 * x5))
            nrm = np.sqrt(max(nn, 0.0))
            q_tr = SQ32 * nrm

            alpha = state[6]
            curve_eval(c_ep, c_sig, c_dmg, c_slope, alpha, res)
            sy = res[0]
            if q_tr - sy <= 1e-9 * max(sy, 1.0):
                stress[0] = t0
                stress[1] = t1
                stress[2] = t2
                stress[3] = t3
                stress[4] = t4
                stress[5] = t5
                return

            inv = 1.0 / max(nrm, 1e-30)
            n0 = x0 * inv
            n1 = x1 * inv
            n2 = x2 * inv
            n3 = x3 * inv
            n4 = x4 * inv
            n5 = x5 * inv

            base = 3.0 * G + H_kin
            dp = max((q_tr - sy) / max(base + res[1], 1e-12 * E), 0.0)
            for _ in range(n_newton):
                curve_eval(c_ep, c_sig, c_dmg, c_slope, alpha + dp, res)
                R = q_tr - base * dp - res[0]
                dR = min(-base - res[1], -1e-12 * E)
                dp = max(dp - R / dR, 0.0)

            c = SQ32 * dp
            state[0] += c * n0
            state[1] += c * n1
            state[2] += c * n2
            state[3] += 2.0 * c * n3
            state[4] += 2.0 * c * n4
            state[5] += 2.0 * c * n5
            state[6] = alpha + dp
            hk = two_third * H_kin * c
            state[7] += hk * n0
            state[8] += hk * n1
            state[9] += hk * n2
            state[10] += hk * n3
            state[11] += hk * n4
            state[12] += hk * n5

            e0 = strain[0] - state[0]
            e1 = strain[1] - state[1]
            e2 = strain[2] - state[2]
            tr = e0 + e1 + e2
            stress[0] = lam * tr + 2.0 * G * e0
            stress[1] = lam * tr + 2.0 * G * e1
            stress[2] = lam * tr + 2.0 * G * e2
            stress[3] = G * (strain[3] - state[3])
            stress[4] = G * (strain[4] - state[4])
            stress[5] = G * (strain[5] - state[5])

        return point

    def warp_params(self, dtype=np.float64):
        return self._prm(dtype), self.curve.pack(dtype)

    @staticmethod
    def warp_point_fn():
        from ..kernels.warp_materials import j2_point

        return j2_point
