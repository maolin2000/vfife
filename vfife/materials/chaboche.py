"""Chaboche mixed isotropic/kinematic hardening with coupled ductile damage.

Backward-Euler return mapping with a proper scalar Newton solve.

Relative to the original MATLAB `material_chaboche.m` three defects are fixed:

  P1  Voigt convention.  The flow rule produces a TENSOR strain increment; the
      original stored it directly into `epsilon_p_vec`, which is then subtracted
      from an ENGINEERING total strain and multiplied by an ENGINEERING-shear
      elasticity matrix.  Plastic shear was therefore off by a factor of two.
      Here `deps_p` is converted with TENSOR_TO_ENG before it touches anything.

  P2  Plastic multiplier.  The original denominator was
          E/(1+nu) + C1 + C2 + H'      =  2G + C1 + C2 + H'
      The consistency condition gives
          3G + sum_i (C_i - gamma_i * (X_i:n)) + H'
      Both the 3G and the gamma_i*(X_i:n) term were wrong/missing, and only one
      linearisation step was taken.  Here the residual

          R(dp) = sqrt(3/2)*[ |xi_tr| + sum x_i
                              - sqrt(6) G dp
                              - sum (x_i + sqrt(2/3) C_i dp)/(1 + gamma_i dp) ]
                  - sigma_y(alpha + dp)

      is solved by Newton (`n_newton`, default 3).  Its derivative at dp = 0
      reproduces the textbook denominator exactly.

  P3  Damage applied twice.  The original multiplied the elastic predictor by
      (1 - D_prev) AND the final stress by (1 - D_new), giving a (1-D)^2
      degradation.  Here yielding is checked on the EFFECTIVE (undamaged)
      stress and the Cauchy stress is (1 - D) * sigma_eff, once.

Damage is regularised with a Duvaut-Lions viscosity so that the softening
branch stops being mesh-size dependent:

    D_visc = (eta * D_prev + dt * D_inviscid) / (eta + dt)

eta = 0 recovers the unregularised model.  eta should be a few times the stable
time increment; it is reported in the run log so it can be checked.

State layout (n_state = 20), one row per integration point:

    [0:6]   eps_p   plastic strain, ENGINEERING Voigt
    [6]     alpha   equivalent plastic strain
    [7:13]  X1      backstress 1
    [13:19] X2      backstress 2
    [19]    D       damage
"""

from __future__ import annotations

import numpy as np

from ..base.voigt import (
    TENSOR_TO_ENG,
    dilatational_wave_speed,
    elastic_matrix,
)
from .base import Material, register

_SQ32 = np.sqrt(1.5)          # sqrt(3/2)
_SQ23 = np.sqrt(2.0 / 3.0)    # sqrt(2/3)
_SQ6 = np.sqrt(6.0)


def _ddot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A:B for two STRESS-like Voigt arrays (shear counts twice)."""
    return (
        a[..., 0] * b[..., 0]
        + a[..., 1] * b[..., 1]
        + a[..., 2] * b[..., 2]
        + 2.0 * (a[..., 3] * b[..., 3] + a[..., 4] * b[..., 4] + a[..., 5] * b[..., 5])
    )


@register("chaboche", verification="constitutive_path",
          output_fields=("stress", "strain", "von_mises", "eps_p", "alpha", "X1", "X2", "D", "D_bar"))
class Chaboche(Material):
    n_state = 21
    state_names = {
        "eps_p": slice(0, 6),
        "alpha": slice(6, 7),
        "X1": slice(7, 13),
        "X2": slice(13, 19),
        "D": slice(19, 20),          # viscous (regularised) damage
        "D_bar": slice(20, 21),      # rate-independent damage it relaxes towards
    }

    def __init__(
        self,
        E: float,
        nu: float,
        rho: float,
        yield_strength: float,
        Q_inf: float = 0.0,
        b: float = 0.0,
        H_iso: float = 0.0,
        C1: float = 0.0,
        gamma1: float = 0.0,
        C2: float = 0.0,
        gamma2: float = 0.0,
        damage_initiation_strain: float = np.inf,
        damage_evolution_rate: float = 0.0,
        damage_max: float = 0.99,
        damage_viscosity: float = 0.0,
        n_newton: int = 3,
        **_ignored,
    ):
        self.E, self.nu, self.rho = float(E), float(nu), float(rho)
        self.sy0 = float(yield_strength)
        self.Q_inf, self.b_iso, self.H_iso = float(Q_inf), float(b), float(H_iso)
        self.C1, self.g1 = float(C1), float(gamma1)
        self.C2, self.g2 = float(C2), float(gamma2)
        self.eps_D = float(damage_initiation_strain)
        self.dD = float(damage_evolution_rate)
        self.D_max = float(damage_max)
        if not 0.0 <= self.D_max < 1.0:
            raise ValueError(f"chaboche damage_max = {self.D_max:g} must lie in [0, 1): at 1 or "
                             "more the stress changes sign")
        self.eta = float(damage_viscosity)
        self.n_newton = int(n_newton)

        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.D0 = elastic_matrix(self.E, self.nu)
        self.wave_speed = dilatational_wave_speed(self.E, self.nu, self.rho)

    # ------------------------------------------------------------------ #

    def _sigma_y(self, alpha):
        return (
            self.sy0
            + self.Q_inf * (1.0 - np.exp(-self.b_iso * alpha))
            + self.H_iso * alpha
        )

    def _sigma_y_prime(self, alpha):
        return self.Q_inf * self.b_iso * np.exp(-self.b_iso * alpha) + self.H_iso

    # ------------------------------------------------------------------ #

    def update(self, strain: np.ndarray, state: np.ndarray, dt: float) -> np.ndarray:
        eps_p = state[:, 0:6]
        alpha = state[:, 6]
        X1 = state[:, 7:13]
        X2 = state[:, 13:19]
        D_prev = state[:, 19]

        # --- elastic predictor on the EFFECTIVE (undamaged) stress ---------
        sig_tr = (strain - eps_p) @ self.D0.T
        p_tr = (sig_tr[:, 0] + sig_tr[:, 1] + sig_tr[:, 2]) / 3.0
        s_tr = sig_tr.copy()
        s_tr[:, 0] -= p_tr
        s_tr[:, 1] -= p_tr
        s_tr[:, 2] -= p_tr

        xi = s_tr - X1 - X2
        xi_norm = np.sqrt(np.maximum(_ddot(xi, xi), 0.0))
        safe = np.maximum(xi_norm, 1e-30)
        N = xi / safe[:, None]                      # unit tensor, N:N = 1
        vm_tr = _SQ32 * xi_norm

        sy = self._sigma_y(alpha)
        f_tr = vm_tr - sy
        plastic = f_tr > 1e-9 * np.maximum(sy, 1.0)

        if not plastic.any():
            # Fully elastic step -- skip the return map entirely.  During the
            # elastic phase of a run this is most of the model, and it is a
            # measurable fraction of total solve time.
            # The Duvaut-Lions relaxation still runs: d_v keeps relaxing towards
            # d_bar whether or not the point is currently yielding, and skipping
            # it here would make the answer depend on whether some OTHER point in
            # the batch happened to yield.
            if self.eta > 0.0:
                state[:, 19] = (self.eta * D_prev + dt * state[:, 20]) / (self.eta + dt)
            return sig_tr * (1.0 - state[:, 19])[:, None]

        x1 = _ddot(X1, N)
        x2 = _ddot(X2, N)
        k1 = _SQ23 * self.C1
        k2 = _SQ23 * self.C2
        base = xi_norm + x1 + x2                     # = s_tr : N

        # --- Newton on the scalar consistency condition -------------------
        # Initial guess = the classic linearised step (correct denominator).
        denom0 = (
            3.0 * self.G
            + self.C1
            + self.C2
            - self.g1 * _SQ32 * x1
            - self.g2 * _SQ32 * x2
            + self._sigma_y_prime(alpha)
        )
        denom0 = np.maximum(denom0, 1e-12 * self.E)
        dp = np.where(plastic, f_tr / denom0, 0.0)
        dp = np.maximum(dp, 0.0)

        for _ in range(self.n_newton):
            d1 = 1.0 + self.g1 * dp
            d2 = 1.0 + self.g2 * dp
            R = _SQ32 * (
                base - _SQ6 * self.G * dp - (x1 + k1 * dp) / d1 - (x2 + k2 * dp) / d2
            ) - self._sigma_y(alpha + dp)
            dR = _SQ32 * (
                -_SQ6 * self.G
                - (k1 - self.g1 * x1) / (d1 * d1)
                - (k2 - self.g2 * x2) / (d2 * d2)
            ) - self._sigma_y_prime(alpha + dp)
            dR = np.minimum(dR, -1e-12 * self.E)     # must stay negative
            step = np.where(plastic, R / dR, 0.0)
            dp = np.maximum(dp - step, 0.0)

        # --- state update --------------------------------------------------
        dp_c = dp[:, None]
        deps_p_tensor = _SQ32 * dp_c * N
        eps_p_new = eps_p + deps_p_tensor * TENSOR_TO_ENG      # <- P1 fix
        alpha_new = alpha + dp
        X1_new = (X1 + k1 * dp_c * N) / (1.0 + self.g1 * dp_c)
        X2_new = (X2 + k2 * dp_c * N) / (1.0 + self.g2 * dp_c)

        # --- damage --------------------------------------------------------
        # D_bar is the rate-independent solution; D relaxes towards it with
        # time constant eta.  Keeping them as separate state variables is what
        # makes this a genuine Duvaut-Lions regularisation: folding the
        # viscosity back into D_bar would merely rescale the damage RATE, which
        # does not restore mesh objectivity.
        D_bar_prev = state[:, 20]
        if self.dD > 0.0 and np.isfinite(self.eps_D):
            a0 = np.maximum(alpha, self.eps_D)
            dD_inc = self.dD * np.maximum(alpha_new - a0, 0.0)
            D_bar = np.minimum(D_bar_prev + dD_inc, self.D_max)
            if self.eta > 0.0:
                D_new = (self.eta * D_prev + dt * D_bar) / (self.eta + dt)
            else:
                D_new = D_bar
        else:
            D_bar = D_bar_prev
            D_new = D_prev

        state[:, 0:6] = eps_p_new
        state[:, 6] = alpha_new
        state[:, 7:13] = X1_new
        state[:, 13:19] = X2_new
        state[:, 19] = D_new
        state[:, 20] = D_bar

        # --- Cauchy stress: effective stress degraded ONCE (P3 fix) --------
        sig_eff = (strain - eps_p_new) @ self.D0.T
        return sig_eff * (1.0 - D_new)[:, None]

    def output_fields(self, state: np.ndarray) -> dict:
        return {"alpha": state[:, 6].copy(), "D": state[:, 19].copy()}

    # ------------------------------------------------------------------ #

    def numba_point_fn(self):
        """Scalar twin of update(), for the Numba/GPU element kernel.

        Same algebra, same order of operations, one integration point at a time.
        `test_backends.py` asserts the two agree to 1e-12 element by element.
        """
        from numba import njit

        E, nu = self.E, self.nu
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        G = self.G
        sy0, Q, b_iso, H_iso = self.sy0, self.Q_inf, self.b_iso, self.H_iso
        C1, g1, C2, g2 = self.C1, self.g1, self.C2, self.g2
        eps_D, dD_rate, D_max, eta = self.eps_D, self.dD, self.D_max, self.eta
        n_newton = self.n_newton
        has_damage = dD_rate > 0.0 and np.isfinite(eps_D)
        SQ32, SQ23, SQ6 = _SQ32, _SQ23, _SQ6

        @njit(cache=True, fastmath=True, inline="always")
        def point(strain, state, dt, k, stress):
            alpha = state[6]
            D_prev = state[19]

            # elastic predictor on the effective (undamaged) stress
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

            p = (t0 + t1 + t2) / 3.0
            xi0 = t0 - p - state[7] - state[13]
            xi1 = t1 - p - state[8] - state[14]
            xi2 = t2 - p - state[9] - state[15]
            xi3 = t3 - state[10] - state[16]
            xi4 = t4 - state[11] - state[17]
            xi5 = t5 - state[12] - state[18]

            nn = (xi0 * xi0 + xi1 * xi1 + xi2 * xi2
                  + 2.0 * (xi3 * xi3 + xi4 * xi4 + xi5 * xi5))
            xi_norm = np.sqrt(nn) if nn > 0.0 else 0.0
            sy = sy0 + Q * (1.0 - np.exp(-b_iso * alpha)) + H_iso * alpha
            f_tr = SQ32 * xi_norm - sy

            if f_tr <= 1e-9 * (sy if sy > 1.0 else 1.0):
                if eta > 0.0:
                    D_prev = (eta * D_prev + dt * state[20]) / (eta + dt)
                    state[19] = D_prev
                one_d = 1.0 - D_prev
                stress[0] = t0 * one_d
                stress[1] = t1 * one_d
                stress[2] = t2 * one_d
                stress[3] = t3 * one_d
                stress[4] = t4 * one_d
                stress[5] = t5 * one_d
                return

            inv = 1.0 / (xi_norm if xi_norm > 1e-30 else 1e-30)
            N0 = xi0 * inv
            N1 = xi1 * inv
            N2 = xi2 * inv
            N3 = xi3 * inv
            N4 = xi4 * inv
            N5 = xi5 * inv

            x1 = (state[7] * N0 + state[8] * N1 + state[9] * N2
                  + 2.0 * (state[10] * N3 + state[11] * N4 + state[12] * N5))
            x2 = (state[13] * N0 + state[14] * N1 + state[15] * N2
                  + 2.0 * (state[16] * N3 + state[17] * N4 + state[18] * N5))
            k1 = SQ23 * C1
            k2 = SQ23 * C2
            base = xi_norm + x1 + x2

            syp = Q * b_iso * np.exp(-b_iso * alpha) + H_iso
            den0 = 3.0 * G + C1 + C2 - g1 * SQ32 * x1 - g2 * SQ32 * x2 + syp
            if den0 < 1e-12 * E:
                den0 = 1e-12 * E
            dp = f_tr / den0
            if dp < 0.0:
                dp = 0.0

            for _ in range(n_newton):
                d1 = 1.0 + g1 * dp
                d2 = 1.0 + g2 * dp
                a_new = alpha + dp
                R = SQ32 * (base - SQ6 * G * dp
                            - (x1 + k1 * dp) / d1
                            - (x2 + k2 * dp) / d2) - (
                    sy0 + Q * (1.0 - np.exp(-b_iso * a_new)) + H_iso * a_new)
                dR = SQ32 * (-SQ6 * G
                             - (k1 - g1 * x1) / (d1 * d1)
                             - (k2 - g2 * x2) / (d2 * d2)) - (
                    Q * b_iso * np.exp(-b_iso * a_new) + H_iso)
                if dR > -1e-12 * E:
                    dR = -1e-12 * E
                dp -= R / dR
                if dp < 0.0:
                    dp = 0.0

            # state update; the flow rule gives a TENSOR increment, so the shear
            # components are doubled on the way into engineering storage
            c = SQ32 * dp
            state[0] += c * N0
            state[1] += c * N1
            state[2] += c * N2
            state[3] += 2.0 * c * N3
            state[4] += 2.0 * c * N4
            state[5] += 2.0 * c * N5

            alpha_new = alpha + dp
            state[6] = alpha_new
            i1 = 1.0 / (1.0 + g1 * dp)
            i2 = 1.0 / (1.0 + g2 * dp)
            kd1 = k1 * dp
            kd2 = k2 * dp
            state[7] = (state[7] + kd1 * N0) * i1
            state[8] = (state[8] + kd1 * N1) * i1
            state[9] = (state[9] + kd1 * N2) * i1
            state[10] = (state[10] + kd1 * N3) * i1
            state[11] = (state[11] + kd1 * N4) * i1
            state[12] = (state[12] + kd1 * N5) * i1
            state[13] = (state[13] + kd2 * N0) * i2
            state[14] = (state[14] + kd2 * N1) * i2
            state[15] = (state[15] + kd2 * N2) * i2
            state[16] = (state[16] + kd2 * N3) * i2
            state[17] = (state[17] + kd2 * N4) * i2
            state[18] = (state[18] + kd2 * N5) * i2

            if has_damage:
                a0 = alpha if alpha > eps_D else eps_D
                inc = alpha_new - a0
                if inc < 0.0:
                    inc = 0.0
                D_bar = state[20] + dD_rate * inc
                if D_bar > D_max:
                    D_bar = D_max
                if eta > 0.0:
                    D_new = (eta * D_prev + dt * D_bar) / (eta + dt)
                else:
                    D_new = D_bar
                state[20] = D_bar
            else:
                D_new = D_prev
            state[19] = D_new

            # Cauchy stress from the UPDATED plastic strain, degraded once
            e0 = strain[0] - state[0]
            e1 = strain[1] - state[1]
            e2 = strain[2] - state[2]
            e3 = strain[3] - state[3]
            e4 = strain[4] - state[4]
            e5 = strain[5] - state[5]
            tr = e0 + e1 + e2
            one_d = 1.0 - D_new
            stress[0] = (lam * tr + 2.0 * G * e0) * one_d
            stress[1] = (lam * tr + 2.0 * G * e1) * one_d
            stress[2] = (lam * tr + 2.0 * G * e2) * one_d
            stress[3] = G * e3 * one_d
            stress[4] = G * e4 * one_d
            stress[5] = G * e5 * one_d

        return point

    # ------------------------------------------------------------------ #

    def warp_params(self, dtype=np.float64):
        """(scalars, packed tables) for the GPU backend. Layout in warp_materials.py."""
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        has_d = 1.0 if (self.dD > 0.0 and np.isfinite(self.eps_D)) else 0.0
        eps_D = self.eps_D if np.isfinite(self.eps_D) else 0.0
        return (
            np.array([lam, self.G, self.E, self.sy0, self.Q_inf, self.b_iso,
                      self.H_iso, self.C1, self.g1, self.C2, self.g2,
                      eps_D, self.dD, self.D_max, self.eta,
                      float(self.n_newton), has_d,
                      _SQ32, _SQ23, _SQ6], dtype=dtype),
            np.zeros(1, dtype=dtype),
        )

    @staticmethod
    def warp_point_fn():
        """Device twin, in kernels/warp_materials.py."""
        from ..kernels.warp_materials import chaboche_point

        return chaboche_point
