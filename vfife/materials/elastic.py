"""Linear elastic material -- the reference case for patch tests and timing."""

from __future__ import annotations

import numpy as np

from ..base.voigt import dilatational_wave_speed, elastic_matrix
from .base import Material, register


@register("linear_elastic", kinematics=("solid_3d", "uniaxial"),
          nonlinear=False, verification="analytical",
          output_fields=("stress", "strain", "von_mises"))
class LinearElastic(Material):
    n_state = 0
    state_names: dict = {}

    def __init__(self, E: float, nu: float, rho: float, **_ignored):
        self.E = float(E)
        self.nu = float(nu)
        self.rho = float(rho)
        if not self.rho > 0.0:
            # it divided by zero below
            raise ValueError("linear_elastic rho must be positive: every solver takes the members' "
                             "own mass and wave speed from it (a member whose inertia is lumped on "
                             "its nodes: give it a small density and the masses in `nodal_masses`)")
        self.D0 = elastic_matrix(self.E, self.nu)
        self.wave_speed = dilatational_wave_speed(self.E, self.nu, self.rho)

    def update(self, strain: np.ndarray, state: np.ndarray, dt: float) -> np.ndarray:
        return strain @ self.D0.T

    # ------------------------------------------------------------------ #

    def numba_point_fn(self):
        """Scalar twin of update(), for the Numba/GPU element kernel.

        The two implementations live side by side on purpose: they are the same
        model written for two different machines, and a test asserts they agree
        to 1e-12 on every formulation.
        """
        from numba import njit

        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        G = self.E / (2.0 * (1.0 + self.nu))

        @njit(cache=True, fastmath=True, inline="always")
        def point(strain, state, dt, k, stress):
            tr = strain[0] + strain[1] + strain[2]
            stress[0] = lam * tr + 2.0 * G * strain[0]
            stress[1] = lam * tr + 2.0 * G * strain[1]
            stress[2] = lam * tr + 2.0 * G * strain[2]
            stress[3] = G * strain[3]
            stress[4] = G * strain[4]
            stress[5] = G * strain[5]

        return point

    # ------------------------------------------------------------------ #

    P_LAM, P_G = 0, 1

    def warp_params(self, dtype=np.float64):
        """(scalars, packed tables) -- the two buffers every material hands the GPU."""
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        prm = np.array([lam, self.E / (2.0 * (1.0 + self.nu))], dtype=dtype)
        return prm, np.zeros(1, dtype=dtype)

    @staticmethod
    def warp_point_fn():
        """Device twin, in kernels/warp_materials.py -- Warp evaluates argument
        annotations against the defining module's globals, which rules out a
        file that uses `from __future__ import annotations`."""
        from ..kernels.warp_materials import elastic_point

        return elastic_point
