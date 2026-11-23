"""Uniaxial steel for reinforcement: bilinear with combined hardening.

This is a ONE-DIMENSIONAL law.  It reads `strain[0]` and writes `stress[0]`,
leaving the other five components at zero, because a reinforcing bar carries
axial force and nothing else.  `uniaxial = True` marks it so, and the continuum
element rejects it -- putting a bar law inside a solid element would silently
produce a material with no shear stiffness at all, which is exactly the kind of
mistake that looks like a converged answer.

Combined hardening, so the Bauschinger effect is represented:

    yield when  |sigma - X| > sigma_y + H_iso alpha
    X += H_kin sign(xi) dp,   alpha += dp

Kinematic hardening translates the (one-dimensional) yield surface, which is
what makes a bar that yielded in tension yield early in compression.  For the
smooth, curved reversal of a Menegotto-Pinto law, use `steel02`; this is the
piecewise-linear version, which is what most RC work uses and what pairs with a
truss element. (`rebar_gmp` used to live in this file and was the same law as
`steel02` -- removed in v0.1; materials/base.RETIRED says so to anyone whose
case still names it.)

State layout (n_state = 3):

    [0]  eps_p   axial plastic strain
    [1]  alpha   accumulated plastic strain
    [2]  X       backstress
"""

from __future__ import annotations

import numpy as np

from .base import Material, register


@register("rebar", kinematics=("uniaxial",), verification="constitutive_path",
          output_fields=("axial_stress", "axial_strain", "eps_p", "alpha", "X"))
class Rebar(Material):
    n_state = 3
    state_names = {"eps_p": slice(0, 1), "alpha": slice(1, 2), "X": slice(2, 3)}
    #: read by the continuum element, which refuses to use a 1-D law
    uniaxial = True

    def __init__(self, E, rho, yield_strength, hardening=0.0,
                 kinematic_fraction=0.5, nu=0.3, rupture_strain=None,
                 **_ignored):
        self.E, self.rho, self.nu = float(E), float(rho), float(nu)
        self.sy = float(yield_strength)
        if not (self.E > 0 and self.sy > 0):
            # a negative yield strength gave -400 MPa at a strain of 1e-6
            raise ValueError("rebar E and yield_strength must be positive (a magnitude; "
                             "the law is symmetric)")
        H = float(hardening)
        if H < 0:
            raise ValueError("rebar hardening modulus must be >= 0")
        b = float(kinematic_fraction)
        if not 0.0 <= b <= 1.0:
            raise ValueError("kinematic_fraction must lie in [0, 1]")
        # split the post-yield modulus between the two mechanisms: purely
        # isotropic hardening gives no Bauschinger effect at all, purely
        # kinematic gives no growth of the elastic range
        self.H_kin = b * H
        self.H_iso = (1.0 - b) * H
        self.H = H
        # RUPTURE, AND WHY IT IS ON ACCUMULATED PLASTIC STRAIN
        #   Without it this law hardens for ever: measured on the solid RC
        #   column, the base bars reached 4.8% plastic strain at the point the
        #   model failed and nothing in the material would have stopped them at
        #   19% either. A real 20 mm bar with fu = 721 MPa does not do that.
        #
        #   The measure is `alpha`, the accumulated plastic strain, because it
        #   is already in the state and is monotone, so rupture is irreversible
        #   by construction and no fourth state column is needed -- which would
        #   have meant changing the layout in three implementations at once.
        #
        #   It is NOT the monotonic ultimate strain eps_u. Accumulated plastic
        #   strain grows on every reversal, so a cycled bar reaches the limit
        #   sooner than a monotonically pulled one. That is closer to low-cycle
        #   fatigue than to a tension test, and it is the conservative
        #   direction, but do not read the number off a mill certificate and
        #   expect a tension test to match.
        self.eps_rupture = (None if rupture_strain is None
                            else float(rupture_strain))
        if self.eps_rupture is not None and self.eps_rupture <= 0.0:
            raise ValueError("rebar rupture_strain must be positive")
        self.wave_speed = float(np.sqrt(self.E / self.rho))
        # a bar has no D0 in the continuum sense; the solver's Underwood mass
        # path asks for one, so give it the axial modulus on the diagonal
        self.D0 = np.diag([self.E, 0.0, 0.0, 0.0, 0.0, 0.0])

    def describe(self) -> str:
        return (f"    E = {self.E / 1e9:.1f} GPa, sigma_y = {self.sy / 1e6:.0f} MPa, "
                f"H = {self.H / 1e9:.4g} GPa "
                f"(kinematic {self.H_kin / max(self.H, 1e-30) * 100:.0f}%)"
                + ("" if self.eps_rupture is None
                   else f", ruptures at eps_p = {self.eps_rupture:.4g}"))

    # ------------------------------------------------------------------ #

    def update(self, strain, state, dt):
        eps_p = state[:, 0]
        alpha = state[:, 1]
        X = state[:, 2]

        sig_tr = self.E * (strain[:, 0] - eps_p)
        xi = sig_tr - X
        f = np.abs(xi) - (self.sy + self.H_iso * alpha)
        plastic = f > 1e-9 * self.sy

        out = np.zeros_like(strain)
        if not plastic.any():
            out[:, 0] = sig_tr
            if self.eps_rupture is not None:
                out[state[:, 1] >= self.eps_rupture, 0] = 0.0
            return out

        dp = np.where(plastic, f / (self.E + self.H_iso + self.H_kin), 0.0)
        np.maximum(dp, 0.0, out=dp)
        if self.eps_rupture is not None:
            # a ruptured bar's state stops, as in the compiled twin: its
            # reported plastic strain went on growing here
            dp[alpha >= self.eps_rupture] = 0.0
        sgn = np.sign(xi)
        state[:, 0] = eps_p + sgn * dp
        state[:, 1] = alpha + dp
        state[:, 2] = X + self.H_kin * sgn * dp
        out[:, 0] = self.E * (strain[:, 0] - state[:, 0])
        if self.eps_rupture is not None:
            out[state[:, 1] >= self.eps_rupture, 0] = 0.0
        return out

    def damage_class(self, state):
        out = np.zeros(len(state), dtype=np.int8)
        out[state[:, 1] > 0.0] = self.DAMAGE_YIELDED
        f = self.failure(state)
        if f is not None:
            out[f > 0.5] = self.DAMAGE_FAILED
        return out

    def failure(self, state):
        """1 where the bar has ruptured.

        The latch is not new -- `update` has zeroed the stress of a ruptured
        point since the bar law was written. What is new is SAYING SO, so that
        the element above can count how many of its fibres are gone instead of
        inferring it from a stress that happens to be zero (which is also what
        an unloaded elastic fibre reports).

        Monotone because `alpha` is: it accumulates |dp| and dp >= 0.
        """
        if self.eps_rupture is None:
            return None
        return (state[:, 1] >= self.eps_rupture).astype(float)

    def output_fields(self, state):
        return {"alpha": state[:, 1].copy(), "eps_p": state[:, 0].copy()}

    # ------------------------------------------------------------------ #

    def numba_point_fn(self):
        from numba import njit

        E, sy, H_iso, H_kin = self.E, self.sy, self.H_iso, self.H_kin
        RUP = 0.0 if self.eps_rupture is None else self.eps_rupture

        @njit(cache=True, fastmath=True, inline="always")
        def point(strain, state, dt, k, stress):
            for i in range(1, 6):
                stress[i] = 0.0
            if RUP > 0.0 and state[1] >= RUP:
                stress[0] = 0.0
                return
            sig_tr = E * (strain[0] - state[0])
            xi = sig_tr - state[2]
            f = abs(xi) - (sy + H_iso * state[1])
            if f <= 1e-9 * sy:
                stress[0] = sig_tr
                return
            dp = f / (E + H_iso + H_kin)
            sgn = 1.0 if xi >= 0.0 else -1.0
            state[0] += sgn * dp
            state[1] += dp
            state[2] += H_kin * sgn * dp
            if RUP > 0.0 and state[1] >= RUP:
                stress[0] = 0.0
                return
            stress[0] = E * (strain[0] - state[0])

        return point

    P = dict(E=0, sy=1, H_iso=2, H_kin=3, tol=4, rupture=5)

    def warp_params(self, dtype=np.float64):
        return (np.array([self.E, self.sy, self.H_iso, self.H_kin,
                          1.0e-9 * self.sy,
                          0.0 if self.eps_rupture is None else self.eps_rupture],
                         dtype=dtype),
                np.zeros(1, dtype=dtype))

    @staticmethod
    def warp_point_fn():
        from ..kernels.warp_materials import rebar_point

        return rebar_point
