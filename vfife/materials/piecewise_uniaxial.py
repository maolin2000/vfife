"""General asymmetric piecewise-linear uniaxial plasticity.

The tension and compression tables contain ``[accumulated plastic strain,
yield-stress magnitude]`` pairs.  ``E`` is both the initial and unloading
modulus.  Isotropic hardening expands the two bounds; kinematic hardening
translates them and therefore produces a Bauschinger effect.

This is deliberately a one-dimensional material point.  It is suitable for
trusses, embedded bars and future fibre sections, not continuum elements.
"""

from __future__ import annotations

import numpy as np

from .base import Material, register


def _curve(values, label):
    table = np.asarray(values, dtype=float)
    if table.ndim != 2 or table.shape[1] != 2 or len(table) < 2:
        raise ValueError(f"{label} needs at least two [plastic_strain, stress] points")
    if not np.isfinite(table).all():
        raise ValueError(f"{label} must contain finite values")
    if abs(table[0, 0]) > 1e-14 or np.any(np.diff(table[:, 0]) <= 0):
        raise ValueError(f"{label} plastic strains must start at zero and strictly increase")
    if np.any(table[:, 1] <= 0) or np.any(np.diff(table[:, 1]) < 0):
        raise ValueError(f"{label} stress magnitudes must be positive and non-decreasing")
    return np.ascontiguousarray(table)


@register(
    "piecewise_uniaxial",
    kinematics=("uniaxial",),
    backends=("numpy",),
    verification="constitutive_path",
    output_fields=(
        "axial_stress", "axial_strain", "eps_p", "alpha", "backstress",
        "external_work", "elastic_energy", "hysteretic_energy",
    ),
    caveat="Piecewise-linear rate-independent law; tables use positive stress magnitudes.",
)
class PiecewiseUniaxial(Material):
    """Asymmetric tabulated 1-D plasticity with selectable hardening split."""

    n_state = 6
    state_names = {
        "eps_p": slice(0, 1),
        "alpha": slice(1, 2),
        "backstress": slice(2, 3),
        "axial_strain": slice(3, 4),
        "axial_stress": slice(4, 5),
        "external_work": slice(5, 6),
    }
    uniaxial = True

    def __init__(self, E, rho, tension, compression=None,
                 hardening_mode="isotropic", kinematic_fraction=0.5,
                 nu=0.3):
        self.E, self.rho, self.nu = float(E), float(rho), float(nu)
        if not np.isfinite(self.E) or self.E <= 0:
            raise ValueError("piecewise_uniaxial E must be finite and positive")
        if not np.isfinite(self.rho) or self.rho <= 0:
            raise ValueError("piecewise_uniaxial rho must be finite and positive")
        self.tension = _curve(tension, "piecewise_uniaxial tension")
        self.compression = _curve(compression if compression is not None else tension,
                                  "piecewise_uniaxial compression")
        # The point loop reads the tables as LISTS OF PAIRS: indexing a numpy
        # array element by element costs more than the arithmetic done with
        # it, and these are three or four rows that never change.
        self._tension = [tuple(map(float, row)) for row in self.tension]
        self._compression = [tuple(map(float, row)) for row in self.compression]
        modes = {"isotropic": 0.0, "kinematic": 1.0, "combined": None}
        if hardening_mode not in modes:
            raise ValueError("hardening_mode must be isotropic, kinematic or combined")
        fraction = float(kinematic_fraction)
        if not 0 <= fraction <= 1:
            raise ValueError("kinematic_fraction must lie in [0, 1]")
        self.hardening_mode = hardening_mode
        self.kinematic_fraction = (fraction if modes[hardening_mode] is None
                                   else modes[hardening_mode])
        self.wave_speed = float(np.sqrt(self.E / self.rho))
        self.D0 = np.diag([self.E, 0.0, 0.0, 0.0, 0.0, 0.0])

    @staticmethod
    def _value(table, alpha):
        """Linear interpolation with terminal-slope extrapolation.

        PLAIN PYTHON OVER A LIST, not `np.interp`. A table has three or four
        rows; np.interp on one of them is about 3 us of numpy call overhead
        against ~0.1 us of arithmetic, and this is called four or five times
        per material point per step. Measured on 480 fibre points and the same
        cyclic path: 22.9 us a point with numpy here, 4.6 us without.
        """
        n = len(table)
        if alpha <= table[0][0]:
            return table[0][1]
        for i in range(1, n):
            x1, k1 = table[i]
            if alpha <= x1:
                x0, k0 = table[i - 1]
                return k0 + (k1 - k0) * (alpha - x0) / (x1 - x0)
        x0, k0 = table[-2]
        x1, k1 = table[-1]
        return k1 + (k1 - k0) / (x1 - x0) * (alpha - x1)

    def _bound(self, table, alpha):
        sy0 = table[0][1]
        return sy0 + (1.0 - self.kinematic_fraction) * (self._value(table, alpha) - sy0)

    def _plastic_increment(self, trial, eps_p, alpha, backstress, direction, table):
        """The consistency equation, solved exactly, one table segment at a time.

        IT IS A LINEAR EQUATION INSIDE EACH SEGMENT, so there is nothing to
        iterate. Writing the residual out (b is the kinematic fraction, k the
        curve, k0 = k(alpha), sy0 = k(0)):

            r(dp) = d(trial - X) + b(k0 - sy0) - E dp - k(alpha + dp)

        -- the b and (1-b) parts of the hardening collapse, which is why the
        split does not appear anywhere below except in the constant. `k` is
        piecewise linear and non-decreasing and E > 0, so E dp + k(alpha + dp)
        is strictly increasing and the root is unique: walk the breakpoints
        from `alpha` upward, solve dp = (C - k_a + H(x_a - alpha))/(E + H) in
        each segment of slope H, and stop at the first root that lands inside
        its own segment. Past the last breakpoint the terminal slope continues.

        This replaces 80 bisection steps, each evaluating the residual twice
        through `np.interp`: ~200 Python-level calls for a number that costs
        one divide. Measured on 480 fibre points and the same cyclic path:
        465 us a point before, 4.6 us after -- 100x, of which the exact solve
        is 20x and reading the table off a list is the rest.
        It is also EXACT rather than converged to 1e-13, and it agrees with a
        200-step bisection to 1.5e-12 over 4 000 random states.

        Still a Python loop, and still 30x the cost of `steel02`'s compiled
        point (147 ns): a table-driven law cannot be inlined into the fibre
        kernel, which is the reason to reach for it only when the table is the
        point of the model.
        """
        b = self.kinematic_fraction
        old_curve = self._value(table, alpha)
        C = direction * (trial - backstress) + b * (old_curve - table[0][1])

        n = len(table)
        j = 0
        while j < n - 2 and table[j + 1][0] <= alpha:
            j += 1
        while True:
            x0, k0 = table[j]
            x1, k1 = table[j + 1]
            H = (k1 - k0) / (x1 - x0)
            dp = (C - k0 + H * (x0 - alpha)) / (self.E + H)
            if dp <= 0.0:
                return 0.0, old_curve
            if alpha + dp <= x1 or j == n - 2:
                return dp, old_curve
            j += 1

    def update(self, strain, state, dt):
        out = np.zeros_like(strain)
        E, b = self.E, self.kinematic_fraction
        tension, compression = self._tension, self._compression
        for i in range(len(strain)):
            eps = float(strain[i, 0])
            eps_p = float(state[i, 0]); alpha = float(state[i, 1])
            backstress = float(state[i, 2])
            trial = E * (eps - eps_p)
            shifted = trial - backstress
            if shifted > self._bound(tension, alpha):
                direction, table = 1.0, tension
            elif shifted < -self._bound(compression, alpha):
                direction, table = -1.0, compression
            else:
                direction, table = 0.0, None
            if direction:
                dp, old_curve = self._plastic_increment(
                    trial, eps_p, alpha, backstress, direction, table)
                new_curve = self._value(table, alpha + dp)
                eps_p += direction * dp
                alpha += dp
                backstress += direction * b * (new_curve - old_curve)
                state[i, 0] = eps_p; state[i, 1] = alpha; state[i, 2] = backstress
            stress = E * (eps - eps_p)
            state[i, 5] += 0.5 * (float(state[i, 4]) + stress) * (eps - float(state[i, 3]))
            state[i, 3] = eps
            state[i, 4] = stress
            out[i, 0] = stress
        return out

    def output_fields(self, state):
        elastic = state[:, 4] ** 2 / (2.0 * self.E)
        return {
            "eps_p": state[:, 0].copy(),
            "alpha": state[:, 1].copy(),
            "backstress": state[:, 2].copy(),
            "axial_strain": state[:, 3].copy(),
            "axial_stress": state[:, 4].copy(),
            "external_work": state[:, 5].copy(),
            "elastic_energy": elastic,
            # Energy per unit volume left after a purely elastic unload.  At a
            # closed cycle this is exactly the stress-strain loop area.
            "hysteretic_energy": np.maximum(state[:, 5] - elastic, 0.0),
        }

    def describe(self):
        return (f"    E = {self.E / 1e9:.4g} GPa; asymmetric piecewise plasticity; "
                f"hardening={self.hardening_mode} "
                f"(kinematic {100*self.kinematic_fraction:.0f}%)")
