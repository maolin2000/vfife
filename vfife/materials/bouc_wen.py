"""The Bouc-Wen hysteretic law (1.22): a smooth elastic-plastic loop with
one internal variable, written from the papers.

    sigma = alpha E eps + (1 - alpha) fy z

    eps_y dz = [A - |z|^n (beta sgn(deps z) + gamma)] deps          (Wen 1976)

Bouc (1967) gave the hysteretic operator, Wen (1976) this differential form
with its exponent `n`: z is the hysteretic part as a multiple of the yield
level, eps_y = fy / E. With A = 1 and beta + gamma = 1 (the defaults,
beta = gamma = 0.5) |z| tends to 1 under monotonic loading: the hysteretic
part saturates at (1 - alpha) fy, the loop's post-yield slope is alpha E,
and `n` sets how sharp the knee is (n -> infinity: bilinear). Loading from
rest with n = 1 it is the closed form

    z = (A / (beta + gamma)) [1 - exp(-(beta + gamma) eps / eps_y)].

IN THE PLANE (Park, Wen & Ang 1986): two components that share the law and
one z vector,

    eps_y dz = A deps - |z|^(n-2) z [beta (|deps_1 z_1| + |deps_2 z_2|) + gamma deps . z],

which is the one-dimensional law along each axis and, for n = 2, Park,
Wen and Ang's: the yield surface is the circle |z| = 1, so a bearing that has
yielded along x carries less along y. A connector's `uy` and `uz` given the
same `material` law of this type are one law in the plane (as `friction` and
`bilinear` are), `elements/connector.py`.

INTEGRATION: the implicit midpoint rule on z over the strain increment of
the step (second order; stable at a sharp knee), solved by Newton to
1e-12, in sub-increments of at most a fiftieth of eps_y
(an explicit step is far smaller than that; a Newton step of a static
analysis may not be). Rate-independent: dt is not read.

State: the strain the step starts from, z, and the energy the loop has
dissipated (per unit volume, the hysteretic part's work less its stored
energy at the end).
"""
from __future__ import annotations

import math

import numpy as np

from .base import Material, register

_SUB = 0.02            # sub-increments of at most this times eps_y
_TOL = 1.0e-12


def _rate(z, de, A, beta, gamma, n):
    """dz/d(eps/eps_y) at z, along de (scalar)."""
    az = abs(z)
    s = 1.0 if de * z >= 0.0 else -1.0
    return A - az ** n * (beta * s + gamma)


def _dz(z, de, ey, A, beta, gamma, n):
    """z after a small increment de (scalar), by the implicit midpoint rule
    (second order, and stable where the knee is sharp)."""
    z0 = z
    x = z0 + _rate(z0, de, A, beta, gamma, n) * de / ey
    for _ in range(50):
        m = 0.5 * (z0 + x)
        g = x - z0 - _rate(m, de, A, beta, gamma, n) * de / ey
        am = abs(m)
        s = 1.0 if de * m >= 0.0 else -1.0
        dr = -n * am ** (n - 1.0) * (1.0 if m >= 0.0 else -1.0) * (beta * s + gamma) if am > 0.0 else 0.0
        dg = 1.0 - 0.5 * dr * de / ey
        step = g / dg
        x -= step
        if abs(step) <= _TOL * max(1.0, abs(x)):
            break
    return x


def _rate2(z, de, A, beta, gamma, n):
    r = math.hypot(z[0], z[1])
    w = r ** (n - 2.0) if r > 0.0 else (1.0 if n == 2.0 else 0.0)
    flow = beta * (abs(de[0] * z[0]) + abs(de[1] * z[1])) + gamma * (de[0] * z[0] + de[1] * z[1])
    return A * de - w * z * flow


def _dz2(z, de, ey, A, beta, gamma, n):
    """The planar z (2,) after a small increment de (2,), implicit midpoint."""
    z0 = np.array(z, float)
    de = np.asarray(de, float)

    def g(x):
        return x - z0 - _rate2(0.5 * (z0 + x), de, A, beta, gamma, n) / ey

    x = z0 + _rate2(z0, de, A, beta, gamma, n) / ey
    for _ in range(50):
        r = g(x)
        h = 1.0e-7 * max(1.0, float(np.abs(x).max()))
        J = np.empty((2, 2))
        for j in range(2):
            xp = x.copy(); xp[j] += h
            J[:, j] = (g(xp) - r) / h
        step = np.linalg.solve(J, r)
        x -= step
        if float(np.abs(step).max()) <= _TOL * max(1.0, float(np.abs(x).max())):
            break
    return x


@register("bouc_wen", kinematics=("uniaxial",), backends=("numpy",), nonlinear=True,
          verification="analytical",
          output_fields=("stress", "z", "hysteretic_energy"))
class BoucWen(Material):
    """Bouc-Wen hysteresis: `E` the initial modulus, `fy` the yield
    stress, `alpha` the post-yield ratio, `n` the sharpness of the knee,
    `beta`, `gamma`, `A` Wen's shape parameters."""

    n_state = 3
    state_names = {"eps_prev": slice(0, 1), "z": slice(1, 2), "hysteretic_energy": slice(2, 3)}
    #: the planar state, a connector's uy and uz together
    n_state_planar = 5
    uniaxial = True

    def __init__(self, E: float, fy: float, rho: float = 1.0, alpha: float = 0.0, n: float = 1.0,
                 beta: float = 0.5, gamma: float = 0.5, A: float = 1.0, **_ignored):
        self.E, self.fy, self.rho, self.nu = float(E), float(fy), float(rho), 0.0
        self.alpha, self.n = float(alpha), float(n)
        self.beta, self.gamma, self.A = float(beta), float(gamma), float(A)
        if not (np.isfinite(self.E) and self.E > 0.0):
            raise ValueError("bouc_wen: E is finite and positive")
        if not (np.isfinite(self.fy) and self.fy > 0.0):
            raise ValueError("bouc_wen: fy is finite and positive")
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError("bouc_wen: alpha, the post-yield ratio, is in [0, 1)")
        if not self.n >= 1.0:
            raise ValueError("bouc_wen: n is 1 or more")
        if not self.A > 0.0:
            raise ValueError("bouc_wen: A is positive")
        if not (self.beta > 0.0 and -self.beta < self.gamma <= self.beta):
            raise ValueError("bouc_wen: beta > 0 and -beta < gamma <= beta (a loop that dissipates, and "
                             "a z that stays bounded)")
        if not self.rho > 0.0:
            raise ValueError("bouc_wen: rho is positive")
        self.ey = self.fy / self.E
        #: the largest tangent the law reaches -- unloading from saturation,
        #: E [alpha + (1 - alpha) A (1 + (beta - gamma) / (beta + gamma))] -- for
        #: the explicit step and the reference stiffness (A = 1, beta = gamma:
        #: E itself)
        self.tangent_max = self.E * (self.alpha + (1.0 - self.alpha) * self.A
                                     * max(1.0, 2.0 * self.beta / (self.beta + self.gamma)))
        self.wave_speed = float(np.sqrt(self.tangent_max / self.rho))

    def _params(self):
        return self.ey, self.A, self.beta, self.gamma, self.n

    def _pieces(self, de):
        if not np.all(np.isfinite(de)):
            raise ValueError("bouc_wen: a strain that is not finite")
        return max(1, int(math.ceil(float(np.max(np.abs(de))) / (_SUB * self.ey))))

    def _advance(self, z, de):
        """(z after the strain increment de, the energy the loop dissipated
        over it per unit volume), in sub-increments: the hysteretic part's
        work less what it stores, h eps_y z^2 / (2 A)."""
        m = self._pieces(de)
        h = (1.0 - self.alpha) * self.fy
        w = 0.0
        for _ in range(m):
            z1 = _dz(z, de / m, *self._params())
            w += h * 0.5 * (z + z1) * de / m - h * self.ey * 0.5 * (z1 * z1 - z * z) / self.A
            z = z1
        return z, w

    def update(self, strain, state, dt):
        eps = np.asarray(strain, dtype=float)[:, 0]
        out = np.zeros_like(strain, dtype=float)
        h = (1.0 - self.alpha) * self.fy
        for i in range(len(eps)):
            e0, z0 = float(state[i, 0]), float(state[i, 1])
            z, w = self._advance(z0, float(eps[i]) - e0)
            state[i, 2] += w
            state[i, 0], state[i, 1] = eps[i], z
            out[i, 0] = self.alpha * self.E * eps[i] + h * z
        return out

    # -- in the plane: a connector's uy and uz -------------------------------
    def init_state_planar(self) -> np.ndarray:
        return np.zeros((1, self.n_state_planar))

    def update_planar(self, eps2, state) -> np.ndarray:
        """(2,) stress at the strains `eps2`, advancing the planar `state`
        (1, 5): the strains the step starts from, z (2,), the energy."""
        eps2 = np.asarray(eps2, float).reshape(2)
        e0 = state[0, 0:2].copy(); z0 = state[0, 2:4].copy()
        de = eps2 - e0
        m = self._pieces(de)
        z = z0.copy()
        h = (1.0 - self.alpha) * self.fy
        for _ in range(m):
            z1 = _dz2(z, de / m, *self._params())
            state[0, 4] += h * float(0.5 * (z + z1) @ (de / m)) - h * self.ey * 0.5 * (float(z1 @ z1) - float(z @ z)) / self.A
            z = z1
        state[0, 0:2] = eps2; state[0, 2:4] = z
        return self.alpha * self.E * eps2 + h * z

    def output_fields(self, state):
        """(the one-dimensional state; a planar one is a connector's, (1, 5))"""
        if state.shape[1] != self.n_state:
            return {"hysteretic_energy": state[:, 4].copy()}
        return {"z": state[:, 1].copy(), "hysteretic_energy": state[:, 2].copy()}

    def describe(self) -> str:
        return (f"    bouc_wen E={self.E:.4g}, fy={self.fy:.4g}, alpha={self.alpha:g}, n={self.n:g}, "
                f"beta={self.beta:g}, gamma={self.gamma:g}, A={self.A:g}")
