"""Cyclic steel with ultra-low-cycle fatigue and fracture (Cai et al. 2023 §3).

The law the paper's collapse analyses run on, and the one v0.1 said the
fracture framework needs: a bare rupture latch drops the stress to zero in one
step, which is a finite jump in the constitutive function -- the shape that put
a bond node in a limit cycle for 8 000 kinetic resets in v0.1. This law falls
to `f_uf` along a degradation curve BEFORE the fibre is cut, so what the latch
finally removes is a small residual rather than the whole force.

FOUR CURVES AND A STATE MACHINE (their Fig. 7):

    monotonic       first loading, either sign          -- eq. (10)
    skeleton        loading after any reversal          -- eq. (11), Ramberg-Osgood
    degradation     once the curve in use reaches f_u   -- eq. (12)
    reloading       from the unloaded state to the peak -- eqs. (14)-(18)

    unloading is elastic at E, and that is the whole of criterion (2).

WHAT THE FRACTURE STRAIN DOES. `eps_cri` is the monotonic critical strain; the
DSPS model (eq. 13) shrinks it as compressive plastic strain accumulates,

    eps_uf = exp(-lambda_DSPS * eps_c_acc) * eps_cri,

so a fibre that has been pushed back and forth in compression fractures at a
tensile strain well below the coupon value. That is the ultra-low-cycle
fatigue: fewer than twenty cycles, but plastic ones.

THREE PLACES WHERE THE PRINTED EQUATIONS DO NOT CLOSE, and what is done here
instead. Each is checked in tests/test_steel_cyclic.py, so the choice is
visible rather than buried:

  * eq. (10), the hardening branch, is printed with a coefficient
    `E(1-k1)/(eps_y (k2-k1)^2)`. With Table 1's numbers that leaves the curve
    discontinuous at eps = k1 eps_y by f_y(1-k1) -- for Q345B a drop of 1.5 GPa
    at first yield. The coefficient that makes the quadratic pass through
    (k1 eps_y, f_y) and (k2 eps_y, f_u) with zero slope at the second is
    `(f_y - f_u)/(eps_y^2 (k2-k1)^2)`, and it reduces to the printed one with
    k3 in place of k1 when f_u = k3 f_y -- which is what Fig. 4 draws. Used
    here.

  * eq. (12), the degradation curve, is printed as
    `f_u + (f_u - f_uf)((eps - eps_uf)/(eps_u - eps_uf))^2`, which gives
    2f_u - f_uf at the necking strain and f_u at fracture: both endpoints
    wrong. The form that matches their Fig. 5 -- flat at necking, steep at
    fracture -- is `f_u + (f_uf - f_u)((eps - eps_u)/(eps_uf - eps_u))^2`.
    Used here.

  * eq. (11) does not meet the elastic line: the skeleton at the yield point
    evaluates to 1.030 f_y for the paper's parameters, a 3% step at first
    yield on every reloading path. `skeleton_fit="shift"` (the default)
    subtracts that constant so the curve starts at f_y and keeps its slope
    everywhere; `"raw"` reproduces the paper exactly and is what the test
    measures the jump with.
"""
from __future__ import annotations

import numpy as np

from .base import Material, register

# ONE DEFINITION OF THE LAW, COMPILED IN PLACE. steel02.py keeps a readable
# reference next to a separately written kernel and pins them against each
# other, because its kernel is a rewrite. Here the scalar functions below are
# the law, and numba compiles the same source: `_point.py_func` is still the
# Python one, so the two cannot drift, and the fused fibre-beam kernel inlines
# `_point` itself.
try:                                            # pragma: no cover - import guard
    from numba import njit
except Exception:                               # pragma: no cover
    njit = None


def _jit(fn):
    return njit(cache=True, inline="always")(fn) if njit is not None else fn


_BATCH = None


def _batch():
    """The compiled loop over fibre points, built once and kept.

    `update` is called with every fibre of every section at every step, so the
    per-call dispatch of a compiled scalar function would dominate: the loop
    has to be inside the compiled code, not around it.
    """
    global _BATCH
    if _BATCH is None and njit is not None:
        @njit(cache=True)
        def batch(strain, state, out, p):
            for i in range(strain.shape[0]):
                out[i, 0] = _point(strain[i, 0], state[i], p)
            return out
        _BATCH = batch
    return _BATCH

# --------------------------------------------------------------------------- #
#  Table 1 of the paper, so a case file can say `grade: Q345B` and stop
# --------------------------------------------------------------------------- #

#: k1, k2, k3, a, b0, b1, b2, fy, fu, eps_cri, lambda_DSPS -- stresses in N/mm^2
GRADES = {
    "Q345B":  dict(k1=4.5, k2=45.0, k3=1.35, a=5.796, b0=2.486, b1=0.608,
                   b2=-0.00046, fy=421.0e6, fu=586.0e6, eps_cri=0.36, lambda_dsps=0.28),
    "Q235B":  dict(k1=5.0, k2=46.0, k3=1.44, a=5.796, b0=2.486, b1=0.608,
                   b2=-0.00046, fy=404.0e6, fu=596.0e6, eps_cri=0.43, lambda_dsps=0.23),
    "HPS485": dict(k1=5.0, k2=46.0, k3=1.2, a=5.796, b0=2.486, b1=0.608,
                   b2=-0.00046, fy=503.0e6, fu=678.0e6, eps_cri=0.29, lambda_dsps=0.28),
    "HT440":  dict(k1=5.0, k2=46.0, k3=1.2, a=5.796, b0=2.486, b1=0.608,
                   b2=-0.00046, fy=501.0e6, fu=800.0e6, eps_cri=0.24, lambda_dsps=0.28),
}

#: branch codes stored in the state
VIRGIN, ELASTIC, RELOAD, SKELETON = 0.0, 1.0, 2.0, 3.0


# --------------------------------------------------------------------------- #
#  the four curves, as scalar functions of |strain|
#
#  Scalar and parameter-array driven on purpose: the compiled twin inlines
#  exactly these, and a law that is written twice is a law that will disagree
#  with itself (the lesson steel02.py records at the top of its own file).
# --------------------------------------------------------------------------- #

@_jit
def _monotonic(e, E, fy, fu, k1, k2, ey):
    """Eq. (10): elastic, yield plateau, quadratic hardening, then f_u."""
    if e < ey:
        return E * e
    if e < k1 * ey:
        return fy
    if e < k2 * ey:
        return fu + (fy - fu) * ((e - k2 * ey) / ((k2 - k1) * ey)) ** 2
    return fu


@_jit
def _skeleton(e, E, fy, a, b0, b1, b2, ey, shift):
    """Eq. (11): Ramberg-Osgood in normalised (eps/eps_y, sigma/f_y)."""
    if e <= ey:
        return E * e
    u = e / ey + a
    return fy * (u / (b0 + b1 * u + b2 * u * u) - shift)


@_jit
def _skeleton_slope(e, E, fy, a, b0, b1, b2, ey):
    if e <= ey:
        return E
    u = e / ey + a
    den = b0 + b1 * u + b2 * u * u
    return E * (b0 - b2 * u * u) / (den * den)


@_jit
def _degradation(e, fu, fuf, eu, euf):
    """Eq. (12), in the form that meets Fig. 5 at both ends."""
    if euf <= eu:
        return fuf
    x = (e - eu) / (euf - eu)
    if x < 0.0:
        x = 0.0
    if x > 1.0:
        x = 1.0
    return fu + (fuf - fu) * x * x


@_jit
def _degradation_slope(e, fu, fuf, eu, euf):
    if euf <= eu:
        return 0.0
    return 2.0 * (fuf - fu) * (e - eu) / ((euf - eu) * (euf - eu))


def _necking_strain(fu, E, fy, a, b0, b1, b2, ey, shift):
    """Where the skeleton reaches f_u -- the necking strain under cycling.

    Bisection rather than the quadratic root, because b2 < 0 puts a pole in
    the Ramberg-Osgood denominator and the algebraic root has to be told which
    side of it to take. The bracket is [eps_y, the pole), and the function is
    monotone on it.
    """
    lo = ey
    hi = ey * 1.0e4
    if b2 < 0.0:                       # the pole, if there is one
        disc = b1 * b1 - 4.0 * b0 * b2
        pole = (-b1 - np.sqrt(disc)) / (2.0 * b2)      # positive root in u
        hi = min(hi, (pole - a) * ey * 0.999)
    if _skeleton(hi, E, fy, a, b0, b1, b2, ey, shift) < fu:
        return hi                       # never reaches f_u: no degradation
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _skeleton(mid, E, fy, a, b0, b1, b2, ey, shift) < fu:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
#  the reloading curve, eqs. (14)-(18)
# --------------------------------------------------------------------------- #

@_jit
def _rho(E, der, dsr, Eh):
    """Eq. (16). `der`, `dsr` are magnitudes; `Eh` keeps its sign."""
    den = E * der - dsr
    if den <= 1.0e-30:
        return 1.0
    return E * der * (1.0 - Eh / E) / den


@_jit
def _zeta(E, der, dsr, sp, fu):
    """Eq. (18), on magnitudes."""
    if dsr <= 1.0e-30:
        return 1.0
    x = E * der
    inner = (x - dsr) * (x - 5.0 * dsr)
    if inner < 0.0:
        inner = 0.0
    z = (x - 3.0 * dsr - np.sqrt(inner)) / (2.0 * dsr) * (sp / max(fu, 1.0e-30))
    return max(z, 1.0e-12)


@_jit
def _eta(r, zeta):
    """Eq. (17): 1 at the peak point, zeta/(zeta+1) at the start."""
    return 1.0 + zeta / (zeta + 1.0) - zeta / (r + zeta)


@_jit
def _reload_stress(e, e0, s0, ep, sp, E, Eh, fu):
    """Eq. (15), with the branch chosen by the secant stiffness E_k (eq. 14).

    Everything below runs on the magnitudes of the two increments, so one
    piece of algebra serves tensile and compressive reloading; `d` puts the
    sign back. Written the other way -- signed throughout -- the square root
    in eq. (18) and the power in eq. (15) both need their own case analysis.
    """
    d = 1.0 if ep >= e0 else -1.0
    der = abs(ep - e0)
    dsr = abs(sp - s0)
    if der <= 1.0e-30:
        return sp
    Ek = dsr / der
    x = abs(e - e0)
    r = x / der
    if r > 1.0:
        r = 1.0
    if Ek >= E * (1.0 - 1.0e-12):                       # elastic
        return s0 + d * E * x
    if Ek >= E / 5.0:                                   # Legeron
        p = _rho(E, der, dsr, Eh)
        return s0 + d * (E * x - (r ** p) * (E * der - dsr))
    z = _zeta(E, der, dsr, abs(sp), fu)                 # Shi and Wang
    return s0 + d * (E * x - _eta(r, z) * (E - Ek) * x)


# --------------------------------------------------------------------------- #

@register("steel_cyclic", kinematics=("uniaxial",), backends=("numpy", "numba"),
          verification="constitutive_path",
          output_fields=("axial_stress", "axial_strain", "eps_max", "eps_acc_c",
                         "eps_uf", "failed"),
          caveat="fracture strain from the DSPS model; f_uf is not in the paper's "
                 "Table 1 and defaults to 0.5 f_u")
class SteelCyclic(Material):
    """Cyclic steel with degradation, ULCF and a fracture strain."""

    n_state = 12
    state_names = {
        "eps_prev": slice(0, 1), "sig_prev": slice(1, 2),
        "branch": slice(2, 3), "direction": slice(3, 4),
        "eps_max": slice(4, 5),
        "eps0": slice(5, 6), "sig0": slice(6, 7),
        "eps_acc_c": slice(7, 8), "failed": slice(8, 9),
        "eps_rev": slice(9, 10), "sig_rev": slice(10, 11),
        "wobbled": slice(11, 12),
    }
    uniaxial = True

    def __init__(self, E=206.0e9, rho=7850.0, nu=0.3, grade=None,
                 fy=None, fu=None, f_uf=None, k1=None, k2=None, k3=None,
                 a=None, b0=None, b1=None, b2=None,
                 eps_cri=None, lambda_dsps=None,
                 skeleton_fit="shift", **_ignored):
        if grade is not None and grade not in GRADES:
            raise ValueError(f"unknown steel grade {grade!r}; Table 1 of the "
                             f"paper has {', '.join(sorted(GRADES))}")
        table = dict(GRADES[grade]) if grade else {}

        def pick(name, given, default=None):
            v = given if given is not None else table.get(name, default)
            if v is None:
                raise ValueError(
                    f"steel_cyclic needs {name} (give it, or a `grade` from "
                    f"{', '.join(sorted(GRADES))})")
            return float(v)

        self.E, self.rho, self.nu = float(E), float(rho), float(nu)
        self.fy = pick("fy", fy)
        self.k3 = pick("k3", k3, 1.0)
        # f_u = k3 f_y is what Fig. 4 draws and what makes eq. (10) continuous;
        # Table 1 also lists a measured f_u, which is 3-25% away from it. The
        # measured value wins when it is given, because it is the one the
        # degradation curve starts from.
        self.fu = pick("fu", fu, self.k3 * self.fy)
        self.f_uf = float(f_uf) if f_uf is not None else 0.5 * self.fu
        if not 0.0 <= self.f_uf <= self.fu:
            # a negative one gave a compressive stress at 36% tension, one
            # above fu a stress past fu
            raise ValueError(f"steel_cyclic f_uf = {self.f_uf:.4g} must lie in [0, fu = {self.fu:.4g}]: "
                             "it is the stress the degradation curve ends at, at fracture")
        self.k1 = pick("k1", k1)
        self.k2 = pick("k2", k2)
        self.a = pick("a", a)
        self.b0 = pick("b0", b0)
        self.b1 = pick("b1", b1)
        self.b2 = pick("b2", b2)
        self.eps_cri = pick("eps_cri", eps_cri)
        self.lambda_dsps = pick("lambda_dsps", lambda_dsps)
        self.grade = grade

        if min(self.E, self.rho, self.fy, self.fu, self.eps_cri) <= 0.0:
            raise ValueError("steel_cyclic E, rho, fy, fu and eps_cri must be positive")
        if self.fu < self.fy:
            raise ValueError("steel_cyclic f_u must not be below f_y")
        if not 1.0 < self.k1 < self.k2:
            raise ValueError("steel_cyclic needs 1 < k1 < k2 (yield plateau, then hardening)")
        if self.lambda_dsps < 0.0:
            raise ValueError("steel_cyclic lambda_dsps must not be negative")
        if skeleton_fit not in ("shift", "raw"):
            raise ValueError("steel_cyclic skeleton_fit is 'shift' or 'raw'")

        self.eps_y = self.fy / self.E
        self.skeleton_fit = skeleton_fit
        # the constant that puts the Ramberg-Osgood branch on the elastic line
        # at first yield; `raw` keeps the paper's 3% step
        u1 = 1.0 + self.a
        self.shift = (0.0 if skeleton_fit == "raw"
                      else u1 / (self.b0 + self.b1 * u1 + self.b2 * u1 * u1) - 1.0)
        # necking: where the curve in use reaches f_u. Two of them, because the
        # monotonic curve and the skeleton reach it at different strains -- the
        # paper's point in Fig. 5, and the reason a cycled fibre necks early.
        self.eps_u_mono = self.k2 * self.eps_y
        self.eps_u_skel = _necking_strain(self.fu, self.E, self.fy, self.a,
                                          self.b0, self.b1, self.b2,
                                          self.eps_y, self.shift)
        # THE ONE ORDERING THE MODEL DEPENDS ON. Fig. 5 exists to say that
        # cycling brings necking forward; if the skeleton reaches f_u LATER
        # than the monotonic curve does, the law says the opposite, and it
        # will say it quietly for the whole run. Table 1's HT440 row is such
        # a set: k3 = 1.2 with f_u/f_y = 1.60, and its four Ramberg-Osgood
        # parameters are the same defaults as every other row.
        if self.eps_u_skel >= self.eps_u_mono:
            raise ValueError(
                f"steel_cyclic: the hysteresis skeleton reaches f_u at "
                f"{self.eps_u_skel*100:.2f}% but the monotonic curve reaches it at "
                f"{self.eps_u_mono*100:.2f}%, so cycling would DELAY necking -- the "
                f"opposite of the model's premise (their Fig. 5). f_u = {self.fu/1e6:.0f} "
                f"MPa is {self.fu/self.fy:.2f} f_y, and the Ramberg-Osgood row "
                f"(a={self.a}, b0={self.b0}, b1={self.b1}, b2={self.b2}) does not get "
                f"there in time. Lower f_u -- k3 f_y is {self.k3*self.fy/1e6:.0f} MPa -- "
                f"or fit a, b0, b1, b2 to this steel")
        self.wave_speed = float(np.sqrt(self.E / self.rho))
        self.D0 = np.diag([self.E, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.sy = self.fy

    # ---- the paper's curves, exposed for tests and for anyone plotting ----
    def monotonic(self, e):
        return _monotonic(abs(e), self.E, self.fy, self.fu, self.k1, self.k2,
                          self.eps_y) * np.sign(e)

    def skeleton(self, e):
        return _skeleton(abs(e), self.E, self.fy, self.a, self.b0, self.b1,
                         self.b2, self.eps_y, self.shift) * np.sign(e)

    def fracture_strain(self, eps_acc_c=0.0):
        """Eq. (13): the DSPS fatigue model."""
        return float(np.exp(-self.lambda_dsps * eps_acc_c) * self.eps_cri)

    def describe(self) -> str:
        g = f"{self.grade} " if self.grade else ""
        return (f"    steel_cyclic {g}fy={self.fy/1e6:.0f}MPa fu={self.fu/1e6:.0f}MPa "
                f"E={self.E/1e9:.0f}GPa necking {self.eps_u_skel*100:.2f}% (cyclic) / "
                f"{self.eps_u_mono*100:.2f}% (monotonic), fracture {self.eps_cri*100:.0f}% "
                f"at zero accumulated compression")

    # ---- the law ---------------------------------------------------------
    def numba_params(self):
        return np.array([self.E, self.fy, self.fu, self.f_uf, self.k1, self.k2,
                         self.a, self.b0, self.b1, self.b2, self.eps_cri,
                         self.lambda_dsps, self.eps_y, self.eps_u_skel,
                         self.shift, self.eps_u_mono], dtype=np.float64)

    def update(self, strain, state, dt, params=None):
        out = np.zeros_like(strain)
        p = self.numba_params() if params is None else params
        batch = _batch()
        if batch is not None:
            # the compiled loop needs a contiguous float64 state; a strided
            # or float32 one was COPIED and the update went into the copy, so
            # every call looked like first loading
            st = np.ascontiguousarray(state, dtype=np.float64)
            res = batch(np.ascontiguousarray(strain, dtype=np.float64), st, out, p)
            if st is not state:
                state[...] = st
            return res
        for i in range(len(strain)):
            out[i, 0] = _point(float(strain[i, 0]), state[i], p)
        return out

    def numba_uniaxial_fn(self):
        """The scalar law itself, for a fibre-beam kernel to inline.

        Not a twin: the same compiled `_point` this material's own `update`
        calls, so the fused path and the reference path cannot disagree.
        """
        return _point if njit is not None else None

    def numba_point_fn(self):
        return None

    def output_fields(self, state):
        p = self.numba_params()
        acc = state[:, 7]
        return {
            "eps_max": state[:, 4].copy(),
            "eps_acc_c": acc.copy(),
            "eps_uf": np.exp(-p[11] * acc) * p[10],
            "failed": state[:, 8].copy(),
        }

    def damage_class(self, state):
        resid = np.abs(state[:, 0] - state[:, 1] / self.E)
        out = np.zeros(len(state), dtype=np.int8)
        out[resid > 0.1 * self.fy / self.E] = self.DAMAGE_YIELDED
        out[state[:, 8] > 0.5] = self.DAMAGE_FAILED
        return out

    def failure(self, state):
        """The latch the element counts.

        The same event the law itself acts on: past `eps_uf` the stress is
        zero, and this says so rather than leaving the element to infer it
        from a stress that an unloaded fibre also reports.
        """
        return state[:, 8].copy()

    def damage_measure(self, state):
        """How far the fibre is into its fatigue life, in [0, 1].

        The ACCUMULATED COMPRESSION over what would exhaust `eps_cri`, not the
        current strain over the current `eps_uf`: the second falls when the
        fibre unloads, and a damage measure may not.
        """
        p = self.numba_params()
        if p[11] <= 0.0:
            return np.clip(state[:, 8], 0.0, 1.0)
        return np.clip(1.0 - np.exp(-p[11] * state[:, 7]), 0.0, 1.0)


@_jit
def _point(eps, st, p):
    """One material point, one step. `st` is mutated; returns the stress.

    THE STATE MACHINE IS THE PAPER'S FIVE CRITERIA, one branch each:

      VIRGIN    first loading, on the monotonic curve (criterion 1)
      ELASTIC   after any reversal, slope E (criterion 2)
      RELOAD    from the unloaded point toward the peak (criteria 3, 4)
      SKELETON  loading past the peak, on the skeleton or the degradation
                curve (criterion 5)

    The peak point is ISOTROPIC -- its strain is the largest magnitude the
    point has ever reached, whichever sign it was reached in -- which is the
    paper's own reading of steel's isotropic hardening, and what puts E in
    Fig. 7 at -eps_1 after A was recorded at +eps_1.
    """
    (E, fy, fu, fuf, k1, k2, a, b0, b1, b2, eps_cri, lam,
     ey, eu_skel, shift, eu_mono) = (p[0], p[1], p[2], p[3], p[4], p[5], p[6],
                                     p[7], p[8], p[9], p[10], p[11], p[12],
                                     p[13], p[14], p[15])

    if st[8] > 0.5:                       # already fractured: it carries nothing
        st[0] = eps
        st[1] = 0.0
        return 0.0

    eps_prev, sig_prev = st[0], st[1]
    branch, direction = st[2], st[3]
    de = eps - eps_prev
    moved = abs(de) > 1.0e-16
    if not moved and branch != VIRGIN:
        # THE SAME STRAIN IS THE SAME STRESS. A repeated call fell back to the
        # branch's direction, and on an elastic unloading branch whose stress
        # still had that sign it jumped onto a reload curve: 340 -> 202 MPa
        # with no change of strain, and a different history ever after.
        # Solvers do call a law twice at one strain.
        return sig_prev
    if moved:
        d = 1.0 if de > 0.0 else -1.0
    else:
        d = direction if direction != 0.0 else (1.0 if eps >= 0.0 else -1.0)
    # A STEP THAT DID NOT MOVE MAY NOT COMMIT A DIRECTION. Every solver here
    # evaluates the internal force once before the first step, at zero strain,
    # and if that call banks `direction = +1` then the first real step
    # downwards looks like a REVERSAL: the point leaves the monotonic curve
    # for the hysteresis skeleton before it has ever been loaded. The two
    # differ by 10% in stress and by a factor of 2.5 in necking strain
    # (Q345B: 9.20% monotonic against 3.61% on the skeleton), so the point
    # then fractured at a strain it should have carried.
    keep_direction = direction if not moved else d

    # ---- a reversal sends every branch to the elastic one ------------------
    if direction != 0.0 and d != direction and branch != ELASTIC:
        st[9], st[10] = eps_prev, sig_prev
        if branch == VIRGIN:
            st[11] = 1.0                  # left the monotonic curve, not yet cycled
        branch = ELASTIC

    eps_uf = np.exp(-lam * st[7]) * eps_cri
    ae = abs(eps)

    if branch == ELASTIC:
        rev_e, rev_s = st[9], st[10]
        # (i) back past where we left the previous branch: resume it
        if (d == direction and ((direction > 0.0 and eps >= rev_e)
                                or (direction < 0.0 and eps <= rev_e))):
            if st[11] > 0.5:
                # A WOBBLE ON THE MONOTONIC CURVE IS NOT A CYCLE. A point that
                # turned back by 1e-5 on its yield plateau and came on again
                # resumed on the hysteresis skeleton: Q345B at 2%, 461 -> 541
                # MPa in one step, and 561 -> 584 in the hardening range.
                # It resumes the curve it left, and moves to
                # the skeleton only where that lies below.
                branch = VIRGIN
            else:
                branch = SKELETON if ae >= st[4] - 1.0e-15 else RELOAD
        else:
            sig = rev_s + E * (eps - rev_e)
            # (ii) the stress now has the sign of travel: this is no longer an
            # unloading, it is a RELOAD toward the peak the other way.
            #
            # THE TEST IS ON `sig` ALONE. It used to also require
            # `rev_s * d < 0`, i.e. that the reversal left a stress of the
            # opposite sign to unload through -- true of every textbook
            # reversal and false of the one that matters: a point that turns
            # round AT ZERO STRESS. Then the elastic branch was never left,
            # and `sig = rev_s + E (eps - rev_e)` ran on with slope E for
            # ever. In a dynamic run every fibre reverses near zero during the
            # first oscillations, so a monotone push afterwards stayed elastic
            # past yield: measured on Cai et al.'s Tube A, fibre stresses of
            # 462 MPa on a law with fy = 288 and fu = 361, and a member squash
            # load 46% too high.
            if sig * d > 0.0:
                st[5] = rev_e - rev_s / E          # the zero-stress point
                st[6] = 0.0
                st[11] = 0.0                       # a real cycle: the paper's curves
                branch = RELOAD
                direction = d
            else:
                st[0], st[1], st[2], st[3] = eps, sig, ELASTIC, direction
                return sig

    if branch == VIRGIN:
        eu = eu_mono
        if ae >= eps_uf and eps > 0.0:
            st[8] = 1.0
            sig = 0.0
        elif ae > eu:
            sig = d_sign(eps) * _degradation(ae, fu, fuf, eu, eps_uf)
        else:
            sig = d_sign(eps) * _monotonic(ae, E, fy, fu, k1, k2, ey)
        if st[11] > 0.5 and sig != 0.0:
            # after a wobble: the monotonic curve until the skeleton (or its
            # degradation) is the lower of the two
            if ae > eu_skel:
                other = _degradation(ae, fu, fuf, eu_skel, eps_uf)
            else:
                other = _skeleton(ae, E, fy, a, b0, b1, b2, ey, shift)
            if other < abs(sig):
                sig = d_sign(eps) * other
        if ae > st[4]:
            st[4] = ae
        direction = keep_direction

    elif branch == RELOAD:
        peak_e = direction * max(st[4], ey)
        ap = abs(peak_e)
        if ap > eu_skel:
            peak_s = direction * _degradation(ap, fu, fuf, eu_skel, eps_uf)
            Eh = _degradation_slope(ap, fu, fuf, eu_skel, eps_uf)
        else:
            peak_s = direction * _skeleton(ap, E, fy, a, b0, b1, b2, ey, shift)
            Eh = _skeleton_slope(ap, E, fy, a, b0, b1, b2, ey)
        past = (direction > 0.0 and eps >= peak_e) or (direction < 0.0 and eps <= peak_e)
        if past:
            # THE ELASTIC CAP MEETS THE SKELETON LATER THAN THE PEAK STRAIN.
            # When the secant to the peak is steeper than E (eq. 14's elastic
            # case) the reload line arrives at the peak strain BELOW the peak
            # stress, and switching to the skeleton there was a step: Q345B
            # pushed to 2%, unloaded into compression and reloaded, 92 MPa in
            # a 1e-4 strain step. It stays on the line at E
            # until the line meets the curve.
            el = st[6] + direction * E * abs(eps - st[5])
            if ae > eu_skel:
                cur = _degradation(ae, fu, fuf, eu_skel, eps_uf)
            else:
                cur = _skeleton(ae, E, fy, a, b0, b1, b2, ey, shift)
            if abs(el) < cur and ae < eps_uf:
                _accumulate(st, eps, eps_prev, el, sig_prev, E, d)
                if ae > st[4]:
                    st[4] = ae
                st[0], st[1], st[2], st[3] = eps, el, RELOAD, direction
                return el
            branch = SKELETON                  # reached the peak: carry on
        else:
            sig = _reload_stress(eps, st[5], st[6], peak_e, peak_s, E, Eh, fu)
            _accumulate(st, eps, eps_prev, sig, sig_prev, E, d)
            st[0], st[1], st[2], st[3] = eps, sig, RELOAD, direction
            return sig

    if branch == SKELETON:
        if ae >= eps_uf and eps > 0.0:
            st[8] = 1.0
            sig = 0.0
        elif ae > eu_skel:
            sig = d_sign(eps) * _degradation(ae, fu, fuf, eu_skel, eps_uf)
        else:
            sig = d_sign(eps) * _skeleton(ae, E, fy, a, b0, b1, b2, ey, shift)
        if ae > st[4]:
            st[4] = ae
        direction = keep_direction

    _accumulate(st, eps, eps_prev, sig, sig_prev, E, d)
    st[0], st[1], st[2], st[3] = eps, sig, branch, direction
    return sig


@_jit
def d_sign(x):
    return 1.0 if x >= 0.0 else -1.0


@_jit
def _accumulate(st, eps, eps_prev, sig, sig_prev, E, d):
    """The accumulated equivalent COMPRESSIVE plastic strain of eq. (13).

    Compressive and plastic, both: the elastic part of the increment is
    removed with dsigma/E, and only increments that shorten the fibre count.
    Tension does not undo it -- the mechanism is damage from reversed plastic
    flow, and it is why `damage_measure` is built on this and not on the
    current strain.
    """
    if d >= 0.0:
        return
    dp = (eps - eps_prev) - (sig - sig_prev) / E
    if dp < 0.0:
        st[7] += -dp
