"""Concrete01 -- Kent-Scott-Park concrete with Karsan-Jirsa unloading, no
tensile strength (the law OpenSees users know by this name and parameters).

Written from the papers; see `_point_py`. History variables per fibre
(n_state = 5): [CminStrain, CunloadSlope, CendStrain, Cstrain, Cstress].
"""
from __future__ import annotations
import math
import numpy as np
from .base import Material, register


# --------------------------------------------------------------------------- #
#  Compiled point loop -- see the note in steel02.py for why this exists.
#  `_update_reference` stays the readable definition; tests/test_uniaxial_numba.py
#  pins the compiled twin against it over a cyclic history.
# --------------------------------------------------------------------------- #

_POINT = None
_KERNEL = None
_COMPILED = False


def _point_py(eps, st, p):
    """One point of Kent-Scott-Park concrete with Karsan-Jirsa unloading,
    committed at every call; no tensile strength. Compression negative.

    Envelope (Kent & Park 1971, as modified by Scott, Park & Priestley
    1982): a parabola fpc (2 r - r^2), r = eps/epsc0, up to epsc0; a straight
    line to (epscu, fpcu); fpcu beyond.

    Unloading from the furthest point reached, (eps_min, sig_min), is a
    straight line to the plastic strain of Karsan & Jirsa (1969),

        eps_p / epsc0 = 0.145 eta^2 + 0.13 eta          eta < 2
                      = 0.707 (eta - 2) + 0.834          eta >= 2,

    eta = eps_min / epsc0, with eta held at epscu / epsc0 once the point has
    crushed, and the line never steeper than the initial slope Ec0 (near the
    origin the plastic strain above would make it so). Reloading follows the
    same line; past eps_p the point is open and carries nothing. The two
    caps were fixed by comparison with the OpenSees program of the same name
    (tests/test_opensees_reference.py).

    State: [eps_min, unloading slope, eps_p, eps, sig].
    `p` = [fpc, fpcu, epsc0, epscu, Ec0], compression quantities negative.
    """
    fpc, fpcu, epsc0, epscu = p[0], p[1], p[2], p[3]
    if eps < st[0]:                                   # past the furthest point: the envelope
        if eps >= epsc0:
            r = eps / epsc0
            sig = fpc * (2.0 * r - r * r)
        elif eps >= epscu:
            sig = fpc + (fpcu - fpc) * (eps - epsc0) / (epscu - epsc0)
        else:
            sig = fpcu
        eta = (eps if eps > epscu else epscu) / epsc0
        if eta < 2.0:
            eps_p = epsc0 * (0.145 * eta * eta + 0.13 * eta)
        else:
            eps_p = epsc0 * (0.707 * (eta - 2.0) + 0.834)
        slope = sig / (eps - eps_p)
        Ec0 = p[4]
        if slope > Ec0:                                # never stiffer than the initial slope
            slope = Ec0
            eps_p = eps - sig / Ec0
        st[0] = eps
        st[1] = slope
        st[2] = eps_p
    elif eps < st[2]:                                 # on the unloading/reloading line
        sig = st[1] * (eps - st[2])
    else:                                             # open
        sig = 0.0
    st[3] = eps
    st[4] = sig
    return sig


def _compile():
    """The same function, compiled: one definition of the law, used by the
    reference loop (as Python), the batch loop and the fused fibre kernel."""
    global _POINT, _KERNEL, _COMPILED
    if _COMPILED:
        return
    _COMPILED = True
    try:
        from numba import njit
    except Exception:
        return
    point = njit(cache=True, inline="always")(_point_py)

    @njit(cache=True)
    def _point_loop(strain, state, out, p):
        for i in range(strain.shape[0]):
            out[i, 0] = point(strain[i, 0], state[i], p)
        return out

    _POINT, _KERNEL = point, _point_loop


def _kernel():
    """The compiled batch loop, or None if numba is unavailable."""
    _compile()
    return _KERNEL



@register(
    "concrete01",
    kinematics=("uniaxial",),
    backends=("numpy", "numba"),
    verification="constitutive_path",
    output_fields=("axial_stress", "axial_strain"),
    nonlinear=True,
    caveat="Kent-Scott-Park envelope, Karsan-Jirsa unloading, no tension.",
)
class Concrete01(Material):
    n_state = 5
    state_names = {
        "CminStrain": slice(0, 1),
        "CunloadSlope": slice(1, 2),
        "CendStrain": slice(2, 3),
        "Cstrain": slice(3, 4),
        "Cstress": slice(4, 5),
    }
    uniaxial = True

    def __init__(self, fpc=None, epsc0=None, fpcu=None, epscu=None,
                 fc=None, eps_c0=None, fcu=None, eps_cu=None,
                 E=None, rho=2400, nu=0.2, **ignored):
        # alias handling: accept either OpenSees names or previous vfife names
        if fpc is None:
            fpc = fc if fc is not None else 30e6
        e0_given = epsc0 if epsc0 is not None else eps_c0
        if E is not None and e0_given is not None:
            # E is Ec0 = 2 fpc / epsc0: with epsc0 given too, E was dropped
            # without a word. Both are fine when they agree.
            implied = 2.0 * abs(float(fpc)) / abs(float(e0_given))
            if abs(float(E) - implied) > 0.01 * implied:
                raise ValueError(
                    f"concrete01 E = {float(E):.4g} disagrees with 2 fpc / epsc0 = {implied:.4g}: "
                    "the law's initial slope is fixed by fpc and epsc0, so give E or epsc0, not both")
        if epsc0 is None:
            if eps_c0 is not None:
                epsc0 = eps_c0
            elif E is not None and fpc is not None:
                # Ec0 = 2*fpc/epsc0 => epsc0 = 2*fpc/E
                epsc0 = 2.0 * abs(float(fpc)) / float(E)
            else:
                epsc0 = 0.002
        if fpcu is None:
            fpcu = fcu if fcu is not None else (0.2 * abs(float(fpc)))
        if epscu is None:
            epscu = eps_cu if eps_cu is not None else 0.0035

        self.fpc = float(fpc)
        self.epsc0 = float(epsc0)
        self.fpcu = float(fpcu)
        self.epscu = float(epscu)
        self.rho = float(rho)
        self.nu = float(nu)

        # make all concrete parameters negative (compression negative)
        if self.fpc > 0:
            self.fpc = -self.fpc
        if self.epsc0 > 0:
            self.epsc0 = -self.epsc0
        if self.fpcu > 0:
            self.fpcu = -self.fpcu
        if self.epscu > 0:
            self.epscu = -self.epscu

        if abs(self.epscu) <= abs(self.epsc0):
            # the softening branch ran backwards: a 24 MPa jump in one step
            # at epsc0
            raise ValueError(f"concrete01 epscu = {self.epscu:g} must be beyond epsc0 = {self.epsc0:g} "
                             "(the crushing strain comes after the peak)")
        self.Ec0 = 2.0 * self.fpc / self.epsc0  # positive
        self.wave_speed = float(math.sqrt(abs(self.Ec0) / self.rho)) if self.rho>0 else 0.0
        self.D0 = np.diag([self.Ec0, 0, 0, 0, 0, 0])
        # expose E for section diagnostics
        self.E = float(self.Ec0)

    def describe(self) -> str:
        return f"    Concrete01 fpc={self.fpc/1e6:.1f}MPa epsc0={self.epsc0:.4f} fpcu={self.fpcu/1e6:.1f}MPa epscu={self.epscu:.4f}"

    def update(self, strain, state, dt):
        out = np.zeros_like(strain)
        fn = _kernel()
        if fn is not None:
            return fn(np.ascontiguousarray(strain, dtype=np.float64), state, out,
                      self.numba_params())
        return self._update_reference(strain, state, out)

    def damage_class(self, state):
        """Crushed past epsc0 (the peak), cracked when the point sits open in
        tension (this law carries no tensile strength, so an open crack is a
        positive strain at zero stress), intact otherwise."""
        cmin, eps = state[:, 0], state[:, 3]
        out = np.zeros(len(state), dtype=np.int8)
        out[eps > 0.0] = self.DAMAGE_CRACKED
        out[cmin < self.epsc0] = self.DAMAGE_CRUSHED
        return out

    def numba_params(self):
        """Law parameters as one array, for the compiled point function."""
        return np.array([self.fpc, self.fpcu, self.epsc0, self.epscu,
                         self.Ec0], dtype=np.float64)

    def numba_uniaxial_fn(self):
        """Scalar twin of the law, for inlining into a fibre-beam kernel."""
        _compile()
        return _POINT

    def _update_reference(self, strain, state, out):
        """The law run as plain Python: the same function the kernel compiles."""
        p = self.numba_params()
        for i in range(strain.shape[0]):
            out[i, 0] = _point_py(float(strain[i, 0]), state[i], p)
        return out
    def output_fields(self, state):
        return {}

    def numba_point_fn(self):
        return None

    def init_state(self, npts: int, dtype=np.float64):
        s = np.zeros((npts, self.n_state), dtype=dtype)
        s[:, 1] = self.Ec0  # CunloadSlope
        return s
