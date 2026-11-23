"""Concrete01 -- faithful port of OpenSees Concrete01.cpp (Kent-Scott-Park).

Direct translation of Concrete01.cpp rev 1.22 (MHS 06/99) envelope/reload/unload
logic. No tensile strength, compression only. History variables per fiber:

  n_state = 5 : [CminStrain, CunloadSlope, CendStrain, Cstrain, Cstress]
Ctangent is derived as CunloadSlope initially and updated via reload/unload.

Reference: OpenSees SRC/material/uniaxial/Concrete01.cpp
"""
from __future__ import annotations
import math
import numpy as np
from .base import Material, register

DBL_EPSILON = float(np.finfo(float).eps)

# --------------------------------------------------------------------------- #
#  Compiled point loop -- see the note in steel02.py for why this exists.
#  `_update_reference` stays the readable definition; tests/test_uniaxial_numba.py
#  pins the compiled twin against it over a cyclic history.
# --------------------------------------------------------------------------- #

_POINT = None
_KERNEL = None
_COMPILED = False


def _compile():
    global _POINT, _KERNEL, _COMPILED
    if _COMPILED:
        return
    _COMPILED = True
    try:
        from numba import njit
    except Exception:
        return

    @njit(cache=True, inline="always")
    def _envelope(eps, fpc, fpcu, epsc0, epscu, Ec0):
        if eps > epsc0:
            eta = eps / epsc0
            return fpc * (2.0 * eta - eta * eta), Ec0 * (1.0 - eta)
        if eps > epscu:
            t = (fpc - fpcu) / (epsc0 - epscu)
            return fpc + t * (eps - epsc0), t
        return fpcu, 0.0

    @njit(cache=True, inline="always")
    def _unload(TminStrain, Tstress, epsc0, epscu, Ec0):
        tempStrain = TminStrain
        if tempStrain < epscu:
            tempStrain = epscu
        eta = tempStrain / epsc0
        if eta < 2.0:
            ratio = 0.145 * eta * eta + 0.13 * eta
        else:
            ratio = 0.707 * (eta - 2.0) + 0.834
        TendStrain = ratio * epsc0
        temp1 = TminStrain - TendStrain
        temp2 = Tstress / Ec0
        TunloadSlope = Ec0
        if temp1 > -DBL_EPSILON:
            TunloadSlope = Ec0
        elif temp1 <= temp2:
            TendStrain = TminStrain - temp1
            if abs(temp1) > 1e-30:
                TunloadSlope = Tstress / temp1
            else:
                TunloadSlope = Ec0
        else:
            TendStrain = TminStrain - temp2
            TunloadSlope = Ec0
        return TendStrain, TunloadSlope

    @njit(cache=True, inline="always")
    def point(eps, st, p):
        fpc, fpcu, epsc0, epscu, Ec0 = p[0], p[1], p[2], p[3], p[4]
        CminStrain = st[0]
        CunloadSlope = st[1]
        CendStrain = st[2]
        Cstrain = st[3]
        Cstress = st[4]

        if (abs(CunloadSlope) < 1e-12 and abs(CminStrain) < 1e-12
                and abs(CendStrain) < 1e-12):
            CunloadSlope = Ec0

        Tstrain = eps
        TminStrain = CminStrain
        TendStrain = CendStrain
        TunloadSlope = CunloadSlope
        Tstress = Cstress

        if abs(Tstrain - Cstrain) < DBL_EPSILON:
            return Tstress

        if Tstrain > 0.0:
            Tstress = 0.0
        else:
            TunloadSlope = CunloadSlope
            tempStress = Cstress + TunloadSlope * Tstrain - TunloadSlope * Cstrain
            if Tstrain < Cstrain:
                if Tstrain <= TminStrain:
                    TminStrain = Tstrain
                    Tstress, _t = _envelope(Tstrain, fpc, fpcu, epsc0, epscu, Ec0)
                    TendStrain, TunloadSlope = _unload(TminStrain, Tstress,
                                                       epsc0, epscu, Ec0)
                elif Tstrain <= TendStrain:
                    Tstress = TunloadSlope * (Tstrain - TendStrain)
                else:
                    Tstress = 0.0
                if tempStress > Tstress:
                    Tstress = tempStress
            elif tempStress <= 0.0:
                Tstress = tempStress
            else:
                Tstress = 0.0

        st[0] = TminStrain
        st[1] = TunloadSlope
        st[2] = TendStrain
        st[3] = Tstrain
        st[4] = Tstress
        return Tstress

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
    caveat="OpenSees Concrete01 Kent-Scott-Park (no tension), faithful port.",
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

    # ---- helpers matching C++ ----
    def _envelope(self, Tstrain):
        if Tstrain > self.epsc0:
            eta = Tstrain / self.epsc0
            Tstress = self.fpc * (2 * eta - eta * eta)
            Ttangent = self.Ec0 * (1.0 - eta)
        elif Tstrain > self.epscu:
            Ttangent = (self.fpc - self.fpcu) / (self.epsc0 - self.epscu)
            Tstress = self.fpc + Ttangent * (Tstrain - self.epsc0)
        else:
            Tstress = self.fpcu
            Ttangent = 0.0
        return Tstress, Ttangent

    def _unload(self, TminStrain, Tstress):
        tempStrain = TminStrain
        if tempStrain < self.epscu:
            tempStrain = self.epscu
        eta = tempStrain / self.epsc0
        if eta < 2.0:
            ratio = 0.145 * eta * eta + 0.13 * eta
        else:
            ratio = 0.707 * (eta - 2.0) + 0.834
        TendStrain = ratio * self.epsc0
        temp1 = TminStrain - TendStrain
        temp2 = Tstress / self.Ec0
        TunloadSlope = self.Ec0
        # temp1 should always be negative
        if temp1 > -DBL_EPSILON:
            TunloadSlope = self.Ec0
        elif temp1 <= temp2:
            TendStrain = TminStrain - temp1
            TunloadSlope = Tstress / temp1 if abs(temp1) > 1e-30 else self.Ec0
        else:
            TendStrain = TminStrain - temp2
            TunloadSlope = self.Ec0
        return TendStrain, TunloadSlope

    def _reload(self, Tstrain, TminStrain, TendStrain, TunloadSlope, CminStrain, CendStrain, CunloadSlope):
        # returns Tstress, Ttangent, TminStrain, TendStrain, TunloadSlope
        if Tstrain <= TminStrain:
            TminStrain = Tstrain
            Tstress, Ttangent = self._envelope(Tstrain)
            TendStrain, TunloadSlope = self._unload(TminStrain, Tstress)
        elif Tstrain <= TendStrain:
            Ttangent = TunloadSlope
            Tstress = Ttangent * (Tstrain - TendStrain)
        else:
            Tstress = 0.0
            Ttangent = 0.0
        return Tstress, Ttangent, TminStrain, TendStrain, TunloadSlope

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
        """Readable definition of the law; the compiled kernel mirrors it."""
        for i, eps in enumerate(strain[:, 0]):
            eps_f = float(eps)
            CminStrain = float(state[i, 0])
            CunloadSlope = float(state[i, 1])
            CendStrain = float(state[i, 2])
            Cstrain = float(state[i, 3])
            Cstress = float(state[i, 4])

            # initial state fix: first call CunloadSlope is 0
            if abs(CunloadSlope) < 1e-12 and abs(CminStrain) < 1e-12 and abs(CendStrain) < 1e-12:
                CunloadSlope = self.Ec0

            Tstrain = eps_f
            TminStrain = CminStrain
            TendStrain = CendStrain
            TunloadSlope = CunloadSlope
            Tstress = Cstress
            # Ctangent not stored, use CunloadSlope as initial
            Ttangent = CunloadSlope

            dStrain = Tstrain - Cstrain
            if abs(dStrain) < DBL_EPSILON:
                out[i, 0] = Tstress
                # no state change
                continue

            if Tstrain > 0.0:
                Tstress = 0.0
                Ttangent = 0.0
            else:
                TunloadSlope = CunloadSlope
                tempStress = Cstress + TunloadSlope * Tstrain - TunloadSlope * Cstrain
                if Tstrain < Cstrain:  # further into compression
                    TminStrain = CminStrain
                    TendStrain = CendStrain
                    Tstress, Ttangent, TminStrain, TendStrain, TunloadSlope = self._reload(
                        Tstrain, TminStrain, TendStrain, TunloadSlope, CminStrain, CendStrain, CunloadSlope
                    )
                    if tempStress > Tstress:
                        Tstress = tempStress
                        Ttangent = TunloadSlope
                elif tempStress <= 0.0:
                    Tstress = tempStress
                    Ttangent = TunloadSlope
                else:
                    Tstress = 0.0
                    Ttangent = 0.0

            # commit
            state[i, 0] = TminStrain
            state[i, 1] = TunloadSlope
            state[i, 2] = TendStrain
            state[i, 3] = Tstrain
            state[i, 4] = Tstress
            out[i, 0] = Tstress
        return out

    def output_fields(self, state):
        return {}

    def numba_point_fn(self):
        return None

    def init_state(self, npts: int, dtype=np.float64):
        s = np.zeros((npts, self.n_state), dtype=dtype)
        s[:, 1] = self.Ec0  # CunloadSlope
        return s
