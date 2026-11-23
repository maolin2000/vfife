"""Concrete02 -- faithful port of OpenSees Concrete02.cpp (Yassin 1993).

Based on FEDEAS Concr2.f, Mohd Yassin 1993, Sze & Filippou 1994.
Parameters: fpc, epsc0, fcu, epscu, rat, ft, Ets
History: ecminP, deptP, epsP, sigP, eP (n_state=5)

Reference: OpenSees SRC/material/uniaxial/Concrete02.cpp rev 1.3
"""
from __future__ import annotations
import math
import numpy as np
from .base import Material, register


# --------------------------------------------------------------------------- #
#  Compiled point loop -- see the note in steel02.py; same reasoning, same
#  structure. The Yassin envelopes are inlined as njit helpers because a method
#  call per fibre point is most of what makes the reference implementation slow.
#  `_update_reference` remains the readable definition and the thing
#  tests/test_uniaxial_numba.py pins this against over a cyclic history.
# --------------------------------------------------------------------------- #

_EPS = 2.220446049250313e-16
_POINT = None
_KERNEL = None
_COMPILED = False


def _compile():
    """Scalar point law + the batch loop that calls it. See steel02.py."""
    global _POINT, _KERNEL, _COMPILED
    if _COMPILED:
        return
    _COMPILED = True
    try:
        from numba import njit
    except Exception:
        return

    @njit(cache=True, inline="always")
    def _tens(epsc, fc, epsc0, ft, Ets):
        Ec0 = 2.0 * fc / epsc0
        eps0 = ft / Ec0
        epsu = ft * (1.0 / Ets + 1.0 / Ec0)
        if epsc <= eps0:
            return epsc * Ec0, Ec0
        if epsc <= epsu:
            return ft - Ets * (epsc - eps0), -Ets
        return 0.0, 1.0e-10

    @njit(cache=True, inline="always")
    def _compr(epsc, fc, epsc0, fcu, epscu):
        Ec0 = 2.0 * fc / epsc0
        r = epsc / epsc0
        if epsc >= epsc0:
            return fc * r * (2.0 - r), Ec0 * (1.0 - r)
        if epsc > epscu:
            return ((fcu - fc) * (epsc - epsc0) / (epscu - epsc0) + fc,
                    (fcu - fc) / (epscu - epsc0))
        return fcu, 1.0e-10

    @njit(cache=True, inline="always")
    def point(eps_section, st, p):
        fc, epsc0, fcu, epscu, ft, Ets, rat = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
        # The fibre's OWN (instantaneous) strain, after the free strains have
        # been taken out of the section's. `p[7]` is 1 for a fibre with no
        # time dependence, which is every fibre in every case before v0.4.
        eps_trial = eps_section * p[7]
        Ec0 = 2.0 * fc / epsc0
        ecminP = st[0]
        deptP = st[1]
        epsP = st[2]
        sigP = st[3]
        eP = st[4]
        if (abs(eP) < 1e-12 and abs(ecminP) < 1e-12 and abs(deptP) < 1e-12
                and abs(epsP) < 1e-12 and abs(sigP) < 1e-12):
            ecminP = 0.0
            deptP = 0.0

        deps = eps_trial - epsP
        if abs(deps) < _EPS:
            return sigP

        eps = eps_trial
        ecmin = ecminP
        dept = deptP

        if eps < ecmin:
            sig, e = _compr(eps, fc, epsc0, fcu, epscu)
            ecmin = eps
        else:
            epsr = (fcu - rat * Ec0 * epscu) / (Ec0 * (1.0 - rat))
            sigmr = Ec0 * epsr
            sigmm, _unused = _compr(ecmin, fc, epsc0, fcu, epscu)
            if abs(ecmin - epsr) > 1e-12:
                er = (sigmm - sigmr) / (ecmin - epsr)
            else:
                er = Ec0
            if abs(er) > 1e-12:
                ept = ecmin - sigmm / er
            else:
                ept = ecmin
            if eps <= ept:
                sigmin = sigmm + er * (eps - ecmin)
                sigmax = er * 0.5 * (eps - ept)
                sig = sigP + Ec0 * deps
                e = Ec0
                if sig <= sigmin:
                    sig = sigmin
                    e = er
                if sig >= sigmax:
                    sig = sigmax
                    e = 0.5 * er
            else:
                epn = ept + dept
                if eps <= epn:
                    sicn, _u2 = _tens(dept, fc, epsc0, ft, Ets)
                    if dept != 0.0:
                        e = sicn / dept
                    else:
                        e = Ec0
                    sig = e * (eps - ept)
                else:
                    sig, e = _tens(eps - ept, fc, epsc0, ft, Ets)
                    dept = eps - ept

        st[0] = ecmin
        st[1] = dept
        st[2] = eps
        st[3] = sig
        st[4] = e
        return sig

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
    "concrete02",
    kinematics=("uniaxial",),
    backends=("numpy", "numba"),
    verification="constitutive_path",
    output_fields=("axial_stress", "axial_strain"),
    nonlinear=True,
    caveat="OpenSees Concrete02 Yassin damage with tension, faithful port.",
)
class Concrete02(Material):
    n_state = 5
    state_names = {
        "ecminP": slice(0, 1),
        "deptP": slice(1, 2),
        "epsP": slice(2, 3),
        "sigP": slice(3, 4),
        "eP": slice(4, 5),
    }
    uniaxial = True

    def __init__(self, fpc=None, epsc0=None, fcu=None, epscu=None,
                 rat=None, ft=None, Ets=None,
                 fc=None, eps_c0=None, fcu2=None, eps_cu=None,
                 rho=2400, nu=0.2, strain_scale=1.0, fpcu=None, **ignored):
        # alias; `fpcu` is OpenSees's name, which Concrete01 here takes and
        # this law silently dropped until unknown keys were refused
        # -- the default 0.2 fpc happened to be what the
        # tests wrote
        if fcu is None and fpcu is not None:
            fcu = fpcu
        if fpc is None:
            fpc = fc if fc is not None else 30e6
        if epsc0 is None:
            epsc0 = eps_c0 if eps_c0 is not None else 0.002
        if fcu is None:
            fcu = fcu2 if fcu2 is not None else (0.2 * abs(float(fpc)))
        if epscu is None:
            epscu = eps_cu if eps_cu is not None else 0.0035

        self.fc_input = float(fpc)
        self.epsc0_input = float(epsc0)
        self.fcu_input = float(fcu)
        self.epscu_input = float(epscu)
        self.rat_input = float(rat) if rat is not None else None
        self.ft_input = float(ft) if ft is not None else None
        self.Ets_input = float(Ets) if Ets is not None else None
        self.rho = float(rho)
        self.nu = float(nu)
        # THE FIBRE'S SHARE OF THE SECTION'S STRAIN, the exact
        # counterpart of `steel02.strain_scale` -- same mechanism, different
        # reason. There it is the anchorage taking part of a bar's elongation;
        # here it is CREEP: the law responds to the INSTANTANEOUS strain, and
        # when a station must also carry the creep of the interval it is
        # arriving in, that strain is
        #
        #     eps_el = (eps_section - eps_free) / (1 + phi_cur)
        #
        # with `eps_free` in the fibre's imposed-strain slot and this factor
        # the `1 / (1 + phi_cur)`. Writing the station's own creep this way
        # rather than adding it to the imposed strain is what makes the step
        # IMPLICIT; the explicit version is not merely first order but
        # marginally unstable on a held strain (v0.4 measured it).
        #
        # 1.0 is a fibre with no time dependence, which is every fibre in
        # every case before v0.4, and the arithmetic is then bit-identical.
        self.strain_scale = float(strain_scale)
        if not 0.0 < self.strain_scale <= 1.0:
            raise ValueError("concrete02 strain_scale must lie in (0, 1]")

        self.fc = float(fpc)
        self.epsc0 = float(epsc0)
        self.fcu = float(fcu)
        self.epscu = float(epscu)

        # defaults as in C++ second constructor
        if self.rat_input is None:
            self.rat = 0.1
        else:
            self.rat = float(self.rat_input)
        if not 0.0 <= self.rat < 1.0:
            # rat = 1 divided by zero on the first update
            raise ValueError(f"concrete02 rat = {self.rat:g} must lie in [0, 1): the unloading slope "
                             "at epscu as a fraction of the initial one")
        if abs(float(self.epscu)) <= abs(float(self.epsc0 if hasattr(self, "epsc0") else self.epsc0_input)):
            raise ValueError(f"concrete02 epscu = {self.epscu:g} must be beyond epsc0 "
                             "(the crushing strain comes after the peak)")
        if self.ft_input is None:
            # ft = 0.1*fc (fc negative, ft positive)
            self.ft = 0.1 * abs(self.fc)
        else:
            self.ft = abs(float(self.ft_input))
        # Ets default 0.1*fc/epsc0  (fc negative, epsc0 negative => positive, but code keeps positive)
        if self.Ets_input is None:
            # 0.1*fc/epsc0 where fc/epsc0 positive (both negative)
            self.Ets = 0.1 * abs(self.fc) / abs(self.epsc0)
        else:
            self.Ets = abs(float(self.Ets_input))

        if self.fc > 0:
            self.fc = -self.fc
        if self.epsc0 > 0:
            self.epsc0 = -self.epsc0
        if self.fcu > 0:
            self.fcu = -self.fcu
        if self.epscu > 0:
            self.epscu = -self.epscu

        self.Ec0 = 2.0 * self.fc / self.epsc0  # positive
        self.wave_speed = float(math.sqrt(abs(self.Ec0) / self.rho)) if self.rho>0 else 0.0
        self.D0 = np.diag([self.Ec0, 0, 0, 0, 0, 0])
        self.E = float(self.Ec0)

    def describe(self) -> str:
        return f"    Concrete02 fpc={self.fc/1e6:.1f}MPa epsc0={self.epsc0:.4f} fcu={self.fcu/1e6:.1f}MPa epscu={self.epscu:.4f} rat={self.rat:.3g} ft={self.ft/1e6:.1f}MPa Ets={self.Ets/1e9:.2f}GPa"

    def _Tens_Envlp(self, epsc):
        Ec0 = 2.0 * self.fc / self.epsc0
        eps0 = self.ft / Ec0
        epsu = self.ft * (1.0 / self.Ets + 1.0 / Ec0)
        if epsc <= eps0:
            sigc = epsc * Ec0
            Ect = Ec0
        else:
            if epsc <= epsu:
                Ect = -self.Ets
                sigc = self.ft - self.Ets * (epsc - eps0)
            else:
                Ect = 1.0e-10
                sigc = 0.0
        return sigc, Ect

    def _Compr_Envlp(self, epsc):
        Ec0 = 2.0 * self.fc / self.epsc0
        ratLocal = epsc / self.epsc0
        if epsc >= self.epsc0:
            sigc = self.fc * ratLocal * (2.0 - ratLocal)
            Ect = Ec0 * (1.0 - ratLocal)
        else:
            if epsc > self.epscu:
                sigc = (self.fcu - self.fc) * (epsc - self.epsc0) / (self.epscu - self.epsc0) + self.fc
                Ect = (self.fcu - self.fc) / (self.epscu - self.epsc0)
            else:
                sigc = self.fcu
                Ect = 1.0e-10
        return sigc, Ect

    def update(self, strain, state, dt):
        out = np.zeros_like(strain)
        fn = _kernel()
        if fn is not None:
            return fn(np.ascontiguousarray(strain, dtype=np.float64), state, out,
                      self.numba_params())
        return self._update_reference(strain, state, out)

    def damage_class(self, state):
        """Cracked once the tensile excursion `dept` has passed the strain
        at f_t (the tension envelope is linear to there and softens after),
        crushed once `ecmin` has passed the peak strain epsc0."""
        ecmin, dept = state[:, 0], state[:, 1]
        Ec0 = 2.0 * self.fc / self.epsc0
        out = np.zeros(len(state), dtype=np.int8)
        out[dept > self.ft / Ec0] = self.DAMAGE_CRACKED
        out[ecmin < self.epsc0] = self.DAMAGE_CRUSHED
        return out

    def numba_params(self):
        """Law parameters as one array, for the compiled point function."""
        return np.array([self.fc, self.epsc0, self.fcu, self.epscu,
                         self.ft, self.Ets, self.rat, self.strain_scale],
                        dtype=np.float64)

    def numba_uniaxial_fn(self):
        """Scalar twin of the law, for inlining into a fibre-beam kernel."""
        _compile()
        return _POINT

    def _update_reference(self, strain, state, out):
        """Readable definition of the law; the compiled kernel mirrors it."""
        Ec0 = 2.0 * self.fc / self.epsc0
        for i, eps_section in enumerate(strain[:, 0]):
            eps_trial = float(eps_section) * self.strain_scale
            ecminP = float(state[i, 0])
            deptP = float(state[i, 1])
            epsP = float(state[i, 2])
            sigP = float(state[i, 3])
            eP = float(state[i, 4])
            # if uninitialized, set defaults
            if abs(eP) < 1e-12 and abs(ecminP) < 1e-12 and abs(deptP) < 1e-12 and abs(epsP) < 1e-12 and abs(sigP) < 1e-12:
                eP = Ec0
                ecminP = 0.0
                deptP = 0.0

            deps = eps_trial - epsP
            if abs(deps) < np.finfo(float).eps:
                out[i, 0] = sigP
                continue

            eps = eps_trial
            # retrieve history
            ecmin = ecminP
            dept = deptP

            if eps < ecmin:
                sig, e = self._Compr_Envlp(eps)
                ecmin = eps
            else:
                epsr = (self.fcu - self.rat * Ec0 * self.epscu) / (Ec0 * (1.0 - self.rat))
                sigmr = Ec0 * epsr
                # previous minimum stress
                sigmm, _ = self._Compr_Envlp(ecmin)
                er = (sigmm - sigmr) / (ecmin - epsr) if abs(ecmin - epsr) > 1e-12 else Ec0
                ept = ecmin - sigmm / er if abs(er) > 1e-12 else ecmin
                if eps <= ept:
                    sigmin = sigmm + er * (eps - ecmin)
                    sigmax = er * 0.5 * (eps - ept)
                    sig_trial = sigP + Ec0 * deps
                    e_trial = Ec0
                    sig = sig_trial
                    e = e_trial
                    if sig <= sigmin:
                        sig = sigmin
                        e = er
                    if sig >= sigmax:
                        sig = sigmax
                        e = 0.5 * er
                else:
                    epn = ept + dept
                    if eps <= epn:
                        # tension reloading branch
                        sicn, _ = self._Tens_Envlp(dept)
                        if dept != 0.0:
                            e = sicn / dept
                        else:
                            e = Ec0
                        sig = e * (eps - ept)
                    else:
                        epstmp = eps - ept
                        sig, e = self._Tens_Envlp(epstmp)
                        dept = eps - ept

            # commit immediately (fiber direct)
            state[i, 0] = ecmin
            state[i, 1] = dept
            state[i, 2] = eps
            state[i, 3] = sig
            state[i, 4] = e
            out[i, 0] = sig
        return out

    def output_fields(self, state):
        return {}

    def numba_point_fn(self):
        return None

    def init_state(self, npts: int, dtype=np.float64):
        s = np.zeros((npts, self.n_state), dtype=dtype)
        s[:, 4] = 2.0 * self.fc / self.epsc0
        return s
