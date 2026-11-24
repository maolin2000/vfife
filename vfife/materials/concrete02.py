"""Concrete02 -- Yassin's concrete with linear tension softening (the law
OpenSees users know by this name and parameters).

Written from the description of the model (Mohd Yassin 1994; Sze & Filippou
1994); see `_point_py`. Parameters: fpc, epsc0, fcu, epscu, rat, ft, Ets.
History (n_state = 5): ecminP, deptP, epsP, sigP, eP.
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


def _point_py(eps_section, st, p):
    """One point of Yassin's concrete with tension softening, committed at
    every call. Compression negative.

    Written from the description of the model (Yassin 1994; Sze & Filippou):

      * compression envelope: fc (2 r - r^2), r = eps/epsc0, to epsc0; a
        straight line to (epscu, fcu); fcu beyond;
      * the unloading lines all aim at one point R, where the initial
        elastic line meets the line through (epscu, fcu) of slope rat Ec0:
            eps_R = (fcu - rat Ec0 epscu) / (Ec0 (1 - rat)),  sig_R = Ec0 eps_R;
        from the furthest point (eps_min, sig_min) the reloading line runs
        with slope Er = (sig_min - sig_R)/(eps_min - eps_R) to zero stress at
        eps_0 = eps_min - sig_min/Er, the unloading line through eps_0 has
        half that slope, and between them the point moves elastically with
        Ec0 = 2 fc/epsc0 -- so a small cycle inside them is elastic;
      * tension: from eps_0, elastic with Ec0 to ft, then softening with
        slope Ets to zero; unloading and reloading in tension go along the
        secant to eps_0 from the furthest tensile point.

    Which of the two lines bounds which way, and the origin of the tension
    branch, were fixed by comparison with the OpenSees program of the same
    name (tests/test_opensees_reference.py).

    State: [eps_min, tensile excursion, eps, sig, tangent].
    `p` = [fc, epsc0, fcu, epscu, ft, Ets, rat, strain_scale].
    """
    fc, epsc0, fcu, epscu, ft, Ets, rat = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    eps = eps_section * p[7]
    if eps == st[2]:                                  # nothing moved: stress and tangent stay
        return st[3]
    Ec0 = 2.0 * fc / epsc0
    if eps < st[0]:                                   # past the furthest point: the envelope
        st[0] = eps
        if eps >= epsc0:
            r = eps / epsc0
            sig = fc * r * (2.0 - r)
            tan = Ec0 * (1.0 - r)
        elif eps > epscu:
            tan = (fcu - fc) / (epscu - epsc0)
            sig = fc + tan * (eps - epsc0)
        else:
            sig = fcu
            tan = 1.0e-10
    else:
        em = st[0]
        if em < 0.0:
            if em >= epsc0:
                r = em / epsc0
                sm = fc * r * (2.0 - r)
            elif em > epscu:
                sm = fc + (fcu - fc) * (em - epsc0) / (epscu - epsc0)
            else:
                sm = fcu
            eps_R = (fcu - rat * Ec0 * epscu) / (Ec0 * (1.0 - rat))
            Er = (sm - Ec0 * eps_R) / (em - eps_R)
            eps_0 = em - sm / Er if Er > 1.0e-12 * Ec0 else 0.0
        else:
            Er = Ec0
            eps_0 = 0.0
        if Er <= 1.0e-12 * Ec0:
            # rat = 0 and crushed: the lines are flat, the point carries nothing
            sig = 0.0
            tan = 1.0e-10
        elif eps <= eps_0:                              # between the reloading and unloading lines
            sig = st[3] + Ec0 * (eps - st[2])
            tan = Ec0
            lo = Er * (eps - eps_0)
            hi = 0.5 * Er * (eps - eps_0)
            if sig < lo:
                sig = lo
                tan = Er
            if sig > hi:
                sig = hi
                tan = 0.5 * Er
        else:                                         # tension, from eps_0
            et = eps - eps_0
            e1 = ft / Ec0
            if et > st[1]:
                st[1] = et
                if et <= e1:
                    sig = Ec0 * et
                    tan = Ec0
                else:
                    sig = ft - Ets * (et - e1)
                    tan = -Ets
                    if sig <= 0.0:
                        sig = 0.0
                        tan = 1.0e-10
            else:
                d = st[1]
                if d <= e1:
                    sd = Ec0 * d
                else:
                    sd = ft - Ets * (d - e1)
                    if sd < 0.0:
                        sd = 0.0
                tan = sd / d if d > 0.0 else Ec0
                sig = tan * et
    st[2] = eps
    st[3] = sig
    st[4] = tan
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
    "concrete02",
    kinematics=("uniaxial",),
    backends=("numpy", "numba"),
    verification="constitutive_path",
    output_fields=("axial_stress", "axial_strain"),
    nonlinear=True,
    caveat="Yassin concrete with linear tension softening.",
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

        # defaults: rat 0.1, ft = fc/10, Ets = (fc/epsc0)/10
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
        s[:, 4] = 2.0 * self.fc / self.epsc0
        return s
