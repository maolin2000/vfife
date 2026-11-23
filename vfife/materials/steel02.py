"""Steel02 (Giuffre-Menegotto-Pinto) reinforcing steel — independent material.

Parameters follow OpenSees Steel02:
  Fy, E0, b, R0, cR1, cR2, a1, a2, a3, a4
plus rho, nu.  Aliases yield_strength/E are accepted for backward compat
with the retired rebar_gmp (removed in v0.1: it was this law, uncompiled).

The law is OpenSees `Steel02::setTrialStrain`, line for line, since v1.0
(reversal targets on the hardening asymptotes, R = R0(1 - cR1 xi/(cR2 + xi)),
isotropic shift a1..a4). It agrees with a transcription of the OpenSees source
to the last bit over a 3000-step random history (tests/test_steel02.py).

State layout (n_state = 13):
  [0] eps_prev            [7]  eps_max -- peak tensile strain, for `rupture_strain`
  [1] sig_prev            [8]  sig_target  (OpenSees sigs0)
  [2] eps_rev  (epsr)     [9]  eps_max_os  (epsmax: largest reversal strain)
  [3] sig_rev  (sigr)     [10] eps_min_os  (epsmin)
  [4] direction (kon: +1 loading, -1 unloading, 0 virgin)
  [5] eps_target (epss0)  [11] eps_pl      (epspl)
  [6] hysteretic_energy   [12] work        (the integral of sig d eps)
"""
from __future__ import annotations
import numpy as np
from .base import Material, register

# --------------------------------------------------------------------------- #
#  Compiled point loop
#
#  The reference implementation below is a Python loop over fibre points, and
#  it costs about 4 us per point -- two orders of magnitude more than the
#  arithmetic in it, because that is Python overhead, not work. Batching does
#  not help: the cost per point is flat from 1 point to 4000. A fibre beam
#  evaluates its internal force twice per relaxation iteration and thousands of
#  times per load stage, so this loop dominates.
#
#  Compiled, the same loop runs at ~68 ns per point (58x at 480 points, the
#  size one internal-force evaluation of cases/case_11 needs). The kernel takes
#  plain scalars rather than closing over `self` so that cache=True works and
#  one compilation is shared by every instance.
#
#  `_update_reference` stays as the readable definition of the law and as the
#  thing tests/test_uniaxial_numba.py pins the compiled path against over a
#  cyclic history -- state is path dependent, so agreement on one call would
#  prove nothing.
# --------------------------------------------------------------------------- #

_POINT = None
_KERNEL = None
_COMPILED = False

# State columns (n_state = 13). 0-7 keep the names and meanings they had, so
# `eps_rev`, `sig_rev`, `hysteretic_energy` and `eps_max` read as before;
# 8-12 are OpenSees Steel02's own history variables.
EPS, SIG, EPSR, SIGR, KON, EPSS0, HYST, EPSMAXT, SIGS0, EPSMAX, EPSMIN, EPSPL, WORK = range(13)


def _point_py(eps_section, st, p):
    """OpenSees Steel02 (Filippou, Popov & Bertero's Menegotto-Pinto), one
    point, committed at every call.

    It follows `Steel02::setTrialStrain` line for line. Until v1.0 this law
    was a variant that aimed each new branch at +-Fy_eff through a "yield"
    strain placed so that the CURVE passed through Fy, and that forgot the
    hardening envelope at every reversal: a 2e-4 wobble at 2% brought the
    stress at 3% from 469.8 to 440.3 MPa (OpenSees: 456.0 with or without the
    wobble), fifty small cycles walked it down to Fy, and a closed loop on a
    branch created energy. Its R was R0 - cR1 xi/(cR2+xi) with OpenSees'
    cR1 = 0.925, so the Bauschinger rounding hardly developed (R stayed near
    19 where OpenSees falls toward 1.5), and a3/a4 were never read.

    Here, as in OpenSees:
      * each reversal aims the new branch at the intersection of the elastic
        line from the reversal point with the hardening asymptote
        Fy + b E (eps - eps_y) of the other side, shifted isotropically by
        a1/a2 (compression) and a3/a4 (tension);
      * R = R0 (1 - cR1 xi / (cR2 + xi)), with xi the plastic excursion of
        the last branch in units of eps_y, fixed for the branch;
      * the envelope stays under its asymptote (the 3.45% overshoot of
        v0.4 is gone).
    `p` = [E, Fy, b, R0, cR1, cR2, a1, a2, strain_scale, rupture, a3, a4].
    """
    E, Fy, b, R0, cR1, cR2, a1, a2 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
    a3, a4 = p[10], p[11]
    # the bar's own strain, after the anchorage has taken its share
    eps = eps_section * p[8]
    rup = p[9]
    # rupture against the last ACCEPTED peak, before this call raises it
    if rup > 0.0 and st[EPSMAXT] >= rup:
        st[EPS] = eps
        st[SIG] = 0.0
        return 0.0
    if eps > st[EPSMAXT]:
        st[EPSMAXT] = eps
    Esh = b * E
    epsy = Fy / E
    eps_p = st[EPS]
    sig_p = st[SIG]
    deps = eps - eps_p
    kon = st[KON]
    if b >= 1.0:                                  # no knee at all: Hooke's law
        sig = E * eps
        st[WORK] += 0.5 * (sig + sig_p) * deps
        st[HYST] = st[WORK] - 0.5 * sig * sig / E
        st[EPS] = eps
        st[SIG] = sig
        return sig
    if kon == 0.0:
        if abs(deps) < 2.2e-15:
            return sig_p                         # nothing has happened yet
        st[EPSMAX] = epsy
        st[EPSMIN] = -epsy
        if deps < 0.0:
            kon = -1.0
            st[EPSS0] = -epsy
            st[SIGS0] = -Fy
            st[EPSPL] = -epsy
        else:
            kon = 1.0
            st[EPSS0] = epsy
            st[SIGS0] = Fy
            st[EPSPL] = epsy
    if kon < 0.0 and deps > 0.0:
        # a reversal from compression to tension
        kon = 1.0
        st[EPSR] = eps_p
        st[SIGR] = sig_p
        if eps_p < st[EPSMIN]:
            st[EPSMIN] = eps_p
        d1 = (st[EPSMAX] - st[EPSMIN]) / (2.0 * (a4 * epsy))
        shft = 1.0 + a3 * d1 ** 0.8
        st[EPSS0] = (Fy * shft - Esh * epsy * shft - sig_p + E * eps_p) / (E - Esh)
        st[SIGS0] = Fy * shft + Esh * (st[EPSS0] - epsy * shft)
        st[EPSPL] = st[EPSMAX]
    elif kon > 0.0 and deps < 0.0:
        # a reversal from tension to compression
        kon = -1.0
        st[EPSR] = eps_p
        st[SIGR] = sig_p
        if eps_p > st[EPSMAX]:
            st[EPSMAX] = eps_p
        d1 = (st[EPSMAX] - st[EPSMIN]) / (2.0 * (a2 * epsy))
        shft = 1.0 + a1 * d1 ** 0.8
        st[EPSS0] = (-Fy * shft + Esh * epsy * shft - sig_p + E * eps_p) / (E - Esh)
        st[SIGS0] = -Fy * shft + Esh * (st[EPSS0] + epsy * shft)
        st[EPSPL] = st[EPSMIN]
    st[KON] = kon
    xi = abs((st[EPSPL] - st[EPSS0]) / epsy)
    R = R0 * (1.0 - (cR1 * xi) / (cR2 + xi))
    den = st[EPSS0] - st[EPSR]
    epsrat = (eps - st[EPSR]) / den if den != 0.0 else 0.0
    dum1 = 1.0 + abs(epsrat) ** R
    dum2 = dum1 ** (1.0 / R)
    sig = b * epsrat + (1.0 - b) * epsrat / dum2
    sig = sig * (st[SIGS0] - st[SIGR]) + st[SIGR]
    # dissipated energy: the work done less what is stored elastically
    st[WORK] += 0.5 * (sig + sig_p) * deps
    st[HYST] = st[WORK] - 0.5 * sig * sig / E
    st[EPS] = eps
    st[SIG] = sig
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


@register("steel02", kinematics=("uniaxial",), backends=("numpy", "numba"),
          verification="constitutive_path",
          output_fields=("axial_stress", "axial_strain", "eps_rev", "sig_rev",
                         "hysteretic_energy", "eps_max"))
class Steel02(Material):
    n_state = 13
    state_names = {
        "eps_prev": slice(0, 1), "sig_prev": slice(1, 2),
        "eps_rev": slice(2, 3), "sig_rev": slice(3, 4),
        "direction": slice(4, 5), "eps_target": slice(5, 6),
        "hysteretic_energy": slice(6, 7), "eps_max": slice(7, 8),
        "sig_target": slice(8, 9), "eps_max_os": slice(9, 10),
        "eps_min_os": slice(10, 11), "eps_pl": slice(11, 12), "work": slice(12, 13),
    }
    uniaxial = True

    def __init__(self, E=None, E0=None, rho=7850, Fy=None, yield_strength=None,
                 b=0.01, R0=20.0, cR1=0.925, cR2=0.15,
                 a1=0.0, a2=1.0, a3=0.0, a4=1.0,
                 nu=0.3, strain_scale=1.0, rupture_strain=None, **_ignored):
        E_val = float(E if E is not None else E0 if E0 is not None else 200e9)
        Fy_val = float(Fy if Fy is not None else yield_strength if yield_strength is not None else 400e6)
        self.E, self.rho, self.nu = E_val, float(rho), float(nu)
        self.Fy = Fy_val
        self.b = float(b)
        self.R0, self.cR1, self.cR2 = float(R0), float(cR1), float(cR2)
        self.a1, self.a2, self.a3, self.a4 = float(a1), float(a2), float(a3), float(a4)
        # Strain penetration. A bar anchored into a joint keeps straining for a
        # length Lsp beyond the member face, so the elongation a section
        # curvature imposes is shared between the bar inside the element and
        # the slip at the anchorage. With that length smeared, the bar sees
        # Le/(Le + Lsp) of the section strain: softer, and yielding later.
        # 1.0 is a perfectly bonded bar, which is what every case had before.
        #
        # This is a PARAMETER of steel02 rather than a separate slip material
        # on purpose. The fused kernel inlines one rebar law; a second law type
        # would push the whole model back to the NumPy path, so the one thing
        # that must not change is the law.
        self.strain_scale = float(strain_scale)
        if not 0.0 < self.strain_scale <= 1.0:
            raise ValueError("steel02 strain_scale must lie in (0, 1]")
        # RUPTURE. THE MEASURE IS THE PEAK *TOTAL TENSILE* STRAIN THE POINT HAS
        # SEEN, AND THAT IS NOT WHAT `rebar.rupture_strain` MEASURES.
        #
        #   `rebar` latches on `alpha`, its accumulated plastic strain, which
        #   grows on every reversal -- the right measure for a column bar being
        #   cycled, and close to low-cycle fatigue. This law has no plastic
        #   strain in its state (Menegotto-Pinto rounds the knee; there is no
        #   yield surface to accumulate against), so the same measure is not
        #   available without inventing one.
        #
        #   What IS available, and what a user actually holds, is the ultimate
        #   strain from a tension test -- A_gt on a mill certificate, 3.5% for
        #   1x7-15.2-1860 strand. This law takes that number literally: the
        #   point is cut when its own strain first reaches it. For a tendon
        #   pulled monotonically to failure, which is the case this was added
        #   for, the two measures agree; for a bar being cycled they do not,
        #   and `rebar` or `steel_cyclic` is the honest choice there. THE KEY
        #   IS SPELT THE SAME ON PURPOSE (one name in the case file) AND THE
        #   MEASURE DIFFERS ON PURPOSE (this law has no other).
        #
        # Compression cannot trigger it: `eps_max` starts at zero and only ever
        # takes a maximum, so it is monotone and the latch is irreversible by
        # construction -- no extra state column for "has ruptured".
        #
        # WHEN IT IS READ: against the peak strain of the last
        # ACCEPTED state, before the current call raises it -- so the call
        # that crosses the limit still carries, and the point is cut from the
        # next one. A staged relaxation re-drives every point from the
        # converged state on every iteration; read after the update, a
        # transient overshoot of the TRIAL strain cut the fibre inside the
        # iteration and the level converged on the broken branch although the
        # intact one was in equilibrium too. On a corroded strand (Li's A7)
        # that broke the strand at 790 MPa with 40 mm elements and 928 MPa
        # with 80 mm ones -- which looked like mesh dependence and was read as
        # such in v0.4. Deferred, both meshes reach the rupture stress at
        # the same moment. In an explicit time history it is one time step.
        #
        # It IS a finite jump in the constitutive function, which is the shape
        # v0.1 lost 8 000 kinetic resets to and the reason `steel_cyclic`
        # exists. Use that law when the drop matters as a process; use this
        # when the fibre simply has to stop carrying.
        self.eps_rupture = (None if rupture_strain is None
                            else float(rupture_strain))
        if self.eps_rupture is not None and self.eps_rupture <= 0.0:
            raise ValueError("steel02 rupture_strain must be positive")
        if self.E <= 0 or self.rho <= 0 or self.Fy <= 0:
            raise ValueError("steel02 E, rho and Fy must be positive")
        if not 0.0 <= self.b <= 1.0:
            raise ValueError("steel02 b must lie in [0, 1]")
        if self.R0 <= 0:
            raise ValueError("steel02 R0 must be positive")
        if not (0.0 <= self.cR1 < 1.0 and self.cR2 > 0.0):
            # R = R0 (1 - cR1 xi/(cR2 + xi)) must stay positive
            raise ValueError("steel02 cR1 must lie in [0, 1) and cR2 be positive (OpenSees: 0.925, 0.15)")
        if self.a2 <= 0.0 or self.a4 <= 0.0:
            raise ValueError("steel02 a2 and a4 must be positive (they divide the plastic excursion)")
        self.wave_speed = float(np.sqrt(self.E / self.rho))
        self.D0 = np.diag([self.E, 0.0, 0.0, 0.0, 0.0, 0.0])
        # For backward compat expose sy
        self.sy = self.Fy
        self.R = self.R0

    def describe(self) -> str:
        return (f"    Steel02 Fy={self.Fy/1e6:.0f}MPa E={self.E/1e9:.1f}GPa b={self.b:.4g} "
                f"R0={self.R0:.4g} a1={self.a1:.3g} a3={self.a3:.3g}"
                + ("" if self.eps_rupture is None
                   else f", ruptures at eps = {self.eps_rupture:.4g} (peak tensile)"))

    def update(self, strain, state, dt, params=None):
        """`params` overrides this instance's own, for one call.

        The fibre beam needs it: strain penetration makes the anchorage share
        of the deformation an ELEMENT property, and one material object cannot
        carry a different value per element. The kernel already reads
        `spar[e, f]`; this lets the NumPy reference read the same rows, so the
        two paths stay comparable instead of the reference quietly being a
        model of something else.
        """
        out = np.zeros_like(strain)
        p = self.numba_params() if params is None else params
        fn = _kernel()
        if fn is not None:
            return fn(np.ascontiguousarray(strain, dtype=np.float64), state, out, p)
        return self._update_reference(strain, state, out, p)

    def damage_class(self, state):
        """Yielded once the point carries a plastic residual: the strain it
        would keep at zero stress, eps - sig/E, past a tenth of the yield
        strain (the Menegotto-Pinto curve rounds the knee, so a strict zero
        would call every fibre yielded)."""
        resid = np.abs(state[:, 0] - state[:, 1] / self.E)
        out = np.zeros(len(state), dtype=np.int8)
        out[resid > 0.1 * self.Fy / self.E] = self.DAMAGE_YIELDED
        f = self.failure(state)
        if f is not None:
            out[f > 0.5] = self.DAMAGE_FAILED
        return out

    def failure(self, state):
        """1 where the point has ruptured.

        Monotone because `eps_max` is: it only ever takes a maximum, so a
        ruptured point cannot come back when the beam unloads. The element
        above counts these rather than inferring rupture from a zero stress,
        which an unloaded elastic fibre also reports.
        """
        if self.eps_rupture is None:
            return None
        return (state[:, 7] >= self.eps_rupture).astype(float)

    def numba_params(self):
        """Law parameters as one array, for the compiled point function."""
        return np.array([self.E, self.Fy, self.b, self.R0,
                         self.cR1, self.cR2, self.a1, self.a2,
                         self.strain_scale,
                         0.0 if self.eps_rupture is None else self.eps_rupture,
                         self.a3, self.a4],
                        dtype=np.float64)

    def numba_uniaxial_fn(self):
        """Scalar twin of the law, for inlining into a fibre-beam kernel."""
        _compile()
        return _POINT

    def _update_reference(self, strain, state, out, params=None):
        """The law run as plain Python: the same function the kernel compiles."""
        p = self.numba_params() if params is None else np.asarray(params, dtype=float)
        for i in range(strain.shape[0]):
            out[i, 0] = _point_py(float(strain[i, 0]), state[i], p)
        return out

    def output_fields(self, state):
        return {
            "eps_rev": state[:, 2].copy(),
            "sig_rev": state[:, 3].copy(),
            "hysteretic_energy": state[:, 6].copy(),
            "eps_max": state[:, 7].copy(),
        }

    def numba_point_fn(self):
        return None
