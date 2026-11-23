"""Bond stress-slip for a reinforcing bar in concrete: fib Model Code 2010.

WHY THIS EXISTS
---------------
Embedded bars in this solver are perfectly bonded, and that was measured to be
the thing setting the limit of the solid RC path: a crack of opening w crossing
a bar segment of length L puts strain w/L into it with nothing to spread it, so
the drift a column survived scaled with the SEGMENT length (32.0 / 18.0 / 11.0
mm for 100 / 50 / 25 mm segments) rather than with anything physical. See
v0.1 for the controlled experiment.

`segment_length` reduced that dependence to 7% by pinning the smearing length
to a physical number. This law removes it: the transfer length comes out of
tau(s) and the bar perimeter, and the analyst does not choose it at all.

THE LAW (MC2010 §6.1.1, ribbed bars, pull-out failure)

    tau = tau_max * (s/s1)**alpha        0 <= s <= s1     ascending
    tau = tau_max                        s1 < s <= s2     plateau
    tau = tau_max - (tau_max - tau_f)
              * (s - s2)/(s3 - s2)       s2 < s <= s3     descending
    tau = tau_f                          s > s3           residual friction

and tau(-s) = -tau(s): the law is odd, because a bar pushed in and a bar pulled
out see the same ribs.

    condition   tau_max        s1      s2      s3       alpha   tau_f
    good        2.5 sqrt(fc)   1.0mm   2.0mm   c_clear  0.4     0.40 tau_max
    poor        1.25 sqrt(fc)  1.8mm   3.6mm   c_clear  0.4     0.40 tau_max

`c_clear` is the clear rib spacing; MC2010 lets it be taken as the bar spacing
in the pull-out case and this class defaults it to 10 mm, which is the usual
value for a 20 mm ribbed bar. All of it can be overridden.

THE OTHER FAILURE MODE, AND WHY ITS NUMBERS ARE NOT HARDCODED
    MC2010 gives two modes. PULL-OUT (above) is the bar shearing a cylinder of
    concrete along the ribs; it needs enough cover and enough transverse steel
    to stop the concrete splitting first. SPLITTING is the other outcome: the
    cover cracks radially and the bar loses its grip before the ribs can shear
    anything.

    Which one governs is a property of the MEMBER, not of the material: cover
    over bar diameter, bar spacing, and the transverse reinforcement crossing
    the potential splitting plane. MC2010 §6.1.1 states the condition; this
    class cannot evaluate it because it does not know the geometry, so
    `failure_mode` is an explicit choice and the default stays "pull_out" --
    changing the default would silently re-rate every existing model.

    The SHAPE of the splitting law is not in doubt and is built in:
      * no plateau -- s2 = s1, the curve turns over at the peak
      * a much shorter descent -- s3 is a fraction of the rib spacing, not a
        multiple, because a split cover unloads fast
      * no friction left when the cover is unconfined -- tau_f = 0. A pulled-out
        bar still rubs; a bar in split concrete has nothing to rub against.
      * a lower peak, scaling as (fcm/20)**0.25 rather than sqrt(fcm)

    The COEFFICIENT in front of that peak comes from MC2010 Table 6.1-1 and is
    NOT hardcoded here, because it was not verified against the code. It is
    `tau_max_coeff`, it defaults to 7.0 (unconfined) / 8.0 (with stirrups),
    and those defaults are a starting point to CHECK, not a citation. Pass
    `tau_max` directly if you have the table in front of you.

    Splitting is the unsafe direction to get wrong: it is weaker than pull-out,
    so modelling a thin-cover member as pull-out overestimates the bond.

UNITS, AND WHY THIS IS NOT A STRESS-STRAIN LAW
    The "strain" this material is handed is a SLIP in metres and the "stress"
    it returns is a bond stress in Pa. It rides the Material interface because
    the state machinery, the three backends and the equivalence tests are all
    there, but it is registered under kinematics ("bond",) so nothing can pick
    it up as a bar or a solid law by accident. The interface element supplies
    the bar perimeter and the tributary length that turn tau into a force.

CYCLIC RULE, AND ITS HONEST LIMITS
    The envelope above is monotonic. Reversals unload on a straight line of
    slope tau_max/s1 -- the secant at the end of the ascending branch -- and
    are bounded by the envelope in both directions. That is the simplest rule
    that gives a closed hysteresis loop and does not create energy.

    It is NOT Eligehausen's cyclic model: there is no frictional plateau on
    reversal, no damage from cycling, and no degradation of tau_max with the
    number of cycles. Under a few large excursions -- a pushover, a handful of
    cycles -- it is reasonable. Under a long cyclic history it will overstate
    the bond that survives, and the direction of that error is unsafe. Do not
    read a cyclic result from this law without saying so.
"""
from __future__ import annotations

import math

import numpy as np

from .base import Material, register


@register(
    "bond_ceb",
    kinematics=("bond",),
    backends=("numpy", "numba", "warp"),
    verification="constitutive_path",
    output_fields=("bond_stress", "slip", "slip_max"),
    nonlinear=True,
    caveat="MC2010 monotonic envelope with a linear unloading rule; not "
           "Eligehausen cyclic degradation -- use bond_cyclic for that, which "
           "reproduces this law bit for bit on a monotonic path.",
)
class BondCEB(Material):
    #: s_max is the largest |slip| the point has seen, and it only grows, so
    #: the envelope cannot be re-entered from inside. s_prev/tau_prev carry the
    #: reversal point the unloading line is drawn from.
    n_state = 3
    state_names = {"s_max": slice(0, 1), "s_prev": slice(1, 2),
                   "tau_prev": slice(2, 3)}
    #: not `uniaxial`: this is a traction-separation law and must not be picked
    #: up by an element expecting stress from strain
    bond = True

    def __init__(self, fc=None, condition="good", failure_mode="pull_out",
                 confinement="unconfined", tau_max=None, tau_max_coeff=None,
                 s1=None, s2=None, s3=None, alpha=0.4, tau_f_ratio=None,
                 s_elastic=None, rho=7850.0, **_ignored):
        if condition not in ("good", "poor"):
            raise ValueError("bond condition must be 'good' or 'poor'")
        if failure_mode not in ("pull_out", "splitting"):
            raise ValueError(
                "bond failure_mode must be 'pull_out' or 'splitting'; which "
                "one governs depends on cover, bar spacing and transverse "
                "reinforcement (MC2010 6.1.1), which this material cannot see")
        if confinement not in ("unconfined", "stirrups"):
            raise ValueError("bond confinement must be 'unconfined' or "
                             "'stirrups'")
        self.failure_mode = failure_mode
        self.confinement = confinement
        if tau_max is None and fc is None:
            raise ValueError(
                "bond_ceb needs fc (to get tau_max from MC2010) or an explicit "
                "tau_max")
        self.condition = condition
        self.fc = None if fc is None else float(fc)
        # MC2010 writes tau_max in MPa against sqrt(fc [MPa]); the factor below
        # is that formula moved into Pa, which is the unit everything else here
        # uses. Getting this wrong is a factor of a thousand and would look
        # like a bond that does not exist.
        poor = condition == "poor"
        if failure_mode == "pull_out":
            k = 1.25 if poor else 2.5
            peak = k * math.sqrt(self.fc / 1.0e6) * 1.0e6 if self.fc else None
            default_s1 = 1.8e-3 if poor else 1.0e-3
            default_s2 = 3.6e-3 if poor else 2.0e-3
            default_s3 = 10.0e-3
            default_tf_ratio = 0.4
        else:
            # SPLITTING. Shape built in, coefficient exposed -- see the module
            # docstring: the peak scales as (fcm/20)^0.25 and the coefficient
            # belongs to MC2010 Table 6.1-1, which was not verified here.
            coeff = (tau_max_coeff if tau_max_coeff is not None
                     else (8.0 if confinement == "stirrups" else 7.0))
            self.tau_max_coeff = float(coeff)
            peak = (self.tau_max_coeff * (self.fc / 20.0e6) ** 0.25 * 1.0e6
                    if self.fc else None)
            if poor:
                # poor bond conditions cut the peak; MC2010 tabulates it, and
                # this factor is the same 0.5 the pull-out column uses
                peak = None if peak is None else 0.5 * peak
            default_s1 = 1.2e-3 if not poor else 0.8e-3
            #: no plateau: splitting turns over at the peak
            default_s2 = default_s1
            #: a split cover unloads over a fraction of the rib spacing, not a
            #: multiple of it
            default_s3 = 5.0e-3
            #: a bar in split concrete has nothing left to rub against
            default_tf_ratio = 0.4 if confinement == "stirrups" else 0.0

        self.tau_max = float(tau_max) if tau_max is not None else peak
        if self.tau_max is None:
            raise ValueError("bond_ceb needs fc or an explicit tau_max")
        self.s1 = float(default_s1 if s1 is None else s1)
        self.s2 = float(default_s2 if s2 is None else s2)
        self.s3 = float(default_s3 if s3 is None else s3)
        self.alpha = float(alpha)
        ratio = default_tf_ratio if tau_f_ratio is None else float(tau_f_ratio)
        self.tau_f = ratio * self.tau_max
        self.rho = float(rho)

        if not 0.0 < self.s1 <= self.s2 <= self.s3:
            raise ValueError("bond_ceb needs 0 < s1 <= s2 <= s3")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("bond_ceb alpha must lie in (0, 1]")
        if self.tau_max <= 0.0:
            raise ValueError("bond_ceb tau_max must be positive")
        if not 0.0 <= self.tau_f <= self.tau_max:
            raise ValueError("bond_ceb tau_f must lie in [0, tau_max]")

        # THE ORIGIN, AND WHY IT NEEDS A CUT-OFF
        #   alpha < 1 makes d(tau)/ds infinite at s = 0. That is fine on paper
        #   and not fine in an explicit scheme: the stability bound an element
        #   builds from this law would be too small, the bar dof would be
        #   under-massed, and an under-massed dof in an explicit scheme is
        #   unstable (elements/stability.py has the bound and the derivation).
        #
        #   It also shows up as a surprise at rest: at a slip of 1e-16 the
        #   power law still returns tau_max * (1e-13)**0.4, about 79 Pa. Small,
        #   but not zero, and enough to fail a rigid-body test written with an
        #   absolute tolerance.
        #
        #   So the branch below `s_elastic` is linear, at the secant through
        #   that point. The default is s1/1000, which caps the tangent at
        #   1000**0.6 = 63 times the s1 secant and moves the envelope by less
        #   than a part in ten thousand of tau_max anywhere above it.
        self.s_elastic = float(self.s1 * 1.0e-3 if s_elastic is None
                               else s_elastic)
        if not 0.0 < self.s_elastic <= self.s1:
            raise ValueError("bond_ceb s_elastic must lie in (0, s1]")
        self.k_initial = (self.tau_max * (self.s_elastic / self.s1) ** self.alpha
                          / self.s_elastic)

        #: the unloading slope. NO SOFTER THAN THE LOADING TANGENT, and it was.
        #   It used to be the s1 secant, tau_max/s1 -- 63x softer than the
        #   capped origin tangent. Below the slip where the envelope's own
        #   tangent falls to that secant (s = s1 * alpha^(1/(1-alpha)), about
        #   0.22 s1) unloading was therefore SOFTER than loading: a point that
        #   slipped 0.7 um had to slide back 43 um to shed its stress, and a
        #   small reversal cycle GAINED energy. Eligehausen's unloading branch
        #   and MC2010's are near-vertical; the steepest thing this law knows
        #   is the capped origin tangent, and that is what unloading uses.
        #   The stability bound `E` below was already k_initial, so nothing
        #   an element sizes from this law changes.
        self.k_unload = self.k_initial
        #: what an element asks for when sizing its stability bound -- the
        #: LARGEST stiffness the law can present, which is the capped tangent
        #: at the origin and not the unloading secant
        self.E = self.k_initial
        self.wave_speed = 0.0        # a traction law has no wave of its own
        self.D0 = np.diag([self.k_unload, 0.0, 0.0, 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------ #

    def describe(self) -> str:
        return (f"    MC2010 {self.condition} bond, {self.failure_mode}"
                f"/{self.confinement}: tau_max = "
                f"{self.tau_max / 1e6:.2f} MPa, s1 = {self.s1 * 1e3:.2f} mm, "
                f"s2 = {self.s2 * 1e3:.2f} mm, s3 = {self.s3 * 1e3:.2f} mm, "
                f"alpha = {self.alpha:.2f}, tau_f = {self.tau_f / 1e6:.2f} MPa")

    def envelope(self, s):
        """The monotonic curve, for |s|, as an array or a scalar."""
        a = np.abs(np.asarray(s, dtype=float))
        out = np.empty_like(a)
        tiny = a < self.s_elastic
        out[tiny] = self.k_initial * a[tiny]
        rising = (a >= self.s_elastic) & (a <= self.s1)
        out[rising] = self.tau_max * (a[rising] / self.s1) ** self.alpha
        flat = (a > self.s1) & (a <= self.s2)
        out[flat] = self.tau_max
        down = (a > self.s2) & (a <= self.s3)
        out[down] = (self.tau_max - (self.tau_max - self.tau_f)
                     * (a[down] - self.s2) / (self.s3 - self.s2))
        out[a > self.s3] = self.tau_f
        return out

    def update(self, strain, state, dt, params=None):
        """`strain[:, 0]` is slip in metres; returns bond stress in Pa.

        The readable definition. The compiled twins mirror it and
        tests/test_bond.py pins them against it along a cyclic path.
        """
        s = np.asarray(strain[:, 0], dtype=float)
        s_max = state[:, 0]
        s_prev = state[:, 1]
        tau_prev = state[:, 2]

        # The line from the last point, held inside the virgin envelope AT
        # THE CURRENT SLIP, both ways. That is the whole rule.
        #
        # THE RULE IT REPLACES HAD A JUMP IN IT. It bounded the line by the
        # envelope at s_max and then, wherever |s| >= s_max, overrode it with
        # the envelope at s -- which assumes the line has already climbed to
        # the envelope by the time |s| reaches s_max. Reloading from the
        # OTHER sign it has not: measured on case_12b, a node committed at
        # -0.71 um and re-evaluated at +0.81 um went from -685 kPa on the
        # line to +810 kPa on the envelope in one step, 1.9 kN on the node
        # with no slip at which the two agree. No equilibrium exists across a
        # force jump, and the relaxation sat in a limit cycle at that node --
        # 8 000 kinetic resets, residual cycling between 1e-2 and 3e-2 -- for
        # as long as it was given. Bounding by the envelope at the CURRENT
        # slip is continuous by construction and, on the same side with a
        # steep unloading slope, identical to the old rule everywhere.
        #
        # The envelope at the largest excursion stays in the bound as well,
        # and for a different reason: past the peak the lugs have sheared and
        # the descending branch is DAMAGE, so a reversal must not rediscover
        # the virgin plateau on the other side (tests/test_bond.py::
        # test_a_reversal_stays_inside_the_envelope_reached_so_far). Taken
        # at max(s_max, |s|) so that virgin loading beyond s_max is not
        # capped by its own past, and continuous at |s| = s_max because the
        # envelope is.
        a = np.abs(s)
        bound = np.minimum(self.envelope(a), self.envelope(np.maximum(s_max, a)))
        line = tau_prev + self.k_unload * (s - s_prev)
        tau = np.clip(line, -bound, bound)

        state[:, 0] = np.maximum(s_max, np.abs(s))
        state[:, 1] = s
        state[:, 2] = tau
        out = np.zeros_like(strain)
        out[:, 0] = tau
        return out

    def output_fields(self, state):
        return {"slip": state[:, 1].copy(), "slip_max": state[:, 0].copy()}

    def init_state(self, npts: int, dtype=np.float64):
        return np.zeros((npts, self.n_state), dtype=dtype)

    # ------------------------------------------------------------------ #

    def numba_point_fn(self):
        from numba import njit

        tmax, s1, s2, s3 = self.tau_max, self.s1, self.s2, self.s3
        alpha, tau_f, ku = self.alpha, self.tau_f, self.k_unload

        s_el, k0 = self.s_elastic, self.k_initial

        @njit(cache=True, fastmath=False, inline="always")
        def envelope(a):
            if a < s_el:
                return k0 * a
            if a <= s1:
                return tmax * (a / s1) ** alpha
            if a <= s2:
                return tmax
            if a <= s3:
                return tmax - (tmax - tau_f) * (a - s2) / (s3 - s2)
            return tau_f

        @njit(cache=True, fastmath=False, inline="always")
        def point(strain, state, dt, k, stress):
            for i in range(1, 6):
                stress[i] = 0.0
            s = strain[0]
            a = abs(s)
            bound = envelope(a)
            reach = envelope(max(state[0], a))
            if reach < bound:
                bound = reach
            tau = state[2] + ku * (s - state[1])
            if tau > bound:
                tau = bound
            if tau < -bound:
                tau = -bound
            if a > state[0]:
                state[0] = a
            state[1] = s
            state[2] = tau
            stress[0] = tau

        return point

    P = dict(tau_max=0, s1=1, s2=2, s3=3, alpha=4, tau_f=5, k_unload=6,
             s_elastic=7, k_initial=8)

    def numba_params(self):
        return np.array([self.tau_max, self.s1, self.s2, self.s3,
                         self.alpha, self.tau_f, self.k_unload,
                         self.s_elastic, self.k_initial], dtype=np.float64)

    def warp_params(self, dtype=np.float64):
        return (self.numba_params().astype(dtype), np.zeros(1, dtype=dtype))

    @staticmethod
    def warp_point_fn():
        from ..kernels.warp_materials import bond_ceb_point

        return bond_ceb_point

    def damage_measure(self, state):
        """No damage variable: erosion has no meaning for an interface."""
        return None
