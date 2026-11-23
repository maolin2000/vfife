"""Cyclic bond stress-slip: the MC2010 envelope plus degradation and pinching.

WHY A SECOND MATERIAL AND NOT A FLAG ON THE FIRST
-------------------------------------------------
`bond_ceb` (materials/bond.py) says, in its own docstring, exactly what it is
not:

    It is NOT Eligehausen's cyclic model: there is no frictional plateau on
    reversal, no damage from cycling, and no degradation of tau_max with the
    number of cycles. [...] Under a long cyclic history it will overstate the
    bond that survives, and the direction of that error is unsafe.

This class is that missing part. It is registered separately and `bond_ceb` is
not touched, for the same reason `EmbeddedTruss` survived `BondedTruss`: the
monotonic law is cheaper, it is right for a pushover, and every result already
produced with it must keep reproducing. Choosing the cyclic law is an explicit
act.

WHAT IS ADDED, AND THE ORDER IT MATTERS IN
------------------------------------------
1.  DEGRADATION. Cycling inside the envelope dissipates energy and wears the
    channel the ribs bear on. A damage variable driven by that energy scales
    the whole envelope down. This is the part whose absence was called unsafe.

2.  PINCHING. On reversal the bar does not pick up the opposite envelope at
    once: it slides back through the channel it has already made, at friction,
    and only re-engages ribs near the far end of its previous travel. Without
    this the loop is a fat parallelogram and the energy it dissipates is far
    too large.

THE RULE, IN FULL
-----------------
    g      = 1 - d,  d = 1 - exp(-c * (E/E_ref)**p)      degradation factor
    s_max  = largest |slip| the point has ever seen
    reach  = g * envelope(s_max)                         what the tip is worth
    fric   = g * tau_f * friction_ratio                  the sliding stress

    if |s| >= s_max:                                     pushing new ground
        tau = sign(s) * g * envelope(|s|)
    else:                                                inside the loop
        tau = tau_prev + k_unload * (s - s_prev)         steep unload/reload
        tau = clip(tau, -cap(-s_max), +cap(+s_max))

    cap(tip) = fric + (reach - fric) * clip(1 - |s - tip|/s_reload, 0, 1)

The cap is what does the pinching, and it is written so that no test of the
direction of travel is needed: it equals `reach` exactly at either tip, so the
interior is continuous with the envelope there, and it falls to `fric` once the
slip is more than `s_reload` away from both. A point unloading from the tip
therefore runs down the steep line, flattens onto the friction plateau when the
line crosses it, slides across, and ramps back up to the envelope over the last
`s_reload` before the far tip -- which is the shape Eligehausen measured.

ONE CHANNEL, NOT TWO
    `s_max` is a single number, not one per direction. A bar driven to +5 mm
    has destroyed the channel for -1 mm as well, so the reversal must not find
    a virgin envelope waiting for it. Tracking the two sides separately and
    letting the negative side start fresh would hand the model bond strength
    that the positive excursion had already spent -- an error in the unsafe
    direction, and the one this class exists to remove.

WHICH NUMBERS ARE PHYSICS AND WHICH ARE CALIBRATION
    The SHAPE above is not in doubt. The COEFFICIENTS in the damage law are:
    `damage_coeff` = 1.6 and `damage_exponent` = 1.1 are the numbers usually
    quoted from Eligehausen, Popov and Bertero (1983), but they were NOT
    verified against the report here, and the energy measure they multiply is
    this class's own (below), which is not identical to the one they used.
    Treat them as a starting point to CALIBRATE, exactly as `tau_max_coeff` is
    treated on the splitting branch of `bond_ceb`. The same goes for
    `friction_ratio` (1.0) and `s_reload` (s1).

    Under a monotonic path all of them are inert -- see `update`.

DAMAGE IS DRIVEN BY THE LOAD PATH, NOT BY THE ITERATION PATH
------------------------------------------------------------
This is the part that cost the most to get right, and it is not a detail.

A dynamic-relaxation solve reaches ONE equilibrium point by taking tens of
thousands of explicit steps, and that approach oscillates: on the block-in-
tension case the slip overshoots its converged value of 24.9 um to 33.9 um and
comes back. To a rule that banks energy every time it is called, every one of
those wiggles is a load reversal. The first version of this class did exactly
that, and the result was not a constitutive law at all:

    monotonic tension, same tolerance, only n_steps changed
        n_steps  60 000     slip 106.0 um     damage 0.0201
        n_steps 200 000     slip  24.3 um     damage 0.0172
        bond_ceb, either     slip  24.9 um     damage 0

A factor of 4.4 in the answer from a solver setting, and a monotonic push
carrying damage at all. The feedback is what makes it violent: spurious damage
softens the bond, softer bond oscillates further, further oscillation banks
more damage.

So `update()` does NOT accumulate anything. It reads the damage and returns a
stress. Energy is banked by `commit()`, which the solver calls when a load
level has CONVERGED (core/solver.py, at the point a staged level is declared
done and at a plain relaxation's convergence). The increment is the trapezoid
between the previous equilibrium point and this one, so the wiggles in between
are discarded by construction rather than filtered by a threshold.

`s_max_cmt` -- the largest excursion of the COMMITTED path -- is what decides
whether a load step was a reversal. The iteration-path `s_max` cannot do that
job: on a monotonic push it sits just above the converged slip because of the
overshoot, so every load step would look like a reversal and the model would
bank the whole envelope.

A material without a `commit` method is untouched by any of this; `bond_ceb`
has none and the solver's hook is a `getattr`.

THE ENERGY MEASURE, STATED PLAINLY
    E accumulates the POSITIVE work increments 0.5*(tau + tau_cmt)*ds taken
    between converged points that lie INSIDE the envelope, per unit bonded
    area (J/m^2).

      * Inside only, so a monotonic push accumulates nothing and reproduces
        `bond_ceb` exactly. Envelope damage is already in the descending
        branch; counting it again would degrade the monotonic curve twice.
      * Positive increments only, which makes E monotone and the damage
        irreversible without a second state column -- the same device
        `rebar.rupture_strain` uses on accumulated plastic strain.
      * It therefore OVER-counts the true loop area by the elastic unloading
        it declines to give back. The normalisation E_ref absorbs the scale
        and `damage_coeff` absorbs the rest; this is calibration, not physics,
        and is why the coefficients above are exposed.

    E_ref is the area under the monotonic envelope from 0 to s3, in closed
    form. It is a fixed property of the law, so E/E_ref is dimensionless and
    comparable between two models with different strengths.
"""
from __future__ import annotations

import numpy as np

from .base import register
from .bond import BondCEB


@register(
    "bond_cyclic",
    kinematics=("bond",),
    backends=("numpy", "numba", "warp"),
    verification="constitutive_path",
    output_fields=("bond_stress", "slip", "slip_max", "bond_damage",
                   "bond_energy"),
    nonlinear=True,
    caveat="Eligehausen-type cyclic degradation on the MC2010 envelope. The "
           "shape is standard; damage_coeff, damage_exponent, friction_ratio "
           "and s_reload are exposed defaults to calibrate, not citations.",
)
class BondCyclic(BondCEB):
    """MC2010 envelope with energy-driven degradation and a friction plateau.

    Every `bond_ceb` parameter is accepted and means the same thing; the four
    below are the additions. They are listed explicitly rather than swallowed
    by **kw so that the registry's parameter contract names them.
    """

    #: `bond_ceb`'s three, plus the energy that drives damage and the three
    #: that describe the last CONVERGED point (see the module docstring: the
    #: iteration path must not drive damage). The first three keep their
    #: positions so the compiled mirrors stay line-for-line with the monotonic
    #: ones and `output_fields` keeps its indices.
    n_state = 7
    state_names = {"s_max": slice(0, 1), "s_prev": slice(1, 2),
                   "tau_prev": slice(2, 3), "energy": slice(3, 4),
                   "s_cmt": slice(4, 5), "tau_cmt": slice(5, 6),
                   "s_max_cmt": slice(6, 7)}

    def __init__(self, damage_coeff=1.6, damage_exponent=1.1,
                 friction_ratio=1.0, s_reload=None, **kw):
        super().__init__(**kw)
        self.damage_coeff = float(damage_coeff)
        self.damage_exponent = float(damage_exponent)
        self.friction_ratio = float(friction_ratio)
        if self.damage_coeff < 0.0:
            raise ValueError("bond_cyclic damage_coeff must not be negative")
        if self.damage_exponent <= 0.0:
            raise ValueError("bond_cyclic damage_exponent must be positive")
        if not 0.0 <= self.friction_ratio <= 1.0:
            raise ValueError("bond_cyclic friction_ratio must lie in [0, 1]; "
                             "the sliding stress is a FRACTION of the "
                             "envelope's residual friction, and above 1 the "
                             "plateau would exceed the envelope it sits under")

        # The slip over which ribs re-engage on reload. s1 is the slip the
        # envelope itself needs to develop tau_max, so it is the scale on
        # which bearing is established; there is no second scale in the law.
        self.s_reload = float(self.s1 if s_reload is None else s_reload)
        if self.s_reload <= 0.0:
            raise ValueError("bond_cyclic s_reload must be positive")

        #: the sliding stress at zero damage
        self.tau_fric0 = self.tau_f * self.friction_ratio

        # Area under the monotonic envelope, 0 to s3, in closed form:
        # power branch + plateau + trapezoid down to the residual. The
        # s_elastic cut-off changes this by less than a part in ten thousand
        # (it moves a sliver under a curve of width s1/1000) and is ignored,
        # which keeps E_ref a clean function of the published parameters.
        self.E_ref = (self.tau_max * self.s1 / (self.alpha + 1.0)
                      + self.tau_max * (self.s2 - self.s1)
                      + 0.5 * (self.tau_max + self.tau_f) * (self.s3 - self.s2))
        if self.E_ref <= 0.0:
            raise ValueError("bond_cyclic E_ref came out non-positive")

        # THE STABILITY BOUND IS UNCHANGED, AND THAT IS A CLAIM, NOT A HOPE.
        #   `self.E` is what an element asks for when sizing its time step, and
        #   it must be the LARGEST stiffness this law can present. Damage only
        #   scales the envelope DOWN (g <= 1), so no degraded branch is steeper
        #   than its virgin twin. The one slope this class adds is the reload
        #   ramp, (reach - fric)/s_reload, which is at most
        #   (tau_max - tau_f)/s1 < tau_max/s1 < k_initial = k_unload.
        #   So `E = k_initial`, inherited, still bounds everything.
        #
        #   `k_unload` is k_initial rather than the s1 secant since the
        #   sign-reversal jump was found on case_12b (materials/bond.py has
        #   the measurement). It also makes THIS class continuous at +-s_max
        #   from either side: the line reaches the pinched cap before the
        #   envelope takes over, which a 63x softer line did not.
        #   tests/test_bond_cyclic.py asserts that arithmetic rather than
        #   trusting this paragraph.

    # ------------------------------------------------------------------ #

    def describe(self) -> str:
        return (super().describe()
                + f"\n    cyclic: damage 1-exp(-{self.damage_coeff:.2f}"
                  f"(E/{self.E_ref:.3g})^{self.damage_exponent:.2f}), "
                  f"friction {self.tau_fric0 / 1e6:.2f} MPa, "
                  f"reload over {self.s_reload * 1e3:.2f} mm")

    def damage(self, energy):
        """Degradation from accumulated cyclic energy. Monotone, in [0, 1)."""
        e = np.maximum(np.asarray(energy, dtype=float), 0.0) / self.E_ref
        return 1.0 - np.exp(-self.damage_coeff * e ** self.damage_exponent)

    def update(self, strain, state, dt, params=None):
        """`strain[:, 0]` is slip in metres; returns bond stress in Pa.

        The readable definition; the compiled twins mirror it and
        tests/test_bond_cyclic.py pins them against it along a cyclic path.

        NOTHING IS ACCUMULATED HERE. Damage is read, never written: this is
        called once per explicit iteration, and a relaxation solve takes tens
        of thousands of them to reach one equilibrium point. `commit()` banks
        the energy, on the converged path. See the module docstring for the
        measurement that forced this.

        MONOTONIC PATHS COME OUT OF HERE UNCHANGED. On a committed path that
        only ever extends |s|, nothing is ever banked, `g` stays exactly 1,
        and the returned stress is the same `bond_ceb` envelope bit for bit.
        That is asserted, not assumed.
        """
        s = np.asarray(strain[:, 0], dtype=float)
        s_max = state[:, 0]
        s_prev = state[:, 1]
        tau_prev = state[:, 2]
        energy = state[:, 3]

        g = 1.0 - self.damage(energy)
        a = np.abs(s)
        inside = a < s_max

        # --- on the envelope: the virgin curve, scaled by what is left ---- #
        tau_env = g * self.envelope(a) * np.sign(s)

        # --- inside: steep line, capped by the pinched loop --------------- #
        reach = g * self.envelope(s_max)
        fric = g * self.tau_fric0
        line = tau_prev + self.k_unload * (s - s_prev)
        # `cap` equals `reach` at its tip and decays to `fric` a distance
        # s_reload away, so the two branches meet continuously at +-s_max.
        ramp_p = np.clip(1.0 - (s_max - s) / self.s_reload, 0.0, 1.0)
        ramp_n = np.clip(1.0 - (s + s_max) / self.s_reload, 0.0, 1.0)
        cap_p = fric + (reach - fric) * ramp_p
        cap_n = fric + (reach - fric) * ramp_n
        tau_in = np.minimum(np.maximum(line, -cap_n), cap_p)

        tau = np.where(inside, tau_in, tau_env)

        state[:, 0] = np.maximum(s_max, a)
        state[:, 1] = s
        state[:, 2] = tau
        out = np.zeros_like(strain)
        out[:, 0] = tau
        return out

    def commit(self, state):
        """This state is an EQUILIBRIUM point: bank the dissipation since the
        last one and remember it.

        Called by the solver when a load level converges, and by nothing else.
        The whole justification is in the module docstring; the short version
        is that the increment below spans two points the analyst asked for,
        while `update` spans two explicit iterations, and only the first of
        those is a load path.

        `inside` is decided against `s_max_cmt`, the largest excursion of the
        COMMITTED path. Deciding it against the iteration-path `s_max` would
        call every monotonic load step a reversal, because dynamic relaxation
        overshoots on its way in.
        """
        s = state[:, 1]
        tau = state[:, 2]
        s_cmt = state[:, 4]
        tau_cmt = state[:, 5]
        s_max_cmt = state[:, 6]

        a = np.abs(s)
        inside = a < s_max_cmt
        d_energy = 0.5 * (tau + tau_cmt) * (s - s_cmt)
        state[:, 3] += np.where(inside, np.maximum(d_energy, 0.0), 0.0)
        state[:, 6] = np.maximum(s_max_cmt, a)
        state[:, 4] = s
        state[:, 5] = tau

    def output_fields(self, state):
        return {"slip": state[:, 1].copy(), "slip_max": state[:, 0].copy(),
                "bond_energy": state[:, 3].copy(),
                "bond_damage": self.damage(state[:, 3])}

    # ------------------------------------------------------------------ #

    def numba_point_fn(self):
        from numba import njit

        tmax, s1, s2, s3 = self.tau_max, self.s1, self.s2, self.s3
        alpha, tau_f, ku = self.alpha, self.tau_f, self.k_unload
        s_el, k0 = self.s_elastic, self.k_initial
        e_ref, c_d, p_d = self.E_ref, self.damage_coeff, self.damage_exponent
        fric0, s_rel = self.tau_fric0, self.s_reload

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
            s_max = state[0]
            tau_prev = state[2]

            e_n = state[3] / e_ref
            if e_n < 0.0:
                e_n = 0.0
            g = np.exp(-c_d * e_n ** p_d)

            if a < s_max:
                reach = g * envelope(s_max)
                fric = g * fric0
                tau = tau_prev + ku * (s - state[1])
                r_p = 1.0 - (s_max - s) / s_rel
                if r_p < 0.0:
                    r_p = 0.0
                elif r_p > 1.0:
                    r_p = 1.0
                r_n = 1.0 - (s + s_max) / s_rel
                if r_n < 0.0:
                    r_n = 0.0
                elif r_n > 1.0:
                    r_n = 1.0
                cap_p = fric + (reach - fric) * r_p
                cap_n = fric + (reach - fric) * r_n
                if tau > cap_p:
                    tau = cap_p
                if tau < -cap_n:
                    tau = -cap_n
                # NOTHING is banked here; see BondCyclic.commit. A twin that
                # accumulated per iteration would disagree with the reference
                # only after a solve, which is the most expensive place to
                # find a backend split.
            else:
                tau = g * envelope(a)
                if s < 0.0:
                    tau = -tau
                elif s == 0.0:
                    tau = 0.0
                state[0] = a

            state[1] = s
            state[2] = tau
            stress[0] = tau

        return point

    P = dict(tau_max=0, s1=1, s2=2, s3=3, alpha=4, tau_f=5, k_unload=6,
             s_elastic=7, k_initial=8, E_ref=9, damage_coeff=10,
             damage_exponent=11, tau_fric0=12, s_reload=13)

    def numba_params(self):
        return np.array([self.tau_max, self.s1, self.s2, self.s3,
                         self.alpha, self.tau_f, self.k_unload,
                         self.s_elastic, self.k_initial, self.E_ref,
                         self.damage_coeff, self.damage_exponent,
                         self.tau_fric0, self.s_reload], dtype=np.float64)

    @staticmethod
    def warp_point_fn():
        from ..kernels.warp_materials import bond_cyclic_point

        return bond_cyclic_point
