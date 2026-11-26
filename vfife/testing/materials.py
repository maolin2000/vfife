"""The questions that do not depend on the material.

v0.1 built this for the ELEMENTS: one minimal model per registered element,
a handful of questions no benchmark states, asked of all of them at once. It
found seven defects in an afternoon, four of them in units that every case in
the library had been running through for months.

The materials had nothing of the kind. Each law has its own test file, deep
and specific -- `test_cdp.py` checks the meridian ratio and the dilation
angle, `test_bond.py` walks the MC2010 envelope branch by branch -- and those
tests are worth more than anything here for the law they were written for.
What was missing is the other axis: the questions that are true of EVERY
constitutive law, asked of every law, so that the fifteenth material inherits
them on the day it is registered rather than on the day someone writes its
test file.

The questions, and what each of them has already cost this library:

    the stress is continuous       `bond_ceb` bounded a reversal by the
    in the strain                  envelope at the largest past excursion
                                   instead of at the current slip. A node
                                   committed at -0.71 um and re-evaluated at
                                   +0.81 um moved 1.5 MPa in one step with no
                                   slip at which the two rules agree. No
                                   equilibrium exists across a jump: case_12b
                                   sat in a limit cycle at that node, 8 000
                                   kinetic resets, for as long as it was
                                   given.

    a closed cycle does not        the thermodynamic floor. A law that gains
    create energy                  energy round a closed strain cycle is a
                                   free energy source, and in an explicit
                                   dynamic run it is an instability with no
                                   step size small enough to fix it.

    unloading is no stiffer        damage and plasticity may only ever take
    than the first loading         stiffness away. An unloading branch that
                                   comes back steeper than E is the signature
                                   of a state variable updated in the wrong
                                   order.

    the points do not talk         the state array is (n_points, n_state) and
    to each other                  every law is vectorised over it. One
                                   `np.max` without an `axis=` and a beam's
                                   worst-damaged point sets the response of
                                   all the others -- which looks like a
                                   converged answer.

    the time step does not         a rate-independent law that reads `dt` is
    change a rate-independent      a law whose answer depends on the analyst's
    law                            time step, which is not a modelling choice
                                   anyone declared.

    rotating the strain            isotropy. It is the material half of
    rotates the stress             v0.1's "a rigid rotation is free", and it
                                   catches the same class of defect: an axis
                                   assumed somewhere in a return map.

    damage does not heal           on a monotonic path. NOT on a cycle: `cdp`
                                   recovers stiffness when a crack closes, by
                                   design (`wc`), so its `d` legitimately
                                   falls there.

    the crack-band energy does     the regularised laws only. `l_ref`, and the
    not depend on the element      six rounds of v0.1 that ended at one
    size                           parameter used on the wrong side of it.

WHAT THIS IS NOT. It does not check that a law is RIGHT -- that a curve
matches GB 50010 or that a peak lands where a test report says. That is what
the per-material test files and the verification contracts do. Every question
here is one whose answer is known before the law is written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..materials import base as mat_base
from .invariants import Finding
from .models import MATERIAL_PARAMS


# --------------------------------------------------------------------------- #
#  Voigt bookkeeping
# --------------------------------------------------------------------------- #
#
# `vfife/voigt.py` states the one convention this package obeys, and the
# ordering of the shear slots is part of it:
#
#     strain  [exx, eyy, ezz, gxy, gyz, gzx]   engineering, g = 2 e
#     stress  [sxx, syy, szz, txy, tyz, tzx]   true shear stress
#
# YZ THEN ZX, not xy/xz/yz. The first version of the rotation check below used
# the other order and passed on linear elasticity anyway -- a consistent
# mislabelling is just a relabelling of the axes, and an isotropic law cannot
# tell -- so it would have gone on to accuse `cdp` of anisotropy on the
# strength of my own bookkeeping. Anything that converts between the two forms
# goes through these three functions.

_SHEAR = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])   # Voigt -> tensor on strain


def to_tensor(voigt: np.ndarray) -> np.ndarray:
    """(6,) engineering-Voigt strain -> (3, 3) symmetric tensor."""
    e = np.asarray(voigt, dtype=float) * _SHEAR
    return np.array([[e[0], e[3], e[5]],
                     [e[3], e[1], e[4]],
                     [e[5], e[4], e[2]]])


def stress_to_tensor(voigt: np.ndarray) -> np.ndarray:
    """(6,) Voigt stress -> (3, 3) tensor. No factor of two anywhere: the
    engineering convention is a property of the STRAIN, and using `to_tensor`
    on a stress halves its shears (which is how the first version of the
    rotation check below reported 4.7e-2 on linear elasticity)."""
    s = np.asarray(voigt, dtype=float)
    return np.array([[s[0], s[3], s[5]],
                     [s[3], s[1], s[4]],
                     [s[5], s[4], s[2]]])


def to_voigt(tensor: np.ndarray, *, engineering: bool = True) -> np.ndarray:
    """(3, 3) symmetric tensor -> (6,) Voigt. `engineering` doubles the shears."""
    t = np.asarray(tensor, dtype=float)
    v = np.array([t[0, 0], t[1, 1], t[2, 2], t[0, 1], t[1, 2], t[0, 2]])
    if engineering:
        v[3:] *= 2.0
    return v


# --------------------------------------------------------------------------- #
#  one specimen per registered material
# --------------------------------------------------------------------------- #

@dataclass
class Spec:
    """How to drive one law: what to build it with, and how far to push it.

    `amplitude` is the scalar the strain direction is multiplied by at the far
    end of a path. Its SIGN is the interesting side of the law -- concrete is
    driven into compression, steel into tension -- and the closed cycle visits
    both regardless.
    """
    name: str
    props: dict
    kinematics: str                 # solid_3d | uniaxial | bond
    amplitude: float
    #: the strain direction, unit 2-norm in Voigt. Set by `__post_init__`.
    direction: np.ndarray = field(default=None)
    isotropic: bool = True          # false = anisotropy is the point of the law
    rate_dependent: bool = False    # false = `dt` must not enter the answer
    steps: int = 200

    def __post_init__(self):
        if self.direction is None:
            self.direction = _DIRECTION[self.kinematics]
        d = np.asarray(self.direction, dtype=float)
        self.direction = d / np.linalg.norm(d)


#: Uniaxial-stress-like for a solid (transverse contraction at a nominal 0.2)
#: with a shear component, so that a law which is right on the diagonal and
#: wrong off it cannot hide; the scalar laws get the one component they read.
_DIRECTION = {
    "solid_3d": np.array([1.0, -0.2, -0.2, 0.25, 0.0, 0.0]),
    "uniaxial": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "bond": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
}

#: The two bond laws are driven by SLIP IN METRES, not strain, and they are
#: not in `models.MATERIAL_PARAMS` because no element model in that registry
#: carries an interface. s1 = 1 mm, s3 = 10 mm, so 3 mm is past the plateau
#: and into the descending branch where the reversal rule lives.
_BOND_PARAMS = {
    "bond_ceb": {"fc": 3.0e7},
    "bond_cyclic": {"fc": 3.0e7},
}

#: amplitude, and the exemptions, per registered material.
_DRIVE = {
    "linear_elastic":     dict(kinematics="solid_3d", amplitude=1.0e-3),
    "orthotropic":        dict(kinematics="solid_3d", amplitude=1.0e-3,
                               isotropic=False),
    "viscoelastic":       dict(kinematics="solid_3d", amplitude=1.0e-3,
                               rate_dependent=True),
    "j2":                 dict(kinematics="solid_3d", amplitude=3.0e-3),
    "chaboche":           dict(kinematics="solid_3d", amplitude=4.0e-3),
    "cdp":                dict(kinematics="solid_3d", amplitude=-3.0e-3),
    # a rupture strain the cycle actually reaches, so that
    # `failure_is_irreversible` is exercised rather than skipped. It lives
    # here and NOT in `models.MATERIAL_PARAMS`, which the element models share:
    # a bar that ruptures halfway through every element invariant would be
    # answering a different question than the one those checks ask.
    "rebar":              dict(kinematics="uniaxial", amplitude=6.0e-3,
                               extra={"rupture_strain": 3.0e-3}),
    # PAST THE NECKING STRAIN (3.61%) and past a shortened fracture strain, so
    # the specimen walks the whole law -- skeleton, reloading, degradation,
    # fracture -- rather than only its hysteresis. `eps_cri` is cut from the
    # coupon's 36% because a check that never reaches the event it is about is
    # a check that passes for the wrong reason (v0.1 says the same of
    # `rebar`).
    "steel_cyclic":       dict(kinematics="uniaxial", amplitude=5.0e-2,
                               extra={"eps_cri": 4.5e-2}),
    "steel02":            dict(kinematics="uniaxial", amplitude=6.0e-3),
    "piecewise_uniaxial": dict(kinematics="uniaxial", amplitude=5.0e-3),
    "concrete01":         dict(kinematics="uniaxial", amplitude=-3.0e-3),
    "concrete02":         dict(kinematics="uniaxial", amplitude=-3.0e-3),
    "parallel":           dict(kinematics="uniaxial", amplitude=6.0e-3),
    "series":             dict(kinematics="uniaxial", amplitude=8.0e-3),
    "gap":                dict(kinematics="uniaxial", amplitude=3.0e-3),
    "bond_ceb":           dict(kinematics="bond", amplitude=3.0e-3),
    "bond_cyclic":        dict(kinematics="bond", amplitude=3.0e-3),
}


def specs() -> Dict[str, Spec]:
    """One `Spec` per material in the registry.

    Building it from `materials.base.definitions()` rather than from a list
    here is the point: a newly registered law with no entry in `_DRIVE` raises
    from `spec()`, so it cannot be added without saying how to drive it.
    """
    return {name: spec(name) for name in mat_base.available()}


def materials() -> List[str]:
    return sorted(mat_base.available())


def spec(name: str) -> Spec:
    if name not in _DRIVE:
        raise ValueError(
            f"material {name!r} is registered but `vfife.testing.materials._DRIVE` "
            "does not say how far to drive it; add an amplitude (and any "
            "exemption) there")
    props = dict(_BOND_PARAMS.get(name) or MATERIAL_PARAMS.get(name) or {})
    if not props:
        raise ValueError(f"no parameter set for material {name!r}")
    drive = dict(_DRIVE[name])
    props.update(drive.pop("extra", {}))
    return Spec(name=name, props=props, **drive)


def make(name: str):
    """(material, spec) for one registered law."""
    s = spec(name)
    return mat_base.create(name, **s.props), s


def spec_for(props: dict, resolved=None) -> Spec:
    """A `Spec` for the material block of a USER's case, not the harness's own.

    How far to drive it and which questions apply come from `_DRIVE`, keyed by
    the registered type; the parameters are the analyst's. The amplitudes there
    are strains, so they carry across a change of units and of stiffness -- a
    3e-3 strain is past yield for steel and past the peak for concrete whatever
    E is. `resolved` is the material object the solver built, used only to fill
    in an `l_ref: auto` that the case left to the mesh.
    """
    name = props.get("type")
    if name not in _DRIVE:
        raise ValueError(f"material {name!r} has no entry in _DRIVE")
    p = {k: v for k, v in props.items() if k != "type"}
    drive = dict(_DRIVE[name])
    drive.pop("extra", None)          # the harness's specimen, not the analyst's
    for key in ("char_length", "l_ref"):
        if str(p.get(key, "")).lower() == "auto":
            p[key] = float(getattr(resolved, key, 1.0)) if resolved is not None else 1.0
    return Spec(name=name, props=p, **drive)


# --------------------------------------------------------------------------- #
#  strain paths and the driver
# --------------------------------------------------------------------------- #

def path(s: Spec, kind: str = "cycle", *, steps: Optional[int] = None) -> np.ndarray:
    """Total strain at each step of a path, shape (n, 6).

    monotonic  0 -> A            everything that must not depend on history
    cycle      0 -> A -> 0 -> -A -> 0   closed, so the work round it is the
                                        dissipation
    unload     0 -> A -> A/2     the unloading branch on its own
    """
    n = int(steps or s.steps)
    a = float(s.amplitude)
    if kind == "monotonic":
        scale = np.linspace(0.0, a, n + 1)[1:]
    elif kind == "unload":
        up = np.linspace(0.0, a, n + 1)[1:]
        scale = np.r_[up, np.linspace(a, 0.5 * a, n + 1)[1:]]
    elif kind == "cycle":
        q = n // 4 + 1
        scale = np.r_[np.linspace(0.0, a, q + 1)[1:],
                      np.linspace(a, 0.0, q + 1)[1:],
                      np.linspace(0.0, -a, q + 1)[1:],
                      np.linspace(-a, 0.0, q + 1)[1:]]
    else:
        raise ValueError(f"unknown path {kind!r}")
    return scale[:, None] * s.direction[None, :]


def drive(material, strains: np.ndarray, *, dt: float = 1.0,
          n_points: int = 1, commit: bool = True, state: np.ndarray = None):
    """Walk a strain path. Returns (stress history, final state).

    `strains` is (n_steps, 6) applied to every point, or (n_steps, n_points, 6)
    to give each point its own history. Each step is one equilibrium point, so
    a law with a `commit` hook (`bond_cyclic` banks its dissipation there) gets
    it once per step -- which is what the solver does at a converged load level
    and nowhere else.
    """
    e = np.asarray(strains, dtype=float)
    if e.ndim == 2:
        e = np.repeat(e[:, None, :], n_points, axis=1)
    n_points = e.shape[1]
    if state is None:
        state = material.init_state(n_points)
    hook = getattr(material, "commit", None) if commit else None
    out = np.empty_like(e)
    for i, step in enumerate(e):
        out[i] = material.update(np.ascontiguousarray(step), state, dt)
        if hook is not None:
            hook(state)
    return out, state


def stiffness(material, s: Spec) -> float:
    """The law's own initial stiffness, MEASURED: the secant of one tiny step
    from the virgin state, along the path's direction.

    Measured rather than read off `material.E` because the bond laws have no
    `E` (their initial stiffness is a slip modulus in Pa/m) and because a law
    whose declared `E` is not the slope it actually starts with is itself a
    defect worth tripping over here.
    """
    # SIGNED: `concrete01` carries no tension at all, so a probe on the
    # positive side measures an initial stiffness of exactly zero and every
    # ratio built on it comes back as an infinity.
    h = 1.0e-9 * s.amplitude
    probe, _ = drive(material, (h * s.direction)[None, :], commit=False)
    # AGAINST THE VIRGIN STRESS, not against zero. A law carrying a residual
    # returns that residual at a strain of 1e-12, which divided by 1e-12 is an
    # enormous stiffness -- and every ratio built on it, including the one that
    # was supposed to REPORT the residual, comes back reassuringly small.
    virgin, _ = drive(material, np.zeros((1, 6)), commit=False)
    return float((probe[0, 0] - virgin[0, 0]) @ s.direction) / h


def rate_dependence(material, s: Spec) -> str:
    """Why this law's answer may legitimately depend on `dt`, or "".

    Asked of the BUILT law, not of the table, because for `cdp` it is a
    PARAMETER: the library's own specimen has `viscosity: 0` and is exactly
    rate-independent, while case_12 declares `viscosity: 1.0` and is not. A
    table keyed by the material name gets that wrong in both directions -- run
    against case_12 it accused the law of reading `dt` (0.93) and of unloading
    19% stiffer than it loaded, both of which are the viscous stress doing
    exactly what the analyst asked it to.
    """
    if s.rate_dependent:
        return "rate-dependent by declaration"
    mu = float(getattr(material, "mu", 0.0) or 0.0)
    if mu > 0.0:
        return f"viscous regularisation, viscosity = {mu:g}"
    if int(getattr(material, "n_prony", 0) or 0):
        return "a Prony series"
    return ""


def work(stress: np.ndarray, strain: np.ndarray) -> float:
    """Trapezoidal integral of sigma : d(epsilon) along one point's path,
    starting from the (stress-free) virgin state."""
    sig = np.r_[np.zeros((1, 6)), np.asarray(stress, dtype=float)]
    eps = np.r_[np.zeros((1, 6)), np.asarray(strain, dtype=float)]
    return float(np.sum(0.5 * (sig[1:] + sig[:-1]) * np.diff(eps, axis=0)))


# --------------------------------------------------------------------------- #
#  the questions
# --------------------------------------------------------------------------- #

def the_virgin_state_is_free(material, s: Spec) -> Finding:
    """Zero strain on a fresh state is zero stress.

    A residual here is a precomputed quantity that was not zeroed, and it
    shows up in a run as a body that moves before it is loaded.
    """
    stress, _ = drive(material, np.zeros((1, 6)), commit=False)
    E0 = abs(stiffness(material, s))
    value = float(np.max(np.abs(stress))) / max(E0 * abs(s.amplitude), 1e-300)
    return Finding("the_virgin_state_is_free", value <= 1e-12, value, 1e-12,
                   "stress at zero strain, over E0 * amplitude")


def an_evaluation_at_zero_costs_nothing(material, s: Spec) -> Finding:
    """Asking the law for the stress at zero strain, before anything moves,
    may not change what it says afterwards.

    THIS IS NOT A HYPOTHETICAL CALL. Every solver in this library assembles
    the internal force once before the first step -- to size the time step, to
    write the t = 0 output frame, to check equilibrium -- so every material
    point in every run is evaluated at zero strain first. A law for which that
    is not free has a different constitutive response in a solver than on a
    bench, which is the hardest kind of difference to find later.

    `steel_cyclic` failed it and the failure was total, not
    marginal: the zero call set a loading DIRECTION without recording a
    stress, the first step the other way then registered a reversal at exactly
    (0, 0), and the elastic branch it entered had no way out. On a monotone
    push afterwards the point ran up the elastic line past yield, past f_u, to
    14 times f_u -- while the same law driven from the first non-zero strain
    was perfectly correct. A member built of it carried 46% more load than its
    own squash capacity.

    BOTH SIGNS ARE DRIVEN, and that is not thoroughness -- it is the whole
    test. The zero call records a loading direction of +1 (there is nothing
    else it could record), so a path that then goes UP agrees with itself
    trivially. Only a path whose FIRST MOVE IS DOWNWARD makes the zero call
    look like a reversal, and that is the case that was broken.

    The tolerance is tight on purpose: the two histories differ by ONE call at
    a strain of zero, so the answer should be identical to round-off.
    """
    e = path(s, "cycle")
    worst = 0.0
    for signed in (e, -e):
        with_zero, _ = drive(mat_base.create(s.name, **s.props),
                             np.vstack([np.zeros((1, 6)), signed]))
        plain, _ = drive(mat_base.create(s.name, **s.props), signed)
        scale = max(float(np.max(np.abs(plain))), 1e-300)
        worst = max(worst, float(np.max(np.abs(with_zero[1:] - plain))) / scale)
    return Finding("an_evaluation_at_zero_costs_nothing", worst <= 1e-12, worst, 1e-12,
                   "worst |sigma(with the zero call) - sigma(without)| / peak |sigma|, "
                   "over the path and its mirror")


def the_stress_is_continuous_in_the_strain(material, s: Spec) -> Finding:
    """At a FROZEN state, the stress is Lipschitz in the strain with a
    constant of the order of the initial stiffness.

    This is the one that `bond_ceb` failed. The state is frozen deliberately:
    a jump in the constitutive FUNCTION is invisible along a path, because the
    state moves with it. Freeze it, step by 1e-6 of the amplitude, and a rule
    that changes branch discontinuously reports a slope hundreds of times E0.

    ONE DISCONTINUITY IS EXEMPT: the point's own DECLARED fracture. A law that
    implements `Material.failure` announces where it stops
    carrying load, and a brittle rupture is a step in the stress by
    construction -- `rebar` with a `rupture_strain` reports 137 E0 at exactly
    that strain, which is the model, not a defect. So an interval is skipped
    when the failure fraction REACHES ONE inside it, or is already one. Note
    how narrow that is: a fraction that merely grows is not exempt, so `cdp`'s
    softening, which climbs towards `damage_max` and never latches, is still
    asked the question over its whole path. Nothing else is forgiven.

    That the exemption is needed at all is worth recording rather than hiding.
    A latched stress drop is the shape v0.1 showed can trap an explicit
    solver in a limit cycle at the bond interface, and it is the argument for
    the paper's degradation curve -- whose stress falls continuously to `f_uf`
    BEFORE the fibre is cut -- over a bare latch.
    """
    eps = path(s, "cycle")
    state = material.init_state(1)
    E0 = abs(stiffness(material, s))
    worst = 0.0
    exempt = 0
    m = 8                     # sub-samples across each step of the path
    for i, step in enumerate(eps):
        # BOTH evaluations start from the same snapshot. Evaluating the second
        # one from the state the first left behind is not a frozen state at
        # all: `viscoelastic` keeps the previous strain, so the two calls saw
        # increments of 2e-5 and 1e-9 and the check reported a slope 67 times
        # E0 for a law that is linear.
        #
        # And the interval is the WHOLE STEP, walked in m pieces, not a tiny
        # nudge from the path point. A nudge only sees a jump if a path point
        # happens to land within the nudge of it: the planted discontinuity in
        # tests/test_material_invariants.py sat between two samples and was
        # missed. Sub-sampling the step means whichever piece contains the jump
        # reports it at m/|de| times its height, wherever in the step it is.
        frozen = state.copy()
        nxt = eps[i + 1] if i + 1 < len(eps) else step + (step - eps[i - 1])
        ts = np.linspace(0.0, 1.0, m + 1)
        de = np.linalg.norm(nxt - step) / m
        if de > 0.0:
            probes, dead = [], []
            for t in ts:
                st = frozen.copy()
                probes.append(material.update(
                    (step + t * (nxt - step))[None, :], st, 1.0)[0])
                f = material.failure(st)
                dead.append(f is not None and float(f[0]) >= 1.0 - 1e-12)
            probes = np.array(probes)
            for j in range(m):
                if dead[j] or dead[j + 1]:
                    exempt += 1
                    continue
                worst = max(worst, float(np.linalg.norm(
                    probes[j + 1] - probes[j])) / de)
        material.update(step[None, :], state, 1.0)     # advance the real path
    value = worst / max(E0, 1e-300)
    detail = "largest frozen-state slope, over E0"
    if exempt:
        detail += f" ({exempt} intervals at or past the declared fracture)"
    return Finding("the_stress_is_continuous_in_the_strain", value <= 2.0,
                   value, 2.0, detail)


def a_closed_cycle_does_not_create_energy(material, s: Spec) -> Finding:
    """The work round a closed strain cycle is the dissipation, and it is
    non-negative. A law that gains energy here is an instability that no time
    step can fix."""
    eps = path(s, "cycle")
    stress, _ = drive(material, eps)
    E0 = abs(stiffness(material, s))
    scale = E0 * s.amplitude ** 2
    value = work(stress[:, 0], eps) / max(scale, 1e-300)
    return Finding("a_closed_cycle_does_not_create_energy", value >= -1e-9,
                   value, -1e-9, "work round the cycle, over E0 * amplitude^2")


def unloading_is_no_stiffer_than_the_first_loading(material, s: Spec) -> Finding:
    """Damage and plasticity take stiffness away and never add it.

    The secant of the first unloading step against the measured initial
    stiffness. Equality is the common case -- plasticity unloads elastically --
    so the limit is one, with room for the discretisation of a curve.

    A viscous law is exempt: on a reversal its viscous stress opposes the new
    direction and the secant is steeper than E for a reason that is not
    stiffness at all (case_12, `viscosity: 1.0`, reported 1.19).
    """
    why = rate_dependence(material, s)
    if why:
        return Finding("unloading_is_no_stiffer_than_the_first_loading",
                       True, 0.0, 0.0, why + ": the secant carries viscous stress",
                       skipped=True)
    eps = path(s, "unload")
    stress, _ = drive(material, eps)
    n = len(eps) // 2
    d = s.direction
    top, next_ = stress[n - 1, 0] @ d, stress[n, 0] @ d
    de = (eps[n] - eps[n - 1]) @ d
    secant = (next_ - top) / de if de != 0.0 else 0.0
    E0 = stiffness(material, s)
    value = abs(secant) / max(abs(E0), 1e-300)
    return Finding("unloading_is_no_stiffer_than_the_first_loading",
                   value <= 1.02, value, 1.02,
                   "unloading secant over the initial stiffness")


def the_points_do_not_talk_to_each_other(material, s: Spec) -> Finding:
    """Four points with four different histories, driven together, give what
    each of them gives alone.

    Every law here is vectorised over the (n_points, n_state) array. One
    reduction without an `axis=` -- a `np.max`, an `np.any` -- and the worst
    point in a mesh sets the response of all of them.
    """
    scales = np.array([0.25, 0.5, 0.75, 1.0])
    base = path(s, "cycle")
    together = base[:, None, :] * scales[None, :, None]
    joint, _ = drive(material, together, n_points=4)
    worst = 0.0
    scale = 0.0
    for k, f in enumerate(scales):
        alone, _ = drive(material, base * f)
        worst = max(worst, float(np.max(np.abs(joint[:, k] - alone[:, 0]))))
        scale = max(scale, float(np.max(np.abs(alone))))
    value = worst / max(scale, 1e-300)
    return Finding("the_points_do_not_talk_to_each_other", value <= 1e-12,
                   value, 1e-12, "four histories together vs one at a time")


def the_time_step_does_not_change_a_rate_independent_law(material, s: Spec) -> Finding:
    """The same strain path at a thousand times the time step.

    Only the strain sequence is held fixed, so this asks one question and not
    two: does `dt` enter a law that declared it does not. A rate-dependent law
    is exempt and says so.
    """
    why = rate_dependence(material, s)
    if why:
        return Finding("the_time_step_does_not_change_a_rate_independent_law",
                       True, 0.0, 0.0, why, skipped=True)
    eps = path(s, "cycle")
    slow, _ = drive(material, eps, dt=1.0e-3)
    fast, _ = drive(material, eps, dt=1.0)
    scale = max(float(np.max(np.abs(slow))), 1e-300)
    value = float(np.max(np.abs(slow - fast))) / scale
    return Finding("the_time_step_does_not_change_a_rate_independent_law",
                   value <= 1e-12, value, 1e-12, "dt = 1e-3 against dt = 1")


def rotating_the_strain_rotates_the_stress(material, s: Spec) -> Finding:
    """Isotropy: run the whole path in a rotated frame and the stress is the
    rotated stress.

    The material half of v0.1's "a rigid rotation is free", and it catches
    the same thing: an axis assumed somewhere inside a return map. An
    anisotropic law is exempt -- for `orthotropic` a failure here would be the
    law working.
    """
    if s.kinematics != "solid_3d":
        return Finding("rotating_the_strain_rotates_the_stress", True, 0.0, 0.0,
                       f"{s.kinematics} law: no frame to rotate", skipped=True)
    if not s.isotropic:
        return Finding("rotating_the_strain_rotates_the_stress", True, 0.0, 0.0,
                       "anisotropic by declaration", skipped=True)
    from ..core.rotations import exp_map

    R = exp_map(np.array([[0.4, -0.7, 0.5]]))[0]
    eps = path(s, "cycle")
    rotated = np.array([to_voigt(R @ to_tensor(e) @ R.T) for e in eps])
    plain, _ = drive(material, eps)
    turned, _ = drive(material, rotated)
    want = np.array([to_voigt(R @ stress_to_tensor(sg) @ R.T, engineering=False)
                     for sg in plain[:, 0]])
    scale = max(float(np.max(np.abs(plain))), 1e-300)
    value = float(np.max(np.abs(turned[:, 0] - want))) / scale
    return Finding("rotating_the_strain_rotates_the_stress", value <= 1e-10,
                   value, 1e-10, "path run in a rotated frame")


def damage_does_not_heal(material, s: Spec) -> Finding:
    """On a MONOTONIC path the damage measure never falls.

    Not on a cycle: `cdp` recovers stiffness when a crack closes under
    compression (`wc`), so its `d` legitimately falls there and the check
    would be asserting the opposite of the model. Monotone loading has no
    closure in it, so the question is clean.
    """
    eps = path(s, "monotonic")
    state = material.init_state(1)
    trace = []
    for step in eps:
        material.update(step[None, :], state, 1.0)
        d = material.damage_measure(state)
        if d is None:
            return Finding("damage_does_not_heal", True, 0.0, 0.0,
                           "the law has no damage measure", skipped=True)
        trace.append(float(d[0]))
    drop = float(np.min(np.diff(trace))) if len(trace) > 1 else 0.0
    span = max(max(trace) - min(trace), 1e-300)
    value = -min(drop, 0.0) / span
    return Finding("damage_does_not_heal", value <= 1e-12, value, 1e-12,
                   "largest fall in the damage measure, over its range")


def failure_is_irreversible(material, s: Spec) -> Finding:
    """`Material.failure` may never fall -- on ANY path, including a cycle.

    The stricter twin of `damage_does_not_heal`, and the reason the two are
    separate checks. Damage is allowed to recover: `cdp` restores stiffness
    when a crack closes (`wc`), so its `d` legitimately falls and that check
    is restricted to a monotonic path. A FAILURE fraction is a latch -- what
    is dead stays dead -- so it is asked on the closed cycle, where a measure
    built on a recovering variable has nowhere to hide.

    A law with no notion of failure is skipped, not passed: `None` and zero
    mean different things one layer up.
    """
    eps = path(s, "cycle")
    state = material.init_state(1)
    trace = []
    for step in eps:
        material.update(step[None, :], state, 1.0)
        f = material.failure(state)
        if f is None:
            return Finding("failure_is_irreversible", True, 0.0, 0.0,
                           "the law has no failure measure", skipped=True)
        value = float(f[0])
        if not 0.0 - 1e-12 <= value <= 1.0 + 1e-12:
            return Finding("failure_is_irreversible", False, value, 1.0,
                           "a failure fraction lies in [0, 1]")
        trace.append(value)
    drop = float(np.min(np.diff(trace))) if len(trace) > 1 else 0.0
    span = max(max(trace) - min(trace), 1e-30)
    value = -min(drop, 0.0) / span
    return Finding("failure_is_irreversible", value <= 1e-12, value, 1e-12,
                   "largest fall in the failure fraction, over its range")


def the_crack_band_energy_does_not_depend_on_the_element_size(
        material, s: Spec) -> Finding:
    """A regularised law dissipates the same energy per unit crack AREA at
    every element size.

    That is the whole content of the word `regularised` in the catalogue: the
    softening branch is stretched by l_e so that energy per volume times l_e
    is the fracture energy. Run the same path bound to two element sizes and
    compare the products.
    """
    if not mat_base.definition(s.name).regularised:
        return Finding("the_crack_band_energy_does_not_depend_on_the_element_size",
                       True, 0.0, 0.0, "not a regularised law", skipped=True)
    l_ref = float(getattr(material, "l_ref", 1.0))
    E = float(getattr(material, "E", 0.0))
    nu = float(getattr(material, "nu", 0.2))
    if E <= 0.0:
        return Finding("the_crack_band_energy_does_not_depend_on_the_element_size",
                       True, 0.0, 0.0, "no elastic modulus to subtract with",
                       skipped=True)
    # THE ENERGY IS PER UNIT CRACK AREA, so it is the integral of stress
    # against the crack OPENING w = (eps - sigma/E) l, not against the strain:
    # the elastic part of the strain is recoverable, it is the same at every
    # element size, and multiplied by l it is a term that scales like l and
    # swamps the answer. The first version of this check integrated against
    # the strain over a path eight times the spec amplitude and reported a
    # 55% size dependence for a law whose own tests show it is objective --
    # all of it the elastic remainder, because tensile damage is capped at
    # `damage_max` and the residual stress keeps doing work for ever.
    #
    # Each size is driven to the same final OPENING for the same reason.
    direction = np.array([1.0, -nu, -nu, 0.0, 0.0, 0.0])
    peak = float(getattr(material, "sigma_t0", 0.0)) or 0.0
    w_end = 8.0 * peak / E * l_ref if peak else 8.0 * abs(s.amplitude) * l_ref
    energies = []
    for size in (l_ref, 2.0 * l_ref):
        fresh = mat_base.create(s.name, **s.props)
        fresh.bind_points(np.array([size]))
        eps_max = peak / E + w_end / size
        scale = np.linspace(0.0, eps_max, 2001)[1:]
        eps = scale[:, None] * direction[None, :]
        stress, _ = drive(fresh, eps)
        axial = stress[:, 0, 0]
        opening = (eps[:, 0] - axial / E) * size
        energies.append(float(np.trapezoid(axial, opening)))
    value = abs(energies[0] - energies[1]) / max(abs(np.mean(energies)), 1e-300)
    return Finding("the_crack_band_energy_does_not_depend_on_the_element_size",
                   value <= 0.05, value, 0.05,
                   f"G at l and 2l: {energies[0]:.4g}, {energies[1]:.4g} J/m2")


def the_envelope_stays_under_its_own_hardening_asymptote(material, s: Spec) -> Finding:
    """A law that declares (Fy, b, E) has declared where its envelope goes:

        sigma_asym(eps) = Fy + b E (eps - Fy/E)

    A monotonic push may approach that line FROM BELOW -- the rounded knee is
    the whole point of Menegotto-Pinto -- but it may not cross it, because the
    line is what the three numbers MEAN. `b` is the post-yield tangent a
    tension test measured; a curve that ends up above `Fy + bE(eps-eps_y)` is
    reporting a steel the analyst did not describe.

    WHY IT IS WORTH A QUESTION OF ITS OWN. Nothing else here would see it. The
    law is continuous, monotone, conservative, rate independent and correct on
    unloading; every other check passes. It is only when you put the curve next
    to the two numbers that wrote it that the gap appears -- and in a section
    it is not small. A prestressing strand is at its plateau stress for the
    whole of a beam's plastic range, so a percent on the plateau is a percent
    on the flexural capacity, every time, in one direction
    (where the model of a corroded-beam campaign's control specimen carries it).

    SKIPPED unless the law declares all three. A bilinear law (`rebar`) has no
    `b` and cannot overshoot anyway; a law with its own ultimate strength and a
    degradation curve (`steel_cyclic`) is describing a different envelope, and
    asking it this question would be asking the wrong one.
    """
    name = "the_envelope_stays_under_its_own_hardening_asymptote"
    fy = getattr(material, "Fy", None)
    if fy is None:
        fy = getattr(material, "sy", None)
    b = getattr(material, "b", None)
    e0 = getattr(material, "E", None)
    if s.kinematics != "uniaxial" or fy is None or b is None or e0 is None:
        return Finding(name, True, 0.0, 1e-3,
                       "the law declares no (Fy, b, E) triple", skipped=True)
    fy, b, e0 = float(fy), float(b), float(e0)
    eps_y = fy / e0
    # far enough past yield that the transition has run its course: at R0 = 20
    # the curve is on its asymptote within a few eps_y
    eps = np.linspace(eps_y, 20.0 * eps_y, 400)
    # the PASSED instance, not a fresh one built from the name: `drive` takes
    # a virgin state anyway, and driving the object handed in is what lets a
    # planted defect be planted (tests/test_material_invariants.py)
    stress, _ = drive(material, eps[:, None] * s.direction[None, :])
    sig = np.abs(stress[:, 0, 0])
    over = float(np.max(sig - (fy + b * e0 * (eps - eps_y)))) / fy
    return Finding(name, over <= 1e-3, max(over, 0.0), 1e-3,
                   "worst (sigma - [Fy + bE(eps - eps_y)]) / Fy, monotonic to "
                   "20 eps_y")


CHECKS = (
    the_virgin_state_is_free,
    an_evaluation_at_zero_costs_nothing,
    the_stress_is_continuous_in_the_strain,
    a_closed_cycle_does_not_create_energy,
    unloading_is_no_stiffer_than_the_first_loading,
    the_points_do_not_talk_to_each_other,
    the_time_step_does_not_change_a_rate_independent_law,
    rotating_the_strain_rotates_the_stress,
    damage_does_not_heal,
    failure_is_irreversible,
    the_crack_band_energy_does_not_depend_on_the_element_size,
    the_envelope_stays_under_its_own_hardening_asymptote,
)


def report(name: str) -> List[Finding]:
    """Every question that applies, asked of one registered law."""
    return report_spec(spec(name))


def report_spec(s: Spec) -> List[Finding]:
    """The same, for a `Spec` built from a user's own material block."""
    out = []
    for check in CHECKS:
        # a fresh law per check: a state left behind by one question is not
        # the starting point of the next one
        out.append(check(mat_base.create(s.name, **s.props), s))
    return out
