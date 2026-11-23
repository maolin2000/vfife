"""The questions that do not depend on the element.

A model is a mesh, a material, a support and a load, and almost everything
that can be wrong with the four of them is wrong in one of a handful of ways
that no benchmark states and every benchmark suffers from. The library found
each of them the expensive way -- a case that would not converge, days of
bisection, and at the end a one-line defect that a generic question would
have asked on the day the feature was written:

    the force is not a gradient      the follower face moment of v0.1: its
                                     tangent was almost purely skew, kinetic
                                     damping (a descent method) thrashed on
                                     it, and the case stalled at load level
                                     26 of 40 with nothing in the output
                                     saying why
    a rigid motion is not free       the spatial frame of v0.1: a member
                                     rotated rigidly WITH A TWIST carried a
                                     spurious moment, so the 45-degree bend
                                     came out 2% stiff -- and the planar
                                     cases, which is all there were, could
                                     not see it
    the mass does not bound the step the recurring "it stalls at the first
                                     load level" of docs/convergence_debugging:
                                     the Underwood row sums are a bound on the
                                     REFERENCE configuration and a measurement
                                     everywhere else
    the reference state is not free  a residual stress from a mis-assembled
                                     precompute, which shows up as a case that
                                     "converges" to the wrong answer

So this module asks them, of a `state.Probe`, in a form that does not know
which solver it is holding. `report(probe)` runs every check that applies and
returns `Finding`s; `tests/test_invariants.py` runs it over every element in
`testing.models`, and `vfife probe` runs it over the user's own case, at the
state their own run stalled in.

WHAT A CHECK COSTS. The tangent is a finite difference over every degree of
freedom: 2 n force evaluations, a second or two at 250 dofs and hopeless at
10 000. That is the honest limit of this approach and the reason `report`
takes a `max_dofs`: past it the tangent checks are skipped and say so rather
than pretending. The rigid-motion and reference-state checks are three force
evaluations and always run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Finding:
    """One question, its answer, and whether the answer is acceptable."""
    name: str
    passed: bool
    value: float
    limit: float
    detail: str = ""
    #: skipped checks carry passed=True and this set
    skipped: bool = False

    def __str__(self) -> str:
        mark = "SKIP" if self.skipped else ("PASS" if self.passed else "FAIL")
        return (f"  {mark}  {self.name}: {self.value:.3e}"
                + (f" (limit {self.limit:.1e})" if not self.skipped else "")
                + (f"  {self.detail}" if self.detail else ""))


def tangent(p, x=None, h=None, external=True):
    """The finite-difference tangent at x, as (K_int, K_ext).

    Central differences of the internal force and of MINUS the external one,
    so that K = K_int + K_ext is the tangent of the residual's negative and a
    stable equilibrium is a positive-definite K. The step is scaled by the
    model's own size: on a result read back from HDF5 the displacement is
    float32, and h = 1e-7 on a float32 field differences pure rounding -- the
    first run of this probe reported a 10% asymmetry that was not there.
    The size is the model's extent, not its largest coordinate: a column in
    survey coordinates (5e5 m east) was differenced at 0.05 m and its modes
    came out 1.8 to 3.7 times too high (`vfife modal`, `spectrum`; 1.4.1,
    as `buckling` since 1.1).
    """
    x = p.get() if x is None else np.asarray(x, float)
    if h is None:
        nodes = np.asarray(p.nodes, float)
        h = 1e-7 * max(1.0, float(np.ptp(nodes, axis=0).max()) if nodes.size else 1.0)
    n = p.n
    Ki = np.empty((n, n))
    Ke = np.zeros((n, n))
    for j in range(n):
        xp = p.perturb(x, j, h)
        xm = p.perturb(x, j, -h)
        Ki[:, j] = (p.internal(xp) - p.internal(xm)) / (2 * h)
        if external:
            Ke[:, j] = -(p.external(xp) - p.external(xm)) / (2 * h)
    return Ki, Ke


def is_a_gradient(p, x=None, K=None, tol=1e-6, what="internal") -> Finding:
    """|K - K^T| / |K| on the free dofs.

    A force with a potential has a symmetric tangent. A skew part means the
    work around a closed loop of states is not zero: kinetic damping, which
    zeroes the velocity when the kinetic energy peaks and is otherwise a
    descent on that potential, has nothing to descend, and the symptom in the
    log is a reset every hundred-odd iterations with the residual bouncing
    between 1e-4 and 1e-3 for the rest of the budget.

    1e-6 is loose against what a correct force scores (1e-9 or below, set by
    the finite difference) and tight against what a wrong one scores (the
    face moment of v0.1 scored 1.8, i.e. the tangent was almost entirely
    the skew part).
    """
    if K is None:
        Ki, Ke = tangent(p, x)
        K = Ki if what == "internal" else Ke
    free = p.free
    Kf = K[np.ix_(free, free)]
    mx = float(np.abs(Kf).max())
    if mx == 0.0:
        return Finding(f"{what} force is a gradient", True, 0.0, tol,
                       "no stiffness on the free dofs")
    v = float(np.abs(Kf - Kf.T).max()) / mx
    return Finding(f"{what} force is a gradient", v <= tol, v, tol)


def a_rigid_translation_is_free(p, x=None, tol=1e-9) -> Finding:
    """Move the DEFORMED body 0.3 of its size and ask again.

    From the reference state this catches a mis-assembled B matrix; from a
    deformed one it also catches a frame that is recovered from absolute
    coordinates rather than from the motion.
    """
    x = p.get() if x is None else np.asarray(x, float)
    f0 = p.internal(x)
    scale = float(np.abs(p.nodes).max()) or 1.0
    t = 0.3 * scale * np.array([1.0, -0.7, 0.4])
    f1 = p.internal(p.rigid(x, np.eye(3), np.zeros(3), t))
    ref = max(float(np.abs(f0).max()), _stiffness_scale(p))
    v = float(np.abs(f1 - f0).max()) / ref
    return Finding("a rigid translation is free", v <= tol, v, tol)


def a_rigid_rotation_is_free(p, x=None, angle=(0.4, -0.9, 0.6), tol=1e-8) -> Finding:
    """Turn the DEFORMED body about its centroid and ask again: the internal
    force must be the old one rotated, component by component.

    A deformed state and a rotation about all three axes at once is the
    combination that matters. This is exactly what the frame solver failed
    before v0.1: subtracting rotation vectors is exact in a plane, so a
    planar benchmark rotated rigidly and stayed force-free, and a member with
    a twist did not.
    """
    from ..core.rotations import exp_map

    x = p.get() if x is None else np.asarray(x, float)
    R = exp_map(np.asarray(angle, float))
    c = p.nodes.mean(axis=0)
    f0 = p.internal(x)
    f1 = p.internal(p.rigid(x, R, c, np.zeros(3)))
    want = f0.copy()
    for idx in (p.translation, p.rotation):
        if idx.size:
            want[idx] = (R @ f0[idx].reshape(-1, 3).T).T.ravel()
    ref = max(float(np.abs(f0).max()), _stiffness_scale(p))
    v = float(np.abs(f1 - want).max()) / ref
    return Finding("a rigid rotation is free", v <= tol, v, tol)


def the_reference_state_is_free(p, tol=1e-10) -> Finding:
    """Zero displacement, zero internal force -- unless the model declares a
    prestress or an initial state, which is why this is a check and not an
    assertion inside the solver."""
    f = p.internal(np.zeros(p.n))
    v = float(np.abs(f).max()) / max(_stiffness_scale(p), 1.0)
    return Finding("the reference state is force-free", v <= tol, v, tol)


def the_mass_bounds_the_step(p, x=None, K=None, safety=None, tol=1.0) -> Finding:
    """max omega^2 of (K_sym, M) against the central-difference limit 4 s^2.

    With dt = 1 the explicit step is stable while omega^2 < 4; the relaxation
    safety factor s enters as s^2, so the fictitious mass is meant to put the
    stiffest mode at 4 s^2. Reporting the ratio says in one number whether a
    stall is the step (ratio near or above 1) or something else -- for the
    rolled cantilever of v0.1 it was 0.49, which is what ruled the mass out
    and sent the search to the load.
    """
    import scipy.linalg as sla

    if K is None:
        Ki, Ke = tangent(p, x)
        K = Ki + Ke
    free = p.free
    Kf = K[np.ix_(free, free)]
    M = p.mass[free]
    if not np.all(M > 0):
        return Finding("the mass bounds the step", True, 0.0, tol,
                       "not a fictitious-mass relaxation", skipped=True)
    w2 = sla.eigh(0.5 * (Kf + Kf.T), np.diag(M), eigvals_only=True)
    s = safety if safety is not None else _safety(p)
    limit = 4.0 * s * s
    v = float(w2[-1]) / limit
    detail = f"max omega^2 = {w2[-1]:.4g} of {limit:.4g}"
    if v > tol and getattr(p.solver, "mass_kind", None) == "underwood":
        deformed = x is not None and float(np.abs(np.asarray(x, float)).max()) > 0
        if deformed:
            detail += ("; the Underwood row sums bound the REFERENCE "
                       "configuration and this one has moved -- `mass: eigenvalue` "
                       "is the bound that survives a rotation, at 1.5-2x the "
                       "iterations")
        else:
            detail += ("; try a smaller `relaxation_safety`, but measure -- "
                       "see docs/convergence_debugging.md")
    return Finding("the mass bounds the step", v <= tol, v, tol, detail)


def the_state_is_stable(p, x=None, K=None, tol=0.0) -> Finding:
    """The smallest eigenvalue of the symmetrised tangent, scaled.

    Not a defect on its own -- a snap-through case is unstable on purpose and
    a free body has zero modes -- but the one number that separates "the
    solver cannot find this equilibrium" from "this state has no stable
    equilibrium to find", which was the last open question of v0.1.
    """
    import scipy.linalg as sla

    if K is None:
        Ki, Ke = tangent(p, x)
        K = Ki + Ke
    free = p.free
    Kf = K[np.ix_(free, free)]
    M = p.mass[free]
    if not np.all(M > 0):
        M = np.ones(Kf.shape[0])
    w2, vec = sla.eigh(0.5 * (Kf + Kf.T), np.diag(M))
    v = float(w2[0])
    detail = ""
    if v < 0:
        d = np.zeros(p.n)
        d[free] = vec[:, 0]
        share = np.abs(d[p.translation].reshape(-1, 3)).sum(axis=0)
        share = share / max(share.sum(), 1e-300)
        detail = f"softest mode x/y/z {np.round(share, 3)}"
    return Finding("the state is stable", v >= -abs(tol), v, -abs(tol), detail)


def _stiffness_scale(p) -> float:
    """A force scale for the tolerances: what the model's own stiffness makes
    of a strain of a thousandth.

    A UNIFORM displacement will not do, which is the joke this function is
    the punchline of -- the first version used one and divided every
    rigid-motion residual by the force a rigid motion produces, which is
    zero. A uniform dilation about the centroid strains every element.
    """
    if getattr(p, "_scale", None) is None:
        L = float(np.abs(p.nodes).max()) or 1.0
        x = np.zeros(p.n)
        x[p.translation] = (1e-3 * (p.nodes - p.nodes.mean(axis=0)) / L).ravel()
        p._scale = max(float(np.abs(p.internal(x)).max()), 1e-300)
    return p._scale


def _relative_residual(p, x=None) -> float:
    x = p.get() if x is None else np.asarray(x, float)
    e = float(np.abs(p.external(x)).max())
    r = float(np.abs(p.residual(x)).max())
    return r / e if e > 0 else (r / max(_stiffness_scale(p), 1e-300))


def _safety(p) -> float:
    try:
        return float(p.solver.p.solver.relaxation_safety)      # the typed case
    except AttributeError:
        return 0.6


#: the checks that need the O(n^2) tangent
TANGENT_CHECKS = ("internal force is a gradient", "external force is a gradient",
                  "the mass bounds the step", "the state is stable")


def self_stress(case) -> str:
    """Why this case's reference configuration is MEANT not to be force-free,
    or "": a tendon's or a bar's prestress, an unstressed
    length, a temperature field, creep, shrinkage or relaxation, a state read
    from another run. `probe` said FAIL "the reference state is force-free"
    (0.88) on every prestressed girder, and "the state is stable" FAIL at an
    arc of a state that is not an equilibrium of anything."""
    if isinstance(case, str):                 # the case file's text, as written
        import yaml
        d = yaml.safe_load(case) or {}
    else:
        d = case.model_dump() if hasattr(case, "model_dump") else dict(case or {})
    why = []
    for name, sec in (d.get("sections") or {}).items():
        if (sec or {}).get("tendons"):
            why.append(f"tendons in section {name!r}")
    for a in d.get("assign") or []:
        if a and (a.get("prestress") is not None or a.get("unstressed_length") is not None
                  or a.get("unstressed_lengths") is not None or a.get("jack") is not None):
            why.append(f"prestress on {a.get('region') or a.get('reinforcement')!r}")
    loads = list(d.get("loads") or []) + [l for st in d.get("steps") or [] for l in (st or {}).get("loads") or []]
    if any((l or {}).get("temperature") is not None for l in loads):
        why.append("a temperature load")
    for name, mat in (d.get("materials") or {}).items():
        for k in ("creep", "shrinkage", "relaxation"):
            if (mat or {}).get(k) is not None:
                why.append(f"{k} on {name!r}")
    if ((d.get("solver") or {}).get("initial_state")):
        why.append("solver.initial_state")
    return ", ".join(dict.fromkeys(why))


def report(p, x=None, max_dofs: int = 4000, external: bool = True,
           reference_state: bool = True, self_stressed: str = "") -> List[Finding]:
    """Every check that applies, at the state x (the solver's own by default).

    The three-force-evaluation checks always run; the tangent ones run when
    the model has at most `max_dofs` degrees of freedom and are reported as
    skipped above it.

    THE INTERNAL TANGENT IS ONLY CONCLUSIVE NEAR EQUILIBRIUM. A co-rotational
    element's tangent is unsymmetric away from equilibrium in proportion to
    the out-of-balance force -- classical, not a defect -- so at a state whose
    relative residual is above a percent the symmetry is measured and reported
    as inconclusive rather than as a failure. The EXTERNAL tangent has no such
    excuse: a load is a function of the state alone, and its symmetry is a
    verdict at any state.

    `self_stressed` (see `self_stress`): the model carries a self-stress on
    purpose, and the state `x` is not an equilibrium the run found; the
    force-free and stability checks are then reported as skipped, with why.
    """
    if reference_state and self_stressed:
        out = [Finding("the reference state is force-free", True, 0.0, 0.0,
                       f"self-stressed ({self_stressed}), check skipped", skipped=True)]
    else:
        out = ([the_reference_state_is_free(p)] if reference_state else [])
    out.append(a_rigid_translation_is_free(p, x))
    if p.rotation.size or p.family in ("solid", "truss"):
        out.append(a_rigid_rotation_is_free(p, x))
    if p.n > max_dofs:
        for name in TANGENT_CHECKS:
            out.append(Finding(name, True, 0.0, 0.0,
                               f"{p.n} dofs is past the {max_dofs} the finite-"
                               "difference tangent can afford", skipped=True))
        return out
    Ki, Ke = tangent(p, x, external=external)
    internal = is_a_gradient(p, K=Ki, what="internal")
    r = _relative_residual(p, x)
    if not internal.passed and r > 1e-2:
        internal.skipped = True
        internal.passed = True
        internal.detail = (f"inconclusive: the relative residual here is {r:.1e}, "
                           "and a co-rotational tangent is unsymmetric away from "
                           "equilibrium in proportion to it")
    out.append(internal)
    if external and np.abs(Ke).max() > 0:
        out.append(is_a_gradient(p, K=Ke, what="external"))
    K = Ki + Ke
    out.append(the_mass_bounds_the_step(p, x, K=K))
    if self_stressed:
        out.append(Finding("the state is stable", True, 0.0, 0.0,
                           f"self-stressed ({self_stressed}) and not an equilibrium of it, "
                           "check skipped; probe a converged frame with --result", skipped=True))
    else:
        out.append(the_state_is_stable(p, x, K=K))
    return out
