"""Selective mass scaling: the least mass that lets an explicit run take a
chosen step.

WHY. The explicit step is set by the one degree of freedom with the largest
stiffness-to-mass ratio, and in a real bridge model that DOF is almost never
one anybody cares about. On the Xihoumen model (810 nodes, 1207 members) it
is the ROTATION of a stay-anchor node inside the tower: the node hangs on a
stiff massless link, carries only pin-ended stays, and its rotary inertia is
the link's own few kg m^2 -- which put the step at 1.07e-7 s, 2.3 hours of
wall time per second of bridge. Nothing that node does is part of any answer.

WHAT. Every free DOF whose stable step `dt_safety * sqrt(m / k_row)` is below
the target gets exactly the mass that lifts it to the target, and nothing else
does. Translational mass is added ISOTROPICALLY per node (the largest of the
three), so a node's mass stays a scalar and a rigid-body motion stays a
rigid-body motion; rotary inertia likewise. The same Gershgorin row sum
`sum |K_ij|` the solvers use for the step is used here, so the step they
compute afterwards is at least the target.

THE ADDED MASS IS REAL MASS. It goes into `physical_mass`, so gravity weighs
it and a ground motion accelerates it. The alternative -- inertia without
weight -- would make a support shaken as a rigid body push those nodes with
less force than their inertia needs, which is a spurious internal force at
exactly the stiff links where it is largest. So the model is the model plus
these masses, the case says so, and the run reports how much was added. A
relaxation run of the same case adds the same masses (they depend only on the
initial stiffness), so a dead state computed by relaxation is in equilibrium
under the same weight a dynamic run from it will carry.

THE CAP. `max_added_fraction` (default 1%) of the model's translational mass:
beyond it the request is refused rather than obeyed, because a target the
model cannot reach without changing its dynamics is not a speed setting any
more.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:                          # pragma: no cover
    from ..io.schema import MassScalingSpec

import numpy as np


def scale(physical_mass: np.ndarray, row: np.ndarray, free: np.ndarray, spec: "Optional[MassScalingSpec]",
          dt_safety: float) -> Optional[dict]:
    """Add the mass in place; return the report, or None when not asked.

    physical_mass, row, free: flat arrays over the solver's DOFs, six per node
    (ux, uy, uz, rx, ry, rz)."""
    if not spec:
        return None
    # `spec`: the typed solver.mass_scaling (io/schema.MassScalingSpec), or None
    target = float(spec.target_dt)
    cap = float(spec.max_added_fraction)
    m = physical_mass.reshape(-1, 6)
    r = np.asarray(row, float).reshape(-1, 6)
    active = (np.asarray(free, bool) & (np.asarray(row) > 1e-20)).reshape(-1, 6)
    need = r * (target / float(dt_safety)) ** 2
    short = np.where(active, np.maximum(need - m, 0.0), 0.0)
    trans = short[:, :3].max(axis=1)
    rot = short[:, 3:].max(axis=1)
    base = float(m[:, 0].sum())
    added = float(trans.sum())
    if base > 0 and added / base > cap:
        raise ValueError(
            f"mass scaling to dt = {target:.3e} s needs {added / 1e3:.1f} t, "
            f"{100 * added / base:.2f}% of the model's {base / 1e3:.1f} t, over the "
            f"{100 * cap:.2f}% allowed (`max_added_fraction`). A target this far "
            "above what the model supports changes its dynamics; choose a smaller "
            "`target_dt`, or raise the cap knowingly.")
    m[:, :3] += trans[:, None]
    m[:, 3:] += rot[:, None]
    return {"target_dt": target, "added_mass": added, "model_mass": base,
            "fraction": added / base if base else 0.0,
            "nodes": int(np.count_nonzero(trans)), "rotary_nodes": int(np.count_nonzero(rot)),
            "added_per_node": trans.copy()}


def summary_line(report: Optional[dict]) -> str:
    if not report:
        return ""
    return (f"\n  mass scaling  +{report['added_mass'] / 1e3:.1f} t "
            f"({100 * report['fraction']:.3f}% of {report['model_mass'] / 1e3:.1f} t) on "
            f"{report['nodes']} nodes, rotary inertia on {report['rotary_nodes']}, "
            f"for dt >= {report['target_dt']:.3e} s")


def rotary_diagonal(R, local) -> np.ndarray:
    """The global-axis diagonal of a member end's rotary inertia
    R^T diag(local) R, R's rows the member's local axes.
    The solvers keep one rotary mass per dof, so the off-diagonal terms of
    a skew member are not kept; for a member along a global axis the
    diagonal is the tensor."""
    R = np.asarray(R, float)
    return (R * R).T @ np.asarray(local, float)


def true_rotary(mass_step: np.ndarray, rotary: np.ndarray, row: np.ndarray, free: np.ndarray,
                extra_floor: Optional[np.ndarray] = None):
    """The mass with its rotary part replaced by the true inertia `rotary`
    (nn, 3), the explicit step untouched (the user's decision: change the
    inertia, keep the stable step).

    `mass_step` is the mass the step is taken from -- the old rotary rule,
    v0.4's scaling included. A free, attached rotation gets
    max(true, row * min(mass_step / row)): the least mass stable at that
    step, so no case slows down and none goes unstable. The floor is per
    axis, in the reference frame: a piece of debris, which can tumble out
    of it, has its three made one at the split (beam_solver
    `_debris_isotropic`). `extra_floor`
    (flat) adds another per-dof lower bound (a dashpot's 2m/c). Returns the
    new flat mass and how many rotations the floor lifted."""
    ms = np.asarray(mass_step, float).reshape(-1, 6)
    r = np.asarray(row, float).reshape(-1, 6)
    act = (np.asarray(free, bool).reshape(-1) & (np.asarray(row).reshape(-1) > 1e-20)).reshape(-1, 6)
    ratio = np.where(act, ms / np.maximum(r, 1e-300), np.inf)
    lo = float(ratio.min()) if np.isfinite(ratio).any() else 0.0
    floor = np.where(act[:, 3:], r[:, 3:] * lo, 0.0)
    if extra_floor is not None:
        floor = np.maximum(floor, np.where(act[:, 3:], np.asarray(extra_floor, float).reshape(-1, 6)[:, 3:], 0.0))
    rot = np.asarray(rotary, float).reshape(-1, 3)
    out = ms.copy()
    out[:, 3:] = np.maximum(rot, floor)
    return out.reshape(-1), int(np.count_nonzero(floor > rot * (1.0 + 1e-12)))
