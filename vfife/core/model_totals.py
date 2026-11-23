"""What a model weighs and carries, in totals (1.1, `vfife summary`).

Taken from the built solver once, before the run, and written into the
result's root attributes (`model_mass`, `model_weight`, `model_line_load`),
so that `vfife summary` can set the reactions beside them without the case
file. The point is a check a reader makes in one glance: under a static dead
load the vertical reaction is the weight plus the line loads, and a model
that weighs 2.8 times what its sections say (the fibre mass
taken from one material's density) shows it here first.

* `mass`: the translational mass summed over every particle, per global
  axis -- the members' own, embedded bars', `nodal_masses`, hydrodynamic and
  `with_mass` masses, and what selective mass scaling added (real mass, by
  that device's design: it weighs and it moves).
* `weight`: the force the `gravity` loads put on the model at the
  end of the step (each at its amplitude's value there): g times the mass that weighs (the weightless part taken off).
  The loads are those of the LAST step -- the model-level ones and that
  step's own -- because the reactions it is set beside are the last frame's.
  None where this cannot be said from the lumped mass: a truss or solid with
  gravity on part of it (they weigh the elements wholly inside the region,
  not the nodes'), a solid with embedded bars (their mass is lumped, their
  weight is not applied).
* `line_load`: the same steps' `line_load`s, q times the length of the
  members they act on, in global axes.
* `point_load`: the same steps' nodal forces (`vector` loads) (1.2).

Every load is taken at its amplitude's value at the step's end (1.2; 1.1
took them at full amplitude).

Each is None when this model cannot say (a solver without the attribute it
is read from); `vfife summary` then leaves the line out.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _translational(solver) -> Optional[np.ndarray]:
    pm = getattr(solver, "physical_mass", None)
    if pm is None:
        pm = getattr(solver, "mass", None)
    nn = getattr(solver, "nn", None)
    if pm is None or not nn:
        return None
    pm = np.asarray(pm, float)
    if pm.size % nn:
        return None
    return pm.reshape(nn, -1)[:, :3]


def totals(solver, p: dict) -> dict:
    """See the module's docstring. A model whose steps build and remove
    members (1.2) is weighed as it stands when the last step ends."""
    if solver.plan is not None:
        from .analysis_steps import as_finished
        with as_finished(solver) as built:
            solver._totals_built = built
            try:
                return _totals(solver, p)
            finally:
                solver._totals_built = None
    return _totals(solver, p)


def _totals(solver, p: dict) -> dict:
    out = {"mass": None, "weight": None, "line_load": None, "point_load": None}
    m = _translational(solver)
    if m is None:
        return out
    out["mass"] = m.sum(axis=0)
    from .added_mass import weighed
    w = weighed(m, getattr(solver, "weightless_mass", None))
    steps = p.get("steps") or []
    loads = list(p.get("loads") or []) + list((steps[-1] or {}).get("loads") or [] if steps else [])
    # each at its amplitude's value at the step's end, not at full amplitude
    # (1.2): a pushover on a cyclic table that ends at zero read 0.73 in the
    # summary's check of a correct run
    try:
        from .amplitude import build as _amps
        amps = _amps(p.get("amplitudes"))
    except Exception:
        amps = {}

    def at_end(ld):
        a = amps.get(ld.get("amplitude")) if ld.get("amplitude") else None
        return float(a(1.0)) if a is not None else 1.0
    region = getattr(solver, "_region", None)
    lumped_regions = hasattr(solver, "_members_in") and type(solver).__name__ != "TrussSolver"
    try:
        weight = np.zeros(3)
        for ld in loads:
            if ld.get("gravity") is None:
                continue
            name = ld.get("region", "all")
            if name != "all" and not lumped_regions:
                raise ValueError("gravity on part of a truss or solid: weighed by element")
            idx = np.arange(len(w)) if name == "all" or region is None else np.asarray(region(name))
            weight += at_end(ld) * np.asarray(ld["gravity"], float) * w[idx].sum(axis=0)
        if any(ld.get("gravity") is not None for ld in loads) and getattr(solver, "rebar", None) \
                and not hasattr(solver, "_members_in"):
            raise ValueError("a solid's embedded bars are lumped but not weighed")
        out["weight"] = weight
    except Exception:
        pass
    try:
        out["line_load"] = _line_loads(solver, loads, region, at_end)
    except Exception:
        pass
    try:
        # the nodal forces (`vector`) (1.2): with them the
        # summary's check covers a pushed pier or a loaded deck too
        point = np.zeros(3)
        for ld in loads:
            if ld.get("vector") is None:
                continue
            v = np.asarray(ld["vector"], float)
            n = len(np.asarray(region(ld.get("region", "all")))) if ld.get("per_node") else 1
            point += at_end(ld) * v * n
        out["point_load"] = point
    except Exception:
        pass
    return out


def _line_loads(solver, loads, region, at_end=lambda ld: 1.0) -> Optional[np.ndarray]:
    chosen = [ld for ld in loads if ld.get("line_load") is not None]
    if not chosen:
        return np.zeros(3)
    conn = getattr(solver, "_elements0", None)
    conn = solver.mesh.elements if conn is None else conn
    nodes = np.asarray(solver.mesh.nodes, float)
    members = np.asarray(conn)[:, :2]
    nn = getattr(solver, "nn", len(nodes))
    total = np.zeros(3)
    for ld in chosen:
        idx = np.asarray(region(ld.get("region", "all")))
        inside = np.zeros(nn, bool); inside[idx] = True
        e = np.flatnonzero(inside[members].all(axis=1))
        built = getattr(solver, "_totals_built", None)
        if built is not None:
            e = e[np.asarray(built)[e]]
        L = np.linalg.norm(nodes[members[e, 1]] - nodes[members[e, 0]], axis=1)
        q = at_end(ld) * np.asarray(ld["line_load"], float)
        if ld.get("coordinate_system", "global") == "local":
            R0 = getattr(solver, "R0", None)
            if R0 is None:
                return None
            # ql = R q for a global q (beam_solver), so q = R^T ql
            total += np.einsum("eji,j,e->i", np.asarray(R0)[e], q, L)
        else:
            total += q * L.sum()
    return total
