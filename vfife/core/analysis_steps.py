"""Analysis steps (1.2): each step its own procedure, members born and
removed, loads and constraints that outlive their step.

A step of a line model may say

    procedure: relaxation | dynamic | static     (default: solver.mode)
    activate:  [regions]     members born in this step
    birth:     tangent | design
    deactivate: [regions]    members removed in this step
    release:   [regions]     inherited constraints on these nodes freed
               [link names]  rigid links freed (1.21)
    rigid_links: [...]       rigid links made at the step's start (1.21)

and its own loads and constraints may say `keep: true`. A case that says
none of this runs exactly as before: `plan` returns None and nothing here is
called again.

WHAT OTHER PROGRAMS DO, AND WHAT IS TAKEN FROM THEM

* Abaqus' `*MODEL CHANGE`: a removed region's forces on the rest are ramped
  to zero over the step, not dropped; a reactivated element is born
  STRAIN FREE in the configuration it finds; distributed loads on it come
  on with it. All three are what this module does.
* OpenSees' `-useInitialDisp`: an element born late remembers its nodes'
  displacements and subtracts them. Its beam version subtracts only
  translations; here the whole local deformation -- the stretch and both
  ends' rotations -- is the birth offset, carried in the slot a temperature
  uses (`_thermal_d`), so every force path sees it.
* A node only the new members reach is, in Abaqus, wherever it was. A
  cantilever cast segment by segment is not built that way: the form is set
  on the tangent of the tip. `birth: tangent` (the default) carries such a
  node rigidly with the standing node it is reached from; `design` leaves
  it at its coordinates.

WHAT A MEMBER THAT IS NOT BUILT IS
    No stiffness (k_local zero, `member_active` false, the batches rebuilt),
    no translational mass on its nodes (its share and its `with_mass` line
    mass are taken off, so it does not weigh), no line load. A node no built
    member reaches is DORMANT: held in every dof, its reaction not reported.
    The member's rotary inertia stays on its nodes -- it changes no static
    answer, and a dynamic step's stable step is taken with it.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from .dof import ROTATION, TRANSLATION

#: the amplitude a removal's or a release's force is taken off with
UNLOAD = "_unload"


class Plan:
    """What every step changes, member by member, read once."""

    def __init__(self, nel: int, nn: int, n_steps: int):
        self.activate: List[np.ndarray] = [np.zeros(0, np.int64)] * n_steps
        self.deactivate: List[np.ndarray] = [np.zeros(0, np.int64)] * n_steps
        self.release: List[np.ndarray] = [np.zeros(0, np.int64)] * n_steps
        #: the rigid links each step's `release` names (1.21)
        self.release_links: List[List[str]] = [[] for _ in range(n_steps)]
        self.procedure: List[str] = []
        self.built0 = np.ones(nel, bool)
        #: nodes freed by a release at or before step k (cumulative): what a
        #: MODEL-LEVEL constraint has lost by step k
        self.released_by: List[np.ndarray] = []
        self.changes = False


def plan(solver) -> Optional[Plan]:
    """The model changes of every step, checked against one another. None
    when no step changes the model, its procedure, or keeps anything."""
    steps = solver.steps or []
    base = solver.p.solver.mode
    if solver.p.static_run:
        base = "static"
    any_change = False
    for st in steps:
        if (st.get("activate") or st.get("deactivate") or st.get("release") or st.get("rigid_links")
                or (st.get("procedure") and st.get("procedure") != base)
                or st.get("total_time") is not None
                or any(x.get("keep") for x in list(st.get("loads") or []) + list(st.get("constraints") or []))):
            any_change = True
    jt = getattr(solver, "_jack_T", None)
    if jt is not None and np.any(jt > 0.0):
        any_change = True                 # a cable jacked in its step: the step has an end to freeze it at
    if not any_change:
        return None
    nel, nn = solver.nel, solver.nn
    pl = Plan(nel, nn, len(steps))
    first: Dict[int, str] = {}
    # the rigid links by name, and the step each is made in (-1: the model's)
    case = solver.p.case
    link_born = {lk.name: -1 for lk in case.rigid_links if lk.name}
    for k, st in enumerate(case.steps):
        for lk in st.rigid_links:
            if lk.name:
                link_born[lk.name] = k
    link_freed: Dict[str, int] = {}
    for k, st in enumerate(steps):
        pl.procedure.append(st.get("procedure") or base)
        for key in ("activate", "deactivate"):
            ids = []
            for r in st.get(key) or []:
                got = solver._members_of(r)
                if not len(got):
                    raise ValueError(f"steps[{k}].{key}: region {r!r} holds no complete member "
                                     "(the members of a group of lines; for a region given by "
                                     "points, the members with both nodes in it)")
                ids.append(got)
            arr = np.unique(np.concatenate(ids)) if ids else np.zeros(0, np.int64)
            getattr(pl, key)[k] = arr.astype(np.int64)
            for e in arr:
                first.setdefault(int(e), key)
        names = [r for r in st.get("release") or [] if r in link_born]
        for r in names:
            if r in solver.mesh.regions:
                raise ValueError(f"steps[{k}].release names {r!r}, a region and a rigid link both: "
                                 "rename one")
            if link_born[r] >= k:
                raise ValueError(f"steps[{k}].release frees rigid link {r!r}, which is made "
                                 + ("in a later step" if link_born[r] > k else "in this step"))
            if r in link_freed:
                raise ValueError(f"steps[{k}].release frees rigid link {r!r}, already freed in "
                                 f"steps[{link_freed[r]}]")
            link_freed[r] = k
        pl.release_links[k] = names
        rel = [np.asarray(solver._region(r), np.int64) for r in st.get("release") or [] if r not in link_born]
        pl.release[k] = np.unique(np.concatenate(rel)) if rel else np.zeros(0, np.int64)
    for e, kind in first.items():
        if kind == "activate":
            pl.built0[e] = False
    if jt is not None and np.any(jt > 0.0):
        # the step each jacked bar is stressed in must have an end the force is
        # reached at: a dynamic step's last force is read one increment short
        for k in range(len(steps)):
            ids = pl.activate[k] if k else np.union1d(pl.activate[0], np.flatnonzero(pl.built0))
            if pl.procedure[k] == "dynamic" and len(ids) and np.any(jt[ids] > 0.0):
                raise ValueError(f"steps[{k}] is dynamic and stresses a cable with `jack`: a jack is "
                                 "worked in a static or a relaxation step; build the cable in one")
    kinds = getattr(solver, "member_kind", None)
    built = pl.built0.copy()
    for k in range(len(steps)):
        a, d = pl.activate[k], pl.deactivate[k]
        if len(a) and np.any(built[a]):
            e = int(a[built[a]][0])
            raise ValueError(f"steps[{k}] activates member {e}, which is already built "
                             "(a member is born once; deactivate it first to build it again)")
        if len(d) and np.any(~built[d]):
            e = int(d[~built[d]][0])
            raise ValueError(f"steps[{k}] deactivates member {e}, which is not built at that step")
        is_fibre = getattr(solver, "_is_fiber", None)
        if len(a) and (kinds is not None or is_fibre is not None):
            # a connector of the frame models is born like any member, free of
            # stress where it stands (1.13); the fibre models'
            # are not yet
            conn = [int(e) for e in a if is_fibre is not None and not is_fibre[e]]
            if conn:
                raise ValueError(f"steps[{k}] activates connector member {conn[0]}: in a fibre model a "
                                 "connector born in a later step is not supported yet (its law reads "
                                 "displacements from the mesh); build it from the start, or model the "
                                 "stage as a beam or a bar")
        built[a] = True
        built[d] = False
    acc = np.zeros(nn, bool)
    for k in range(len(steps)):
        if len(pl.release[k]):
            held = np.zeros(nn, bool)
            for c in list(solver.p.get("constraints") or []) + [
                    c for j in range(k) for c in steps[j].get("constraints") or [] if c.get("keep")]:
                held[np.asarray(solver._region(c["region"]), np.int64)] = True
            if not held[pl.release[k]].any():
                raise ValueError(f"steps[{k}].release frees nodes no inherited constraint holds "
                                 "(a model-level one, or one an earlier step kept); a step's own "
                                 "constraint is that step's alone and needs no release")
        acc[pl.release[k]] = True
        pl.released_by.append(np.flatnonzero(acc))
    pl.changes = bool(any(len(x) for x in pl.activate + pl.deactivate + pl.release + pl.release_links)
                      or any(st.rigid_links for st in case.steps))
    return pl


# --------------------------------------------------------------------------- #
#  setup
# --------------------------------------------------------------------------- #

def install(solver) -> None:
    """After the masses are set: take the members that are born later out of
    the model, and remember what to put back."""
    pl = solver.plan
    solver.member_built = pl.built0.copy()
    from .step_state import Staging
    solver.staging = Staging()                 # the solver's own (core/step_state)
    solver.staging.birth_d = None
    solver.staging.k_built = (None if _is_fibre(solver) else np.array(solver.k_local, dtype=float, copy=True))
    solver.staging.k_ref_built = (None if getattr(solver, "_k_ref", None) is None
                           else np.array(solver._k_ref, dtype=float, copy=True))
    solver.staging.dormant = np.zeros(solver.nn, bool)
    solver.staging.step_extra = {}
    solver.staging.born_now = np.zeros(0, np.int64)
    solver.staging.clock = 0.0
    solver.staging.base_mode = solver.mode
    if any(len(x) for x in pl.activate + pl.deactivate + pl.release) and _is_fibre(solver) \
            and getattr(solver, "_relax", None):
        # creep and shrinkage on a fibre section run on each segment's own
        # clock since 1.8; a tendon's relaxation does not:
        # it is dated from the model's stressing age
        raise ValueError("steps that activate, deactivate or release in a fibre model whose tendons "
                         "relax: a tendon born late would relax from the model's stressing age; "
                         "not supported yet")
    #: each member's own concrete clock (1.2): its age is the model's + shift,
    #: and nothing is dated before `floor`, its age at birth
    solver.staging.age_shift = np.zeros(solver.nel)
    solver.staging.age_floor = np.full(solver.nel, -np.inf)
    if len(set(pl.procedure)) > 1:
        # the ledger is kept by the time-stepping loops only: a run that mixes
        # them audits part of its history and reads as an error (review, 1.2)
        solver.staging.mixed_procedures = True
    #: the result's time axis: the clock when any step is dynamic
    solver.staging.time_axis = "time" if "dynamic" in pl.procedure else "load_factor"
    if _is_fibre(solver):
        # the fibres' first state, for a member built again after a removal:
        # it is new concrete, not the one that was taken out (review, 1.2)
        solver.staging.state0 = {}
        for name in ("_state_c", "_state_s", "_com_c", "_com_s", "_conn_slip", "_conn_committed",
                     "_sl_gcr_t", "_sl_gmax_t", "_sl_gcr_c", "_sl_gmax_c", "_sl_gamma"):
            v = getattr(solver, name, None)
            if isinstance(v, np.ndarray) and v.shape[:1] == (solver.nel,):
                solver.staging.state0[name] = v.copy()
        solver.staging.fstate0 = [np.array(a, copy=True) for a in getattr(solver, "fiber_states", [])]
        solver.staging.fcom0 = [np.array(a, copy=True) for a in getattr(solver, "fiber_committed", [])]
    off = np.flatnonzero(~pl.built0)
    if len(off):
        _stiffness_off(solver, off)
        _mass(solver, off, -1.0)
        _update_dormant(solver)
    if solver.mode == "dynamic" and len(off):
        _dynamic_step(solver, solver.total_time)


def _member_nodes(solver, ids):
    return np.unique(solver.mesh.elements[np.asarray(ids, int), :2].ravel())


def _update_dormant(solver):
    reach = np.zeros(solver.nn, bool)
    built = np.flatnonzero(solver.member_built)
    reach[solver.mesh.elements[built, :2].ravel()] = True
    solver.staging.dormant = ~reach


def dormant_dofs(solver) -> Optional[np.ndarray]:
    if solver.plan is None or not solver.staging.dormant.any():
        return None
    return solver.dofs.node_dofs(np.flatnonzero(solver.staging.dormant), solver.dofs.components).ravel()


def _is_fibre(solver) -> bool:
    return hasattr(solver, "_is_fiber")


class _probe:
    """A force evaluation that leaves no history: the frame solvers' own
    `_probe`, or, on the fibre beam, its fibres' states put back after."""

    def __init__(self, solver):
        self.s = solver

    def __enter__(self):
        s = self.s
        if hasattr(s, "_probe"):
            self.cm = s._probe()
            return self.cm.__enter__()
        self.cm = None
        self.saved = {}
        for name in ("_state_c", "_state_s", "_conn_slip", "_sl_gcr_t", "_sl_gmax_t", "_sl_gamma"):
            v = getattr(s, name, None)
            if isinstance(v, np.ndarray):
                self.saved[name] = v.copy()
        self.fs = [np.array(a, copy=True) for a in getattr(s, "fiber_states", [])]
        return self

    def __exit__(self, *exc):
        s = self.s
        if self.cm is not None:
            return self.cm.__exit__(*exc)
        for name, v in self.saved.items():
            getattr(s, name)[...] = v
        for a, b in zip(getattr(s, "fiber_states", []), self.fs):
            a[...] = b
        return False


def _stiffness_off(solver, ids):
    ids = np.asarray(ids, int)
    if _is_fibre(solver):
        solver._active[ids] = 0
        solver.member_active[ids] = False
        solver._refresh_kernel_mask()
        solver._rebuild_reference_stiffness()
        return
    solver.k_local[ids] = 0.0
    if getattr(solver, "_k_ref", None) is not None:
        solver._k_ref[ids] = 0.0
    solver.member_active[ids] = False
    _reference_K(solver, ids, -1.0)
    _rebatch(solver)


def _reference_K(solver, ids, sign):
    """beta K v is on the reference K: a member not built must not damp its
    nodes' relative motion (review, 1.2: a cantilever 14x overdamped)."""
    if not getattr(solver, "beta", 0.0) or getattr(solver, "_kg_ref", None) is None or not len(ids):
        return
    import scipy.sparse as sp
    K = sp.lil_matrix(solver.K)
    for e in np.asarray(ids, int):
        dof = solver.dofs.element_dofs(solver.mesh.elements[e])
        K[np.ix_(dof, dof)] = K[np.ix_(dof, dof)].toarray() + sign * solver._kg_ref[e]
        solver.record_k_block(e, dof if sign > 0 else None)
    solver.K = K.tocsr()


def _stiffness_on(solver, ids):
    ids = np.asarray(ids, int)
    if _is_fibre(solver):
        solver._active[ids] = 1
        solver.member_active[ids] = True
        solver._refresh_kernel_mask()
        solver._rebuild_reference_stiffness()
        return
    kf = getattr(solver, "_k_factor", 1.0)
    f = np.asarray(kf, float)[ids][:, None, None] if np.ndim(kf) else float(kf)
    solver.k_local[ids] = solver.staging.k_built[ids] * f
    if getattr(solver, "_k_ref", None) is not None and solver.staging.k_ref_built is not None:
        solver._k_ref[ids] = solver.staging.k_ref_built[ids]
    solver.member_active[ids] = True
    _reference_K(solver, ids, +1.0)
    _rebatch(solver)


def _rebatch(solver):
    solver.rebatch()


def _mass(solver, ids, sign):
    """Put member ids' translational mass on (+1) or take it off (-1) their
    nodes: their own share and their `with_mass` line mass (weightless)."""
    stride = solver.dofs.stride
    pm = solver.physical_mass.reshape(-1, stride)
    line = getattr(solver, "line_mass_per_end", None)
    for e in np.asarray(ids, int):
        full = solver._element_mass_share(int(e))
        share = full[:3]
        lm = float(line[e]) if line is not None and e < len(line) else 0.0
        zone = solver.member_zone_mass(int(e)) if hasattr(solver, "member_zone_mass") else np.zeros(2)
        for k, n in enumerate(solver.mesh.elements[e, :2]):
            pm[n, :3] = np.maximum(pm[n, :3] + sign * (share + lm + zone[k]), 0.0)
            if stride > 3:
                # the rotary inertia goes and comes with the member (review, 1.2)
                pm[n, 3:6] = np.maximum(pm[n, 3:6] + sign * full[3:6], 0.0)
            if lm:
                w = np.array(solver.weightless_mass, float, copy=True)
                w[n] = np.maximum(w[n] + sign * lm, 0.0)
                solver.weightless_mass = w
    solver.physical_mass = pm.reshape(-1)
    if solver.mode != "relaxation":
        solver.mass = np.maximum(solver.physical_mass, 1e-20)


# --------------------------------------------------------------------------- #
#  procedures
# --------------------------------------------------------------------------- #

def _row(solver):
    return np.asarray(abs(solver.K).sum(axis=1)).ravel()


def _free(solver):
    free = ~solver._held_dofs()
    d = dormant_dofs(solver)
    if d is not None:
        free[d] = False
    return free


def _dashpot_rows(solver):
    """Row sums of the connector dashpots (linear ones and the damper laws'
    linear pieces) of the members now active, or None without connectors."""
    from ..elements import connector
    if hasattr(solver, "_connector_props"):                       # the fibre solver
        ids = [e for e in getattr(solver, "_connector_ids", ()) if solver._active[e]]
        props_of = lambda e: solver._connector_props[int(solver._connector_of[e])]
    elif hasattr(solver, "member_kind") and hasattr(solver, "member_props"):
        active = getattr(solver, "member_active", None)
        ids = [int(e) for e in np.flatnonzero(np.asarray(solver.member_kind) == "connector2")
               if active is None or active[e]]
        props_of = lambda e: solver.member_props[e]
    else:
        return None
    if not len(ids):
        return None
    return connector.dashpot_rows(ids, solver.mesh.elements, solver.mesh.nodes, solver.R0, props_of,
                                  solver.dofs, solver.ndof)


def _dynamic_step(solver, total_time):
    """The stable step of the CURRENT masses: a member not yet built has
    taken its mass off its nodes, and the step set up with it is too long."""
    sol = solver.p.solver
    row = _row(solver)
    mass = np.maximum(solver.physical_mass, 1e-20)
    free = _free(solver) & (row > 1e-20)
    # A CARRIED PARTICLE IS NOT INTEGRATED: a parked one stays where it
    # stopped, an appendage's tips follow their anchor (`_carry_appendages`),
    # and neither has any mass of its own. In the bound they set a step of
    # 1e-10 of the step's and a step begun after a park never ended
    carried = solver.carried_nodes() if hasattr(solver, "carried_nodes") else ()
    moving = _free(solver)
    if len(carried):
        moving[solver.dofs.node_dofs(np.asarray(carried, np.int64)).ravel()] = False
    free &= moving
    if not free.any():
        return
    ratio = mass[free] / row[free]
    tie = solver.constraint_layer().carried if hasattr(solver, "constraint_layer") else None
    if tie is not None:
        # the members' mass set off their nodes: the full mass's bound there
        # (`CarriedMass.step_ratio`, as the setup's `_carried_step`)
        ratio = np.minimum(ratio, tie.step_ratio(mass, row, free)[free])
    root = float(np.sqrt(np.min(ratio)))
    dt = float(sol.dt_safety) * root
    f = solver._damping_step_factor(1.0 / root) if hasattr(solver, "_damping_step_factor") else 1.0
    dt *= f
    base = getattr(solver, "stable_dt", None) or solver.dt
    dt = min(dt, base) if solver.staging.base_mode == "dynamic" else dt
    # THE CONNECTOR DASHPOTS' BOUND, dt < 2 m / c, HERE TOO (found in the
    # review of 1.10). The solvers' own setup applies it
    # under a dynamic base mode only; a dynamic step after a relaxation --
    # gravity, then the earthquake -- took its step from the stiffness alone,
    # and a damper law (which a relaxation base does not refuse, as it does
    # `connector_damping`) went non-finite, or chattered about zero velocity.
    rows = _dashpot_rows(solver)
    if rows is not None:
        on = moving & (rows > 1e-20)
        if on.any():
            dt = min(dt, float(2.0 * float(sol.dt_safety) * np.min(mass[on] / rows[on])))
    from ..elements.stability import aligned_step
    solver.dt, solver.n_steps = aligned_step(float(total_time), dt)
    solver.total_time = float(total_time)
    solver.mass = mass


def configure(solver, k: int) -> None:
    """The step's procedure: its mass, step and damping."""
    pl = solver.plan
    mode = pl.procedure[k]
    if mode == "static":
        mode = "relaxation"                 # the implicit solver drives a relaxation-built model
    sol = solver.p.solver
    step = solver.steps[k]
    if not hasattr(solver, "_mode_cfg"):
        solver._mode_cfg = {solver.staging.base_mode: _snapshot(solver)}
    if mode == solver.staging.base_mode and not pl.changes:
        _restore_cfg(solver, solver._mode_cfg[mode])
        if mode == "dynamic" and step.get("total_time"):
            _dynamic_step(solver, float(step["total_time"]))
        _relaxation_damping(solver, pl)
        return
    solver.mode = mode
    if mode == "relaxation":
        if solver.staging.base_mode == "relaxation":
            _restore_cfg(solver, solver._mode_cfg["relaxation"])
        else:
            safety = float(sol.relaxation_safety)
            solver.mass = np.maximum(_row(solver) / safety ** 2, 1e-20)
            solver.dt = 1.0
            solver.damp_kind = "kinetic"
            solver.zeta = solver.alpha = solver.beta = 0.0
            n = step.get("n_steps") or sol.n_steps or 1
            solver.n_steps = int(n); solver.total_time = float(n)
            solver._n_steps_solver = solver.n_steps
        _relaxation_damping(solver, pl)
        if hasattr(solver, "_refresh_fictitious_mass"):
            solver._refresh_fictitious_mass()
    else:
        T = float(step.get("total_time") or sol.total_time or 0.0)
        if T <= 0.0:
            raise ValueError(f"steps[{k}] is dynamic and has no duration: give it total_time")
        if solver.staging.base_mode == "dynamic":
            _restore_cfg(solver, solver._mode_cfg["dynamic"])
        else:
            damp = sol.damping
            solver.damp_kind = damp.type or "viscous"
            if solver.damp_kind == "kinetic":
                raise ValueError(f"steps[{k}] is dynamic, and solver.damping is kinetic -- a device "
                                 "for finding a static answer that removes the dynamics; give the "
                                 "dynamic steps rayleigh or viscous damping (or none)")
            solver.zeta = float(damp.coeff)
            solver.alpha = float(damp.alpha)
            solver.beta = float(damp.beta)
        _dynamic_step(solver, T)


def _relaxation_damping(solver, pl):
    """In a model with dynamic steps, `solver.damping` is the dynamic steps'
    (it may not be kinetic there); its relaxation steps take kinetic damping."""
    if solver.mode == "relaxation" and "dynamic" in pl.procedure and solver.damp_kind != "kinetic":
        solver.damp_kind = "kinetic"
        solver.zeta = solver.alpha = solver.beta = 0.0


_CFG = ("mode", "mass", "dt", "n_steps", "total_time", "damp_kind", "zeta", "alpha", "beta")


def _snapshot(solver):
    return {k: (np.array(getattr(solver, k), copy=True) if isinstance(getattr(solver, k, None), np.ndarray)
                else getattr(solver, k, None)) for k in _CFG}


def _restore_cfg(solver, cfg):
    for k, v in cfg.items():
        setattr(solver, k, np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
    if solver.mode != "relaxation":
        solver.mass = np.maximum(solver.physical_mass, 1e-20)


# --------------------------------------------------------------------------- #
#  the step's model change
# --------------------------------------------------------------------------- #

def begin_step(solver, k: int, resumed: bool = False) -> None:
    """At the start of step k, before its loads are read: the procedure,
    then the removals and releases (their forces read from the state the
    previous step left), then the births. A step resumed from a checkpoint
    has had its model change already: only its procedure is set again."""
    pl = solver.plan
    solver.step_index = k
    solver._jack_live = True          # a run's step is in progress: its jacks hold their force
    if resumed:
        configure(solver, k)
        rt = solver.staging.resumed_time
        solver.staging.resumed_time = None
        if rt is not None and solver.mode == "dynamic" and float(solver.total_time) == rt[2]:
            # THE STEP THE RUN WAS ON, not one taken from the masses now: a
            # split, a park or an appendage has changed them since the step
            # began, and the restart took another step than the run it
            # continues (the review of v1.19). A restart that RAISES
            # the step's duration -- a budget, which may change -- takes the
            # step `configure` gave it (review of v1.19).
            solver.dt, solver.n_steps = float(rt[0]), int(rt[1])
        return
    jack_freeze(solver, k)
    extra = []
    d = pl.deactivate[k]
    if len(d):
        extra.append(_removal_force(solver, d, k))
    rel = pl.release[k]
    if len(rel):
        extra.append(_release_force(solver, rel, k))
    ties = solver.ties
    if ties is not None and pl.release_links[k]:
        # a released rigid link's force, taken off over the step as a
        # support's reaction is -- in the release's slot, the same amplitude
        # (the checkpoint keeps two slots)
        F = ties.release(pl.release_links[k])
        if extra and extra[-1] is not None and len(rel):
            extra[-1] = extra[-1] + F
        else:
            extra.append(F)
    a = pl.activate[k]
    solver.staging.born_now = a
    if len(a):
        _build(solver, a, solver.steps[k].get("birth") or "tangent")
    if ties is not None:
        ties.birth(k)                     # the step's own links, where the built members put their nodes
    solver.staging.step_extra[k] = [F for F in extra if F is not None]
    configure(solver, k)


def _removal_force(solver, ids, k):
    """What the members being removed did to the rest -- the loads that act
    on them (their weight, their line loads, at the values the step before
    ended with) less the internal force they put on their nodes -- as a
    load that is taken off over the step (Abaqus' removal). Then they are
    gone."""
    with _probe(solver):
        f_before = solver.internal_force()[0].copy()
    _stiffness_off(solver, ids)
    with _probe(solver):
        f_after = solver.internal_force()[0].copy()
    f_rem = f_before - f_after
    W = _member_loads_at_end(solver, ids, k - 1) if k > 0 else np.zeros(solver.ndof)
    _mass(solver, ids, -1.0)
    solver.member_built[ids] = False
    _update_dormant(solver)
    return W - f_rem


def _step_loads(solver, j):
    """The loads acting in step j as `step_data` reads them: the model's
    (held after the first step), the kept ones of earlier steps, step j's."""
    base = [dict(x, amplitude="hold") if j and x.get("amplitude") is None else dict(x)
            for x in solver.p.get("loads") or []]
    return inherited(solver, j, base, "loads") + [dict(x) for x in solver.steps[j].get("loads") or []]


def _end_factor(solver, ld, j):
    """A load's amplitude at the end of step j (1 for the defaults: every
    default amplitude ends at full value)."""
    name = ld.get("amplitude")
    if not name:
        return 1.0
    a = solver.amplitudes.get(name)
    return 1.0 if a is None else float(a(1.0))


def _member_loads_at_end(solver, ids, j):
    """(ndof,) the gravity and line loads members ids carried at the end of
    step j -- what leaves with them when they are removed (review, 1.2: the
    line loads and the kept gravity were missing)."""
    from .dof import TRANSLATION
    out = np.zeros(solver.ndof)
    ids = np.asarray(ids, int)
    for ld in _step_loads(solver, j):
        f = _end_factor(solver, ld, j)
        if not f:
            continue
        inside = np.zeros(solver.nn, bool)
        inside[solver._region(ld.get("region", "all"))] = True
        if ld.get("gravity") is not None:
            g = np.asarray(ld["gravity"], float) * f
            for e in ids:
                share = solver._element_mass_share(int(e))[:3]
                zone = solver.member_zone_mass(int(e)) if hasattr(solver, "member_zone_mass") else np.zeros(2)
                for k, n in enumerate(solver.mesh.elements[e, :2]):
                    if inside[n]:
                        solver.dofs.add_nodal(out, np.array([n]), (share + zone[k]) * g, TRANSLATION)
        elif ld.get("line_load") is not None:
            q = np.asarray(ld["line_load"], float) * f
            kinds = getattr(solver, "member_kind", None)
            for e in ids:
                if not inside[solver.mesh.elements[e, :2]].all():
                    continue
                if kinds is not None and kinds[e] not in solver.BEAM_TYPES:
                    continue
                _fl, g = solver.member_line_load(int(e), q, ld.get("coordinate_system", "global") == "local")
                out[solver.dofs.element_dofs(solver.mesh.elements[e])] += g
    return out


def _release_force(solver, nodes, k):
    """The reaction the freed supports carried at the end of the step
    before, as a load taken off over this step."""
    F = np.zeros(solver.ndof)
    rf = getattr(solver, "last_reaction_force", None)
    rm = getattr(solver, "last_reaction_moment", None)
    if rf is None:
        return None
    solver.dofs.add_nodal(F, nodes, np.asarray(rf)[nodes], TRANSLATION)
    if rm is not None:
        solver.dofs.add_nodal(F, nodes, np.asarray(rm)[nodes], ROTATION)
    return F


def _build(solver, ids, birth):
    ids = np.asarray(ids, int)
    brk = getattr(solver, "breaking", None)
    if brk is not None and brk.member_failed[ids].any():
        # a timed removal took it out before it was born: it stays out (review, 1.2)
        ids = ids[~brk.member_failed[ids]]
        if not len(ids):
            return
    new_nodes = [int(n) for n in _member_nodes(solver, ids) if solver.staging.dormant[n]]
    qv = solver.dofs.view(solver.q); vv = solver.dofs.view(solver.v)
    if new_nodes and birth == "tangent":
        _place_tangent(solver, ids, set(new_nodes))
    elif new_nodes:
        qv[new_nodes] = 0.0
        solver.orient[new_nodes] = np.eye(3)
    if new_nodes:
        vv[new_nodes] = 0.0
        # a node built again is a new node: a support that held it where it
        # stood before it was removed holds it at the support now (found in
        # review, v1.14: it was born at zero and dragged to the old value,
        # the new member stressed by the way)
        st = solver.__dict__.get("_hold")
        if st is not None and len(st["at"]) == solver.ndof:
            st["at"].reshape(solver.nn, -1)[new_nodes] = np.nan
        # A SUPPORT IS WHERE IT IS: a new node's held translations stay at the
        # support, whatever the tangent says (a roller at the far pier, or a
        # plane restraint, which holds one direction of every node: under it
        # the tangent still sets the others -- review, 1.2)
        held = _held_translations(solver)
        for n in new_nodes:
            qv[n, :3][held[n]] = 0.0
        # ...AND SO IS A HELD ROTATION: a new node whose three rotations a
        # support holds is born unturned. Found in review for a
        # node only new connectors reach -- a spring to a fixed ground point
        # was born with the tip's rotation on that point, zero offset, and
        # the support then turned it back over the step, the full rotation in
        # a spring that was to be born free of force. A beam's new end at a
        # clamped support had the same history and kept it until 1.14:
        # the member is now born free of stress between the
        # standing end as it stands and the support as it is.
        turned = _held_rotations(solver)
        for n in new_nodes:
            if turned[n].all():
                solver.orient[n] = np.eye(3)
                qv[n, 3:] = 0.0
    solver.member_built[ids] = True
    _update_dormant(solver)
    _mass(solver, ids, +1.0)
    if _is_fibre(solver):
        _fresh_fibres(solver, ids)
    _stiffness_on(solver, ids)
    # BORN WITHOUT STRESS where it stands: its local deformation now is its
    # reference, less the temperature it is born at -- read when the step's
    # loads are (`birth_offsets`), not from the last force evaluation, which
    # holds the step before's temperature and the members' earlier offsets
    # (review, 1.2: a member born twice was born stressed)
    solver._birth_raw = (ids, raw_local(solver, ids))
    if solver.staging.birth_d is None:
        solver.staging.birth_d = np.zeros((solver.nel, 12))
    solver.staging.birth_d[ids] = solver._birth_raw[1]
    _connector_birth(solver, ids)


def _connector_birth(solver, ids):
    """A CONNECTOR IS BORN FREE OF FORCE TOO: the six
    deformations it has where its ends stand now are its reference, and its
    laws start from nothing (a bearing built again is a new bearing). A
    saddle's links wait for the cable they carry; a bearing for its girder."""
    kinds = getattr(solver, "member_kind", None)
    if kinds is None or not hasattr(solver, "_conn_committed"):
        return
    ids = np.asarray(ids, int)
    cs = ids[np.asarray(kinds)[ids] == "connector2"]
    if not len(cs):
        return
    from ..elements import connector
    if solver._conn_birth is None:
        solver._conn_birth = np.zeros((solver.nel, 6))
    qv = solver.dofs.view(solver.q)
    for e in cs:
        conn = solver.mesh.elements[e, :2]
        d, _J = connector._deformation_and_jacobian(qv[conn, :3], solver.orient[conn], solver.mesh.nodes[conn],
                                                    solver.R0[e], solver.member_props[e])
        solver._conn_birth[e] = d
    solver._conn_committed[cs] = 0.0
    solver._conn_slip[cs] = 0.0
    if solver.connector_materials:
        solver.connector_materials.reset(cs)            # a `material` law's history starts at its birth
    # the row a bar's stretch offset takes means nothing to a connector
    solver.staging.birth_d[cs] = 0.0
    solver._birth_raw[1][np.isin(solver._birth_raw[0], cs)] = 0.0


def _fresh_fibres(solver, ids):
    """A fibre member born again starts from its fibres' first state."""
    for name, v0 in solver.staging.state0.items():
        v = getattr(solver, name, None)
        if isinstance(v, np.ndarray) and v.shape == v0.shape:
            v[ids] = v0[ids]
    for a, a0 in zip(getattr(solver, "fiber_states", []), solver.staging.fstate0):
        if a.shape == a0.shape and a.shape[:1] == (solver.nel,):
            a[ids] = a0[ids]
    for a, a0 in zip(getattr(solver, "fiber_committed", []), solver.staging.fcom0):
        if a.shape == a0.shape and a.shape[:1] == (solver.nel,):
            a[ids] = a0[ids]
    if solver.connector_materials:
        # a connector's `material` law starts from its first state too, as its
        # slips do above (found in review: `_connector_birth` returns before
        # its reset on a fibre model)
        solver.connector_materials.reset(ids)


def birth_offsets(solver):
    """Called from `_force`, once the step's temperatures are known: a member
    born this step is stress-free at the temperature it is born at."""
    pending = getattr(solver, "_birth_raw", None)
    if pending is None:
        return
    solver._birth_raw = None
    ids, raw = pending
    d = raw.copy()
    if getattr(solver, "time_dependent", False) and (solver.staging.age_shift is not None):
        # THE SEGMENT'S OWN CLOCK: born at `activate_age` (solver.age by
        # default) when the model's clock reads the step's start
        k = int(getattr(solver, "step_index", 0) or 0)
        a0 = solver.steps[k].get("activate_age") or float(solver.p.solver.age or 28.0)
        start = float(getattr(solver, "_step_ages", (solver._current_age,))[0])
        solver.staging.age_shift[ids] = float(a0) - start
        solver.staging.age_floor[ids] = float(a0)
        if hasattr(solver, "_relax_start"):
            # a prestressed bar born now relaxes from now (a jacked one from
            # its lock-off: its clock is not started here)
            jt = getattr(solver, "_jack_T", None)
            born_pre = ids if jt is None else ids[~(np.asarray(jt)[ids] > 0.0)]
            solver._relax_start(born_pre, start, restart=True)
        # ...and its shrinkage before then happened in the formwork, free
        from .creep import shrinkage_strain
        for g in getattr(solver, "_tgroups", []):
            if g["shrink"] is None:
                continue
            inside = np.isin(ids, g["ids"])
            if inside.any():
                eps = shrinkage_strain(g["shrink"], float(a0))
                d[inside, 6] -= eps * np.asarray(solver.length, float)[ids[inside]]
        # and an old history (a member built, removed and built again) is gone
        h = getattr(solver, "_history", None)
        if h is not None:
            for inc in h.increments:
                inc[ids] = 0.0
            if h._last is not None:
                h._last[ids] = 0.0
    for a, dd in ((() if _is_fibre(solver) else getattr(solver, "_thermal", ()))):
        # a frame's field (nel, 12); a fibre's is per fibre and is not
        # subtracted at birth (a documented limitation) -- a section of
        # twelve fibres has the frame's shape, hence the test on the solver
        dd = np.asarray(dd)
        if dd.ndim == 2 and dd.shape[1] == 12:
            d -= dd[ids] * a(0.0)
    solver.staging.birth_d[ids] = d


def _held_translations(solver) -> np.ndarray:
    """(nn, 3) translations a constraint of the current step holds."""
    k = int(getattr(solver, "step_index", 0) or 0)
    step = solver.steps[k] if k < len(solver.steps) else {}
    out = np.zeros((solver.nn, 3), bool)
    for c in constrained_items(solver, k):
        idx = np.asarray(c["_nodes"])
        for d in range(3):
            if c.get("dofs", [0] * 6)[d]:
                out[idx, d] = True
    return out


def _held_rotations(solver) -> np.ndarray:
    """(nn, 3) rotations a constraint of the current step holds."""
    k = int(getattr(solver, "step_index", 0) or 0)
    out = np.zeros((solver.nn, 3), bool)
    for c in constrained_items(solver, k):
        idx = np.asarray(c["_nodes"])
        dofs = list(c.get("dofs", [0] * 6)) + [0] * 6
        for d in range(3):
            if dofs[3 + d]:
                out[idx, d] = True
    return out


def constrained_items(solver, k):
    """The constraints of step k with the nodes each still holds: the
    inherited ones less the nodes a release has freed SINCE they began
    (a constraint kept after a release is not freed by it -- review, 1.2),
    and the step's own."""
    base = [dict(x, _from=-1) for x in solver.p.get("constraints") or []]
    items = inherited(solver, k, base, "constraints")
    out = []
    for c in items:
        idx = np.asarray(solver._region(c["region"]), np.int64)
        freed = freed_since(solver, k, c.get("_from", -1))
        if freed is not None:
            idx = np.setdiff1d(idx, freed)
        out.append(dict(c, _nodes=idx))
    for c in solver.steps[k].get("constraints") or []:
        out.append(dict(c, _nodes=np.asarray(solver._region(c["region"]), np.int64)))
    return out


def freed_since(solver, k, origin) -> Optional[np.ndarray]:
    """The nodes released by the steps after `origin` up to and including k."""
    pl = solver.plan
    if pl is None:
        return None
    sets = [pl.release[j] for j in range(max(origin + 1, 0), k + 1) if len(pl.release[j])]
    if not sets:
        return None
    return np.unique(np.concatenate(sets))


def _place_tangent(solver, ids, dormant):
    """Each node only the new members reach is carried rigidly by the
    standing node it is reached from first (breadth first, from every
    standing node at once): its displacement and its rotation."""
    ids = np.asarray(ids, int)
    # A CABLE BEING JACKED AND A CONNECTOR CARRY NOTHING (1.13):
    # a rib segment hung from a tower was reached first through its buckle
    # cable and the saddle's links, and was put where the tower top's
    # rotation, a thousandth of a radian over 270 m, sent it. Their own new
    # nodes (a saddle, a link's middle point) stay where the model has them.
    # NOR DOES A CABLE CUT TO LENGTH (1.15): a stay with a
    # `prestress` or an unstressed length, or one that carries tension only,
    # is no more a rigid arm than a jacked one -- a new deck segment reached
    # through it from a standing tower was put where the tower top sent it,
    # not on the deck's tangent.
    kinds = getattr(solver, "member_kind", None)
    jt = getattr(solver, "_jack_T", None)
    if kinds is not None:
        carry = np.asarray(kinds)[ids] != "connector2"
        if jt is not None:
            carry &= ~(jt[ids] > 0.0)
        lu = getattr(solver, "member_Lu", None)
        length = getattr(solver, "length", None)
        if lu is not None and length is not None:
            carry &= ~(np.abs(np.asarray(lu)[ids] - np.asarray(length)[ids]) > 1e-12 * np.abs(np.asarray(length)[ids]))
        cable = np.zeros(len(ids), bool)
        if lu is not None and length is not None:
            cable |= np.abs(np.asarray(lu)[ids] - np.asarray(length)[ids]) > 1e-12 * np.abs(np.asarray(length)[ids])
        tonly = getattr(solver, "member_tension_only", None)
        if tonly is not None:
            carry &= ~np.asarray(tonly)[ids]
            cable |= np.asarray(tonly)[ids]
        cables = ids[cable]
        ids = ids[carry]
    else:
        cables = ids[:0]
    conn = solver.mesh.elements[ids, :2]
    adj: Dict[int, List[int]] = {}
    for a, b in conn:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    root = {}
    queue = deque()
    for n in adj:
        if n not in dormant:
            root[n] = n
            queue.append(n)
    # a group of new members that touches nothing standing (the first span,
    # a segment on its own falsework) is where it was designed: its nodes
    # keep their coordinates, as under `design`
    X = solver.mesh.nodes
    qv = solver.dofs.view(solver.q)
    moved = []
    while queue:
        n = queue.popleft()
        for m in adj[n]:
            if m in root:
                continue
            r = root[n]
            root[m] = r
            Rr = solver.orient[r]
            x = X[r] + qv[r, :3] + Rr @ (X[m] - X[r])
            qv[m, :3] = x - X[m]
            solver.orient[m] = Rr
            moved.append(m)
            queue.append(m)
    for m in dormant:
        if m not in root:
            qv[m, :3] = 0.0
            solver.orient[m] = np.eye(3)
    _between_cable_ends(solver, cables, dormant, root, qv)
    from .rotations import log_map
    nodes = list(dormant)
    if nodes:
        qv[nodes, 3:] = log_map(solver.orient[nodes])


def _between_cable_ends(solver, cables, dormant, root, qv):
    """The new nodes inside a cable that places nothing (one cut to length or
    tension-only, meshed in several bars) go between its ends: each one's
    displacement is the ends' interpolated along the cable's design length
    (a weighted graph Laplacian, the ends held), so the stay is born straight
    between where its ends are, not kinked into its last bar (review,
    v1.16). Rotations stay as they are: a bar has none to carry. A
    cable that touches nothing placed or standing stays where it was designed."""
    if len(cables) == 0:
        return
    conn = solver.mesh.elements[np.asarray(cables, int), :2]
    inner = sorted({int(n) for n in conn.ravel() if int(n) in dormant and int(n) not in root})
    if not inner:
        return
    pos = {n: i for i, n in enumerate(inner)}
    X = solver.mesh.nodes
    nu = len(inner)
    rows, cols, vals = [], [], []
    rhs = np.zeros((nu, 3))
    for a, b in conn:
        a, b = int(a), int(b)
        w = 1.0 / max(float(np.linalg.norm(X[b] - X[a])), 1e-12)
        for u, v in ((a, b), (b, a)):
            if u not in pos:
                continue
            rows.append(pos[u]); cols.append(pos[u]); vals.append(w)  # noqa: E702
            if v in pos:
                rows.append(pos[u]); cols.append(pos[v]); vals.append(-w)  # noqa: E702
            else:
                rhs[pos[u]] += w * qv[v, :3]
    # a chain with no end held anywhere: the small diagonal keeps it at its
    # design position (zero displacement) instead of leaving A singular
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    A = sp.csc_matrix((vals, (rows, cols)), shape=(nu, nu))
    A = A + sp.identity(nu, format="csc") * (1e-12 * float(A.diagonal().max()))
    qv[inner, :3] = np.asarray(spsolve(A, rhs)).reshape(nu, 3)


def raw_local(solver, ids) -> np.ndarray:
    """(len(ids), 12): the local deformation of members ids as the force law
    reads it -- the stretch from the unstressed length and, for a beam, both
    ends' local rotations -- with no offset taken off.

    A BAR WITH A DESIGNED LENGTH KEEPS IT: a stay with `prestress` or an
    `unstressed_length` is installed to that length -- tensioned to it, or
    slack by it -- not stress-free where it is put (review, 1.2); its stretch
    offset is zero, and it pulls as designed from the moment it is born."""
    from .rotations import beam_frames_from_triads
    ids = np.asarray(ids, int)
    conn = solver.mesh.elements[ids, :2]
    dofs = np.array([solver.dofs.element_dofs(c) for c in conn]).reshape(len(ids), 12)
    q = solver.q[dofs].reshape(len(ids), 2, 6)
    x = solver.mesh.nodes[conn] + q[:, :, :3]
    x = solver.geometry.ends(ids, x, solver.orient)
    R0 = np.asarray(solver.R0)[ids]
    _R, th, L = beam_frames_from_triads(x, solver.orient[conn], R0)
    Lu = getattr(solver, "member_Lu", None)
    length = np.asarray(solver.length, float)
    ref = length if Lu is None else np.asarray(Lu, float)
    out = np.zeros((len(ids), 12))
    out[:, 6] = L - ref[ids]
    kinds = getattr(solver, "member_kind", None)
    beam = np.ones(len(ids), bool) if kinds is None else np.isin(np.asarray(kinds)[ids], tuple(solver.BEAM_TYPES))
    out[beam, 3:6] = th[beam, 0]
    out[beam, 9:12] = th[beam, 1]
    if Lu is not None:
        designed = ~beam & (np.abs(ref[ids] - length[ids]) > 1e-12 * length[ids])
        out[designed, 6] = 0.0
    return out


# --------------------------------------------------------------------------- #
#  what `step_data` reads
# --------------------------------------------------------------------------- #

def inherited(solver, k, items, kind):
    """The model-level items, then every kept item of the steps before k,
    each held at the value it ended its own step with."""
    out = list(items)
    for j in range(k):
        for it in solver.steps[j].get(kind) or []:
            if it.get("keep"):
                out.append({**it, "amplitude": _held_at_end(solver, it.get("amplitude"), j),
                            "_kept": True, "_from": j})
    return out


def _held_at_end(solver, name, j):
    if not name:
        return "hold"
    a = solver.amplitudes.get(name)
    v = 1.0 if a is None else float(a(1.0))
    if v == 1.0:
        return "hold"
    from .amplitude import Amplitude
    key = f"_kept_{v:.17g}"
    if key not in solver.amplitudes:
        solver.amplitudes[key] = Amplitude(key, [[0.0, v], [1.0, v]])
    return key


def extra_groups(solver, k):
    """The step's unloading forces (removals, releases) with the unloading
    amplitude, for `step_data`'s groups."""
    out = []
    F_list = solver.staging.step_extra.get(k) or []
    if not F_list:
        return out
    from .amplitude import Amplitude
    if UNLOAD not in solver.amplitudes:
        solver.amplitudes[UNLOAD] = Amplitude(UNLOAD, [[0.0, 1.0], [1.0, 0.0]])
    for F in F_list:
        out.append((solver.amplitudes[UNLOAD], F))
    return out


def newborn_weight(solver, idx, g) -> Optional[np.ndarray]:
    """(ndof,) the weight of the members born this step on the nodes idx --
    what an inherited (held) gravity load must bring on gradually."""
    born = solver.staging.born_now
    if born is None or not len(born):
        return None
    inside = np.zeros(solver.nn, bool); inside[idx] = True
    out = np.zeros(solver.ndof)
    for e in born:
        share = solver._element_mass_share(int(e))[:3]
        zone = solver.member_zone_mass(int(e)) if hasattr(solver, "member_zone_mass") else np.zeros(2)
        for k, n in enumerate(solver.mesh.elements[e, :2]):
            if inside[n]:
                solver.dofs.add_nodal(out, np.array([n]), (share + zone[k]) * np.asarray(g, float),
                                      TRANSLATION)
    return out


def built_mask(solver) -> Optional[np.ndarray]:
    return getattr(solver, "member_built", None) if solver.plan is not None else None


def offsets(solver) -> Optional[np.ndarray]:
    return solver.staging.birth_d if solver.plan is not None else None


def final_built(solver) -> Optional[np.ndarray]:
    """The members standing when the last step ends."""
    pl = solver.plan
    if pl is None:
        return None
    built = pl.built0.copy()
    for a, d in zip(pl.activate, pl.deactivate):
        built[a] = True
        built[d] = False
    return built


class as_finished:
    """The masses as they are at the end of the run, for `vfife summary`'s
    totals (review, 1.2: a staged result weighed the members built at the
    start); put back on exit."""

    def __init__(self, solver):
        self.s = solver

    def __enter__(self):
        s = self.s
        fb = final_built(s)
        self.saved = None
        if fb is None:
            return fb
        self.saved = (np.array(s.physical_mass, copy=True),
                      None if getattr(s, "weightless_mass", None) is None else np.array(s.weightless_mass, copy=True),
                      getattr(s, "mass", None))
        now = s.member_built
        _mass(s, np.flatnonzero(fb & ~now), +1.0)
        _mass(s, np.flatnonzero(~fb & now), -1.0)
        return fb

    def __exit__(self, *exc):
        if self.saved is not None:
            s = self.s
            s.physical_mass, w, m = self.saved
            if w is not None:
                s.weightless_mass = w
            if m is not None:
                s.mass = m
        return False


# --------------------------------------------------------------------------- #
#  the clock, and the checkpoint
# --------------------------------------------------------------------------- #

def end_step(solver, k):
    """After step k: a dynamic step moves the clock by its duration."""
    if solver.plan is None:
        return
    if solver.plan.procedure[k] == "dynamic":
        solver.staging.clock = float(solver.staging.clock + solver.total_time)
    jack_freeze(solver, k + 1)
    solver._jack_live = False


# --------------------------------------------------------------------------- #
#  cables stressed to a force (1.13)
# --------------------------------------------------------------------------- #
#
# `jack: T` on a bar: in the step that builds it (the first, for a bar built
# from the start) its force is T times the step's progress, whatever its
# length -- the jack holds a force, not a length -- and at the step's end its
# unstressed length is what gives T at the length it then has. From then on it
# is an ordinary bar. It is done through the unstressed length the bar's force
# law already reads: before every force evaluation of that step,
#
#     L_u = (L - d) / (1 + p T / (E A kf)),       so that  E A kf (L - L_u - d) / L_u = p T
#
# with L its length now, d the offsets the law takes off (its stretch at
# birth, a temperature), kf the creep and ageing factor. No kernel knows.

def _jack_step(solver) -> Optional[np.ndarray]:
    """The step each member is jacked in (-1: none)."""
    js = getattr(solver, "_jack_step_of", None)
    if js is None:
        pl = solver.plan
        js = np.full(solver.nel, -1, np.int64)
        jacked = solver._jack_T > 0.0
        js[jacked & pl.built0] = 0            # built from the start: the first step
        # ...else the FIRST step that builds it. A bar removed and built again
        # is not jacked again: it keeps the length its jack left it (found in
        # review: one built from the start, removed and rebuilt was given
        # the rebuilding step, and stood unstressed through the first)
        for k in range(len(pl.activate) - 1, -1, -1):
            a = pl.activate[k]
            js[a[jacked[a] & ~pl.built0[a]]] = k
        solver._jack_step_of = js
    return js


def _jack_set(solver, ids, p):
    if not len(ids):
        return
    conn = solver.mesh.elements[ids, :2]
    qv = solver.dofs.view(solver.q)[:, :3]
    x = solver.mesh.nodes[conn] + qv[conn]
    L = np.linalg.norm(x[:, 1] - x[:, 0], axis=1)
    dth = getattr(solver, "_thermal_d", None)
    d = 0.0 if dth is None else np.asarray(dth)[ids, 6]
    EA = np.array([float(solver.member_E[e]) * float(solver.member_props[e]["area"]) for e in ids])
    kf = getattr(solver, "_k_factor", 1.0)
    if np.ndim(kf):
        EA = EA * np.asarray(kf)[ids]
    solver.member_Lu[ids] = (L - d) / (1.0 + p * solver._jack_T[ids] / EA)


def jack_lengths(solver) -> None:
    """Before a force evaluation: the unstressed lengths of the bars being
    jacked in the current step, for the force the jack holds now."""
    # ONLY WHILE A RUN'S STEP IS IN PROGRESS (found in review): on a solver
    # that is merely built -- a modal or a buckling analysis, a probe -- the
    # same lines made the cable follow every perturbation with no force, and
    # its stiffness was missing from the tangent without a word. There it
    # is the bar it is until jacked.
    if solver.plan is None or not getattr(solver, "_jack_live", False):
        return
    k = int(getattr(solver, "step_index", 0) or 0)
    js = _jack_step(solver)
    ids = np.flatnonzero((js == k) & ~solver._jack_done & solver.member_built)
    _jack_set(solver, ids, min(max(float(solver._jack_p), 0.0), 1.0))


def jack_freeze(solver, k) -> None:
    """Before step k begins (and after the last): the bars jacked in an
    earlier step keep the length that gives their full force in the state
    that step ended in."""
    if solver.plan is None or not hasattr(solver, "_jack_T") or not solver._jack_T.any():
        return
    js = _jack_step(solver)
    ids = np.flatnonzero((js >= 0) & (js < k) & ~solver._jack_done & solver.member_built)
    _jack_set(solver, ids, 1.0)
    if hasattr(solver, "_relax_start") and len(ids):
        # locked off: its relaxation starts now, from the strain the jack left
        solver._relax_start(ids, float(getattr(solver, "_current_age", 0.0)), jacked=True)
    solver._jack_done[(js >= 0) & (js < k)] = True
    solver._jack_p = 0.0


def checkpoint(solver, ck: dict) -> dict:
    if solver.plan is None:
        return ck
    ck["member_built"] = solver.member_built.astype(np.int8)
    ck["birth_offset"] = (np.zeros((solver.nel, 12)) if solver.staging.birth_d is None
                          else np.asarray(solver.staging.birth_d, float).copy())
    if hasattr(solver, "_jack_T"):
        # the lengths the jacks left, and which are behind us
        ck["jack_length"] = np.stack([np.asarray(solver.member_Lu, float),
                                      solver._jack_done.astype(float)], axis=1)
    if hasattr(solver, "_conn_committed"):
        cb = getattr(solver, "_conn_birth", None)
        ck["connector_birth"] = np.zeros((solver.nel, 6)) if cb is None else np.asarray(cb, float).copy()
    ck["physical_mass_steps"] = np.asarray(solver.physical_mass, float).copy()
    ck["step_clock"] = np.array([solver.staging.clock])
    # the step the run is on (`begin_step`)
    ck["step_time"] = np.array([float(solver.dt), float(solver.n_steps), float(solver.total_time)])
    if (solver.staging.age_shift is not None):
        ck["age_shift"] = np.stack([solver.staging.age_shift, solver.staging.age_floor], axis=1)
    k = int(getattr(solver, "step_index", 0) or 0)
    # FIXED SHAPES: a restart is checked array by array against a fresh
    # model's -- two slots for the step's unloading forces, a mask for the born
    extra = list(solver.staging.step_extra.get(k) or [])[:2]
    slots = np.zeros((2, solver.ndof))
    for i, F in enumerate(extra):
        slots[i] = F
    ck["step_unload"] = slots
    ck["step_unload_n"] = np.array([len(extra)])
    born = np.zeros(solver.nel, np.int8)
    born[np.asarray(solver.staging.born_now, np.int64)] = 1
    ck["born_now"] = born
    # the reactions the next step's release reads (review, 1.2: a restart at a
    # step boundary lost them)
    rf = getattr(solver, "last_reaction_force", None)
    rm = getattr(solver, "last_reaction_moment", None)
    ck["last_reaction"] = np.concatenate([
        np.zeros((solver.nn, 3)) if rf is None else np.asarray(rf, float),
        np.zeros((solver.nn, 3)) if rm is None else np.asarray(rm, float)], axis=1)
    return ck


def restore(solver, ck: dict) -> None:
    if solver.plan is None or "member_built" not in ck:
        return
    built = np.asarray(ck["member_built"]).astype(bool)
    on, off = np.flatnonzero(built & ~solver.member_built), np.flatnonzero(~built & solver.member_built)
    brk = getattr(solver, "breaking", None)
    if brk is not None:
        on = on[~brk.member_failed[on]]           # a broken member stays broken
    if len(off):
        _stiffness_off(solver, off)
    if len(on):
        _stiffness_on(solver, on)
    solver.member_built = built
    solver.staging.birth_d = np.asarray(ck["birth_offset"], float).copy()
    if "jack_length" in ck and hasattr(solver, "_jack_T") and solver._jack_T.any():
        jl = np.asarray(ck["jack_length"], float)
        ids = solver._jack_T > 0.0
        solver.member_Lu[ids] = jl[ids, 0]
        solver._jack_done = jl[:, 1] > 0.5
    if "connector_birth" in ck and hasattr(solver, "_conn_committed"):
        cb = np.asarray(ck["connector_birth"], float)
        solver._conn_birth = cb.copy() if cb.any() else None
    solver.physical_mass = np.asarray(ck["physical_mass_steps"], float).copy()
    solver.staging.clock = float(np.asarray(ck["step_clock"]).ravel()[0])
    if "step_time" in ck:
        st = np.asarray(ck["step_time"], float).ravel()
        solver.staging.resumed_time = (float(st[0]), int(round(st[1])), float(st[2]))
    if "age_shift" in ck and (solver.staging.age_shift is not None):
        a = np.asarray(ck["age_shift"], float)
        solver.staging.age_shift, solver.staging.age_floor = a[:, 0].copy(), a[:, 1].copy()
    if "step_unload" in ck:
        k = int(np.asarray(ck.get("loop", [0])).ravel()[0]) if "loop" in ck else int(getattr(solver, "step_index", 0) or 0)
        n = int(np.asarray(ck.get("step_unload_n", [2])).ravel()[0])
        solver.staging.step_extra = {k: [np.asarray(x, float) for x in np.asarray(ck["step_unload"])[:n]]}
        solver.staging.born_now = np.flatnonzero(np.asarray(ck["born_now"]) > 0).astype(np.int64)
    if "last_reaction" in ck:
        lr = np.asarray(ck["last_reaction"], float)
        solver.last_reaction_force = lr[:, :3].copy()
        solver.last_reaction_moment = lr[:, 3:].copy()
    if solver.mode != "relaxation":
        solver.mass = np.maximum(solver.physical_mass, 1e-20)
    _update_dormant(solver)
