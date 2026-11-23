"""Where a run was when its checkpoint was taken, and how to pick it up there.

The frame, fibre-beam and implicit solvers saved their state -- positions,
velocities, triads, material state -- but not their place in the loops that
drive it, so `--restart` began again at step 0, t = 0 and load factor 0 on
top of the restored state: a ground motion replayed from its first second,
a staged push reloaded from zero (which UNLOADS a plastic model), a vehicle
started again at the abutment. The state was right and the run was wrong.

A frame-family loop position is five integers, `solver._loop`:

    [k, step_start, stage, stage_step0, phase]

`k` the analysis step, `step_start` the global iteration count at which it
began (so the local counter is `step - step_start`, and with it the clock
and the load factor), `stage` the load level of a staged relaxation (0 when
unstaged), `stage_step0` the count at which that level began, and `phase`
0 iterating, 1 converged but not yet recorded, 2 done. With it go the
integrator's own memory (`_half`, `ke_prev`, `n_resets`), the prescribed
starting positions a level blends from, the fictitious mass, the vehicles'
states and the energy ledger.

`initial_state` (api.start_from) uses the same checkpoint but STARTS a run:
it clears `_resume`, so none of the loop position is used -- only the
material and geometric state, and the creep history, which is the
material's memory and belongs to the concrete, not to the run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .diagnostics import UserError

# `good`, `dl0`: the arc-length stall recovery; an older
# checkpoint has fewer entries, and zip() leaves them to their defaults
_STATIC_KEYS = ("kind", "lam", "dl", "inc", "count", "its", "cuts", "side0", "good", "dl0", "relaxing")
_VEHICLE_FIELDS = ("x", "v", "a", "contact", "_surf_prev", "_t_done", "_out", "lift_off",
                   "lift_tyre", "tension_rigid", "lift_first")     # the last three: v1.0
_VEHICLE_INTS = ("lift_off", "lift_tyre", "tension_rigid")


@dataclass
class StageRecord:
    start: List[np.ndarray]
    iterations: int
    converged: bool
    resid: float
    kr: float
    ke: float


@dataclass
class Resume:
    k: int
    step_start: int
    stage: int
    stage_step0: int
    phase: int
    measures: tuple
    start: List[np.ndarray]
    step: int
    mass: Optional[np.ndarray] = None
    vehicles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    step_ages: Optional[tuple] = None
    current_age: Optional[float] = None
    #: the implicit static solver's increment: load factor, increment size,
    #: counts, and for an arc its last direction (vfife/implicit.py)
    static: Dict[str, Any] = field(default_factory=dict)
    #: the peak force and displacement the residual floors are a millionth
    #: of (`_force_scale`, `_motion_scale`): a restart without them judged a
    #: later, lighter step on another scale and converged elsewhere:
    scales: Optional[tuple] = None

    def first_stage(self) -> int:
        """The level to go on with: this one, or the next if it was done."""
        return self.stage + 1 if self.phase == 2 else max(self.stage, 1)

    def in_stage(self, stage: int) -> Optional[StageRecord]:
        if stage != self.stage or self.phase == 2:
            return None
        resid, kr, ke = self.measures
        return StageRecord([np.array(a, dtype=float) for a in self.start],
                           self.step - self.stage_step0, self.phase == 1,
                           float(resid), float(kr), float(ke))

    def after_step_data(self, solver) -> None:
        """What `step_data` rebuilt from the case, put back as it was."""
        solver.step_start = self.step_start
        if self.mass is not None and np.shape(self.mass) == np.shape(solver.mass):
            solver.mass = np.array(self.mass, dtype=float)
        for name, st in self.vehicles.items():
            veh = getattr(solver, "_vehicles", {}).get(name)
            if veh is None:
                continue
            for f in _VEHICLE_FIELDS:
                if f in st:
                    setattr(veh, f, st[f])
            diag = getattr(solver, "diagnostics", None)
            if getattr(veh, "lift_off", 0) and isinstance(diag, dict):
                diag.setdefault("vehicle_contact", {})[name] = veh.contact_report()
        if self.step_ages is not None:
            solver._step_ages = self.step_ages
        if self.current_age is not None:
            solver._current_age = self.current_age


#: a restored checkpoint with no loop position (see loop_restore)
_NO_POSITION = object()


def resume_position(solver, n_steps: int) -> Optional[Resume]:
    """The position to resume from, once; None for a fresh run."""
    r = getattr(solver, "_resume", None)
    solver._resume = None
    if r is None:
        return None
    solver._clock_floor = None
    if r is _NO_POSITION:
        raise UserError("RESTART_OLD_CHECKPOINT",
                        "this checkpoint was written before vfife recorded where in the analysis "
                        "a run was, so it cannot be resumed: the run would begin "
                        "again from step 0 on top of the state it reached",
                        "start a new run from it with `solver.initial_state: <that result>` "
                        "(the state at rest, t = 0), or run the case again from the start")
    if r.k >= n_steps:
        raise UserError("RESTART_BUDGET_EXHAUSTED",
                        "this checkpoint is at the end of the analysis: there is nothing left to run",
                        "add a step, or raise the one that ended, and restart again")
    if r.scales is not None:
        # a resumed run only: `initial_state` starts a new one, whose peak
        # starts at zero
        solver._force_scale, solver._motion_scale = (float(x) for x in r.scales)
    return r


def check_resumed(solver, resume: Optional[Resume]) -> None:
    """A restart that ran nothing is a mistake the user should hear about,
    not an empty result reported as CONVERGED."""
    if resume is not None and solver.step == resume.step and not solver.times:
        raise UserError("RESTART_BUDGET_EXHAUSTED",
                        f"this restart had nothing left to run: the checkpoint is at step {solver.step}, "
                        "where the analysis it came from had already finished",
                        "raise solver.n_steps (or total_time), or add a step, and restart again")


# ------------------------------------------------------------ checkpoint
def loop_checkpoint(solver) -> Dict[str, np.ndarray]:
    ck: Dict[str, np.ndarray] = {}
    loop = getattr(solver, "_loop", None)
    if loop is None:
        # never run: the position is the start
        loop = [0, int(getattr(solver, "step", 0)), 0, int(getattr(solver, "step", 0)), 0]
    ck["loop"] = np.asarray(loop, dtype=np.int64)
    ck["loop_measures"] = np.asarray(getattr(solver, "_loop_measures", (0.0, 0.0, 0.0)), float)
    for i, a in enumerate(getattr(solver, "_stage_start_q", None) or []):
        ck[f"loop_start{i}"] = np.asarray(a, float)
    ck["half"] = np.array([1 if getattr(solver, "_half", False) else 0])
    ck["ke_prev"] = np.array([float(getattr(solver, "ke_prev", 0.0))])
    ck["n_resets"] = np.array([int(getattr(solver, "n_resets", 0))])
    # the residual floors' peaks
    ck["scales"] = np.array([float(getattr(solver, "_force_scale", 0.0) or 0.0),
                             float(getattr(solver, "_motion_scale", 0.0) or 0.0)])
    ck["mass"] = np.asarray(solver.mass, float)
    if hasattr(solver, "_hold_checkpoint"):
        solver._hold_checkpoint(ck)                # supports holding a node where it stood (1.14)
    for i, (name, veh) in enumerate(sorted(getattr(solver, "_vehicles", {}).items())):
        ck[f"vehicle{i}_name"] = np.asarray(name.encode())
        for f in _VEHICLE_FIELDS:
            val = getattr(veh, f, None)
            if val is not None:
                ck[f"vehicle{i}_{f}"] = np.asarray(val, float)
    if getattr(solver, "_step_ages", None) is not None and getattr(solver, "time_dependent", False):
        a0, a1, kind = solver._step_ages
        ck["step_ages"] = np.array([a0, np.nan if a1 is None else a1], float)
        ck["step_ages_kind"] = np.asarray(str(kind).encode())
    if hasattr(solver, "diagnostics"):
        ck["energy"] = solver.diagnostics["energy"].checkpoint()
    st = getattr(solver, "_static_state", None)
    if st:
        ck["static_state"] = np.array([st.get(k, 0.0) for k in _STATIC_KEYS], float)
        if st.get("prev") is not None:
            ck["static_prev"] = np.asarray(st["prev"], float)
    ck.update(creep_checkpoint(solver))
    ck.update(split_checkpoint(solver, ck))
    return ck


# ---------------------------------------------------------------- splits
def split_checkpoint(solver, ck) -> Dict[str, np.ndarray]:
    """Which particle every split end came from (the bond graph's `parents`
    and `made_for`), and the frame solver's list of splits. Without them a
    restart knew where the ends were but not where they had come from: a
    rigid surface restricted to a region no longer caught the debris split
    off its nodes, which fell through it, and the restarted result had no
    `member_splits` table."""
    out: Dict[str, np.ndarray] = {}
    b = getattr(solver, "bonds", None)
    if b is not None and getattr(b, "parents", None):
        out["bond_parents"] = np.asarray(b.parents, dtype=np.int64)
        out["bond_made_for"] = np.asarray(b.made_for, dtype=np.int64).reshape(-1, 2)
    if "splits" not in ck and getattr(solver, "splits", None):
        out["splits"] = np.asarray(solver.splits, float).reshape(-1, 4)
    return out


def split_restore(solver, ck) -> None:
    b = getattr(solver, "bonds", None)
    if b is not None and "bond_parents" in ck:
        b.parents = [int(p) for p in np.asarray(ck["bond_parents"]).ravel()]
        b.made_for = [(int(e), int(k)) for e, k in np.asarray(ck["bond_made_for"]).reshape(-1, 2)]
        # a rigid surface follows the splits it has not seen: all of them
        contact = getattr(solver, "contact", None)
        if contact is not None and hasattr(contact, "_seen_splits"):
            contact._seen_splits = 0
    if "splits" in ck and isinstance(getattr(solver, "splits", None), list) and not solver.splits:
        solver.splits[:] = [(float(t), int(e), int(p), int(n))
                            for t, e, p, n in np.asarray(ck["splits"], float).reshape(-1, 4)]


def loop_restore(solver, ck) -> None:
    creep_restore(solver, ck)
    if hasattr(solver, "_hold_restore"):
        solver._hold_restore(ck)
    split_restore(solver, ck)
    solver._restored_energy = ck.get("energy")
    if "half" in ck:
        solver._half = bool(np.asarray(ck["half"]).ravel()[0])
    if "ke_prev" in ck:
        solver.ke_prev = float(np.asarray(ck["ke_prev"]).ravel()[0])
    if "n_resets" in ck:
        solver.n_resets = int(np.asarray(ck["n_resets"]).ravel()[0])
    if "loop" not in ck:
        # a checkpoint written before v1.0 says where the state
        # is, not where the run was: resuming from it would replay the
        # analysis from step 0 on top of it. `initial_state` (which clears
        # this) can still start a new run from it.
        solver._resume = _NO_POSITION
        return
    k, step_start, stage, stage_step0, phase = (int(x) for x in np.asarray(ck["loop"]).ravel()[:5])
    starts, i = [], 0
    while f"loop_start{i}" in ck:
        starts.append(np.asarray(ck[f"loop_start{i}"], float)); i += 1
    vehicles, i = {}, 0
    while f"vehicle{i}_name" in ck:
        name = bytes(np.asarray(ck[f"vehicle{i}_name"]).item()).decode()
        st = {}
        for f in _VEHICLE_FIELDS:
            key = f"vehicle{i}_{f}"
            if key in ck:
                a = np.array(ck[key], dtype=float)
                st[f] = (float(a.reshape(-1)[0]) if f in ("_t_done", "lift_first")
                         else int(a.reshape(-1)[0]) if f in _VEHICLE_INTS else a)
        vehicles[name] = st; i += 1
    ages = None
    if "step_ages" in ck:
        a = np.asarray(ck["step_ages"], float)
        ages = (float(a[0]), None if np.isnan(a[1]) else float(a[1]),
                bytes(np.asarray(ck["step_ages_kind"]).item()).decode())
    static = {}
    if "static_state" in ck:
        static = dict(zip(_STATIC_KEYS, (float(x) for x in np.asarray(ck["static_state"], float))))
        static["kind"] = int(static["kind"])
        if "static_prev" in ck:
            static["prev"] = np.array(ck["static_prev"], float)
    solver._resume = Resume(
        k=k, step_start=step_start, stage=stage, stage_step0=stage_step0, phase=phase,
        measures=tuple(np.asarray(ck.get("loop_measures", np.zeros(3)), float)[:3]),
        start=starts, step=int(np.asarray(ck["step"]).ravel()[0]),
        mass=np.array(ck["mass"], float) if "mass" in ck else None,
        vehicles=vehicles, step_ages=ages, static=static,
        current_age=getattr(solver, "_current_age", None) if ages is not None else None,
        scales=tuple(np.asarray(ck["scales"], float).ravel()[:2]) if "scales" in ck else None)
    solver.step_index = k
    solver.step_start = step_start


# ----------------------------------------------------------------- creep
def creep_checkpoint(solver) -> Dict[str, np.ndarray]:
    """The concrete's memory: every recorded elastic increment and its age.
    A continuation began with none of it and crept again from the elastic
    value (case_29: 0.0100 where 0.0300 had been reached)."""
    if not getattr(solver, "time_dependent", False):
        return {}
    # a frame's phi is one per member since v1.0; the clock keeps
    # the largest for a reader of the old layout
    phi = np.atleast_1d(np.asarray(getattr(solver, "_phi_cur", 0.0), float))
    ck = {"creep_clock": np.array([
        float(getattr(solver, "_current_age", np.nan)),
        np.nan if getattr(solver, "_prev_station", None) is None else float(solver._prev_station),
        float(phi.max()) if phi.size else 0.0])}
    if phi.size > 1:
        ck["creep_phi"] = phi
    # where each tendon's relaxation clock started: a continuation from
    # `initial_state` keeps it, and does not relax from zero again
    t0 = _relax_starts(solver)
    if t0 is not None:
        ck["relax_t0"] = t0
    h = getattr(solver, "_history", None)
    if h is not None and h.ages:
        ck["creep_ages"] = np.asarray(h.ages, float)
        ck["creep_increments"] = np.stack([np.asarray(x, float) for x in h.increments])
        ck["creep_last"] = np.asarray(h._last, float)
        if getattr(h, "windows", None):
            # each station's interval, for a member on its own clock (1.2)
            ck["creep_windows"] = np.asarray([[np.nan if a is None else a, b] for a, b in h.windows], float)
    return ck


def creep_restore(solver, ck) -> None:
    if "creep_clock" not in ck or not getattr(solver, "time_dependent", False):
        return
    age, prev, phi = (float(x) for x in np.asarray(ck["creep_clock"], float)[:3])
    solver._current_age = age
    solver._prev_station = None if np.isnan(prev) else prev
    solver._phi_cur = np.array(ck["creep_phi"], float) if "creep_phi" in ck else phi
    # the clock a continuation may not run back from; a
    # restart clears it (`resume_position`), its own step being mid-way
    solver._clock_floor = age
    h = getattr(solver, "_history", None)
    if h is not None and "creep_ages" in ck:
        h.ages = [float(a) for a in np.asarray(ck["creep_ages"], float)]
        h.increments = [np.array(x, float) for x in np.asarray(ck["creep_increments"], float)]
        h._last = np.array(ck["creep_last"], float)
        if "creep_windows" in ck:
            h.windows = [(None if np.isnan(a) else float(a), float(b))
                         for a, b in np.asarray(ck["creep_windows"], float)]
    if "relax_t0" in ck:
        _restore_relax_starts(solver, np.asarray(ck["relax_t0"], float).ravel())
    if hasattr(solver, "_age_stiffness"):
        solver._age_stiffness(age)


def _relax_starts(solver):
    """The relaxing tendons' clocks: the fibre tendons' stressing ages
    (`_relax`, one per tendon), or the frame path's prestressed bars, bar by
    bar since v1.16 -- (n, 3) of each bar's start (NaN: not yet),
    prestrain and held length."""
    rows = getattr(solver, "_relax_members", None)
    if rows:
        return np.concatenate([np.stack([np.asarray(r[2], float), np.asarray(r[3], float),
                                         np.asarray(r[4], float)], axis=1) for r in rows])
    rows = getattr(solver, "_relax", None)
    if not rows:
        return None
    return np.array([float(r[2]) for r in rows], float)


def _restore_relax_starts(solver, t0) -> None:
    rows = getattr(solver, "_relax_members", None)
    if rows:
        n = sum(len(r[0]) for r in rows)
        if t0.size == 3 * n:
            state = t0.reshape(n, 3)
            at = 0
            for ids, _spec, t, e, b in rows:
                m = len(ids)
                t[:] = state[at:at + m, 0]; e[:] = state[at:at + m, 1]; b[:] = state[at:at + m, 2]
                at += m
        elif t0.size == len(rows):                        # a checkpoint from before v1.16: one start a group
            for (ids, _spec, t, _e, _b), v in zip(rows, t0):
                t[np.isfinite(t)] = float(v)
        return
    rows = getattr(solver, "_relax", None)
    if rows and len(rows) == len(t0):
        solver._relax = [tuple(r[:2]) + (float(t),) + tuple(r[3:]) for r, t in zip(rows, t0)]
