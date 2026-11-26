"""The explicit dynamic step as one compiled call.

`BeamSolver.run` asks `stepper(...)` once per analysis step. When the model
and the step are ones the compiled step (kernels/numba_step) reproduces, it
returns a `Stepper` and the loop hands it every run of steps between two
points where the loop has work of its own -- a progress check, an output
frame, the end of the step. Otherwise it returns None and the loop takes its
steps one at a time, as before; `refusal(...)` says why.

What it reproduces, to the bit: frame members (`beam2`, `beam2_timoshenko`,
the elastic members of a mixed line mesh), axial members (bars, cables,
hangers, tension-only or not) and, since v1.4, connectors and
bearings with their laws and dashpots (kernels/numba_connector); fibre
models (`beam2_fiber`, `FiberStepper`) through their fused kernel, with
their bearings; loads that are fixed vectors times their amplitudes, a
uniform ground motion; no damping, Rayleigh or viscous; held dofs. Since
1.6 also prescribed motions and supports driven by
records, plastic hinges, breaking members and their fragments (a break is
an EVENT the loop takes), fibre fracture and splitting (the same), moving
loads and vehicles -- each written ONCE as a compiled function the loop
runs too (kernels/numba_step.hinge_update, carry_appendages, _prescribe;
kernels/numba_loads.axle_actions, vehicle_step), so the two paths agree
by construction. A fibre section's cracking shear law is in the fused
kernel since 1.7, so it is in the compiled step too.
Since 1.12 the contact between members
(core/member_contact) as well: one evaluation is the compiled
`contact_step`, which the contact's component (core/components) calls
after the members' forces in the loop and both chunks call themselves;
`_components_refusal` is how `refusal` tells that component from one it
would step past.
What is left to the loop: a prescribed rotation, rigid-surface contact,
temperatures and
time-dependent strains, rigid diaphragms, soil springs, members born in a
step, analysis steps (1.2), a load group that is a script's own callable,
and a fibre run too short to pay for compiling its step (`SHORT_FIBRE_RUN`).
Frame models need `backend: numba`; a fibre model takes it whenever its
fused kernel runs.

`VFIFE_COMPILED_STEP=0` in the environment turns it off; `=1` takes it
even for a fibre run too short to pay for its compilation.
"""
from __future__ import annotations

import os

import numpy as np

from ..base.dof import TRANSLATION
from .capabilities import refuse

#: amplitude factors and ground accelerations are precomputed per run of
#: steps; a long run is cut into pieces this long
MAX_CHUNK = 4096


#: the methods of one step; any of them replaced on the solver itself is
#: code the compiled step would not run
_HOOKS = ("internal_force", "_member_forces", "_internal_force_numba", "_force", "_advance", "_apply_bc",
          "commit_state", "_commit_members", "constraint_layer", "_check_failure", "apply_pending_splits",
          "_carry_appendages", "_ledger_force", "_stacked_groups", "_beam_forces_batched",
          "_truss_forces_batched", "kernel_members", "_kernel_dth")


_ORIGINALS: dict = {}


def _original(kind, name):
    """The method `name` of the solver class as this module first saw it: a
    script that replaces it on the class is caught as one that replaces it on
    the instance is (review)."""
    key = (kind, name)
    if key not in _ORIGINALS:
        _ORIGINALS[key] = getattr(kind, name, None)
    return _ORIGINALS[key]


#: the fibre solver's
_HOOKS_FIBER = ("internal_force", "_member_forces", "_internal_force_kernel", "_force", "_refresh_imposed",
                "_advance", "_apply_bc", "commit_state", "_commit_members", "_latch_fracture", "_check_failure", "_connector_force",
                "_connector_kernel", "_set_actions_abs", "_ledger_force", "apply_pending_splits",
                "kernel_connectors", "fused_kernel_prepare", "fused_kernel_section", "fused_kernel_kind",
                "fused_kernel_connector_state", "shear_law_args", "fracture_kernel_arrays")


#: a fibre run shorter than this many steps keeps the loop unless the
#: compiled step is asked for (`VFIFE_COMPILED_STEP=1`): compiling it, about
#: 3 s once per process, costs more than it saves (0.3 ms a step on
#: Hangzhou Bay). The answer is the same either way, to the bit.
SHORT_FIBRE_RUN = 20000


def _setting() -> str:
    return os.environ.get("VFIFE_COMPILED_STEP", "").strip().lower()


def enabled() -> bool:
    return _setting() not in ("0", "off", "false", "no")


def forced() -> bool:
    return _setting() in ("1", "on", "true", "yes")


_CONTACT_METHODS = ("force", "prepare", "masses", "book", "commit", "_grant", "_active")


def _components_refusal(solver):
    """None when every component attached to the solver (core/components)
    is one the compiled step runs itself -- the contact between members, its
    own methods untouched (1.12) -- else why not. Rigid
    surfaces, or a hook put on the contact itself (its force, its masses
    ...), run in the loop and would be stepped past here (review)."""
    from .member_contact import MemberContact, MemberContactComponent
    # ONE contact between members, the solver's own: the compiled step runs
    # `solver.member_contact` and nothing else (a second one attached by a
    # script ran in the loop only -- review of v1.18)
    if sum(type(c) is MemberContactComponent for c in solver.components) > 1:
        return refuse("two_contacts")
    for c in solver.components:
        if type(c) is not MemberContactComponent:
            return (refuse("rigid_contact") if type(c).__name__ == "RigidContactComponent"
                    else refuse("component", name=type(c).__name__))
        if vars(c).keys() - {"contact", "_nodes", "_zeros"} or any(
                getattr(MemberContactComponent, m) is not _original(MemberContactComponent, m)
                for m in ("line_force", "commit", "fields", "checkpoint", "restore")):
            return refuse("contact_component_hooked")
        mc = c.contact
        if mc is not getattr(solver, "member_contact", None):
            return refuse("foreign_contact")
        for m in _CONTACT_METHODS:
            if m in vars(mc) or getattr(type(mc), m, None) is not _original(MemberContact, m):
                return refuse("contact_hooked", method=m)
    return None


def refusal(solver, fixed, moving, prescribed, groups, ground):
    """None when the compiled step reproduces this step of `solver`, else why not."""
    if not enabled():
        return refuse("off")
    from ..kernels import numba_step
    if not numba_step.available():
        return refuse("no_numba")
    from .beam_solver import BeamSolver
    from .mixed_structural_solver import MixedStructuralSolver
    from .fiber_beam_solver import FiberBeamSolver
    kind = type(solver)
    if kind not in (BeamSolver, MixedStructuralSolver, FiberBeamSolver):
        return refuse("own_forces", kind=kind.__name__)
    if solver.mode != "dynamic":
        return refuse("not_dynamic")
    fiber = kind is FiberBeamSolver
    # a hook installed on the instance -- rigid contact wraps the force
    # evaluation (core/rigid_contact), a script may wrap anything -- is code
    # the compiled step would step past
    hooked = [n for n in (_HOOKS_FIBER if fiber else _HOOKS)
              if n in vars(solver) or getattr(kind, n, None) is not _original(kind, n)]
    comp = _components_refusal(solver)
    if hooked or comp is not None or getattr(solver, "contact", None) is not None:
        return refuse("hooked", what=", ".join(hooked) or comp or refuse("rigid_contact"))
    if solver.constraint_layer().links is not None:
        return refuse("rigid_links")
    if solver.damp_kind == "rayleigh" and solver.regional_damping() is not None:
        return refuse("damping_regions")
    if not fiber and getattr(solver, "backend", "numpy") != "numba":
        return refuse("backend")
    if solver.dofs.stride != 6 or solver.q.size != 6 * solver.nn:
        return refuse("six_dofs")
    if solver.plan is not None:
        return refuse("steps")
    # prescribed motions and supports driven by records (1.6): the kinds
    # `orientation._apply_bc` knows, on translations; a prescribed ROTATION
    # rebuilds the point's orientation from the vector with NumPy's exp_map,
    # which the kernel's scalar exp need not match to the bit
    for dofs, _value, a, motion in prescribed:
        if np.any((np.asarray(dofs) % solver.dofs.stride) >= 3):
            return refuse("rotation")
        if motion not in ("displacement", "velocity", "acceleration") and not hasattr(a, "motion"):
            return refuse("motion_kind", motion=motion)
    # moving loads and vehicles are taken (1.6): their arithmetic is the
    # compiled functions of kernels/numba_loads, which the loop runs too; any
    # other callable in a load group is a script's
    from .moving_load import MovingLoad
    from .vehicle import Vehicle
    for _a, F in groups:
        if callable(F) and not isinstance(F, (MovingLoad, Vehicle)):
            return refuse("callable")
    if getattr(solver, "_thermal", None) or solver.time_dependent:
        return refuse("thermal")
    if fiber:
        if solver.constraint_layer().diaphragms is not None:
            return refuse("diaphragms")
        why = _fiber_refusal(solver)
        if why is None and not forced():
            left = int(solver.n_steps) - (int(solver.step) - int(getattr(solver, "step_start", 0)))
            from ..kernels.numba_step import fiber_chunk_ready
            if left < SHORT_FIBRE_RUN and not fiber_chunk_ready(solver._kernel):
                why = refuse("short_fibre", left=left)
        return why
    if getattr(solver, "_thermal_d", None) is not None:
        return refuse("imposed")
    if solver.constraint_layer().diaphragms is not None:
        return refuse("diaphragms")
    if solver.geometry.any:
        return refuse("end_offsets")
    # plastic hinges, breaking and split-off members are taken (1.6): the
    # hinge update and the failure criterion run in the step, a break is an
    # event the loop takes, parked particles and appendages are carried;
    # what must not be left over is work the loop does at the END of a step
    if getattr(solver, "_pending_split", None) or getattr(solver, "_park_check", None) \
            or getattr(solver, "_reload_step", False):
        return refuse("split_waiting")
    soil = getattr(solver, "soil", None)
    if soil is not None and len(soil):
        return refuse("soil")
    if solver.damp_kind == "rayleigh":
        K = solver.K
        if getattr(K, "format", None) != "csr":
            return refuse("csr")
    for name, shape in (("q", (solver.ndof,)), ("v", (solver.ndof,)), ("mass", (solver.ndof,)),
                        ("orient", (solver.nn, 3, 3))):
        a = getattr(solver, name)
        if not isinstance(a, np.ndarray) or a.shape != shape or a.dtype != np.float64 \
                or not a.flags.c_contiguous:
            return refuse("arrays", name=name, shape=shape)
    if kind is MixedStructuralSolver:
        solver._beam_batch(); solver._truss_batch()
        if len(getattr(solver, "_connector_ids", ())) and solver._connector_kernel() is None:
            return refuse("connectors")
        known = np.concatenate([solver._truss_ids, getattr(solver, "_connector_ids", np.zeros(0, int))])
        if len(solver._other_ids) != len(known) or \
                not np.array_equal(np.sort(solver._other_ids), np.sort(known)):
            return refuse("other_members")
        if len(solver._truss_ids) and list(solver._truss_T) != [0, 1, 2]:
            return refuse("bars_dofs")
        if np.ndim(getattr(solver, "_k_factor", 1.0)):
            return refuse("bars_creep")
    return None


def _arrays_refusal(solver):
    if solver.damp_kind == "rayleigh":
        K = solver.K
        if getattr(K, "format", None) != "csr":
            return refuse("csr")
    for name, shape in (("q", (solver.ndof,)), ("v", (solver.ndof,)), ("mass", (solver.ndof,)),
                        ("orient", (solver.nn, 3, 3))):
        a = getattr(solver, name)
        if not isinstance(a, np.ndarray) or a.shape != shape or a.dtype != np.float64 \
                or not a.flags.c_contiguous:
            return refuse("arrays", name=name, shape=shape)
    return None


def _fiber_refusal(s):
    """What the fibre step adds to the frame step's conditions:
    the fused kernel and the compiled connectors carry its forces."""
    why = s.fused_kernel_refusal()
    if why is not None:
        return why
    if getattr(s, "fracture", None) is not None and s.fracture_kernel_arrays() is None:
        return refuse("fibre_fracture")
    if s.split_pending():
        return refuse("split_waiting")
    if s.in_trial():
        return refuse("trial")
    soil = getattr(s, "soil", None)
    if soil is not None and len(soil):
        return refuse("soil")
    if s.geometry.any:
        return refuse("end_offsets")
    if s.staging.birth_d is not None:
        return refuse("born")
    live, _kinds, ck = s.kernel_connectors()
    if len(live) and ck is None:
        # a connector law with no compiled form (a `material` law, 1.21):
        # the compiled step would carry its forces as nothing
        return refuse("connectors")
    return _arrays_refusal(s)


def stepper(solver, fixed, moving, prescribed, groups, ground):
    """A `Stepper` for this step of the dynamic loop, or None (`solver.
    compiled_step_refusal` then says why)."""
    why = refusal(solver, fixed, moving, prescribed, groups, ground)
    solver.compiled_step_refusal = why
    if why is not None:
        return None
    from .fiber_beam_solver import FiberBeamSolver
    if type(solver) is FiberBeamSolver:
        return FiberStepper(solver, fixed, moving, prescribed, groups, ground)
    return Stepper(solver, fixed, moving, prescribed, groups, ground)


class Stepper:
    """The arrays of one analysis step, gathered once; `advance` runs steps."""

    def __init__(self, s, fixed, moving, prescribed, groups, ground):
        self.s = s
        self.groups = groups
        nd = s.ndof
        self._members(s)
        # the loads: the stacked groups (BeamSolver._stacked_groups), else
        # each group's vector and amplitude
        stacked = s._stacked_groups(groups)
        self.s_mode = 0
        self.s_P = None
        if stacked is not None:
            G, P, A, rest = stacked
            Gc = G.tocsc() if G.format != "csc" else G
            self.s_indptr = np.ascontiguousarray(Gc.indptr, dtype=np.int64)
            self.s_indices = np.ascontiguousarray(Gc.indices, dtype=np.int64)
            self.s_data = np.ascontiguousarray(Gc.data, dtype=float)
            if P is not None:
                self.s_mode, self.s_P, self.s_A = 1, np.asarray(P, float), np.ascontiguousarray(A, dtype=float)
                self.s_amps = None
            else:
                self.s_mode, self.s_A, self.s_amps = 2, np.zeros((0, 2)), list(A)
            dense = list(rest)
            self.load_groups = rest
        else:
            self.s_indptr = np.zeros(1, np.int64); self.s_indices = np.zeros(0, np.int64)
            self.s_data = np.zeros(0); self.s_A = np.zeros((0, 2)); self.s_amps = None
            dense = list(groups)
            self.load_groups = groups
        self._groups(dense)
        # the factors of a run of steps are precomputed, one row per step:
        # a few million numbers at most (2000 groups with tables of their own
        # were 65 MB a run, found in review)
        width = len(self.g_amps) + (len(self.s_amps) if self.s_amps else 0)
        self.max_chunk = max(1, min(MAX_CHUNK, 2_000_000 // max(width, 1)))
        self.ground = [(np.asarray(ts, float), np.asarray(ac, float), np.asarray(d, float).reshape(3))
                       for ts, ac, d in ground]
        self.m_dir = (np.ascontiguousarray(np.stack([d for _t, _a, d in self.ground]))
                      if self.ground else np.zeros((0, 3)))
        self.m_trans = (np.ascontiguousarray(s.dofs.field(s.physical_mass, TRANSLATION), dtype=float)
                        if self.ground else np.zeros((s.nn, 3)))
        # damping
        self.dkind = {"rayleigh": 1, "viscous": 2}.get(s.damp_kind, 0)
        if self.dkind == 1:
            K = s.K
            self.K = (np.ascontiguousarray(K.indptr, dtype=np.int64),
                      np.ascontiguousarray(K.indices, dtype=np.int64),
                      np.ascontiguousarray(K.data, dtype=float))
        else:
            self.K = (np.zeros(nd + 1, np.int64), np.zeros(0, np.int64), np.zeros(0))
        # the supports (orientation._apply_bc, nothing prescribed)
        self.fixed = np.ascontiguousarray(fixed, dtype=np.bool_)
        fixed_zero = fixed & ~moving
        self.fixed_zero = np.ascontiguousarray(fixed_zero, dtype=np.bool_)
        self.clamped_mask = np.ascontiguousarray(s.dofs.view(fixed_zero)[:, 3:].all(axis=1), dtype=np.bool_)
        self.F = np.zeros(nd); self.fint = np.zeros(nd); self.damp = np.zeros(nd)
        self.u_prev = np.zeros(nd); self.fe_prev = np.zeros(nd); self.fi_prev = np.zeros(nd)
        self._prescribed(prescribed)
        self._breaking()

    def _members(self, s):
        """The members' arrays (frames, bars, connectors) and the gather of
        their rows into the dofs: what the dynamic step and the compiled
        relaxation (1.9) share."""
        from .mixed_structural_solver import MixedStructuralSolver
        self.mixed = isinstance(s, MixedStructuralSolver)
        nd = s.ndof
        # the members' arrays are the solver's to give
        km = s.kernel_members()
        self.f, self.collapse = km.f, km.collapse
        self.t_ids, self.t_conn, self.t_EA = km.t_ids, km.t_conn, km.t_EA
        self.t_collapse = self.collapse
        self.threads = km.threads
        # the connectors, compiled, after the axial members
        ck = km.connectors
        if ck is not None:
            self.c_ids, self.c_kind = ck["ids"], ck["kind"]
            self.c_enc, self.c_spins = ck["enc"], ck["spins"]
        else:
            from ..kernels.numba_connector import encode, spins
            self.c_ids = np.zeros(0, np.int64); self.c_kind = np.zeros(0, np.int64)
            self.c_enc, self.c_spins = encode([{}]), spins()
        # every member writes its own row; each dof gathers its entries in
        # the order the serial kernels add them: the frames member by member,
        # then the axial members (the first end, then the second)
        f_dofs = self.f[5]
        nf, nt = len(self.f[0]), len(self.t_ids)
        dof = [np.asarray(f_dofs, np.int64).reshape(-1)]
        if nt:
            c = self.t_conn
            dof.append(np.concatenate([6 * c[:, :1] + np.arange(3), 6 * c[:, 1:2] + np.arange(3)],
                                      axis=1).reshape(-1))
        ncn = len(self.c_ids)
        if ncn:
            cc = np.asarray(s.mesh.elements)[self.c_ids][:, :2].astype(np.int64)
            dof.append(np.concatenate([6 * cc[:, :1] + np.arange(6), 6 * cc[:, 1:2] + np.arange(6)],
                                      axis=1).reshape(-1))
        dof = np.concatenate(dof)
        row = np.arange(nf * 12 + nt * 6 + ncn * 12, dtype=np.int64)
        order = np.lexsort((row, dof))
        self.inc_idx = np.ascontiguousarray(row[order], dtype=np.int64)
        self.inc_ptr = np.ascontiguousarray(
            np.concatenate([[0], np.cumsum(np.bincount(dof, minlength=nd))]), dtype=np.int64)
        self.end_actions = np.zeros((s.nel, 12)); self.actions = np.zeros((s.nel, 6))
        self.cdeform = np.zeros((s.nel, 6))

    def _groups(self, dense):
        """The dense load groups by kind (1.6): the fixed vectors stacked in
        g_F, the callables -- moving loads and vehicles -- in `movers`;
        g_kind/g_idx keep the order the loop adds them in."""
        nd = self.s.ndof
        self.g_amps = [a for a, _F in dense]
        fixed = [np.asarray(F, float) for _a, F in dense if not callable(F)]
        self.g_F = np.ascontiguousarray(np.stack(fixed)) if fixed else np.zeros((0, nd))
        self.movers = [F for _a, F in dense if callable(F)]
        kind, idx, nf, nm = [], [], 0, 0
        for _a, F in dense:
            if callable(F):
                kind.append(1); idx.append(nm); nm += 1
            else:
                kind.append(0); idx.append(nf); nf += 1
        self.g_kind = np.ascontiguousarray(np.asarray(kind, dtype=np.int64))
        self.g_idx = np.ascontiguousarray(np.asarray(idx, dtype=np.int64))
        self._movers()

    def _movers(self):
        """The moving loads' and the vehicles' fixed arrays (1.6): the paths'
        members flattened (axes, lengths, dofs, releases), each moving load's
        axles with their group row and force, and the vehicles' parameters
        padded to the largest of them (kernels/numba_step.moving_step)."""
        from .moving_load import MovingLoad
        s = self.s
        P_R, P_L, P_E, P_rel, P_has = [], [], [], [], []
        ax_group, ax_p = [], []
        self.ml, self.veh = [], []
        off = 0
        for row, F in enumerate(self.movers):
            if isinstance(F, MovingLoad):
                if not hasattr(F, "_signed"):
                    F._prepare()
                rel = getattr(F, "releases", None)
                self.ml.append((row, F, off))
                ax_group.extend([row] * len(F.offsets))
                ax_p.append(F.magnitudes[:, None] * F.direction[None, :])
            else:
                if not F._prepared:
                    F._prepare()
                rel = F.releases if F._has_rel else None
                self.veh.append((row, F, off))
            n_p = len(F.path.elements)
            P_R.append(np.asarray(F._R, float)); P_L.append(np.asarray(F._L, float))
            P_E.append(np.asarray(F._edofs, np.int64))
            P_rel.append(rel); P_has.append(np.full(n_p, rel is not None, dtype=np.bool_))
            off += n_p
        c = np.ascontiguousarray
        if off:
            self.P_R = c(np.concatenate(P_R)); self.P_L = c(np.concatenate(P_L))
            self.P_edofs = c(np.concatenate(P_E), dtype=np.int64); self.P_has = c(np.concatenate(P_has))
            if any(r is not None for r in P_rel):
                self.P_rel = c(np.concatenate([np.asarray(r, float) if r is not None
                                               else np.tile(np.eye(12), (len(h), 1, 1))
                                               for r, h in zip(P_rel, P_has)]))
            else:
                self.P_rel = np.zeros((1, 12, 12))
        else:
            self.P_R = np.zeros((1, 3, 3)); self.P_L = np.zeros(1); self.P_edofs = np.zeros((1, 12), np.int64)
            self.P_has = np.zeros(1, np.bool_); self.P_rel = np.zeros((1, 12, 12))
        self.ax_group = c(np.asarray(ax_group, dtype=np.int64))
        self.ax_p = c(np.concatenate(ax_p)) if ax_p else np.zeros((0, 3))
        self.mv = np.zeros((max(len(self.movers), 1), s.ndof))
        nv = len(self.veh)
        vs = [F for _r, F, _o in self.veh]
        na_max = max([F.na for F in vs] + [1]); n_max = max([len(F.m) for F in vs] + [1])
        nf_max = max([len(F.fi) for F in vs] + [1])
        self.na_max, self.n_max = na_max, n_max
        V = dict(group=np.zeros(nv, np.int64), na=np.zeros(nv, np.int64), n=np.zeros(nv, np.int64),
                 nf=np.zeros(nv, np.int64), nb=np.zeros(nv, np.int64), fi=np.zeros((nv, nf_max), np.int64),
                 flex=np.zeros((nv, na_max), np.bool_), m=np.zeros((nv, n_max)), B=np.zeros((nv, na_max, n_max)),
                 rigid=np.zeros((nv, na_max)), ks=np.zeros((nv, na_max)), cs=np.zeros((nv, na_max)),
                 kt=np.zeros((nv, na_max)), ct=np.zeros((nv, na_max)), weight=np.zeros((nv, n_max)),
                 up=np.zeros((nv, 3)))
        for vi, (row, F, _off) in enumerate(self.veh):
            na, n, nf = F.na, len(F.m), len(F.fi)
            V["group"][vi] = row; V["na"][vi] = na; V["n"][vi] = n; V["nf"][vi] = nf; V["nb"][vi] = F.nb
            V["fi"][vi, :nf] = F.fi; V["flex"][vi, :na] = F.flex_mask
            V["m"][vi, :n] = F.m; V["weight"][vi, :n] = F.weight; V["B"][vi, :na, :n] = F.B
            for name in ("rigid", "ks", "cs", "kt", "ct"):
                V[name][vi, :na] = getattr(F, name)
            V["up"][vi] = F.up
        self.V = tuple(V[k] for k in ("group", "na", "n", "nf", "nb", "fi", "flex", "m", "B", "rigid",
                                      "ks", "cs", "kt", "ct", "weight", "up"))

    def _moving(self, p0):
        """The movers' arrays of one chunk (1.6): where every axle is in each
        step -- the path member under it (flattened), xi, on the path or
        not -- the road under the vehicles' axles, and the vehicles' state,
        which the kernel advances in place and `_book_movers` hands back."""
        s = self.s
        n_do = len(p0)
        t = p0 * s.total_time
        nax = len(self.ax_group)
        mk = np.zeros((n_do, nax), np.int64); ma = np.zeros((n_do, nax)); mon = np.zeros((n_do, nax), np.bool_)
        col = 0
        for _row, F, off in self.ml:
            # MovingLoad.station and __call__, every step at once
            sref = F.start + F.speed * t if F.speed is not None else F.start + p0 * (F.stop - F.start)
            sm = sref[:, None] + F._signed[None, :]
            on = (sm >= 0.0) & (sm <= F.path.total)
            k, a = F.path.place(sm)
            na = len(F.offsets)
            mk[:, col:col + na] = k + off; ma[:, col:col + na] = a; mon[:, col:col + na] = on
            col += na
        nv = len(self.veh)
        na_max, n_max = self.na_max, self.n_max
        V_mk = np.zeros((n_do, nv, na_max), np.int64); V_ma = np.zeros((n_do, nv, na_max))
        V_mon = np.zeros((n_do, nv, na_max), np.bool_); V_rough = np.zeros((n_do, nv, na_max))
        V_x = np.zeros((nv, n_max)); V_v = np.zeros((nv, n_max)); V_a = np.zeros((nv, n_max))
        V_contact = np.zeros((nv, na_max)); V_surf = np.zeros((nv, na_max)); V_has = np.zeros(nv, np.uint8)
        for vi, (_row, F, off) in enumerate(self.veh):
            na, n = F.na, len(F.m)
            sm = (F.start + F.speed * t)[:, None] + F.offsets[None, :]          # Vehicle.stations
            on = (sm >= 0.0) & (sm <= F.path.total)
            k, a = F.path.place(sm)
            V_mk[:, vi, :na] = k + off; V_ma[:, vi, :na] = a; V_mon[:, vi, :na] = on
            if F.rough is not None:
                V_rough[:, vi, :na] = F.rough(sm)
            surf, has = F._kernel_state()
            V_x[vi, :n] = F.x; V_v[vi, :n] = F.v; V_a[vi, :n] = F.a
            V_contact[vi, :na] = F.contact; V_surf[vi, :na] = surf; V_has[vi] = has[0]
        V_cnt = np.zeros((n_do, nv, 2), np.int64)
        self._chunk_movers = (mk, ma, mon, V_mk, V_ma, V_mon, V_rough, V_x, V_v, V_a, V_contact, V_surf,
                              V_has, V_cnt)
        return (self.g_kind, self.g_idx, self.mv, mk, ma, mon, self.ax_group, self.ax_p,
                self.P_R, self.P_L, self.P_edofs, self.P_rel, self.P_has, *self.V,
                V_mk, V_ma, V_mon, V_rough, V_x, V_v, V_a, V_contact, V_surf, V_has, V_cnt)

    def _book_movers(self, last_eval, p0):
        """What the loop's calls to the movers leave behind, after a chunk
        whose last evaluated step is `last_eval` (-1: none): the vehicles'
        state, their contact losses on the run's clock in step order, and,
        for the last step, each mover's force, members' share and instant
        (a probe that asks again at the same instant is handed the force)."""
        if not self.movers:
            return
        s = self.s
        (mk, ma, mon, V_mk, V_ma, V_mon, V_rough, V_x, V_v, V_a, V_contact, V_surf,
         V_has, V_cnt) = self._chunk_movers
        for vi, (_row, F, _off) in enumerate(self.veh):
            na, n = F.na, len(F.m)
            F.x[:] = V_x[vi, :n]; F.v[:] = V_v[vi, :n]; F.a[:] = V_a[vi, :n]
            F.contact[:] = V_contact[vi, :na]; F._surf_buf[:] = V_surf[vi, :na]; F._has_prev[0] = V_has[vi]
            if not V_has[vi]:
                # no step filled it (the kernel left before its first): as
                # the loop has it, and as a checkpoint writes it
                F._surf_prev = None
        if last_eval < 0:
            return
        rows = last_eval + 1
        if self.veh:
            for n_, vi in np.argwhere(V_cnt[:rows].any(axis=2)):
                F = self.veh[vi][1]
                t0 = float(p0[n_] * s.total_time)
                F._contact_lost(t0 + float(getattr(F, "clock_offset", 0.0)),
                                int(V_cnt[n_, vi, 0]), int(V_cnt[n_, vi, 1]))
        t0 = float(p0[last_eval] * s.total_time)
        for vi, (row, F, off) in enumerate(self.veh):
            na = F.na
            on = V_mon[last_eval, vi, :na].copy(); k = V_mk[last_eval, vi, :na] - off
            fl = np.zeros((na, 12))
            if on.any():
                sl = F.stations(t0)[on]
                _k, fl_on, _fg = F.path.shape(sl, F._R, F._L, np.broadcast_to(F.up, (len(sl), 3)),
                                               F.releases if F._has_rel else None)
                fl[on] = fl_on
            F._latch(self.mv[row].copy(), on, k, fl, t0)
        for _row, F, _off in self.ml:
            F(float(p0[last_eval]), t0, np.zeros(s.ndof))

    def _breaking(self):
        """The plastic hinges' and the failure table's arrays (1.6): the
        hinged members with their rows in the frame batch, Breaking's
        criterion arrays, and the parked particles and appendages."""
        s = self.s
        b = getattr(s, "breaking", None)
        nel = s.nel
        hinged = getattr(s, "_hinged", None)
        if hinged is not None and len(hinged) and getattr(s, "_plastic_d", None) is not None:
            ids = hinged[s.member_active[hinged]]
            where = {int(e): k for k, e in enumerate(self.f[0])}
            self.h_el = np.ascontiguousarray([int(e) for e in ids], dtype=np.int64)
            self.h_batch = np.ascontiguousarray([where[int(e)] for e in ids], dtype=np.int64)
            self.h_cap = np.ascontiguousarray(s._hinge_cap, dtype=float)
            self.plastic_d, self.hinge_rot = s._plastic_d, s._hinge_rotation
        else:
            self.h_el = np.zeros(0, np.int64); self.h_batch = np.zeros(0, np.int64)
            self.h_cap = np.zeros((1, 3)); self.plastic_d = np.zeros((1, 12)); self.hinge_rot = np.zeros((1, 2))
        if b is not None:
            self.brk = (bool(b.can_break), b._bar_cap, b._beam_on, b._beam_cap, b.member_failed,
                        b.member_active, b.end_broken, b.hinged, b.hinge_damage, b.rotation_capacity,
                        b.capacity_ratio)
        else:
            self.brk = (False, np.full(1, np.inf), np.zeros(1, bool), np.full((1, 4), np.inf), np.zeros(1, bool),
                        np.ones(1, bool), np.zeros((1, 2), bool), np.zeros(1, bool), np.zeros((1, 2)),
                        np.full(1, np.inf), np.full(1, np.nan))
        arrays = s._carry_arrays() if hasattr(s, "_carry_arrays") else None
        if arrays is None:
            arrays = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 3, 3)),
                      np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros((0, 6)))
        self.carry = arrays

    def _prescribed(self, prescribed):
        """The prescribed motions (orientation._apply_bc, 1.6): one entry per
        (dofs, value, amplitude, motion), its dofs flattened; the kinds the
        kernel's `_prescribe` takes; and the driven mask for the ledger."""
        s = self.s
        self.pres = list(prescribed)
        dof, ent, kind = [], [], []
        for i, (dofs, _value, a, motion) in enumerate(self.pres):
            d = np.asarray(dofs, dtype=np.int64).ravel()
            dof.append(d); ent.append(np.full(len(d), i, np.int64))
            kind.append(0 if motion == "displacement" else 1 if hasattr(a, "motion")
                        else 2 if motion == "velocity" else 3)
        self.p_dof = np.ascontiguousarray(np.concatenate(dof) if dof else np.zeros(0, np.int64), dtype=np.int64)
        self.p_entry = np.ascontiguousarray(np.concatenate(ent) if ent else np.zeros(0, np.int64), dtype=np.int64)
        self.p_kind = np.ascontiguousarray(np.asarray(kind, dtype=np.int64))
        driven = getattr(s, "_driven", None)
        self.driven = np.ascontiguousarray(driven if driven is not None else np.zeros(s.ndof, bool),
                                           dtype=np.bool_)
        self.half_dt2 = .5 * float(s.dt) ** 2

    def _targets(self, local):
        """(p_tq, p_tv) of the steps `local`: the boundary conditions are
        placed at the END of each step, progress = local / n_steps."""
        s = self.s
        n_do = len(local)
        E = len(self.pres)
        tq = np.zeros((n_do, E)); tv = np.zeros((n_do, E))
        if not E:
            return tq, tv
        prog = local / s.n_steps
        for i, (dofs, value, a, motion) in enumerate(self.pres):
            k = int(self.p_kind[i])
            if k == 1:
                for r, pg in enumerate(prog):
                    u, du = a.motion(float(pg))
                    tq[r, i] = value * u; tv[r, i] = value * du
                continue
            tq[:, i] = value * _factors(a, prog)
            if k == 0:
                rate = getattr(a, "rate", None)
                if rate is not None:
                    tv[:, i] = np.array([value * rate(float(pg)) for pg in prog], dtype=float)
        return np.ascontiguousarray(tq), np.ascontiguousarray(tv)

    def advance(self, first, last):
        """Steps `first` .. `last` (the loop's `local`, from 1) of this
        analysis step. Returns (status, F): status 0, or "NONFINITE" with
        the step that went non-finite counted in `solver.step`; F is the
        external force of the last step taken. A collapsed member raises
        ValueError, as the force evaluation does."""
        from ..kernels.numba_step import dynamic_chunk, dynamic_chunk_threaded
        from ..kernels.numba_frame import with_threads
        s = self.s
        local = np.arange(first, last + 1)
        p0 = (local - 1) / s.n_steps                       # (local - 1)/self.n_steps
        n_do = len(local)
        g_amp = (np.ascontiguousarray(np.stack([_factors(a, p0) for a in self.g_amps], axis=1))
                 if self.g_amps else np.zeros((n_do, 0)))
        s_i = np.zeros(n_do, np.int64); s_w = np.zeros(n_do); s_direct = np.zeros((n_do, 0))
        if self.s_mode == 1:
            P = self.s_P
            s_i = np.minimum(np.maximum(np.searchsorted(P, p0, side="right") - 1, 0),
                             len(P) - 2).astype(np.int64)
            s_w = np.minimum(np.maximum((p0 - P[s_i]) / (P[s_i + 1] - P[s_i]), 0.0), 1.0)
        elif self.s_mode == 2:
            s_direct = np.ascontiguousarray(np.stack([_factors(a, p0) for a in self.s_amps], axis=1))
        if self.ground:
            t = p0 * s.total_time
            m_ag = np.ascontiguousarray(np.stack([np.interp(t, ts, ac, left=0., right=0.)
                                                  for ts, ac, _d in self.ground], axis=1))
        else:
            m_ag = np.zeros((n_do, 0))
        # the members' live values, as the force evaluation reads them
        if len(self.t_ids):
            t_Lu = np.ascontiguousarray(s.member_Lu[self.t_ids], dtype=float)
            t_tonly = np.ascontiguousarray(s.member_tension_only[self.t_ids], dtype=np.bool_)
        else:
            t_Lu = np.zeros(0); t_tonly = np.zeros(0, np.bool_)
        t_dth6 = np.zeros(len(self.t_ids))
        led = s.diagnostics["energy"]
        led_on = bool(led.available)
        led_prev = led_on and led._u is not None
        if led_prev:
            self.u_prev[:] = led._u; self.fe_prev[:] = led._fext; self.fi_prev[:] = led._fint
        sums = np.array([led.external_work, led.internal_work, led.dissipated], float)
        kin = getattr(s, "_kinematic_v", None)
        seed = kin is None or kin.shape != s.v.shape
        kin_v = np.zeros(s.ndof) if seed else kin
        if getattr(s, "_rot_scratch", None) is None or s._rot_scratch.shape != (s.nn, 3):
            s._rot_scratch = np.zeros((s.nn, 3))
        f_ids, f_conn, f_R0, f_k, f_L0, _f_dofs, f_dth = self.f
        p_tq, p_tv = self._targets(local)
        pos = np.zeros(1, np.int64)
        n = self.threads
        fn = dynamic_chunk_threaded if n > 1 else dynamic_chunk
        args = (n_do, bool(s._half), float(s.dt), s.q, s.v, s.mass, s.orient, s._rot_scratch,
                s.mesh.nodes, f_ids, f_conn, f_R0, f_k, f_L0, f_dth, 4 * n if n > 1 else 1,
                self.t_conn, self.t_EA, t_Lu, t_dth6, t_tonly, self.t_ids,
                self.inc_ptr, self.inc_idx,
                self.c_ids, self.c_kind, np.ascontiguousarray(np.asarray(s.mesh.elements)[:, :2], dtype=np.int64),
                np.ascontiguousarray(np.asarray(s.R0, dtype=float)), self.c_spins, *self.c_enc,
                *self._slip_arrays(), self.cdeform, True,
                self.g_F, g_amp,
                self.s_mode, self.s_indptr, self.s_indices, self.s_data, self.s_A, s_i, s_w, s_direct,
                self.m_trans, m_ag, self.m_dir,
                self.dkind, s.alpha, s.beta, s.zeta,
                *self.K,
                self.fixed, self.fixed_zero, self.clamped_mask, kin_v, seed,
                self.driven, self.half_dt2, self.p_dof, self.p_entry, self.p_kind, p_tq, p_tv,
                self.h_el, self.h_batch, self.h_cap, self.brk[6], self.brk[5], self.plastic_d, self.hinge_rot,
                self.brk[9], self.brk[8],
                self.brk[0], self.brk[1], self.brk[2], self.brk[3], self.brk[4], self.brk[7], self.brk[10],
                *self.carry, *self._moving(p0), *self._contact(),
                led_on, led_prev, self.u_prev, self.fe_prev, self.fi_prev, sums,
                self.F, self.fint, self.damp, self.end_actions, self.actions, pos)
        try:
            status, done, half = with_threads(n, fn, *args) if n > 1 else fn(*args)
        except BaseException:
            # raised inside a step (a member that turned inside out, in a
            # kernel that still divides as Python does), or a KeyboardInterrupt
            # delivered as the call returns: `pos` counts the steps the state
            # has taken, and they are booked as the loop would have booked them
            # (found in review)
            done = int(pos[0])
            self._book(done, False, seed, kin_v, led_on, led, sums, p0)
            self._book_movers(done - 1, p0)
            self._book_contact(done)
            raise
        self._book(int(done), half, seed, kin_v, led_on, led, sums, p0)
        # the contact was evaluated in every step taken (a collapse or an
        # event leaves the kernel before the contact of its step)
        self._book_contact(int(done))
        # a collapse or an event stopped the kernel in step `done`, whose
        # external force -- the movers included -- it had evaluated
        self._book_movers(int(done) if status >= 2 else int(done) - 1, p0)
        if status == 3:
            # a member reached its capacity in step first + done (1.6), or
            # the contact's neighbour list outgrew its arrays there:
            # the loop takes that step
            self.done = int(done)
            return "HANDBACK", self.F.copy()
        if status >= 2:
            # the force evaluation of the next step ran up to the collapse
            # check, as the loop's `_force` had: its load instant is the one
            # left behind
            s._member_load_at = (self.load_groups, float(p0[min(int(done), n_do - 1)]))
            raise ValueError(self.collapse)
        return ("NONFINITE" if status == 1 else 0), self.F.copy()

    def _contact(self):
        """The contact between members for the kernel."""
        from .member_contact import no_chunk_args
        mc = getattr(self.s, "member_contact", None)
        return no_chunk_args() if mc is None else mc.chunk_args()

    def _book_contact(self, evaluations):
        mc = getattr(self.s, "member_contact", None)
        if mc is not None:
            mc.booked(evaluations)

    def _slip_arrays(self):
        s = self.s
        if len(self.c_ids):
            return s._conn_committed, s._conn_slip
        z = getattr(self, "_zslip", None)
        if z is None:
            z = self._zslip = (np.zeros((1, 6)), np.zeros((1, 6)))
        return z

    def _book(self, done, half, seed, kin_v, led_on, led, sums, p0):
        """What the loop leaves behind after `done` steps."""
        s = self.s
        s.step += int(done)
        if done:
            s._half = bool(half)
            if seed:
                s._kinematic_v = kin_v
            if led_on:
                led.external_work, led.internal_work, led.dissipated = (float(x) for x in sums)
                led._u, led._fext, led._fint = self.u_prev.copy(), self.fe_prev.copy(), self.fi_prev.copy()
            s.end_actions = self.end_actions.copy()
            if self.mixed:
                s.connector_deformation = self.cdeform.copy()
            s._member_load_at = (self.load_groups, float(p0[int(done) - 1]))


def _factors(a, p0):
    """a(p) at every p of `p0`: one `np.interp` for a table (the same numbers
    as one call per p -- `Amplitude.__call__` is that interpolation), one
    call per p otherwise."""
    from .amplitude import Amplitude
    if type(a) is Amplitude:
        return np.interp(p0, a.p, a.a).astype(float)
    return np.array([a(float(p)) for p in p0], dtype=float)


def drive(s, k, data, plan, on_output, output_every, on_progress, progress_every, what):
    """The dynamic steps of analysis step k of solver `s`: the runs of steps
    between two points where the loop has work of its own -- a progress
    check, an output frame, a removal that falls due -- through one compiled
    call each when the model and the step are ones the compiled step
    reproduces, the loop's own steps otherwise. `data` is the list [fixed,
    moving, prescribed, groups, ground] and is reloaded in place when a
    split moves mass; `what` names the solver in a non-finite message.

    ONE driver for the frame and the fibre solvers: each had
    its own copy, and a new status of the kernel was two edits that had to
    agree. What differs is the solver's own:

    * `s.compiled_pending()`: work a probe left for the end of the next
      step (a split, a parked particle) -- the loop takes that step;
    * `s.compiled_event_stop(k, local, last, plan)`: the first step of the
      run at which a scheduled removal falls due, or None;
    * `s.compiled_event(k, local, F, data, after)`: an EVENT the kernel
      finished the step of (a fibre point that failed), booked as the loop
      books it after that step's forces.

    HANDBACK is a step the kernel did not take (a member at its capacity,
    a contact list that outgrew its arrays): the loop takes it, and the
    compiled step is rebuilt on what it leaves."""
    from .diagnostics import finish as finish_diagnostics, NonFiniteFailure, first_nonfinite
    stepper_ = stepper(s, *data)
    compiled = stepper_ is not None
    local = s.step - s.step_start + 1

    def after(local, F):
        s.after_dynamic_step(k, local, F, data[0], plan, on_output, output_every, on_progress, progress_every)

    def loop_step(local):
        # one step by the loop's own body, and the compiled step rebuilt on
        # what it leaves (a break, a split, a removal, a reload)
        F = s.one_dynamic_step(k, local, data)
        after(local, F)
        if not compiled:
            return None
        if getattr(s, "_reload_step", False):
            data[:] = s.step_data(k)
            s.current_fixed = data[0]; s._reload_step = False
        return stepper(s, *data)

    while local <= s.n_steps:
        if stepper_ is not None and s.compiled_pending():
            stepper_ = None
        if stepper_ is None:
            stepper_ = loop_step(local)
            local += 1
            continue
        last = next_stop(local, s.n_steps, progress_every, output_every, stepper_.max_chunk)
        due = s.compiled_event_stop(k, local, last, plan)
        if due is not None and due == local:
            # a removal falls due in this very step: the loop's
            stepper_ = loop_step(local)
            local += 1
            continue
        if due is not None:
            last = due - 1
        status, F = stepper_.advance(local, last)
        if status == "NONFINITE":
            location = first_nonfinite(displacement=s.q, velocity=s.v)
            message = f"{what} dynamic integration became non-finite"
            finish_diagnostics(s.diagnostics, "NONFINITE", step=s.step, load_step=k + 1,
                               message=message, location=location)
            raise NonFiniteFailure(message, location=location)
        if status == "HANDBACK":
            local += stepper_.done
            try:
                stepper_ = loop_step(local)
            finally:
                # member forces the kernel left for this step's evaluation
                # (FiberStepper, status 5) are that evaluation's alone: a
                # step that failed before it must not leave them behind
                s.__dict__.pop("_kernel_resume", None)
            local += 1
            continue
        if status == "EVENT":
            local += stepper_.done - 1
            s.compiled_event(k, local, F, data, after)
            local += 1
            stepper_ = stepper(s, *data)
            continue
        after(last, F)
        local = last + 1


def next_stop(local, n_steps, progress_every, output_every, cap=MAX_CHUNK):
    """The first step at or after `local` where the loop has work of its own,
    at most `cap` steps on."""
    def up(m):
        return -(-local // m) * m
    return min(up(max(int(progress_every), 1)), up(max(int(output_every), 1)), n_steps,
               local + max(int(cap), 1) - 1)


class FiberStepper(Stepper):
    """The fibre beam's step: the fused fibre kernel and the
    compiled connectors carry the member forces; the loads, the damping, the
    ledger, the update and the supports are the frame step's own blocks."""

    def __init__(self, s, fixed, moving, prescribed, groups, ground):
        self.s = s
        self.groups = groups
        self.mixed = False
        nd = s.ndof
        self.load_groups = groups
        self._groups(list(groups))
        self.max_chunk = max(1, min(MAX_CHUNK, 2_000_000 // max(len(self.g_amps), 1)))
        self.ground = [(np.asarray(ts, float), np.asarray(ac, float), np.asarray(d, float).reshape(3))
                       for ts, ac, d in ground]
        self.m_dir = (np.ascontiguousarray(np.stack([d for _t, _a, d in self.ground]))
                      if self.ground else np.zeros((0, 3)))
        self.m_trans = (np.ascontiguousarray(s.dofs.field(s.physical_mass, TRANSLATION), dtype=float)
                        if self.ground else np.zeros((s.nn, 3)))
        self.dkind = {"rayleigh": 1, "viscous": 2}.get(s.damp_kind, 0)
        if self.dkind == 1:
            K = s.K
            self.K = (np.ascontiguousarray(K.indptr, dtype=np.int64),
                      np.ascontiguousarray(K.indices, dtype=np.int64),
                      np.ascontiguousarray(K.data, dtype=float))
        else:
            self.K = (np.zeros(nd + 1, np.int64), np.zeros(0, np.int64), np.zeros(0))
        self.fixed = np.ascontiguousarray(fixed, dtype=np.bool_)
        fixed_zero = fixed & ~moving
        self.fixed_zero = np.ascontiguousarray(fixed_zero, dtype=np.bool_)
        self.clamped_mask = np.ascontiguousarray(s.dofs.view(fixed_zero)[:, 3:].all(axis=1), dtype=np.bool_)
        self.threads = int(getattr(s, "_kernel_threads", 1) or 1)
        self.F = np.zeros(nd); self.fint = np.zeros(nd); self.damp = np.zeros(nd)
        self.end_actions = np.zeros((s.nel, 12)); self.actions = np.zeros((s.nel, 10))
        self.u_prev = np.zeros(nd); self.fe_prev = np.zeros(nd); self.fi_prev = np.zeros(nd)
        self._prescribed(prescribed)
        self.c_ids, self.c_kind, ck = s.kernel_connectors()
        if ck is not None:
            self.c_enc, self.c_spins = ck["enc"], ck["spins"]
        else:
            from ..kernels.numba_connector import encode, spins
            self.c_enc, self.c_spins = encode([{}]), spins()
        self.collapse = None

    def advance(self, first, last):
        from ..kernels.numba_step import fiber_chunk
        from ..kernels.numba_frame import with_threads
        s = self.s
        local = np.arange(first, last + 1)
        p0 = (local - 1) / s.n_steps
        n_do = len(local)
        g_amp = (np.ascontiguousarray(np.stack([_factors(a, p0) for a in self.g_amps], axis=1))
                 if self.g_amps else np.zeros((n_do, 0)))
        if self.ground:
            t = p0 * s.total_time
            m_ag = np.ascontiguousarray(np.stack([np.interp(t, ts, ac, left=0., right=0.)
                                                  for ts, ac, _d in self.ground], axis=1))
        else:
            m_ag = np.zeros((n_do, 0))
        led = s.diagnostics["energy"]
        led_on = bool(led.available)
        led_prev = led_on and led._u is not None
        if led_prev:
            self.u_prev[:] = led._u; self.fe_prev[:] = led._fext; self.fi_prev[:] = led._fint
        sums = np.array([led.external_work, led.internal_work, led.dissipated], float)
        kin = getattr(s, "_kinematic_v", None)
        seed = kin is None or kin.shape != s.v.shape
        kin_v = np.zeros(s.ndof) if seed else kin
        s.fused_kernel_prepare()
        p_tq, p_tv = self._targets(local)
        fr = s.fracture_kernel_arrays()
        if fr is None:
            fr = (np.zeros(0, np.int64), np.zeros(0), np.zeros((s.nel, 0)), np.zeros((s.nel, 0)))
        pos = np.zeros(1, np.int64)
        fn = fiber_chunk(s.fused_kernel_kind())
        args = (n_do, bool(s._half), float(s.dt), s.q, s.v, s.mass, s.orient, *s.fused_kernel_section(),
                self.c_ids, self.c_kind, np.ascontiguousarray(np.asarray(s.mesh.elements)[:, :2], dtype=np.int64),
                s.mesh.nodes, self.c_spins, *self.c_enc, *s.fused_kernel_connector_state(), True,
                self.g_F, g_amp, self.m_trans, m_ag, self.m_dir,
                self.dkind, s.alpha, s.beta, s.zeta, *self.K,
                self.fixed, self.fixed_zero, self.clamped_mask, kin_v, seed,
                self.driven, self.half_dt2, self.p_dof, self.p_entry, self.p_kind, p_tq, p_tv,
                len(fr[0]) > 0, *fr, *s.shear_law_args(), *self._moving(p0),
                *self._contact(),
                led_on, led_prev, self.u_prev, self.fe_prev, self.fi_prev, sums,
                self.F, self.fint, self.damp, self.actions, self.end_actions, pos)
        n = self.threads
        try:
            status, done, half = with_threads(n, fn, *args) if n > 1 else fn(*args)
        except BaseException:
            done = int(pos[0])
            self._book(done, False, seed, kin_v, led_on, led, sums, p0)
            self._book_movers(done - 1, p0)
            self._book_contact(done)
            raise
        self._book(int(done), half, seed, kin_v, led_on, led, sums, p0)
        self._book_movers(int(done) - 1, p0)
        self._book_contact(int(done))
        if status == 5:
            # a fibre point failed in step first + done of a model with
            # contact between members: the kernel made the member forces of
            # that step and stopped; the loop's force evaluation takes them
            # (FiberBeamSolver._internal_force_kernel), latches, and goes on
            # to the contact
            s._kernel_resume = (int(s.step), self.fint.copy(), self.actions.copy(), self.end_actions.copy())
            self.done = int(done)
            return "HANDBACK", self.F.copy()
        if status == 4:
            # the contact's neighbour list has outgrown its arrays in step
            # first + done, of which nothing has been done: the loop takes it
            # and makes them longer
            self.done = int(done)
            return "HANDBACK", self.F.copy()
        if status == 3:
            # a fibre point failed in the last step taken (first + done - 1):
            # the step is complete, and the loop latches the member (1.6)
            self.done = int(done)
            return "EVENT", self.F.copy()
        return ("NONFINITE" if status == 1 else 0), self.F.copy()

    def _book(self, done, half, seed, kin_v, led_on, led, sums, p0):
        s = self.s
        super()._book(done, half, seed, kin_v, led_on, led, sums, p0)
        if done:
            # the fibre loop's clock at the start of the last step taken, and
            # the actions as _internal_force_kernel leaves them
            s.time = (s.step_index + float(p0[int(done) - 1])) * s.total_time
            station_abs = np.where(s._kernel_active[:, None].astype(bool), self.actions[:, 6:10], np.nan)
            s._set_actions_abs(np.ascontiguousarray(self.actions[:, :6]), station_abs)


# ------------------------------------------------------------ relaxation
def relax_refusal(solver, prescribed, start):
    """None when the compiled relaxation (1.9) reproduces a
    level of `solver`'s staged relaxation (`BeamSolver._relax_stage`), else
    why not. The frames, bars and connectors of the dynamic step, under
    `backend: numba`; the fibre models keep their loop (their kernel is the
    cost there, and `solver.threads` is what speeds it up)."""
    if not enabled():
        return refuse("off")
    from ..kernels import numba_step
    if not numba_step.available() or numba_step.relax_chunk is None:
        return refuse("no_numba")
    from .beam_solver import BeamSolver
    from .mixed_structural_solver import MixedStructuralSolver
    kind = type(solver)
    if kind not in (BeamSolver, MixedStructuralSolver):
        from .fiber_beam_solver import FiberBeamSolver
        return refuse("fibre_relax" if kind is FiberBeamSolver else "own_forces", kind=kind.__name__)
    if getattr(solver, "backend", "numpy") != "numba":
        return refuse("backend")
    hooked = [n for n in _HOOKS + ("_relax_stage", "_convergence_measures", "_residual_norm")
              if n in vars(solver) or getattr(kind, n, None) is not _original(kind, n)]
    if hooked or solver.components or getattr(solver, "contact", None) is not None:
        # any component: the relaxation chunk runs no contact at all
        return refuse("hooked", what=", ".join(hooked) or "contact")
    if solver.constraint_layer().links is not None:
        return refuse("rigid_links")
    if solver.dofs.stride != 6 or solver.q.size != 6 * solver.nn:
        return refuse("six_dofs")
    if solver.plan is not None:
        return refuse("steps")
    for dofs, _value, a, motion in prescribed:
        if motion != "displacement":
            return refuse("relax_motion", motion=motion)
        if np.any((np.asarray(dofs) % 6) >= 3):
            return refuse("rotation")
    if start is not None and len(start) != len(prescribed):
        return refuse("relax_start")
    if getattr(solver, "_thermal", None) or solver.time_dependent \
            or getattr(solver, "_thermal_d", None) is not None:
        return refuse("thermal")
    if getattr(solver, "_plastic_d", None) is not None:
        return refuse("hinges")
    if solver.constraint_layer().diaphragms is not None:
        return refuse("diaphragms")
    soil = getattr(solver, "soil", None)
    if soil is not None and len(soil):
        return refuse("soil")
    if solver.geometry.any:
        return refuse("end_offsets")
    kv = getattr(solver, "_kinematic_v", None)
    if kv is not None and kv.shape != solver.v.shape:
        return refuse("grown")
    led = solver.diagnostics.get("energy") if isinstance(getattr(solver, "diagnostics", None), dict) else None
    if led is None:
        return refuse("ledger")
    why = _arrays_refusal_relax(solver)
    if why is not None:
        return why
    if kind is MixedStructuralSolver:
        solver._beam_batch(); solver._truss_batch()
        if len(getattr(solver, "_connector_ids", ())) and solver._connector_kernel() is None:
            return refuse("connectors")
        known = np.concatenate([solver._truss_ids, getattr(solver, "_connector_ids", np.zeros(0, int))])
        if len(solver._other_ids) != len(known) or \
                not np.array_equal(np.sort(solver._other_ids), np.sort(known)):
            return refuse("other_members")
        if len(solver._truss_ids) and list(solver._truss_T) != [0, 1, 2]:
            return refuse("bars_dofs")
        if np.ndim(getattr(solver, "_k_factor", 1.0)):
            return refuse("bars_creep")
    return None


def _arrays_refusal_relax(solver):
    for name, shape in (("q", (solver.ndof,)), ("v", (solver.ndof,)), ("mass", (solver.ndof,)),
                        ("orient", (solver.nn, 3, 3))):
        a = getattr(solver, name)
        if not isinstance(a, np.ndarray) or a.shape != shape or a.dtype != np.float64 \
                or not a.flags.c_contiguous:
            return refuse("arrays", name=name, shape=shape)
    return None


def relax_stepper(solver, fixed, moving, prescribed, start):
    """A `RelaxStepper` for the level, or None (`solver.compiled_relax_refusal`
    says why). The member arrays are kept between levels while the members
    and the supports are the same."""
    why = relax_refusal(solver, prescribed, start)
    solver.compiled_relax_refusal = why
    if why is not None:
        return None
    from .mixed_structural_solver import MixedStructuralSolver
    batch = solver._beam_batch() if isinstance(solver, MixedStructuralSolver) else solver._frame_batch()
    key = (id(batch), id(getattr(solver, "_truss_ids", None)), id(getattr(solver, "_connector_ids", None)),
           solver.nel, solver.ndof)
    rs = getattr(solver, "_relax_stepper", None)
    if rs is None or rs.key != key or rs.batch is not batch:
        rs = RelaxStepper(solver)
        rs.key, rs.batch = key, batch
        solver._relax_stepper = rs
    return rs


class RelaxStepper(Stepper):
    """The compiled relaxation (1.9): the iterations of one
    level of `BeamSolver._relax_stage` in chunks of the kernel
    `relax_chunk`, ending at every progress check (where the loop reports
    and a checkpoint can be written) -- the same numbers as the loop, to the
    bit, with the loop's work in Python left at the checks."""

    def __init__(self, s):
        self.s = s
        self._members(s)
        nd = s.ndof
        self.fint = np.zeros(nd)
        self.u_prev = np.zeros(nd); self.fe_prev = np.zeros(nd); self.fi_prev = np.zeros(nd)
        if getattr(s, "_rot_scratch", None) is None or s._rot_scratch.shape != (s.nn, 3):
            s._rot_scratch = np.zeros((s.nn, 3))

    def _supports(self, fixed, moving, prescribed, progress, start):
        s = self.s
        fixed_zero = fixed & ~moving
        self.fixed = np.ascontiguousarray(fixed, dtype=np.bool_)
        self.fixed_zero = np.ascontiguousarray(fixed_zero, dtype=np.bool_)
        self.clamped_mask = np.ascontiguousarray(s.dofs.view(fixed_zero)[:, 3:].all(axis=1), dtype=np.bool_)
        self.free_idx = np.ascontiguousarray(np.flatnonzero(~fixed), dtype=np.int64)
        dof, ptr, tgt, st, vel = [], [0], [], [], []
        for i, (dofs, value, a, _motion) in enumerate(prescribed):
            d = np.asarray(dofs, dtype=np.int64).ravel()
            # orientation._apply_bc: the same expressions, per entry
            target = np.broadcast_to(np.asarray(value * a(progress), float), d.shape)
            rate = getattr(a, "rate", None)
            v = np.broadcast_to(np.asarray(value * rate(progress) if rate is not None else 0., float), d.shape)
            dof.append(d); tgt.append(target); vel.append(v); ptr.append(ptr[-1] + len(d))
            st.append(np.broadcast_to(np.asarray(start[i], float), d.shape) if start is not None
                      else np.zeros(len(d)))
        c = np.ascontiguousarray
        cat = (lambda xs, t=float: c(np.concatenate(xs), dtype=t) if xs else np.zeros(0, t))
        self.p_dof = cat(dof, np.int64); self.p_ptr = c(ptr, dtype=np.int64)
        self.p_target = cat(tgt); self.p_start = cat(st); self.p_vel = cat(vel)
        self.has_start = start is not None

    def relax(self, F, fixed, moving, prescribed, progress, start, tol, budget, ramp, it0,
              progress_every, report, cv_s, stage, what, measures):
        from ..kernels.numba_step import relax_chunk, relax_chunk_threaded
        from ..kernels.numba_frame import with_threads
        from .diagnostics import finish as finish_diagnostics, NonFiniteFailure, StopAnalysis, first_nonfinite
        s = self.s
        self._supports(fixed, moving, prescribed, progress, start)
        F = np.ascontiguousarray(F, dtype=float)
        free = ~fixed
        from ..kernels.numba_step import norm2
        F_norm = float(norm2(np.ascontiguousarray(F[free])))   # the loop's _norm_fn
        c1 = 1.0/(1.0 + .5*cv_s*s.dt); c2 = c1*(1.0 - .5*cv_s*s.dt)
        resid, kr, ke = measures if measures is not None else (np.inf, np.inf, 0.0)
        meas = np.zeros(6)
        meas[:3] = getattr(s, "_loop_measures", (0.0, 0.0, 0.0))
        had_fs = hasattr(s, "_force_scale"); had_ms = hasattr(s, "_motion_scale")
        fst = np.array([float(s.ke_prev), float(getattr(s, "_force_scale", 0.0)),
                        float(getattr(s, "_motion_scale", 0.0))])
        ist = np.array([int(s.n_resets)], np.int64)
        led = s.diagnostics["energy"]
        f_ids, f_conn, f_R0, f_k, f_L0, _f_dofs, f_dth = self.f
        if len(self.t_ids):
            t_Lu = np.ascontiguousarray(s.member_Lu[self.t_ids], dtype=float)
            t_tonly = np.ascontiguousarray(s.member_tension_only[self.t_ids], dtype=np.bool_)
        else:
            t_Lu = np.zeros(0); t_tonly = np.zeros(0, np.bool_)
        t_dth6 = np.zeros(len(self.t_ids))
        conn2 = np.ascontiguousarray(np.asarray(s.mesh.elements)[:, :2], dtype=np.int64)
        R0 = np.ascontiguousarray(np.asarray(s.R0, dtype=float))
        n = self.threads
        fn = relax_chunk_threaded if n > 1 else relax_chunk
        pe = max(int(progress_every), 1)
        it = it0 + 1
        converged = False
        while it <= budget:
            last = min(budget, -(-it // pe) * pe)
            led_on = bool(led.available)
            led_prev = led_on and led._u is not None
            if led_prev:
                self.u_prev[:] = led._u; self.fe_prev[:] = led._fext; self.fi_prev[:] = led._fint
            sums = np.array([led.external_work, led.internal_work, led.dissipated], float)
            kin = getattr(s, "_kinematic_v", None)
            seed = kin is None
            kin_v = np.zeros(s.ndof) if seed else kin
            pos = np.zeros(1, np.int64)
            args = (it, last, float(ramp), float(tol), float(s.kinetic_tolerance), bool(s._half), float(s.dt),
                    c1, c2, float(cv_s),
                    s.q, s.v, s.mass, s.orient, s._rot_scratch, s.mesh.nodes,
                    f_ids, f_conn, f_R0, f_k, f_L0, f_dth, 4 * n if n > 1 else 1,
                    self.t_conn, self.t_EA, t_Lu, t_dth6, t_tonly, self.t_ids,
                    self.inc_ptr, self.inc_idx,
                    self.c_ids, self.c_kind, conn2, R0, self.c_spins, *self.c_enc,
                    *self._slip_arrays(), self.cdeform,
                    F, F_norm, self.free_idx, self.fixed, self.fixed_zero, self.clamped_mask, kin_v, seed,
                    self.p_dof, self.p_ptr, self.p_target, self.p_start, self.p_vel, self.has_start,
                    led_on, led_prev, self.u_prev, self.fe_prev, self.fi_prev, sums,
                    fst, ist, meas, self.fint, self.end_actions, self.actions, pos)
            try:
                status, done, conv, half = with_threads(n, fn, *args) if n > 1 else fn(*args)
            except BaseException:
                self._book_relax(int(pos[0]), bool(s._half), seed, kin_v, led_on, led, sums, fst, ist,
                                 had_fs, had_ms)
                raise
            done = int(done)
            self._book_relax(done, half, seed, kin_v, led_on, led, sums, fst, ist, had_fs, had_ms)
            if status != 0 and (done > 1 or (done == 1 and status >= 2)):
                # the last iteration that finished its measures
                s._loop_measures = (float(meas[0]), float(meas[1]), float(meas[2]))
            if status == 1:
                location = first_nonfinite(displacement=s.q, velocity=s.v)
                finish_diagnostics(s.diagnostics, "NONFINITE", step=s.step, load_step=stage,
                                   message=f"{what} became non-finite", location=location)
                raise NonFiniteFailure(f"{what} became non-finite; reduce relaxation_safety or check for "
                                       "a structural mechanism", location=location)
            if status >= 2:
                raise ValueError(self.collapse)
            if done:
                resid, kr, ke = float(meas[0]), float(meas[1]), float(meas[2])
                lastit = it + done - 1
                s._loop_measures = (resid, kr, ke)
                if lastit % pe == 0:
                    s.residuals.append(float(resid)); s.kinetic.append(float(ke))
                    if report is not None and report(resid, ke) is False:
                        raise StopAnalysis()
                it = lastit + 1
            if conv:
                converged = True
                break
        return converged, resid, kr, ke

    def _book_relax(self, done, half, seed, kin_v, led_on, led, sums, fst, ist, had_fs, had_ms):
        s = self.s
        s.step += int(done)
        if not done:
            return
        s._half = bool(half)
        s.ke_prev = float(fst[0])
        s.n_resets = int(ist[0])
        s._force_scale = float(fst[1]); s._motion_scale = float(fst[2])
        if seed:
            s._kinematic_v = kin_v
        if led_on:
            led.external_work, led.internal_work, led.dissipated = (float(x) for x in sums)
            led._u, led._fext, led._fint = self.u_prev.copy(), self.fe_prev.copy(), self.fi_prev.copy()
        s.end_actions = self.end_actions.copy()
        if self.mixed:
            s.connector_deformation = self.cdeform.copy()
