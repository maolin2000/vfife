"""The members of the mixed structural solver, by family (P2 of the
architecture review).

MixedStructuralSolver holds frame members, axial bars and connectors on one
six-dof particle system. Until 1.19 each kind's arrays and its force were
methods and attributes of the solver class (`_beam_batch`,
`_beam_forces_batched`, `_truss_batch`, `_truss_forces_batched`, the
connector loop inside `_member_forces`, a branch per kind in
`_refresh_fictitious_mass`). Here each kind is a family: an object that
gathers its members' arrays once and gives

    forces(s, out, actions, end_actions[, deformation])   its internal force
    turned_blocks(s)                                      its stiffness turned
                                                          into the current frame

with the solver `s` the family belongs to (its `q`, `orient`, `mesh`,
`dofs`, backend and imposed deformations are the solver's). The arithmetic
is the solver's own, moved and not rewritten: the answers are the same to
the bit (`tools/state_baseline.py`).

A family is built from the members active when it is built, and rebuilt
when the solver's members change (`MixedStructuralSolver.rebatch`). The
solver keeps the names other modules read (`_beam_ids`, `_other_ids`,
`_truss_ids`, `_truss_T`, `_connector_ids`) as plain aliases of the
families' arrays. What a family reads of its solver's private state is
counted per file by tests/test_solver_boundary.py, as any other module's.

This is the first step of "one solver and member families"
(as revised): the frame and fibre-beam solvers are to take their
members from families too, routed inside the solver (rewriting a case to
route it changes a checkpoint's layout and refuses a restart).
"""
from __future__ import annotations

import numpy as np

from ..base.dof import TRANSLATION, repeated_rotation
from ..base.rotations import beam_frames_from_triads

#: what a member that collapses to a point is told
COLLAPSED = "mixed structural member collapsed to zero length"


def _active(s):
    active = getattr(s, "member_active", None)
    return np.ones(s.nel, bool) if active is None else active


class FrameMembers:
    """The frame members (`BEAM_TYPES`): co-rotational beams with the
    classical local 12x12 law, reverse motion through the points' triads.

    Two ways to have them: `FrameMembers(s)`, the mixed solver's frame
    members among its other kinds (gathered copies of their arrays), and
    `FrameMembers.whole(nel, dofs, conn, R0, k, L0, zero_dth)`, every member
    of the frame solver (BeamSolver) over its own batch
    (`_frame_batch`), the NumPy law read live from `k_local`, `R0`,
    `length` as that solver always read it (a broken member's stiffness is
    zeroed in place, without a rebatch), and its imposed deformation
    `_frame_eigen_d` (no hinges: they are the mixed solver's)."""

    #: every member of the solver, read live (`whole`)
    is_whole = False
    #: the members' end offsets gathered for the kernel (core/member_geometry), at first use
    _offs = None
    #: what a member that collapses to a point is told
    collapse = COLLAPSED

    @classmethod
    def whole(cls, nel, dofs, conn, R0, k, L0, zero_dth):
        """Every member of the frame solver, over the arrays of its batch
        (`BeamSolver._frame_batch`), which this shares and does not copy."""
        f = cls.__new__(cls)
        f.is_whole = True
        f.collapse = "beam2 member collapsed to zero length"
        f.dofs = f.dofs64 = dofs
        f.conn64, f.R0c, f.kc, f.L0c, f.zero_dth = conn, R0, k, L0, zero_dth
        f.ids = f.ids64 = np.arange(nel, dtype=np.int64)
        f.other_ids = np.zeros(0, dtype=np.int64)
        return f

    def __init__(self, s):
        active = _active(s)
        ids = np.flatnonzero(np.isin(s.member_kind, tuple(s.BEAM_TYPES)) & active)
        self.ids = ids
        self.dofs = (np.array([s.dofs.element_dofs(s.mesh.elements[e]) for e in ids])
                     .reshape(len(ids), 12) if len(ids) else np.zeros((0, 12), dtype=int))
        self.conn = s.mesh.elements[ids]
        self.R0 = s.R0[ids]
        self.k = s.k_local[ids]
        self.L0 = s.length[ids]
        #: every other member that carries force (the axial families')
        self.other_ids = np.flatnonzero(~np.isin(s.member_kind, tuple(s.BEAM_TYPES)) & active)
        # contiguous, typed copies for the kernel
        self.ids64 = np.ascontiguousarray(ids, dtype=np.int64)
        self.conn64 = np.ascontiguousarray(self.conn[:, :2], dtype=np.int64)
        self.dofs64 = np.ascontiguousarray(self.dofs, dtype=np.int64)
        self.R0c = np.ascontiguousarray(np.asarray(self.R0, dtype=float))
        self.kc = np.ascontiguousarray(np.asarray(self.k, dtype=float))
        self.L0c = np.ascontiguousarray(np.asarray(self.L0, dtype=float))
        self.zero_dth = np.zeros((len(ids), 12))
        if getattr(s, "backend", "numpy") == "numba":
            from ..kernels.numba_frame import configure_threads
            s._threads = configure_threads(s.p.solver.threads)

    def forces(self, s, out, actions, end_actions):
        """VFIFE reverse motion for every frame member at once.

        The co-rotational frame and the local nodal rotations come from
        `rotations.beam_frames_from_triads` (exact in space since
        v0.1: the frame follows the mean nodal triad as well as the
        chord, and the local rotation is the relative rotation of the node's
        triad, not a difference of rotation vectors); the triads are the
        points' orientations. The classical local 12x12 law
        then gives end actions that are rotated back, spatial moments that
        drive the points' spin directly.
        """
        ids = self.ids
        if not len(ids):
            return
        whole = self.is_whole
        geometry = s.geometry
        offset = geometry.any
        if getattr(s, "backend", "numpy") == "numba":
            from ..kernels.numba_frame import frame_forces_subset, frame_forces_subset_threaded
            dth = (s._frame_eigen_d() if whole else s._eigen_d())
            dth = self.zero_dth if dth is None else np.ascontiguousarray(dth if whole else dth[ids])
            args = (self.ids64, s.mesh.nodes, self.conn64,
                    s.dofs.view(s.q), s.orient, self.R0c, self.kc,
                    self.L0c, self.dofs64, dth, out, end_actions, actions)
            if offset:
                # members with end offsets (1.21): their own kernel, serial
                from ..kernels.numba_frame import frame_forces_offset
                if self._offs is None:
                    self._offs = geometry.kernel_arrays(ids)
                if frame_forces_offset(*args, *self._offs):
                    raise ValueError(self.collapse)
                return
            n = getattr(s, "_threads", 1)
            # threads: the same sum, to the bit
            if n > 1:
                from ..kernels.numba_frame import with_threads
                bad = with_threads(n, frame_forces_subset_threaded, *args, 4 * n)
            else:
                # every member too: `frame_forces_subset` with ids 0..n-1 is
                # `frame_forces` statement for statement
                bad = frame_forces_subset(*args)
            if bad:
                raise ValueError(self.collapse)
            return
        dofs = self.dofs
        if whole:                                  # read live (see the class)
            conn, R0, L0, k = s.mesh.elements, s.R0, s.length, np.asarray(s.k_local)
        else:
            conn, R0, L0, k = self.conn, self.R0, self.L0, self.k
        q = s.q[dofs].reshape(-1, 2, 6)
        x = s.mesh.nodes[conn] + q[:, :, :3]
        x = geometry.ends(ids, x, s.orient)
        R, theta_loc, L = beam_frames_from_triads(x, s.orient[conn], R0)
        if np.any(L < 1e-14):
            raise ValueError(self.collapse)
        local = np.zeros((len(ids), 12))
        local[:, 6] = L - L0
        local[:, 3:6] = theta_loc[:, 0]
        local[:, 9:12] = theta_loc[:, 1]
        eig = (s._frame_eigen_d() if whole else s._eigen_d())
        if eig is not None:
            # f = k (d - d_th - d_p), core/temperature, v0.5
            local = local - (eig if whole else eig[ids])
        fl = np.einsum("eij,ej->ei", k, local)
        end_actions[ids] = fl
        # T.T @ fl with T = blockdiag(R, R, R, R): each 3-block is R.T @ block
        contrib = np.einsum("eji,ekj->eki", R, fl.reshape(-1, 4, 3)).reshape(-1, 12)
        geometry.forces_to_nodes(ids, contrib, s.orient)
        np.add.at(out, dofs, contrib)
        actions[ids] = np.maximum(np.abs(fl[:, :6]), np.abs(fl[:, 6:]))

    def rotations(self, s):
        """R for every frame member in the CURRENT configuration -- the same
        frame the force evaluation uses, without the forces."""
        q = s.q[self.dofs].reshape(-1, 2, 6)
        x = s.mesh.nodes[self.conn] + q[:, :, :3]
        x = s.geometry.ends(self.ids, x, s.orient)
        return beam_frames_from_triads(x, s.orient[self.conn], self.R0)[0]

    def turned_blocks(self, s):
        """(position, element, T^T k T) of each frame member, its stiffness
        turned into its current frame (the relaxation mass,
        `BeamFamily._refresh_fictitious_mass`)."""
        ids = self.ids
        R = self.rotations(s)
        T = np.zeros((len(ids), 12, 12))
        for blk in range(4):
            T[:, 3*blk:3*blk+3, 3*blk:3*blk+3] = R
        kg = np.einsum("eji,ejk,ekl->eil", T, self.k, T)     # T^T k T
        geometry = s.geometry
        r = geometry.now(ids, s.orient)
        for i, e in enumerate(ids):
            yield i, int(e), (kg[i] if r is None else geometry.stiffness_to_nodes(int(e), kg[i], r[i]))


class AxialMembers:
    """The bars (`truss2`), batched since v0.4: "trusses and
    connectors are few and stay in the loop" was true of every frame this
    solver was written for, and not of a cable bridge -- the Xihoumen model
    has 346 cable members against 847 beams, and the per-element loop over
    them was 35 of a 39 ms force evaluation. The arithmetic is
    `MixedStructuralSolver._truss_force` element for element (E A first,
    then the strain).

    Built together with the connectors (`ConnectorMembers`): the two were
    one batch, and a model's other members are exactly these two."""

    def __init__(self, s):
        active = _active(s)
        ids = np.flatnonzero((s.member_kind == "truss2") & active)
        self.ids = ids
        self.conn = np.ascontiguousarray(s.mesh.elements[ids][:, :2])
        self.EA = np.array([s.member_E[e] * float(s.member_props[e]["area"]) for e in ids])
        self.T = s.dofs.component_indices(TRANSLATION)
        self._c64_for = None

    def forces(self, s, out, actions, end_actions):
        ids = self.ids
        if not len(ids):
            return
        if getattr(s, "backend", "numpy") == "numba" and s.dofs.stride == 6 \
                and list(self.T) == [0, 1, 2]:
            # one compiled loop instead of temporaries and two np.add.at
            # scatters: 0.30 -> 0.03 ms of a Xihoumen step
            from ..kernels.numba_frame import truss_forces
            if self._c64_for is not ids:
                self._c64_for = ids
                self.c64 = np.ascontiguousarray(self.conn, dtype=np.int64)
                self.ids64 = np.ascontiguousarray(ids, dtype=np.int64)
                self.EAc = np.ascontiguousarray(self.EA, dtype=float)
                self.zero = np.zeros(len(ids))
            # read live, as the NumPy path does: a script (or a later stage)
            # that changes a cable's stress-free length or its tension-only
            # flag after the first evaluation was ignored by a cached copy
            # (found in review); 350 gathers cost nothing
            lu = np.ascontiguousarray(s.member_Lu[ids], dtype=float)
            tonly = np.ascontiguousarray(s.member_tension_only[ids], dtype=np.bool_)
            dth = getattr(s, "_thermal_d", None)
            dth6 = self.zero if dth is None else np.ascontiguousarray(dth[ids, 6], dtype=float)
            EAc = self.EAc
            kf = getattr(s, "_k_factor", 1.0)
            if np.ndim(kf):
                EAc = np.ascontiguousarray(self.EA * kf[ids], dtype=float)
            bad = truss_forces(self.c64, s.mesh.nodes, s.dofs.view(s.q), EAc,
                               lu, dth6, tonly, self.ids64,
                               out, end_actions, actions)
            if bad:
                raise ValueError(COLLAPSED)
            return
        c = self.conn
        qv = s.dofs.view(s.q)[:, self.T]
        x0 = s.mesh.nodes[c[:, 0]] + qv[c[:, 0]]
        x1 = s.mesh.nodes[c[:, 1]] + qv[c[:, 1]]
        axis = x1 - x0
        L = np.sqrt(np.einsum("ij,ij->i", axis, axis))
        if np.any(L < 1e-14):
            raise ValueError(COLLAPSED)
        Lu = s.member_Lu[ids]
        dth = getattr(s, "_thermal_d", None)
        EA = self.EA
        kf = getattr(s, "_k_factor", 1.0)
        if np.ndim(kf):
            # a bar's creep and ageing: the 1/(1 + phi) the
            # beams take through k_local. Without it a held bar's force saw the
            # creep strain but not the station's own creep, and crept back
            EA = EA * kf[ids]
        if dth is not None:
            # f = k (d - d_th) for a bar: its free thermal elongation comes off
            # the stretch before E A sees it
            N = EA * (L - Lu - dth[ids, 6]) / Lu
        else:
            N = EA * (L - Lu) / Lu
        N = np.where(s.member_tension_only[ids] & (N < 0.0), 0.0, N)
        f = (N / L)[:, None] * axis
        ov = s.dofs.view(out)
        T = self.T
        np.add.at(ov, (c[:, 0][:, None], T[None, :]), -f)
        np.add.at(ov, (c[:, 1][:, None], T[None, :]), f)
        end_actions[ids, 0] = -N; end_actions[ids, 6] = N
        actions[ids, 0] = np.abs(N)

    @staticmethod
    def mass_block(s, e):
        """Bar `e`'s axial stiffness along its CURRENT axis (a cable's along
        every axis, `_assemble_reference` says why)."""
        conn = s.mesh.elements[e]; dof = s.dofs.element_dofs(conn)
        x = s.mesh.nodes[conn] + s.q[dof].reshape(2, 6)[:, :3]
        ka = s.member_E[e] * float(s.member_props[e]["area"]) / s.length[e]
        if s._is_cable(s.member_props[e]):
            kg = np.zeros((12, 12))            # see _assemble_reference
            for a_, b_ in ((0, 6), (6, 0)):
                kg[a_:a_+3, a_:a_+3] += ka*np.eye(3); kg[a_:a_+3, b_:b_+3] -= ka*np.eye(3)
            return dof, kg
        kl = np.zeros((12, 12)); kl[0, 0] = kl[6, 6] = ka; kl[0, 6] = kl[6, 0] = -ka
        Rt = s._frame(x[0], x[1], s.R0[e, 1])[0]
        T1 = repeated_rotation(Rt)
        return dof, T1.T @ kl @ T1


class ConnectorMembers:
    """The connectors (`connector2`): `elements.connector.response`, member
    by member, or the compiled one under `backend: numba`
    (kernels/numba_connector)."""

    def __init__(self, s):
        self.ids = np.flatnonzero((s.member_kind == "connector2") & _active(s))
        self._kernel = None
        #: a connector of these has a law the kernel has no code for
        self._declined = False

    def kernel(self, s):
        """The compiled connectors' arrays under `backend: numba`, or None:
        no connectors, another backend, no numba."""
        ids = self.ids
        if not len(ids) or getattr(s, "backend", "numpy") != "numba" or s.dofs.stride != 6:
            return None
        if self._kernel is not None:
            return self._kernel
        if self._declined:
            return None
        from ..elements import connector as _c
        if any(_c.has_material_law(s.member_props[e]) for e in ids):
            self._declined = True              # a `material` law (1.21) has no compiled form
            return None
        try:
            from ..kernels import numba_connector as nc
        except ImportError:                                   # pragma: no cover
            return None
        if not nc.available():
            return None
        self._kernel = {
            "for": ids, "ids": np.ascontiguousarray(ids, dtype=np.int64),
            "kind": np.arange(len(ids), dtype=np.int64),
            "enc": nc.encode([s.member_props[e] for e in ids]), "spins": nc.spins(),
            "no_rows": np.zeros((1, 12)), "no_birth": nc.NO_BIRTH}
        return self._kernel

    def forces(self, s, out, actions, end_actions, deformation):
        from ..elements import connector
        ck = self.kernel(s)
        if ck is not None:
            # compiled: the same mechanics to round-off, in
            # element order, the meeting of a connector's ends refused as below
            from ..kernels.numba_connector import respond
            bad = respond(ck["ids"], ck["kind"],
                          np.ascontiguousarray(np.asarray(s.mesh.elements)[:, :2], dtype=np.int64),
                          s.mesh.nodes, s.dofs.view(s.q), s.orient, s.v,
                          np.ascontiguousarray(np.asarray(s.R0, dtype=float)), ck["spins"], *ck["enc"],
                          s._conn_committed, s._conn_slip, s.mode == "dynamic",
                          out, actions, end_actions, deformation, 6, ck["no_rows"], False, True,
                          ck["no_birth"] if s._conn_birth is None else s._conn_birth)
            if bad >= 0:
                raise ValueError(COLLAPSED)
            return
        for e in self.ids:
            conn = s.mesh.elements[e]
            dof = s.dofs.element_dofs(conn)
            qe = s.q[dof].reshape(2, 6); x = s.mesh.nodes[conn] + qe[:, :3]
            axis = x[1] - x[0]; L = np.linalg.norm(axis)
            if L < 1e-14:
                raise ValueError(COLLAPSED)
            fe, act, deform = connector.response(
                qe[:, :3], s.orient[conn], s.v[dof], s.mesh.nodes[conn],
                s.R0[e], s.member_props[e], include_damping=s.mode == "dynamic",
                slip_committed=s._conn_committed[e], slip_out=s._conn_slip[e],
                offset=None if s._conn_birth is None else s._conn_birth[e],
                materials=s.connector_materials.of(e) if s.connector_materials else None)
            out[dof] += fe; actions[e] = np.abs(act)
            end_actions[e, :6] = -act; end_actions[e, 6:] = act
            deformation[e] = deform

