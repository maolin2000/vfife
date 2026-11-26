"""The constraint layer (1.21): the multi-point constraints other than the
supports, each kind written once and every procedure taking it through one
interface.

WHY. Until 1.21 the rigid diaphragms were written three times -- the
acceleration projection of the explicit step and the relaxation
(`BeamSolver._project_diaphragms`), the transformation of the Newton system
(`implicit.Newton._diaphragm_map`) and the one of the modal and buckling
analyses (`modal._diaphragm_map`) -- each with its own check that a floor is
not also held by a support, and the rigid links were about to be written a
fourth time. They also hung on the frame solver, which the fibre-beam solver
is a sibling of, not a child: what a fibre model could not do was a matter of
where the code sat, not of what it computes. See the design note of 1.21
(the constraint layer).

WHAT. A solver of the beam family has one `ConstraintLayer` (`ties`, built by
`BeamFamily.constraint_layer()`). Each kind of constraint is one class here:

    Diaphragms    the rigid floors: linear, in the reference geometry
    RigidLinks    master-slave links in large rotation (`rigid_links`)

and the layer gives each procedure what it takes:

    project(acc, fixed)        the explicit step and the relaxation
    correct(fixed)             after each position update (`_apply_bc`)
    residual_norm(r, free, n)  the relaxation's convergence measure
    map / free_map             the transformation q = T q_m of the Newton
                               system; the modal and buckling ones
    check_free(free)           a slave dof held in a step: refused
    checkpoint / restore       a restart

The supports are not here: they are one mechanism already (`step_data`,
`_apply_bc`) and every procedure reads their masks.

THE ARITHMETIC of the floors: the projection's 3x3 solve about the floor's
centre of mass, and the one map (`map`) of the Newton, modal and buckling
systems with its three masters at the same point (step 1 of the layer moved
the code and changed no result in a bit; step 2 put the Newton map's masters
at the centre of mass too -- they were at the nodes' mean point, the same
constraint: round-off).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..base.rotations import exp_map, log_map


# ------------------------------------------------------------------ floors
class Diaphragms:
    """Rigid floors: each floor's nodes move as a rigid body in the plane
    normal to `normal` -- the two in-plane translations and the rotation about
    the normal -- and stay free out of it. `floors` is the list of dicts
    `BeamFamily._tie_floors` reads from the case (idx, n, a, b, x, name)."""

    def __init__(self, solver, floors):
        self.s = solver
        self.floors = floors

    def project(self, acc):
        """Replace the floor's nodal accelerations by the rigid-floor motion
        their own resultant produces."""
        A = acc.reshape(-1, 6)
        M = self.s.mass.reshape(-1, 6)
        for d in self.floors:
            idx, n, a, b = d["idx"], d["n"], d["a"], d["b"]
            ma, mb, mn = M[idx, a], M[idx, b], M[idx, 3 + n]
            # the centre is the floor's own centre of mass, so that a pure
            # resultant through it produces no spin
            total = ma.sum() + mb.sum()
            centre = ((ma + mb)[:, None] * d["x"][:, (a, b)]).sum(axis=0) / max(total, 1e-300)
            ra = d["x"][:, a] - centre[0]
            rb = d["x"][:, b] - centre[1]
            fa, fb = ma * A[idx, a], mb * A[idx, b]
            torque = (mn * A[idx, 3 + n]).sum() + (ra * fb - rb * fa).sum()
            # the 3x3 rigid-floor mass matrix; the off-diagonals vanish at the
            # centre of mass only when the two in-plane masses agree, which is
            # the usual case and not one worth assuming
            Ma, Mb = ma.sum(), mb.sum()
            Sab, Sba = (ma * rb).sum(), (mb * ra).sum()
            J = (mn.sum() + (ma * rb * rb).sum() + (mb * ra * ra).sum())
            K = np.array([[Ma, 0.0, -Sab],
                          [0.0, Mb, Sba],
                          [-Sab, Sba, J]])
            rhs = np.array([fa.sum(), fb.sum(), torque])
            try:
                Aa, Ab, alpha = np.linalg.solve(K, rhs)
            except np.linalg.LinAlgError:
                raise ValueError(f"diaphragm {d['name']!r} has no mass to accelerate")
            A[idx, a] = Aa - alpha * rb
            A[idx, b] = Ab + alpha * ra
            A[idx, 3 + n] = alpha
        return acc

    def condense(self, rr):
        """The floors' own equations out of a residual `rr` that is zero off
        the free dofs: each floor's in-plane resultants and its torque about
        the normal (returned), its nodes' in-plane components zeroed in place.
        At a floor's nodes the nodal residual holds the constraint forces
        that keep the floor rigid, which never vanish (1.9)."""
        R = rr.reshape(-1, 6)
        M = self.s.mass.reshape(-1, 6)
        extra = []
        for d in self.floors:
            idx, n, a, b = d["idx"], d["n"], d["a"], d["b"]
            ma, mb = M[idx, a], M[idx, b]
            total = ma.sum() + mb.sum()
            centre = ((ma + mb)[:, None] * d["x"][:, (a, b)]).sum(axis=0) / max(total, 1e-300)
            ra = d["x"][:, a] - centre[0]
            rb = d["x"][:, b] - centre[1]
            fa, fb = R[idx, a], R[idx, b]
            extra += [fa.sum(), fb.sum(), R[idx, 3 + n].sum() + (ra * fb - rb * fa).sum()]
            R[idx, a] = 0.0; R[idx, b] = 0.0; R[idx, 3 + n] = 0.0
        return extra

    def entries(self, col, start, mass):
        """The floors' rows of the transformation q = T q_m (see
        `ConstraintLayer.map`): three masters per floor from column `start`
        on -- U_a, U_b and alpha about the floor's centre of mass (`mass` per
        dof; the nodes' mean point when they carry none), u_a = U_a - alpha
        r_b, u_b = U_b + alpha r_a, theta_n = alpha in the reference geometry.
        The same kinematics `project` imposes on the accelerations: one
        parametrisation for every procedure since 1.21 (the Newton system's
        was about the mean point, the same constraint to round-off)."""
        rows, cols, vals = [], [], []
        M = np.asarray(mass, float).reshape(-1, 6)
        for k, d in enumerate(self.floors):
            idx, a, b, nrm = d["idx"], d["a"], d["b"], d["n"]
            x = np.asarray(d["x"], float)
            ma, mb = M[idx, a], M[idx, b]
            total = float((ma + mb).sum())
            if total > 0.0:
                centre = ((ma + mb)[:, None] * x[:, (a, b)]).sum(axis=0) / total
            else:
                centre = x[:, (a, b)].mean(axis=0)
            ra = x[:, a] - centre[0]; rb = x[:, b] - centre[1]
            m0 = start + 3 * k
            n = len(idx)
            rows += [6 * idx + a, 6 * idx + b, 6 * idx + a, 6 * idx + b, 6 * idx + 3 + nrm]
            cols += [np.full(n, m0), np.full(n, m0 + 1), np.full(n, m0 + 2), np.full(n, m0 + 2), np.full(n, m0 + 2)]
            vals += [np.ones(n), np.ones(n), -rb, ra, np.ones(n)]
        return rows, cols, vals

    def slave_dofs(self):
        return np.concatenate([6 * d["idx"] + c for d in self.floors for c in (d["a"], d["b"], 3 + d["n"])])

    def check_free(self, free):
        """Refuse a floor whose in-plane dof is held or driven in this step
        (`free` the step's free mask over all dofs): the always-held ones are
        refused where the floors are read (`BeamFamily._tie_floors`)."""
        F = np.asarray(free, bool).reshape(-1, 6)
        for d in self.floors:
            if not F[d["idx"]][:, (d["a"], d["b"], 3 + d["n"])].all():
                raise ValueError(_held_floor(d))


def _held_floor(d):
    return (f"diaphragm {d['name']!r}: an in-plane dof of the floor is held or driven in this step; a floor "
            "is either held by a support or moved by a diaphragm, not both")


# ------------------------------------------------------------- rigid links
#
# MASTER-SLAVE RIGID LINKS (1.21): `rigid_links: [{master, slaves, dofs, arm, name}]`.
#
# Each slave's tied components -- `dofs`, six flags in the global axes ux uy
# uz rx ry rz, as Midas' *RIGIDLINK and OpenSees' equalDOF have them --
# follow the master as a rigid body does, in large rotation:
#
#     translation k   (x_s)_k = (x_m + R_m R_m0^T r0)_k     (arm: true)
#                     (x_s)_k = (x_m + r0)_k                (arm: false, equalDOF)
#     rotation        R_s = R_m P0 when all three are tied; else the
#                     components k of phi = log(R_s (R_m P0)^T) are zero
#                     (a partial tie of the rotations is exact while the
#                     relative rotation stays small: the spins are tied
#                     component by component, omega_s,k = omega_m,k, which
#                     is the derivative of phi_k = 0 at phi = 0)
#
# with r0 = x_s - x_m, R_m0 and P0 = R_m0^T R_s0 taken where the link is
# made: it is born free of force where its nodes stand. Rotations are the
# points' spatial orientations and the spins the solver integrates are
# spatial, so a tied rotation component is simply omega_s,k = omega_m,k.
#
# HOW IT IS IMPOSED, in the explicit step and the dynamic relaxation: on the
# ACCELERATION, as the rigid diaphragms are (`Diaphragms.project` above).
# With T_s the map of the master's six velocities to the slave's tied
# components -- v_s,k = v_m,k + (omega_m x r)_k, omega_s,k = omega_m,k -- and
# b_s the centripetal part of the slave's tied acceleration, omega_m x
# (omega_m x r), the master accelerates under its own force and the tied
# forces of its slaves, through its own mass and theirs:
#
#     (M_m + sum T_s^T M_s T_s) a_m = f_m + sum T_s^T (f_s - M_s b_s)
#
# and each slave's tied acceleration is T_s a_m + b_s. That is the exact
# projection of the unconstrained accelerations onto the constraint (the
# Lagrange multiplier eliminated), for a master whose slaves are tied to it
# alone. After each position update the tied components are set exactly
# from the master (`correct`), which removes the round-off the integration
# leaves and makes a master held or driven by a support carry its slaves
# with it.
#
# What is not taken here (refused where it would be met): a slave that is a
# master or another link's slave, a slave in a rigid diaphragm, a tied
# component held by a support, links made in an analysis step, and the
# compiled step and relaxation (they hand the step to the loop).

def _skew(r):
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


@dataclass
class Link:
    name: str
    master: int
    slaves: np.ndarray          # (ns,) node indices
    dofs: np.ndarray            # (6,) bool
    arm: bool
    r0: np.ndarray              # (ns, 3) the arm where it was made
    Rm0: np.ndarray             # (3, 3) the master's orientation then
    P0: np.ndarray              # (ns, 3, 3) R_m0^T R_s0
    #: the analysis step it is made in (1.21; -1: the model's, made at the start)
    born: int = -1
    #: whether it ties its slaves now (a step's is made at its step's start,
    #: and a released one ties nothing from its step on)
    active: bool = True
    #: the force it put on each slave at the last output frame, (ns, 6) on the
    #: tied components: what a release takes off over its step
    force: np.ndarray = None


class RigidLinks:
    def __init__(self, solver, specs, region, held, floors):
        """`region(name)` gives a region's node indices, `held` the dofs any
        support of the model or of a step names (nn x 6, held or driven),
        `floors` the rigid diaphragms' node sets."""
        self.s = solver
        #: the dofs held or driven in the step running, as `_apply_bc` last
        #: had them: a master's held dofs do not accelerate (`project`)
        self.fixed = None
        #: whether a step's `release` names one of these links (set by the
        #: solver when it reads the steps)
        self.released_any = False
        if solver.dofs.stride != 6:
            raise ValueError("rigid_links need six dofs a node (a frame model)")
        self.links: List[Link] = []
        x = self._positions()
        slave_of = {}
        masters = set()
        names = set()
        seen = {}
        for i, spec in enumerate(specs):
            spec = dict(spec)
            born = int(spec.get("born", -1))
            j = seen[born] = seen.get(born, -1) + 1          # its place in its own list
            name = spec.get("name") or (f"rigid_links[{j}]" if born < 0 else f"steps[{born}].rigid_links[{j}]")
            if spec.get("name"):
                if name in names:
                    raise ValueError(f"two rigid links are named {name!r}: a name is what a step's `release` "
                                     "frees, one link each")
                names.add(name)
            m = np.asarray(region(spec["master"]), dtype=np.int64)
            if len(m) != 1:
                raise ValueError(f"rigid link {name!r}: `master` {spec['master']!r} is {len(m)} nodes; a master is one")
            m = int(m[0])
            sl = np.asarray(region(spec["slaves"]), dtype=np.int64)
            if not len(sl):
                raise ValueError(f"rigid link {name!r}: `slaves` {spec['slaves']!r} holds no node")
            if m in set(sl.tolist()):
                raise ValueError(f"rigid link {name!r}: the master node {m} is among its own slaves")
            dofs = np.asarray(spec.get("dofs", [1] * 6), dtype=bool)
            if dofs.shape != (6,) or not dofs.any():
                raise ValueError(f"rigid link {name!r}: `dofs` is six flags with at least one set")
            for n in sl.tolist():
                if n in slave_of:
                    raise ValueError(f"rigid link {name!r}: node {n} is already a slave of {slave_of[n]!r}; "
                                     "a node follows one master")
                slave_of[n] = name
            masters.add(m)
            arm = bool(spec.get("arm", True))
            orient = solver.orient
            Rm0 = orient[m].copy()
            P0 = np.einsum("ji,njk->nik", Rm0, orient[sl])
            self.links.append(Link(name, m, sl, dofs, arm, x[sl] - x[m], Rm0, P0, born=born, active=born < 0,
                                   force=np.zeros((len(sl), 6))))
        chain = masters & set(slave_of)
        if chain:
            n = sorted(chain)[0]
            raise ValueError(f"node {n} is the master of one rigid link and a slave of {slave_of[n]!r}: "
                             "links in a chain are not taken (tie each slave to the chain's first master)")
        on_floor = {int(i) for idx in floors for i in idx}
        both = on_floor & set(slave_of)
        if both:
            raise ValueError(f"node {sorted(both)[0]} is a slave of rigid link {slave_of[sorted(both)[0]]!r} "
                             "and in a rigid diaphragm: a node is moved by one of them")
        both = on_floor & masters
        if both:
            raise ValueError(f"node {sorted(both)[0]} is the master of a rigid link and in a rigid diaphragm: "
                             "not taken in this version (the floor's projection and the link's are not combined)")
        held = np.asarray(held).reshape(-1, 6)
        # every link, a step's too (found in review: a step's link on a held
        # slave was not refused and the relaxation did not settle), against
        # every support of the model and of the steps
        for L in self.links:
            bad = held[L.slaves][:, L.dofs]
            if bad.any():
                n = int(L.slaves[np.flatnonzero(bad.any(axis=1))[0]])
                raise ValueError(f"rigid link {L.name!r}: a tied dof of slave node {n} is also held by a "
                                 "support (of the model or of a step, held or driven); a dof is held by a "
                                 "support or tied to a master, not both")

    # ------------------------------------------------------------- state
    def _positions(self):
        s = self.s
        return s.mesh.nodes + s.dofs.view(s.q)[:, :3]

    def _arm(self, L):
        """The arms as they are now: (ns, 3)."""
        if not L.arm:
            return L.r0
        return (self.s.orient[L.master] @ L.Rm0.T @ L.r0.T).T

    def _maps(self, L, r):
        """T (ns, 6, 6) of each slave: the slave's velocities from the
        master's, on the tied rows (others zero)."""
        ns = len(L.slaves)
        T = np.zeros((ns, 6, 6))
        for j in range(ns):
            T[j, :3, :3] = np.eye(3)
            if L.arm:
                T[j, :3, 3:] = -_skew(r[j])          # omega x r = -[r]x omega
            T[j, 3:, 3:] = np.eye(3)
        T[:, ~L.dofs, :] = 0.0
        return T

    # ------------------------------------------------------- the projection
    def project(self, acc, fixed=None):
        """The accelerations with the links imposed (see the module note).
        A master's held or driven dofs (`fixed`, or the step's as `_apply_bc`
        last had them) do not accelerate here: the master's free dofs are
        solved for alone, its held ones at zero -- a held dof's reaction
        must not leak into the free ones through the arm's coupling (found
        in review). The supports then move it, and `correct` its slaves."""
        s = self.s
        fixed = self.fixed if fixed is None else fixed
        FX = None if fixed is None else np.asarray(fixed, bool).reshape(-1, 6)
        A = acc.reshape(-1, 6)
        M = s.mass.reshape(-1, 6)
        V = s.v.reshape(-1, 6)
        for L in self.live():
            r = self._arm(L)
            T = self._maps(L, r)
            m = L.master
            w = V[m, 3:]
            Mm = np.diag(M[m]).astype(float)
            f = M[m] * A[m]
            B = np.zeros((len(L.slaves), 6))
            for j, n in enumerate(L.slaves):
                if L.arm:
                    B[j, :3] = np.cross(w, np.cross(w, r[j]))
                B[j, ~L.dofs] = 0.0
                Ms = M[n] * L.dofs
                Mm += T[j].T @ (Ms[:, None] * T[j])
                f += T[j].T @ (M[n] * A[n] * L.dofs - Ms * B[j])
            free = np.ones(6, bool) if FX is None else ~FX[m]
            am = np.zeros(6)
            if free.any():
                try:
                    am[free] = np.linalg.solve(Mm[np.ix_(free, free)], f[free])
                except np.linalg.LinAlgError:
                    raise ValueError(f"rigid link {L.name!r}: its master and slaves have no mass to accelerate")
            A[m] = am
            for j, n in enumerate(L.slaves):
                A[n] = np.where(L.dofs, T[j] @ am + B[j], A[n])
        return acc

    def correct(self, fixed=None):
        """Set the slaves' tied components exactly from the master: position,
        velocity, orientation and spin (after each position update)."""
        s = self.s
        if fixed is not None:
            self.fixed = fixed
        qv = s.dofs.view(s.q); vv = s.dofs.view(s.v)
        x = self._positions()
        for L in self.live():
            r = self._arm(L)
            m = L.master
            wm = vv[m, 3:]
            tt = L.dofs[:3]; tr = L.dofs[3:]
            for j, n in enumerate(L.slaves):
                if tt.any():
                    want = x[m] + r[j]
                    qv[n, :3] = np.where(tt, want - s.mesh.nodes[n], qv[n, :3])
                    vel = vv[m, :3] + (np.cross(wm, r[j]) if L.arm else 0.0)
                    vv[n, :3] = np.where(tt, vel, vv[n, :3])
                if tr.any():
                    target = s.orient[m] @ L.P0[j]
                    if tr.all():
                        s.orient[n] = target
                    else:
                        phi = log_map((s.orient[n] @ target.T)[None])[0]
                        phi[tr] = 0.0
                        s.orient[n] = exp_map(phi[None])[0] @ target
                    qv[n, 3:] = log_map(s.orient[n][None])[0]
                    vv[n, 3:] = np.where(tr, wm, vv[n, 3:])

    def condense(self, r):
        """A residual with each slave's tied components carried to its master
        (as a force and the moment of its arm) and zeroed: what has to vanish
        at a link is the master's equations, not the constraint forces."""
        R = r.reshape(-1, 6)
        for L in self.live():
            arm = self._arm(L)
            T = self._maps(L, arm)
            for j, n in enumerate(L.slaves):
                R[L.master] += T[j].T @ np.where(L.dofs, R[n], 0.0)
                R[n] = np.where(L.dofs, 0.0, R[n])
        return r

    # ------------------------------------------------- the linear map
    def slave_dofs(self):
        rows = [(6 * L.slaves[:, None] + np.flatnonzero(L.dofs)[None, :]).ravel() for L in self.live()]
        return np.concatenate(rows) if rows else np.zeros(0, np.int64)

    def entries(self, col):
        """The slaves' tied rows of the transformation q = T q_m at the
        current configuration (`ConstraintLayer.map`): v_s,k = v_m,k +
        (omega_m x r)_k for a translation, omega_s,k = omega_m,k for a
        rotation; `col` gives the column of each of the master's dofs. The
        arm is the one the master has now: the map is rebuilt as it turns
        (the constraint's curvature is not in the static tangent, which costs
        Newton its quadratic rate where the links carry large forces, not its
        answer)."""
        rows, cols, vals = [], [], []
        for L in self.live():
            r = self._arm(L)
            T = self._maps(L, r)
            m = L.master
            for j, n in enumerate(L.slaves):
                for k in np.flatnonzero(L.dofs):
                    for c in np.flatnonzero(T[j, k] != 0.0):
                        rows.append(6 * int(n) + int(k)); cols.append(int(col[6 * m + c])); vals.append(float(T[j, k, c]))
        return [np.asarray(rows, np.int64)], [np.asarray(cols, np.int64)], [np.asarray(vals, float)]

    # --------------------------------------------------------- checkpoint
    def checkpoint(self, out):
        out["rigid_link_r0"] = np.concatenate([L.r0 for L in self.links])
        out["rigid_link_Rm0"] = np.stack([L.Rm0 for L in self.links])
        out["rigid_link_P0"] = np.concatenate([L.P0 for L in self.links])
        # which tie now, and the force each put on its slaves at the last
        # frame (a release in the next step takes it off; 1.21)
        out["rigid_link_active"] = np.array([L.active for L in self.links], np.int8)
        out["rigid_link_force"] = np.concatenate([L.force for L in self.links])
        return out

    def restore(self, ck):
        if "rigid_link_r0" not in ck:
            # a state written without links (an `initial_state` relaxed
            # before the links were added): the links are this model's own,
            # made where the run starts, and are kept
            return
        r0 = np.asarray(ck["rigid_link_r0"], float); P0 = np.asarray(ck["rigid_link_P0"], float)
        Rm0 = np.asarray(ck["rigid_link_Rm0"], float)
        if len(r0) != sum(len(L.slaves) for L in self.links) or len(Rm0) != len(self.links):
            raise ValueError("the checkpoint's rigid links are not this model's")
        active = np.asarray(ck["rigid_link_active"]).astype(bool) if "rigid_link_active" in ck else None
        force = np.asarray(ck["rigid_link_force"], float) if "rigid_link_force" in ck else None
        at = 0
        for i, L in enumerate(self.links):
            n = len(L.slaves)
            L.r0 = r0[at:at + n].copy(); L.P0 = P0[at:at + n].copy(); L.Rm0 = Rm0[i].copy()
            if active is not None:
                L.active = bool(active[i])
            if force is not None:
                L.force = force[at:at + n].copy()
            at += n

    # ------------------------------------------------- analysis steps (1.21)
    def live(self):
        return [L for L in self.links if L.active]

    def staged(self):
        """Whether any link is made or released in an analysis step."""
        return any(L.born >= 0 for L in self.links) or self.released_any

    def names(self):
        return {L.name for L in self.links}

    def record(self, residual):
        """The force each live link puts on its slaves, from an output
        frame's f_int - F (at a slave's tied components the structure's own
        force less the load is what the link supplies)."""
        R = np.asarray(residual, float).reshape(-1, 6)
        for L in self.live():
            L.force = R[L.slaves] * L.dofs

    def birth(self, k):
        """The links of step k made where their nodes stand now: born free of
        force, as a member born in a step is (the arm and the relative
        orientation taken here)."""
        x = self._positions()
        orient = self.s.orient
        for L in self.links:
            if L.born == k:
                L.r0 = x[L.slaves] - x[L.master]
                L.Rm0 = orient[L.master].copy()
                L.P0 = np.einsum("ji,njk->nik", L.Rm0, orient[L.slaves])
                L.force = np.zeros((len(L.slaves), 6))
                L.active = True

    def release(self, names):
        """The links `names` freed: the force each put on its slaves at the
        end of the step before, and the opposite on its master through the
        arm (-T^T f), as a load the step takes off over its course -- as a
        released support's reaction is (`analysis_steps._release_force`).
        Returns that load (nodal, over all dofs)."""
        F = np.zeros(self.s.ndof)
        Fv = F.reshape(-1, 6)
        for L in self.links:
            if L.name not in names:
                continue
            r = self._arm(L)
            T = self._maps(L, r)
            for j, n in enumerate(L.slaves):
                f = L.force[j] * L.dofs
                Fv[n] += f
                Fv[L.master] -= T[j].T @ f
            L.active = False
            L.force = np.zeros((len(L.slaves), 6))
        return F




# -------------------------------------------------------------- the layer
class CarriedMass:
    """The mass of members set off their nodes, where it is (1.21, step 3 of
    the member offsets' design): a member's share at its node n is at its
    elastic end, x_n + Q_n e -- a point carried rigidly by the node, with no
    dof of its own. That is a rigid link whose slave has no dofs (the
    projection of `RigidLinks.project` with T = [I, -[r]x]), summed per node
    into two numbers of the reference: the first moment s0 = sum m e and the
    second P0 = sum m (|e|^2 I - e e^T). Now S = Q s0, J = Q P0 Q^T, and the
    node's six accelerations solve

        [[M I, -[S]x], [[S]x, J_n + J]] [a; alpha] =
            [f - w x (w x S); m - w x (J w)]

    -- the parallel-axis inertia, the coupling of translation and rotation,
    the centripetal terms (Abaqus' exact rotary inertia of an offset beam;
    the node's own diagonal M already holds the members' translational
    mass). Only the explicit dynamic step and the modes see it: a static
    solution and a relaxation do not depend on the mass. The weight and a
    ground motion's inertia act at the same points (`body_moments`), turned
    with the node: S x b, b the field's acceleration -- a mass hung beside
    its node swings as a pendulum, and a girder drawn on its top flange feels
    the height of its weight as it twists.

    A rigid zone's own mass (`BeamFamily.member_zone_mass`) is carried at
    the zone's middle, half the zone back from the elastic end along the
    elastic axis (a point mass there: the zone's own rotary inertia is left
    out).

    The members counted are those whose mass is on their nodes (built: a
    broken member keeps its mass), on the connectivity now (a split moves a
    member's end, and its mass, to a spare particle): the tables follow
    both. A node a rigid floor or a rigid link of the model ties (`skip`) is
    left to that constraint, everywhere -- the step, the modes, the energy
    and the step's bound (their projection takes the diagonal)."""

    def __init__(self, solver, skip=None):
        self.s = solver
        self.skip = None if skip is None or not len(skip) else np.unique(np.asarray(skip, np.int64))
        self._key = None
        self.nodes = np.zeros(0, np.int64)
        #: the body fields of the step: (amplitude, acceleration (3,), nodes)
        #: of each gravity load (`BeamFamily.step_data`)
        self.fields = []

    def _members(self):
        s = self.s
        keep = np.zeros(s.nel, bool); keep[s.geometry.members] = True
        built = getattr(s, "member_built", None) if getattr(s, "plan", None) is not None else None
        if built is not None:
            keep &= np.asarray(built, bool)
        return np.flatnonzero(keep)

    def tables(self):
        """(nodes, s0 (n, 3), P0 (n, 3, 3)) for the nodes that carry an
        offset mass, rebuilt when the members or their nodes change."""
        s = self.s
        ids = self._members()
        conn = np.asarray(s.mesh.elements)[:, :2]
        key = (ids.tobytes(), conn[ids].tobytes())
        if key != self._key:
            off = s.geometry.offset
            s0 = {}; P0 = {}
            for e in ids:
                m = float(s.member_mass_share(int(e))[0])
                if m <= 0.0:
                    continue
                zone = s.member_zone_mass(int(e))
                zl = s.member_zones(int(e))
                ex = np.asarray(s.R0[e])[0]
                for k in range(2):
                    ev = off[e, k]
                    if not np.any(ev):
                        continue
                    n = int(conn[e, k])
                    # the elastic part's share at the elastic end, the rigid
                    # zone's at its middle, half its length back from it
                    back = -1.0 if k == 0 else 1.0
                    for mk, c in ((m, ev), (float(zone[k]), ev + back * 0.5 * zl[k] * ex)):
                        if mk <= 0.0 or not np.any(c):
                            continue
                        s0[n] = s0.get(n, 0.0) + mk * c
                        P0[n] = P0.get(n, 0.0) + mk * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
            nodes = np.asarray(sorted(s0), np.int64)
            if self.skip is not None:
                nodes = nodes[~np.isin(nodes, self.skip)]
            self.nodes = nodes
            self.s0 = np.asarray([s0[n] for n in nodes], float).reshape(-1, 3)
            self.P0 = np.asarray([P0[n] for n in nodes], float).reshape(-1, 3, 3)
            self._key = key
        return self.nodes, self.s0, self.P0

    def blocks(self, orient=None):
        """(nodes, S (n, 3), J (n, 3, 3)) now -- at the triads `orient` (the
        solver's when None)."""
        nodes, s0, P0 = self.tables()
        Q = (self.s.orient if orient is None else orient)[nodes]
        return nodes, np.einsum("nij,nj->ni", Q, s0), np.einsum("nij,njk,nlk->nil", Q, P0, Q)

    def project(self, acc, fixed=None):
        """The accelerations of the carrying nodes from their six equations
        (the class note); `fixed` dofs do not accelerate."""
        s = self.s
        if getattr(s, "mode", None) != "dynamic":
            return acc
        nodes, S, J = self.blocks()
        if not len(nodes):
            return acc
        A = acc.reshape(-1, 6)
        M = s.mass.reshape(-1, 6)[nodes]
        w = s.v.reshape(-1, 6)[nodes, 3:]
        n = len(nodes)
        M6 = np.zeros((n, 6, 6))
        M6[:, np.arange(6), np.arange(6)] = M
        sk = _skews(S)
        M6[:, :3, 3:] -= sk
        M6[:, 3:, :3] += sk
        M6[:, 3:, 3:] += J
        f = M * A[nodes]
        f[:, :3] -= np.cross(w, np.cross(w, S))
        f[:, 3:] -= np.cross(w, np.einsum("nij,nj->ni", J, w))
        if fixed is not None:
            FX = np.asarray(fixed, bool).reshape(-1, 6)[nodes]
            if FX.any():
                # a held dof: its row and column out, zero acceleration
                M6 = np.where(FX[:, :, None] | FX[:, None, :], 0.0, M6)
                M6[:, np.arange(6), np.arange(6)] += FX
                f = np.where(FX, 0.0, f)
        A[nodes] = np.linalg.solve(M6, f[..., None])[..., 0]
        return acc

    def coupling(self, orient=None):
        """The mass matrix's part beyond the node's diagonal, as sparse COO
        entries (rows, cols, vals) on the solver's dofs: the modes'."""
        nodes, S, J = self.blocks(orient)
        rows, cols, vals = [], [], []
        sk = _skews(S)
        for i, n in enumerate(nodes):
            C = np.zeros((6, 6))
            C[:3, 3:] = -sk[i]; C[3:, :3] = sk[i]; C[3:, 3:] = J[i]
            r, c = np.nonzero(C)
            rows.append(6 * n + r); cols.append(6 * n + c); vals.append(C[r, c])
        if not rows:
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
        return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

    def kinetic(self):
        """The kinetic energy the diagonal does not count: v . (w x S) +
        1/2 w^T J w at each carrying node."""
        nodes, S, J = self.blocks()
        if not len(nodes):
            return 0.0
        V = self.s.v.reshape(-1, 6)[nodes]
        v, w = V[:, :3], V[:, 3:]
        return float(np.sum(np.einsum("ni,ni->n", v, np.cross(w, S)))
                     + 0.5 * np.sum(np.einsum("ni,nij,nj->n", w, J, w)))

    def body_moments(self, out, progress, ground_a=None):
        """Add to `out` (ndof) the moment about each carrying node of its
        carried masses under the step's body fields at `progress` -- each
        gravity load's acceleration times its amplitude on its nodes, and a
        ground motion's -a_g (`ground_a`, (3,), on every node): S x b with S
        = Q s0 now. Each force evaluation asks for it."""
        if not self.fields and ground_a is None:
            return out
        nodes, S, _J = self.blocks()
        if not len(nodes):
            return out
        b = np.zeros((len(nodes), 3))
        for a, g, inside in self.fields:
            b += np.where(inside[nodes][:, None], float(a(progress)) * g, 0.0)
        if ground_a is not None:
            b += np.asarray(ground_a, float)
        out.reshape(-1, 6)[nodes, 3:6] += np.cross(S, b)
        return out

    def step_ratio(self, mass, row, free):
        """(ndof,) mass over stiffness of the carrying nodes' dofs for the
        explicit step's bound, from the full mass: Gershgorin on M6^-1 |K|
        -- omega^2 <= sum_k |M6^-1|_ik row_k, row the rows of |K| -- in place
        of the diagonal's m_i / row_i; inf elsewhere and on held dofs."""
        out = np.full(len(mass), np.inf)
        nodes, s0, P0 = self.tables()
        if not len(nodes):
            return out
        M = np.asarray(mass, float).reshape(-1, 6); RW = np.asarray(row, float).reshape(-1, 6)
        F = np.asarray(free, bool).reshape(-1, 6); O = out.reshape(-1, 6)
        sk = _skews(s0)
        for i, n in enumerate(nodes):
            f = F[n]
            if not f.any():
                continue
            M6 = np.diag(M[n]).astype(float)
            M6[:3, 3:] -= sk[i]; M6[3:, :3] += sk[i]; M6[3:, 3:] += P0[i]
            inv = np.linalg.inv(M6[np.ix_(f, f)])
            O[n, np.flatnonzero(f)] = 1.0 / np.maximum(np.abs(inv) @ RW[n, f], 1e-300)
        return out


def _skews(r):
    """[r]x of each row of r (n, 3): (n, 3, 3)."""
    out = np.zeros((len(r), 3, 3))
    out[:, 0, 1], out[:, 0, 2] = -r[:, 2], r[:, 1]
    out[:, 1, 0], out[:, 1, 2] = r[:, 2], -r[:, 0]
    out[:, 2, 0], out[:, 2, 1] = -r[:, 1], r[:, 0]
    return out


class ConstraintLayer:
    """The multi-point constraints of one solver (see the module note).
    `floors` as `BeamFamily._tie_floors` reads them, `links` the case's
    `rigid_links` (dicts), `region` the solver's region lookup and `named`
    every dof a support of the model or of a step names (nn x 6), for the
    links' checks."""

    def __init__(self, solver, floors, links=(), region=None, named=None):
        self.floors = floors
        self.diaphragms = Diaphragms(solver, floors) if floors else None
        self.links = (RigidLinks(solver, list(links), region, named, [d["idx"] for d in floors])
                      if links else None)
        # the mass of members set off their nodes (no dofs of its own: not
        # in `empty`, `map` or the checks)
        geometry = getattr(solver, "geometry", None)
        self.carried = (CarriedMass(solver, self._tied()) if geometry is not None and geometry.any
                        else None)

    def empty(self):
        return self.diaphragms is None and self.links is None

    def project(self, acc, fixed=None):
        """The accelerations with the constraints imposed: the floors', then
        the links' (a link's master may not be on a floor)."""
        if self.carried is not None:
            acc = self.carried.project(acc, fixed)
        if self.diaphragms is not None:
            acc = self.diaphragms.project(acc)
        if self.links is not None:
            acc = self.links.project(acc, fixed)
        return acc

    def _tied(self):
        """The nodes a floor or a link ties (the carried mass leaves them)."""
        out = []
        if self.diaphragms is not None:
            out += [np.asarray(d["idx"], np.int64) for d in self.floors]
        if self.links is not None:
            # every link of the run, made now or in a later step
            for L in self.links.links:
                out += [np.asarray(L.slaves, np.int64), np.asarray([L.master], np.int64)]
        return np.concatenate(out) if out else None

    def correct(self, fixed=None):
        if self.links is not None:
            self.links.correct(fixed)

    def residual_norm(self, r, free, norm):
        """|r| over the free dofs, with each constraint's own equations in
        place of its constraint forces: a link's slave's tied residual is
        carried to its master, a floor's nodes give the floor's resultants."""
        if self.links is not None:
            r = self.links.condense(np.where(free, r, 0.0))
        if self.diaphragms is None:
            return float(norm(r[free]))
        rr = np.where(free, r, 0.0)
        extra = self.diaphragms.condense(rr)
        return float(np.sqrt(np.dot(rr, rr) + np.dot(extra, extra)))

    def map(self, n, mass):
        """The multi-point constraints as a transformation of the dofs, q = T
        q_m, at the current configuration: every dof that no constraint
        moves is its own master, a floor adds three masters, a link's
        slave's tied components follow its master's dofs. Returns (T, own,
        m, slave): T (n x m, CSC), the dofs that are their own masters (the
        first len(own) columns), the number of masters, the slave mask.
        The Newton system is T^T K T (`implicit.Newton`), the modal and
        buckling ones the same on the free dofs (`free_map`)."""
        import scipy.sparse as sp
        slave = np.zeros(n, bool)
        for tie in (self.diaphragms, self.links):
            if tie is not None:
                slave[tie.slave_dofs()] = True
        own = np.flatnonzero(~slave)
        col = np.full(n, -1, np.int64); col[own] = np.arange(len(own))
        rows = [own]; cols = [np.arange(len(own))]; vals = [np.ones(len(own))]
        m = len(own)
        if self.diaphragms is not None:
            r, c, v = self.diaphragms.entries(col, m, mass)
            rows += r; cols += c; vals += v
            m += 3 * len(self.floors)
        if self.links is not None:
            r, c, v = self.links.entries(col)
            rows += r; cols += c; vals += v
        T = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, m)).tocsc()
        return T, own, m, slave

    def free_map(self, p, free, mass):
        """`map` on the free dofs `free` (indices) of the probe `p` (whose
        dofs are the solver's), for the modal and buckling analyses: u_free =
        T u_reduced. A master's held dof is no column (it does not move); a
        slave's tied dof held in this step is refused."""
        T, own, m, slave = self.map(p.n, mass)
        is_free = np.zeros(p.n, bool); is_free[free] = True
        self.check_free(is_free)
        T = T.tocsr()[free]
        keep = np.concatenate([np.flatnonzero(is_free[own]), np.arange(len(own), m)])
        return T[:, keep].tocsr()

    def check_free(self, free):
        """Refuse a constraint whose slave dof is held or driven in this step
        (`free` the step's free mask over all dofs)."""
        if self.diaphragms is not None:
            self.diaphragms.check_free(free)
        if self.links is not None:
            bad = ~np.asarray(free, bool)[self.links.slave_dofs()]
            if bad.any():
                raise ValueError("a rigid link's slave's tied dof is held or driven in this step; a dof is held "
                                 "by a support or tied to a master, not both")

    def record(self, residual):
        """An output frame's f_int - F: the force each live link carries."""
        if self.links is not None:
            self.links.record(residual)

    def release(self, names):
        """At the start of an analysis step (`analysis_steps.begin_step`):
        the links `names` freed; their force returned, to be taken off over
        the step."""
        return self.links.release(set(names))

    def birth(self, k):
        """At the start of analysis step k, after its members are built: its
        own links made where their nodes stand."""
        if self.links is not None:
            self.links.birth(k)

    def checkpoint(self, out):
        return self.links.checkpoint(out) if self.links is not None else out

    def restore(self, ck):
        if self.links is not None:
            self.links.restore(ck)


def of(solver):
    """The solver's constraint layer, or None for a solver outside the beam
    family (the solids and the trusses have none)."""
    return solver.constraint_layer() if hasattr(solver, "constraint_layer") else None
