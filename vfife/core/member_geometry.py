"""The members' geometry (1.21): where each line member's elastic part is,
written once and every procedure taking it through one object.

WHY. A member's elastic part need not run from node to node (`end_offsets`,
core/end_offsets reads the input): a rigid end zone inside a column or a
pier cap, and a section set off its nodes -- a girder drawn on its top line
with its centroid below. Until 1.21 every procedure that needed a member's
ends probed the solver for the offsets itself (22 places), and the ones that
did not -- moving loads, vehicles, contact between members, buckling,
breaking and removal -- refused members that have them. Here is the one
place; see the design note of 1.21 (member offsets: geometry and mass, each
written once).

WHAT. A solver of the beam family has one `MemberGeometry` (`geometry`); a
model without offsets has one too, with `offset` None, and every method is
then the identity (the arrays returned are the ones given, not copies), so
such a model computes what it did, to the bit.

KINEMATICS. With the node at x and its triad Q (`solver.orient`, the
identity in the reference state), the elastic end is at x + Q e, e the
offset in the reference axes -- it turns with the node. The member's
co-rotational frame, length and local rotations are those of its elastic
ends (their triads are the nodes': a rigid stretch does not rotate
relative to its node). The reference frame `R0` and length come from the
elastic ends at the reference state.

FORCES. The elastic end's force f and moment m reach the node as f and
m + r x f, r = Q e the offset now: the transpose of the rigid kinematics,
so work is the same at the node and at the elastic end. In matrix form
B = [[I, -[r]x], [0, I]] per end, and the reference stiffness is
B^T T^T k T B.

LOADS ARE ON THE DRAWN LINE (1.21, the design's rule): a member's line
load is per unit length of the line between its NODES, and a moving load's
station is measured along that line. A load over a rigid zone goes to the
zone's node rigidly, force and the moment of its lever; a load beside the
elastic part goes to the elastic axis at the same place along it, with the
moment of the lever across (`drawn_loads`). The elastic part's share is
the consistent one -- the Hermite functions for a force, their slopes for a
moment -- so a load anywhere is work-equivalent, and a vehicle reads the
deck's motion under its wheel back through the same vector.

MASS is the elastic part's, at its ends: the nodes carry it there
(`constraints.CarriedMass`), and its weight acts there. The rigid stretch
across the member has none; a rigid zone weighs as its member, at its middle
(`BeamFamily.member_zone_mass`). End actions are reported at the elastic ends.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


class MemberGeometry:
    """Each member's two end offsets and what follows from them.

    `offset` (nel, 2, 3): the offsets as global vectors in the reference
    state (= in the nodes' own axes, the triads starting at the identity),
    or None when no member has one; `members`: the members that have one.
    `mesh` is the solver's (its nodes and connectivity, read live)."""

    def __init__(self, mesh=None, offset: Optional[np.ndarray] = None, full=None):
        self.mesh = mesh
        #: (nel,) bool, the members with `axial: full`, or None (`full_axial`)
        self.full = None if offset is None or full is None or not np.any(full) else np.asarray(full, bool)
        if offset is not None and not np.any(offset != 0.0):
            offset = None
        self.offset = offset
        self.members = (np.zeros(0, np.int64) if offset is None else
                        np.flatnonzero(np.any(offset != 0.0, axis=(1, 2))).astype(np.int64))

    @classmethod
    def build(cls, mesh, nel: int,
              groups: Iterable[Tuple[np.ndarray, dict, Optional[list], str]]) -> "MemberGeometry":
        """From the assignments that give `end_offsets`: (element ids, spec,
        orientation, name) each (core/end_offsets.offsets reads them)."""
        from .end_offsets import full_axial, offsets
        groups = list(groups)
        return cls(mesh, offsets(mesh.nodes, mesh.elements, nel, groups), full_axial(nel, groups))

    @property
    def any(self) -> bool:
        """Does any member have an offset?"""
        return self.offset is not None

    # ------------------------------------------------------------------ #
    #  positions
    # ------------------------------------------------------------------ #

    def reference_ends(self, e: int) -> Tuple[np.ndarray, np.ndarray]:
        """The two elastic ends of member `e` in the reference state."""
        n1, n2 = self.mesh.elements[e, 0], self.mesh.elements[e, 1]
        x1 = np.asarray(self.mesh.nodes[n1], float)
        x2 = np.asarray(self.mesh.nodes[n2], float)
        if self.offset is None:
            return x1, x2
        return x1 + self.offset[e, 0], x2 + self.offset[e, 1]

    def now(self, ids, orient: np.ndarray) -> Optional[np.ndarray]:
        """(len(ids), 2, 3): the offsets of members `ids` now, Q e (`orient`
        the points' triads); None when the model has none."""
        if self.offset is None:
            return None
        ids = np.asarray(ids, int)
        conn = self.mesh.elements[ids, :2]
        return np.einsum("enij,enj->eni", orient[conn], self.offset[ids])

    def ends(self, ids, x: np.ndarray, orient: np.ndarray) -> np.ndarray:
        """x (len(ids), 2, 3), the members' NODE positions, moved to their
        elastic ends (a copy; `x` itself when the model has no offsets)."""
        r = self.now(ids, orient)
        return x if r is None else x + r

    # ------------------------------------------------------------------ #
    #  forces and stiffness
    # ------------------------------------------------------------------ #

    def forces_to_nodes(self, ids, contrib: np.ndarray, orient: np.ndarray) -> None:
        """contrib (len(ids), 12): each member's end forces and moments in
        global axes at its ELASTIC ends, [f1, m1, f2, m2]; adds r x f to each
        moment, r = Q e now, so it is what the nodes get. In place; nothing
        without offsets."""
        r = self.now(ids, orient)
        if r is None:
            return
        _add_lever(contrib, r)

    def loads_to_nodes(self, ids, contrib: np.ndarray) -> None:
        """`forces_to_nodes` for a LOAD: the lever is the offset as drawn
        (the reference e), not as turned now -- a load vector is put together
        once and taken off as it was put on (a step that removes a loaded
        member), so both ends of its life see the same moments."""
        if self.offset is None:
            return
        _add_lever(contrib, self.offset[np.asarray(ids, int)])

    def load_to_nodes(self, e: int, g: np.ndarray) -> np.ndarray:
        """g (12,): a load at member `e`'s elastic ends in global axes, as its
        nodes get it (`loads_to_nodes` of one member; g itself without
        offsets)."""
        if self.offset is None:
            return g
        g = g.reshape(1, 12).copy()
        self.loads_to_nodes([e], g)
        return g[0]

    def stiffness_to_nodes(self, e: int, kg: np.ndarray, r=None) -> np.ndarray:
        """B^T kg B for member `e` (kg its 12x12 block in global axes at its
        elastic ends), at the offsets `r` ((2, 3); the reference ones when
        None). kg itself for a member without offsets."""
        if self.offset is None or not np.any(self.offset[e]):
            return kg
        if r is None:
            r = self.offset[e]
        b = lever_matrix(r[0], r[1])
        return b.T @ kg @ b

    def line_load(self, e: int, ql: np.ndarray, R: np.ndarray, L: float, condense) -> Tuple[np.ndarray, np.ndarray]:
        """(fl, g): a uniform line load on member `e` -- `ql` (3,) in its
        local axes `R` -- as the member-local fixed-end vector of its
        elastic part (through `condense`, its end releases) and the global
        vector at its nodes. Without offsets the closed form over the
        elastic length L (as every version has had it); with them, per unit
        length of the drawn line, the zones' share rigidly to their nodes
        (`drawn_loads`)."""
        if self.offset is None or not np.any(self.offset[e]):
            fl = np.zeros(12)
            fl[[0, 6]] = ql[0]*L/2
            fl[[1, 7]] = ql[1]*L/2
            fl[5], fl[11] = ql[1]*L**2/12, -ql[1]*L**2/12
            fl[[2, 8]] = ql[2]*L/2
            fl[4], fl[10] = -ql[2]*L**2/12, ql[2]*L**2/12
            fl = condense(fl)
            from .dof import repeated_rotation
            return fl, self.load_to_nodes(e, repeated_rotation(R).T @ fl)
        x1, x2 = self._drawn(e)
        ts, ws = _line_points(x1, x2, self.offset[e, 0], self.offset[e, 1], R, L)
        Ld = float(np.linalg.norm(x2 - x1))
        F = np.outer(ws * Ld, R.T @ np.asarray(ql, float))
        return self._carried(e, ts, F, R, L, condense)

    def _drawn(self, e):
        n1, n2 = self.mesh.elements[e, 0], self.mesh.elements[e, 1]
        return np.asarray(self.mesh.nodes[n1], float), np.asarray(self.mesh.nodes[n2], float)

    def _carried(self, e, ts, F, R, L, condense):
        x1, x2 = self._drawn(e)
        fl, g = drawn_loads(x1, x2, self.offset[e, 0], self.offset[e, 1], R, L, ts, F)
        if condense is not None:
            fl = condense(fl)
        from .dof import repeated_rotation
        return fl, g + self.load_to_nodes(e, repeated_rotation(R).T @ fl)

    def full_axial(self, e: int, kl: np.ndarray, L: float) -> np.ndarray:
        """Member `e`'s local stiffness kl (12x12, over its elastic length L)
        with its axial and torsional terms over the length between its NODES
        when its offsets say `axial: full` (kl itself otherwise): EA / L and
        GJ / L scaled by L / L_drawn."""
        if self.full is None or not self.full[e]:
            return kl
        x1, x2 = self._drawn(e)
        f = float(L) / float(np.linalg.norm(x2 - x1))
        kl = np.array(kl, float, copy=True)
        for a, b in ((0, 6), (3, 9)):
            kl[np.ix_([a, b], [a, b])] *= f
        return kl

    # ------------------------------------------------------------------ #
    #  the compiled kernels
    # ------------------------------------------------------------------ #

    def kernel_arrays(self, ids) -> Tuple[np.ndarray, np.ndarray]:
        """(offs (n, 2, 3), has (n,) uint8) gathered to members `ids`, for the
        compiled member kernels: zeros and 0 where a member has none."""
        ids = np.asarray(ids, int)
        if self.offset is None:
            return np.zeros((len(ids), 2, 3)), np.zeros(len(ids), np.uint8)
        o = np.ascontiguousarray(self.offset[ids], dtype=float)
        return o, np.ascontiguousarray(np.any(o != 0.0, axis=(1, 2)).astype(np.uint8))


#: a model without offsets: every method the identity
NONE = MemberGeometry()


def of(solver) -> MemberGeometry:
    """The solver's member geometry; `NONE` for a solver that has no line
    members (the solids, the shells)."""
    return solver.geometry if hasattr(solver, "geometry") else NONE


def _elastic_place(x1, x2, e1, e2, R, L):
    """(alpha, beta): the place along the elastic part, xi = alpha + beta t,
    of the point at t along the drawn line x1 -> x2 (its projection on the
    elastic axis R[0]); beta > 0 (end_offsets refuses an elastic part
    turned round)."""
    ex = R[0]
    E1 = x1 + e1
    return float(np.dot(x1 - E1, ex)) / L, float(np.dot(x2 - x1, ex)) / L


def _line_points(x1, x2, e1, e2, R, L):
    """(t, w): points along the drawn line and their weights (summing to 1)
    that integrate a uniform load on it exactly through `drawn_loads`: each
    rigid zone's share at its middle, the elastic part's by three Gauss
    points (the shapes are cubic in t there, the lever linear)."""
    a, b = _elastic_place(x1, x2, e1, e2, R, L)
    t0 = min(max(-a / b, 0.0), 1.0)
    t1 = min(max((1.0 - a) / b, 0.0), 1.0)
    ts, ws = [], []
    if t0 > 0.0:
        ts.append(0.5 * t0); ws.append(t0)
    if t1 > t0:
        g = np.sqrt(0.6)
        for x, w in ((-g, 5.0 / 9.0), (0.0, 8.0 / 9.0), (g, 5.0 / 9.0)):
            ts.append(0.5 * (t0 + t1) + 0.5 * (t1 - t0) * x); ws.append(0.5 * (t1 - t0) * w)
    if t1 < 1.0:
        ts.append(0.5 * (1.0 + t1)); ws.append(1.0 - t1)
    return np.asarray(ts), np.asarray(ws)


def drawn_loads(x1, x2, e1, e2, R, L, ts, F) -> Tuple[np.ndarray, np.ndarray]:
    """Forces F (n, 3, global) at points t (n,) of a member's drawn line
    x1 -> x2 (reference positions; e1, e2 its offsets, R its elastic axes,
    L its elastic length): (fl, g) -- the member-local consistent vector of
    the elastic part's share (uncondensed) and the global vector at the
    NODES of what falls on a rigid zone.

    A point whose place along the elastic part (`_elastic_place`) is before
    its start is on the first zone: its force goes to the first node with
    the moment (P - x1) x F, and likewise after its end. Any other point P
    is carried to the elastic axis at C, the same place along it: the force
    at C and the moment (P - C) x F there, each by its consistent shape."""
    a0, b0 = _elastic_place(x1, x2, e1, e2, R, L)
    E1, E2 = x1 + e1, x2 + e2
    d = x2 - x1
    fl = np.zeros(12); g = np.zeros(12)
    for t, f in zip(np.asarray(ts, float), np.asarray(F, float)):
        P = x1 + t * d
        xi = a0 + b0 * t
        if xi < 0.0:
            g[0:3] += f; g[3:6] += np.cross(P - x1, f)
            continue
        if xi > 1.0:
            g[6:9] += f; g[9:12] += np.cross(P - x2, f)
            continue
        C = E1 + xi * (E2 - E1)
        fl += hermite_force(xi, L, R @ f) + hermite_moment(xi, L, R @ np.cross(P - C, f))
    return fl, g


def hermite_force(a: float, L: float, pl: np.ndarray) -> np.ndarray:
    """The twelve local end actions of a local force `pl` at `a` (0 at end
    1, 1 at end 2) of a beam of length L: `moving_load.consistent_point_load`'s
    arithmetic."""
    b = 1.0 - a
    n1 = 1.0 - 3.0*a*a + 2.0*a**3; n2 = 3.0*a*a - 2.0*a**3
    m1 = L*a*b*b; m2 = -L*a*a*b
    fl = np.zeros(12)
    fl[0], fl[6] = pl[0]*b, pl[0]*a
    fl[1], fl[7] = pl[1]*n1, pl[1]*n2
    fl[5], fl[11] = pl[1]*m1, pl[1]*m2
    fl[2], fl[8] = pl[2]*n1, pl[2]*n2
    fl[4], fl[10] = -pl[2]*m1, -pl[2]*m2
    return fl


def hermite_moment(a: float, L: float, ml: np.ndarray) -> np.ndarray:
    """The twelve local end actions of a local point MOMENT `ml` at `a`: the
    slopes of the same functions (a twist by the linear ones), so the work
    of the moment through the beam's rotation there is the work of these
    through the ends' motion."""
    b = 1.0 - a
    s = 6.0 * a * b / L                      # -dN1/dx = dN2/dx
    r1 = b * b - 2.0 * a * b                 # d(L a b^2)/dx
    r2 = a * a - 2.0 * a * b                 # d(-L a^2 b)/dx
    fl = np.zeros(12)
    fl[3], fl[9] = ml[0] * b, ml[0] * a
    # about y: theta_y = -dw/dx
    fl[2], fl[8] = ml[1] * s, -ml[1] * s
    fl[4], fl[10] = ml[1] * r1, ml[1] * r2
    # about z: theta_z = dv/dx
    fl[1], fl[7] = -ml[2] * s, ml[2] * s
    fl[5], fl[11] = ml[2] * r1, ml[2] * r2
    return fl


def _add_lever(contrib: np.ndarray, r: np.ndarray) -> None:
    contrib[:, 3:6] += np.cross(r[:, 0], contrib[:, 0:3])
    contrib[:, 9:12] += np.cross(r[:, 1], contrib[:, 6:9])


def lever_matrix(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """12x12 B: the nodes' motion to the elastic ends' (u_end = u + theta x r);
    its transpose takes the ends' forces to the nodes."""
    out = np.eye(12)
    for blk, r in ((0, r1), (6, r2)):
        x, y, z = r
        # u_end = u - [r]x theta
        out[blk:blk + 3, blk + 3:blk + 6] = -np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return out
