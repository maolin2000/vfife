"""A load that travels along the structure.

Every other load in this library is a fixed distribution of nodal forces
scaled by an amplitude: the shape is decided once, in `step_data`, and the
step only changes how much of it is applied. A vehicle crossing a bridge is
not that. Its magnitude never changes; what changes is WHERE it acts, and
between two nodes it acts nowhere the mesh has a degree of freedom.

So a moving load is two separate things:

  * a PATH -- the members the load runs along, ordered end to end, with an
    arc-length coordinate `s` measured along the undeformed geometry; and
  * a rule that turns a point force at `s` into end forces on the one member
    that contains it.

The second is the consistent (work-equivalent) distribution: the Hermite
shape functions of the beam element, which for a prismatic member reproduce
the exact deflection at the nodes for a point load anywhere between them --
the same functions the element's own stiffness is built from, and the same
ones that integrate to the `line_load` end forces when a uniform load is
summed over the member. The cheap alternative, the lever rule (1-xi, xi)
with no end moments, is a different structure: with the load in the middle of
a member it loses 27% of the midspan deflection on one element per half span,
and it is never exact, at any refinement. It is not offered.

WHAT THIS IS NOT. This is a moving FORCE, not a moving mass: the vehicle has
no inertia of its own, does not bounce on its suspension, and does not follow
the deck down as it deflects (the force keeps its direction and its station
on the UNDEFORMED path). That is the standard first model of a vehicle on a
bridge, and it is the one whose closed-form solution exists to check against
(Fryba); vehicle-bridge interaction is a later thing that needs the vehicle
as a model of its own.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np



def consistent_point_load(L: float, R: np.ndarray, xi: float, p: np.ndarray) -> np.ndarray:
    """The twelve local end actions of a point force `p` (global) acting at
    `xi` (0 at node 1, 1 at node 2) along a beam member of length `L` whose
    local axes are the rows of `R`.

    Axial: the lever rule, which IS the consistent distribution for the bar's
    linear axial field. Transverse: the cubic shape functions, so the member
    carries the load's fixed-end moments as well as its shears. Integrating
    these over a uniform xi reproduces the `line_load` block exactly
    (qL/2 and +-qL^2/12), which is the check that the signs agree with the
    rest of the library.
    """
    pl = R @ np.asarray(p, float)
    a = float(np.clip(xi, 0.0, 1.0)); b = 1.0 - a
    n1 = 1.0 - 3.0*a*a + 2.0*a**3            # deflection shape at node 1
    n2 = 3.0*a*a - 2.0*a**3
    m1 = L*a*b*b                             # fixed-end moment, node 1
    m2 = -L*a*a*b
    fl = np.zeros(12)
    fl[0], fl[6] = pl[0]*b, pl[0]*a
    fl[1], fl[7] = pl[1]*n1, pl[1]*n2
    fl[5], fl[11] = pl[1]*m1, pl[1]*m2
    fl[2], fl[8] = pl[2]*n1, pl[2]*n2
    fl[4], fl[10] = -pl[2]*m1, -pl[2]*m2
    return fl


class LoadPath:
    """The members of a region, ordered end to end, with arc length along them.

    A region is a set of nodes; the members wholly inside it must form ONE
    open chain, which is what a carriageway is. Anything else -- a fork, a
    ring, two disconnected runs -- is refused by name rather than silently
    traversed in mesh order, because mesh order is not travel order and a
    load that jumps about would still produce plausible-looking numbers.

    `s = 0` is the end of the chain that comes first in (x, y, z), so on the
    usual bridge laid out along +x the load enters at the left abutment and
    positive speed drives it towards the right.
    """

    def __init__(self, mesh, elements: Sequence[int], what: str = "moving_load"):
        elements = np.asarray(sorted(int(e) for e in elements), int)
        if not len(elements):
            raise ValueError(f"{what} region contains no complete members")
        conn = mesh.elements[elements][:, :2]
        adj: Dict[int, List[Tuple[int, int]]] = {}
        for e, (i, j) in zip(elements, conn):
            adj.setdefault(int(i), []).append((int(e), int(j)))
            adj.setdefault(int(j), []).append((int(e), int(i)))
        ends = [n for n, links in adj.items() if len(links) == 1]
        forks = [n for n, links in adj.items() if len(links) > 2]
        if forks:
            raise ValueError(f"{what} path branches at node(s) {sorted(forks)[:4]}: "
                             "the load must travel one chain of members")
        if len(ends) != 2:
            raise ValueError(f"{what} path is not one open chain: it has {len(ends)} "
                             "free end(s) (a closed ring, or more than one run)")
        start = min(ends, key=lambda n: tuple(np.round(mesh.nodes[n], 9)))
        order: List[int] = []; nodes = [start]
        seen = set(); node = start
        while True:
            nxt = [(e, o) for e, o in adj[node] if e not in seen]
            if not nxt:
                break
            e, o = nxt[0]; seen.add(e); order.append(e); nodes.append(o); node = o
        if len(order) != len(elements):
            raise ValueError(f"{what} path is not one open chain: "
                             f"{len(elements) - len(order)} member(s) are not reachable "
                             "from its end")
        self.elements = np.asarray(order, int)
        self.nodes = np.asarray(nodes, int)
        # a member numbered against the travel: its xi runs from ITS node i,
        # which is the far end of the path here (v1.0: the mirrored
        # distribution gave 3.8x the midspan deflection)
        self.flipped = np.array([int(mesh.elements[e][0]) != int(n)
                                 for e, n in zip(order, nodes[:-1])], bool)
        xyz = mesh.nodes[self.nodes]
        self.lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        self.stations = np.concatenate([[0.0], np.cumsum(self.lengths)])
        self.total = float(self.stations[-1])

    def locate(self, s: float) -> Optional[Tuple[int, float]]:
        """`(element, xi)` for arc length `s`, or None when `s` is off the
        path -- which is how a vehicle drives on and off the bridge."""
        if not (0.0 <= s <= self.total):
            return None
        k = int(np.searchsorted(self.stations, s, side="right") - 1)
        k = min(max(k, 0), len(self.elements) - 1)
        L = self.lengths[k]
        xi = float((s - self.stations[k]) / L) if L else 0.0
        return int(self.elements[k]), (1.0 - xi if self.flipped[k] else xi)

    def place(self, s: np.ndarray):
        """(k, xi) for stations `s` on the path: the path index of the member
        under each and the position from the member's own node i."""
        st = self.stations
        k = np.clip(np.searchsorted(st, s, side="right") - 1, 0, len(self.elements) - 1)
        Lp = self.lengths[k]
        a = np.clip(np.where(Lp > 0, (s - st[k]) / np.where(Lp > 0, Lp, 1.0), 0.0), 0.0, 1.0)
        a = np.where(self.flipped[k], 1.0 - a, a)
        return k, a

    def shape(self, s: np.ndarray, R: np.ndarray, L: np.ndarray, p: np.ndarray,
              releases: Optional[np.ndarray] = None):
        """For stations `s` (all on the path): the path index k of the member
        under each, and the member-local (n, 12) and global (n, 12) consistent
        end actions of the forces `p` (n, 3, global) there. `R`, `L` are the
        path members' axes and lengths. xi is from each member's own node i.
        `releases` (npath, 12, 12), from `path_releases`, condenses the local
        vector of a member with end releases, as a `line_load`'s is."""
        k, a = self.place(s)
        try:
            from ..kernels.numba_loads import path_actions as _pa
        except ImportError:                                   # pragma: no cover
            _pa = None
        if _pa is not None:
            # the compiled version (1.6), which the compiled step runs too:
            # one arithmetic for both paths, to the bit
            n = len(s)
            fl = np.zeros((n, 12)); fg = np.zeros((n, 12))
            has = releases is not None
            rel = np.ascontiguousarray(releases, dtype=float) if has else np.zeros((1, 12, 12))
            _pa(np.ascontiguousarray(k, dtype=np.int64), np.ascontiguousarray(a, dtype=float),
                np.ascontiguousarray(R, dtype=float), np.ascontiguousarray(L, dtype=float),
                np.ascontiguousarray(np.broadcast_to(p, (n, 3)), dtype=float), rel, has, fl, fg)
            return k, fl, fg
        b = 1.0 - a
        Rk = R[k]; Lk = L[k]
        pl = np.einsum("nij,nj->ni", Rk, p)
        n1 = 1.0 - 3.0*a*a + 2.0*a**3; n2 = 3.0*a*a - 2.0*a**3
        m1 = Lk*a*b*b; m2 = -Lk*a*a*b
        fl = np.zeros((len(s), 12))
        fl[:, 0], fl[:, 6] = pl[:, 0]*b, pl[:, 0]*a
        fl[:, 1], fl[:, 7] = pl[:, 1]*n1, pl[:, 1]*n2
        fl[:, 5], fl[:, 11] = pl[:, 1]*m1, pl[:, 1]*m2
        fl[:, 2], fl[:, 8] = pl[:, 2]*n1, pl[:, 2]*n2
        fl[:, 4], fl[:, 10] = -pl[:, 2]*m1, -pl[:, 2]*m2
        if releases is not None:
            # a released end takes no moment: the fixed-end actions of a
            # member with releases are condensed through them
            # (the uncondensed moment went to the node and bent the NEXT
            # member -- a hinge deflected 30% too far)
            fl = np.einsum("nij,nj->ni", releases[k], fl)
        fg = np.einsum("nji,nbj->nbi", Rk, fl.reshape(-1, 4, 3)).reshape(-1, 12)   # T^T fl
        return k, fl, fg


def path_releases(solver, elements) -> Optional[np.ndarray]:
    """(npath, 12, 12): the linear map a member's end releases apply to its
    local fixed-end vector (`_apply_load_releases`, the rule a `line_load`
    follows), identity where it has none; None when no path member has any."""
    apply = getattr(solver, "_apply_load_releases", None)
    if apply is None:
        return None
    props_of = ((lambda e: solver.member_props[e]) if hasattr(solver, "member_props")
                else (lambda e: solver.p["element"]))
    out = None
    for i, e in enumerate(np.asarray(elements, int)):
        props = props_of(int(e))
        if not solver._release_indices(props):
            continue
        if out is None:
            out = np.tile(np.eye(12), (len(elements), 1, 1))
        L = float(solver.length[int(e)])
        mat = solver._material_of(int(e)) if hasattr(solver, "_material_of") else None
        out[i] = np.stack([apply(col, L, props, mat) for col in np.eye(12)], axis=1)
    return out


class MovingLoad:
    """One group of axles travelling along a `LoadPath`.

    Called with the step's `progress` and physical time `t`, it returns the
    nodal force vector for where the axles are at that instant. It is handed
    to the solver as an ordinary load group, so an amplitude still scales it
    (`hold`, a ramp, or anything else) and everything downstream -- reactions,
    monitors, the energy audit -- sees a perfectly normal external force.

    Position: `speed` gives s = start + speed * t, the physical crossing.
    `stop` instead sweeps s from `start` to `stop` over the step's progress,
    which with `load_steps: n` is n static positions -- an influence line for
    the cost of n relaxations and no dynamics at all.
    """

    def __init__(self, spec: dict, path: LoadPath, dofs, mesh, length, R0, total_time: float):
        self.path, self.dofs, self.mesh = path, dofs, mesh
        self.length, self.R0 = length, R0
        self.total_time = float(total_time)
        axles = np.asarray(spec["axles"], float).reshape(-1, 2)
        self.offsets, self.magnitudes = axles[:, 0], axles[:, 1]
        d = np.asarray(spec.get("direction", [0.0, 0.0, -1.0]), float)
        n = float(np.linalg.norm(d))
        if n == 0.0:
            raise ValueError("moving_load direction must not be the zero vector")
        self.direction = d / n
        self.start = float(spec.get("start", 0.0))
        self.speed = spec.get("speed"); self.stop = spec.get("stop")
        self.speed = None if self.speed is None else float(self.speed)
        self.stop = None if self.stop is None else float(self.stop)
        self.ndof = dofs.ndof if hasattr(dofs, "ndof") else None

    def station(self, progress: float, t: Optional[float]) -> float:
        """Where the reference point of the axle group is at this instant."""
        if self.speed is not None:
            # `t` is None on the quasi-static path (and in the invariant
            # probes, which ask for the force at progress = 1 without a
            # clock). The step's own clock is progress * total_time, which
            # is what the dynamic loop passes in anyway.
            when = t if t is not None else progress * self.total_time
            return self.start + self.speed * float(when)
        return self.start + float(progress) * (self.stop - self.start)

    def _prepare(self):
        """Per-path arrays the vectorised evaluation needs, built once."""
        el = self.path.elements
        self._R = np.asarray(self.R0)[el]                              # (n, 3, 3)
        self._L = np.asarray(self.length, float)[el]
        self._edofs = np.stack([self.dofs.element_dofs(self.mesh.elements[e][:2]) for e in el])
        # an offset is measured along the direction of TRAVEL (docs/moving_load.md:
        # a trailing axle is -4.2 whichever way the group goes), so a group
        # driven towards decreasing s carries its axles mirrored in s
        if self.speed is not None:
            way = -1.0 if self.speed < 0 else 1.0
        else:
            way = -1.0 if self.stop < self.start else 1.0
        self._signed = way * self.offsets

    def __call__(self, progress: float, t: Optional[float], out: np.ndarray) -> np.ndarray:
        """All axles at once: one searchsorted, the Hermite shape functions as
        arrays, and one scatter-add -- a platoon of a hundred axles costs about
        what one did."""
        if not hasattr(self, "_signed"):
            self._prepare()
        s = self.station(progress, t) + self._signed
        on = (s >= 0.0) & (s <= self.path.total)       # axles not on the path yet
        self.member_load = None
        if not on.any():
            return out
        s = s[on]; mag = self.magnitudes[on]
        k, fl, fg = self.path.shape(s, self._R, self._L, mag[:, None] * self.direction[None, :],
                                    getattr(self, "releases", None))
        # the member-local share, for the members' end actions
        self.member_load = (self.path.elements[k], fl)
        np.add.at(out, self._edofs[k].ravel(), fg.ravel())
        return out


def member_loads(solver, groups, progress: float) -> Optional[np.ndarray]:
    """(nel, 12) member-local equivalent loads acting INSIDE the members at
    the last force evaluation -- the line loads and the axles -- or None.

    A member's true end actions are k d minus these: the nodal equivalent
    carries the load to the nodes, and its fixed-end moments belong to the
    member. Without it an axle between nodes read its member's end moment
    6.7% low."""
    out = None
    line = getattr(solver, "_line_local", None) or {}
    if line:
        amps = {a.name: a for a in solver.amplitudes.values()}
        for name, arr in line.items():
            part = arr * amps[name](progress)
            out = part if out is None else out + part
    for a, F in groups:
        ml = getattr(F, "member_load", None) if callable(F) else None
        if ml is not None:
            if out is None:
                out = np.zeros((solver.nel, 12))
            np.add.at(out, ml[0], ml[1] * a(progress))
    if out is not None:
        # a member that carries nothing has no end actions to correct: a
        # removed or fractured one read -fl, a fixed-end moment on a member
        # that is gone
        dead = np.zeros(len(out), bool)
        active = getattr(solver, "member_active", None)
        if active is not None and np.shape(active) == (len(out),):
            dead |= ~np.asarray(active, bool)
        live = getattr(solver, "_active", None)
        if live is not None and np.shape(live) == (len(out),):
            dead |= np.asarray(live) == 0
        if dead.any():
            out = out.copy()
            out[dead] = 0.0
    return out
