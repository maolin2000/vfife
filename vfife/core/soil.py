"""Soil springs on piles: p-y curves by depth (1.2).

    soil_springs:
      - region: pile                 # the pile's members (a line region)
        ground: 0.0                  # the ground surface, on the `vertical` axis
        vertical: z
        curves:                      # p per unit length of pile against y,
          - {depth: 0.0,  p_y: [[0, 0], [0.005, 40e3], [0.02, 90e3], [0.05, 120e3]]}
          - {depth: 10.0, p_y: [[0, 0], [0.005, 160e3], [0.02, 400e3], [0.05, 520e3]]}

Each node of the region below the ground gets a horizontal spring: its force
is the curve at the node's depth times the length of pile the node stands
for (half of each member next to it, the part below ground). Between two
depths the curves are interpolated linearly (each at the same y); above the
first depth the first curve, below the last the last. Beyond a curve's last
point p stays at its last value (the ultimate resistance). The curves are the
user's: the code formulas (API, JTG 3363) are not in the library -- they are
the geotechnical engineer's to choose and to own.

NONLINEAR ELASTIC (the user's decision, 2026-11-05): loading and unloading
follow the same curve. There is no hysteresis and no gap behind the pile, so
a seismic run dissipates nothing in the soil (unconservative for the damping
it leaves out); a pushover, a dead-load state, modes and buckling are what
it is for. In the plane (`direction: horizontal`, the default) the spring is
radial -- the resistance follows the resultant of the two horizontal
displacements, so a 45-degree push meets the same soil as an axial one -- and
it has a potential, so the tangent `modal`, `buckling` and the static solver
take from the internal force is symmetric. `direction: x` (or `y`) is one
axis only, for a plane model.

The forces join the internal force of the frame families (beam2,
beam2_timoshenko, mixed_line, beam2_fiber); the springs' stiffness joins the
reference stiffness the explicit step and the relaxation masses are taken
from (the curves' steepest slope); and the soil's force on the pile is in
the `reaction_force` field and the reaction monitors, as a support's is.
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np

_AXES = "xyz"


class SoilSprings:
    def __init__(self, specs: List[dict], nodes: np.ndarray, members: np.ndarray,
                 region: Callable, members_in: Callable):
        #: one group per spec: (node ids (m,), axes, y grid (ny,), P (m, ny))
        #: -- P is each node's force curve, its length times p, on the union
        #: of the spec's y points (each curve is exact on it), so one force
        #: evaluation is a few array operations, not a loop over the nodes
        #: (a 1000-member pile model ran 17 times slower with the loop)
        self.groups = []
        nodes = np.asarray(nodes, float)
        for k, spec in enumerate(specs):
            where = f"soil_springs[{k}]"
            up = _AXES.index(spec.get("vertical", "z"))
            ground = float(spec.get("ground", 0.0))
            elems = np.asarray(members_in(np.asarray(region(spec["region"]))))
            if not len(elems):
                raise ValueError(f"{where}: region {spec['region']!r} contains no complete members: the "
                                 "springs stand on a pile's members, whose lengths they share")
            curves = sorted(spec["curves"], key=lambda c: float(c["depth"]))
            depths = np.array([float(c["depth"]) for c in curves])
            tables = [np.asarray(c["p_y"], float) for c in curves]
            trib = {}
            for e in elems:
                a, b = (int(i) for i in members[e][:2])
                za, zb = ground - nodes[a, up], ground - nodes[b, up]       # depths, + below ground
                L = float(np.linalg.norm(nodes[b] - nodes[a]))
                if L <= 0.0:
                    continue
                # THE BURIED PART, SHARED AS ITS LINEAR SHAPE FUNCTIONS SHARE
                # IT. Half of it to each end, and nothing to an end above the
                # ground, lost the soil next to a ground surface inside an
                # element: first-order convergence, +23% on 30 elements with
                # the ground 0.13 m below a node (review of 1.2)
                if za < 0.0 and zb < 0.0:
                    continue
                if za >= 0.0 and zb >= 0.0:
                    s0, s1 = 0.0, 1.0
                else:
                    sc = za / (za - zb)                    # where the member crosses the ground
                    s0, s1 = (0.0, sc) if za >= 0.0 else (sc, 1.0)
                share_b = 0.5 * (s1 ** 2 - s0 ** 2) * L
                share_a = (s1 - s0) * L - share_b
                trib[a] = trib.get(a, 0.0) + share_a
                trib[b] = trib.get(b, 0.0) + share_b
            trib = {n: v for n, v in trib.items() if v > 0.0}
            if not trib:
                raise ValueError(f"{where}: no member of {spec['region']!r} is below the ground "
                                 f"({spec.get('vertical', 'z')} = {ground:g})")
            direction = spec.get("direction", "horizontal")
            axes = [i for i in range(3) if i != up] if direction == "horizontal" else [_AXES.index(direction)]
            grid = np.unique(np.concatenate([t[:, 0] for t in tables]))
            ids = np.array(sorted(trib), dtype=np.int64)
            P = np.empty((len(ids), len(grid)))
            for r, n in enumerate(ids):
                # an end just above the ground (its member crosses it) takes
                # the curve at the surface
                y, pv = _curve_at(depths, tables, max(ground - nodes[n, up], 0.0))
                P[r] = trib[int(n)] * np.interp(grid, y, pv)
            self.groups.append((ids, axes, grid, P))

    @property
    def node_ids(self) -> np.ndarray:
        if not self.groups:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate([g[0] for g in self.groups]))

    def __len__(self):
        return int(sum(len(g[0]) for g in self.groups))

    def force(self, u: np.ndarray) -> np.ndarray:
        """(nn, 3): the springs' share of the internal force (the soil's force
        on the pile is minus this), at translations u (nn, 3)."""
        u = np.asarray(u, float)
        out = np.zeros_like(u)
        for ids, axes, grid, P in self.groups:
            uh = u[np.ix_(ids, axes)]
            r = np.linalg.norm(uh, axis=1)
            # piecewise-linear on the grid, flat past its last point
            rc = np.minimum(r, grid[-1])
            j = np.clip(np.searchsorted(grid, rc, side="right") - 1, 0, len(grid) - 2)
            g0, g1 = grid[j], grid[j + 1]
            rows = np.arange(len(ids))
            w = (rc - g0) / (g1 - g0)
            f = (1.0 - w) * P[rows, j] + w * P[rows, j + 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                dirn = np.where(r[:, None] > 0.0, uh / r[:, None], 0.0)
            np.add.at(out, (ids[:, None], np.asarray(axes)[None, :]), f[:, None] * dirn)
        return out

    def stiffness(self, nn: int) -> np.ndarray:
        """(nn, 3): each spring's steepest stiffness, for the stable step."""
        out = np.zeros((nn, 3))
        for ids, axes, grid, P in self.groups:
            k = np.max(np.diff(P, axis=1) / np.diff(grid)[None, :], axis=1)
            for a in axes:
                np.add.at(out[:, a], ids, k)
        return out

    def mask(self, nn: int) -> np.ndarray:
        """(nn, 3) bool: the translations the springs act on."""
        out = np.zeros((nn, 3), bool)
        for ids, axes, _g, _P in self.groups:
            out[np.ix_(ids, axes)] = True
        return out


def _curve_at(depths, tables, depth):
    """The p-y curve at `depth`: the two neighbouring curves interpolated
    at the union of their y points (each held flat past its last)."""
    if depth <= depths[0]:
        t = tables[0]
        return t[:, 0].copy(), t[:, 1].copy()
    if depth >= depths[-1]:
        t = tables[-1]
        return t[:, 0].copy(), t[:, 1].copy()
    j = int(np.searchsorted(depths, depth)) - 1
    w = (depth - depths[j]) / (depths[j + 1] - depths[j])
    a, b = tables[j], tables[j + 1]
    y = np.union1d(a[:, 0], b[:, 0])
    return y, (1.0 - w) * np.interp(y, a[:, 0], a[:, 1]) + w * np.interp(y, b[:, 0], b[:, 1])


# --------------------------------------------------------------------------- the solvers' side
def attach(solver):
    """Build `solver.soil` from its problem, or None."""
    specs = (getattr(solver, "p", None) or {}).get("soil_springs") or []
    if not specs:
        solver.soil = None
        return None
    conn = getattr(solver, "_elements0", None)
    conn = solver.mesh.elements if conn is None else conn
    solver.soil = SoilSprings(specs, solver.mesh.nodes[:solver.nn], np.asarray(conn),
                              solver._region, solver._members_in)
    return solver.soil


def add_force(solver, out: np.ndarray) -> np.ndarray:
    """`out` (the flat internal force) with the springs' forces added."""
    soil = getattr(solver, "soil", None)
    if soil is None or not len(soil):
        return out
    from ..base.dof import TRANSLATION
    u = solver.dofs.field(solver.q, TRANSLATION)
    f = soil.force(u[:solver.nn])
    solver.dofs.add_nodal(out, np.arange(solver.nn), f, TRANSLATION)
    solver.soil_force = f
    return out


def add_stiffness(solver):
    """The springs' stiffness on the diagonal of `solver.K`."""
    soil = getattr(solver, "soil", None)
    if soil is None or not len(soil):
        return
    import scipy.sparse as sp
    from ..base.dof import TRANSLATION
    k = soil.stiffness(solver.nn)
    diag = np.zeros(solver.ndof)
    solver.dofs.add_nodal(diag, np.arange(solver.nn), k, TRANSLATION)
    solver.K = (solver.K + sp.diags(diag)).tocsr()


def reaction_residual(solver, residual):
    """(the residual the reactions are read from, the springs' dofs as a
    mask). At a held dof it is the supports' force on the structure, the
    soil's included (the residual less the springs' share); at a spring's
    free dof it is the soil's force on the pile, minus the springs' share --
    NOT the residual there, which in a dynamic run holds the inertia and the
    damping: a ramped head load read as -99.7 kN of soil reaction where the
    soil carried -16 kN (review of 1.2)."""
    mask = np.zeros(len(residual), bool)
    soil = getattr(solver, "soil", None)
    f = getattr(solver, "soil_force", None)
    if soil is None or not len(soil) or f is None:
        return residual, mask
    from ..base.dof import TRANSLATION
    flat = np.zeros(len(residual))
    solver.dofs.add_nodal(flat, np.arange(solver.nn), f[:solver.nn], TRANSLATION)
    on = np.zeros(len(residual), bool)
    on_nodes = soil.mask(solver.nn)
    rows = solver.dofs.node_dofs(np.arange(solver.nn), TRANSLATION)
    on[np.asarray(rows)[on_nodes]] = True
    held = np.asarray(solver.current_fixed, bool)
    out = np.array(residual, float, copy=True)
    out[held] -= flat[held]
    free = on & ~held
    out[free] = -flat[free]
    return out, on
