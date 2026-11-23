"""A line-member mesh written in the case (`mesh: {type: lines}`, 1.1).

A bridge's frame model is a few dozen points and the members between them,
and until 1.1 it had to be drawn in Gmsh: a script of points, lines,
transfinite divisions and physical groups, exported to a .msh the case then
read. That is the biggest step between a bridge engineer and a first run,
and none of it is mechanics. Here the same model is written in the case:

    mesh:
      type: lines
      points:
        A: [0, 0, 0]
        B: [40, 0, 0]
        C: [80, 0, 0]
        P: [40, 0, -15]
      lines:
        girder: {through: [A, B, C], size: 2.0}
        pier:   {through: [P, B], divisions: 6}
        arch:   {through: [A, C], divisions: 24, arc: {rise: 12}}

What it builds is what `io.gmsh.read` returns for a Line2 file, so every
consumer downstream -- the sections' `assign`, regions, preflight, the
solvers, `view`, the fingerprint -- sees no difference:

* one Line2 block (`truss2`), the members of each line in order from its
  first point, each segment divided into `divisions` elements (or
  ceil(length / size));
* a region per point (its one node) and per line (its nodes), and an element
  region per line (its members), as a Gmsh physical group gives;
* named points are nodes: lines that pass through the same point share it.
  Lines meet nowhere else: two nodes at one place (a line laid over part of
  another, a line attached at another's division node, two points at one
  place) and two members on one pair of nodes are refused, in these terms.
  A point written as coordinates is the named point there, or a new one;
* `groups: {name: [line, ...]}` joins lines into one ELEMENT region, as a
  Gmsh physical group of several pieces is (for `assign` and `removals`); a
  `regions:` union of lines is a set of nodes only.

Arcs (`arc: {rise, shape, up}`) are drawn between a line's two points: a
parabola with its nodes evenly along the chord (an arch axis), or a circle
evenly by angle, rising `rise` at mid-chord along `up`.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from ..core.mesh import Mesh


def _xyz(x) -> str:
    return "[" + ", ".join(f"{float(v):.6g}" for v in x) + "]"


def _line_of(element_regions, e):
    return next((k for k, v in element_regions.items() if e in v), "?")


def _no_overlap(nodes, conn, element_regions, point_node, tol):
    """Refuse two nodes at one place and two members on one pair of nodes.

    A line laid over part of another (the natural way to write "the main
    span" of a girder) made the shared stretch two members side by side, or
    two node chains at one place that preflight then refused as a duplicate
    node in Gmsh's words; a line attached to another's division node by
    coordinates did the same (review of 1.1). Said here, in the mesh's terms."""
    from scipy.spatial import cKDTree
    pairs = cKDTree(nodes).query_pairs(tol) if len(nodes) > 1 else set()
    if pairs:
        i, j = sorted(next(iter(pairs)))
        def owner(n):
            rows = np.flatnonzero((conn == n).any(axis=1))
            return _line_of(element_regions, int(rows[0])) if len(rows) else "?"
        raise ValueError(f"mesh.lines.{owner(i)} and mesh.lines.{owner(j)} both put a node at {_xyz(nodes[i])} "
                         "without sharing it: lines meet only at a named point. Name the point in "
                         "`points` and put it in both lines' `through`; for a part of a line (a span "
                         "of a girder), write the parts as lines and join them in `groups`")
    key = np.sort(conn, axis=1)
    _u, first, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    if (counts > 1).any():
        k = int(np.flatnonzero(counts > 1)[0])
        dup = np.flatnonzero((key == key[first[k]]).all(axis=1))
        raise ValueError(f"mesh.lines.{_line_of(element_regions, int(dup[0]))} and "
                         f"mesh.lines.{_line_of(element_regions, int(dup[1]))} both have a member from "
                         f"{_xyz(nodes[key[first[k], 0]])} to {_xyz(nodes[key[first[k], 1]])}: two members "
                         "side by side. For a part of a line, write the parts as lines and join them "
                         "in `groups`")


def _arc_points(a, b, n, arc) -> List[np.ndarray]:
    """The n - 1 interior points of an arc from a to b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    chord = b - a
    L = float(np.linalg.norm(chord))
    up = np.asarray(arc.get("up", [0.0, 0.0, 1.0]), float)
    up = up - (up @ chord) / (L * L) * chord              # the part across the chord
    if not np.linalg.norm(arc.get("up", [0, 0, 1])) > 0.0:
        raise ValueError("the arc's `up` is [0, 0, 0]: give the direction it rises in")
    if np.linalg.norm(up) <= 1e-12 * float(np.linalg.norm(arc.get("up", [0, 0, 1]))):
        raise ValueError("the arc's `up` is along its chord: give a direction across it")
    up /= np.linalg.norm(up)
    f = float(arc["rise"])
    if f == 0.0:
        return [a + chord * (k / n) for k in range(1, n)]
    if arc.get("shape", "parabola") == "parabola":
        return [a + chord * s + up * (4.0 * f * s * (1.0 - s)) for s in (k / n for k in range(1, n))]
    # a circle through a, b and the apex: radius from the sagitta
    R = (L * L / 4.0 + f * f) / (2.0 * abs(f))
    side = math.copysign(1.0, f)
    centre = a + chord / 2.0 - side * up * (R - abs(f))
    half = math.asin(min(1.0, (L / 2.0) / R))
    if abs(f) > L / 2.0:                                  # more than a half circle
        half = math.pi - half
    ex = chord / L
    out = []
    for k in range(1, n):
        t = -half + 2.0 * half * k / n
        out.append(centre + R * (math.sin(t) * ex + side * math.cos(t) * up))
    return out


def build(spec) -> Mesh:
    """The mesh of a `MeshSpec` of type `lines` (or its dict)."""
    spec = spec.model_dump() if hasattr(spec, "model_dump") else dict(spec)
    named: Dict[str, np.ndarray] = {k: np.asarray(v, float) for k, v in (spec.get("points") or {}).items()}
    lines = spec.get("lines") or {}
    coords = [p for p in named.values()]
    for line in lines.values():
        coords += [np.asarray(p, float) for p in line["through"] if not isinstance(p, str)]
    span = float(np.ptp(np.array(coords), axis=0).max()) if len(coords) > 1 else 1.0
    tol = 1e-9 * max(span, 1.0)

    nodes: List[np.ndarray] = []
    point_node: Dict[str, int] = {}
    for name, x in named.items():
        # TWO POINTS AT ONE PLACE ARE REFUSED. They would be two nodes no
        # member can join (a member has a length) that preflight then refuses
        # as a duplicate node; a bearing's two sides are two points a little
        # apart, joined by a connector (review of 1.1)
        for other, j in point_node.items():
            if np.linalg.norm(nodes[j] - x) <= tol:
                raise ValueError(f"mesh.points.{name} and mesh.points.{other} are at one place, "
                                 f"{_xyz(x)}: a point is a node, and two nodes at one place cannot be "
                                 "joined by a member; give a bearing's two sides a small offset and a "
                                 "connector between them")
        point_node[name] = len(nodes)
        nodes.append(x)
    anonymous: List[int] = []

    def node_of(p, where):
        if isinstance(p, str):
            return point_node[p]
        x = np.asarray(p, float)
        here = [n for n in list(point_node.values()) + anonymous if np.linalg.norm(nodes[n] - x) <= tol]
        if here:
            return here[0]
        anonymous.append(len(nodes))
        nodes.append(x)
        return anonymous[-1]

    elements: List[List[int]] = []
    element_regions: Dict[str, np.ndarray] = {}
    node_regions: Dict[str, np.ndarray] = {}
    for name, line in lines.items():
        where = f"mesh.lines.{name}"
        ends = [node_of(p, where) for p in line["through"]]
        mine: List[int] = []
        touched: List[int] = []
        for i, (a, b) in enumerate(zip(ends[:-1], ends[1:])):
            if a == b:
                raise ValueError(f"{where}: segment {i + 1} starts and ends at the same node")
            xa, xb = nodes[a], nodes[b]
            L = float(np.linalg.norm(xb - xa))
            if L <= tol:
                raise ValueError(f"{where}: segment {i + 1} has no length (its two points are at one "
                                 "place); a member joins two different places -- a connector too: give "
                                 "its two points a small offset")
            n = line.get("divisions")
            arc = line.get("arc")
            if n is None and line.get("size"):
                # along the curve, not the chord: a 12 m rise on a 60 m span
                # is 6% longer than its chord
                length = L
                if arc:
                    pts = [xa] + _arc_points(xa, xb, 400, arc) + [xb]
                    length = float(np.sum(np.linalg.norm(np.diff(np.asarray(pts), axis=0), axis=1)))
                n = int(math.ceil(length / float(line["size"]) - 1e-9))
            n = int(n or 1)
            if arc:
                try:
                    _arc_points(xa, xb, 2, arc)
                except ValueError as exc:
                    raise ValueError(f"{where}: {exc}") from None
            inner = (_arc_points(xa, xb, n, arc) if arc else
                     [xa + (xb - xa) * (k / n) for k in range(1, n)])
            chain = [a]
            for x in inner:
                chain.append(len(nodes))
                nodes.append(np.asarray(x, float))
            chain.append(b)
            for p, q in zip(chain[:-1], chain[1:]):
                mine.append(len(elements))
                elements.append([p, q])
            touched += chain
        element_regions[name] = np.asarray(mine, dtype=np.int64)
        node_regions[name] = np.unique(np.asarray(touched, dtype=np.int64))

    used = np.zeros(len(nodes), bool)
    if elements:
        used[np.asarray(elements).ravel()] = True
    unused = [k for k, v in point_node.items() if not used[v]]
    if unused:
        raise ValueError(f"mesh.points.{unused[0]} is on no line: a node no member reaches has no "
                         "stiffness (a point for a load or a support goes on a line's `through`)")
    conn = np.ascontiguousarray(np.asarray(elements, dtype=np.int32).reshape(-1, 2))
    _no_overlap(np.asarray(nodes, float), conn, element_regions, point_node, tol)
    for gname, members in (spec.get("groups") or {}).items():
        element_regions[gname] = np.unique(np.concatenate([element_regions[m] for m in members]))
        node_regions[gname] = np.unique(np.concatenate([node_regions[m] for m in members]))
    mesh = Mesh(nodes=np.ascontiguousarray(np.asarray(nodes, float)), elements=conn,
                cell_blocks={"truss2": conn},
                element_regions={k: {"truss2": v} for k, v in element_regions.items()})
    mesh.regions.update({k: np.asarray([v], dtype=np.int64) for k, v in point_node.items()})
    mesh.regions.update(node_regions)
    mesh.import_notes = []
    return mesh
