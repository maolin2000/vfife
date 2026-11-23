"""Region selectors: naming a set of nodes without writing a bounding box.

A region is one primitive, or a boolean combination of regions named earlier or
later in the file.  Order does not matter -- references are resolved by
dependency, so a region can be built from parts declared below it.

    regions:
      base:     {plane: {point: [0, 0, 0], normal: [0, 0, 1]}}
      support:  {where: {x: 0.10, z: 0}}                  # a line: two axes pinned
      pin:      {where: {x: 0.10, y: 0.10, z: 0}}         # a point: three
      web:      {where: {y: {min: -0.05, max: 0.05}}}
      diagonal: {line: {from: [0, 0, 0], to: [2, 0, 0.3]}}
      shear:    {intersect: [base, web]}
      ends:     {union: [left_face, right_face]}      # gmsh physical groups
      interior: {difference: [all, surface]}

WHY THIS REPLACED `box`
-----------------------
The previous version offered a six-number bounding box or an explicit list of
node ids, and nothing else.  On a generated box mesh that is tolerable.  On any
real geometry it is guesswork: you cannot select a bolt circle, you cannot say
"the part of this Gmsh group above mid-height", and you cannot combine two
groups at all.  Every case file grew a column of near-identical boxes with
hand-tuned 1e-3 tolerances, and a tolerance that is slightly wrong selects
slightly the wrong nodes -- silently, because an almost-right support looks
exactly like a right one until you read the reactions.

TOLERANCES
----------
Every primitive with a tolerance takes it as an ABSOLUTE length, defaulting to
`tol` (1e-6 of the mesh diagonal).  A relative default is the right one here:
the same case file meshed in millimetres and in metres should select the same
nodes, and an absolute default cannot do that.
"""

from __future__ import annotations

from typing import Callable, Dict, List
import numpy as np

_PRIMITIVES = ("box", "where", "plane", "line", "cylinder", "sphere", "group",
               "node_ids", "all", "nearest")
_COMBINATORS = ("union", "intersect", "difference")

def resolve_all(nodes: np.ndarray, specs: Dict[str, dict],
                predefined: Dict[str, np.ndarray] | None = None,
                lazy: Dict[str, Callable[[], np.ndarray]] | None = None,
                notes: List[str] | None = None,
                ) -> Dict[str, np.ndarray]:
    """Resolve every named region to a sorted array of node indices.

    `predefined` holds regions that already exist -- Gmsh physical groups --
    which a spec may reference by name and combine with.

    `notes` collects, when given, one line per pinned `where` axis that had
    to SNAP to the nearest node line (see the `where` primitive); preflight
    reports them so the coordinate actually used is on the record.

    `lazy` holds regions that are available but not yet computed: a name maps
    to the function that would produce it.  One is evaluated only if a spec
    actually names it, and the result is a plain dict either way, so callers
    that do `dict(mesh.regions)` are unaffected.

    This exists for `surface`, which costs a pass over every element face and
    which no case in the library uses.  It was computed eagerly, twice per run
    -- once in preflight and once in the solver -- so every analysis paid for a
    node set nothing asked for; about 11 seconds of it at 10^6 elements.
    """
    out = dict(predefined or {})
    thunks = dict(lazy or {})
    order = _dependency_order(specs, set(out) | set(thunks))
    diag = float(np.linalg.norm(nodes.max(axis=0) - nodes.min(axis=0)))
    tol = 1e-6 * max(diag, 1.0)
    for name in order:
        idx = _resolve_one(nodes, name, specs[name], out, tol, thunks, notes)
        if idx.size == 0:
            raise ValueError(
                f"region {name!r} selected no nodes. Check the coordinates and "
                f"the tolerance (the default is {tol:.3e}, 1e-6 of the mesh "
                "diagonal); a plane or box that misses by less than a node "
                "spacing selects nothing and looks like a modelling error "
                "further downstream."
            )
        out[name] = idx
    return out

def _dependency_order(specs: Dict[str, dict], known: set) -> List[str]:
    """Topological order, so a region may be built from one declared later."""
    deps = {}
    for name, spec in specs.items():
        d = []
        for key in _COMBINATORS:
            if key in spec and spec[key] is not None:
                d.extend(spec[key])
        if spec.get("group") is not None:
            d.append(spec["group"])
        deps[name] = [x for x in d if x not in known]

    order, state = [], {}
    def visit(n, chain):
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            raise ValueError(
                "regions refer to each other in a cycle: "
                + " -> ".join(chain + [n])
            )
        if n not in deps:
            raise ValueError(
                f"region {n!r} is referenced but never defined. Defined here: "
                + (", ".join(sorted(specs)) or "(none)")
                + (f"; from the mesh: {', '.join(sorted(known))}" if known else "")
            )
        state[n] = 1
        for m in deps[n]:
            visit(m, chain + [n])
        state[n] = 2
        order.append(n)

    for name in specs:
        visit(name, [])
    return order

def _resolve_one(nodes, name, spec, known, tol, thunks=None, notes=None) -> np.ndarray:
    given = [k for k in _PRIMITIVES + _COMBINATORS
             if spec.get(k) is not None and spec.get(k) is not False]
    if len(given) != 1:
        raise ValueError(
            f"region {name!r} must give exactly one of "
            f"{', '.join(_PRIMITIVES + _COMBINATORS)}; it gives "
            + (", ".join(given) if given else "nothing")
        )
    kind = given[0]
    t = float(spec.get("tol") or tol)

    if kind in _COMBINATORS:
        parts = [_lookup(known, n, name, thunks) for n in spec[kind]]
        if kind == "union":
            return np.unique(np.concatenate(parts))
        if kind == "intersect":
            acc = parts[0]
            for p in parts[1:]:
                acc = np.intersect1d(acc, p)
            return acc
        acc = parts[0]                              # difference: first minus rest
        for p in parts[1:]:
            acc = np.setdiff1d(acc, p)
        return acc

    if kind == "group":
        return _lookup(known, spec["group"], name, thunks)
    if kind == "node_ids":
        ids = np.unique(np.asarray(spec["node_ids"], dtype=np.int64))
        bad = ids[(ids < 0) | (ids >= nodes.shape[0])]
        if bad.size:
            # -1 was the last node, silently
            raise ValueError(f"region {name!r}: node_ids {', '.join(map(str, bad[:5]))} are outside "
                             f"0 .. {nodes.shape[0] - 1} (node numbers start at 0)")
        return ids
    if kind == "all":
        return np.arange(nodes.shape[0], dtype=np.int64)
    if kind == "nearest":
        p = np.asarray(spec["nearest"], dtype=float)
        return np.array([int(np.argmin(np.linalg.norm(nodes - p, axis=1)))],
                        dtype=np.int64)

    if kind == "box":
        b = np.asarray(spec["box"], dtype=float)
        m = np.ones(nodes.shape[0], dtype=bool)
        for i in range(3):
            m &= (nodes[:, i] >= b[2 * i] - t) & (nodes[:, i] <= b[2 * i + 1] + t)
        return np.flatnonzero(m).astype(np.int64)

    if kind == "where":
        # any subset of the three axes, either bound optional -- a half-space,
        # a slab or a box, without having to invent numbers for the axes you
        # do not care about
        # A bare number pins that axis: {x: 0.1, z: 0} is the line x = 0.1
        # on the bottom face, {x: 0.1, y: 0.1, z: 0} a point on it. This is
        # what every support and load line in a beam case needs, and it
        # replaced six-number boxes with hand-tuned 1e-3 tolerances in every
        # one of them.
        m = np.ones(nodes.shape[0], dtype=bool)
        pinned = {a: float(v) for a, v in spec["where"].items() if isinstance(v, (int, float))}
        for axis, lim in spec["where"].items():
            if isinstance(lim, (int, float)):
                continue
            i = {"x": 0, "y": 1, "z": 2}[axis]
            if lim.get("min") is not None:
                m &= nodes[:, i] >= float(lim["min"]) - t
            if lim.get("max") is not None:
                m &= nodes[:, i] <= float(lim["max"]) + t
        # THE PINNED AXES ARE SNAPPED TOGETHER, to the node nearest the point
        # they describe among those the ranges allow. Snapped one axis at a
        # time over all nodes, {x: 0.99, z: 1} took x from a bottom-row node
        # and then found nothing at z = 1.
        snapped = {}
        if pinned and m.any():
            idx_ax = [{"x": 0, "y": 1, "z": 2}[a] for a in pinned]
            target = np.array(list(pinned.values()))
            cand = np.flatnonzero(m)
            d = np.linalg.norm(nodes[np.ix_(cand, idx_ax)] - target, axis=1)
            best = nodes[cand[int(np.argmin(d))]]
            snapped = {a: float(best[i]) for a, i in zip(pinned, idx_ax)}
        for axis, lim in spec["where"].items():
            i = {"x": 0, "y": 1, "z": 2}[axis]
            if isinstance(lim, (int, float)):
                # A pinned axis means "the node line at x"; a support at
                # x = 0.10 on a mesh whose nearest line is at 0.0833 is the
                # ordinary case, not an error, and the six-number boxes this
                # replaced were doing the same snap with a +-26 mm box and
                # no record of it. Snap to the nearest line, and say so when
                # the snap is more than the tolerance, so the coordinate
                # actually used is in the preflight report.
                target = float(lim)
                nearest = snapped.get(axis, target)
                if abs(nearest - target) > t and notes is not None:
                    # in the model's own length unit: "mm away" assumed metres
                    notes.append(f"region {name!r}: {axis} = {target:.6g} snapped to the "
                                 f"nearest node line at {axis} = {nearest:.6g} "
                                 f"({abs(nearest - target):.3g} away, in the model's length unit)")
                m &= np.abs(nodes[:, i] - nearest) <= t
        return np.flatnonzero(m).astype(np.int64)

    if kind == "line":
        # nodes within `tol` of the SEGMENT from `from` to `to` -- not the
        # infinite line, so a support line does not pick up the far end
        l = spec["line"]
        a = np.asarray(l["from"], dtype=float)
        b = np.asarray(l["to"], dtype=float)
        ab = b - a
        L2 = float(ab @ ab)
        if L2 <= 0.0:
            raise ValueError(f"region {name!r}: line from and to coincide; use `nearest` for a point")
        s_ = np.clip(((nodes - a) @ ab) / L2, 0.0, 1.0)
        d = np.linalg.norm(nodes - (a + s_[:, None] * ab[None, :]), axis=1)
        return np.flatnonzero(d <= float(l.get("tol") or t)).astype(np.int64)

    if kind == "plane":
        p = spec["plane"]
        pt = np.asarray(p["point"], dtype=float)
        n = np.asarray(p["normal"], dtype=float)
        n = n / np.linalg.norm(n)
        d = np.abs((nodes - pt) @ n)
        return np.flatnonzero(d <= float(p.get("tol") or t)).astype(np.int64)

    if kind == "sphere":
        s = spec["sphere"]
        c = np.asarray(s["centre"], dtype=float)
        r = np.linalg.norm(nodes - c, axis=1)
        m = r <= float(s["radius"]) + t
        if s.get("rmin") is not None:
            m &= r >= float(s["rmin"]) - t
        return np.flatnonzero(m).astype(np.int64)

    if kind == "cylinder":
        c = spec["cylinder"]
        pt = np.asarray(c["point"], dtype=float)
        ax = np.asarray(c["axis"], dtype=float)
        ax = ax / np.linalg.norm(ax)
        rel = nodes - pt
        along = rel @ ax
        radial = np.linalg.norm(rel - along[:, None] * ax[None, :], axis=1)
        m = radial <= float(c["radius"]) + t
        if c.get("rmin") is not None:
            m &= radial >= float(c["rmin"]) - t
        if c.get("length") is not None:
            m &= (along >= -t) & (along <= float(c["length"]) + t)
        return np.flatnonzero(m).astype(np.int64)

    raise ValueError(f"region {name!r}: unknown selector {kind!r}")


def _lookup(known, ref, by, thunks=None):
    if ref not in known and thunks and ref in thunks:
        known[ref] = thunks[ref]()          # computed here, because it is used
    try:
        return known[ref]
    except KeyError:
        raise ValueError(
            f"region {by!r} refers to {ref!r}, which is not defined. "
            f"Available: {', '.join(sorted(known)) or '(none)'}"
        ) from None


def surface_provider(mesh, element) -> Callable[[], np.ndarray]:
    """A thunk for the `surface` region, memoised on the mesh.

    Both preflight and the solver want it, and they share a mesh, so without
    the cache a run that does use `surface` would pay for it twice.

    `element` is one element name, or several for a mixed mesh.  The face count
    is global either way: a quadrilateral shared by a hexahedron and a pyramid
    is seen twice and is correctly interior, which is exactly what makes the
    transition region come out right.
    """
    def compute() -> np.ndarray:
        cached = getattr(mesh, "_surface_nodes", None)
        if cached is None:
            names = [element] if isinstance(element, str) else list(element)
            blocks = getattr(mesh, "cell_blocks", None) or {}
            cached = surface_nodes(
                [(n, blocks[n]) for n in names] if len(names) > 1
                else mesh.elements,
                names[0] if len(names) == 1 else names, mesh.n_nodes)
            try:
                mesh._surface_nodes = cached
            except AttributeError:          # a frozen or slotted Mesh
                pass
        return cached
    return compute


def surface_nodes(conn, element, n_nodes: int) -> np.ndarray:
    """Nodes on the outside of the mesh: those on a face no two elements share.

    Provided as a predefined region called `surface`, because "everything except
    the outside" and "the outside near here" are both common and neither can be
    written as a box.
    """
    from ..elements import library

    # one (name, connectivity) pair per element type present
    blocks = (conn if isinstance(conn, list)
              else [(element if isinstance(element, str) else element[0], conn)])
    # Pyramid5 has mixed quad (4) and tri (3) faces; Counter handles variable width
    from collections import Counter
    counter = Counter()
    for name, block_conn in blocks:
        for f in library.get(name).faces:
            for elem in block_conn:
                counter[tuple(sorted(int(elem[i]) for i in f))] += 1
    outer_nodes = set()
    for tup, cnt in counter.items():
        if cnt == 1:
            outer_nodes.update(tup)
    if not outer_nodes:
        return np.zeros(0, dtype=np.int64)
    return np.array(sorted(outer_nodes), dtype=np.int64)
