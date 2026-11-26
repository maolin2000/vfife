"""Rigid end offsets of line members (1.21).

A member's elastic part need not run from node to node. `end_offsets` puts
each end of it at a fixed vector from its node: the stretch between the
node and the elastic end is RIGID, carried by the node's own rotation. Two
things are this:

* a rigid end zone -- the part of a beam inside a column or a pier cap,
  Midas' *OFFSET -- an offset along the member;
* a section set off its nodes -- a girder drawn on its top line with its
  centroid 2 m below (Midas' CT) -- an offset across the member, the same
  at both ends of a member of constant depth and different at the two ends
  of a tapered one.

The second was a centroid node and a rigid arm per node (`offset_arms`):
an extra point with its own mass and a "rigid" member whose stiffness had
to be picked between too soft (the arm bends) and too stiff (the iteration
stalls on round-off). Here the offset is exact and there is no extra point.

This module reads the input: `offsets` turns the assignments' specs into
each member's two offsets. What they do -- the elastic ends, the forces at
the nodes, the stiffness -- is `core/member_geometry.MemberGeometry`, the
solver's `geometry`.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

#: what an end_offsets spec may hold
KEYS = {"i", "j", "axes", "per", "axial"}


def parse(spec, where: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """(e_i, e_j, axes) of one `end_offsets` spec; raises with `where`.
    (`per` is read by `per_of`.)"""
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: end_offsets is {{i: [x, y, z], j: [x, y, z], axes: local|global, "
                         "per: line|member}}")
    bad = set(spec) - KEYS
    if bad:
        raise ValueError(f"{where}: end_offsets has no key {sorted(bad)[0]!r} "
                         "(i, j, axes, per, axial)")
    if spec.get("axial", "elastic") not in ("elastic", "full"):
        raise ValueError(f"{where}: end_offsets.axial is elastic or full, not {spec.get('axial')!r}")
    axes = spec.get("axes", "local")
    if axes not in ("local", "global"):
        raise ValueError(f"{where}: end_offsets.axes is local or global, not {axes!r}")
    per_of(spec, where)
    out = []
    for end in ("i", "j"):
        v = spec.get(end, [0.0, 0.0, 0.0])
        try:
            v = np.asarray(v, dtype=float).reshape(3)
        except (TypeError, ValueError):
            raise ValueError(f"{where}: end_offsets.{end} is three numbers, not {v!r}") from None
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{where}: end_offsets.{end} is not finite")
        out.append(v)
    if spec.get("i") is None and spec.get("j") is None:
        raise ValueError(f"{where}: end_offsets gives neither i nor j")
    return out[0], out[1], axes


def per_of(spec, where: str) -> str:
    """`line` (the default: the region is one line, `i` at its start and
    `j` at its end) or `member` (every member of the region has the same
    two offsets, `i` at its first node and `j` at its second)."""
    per = spec.get("per", "line")
    if per not in ("line", "member"):
        raise ValueError(f"{where}: end_offsets.per is line or member, not {per!r}")
    return per


def _axes(xi, xj, reference, where="end_offsets") -> np.ndarray:
    """A member's local axes as rows (`BeamFamily._frame`'s rule): x along
    it, y the orientation made normal to it, z = x cross y."""
    ex = np.asarray(xj, float) - np.asarray(xi, float)
    ex = ex / np.linalg.norm(ex)
    ref = np.asarray(reference if reference is not None else [0.0, 0.0, 1.0], float)
    ey = ref - ex * np.dot(ref, ex)
    if np.linalg.norm(ey) < 1e-10:
        raise ValueError(f"{where}: end_offsets: the orientation is parallel to a member")
    ey = ey / np.linalg.norm(ey)
    return np.vstack([ex, ey, np.cross(ex, ey)])


def _chain(elements, ids):
    """The members `ids` as one open chain from its start: [(member,
    reversed)], the nodes in order, or None when they are not one. The
    start is the end node that is its member's FIRST node (a line meshed
    from A to B starts at A)."""
    ids = [int(e) for e in ids]
    if len(ids) == 1:
        return [(ids[0], False)], [int(elements[ids[0], 0]), int(elements[ids[0], 1])]
    touching = {}
    for e in ids:
        for n in elements[e, :2]:
            touching.setdefault(int(n), []).append(e)
    ends = [n for n, m in touching.items() if len(m) == 1]
    if len(ends) != 2 or any(len(m) > 2 for m in touching.values()):
        return None
    starts = [n for n in ends if int(elements[touching[n][0], 0]) == n]
    if len(starts) != 1:
        return None
    n, out, nodes, seen = starts[0], [], [starts[0]], set()
    while True:
        nxt = [e for e in touching[n] if e not in seen]
        if not nxt:
            break
        e = nxt[0]
        seen.add(e)
        rev = int(elements[e, 0]) != n
        n = int(elements[e, 0] if rev else elements[e, 1])
        out.append((e, rev))
        nodes.append(n)
    return (out, nodes) if len(out) == len(ids) else None


def offsets(nodes, elements, nel: int,
            groups: Iterable[Tuple[np.ndarray, dict, Optional[list], str]]) -> Optional[np.ndarray]:
    """(nel, 2, 3): each member's two offsets as global vectors in the
    reference state (= in its nodes' own axes, the triads starting at the
    identity), or None when no assignment gives one.

    `groups`: (element ids, end_offsets spec, orientation, name) of each
    assignment that gives one. The offsets are the LINE's: a region meshed
    into several members is one open chain, `i` at its start and `j` at its
    end. Along the chain (x) an offset is a rigid zone, and only the chain's
    two ends have one; across it (y, z) it is the section set off its nodes,
    linear in the arc length from `i` to `j` at every node (constant when
    the two are equal). `local` axes are each member's own, from its NODES'
    chord in the chain's direction and its orientation (y the orientation
    made normal to it, z = x cross y); `global` offsets are a vector,
    linear in the arc length from `i` to `j` at every node, the ends
    included (no rigid zone of their own: that is an offset along the line,
    in local axes).

    `per: member` takes the members one by one instead: each has `i` at its
    first node and `j` at its second, in its own local axes (or global),
    along and across alike -- a region of members that need not be a
    line, as a converted model's are (each Midas element its own offsets,
    grouped by them)."""
    off = None
    for ids, spec, orientation, name in groups:
        if spec is None:
            continue
        where = f"assignment on {name!r}"
        ei, ej, axes = parse(spec, where)
        ids = np.asarray(ids, int)
        if not len(ids):
            continue

        if off is None:
            off = np.zeros((nel, 2, 3))
        if per_of(spec, where) == "member":
            for e in ids:
                if axes == "global":
                    a, b = ei, ej
                else:
                    R = _axes(nodes[elements[e, 0]], nodes[elements[e, 1]], orientation, where)
                    a, b = R.T @ ei, R.T @ ej
                _check_length(nodes, elements, e, a, b, where)
                off[e, 0], off[e, 1] = a, b
            continue
        ch = _chain(elements, ids)
        if ch is None:
            raise ValueError(
                f"{where}: end_offsets need the region's members to be one open chain with one "
                "start (a line meshed from its first point to its last: no branch, no loop, and "
                "exactly one end that is its member's first node); give each line its own "
                "assignment")
        chain, order = ch

        def axes_of(e, rev):
            a, b = (elements[e, 1], elements[e, 0]) if rev else (elements[e, 0], elements[e, 1])
            return _axes(nodes[a], nodes[b], orientation, where)
        # arc length of every node of the chain
        s = np.concatenate([[0.0], np.cumsum([np.linalg.norm(nodes[order[k + 1]] - nodes[order[k]])
                                              for k in range(len(chain))])])
        S = s[-1]
        for k, (e, rev) in enumerate(chain):
            R = axes_of(e, rev)
            ends = []
            for at in (k, k + 1):                       # the member's two nodes, in chain order
                t = s[at] / S
                if axes == "global":
                    # the vector itself, linear along the line: one point per node whatever
                    # the line's turns (a rigid zone along the line is given in local axes)
                    ends.append((1.0 - t) * ei + t * ej)
                    continue
                v = np.zeros(3)
                v[1:] = (1.0 - t) * ei[1:] + t * ej[1:]
                if at == 0:
                    v[0] = ei[0]
                if at == len(chain):
                    v[0] = ej[0]
                ends.append(R.T @ v)
            a, b = (ends[1], ends[0]) if rev else (ends[0], ends[1])
            _check_length(nodes, elements, e, a, b, where)
            off[e, 0], off[e, 1] = a, b
    return off


def full_axial(nel: int, groups) -> Optional[np.ndarray]:
    """(nel,) bool: the members whose assignment says `axial: full` -- the
    axial and torsional stiffness over the length between the NODES, the
    rigid zones only rigid in bending and shear (Midas, ETABS); None when
    none does. `elastic` (the default) keeps them over the elastic part (a
    rigid zone rigid in every direction, OpenSees' and Nastran's)."""
    out = np.zeros(nel, bool)
    for ids, spec, _orientation, _name in groups:
        if spec is not None and len(ids):
            # a later assignment wins, as its offsets do
            out[np.asarray(ids, int)] = spec.get("axial", "elastic") == "full"
    return out if out.any() else None


def _check_length(nodes, elements, e, a, b, where):
    """Raise when offsets a, b leave member e no elastic length (or turn it
    round)."""
    n1, n2 = elements[e, 0], elements[e, 1]
    chord = nodes[n2] - nodes[n1]
    elastic = (nodes[n2] + b) - (nodes[n1] + a)
    Ln = float(np.linalg.norm(chord))
    if np.linalg.norm(elastic) < 1e-6 * Ln or float(np.dot(elastic, chord)) <= 0.0:
        raise ValueError(
            f"{where}: the end offsets of member {int(e)} leave no elastic length "
            "between them (or turn it round): a rigid zone is longer than the "
            "member -- mesh it finer or shorten the zone")

