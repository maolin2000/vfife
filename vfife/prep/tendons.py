"""Tendons of a frame model, written one to an entry and expanded on reading.

A case's top-level `tendons:` describes each tendon the way a designer does
-- its line, area, jacking, friction, draw-in and relaxation -- and this
module turns it into what the solver reads: a node of the tendon's own at
each node of the girder it is bonded to (and at its two anchors), each on a
rigid arm to the girder's node, and a bar between each two, jacked to the
force the tendon has there in the step that stresses it. It is what the Midas
converter did since 1.15, moved here so that a
case written by hand and one converted from Midas expand the same way, and
the converter can write the short form.

The expansion is in memory, every time the case is read; `vfife expand`
writes the long form only to look at. What it adds goes after the mesh's own
nodes and groups, so nothing that was there moves.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import brentq


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TendonLine(_Strict):
    #: a point of the tendon's reference line, where its profile's x is 0
    origin: List[float] = Field(min_length=3, max_length=3)
    #: the reference line's direction (x); its y is horizontal, across it, and z the third axis (up for a line that is not vertical)
    direction: List[float] = Field(min_length=3, max_length=3)
    #: the y and z axes turned about x by this angle (degrees)
    turn: float = 0.0


class TendonStress(_Strict):
    #: the step that stresses it (its name); none: the step in which the last of the members it is bonded to is built
    step: Optional[str] = None
    #: the end or ends it is jacked at: `begin`, `end`, `both`
    ends: Literal["begin", "end", "both"] = "begin"
    #: the stress at the jack, at its begin and its end (stress unit), or ...
    stress: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    #: ... the whole tendon's force at the jack (force unit); one of the two
    force: Optional[List[float]] = Field(None, min_length=2, max_length=2)

    @model_validator(mode="after")
    def _one(self):
        if (self.stress is None) == (self.force is None):
            raise ValueError("give one of `stress` or `force` at the jack")
        return self


class TendonFriction(_Strict):
    #: the friction coefficient mu, per radian of the line's turning
    mu: float = Field(0.0, ge=0.0)
    #: the wobble coefficient k, per unit of length
    k: float = Field(0.0, ge=0.0)


class TendonDrawIn(_Strict):
    #: the anchorage's draw-in at the begin (length unit)
    begin: float = Field(0.0, ge=0.0)
    #: the anchorage's draw-in at the end (length unit)
    end: float = Field(0.0, ge=0.0)


class TendonRelaxation(_Strict):
    #: the code whose loss it follows (`vfife.codes`): `jtg3362` for now
    code: Literal["jtg3362"] = "jtg3362"
    #: the characteristic strength fpk (stress unit)
    fpk: float = Field(gt=0.0)
    #: the code's relaxation coefficient zeta
    zeta: float = Field(gt=0.0)
    #: over-stressed: psi 0.9 instead of 1.0
    overstressed: bool = False


class TendonSpec(_Strict):
    #: its name: what the results report it by, and the start of the regions it expands to (`name/bar1`, `name/arm1`)
    name: str
    #: the girder nodes it is bonded to: a group of the mesh; the ones along its length are used, and the nearest to each anchor
    nodes: str
    #: a bar section: the area of one strand or bar, and its material
    section: str
    #: strands or bars in the duct: the area is times this
    count: int = Field(1, ge=1)
    #: its reference line
    line: TendonLine
    #: [oy, oz] added to the profile everywhere
    offset: List[float] = Field([0.0, 0.0], min_length=2, max_length=2)
    #: across the line: [x, y, radius] points; none: on the line
    profile_y: List[List[float]] = []
    #: its drape: [x, z, radius] points, each corner rounded to its radius
    profile_z: List[List[float]] = Field(min_length=2)
    #: its jacking
    stress: TendonStress
    #: its friction losses
    friction: TendonFriction = TendonFriction()
    #: its anchorages' draw-in
    draw_in: TendonDrawIn = TendonDrawIn()
    #: its relaxation (none: it does not relax)
    relaxation: Optional[TendonRelaxation] = None
    #: an arm's stiffness: this times the tendon's stiffest bar (4 E A / L)
    arm_factor: float = Field(30.0, gt=0.0)

    @model_validator(mode="after")
    def _points(self):
        for which, pts in (("profile_y", self.profile_y), ("profile_z", self.profile_z)):
            if any(len(p) != 3 for p in pts):
                raise ValueError(f"tendon {self.name!r}: each point of {which} is [x, value, radius]")
            if any(b[0] <= a[0] for a, b in zip(pts, pts[1:])):
                raise ValueError(f"tendon {self.name!r}: the points of {which} do not run forward along its line")
        return self


def free_axis(a, b):
    """The orientation of a member whose section has no axes of its own
    (a pipe, a round bar, a rigid arm): global Z, or global X for a member
    that is nearer the vertical than 45 degrees."""
    x = [b[i] - a[i] for i in range(3)]
    n = math.sqrt(sum(c * c for c in x)) or 1.0
    return (1.0, 0.0, 0.0) if abs(x[2]) / n > 0.7071 else (0.0, 0.0, 1.0)


def _rounded(points):
    """A polyline [(x, v, radius)] with each corner rounded to its radius:
    [(x, v)], the arcs drawn in short chords."""
    pts = [(float(x), float(v)) for x, v, _r in points]
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        r = float(points[i][2] or 0.0)
        (xa, va), (xb, vb), (xc, vc) = out[-1], pts[i], pts[i + 1]
        d1 = (xb - xa, vb - va)
        d2 = (xc - xb, vc - vb)
        l1, l2 = math.hypot(*d1), math.hypot(*d2)
        if r <= 0.0 or l1 == 0.0 or l2 == 0.0:
            out.append(pts[i])
            continue
        d1, d2 = (d1[0] / l1, d1[1] / l1), (d2[0] / l2, d2[1] / l2)
        turn = math.atan2(d1[0] * d2[1] - d1[1] * d2[0], d1[0] * d2[0] + d1[1] * d2[1])
        if abs(turn) < 1.0e-9:
            out.append(pts[i])
            continue
        t = min(r * math.tan(abs(turn) / 2.0), 0.5 * l1, 0.5 * l2)
        r_ = t / math.tan(abs(turn) / 2.0)
        a = (xb - d1[0] * t, vb - d1[1] * t)
        side = 1.0 if turn > 0 else -1.0
        centre = (a[0] - side * d1[1] * r_, a[1] + side * d1[0] * r_)
        start = math.atan2(a[1] - centre[1], a[0] - centre[0])
        n = max(4, int(abs(turn) / math.radians(2.0)) + 1)
        for j in range(n + 1):
            ang = start + side * abs(turn) * j / n
            out.append((centre[0] + r_ * math.cos(ang), centre[1] + r_ * math.sin(ang)))
    out.append(pts[-1])
    return out


def _drawn_in(s, force, slip, ea):
    """The force along a tendon after its anchorage at s = 0 draws in by
    `slip`: within the length l where the draw-in is taken up, the force is
    the friction curve mirrored about its value at l (friction acts the other
    way as the tendon slides back), sum over l of the loss equal to slip EA."""
    force = np.asarray(force, dtype=float)
    if slip <= 0.0 or len(s) < 2:
        return force
    need = slip * ea

    def taken(i):                                     # what a mirror about the force at s[i] takes up
        loss = 2.0 * (force[: i + 1] - force[i])
        return float(np.sum(0.5 * (loss[1:] + loss[:-1]) * np.diff(s[: i + 1]))) if i > 0 else 0.0
    i = next((j for j in range(1, len(s)) if taken(j) >= need), None)
    out = force.copy()
    if i is None:                                     # the whole tendon slides back: a further even drop
        drop = (need - taken(len(s) - 1)) / max(s[-1], 1.0e-12)
        out = 2.0 * force[-1] - force - drop
        return np.maximum(out, 0.0)
    lo, hi = s[i - 1], s[i]
    for _ in range(40):                               # the length, between two points of the curve
        mid = 0.5 * (lo + hi)
        fm = float(np.interp(mid, s, force))
        inside = s < mid
        loss = 2.0 * (np.concatenate([force[inside], [fm]]) - fm)
        ss = np.concatenate([s[inside], [mid]])
        got = float(np.sum(0.5 * (loss[1:] + loss[:-1]) * np.diff(ss)))
        lo, hi = (mid, hi) if got < need else (lo, mid)
    fm = float(np.interp(lo, s, force))
    out[s < lo] = 2.0 * fm - force[s < lo]
    return out


def _drawn_in_both(s, force, slip_b, slip_e, ea):
    """The force along a tendon jacked at both ends after both anchorages
    draw in (by slip_b at s = 0, slip_e at the far end).

    Where an end's draw-in moves the tendon, the force grows from that
    anchorage inward by the friction it had, |dF/ds| -- the curve mirrored
    on the jack's own side, parallel to it on the other jack's side -- so it
    is a + V(s) from the near end and b + W(s) from the far one, with V, W
    the friction's accumulated |dF| from each end. The force is the least of
    the three, F, a + V, b + W: an end's draw-in reaches where its line meets
    the force as jacked, and if the two lines meet first, that point is the
    one that does not move, and each end moves the tendon up to it. a and b
    are what makes each end's lost stretch, the area it takes from F over
    EA, its slip. With the two draw-ins apart this is each end's mirrored
    curve (JTG D62 6.2.3); with a short tendon it is the even drop of l_f >
    l/2, and unequal halves move the still point instead of stepping there
    (review)."""
    s = np.asarray(s, dtype=float)
    f = np.asarray(force, dtype=float)
    if len(s) < 2 or (slip_b <= 0.0 and slip_e <= 0.0):
        return f.copy()
    v = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(f)))])
    w = v[-1] - v
    top = float(f.max())
    big = 4.0 * top + v[-1]

    eps = 1.0e-7 * top
    ds = np.diff(s)

    def held(d):                                      # the antiderivative of the near end's share, 1 to 0 across +-eps
        mid = -eps + 0.5 * (d + eps) - (d * d - eps * eps) / (4.0 * eps)
        return np.where(d <= -eps, d, np.where(d >= eps, 0.0, mid))

    def lost(a, b):
        """The stretch each end takes back, as force times length. Where the
        two lines are within eps of each other -- a flat stretch of F, or a
        crossing inside an interval -- the share goes over smoothly, so each
        end's area is continuous in a and b (review: a tie given to the far
        end left the near one half its slip on a straight middle)."""
        g = np.minimum(f, np.minimum(a + v, b + w))
        loss = f - g
        d = (a + v) - (b + w)
        d0, d1 = d[:-1], d[1:]
        flat = np.abs(d1 - d0) < 1.0e-12 * top
        share0 = np.clip(0.5 - d0 / (2.0 * eps), 0.0, 1.0)
        span = np.where(flat, 1.0, d1 - d0)
        share = np.where(flat, share0, (held(d1) - held(d0)) / span)
        whole = 0.5 * (loss[1:] + loss[:-1]) * ds
        near = float(np.sum(whole * share))
        return near, float(np.sum(whole)) - near

    need_b, need_e = max(slip_b, 0.0) * ea, max(slip_e, 0.0) * ea

    def level(delta, need, part):
        """a, with b = a + delta, at which `part` of the loss (0 near, 1 far,
        2 both) is `need`: shifting both lines down only takes more."""
        def gap(a):
            got = lost(a, a + delta)
            return (got[0] + got[1] if part == 2 else got[part]) - need
        lo, hi = -big - abs(delta), big + abs(delta)
        if gap(hi) >= 0.0:
            return hi
        if gap(lo) <= 0.0:
            return lo
        return brentq(gap, lo, hi, xtol=1.0e-9 * top, rtol=1.0e-14, maxiter=200)
    if need_b <= 0.0:                                 # one end only: its line alone, the other far above
        b = level(-2.0 * big, need_e, 1) - 2.0 * big
        a = big
    elif need_e <= 0.0:
        a, b = level(2.0 * big, need_b, 0), big
    else:
        # the two lines' offset puts the still point where each end takes
        # its own slip: with the total held, a higher far line leaves the
        # near end more (review: an alternating solve of a
        # and b stalled on a straight middle where the two lines lie level)
        def share(delta):
            a_ = level(delta, need_b + need_e, 2)
            return lost(a_, a_ + delta)[0] - need_b
        d_lo, d_hi = -2.0 * big, 2.0 * big
        if share(d_lo) >= 0.0:
            delta = d_lo
        elif share(d_hi) <= 0.0:
            delta = d_hi
        else:
            delta = brentq(share, d_lo, d_hi, xtol=1.0e-9 * top, rtol=1.0e-14, maxiter=200)
        a = level(delta, need_b + need_e, 2)
        b = a + delta
    return np.maximum(np.minimum(f, np.minimum(a + v, b + w)), 0.0)




# --------------------------------------------------------------------------- #
#  the expansion
# --------------------------------------------------------------------------- #

def _element_steps(mesh, steps) -> np.ndarray:
    """For each member, the step it is built in: 0 for one no step
    activates, else the first that does. A name is a group of the mesh --
    its members, or a node group's members with both nodes in it, as the
    solver reads `activate` (`Mesh.region_elements`)."""
    conn = (mesh.cell_blocks or {}).get("truss2", mesh.elements)
    first = np.zeros(conn.shape[0], dtype=np.int64)
    seen = np.zeros(conn.shape[0], dtype=bool)
    for i, st in enumerate(steps):
        for name in (st.get("activate") or []):
            if name not in (mesh.element_regions or {}) and name not in mesh.regions:
                raise ValueError(f"step {st.get('name')!r} activates {name!r}, which is not a group of the mesh: "
                                 "with `tendons`, a step builds mesh groups (the tendons' arms are born with them)")
            rows = np.asarray(mesh.region_elements(name, "truss2", conn), np.int64)
            fresh = rows[~seen[rows]]
            first[fresh] = i
            seen[fresh] = True
    return first


def _built_step(mesh, first) -> np.ndarray:
    """For each node, the first step in which a member at it stands."""
    conn = (mesh.cell_blocks or {}).get("truss2", mesh.elements)
    out = np.full(mesh.nodes.shape[0], np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(out, conn[:, 0], first)
    np.minimum.at(out, conn[:, 1], first)
    out[out == np.iinfo(np.int64).max] = 0
    return out


def expand(d: Dict[str, Any], mesh, warn=None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """(the case without `tendons`, with what they are; the mesh overlay,
    for `gmsh.read(..., overlay=)`; a report: per tendon its step, jacking
    forces and bars). `mesh` is the case's mesh as read, with the rows' tags
    (`mesh.node_tags`). `warn(who, text)` hears what is followed only roughly."""
    from ..io.model import SectionSpec
    from ..codes import jtg3362
    from ..core.units import scales, to_case

    warn = warn or (lambda _who, _t: None)
    d = copy.deepcopy(d)
    specs = [TendonSpec(**t) for t in (d.pop("tendons", None) or [])]
    names = [t.name for t in specs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"tendon name {dup[0]!r} is used twice")
    materials = d.setdefault("materials", {})
    sections = d.setdefault("sections", {})
    if specs and any(isinstance(v, dict) and v.get("type") == "fiber" for v in sections.values()):
        raise ValueError("the case's `tendons` are for frame models; a fibre beam's tendons are written in "
                         "its section (`sections.<name>.tendons`)")
    assign = d.setdefault("assign", [])
    steps = d.get("steps") or []
    step_of: Dict[Any, List[int]] = {}
    for i, st in enumerate(steps):
        step_of.setdefault(st.get("name"), []).append(i)
    # the lengths below are a metre's fractions: in the case's own unit
    # (`units`), and the weightless density 1e-6 kg/m^3 too
    L = scales(d.get("units")).get("length") or 1.0
    light = to_case(1.0e-6, "density", d.get("units")) or 1.0e-6

    def key(p):                              # a place, to a micrometre
        return tuple(round(float(c) * L, 6) for c in p)
    xyz = np.asarray(mesh.nodes, float)
    tags = np.asarray(getattr(mesh, "node_tags"), np.int64)
    el_step = _element_steps(mesh, steps) if specs else None
    built = _built_step(mesh, el_step) if specs else None
    conn = (mesh.cell_blocks or {}).get("truss2", mesh.elements)
    place_of = {key(p): True for p in xyz}
    taken_sections = set(sections)
    taken_materials = set(materials)

    def new_material(name, value):           # one this makes, never one of the case's
        if name in taken_materials:
            raise ValueError(f"a material named {name!r} is already in `materials`; the tendons make one of that name")
        materials.setdefault(name, value)
    new_nodes: List[Tuple[float, float, float]] = []
    lines: List[Tuple[str, List[Tuple[int, int]]]] = []
    report: Dict[str, Any] = {}

    for tp in specs:
        who = f"tendon {tp.name!r}"
        if tp.nodes not in mesh.regions:
            raise ValueError(f"{who}: `nodes: {tp.nodes}` is not a group of the mesh")
        if tp.section not in sections:
            raise ValueError(f"{who}: `section: {tp.section}` is not in `sections`")
        sec = SectionSpec(**sections[tp.section])
        if sec.type != "bar":
            raise ValueError(f"{who}: `section: {tp.section}` is a {sec.type} section; a tendon takes a bar section")
        mat = sec.material
        if mat not in materials or "E" not in materials[mat]:
            raise ValueError(f"{who}: its section's material {mat!r} has no elastic modulus E")
        stress_step = None
        if tp.stress.step is not None:
            if tp.stress.step not in step_of:
                raise ValueError(f"{who}: `stress.step: {tp.stress.step}` is not a step of the case")
            if len(step_of[tp.stress.step]) > 1:
                raise ValueError(f"{who}: `stress.step: {tp.stress.step}` names {len(step_of[tp.stress.step])} steps; "
                                 "give the steps distinct names")
            stress_step = step_of[tp.stress.step][0]
        # its reference line
        dv = [float(c) for c in tp.line.direction]
        n_d = math.sqrt(sum(c * c for c in dv))
        if n_d == 0.0:
            raise ValueError(f"{who}: the line's direction is zero")
        ex = tuple(c / n_d for c in dv)
        origin = tuple(float(c) for c in tp.line.origin)
        ey = [-ex[1], ex[0], 0.0]
        n_y = math.sqrt(ey[0] ** 2 + ey[1] ** 2)
        if n_y < 1.0e-9:
            raise ValueError(f"{who}: a tendon on a vertical reference line is not followed")
        ey = [c / n_y for c in ey]
        ez = [ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2], ex[0] * ey[1] - ex[1] * ey[0]]
        if tp.line.turn:
            c_, s_ = math.cos(math.radians(tp.line.turn)), math.sin(math.radians(tp.line.turn))
            ey, ez = [c_ * ey[i] + s_ * ez[i] for i in range(3)], [-s_ * ey[i] + c_ * ez[i] for i in range(3)]
        off_y, off_z = float(tp.offset[0]), float(tp.offset[1])
        line_y = _rounded([tuple(p) for p in tp.profile_y]) if tp.profile_y else [(0.0, 0.0)]
        line_z = _rounded([tuple(p) for p in tp.profile_z])
        x0, x1 = line_z[0][0], line_z[-1][0]

        def at(x):
            yy = float(np.interp(x, [q[0] for q in line_y], [q[1] for q in line_y])) + off_y
            zz = float(np.interp(x, [q[0] for q in line_z], [q[1] for q in line_z])) + off_z
            return tuple(origin[i] + x * ex[i] + yy * ey[i] + zz * ez[i] for i in range(3))
        # where it is joined to the girder: its nodes along its line, and its anchors
        own = sorted(int(n) for n in np.asarray(mesh.regions[tp.nodes]).ravel())
        if not own:
            raise ValueError(f"{who}: its group {tp.nodes!r} has no node")
        along = {n: sum((float(xyz[n][i]) - origin[i]) * ex[i] for i in range(3)) for n in own}
        span_ = max(along.values()) - min(along.values())
        tol = 1.0e-3 * max(span_, 1.0 / L)
        stops = [(along[n], n) for n in own if x0 + tol < along[n] < x1 - tol]
        for xa in (x0, x1):
            near_ = min(own, key=lambda n: abs(along[n] - xa))
            stops.append((xa, near_))
        stops.sort()
        stops = [q for i, q in enumerate(stops) if i == 0 or q[0] - stops[i - 1][0] > tol]
        if len(stops) < 2 or len({n for _x, n in stops}) < 2:
            raise ValueError(f"{who}: it reaches only one node of its group {tp.nodes!r}")
        own_set = set(own)
        inside = [k for k in range(conn.shape[0]) if int(conn[k, 0]) in own_set and int(conn[k, 1]) in own_set]
        reach_m = max([abs(along[int(conn[k, 0])] - along[int(conn[k, 1])]) for k in inside] or [span_])
        if any(abs(xa - along[n]) > 0.5 * reach_m for xa, n in stops):
            warn(who, "anchored beyond its members (joined to the nearest node by a long arm)")
        side = max(math.sqrt(max(sum((float(xyz[n][i]) - origin[i]) ** 2 for i in range(3)) - along[n] ** 2, 0.0))
                   for n in own)
        if side > 0.02 * max(span_, 1.0 / L) + 0.5 * max(abs(v) for _x, v, _r in list(tp.profile_y) + list(tp.profile_z)) + 0.05 / L:
            warn(who, "on members off its straight reference line (a curved girder: the tendon is taken straight)")
        # the force along it: jacking, friction, draw-in
        xs = sorted(set([q[0] for q in line_z] + [q[0] for q in line_y if x0 <= q[0] <= x1] + [q[0] for q in stops]
                        + list(np.linspace(x0, x1, 200))))
        # (points closer than a micrometre are one: the turn between two such chords is noise, up to pi --
        # it took half of a 22 m tendon's force in friction)
        xs = [x for i, x in enumerate(xs) if i == 0 or x - xs[i - 1] > 1.0e-6 / L]
        pts = np.array([at(x) for x in xs])
        seg = np.diff(pts, axis=0)
        ds = np.linalg.norm(seg, axis=1)
        s_cum = np.concatenate([[0.0], np.cumsum(ds)])
        u_ = seg / np.maximum(ds, 1.0e-300)[:, None]
        turn_ = np.arccos(np.clip(np.sum(u_[1:] * u_[:-1], axis=1), -1.0, 1.0))
        th_cum = np.concatenate([[0.0], np.cumsum(np.concatenate([[0.0], turn_]))])[: len(xs)]
        area_t = sec.geometry()["area"] * tp.count
        e_t = float(materials[mat]["E"])
        if tp.stress.stress is not None:
            p_b, p_e = tp.stress.stress[0] * area_t, tp.stress.stress[1] * area_t
        else:
            p_b, p_e = float(tp.stress.force[0]), float(tp.stress.force[1])
        mu, kw = tp.friction.mu, tp.friction.k
        total_s, total_t = s_cum[-1], th_cum[-1]
        ahead = p_b * np.exp(-(mu * th_cum + kw * s_cum))
        back = p_e * np.exp(-(mu * (total_t - th_cum) + kw * (total_s - s_cum)))
        slip_b, slip_e, ea_t = tp.draw_in.begin, tp.draw_in.end, e_t * area_t
        if tp.stress.ends == "begin":
            force_ = _drawn_in(s_cum, ahead, slip_b, ea_t)
        elif tp.stress.ends == "end":
            force_ = _drawn_in(total_s - s_cum[::-1], back[::-1], slip_e, ea_t)[::-1]
        else:
            # JACKED AT BOTH ENDS: both anchorages draw in on
            # the force both jacks left (see `_drawn_in_both`)
            force_ = _drawn_in_both(s_cum, np.maximum(ahead, back), slip_b, slip_e, ea_t)
        # its materials: the steel without weight (the girder's weight has
        # the duct in it), and with its relaxation from each bar's own
        # stress after anchoring, bars of a final loss
        # within 1e-4 sharing one
        mat_t = f"{mat}_tendon"
        new_material(mat_t, dict(materials[mat], rho=light))
        mat_arm = f"{mat}_tendon_arm"
        new_material(mat_arm, {"type": "linear_elastic", "E": e_t, "nu": 0.3, "rho": light})
        rel = tp.relaxation
        above = False
        # its nodes
        t_nodes = []
        for xa, n in stops:
            p_ = list(at(xa))
            while key(p_) in place_of:
                p_ = [p_[i] + 1.0e-3 / L * ey[i] for i in range(3)]      # (another node there: a millimetre aside)
            place_of[key(p_)] = True
            new_nodes.append(tuple(p_))
            t_nodes.append(len(new_nodes) - 1)
        # its bars, each jacked to the force at its middle
        if stress_step is None:
            # when the last of the members it runs along is built (its
            # group's members, both ends in the group)
            stress_step = int(max([el_step[k] for k in inside] or [built[n] for _x, n in stops]))
        bars_out = []
        ea_max = 0.0
        for i in range(len(stops) - 1):
            xm = 0.5 * (stops[i][0] + stops[i + 1][0])
            sm = float(np.interp(xm, xs, s_cum))
            pm = float(np.interp(sm, s_cum, force_))
            m_name = mat_t
            if rel is not None:
                fin = jtg3362.relaxation_final(pm / area_t, rel.fpk, rel.zeta, 0.9 if rel.overstressed else 1.0)
                if pm / area_t > rel.fpk:
                    above = True
                    fin = 0.0
                b_ = int(round(fin * 1.0e4))
                if b_ > 0:
                    m_name = f"{mat_t}_r{b_}"
                    new_material(m_name, dict(materials[mat_t], relaxation=jtg3362.relaxation_curve_of(b_ * 1.0e-4)))
            sname = f"tendon_{area_t:.12g}_{m_name}"
            if sname in taken_sections:
                raise ValueError(f"{who}: a section named {sname!r} is already in `sections`")
            sections.setdefault(sname, {"type": "bar", "material": m_name, "shape": "general", "area": area_t})
            region = f"{tp.name}/bar{i + 1}"
            lines.append((region, [(-(t_nodes[i] + 1), -(t_nodes[i + 1] + 1))]))
            assign.append({"region": region, "section": sname, "jack": pm})
            a_, b_xyz = new_nodes[t_nodes[i]], new_nodes[t_nodes[i + 1]]
            ea_max = max(ea_max, e_t * area_t / max(math.dist(a_, b_xyz), 1.0e-6))
            bars_out.append((region, pm, stress_step))     # (as assigned: the result's tendon table reads it)
        if above:
            warn(who, "stressed above its fpk somewhere (its relaxation there not followed)")
        # THE ARM AND THE TENDON'S NODE ARE BUILT WITH THE GIRDER AT IT, THE
        # BAR WHEN IT IS STRESSED: a tendon's node born later, at its design
        # place, under a girder that has moved, is set off by the girder's
        # deflection (found in review: a cantilever's tendon 26% stronger).
        # A TENDON'S ARM IS A CANTILEVER: the tendon's node
        # turns freely on its bars, so its arm bends as 3 E I / r^3; four
        # times the bars' E A / L, `arm_factor` times over, along and across
        arms_out = []
        along_k = 4.0 * ea_max
        for j, ((xa, n), c) in enumerate(zip(stops, t_nodes)):
            reach_ = math.dist(tuple(float(v) for v in xyz[n]), new_nodes[c])
            i_arm = tp.arm_factor * max(0.0 * reach_, along_k * reach_ ** 3 / 12.0) / e_t
            sname = f"{tp.name}/arm{j + 1}"
            if sname in sections:
                raise ValueError(f"{who}: a section named {sname!r} is already in `sections`")
            sections[sname] = {"type": "beam", "material": mat_arm, "shape": "general",
                               "area": tp.arm_factor * along_k * reach_ / e_t, "iy": i_arm, "iz": i_arm, "j": 2.0 * i_arm}
            region = sname
            lines.append((region, [(int(tags[n]), -(c + 1))]))
            assign.append({"region": region, "section": sname,
                           "orientation": list(free_axis(tuple(float(v) for v in xyz[n]), new_nodes[c]))})
            arms_out.append((region, int(min(built[n], stress_step))))
        # their steps
        for region, k in arms_out + [(r, k) for r, _p, k in bars_out]:
            if k > 0:
                steps[k].setdefault("activate", []).append(region)
        report[tp.name] = {"step": stress_step, "count": tp.count, "area": area_t,
                           "jacking": [p_b, p_e],
                           "segments": [(r, p) for r, p, _k in bars_out]}
    if steps:
        d["steps"] = steps
    overlay = {"nodes": np.asarray(new_nodes, float).reshape(-1, 3), "lines": lines}
    return d, overlay, report
