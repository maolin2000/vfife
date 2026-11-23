"""Parametric bridge sections (1.2): a box girder and a T girder from the
dimensions on the drawing, and bars laid in layers.

One outline serves both kinds of member. A `fiber` section gets the outline
as quadrilateral patches (the `shape: patches` the solver already reads,
v0.3); a `beam` section gets the area, second moments, torsion
constant and shear areas integrated from the very same polygons. So a beam
model and a fibre model of one girder agree on A, Iy and Iz exactly, and the
fibre grid's own discretisation error is the only difference between them.

AXES AND ORIGIN
    y across the deck, z up, as every section in this library. The outline
    is drawn with z = 0 at the top of the deck and then moved so that the
    section's origin -- the point the member line passes through -- is its
    centroid (the default), the top of the deck or the bottom of the girder
    (`origin`, fibre sections only: a beam's properties are about its
    centroid). Bars, bar layers and tendons are written in these shifted
    axes; a bar layer's `z_from: top | bottom` measures from a face instead.

BOX GIRDER (`shape: box_girder`)
    height             H, top of deck to bottom of the bottom slab
    top_width          the whole deck, tip to tip
    bottom_width       the bottom slab, outside face to outside face
    cantilever         from a tip to the outer face of the web, measured at
                       the top of the deck
    top_thickness      the top slab between the webs
    bottom_thickness   the bottom slab
    web_thickness      each web, measured HORIZONTALLY (what a section
                       drawing dimensions on an inclined web)
    cantilever_thickness   [at the web, at the tip], or one value;
                       default top_thickness
    top_haunch, bottom_haunch   [width, height] of the fillet in each inside
                       corner of the cell(s); default none
    cells              1 (default) or more: interior webs, vertical,
                       spaced evenly between the outer webs' centre lines
    The outer webs run straight from the bottom slab's corners to the
    points `cantilever` in from the tips, so `bottom_width` smaller than
    `top_width - 2 cantilever` makes them lean out.

T GIRDER (`shape: t_girder`)
    height, top_width, top_thickness, web_thickness as above (the flange
    is the top slab, `top_thickness` its thickness at the web);
    `cantilever_thickness: [top_thickness, tip]` tapers it;
    `bottom_width` + `bottom_thickness` add a bottom flange (an I or a bulb-T).

FIBRES
    Each wall is divided into `layers` fibres through its thickness
    (default 3) and into pieces no longer than `fiber_size` along it
    (default height / 8): about two hundred fibres for a single-cell box,
    the count the Hangzhou Bay girder was run with (case_24d, 166).

TORSION AND SHEAR (beam sections; a fibre section gets the same `j`)
    A box: Bredt on the wall centre lines, a multi-cell box by the cells'
    shear-flow equations, plus int t^3 dy / 3 for the cantilevers (a few
    percent low for thick walls). A T: Saint-Venant's problem on its own
    outline (`saint_venant_j`); the sum of its Roark rectangles was 11-26%
    low (review, 1.2). Shear areas: `az` the
    webs' area times cos^2 of their lean, `ay` the slabs' area (5/6 of the
    flanges' for a T). These are the hand-check values, not exact for thick
    walls; give `j`, `ay` or `az` to override any of them.
"""
from __future__ import annotations

import math

import numpy as np
from typing import Any, Dict, List, Optional, Sequence

GIRDER_SHAPES = ("box_girder", "t_girder")

#: the keys a girder shape reads, beside `height` and the overrides
GIRDER_KEYS = {
    "box_girder": {"height", "top_width", "bottom_width", "cantilever", "top_thickness",
                   "bottom_thickness", "web_thickness", "cantilever_thickness",
                   "top_haunch", "bottom_haunch", "cells"},
    "t_girder": {"height", "top_width", "top_thickness", "web_thickness",
                 "cantilever_thickness", "bottom_width", "bottom_thickness"},
}
FIBRE_KEYS = {"layers", "fiber_size", "origin"}


# --------------------------------------------------------------------------- #
#  polygons
# --------------------------------------------------------------------------- #

def _poly_integrals(pts: Sequence[Sequence[float]]):
    """(A, Sy, Sz, Iyy, Izz, Iyz) of a simple polygon, either winding, about
    the axes' origin: Sy = int z dA, Sz = int y dA, Iyy = int z^2 dA,
    Izz = int y^2 dA."""
    n = len(pts)
    a = sy = sz = iyy = izz = iyz = 0.0
    for i in range(n):
        y0, z0 = pts[i]
        y1, z1 = pts[(i + 1) % n]
        c = y0 * z1 - y1 * z0
        a += c
        sz += (y0 + y1) * c
        sy += (z0 + z1) * c
        izz += (y0 * y0 + y0 * y1 + y1 * y1) * c
        iyy += (z0 * z0 + z0 * z1 + z1 * z1) * c
        iyz += (y0 * z1 + 2 * y0 * z0 + 2 * y1 * z1 + y1 * z0) * c
    s = 1.0 if a >= 0 else -1.0
    return (s * a / 2, s * sy / 6, s * sz / 6, s * iyy / 12, s * izz / 12, s * iyz / 24)


def _lerp(p, q, s):
    return (p[0] + (q[0] - p[0]) * s, p[1] + (q[1] - p[1]) * s)


def _need(spec, shape, *keys):
    missing = [k for k in keys if spec.get(k) is None]
    if missing:
        raise ValueError(f"a {shape} section needs " + ", ".join(missing))
    return [float(spec[k]) for k in keys]


def _pair(v, default, what):
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v), float(v)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return float(v[0]), float(v[1])
    raise ValueError(f"{what} is one value or two [at the web, at the tip]")


def _fillet(v, what):
    if v is None:
        return None
    if not isinstance(v, (list, tuple)) or len(v) != 2 or min(float(x) for x in v) <= 0:
        raise ValueError(f"{what} is [width, height], both positive")
    return float(v[0]), float(v[1])


# --------------------------------------------------------------------------- #
#  the outlines: a list of walls, each a quadrilateral (or a triangle, a
#  quad with its last two corners equal) ordered so that corner 0 -> 1 runs
#  ALONG the wall and 1 -> 2 THROUGH it
# --------------------------------------------------------------------------- #

class Wall:
    __slots__ = ("corners", "kind", "thickness")

    def __init__(self, corners, kind, thickness):
        self.corners = [tuple(map(float, c)) for c in corners]
        self.kind = kind              # slab | web | cantilever | fillet
        self.thickness = float(thickness)


def _box(spec) -> Dict[str, Any]:
    H, B, b, c, tt, tb, tw = _need(spec, "box_girder", "height", "top_width", "bottom_width",
                                   "cantilever", "top_thickness", "bottom_thickness",
                                   "web_thickness")
    if spec.get("cantilever_thickness") is not None and float(spec["cantilever"]) == 0.0:
        raise ValueError("box_girder: cantilever_thickness without cantilevers (cantilever: 0)")
    tr, tp = _pair(spec.get("cantilever_thickness"), (tt, tt), "cantilever_thickness")
    th = _fillet(spec.get("top_haunch"), "top_haunch")
    bh = _fillet(spec.get("bottom_haunch"), "bottom_haunch")
    cells = int(spec.get("cells") or 1)
    if cells < 1:
        raise ValueError("box_girder: cells is 1 or more")
    if c < 0:
        raise ValueError("box_girder: cantilever must not be negative")
    for name, v in (("height", H), ("top_width", B), ("bottom_width", b),
                    ("top_thickness", tt), ("bottom_thickness", tb), ("web_thickness", tw),
                    ("cantilever_thickness", min(tr, tp))):
        if v <= 0:
            raise ValueError(f"box_girder: {name} must be positive")
    if tt + tb >= H:
        raise ValueError(f"box_girder: the slabs ({tt:g} + {tb:g}) fill the height {H:g}")
    yt = B / 2 - c                        # web outer face at the top of the deck
    if yt <= tw:
        raise ValueError(f"box_girder: the cantilevers ({c:g} each) leave no room for the webs "
                         f"on a {B:g} deck")
    if b / 2 <= tw:
        raise ValueError(f"box_girder: bottom_width {b:g} leaves no room for two webs")
    if c > 0 and max(tr, tp) >= H - tb:
        raise ValueError("box_girder: a cantilever as thick as the girder is not a cantilever")

    def yo(z):                            # outer face of the right web; z from 0 (top) to -H
        return b / 2 + (yt - b / 2) * (z + H) / H

    def yi(z):                            # its inner face
        return yo(z) - tw

    z1, z2 = -tt, -H + tb                 # underside of the top slab, top of the bottom slab
    if c > 0 and max(yo(-tr), yo(-tp)) >= B / 2 - 1e-9 * B:
        raise ValueError("box_girder: the webs lean out past the cantilever tips")
    if min(yi(z1), yi(z2)) <= 0:
        raise ValueError("box_girder: the webs meet inside the girder")
    cos2 = 1.0 / (1.0 + ((yt - b / 2) / H) ** 2)
    walls: List[Wall] = []
    # top slab between the outer web faces (edges follow the webs' lean)
    walls.append(Wall([(-yo(0), 0), (yo(0), 0), (yo(z1), z1), (-yo(z1), z1)], "slab", tt))
    # bottom slab
    walls.append(Wall([(-yo(-H), -H), (yo(-H), -H), (yo(z2), z2), (-yo(z2), z2)], "slab", tb))
    for s in (1.0, -1.0):
        # web, from the top slab's underside to the bottom slab's top, along z
        walls.append(Wall([(s * yo(z2), z2), (s * yo(z1), z1), (s * yi(z1), z1), (s * yi(z2), z2)],
                          "web", tw))
        # cantilever: its inner edge on the web's outer face line (none on a
        # box without them: a box pier, a steel box)
        if c > 0:
            walls.append(Wall([(s * yo(0), 0), (s * B / 2, 0), (s * B / 2, -tp), (s * yo(-tr), -tr)],
                              "cantilever", 0.5 * (tr + tp)))
    # interior webs: vertical, evenly between the outer webs' centre lines
    zm = 0.5 * (z1 + z2)
    ym = yo(zm) - tw / 2
    inner = [-ym + 2 * ym * k / cells for k in range(1, cells)]
    for y in inner:
        # clear of each other, and of the outer webs' inner faces at BOTH
        # slabs: a steep web closes on an interior one near one of them
        if (any(abs(y - q) < tw for q in inner if q != y)
                or min(yi(z1), yi(z2)) - (abs(y) + tw / 2) <= 0):
            raise ValueError(f"box_girder: {cells} cells leave no room between the webs")
        walls.append(Wall([(y + tw / 2, z2), (y + tw / 2, z1), (y - tw / 2, z1), (y - tw / 2, z2)],
                          "web", tw))
    # fillets in each inside corner
    faces = [(-1.0, None), (1.0, None)] + [(0.0, y) for y in inner]
    for fil, zs, sgn in ((th, z1, -1.0), (bh, z2, 1.0)):
        if fil is None:
            continue
        w, h = fil
        if (th[1] if th else 0.0) + (bh[1] if bh else 0.0) >= z1 - z2:
            raise ValueError("box_girder: the haunches meet across the cell")
        for side, y in faces:
            if y is None:                  # an outer web's inner face, the corner inboard of it
                p0 = (side * yi(zs), zs)
                p1 = (side * (yi(zs) - w), zs)
                p2 = (side * yi(zs + sgn * h), zs + sgn * h)
                walls.append(Wall([p0, p1, p2, p2], "fillet", h))
            else:
                for d in (1.0, -1.0):
                    e = y + d * tw / 2
                    walls.append(Wall([(e, zs), (e + d * w, zs), (e, zs + sgn * h), (e, zs + sgn * h)],
                                      "fillet", h))
    # each cell stays open after its fillets: its clear width at each slab
    # face takes a fillet from each side
    for fil, zs in ((th, z1), (bh, z2)):
        if fil is None:
            continue
        edges = [-yi(zs)] + [e for y in inner for e in (y - tw / 2, y + tw / 2)] + [yi(zs)]
        for k in range(cells):
            if edges[2 * k + 1] - edges[2 * k] <= 2 * fil[0]:
                raise ValueError("box_girder: the haunches close the cell")
    # TORSION: Bredt on the centre lines, cells by their shear flows
    zt_m, zb_m = -tt / 2, -H + tb / 2
    def ymid(z):
        return yo(z) - tw / 2
    lines = [lambda z: -ymid(z)] + [(lambda yy: (lambda z: yy))(y) for y in inner] + [ymid]
    hm = zt_m - zb_m
    A, top_len, bot_len, side = [], [], [], []
    for k in range(cells):
        L, R = lines[k], lines[k + 1]
        wt, wb = R(zt_m) - L(zt_m), R(zb_m) - L(zb_m)
        A.append(0.5 * (wt + wb) * hm)
        top_len.append(wt); bot_len.append(wb)
    outer_web = hm / (tw * cos2)          # ds/t of an inclined web: (h/cos)/(tw cos)
    inner_web = hm / tw
    n = cells
    D = [[0.0] * n for _ in range(n)]
    for k in range(n):
        D[k][k] = top_len[k] / tt + bot_len[k] / tb
        D[k][k] += outer_web if k == 0 else inner_web
        D[k][k] += outer_web if k == n - 1 else inner_web
        if k > 0:
            D[k][k - 1] = D[k - 1][k] = -inner_web
    q = _solve(D, [2 * a for a in A])
    j = sum(2 * a * qq for a, qq in zip(A, q))
    j += 2 * (B / 2 - yo(0)) * _t3(tr, tp) / 3 if c > 0 else 0.0
    web_area = sum(_poly_integrals(w.corners)[0] for w in walls if w.kind == "web")
    outer_web_area = 2 * _poly_integrals(walls[2].corners)[0]
    slab_area = sum(_poly_integrals(w.corners)[0] for w in walls if w.kind == "slab")
    # the cells, for the tendons' outline check: an external tendon runs in one
    edges = lambda z: [-yi(z)] + [e for y in inner for e in (y - tw / 2, y + tw / 2)] + [yi(z)]
    lo, hi = edges(z2), edges(z1)
    voids = [Wall([(lo[2 * k], z2), (lo[2 * k + 1], z2), (hi[2 * k + 1], z1), (hi[2 * k], z1)],
                  "cell", 0.0) for k in range(cells)]
    return {"walls": walls, "j": j, "voids": voids,
            "az": (web_area - outer_web_area) + outer_web_area * cos2,
            "ay": slab_area, "width": B, "height": H}


def _t3(t0, t1):
    """The mean of t^3 over a thickness tapering linearly from t0 to t1."""
    if abs(t0 - t1) <= 1e-12 * max(t0, t1):
        return t0 ** 3
    return (t0 ** 4 - t1 ** 4) / (4.0 * (t0 - t1))


def _t(spec) -> Dict[str, Any]:
    H, B, tf, tw = _need(spec, "t_girder", "height", "top_width", "top_thickness", "web_thickness")
    tr, tp = _pair(spec.get("cantilever_thickness"), (tf, tf), "cantilever_thickness")
    if abs(tr - tf) > 1e-9 * tf:
        # the flange IS top_thickness thick at the web: a root of its own
        # either overrode it (thicker) or left a step at the web's face
        # (thinner) -- found in review (1.2)
        raise ValueError(f"t_girder: the flange is top_thickness ({tf:g}) thick at the web; write "
                         f"cantilever_thickness as [{tf:g}, <tip>]")
    bb, tb = spec.get("bottom_width"), spec.get("bottom_thickness")
    if (bb is None) != (tb is None):
        raise ValueError("t_girder: a bottom flange needs both bottom_width and bottom_thickness")
    bb = None if bb is None else float(bb)
    tb = 0.0 if tb is None else float(tb)
    for name, v in (("height", H), ("top_width", B), ("top_thickness", tf),
                    ("web_thickness", tw), ("cantilever_thickness", tp)):
        if v <= 0:
            raise ValueError(f"t_girder: {name} must be positive")
    if B <= tw:
        raise ValueError("t_girder: top_width must be wider than the web")
    if bb is not None and (bb <= tw or tb <= 0):
        raise ValueError("t_girder: the bottom flange must be wider than the web and of positive thickness")
    if tf + tb >= H:
        raise ValueError(f"t_girder: the flanges fill the height {H:g}")
    walls: List[Wall] = []
    # the flange over the web, and the two outstands (tapered when asked)
    walls.append(Wall([(-tw / 2, 0), (tw / 2, 0), (tw / 2, -tf), (-tw / 2, -tf)], "slab", tf))
    for s in (1.0, -1.0):
        walls.append(Wall([(s * tw / 2, 0), (s * B / 2, 0), (s * B / 2, -tp), (s * tw / 2, -tf)],
                          "cantilever", 0.5 * (tf + tp)))
    zb = -H + tb
    walls.append(Wall([(tw / 2, zb), (tw / 2, -tf), (-tw / 2, -tf), (-tw / 2, zb)], "web", tw))
    if bb is not None:
        walls.append(Wall([(-bb / 2, -H), (bb / 2, -H), (bb / 2, zb), (-bb / 2, zb)], "slab", tb))
    flange = B * 0.5 * (tf + tp) + (bb * tb if bb is not None else 0.0)
    # TORSION: Saint-Venant's problem on the outline itself. The sum of the
    # rectangles' b t^3 / 3 was 11-26% low against it (review, 1.2): it
    # leaves out the junctions, where a T keeps most of its stiffness
    return {"walls": walls, "j": saint_venant_j(walls), "az": tw * (H - tf - tb),
            "ay": 5.0 / 6.0 * flange, "width": B, "height": H}


def _inside(walls, P, tol=0.0):
    """Points P (n, 2) inside any of the (convex) walls, or within tol of one."""
    ins = np.zeros(len(P), bool)
    for w in walls:
        c = np.asarray(w.corners if w.corners[2] != w.corners[3] else w.corners[:3], float)
        m = len(c)
        sgn = np.sign(sum(c[i, 0] * c[(i + 1) % m, 1] - c[(i + 1) % m, 0] * c[i, 1] for i in range(m)))
        ok = np.ones(len(P), bool)
        for i in range(m):
            e = c[(i + 1) % m] - c[i]
            L = float(np.hypot(*e)) or 1.0
            cross = (e[0] * (P[:, 1] - c[i, 1]) - e[1] * (P[:, 0] - c[i, 0])) * sgn / L
            ok &= cross >= -tol
        ins |= ok
    return ins


_J_CACHE: Dict[Any, float] = {}


def saint_venant_j(walls, cells_per_thickness: int = 24) -> float:
    key = (tuple(tuple(w.corners) for w in walls), cells_per_thickness)
    if key not in _J_CACHE:
        if len(_J_CACHE) > 256:
            _J_CACHE.clear()
        _J_CACHE[key] = _saint_venant_j(walls, cells_per_thickness)
    return _J_CACHE[key]


def _saint_venant_j(walls, cells_per_thickness: int = 24) -> float:
    """Saint-Venant's torsion constant of a simply connected outline (a T,
    an I): Prandtl's stress function, laplace(phi) = -2 with phi = 0 on the
    boundary, J = 2 int phi, by finite differences on square cells -- the
    cells whose centres lie inside, a ghost of opposite sign across each
    face that borders the outside, so the boundary sits on the faces.
    Second order where the outline runs along the grid, first order along a
    taper. Against the exact series for a 0.3 x 1.3 rectangle and on a T
    refined to 100 cells a thickness: within 1% at the default 24."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    pts = np.array([c for w in walls for c in w.corners])
    tmin = min(w.thickness for w in walls if w.kind != "fillet")
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    h = tmin / cells_per_thickness
    h = max(h, float(np.sqrt(np.prod(hi - lo) / 4.0e5)))      # at most ~0.4 million cells
    n = np.maximum(np.ceil((hi - lo) / h).astype(int), 1)
    ys = lo[0] + (np.arange(n[0]) + 0.5) * h
    zs = lo[1] + (np.arange(n[1]) + 0.5) * h
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    ins = _inside(walls, np.c_[Y.ravel(), Z.ravel()]).reshape(Y.shape)
    idx = -np.ones(ins.shape, int)
    idx[ins] = np.arange(int(ins.sum()))
    N = int(ins.sum())
    pad = np.pad(ins, 1)
    I, J = np.nonzero(ins)
    k = idx[I, J]
    diag = np.zeros(N)
    rows, cols = [k], [k]
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = pad[I + 1 + di, J + 1 + dj]
        diag += np.where(nb, 1.0, 2.0)
        rows.append(k[nb]); cols.append(idx[I[nb] + di, J[nb] + dj])
    vals = [diag] + [-np.ones(len(r)) for r in rows[1:]]
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N, N))
    phi = spl.spsolve(A, np.full(N, 2.0 * h * h))
    return float(2.0 * phi.sum() * h * h)


def _solve(D, r):
    n = len(r)
    M = [row[:] + [r[i]] for i, row in enumerate(D)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        M[k], M[p] = M[p], M[k]
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            for jj in range(k, n + 1):
                M[i][jj] -= f * M[k][jj]
    x = [0.0] * n
    for k in range(n - 1, -1, -1):
        x[k] = (M[k][n] - sum(M[k][jj] * x[jj] for jj in range(k + 1, n))) / M[k][k]
    return x


_OUTLINES = {"box_girder": _box, "t_girder": _t}


# --------------------------------------------------------------------------- #
#  what the rest of the library reads
# --------------------------------------------------------------------------- #

def girder(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The girder's outline and properties, about its centroid.

    Returns {walls, area, centroid (y, z) in the drawing axes (z = 0 at the
    top of the deck), iy, iz, j, ay, az, top, bottom (z of the faces about
    the centroid), width, height}."""
    shape = spec.get("shape")
    if spec.get("height") is None and spec.get("h") is not None:       # sections.py's letter
        spec = {**spec, "height": spec["h"]}
    out = _OUTLINES[shape](spec)
    A = Sy = Sz = Iyy = Izz = 0.0
    for w in out["walls"]:
        pts = w.corners if w.corners[2] != w.corners[3] else w.corners[:3]
        a, sy, sz, iyy, izz, _ = _poly_integrals(pts)
        A += a; Sy += sy; Sz += sz; Iyy += iyy; Izz += izz
    yc, zc = Sz / A, Sy / A
    out.update(area=A, centroid=(yc, zc), iy=Iyy - A * zc * zc, iz=Izz - A * yc * yc,
               top=-zc, bottom=-float(spec["height"]) - zc)
    return out


def origin_shift(spec, g) -> float:
    """dz added to the drawing's z to put the chosen origin at z = 0."""
    where = spec.get("origin") or "centroid"
    return {"centroid": -g["centroid"][1], "top": 0.0, "bottom": float(spec["height"])}[where]


def patches(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The fibre patches of a girder in the section's own axes, and its faces:
    {patches, top, bottom, j, section_size}."""
    g = girder(spec)
    dz = origin_shift(spec, g)
    layers = int(spec.get("layers") or 3)
    size = float(spec.get("fiber_size") or float(spec["height"]) / 8.0)
    if layers < 1 or size <= 0:
        raise ValueError("layers is 1 or more and fiber_size positive")
    out = []
    for w in g["walls"]:
        c = [(y, z + dz) for y, z in w.corners]
        along = max(math.dist(c[0], c[1]), math.dist(c[3], c[2]))
        if w.kind == "fillet":
            ny, nz = 2, 2
        else:
            ny, nz = max(1, math.ceil(along / size - 1e-9)), layers
        out.append({"corners": [list(p) for p in c], "ny": ny, "nz": nz})
    shifted = lambda ws: [[[y, z + dz] for y, z in w.corners] for w in ws]
    return {"patches": out, "top": dz, "bottom": -float(spec["height"]) + dz, "j": g["j"],
            "walls": shifted(g["walls"]), "voids": shifted(g.get("voids") or []),
            "section_size": [g["width"], g["height"]]}


def contains(quads, y, z, tol=1e-9):
    """Points (y, z) inside the union of `quads` (each four [y, z], a
    triangle repeating its last corner), to within `tol` of the outline's
    size. A girder's outline is exact, so the 1% a hand-drawn patch section
    is allowed (fiber_section.PATCH_OUTLINE_TOLERANCE) does not apply: 0.12 m
    on a 12 m deck let a row of bars sit above it (review, 1.2)."""
    P = np.c_[np.atleast_1d(np.asarray(y, float)), np.atleast_1d(np.asarray(z, float))]
    walls = [Wall(q, "", 0.0) for q in quads]
    span = float(np.ptp(np.array([c for w in walls for c in w.corners]), axis=0).max())
    return _inside(walls, P, tol * span)


def properties(spec: Dict[str, Any]) -> Dict[str, float]:
    """A beam section's integrated properties (sections.properties)."""
    g = girder(spec)
    return {"area": g["area"], "iy": g["iy"], "iz": g["iz"], "j": g["j"],
            "ay": g["ay"], "az": g["az"], "perimeter": None}


# --------------------------------------------------------------------------- #
#  bars in layers
# --------------------------------------------------------------------------- #

def layer_bars(layer: Dict[str, Any], top: Optional[float], bottom: Optional[float],
               where: str) -> List[Dict[str, Any]]:
    """One straight layer of identical bars -> [{y, z, section}].

    {section, from: [y, z], to: [y, z], count: n | spacing: s, z_from}
    `count` bars from `from` to `to`, both ends included (one bar sits at
    the middle); `spacing` puts as many bars as fit at that spacing or
    closer, both ends included -- d16 @ 150 over 1.0 m is 8 bars at 143 mm.
    `z_from: top` measures both z downward from the section's top face,
    `bottom` upward from its bottom face, `origin` (default) is the section
    axes."""
    allowed = {"section", "from", "to", "count", "spacing", "z_from"}
    bad = sorted(set(layer) - allowed)
    if bad:
        raise ValueError(f"{where} does not read {', '.join('`' + k + '`' for k in bad)} "
                         "(a layer is {section, from: [y, z], to: [y, z], count | spacing, z_from?})")
    for k in ("section", "from", "to"):
        if layer.get(k) is None:
            raise ValueError(f"{where} needs `{k}`")
    p, q = layer["from"], layer["to"]
    for nm, v in (("from", p), ("to", q)):
        if not isinstance(v, (list, tuple)) or len(v) != 2 or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            raise ValueError(f"{where}: `{nm}` is [y, z]")
    if (layer.get("count") is None) == (layer.get("spacing") is None):
        raise ValueError(f"{where} needs one of `count` or `spacing`")
    zf = layer.get("z_from") or "origin"
    if zf not in ("origin", "top", "bottom"):
        raise ValueError(f"{where}: z_from is origin, top or bottom")

    def place(z):
        if zf == "top":
            if top is None:
                raise ValueError(f"{where}: this section has no top face to measure from")
            return top - float(z)
        if zf == "bottom":
            if bottom is None:
                raise ValueError(f"{where}: this section has no bottom face to measure from")
            return bottom + float(z)
        return float(z)
    a = (float(p[0]), place(p[1]))
    b = (float(q[0]), place(q[1]))
    L = math.dist(a, b)
    if layer.get("count") is not None:
        if isinstance(layer["count"], bool) or not isinstance(layer["count"], (int, float)):
            raise ValueError(f"{where}: count is a whole number, 1 or more")
        n = int(layer["count"])
        if n < 1 or n != layer["count"]:
            raise ValueError(f"{where}: count is a whole number, 1 or more")
    else:
        s = float(layer["spacing"])
        if s <= 0:
            raise ValueError(f"{where}: spacing must be positive")
        if L <= 0:
            raise ValueError(f"{where}: a layer laid by spacing needs `from` and `to` apart")
        n = int(math.ceil(L / s - 1e-9)) + 1
    if n == 1:
        pts = [_lerp(a, b, 0.5)]
    else:
        if L <= 0:
            raise ValueError(f"{where}: {n} bars at one point -- use `count` on a single bar instead")
        pts = [_lerp(a, b, k / (n - 1)) for k in range(n)]
    return [{"y": y, "z": z, "section": layer["section"]} for y, z in pts]
