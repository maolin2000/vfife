"""Sweep the fibre sections along the deformed members: a surface to look at.

A fibre-beam result is nodes and lines, and a picture of nodes and lines
falling is a poor account of a girder breaking. Everything needed for a
better one is already in the model: every member has a section outline
(the rectangle, box or tube its fibres were laid in) and every particle
has an exact orientation, so the outline can be carried along
the deformed axis by the particles' triads -- the section at node n is
x_n + R_n (y e_y + z e_z) in the member's reference frame -- and the
faces between the two ends' outlines are a surface that bends, twists and,
once a member has split, falls in two pieces.

The faces are painted with what the section knows: with `fiber_stress`
and `fiber_damage` in the result (output.fields), every face takes the
stress and the damage class (0 intact, 1 cracked, 2 crushed, 3 yielded,
4 failed -- materials/base.damage_class) of the fibre nearest to it, so a
crack band, a crushed edge or a yielded tendon shows where it is on the
section; the members' actions and `fac` / `fractured` come along as cell
data too.

Output is a ParaView collection (.pvd) of one .vtu per frame, written with
meshio -- the surface changes shape every frame, and a collection of
files is the plain way to say so. The bars and tendons are a second
collection of the same kind. ParaView's Tube filter wants vtkPolyData and
refuses a .vtu with "Input must be vtkPolyData": tick "Render Lines As
Tubes" in the Display properties, or put an Extract Surface in front of
the Tube.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np

from . import api
from .io.writer import open_result
from .core.rotations import exp_map
from .core.structural import make_structural_solver


def section_outline(spec, n_round=None):
    """Closed loops [(y, z), ...] of one fibre-section spec, outer loop first.

    The edges are subdivided to the section's own fibre grid -- `ny` along
    the width, `nz` along the height, `nt` round a tube -- so that every
    face of the swept surface lies against one boundary fibre and a crack
    map painted face by face has the resolution the fibres have."""
    stype = spec.get("type", "rect")
    if stype == "rect":
        w = float(spec.get("width", 0.1)); h = float(spec.get("height", 0.2))
        return [_rect(w, h, int(spec.get("ny", 10)), int(spec.get("nz", 10)))]
    if stype == "box":
        w = float(spec["width"]); h = float(spec["height"]); t = float(spec["thickness"])
        ny, nz = int(spec.get("ny", 4)), int(spec.get("nz", 4))
        return [_rect(w, h, ny, nz), _rect(w - 2 * t, h - 2 * t, ny, nz)]
    if stype == "tube":
        d = float(spec["diameter"]); t = float(spec["thickness"])
        n = n_round or int(spec.get("nt", 20))
        return [_circle(d / 2, n), _circle(d / 2 - t, n)]
    if stype == "circle":
        n = n_round or int(spec.get("nt", 8))
        return [_circle(float(spec.get("diameter", 0.2)) / 2, n)]
    if stype == "patches":
        # every patch's own boundary: the surface mode draws the patches
        return [np.asarray(p["corners"], dtype=float) for p in spec["patches"]]
    raise ValueError(f"no outline for fiber_section.type {stype!r}")


def fiber_cells(spec):
    """One quadrilateral (4 corners in (y, z)) per concrete fibre of a
    fibre-section spec, in the order the section lays its fibres
    (elements/fiber_section.py): the cell each fibre stands for, so that
    the member can be drawn as a solid of fibre cells rather than as a
    shell. Ring cells (tube, circle) have their arcs drawn straight."""
    stype = spec.get("type", "rect")
    cells = []
    if stype == "rect":
        w = float(spec.get("width", 0.1)); h = float(spec.get("height", 0.2))
        ny, nz = int(spec.get("ny", 10)), int(spec.get("nz", 10))
        dy, dz = w / ny, h / nz
        for i in range(ny):                       # meshgrid(..., indexing="ij"): y outer
            y0 = -w / 2 + i * dy
            for j in range(nz):
                z0 = -h / 2 + j * dz
                cells.append([(y0, z0), (y0 + dy, z0), (y0 + dy, z0 + dz), (y0, z0 + dz)])
    elif stype == "box":
        w = float(spec["width"]); h = float(spec["height"]); t_all = float(spec["thickness"])
        nw, nh, nr = int(spec.get("ny", 4)), int(spec.get("nz", 4)), int(spec.get("nr", 1))
        t = t_all / nr
        for ir in range(nr):
            z_in = h / 2 - t_all + t * ir
            dy = w / nw
            for i in range(nw):
                y0 = -w / 2 + dy * i
                for sign in (+1.0, -1.0):
                    za, zb = sign * z_in, sign * (z_in + t)
                    cells.append([(y0, za), (y0 + dy, za), (y0 + dy, zb), (y0, zb)])
            y_in = w / 2 - t_all + t * ir
            inner_h = h - 2 * t_all
            dz = inner_h / nh
            for i in range(nh):
                z0 = -inner_h / 2 + dz * i
                for sign in (+1.0, -1.0):
                    ya, yb = sign * y_in, sign * (y_in + t)
                    cells.append([(ya, z0), (yb, z0), (yb, z0 + dz), (ya, z0 + dz)])
    elif stype == "patches":
        from .elements.fiber_section import patch_fibers
        return patch_fibers(spec["patches"])[3]
    elif stype in ("tube", "circle"):
        if stype == "tube":
            r_out = float(spec["diameter"]) / 2; r_in = r_out - float(spec["thickness"])
            nt, nr = int(spec.get("nt", 20)), int(spec.get("nr", 1))
            rings = [(r_in + (r_out - r_in) * ir / nr, r_in + (r_out - r_in) * (ir + 1) / nr, nt)
                     for ir in range(nr)]
        else:
            r_max = float(spec.get("diameter", 0.2)) / 2
            nr, nt = int(spec.get("nr", 5)), int(spec.get("nt", 8))
            rings = []
            for ir in range(nr):
                r0, r1 = r_max * ir / nr, r_max * (ir + 1) / nr
                r_mid = 0.5 * (r0 + r1)
                n_t = max(1, int(round(nt * r_mid / r_max))) if r_mid > 0 else 1
                rings.append((r0, r1, n_t))
        for r0, r1, n_t in rings:
            for it in range(n_t):
                a0, a1 = 2 * np.pi * it / n_t, 2 * np.pi * (it + 1) / n_t
                cells.append([(r0 * np.cos(a0), r0 * np.sin(a0)), (r1 * np.cos(a0), r1 * np.sin(a0)),
                              (r1 * np.cos(a1), r1 * np.sin(a1)), (r0 * np.cos(a1), r0 * np.sin(a1))])
    else:
        raise ValueError(f"no fibre cells for fiber_section.type {stype!r}")
    return np.asarray(cells, dtype=float)


def _rect(w, h, ny=1, nz=1):
    """The rectangle's perimeter, counter-clockwise from (-w/2, -h/2), with
    `ny` segments on the edges along y and `nz` on the edges along z."""
    ys = np.linspace(-w / 2, w / 2, ny + 1); zs = np.linspace(-h / 2, h / 2, nz + 1)
    pts = [(y, -h / 2) for y in ys[:-1]] + [(w / 2, z) for z in zs[:-1]]
    pts += [(y, h / 2) for y in ys[::-1][:-1]] + [(-w / 2, z) for z in zs[::-1][:-1]]
    return np.array(pts)


def _circle(r, n):
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


class Sweep:
    """The swept surface of one fibre-beam case; `frame()` builds it for a
    displacement / rotation / connectivity state."""

    def __init__(self, source):
        prep = api.prepare_case(source)
        problem = api._with_element(prep.case, prep.element, prep.case_path)
        sol = make_structural_solver(problem, prep.mesh, prep.material)
        if not hasattr(sol, "_section_specs"):
            raise ValueError("vfife sweep draws fibre-beam models (beam2_fiber); this case "
                             f"lowers to {sol.p.case.element.type!r}")
        self.sol = sol
        self.nodes0 = sol.mesh.nodes
        self.R0 = np.asarray(sol.R0)
        self._ends = None                     # lazily: see `_bar_ends`
        entries = sol._elem_spec.get("fiber_sections") or [
            {"fiber_section": sol._elem_spec.get("fiber_section") or {}}]
        self.loops = [section_outline(m["fiber_section"] or {}) for m in entries]
        self.section_of = (sol._section_of if sol._section_of is not None
                           else np.zeros(sol.nel, dtype=np.int64))
        # THE MEMBERS THAT HAVE A SECTION TO SWEEP. A fibre mesh can carry
        # `connector2` bearings since v0.4, and a connector has no
        # section: its row in `section_of` is a pad that is never read, so
        # every loop below would have drawn each bearing as a 0.5 m stub of
        # whatever section happened to sit in row 0 -- on case_24f, two
        # box-girder segments pointing down into the piers. It was the tenth
        # touch point of that stage and it was not on the list.
        #
        # ONE list, used by every loop here, because `paint`, `paint_along`
        # and `_face_columns` walk the cells in the same order the geometry
        # was built in: filtering in one place and not another would not drop
        # a member, it would shift every colour by one.
        self.members = np.flatnonzero(
            getattr(sol, "_is_fiber", np.ones(sol.nel, dtype=bool)))
        # the fibre cells of each section, checked against the section's own
        # fibre positions (a cell's centroid must be its fibre, ring cells
        # excepted -- their fibre sits at the RMS radius)
        self.cells = []
        for k, m in enumerate(entries):
            spec = m["fiber_section"] or {}
            cells = fiber_cells(spec)
            e0 = int(sol._rows[k][0]); n_c = sol._layouts[k].n_c
            if len(cells) != n_c:
                raise ValueError(f"fibre section {k}: {len(cells)} cells for {n_c} fibres")
            if spec.get("type", "rect") in ("rect", "box"):
                cen = cells.mean(axis=1)
                if not (np.allclose(cen[:, 0], sol.fiber_y[e0, :n_c], atol=1e-9)
                        and np.allclose(cen[:, 1], sol.fiber_z[e0, :n_c], atol=1e-9)):
                    raise ValueError(f"fibre section {k}: the cells do not sit on the fibres")
            self.cells.append(cells)
        # per section: the outline vertices (y, z) and, per face, the nearest
        # fibre column (for painting) and which loop the face belongs to
        self._verts, self._face_fiber = [], []
        for k, loops in enumerate(self.loops):
            e0 = int(sol._rows[k][0])
            fy, fz, fa = sol.fiber_y[e0], sol.fiber_z[e0], sol.fiber_area[e0]
            live = fa > 0
            verts, near = [], []
            for loop in loops:
                n = len(loop)
                for i in range(n):
                    a, b = loop[i], loop[(i + 1) % n]
                    c = 0.5 * (a + b)
                    d2 = (fy - c[0]) ** 2 + (fz - c[1]) ** 2
                    d2[~live] = np.inf
                    near.append(int(np.argmin(d2)))
                verts.extend(loop.tolist())
            self._verts.append(np.asarray(verts))
            self._face_fiber.append(np.asarray(near))

    def frame(self, u, rot, conn=None):
        """(points, quads, face_element) of the surface for nodal
        displacements `u` (nn, 3) and rotation vectors `rot` (nn, 3);
        `conn` is the members' connectivity for this frame (a split moves a
        member onto its spare particles), the mesh's when None."""
        conn = self.sol.mesh.elements if conn is None else np.asarray(conn, dtype=np.int64)
        x = self.nodes0 + u
        R = exp_map(np.asarray(rot, dtype=float))
        pts, quads, owner, blend = [], [], [], []
        for e in self.members:
            k = int(self.section_of[e])
            verts = self._verts[k]
            # section axes of the member's reference frame, rotated with each end
            ey, ez = self.R0[e, 1], self.R0[e, 2]
            local = verts[:, 0:1] * ey + verts[:, 1:2] * ez         # (nv, 3)
            base = len(pts)
            nv = len(verts)
            for end in (0, 1):
                n = int(conn[e, end])
                pts.extend((x[n] + local @ R[n].T).tolist())
                blend.extend([(int(conn[e, 0]), int(conn[e, 1]), float(end))] * nv)
            # faces between the two ends, loop by loop
            start = 0
            for loop in self.loops[k]:
                m = len(loop)
                for i in range(m):
                    a = start + i; b = start + (i + 1) % m
                    quads.append([base + a, base + b, base + nv + b, base + nv + a])
                    owner.append(e)
                start += m
        self._blend = np.asarray(blend, dtype=float) if blend else np.zeros((0, 3))
        return np.asarray(pts), np.asarray(quads, dtype=np.int64), np.asarray(owner, dtype=np.int64)

    @property
    def slices(self):
        """Where a member is cut along its length so that every integration
        station owns one slab: the midpoints between stations, plus the
        ends -- [0, 0.5, 1] for 2-point Gauss, [0, 0.25, 0.75, 1] for
        3-point Lobatto. One cell per fibre per station, painted with that
        station's own value, and nothing averaged away."""
        g = np.asarray(self.sol.gauss_s, dtype=float)
        return np.concatenate([[0.0], 0.5 * (g[:-1] + g[1:]), [1.0]])

    def along(self, t, values):
        """`values` (..., n_stations) interpolated along the member to the
        fractions `t`: linear between stations, linear extrapolation past
        the first and last (Gauss stations sit inside the member; Lobatto's
        are the ends themselves)."""
        g = np.asarray(self.sol.gauss_s, dtype=float)
        t = np.asarray(t, dtype=float)
        v = np.asarray(values, dtype=float)
        out = np.empty(v.shape[:-1] + t.shape)
        for i, ti in enumerate(t):
            if len(g) == 1:
                out[..., i] = v[..., 0]
            elif ti <= g[0]:
                out[..., i] = v[..., 0] + (v[..., 1] - v[..., 0]) * (ti - g[0]) / (g[1] - g[0])
            elif ti >= g[-1]:
                out[..., i] = v[..., -1] + (v[..., -1] - v[..., -2]) * (ti - g[-1]) / (g[-1] - g[-2])
            else:
                j = int(np.searchsorted(g, ti)) - 1
                w = (ti - g[j]) / (g[j + 1] - g[j])
                out[..., i] = v[..., j] * (1 - w) + v[..., j + 1] * w
        return out

    def solid_frame(self, u, rot, conn=None):
        """(points, hexahedra, cell_element, cell_column, cell_station):
        every concrete fibre of every member as one hexahedron per
        integration station (the slabs of `slices`), so the member is a
        solid that can be clipped and sliced, with one cell per fibre per
        station to paint with that station's own value. Interior slices
        interpolate the two end sections linearly."""
        sol = self.sol
        conn = sol.mesh.elements if conn is None else np.asarray(conn, dtype=np.int64)
        x = self.nodes0 + u
        R = exp_map(np.asarray(rot, dtype=float))
        ts = self.slices
        n_sl = len(ts)
        pts, hexa, owner, cols, stat, blend = [], [], [], [], [], []
        for e in self.members:
            k = int(self.section_of[e])
            cells = self.cells[k]                                   # (n_c, 4, 2)
            ey, ez = self.R0[e, 1], self.R0[e, 2]
            local = (cells[:, :, 0:1] * ey + cells[:, :, 1:2] * ez).reshape(-1, 3)   # (4 n_c, 3)
            n_c = len(cells)
            ni, nj = int(conn[e, 0]), int(conn[e, 1])
            ring_i = x[ni] + local @ R[ni].T
            ring_j = x[nj] + local @ R[nj].T
            base = len(pts)
            for t in ts:
                pts.extend(((1.0 - t) * ring_i + t * ring_j).tolist())
                blend.extend([(ni, nj, float(t))] * len(ring_i))
            for st in range(n_sl - 1):
                for f in range(n_c):
                    a = base + st * 4 * n_c + 4 * f; b = a + 4 * n_c
                    hexa.append([a, a + 1, a + 2, a + 3, b, b + 1, b + 2, b + 3])
                    owner.append(e); cols.append(f); stat.append(st)
        self._blend = np.asarray(blend, dtype=float) if blend else np.zeros((0, 3))
        return (np.asarray(pts), np.asarray(hexa, dtype=np.int64),
                np.asarray(owner, dtype=np.int64), np.asarray(cols, dtype=np.int64),
                np.asarray(stat, dtype=np.int64))

    def paint_solid(self, owner, cols, stat, fiber_stress=None, fiber_damage=None, cell_fields=None):
        """Cell data for the fibre hexahedra: each cell is one fibre at one
        integration station, and carries that station's own value."""
        out = {"element": owner.astype(float), "fiber": cols.astype(float), "station": stat.astype(float)}
        for name, arr in (cell_fields or {}).items():
            a = np.asarray(arr, dtype=float)
            if a.ndim == 1 and a.shape[0] == self.sol.nel:
                out[name] = a[owner]
        if fiber_stress is not None:
            out["fiber_stress"] = np.asarray(fiber_stress, dtype=float)[owner, stat, cols]
        if fiber_damage is not None:
            out["fiber_damage"] = np.asarray(fiber_damage, dtype=float)[owner, stat, cols]
        return out

    def point_stress(self, owner, cols, stat, n_points, fiber_stress):
        """`fiber_stress` as POINT data on the solid: each fibre's station
        values interpolated along the member to every slice (linearly
        between stations, extrapolated to the ends past Gauss stations;
        Lobatto's ends are stations), so the shading runs along the member
        instead of stepping slab by slab. Adjacent members' ends coincide,
        so `Clean to Grid` in ParaView joins the shading across them."""
        fs = np.asarray(fiber_stress, dtype=float)
        ts = self.slices
        out = np.zeros(n_points)
        pos = 0
        for e in self.members:
            n_c = len(self.cells[int(self.section_of[e])])
            v = self.along(ts, fs[e, :, :n_c].T)               # (n_c, n_slices)
            for si in range(len(ts)):
                block = pos + si * 4 * n_c
                out[block: block + 4 * n_c] = np.repeat(v[:, si], 4)
            pos += 4 * n_c * len(ts)
        return out

    def _bar_ends(self):
        """{(element, column): (y_i, z_i, y_j, z_j)} for every steel row
        whose position VARIES ALONG THE MEMBER -- the tendons.

        A rebar sits at one (y, z) and its two ends are that value. A
        draped tendon does not: the section integration samples it once per
        element, at the midpoint, and drawing THAT at both ends turns the
        drape into a staircase of horizontal chords -- which is how a
        tendon came out looking straight in ParaView. The solver keeps the
        profile's value at each element's two ends for exactly this.
        """
        ends = {}
        for t in getattr(self.sol, "tendons", []):
            if "y_ends" not in t:
                continue
            for k, e in enumerate(np.asarray(t["elements"], dtype=np.int64)):
                ends[(int(e), int(t["column"]))] = (float(t["y_ends"][k, 0]), float(t["z_ends"][k, 0]),
                                                    float(t["y_ends"][k, 1]), float(t["z_ends"][k, 1]))
        return ends

    def bars_frame(self, u, rot, conn=None):
        """(points, lines, line_element, line_column) of every steel row --
        rebars and tendons -- as a polyline per element: a bar at its own
        (y, z), a draped tendon between the profile's values at the
        element's two ends, so the drawn tendon is the continuous drape and
        not a staircase. Painted by `paint_bars`."""
        sol = self.sol
        conn = sol.mesh.elements if conn is None else np.asarray(conn, dtype=np.int64)
        x = self.nodes0 + u
        R = exp_map(np.asarray(rot, dtype=float))
        ts = self.slices
        if self._ends is None:
            self._ends = self._bar_ends()
        pts, lines, owner, cols, stat, blend = [], [], [], [], [], []
        for e in self.members:
            lay = sol._layouts[int(self.section_of[e])]
            if lay.n_s == 0:
                continue
            ey, ez = self.R0[e, 1], self.R0[e, 2]
            ni, nj = int(conn[e, 0]), int(conn[e, 1])
            for c in range(lay.n_c, lay.n_c + lay.n_s):
                if sol.fiber_area[e, c] <= 0.0:
                    continue
                y0, z0, y1, z1 = self._ends.get(
                    (e, c), (sol.fiber_y[e, c], sol.fiber_z[e, c], sol.fiber_y[e, c], sol.fiber_z[e, c]))
                pi = x[ni] + R[ni] @ (y0 * ey + z0 * ez)
                pj = x[nj] + R[nj] @ (y1 * ey + z1 * ez)
                base = len(pts)
                pts.extend([((1.0 - t) * pi + t * pj).tolist() for t in ts])
                blend.extend([(ni, nj, float(t)) for t in ts])
                for st in range(len(ts) - 1):
                    lines.append([base + st, base + st + 1]); owner.append(e); cols.append(c); stat.append(st)
        self._blend = np.asarray(blend, dtype=float) if blend else np.zeros((0, 3))
        return (np.asarray(pts).reshape(-1, 3), np.asarray(lines, dtype=np.int64).reshape(-1, 2),
                np.asarray(owner, dtype=np.int64), np.asarray(cols, dtype=np.int64),
                np.asarray(stat, dtype=np.int64))

    def paint_bars(self, owner, cols, stat, fiber_stress=None, fiber_damage=None):
        out = {"element": owner.astype(float), "column": cols.astype(float), "station": stat.astype(float)}
        if len(owner) == 0:
            return out
        if fiber_stress is not None:
            out["fiber_stress"] = np.asarray(fiber_stress, dtype=float)[owner, stat, cols]
        if fiber_damage is not None:
            out["fiber_damage"] = np.asarray(fiber_damage, dtype=float)[owner, stat, cols]
        return out

    def point_nodal(self, values):
        """A nodal field carried onto the points of the LAST frame built.

        THE SWEPT MESH IS ALREADY THE DEFORMED SHAPE -- its points are
        `x0 + u` -- so `displacement` was consumed as geometry and written
        nowhere, and ParaView had nothing to colour a member by and nothing
        for `Warp By Vector` (asked for in this session).

        Each swept point sits between its member's two ends at a known
        parameter, so any nodal field rides along the same blend:
        `(1-t) f[i] + t f[j]`. The blend is recorded IN the loop that places
        the points, not rebuilt afterwards -- the ordering here is the one
        thing that cannot be allowed to drift from the geometry.
        """
        v = np.asarray(values, dtype=float)
        if not hasattr(self, "_blend") or len(self._blend) == 0:
            raise ValueError("point_nodal needs a frame built first "
                             "(frame / solid_frame / bars_frame)")
        i = self._blend[:, 0].astype(np.int64)
        j = self._blend[:, 1].astype(np.int64)
        s = self._blend[:, 2]
        if v.ndim == 1:
            return (1.0 - s) * v[i] + s * v[j]
        return (1.0 - s)[:, None] * v[i] + s[:, None] * v[j]

    def _face_columns(self, owner):
        face_col = np.empty(len(owner), dtype=np.int64)
        pos = 0
        for e in self.members:
            k = int(self.section_of[e]); n = len(self._face_fiber[k])
            face_col[pos:pos + n] = self._face_fiber[k]; pos += n
        return face_col

    def paint(self, owner, fiber_stress=None, cell_fields=None, fiber_damage=None):
        """Cell data for the faces: every per-element field of the frame
        spread over the element's faces; the nearest fibre's stress (mean
        over the stations, and at each station) when given; and the nearest
        fibre's damage class (the worse of the stations: 0 intact, 1 cracked,
        2 crushed, 3 yielded, 4 failed) when given."""
        out = {}
        for name, arr in (cell_fields or {}).items():
            a = np.asarray(arr, dtype=float)
            if a.ndim == 1 and a.shape[0] == self.sol.nel:
                out[name] = a[owner]
        face_col = self._face_columns(owner) if (fiber_stress is not None or fiber_damage is not None) else None
        if fiber_stress is not None:
            fs = np.asarray(fiber_stress, dtype=float)          # (nel, stations, fibres)
            picked = fs[owner, :, :][np.arange(len(owner)), :, face_col]      # (faces, stations)
            out["fiber_stress"] = picked.mean(axis=1)
            for st in range(picked.shape[1]):                   # every station, by index
                out[f"fiber_stress_s{st}"] = picked[:, st]
        if fiber_damage is not None:
            fd = np.asarray(fiber_damage, dtype=float)
            picked = fd[owner, :, :][np.arange(len(owner)), :, face_col]
            out["fiber_damage"] = picked.max(axis=1)
        return out


def _nodal(sw, u, rot):
    """The nodal fields every frame carries onto its points.

    `displacement` because the swept mesh IS the deformed shape and so had no
    array to colour by (and nothing for `Warp By Vector`, which is how most
    people scale a deflection up to see it); `displacement_magnitude` because
    that is what a first look wants; `rotation` because a box girder that is
    twisting says so here and nowhere else in this file.
    """
    du = sw.point_nodal(np.asarray(u, dtype=float))
    return {"displacement": du,
            "displacement_magnitude": np.linalg.norm(du, axis=1),
            "rotation": sw.point_nodal(np.asarray(rot, dtype=float))}


def write_series(source, result, out_dir=None, every=1, log=print, surface=False):
    """Every frame of `result` swept and written as <stem>_sweep/frame_NNNN.vtu
    plus <stem>_sweep.pvd; returns the .pvd path. The frames are solids of
    fibre hexahedra (one cell per concrete fibre per member), or the outline
    surface with `surface=True`; the bars and tendons are a second series."""
    import meshio
    # `--every 0` was a ZeroDivisionError traceback
    if int(every) < 1:
        from .core.diagnostics import UserError
        raise UserError("SWEEP_EVERY", f"every = {every}: write every k-th frame, k >= 1",
                        "`--every 1` writes them all")
    sw = Sweep(source)
    result = os.path.abspath(result)
    stem = os.path.splitext(result)[0]
    out_dir = pathlib.Path(out_dir or stem + "_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    # CLEAR THE OLD SERIES FIRST. The frames are numbered
    # 0, 1, 2, ... of the frames KEPT, so re-sweeping the same result with a
    # different `--every` (or after a fix) leaves the old files behind
    # interleaved with the new ones. The .pvd lists only the new ones, but
    # anything opened by hand out of the folder is then a mixture of two
    # runs -- which cost an hour once.
    for stale in sorted(out_dir.glob("frame_*.vtu")) + sorted(out_dir.glob("bars_*.vtu")):
        stale.unlink()
    entries, bar_entries = [], []
    with open_result(result, "r") as h:
        frames = sorted(h["steps"], key=int)
        # THE CLOCK OF A LOAD-FACTOR RESULT IS ITS RECORD INDEX, as in the
        # XDMF: lambda runs backwards on an arc-length path,
        # and ParaView orders a .pvd by its timesteps
        by_record = str(h.attrs.get("time_axis", "")) == "load_factor"

        def clock(g, fi):
            return float(fi) if by_record else float(g.attrs.get("time", fi))
        for fi, name in enumerate(frames):
            if fi % every and fi != len(frames) - 1:
                continue
            g = h["steps"][name]
            if "displacement" not in g or "rotation" not in g:
                raise ValueError(f"frame {name} of {result} lacks displacement/rotation; "
                                 "add both to output.fields")
            u = g["displacement"][...].astype(float)
            rot = g["rotation"][...].astype(float)
            conn = g["connectivity"][...].astype(np.int64) if "connectivity" in g else None
            fields = {k: g[k][...] for k in g if k not in ("displacement", "rotation", "connectivity",
                                                           "fiber_stress", "fiber_damage",
                                                           "reaction_force", "reaction_moment")}
            fs = g["fiber_stress"][...] if "fiber_stress" in g else None
            fd = g["fiber_damage"][...] if "fiber_damage" in g else None
            if surface:
                pts, quads, owner = sw.frame(u, rot, conn)
                cell = sw.paint(owner, fs, fields, fd)
                cell["element"] = owner.astype(float)
                mesh = meshio.Mesh(pts, [("quad", quads)], point_data=_nodal(sw, u, rot),
                                   cell_data={k: [v] for k, v in cell.items()})
            else:
                pts, quads, owner, cols, stat = sw.solid_frame(u, rot, conn)
                cell = sw.paint_solid(owner, cols, stat, fs, fd, fields)
                point = dict(_nodal(sw, u, rot))
                if fs is not None:
                    point["fiber_stress_nodal"] = sw.point_stress(owner, cols, stat, len(pts), fs)
                mesh = meshio.Mesh(pts, [("hexahedron", quads)], point_data=point,
                                   cell_data={k: [v] for k, v in cell.items()})
            fn = out_dir / f"frame_{fi:04d}.vtu"
            mesh.write(fn)
            entries.append((clock(g, fi), fn.name))
            if sw.sol.n_steel:
                # the bars and tendons, as lines, in a series of their own
                bp, bl, bo, bc, bst = sw.bars_frame(u, rot, conn)
                bcell = sw.paint_bars(bo, bc, bst, fs, fd)
                bm = meshio.Mesh(bp, [("line", bl)], point_data=_nodal(sw, u, rot),
                                 cell_data={k: [v] for k, v in bcell.items()})
                bfn = out_dir / f"bars_{fi:04d}.vtu"
                bm.write(bfn)
                bar_entries.append((clock(g, fi), bfn.name))
            painted = [n for n, v in (("fiber_stress", fs), ("fiber_damage", fd)) if v is not None]
            log(f"  sweep         frame {name}: {len(pts)} points, {len(quads)} "
                f"{'faces' if surface else 'fibre cells (' + str(len(sw.slices) - 1) + ' per fibre, one per station)'}"
                + (f", painted with {' + '.join(painted)}" if painted else ""))
    pvd_path = pathlib.Path(stem + "_sweep.pvd")
    # the frames' folder as seen FROM the .pvd: its bare name was right only
    # when the folder sat beside the result
    try:
        where = pathlib.Path(os.path.relpath(out_dir.resolve(), pvd_path.resolve().parent)).as_posix()
    except ValueError:                        # Windows, another drive
        where = out_dir.resolve().as_posix()
    pvd = _collection(pvd_path, where, entries)
    if bar_entries:
        _collection(pathlib.Path(stem + "_sweep_bars.pvd"), where, bar_entries)
        log(f"  bars          {stem}_sweep_bars.pvd (rebars and tendons as lines; Display > Render Lines As Tubes)")
    return str(pvd)


def _collection(pvd, dirname, entries):
    lines = ['<?xml version="1.0"?>', '<VTKFile type="Collection" version="0.1">', '  <Collection>']
    lines += [f'    <DataSet timestep="{t:.10g}" file="{dirname}/{fn}"/>' for t, fn in entries]
    lines += ['  </Collection>', '</VTKFile>', '']
    pvd.write_text("\n".join(lines), encoding="utf-8")
    return str(pvd)
