"""Sweep the reconstructed box section along the deformed girder and write a
ParaView collection.

    python benchmarks/sweep_confederation.py [case_28_confederation_trucks]
    python benchmarks/sweep_confederation.py case_28b_confederation_thermal --scale 200

`vfife sweep` draws fibre-beam models, and case_28 carries `general` sections
-- area and second moments, no fibres -- so the library has nothing to sweep.
It does have the outline that produced those numbers, which is what this walks
along the deformed axis instead: one closed ring per node, quads between
consecutive rings, one .vtu per output frame plus a .pvd beside them.

POINT data along the sweep: `bending_moment` and `axial_force`, SIGNED and
linear along each element, so ParaView draws the real diagram. The `bending_z`
the solver writes as an element field is `max(|M_i|, |M_j|)` -- a magnitude, on
purpose, because averaging a linearly varying moment halves a cantilever's
support moment -- and colouring by it puts a jump wherever the moment changes
sign inside an element. The signed end actions are what a diagram needs, and
the internal moment is `-M_i` at the i end and `+M_j` at the j end.

Each element gets its own pair of rings, so the moment really is discontinuous
where it should be: at the pier, where the shaft delivers its own moment into
the joint and the girder's diagram jumps by it.

Cell data: `depth`, so the haunch reads as a field and not just a shape.

    results/<case>_box/frame_0000.vtu ...
    results/<case>_box.pvd          <- open THIS in ParaView
"""
import argparse
import pathlib
import sys

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
NWEB = 12               #: samples down the web; see _ring for the rest


def _ring(cb, depth, n=NWEB):
    """The section's OUTER outline as (across, up) offsets from the centroid.

    Built from the outline's own LANDMARKS -- the deck tip, the fourteen points
    of the cantilever soffit, the web face, the soffit corner -- and not from
    `linspace` down the depth. Sampling by depth gives a 14 m section one point
    in its top 650 mm and a 4.5 m one four, so the same point index means a
    different feature on neighbouring rings and the quads between them twist
    the deck into the web.
    """
    t_bot, t_web = cb.thickness(depth)
    yc = cb.section(depth)[1]
    xw = cb.X_WEB_TOP_OUT - cb.WEB_LEAN * np.linspace(cb.T_DECK_WEB, depth, n)
    ya = np.r_[0.0, cb.CANT_T[::-1], np.linspace(cb.T_DECK_WEB, depth, n)]
    xa = np.r_[cb.HALF_DECK, cb.CANT_X[::-1], xw]
    a = np.r_[xa, -xa[::-1]] / 1e+3                 # right side down, left up
    u = np.r_[yc - ya, yc - ya[::-1]] / 1e+3
    return np.column_stack([a, u])


def _pier_ring(cb, d_out, n=48):
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return 0.5 * d_out * np.column_stack([np.cos(th), np.sin(th)])


def _pier_diameter(cb, xyz, chain, node):
    base = cb.Z_BASE["P32" if xyz[node, 0] < 0 else "P31"]
    h = xyz[chain[0], 2] - base
    f_ = float(np.clip((xyz[node, 2] - base) / h, 0.0, 1.0))
    return (cb.PIER_D_TOP if f_ >= cb.PIER_SHAFT_FRACTION else
            float(np.interp(f_, [0.0, cb.PIER_SHAFT_FRACTION],
                            [cb.PIER_D_BASE, cb.PIER_D_MID])))


def _vtu(path, pts, quads, cell_data, point_data):
    n, m = len(pts), len(quads)
    x = "\n".join("%.6g %.6g %.6g" % tuple(p) for p in pts)
    c = "\n".join("%d %d %d %d" % tuple(q) for q in quads)
    off = "\n".join(str(4 * (k + 1)) for k in range(m))
    typ = "\n".join("9" for _ in range(m))
    def arrays(d):
        return "".join(
            '<DataArray type="Float32" Name="%s" format="ascii">\n%s\n</DataArray>\n'
            % (k, "\n".join("%.6g" % v for v in vals)) for k, vals in d.items())
    fields, pfields = arrays(cell_data), arrays(point_data)
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        '<UnstructuredGrid>\n<Piece NumberOfPoints="%d" NumberOfCells="%d">\n'
        '<Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">\n'
        '%s\n</DataArray></Points>\n'
        '<Cells>\n<DataArray type="Int32" Name="connectivity" format="ascii">\n%s\n</DataArray>\n'
        '<DataArray type="Int32" Name="offsets" format="ascii">\n%s\n</DataArray>\n'
        '<DataArray type="UInt8" Name="types" format="ascii">\n%s\n</DataArray>\n</Cells>\n'
        '<PointData>\n%s</PointData>\n<CellData>\n%s</CellData>\n'
        '</Piece>\n</UnstructuredGrid>\n</VTKFile>\n'
        % (n, m, x, c, off, typ, pfields, fields), encoding="utf-8")


def main(argv=None):
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import generate_confederation as cb

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case", nargs="?", default="case_28_confederation_trucks")
    ap.add_argument("--scale", type=float, default=200.0,
                    help="displacement magnification (the deflections are mm on"
                         " a 440 m bridge; 1.0 shows nothing)")
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args(argv)
    stem = a.case[:-5] if a.case.endswith(".yaml") else a.case
    h5 = ROOT / ("results/%s.h5" % pathlib.Path(stem).name)
    out_dir = ROOT / ("results/%s_box" % pathlib.Path(stem).name)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("frame_*.vtu"):
        stale.unlink()

    with h5py.File(h5) as f:
        xyz = f["mesh/nodes"][:]
        conn = f["mesh/elements"][:]
        keys = sorted(f["steps"].keys(), key=int)[::a.every]
        need = ("bending_z_i", "bending_z_j", "axial_force_i", "axial_force_j")
        if any(n not in f["steps"][keys[0]] for n in need):
            raise SystemExit(
                "this result has no signed end actions: add "
                "bending_z_i, bending_z_j, axial_force_i, axial_force_j to the "
                "case's output.fields and run it again (the plain `bending_z` "
                "is max(|M_i|, |M_j|), which is a magnitude and not a diagram)")
        frames = [tuple(f["steps"][k][n][:] for n in
                        ("displacement",) + need) for k in keys]

    # Which nodes are girder and which are pier is decided by the ELEMENTS, not
    # by a height: the pier's top two nodes sit at z = 33.8 and 24.6 m, so any
    # "above 20 m" test hands them to the girder, which then sweeps a 14 m box
    # section at x = +/-125 and fans it out to the neighbours at +/-120 and 130.
    d = xyz[conn[:, 1]] - xyz[conn[:, 0]]
    horizontal = np.abs(d[:, 0]) > np.abs(d[:, 2])
    girder_nodes = np.unique(conn[horizontal])
    chains = [girder_nodes[np.argsort(xyz[girder_nodes, 0])]]
    for xp in (-cb.X_PIER, cb.X_PIER):
        col = conn[(~horizontal) & (np.abs(xyz[conn[:, 0], 0] - xp) < 1e-6)]
        idx = np.unique(col)
        chains.append(idx[np.argsort(-xyz[idx, 2])])

    elem_of = {}
    for e, (i, j) in enumerate(conn):
        elem_of[frozenset((int(i), int(j)))] = e

    entries = []
    for fi, (disp, mi, mj, ni, nj) in enumerate(frames):
        pts, quads = [], []
        pd = {"bending_moment": [], "axial_force": [], "part": []}
        cd = {"depth": []}
        for ci, chain in enumerate(chains):
            for k in range(len(chain) - 1):
                e = elem_of.get(frozenset((int(chain[k]), int(chain[k + 1]))))
                flip = e is not None and conn[e, 0] != chain[k]
                m_end = (mj[e], -mi[e]) if flip else (-mi[e], mj[e])
                n_end = (nj[e], -ni[e]) if flip else (-ni[e], nj[e])
                base = len(pts)
                for side in (0, 1):
                    node = chain[k + side]
                    p = xyz[node] + a.scale * disp[node]
                    nxt = chain[min(k + 1 + side, len(chain) - 1)]
                    prv = chain[max(k - 1 + side, 0)]
                    t = ((xyz[nxt] + a.scale * disp[nxt])
                         - (xyz[prv] + a.scale * disp[prv]))
                    t /= np.linalg.norm(t)
                    across = np.array([0.0, 1.0, 0.0])
                    up = np.cross(t, across)       # x_hat X y_hat = +z_hat
                    up /= np.linalg.norm(up)
                    if ci == 0:
                        d = cb.depth_at(abs(abs(xyz[node, 0]) - cb.X_PIER) * 1e+3)
                        r = _ring(cb, d)
                    else:
                        if k + side == 0:
                            # the pier meets the girder at the section CENTROID,
                            # so a shaft drawn from the node up would stand
                            # inside the box; start it at the soffit instead
                            dg = cb.depth_at(0.0)
                            p = p - np.array([0.0, 0.0,
                                              (dg - cb.section(dg)[1]) / 1e+3])
                        d = _pier_diameter(cb, xyz, chain, node) * 1e+3
                        r = _pier_ring(cb, d / 1e+3)
                    pts.extend(p + r[:, 0:1] * across + r[:, 1:2] * up)
                    pd["bending_moment"].extend([m_end[side] / 1e+6] * len(r))
                    pd["axial_force"].extend([n_end[side] / 1e+6] * len(r))
                    pd["part"].extend([float(ci)] * len(r))   # 0 girder, 1/2 pier
                    if side == 0:
                        nr, d0 = len(r), d
                for s_ in range(nr):
                    s2 = (s_ + 1) % nr
                    quads.append((base + s_, base + s2,
                                  base + nr + s2, base + nr + s_))
                    cd["depth"].append(0.5 * (d0 + d) / 1e+3)
        fn = out_dir / ("frame_%04d.vtu" % fi)
        _vtu(fn, np.asarray(pts), quads, cd, pd)
        entries.append((fi, "%s/%s" % (out_dir.name, fn.name)))

    pvd = ROOT / ("results/%s_box.pvd" % pathlib.Path(stem).name)
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
             '  <Collection>']
    lines += ['    <DataSet timestep="%d" group="" part="0" file="%s"/>' % e
              for e in entries]
    lines += ['  </Collection>', '</VTKFile>', '']
    pvd.write_text("\n".join(lines), encoding="utf-8")
    print("  written       %s (%d frames, displacement x%g)"
          % (pvd, len(entries), a.scale))
    print("                bending_moment and axial_force are point data in"
          " MN m / MN, signed and linear along each element")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
