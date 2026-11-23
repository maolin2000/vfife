"""The swept surface of a fibre-beam result.

Every member's section outline is carried along the deformed axis by the
particles' exact triads and the faces between the two ends are painted
with the nearest fibre's stress. Checked on a cantilever bent by a tip
load: the tip section sits at the tip's displacement and is tilted by the
tip's rotation, the fixed end's top faces are in tension and its bottom
faces in compression, and the ParaView collection is written.
"""
import pathlib

import numpy as np
import pytest

from vfife.api import run_case
from vfife.cli import main
from vfife.core.rotations import exp_map
from vfife.sweep import Sweep, fiber_cells, section_outline, write_series

L, W, H = 4.0, 0.4, 0.2      # depth W along local y = global z; H along local z


def _mesh(td, n):
    msh = pathlib.Path(td) / "cant.msh"
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "3",
           '0 1 "base"', '0 2 "tip"', '1 3 "beam"', "$EndPhysicalNames", "$Nodes", str(n + 1)]
    out += [f"{k + 1} {L * k / n:.12g} 0 0" for k in range(n + 1)]
    out += ["$EndNodes", "$Elements", str(n + 2), "1 15 2 1 1 1", f"2 15 2 2 2 {n + 1}"]
    out += [f"{k + 3} 1 2 3 3 {k + 1} {k + 2}" for k in range(n)]
    out += ["$EndElements", ""]
    msh.write_text("\n".join(out), encoding="utf-8")
    return msh


def _case(td, n=4, shape="rect", fields="[displacement, rotation, bending_z, fiber_stress]"):
    msh = _mesh(td, n)
    sec = (f"shape: rect, width: {W}, height: {H}, ny: 8, nz: 4" if shape == "rect" else
           f"shape: box, width: {W}, height: {H}, t: 0.02, ny: 4, nz: 4, nr: 1")
    yp = pathlib.Path(td) / "cant.yaml"
    yp.write_text(f"""
name: sweep_cantilever
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
  s: {{type: linear_elastic, E: 2.0e11, nu: 0.0, rho: 7850}}
sections:
  m: {{type: fiber, material: s, {sec}}}
assign: [{{region: all, section: m}}]
regions: {{}}
constraints:
  - {{region: base, dofs: [1, 1, 1, 1, 1, 1]}}
loads:
  - {{region: tip, vector: [0, 0, -1.0e5]}}
solver: {{backend: numpy, mode: relaxation, load_steps: 1, max_steps_per_load: 20000,
          stage_ramp_iterations: 1, tolerance: 1.0e-6, kinetic_tolerance: 1.0e-9, n_steps: 50}}
output:
  path: {pathlib.Path(td) / 'cant.h5'}
  fields: {fields}
  monitors:
    w_tip: {{kind: displacement, region: tip, dof: z}}
""", encoding="utf-8")
    return str(yp), str(pathlib.Path(td) / "cant.h5")


def test_section_outlines():
    """Edges are subdivided to the fibre grid: ny along y, nz along z."""
    rect = section_outline({"type": "rect", "width": 0.4, "height": 0.2, "ny": 4, "nz": 2})[0]
    assert rect.shape == (2 * (4 + 2), 2) and np.allclose(rect[0], [-0.2, -0.1])
    assert section_outline({"type": "rect", "width": 0.4, "height": 0.2, "ny": 1, "nz": 1})[0].shape == (4, 2)
    box = section_outline({"type": "box", "width": 0.4, "height": 0.2, "thickness": 0.02, "ny": 1, "nz": 1})
    assert len(box) == 2 and np.allclose(np.abs(box[1]).max(axis=0), [0.18, 0.08])
    tube = section_outline({"type": "tube", "diameter": 0.3, "thickness": 0.01}, n_round=12)
    assert len(tube) == 2 and tube[0].shape == (12, 2)
    assert np.allclose(np.linalg.norm(tube[1], axis=1), 0.14)
    with pytest.raises(ValueError, match="outline"):
        section_outline({"type": "hexagon"})


def test_fiber_cells_sit_on_the_fibres():
    """One cell per fibre, in the section's own order, centred on it (rect,
    box); ring cells sit at the fibre's angle with the fibre at the RMS
    radius inside them."""
    from vfife.elements.fiber_section import box_fibers, rect_fibers, tube_fibers
    c = fiber_cells({"type": "rect", "width": 0.4, "height": 0.2, "ny": 8, "nz": 4})
    y, z, a = rect_fibers(0.4, 0.2, 8, 4)
    assert c.shape == (32, 4, 2) and np.allclose(c.mean(axis=1)[:, 0], y) and np.allclose(c.mean(axis=1)[:, 1], z)
    c = fiber_cells({"type": "box", "width": 0.4, "height": 0.2, "thickness": 0.02, "ny": 4, "nz": 3, "nr": 2})
    y, z, a = box_fibers(0.4, 0.2, 0.02, 4, 3, 2)
    assert len(c) == len(y) and np.allclose(c.mean(axis=1)[:, 0], y) and np.allclose(c.mean(axis=1)[:, 1], z)
    c = fiber_cells({"type": "tube", "diameter": 0.3, "thickness": 0.01, "nt": 12, "nr": 2})
    y, z, a = tube_fibers(0.3, 0.01, 12, 2)
    assert len(c) == 24
    ang = np.arctan2(c.mean(axis=1)[:, 1], c.mean(axis=1)[:, 0]); fib = np.arctan2(z, y)
    assert np.allclose(np.cos(ang - fib), 1.0)
    r0 = np.linalg.norm(c[:, 0], axis=1); r1 = np.linalg.norm(c[:, 1], axis=1)
    assert np.all((np.hypot(y, z) > r0) & (np.hypot(y, z) < r1))


def test_the_tip_section_follows_the_tip_particle(tmp_path):
    case, h5 = _case(tmp_path, n=4)
    outcome = run_case(case, progress_every=10 ** 9)
    assert outcome.status.startswith("CONVERGED") or outcome.converged
    import h5py
    with h5py.File(h5) as h:
        last = sorted(h["steps"], key=int)[-1]
        u = h["steps"][last]["displacement"][...].astype(float)
        rot = h["steps"][last]["rotation"][...].astype(float)
        fs = h["steps"][last]["fiber_stress"][...]
        mz = h["steps"][last]["bending_z"][...]
    sw = Sweep(case)
    pts, quads, owner = sw.frame(u, rot)
    assert len(quads) == 4 * 24 and len(pts) == 4 * 48 and set(owner) == {0, 1, 2, 3}
    # the last element's far end: its four vertices average to the tip point
    tip_ring = pts[-24:]
    tip = sw.nodes0[-1] + u[-1]
    assert np.allclose(tip_ring.mean(axis=0), tip, atol=1e-9)
    # ... and lie in the plane whose normal is the rotated member axis
    normal = exp_map(rot[-1:])[0] @ np.array([1.0, 0.0, 0.0])
    assert np.allclose((tip_ring - tip) @ normal, 0.0, atol=1e-9)
    assert abs(rot[-1, 1]) > 1e-3                      # the tip did rotate about y
    # painting: the top faces of the fixed element in tension, the bottom in
    # compression (a downward tip load hogs a cantilever; local y = global z)
    cell = sw.paint(owner, fs, {"bending_z": mz})
    faces0 = np.flatnonzero(owner == 0)
    centres = np.array([pts[q].mean(axis=0) for q in quads[faces0]])
    top = faces0[np.argmax(centres[:, 2])]; bottom = faces0[np.argmin(centres[:, 2])]
    assert cell["fiber_stress"][top] > 0.0 > cell["fiber_stress"][bottom]
    assert cell["bending_z"][top] == pytest.approx(mz[0])
    assert fs.shape == (4, 2, 32)


def test_the_series_is_written_and_the_cli_finds_the_result(tmp_path):
    case, h5 = _case(tmp_path, n=3, shape="box")
    run_case(case, progress_every=10 ** 9)
    log = []
    pvd = write_series(case, h5, every=1, log=log.append)
    assert pathlib.Path(pvd).exists()
    frames = sorted(pathlib.Path(pvd).with_suffix("").glob("frame_*.vtu"))
    assert len(frames) >= 1 and "painted" in log[-1]
    import meshio
    m = meshio.read(frames[-1])
    # a solid of fibre cells: the box has 2 x 4 + 2 x 4 fibres per member,
    # one cell per fibre per integration station (2-point Gauss: two slabs)
    assert m.cells[0].type == "hexahedron" and m.cells[0].data.shape[0] == 3 * 16 * 2
    assert "fiber_stress" in m.cell_data and "element" in m.cell_data and "fiber" in m.cell_data
    assert sorted(set(m.cell_data["station"][0])) == [0.0, 1.0]
    # every slab carries its own station's value, nothing averaged
    import h5py
    with h5py.File(h5) as hh:
        fs = hh["steps"][sorted(hh["steps"], key=int)[-1]]["fiber_stress"][...]
    el = m.cell_data["element"][0].astype(int); fb = m.cell_data["fiber"][0].astype(int)
    st = m.cell_data["station"][0].astype(int)
    assert np.allclose(m.cell_data["fiber_stress"][0], fs[el, st, fb], rtol=1e-6, atol=1e-3)
    # the nodal stress: the station values interpolated along the member,
    # equal on the four corners of a fibre's slice; a Gauss member's end
    # values are extrapolated so that the two stations' mean is the middle
    ps = m.point_data["fiber_stress_nodal"]; hx = m.cells[0].data
    assert ps.shape == (len(m.points),)
    for c in range(len(hx)):
        assert np.allclose(ps[hx[c][:4]], ps[hx[c][0]]) and np.allclose(ps[hx[c][4:]], ps[hx[c][4]])
    rc = main(["sweep", case, "--result", h5, "-o", str(tmp_path / "frames"), "--surface"])
    assert rc == 0 and (tmp_path / "frames" / "frame_0000.vtu").exists()
    m = meshio.read(tmp_path / "frames" / "frame_0000.vtu")
    assert m.cells[0].type == "quad" and m.cells[0].data.shape[0] == 3 * 2 * 16   # two loops, ny = nz = 4


def test_a_result_without_fiber_stress_still_sweeps_and_says_so(tmp_path):
    case, h5 = _case(tmp_path, n=2, fields="[displacement, rotation]")
    run_case(case, progress_every=10 ** 9)
    log = []
    write_series(case, h5, log=log.append)
    assert "painted" not in log[-1]


def test_two_sections_each_get_their_own_outline(tmp_path):
    """A model with several fibre sections: each member is swept
    with the outline of the section its region names."""
    msh = _mesh(tmp_path, 4)
    yp = pathlib.Path(tmp_path) / "two.yaml"
    yp.write_text(f"""
name: sweep_two
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
  s: {{type: linear_elastic, E: 2.0e11, nu: 0.0, rho: 7850}}
sections:
  deep: {{type: fiber, material: s, shape: rect, width: 0.4, height: 0.2, ny: 4, nz: 2}}
  thin: {{type: fiber, material: s, shape: box, width: 0.2, height: 0.2, t: 0.02, ny: 2, nz: 2, nr: 1}}
assign:
  - {{region: inner, section: deep}}
  - {{region: outer, section: thin}}
regions:
  inner: {{box: [-0.01, 2.01, -1, 1, -1, 1]}}
  outer: {{box: [1.99, 4.01, -1, 1, -1, 1]}}
constraints:
  - {{region: base, dofs: [1, 1, 1, 1, 1, 1]}}
loads:
  - {{region: tip, vector: [0, 0, -1.0e4]}}
solver: {{backend: numpy, mode: relaxation, load_steps: 1, max_steps_per_load: 20000,
          stage_ramp_iterations: 1, tolerance: 1.0e-6, kinetic_tolerance: 1.0e-9, n_steps: 50}}
output:
  path: {pathlib.Path(tmp_path) / 'two.h5'}
  fields: [displacement, rotation, fiber_stress]
  monitors: {{}}
""", encoding="utf-8")
    run_case(str(yp), progress_every=10 ** 9)
    sw = Sweep(str(yp))
    assert [len(l) for l in sw.loops] == [1, 2]
    pts, quads, owner = sw.frame(np.zeros((5, 3)), np.zeros((5, 3)))
    assert np.bincount(owner).tolist() == [12, 12, 16, 16]     # rect 4+2 per side pair; box 2 loops of 8
    assert len(pts) == 2 * (2 * 12 + 2 * 16)
    pvd = write_series(str(yp), str(pathlib.Path(tmp_path) / "two.h5"), log=lambda *_: None)
    assert pathlib.Path(pvd).exists()


def test_an_rc_cantilever_is_painted_with_its_damage_classes_and_its_bars(tmp_path):
    """Concrete02 cover, steel02 bars, pushed past cracking and yield: the
    base element's tension faces read cracked (1), its compression faces
    intact or crushed, and the bars on the tension side read yielded (3) in
    the separate bars series."""
    msh = _mesh(tmp_path, 2)
    yp = pathlib.Path(tmp_path) / "rc.yaml"
    h5 = pathlib.Path(tmp_path) / "rc.h5"
    yp.write_text(f"""
name: sweep_rc
units: {{length: m, force: N, stress: Pa}}
mesh: {{type: gmsh, path: {msh}}}
materials:
  conc: {{type: concrete02, fpc: -30.0e6, epsc0: -0.002, fcu: -6.0e6, epscu: -0.006, rat: 0.1, ft: 3.0e6, Ets: 1.5e9, rho: 2400}}
  steel: {{type: steel02, E: 2.0e11, Fy: 4.0e8, b: 0.01, rho: 7850}}
sections:
  bar: {{type: bar, material: steel, shape: circle, d: 0.02}}
  col: {{type: fiber, material: conc, shape: rect, width: 0.4, height: 0.3, ny: 8, nz: 4,
        bars: [{{y: 0.15, z: 0.1, section: bar}}, {{y: 0.15, z: -0.1, section: bar}},
               {{y: -0.15, z: 0.1, section: bar}}, {{y: -0.15, z: -0.1, section: bar}}]}}
assign: [{{region: all, section: col}}]
regions: {{}}
constraints:
  - {{region: base, dofs: [1, 1, 1, 1, 1, 1]}}
steps:
  - constraints: [{{region: tip, dofs: [0, 0, 1, 0, 0, 0], value: -0.06}}]
    load_steps: 3
solver: {{backend: numpy, mode: relaxation, max_steps_per_load: 20000,
          tolerance: 5.0e-4, kinetic_tolerance: 1.0e-6, n_steps: 50}}
output:
  path: {h5}
  fields: [displacement, rotation, fiber_stress, fiber_damage]
  monitors: {{}}
""", encoding="utf-8")
    out = run_case(str(yp), progress_every=10 ** 9)
    assert out.converged, out.status
    log = []
    pvd = write_series(str(yp), str(h5), log=log.append)
    assert "fiber_stress + fiber_damage" in log[-3]
    import meshio
    frames = sorted(pathlib.Path(pvd).with_suffix("").glob("frame_*.vtu"))
    bars = sorted(pathlib.Path(pvd).with_suffix("").glob("bars_*.vtu"))
    assert len(bars) == len(frames) and pathlib.Path(str(pvd).replace("_sweep.pvd", "_sweep_bars.pvd")).exists()
    m = meshio.read(frames[-1])
    dmg = m.cell_data["fiber_damage"][0]; el = m.cell_data["element"][0]
    P = m.points; q = m.cells[0].data
    assert m.cells[0].type == "hexahedron" and q.shape[0] == 2 * 32 * 2      # two members, two stations
    centres = np.array([P[qq].mean(axis=0) for qq in q])
    base = el == 0
    # pushed down (-z): the top (+z) row of fibre cells of the base element
    # is in tension, the bottom row in compression
    top = base & (centres[:, 2] > 0.15); bottom = base & (centres[:, 2] < -0.15)
    assert top.sum() == 4 * 2 and bottom.sum() == 4 * 2                     # four fibres x two stations
    assert np.all(dmg[top] == 1.0)                       # cracked
    assert np.all(np.isin(dmg[bottom], [0.0, 2.0]))      # intact or crushed
    b = meshio.read(bars[-1])
    bd = b.cell_data["fiber_damage"][0]; bel = b.cell_data["element"][0]
    bp = b.points; bl = b.cells[0].data
    bz = np.array([bp[l].mean(axis=0)[2] for l in bl])
    assert np.all(bd[(bel == 0) & (bz > 0)] == 3.0)      # the base element's tension bars yielded
    assert bl.shape[0] == 2 * 4 * 2                       # two members x four bars x two stations


def test_a_bearing_is_not_drawn_as_a_length_of_box_girder(tmp_path):
    """THE TENTH TOUCH POINT OF v0.4, and it was not on the list.

    A fibre mesh can carry `connector2` bearings now. A connector has no
    section, and the solver pads its row in `section_of` with section 0 -- a
    row it documents as "never read". This file read it: every loop here ran
    over `range(nel)`, so each bearing came out as a stub of whatever section
    sat in row 0. On case_24f that is two 0.5 m lengths of 3.97 x 13.8 m box
    girder pointing down into the piers, which is not subtle in ParaView and
    is exactly the kind of thing nobody reports because they assume it is
    their own filter settings.

    The arithmetic is the assertion: 2 connectors x 166 concrete fibres x 3
    stations = 996 cells that should not be there.
    """
    import test_fiber_connector as fc                # the miniature bridge

    sw = Sweep(fc._write(tmp_path))
    sol = sw.sol
    assert list(sw.members) == [0, 1, 2]             # not the connector, element 3

    u = np.zeros((sol.nn, 3)); rot = np.zeros((sol.nn, 3))
    _pts, _hexa, owner, _cols, _stat = sw.solid_frame(u, rot)
    assert fc.LINK not in set(owner.tolist())

    # and the faces and their colours stay in step: `paint` and
    # `_face_columns` walk the cells in the order the geometry was built, so
    # a filter applied in one place and not another shifts every colour by one
    _p, quads, face_owner = sw.frame(u, rot)
    assert fc.LINK not in set(np.asarray(face_owner).tolist())
    assert len(sw._face_columns(face_owner)) == len(face_owner) == len(quads)
    # the bars too: a connector has no rebar row either
    _bp, _bl, bar_owner, _bc, _bs = sw.bars_frame(u, rot)
    assert fc.LINK not in set(np.asarray(bar_owner).tolist())


def test_the_swept_points_carry_the_displacement(tmp_path):
    """The swept mesh IS the deformed shape, so `displacement` was consumed as
    geometry and written nowhere -- ParaView had no array to colour a member
    by and nothing for `Warp By Vector`, which is how a small deflection is
    normally made visible. Asked for in this session.

    The check that matters is the ORDERING: the blend is recorded in the loop
    that places the points, so a point's displacement must be exactly the
    convex combination of its member's two ends -- and the extremes over the
    whole swept mesh must be the nodal extremes, not something interpolated
    past them.
    """
    case, h5 = _case(tmp_path, n=4)
    run_case(case, progress_every=10 ** 9)
    import h5py
    with h5py.File(h5) as h:
        last = sorted(h["steps"], key=int)[-1]
        u = h["steps"][last]["displacement"][...].astype(float)
        rot = h["steps"][last]["rotation"][...].astype(float)

    sw = Sweep(case)
    pts, _quads, _owner = sw.frame(u, rot)
    du = sw.point_nodal(u)
    assert du.shape == (len(pts), 3)
    # a convex blend cannot leave the nodal range
    assert du[:, 2].min() == pytest.approx(u[:, 2].min(), abs=1e-12)
    assert du[:, 2].max() == pytest.approx(u[:, 2].max(), abs=1e-12)
    # the surface path blends at t = 0 and t = 1 only, so every point sits on
    # a node exactly -- the tip ring on the tip
    assert du[-24:] == pytest.approx(np.tile(u[-1], (24, 1)))

    # the solid interpolates along the member, and a cantilever's tip is its
    # largest displacement whichever path built the points
    pts_s, _hexa, _o, _c, _s = sw.solid_frame(u, rot)
    ds = sw.point_nodal(u)
    assert len(ds) == len(pts_s)
    assert abs(ds[:, 2]).max() == pytest.approx(abs(u[:, 2]).max(), abs=1e-12)
    assert np.linalg.norm(ds, axis=1).max() == pytest.approx(
        np.linalg.norm(u, axis=1).max(), abs=1e-12)

    # and the swept POINT is its undeformed position plus that displacement
    sw.frame(u, rot)
    du = sw.point_nodal(u)
    rings = pts[:24] - du[:24]                 # element 0's near end, undeformed
    assert np.allclose(rings[:, 0], sw.nodes0[0, 0], atol=1e-9)


def test_the_written_frames_carry_displacement_as_a_point_array(tmp_path):
    """End to end, because the arrays are what a user actually opens."""
    meshio = pytest.importorskip("meshio")
    case, h5 = _case(tmp_path, n=3)
    run_case(case, progress_every=10 ** 9)
    pvd = write_series(case, h5, out_dir=str(tmp_path / "sw"), log=lambda *a: None)
    frame = sorted((tmp_path / "sw").glob("frame_*.vtu"))[-1]
    m = meshio.read(str(frame))
    assert {"displacement", "displacement_magnitude", "rotation"} <= set(m.point_data)
    assert m.point_data["displacement"].shape == (len(m.points), 3)
    assert m.point_data["displacement_magnitude"] == pytest.approx(
        np.linalg.norm(m.point_data["displacement"], axis=1))
    bars = sorted((tmp_path / "sw").glob("bars_*.vtu"))
    if bars:
        assert "displacement" in meshio.read(str(bars[-1])).point_data
    assert pathlib.Path(pvd).exists()
