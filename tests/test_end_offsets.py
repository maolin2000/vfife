"""Rigid end offsets of beams (1.21, core/end_offsets.py, core/member_geometry.py): the elastic part
of a member runs between its nodes moved by `end_offsets`, and the stretch
between a node and its elastic end is rigid.

Each test is a closed form: a rigid end zone shortens the elastic span, a
section set off its nodes turns an axial load into a bending moment, and a
member drawn off its line with offsets back onto it is the member on its
line."""
import numpy as np
import pytest

from vfife.api import run_case

E, I, A, L, P = 2.0e11, 2.0e-4, 1.0e-2, 4.0, 200.0   # small: the geometry stays linear to 1e-4


def _case(tmp_path, *, offsets=None, load=None, n=4, points=None, kind="beam", backend="numpy",
          constraints=None, moment=None, extra_assign=None, mode="static", output=None):
    pts = points or {"A": [0, 0, 0], "B": [L, 0, 0]}
    a = {"region": "beam", "section": "s", "orientation": [0, 1, 0]}
    if offsets is not None:
        a["end_offsets"] = offsets
    loads = []
    if load is not None:
        loads.append({"region": "B", "vector": load})
    if moment is not None:
        loads.append({"region": "B", "moment": moment})
    return {
        "version": 1, "name": "off", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "lines", "points": pts,
                 "lines": {"beam": {"through": ["A", "B"], "size": L / n}}},
        "materials": {"steel": {"type": "linear_elastic", "E": E, "nu": 0.3, "rho": 7850.0}},
        "sections": {"s": {"type": "beam", "material": "steel", "shape": "general",
                           "area": A, "iy": I, "iz": I, "j": 2 * I,
                           **({"theory": "timoshenko", "ay": A, "az": A} if kind == "timoshenko" else {})}},
        "assign": [a] + (extra_assign or []),
        "constraints": constraints or [{"region": "A", "dofs": [1, 1, 1, 1, 1, 1]}],
        "loads": loads, "solver": {"mode": mode, "backend": backend, "tolerance": 1e-7},
        "output": output or {"path": str(tmp_path / "off.h5"),
                             "monitors": {"w": {"kind": "displacement", "region": "B", "dof": "z"},
                                          "u": {"kind": "displacement", "region": "B", "dof": "x"}}},
    }


def _run(case):
    got = {}
    out = run_case(case, on_ready=lambda s, _p: got.__setitem__("s", s))
    assert out.status == "CONVERGED", out.message
    m = {k: float(np.asarray(v)[-1]) for k, v in out.monitors.items()}
    s = got["s"]
    tip = int(np.argmax(s.mesh.nodes[:, 0]))
    from scipy.spatial.transform import Rotation
    m["ry"] = float(Rotation.from_matrix(s.orient[tip]).as_rotvec()[1])
    return m, s


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_a_rigid_end_zone_shortens_the_span(tmp_path, backend):
    """One member from the support to the tip, its first `a` rigid: the
    tip of a cantilever of length L - a."""
    if backend == "numba":
        pytest.importorskip("numba")
    a = 1.0
    m, s = _run(_case(tmp_path, offsets={"i": [a, 0, 0]}, load=[0, 0, -P], n=1, backend=backend))
    assert m["w"] == pytest.approx(-P * (L - a) ** 3 / (3 * E * I), rel=3e-4)
    assert m["ry"] == pytest.approx(P * (L - a) ** 2 / (2 * E * I), rel=3e-4)
    assert float(s.length[0]) == pytest.approx(L - a)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_a_section_below_its_nodes_bends_under_an_axial_load(tmp_path, backend):
    """Nodes on the top line, the centroid c below: an axial pull at the
    tip node is a pull at the centroid and a moment P c, uniform."""
    if backend == "numba":
        pytest.importorskip("numba")
    c = 0.5
    off = {"i": [0, 0, -c], "j": [0, 0, -c], "axes": "global"}
    m, _s = _run(_case(tmp_path, offsets=off, load=[P, 0, 0], backend=backend))
    M = P * c                               # the top fibre (the nodes) pulled: hogging, the tip goes down
    theta = M * L / (E * I)
    assert m["ry"] == pytest.approx(theta, rel=3e-4)
    # the tip node: the centroid's stretch, plus the turn of the rigid stretch c above it
    assert m["u"] == pytest.approx(P * L / (E * A) + theta * c, rel=3e-4)
    assert m["w"] == pytest.approx(-M * L ** 2 / (2 * E * I), rel=3e-4)


def test_local_and_global_axes_are_the_same_offset(tmp_path):
    """The member along x with orientation y: local z is global z."""
    c = 0.3
    g, _ = _run(_case(tmp_path, offsets={"i": [0, 0, -c], "j": [0, 0, -c], "axes": "global"}, load=[P, 0, 0]))
    loc, _ = _run(_case(tmp_path, offsets={"i": [0, 0, -c], "j": [0, 0, -c]}, load=[P, 0, 0]))
    for k in g:
        assert loc[k] == pytest.approx(g[k], rel=1e-12, abs=1e-15)


def test_a_member_drawn_off_its_line_and_offset_back_is_the_member_on_it(tmp_path):
    """Nodes 1 m above a cantilever's line, offsets 1 m down at both ends,
    supported and loaded through the rigid stretches: the tip moves as the
    member on the line would, plus the rigid stretch turning."""
    h = 1.0
    on, _ = _run(_case(tmp_path, load=[0, 0, -P]))
    pts = {"A": [0, 0, h], "B": [L, 0, h]}
    off, _ = _run(_case(tmp_path, points=pts, offsets={"i": [0, 0, -h], "j": [0, 0, -h], "axes": "global"},
                        load=[0, 0, -P]))
    assert off["w"] == pytest.approx(on["w"], rel=3e-4)
    assert off["ry"] == pytest.approx(on["ry"], rel=3e-4)
    assert off["u"] == pytest.approx(on["u"] + on["ry"] * h, rel=3e-4, abs=1e-12)


def test_timoshenko_members_take_offsets_too(tmp_path):
    a = 1.0
    m, _ = _run(_case(tmp_path, offsets={"i": [a, 0, 0]}, load=[0, 0, -P], n=1, kind="timoshenko"))
    G = E / 2.6
    assert m["w"] == pytest.approx(-(P * (L - a) ** 3 / (3 * E * I) + P * (L - a) / (G * A)), rel=3e-4)


def test_offsets_that_leave_no_elastic_length_are_refused(tmp_path):
    with pytest.raises(Exception, match="no elastic length"):
        _run(_case(tmp_path, offsets={"i": [L, 0, 0]}, n=1, load=[0, 0, -P]))


def test_an_unknown_offset_key_is_refused(tmp_path):
    with pytest.raises(Exception, match="no key"):
        _run(_case(tmp_path, offsets={"i": [0.1, 0, 0], "k": [0, 0, 0]}, load=[0, 0, -P]))


def _zone_or_short(tmp_path, mode, offset, backend="numpy"):
    """The rigid zone [0, a] of a one-member cantilever, or the cantilever
    drawn from a: the same structure, node for node."""
    a = 1.0
    if offset:
        c = _case(tmp_path, offsets={"i": [a, 0, 0]}, load=[0, 0, -P], n=1, mode=mode, backend=backend)
    else:
        c = _case(tmp_path, points={"A": [a, 0, 0], "B": [L, 0, 0]}, load=[0, 0, -P], n=1,
                  mode=mode, backend=backend)
        c["mesh"]["lines"]["beam"]["size"] = L - a
    if mode == "dynamic":
        c["solver"].update(total_time=0.01, damping={"type": "viscous", "coeff": 0.0})
    elif mode == "relaxation":
        c["solver"].update(n_steps=200000, tolerance=1e-6)
    return c


def test_a_rigid_zone_moves_as_the_member_drawn_without_it(tmp_path):
    """In time (the stable step, the masses) the zone model and the short
    one go through the same states."""
    out = {}
    for offset in (True, False):
        r = run_case(_zone_or_short(tmp_path / str(offset), "dynamic", offset))
        assert r.status in ("COMPLETED", "COMPLETED_DYNAMIC"), r.message
        out[offset] = np.asarray(r.monitors["w"])
    assert len(out[True]) == len(out[False])
    np.testing.assert_allclose(out[True], out[False], rtol=1e-9, atol=1e-15)


def test_the_relaxation_masses_see_the_offsets(tmp_path):
    """The fictitious masses come from the stiffness rows, and the rows hold
    the rigid zone: the tip's translations and its axial turn are the short
    cantilever's, the rest no lighter (the row sum takes in the zone's
    lever on the support's rotations, held as they are)."""
    m = {}
    for offset in (True, False):
        got = {}

        def ready(s, _p):
            got["s"] = s
            raise KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            run_case(_zone_or_short(tmp_path / str(offset), "relaxation", offset), on_ready=ready)
        s = got["s"]
        tip = int(np.argmax(s.mesh.nodes[:, 0]))
        m[offset] = np.asarray(s.mass).reshape(-1, 6)[tip]
    np.testing.assert_allclose(m[True][[0, 3]], m[False][[0, 3]], rtol=1e-12)
    assert np.all(m[True] >= m[False] * (1 - 1e-12))


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_a_frame_model_takes_them_as_a_mixed_one_does(tmp_path, backend):
    """The section on `all` makes a plain beam2 model (BeamSolver), on a
    region a mixed_line one: the same answer."""
    if backend == "numba":
        pytest.importorskip("numba")
    off = {"i": [0, 0, -0.5], "j": [0, 0, -0.5], "axes": "global"}
    mixed, s1 = _run(_case(tmp_path, offsets=off, load=[P, 0, -P], backend=backend))
    c = _case(tmp_path, offsets=off, load=[P, 0, -P], backend=backend)
    c["assign"][0]["region"] = "all"
    plain, s2 = _run(c)
    assert type(s2).__name__ == "BeamSolver" and type(s1).__name__ == "MixedStructuralSolver"
    for k in mixed:
        assert plain[k] == pytest.approx(mixed[k], rel=1e-9, abs=1e-15)


def test_a_line_load_on_a_member_off_its_nodes(tmp_path):
    """The line load is on the drawn line, straight above the elastic part
    (no lever for a vertical load); its end forces reach the nodes through
    the rigid stretches: the same tip as the member on its line, plus the
    stretch turning."""
    h, q = 1.0, -50.0
    ll = [{"region": "beam", "line_load": [0, 0, q]}]
    c_on = _case(tmp_path)
    c_on["loads"] = ll
    on, _ = _run(c_on)
    c_off = _case(tmp_path, points={"A": [0, 0, h], "B": [L, 0, h]},
                  offsets={"i": [0, 0, -h], "j": [0, 0, -h], "axes": "global"})
    c_off["loads"] = ll
    off, _ = _run(c_off)
    assert on["w"] == pytest.approx(q * L ** 4 / (8 * E * I), rel=1e-3)
    assert off["w"] == pytest.approx(on["w"], rel=3e-4)
    assert off["u"] == pytest.approx(on["u"] + on["ry"] * h, rel=3e-4, abs=1e-12)


def test_local_offsets_turn_with_the_member(tmp_path):
    """The same cantilever along y: `local` offsets follow its axes, and the
    answer is the one along x turned a quarter round."""
    off = {"i": [0, 0, -0.5], "j": [0, 0, -0.5]}
    along_x, _ = _run(_case(tmp_path, offsets=off, load=[P, 0, 0]))
    c = _case(tmp_path, points={"A": [0, 0, 0], "B": [0, L, 0]}, offsets=off, load=[0, P, 0])
    c["assign"][0]["orientation"] = [-1, 0, 0]
    c["output"]["monitors"]["u"]["dof"] = "y"
    got = {}
    r = run_case(c, on_ready=lambda s, _p: got.__setitem__("s", s))
    assert r.status == "CONVERGED", r.message
    assert float(np.asarray(r.monitors["u"])[-1]) == pytest.approx(along_x["u"], rel=1e-9)
    assert float(np.asarray(r.monitors["w"])[-1]) == pytest.approx(along_x["w"], rel=1e-9)


def test_a_member_built_in_a_step_is_born_stress_free_at_its_elastic_ends(tmp_path):
    """Built in the second step after the structure has moved, the member
    carries nothing until loaded."""
    pts = {"A": [0, 0, 0], "B": [L, 0, 0], "C": [2 * L, 0, 0]}
    c = _case(tmp_path, points=pts, offsets={"i": [0, 0, -0.5], "j": [0, 0, -0.5], "axes": "global"})
    c["mesh"]["lines"] = {"beam": {"through": ["A", "B"], "size": L / 2},
                          "tail": {"through": ["B", "C"], "size": L / 2}}
    tail = {"region": "tail", "section": "s", "orientation": [0, 1, 0],
            "end_offsets": {"i": [0, 0, -0.5], "j": [0, 0, -0.5], "axes": "global"}}
    c["assign"].append(tail)
    c["loads"] = []
    c["steps"] = [{"name": "load", "deactivate": ["tail"],
                   "loads": [{"region": "B", "vector": [0, 0, -P], "keep": True}]},
                  {"name": "build", "activate": ["tail"]}]
    got = {}
    r = run_case(c, on_ready=lambda s, _p: got.__setitem__("s", s))
    assert r.status == "CONVERGED", r.message
    s = got["s"]
    built = np.flatnonzero(np.asarray(s.member_kind) == "beam2")
    ea = np.abs(np.asarray(s.end_actions))
    x = s.mesh.nodes[s.mesh.elements[:, 0], 0]
    born = built[x[built] >= L - 1e-9]
    assert ea[born].max() < 1e-6 * P


def _at(tmp_path, station, *, offsets=None, points=None, direction=(0, 0, -1), n=2):
    """A cantilever with one axle standing at `station` along its drawn line
    (a moving load at rest: start = stop)."""
    c = _case(tmp_path, offsets=offsets, points=points, n=n)
    c["loads"] = [{"region": "beam", "moving_load": {"axles": [[0.0, P]], "start": station, "stop": station,
                                                     "direction": list(direction)}}]
    return c


@pytest.mark.parametrize("station", [0.5, 1.7, 3.2])
def test_an_axle_on_a_rigid_zone_goes_to_its_node_and_beyond_it_on_the_short_span(tmp_path, station):
    """Stations along the DRAWN line: on the root's rigid zone [0, a] the
    axle goes into the support and the tip does not move; past it, it
    stands c = s - a along the elastic cantilever of L - a."""
    a = 1.0
    m, s = _run(_at(tmp_path, station, offsets={"i": [a, 0, 0]}))
    Le, c = L - a, station - a
    want = 0.0 if c <= 0 else -P * c * c * (3 * Le - c) / (6 * E * I)
    assert m["w"] == pytest.approx(want, rel=1e-6, abs=1e-12)


def test_an_axle_beside_the_elastic_axis_twists_it_by_its_lever(tmp_path):
    """A girder drawn on its top line, its centroid h below: a vertical
    axle has no lever (the same as the member on its line), a sideways one
    twists it by P h -- the tip turns by P h c / (G J)."""
    h, st = 0.8, 2.5
    off = {"i": [0, 0, -h], "j": [0, 0, -h], "axes": "global"}
    top = {"A": [0, 0, h], "B": [L, 0, h]}
    on, _ = _run(_at(tmp_path, st))
    down, _ = _run(_at(tmp_path, st, offsets=off, points=top))
    assert down["w"] == pytest.approx(on["w"], rel=3e-4)
    c = _at(tmp_path, st, offsets=off, points=top, direction=(0, 1, 0))
    c["output"]["monitors"]["v"] = {"kind": "displacement", "region": "B", "dof": "y"}
    side, s = _run(c)
    from scipy.spatial.transform import Rotation
    tip = int(np.argmax(s.mesh.nodes[:, 0]))
    rx = float(Rotation.from_matrix(s.orient[tip]).as_rotvec()[0])
    G = E / (2 * 1.3)
    assert rx == pytest.approx(-P * h * st / (G * 2 * I), rel=1e-6)
    assert side["v"] == pytest.approx(P * st ** 2 * (3 * L - st) / (6 * E * I) - rx * h, rel=1e-6)


def test_a_line_load_on_a_rigid_zone_goes_to_its_node(tmp_path):
    """A line load is per length of the DRAWN line: over the root's zone it
    goes into the support, over the rest it loads the cantilever of L - a
    (whose tip deflection is the short member's), and the support carries
    the whole q L."""
    a, q = 1.0, -50.0
    c = _case(tmp_path, offsets={"i": [a, 0, 0]}, n=1)
    c["loads"] = [{"region": "beam", "line_load": [0, 0, q]}]
    c["output"]["monitors"]["R"] = {"kind": "reaction", "region": "A", "dof": "z"}
    m, _s = _run(c)
    assert m["w"] == pytest.approx(q * (L - a) ** 4 / (8 * E * I), rel=1e-6)
    assert m["R"] == pytest.approx(-q * L, rel=1e-9)


def test_a_vehicle_on_a_girder_drawn_on_its_top_line_is_the_vehicle_on_its_centroid(tmp_path):
    """The deck drawn h above its centroid with the offset down, and the deck
    drawn on its centroid: the same crossing (a vertical wheel has no lever,
    and the deck's vertical motion under it is the same)."""
    h = 0.5
    veh = {"name": "q", "speed": 20.0, "start": 6.0, "body": {"mass": 2000.0},
           "axles": [{"offset": 0.0, "unsprung_mass": 200.0, "suspension": {"stiffness": 2e5, "damping": 1e4},
                      "tyre": {"stiffness": 1.6e6}}]}
    runs = []
    for drawn in (False, True):
        pts = {"A": [0, 0, h if drawn else 0], "B": [4 * L, 0, h if drawn else 0]}
        c = _case(tmp_path / str(drawn), points=pts, n=16, mode="dynamic",
                  offsets={"i": [0, 0, -h], "j": [0, 0, -h], "axes": "global"} if drawn else None,
                  constraints=[{"region": "A", "dofs": [1] * 6}, {"region": "B", "dofs": [1, 1, 1, 1, 0, 0]}])
        c["loads"] = [{"region": "beam", "vehicle": veh}]
        c["solver"]["total_time"] = 0.003                   # the deck's small step: a few thousand
        c["output"]["monitors"] = {"z": {"kind": "vehicle", "of": "q", "field": "heave"},
                                   "P": {"kind": "vehicle", "of": "q", "field": "contact_force"},
                                   "w": {"kind": "vehicle", "of": "q", "field": "wheel"}}
        (tmp_path / str(drawn)).mkdir(exist_ok=True)
        r = run_case(c)
        assert r.status == "COMPLETED_DYNAMIC", r.message
        runs.append({k: np.asarray(v, float) for k, v in r.monitors.items()})
    assert np.abs(runs[0]["P"]).max() > 0.0
    # the drawn deck's mass is carried on its centroid line (`CarriedMass`):
    # the same crossing to the steps' difference (the levers stiffen the drawn
    # deck's rotations; with its mass at its nodes it was 0.13% off)
    for k in ("z", "P", "w"):
        assert runs[1][k][-1] == pytest.approx(runs[0][k][-1], rel=3e-5)


def test_a_member_with_offsets_breaks_and_its_fragment_keeps_them(tmp_path):
    """The rigid stretch follows the member's end onto its new particle
    (the design's rule): the root member of a cantilever under its own
    weight breaks off the support, zone and all, and falls with its
    elastic part as long as it was."""
    c = _case(tmp_path, offsets={"i": [0.5, 0, 0]}, mode="dynamic")
    c["assign"][0]["failure"] = {"yield_strength": 1.0e4, "extreme_fibre": [0.1, 0.1]}
    c["solver"]["total_time"] = 0.01
    c["loads"] = [{"region": "all", "gravity": [0, 0, -9.81]}]
    got = {}
    r = run_case(c, on_ready=lambda s, _p: got.__setitem__("s", s))
    assert r.status == "COMPLETED_DYNAMIC", r.message
    s = got["s"]
    root = int(np.flatnonzero(np.any(s.geometry.offset != 0, axis=(1, 2)))[0])
    n1, n2 = s.mesh.elements[root, :2]
    assert 0 not in (n1, n2) and max(n1, n2) >= 5       # off the support, onto a spare (after the mesh's five points)
    x = s.mesh.nodes + s.q.reshape(-1, 6)[:, :3]
    e1 = x[n1] + s.orient[n1] @ s.geometry.offset[root, 0]
    e2 = x[n2] + s.orient[n2] @ s.geometry.offset[root, 1]
    assert np.linalg.norm(e2 - e1) == pytest.approx(s.length[root], rel=1e-6)
    assert x[n1, 2] < -1e-5 and x[n2, 2] < -1e-5                            # falling


def test_a_removed_member_with_offsets_takes_its_line_load_with_it(tmp_path):
    """Loaded on the drawn line in the first step, the tail removed in the
    second: what is left is the cantilever under its own share alone."""
    q = -50.0
    pts = {"A": [0, 0, 0.5], "B": [L, 0, 0.5], "C": [2 * L, 0, 0.5]}
    off = {"i": [0, 0, -0.5], "j": [0, 0, -0.5], "axes": "global"}
    c = _case(tmp_path, points=pts, offsets=off)
    c["mesh"]["lines"] = {"beam": {"through": ["A", "B"], "size": L / 2},
                          "tail": {"through": ["B", "C"], "size": L / 2}}
    c["assign"].append({"region": "tail", "section": "s", "orientation": [0, 1, 0], "end_offsets": off})
    c["loads"] = []
    c["steps"] = [{"name": "load", "loads": [{"region": "all", "line_load": [0, 0, q], "keep": True}]},
                  {"name": "cut", "deactivate": ["tail"]}]
    r = run_case(c)
    assert r.status == "CONVERGED", r.message
    w = float(np.asarray(r.monitors["w"])[-1])
    assert w == pytest.approx(q * L ** 4 / (8 * E * I), rel=3e-4)


def test_contact_is_between_the_elastic_axes(tmp_path):
    """A beam drawn d above its axis (the offset down) dropped on another:
    at rest its AXIS is the two radii above the other's, its drawn line d
    more -- the capsule is on the elastic part."""
    from test_member_contact import _two_beams, _go
    d_off, r1, r2 = 0.2, 0.1, 0.15
    c = _two_beams(tmp_path, "restoff", h=0.3 + d_off, r_up=r1, r_lo=r2, total=3.0, damping=1.0,
                   struct_damping={"type": "viscous", "coeff": 6.0})
    sec = c["assign"][0]["section"]
    c["assign"] = [{"region": "lower", "section": sec, "orientation": [0, 0, 1]},
                   {"region": "upper", "section": sec, "orientation": [0, 0, 1],
                    "end_offsets": {"i": [0, 0, -d_off], "j": [0, 0, -d_off], "axes": "global"}}]
    out, s = _go(c)
    assert out.status == "COMPLETED_DYNAMIC", out.message
    x = s.mesh.nodes + s.q.reshape(-1, 6)[:, :3]
    up = s.mesh.regions["upper"]; lo = s.mesh.regions["lower"]
    mid_up = up[np.argmin(np.abs(s.mesh.nodes[up][:, 1]))]
    mid_lo = lo[np.argmin(np.abs(s.mesh.nodes[lo][:, 0]))]
    assert x[mid_up, 2] - x[mid_lo, 2] == pytest.approx(r1 + r2 + d_off, abs=1e-4)
    assert s.member_contact.initial_pairs == 0 and s.member_contact.n_active >= 1


def _offset_or_centroid(tmp_path, off, *, mode="static", n=4, rho=7850.0):
    """A cantilever along x with its section set `off` (global) from the
    drawn line, or drawn on its centroid line: the same structure."""
    pts = {"A": [0, 0, 0], "B": [L, 0, 0]}
    if off is None:
        return _case(tmp_path, n=n, mode=mode)
    pts = {k: [float(a - b) for a, b in zip(v, off)] for k, v in pts.items()}
    c = _case(tmp_path, points=pts, n=n, mode=mode,
              offsets={"i": [float(v) for v in off], "j": [float(v) for v in off], "axes": "global"})
    c["materials"]["steel"]["rho"] = rho
    return c


def test_the_weight_of_a_section_beside_its_nodes_acts_on_its_centroid(tmp_path):
    """A cantilever drawn c beside its centroid: its weight acts on the
    centroid line, through the member's axis -- it bends as the member on
    its line and does not twist. (Hung at the drawn nodes, as before 1.21's
    carried mass, the weight twisted it by w c L^2 / (2 G J).)"""
    cy, g = 0.4, 9.81
    c = _offset_or_centroid(tmp_path, [0.0, cy, 0.0])
    c["loads"] = [{"region": "all", "gravity": [0, 0, -g]}]
    m, s = _run(c)
    on = _offset_or_centroid(tmp_path / "on", None)
    on["loads"] = c["loads"]
    on, _ = _run(on)
    w = 7850.0 * A * g
    from scipy.spatial.transform import Rotation
    tip = int(np.argmax(s.mesh.nodes[:, 0]))
    rx = float(Rotation.from_matrix(s.orient[tip]).as_rotvec()[0])
    G = E / 2.6
    assert abs(rx) < 1e-6 * w * cy * L ** 2 / (2 * G * 2 * I)
    assert m["w"] == pytest.approx(on["w"], rel=1e-6)
    assert on["w"] == pytest.approx(-w * L ** 4 / (8 * E * I), rel=3e-2)      # (lumped at four members' ends)


def test_the_modes_of_a_section_set_off_its_nodes_are_those_of_the_member_on_its_line(tmp_path):
    """The mass of a member drawn off its centroid is carried where it is
    (`CarriedMass`): the parallel axis, the coupling of translation and
    rotation -- the frequencies and the modal masses of the member drawn on
    its centroid line, to round-off."""
    from vfife import modal
    import yaml
    from test_v11_buckling import _solver
    got = []
    for off in (None, [0.0, 0.3, -0.5]):
        c = _offset_or_centroid(tmp_path / str(off is None), off)
        c["output"] = {"fields": ["displacement"], "every": 100000}
        (tmp_path / str(off is None)).mkdir(exist_ok=True)
        path = tmp_path / str(off is None) / "m.yaml"
        path.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
        got.append(modal.analyse(_solver(path), n_modes=6))
    np.testing.assert_allclose(got[1].frequencies, got[0].frequencies, rtol=1e-7)
    np.testing.assert_allclose(got[1].total_mass, got[0].total_mass, rtol=1e-12)
    # (the bending pairs are degenerate: their basis is arbitrary, their sum is not)
    np.testing.assert_allclose(got[1].participation.sum(axis=0), got[0].participation.sum(axis=0),
                               rtol=1e-6, atol=1e-9)


def test_a_ground_motion_shakes_a_section_set_off_its_nodes_as_the_member_on_its_line(tmp_path):
    """Shaken across, the member drawn off its centroid moves as the one on
    it: the centroid's motion (the node's carried by the lever) the same in
    time."""
    off = np.array([0.0, 0.3, -0.5])
    rec = tmp_path / "ag.dat"
    rec.write_text("0 0\n0.002 2.0\n0.02 -2.0\n0.05 0\n", encoding="utf-8")
    tips = []
    for o in (None, off):
        c = _offset_or_centroid(tmp_path, None if o is None else list(o), mode="dynamic")
        c["loads"] = [{"region": "all", "ground_motion": {"path": str(rec), "direction": [0, 0.6, 0.8]}}]
        c["solver"].update(total_time=0.02, dt_safety=0.3)
        got = {}
        r = run_case(c, on_ready=lambda s, _p: got.__setitem__("s", s))
        assert r.status == "COMPLETED_DYNAMIC", r.message
        s = got["s"]
        tip = int(np.argmax(s.mesh.nodes[:, 0]))
        x = s.mesh.nodes[tip] + s.q.reshape(-1, 6)[tip, :3]
        if o is not None:
            x = x + s.orient[tip] @ o - o                    # the centroid, as a displacement
        tips.append(x - s.mesh.nodes[tip] if o is None else x - (s.mesh.nodes[tip]))
    np.testing.assert_allclose(tips[1], tips[0], rtol=2e-3, atol=2e-3 * np.abs(tips[0]).max())


def test_a_member_hung_beside_its_pins_swings_as_a_pendulum(tmp_path):
    """One member pinned at both nodes, its section 1 m beside the pin line,
    let go from level: its weight acts on the centroid turned with the nodes
    (`CarriedMass.body_moments`), so it swings down and never has more
    kinetic energy than m g r (a weight moment as drawn pumped it to 5 m g r
    in 1.2 s, found in review)."""
    off = [0.0, 1.0, 0.0]
    c = _case(tmp_path, n=1, mode="dynamic", offsets={"i": off, "j": off, "axes": "global"},
              constraints=[{"region": "A", "dofs": [1, 1, 1, 0, 0, 0]}, {"region": "B", "dofs": [1, 1, 1, 0, 0, 0]}])
    c["loads"] = [{"region": "all", "gravity": [0, 0, -9.81], "amplitude": "hold"}]
    c["solver"].update(total_time=0.9)
    c["output"] = {"fields": ["displacement"], "every": 100000}
    got = {}
    r = run_case(c, on_ready=lambda s, _p: got.__setitem__("s", s))
    assert r.status == "COMPLETED_DYNAMIC", r.message
    s = got["s"]
    mgr = 7850.0 * A * L * 9.81 * 1.0
    h = s.diagnostics["energy"].history
    ke = np.array([x["kinetic_energy"] for x in h])
    assert ke.max() <= 1.001 * mgr and ke.max() > 0.95 * mgr           # down through the bottom
    assert h[-1]["kinetic_energy"] == pytest.approx(h[-1]["external_work"], rel=2e-3)


def test_a_rigid_zone_weighs_as_the_member_it_is_part_of(tmp_path):
    """A cantilever with a rigid zone a at its tip: the whole drawn length
    weighs (the zone as the member it is part of), the zone's share at its
    middle -- the support carries rho A g L and the moment of the lot, and
    the elastic part bends as the short cantilever L - a under its own
    weight with the zone's weight hung a / 2 beyond its end."""
    a, g = 1.0, 9.81
    c = _case(tmp_path, offsets={"j": [-a, 0, 0]}, n=1)
    c["loads"] = [{"region": "all", "gravity": [0, 0, -g]}]
    c["output"]["monitors"]["R"] = {"kind": "reaction", "region": "A", "dof": "z"}
    m, s = _run(c)
    w = 7850.0 * A * g
    assert m["R"] == pytest.approx(w * L, rel=1e-7)
    assert abs(float(s.last_reaction_moment[int(np.argmin(s.mesh.nodes[:, 0])), 1])) == pytest.approx(
        w * L * L / 2, rel=1e-7)
    Le, Pz = L - a, w * a
    short = _case(tmp_path / "s", points={"A": [0, 0, 0], "B": [Le, 0, 0]}, n=1)
    short["loads"] = [{"region": "all", "gravity": [0, 0, -g]},
                      {"region": "B", "vector": [0, 0, -Pz]}, {"region": "B", "moment": [0, Pz * a / 2, 0]}]
    sm, _ = _run(short)
    assert m["ry"] == pytest.approx(sm["ry"], rel=1e-6)
    assert m["w"] == pytest.approx(sm["w"] - sm["ry"] * a, rel=1e-6)


@pytest.mark.parametrize("axial", ["elastic", "full"])
def test_axial_full_takes_the_stretch_over_the_length_between_the_nodes(tmp_path, axial):
    """A bar with a rigid zone a at each end pulled by N: its ends part by
    N (L - 2a) / EA with the zones rigid in every direction (`elastic`), by
    N L / EA with them rigid in bending and shear only (`full`, Midas'); the
    twist under a torque the same way."""
    a, N, T = 0.5, 1.0e4, 50.0
    c = _case(tmp_path, offsets={"i": [a, 0, 0], "j": [-a, 0, 0], "axial": axial}, load=[N, 0, 0],
              moment=[T, 0, 0])
    m, s = _run(c)
    Lx = L if axial == "full" else L - 2 * a
    assert m["u"] == pytest.approx(N * Lx / (E * A), rel=1e-6)
    from scipy.spatial.transform import Rotation
    tip = int(np.argmax(s.mesh.nodes[:, 0]))
    rx = float(Rotation.from_matrix(s.orient[tip]).as_rotvec()[0])
    assert rx == pytest.approx(T * Lx / (E / 2.6 * 2 * I), rel=1e-6)


def test_axial_full_is_for_frame_beams(tmp_path):
    c = _fibre(_case(tmp_path, offsets={"i": [0.5, 0, 0], "axial": "full"}, load=[0, 0, -P]), "linear_elastic")
    with pytest.raises(Exception, match="axial: full is for frame beams"):
        run_case(c)


@pytest.mark.parametrize("off, zones", [
    ({"i": [0, 0, -1], "j": [0, 0, -1], "axes": "global"}, 0.0),      # an inclined member hung below its nodes
    ({"i": [-0.5, 0, 0]}, 0.0),                                        # an elastic part longer than the span
    ({"i": [0.5, 0, 0], "j": [-0.3, 0, 0]}, 0.8)])                     # two zones
def test_only_a_rigid_zone_weighs_beyond_the_elastic_part(tmp_path, off, zones):
    """The member's mass is the elastic part's and its zones', no more: an
    offset that shifts the elastic part along itself (a vector in global
    axes on an inclined member) or lengthens it is no zone (found in
    review: 35% too heavy)."""
    pts = {"A": [0, 0, 0], "B": [L / np.sqrt(2), 0, L / np.sqrt(2)]}
    pts = {k: [float(v) for v in p] for k, p in pts.items()}
    c = _case(tmp_path, points=pts, offsets=off, n=1)
    c["assign"][0]["orientation"] = [0, 1, 0]
    got = {}
    run_case(c, on_ready=lambda s, _p: got.__setitem__("s", s))
    s = got["s"]
    m = s.physical_mass.reshape(-1, 6)[:, 0].sum()
    assert m == pytest.approx(7850.0 * A * (float(s.length[0]) + zones), rel=1e-12)


def test_buckling_takes_rigid_zones_and_their_levers(tmp_path):
    """A pinned column with a rigid zone a at each end: the symmetric mode
    of the elastic part l = L - 2a held at the zones' ends by their levers,
    k a tan(k l / 2) = 1 with k^2 = P / EI (k l = pi without zones)."""
    import yaml
    from scipy.optimize import brentq
    from test_v11_buckling import _solver
    from vfife import buckling
    a = 0.3
    c = _case(tmp_path, points={"A": [0, 0, 0], "B": [0, 0, L]}, n=8, offsets={"i": [a, 0, 0], "j": [-a, 0, 0]},
              constraints=[{"region": "A", "dofs": [1, 1, 1, 0, 0, 1]}, {"region": "B", "dofs": [1, 1, 0, 0, 0, 0]}])
    c["assign"][0]["orientation"] = [1, 0, 0]
    c["loads"] = [{"region": "B", "vector": [0, 0, -1000.0]}]
    c["solver"] = {"mode": "relaxation", "backend": "numpy", "n_steps": 200000, "tolerance": 1e-9}
    c["output"] = {"fields": ["displacement"], "every": 100000}
    path = tmp_path / "col.yaml"
    path.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    r = buckling.analyse(_solver(path), n_modes=2)
    l = L - 2 * a
    k = brentq(lambda k: k * a * np.tan(k * l / 2) - 1.0, 1e-6, np.pi / l - 1e-9)
    assert r.factors[0] * 1000.0 == pytest.approx(k * k * E * I, rel=1e-4)
    assert k * k * E * I < np.pi ** 2 * E * I / l ** 2         # the levers soften it below the short column


@pytest.mark.parametrize("n", [1, 4])
def test_a_zone_at_each_end_of_a_line_meshed_into_members(tmp_path, n):
    """`i` at the line's first point, `j` at its last, whatever the mesh: a
    fixed-fixed beam with a zone a at each end is a fixed-fixed clear span
    L - 2a -- its fixed-end moment q Lc^2 / 12 at the faces, its midspan
    deflection q Lc^4 / (384 E I); the zones' share of the load goes to
    the supports."""
    a, q = 0.5, -50.0
    Lc = L - 2 * a
    c = _case(tmp_path, offsets={"i": [a, 0, 0], "j": [-a, 0, 0]}, n=n,
              constraints=[{"region": "A", "dofs": [1, 1, 1, 1, 1, 1]},
                           {"region": "B", "dofs": [1, 1, 1, 1, 1, 1]}])
    c["loads"] = [{"region": "beam", "line_load": [0, 0, q]}]
    if n == 4:
        c["mesh"]["points"]["M"] = [L / 2, 0, 0]
        c["mesh"]["lines"]["beam"]["through"] = ["A", "M", "B"]
        c["output"]["monitors"]["mid"] = {"kind": "displacement", "region": "M", "dof": "z"}
    got = {}
    r = run_case(c, on_ready=lambda s, _p: got.__setitem__("s", s))
    assert r.status == "CONVERGED", r.message
    s = got["s"]
    assert float(np.sum(s.length)) == pytest.approx(Lc)
    zoned = np.flatnonzero(np.any(s.geometry.offset[:, :, 0] != 0, axis=1))
    assert len(zoned) == (1 if n == 1 else 2)
    if n == 1:
        # held at both nodes, the one member carries nothing: the load goes to the supports --
        # the clear span's face moment q Lc^2 / 12 and shear q Lc / 2 over the zone's lever a,
        # and the zone's own share q a at a / 2 (the load is on the drawn line)
        A_ = int(np.argmin(s.mesh.nodes[:, 0]))
        assert float(s.last_reaction_force[A_, 2]) == pytest.approx(-q * L / 2, rel=1e-9)
        assert abs(float(s.last_reaction_moment[A_, 1])) == pytest.approx(
            -q * Lc ** 2 / 12 - q * Lc / 2 * a - q * a * a / 2, rel=1e-9)
    else:
        assert float(np.asarray(r.monitors["mid"])[-1]) == pytest.approx(q * Lc ** 4 / (384 * E * I), rel=1e-3)


def test_different_offsets_at_the_two_ends_vary_along_the_line(tmp_path):
    """A tapered girder drawn on its top line: the centroid's depth below
    it is linear in the arc length from `i` to `j` at every node."""
    c = _case(tmp_path, offsets={"i": [0, 0, -2.0], "j": [0, 0, -1.0]}, n=4, load=[0, 0, -P])
    got = {}
    r = run_case(c, on_ready=lambda s, _p: got.__setitem__("s", s))
    assert r.status == "CONVERGED", r.message
    s = got["s"]
    for e in range(s.nel):
        for end in (0, 1):
            x = s.mesh.nodes[s.mesh.elements[e, end], 0]
            assert s.geometry.offset[e, end, 2] == pytest.approx(-2.0 + x / L, abs=1e-12)
            assert s.geometry.offset[e, end, 0] == 0.0


def test_a_region_that_is_not_one_chain_has_no_ends():
    from vfife.core.end_offsets import _chain
    el = np.array([[0, 1], [1, 2], [1, 3]])
    assert _chain(el, [0, 1, 2]) is None                       # a branch
    assert _chain(el, [0, 1]) == ([(0, False), (1, False)], [0, 1, 2])
    assert _chain(np.array([[1, 0], [1, 2]]), [0, 1]) is None    # two first nodes at the ends: no start
    assert _chain(np.array([[0, 1], [2, 1]]), [0, 1]) is None    # both drawn inwards: no start either
    assert _chain(np.array([[0, 1], [2, 1], [2, 3]]), [0, 1, 2]) == (
        [(0, False), (1, True), (2, False)], [0, 1, 2, 3])


def test_a_horizontal_line_load_on_a_girder_off_its_nodes_twists_nothing_it_should_not(tmp_path):
    """A load across the member with the offset below the nodes: on the drawn
    line, its lever twists the member by h q per length, and the centroid
    sways as the member drawn on its line, loaded the same."""
    h, q = 1.0, -50.0
    ll = [{"region": "beam", "line_load": [0, q, 0]}]
    c_on = _case(tmp_path)
    c_on["loads"] = ll
    c_on["output"]["monitors"]["v"] = {"kind": "displacement", "region": "B", "dof": "y"}
    on, _ = _run(c_on)
    c_off = _case(tmp_path, points={"A": [0, 0, h], "B": [L, 0, h]},
                  offsets={"i": [0, 0, -h], "j": [0, 0, -h], "axes": "global"})
    c_off["loads"] = ll
    c_off["output"]["monitors"]["v"] = {"kind": "displacement", "region": "B", "dof": "y"}
    off, s = _run(c_off)
    assert on["v"] == pytest.approx(q * L ** 4 / (8 * E * I), rel=1e-3)
    # the lever h of the load on the drawn line is a torque -h q per length: the tip turns
    # by -h q L^2 / (2 G J), and the drawn node sways by the centroid's sway less rx h
    rx = -h * q * L ** 2 / (2 * E / 2.6 * 2 * I)
    assert off["v"] == pytest.approx(on["v"] - rx * h, rel=3e-4)


def test_the_three_force_paths_agree_on_a_turned_state(tmp_path):
    """NumPy and the numba kernel, on a 3D frame with offsets local and
    global, every node moved and turned. (The element-at-a-time reference,
    `_internal_force_reference`, agreed to 1e-16 when this was written.)"""
    pytest.importorskip("numba")
    from scipy.spatial.transform import Rotation
    pts = {"A": [0, 0, 0], "B": [3, 0, 0], "C": [3, 2, 1], "D": [0, 2, 3]}
    c = _case(tmp_path, points=pts, offsets={"i": [0.3, 0.1, -0.4], "j": [-0.2, 0, -0.5]}, load=[0, 0, -P])
    c["mesh"]["lines"] = {"beam": {"through": ["A", "B", "C"], "size": 1.0},
                          "post": {"through": ["C", "D"], "size": 1.0}}
    c["assign"].append({"region": "post", "section": "s", "orientation": [1, 0, 0],
                        "end_offsets": {"i": [0, 0.2, 0.1], "axes": "global"}})
    got = {}

    def ready(s, _p):
        got["s"] = s
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_case(c, on_ready=ready)
    s = got["s"]
    rng = np.random.default_rng(3)
    qv = s.dofs.view(s.q)
    qv[:, :3] = 0.05 * rng.standard_normal((s.nn, 3))
    s.orient[:] = Rotation.from_rotvec(0.3 * rng.standard_normal((s.nn, 3))).as_matrix()
    out = {}
    for be in ("numpy", "numba"):
        s.backend = be
        s.rebatch()
        out[be] = np.array(s.internal_force()[0], copy=True)
    scale = np.abs(out["numpy"]).max()
    assert np.abs(out["numba"] - out["numpy"]).max() < 1e-12 * scale


# ---------------------------------------------------------------- fibre beams
def _fibre(case, law):
    """The case with a fibre tube of `law` in place of the frame section."""
    mats = {"linear_elastic": {"type": "linear_elastic", "E": E, "nu": 0.3, "rho": 7850.0},
            "steel_cyclic": {"type": "steel_cyclic", "grade": "Q345B", "E": E, "rho": 7850.0}}
    case["materials"] = {"steel": mats[law]}
    case["sections"] = {"s": {"type": "fiber", "material": "steel", "shape": "tube",
                              "diameter": 0.3, "t": 0.01, "nt": 16}}
    return case


@pytest.mark.parametrize("law", ["linear_elastic", "steel_cyclic"])
def test_a_fibre_member_drawn_off_its_line_and_offset_back_is_the_member_on_it(tmp_path, law):
    """The fibre beam's own path (the fused kernel for steel_cyclic, the
    NumPy one where it does not take the law): the same rule as a frame's."""
    h = 1.0
    on, s_on = _run(_fibre(_case(tmp_path, load=[0, 0, -P]), law))
    pts = {"A": [0, 0, h], "B": [L, 0, h]}
    off, s = _run(_fibre(_case(tmp_path, points=pts, offsets={"i": [0, 0, -h], "j": [0, 0, -h], "axes": "global"},
                               load=[0, 0, -P]), law))
    assert type(s).__name__ == "FiberBeamSolver"
    assert ("fused kernel" in s.kernel_status and "NumPy" not in s.kernel_status) == (law == "steel_cyclic"), \
        s.kernel_status
    assert off["w"] == pytest.approx(on["w"], rel=3e-4)
    assert off["ry"] == pytest.approx(on["ry"], rel=3e-4)
    assert off["u"] == pytest.approx(on["u"] + on["ry"] * h, rel=3e-4, abs=1e-12)


@pytest.mark.parametrize("law", ["linear_elastic", "steel_cyclic"])
def test_a_fibre_members_rigid_zone_is_the_short_member(tmp_path, law):
    """A one-member fibre cantilever with its first metre rigid, and the
    cantilever drawn from that metre: the same tip."""
    zone, _ = _run(_fibre(_zone_or_short(tmp_path / "z", "static", True), law))
    short, _ = _run(_fibre(_zone_or_short(tmp_path / "s", "static", False), law))
    assert zone["w"] == pytest.approx(short["w"], rel=1e-6)
    assert zone["ry"] == pytest.approx(short["ry"], rel=1e-6)


def test_a_fibre_member_with_offsets_removed_takes_its_line_load_with_it(tmp_path):
    """The fibre solver's removal with a line load on the drawn line (it
    stopped with an AttributeError before 1.21, offsets or not)."""
    q = -50.0
    pts = {"A": [0, 0, 0.5], "B": [L, 0, 0.5], "C": [2 * L, 0, 0.5]}
    off = {"i": [0, 0, -0.5], "j": [0, 0, -0.5], "axes": "global"}
    c = _fibre(_case(tmp_path, points=pts, offsets=off), "linear_elastic")
    c["mesh"]["lines"] = {"beam": {"through": ["A", "B"], "size": L / 2},
                          "tail": {"through": ["B", "C"], "size": L / 2}}
    c["assign"].append({"region": "tail", "section": "s", "orientation": [0, 1, 0], "end_offsets": off})
    c["loads"] = []
    c["steps"] = [{"name": "load", "loads": [{"region": "all", "line_load": [0, 0, q], "keep": True}]},
                  {"name": "cut", "deactivate": ["tail"]}]
    r = run_case(c)
    assert r.status == "CONVERGED", r.message
    EI = E * np.pi / 64 * (0.3 ** 4 - 0.28 ** 4)
    assert float(np.asarray(r.monitors["w"])[-1]) == pytest.approx(q * L ** 4 / (8 * EI), rel=2e-2)


def test_global_offsets_on_a_line_that_turns_are_one_vector_at_every_node(tmp_path):
    """An L in plan, a uniform sideways offset in global axes: every node's
    offset is that vector -- it does not fade or turn into a rigid zone."""
    pts = {"A": [0, 0, 0], "K": [2, 0, 0], "B": [2, 2, 0]}
    c = _case(tmp_path, points=pts, offsets={"i": [0, -0.5, 0], "j": [0, -0.5, 0], "axes": "global"},
              load=[0, 0, -P])
    c["mesh"]["lines"]["beam"] = {"through": ["A", "K", "B"], "size": 1.0}
    c["assign"][0]["orientation"] = [0, 0, 1]
    got = {}
    r = run_case(c, on_ready=lambda s, _p: got.__setitem__("s", s))
    assert r.status == "CONVERGED", r.message
    np.testing.assert_allclose(got["s"].geometry.offset, np.tile([0.0, -0.5, 0.0], (got["s"].nel, 2, 1)), atol=1e-15)


@pytest.mark.parametrize("kind", ["frame", "fibre"])
def test_a_member_born_later_then_loaded_carries_the_load(tmp_path, kind):
    """Built after the structure has moved, then loaded at its tip: the
    tail is a cantilever from B, its shear the load."""
    pts = {"A": [0, 0, 0], "B": [L, 0, 0], "C": [2 * L, 0, 0]}
    off = {"i": [0, 0, -0.5], "j": [0, 0, -0.5], "axes": "global"}
    c = _case(tmp_path, points=pts, offsets=off)
    c["mesh"]["lines"] = {"beam": {"through": ["A", "B"], "size": L / 2},
                          "tail": {"through": ["B", "C"], "size": L / 2}}
    c["assign"].append({"region": "tail", "section": "s", "orientation": [0, 1, 0], "end_offsets": off})
    if kind == "fibre":
        c = _fibre(c, "linear_elastic")
    c["steps"] = [{"name": "load", "deactivate": ["tail"],
                   "loads": [{"region": "B", "vector": [0, 0, -P], "keep": True}]},
                  {"name": "build", "activate": ["tail"]},
                  {"name": "tip", "loads": [{"region": "C", "vector": [0, 0, -P], "keep": True}]}]
    got = {}
    r = run_case(c, on_ready=lambda s, _p: got.__setitem__("s", s))
    assert r.status == "CONVERGED", r.message
    s = got["s"]
    x = s.mesh.nodes[s.mesh.elements[:, 0], 0]
    tail = np.flatnonzero(x >= L - 1e-9)
    shear = np.abs(np.asarray(s.end_actions)[tail][:, [2, 8]])
    np.testing.assert_allclose(shear, P, rtol=1e-3)


def test_per_member_offsets_are_each_members_own(tmp_path):
    """`per: member` (1.21, the Midas converter's form): every member of the
    region has the two offsets, `i` at its first node and `j` at its second
    -- four members of a cantilever, a zone a at each end of each and the
    second end dropped 0.2: the same as each member given its offsets on a
    line of its own."""
    a, n = 0.1, 4
    c = _case(tmp_path, offsets={"i": [a, 0, 0], "j": [-a, 0, -0.2], "per": "member"}, n=n,
              constraints=[{"region": "A", "dofs": [1, 1, 1, 1, 1, 1]}], load=[0, 0, -P])
    m, s = _run(c)
    assert float(np.sum(s.length)) == pytest.approx(n * np.hypot(L / n - 2 * a, 0.2))
    np.testing.assert_allclose(s.geometry.offset[:, 0], [[a, 0, 0]] * n, atol=1e-15)
    np.testing.assert_allclose(s.geometry.offset[:, 1], [[-a, 0, -0.2]] * n, atol=1e-15)
    # the same offsets on each member given line by line
    pts = {f"P{k}": [k * L / n, 0, 0] for k in range(n + 1)}
    c2 = _case(tmp_path, n=1, load=[0, 0, -P])
    c2["mesh"]["points"] = pts
    c2["mesh"]["lines"] = {f"m{k}": {"through": [f"P{k}", f"P{k + 1}"], "size": L / n} for k in range(n)}
    c2["assign"] = [{"region": f"m{k}", "section": "s", "orientation": [0, 1, 0],
                     "end_offsets": {"i": [a, 0, 0], "j": [-a, 0, -0.2]}} for k in range(n)]
    c2["constraints"] = [{"region": "P0", "dofs": [1, 1, 1, 1, 1, 1]}]
    c2["loads"] = [{"region": f"P{n}", "vector": [0, 0, -P]}]
    c2["output"]["monitors"] = {"w": {"kind": "displacement", "region": f"P{n}", "dof": "z"},
                                "u": {"kind": "displacement", "region": f"P{n}", "dof": "x"}}
    m2, _s2 = _run(c2)
    for k in ("w", "u", "ry"):
        assert m[k] == pytest.approx(m2[k], rel=1e-9, abs=1e-15)


def test_per_is_line_or_member(tmp_path):
    with pytest.raises(Exception, match="per is line or member"):
        _run(_case(tmp_path, offsets={"i": [0.5, 0, 0], "per": "each"}, load=[0, 0, -P]))


def test_the_static_tangent_is_taken_member_by_member_with_offsets(tmp_path, capfd):
    """Under numba a mixed model's static tangent differences each member
    (`_tangent_picks`); with end offsets the members' offset kernel (1.21)
    -- the converted bridges' path. The same answer as the NumPy backend."""
    pytest.importorskip("numba")
    off = {"i": [0.3, 0, -0.5], "j": [-0.2, 0, -0.4], "per": "member"}
    ref, _ = _run(_case(tmp_path, offsets=off, load=[P, 0, -P]))
    got, s = _run(_case(tmp_path, offsets=off, load=[P, 0, -P], backend="numba"))
    assert type(s).__name__ == "MixedStructuralSolver"
    assert "evaluations of each member" in capfd.readouterr().out          # not of the whole force
    for k in ref:
        assert got[k] == pytest.approx(ref[k], rel=1e-7, abs=1e-14)
