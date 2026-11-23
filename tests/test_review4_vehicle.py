"""Vehicles and moving loads: the fourth review's findings,
one test each.

1. a member numbered against the travel mirrored the axle's distribution
2. two vehicles with one (default) name were one vehicle counted twice
3. a moving_load under arc length mixed its end stations
4. after a split the vehicle pushed the particles the fragment had left
5. a moving load on a fibre deck whose member fractured and split
6. moving_load / vehicle on a shell deck died with AttributeError
7. an axle between nodes: the end moments lacked its fixed-end moments
8. a vehicle's amplitude was accepted and ignored
9. vehicle monitors were not checked (NaN record, IndexError)
10. tyre lift-off and a rigid axle in tension were counted, not reported
11. a `speed` moving load in relaxation/static moved on the load factor
12. the tyre's dashpot pushed a wheel that was above the road
13. the vehicle ran one step behind the step's clock
14. the PSD road: misspelt keys, `scale` ignored, a coarse table
15. `vehicle.direction: [0, 0, 0]` ran to NONFINITE
"""
import copy
import json

import h5py
import numpy as np
import pytest
import yaml

from vfife.api import _with_element, prepare_case, run_case
from vfife.core.structural import make_structural_solver
from vfife.core.vehicle import roughness_profile
from vfife.io import schema

from test_member_failure import _run, _write_msh

UNITS = {"length": "m", "force": "N", "stress": "Pa"}
STEEL = {"type": "linear_elastic", "E": 2.1e11, "nu": 0.3, "rho": 7850.0}
SEC = {"type": "beam", "material": "steel", "shape": "general",
       "area": 0.10, "iy": 9.7e-4, "iz": 9.7e-4, "j": 1.94e-3}
CAR = {"name": "car", "speed": 27.78, "start": 6.0, "body": {"mass": 5750.0},
       "axles": [{"offset": 0.0, "suspension": {"stiffness": 1.595e6}}]}


def _beam(tmp_path, name, loads, *, n=4, L=20.0, els=None, solver=None, monitors=None,
          fields=("displacement",), extra=None):
    """A simply supported steel beam of n members along +x."""
    nodes = [(L * i / n, 0.0, 0.0) for i in range(n + 1)]
    els = els or [(i, i + 1) for i in range(n)]
    _write_msh(tmp_path / f"{name}.msh", nodes,
               {"left": (0, [0]), "right": (0, [n]), "mid": (0, [n // 2]), "deck": (1, els)})
    d = {"units": UNITS, "mesh": {"type": "gmsh", "path": f"{name}.msh"},
         "materials": {"steel": STEEL}, "sections": {"deck": SEC},
         "assign": [{"region": "all", "section": "deck", "orientation": [0, 1, 0]}],
         "constraints": [{"region": "left", "dofs": [1, 1, 1, 1, 0, 0]},
                         {"region": "right", "dofs": [0, 1, 1, 1, 0, 0]}],
         "loads": loads,
         "solver": solver or {"backend": "numpy", "mode": "dynamic", "total_time": 0.01},
         "output": {"fields": list(fields), "every": 10, "monitors": monitors or {}}}
    d.update(extra or {})
    d["name"] = name
    d["output"]["path"] = str(tmp_path / f"{name}.h5")
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return p


def _solver(path):
    prep = prepare_case(str(path))
    return make_structural_solver(_with_element(prep.case, prep.element, prep.case_path),
                                  prep.mesh, prep.material)


def _movers(s):
    return [F for _a, F in s.step_data(0)[3] if callable(F)]


def _refused(path, *words):
    with pytest.raises(Exception) as e:
        prepare_case(str(path))
    for w in words:
        assert w in str(e.value), str(e.value)


# 1 ------------------------------------------------------------------------ #
@pytest.mark.parametrize("kind", ["moving_load", "vehicle"])
def test_members_numbered_against_the_travel_carry_the_axle_as_forward_ones_do(tmp_path, kind):
    ml = {"moving_load": {"axles": [[0.0, 1.0e5]], "start": 6.0, "stop": 6.0}}
    veh = {"vehicle": CAR}
    load = dict(ml if kind == "moving_load" else veh, region="deck")
    solver = (None if kind == "vehicle" else
              {"backend": "numpy", "mode": "relaxation", "n_steps": 10, "load_steps": 1})
    forces = []
    for tag, els in (("fwd", None), ("rev", [(i + 1, i) for i in range(4)]),
                     ("mixed", [(1, 0), (2, 3), (2, 1), (4, 3)])):
        s = _solver(_beam(tmp_path, f"{kind}_{tag}", [load], els=els, solver=solver))
        F = _movers(s)[0]
        forces.append(F(1.0, 0.0 if kind == "vehicle" else None, np.zeros(s.ndof)))
    # 6.0 m is 20% along the member 5-10 m: Hermite weights 0.896 / 0.104
    uz = forces[0].reshape(-1, 6)[:, 2]
    assert abs(uz[1]) / abs(uz[1] + uz[2]) == pytest.approx(1 - 3 * .04 + 2 * .008, rel=1e-9)
    for f in forces[1:]:
        np.testing.assert_allclose(f, forces[0], rtol=1e-12, atol=1e-9)


# 2 ------------------------------------------------------------------------ #
def test_unnamed_vehicles_get_their_own_names_and_a_shared_name_is_refused(tmp_path):
    heavy = {"speed": 20.0, "start": 5.0, "body": {"mass": 20000.0},
             "axles": [{"offset": 0.0, "suspension": {"stiffness": 4e6}}]}
    light = dict(heavy, start=15.0, body={"mass": 1000.0})
    p = _beam(tmp_path, "two", [{"region": "deck", "vehicle": heavy}, {"region": "deck", "vehicle": light}],
              monitors={"b": {"kind": "vehicle", "of": "vehicle_2", "field": "heave"}})
    s = _solver(p)
    movers = _movers(s)
    assert sorted(v.name for v in movers) == ["vehicle", "vehicle_2"]
    f = sum(F(0.0, 0.0, np.zeros(s.ndof)) for F in movers)
    assert -f.reshape(-1, 6)[:, 2].sum() == pytest.approx(21000 * 9.80665, rel=1e-9)
    same = _beam(tmp_path, "same", [{"region": "deck", "vehicle": dict(heavy, name="a")},
                                    {"region": "deck", "vehicle": dict(light, name="a")}])
    _refused(same, "two vehicles named 'a'")


# 3 ------------------------------------------------------------------------ #
def test_a_moving_load_under_arc_length_is_refused(tmp_path):
    p = _beam(tmp_path, "arc", [{"region": "deck", "moving_load": {"axles": [[0.0, 1e5]], "start": 1.0, "stop": 19.0}}],
              solver={"backend": "numpy", "mode": "static", "static": {"arc_length": {"initial": 0.1}}})
    _refused(p, "moving_load under static.arc_length")


# 4 ------------------------------------------------------------------------ #
def test_after_a_split_the_vehicle_pushes_the_fragment_s_particles(tmp_path):
    p = _beam(tmp_path, "vsplit", [{"region": "deck", "vehicle": {
        "name": "car", "speed": 1.0, "start": 1.5, "body": {"mass": 100.0},
        "axles": [{"offset": 0.0, "suspension": {"stiffness": 1e5}}]}}],
        L=4.0, solver={"backend": "numpy", "mode": "dynamic", "total_time": 0.04,
                       "damping": {"type": "viscous", "coeff": 20.0}},
        extra={"removals": [{"region": "member1", "time": 0.02, "split": True}]})
    # a region naming member 1's two nodes: its member is the one removed
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    nodes = [(1.0 * i, 0.0, 0.0) for i in range(5)]
    _write_msh(tmp_path / "vsplit.msh", nodes,
               {"left": (0, [0]), "right": (0, [4]), "mid": (0, [2]), "member1": (0, [1, 2]),
                "deck": (1, [(i, i + 1) for i in range(4)])})
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    s = _solver(p)
    s.run()
    assert s.splits, "the member did not split"
    now = set(int(n) for n in s.mesh.elements[1, :2])
    assert now.isdisjoint({1, 2})
    veh = s._vehicles["car"]
    uz = veh._out.reshape(-1, 6)[:, 2]
    assert set(np.flatnonzero(np.abs(uz) > 1e-9)) == now


# 5 ------------------------------------------------------------------------ #
@pytest.mark.parametrize("kind", ["moving_load", "vehicle"])
def test_a_fibre_deck_that_fractures_and_splits_under_the_axle_runs_on(tmp_path, kind):
    load = ({"moving_load": {"axles": [[0.0, 6.0e4]], "start": 1.9, "speed": 1.0}} if kind == "moving_load" else
            {"vehicle": {"name": "car", "speed": 1.0, "start": 1.9, "body": {"mass": 6000.0},
                         "axles": [{"offset": 0.0, "suspension": {"stiffness": 1e7, "damping": 1e5}}]}})
    p = _beam(tmp_path, f"ffr_{kind}", [dict(load, region="deck")], L=4.0,
              solver={"backend": "numpy", "mode": "dynamic", "total_time": 0.02,
                      "damping": {"type": "viscous", "coeff": 50.0}},
              extra={"materials": {"steel": {"type": "rebar", "E": 200.0e9, "yield_strength": 4.0e8,
                                             "hardening": 2.0e9, "rho": 7850.0, "rupture_strain": 3.0e-3}},
                     "sections": {"deck": {"type": "fiber", "material": "steel", "shape": "rect",
                                           "width": 0.05, "height": 0.05, "ny": 3, "nz": 3,
                                           "fracture": {"threshold": 0.3, "split": True}}}})
    s = _solver(p)
    s.run()
    assert s.splits
    f = (_movers(s)[0](1.0, s.total_time, np.zeros(s.ndof)) if kind == "moving_load"
         else s._vehicles["car"]._out)
    hit = set(np.flatnonzero(np.abs(f.reshape(-1, 6)[:, 2]) > 1e-9))
    # every particle the axle pushes is one a member stands on now
    assert hit and hit <= set(int(n) for n in s.mesh.elements[:, :2].ravel())


# 6 ------------------------------------------------------------------------ #
@pytest.mark.parametrize("kind", ["moving_load", "vehicle"])
def test_a_travelling_load_on_a_shell_deck_is_refused_by_name(kind):
    d = {"version": 1, "units": UNITS, "mesh": {"type": "gmsh", "path": "x.msh"},
         "materials": {"steel": STEEL}, "sections": {"s": {"type": "shell", "material": "steel", "thickness": 0.2}},
         "assign": [{"region": "all", "section": "s"}],
         "loads": [{"region": "lane", **({"moving_load": {"axles": [[0.0, 1e4]], "start": 5.0, "stop": 5.5}}
                                         if kind == "moving_load" else {"vehicle": CAR})}],
         "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.01}}
    with pytest.raises(Exception, match="travel along beam members"):
        schema.from_dict(d)


# 7 ------------------------------------------------------------------------ #
@pytest.mark.parametrize("kind", ["moving_load", "line_load"])
def test_end_moments_carry_the_fixed_end_moments_of_a_load_between_nodes(tmp_path, kind):
    """A simply supported 20 m beam on four members, solved linearly: the
    written end moments must be the statics' M(x) at the nodes."""
    P, q, L = 1.0e3, 50.0, 20.0
    load = ({"moving_load": {"axles": [[0.0, P]], "start": 7.5, "stop": 7.5}} if kind == "moving_load"
            else {"line_load": [0.0, 0.0, -q]})
    s = _solver(_beam(tmp_path, f"fem_{kind}", [dict(load, region="deck")],
                      solver={"backend": "numpy", "mode": "relaxation", "n_steps": 10, "load_steps": 1}))
    fixed, _m, _p, groups, ground = s.step_data(0)
    F = s._force(groups, ground, 1.0)
    free = ~fixed
    from scipy.sparse.linalg import spsolve
    s.q[free] = spsolve(s.K.tocsr()[free][:, free], F[free])
    s.sync_orientation()
    fields = s.nodal_fields(s.internal_force()[1])
    x = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    if kind == "moving_load":
        M = np.where(x <= 7.5, P * (L - 7.5) * x / L, P * 7.5 * (L - x) / L)
    else:
        M = q * x * (L - x) / 2
    # node i of member e is x[e], node j is x[e + 1]; the j end reads -M
    np.testing.assert_allclose(np.abs(fields["bending_y_i"]), M[:-1], rtol=1e-6, atol=1e-6 * M.max())
    np.testing.assert_allclose(np.abs(fields["bending_y_j"]), M[1:], rtol=1e-6, atol=1e-6 * M.max())
    # signed since v1.0: the larger end with its internal-force sign
    # (-f_i or +f_j); the envelope keeps the old magnitude
    np.testing.assert_allclose(fields["bending_y_max_abs"], np.maximum(M[:-1], M[1:]), rtol=1e-6)
    np.testing.assert_allclose(np.abs(fields["bending_y"]), np.maximum(M[:-1], M[1:]), rtol=1e-6)
    ends = np.where(np.abs(fields["bending_y_j"]) > np.abs(fields["bending_y_i"]) * (1 + 1e-9),
                    fields["bending_y_j"], -fields["bending_y_i"])
    np.testing.assert_allclose(fields["bending_y"], ends, rtol=1e-12)


# 8 ------------------------------------------------------------------------ #
def test_a_vehicle_with_an_amplitude_is_refused(tmp_path):
    p = _beam(tmp_path, "amp", [{"region": "deck", "vehicle": CAR, "amplitude": "zero"}],
              extra={"amplitudes": {"zero": [[0.0, 0.0], [1.0, 0.0]]}})
    _refused(p, "a vehicle takes no amplitude")


# 9 ------------------------------------------------------------------------ #
def test_vehicle_monitors_name_a_vehicle_and_one_of_its_axles(tmp_path):
    bad_of = _beam(tmp_path, "badof", [{"region": "deck", "vehicle": CAR}],
                   monitors={"a": {"kind": "vehicle", "of": "Car", "field": "acceleration"}})
    _refused(bad_of, "vehicle 'Car', which no load names", "car")
    bad_axle = _beam(tmp_path, "badaxle", [{"region": "deck", "vehicle": CAR}],
                     monitors={"P": {"kind": "vehicle", "of": "car", "field": "contact_force", "axle": 1}})
    _refused(bad_axle, "has 1 axle(s)", "no axle 1")


# 10 ----------------------------------------------------------------------- #
def test_a_wheel_off_the_road_is_reported_with_its_count_and_first_time(tmp_path):
    s = np.array([-200.0, -95.0, -94.99, 0.0]); z = np.array([0.0, 0.0, -0.04, -0.04])
    np.savetxt(tmp_path / "step.txt", np.column_stack([s, z]), encoding="utf-8")
    veh = {"name": "q", "speed": 20.0, "start": -100.0, "body": {"mass": 4000.0},
           "axles": [{"offset": 0.0, "unsprung_mass": 400.0, "suspension": {"stiffness": 2e5, "damping": 1e4},
                      "tyre": {"stiffness": 1.6e6, "damping": 3e4}}],
           "roughness": {"profile": str(tmp_path / "step.txt")}}
    from test_vehicle import SEC as VSEC
    nodes = [(i * 2.0, 0, 0) for i in range(6)]
    _write_msh(tmp_path / "lift.msh", nodes, {"ends": (0, [0, 5]), "deck": (1, [(i, i + 1) for i in range(5)])})
    d = {"version": 1, "units": UNITS, "mesh": {"type": "gmsh", "path": "lift.msh"},
         "materials": {"steel": dict(STEEL, E=2.0e8)}, "sections": {"c": VSEC},
         "assign": [{"region": "deck", "section": "c", "orientation": [0, 1, 0]}],
         "constraints": [{"region": "ends", "dofs": [1] * 6}],
         "loads": [{"region": "deck", "vehicle": veh}],
         "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.5},
         "output": {"fields": ["displacement"], "every": 50, "monitors": {}}}
    out, h5 = _run(tmp_path, "lift", d)
    c = out.diagnostics["vehicle_contact"]["q"]
    assert c["tyre_lift_off_steps"] > 0 and c["rigid_axle_tension_steps"] == 0
    assert 0.25 < c["first_time"] < 0.3                  # the step is at -95 m: 0.25 s in
    with h5py.File(h5) as f:
        assert json.loads(f["diagnostics"].attrs["vehicle_contact_json"])["q"] == c


# 11 ----------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["relaxation", "static"])
def test_a_speed_driven_moving_load_outside_a_dynamic_run_is_refused(tmp_path, mode):
    solver = {"backend": "numpy", "mode": mode}
    if mode == "relaxation":
        solver["n_steps"] = 100
    p = _beam(tmp_path, f"spd_{mode}", [{"region": "deck", "moving_load": {"axles": [[0.0, 1e5]], "start": 5.0,
                                                                            "speed": 1.0}}], solver=solver)
    _refused(p, "needs solver.mode = dynamic", "give `stop`")


# 12 ----------------------------------------------------------------------- #
def test_a_tyre_dashpot_does_not_push_a_wheel_above_the_road(tmp_path):
    veh = {"name": "q", "speed": 20.0, "start": -100.0, "body": {"mass": 4000.0},
           "axles": [{"offset": 0.0, "unsprung_mass": 400.0, "suspension": {"stiffness": 2e5, "damping": 1e4},
                      "tyre": {"stiffness": 1.6e6, "damping": 3e4}}]}
    v = _movers(_solver(_beam(tmp_path, "damper", [{"region": "deck", "vehicle": veh}])))[0]
    x = v.x.copy(); x[1] = 0.005                          # 5 mm above the road ...
    vel = np.zeros_like(v.v); vel[1] = -1.0               # ... closing at 1 m/s
    f, P = v._forces(x, vel, np.zeros(1), np.zeros(1))
    assert P[0] == 0.0
    # and the wheel feels no tyre force either: only its suspension and weight
    S = -2e5 * (x[0] - x[1]) - 1e4 * (vel[0] - vel[1])
    assert f[1] == pytest.approx(-S - 400.0 * 9.80665)
    # pressed into the road it pushes, spring and dashpot together
    x[1] = -0.001
    _f, P = v._forces(x, vel, np.zeros(1), np.zeros(1))
    assert P[0] == pytest.approx(1.6e6 * 0.001 + 3e4 * 1.0)


# 13 ----------------------------------------------------------------------- #
def test_the_vehicle_acts_where_it_is_at_the_step_s_start(tmp_path):
    s = _solver(_beam(tmp_path, "clock", [{"region": "deck", "vehicle": dict(CAR, start=5.0)}]))
    v = _movers(s)[0]
    t = 40 * s.dt
    f = v(0.0, t, np.zeros(s.ndof)).reshape(-1, 6)[:, 2]
    a = (5.0 + CAR["speed"] * t - 5.0) / 5.0              # along the member 5-10 m
    assert f[2] / (f[1] + f[2]) == pytest.approx(3 * a * a - 2 * a ** 3, rel=1e-9)
    assert v.value("station") == pytest.approx(5.0 + CAR["speed"] * t)


# 14 ----------------------------------------------------------------------- #
def test_the_psd_road_refuses_a_misspelt_key_honours_scale_and_is_the_sum_itself():
    with pytest.raises(Exception, match="sead"):
        schema.RoughnessSpec(psd={"g0": 64e-6, "sead": 5})
    spec = {"psd": {"g0": 64e-6, "n_min": 0.5, "n_max": 3.0, "count": 50, "seed": 2}}
    r1 = roughness_profile(spec, 0.0, 100.0)
    r10 = roughness_profile(dict(spec, scale=10.0), 0.0, 100.0)
    x = np.linspace(0.0, 100.0, 7777)
    np.testing.assert_allclose(r10(x), 10.0 * r1(x), rtol=1e-12, atol=1e-15)
    # the harmonic sum evaluated exactly: no table between the stations
    rng = np.random.default_rng(2)
    edges = np.linspace(0.5, 3.0, 51); n = 0.5 * (edges[1:] + edges[:-1]); dn = np.diff(edges)
    amp = np.sqrt(2 * 64e-6 * (n / 0.1) ** -2.0 * dn); phi = rng.uniform(0, 2 * np.pi, 50)
    exact = (amp[None, :] * np.cos(2 * np.pi * np.outer(x, n) + phi[None, :])).sum(1)
    np.testing.assert_allclose(r1(x), exact, rtol=0, atol=1e-14)


# 15 ----------------------------------------------------------------------- #
def test_a_zero_vehicle_direction_is_refused(tmp_path):
    p = _beam(tmp_path, "zdir", [{"region": "deck", "vehicle": dict(CAR, direction=[0.0, 0.0, 0.0])}])
    _refused(p, "vehicle.direction must not be the zero vector")


# =========================================================================== #
#  1.0 re-review: the review of the v1.0 fixes
# =========================================================================== #
def _hinged_beam(tmp_path, name, load, solver):
    """0-20 m clamped at both ends on four members; member 5-10 m released
    at its j end (a hinge at x = 10)."""
    nodes = [(5.0 * i, 0.0, 0.0) for i in range(5)]
    from test_member_failure import _write_msh as w
    w(tmp_path / f"{name}.msh", nodes,
      {"ends": (0, [0, 4]), "hinged": (1, [(1, 2)]), "plain": (1, [(0, 1), (2, 3), (3, 4)])})
    d = {"name": name, "units": UNITS, "mesh": {"type": "gmsh", "path": f"{name}.msh"},
         "materials": {"steel": STEEL}, "sections": {"deck": SEC},
         "assign": [{"region": "plain", "section": "deck", "orientation": [0, 1, 0]},
                    {"region": "hinged", "section": "deck", "orientation": [0, 1, 0],
                     "releases": ["j_ry", "j_rz"]}],
         "constraints": [{"region": "ends", "dofs": [1] * 6}],
         "loads": [dict(load, region="hinged")], "solver": solver,
         "output": {"fields": ["displacement"], "every": 10, "path": str(tmp_path / f"{name}.h5")}}
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return _solver(p)


@pytest.mark.parametrize("kind", ["moving_load", "vehicle"])
def test_147_an_axle_on_a_released_member_is_condensed_like_a_line_load(tmp_path, kind):
    """The axle's fixed-end moment went to the node of a released end and
    bent the next member (the hinge deflected 30% too far), and the end
    actions then read -P L/8 at the hinge."""
    P, EI = 1.0e3, 2.1e11 * 9.7e-4
    if kind == "moving_load":
        s = _hinged_beam(tmp_path, "rel_ml", {"moving_load": {"axles": [[0.0, P]], "start": 2.5, "stop": 2.5}},
                         {"backend": "numpy", "mode": "relaxation", "n_steps": 10, "load_steps": 1})
        fixed, _m, _p, groups, ground = s.step_data(0)
        F = s._force(groups, ground, 1.0)
    else:
        veh = {"name": "car", "speed": 1e-9, "start": 2.5, "body": {"mass": P / 9.80665},
               "axles": [{"offset": 0.0, "suspension": {"stiffness": 1e6}}]}
        s = _hinged_beam(tmp_path, "rel_v", {"vehicle": veh}, {"backend": "numpy", "mode": "dynamic",
                                                                "total_time": 0.01})
        fixed, _m, _p, groups, ground = s.step_data(0)
        F = s._force(groups, ground, 0.0, 0.0)
    from scipy.sparse.linalg import spsolve
    free = ~fixed
    s.q[free] = spsolve(s.K.tocsr()[free][:, free], F[free])
    s.sync_orientation()
    f = s.nodal_fields(s.internal_force()[1])
    # statics: two 10 m cantilevers joined by the hinge's shear V
    d = P * 7.5 ** 2 * (30.0 - 7.5) / (6 * EI)
    V = 3 * EI * d / (2 * 1000.0)
    assert s.q.reshape(-1, 6)[2, 2] == pytest.approx(-V * 1000.0 / (3 * EI), rel=1e-6)
    e = int(np.flatnonzero((s.mesh.elements[:, :2] == [1, 2]).all(axis=1))[0])
    assert abs(f["bending_y_j"][e]) < 1e-6 * P
    assert abs(f["bending_y_i"][e]) == pytest.approx(P * 2.5 - V * 5.0, rel=1e-6)


def _two_crossings(tmp_path, name, vehicles, monitors):
    car = lambda start, **kw: dict({"speed": 20.0, "start": start, "body": {"mass": 5000.0},
                                    "axles": [{"offset": 0.0, "suspension": {"stiffness": 1.6e6}}]}, **kw)
    p = _beam(tmp_path, name, [], monitors=monitors,
              solver={"backend": "numpy", "mode": "dynamic", "total_time": 0.1})
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    d["steps"] = [{"name": f"s{i}", "loads": [{"region": "deck", "vehicle": car(2.0 + 10 * i, **kw)}]}
                  for i, kw in enumerate(vehicles)]
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    s = _solver(p)
    s.log = lambda *a: None
    s.run(output_every=10 ** 6)
    return s


def test_147_unnamed_vehicles_are_named_per_step_and_a_monitor_follows_each(tmp_path):
    s = _two_crossings(tmp_path, "percar", [{}, {}],
                       {"s": {"kind": "vehicle", "of": "vehicle", "field": "station"}})
    t, v = np.array(s.times), np.array(s.monitors["s"].values)
    assert v[np.isclose(t, 0.1)][0] == pytest.approx(2.0 + 20.0 * 0.1, abs=0.05)
    assert v[np.isclose(t, 0.2)][0] == pytest.approx(12.0 + 20.0 * 0.1, abs=0.05)
    # a vehicle that is not in this step is NaN, not its last state
    s = _two_crossings(tmp_path, "named", [{"name": "a"}, {"name": "b"}],
                       {"a": {"kind": "vehicle", "of": "a", "field": "station"}})
    t, v = np.array(s.times), np.array(s.monitors["a"].values)
    assert np.isfinite(v[np.isclose(t, 0.1)][0]) and np.isnan(v[np.isclose(t, 0.2)][0])


def test_147_a_wheel_off_the_road_is_timed_on_the_run_s_clock(tmp_path):
    s = np.array([-200.0, -95.0, -94.99, 0.0]); z = np.array([0.0, 0.0, -0.04, -0.04])
    np.savetxt(tmp_path / "step.txt", np.column_stack([s, z]), encoding="utf-8")
    veh = {"name": "q", "speed": 20.0, "start": -100.0, "body": {"mass": 4000.0},
           "axles": [{"offset": 0.0, "unsprung_mass": 400.0, "suspension": {"stiffness": 2e5, "damping": 1e4},
                      "tyre": {"stiffness": 1.6e6, "damping": 3e4}}],
           "roughness": {"profile": str(tmp_path / "step.txt")}}
    from test_vehicle import SEC as VSEC
    nodes = [(i * 2.0, 0, 0) for i in range(6)]
    _write_msh(tmp_path / "lift2.msh", nodes, {"ends": (0, [0, 5]), "deck": (1, [(i, i + 1) for i in range(5)])})
    d = {"version": 1, "units": UNITS, "mesh": {"type": "gmsh", "path": "lift2.msh"},
         "materials": {"steel": dict(STEEL, E=2.0e8)}, "sections": {"c": VSEC},
         "assign": [{"region": "deck", "section": "c", "orientation": [0, 1, 0]}],
         "constraints": [{"region": "ends", "dofs": [1] * 6}],
         "steps": [{"name": "rest"}, {"name": "drive", "loads": [{"region": "deck", "vehicle": veh}]}],
         "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.5},
         "output": {"fields": ["displacement"], "every": 50, "monitors": {}}}
    out, _h5 = _run(tmp_path, "lift2", d)
    assert 0.75 < out.diagnostics["vehicle_contact"]["q"]["first_time"] < 0.8   # 0.5 s + 0.25 s
