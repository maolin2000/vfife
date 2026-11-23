"""The explicit dynamic loop, connectors, bearings and contact, as a third
reader found them. One test per reproduced defect."""
import copy
import pathlib
import sys

import numpy as np
import pytest
import yaml

from vfife import api
from vfife.elements import connector

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from test_member_failure import _write_msh  # noqa: E402

STEEL = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}


def _write(tmp_path, d, name):
    d = copy.deepcopy(d)
    d["name"] = name
    d.setdefault("version", 1)
    d.setdefault("output", {})["path"] = str(tmp_path / f"{name}.h5")
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return p


def _solver(tmp_path, d, name="s"):
    from vfife.cli import _with_element, make_solver
    prep = api.prepare_case(str(_write(tmp_path, d, name)))
    return make_solver(_with_element(prep.case, prep.element, None), prep.mesh, prep.material)


def _cantilever(tmp_path, n=4, L=2.0):
    _write_msh(tmp_path / "c.msh", [(L * i / n, 0, 0) for i in range(n + 1)],
               {"root": (0, [0]), "tip": (0, [n]), "bar": (1, [(i, i + 1) for i in range(n)])})
    return {"units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "gmsh", "path": str(tmp_path / "c.msh")},
            "materials": {"steel": dict(STEEL)},
            "sections": {"beam": {"type": "beam", "material": "steel", "shape": "rect",
                                  "width": 0.1, "height": 0.1}},
            "assign": [{"region": "bar", "section": "beam", "orientation": [0, 1, 0]}],
            "constraints": [{"region": "root", "dofs": [1] * 6}],
            "loads": [{"region": "tip", "vector": [0, 0, -1.0e3]}],
            "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 20000,
                       "tolerance": 1e-6},
            "output": {"every": 100000, "fields": ["displacement"],
                       "monitors": {"w": {"kind": "displacement", "region": "tip", "dof": "z"}}}}


# ------------------------------------------------------------------ dynamics
def test_kinetic_damping_does_not_reset_on_every_step(tmp_path):
    """The reset kept the pre-reset energy as the one to beat: 73 941 resets
    in 74 043 iterations of a continuous relaxation."""
    s = _solver(tmp_path, _cantilever(tmp_path), "k")
    s.run(output_every=100000, progress_every=1000)
    P, L, E, I = 1e3, 2.0, 2e11, 0.1 ** 4 / 12
    tip = s.q[6 * 4 + 2]
    assert tip == pytest.approx(-P * L ** 3 / (3 * E * I), rel=1e-3)
    assert s.n_resets < s.step / 3                 # every step, until v1.0


def test_a_steps_own_budget_is_the_steps(tmp_path):
    d = _cantilever(tmp_path)
    d["solver"]["n_steps"] = 300                    # far too few
    d["steps"] = [{"name": "only", "n_steps": 20000}]
    assert api.run_case(_write(tmp_path, d, "b")).converged


@pytest.mark.parametrize("damping, mode, match", [
    ({"type": "kinetic"}, "dynamic", "relaxation device"),
    ({"type": "viscous", "coeff": 1.0, "alpha": 0.5}, "dynamic", "would not be applied"),
    ({"type": "none", "alpha": 0.5}, "relaxation", "would not be applied"),
    ({"type": "kinetic", "beta": 1e-4}, "relaxation", "needs type: rayleigh"),
])
def test_damping_that_would_be_dropped_is_refused(tmp_path, damping, mode, match):
    d = _cantilever(tmp_path)
    d["solver"] = {"mode": mode, "n_steps": 100, "damping": damping,
                   **({"total_time": 0.01} if mode == "dynamic" else {})}
    with pytest.raises(Exception, match=match):
        api.prepare_case(str(_write(tmp_path, d, "d")))


def test_alpha_alongside_kinetic_damping_acts_on_a_frame(tmp_path):
    """v0.1's combination for softening problems: applied by the solid
    solver only; the frame run was bit-identical with and without it."""
    runs = []
    for a in (0.0, 0.05):
        d = _cantilever(tmp_path)
        d["solver"]["damping"] = {"type": "kinetic", "alpha": a}
        s = _solver(tmp_path, d, f"a{a}")
        s.run(output_every=100000, progress_every=1000)
        runs.append((s.step, s.q[6 * 4 + 2]))
    assert runs[0][0] != runs[1][0]
    assert runs[1][1] == pytest.approx(runs[0][1], rel=1e-4)


@pytest.mark.parametrize("spec, match", [
    ({"path": "x.dat", "direction": [0, 0, 2]}, "unit vector"),
    ({"path": "x.dat", "factor": 9.81}, "not factor"),
])
def test_a_ground_motion_says_what_it_takes(tmp_path, spec, match):
    d = _cantilever(tmp_path)
    d["solver"] = {"mode": "dynamic", "total_time": 0.01}
    d["loads"] = [{"region": "all", "ground_motion": spec}]
    with pytest.raises(Exception, match=match):
        api.prepare_case(str(_write(tmp_path, d, "g")))


def test_a_ground_motion_takes_no_amplitude(tmp_path):
    d = _cantilever(tmp_path)
    d["solver"] = {"mode": "dynamic", "total_time": 0.01}
    d["amplitudes"] = {"off": {"table": [[0, 0], [1, 0]]}}
    d["loads"] = [{"region": "all", "ground_motion": {"path": "x.dat"}, "amplitude": "off"}]
    with pytest.raises(Exception, match="no amplitude"):
        api.prepare_case(str(_write(tmp_path, d, "ga")))


@pytest.mark.parametrize("where", ["solver", "step"])
def test_load_levels_are_refused_in_a_dynamic_run(tmp_path, where):
    d = _cantilever(tmp_path)
    d["solver"] = {"mode": "dynamic", "total_time": 0.01}
    if where == "solver":
        d["solver"]["load_steps"] = 10
    else:
        d["steps"] = [{"name": "s", "load_steps": 10}]
    with pytest.raises(Exception, match="load_steps"):
        api.prepare_case(str(_write(tmp_path, d, "l")))


def test_target_dt_is_refused_where_nothing_reads_it(tmp_path):
    d = _cantilever(tmp_path)
    d["solver"] = {"mode": "dynamic", "total_time": 0.01, "target_dt": 1e-3}
    with pytest.raises(ValueError, match="target_dt"):
        _solver(tmp_path, d)


def test_the_load_is_read_where_the_state_is(tmp_path):
    """The force of step n is F(t_n): a load ramped from zero moves nothing in
    the first step. Read at t_(n+1) it made the time history first-order in
    dt (convergence ratio 2.00 on halving the step, 4.00 with this)."""
    d = _cantilever(tmp_path)
    d["solver"] = {"mode": "dynamic", "total_time": 0.002, "damping": {"type": "none"}}
    d["amplitudes"] = {"up": {"table": [[0, 0], [1, 1]]}}
    d["loads"][0]["amplitude"] = "up"
    d["output"]["every"] = 1
    w = api.run_case(_write(tmp_path, d, "o")).monitors["w"]
    assert w[0] == 0.0 and w[1] != 0.0             # the first step's force is F(0) = 0


def _truss(tmp_path, rho=7850.0):
    _write_msh(tmp_path / "t.msh", [(0, 0, 0), (2, 0, 0), (1, 0, 1)],
               {"a": (0, [0]), "b": (0, [1]), "apex": (0, [2]),
                "bars": (1, [(0, 2), (1, 2)])})
    return {"units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "gmsh", "path": str(tmp_path / "t.msh")},
            "materials": {"s": {**STEEL, "rho": rho}},
            "sections": {"bar": {"type": "bar", "material": "s", "shape": "general", "area": 1e-4}},
            "assign": [{"region": "all", "section": "bar"}],
            "constraints": [{"region": "a", "dofs": [1, 1, 1]}, {"region": "b", "dofs": [1, 1, 1]},
                            {"region": "apex", "dofs": [0, 1, 0]}],
            "nodal_masses": [{"region": "apex", "mass": 100.0}],
            "loads": [{"region": "apex", "vector": [0, 0, -1000.0], "amplitude": "hold"}],
            "solver": {"mode": "dynamic", "total_time": 0.02, "damping": {"type": "none"}},
            "output": {"every": 10, "fields": ["displacement"],
                       "monitors": {"w": {"kind": "displacement", "region": "apex", "dof": "z"}}}}


def test_a_dynamic_truss_keeps_its_velocity_across_steps(tmp_path):
    one = _truss(tmp_path)
    two = _truss(tmp_path)
    two["solver"]["total_time"] = 0.01
    two["steps"] = [{"name": "s1"}, {"name": "s2"}]
    for d in (one, two):
        d["solver"]["dt_safety"] = 0.05             # the apex's own step is coarse
    a = api.run_case(_write(tmp_path, one, "one")).monitors["w"][-1]
    b = api.run_case(_write(tmp_path, two, "two")).monitors["w"][-1]
    assert b == pytest.approx(a, rel=1e-2)          # it was half


def test_the_truss_step_sees_its_lumped_masses(tmp_path):
    """rho = 1 with 100 kg on the apex took the bar's wave-speed step, 500
    times too small."""
    s = _solver(tmp_path, _truss(tmp_path, rho=1.0))
    k = 2 * 2e11 * 1e-4 / np.sqrt(2) * 0.5
    assert s.dt > 0.1 * np.sqrt(100.0 / k)


# ----------------------------------------------------------- connectors
def _bearing(tmp_path, c):
    _write_msh(tmp_path / "b.msh", [(0, 0, 0), (0, 0, 1)],
               {"anchor": (0, [0]), "block": (0, [1]), "bearing": (1, [(0, 1)])})
    return {"units": {"length": "m", "force": "N"},
            "mesh": {"type": "gmsh", "path": str(tmp_path / "b.msh")},
            "materials": {"s": dict(STEEL)},
            "sections": {"b": {"type": "connector", "material": "s", "stiffness": [1e8, 1e6, 1e8, 1e6, 1e6, 1e6],
                               "connector_damping": [0, c, 0, 0, 0, 0]}},
            "assign": [{"region": "all", "section": "b", "orientation": [1, 0, 0]}],
            "nodal_masses": [{"region": "block", "mass": 1000.0, "inertia": [100.0] * 3}],
            "constraints": [{"region": "anchor", "dofs": [1] * 6},
                            {"region": "block", "dofs": [0, 1, 1, 0, 0, 0]}],
            "loads": [{"region": "block", "vector": [1e4, 0, 0]}],
            "solver": {"mode": "dynamic", "total_time": 1.0, "damping": {"type": "none"}},
            "output": {"every": 100000, "fields": ["displacement"]}}


def test_a_dashpot_to_a_clamped_anchor_leaves_the_step_alone(tmp_path):
    """dt 6e-25 s: the held anchor's 1e-20 mass floor was in the bound."""
    plain = _solver(tmp_path, _bearing(tmp_path, 0.0), "p").dt
    damped = _solver(tmp_path, _bearing(tmp_path, 6324.6), "d").dt
    assert damped == pytest.approx(plain, rel=1e-9)


def _state(q):
    return q.reshape(2, 6)[:, :3], None


def _geometry():
    from vfife.core.beam_solver import BeamSolver
    x0 = np.array([[0., 0., 0.], [0., 0., 0.05]])
    R0, _ = BeamSolver._frame(x0[0], x0[1], [1, 0, 0])
    return x0, R0


def test_friction_on_uy_and_uz_is_one_law_in_the_plane():
    x0, R0 = _geometry()
    law = {"type": "friction", "mu": 0.2, "normal": "ux", "stiffness": 1e8}
    props = {"type": "connector2", "coordinate_system": "local",
             "stiffness": [1e9, 0, 0, 1e6, 1e6, 1e6], "laws": {"uy": law, "uz": dict(law)}}
    q = np.zeros(12)
    d = connector.deformation(np.zeros((2, 3)), None, x0, R0, props)
    # shorten by 1e-4 (N = 1e5) and drag at 45 degrees in the local y-z plane
    ax = R0.T
    move = -1e-4 * ax[:, 0] + 1e-3 * (ax[:, 1] + ax[:, 2])
    q[6:9] = move
    _f, act, _d = connector.response(q.reshape(2, 6)[:, :3], None, np.zeros(12), x0, R0, props,
                                     slip_committed=np.zeros(6), slip_out=np.zeros(6))
    assert np.hypot(act[1], act[2]) == pytest.approx(0.2 * 1e5, rel=1e-6)     # was 1.414 mu N
    assert d is not None


def test_a_dashpot_on_an_open_gap_does_nothing():
    x0, R0 = _geometry()
    props = {"type": "connector2", "coordinate_system": "local",
             "stiffness": [0, 1e6, 1e6, 1e6, 1e6, 1e6], "connector_damping": [1e7, 0, 0, 0, 0, 0],
             "laws": {"ux": {"type": "gap", "lo": -0.006, "stiffness": 1e9}}}
    q = np.zeros(12); v = np.zeros(12)
    q[6:9] = -0.002 * R0.T[:, 0]                   # 2 mm into a 6 mm gap
    v[6:9] = -0.5 * R0.T[:, 0]
    _f, act, _d = connector.response(q.reshape(2, 6)[:, :3], None, v, x0, R0, props,
                                     include_damping=True)
    assert act[0] == 0.0


def test_a_global_connector_with_length_balances_moments():
    x0 = np.array([[0., 0., 0.], [0., 0., 0.05]])
    from vfife.core.beam_solver import BeamSolver
    R0, _ = BeamSolver._frame(x0[0], x0[1], [1, 0, 0])
    props = {"type": "connector2", "coordinate_system": "global",
             "stiffness": [1e6, 1e6, 1e10, 1e8, 1e8, 1e8]}
    u = np.array([[0., 0., 0.], [0.2, 0., -0.001]])
    fe, _a, _d = connector.response(u, None, np.zeros(12), x0, R0, props)
    x = x0 + u
    net = np.cross(x[0], fe[0:3]) + np.cross(x[1], fe[6:9]) + fe[3:6] + fe[9:12]
    assert np.abs(net).max() < 1e-6 * np.abs(fe).max()     # was the whole P-Delta moment


def test_a_named_normal_on_global_axes_is_refused():
    x0, R0 = _geometry()
    props = {"type": "connector2", "coordinate_system": "global",
             "stiffness": [0, 0, 1e9, 1e6, 1e6, 1e6],
             "laws": {"ux": {"type": "friction", "mu": 0.2, "normal": "uz", "stiffness": 1e8}}}
    with pytest.raises(ValueError, match="node"):
        connector.reference_stiffness(x0, R0, props)


def test_the_friction_normal_is_the_elongation():
    from vfife.io.schema import ConnectorLawSpec
    with pytest.raises(Exception, match="ux"):
        ConnectorLawSpec(type="friction", mu=0.2, normal="uz", stiffness=1e8)


def test_the_compression_only_seat_is_lo_zero(tmp_path):
    """The docs' {lo: null, hi: 0} is a tension-only tie; {lo: 0} carries a
    block resting on it."""
    _write_msh(tmp_path / "s.msh", [(0, 0, 0), (0, 0, 1)],
               {"ground": (0, [0]), "block": (0, [1]), "seat": (1, [(0, 1)])})
    d = {"units": {"length": "m", "force": "N"},
         "mesh": {"type": "gmsh", "path": str(tmp_path / "s.msh")},
         "materials": {"s": dict(STEEL)},
         "sections": {"seat": {"type": "connector", "material": "s", "stiffness": [0, 1e8, 1e8, 1e6, 1e6, 1e6],
                               "laws": {"ux": {"type": "gap", "lo": 0.0, "stiffness": 1e9}}}},
         "assign": [{"region": "all", "section": "seat", "orientation": [1, 0, 0]}],
         "constraints": [{"region": "ground", "dofs": [1] * 6}],
         "loads": [{"region": "block", "vector": [0, 0, -1e5]}],
         "solver": {"mode": "relaxation", "n_steps": 20000, "tolerance": 1e-8},
         "output": {"every": 100000, "fields": ["displacement", "connector_force"],
                    "monitors": {"w": {"kind": "displacement", "region": "block", "dof": "z"}}}}
    out = api.run_case(_write(tmp_path, d, "seat"))
    assert out.converged
    assert out.monitors["w"][-1] == pytest.approx(-1e-4, rel=1e-3)


def test_connector_forces_keep_their_sign(tmp_path):
    import h5py
    d = {"units": {"length": "m", "force": "N"}}
    _write_msh(tmp_path / "s.msh", [(0, 0, 0), (0, 0, 1)],
               {"ground": (0, [0]), "block": (0, [1]), "post": (1, [(0, 1)])})
    d.update({"mesh": {"type": "gmsh", "path": str(tmp_path / "s.msh")},
              "materials": {"s": dict(STEEL)},
              "sections": {"p": {"type": "connector", "material": "s", "stiffness": [1e9, 1e8, 1e8, 1e6, 1e6, 1e6]}},
              "assign": [{"region": "all", "section": "p", "orientation": [1, 0, 0]}],
              "constraints": [{"region": "ground", "dofs": [1] * 6}],
              "solver": {"mode": "relaxation", "n_steps": 20000, "tolerance": 1e-8},
              "output": {"every": 100000, "fields": ["connector_force"]}})
    signs = []
    for f in (-1e5, 1e5):
        d["loads"] = [{"region": "block", "vector": [0, 0, f]}]
        out = api.run_case(_write(tmp_path, d, f"sg{f > 0}"))
        with h5py.File(out.result_path) as h:
            last = sorted(h["steps"], key=int)[-1]
            signs.append(float(np.asarray(h["steps"][last]["connector_force_x"])[0]))
    assert signs[0] * signs[1] < 0                  # both came out positive


def test_a_zero_length_connector_is_reported_not_a_traceback(tmp_path):
    _write_msh(tmp_path / "z.msh", [(0, 0, 0), (0, 0, 0)],
               {"a": (0, [0]), "b": (0, [1]), "c": (1, [(0, 1)])})
    d = {"units": {"length": "m", "force": "N"},
         "mesh": {"type": "gmsh", "path": str(tmp_path / "z.msh")},
         "materials": {"s": dict(STEEL)},
         "sections": {"p": {"type": "connector", "material": "s", "stiffness": [1e9] * 6}},
         "assign": [{"region": "all", "section": "p", "orientation": [1, 0, 0]}],
         "constraints": [{"region": "a", "dofs": [1] * 6}],
         "loads": [{"region": "b", "vector": [0, 0, -1.0]}],
         "solver": {"mode": "relaxation", "n_steps": 100}}
    prep = api.prepare_case(str(_write(tmp_path, d, "z")))
    codes = {i["code"] for i in prep.preflight["issues"]}
    assert "ZERO_LENGTH" in codes


def test_a_removed_connector_in_a_fibre_model_stops_carrying(tmp_path):
    """Reported failed by `member_failed` and still 20 kN in the prop."""
    _write_msh(tmp_path / "fr.msh", [(0, 0, 0), (0.4, 0, 0), (0.8, 0, 0), (0.8, 0, -0.2)],
               {"root": (0, [0]), "tip": (0, [2]), "ground": (0, [3]),
                "beam": (1, [(0, 1), (1, 2)]), "prop": (1, [(2, 3)])})
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": str(tmp_path / "fr.msh")},
         "materials": {"st": {"type": "steel02", "E": 2e11, "Fy": 4e9, "b": 0.01, "R0": 20.0, "rho": 7850.0}},
         "sections": {"f": {"type": "fiber", "material": "st", "shape": "rect", "width": 0.1,
                            "height": 0.1, "ny": 4, "nz": 4},
                      "pot": {"type": "connector", "material": "st",
                              "stiffness": [5e9, 1e8, 1e8, 1e6, 1e6, 1e6]}},
         "assign": [{"region": "beam", "section": "f", "orientation": [0, 1, 0]},
                    {"region": "prop", "section": "pot", "orientation": [1, 0, 0]}],
         "constraints": [{"region": "root", "dofs": [1] * 6}, {"region": "ground", "dofs": [1] * 6}],
         "loads": [{"region": "tip", "vector": [0, 0, -2e4], "amplitude": "hold"}],
         "removals": [{"region": "prop", "time": 0.02}],
         "solver": {"mode": "dynamic", "backend": "numpy", "total_time": 0.1,
                    "damping": {"type": "rayleigh", "alpha": 60.0}},
         "output": {"every": 1000, "fields": ["displacement"],
                    "monitors": {"Rg": {"kind": "reaction", "region": "ground", "dof": "z"}}}}
    out = api.run_case(_write(tmp_path, d, "fr"))
    assert abs(out.monitors["Rg"][-1]) < 1e-6


def test_a_contact_stiffness_past_the_explicit_limit_is_capped(tmp_path):
    """A beam resting on a 5e10 N/m floor was thrown 1.4 mm into the air."""
    from vfife.core.rigid_contact import RigidContact

    class _Mesh:
        nodes = np.array([[0.0, 0.0, 0.0]])
    c = RigidContact({"rigid": [{"plane": {"point": [0, 0, 0], "normal": [0, 0, 1]},
                                 "stiffness": 5e10}]}, _Mesh(), {}, "dynamic")
    m, dt = np.array([10.0]), 1e-4
    f = c.force(np.array([[0.0, 0.0, -1e-6]]), np.zeros((1, 3)), m, dt)
    assert f[0, 2] == pytest.approx(m[0] * (2.0 / dt) ** 2 / 4.0 * 1e-6)
    assert c.stiffness_capped


def test_a_pressure_on_a_physical_group_loads_its_own_elements(tmp_path):
    """Two wheel patches one element apart: the strip between them has all
    its nodes in the patches, and took the pressure too (+50%)."""
    sys.path.insert(0, str(ROOT / "tests"))
    from mshwrite import write22
    nodes = [(x, y, 0.0) for y in range(4) for x in range(3)]
    q = lambda i, j: [j * 3 + i + 1, j * 3 + i + 2, (j + 1) * 3 + i + 2, (j + 1) * 3 + i + 1]
    els = [(3, 1, q(i, 0)) for i in range(2)] + [(3, 2, q(i, 1)) for i in range(2)] \
        + [(3, 1, q(i, 2)) for i in range(2)] + [(1, 3, [1, 4]), (1, 3, [4, 7]), (1, 3, [7, 10]),
                                                 (1, 3, [3, 6]), (1, 3, [6, 9]), (1, 3, [9, 12])]
    write22(tmp_path / "p.msh", nodes, els, [(2, 1, "wheels"), (2, 2, "gap"), (1, 3, "edges")])
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": str(tmp_path / "p.msh")},
         "materials": {"s": dict(STEEL)},
         "sections": {"p": {"type": "shell", "material": "s", "thickness": 0.05}},
         "assign": [{"region": "all", "section": "p"}],
         "constraints": [{"region": "edges", "dofs": [1] * 6}],
         "loads": [{"region": "wheels", "pressure": 1000.0}],
         "solver": {"mode": "relaxation", "n_steps": 20000, "tolerance": 1e-6},
         "output": {"every": 100000, "fields": ["displacement"],
                    "monitors": {"R": {"kind": "reaction", "region": "edges", "dof": "z"}}}}
    out = api.run_case(_write(tmp_path, d, "pw"))
    assert abs(out.monitors["R"][-1]) == pytest.approx(1000.0 * 4, rel=1e-4)    # was 6000


def test_every_shell_model_is_told_it_is_experimental(tmp_path):
    from mshwrite import write22
    write22(tmp_path / "t.msh", [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(2, 1, [1, 2, 3]), (1, 2, [1, 2])],
            [(2, 1, "plate"), (1, 2, "edge")])
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": str(tmp_path / "t.msh")},
         "materials": {"s": dict(STEEL)},
         "sections": {"p": {"type": "shell", "material": "s", "thickness": 0.05}},
         "assign": [{"region": "all", "section": "p"}],
         "constraints": [{"region": "edge", "dofs": [1] * 6}],
         "loads": [{"region": "plate", "pressure": 1000.0}],
         "solver": {"mode": "relaxation", "n_steps": 100, "backend": "numba"}}
    prep = api.prepare_case(str(_write(tmp_path, d, "t3")))
    assert "SHELL_EXPERIMENTAL" in {i["code"] for i in prep.preflight["issues"]}
    with pytest.raises(ValueError, match="numpy backend only"):
        _solver(tmp_path, d, "t3s")
