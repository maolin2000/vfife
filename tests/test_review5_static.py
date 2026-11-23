"""Review round 5, the implicit static solver and the numba/NumPy agreement.
One test per finding, each the smallest model that showed it.
"""
import h5py
import numpy as np
import pytest
import yaml

from test_member_failure import _run, _write_msh

E, I = 2.0e11, 1.0e-5
STEEL = {"type": "linear_elastic", "E": E, "nu": 0.3, "rho": 7850.0}
BEAM = {"type": "beam", "material": "s", "shape": "general", "area": 1e-3, "iy": I, "iz": I, "j": 2 * I}
UNITS = {"length": "m", "force": "N", "stress": "Pa"}


def _mon(h5):
    with h5py.File(h5) as f:
        return {k: np.asarray(f["monitors"][k]) for k in f["monitors"]}


def _frames(h5):
    with h5py.File(h5) as f:
        return len(f["steps"])


def _cantilever(tmp_path, name, n=10, L=2.0):
    _write_msh(tmp_path / f"{name}.msh", [(L * i / n, 0, 0) for i in range(n + 1)],
               {"root": (0, [0]), "tip": (0, [n]), "mid": (0, [n // 2]),
                "b": (1, [(i, i + 1) for i in range(n)])})
    return {"units": UNITS, "mesh": {"type": "gmsh", "path": f"{name}.msh"},
            "materials": {"s": STEEL}, "sections": {"b": BEAM},
            "assign": [{"region": "b", "section": "b", "orientation": [0, 1, 0]}],
            "constraints": [{"region": "root", "dofs": [1] * 6}],
            "loads": [{"region": "tip", "vector": [0, 0, -1000.0]}],
            "output": {"fields": ["displacement"], "every": 1,
                       "monitors": {"w": {"kind": "displacement", "region": "tip", "dof": "z"},
                                    "R": {"kind": "reaction", "region": "root", "dof": "z"}}}}


# ---------------------------------------------------------------- 1
@pytest.mark.parametrize("mode", ["static", "relaxation"])
def test_the_solvers_load_steps_stage_a_named_step(tmp_path, mode):
    """`steps: [{name: one}]` under `solver.load_steps: 4`: the step's dict
    carries load_steps = None, and `.get(key, default)` never fell back, so
    the load took `ramp_hold` -- full at the second of four levels."""
    d = _cantilever(tmp_path, f"ls_{mode}")
    d["steps"] = [{"name": "one"}]
    d["solver"] = {"mode": mode, "tolerance": 1e-8, "load_steps": 4}
    if mode == "relaxation":
        d["solver"].update(max_steps_per_load=100000, n_steps=400000)
    out, h5 = _run(tmp_path, f"ls_{mode}", d)
    assert out.status == "CONVERGED"
    assert _mon(h5)["R"] == pytest.approx([250.0, 500.0, 750.0, 1000.0], rel=1e-6)


def test_the_default_ten_static_levels_are_tenths_with_a_steps_block(tmp_path):
    d = _cantilever(tmp_path, "ls10")
    d["steps"] = [{"name": "one"}]
    d["solver"] = {"mode": "static", "tolerance": 1e-8}
    out, h5 = _run(tmp_path, "ls10", d)
    assert _mon(h5)["R"] == pytest.approx(100.0 * np.arange(1, 11), rel=1e-6)


# ---------------------------------------------------------------- 2
def _solver(tmp_path, name, d):
    from vfife.api import prepare_case, _with_element
    from vfife.cli import make_solver
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(dict(d, name=name), sort_keys=False), encoding="utf-8")
    prep = prepare_case(str(p))
    return make_solver(_with_element(prep.case, prep.element, prep.case_path), prep.mesh, prep.material)


@pytest.mark.parametrize("kernel,threads", [("whole", 1), ("whole", 2), ("subset", 1), ("subset", 2)])
def test_the_numba_frame_kernel_subtracts_every_imposed_component(tmp_path, kernel, threads):
    """Creep's history imposes the end twists (local components 3 and 9)
    too; the kernels subtracted 4, 5, 6, 10, 11 only, and a held torque crept
    16-27% short of 1 + phi on numba alone. `whole`: BeamSolver's
    path (every member, `frame_forces_subset` over ids 0..n-1 since v1.19,
    and its threaded twin); `subset`: mixed_line's."""
    pytest.importorskip("numba")
    from vfife.core.beam_solver import BeamSolver
    from vfife.core.rotations import exp_map
    n = 4
    nodes = [(0.5 * i, 0.1 * i * i, 0.05 * i) for i in range(n + 1)]
    _write_msh(tmp_path / "k.msh", nodes, {"a": (0, [0]), "b": (1, [(i, i + 1) for i in range(n)])})
    forces = {}
    for be in ("numpy", "numba"):
        d = {"units": UNITS, "mesh": {"type": "gmsh", "path": "k.msh"},
             "materials": {"s": STEEL}, "sections": {"b": BEAM},
             "assign": [{"region": "b", "section": "b", "orientation": [0, 0, 1]}],
             "constraints": [{"region": "a", "dofs": [1] * 6}], "loads": [],
             "solver": {"mode": "relaxation", "backend": be, "threads": threads, "n_steps": 10},
             "output": {"fields": ["displacement"], "monitors": {}}}
        s = _solver(tmp_path, f"k{kernel}{be}{threads}", d)
        rng = np.random.default_rng(3)
        s.q[:] = 1e-3 * rng.standard_normal(s.ndof)
        w = s.dofs.view(s.q)[:, 3:]
        s.orient[:] = np.einsum("nij,njk->nik", exp_map(w), s.orient)
        s._thermal_d = 1e-4 * rng.standard_normal((s.nel, 12))     # every component
        force = (BeamSolver._member_forces if kernel == "whole" else type(s)._member_forces)
        forces[be] = force(s)[0].copy()
    scale = np.abs(forces["numpy"]).max()
    assert np.abs(forces["numba"] - forces["numpy"]).max() < 1e-12 * scale


def test_a_held_torque_creeps_alike_on_numpy_and_numba(tmp_path):
    pytest.importorskip("numba")
    creep = yaml.safe_load(open(__import__("pathlib").Path(__file__).resolve().parents[1]
                                / "cases" / "case_29_creep_bar.yaml", encoding="utf-8"))["materials"]["concrete"]["creep"]
    n, L = 4, 2.0
    _write_msh(tmp_path / "tc.msh", [(L * i / n, 0, 0) for i in range(n + 1)],
               {"left": (0, [0]), "right": (0, [n]), "bar": (1, [(i, i + 1) for i in range(n)])})
    out = {}
    for be in ("numpy", "numba"):
        d = {"units": UNITS, "mesh": {"type": "gmsh", "path": "tc.msh"},
             "materials": {"c": {"type": "linear_elastic", "E": 3e10, "nu": 0.2, "rho": 2400, "creep": creep}},
             "sections": {"bar": {"type": "beam", "material": "c", "shape": "general", "area": 0.01,
                                  "iy": 8.33e-6, "iz": 8.33e-6, "j": 1.41e-5}},
             "assign": [{"region": "bar", "section": "bar", "orientation": [0, 0, 1]}],
             "constraints": [{"region": "all", "dofs": [1, 1, 1, 0, 1, 1]}, {"region": "left", "dofs": [1] * 6}],
             "loads": [{"region": "right", "moment": [1.0e3, 0, 0]}],
             "steps": [{"name": "load", "age": 28.0, "load_steps": 1, "max_steps_per_load": 20000},
                       {"name": "creep", "to_age": 1028.0, "load_steps": 3, "max_steps_per_load": 20000}],
             "solver": {"mode": "relaxation", "backend": be, "n_steps": 300000, "age": 28.0,
                        "damping": {"type": "kinetic"}, "tolerance": 1e-7},
             "output": {"fields": ["rotation"], "every": 1, "monitors": {}}}
        res, h5 = _run(tmp_path, f"tc_{be}", d)
        with h5py.File(h5) as f:
            out[be] = np.array([f["steps"][k]["rotation"][n, 0] for k in sorted(f["steps"])])
    assert out["numba"] == pytest.approx(out["numpy"], rel=1e-6)
    assert out["numpy"][-1] / out["numpy"][0] == pytest.approx(3.0, rel=2e-3)    # 1 + phi


# ---------------------------------------------------------------- 3
def test_a_simply_supported_girder_bows_under_a_gradient_by_newton(tmp_path):
    """Statically determinate: the answer carries no force, and the old
    relative residual was round-off over round-off -- "a limit point"."""
    n, L = 10, 10.0
    _write_msh(tmp_path / "ss.msh", [(L * i / n, 0, 0) for i in range(n + 1)],
               {"left": (0, [0]), "right": (0, [n]), "mid": (0, [n // 2]), "b": (1, [(i, i + 1) for i in range(n)])})
    d = {"units": UNITS, "mesh": {"type": "gmsh", "path": "ss.msh"},
         "materials": {"c": {"type": "linear_elastic", "E": 3.5e10, "nu": 0.2, "rho": 2500, "thermal_expansion": 1e-5}},
         "sections": {"g": {"type": "beam", "material": "c", "shape": "general", "area": 0.5, "iy": 0.05, "iz": 0.05, "j": 0.1}},
         "assign": [{"region": "b", "section": "g", "orientation": [0, 1, 0]}],
         "constraints": [{"region": "left", "dofs": [1, 1, 1, 1, 0, 0]}, {"region": "right", "dofs": [0, 1, 1, 1, 0, 0]}],
         "loads": [{"region": "all", "temperature": {"gradient": [0.0, 20.0]}}],
         "solver": {"mode": "static", "tolerance": 1e-6, "load_steps": 2},
         "output": {"fields": ["displacement"], "every": 1,
                    "monitors": {"w": {"kind": "displacement", "region": "mid", "dof": "z"}}}}
    out, h5 = _run(tmp_path, "ss", d)
    assert out.status == "CONVERGED"
    assert abs(_mon(h5)["w"][-1]) == pytest.approx(1e-5 * 20 * L ** 2 / 8, rel=1e-3)


@pytest.mark.parametrize("section,law", [("beam", "elastic"), ("fibre", "elastic"), ("fibre", "steel02")])
def test_a_support_released_in_a_later_step_lets_the_member_spring_back(tmp_path, section, law):
    n = 4
    _write_msh(tmp_path / "rel.msh", [(i / n, 0, 0) for i in range(n + 1)],
               {"root": (0, [0]), "tip": (0, [n]), "b": (1, [(i, i + 1) for i in range(n)])})
    secs = {"fibre": {"type": "fiber", "material": "m", "shape": "rect", "width": 0.1, "height": 0.1, "ny": 1, "nz": 20},
            "beam": {"type": "beam", "material": "m", "shape": "rect", "width": 0.1, "height": 0.1}}
    mats = {"elastic": STEEL, "steel02": {"type": "steel02", "E": 2.0e11, "Fy": 3.5e8, "b": 0.02, "R0": 20.0, "rho": 7850}}
    d = {"units": UNITS, "mesh": {"type": "gmsh", "path": "rel.msh"},
         "materials": {"m": mats[law]}, "sections": {"s": secs[section]},
         "assign": [{"region": "b", "section": "s", "orientation": [0, 1, 0]}],
         "constraints": [{"region": "root", "dofs": [1] * 6}, {"region": "all", "dofs": [0, 1, 0, 1, 0, 1]}],
         "loads": [],
         "steps": [{"name": "drive", "load_steps": 4,
                    "constraints": [{"region": "tip", "dofs": [0, 0, 1, 0, 0, 0], "value": -0.03}]},
                   {"name": "release", "load_steps": 2}],
         "solver": {"backend": "numpy", "mode": "static", "tolerance": 1e-6},
         "output": {"fields": ["displacement"], "every": 1,
                    "monitors": {"w": {"kind": "displacement", "region": "tip", "dof": "z"},
                                 "R": {"kind": "reaction", "region": "root", "dof": "z"}}}}
    out, h5 = _run(tmp_path, f"rel_{section}_{law}", d)
    assert out.status == "CONVERGED", out.message
    m = _mon(h5)
    assert abs(m["R"][-1]) < 1e-6 * abs(m["R"][3])                 # nothing left to hold
    if law == "elastic":
        assert abs(m["w"][-1]) < 1e-9                               # straight again
    else:
        assert -0.03 < m["w"][-1] < -0.005                          # a plastic set remains


def test_a_tolerance_below_round_off_is_named_not_called_a_limit_point(tmp_path):
    d = _cantilever(tmp_path, "ro")
    d["solver"] = {"mode": "static", "tolerance": 1e-12, "load_steps": 2}
    out, _h5 = _run(tmp_path, "ro", d)
    assert out.status == "MAX_STEPS"
    assert "round-off" in out.message and "not a limit point" in out.message


# ---------------------------------------------------------------- 4
@pytest.mark.parametrize("every", [None, 3])
def test_static_monitors_are_written_at_every_increment(tmp_path, every):
    d = _cantilever(tmp_path, f"ev{every}")
    d["solver"] = {"mode": "static", "tolerance": 1e-8, "load_steps": 10}
    if every is None:
        d["output"].pop("every")                       # the default, 200
    else:
        d["output"]["every"] = every
    out, h5 = _run(tmp_path, f"ev{every}", d)
    m = _mon(h5)
    assert len(m["time"]) == len(m["w"]) == len(m["residual"]) == 10
    assert m["R"] == pytest.approx(100.0 * np.arange(1, 11), rel=1e-6)
    # without `every` a static run frames every increment (v1.0: the
    # default 200 counted increments and wrote only the last); 3, 6, 9 and the last
    assert _frames(h5) == (10 if every is None else 4)


# ---------------------------------------------------------------- 5
def test_a_steps_own_tolerance_is_honoured_by_newton(tmp_path):
    L = 2.0
    M = 0.8 * np.pi / 2 * E * I / L                    # a large rotation: several iterations
    d = _cantilever(tmp_path, "stol", n=10, L=L)
    d["loads"] = []
    d["steps"] = [{"name": "s1", "load_steps": 2, "tolerance": 0.3,
                   "loads": [{"region": "tip", "moment": [0, -M, 0]}]}]
    d["solver"] = {"mode": "static", "tolerance": 1e-10}
    out, h5 = _run(tmp_path, "stol", d)
    res = _mon(h5)["residual"]
    assert out.status == "CONVERGED"
    assert np.all(res <= 0.3) and np.all(res > 1e-8)    # stopped at the step's, not the solver's


# ---------------------------------------------------------------- 6
def _von_mises(tmp_path, name, static, every=1):
    a, h, EA = 1.0, 0.1, 1.0e6
    _write_msh(tmp_path / f"{name}.msh", [(-a, 0, 0), (0, 0, h), (a, 0, 0)],
               {"sup": (0, [0, 2]), "apex": (0, [1]), "bars": (1, [(0, 1), (1, 2)])})
    d = {"units": UNITS, "mesh": {"type": "gmsh", "path": f"{name}.msh"},
         "materials": {"s": {"type": "linear_elastic", "E": EA / 1e-3, "nu": 0.3, "rho": 7850}},
         "sections": {"b": {"type": "bar", "material": "s", "shape": "general", "area": 1e-3}},
         "assign": [{"region": "bars", "section": "b"}],
         "constraints": [{"region": "sup", "dofs": [1] * 6}, {"region": "apex", "dofs": [0, 1, 0, 1, 1, 1]}],
         "loads": [{"region": "apex", "vector": [0, 0, -1.5 * 381.087]}],
         "solver": {"mode": "static", "tolerance": 1e-9, "static": static},
         "output": {"fields": ["displacement"], "every": every,
                    "monitors": {"w": {"kind": "displacement", "region": "apex", "dof": "z"}}}}
    out, h5 = _run(tmp_path, name, d)
    return out, h5


def test_a_modified_tangent_under_arc_length_is_reused(tmp_path):
    arc = {"initial": 0.02, "max_increments": 400}
    full, h_full = _von_mises(tmp_path, "vmf", {"arc_length": arc})
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod, h_mod = _von_mises(tmp_path, "vmm", {"arc_length": arc, "tangent": "modified"})
    assert full.status == mod.status == "CONVERGED"
    line = [l for l in buf.getvalue().splitlines() if "Newton iterations" in l][-1]
    words = line.split()
    its, tans = int(words[words.index("Newton") - 1]), int(words[words.index("tangents,") - 1])
    assert tans < its                                  # it was one tangent per iteration
    assert _mon(h_mod)["w"][-1] == pytest.approx(_mon(h_full)["w"][-1], rel=1e-6)


def test_output_every_thins_arc_length_frames_not_monitors(tmp_path):
    arc = {"initial": 0.02, "max_increments": 400}
    out1, h1 = _von_mises(tmp_path, "vm1", {"arc_length": arc}, every=1)
    out5, h5 = _von_mises(tmp_path, "vm5", {"arc_length": arc}, every=5)
    m1, m5 = _mon(h1), _mon(h5)
    assert len(m5["time"]) == len(m1["time"]) == _frames(h1)
    assert np.array_equal(m5["w"], m1["w"])
    n = len(m1["time"])
    assert _frames(h5) == n // 5 + (1 if n % 5 else 0)  # every fifth, and the last


# ---------------------------------------------------------------- 8
def _preflight(tmp_path, d, name="pf"):
    from vfife import api
    return api.prepare_case(api.CaseDraft(dict(d, name=name), base=tmp_path)).preflight


def test_settings_that_do_nothing_are_warned_by_name(tmp_path):
    d = _cantilever(tmp_path, "pf")
    d["solver"] = {"mode": "static", "damping": {"type": "rayleigh", "alpha": 5.0}}
    codes = [i["code"] for i in _preflight(tmp_path, d)["issues"]]
    assert "STATIC_IGNORES_DAMPING" in codes
    d["solver"] = {"mode": "relaxation", "n_steps": 100, "static": {"arc_length": {"initial": 0.1}}}
    codes = [i["code"] for i in _preflight(tmp_path, d)["issues"]]
    assert "STATIC_BLOCK_IGNORED" in codes
    d["solver"] = {"mode": "static", "static": {"line_search": True, "arc_length": {"initial": 0.1}}}
    codes = [i["code"] for i in _preflight(tmp_path, d)["issues"]]
    assert "ARC_NO_LINE_SEARCH" in codes
    d["solver"] = {"mode": "static", "static": {"arc_length": {"initial": 0.1}}}
    codes = [i["code"] for i in _preflight(tmp_path, d)["issues"]]
    assert not {"STATIC_IGNORES_DAMPING", "STATIC_BLOCK_IGNORED", "ARC_NO_LINE_SEARCH"} & set(codes)


def test_a_static_run_is_summarised_as_static(tmp_path):
    d = _cantilever(tmp_path, "sm")
    d["solver"] = {"mode": "static", "load_steps": 1}
    s = _solver(tmp_path, "sm", d)
    line = [l for l in s.summary().splitlines() if l.strip().startswith("mode")][0]
    assert "static" in line and "relaxation" not in line


# ---------------------------------------------------------------- 9
def test_fibre_laws_without_a_scalar_twin_are_not_offered_on_numba(tmp_path):
    from vfife.catalog import compatibility
    for law in ("linear_elastic", "rebar", "piecewise_uniaxial"):
        ok, why = compatibility("beam2_fiber", law, backend="numba")
        # the twin is the reason only where nothing else refuses first
        assert not ok and any("scalar" in w or "backend numba" in w for w in why), law
    for law in ("steel02", "concrete02", "concrete01", "steel_cyclic"):
        assert compatibility("beam2_fiber", law, backend="numba")[0], law
    assert compatibility("beam2_fiber", "linear_elastic", backend="numpy")[0]
    _write_msh(tmp_path / "cap.msh", [(0, 0, 0), (0.5, 0, 0), (1, 0, 0)],
               {"root": (0, [0]), "tip": (0, [2]), "b": (1, [(0, 1), (1, 2)])})
    d = {"units": UNITS, "mesh": {"type": "gmsh", "path": "cap.msh"},
         "materials": {"m": STEEL},
         "sections": {"f": {"type": "fiber", "material": "m", "shape": "rect", "width": 0.1, "height": 0.1,
                            "ny": 1, "nz": 4}},
         "assign": [{"region": "b", "section": "f", "orientation": [0, 1, 0]}],
         "constraints": [{"region": "root", "dofs": [1] * 6}], "loads": [{"region": "tip", "vector": [0, 0, -1e3]}],
         "solver": {"backend": "numba", "mode": "static"}, "output": {"fields": ["displacement"], "monitors": {}}}
    rep = _preflight(tmp_path, d, "cap")
    assert rep["status"] == "FAIL" and any(i["code"] == "CAPABILITY_MISMATCH" for i in rep["issues"])


# =============================================================== 1.0 re-review
# The v1.0 fixes, reviewed again: one test per finding.

def _string(tmp_path, name):
    """A 10 m pinned-pinned member with almost no bending stiffness: a string.
    10 kN at midspan in step 1, removed, and 10 N in step 2. Its linear
    predictor from the straight state is metres long, with forces of 1e12 N."""
    n, L = 10, 10.0
    _write_msh(tmp_path / f"{name}.msh", [(L * i / n, 0, 0) for i in range(n + 1)],
               {"left": (0, [0]), "right": (0, [n]), "mid": (0, [n // 2]),
                "b": (1, [(i, i + 1) for i in range(n)])})
    pin = [0, 1, 0, 1, 0, 1]
    return {"units": UNITS, "mesh": {"type": "gmsh", "path": f"{name}.msh"},
            "materials": {"s": STEEL},
            "sections": {"b": dict(BEAM, iy=1e-10, iz=1e-10, j=2e-10)},
            "assign": [{"region": "b", "section": "b", "orientation": [0, 1, 0]}],
            "constraints": [{"region": "left", "dofs": [1, 1, 1, 1, 0, 1]},
                            {"region": "right", "dofs": [1, 1, 1, 1, 0, 1]},
                            {"region": "all", "dofs": pin}],
            "steps": [{"name": "big", "load_steps": 1, "loads": [{"region": "mid", "vector": [0, 0, -1e4]}]},
                      {"name": "small", "load_steps": 1, "loads": [{"region": "mid", "vector": [0, 0, -10.0]}]}],
            "solver": {"mode": "static"},                     # the default tolerance, 1e-4
            "output": {"fields": ["displacement"], "every": 1,
                       "monitors": {"wm": {"kind": "displacement", "region": "mid", "dof": "z"},
                                    "R": {"kind": "reaction", "region": "left", "dof": "z"}}}}


def test_the_residual_floor_is_raised_by_accepted_states_only(tmp_path):
    """The floor (a millionth of the peak force) was raised by every Newton
    iterate: the string's predictor put 9.8e11 N in it, and step 2 was
    accepted with 82 N out of balance on its 10 N -- R = 41.7 N, twice the
    deflection. Against the same run at tolerance 1e-9."""
    out, h5 = _run(tmp_path, "str", _string(tmp_path, "str"))
    tight = _string(tmp_path, "strt")
    tight["solver"]["tolerance"] = 1e-9
    out_t, h5_t = _run(tmp_path, "strt", tight)
    assert out.status == out_t.status == "CONVERGED"
    m, t = _mon(h5), _mon(h5_t)
    assert m["R"][-1] == pytest.approx(5.0, rel=1e-3)
    assert m["wm"][-1] == pytest.approx(t["wm"][-1], rel=1e-3)


def test_a_static_restart_carries_the_residual_floor(tmp_path):
    """`_force_scale` (and relaxation's `_motion_scale`) were not in the
    checkpoint: a restart judged its later steps on another scale."""
    from vfife import api
    from vfife.core.diagnostics import AnalysisInterrupted
    d = _string(tmp_path, "rs")
    p = tmp_path / "rs.yaml"
    p.write_text(yaml.safe_dump(dict(d, name="rs"), sort_keys=False), encoding="utf-8")
    got = {}
    api.run_case(p, output=str(tmp_path / "full.h5"), progress_every=1,
                 on_ready=lambda s, _p: got.__setitem__("full", s))
    with pytest.raises(AnalysisInterrupted):
        api.run_case(p, output=str(tmp_path / "half.h5"), progress_every=1,
                     on_ready=lambda s, _p: got.__setitem__("half", s),
                     on_progress=lambda *a: got["half"].step < 1)
    api.run_case(p, output=str(tmp_path / "rest.h5"), restart=str(tmp_path / "half.h5"), progress_every=1,
                 on_ready=lambda s, _p: got.__setitem__("rest", s))
    a, b = got["full"].checkpoint(), got["rest"].checkpoint()
    assert "scales" in a and a["scales"][0] > 0
    for k in a:
        if np.asarray(a[k]).dtype.kind not in "SUO":
            assert np.array_equal(a[k], b[k]), k


def test_a_modified_tangent_is_not_stopped_while_it_still_converges(tmp_path):
    """Corrections of 1e-12 m on a 2 m cantilever, the residual falling
    linearly: the stall detector called it the arithmetic's floor at 2.8e-10
    on its way to 1e-10 (the old solver converged there)."""
    L = 2.0
    d = _cantilever(tmp_path, "modt", n=10, L=L)
    d["loads"] = [{"region": "tip", "moment": [0, -0.8 * np.pi / 2 * E * I / L, 0]}]
    d["solver"] = {"mode": "static", "tolerance": 1e-10, "load_steps": 2,
                   "static": {"tangent": "modified", "max_iterations": 200}}
    out, _h5 = _run(tmp_path, "modt", d)
    assert out.status == "CONVERGED", out.message


def test_round_off_is_judged_by_the_distance_from_the_origin(tmp_path):
    """A 2 m cantilever at UTM coordinates carries 7e-10 m of coordinate
    round-off; it was judged against 1e-11 x 2 m and called a limit point."""
    n, L = 10, 2.0
    x0, y0 = 5.0e5, 3.0e6
    _write_msh(tmp_path / "utm.msh", [(x0 + L * i / n, y0, 0) for i in range(n + 1)],
               {"root": (0, [0]), "tip": (0, [n]), "b": (1, [(i, i + 1) for i in range(n)])})
    d = {"units": UNITS, "mesh": {"type": "gmsh", "path": "utm.msh"},
         "materials": {"s": STEEL}, "sections": {"b": BEAM},
         "assign": [{"region": "b", "section": "b", "orientation": [0, 1, 0]}],
         "constraints": [{"region": "root", "dofs": [1] * 6}],
         "loads": [{"region": "tip", "moment": [0, -0.8 * np.pi / 2 * E * I / L, 0]}],
         "solver": {"mode": "static", "tolerance": 1e-11, "load_steps": 2},
         "output": {"fields": ["displacement"], "monitors": {}}}
    out, _h5 = _run(tmp_path, "utm", d)
    assert out.status == "MAX_STEPS"
    assert "round-off" in out.message and "not a limit point" in out.message


def test_load_control_past_a_limit_point_ends(tmp_path):
    """The two-bar truss at 1.5 times its limit load under load control: the
    increment shrank across successes to 4e-17, lam + dl was lam, and the
    run never ended. It ends at the smallest increment `cutbacks` allows."""
    import time
    t0 = time.perf_counter()
    out, h5 = _von_mises(tmp_path, "vmlc", {})
    assert out.status == "MAX_STEPS" and "limit point" in out.message
    assert time.perf_counter() - t0 < 60
    assert len(_mon(h5)["w"]) < 100


def test_line_search_under_arc_length_is_noted_only_where_it_does_little(tmp_path):
    d = _cantilever(tmp_path, "pfa")
    d["steps"] = [{"name": "dead"}, {"name": "live"}]
    d["solver"] = {"mode": "static", "static": {"line_search": True,
                                                "arc_length": {"initial": 0.1, "steps": ["live"]}}}
    codes = [i["code"] for i in _preflight(tmp_path, d, "pfa")["issues"]]
    assert "ARC_NO_LINE_SEARCH" not in codes          # the dead step is load-controlled
    d["solver"]["static"]["arc_length"]["steps"] = ["dead", "live"]
    d["solver"]["static"]["line_search"] = False
    issues = [i for i in _preflight(tmp_path, d, "pfb")["issues"] if i["code"] == "ARC_NO_LINE_SEARCH"]
    assert len(issues) == 1 and issues[0]["severity"] == "INFO"
    assert "load factor 0" in issues[0]["message"] and "load factor 1" in issues[0]["message"]


def test_load_steps_zero_is_refused(tmp_path):
    from vfife.io.schema import SolverSpec, StepSpec
    with pytest.raises(Exception, match="greater than or equal to 1"):
        StepSpec(load_steps=0)
    with pytest.raises(Exception, match="greater than or equal to 1"):
        SolverSpec(mode="relaxation", load_steps=0)
