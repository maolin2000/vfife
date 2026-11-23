"""The implicit static solver as a second reader found it.

Each test is one defect a review of the static solver (v1.0) reproduced, pinned so it
stays fixed: the default levels, a fibre that ruptures at a commit, a
temperature inside an arc-length step, an arc pattern that moves nothing,
an arc that runs out of increments, a misspelt step name, and a stop the
step starts beyond.
"""
import h5py
import numpy as np
import pytest

from test_member_failure import _run, _write_msh

E, I = 2.0e11, 1.0e-5
BEAM = {"type": "beam", "material": "s", "shape": "general", "area": 1e-3, "iy": I, "iz": I, "j": 2 * I}


def _steel(**extra):
    return dict({"type": "linear_elastic", "E": E, "nu": 0.3, "rho": 7850.0}, **extra)


def _cant(tmp_path, name, loads, solver, n=10, L=2.0, constraints=None, monitors=None,
          steps=None, material=None):
    _write_msh(tmp_path / f"{name}.msh", [(L * i / n, 0, 0) for i in range(n + 1)],
               {"root": (0, [0]), "tip": (0, [n]), "mid": (0, [n // 2]),
                "b": (1, [(i, i + 1) for i in range(n)])})
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": f"{name}.msh"},
         "materials": {"s": material or _steel()}, "sections": {"b": BEAM},
         "assign": [{"region": "b", "section": "b", "orientation": [0, 1, 0]}],
         "constraints": constraints or [{"region": "root", "dofs": [1] * 6}],
         "loads": loads, "solver": solver,
         "output": {"fields": ["displacement"], "every": 1,
                    "monitors": monitors or {"w": {"kind": "displacement", "region": "tip", "dof": "z"}}}}
    if steps is not None:
        d["steps"] = steps
    out, h5 = _run(tmp_path, name, d)
    m = {}
    if h5 is not None:
        with h5py.File(h5) as f:
            m = {k: np.asarray(f["monitors"][k]) for k in f["monitors"]}
    return m, out


def test_without_load_steps_the_ten_levels_are_tenths_of_the_load(tmp_path):
    """Unset, a step was UNSTAGED to `step_data`: its load took the
    ramp_hold amplitude (full at 0.5), and the ten levels put it all on in
    five and repeated it, labelled 0.6 ... 1.0."""
    P = 1.0e3
    m, out = _cant(tmp_path, "lin", [{"region": "tip", "vector": [0, 0, -P]}],
                   {"mode": "static", "tolerance": 1e-8})
    w_exact = -P * 2.0 ** 3 / (3 * E * I)
    assert out.status == "CONVERGED"
    assert m["time"] == pytest.approx(np.linspace(0.1, 1.0, 10))
    assert m["w"] / w_exact == pytest.approx(np.linspace(0.1, 1.0, 10), rel=1e-5)


def test_a_fibre_that_ruptures_at_a_commit_is_re_solved_not_recorded_unbalanced(tmp_path):
    n = 4
    _write_msh(tmp_path / "rup.msh", [(i / n, 0, 0) for i in range(n + 1)],
               {"root": (0, [0]), "tip": (0, [n]), "b": (1, [(i, i + 1) for i in range(n)])})
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": "rup.msh"},
         "materials": {"st": {"type": "steel02", "E": 2.0e11, "Fy": 3.5e8, "b": 0.02, "R0": 20.0,
                              "rho": 7850, "rupture_strain": 0.006}},
         "sections": {"f": {"type": "fiber", "material": "st", "shape": "rect", "width": 0.1,
                            "height": 0.1, "ny": 1, "nz": 20, "integration": "lobatto3"}},
         "assign": [{"region": "b", "section": "f", "orientation": [0, 1, 0]}],
         "constraints": [{"region": "root", "dofs": [1] * 6}, {"region": "all", "dofs": [0, 1, 0, 1, 0, 1]}],
         "loads": [],
         "steps": [{"name": "push", "load_steps": 8,
                    "constraints": [{"region": "tip", "dofs": [0, 0, 1, 0, 0, 0], "value": -0.04}]}],
         "solver": {"backend": "numpy", "mode": "static", "tolerance": 1e-6},
         "output": {"fields": ["displacement"], "every": 1,
                    "monitors": {"R": {"kind": "reaction", "region": "root", "dof": "z"},
                                 "Rt": {"kind": "reaction", "region": "tip", "dof": "z"}}}}
    out, h5 = _run(tmp_path, "rup", d)
    assert out.status == "CONVERGED"
    with h5py.File(h5) as f:
        R, Rt = np.asarray(f["monitors/R"]), np.asarray(f["monitors/Rt"])
    # a free cantilever driven at its tip: root and tip reactions balance at
    # every recorded level, the ruptured ones included (it was 70 kN out)
    assert np.abs(R + Rt).max() < 1e-4 * np.abs(R).max()
    assert R.max() > 1.05 * R[-1]                 # it did rupture and lose load


def test_a_temperature_in_an_arc_length_step_follows_the_load_factor(tmp_path):
    """`_force` refreshes the imposed deformation as a side effect; an arc
    step built F(1) last and kept the full temperature from lambda = 0."""
    cons = [{"region": "root", "dofs": [1] * 6}, {"region": "tip", "dofs": [1, 0, 0, 0, 0, 0]}]
    mons = {"Rx": {"kind": "reaction", "region": "tip", "dof": "x"}}
    loads = [{"region": "all", "temperature": {"uniform": 50.0}}, {"region": "mid", "vector": [0, 0, -10.0]}]
    m, out = _cant(tmp_path, "arcT", loads,
                   {"mode": "static", "tolerance": 1e-8, "load_steps": 4,
                    "static": {"arc_length": {"initial": 0.25}}},
                   n=4, constraints=cons, monitors=mons, material=_steel(thermal_expansion=1.2e-5))
    assert out.status == "CONVERGED"
    full = E * 1e-3 * 1.2e-5 * 50.0
    # the restraint force grows with the load factor, and is full at 1
    assert m["Rx"] == pytest.approx(-full * m["time"], rel=1e-3)


def test_an_arc_pattern_that_moves_nothing_free_runs_by_load_control(tmp_path):
    """A prescribed tip rotation only: the arc had length zero, 'converged'
    at lambda = 0 for max_increments and reported CONVERGED with the tip
    where it started."""
    cons = [{"region": "root", "dofs": [1] * 6},
            {"region": "tip", "dofs": [0, 0, 0, 0, 1, 0], "value": 0.1}]
    base = {"mode": "static", "tolerance": 1e-8, "load_steps": 4}
    lc, _ = _cant(tmp_path, "lc", [], base, constraints=cons)
    arc, out = _cant(tmp_path, "arc", [], dict(base, static={"arc_length": {"initial": 0.25}}),
                     constraints=cons)
    assert out.status == "CONVERGED"
    assert arc["w"][-1] == pytest.approx(lc["w"][-1], rel=1e-9) and abs(arc["w"][-1]) > 0.05


def test_running_out_of_arc_increments_is_a_failure_not_an_answer(tmp_path):
    P = 1.0e3
    m, out = _cant(tmp_path, "short", [{"region": "tip", "vector": [0, 0, -P]}],
                   {"mode": "static", "tolerance": 1e-8,
                    "static": {"arc_length": {"initial": 0.01, "max_increments": 3}}})
    assert out.status == "MAX_STEPS"
    assert "max_increments" in out.message


def test_a_misspelt_arc_step_name_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no step is called"):
        _cant(tmp_path, "typo", [], {"mode": "static", "static": {"arc_length": {"steps": ["snpa"]}}},
              steps=[{"name": "snap", "load_steps": 2,
                      "loads": [{"region": "tip", "vector": [0, 0, -1.0e3]}]}])


def test_a_stop_is_a_crossing_from_where_the_step_starts(tmp_path):
    """A settle step leaves the tip below the stop value; the arc step that
    follows used to stop after one increment ('already past'). The stop now
    means crossing the value from the side the step starts on -- and a stop
    on anything but the last step is refused, since the next step would
    start from its loads held at their full value."""
    P = 1.0e3
    w1 = -P * 2.0 ** 3 / (3 * E * I)
    m, out = _cant(tmp_path, "cross", [{"region": "tip", "vector": [0, 0, -P]}],
                   {"mode": "static", "tolerance": 1e-8, "load_steps": 1,
                    "static": {"arc_length": {"initial": 0.1, "steps": ["more"],
                                              "stop": {"region": "tip", "dof": "z", "value": 2.5 * w1}}}},
                   steps=[{"name": "settle"},
                          {"name": "more", "loads": [{"region": "tip", "vector": [0, 0, -2 * P]}]}])
    assert out.status == "CONVERGED"
    assert m["w"][0] == pytest.approx(w1, rel=1e-6)
    assert m["w"][-1] <= 2.5 * w1 < m["w"][-2]          # stopped where it crossed, not at once
    with pytest.raises(ValueError, match="only\\s+apply to the last step"):
        _cant(tmp_path, "early", [{"region": "tip", "vector": [0, 0, -P]}],
              {"mode": "static", "static": {"arc_length": {"stop": {"region": "tip", "dof": "z", "value": -1.0}}}},
              steps=[{"name": "a", "load_steps": 2}, {"name": "b", "load_steps": 2}])
