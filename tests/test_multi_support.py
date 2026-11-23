"""Multiple-support excitation: a constraint with a
`record` makes its region follow its own ground motion, in absolute
coordinates.

Three things are checked against something that is not the new code:

* shaken the same way at every support, the motion RELATIVE to the ground
  is what the old uniform `ground_motion` load (relative formulation) gives;
* a slow differential settlement d of one end of a fixed-fixed beam gives
  the textbook reactions 12 E I d / L^3, and the midspan moves d / 2;
* a travelling wave reaches a node (x - origin) . e / c late.
"""
import h5py
import numpy as np
import pytest

from test_member_failure import _run, _write_msh

E = 2.0e11
STEEL = {"type": "linear_elastic", "E": E, "nu": 0.3, "rho": 7850.0}
A, I = 1.0e-2, 1.3e-5
SEC = {"type": "beam", "material": "steel", "shape": "general", "area": A, "iy": I, "iz": I, "j": 2 * I}


def _pulse(tmp_path, name, amp=2.0, f=3.0, t_end=None, n=4001):
    """One full sine of ground acceleration, A sin(w t) for t in [0, 1/f],
    and the displacement it integrates to exactly (zero initial velocity):
    u = A/w (t - sin(w t)/w), ending at rest, displaced by A/(w f)."""
    w = 2 * np.pi * f
    T = 1.0 / f
    t = np.linspace(0.0, T, n)
    acc = amp * np.sin(w * t)
    disp = amp / w * (t - np.sin(w * t) / w)
    a_path = tmp_path / f"{name}_acc.txt"
    u_path = tmp_path / f"{name}_disp.txt"
    np.savetxt(a_path, np.column_stack([t, acc]), encoding="utf-8")
    np.savetxt(u_path, np.column_stack([t, disp]), encoding="utf-8")
    return str(a_path), str(u_path), amp / (w * f)


def _column(tmp_path, name, base, loads, total=0.8, damping=None):
    """A 2 m cantilever column of four members with 500 kg on top, shaken
    along x."""
    H = 2.0
    nodes = [(0, 0, H * i / 4) for i in range(5)]
    _write_msh(tmp_path / f"{name}.msh", nodes,
               {"base": (0, [0]), "top": (0, [4]), "col": (1, [(i, i + 1) for i in range(4)])})
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": f"{name}.msh"},
        "materials": {"steel": STEEL},
        "sections": {"c": SEC},
        "assign": [{"region": "col", "section": "c", "orientation": [1, 0, 0]}],
        "nodal_masses": [{"region": "top", "mass": 500.0}],
        "constraints": base if isinstance(base, list) else [base],
        "loads": loads,
        "solver": {"backend": "numpy", "mode": "dynamic", "total_time": total},
        "output": {"fields": ["displacement"], "every": 50,
                   "monitors": {"rel": {"kind": "displacement", "region": "top", "dof": "x",
                                        "relative_to": "base"},
                                "base": {"kind": "displacement", "region": "base", "dof": "x"},
                                "shear": {"kind": "reaction", "region": "base", "dof": "x"}}},
    }
    if damping:
        d["solver"]["damping"] = damping
    out, h5 = _run(tmp_path, name, d)
    with h5py.File(h5) as f:
        m = {k: np.asarray(f["monitors"][k]) for k in f["monitors"]}
        g = f["diagnostics"]
        energy = {k: np.asarray(g[k]) for k in ("external_work", "internal_work", "kinetic_energy",
                                                "dissipated_energy", "relative_energy_error")}
    return m, out, energy


def test_the_same_record_at_every_support_is_the_uniform_ground_motion(tmp_path):
    a_path, u_path, _final = _pulse(tmp_path, "p")
    rel, _o, _e = _column(tmp_path, "rel", {"region": "base", "dofs": [1] * 6},
                          [{"ground_motion": {"path": a_path, "direction": [1, 0, 0]}}])
    ab, _o, _e = _column(tmp_path, "abs",
                         [{"region": "base", "dofs": [1] * 6},
                          {"region": "base", "dofs": [1, 0, 0, 0, 0, 0],
                           "record": {"path": u_path, "quantity": "displacement"}}], [])
    assert np.allclose(rel["time"], ab["time"])
    peak = np.abs(rel["rel"]).max()
    assert peak > 1e-3                                   # it was shaken
    # the relative motion: the same history, to the one-step lag the
    # uniform load has always had (its force is read at the step's end)
    assert np.abs(ab["rel"] - rel["rel"]).max() < 0.02 * peak
    # the base followed the record, and the reaction is the column's shear
    t = ab["time"]
    w = 2 * np.pi * 3.0
    ug = np.where(t < 1 / 3.0, 2.0 / w * (t - np.sin(w * t) / w), 2.0 / (w * 3.0))
    assert ab["base"] == pytest.approx(ug, abs=1e-8)        # the record, linearly interpolated
    assert np.abs(ab["shear"]).max() == pytest.approx(np.abs(rel["shear"]).max(), rel=0.03)


def test_an_acceleration_record_at_the_support_is_integrated_to_the_same_motion(tmp_path):
    a_path, u_path, final = _pulse(tmp_path, "q")
    clamp = {"region": "base", "dofs": [1] * 6}
    by_u, _o, _e = _column(tmp_path, "by_u", [clamp, {"region": "base", "dofs": [1, 0, 0, 0, 0, 0],
                                                      "record": {"path": u_path}}], [])
    by_a, _o, _e = _column(tmp_path, "by_a", [clamp, {"region": "base", "dofs": [1, 0, 0, 0, 0, 0],
                                                      "record": {"path": a_path,
                                                                 "quantity": "acceleration"}}], [])
    # the support's own integration of the acceleration arrives at the same
    # place, to first order in the step
    assert by_a["base"][-1] == pytest.approx(final, rel=1e-2)
    peak = np.abs(by_u["rel"]).max()
    assert np.abs(by_a["rel"] - by_u["rel"]).max() < 0.02 * peak


def test_a_slow_settlement_of_one_end_gives_the_textbook_reactions(tmp_path):
    # the right end sinks 10 mm in y over 1 s (a smooth ramp, long against the
    # beam's 0.1 s period) and stays there; heavy mass damping settles it
    t = np.linspace(0.0, 1.0, 2001)
    dy = 0.01
    s = t - np.sin(2 * np.pi * t) / (2 * np.pi)                   # 0 -> 1, at rest at both ends
    path = tmp_path / "settle.txt"
    np.savetxt(path, np.column_stack([t, dy * s]), encoding="utf-8")
    # the usual clamp, and a record on the one dof that moves: the record
    # overrides the clamp's zero there
    right = [{"region": "right", "dofs": [1] * 6},
             {"region": "right", "dofs": [0, 1, 0, 0, 0, 0], "record": {"path": str(path)}}]
    m, out, L = _settle(tmp_path, right)
    k = 12 * E * I / L ** 3
    assert m["mid"][-1] == pytest.approx(dy / 2, rel=2e-3)
    assert m["rr"][-1] == pytest.approx(k * dy, rel=5e-3)
    assert m["rl"][-1] == pytest.approx(-k * dy, rel=5e-3)


def _settle(tmp_path, constraints):
    L = 4.0
    n = 8
    nodes = [(L * i / n, 0, 0) for i in range(n + 1)]
    _write_msh(tmp_path / "s2.msh", nodes,
               {"left": (0, [0]), "right": (0, [n]), "mid": (0, [n // 2]),
                "beam": (1, [(i, i + 1) for i in range(n)])})
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": "s2.msh"},
        "materials": {"steel": STEEL},
        "sections": {"c": SEC},
        "assign": [{"region": "beam", "section": "c", "orientation": [0, 0, 1]}],
        "constraints": [{"region": "left", "dofs": [1] * 6}] + constraints,
        "loads": [],
        "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 2.5,
                   "damping": {"type": "viscous", "coeff": 60.0}},
        "output": {"fields": ["displacement"], "every": 500,
                   "monitors": {"mid": {"kind": "displacement", "region": "mid", "dof": "y"},
                                "rl": {"kind": "reaction", "region": "left", "dof": "y"},
                                "rr": {"kind": "reaction", "region": "right", "dof": "y"}}},
    }
    out, h5 = _run(tmp_path, "s2", d)
    with h5py.File(h5) as f:
        m = {k: np.asarray(f["monitors"][k]) for k in f["monitors"]}
    return m, out, L


@pytest.mark.parametrize("damping", [None, {"type": "rayleigh", "alpha": 0.0, "beta": 2e-4}])
def test_the_energy_ledger_closes_with_the_ground_doing_the_work(tmp_path, damping):
    """The ground does work on the structure through the reactions of the
    driven supports; the ledger books fint + damp there and leaves the
    support particle's own kinetic energy to the ground. Without that the
    external work of a record-driven run is zero and every joule of the
    response is booked as error."""
    _a, u_path, _f = _pulse(tmp_path, "e")
    _m, _o, en = _column(tmp_path, "e", [{"region": "base", "dofs": [1] * 6},
                                         {"region": "base", "dofs": [1, 0, 0, 0, 0, 0],
                                          "record": {"path": u_path}}], [], damping=damping)
    assert en["external_work"].max() > 1.0                   # joules went in
    assert en["relative_energy_error"].max() < 5e-3
    if damping:
        assert en["dissipated_energy"][-1] > 0.0


def test_a_travelling_wave_reaches_each_support_late_by_its_distance(tmp_path):
    """Both ends of a 4 m beam follow one record, the wave running along +x
    at 200 m/s: the far end moves 20 ms after the near one, and the member
    between them is strained by the difference."""
    L, c = 4.0, 200.0
    _a, u_path, _f = _pulse(tmp_path, "w", f=10.0)
    n = 8
    nodes = [(L * i / n, 0, 0) for i in range(n + 1)]
    _write_msh(tmp_path / "w.msh", nodes,
               {"ends": (0, [0, n]), "left": (0, [0]), "right": (0, [n]),
                "beam": (1, [(i, i + 1) for i in range(n)])})
    wave = {"speed": c, "direction": [1, 0, 0], "origin": [0, 0, 0]}
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": "w.msh"},
        "materials": {"steel": STEEL},
        "sections": {"c": SEC},
        "assign": [{"region": "beam", "section": "c", "orientation": [0, 0, 1]}],
        "constraints": [{"region": "ends", "dofs": [1] * 6},
                        {"region": "ends", "dofs": [0, 1, 0, 0, 0, 0],
                         "record": {"path": u_path, "wave": wave}}],
        "loads": [],
        "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.2},
        "output": {"fields": ["displacement"], "every": 20,
                   "monitors": {"l": {"kind": "displacement", "region": "left", "dof": "y"},
                                "r": {"kind": "displacement", "region": "right", "dof": "y"}}},
    }
    _out, h5 = _run(tmp_path, "w", d)
    with h5py.File(h5) as f:
        t = np.asarray(f["monitors/time"]); ul = np.asarray(f["monitors/l"]); ur = np.asarray(f["monitors/r"])
    w = 2 * np.pi * 10.0

    def ug(tt):
        tt = np.clip(tt, 0.0, 0.1)
        return 2.0 / w * (tt - np.sin(w * tt) / w)
    assert ul == pytest.approx(ug(t), abs=1e-8)
    assert ur == pytest.approx(ug(t - L / c), abs=1e-8)
    assert np.abs(ul - ur).max() > 1e-3                          # the ends moved apart


def test_the_record_helper_delays_holds_and_differentiates(tmp_path):
    from vfife.core.ground import RecordMotion, support_prescriptions
    p = tmp_path / "r.txt"
    np.savetxt(p, [[0.0, 0.0], [1.0, 2.0], [2.0, 2.0]], encoding="utf-8")
    r = RecordMotion({"path": str(p), "delay": 0.5}, duration=4.0)
    assert r(0.0) == 0.0 and r(0.125) == 0.0                    # t = 0.5 - 0.5 = 0
    # t = 0.5: a displacement record is read C1 since v1.0 -- the
    # cubic Hermite with PCHIP knot slopes (1.0 re-review) 3 (the
    # shape-preserving end slope) and 0 (the sample is where the record turns
    # flat), not the chord's 1.0 and 2.0
    assert r(0.25) == pytest.approx(1.375)
    assert r(0.375) == pytest.approx(2.0)                         # t = 1, the sample itself
    assert r(1.0) == pytest.approx(2.0)                           # held after the end
    assert r.rate(0.25) == pytest.approx(2.25)
    assert r.rate(1.0) == 0.0
    a = RecordMotion({"path": str(p), "quantity": "acceleration"}, duration=4.0)
    assert a(1.0) == 0.0                                          # an acceleration is not held
    # a wave that reaches the region before its origin is refused
    nodes = np.array([[0.0, 0, 0], [-5.0, 0, 0]])
    c = {"dofs": [1, 0, 0, 0, 0, 0], "record": {"path": str(p), "wave": {"speed": 100.0,
                                                                          "direction": [1, 0, 0]}}}
    with pytest.raises(ValueError, match="before its origin"):
        support_prescriptions(c, np.array([0, 1]), lambda i, d: i, nodes, 1.0)
    out = support_prescriptions(c, np.array([0]), lambda i, d: i, nodes, 1.0)
    assert len(out) == 1 and out[0][2].delay == 0.0


def test_a_record_constraint_is_refused_where_it_would_mean_nothing(tmp_path):
    from vfife.io.schema import ConstraintSpec
    rec = {"path": "r.txt"}
    with pytest.raises(ValueError, match="takes its motion from the record"):
        ConstraintSpec(region="a", dofs=[1, 0, 0, 0, 0, 0], value=1.0, record=rec)
    with pytest.raises(ValueError, match="drives translations"):
        ConstraintSpec(region="a", dofs=[1, 0, 0, 1, 0, 0], record=rec)
    with pytest.raises(ValueError, match="drives translations"):
        ConstraintSpec(region="a", dofs=[0, 0, 0, 0, 0, 0], record=rec)
    assert ConstraintSpec(region="a", dofs=[1, 1, 1], record=rec).record.quantity == "displacement"
    # a relaxation run has no clock to play a record on
    _a, u_path, _f = _pulse(tmp_path, "x")
    with pytest.raises(Exception, match="requires solver.mode = dynamic"):
        _write_msh(tmp_path / "x.msh", [(0, 0, 0), (1, 0, 0)],
                   {"a": (0, [0]), "b": (1, [(0, 1)])})
        _run(tmp_path, "x", {
            "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "gmsh", "path": "x.msh"}, "materials": {"steel": STEEL},
            "sections": {"c": SEC}, "assign": [{"region": "b", "section": "c", "orientation": [0, 0, 1]}],
            "constraints": [{"region": "a", "dofs": [1] * 6},
                            {"region": "a", "dofs": [1, 0, 0, 0, 0, 0], "record": {"path": u_path}}],
            "loads": [], "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 10},
            "output": {"fields": ["displacement"], "monitors": {}}})


def test_an_acceleration_constraint_on_a_frame_moves_its_node(tmp_path):
    """Before v1.0 the frame solvers zeroed a held dof's velocity
    before every position update and then integrated a prescribed
    acceleration from it: the node crept by a dt^2/2 per step instead of
    moving a t^2/2. The truss solver, the only one tested, had it right."""
    _write_msh(tmp_path / "k.msh", [(0, 0, 0), (1, 0, 0)],
               {"a": (0, [0]), "b": (1, [(0, 1)])})
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": "k.msh"}, "materials": {"steel": STEEL},
         "sections": {"c": SEC}, "assign": [{"region": "b", "section": "c", "orientation": [0, 0, 1]}],
         "constraints": [{"region": "a", "dofs": [1] * 6},
                         {"region": "a", "dofs": [0, 0, 1, 0, 0, 0], "value": 2.0,
                          "motion": "acceleration", "amplitude": "hold"}],
         "loads": [], "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.05},
         "output": {"fields": ["displacement"], "every": 1000,
                    "monitors": {"z": {"kind": "displacement", "region": "a", "dof": "z"}}}}
    _out, h5 = _run(tmp_path, "k", d)
    with h5py.File(h5) as f:
        t = np.asarray(f["monitors/time"]); z = np.asarray(f["monitors/z"])
    assert z[-1] == pytest.approx(0.5 * 2.0 * t[-1] ** 2, rel=1e-2)


def test_case_36_meets_its_contract(tmp_path):
    """The shipped case against its closed-form contract, through the same
    reader `vfife verify` uses."""
    from pathlib import Path
    from vfife.api import run_case
    from vfife.verification import verify
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "case_36.h5"
    run_case(str(root / "cases" / "case_36_multi_support_wave.yaml"), output=str(out))
    report = verify(root / "verification" / "case_36_multi_support_wave.yaml", out)
    assert report["status"] == "PASS", [c for c in report["checks"] if c["status"] != "PASS"]
