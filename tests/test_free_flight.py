"""A free body under dynamics: the floating beam of Wu &
Ting (2020), example 1, in miniature.

THE THREE THINGS THIS PINS
    The preflight lets a free body fly under `mode: dynamic` (and still
    refuses it under relaxation); a `follower: true` pressure turns with the
    face it acts on; and the energy audit closes on a free elastic flight --
    it used to pair step n's forces with step n+1's displacement increment,
    and on the flexible floating beam that lag drifted the internal work to
    -18 J of a 91 J budget while the kinetic energy stayed put.
"""
import numpy as np
import pytest

from vfife.api import prepare_case, run_case
from vfife.core.mesh import FollowerPressure, surface_pressure
from vfife.core.rotations import exp_map


@pytest.fixture(autouse=True)
def _own_folder(tmp_path, monkeypatch):
    """Run in a folder of the test's own: these tests write the default
    `results.h5` (or results/...) beside the working directory, and parallel
    workers (pytest -n) wrote the same file at once."""
    monkeypatch.chdir(tmp_path)


def _flight(E=90.0e9, total=0.008, divisions=(1, 10, 1), mode="dynamic", follower=True):
    return {
        "name": "flight_test", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "box", "dimensions": [0.01, 0.2, 0.01], "divisions": list(divisions)},
        "materials": {"m": {"type": "linear_elastic", "E": E, "nu": 0.3, "rho": 1000.0}},
        "sections": {"body": {"type": "solid", "material": "m", "formulation": "reduced"}},
        "regions": {"near": {"box": [-0.001, 0.011, -0.001, 0.001, -0.001, 0.011]},
                    "far": {"box": [-0.001, 0.011, 0.199, 0.201, -0.001, 0.011]},
                    "A": {"nearest": [0.005, 0.2, 0.005]}, "C": {"nearest": [0.005, 0.0, 0.005]}},
        "amplitudes": {"pulse": {"table": [[0.0, 0.0], [0.0625, 1.0], [0.125, 0.0], [1.0, 0.0]]}},
        "loads": [{"region": "near", "pressure": 60.0e6, "amplitude": "pulse", "follower": follower},
                  {"region": "far", "pressure": 30.0e6, "amplitude": "pulse", "follower": follower}],
        "solver": {"backend": "numpy", "mode": mode, "total_time": total, "dt_safety": 0.6,
                   "damping": {"type": "viscous", "coeff": 0.0}, "n_steps": 1000, "tolerance": 1e-4},
        "output": {"path": "results/_flight_test.h5", "every": 10**7,
                   "monitors": {"y_cm": {"kind": "displacement", "region": "all", "dof": "y"},
                                "y_rel": {"kind": "displacement", "region": "A", "dof": "y", "relative_to": "C"}}},
    }


def test_a_free_body_is_a_warning_under_dynamics_and_an_error_under_relaxation():
    dyn = prepare_case(_flight()).preflight
    assert dyn["status"] == "PASS"
    codes = {i["code"]: i["severity"] for i in dyn["issues"]}
    assert codes.get("RIGID_BODY_MODE") == "WARNING"
    rel = prepare_case(_flight(mode="relaxation")).preflight
    assert rel["status"] != "PASS"
    assert {i["code"]: i["severity"] for i in rel["issues"]}.get("RIGID_BODY_MODE") == "ERROR"


def test_a_follower_pressure_turns_with_its_face():
    """On the rotated mesh the follower force is the reference force
    rotated; the fixed-direction one is not."""
    pr = prepare_case(_flight())
    mesh = pr.mesh; idx = mesh.regions["near"]
    fp = FollowerPressure(mesh.nodes, mesh.elements, "hexa8", idx, 1.0)
    f0 = fp(mesh.nodes)
    assert np.allclose(f0, surface_pressure(mesh.nodes, mesh.elements, "hexa8", idx, 1.0))
    Q = exp_map(np.array([0.7, -0.4, 1.1]))
    x = (Q @ mesh.nodes.T).T + 0.3
    f1 = fp(x)
    assert np.allclose(f1, (Q @ f0.T).T, atol=1e-12 * np.abs(f0).max())
    assert not np.allclose(f1, f0, atol=1e-3 * np.abs(f0).max())


def test_the_energy_audit_closes_on_a_free_elastic_flight():
    """Momentum is the impulse, and once the pulse is over the external
    work is spent: kinetic + strain stays equal to it, to 2e-3 on this
    coarse mesh (3e-5 on case_19), on the
    flexible beam where the two exchange all the way through."""
    r = run_case(_flight(E=3.6e9, total=0.008))
    assert r.status == "COMPLETED_DYNAMIC"
    import h5py
    with h5py.File(r.result_path) as h:
        err = h["diagnostics/relative_energy_error"][...]
        W = h["diagnostics/external_work"][...]
    assert err[-1] < 2e-3, err[-1]                    # 20 mm cubes, ~10 steps per period of the top mode
    assert W[-1] > 0.0
    # impulse: (60 - 30) MPa x 1e-4 m^2 x 0.5 ms = +1.5 N s on 0.02 kg = 75 m/s
    # from the pulse centroid at 0.5 ms -> 0.5625 m at 8 ms
    assert abs(r.monitors["y_cm"][-1] / 0.5625 - 1.0) < 0.02
    # equal, collinear-in-the-plane pulses on full faces: no spin, so the
    # chord keeps its length to the axial vibration
    assert abs(r.monitors["y_rel"][-1]) < 0.01


def test_a_contract_can_read_the_energy_audit(tmp_path):
    from vfife.verification import verify
    r = run_case(_flight(E=90.0e9, total=0.004))
    contract = tmp_path / "c.yaml"
    contract.write_text(
        "version: 1\nclassification: independent_reference\ncase: flight_test\nmetrics:\n"
        "  - {id: e, source: diagnostic, name: relative_energy_error, reduction: final, expect: {upper: 1.0e-3}}\n"
        "  - {id: w, source: diagnostic, name: external_work, reduction: final, expect: {lower: 0.0}}\n", encoding="utf-8")
    rep = verify(contract, r.result_path)
    assert rep["status"] == "PASS", rep["checks"]


def test_a_solver_block_n_steps_does_not_truncate_a_dynamic_run():
    """`n_steps` is a relaxation budget. It used to override the step count
    of a dynamic run too, and since the amplitude progress is per step it
    compressed the load history with it: a 1 ms pulse on this flight became
    0.34 ms and the beam left at a third of its momentum."""
    p = _flight(E=90.0e9, total=0.008); p["solver"]["n_steps"] = 1000
    r = run_case(p)
    assert abs(r.monitors["y_cm"][-1] / 0.5625 - 1.0) < 0.02
