"""Vehicles with mass, suspension and tyres.

Checked against what does not come from this library: a quarter car on a
sinusoidal road against its 2 x 2 frequency response; a two-axle truck's
axle loads against the lever rule; a PSD roughness's variance against the
integral of its PSD; and (case_38) the sprung-mass crossing against a modal
solution integrated by scipy.
"""
import h5py
import numpy as np
import pytest

from test_member_failure import _run, _write_msh

STEEL = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}
SEC = {"type": "beam", "material": "steel", "shape": "general", "area": 0.5, "iy": 0.2, "iz": 0.2, "j": 0.4}


def _road_case(tmp_path, name, vehicle, total, monitors, roughness=None):
    """A short heavy beam far ahead; the vehicle stays on the rigid approach
    (negative stations) the whole run, so the road is the roughness alone."""
    nodes = [(i * 2.0, 0, 0) for i in range(6)]
    _write_msh(tmp_path / f"{name}.msh", nodes,
               {"ends": (0, [0, 5]), "deck": (1, [(i, i + 1) for i in range(5)])})
    if roughness is not None:
        vehicle = dict(vehicle, roughness=roughness)
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": f"{name}.msh"},
        # soft: the beam is only there to carry the path, and its step is
        # the run's step
        "materials": {"steel": dict(STEEL, E=2.0e8)},
        "sections": {"c": SEC},
        "assign": [{"region": "deck", "section": "c", "orientation": [0, 1, 0]}],
        "constraints": [{"region": "ends", "dofs": [1] * 6}],
        "loads": [{"region": "deck", "vehicle": vehicle}],
        "solver": {"backend": "numpy", "mode": "dynamic", "total_time": total},
        "output": {"fields": ["displacement"], "every": 10, "monitors": monitors},
    }
    out, h5 = _run(tmp_path, name, d)
    with h5py.File(h5) as f:
        return {k: np.asarray(f["monitors"][k]) for k in f["monitors"]}


def test_a_quarter_car_on_a_sine_road_follows_its_frequency_response(tmp_path):
    mb, mu, ks, cs, kt = 4000.0, 400.0, 2.0e5, 1.0e4, 1.6e6
    lam, amp, V = 5.0, 0.005, 20.0
    s = np.linspace(-1000.0, 0.0, 200001)
    np.savetxt(tmp_path / "sine.txt", np.column_stack([s, amp * np.sin(2 * np.pi * s / lam)]), encoding="utf-8")
    veh = {"name": "q", "speed": V, "start": -900.0, "body": {"mass": mb},
           "axles": [{"offset": 0.0, "unsprung_mass": mu, "suspension": {"stiffness": ks, "damping": cs},
                      "tyre": {"stiffness": kt}}]}
    m = _road_case(tmp_path, "sine", veh, 10.0,
                   {"z": {"kind": "vehicle", "of": "q", "field": "heave"},
                    "w": {"kind": "vehicle", "of": "q", "field": "wheel"},
                    "P": {"kind": "vehicle", "of": "q", "field": "contact_force"}},
                   roughness={"profile": str(tmp_path / "sine.txt")})
    w = 2 * np.pi * V / lam
    M = np.diag([mb, mu]); C = np.array([[cs, -cs], [-cs, cs]]); K = np.array([[ks, -ks], [-ks, ks + kt]])
    X = np.linalg.solve(-w * w * M + 1j * w * C + K, np.array([0.0, kt * amp]))
    late = m["time"] > 6.0                                   # the sprung mode (zeta 0.18) is gone
    zb = m["z"][late]; zu = m["w"][late]
    z_static = -(mb + mu) * 9.80665 / kt - mb * 9.80665 / ks
    assert 0.5 * (zb.max() - zb.min()) == pytest.approx(abs(X[0]), rel=1e-2)
    assert 0.5 * (zu.max() - zu.min()) == pytest.approx(abs(X[1]), rel=1e-2)
    assert zb.mean() == pytest.approx(z_static, abs=0.02 * abs(X[0]))
    # the tyre force swings about the weight, never leaving the road here
    P = m["P"][late]
    assert P.min() > 0 and 0.5 * (P.max() + P.min()) == pytest.approx((mb + mu) * 9.80665, rel=1e-2)


def test_a_truck_carries_its_axle_loads_by_the_lever_rule(tmp_path):
    """Two axles 4 m apart, the body's centre 1 m behind the front one: on a
    flat road the axles carry 3/4 and 1/4 of the body, plus their own
    wheels, and nothing moves."""
    mb, mu = 12000.0, 500.0
    veh = {"name": "t", "speed": 10.0, "start": -500.0,
           "body": {"mass": mb, "pitch_inertia": 3.0e4, "cg": -1.0},
           "axles": [{"offset": 0.0, "unsprung_mass": mu, "suspension": {"stiffness": 4e5, "damping": 2e4},
                      "tyre": {"stiffness": 2e6}},
                     {"offset": -4.0, "unsprung_mass": mu, "suspension": {"stiffness": 6e5, "damping": 2e4},
                      "tyre": {"stiffness": 2e6}}]}
    m = _road_case(tmp_path, "truck", veh, 1.0,
                   {"P0": {"kind": "vehicle", "of": "t", "field": "contact_force", "axle": 0},
                    "P1": {"kind": "vehicle", "of": "t", "field": "contact_force", "axle": 1},
                    "th": {"kind": "vehicle", "of": "t", "field": "pitch"},
                    "s1": {"kind": "vehicle", "of": "t", "field": "station", "axle": 1}})
    g = 9.80665
    assert m["P0"] == pytest.approx((0.75 * mb + mu) * g, rel=1e-9)
    assert m["P1"] == pytest.approx((0.25 * mb + mu) * g, rel=1e-9)
    assert np.ptp(m["th"]) < 1e-12
    # the rear axle is 4 m behind the front one, whichever way it drives
    assert m["s1"][-1] == pytest.approx(-500.0 + 10.0 * 1.0 - 4.0, abs=0.05)


def test_a_psd_road_has_the_variance_its_psd_integrates_to():
    from vfife.core.vehicle import roughness_profile
    g0, n0, w = 64e-6, 0.1, 2.0
    r = roughness_profile({"psd": {"g0": g0, "n0": n0, "w": w, "n_min": 0.05, "n_max": 2.0,
                                   "count": 800, "seed": 3}}, 0.0, 5000.0)
    x = np.linspace(0.0, 5000.0, 100001)
    expected = g0 * n0 ** w * (0.05 ** (1 - w) - 2.0 ** (1 - w)) / (w - 1)
    assert np.var(r(x)) == pytest.approx(expected, rel=0.05)
    # the same seed, the same road
    r2 = roughness_profile({"psd": {"g0": g0, "n_min": 0.05, "n_max": 2.0, "count": 800, "seed": 3}},
                           0.0, 5000.0)
    assert np.allclose(r(x[:1000]), r2(x[:1000]))


def test_a_vehicle_is_refused_where_it_would_mean_nothing(tmp_path):
    from vfife.io.schema import VehicleSpec, VehicleAxleSpec, MonitorSpec
    body = {"mass": 1000.0}
    ax = {"offset": 0.0, "suspension": {"stiffness": 1e5}}
    with pytest.raises(ValueError, match="speed 0"):
        VehicleSpec(body=body, axles=[ax], speed=0.0)
    with pytest.raises(ValueError, match="pitch_inertia"):
        VehicleSpec(body=body, axles=[ax, dict(ax, offset=-3.0)], speed=10.0)
    with pytest.raises(ValueError, match="both or neither"):
        VehicleAxleSpec(offset=0.0, suspension={"stiffness": 1e5}, tyre={"stiffness": 1e6})
    with pytest.raises(ValueError, match="needs 'of'"):
        MonitorSpec(kind="vehicle", field="heave")
    with pytest.raises(ValueError, match="heave, pitch"):
        MonitorSpec(kind="vehicle", of="v", field="speed")
    # a relaxation run has no clock; a tyre too stiff for the step is named
    veh = {"name": "v", "speed": 10.0, "start": -50.0, "body": body, "axles": [ax]}
    with pytest.raises(Exception, match="dynamic"):
        _write_msh(tmp_path / "r.msh", [(0, 0, 0), (2, 0, 0)], {"ends": (0, [0, 1]), "deck": (1, [(0, 1)])})
        _run(tmp_path, "r", {
            "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "gmsh", "path": "r.msh"}, "materials": {"steel": STEEL},
            "sections": {"c": SEC}, "assign": [{"region": "deck", "section": "c", "orientation": [0, 1, 0]}],
            "constraints": [{"region": "ends", "dofs": [1] * 6}],
            "loads": [{"region": "deck", "vehicle": veh}],
            "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 10},
            "output": {"fields": ["displacement"], "monitors": {}}})
    stiff = dict(veh, axles=[{"offset": 0.0, "unsprung_mass": 1.0, "suspension": {"stiffness": 1e5},
                              "tyre": {"stiffness": 1e12}}])
    with pytest.raises(Exception, match="stiffest mode"):
        _road_case(tmp_path, "stiff", stiff, 0.01, {})


def test_case_38_meets_its_contract(tmp_path):
    """The sprung-mass crossing against a modal solution integrated by scipy."""
    from pathlib import Path
    from vfife.api import run_case
    from vfife.verification import verify
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "case_38.h5"
    run_case(str(root / "cases" / "case_38_sprung_mass_crossing.yaml"), output=str(out))
    report = verify(root / "verification" / "case_38_sprung_mass_crossing.yaml", out)
    assert report["status"] == "PASS", [c for c in report["checks"] if c["status"] != "PASS"]
