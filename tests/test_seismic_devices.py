"""The devices of a bridge's seismic design as connector laws (1.10): a nonlinear fluid viscous damper, F = c sign(v) |v|^alpha,
and a friction pendulum bearing, friction mu N plus the restoring force
(N / R) d of its spherical surface.

* The damper's law, and its linear piece below `linear_below`, continuous
  where the two meet.
* A mass on a spring and a damper under a force put on at once: the
  displacement history is the one a fine Runge-Kutta integration of
  m x'' + c sign(x') |x'|^alpha + k x = F gives -- for alpha = 0.3 and for alpha = 1, where the envelope is the
  exponential of a linear dashpot.
* A block on a pendulum bearing under its weight, pushed: nothing until
  mu W, then d = (H - mu W) R / W; with twice the weight, the same
  displacement at twice the push (the bearing's period does not know the
  mass).
* The compiled connectors give the reference's force for both laws, and the
  compiled step is the loop to the bit.
* What a law cannot mean is refused by name.
"""
import os

import h5py
import numpy as np
import pytest

from test_connector_laws import _bearing_case, _run, G, K_V
from vfife.elements import connector


def test_the_damper_law_and_its_linear_piece():
    law = {"type": "damper", "c": 2.0e6, "exponent": 0.3, "linear_below": 1e-3}
    for v in (0.02, 0.5, -0.3):
        assert connector.damper_force(law, v) == pytest.approx(np.sign(v) * 2.0e6 * abs(v) ** 0.3, rel=1e-14)
    # linear below v0, and continuous there
    assert connector.damper_force(law, 4e-4) == pytest.approx(2.0e6 * 1e-3 ** -0.7 * 4e-4, rel=1e-14)
    assert connector.damper_force(law, 1e-3 * (1 - 1e-12)) == pytest.approx(connector.damper_force(law, 1e-3), rel=1e-9)
    assert connector.damper_force(law, 0.0) == 0.0
    # the largest slope is the linear piece's: what the explicit bound reads
    assert connector.damper_linear(law) == pytest.approx(2.0e6 * 1e-3 ** -0.7)
    assert connector.damping_matrix({"laws": {"ux": law}, "connector_damping": [0, 5.0, 0, 0, 0, 0]})[0, 0] \
        == pytest.approx(2.0e6 * 1e-3 ** -0.7)
    lin = {"type": "damper", "c": 3.0, "exponent": 1.0}
    assert connector.damper_force(lin, -0.25) == -0.75 and connector.damper_linear(lin) == 3.0


def _oscillator(tmp_path, alpha, c, backend="numpy"):
    """A block on a vertical connector: its local ux (global z) carries a
    spring k and the damper; released from x0 with no velocity."""
    m, k = 2.0e4, 8.0e5                                  # 1.0 Hz
    law = {"type": "damper", "c": c, "exponent": alpha, "stiffness": k}
    if alpha < 1.0:
        law["linear_below"] = 1e-4
    d = _bearing_case(tmp_path, {"ux": law}, [0, 1e9, 1e9, 1e6, 1e6, 1e6])
    d["constraints"].append({"region": "block", "dofs": [1, 1, 0, 1, 1, 1]})
    d["nodal_masses"] = [{"region": "block", "mass": m}]
    x0 = 0.05
    # a force k x0 put on at once and held: the step response about x0
    d["loads"] = [{"region": "block", "vector": [0, 0, k * x0], "amplitude": "hold"}]
    d["solver"] = {"backend": backend, "mode": "dynamic", "total_time": 3.0, "damping": {"type": "none"},
                   # a 1 Hz oscillator is the only mode: resolve it (the damper's
                   # linear piece already holds the step down when alpha < 1)
                   "dt_safety": 0.02 if alpha == 1.0 else 0.5}
    d["output"] = dict(d["output"], fields=["displacement"], every=10)
    d["output"]["monitors"] = {"u": {"kind": "displacement", "region": "block", "dof": "z"}}
    return d, m, k, x0


def _reference(m, k, c, alpha, x0, t):
    from scipy.integrate import solve_ivp

    def rhs(_t, y):
        v = y[1]
        return [v, (k * x0 - c * np.sign(v) * abs(v) ** alpha - k * y[0]) / m]
    sol = solve_ivp(rhs, (0.0, t[-1]), [0.0, 0.0], t_eval=t, rtol=1e-10, atol=1e-12, max_step=1e-3)
    return sol.y[0]


@pytest.mark.parametrize("alpha,c", [(0.3, 6.0e3), (1.0, 2.0e4)])
def test_a_mass_on_a_spring_and_a_damper_follows_the_integrated_equation(tmp_path, alpha, c):
    d, m, k, x0 = _oscillator(tmp_path, alpha, c)
    _run(tmp_path, f"osc{int(alpha * 10)}", d)
    with h5py.File(tmp_path / f"osc{int(alpha * 10)}.h5") as f:
        t = np.asarray(f["monitors/time"]); u = np.asarray(f["monitors/u"])
    ref = _reference(m, k, c, alpha, x0, t)
    assert np.abs(u - ref).max() < 0.01 * x0
    assert u.max() > 1.3 * x0 and np.abs(u[-20:] - x0).max() < 0.6 * x0     # it overshot, and decayed
    if alpha == 1.0:
        zeta = c / (2 * np.sqrt(k * m))
        w = u - x0
        peaks = w[1:-1][(w[1:-1] > w[:-2]) & (w[1:-1] > w[2:])]
        assert peaks[1] / peaks[0] == pytest.approx(np.exp(-2 * np.pi * zeta / np.sqrt(1 - zeta ** 2)), rel=2e-2)


K_STICK = 2.0e7


def _pendulum(tmp_path, W, H, name, R=2.0, mu=0.05):
    law = {"type": "pendulum", "mu": mu, "radius": R, "normal": "ux", "stiffness": K_STICK}
    d = _bearing_case(tmp_path, {"uy": dict(law), "uz": dict(law)}, [1.0e8, 0, 0, 1e6, 1e6, 1e6])
    d["constraints"].append({"region": "block", "dofs": [0, 0, 0, 1, 1, 1]})
    d["loads"] = [{"region": "block", "vector": [0, 0, -W], "amplitude": "hold"}]
    d["steps"] = [{"loads": [{"region": "block", "vector": [0.6 * H, 0.8 * H, 0]}],
                   "load_steps": 2, "max_steps_per_load": 20000, "tolerance": 1e-6}]
    d["solver"] = {"backend": "numpy", "mode": "relaxation", "damping": {"type": "kinetic"}, "tolerance": 1e-6}
    d["output"] = dict(d["output"], fields=["displacement"])
    d["output"]["monitors"] = {"ux": {"kind": "displacement", "region": "block", "dof": "x"},
                               "uy": {"kind": "displacement", "region": "block", "dof": "y"}}
    out, _ = _run(tmp_path, name, d)
    return np.asarray(out.monitors["ux"]), np.asarray(out.monitors["uy"])


def test_a_pendulum_bearing_slides_at_mu_w_and_restores_with_w_over_r(tmp_path):
    W, R, mu = 2.0e6, 2.0, 0.05
    H = 0.08 * W                                   # two levels: 0.04 W sticks, 0.08 W is past mu W
    ux, uy = _pendulum(tmp_path, W, H, "fp1")
    # in the plane the bearing moves along the push: one law, not two
    levels = np.arange(1, 3) / 2 * H
    r = np.hypot(ux, uy)
    expect = np.where(levels > mu * W, (levels - mu * W) * R / W, levels / (K_STICK + W / R))   # stuck: both springs
    assert np.allclose(r, expect, rtol=2e-3, atol=1e-6)
    assert np.allclose(uy[-1] / ux[-1], 0.8 / 0.6, rtol=1e-4)
    # twice the weight, twice the push: the same displacement
    ux2, uy2 = _pendulum(tmp_path, 2 * W, 2 * H, "fp2")
    assert np.allclose(np.hypot(ux2, uy2)[1:], r[1:], rtol=2e-3, atol=1e-6)


def test_an_uplifted_pendulum_carries_nothing():
    law = {"type": "pendulum", "mu": 0.05, "radius": 2.0, "normal": "ux", "stiffness": 1.0e8}
    f, s = connector.law_force(law, 0.01, 0.0, -5.0)            # tension on ux: lifted off
    assert f == 0.0 and s == 0.01
    f, _ = connector.law_force(law, 0.01, 0.01, 1.0e6)           # sliding state, pressed
    assert f == pytest.approx(1.0e6 / 2.0 * 0.01)


@pytest.mark.parametrize("planar", [False, True])
def test_the_compiled_connectors_give_the_reference_for_both_laws(planar):
    nc = pytest.importorskip("vfife.kernels.numba_connector")
    if not nc.available():
        pytest.skip("no numba")
    fp = {"type": "pendulum", "mu": 0.06, "radius": 3.0, "normal": "ux", "stiffness": 2.0e8}
    laws = {"rx": {"type": "damper", "c": 4.0e5, "exponent": 0.4, "linear_below": 1e-3, "stiffness": 3.0e6},
            "uy": dict(fp)}
    if planar:
        laws["uz"] = dict(fp)
    else:
        laws["uz"] = {"type": "damper", "c": 1.5e6, "exponent": 0.25, "linear_below": 2e-3}
    props = {"stiffness": [2e9, 0, 0, 0, 1e7, 1e7], "laws": laws, "coordinate_system": "local"}
    K, C, local, has_c, ncur, cx, cy, law, lp, pl = nc.encode([props])
    assert bool(pl[0]) is planar
    rng = np.random.default_rng(5)
    x0 = np.array([[0.0, 0, 0], [0.0, 0, 0.4]])
    R0 = np.array([[0.0, 0, 1], [1.0, 0, 0], [0, 1.0, 0]])
    sc = np.zeros(6)
    worst = 0.0
    for step in range(40):
        u = rng.standard_normal((2, 3)) * 2e-3
        u[1, 2] -= 1e-3                                         # mostly pressed, sometimes not
        from vfife.core.rotations import exp_map
        T = exp_map(rng.standard_normal((2, 3)) * 1e-3)
        ve = rng.standard_normal(12) * (10.0 ** rng.uniform(-4, 0))
        so_ref = sc.copy()
        fe_ref, act_ref, d_ref = connector.response(u, T, ve, x0, R0, props, include_damping=True,
                                                    slip_committed=sc, slip_out=so_ref)
        fe = np.empty(12); act = np.empty(6); d = np.empty(6); so = sc.copy()
        nc.response_one(u, np.ascontiguousarray(T), ve, x0, R0, nc.spins(), K[0], C[0], local[0], has_c[0],
                        ncur[0], cx[0], cy[0], law[0], lp[0], pl[0], sc, fe, act, d, so, True, np.zeros(6))
        scale = max(np.abs(act_ref).max(), 1.0)
        worst = max(worst, np.abs(act - act_ref).max() / scale)
        assert np.allclose(so, so_ref, rtol=1e-9, atol=1e-13)
        sc = so_ref
    assert worst < 1e-7, worst


def test_the_compiled_step_takes_a_damper_to_the_bit(tmp_path):
    """A girder stub on a damper and a pendulum bearing: the compiled step
    (frames and connectors under backend numba) against the loop."""
    pytest.importorskip("numba")
    from test_connector_laws import _write_msh
    from vfife.api import run_case
    _write_msh(tmp_path / "g.msh", [(0, 0, 0), (0, 0, 0.5), (2.0, 0, 0.5), (4.0, 0, 0.5), (4.0, 0, 0)],
               {"anchors": (0, [0, 4]), "mid": (0, [2]), "damper": (1, [(0, 1)]), "bearing": (1, [(4, 3)]),
                "girder": (1, [(1, 2), (2, 3)])})
    fp = {"type": "pendulum", "mu": 0.05, "radius": 2.0, "normal": "ux", "stiffness": 1.0e8}
    d = {"units": {"length": "m", "force": "N"},
         "mesh": {"type": "gmsh", "path": str(tmp_path / "g.msh")},
         "materials": {"s": {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}},
         "sections": {"g": {"type": "beam", "material": "s", "shape": "general", "area": 0.05, "iy": 2e-3,
                            "iz": 2e-3, "j": 1e-3},
                      "d": {"type": "connector", "stiffness": [1e9, 0, 1e8, 1e7, 1e7, 1e7],
                            "laws": {"uy": {"type": "damper", "c": 5.0e4, "exponent": 0.3,
                                            "linear_below": 1e-3, "stiffness": 2.0e6}}},
                      "b": {"type": "connector", "stiffness": [1e9, 0, 0, 1e7, 1e7, 1e7],
                            "laws": {"uy": dict(fp), "uz": dict(fp)}}},
         "assign": [{"region": "girder", "section": "g", "orientation": [0, 0, 1]},
                    {"region": "damper", "section": "d", "orientation": [1, 0, 0]},
                    {"region": "bearing", "section": "b", "orientation": [1, 0, 0]}],
         "constraints": [{"region": "anchors", "dofs": [1] * 6}],
         "amplitudes": {"kick": [[0, 0], [0.2, 1.0], [0.4, -1.0], [0.6, 0.0], [1.0, 0.0]]},
         "loads": [{"region": "mid", "vector": [0, 0, -5.0e5], "amplitude": "hold"},
                   {"region": "mid", "vector": [2.0e5, 5.0e4, 0], "amplitude": "kick"}],
         "solver": {"backend": "numba", "mode": "dynamic", "total_time": 0.3, "damping": {"type": "none"}},
         "output": {"fields": ["displacement"], "every": 100}}
    ends = []
    for flag in ("0", "1"):
        d["name"] = f"cs{flag}"
        d["output"]["path"] = str(tmp_path / f"cs{flag}.h5")
        old = os.environ.get("VFIFE_COMPILED_STEP")
        os.environ["VFIFE_COMPILED_STEP"] = flag
        box = {}
        try:
            out = run_case(d, on_ready=lambda s, p: box.setdefault("s", s))
        finally:
            if old is None:
                os.environ.pop("VFIFE_COMPILED_STEP", None)
            else:
                os.environ["VFIFE_COMPILED_STEP"] = old
        assert out.status == "COMPLETED_DYNAMIC", out.message
        s_ = box["s"]
        ends.append((s_.q.copy(), s_.v.copy(), s_._conn_committed.copy(), s_.compiled_step_refusal))
    assert ends[1][3] is None, ends[1][3]
    for a_, b_ in zip(ends[0][:3], ends[1][:3]):
        assert np.array_equal(a_, b_)
    assert np.abs(ends[0][0]).max() > 1e-4 and np.abs(ends[0][2]).max() > 0.0       # it moved, and slid


@pytest.mark.parametrize("law,why", [
    ({"type": "damper", "c": 1e6, "exponent": 0.3}, "linear_below"),
    ({"type": "damper", "c": 1e6, "exponent": 1.5}, "less than or equal to 1"),
    ({"type": "damper", "c": 1e6, "exponent": 0.3, "linear_below": 1e-3, "mu": 0.1}, "does not take"),
    ({"type": "pendulum", "mu": 0.05, "normal": "ux", "stiffness": 1e8}, "radius"),
    ({"type": "pendulum", "mu": 0.05, "normal": "uy", "stiffness": 1e8, "radius": 2.0}, "normal"),
])
def test_what_a_device_law_cannot_mean_is_refused(law, why):
    from vfife.io.schema import ConnectorLawSpec
    with pytest.raises(Exception, match=why):
        ConnectorLawSpec(**law)


def test_a_dynamic_step_after_a_relaxation_takes_the_dampers_bound(tmp_path):
    """Gravity by relaxation, then the earthquake as a `procedure: dynamic`
    step: the step's own time step was taken from the stiffness alone, and a
    stiff damper went non-finite (c dt / m far above 2; found in the review
    of 1.10). It takes the dashpots' bound, dt < 2 m / c, as a dynamic base
    mode does."""
    from test_connector_laws import _write_msh
    from vfife.api import run_case
    _write_msh(tmp_path / "g.msh", [(0, 0, 0), (0, 0, 0.5), (2.0, 0, 0.5), (4.0, 0, 0.5), (4.0, 0, 0)],
               {"anchors": (0, [0, 4]), "mid": (0, [2]), "damper": (1, [(0, 1)]), "bearing": (1, [(4, 3)]),
                "girder": (1, [(1, 2), (2, 3)])})
    d = {"units": {"length": "m", "force": "N"},
         "mesh": {"type": "gmsh", "path": str(tmp_path / "g.msh")},
         "materials": {"s": {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}},
         "sections": {"g": {"type": "beam", "material": "s", "shape": "general", "area": 0.05, "iy": 2e-3,
                            "iz": 2e-3, "j": 1e-3},
                      "d": {"type": "connector", "stiffness": [1e9, 0, 1e8, 1e7, 1e7, 1e7],
                            "laws": {"uy": {"type": "damper", "c": 3.0e7, "exponent": 1.0, "stiffness": 2.0e6}}},
                      "b": {"type": "connector", "stiffness": [1e9, 2e6, 1e8, 1e7, 1e7, 1e7]}},
         "assign": [{"region": "girder", "section": "g", "orientation": [0, 0, 1]},
                    {"region": "damper", "section": "d", "orientation": [1, 0, 0]},
                    {"region": "bearing", "section": "b", "orientation": [1, 0, 0]}],
         "constraints": [{"region": "anchors", "dofs": [1] * 6}],
         "loads": [{"region": "mid", "vector": [0, 0, -5.0e5], "amplitude": "hold"}],
         "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 20000, "tolerance": 1e-6},
         "steps": [{"name": "g", "n_steps": 20000},
                   {"name": "eq", "procedure": "dynamic", "total_time": 0.05,
                    "loads": [{"region": "mid", "vector": [2.0e5, 5.0e4, 0]}]}],
         "output": {"fields": ["displacement"], "every": 1000, "path": str(tmp_path / "st.h5")}}
    box = {}
    out = run_case(d, on_ready=lambda s, p: box.setdefault("s", s))
    assert out.status in ("COMPLETED_DYNAMIC", "CONVERGED"), (out.status, out.message)
    s = box["s"]
    rows = connector.dashpot_rows([0], s.mesh.elements, s.mesh.nodes, s.R0, lambda e: s.member_props[e],
                                  s.dofs, s.ndof)
    on = rows > 1e-20
    on &= ~s._held_dofs()
    assert s.dt <= 2.0 * np.min(s.mass[on] / rows[on])
    assert np.isfinite(s.q).all()
