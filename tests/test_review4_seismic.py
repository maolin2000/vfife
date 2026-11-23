"""Seismic input, modes, the response spectrum and rotary inertia: the
fourth review's findings (section 4), one test each.
"""
import h5py
import numpy as np
import pytest
import yaml

from test_member_failure import _run, _write_msh
from test_multi_support import _column, _pulse
from test_response_spectrum import TABLE, _closed_form, E as CE, A as CA, L1, L2, M1, M2


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _solver(tmp_path, name, d):
    from vfife.api import prepare_case, _with_element
    from vfife.core.structural import make_structural_solver
    p = tmp_path / f"{name}.yaml"
    d = dict(d, name=name)
    d["output"] = dict(d.get("output", {}), path=str(tmp_path / f"{name}.h5"))
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    prep = prepare_case(str(p))
    return make_structural_solver(_with_element(prep.case, prep.element, prep.case_path),
                                  prep.mesh, prep.material)


def _spectrum(h5):
    with h5py.File(h5) as f:
        g = f["response_spectrum"]
        return {k: np.asarray(g[k]) for k in g}


def _chain(tmp_path, name, rs=None, kind="beam", steps=None):
    """The two-mass chain of test_response_spectrum, as beams or as bars."""
    _write_msh(tmp_path / f"{name}.msh", [(0, 0, 0), (L1, 0, 0), (L1 + L2, 0, 0)],
               {"base": (0, [0]), "m1": (0, [1]), "m2": (0, [2]), "free": (0, [1, 2]),
                "s1": (1, [(0, 1)]), "s2": (1, [(1, 2)])})
    light = {"type": "linear_elastic", "E": CE, "nu": 0.3, "rho": 1.0e-3}
    if kind == "beam":
        sec = {"type": "beam", "material": "light", "shape": "general", "area": CA,
               "iy": 1e-6, "iz": 1e-6, "j": 2e-6}
        cons = [{"region": "base", "dofs": [1] * 6}, {"region": "free", "dofs": [0, 1, 1, 1, 1, 1]}]
        asg = [{"region": r, "section": "b", "orientation": [0, 0, 1]} for r in ("s1", "s2")]
    else:
        sec = {"type": "bar", "material": "light", "shape": "general", "area": CA}
        cons = [{"region": "base", "dofs": [1, 1, 1]}, {"region": "free", "dofs": [0, 1, 1]}]
        asg = [{"region": "all", "section": "b"}]
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": f"{name}.msh"},
         "materials": {"light": light}, "sections": {"b": sec}, "assign": asg,
         "nodal_masses": [{"region": "m1", "mass": M1}, {"region": "m2", "mass": M2}],
         "constraints": cons, "loads": [],
         "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 50, "tolerance": 1e-8},
         "output": {"fields": ["displacement"], "every": 10}}
    if rs is not None:
        d["response_spectrum"] = rs
    if steps is not None:
        d["steps"] = steps
    return d


RS2 = {"spectra": [{"direction": [1, 0, 0], "table": TABLE}], "combination": "cqc",
       "damping": 0.05, "modes": 2}


# --------------------------------------------------------------------------- #
#  1  the LAST step's supports
# --------------------------------------------------------------------------- #
def test_the_spectrum_and_modal_result_take_the_last_steps_supports(tmp_path):
    # step 1 holds m2 on a temporary support, step 2 releases it: the
    # spectrum at the end is the free chain's
    steps = [{"name": "erect", "constraints": [{"region": "m2", "dofs": [1, 0, 0, 0, 0, 0]}]},
             {"name": "released"}]
    d = _chain(tmp_path, "steps", RS2, steps=steps)
    _out, h5 = _run(tmp_path, "steps", d)
    r = _spectrum(h5)
    ref = _closed_form(0.05, "cqc")
    assert r["period"] == pytest.approx(ref["T"], rel=1e-6)
    assert r["displacement"][2, 0] == pytest.approx(ref["u2"], rel=1e-5)
    # `vfife modal --result`: a fresh solver put on the step of the frame
    from vfife import modal
    from vfife.testing.state import probe
    s = _solver(tmp_path, "steps_fresh", _chain(tmp_path, "steps_fresh", steps=steps))
    held0 = int(probe(s).fixed.sum())
    assert modal.use_step_at(s, 2.0) == 1
    assert int(probe(s).fixed.sum()) == held0 - 1
    assert modal.use_step_at(s, 0.5) == 0 and int(probe(s).fixed.sum()) == held0


# --------------------------------------------------------------------------- #
#  2  rigid floors in the modes and the spectrum
# --------------------------------------------------------------------------- #
def _floor(tmp_path, name, rs=None):
    E, H, m, I1, I2 = 2e11, 3.0, 1000.0, 1e-5, 4e-5
    n = 4
    nodes = [(0, 0, H * i / n) for i in range(n + 1)] + [(0, 4.0, H * i / n) for i in range(n + 1)]
    _write_msh(tmp_path / f"{name}.msh", nodes,
               {"base": (0, [0, n + 1]), "top": (0, [n, 2 * n + 1]), "top1": (0, [n]), "top2": (0, [2 * n + 1]),
                "c1": (1, [(i, i + 1) for i in range(n)]), "c2": (1, [(n + 1 + i, n + 2 + i) for i in range(n)])})
    sec = lambda I: {"type": "beam", "material": "s", "shape": "general", "area": 1e-2, "iy": I, "iz": I, "j": 2 * I}
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": f"{name}.msh"},
         "materials": {"s": {"type": "linear_elastic", "E": E, "nu": 0.3, "rho": 1e-3}},
         "sections": {"a": sec(I1), "b": sec(I2)},
         "assign": [{"region": "c1", "section": "a", "orientation": [1, 0, 0]},
                    {"region": "c2", "section": "b", "orientation": [1, 0, 0]}],
         "nodal_masses": [{"region": "top1", "mass": m}, {"region": "top2", "mass": m}],
         "constraints": [{"region": "base", "dofs": [1] * 6}], "loads": [],
         "diaphragms": [{"region": "top", "normal": "z"}],
         "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 50, "tolerance": 1e-8},
         "output": {"fields": ["displacement"], "every": 10}}
    if rs:
        d["response_spectrum"] = rs
    # the floor's two in-plane dofs: U (x) and the twist alpha about z; the
    # tops sit at y = -2, +2 from the centre, so u_x = U + 2 alpha, U - 2 alpha
    G = E / 2.6
    k1, k2 = 3 * E * I1 / H ** 3, 3 * E * I2 / H ** 3
    K = np.array([[k1 + k2, 2 * (k1 - k2)], [2 * (k1 - k2), 4 * (k1 + k2) + G * 2 * (I1 + I2) / H]])
    M = np.diag([2 * m, 8 * m])
    return d, K, M, n


def test_modes_and_the_spectrum_move_a_rigid_floor_as_one(tmp_path):
    import scipy.linalg
    from vfife import modal
    d, K, M, n = _floor(tmp_path, "floor")
    w2, v = scipy.linalg.eigh(K, M)
    f_ref = np.sqrt(w2) / (2 * np.pi)                     # 2.9603, 5.2441 Hz
    res = modal.analyse(_solver(tmp_path, "floor", d), n_modes=4)
    # the two x-twist modes, and y at sqrt((k1 + k2) / 2m) between them
    fx = res.frequencies[res.participation[:, 0] > 1e-3]
    assert fx == pytest.approx(f_ref, rel=2e-4)
    assert res.participation[:, 0].sum() == pytest.approx(1.0, rel=1e-4)
    # every mode: the tops' y equal (no stretching of the floor)
    sh = res.shapes.reshape(len(res.frequencies), -1, 6)
    assert np.abs(sh[:, n, 1] - sh[:, 2 * n + 1, 1]).max() < 1e-9
    # the spectrum: the same two modes, CQC, a flat 2 m/s^2
    rs = {"spectra": [{"direction": [1, 0, 0], "table": [[0, 2.0], [10, 2.0]]}], "combination": "cqc",
          "modes": 4}
    d2, *_ = _floor(tmp_path, "floor_rs", rs)
    _out, h5 = _run(tmp_path, "floor_rs", d2)
    r = _spectrum(h5)
    phi = v                                                # mass-normalised
    gam = phi.T @ M @ np.array([1.0, 0.0])
    U = phi * (gam * 2.0 / w2)[None, :]
    from vfife.spectrum import cqc_rho
    rho = cqc_rho(np.sqrt(w2), 0.05)
    comb = lambda x: float(np.sqrt(x @ rho @ x))
    assert r["displacement"][n, 0] == pytest.approx(comb(U[0] + 2 * U[1]), rel=1e-3)
    assert r["displacement"][2 * n + 1, 0] == pytest.approx(comb(U[0] - 2 * U[1]), rel=1e-3)


# --------------------------------------------------------------------------- #
#  3, 4  rotary inertia: true, rotated, and the step unchanged
# --------------------------------------------------------------------------- #
EB, NU, RHO, LB, B, H = 2.0e11, 0.25, 7850.0, 10.0, 0.2, 0.4


def _bar(tmp_path, name, kind, n, axis, mode="relaxation"):
    e = np.eye(3)[axis]
    nodes = [tuple(LB * i / n * e) for i in range(n + 1)]
    _write_msh(tmp_path / f"{name}.msh", nodes, {"base": (0, [0]), "col": (1, [(i, i + 1) for i in range(n)])})
    mat = {"type": "linear_elastic", "E": EB, "nu": NU, "rho": RHO}
    if kind == "fiber":
        sec = {"type": "fiber", "material": "s", "shape": "rect", "width": B, "height": H, "ny": 8, "nz": 4}
    else:
        sec = {"type": "beam", "material": "s", "shape": "rect", "width": B, "height": H}
        if kind == "timo":
            sec["theory"] = "timoshenko"
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": f"{name}.msh"}, "materials": {"s": mat}, "sections": {"c": sec},
         "assign": [{"region": "col", "section": "c", "orientation": [0, 0, 1] if axis == 0 else [1, 0, 0]}],
         "constraints": [{"region": "base", "dofs": [1] * 6}], "loads": [],
         "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 50},
         "output": {"fields": ["displacement"], "every": 10}}
    if mode == "dynamic":
        d["solver"] = {"backend": "numpy", "mode": "dynamic", "total_time": 1e-3}
    return _solver(tmp_path, name, d)


def _torsion_frequency(res, axis):
    for i in range(len(res.frequencies)):
        sh = res.shapes[i].reshape(-1, 6)
        if res.rotational[i] and np.abs(sh[:, 3:]).max(axis=0).argmax() == axis:
            return res.frequencies[i]
    raise AssertionError("no torsion mode")


def test_beam2_torsion_frequency_converges_to_the_bar(tmp_path):
    from vfife import modal
    G = EB / (2 * (1 + NU))
    a, b = H / 2, B / 2
    J = a * b ** 3 * (16 / 3 - 3.36 * b / a * (1 - b ** 4 / (12 * a ** 4)))
    Ip = B * H * (B * B + H * H) / 12
    exact = np.sqrt(G * J / (RHO * Ip)) / (4 * LB)
    for n in (4, 16):
        f = _torsion_frequency(modal.analyse(_bar(tmp_path, f"t{n}", "beam", n, 2), n_modes=12), 2)
        # a lumped fixed-free rod: (4n/pi) sin(pi/4n) of the exact
        lumped = exact * (4 * n / np.pi) * np.sin(np.pi / (4 * n))
        assert f == pytest.approx(lumped, rel=2e-3), n       # it grew with n (ratio 0.41 at n = 10)


@pytest.mark.parametrize("kind", ["beam", "timo", "fiber"])
def test_rotary_inertia_does_not_depend_on_the_members_orientation(tmp_path, kind):
    from vfife import modal
    sx, sz = _bar(tmp_path, f"{kind}x", kind, 8, 0), _bar(tmp_path, f"{kind}z", kind, 8, 2)
    rx, rz = modal.analyse(sx, n_modes=10), modal.analyse(sz, n_modes=10)
    if kind != "fiber":
        assert rx.frequencies == pytest.approx(rz.frequencies, rel=1e-6)
    else:
        # the old step (the fibre's old rule was orientation-dependent too)
        # needs two of the z-bar's bending rotations above their true
        # inertia: floored there, 5e-4 on the higher modes
        assert (sx.rotary_floored, sz.rotary_floored) == (0, 2)
        assert rx.frequencies == pytest.approx(rz.frequencies, rel=1e-3)
    assert _torsion_frequency(rx, 0) == pytest.approx(_torsion_frequency(rz, 2), rel=1e-6)


def test_the_rotated_diagonal_and_the_fibre_torsion_mode(tmp_path):
    from vfife import modal
    from vfife.core.mass_scaling import rotary_diagonal
    # a member along z: local x (the twist axis) is global z
    R = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert rotary_diagonal(R, [3.0, 1.0, 2.0]) == pytest.approx([1.0, 2.0, 3.0])
    # the fibre beam twists with G (Iy + Iz) (its default J) against rho Ip:
    # f = sqrt(G / rho) / 4 L, whichever way the bar points
    G = EB / (2 * (1 + NU))
    res = modal.analyse(_bar(tmp_path, "fz", "fiber", 20, 2), n_modes=12)
    lumped = np.sqrt(G / RHO) / (4 * LB) * (80 / np.pi) * np.sin(np.pi / 80)
    assert _torsion_frequency(res, 2) == pytest.approx(lumped, rel=2e-3)


@pytest.mark.parametrize("kind", ["beam", "timo", "fiber"])
def test_the_explicit_step_is_the_old_rules(tmp_path, kind):
    """用户定：改，稳定步长不变 -- the step from the old rotary rule, the
    motion from the true inertia, floored only where that step needs it."""
    from vfife.elements.stability import aligned_step
    s = _bar(tmp_path, f"dt_{kind}", kind, 6, 1, mode="dynamic")      # along y: rotation matters
    A, Iy, Iz = B * H, B * H ** 3 / 12, H * B ** 3 / 12
    if kind == "fiber":
        Iy, Iz = float(s.Iy[0]), float(s.Iz[0])
        A = float(s.total_area[0])
    L = LB / 6
    m = np.zeros((7, 6))
    for i in range(6):
        m[[i, i + 1], :3] += RHO * A * L / 2
        if kind == "beam":
            old = np.full(3, RHO * A * L ** 3 / 24)
        elif kind == "timo":
            J = float(s.member_props[0]["j"])
            old = RHO * L * np.array([J, Iy, Iz]) / 2
        else:
            old = RHO * L * np.array([Iy + Iz, Iy, Iz]) / 2
        m[[i, i + 1], 3:] += old
    row = np.asarray(abs(s.K).sum(axis=1)).ravel()
    free = np.ones(42, bool); free[:6] = False
    ratio = (m.ravel() / np.maximum(row, 1e-20))[free & (row > 1e-20)]
    from vfife.io.schema import solver_setting
    dt = float(solver_setting(s.p["solver"], "dt_safety")) * np.sqrt(ratio.min())
    assert s.dt == pytest.approx(aligned_step(1e-3, dt)[0], rel=1e-12)
    # and the mass it runs with is stable at that step, dof by dof
    act = free & (row > 1e-20)
    assert (s.mass[act] / row[act]).min() >= ratio.min() * (1 - 1e-9)
    # the rotary masses did change (true, rotated) somewhere
    assert np.abs(s.physical_mass.reshape(-1, 6)[:, 3:] - m[:, 3:]).max() > 0


# --------------------------------------------------------------------------- #
#  5  a mode with no translation
# --------------------------------------------------------------------------- #
def test_a_rotation_only_mode_is_scaled_by_its_rotation_and_flagged(tmp_path):
    from vfife import modal
    res = modal.analyse(_bar(tmp_path, "rot", "beam", 10, 2), n_modes=12)
    assert res.rotational.any()
    for i in np.flatnonzero(res.rotational):
        assert np.abs(res.shapes[i]).max() == pytest.approx(1.0)       # was 5.6e14
        assert res.norm[i] > 1e-6
        assert np.abs(res.gamma[i]).max() < 1e-6 * np.abs(res.gamma).max()
    assert "rotation only" in res.table()
    # every other mode still has a unit largest translation
    tr = res.shapes[~res.rotational].reshape(int((~res.rotational).sum()), -1, 6)[:, :, :3]
    assert np.abs(tr).max(axis=(1, 2)) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
#  6  velocity and acceleration records: exact, like a displacement record
# --------------------------------------------------------------------------- #
def test_the_record_integrals_are_exact_on_the_interpolation():
    from vfife.core.ground import RecordMotion
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r.txt")
        np.savetxt(p, [[0.0, 0.0], [1.0, 2.0], [3.0, 2.0]])
        vel = RecordMotion({"path": p, "quantity": "velocity"}, duration=4.0)
        acc = RecordMotion({"path": p, "quantity": "acceleration"}, duration=4.0)
    # v = 2t on [0, 1], 2 after: u = t^2, then 1 + 2 (t - 1); held past the end
    assert vel.motion(0.5 / 4) == pytest.approx((0.25, 1.0))
    assert vel.motion(2.0 / 4) == pytest.approx((3.0, 2.0))
    assert vel.motion(3.5 / 4) == pytest.approx((5.0, 0.0))
    # a = 2t on [0, 1], 2 after: v = t^2, u = t^3 / 3; then v = 1 + 2 s,
    # u = 1/3 + s + s^2; coasting past the end at v = 5
    assert acc.motion(0.5 / 4) == pytest.approx((0.125 / 3, 0.25))
    assert acc.motion(2.0 / 4) == pytest.approx((1 / 3 + 2.0, 3.0))
    assert acc.motion(3.5 / 4) == pytest.approx((1 / 3 + 2 + 4 + 5 * 0.5, 5.0))


@pytest.mark.parametrize("quantity", ["velocity", "acceleration"])
def test_a_support_driven_by_velocity_or_acceleration_is_step_independent(tmp_path, quantity):
    amp, f = 2.0, 3.0
    w = 2 * np.pi * f
    t = np.linspace(0, 1 / f, 4001)
    col = amp * np.sin(w * t) if quantity == "acceleration" else amp / w * (1 - np.cos(w * t))
    p = tmp_path / f"{quantity}.txt"
    np.savetxt(p, np.column_stack([t, col]))
    clamp = {"region": "base", "dofs": [1] * 6}
    m, _o, _e = _column(tmp_path, quantity, [clamp, {"region": "base", "dofs": [1, 0, 0, 0, 0, 0],
                                                     "record": {"path": str(p), "quantity": quantity}}],
                        [], total=0.2)
    tt = m["time"]
    exact = amp / w * (tt - np.sin(w * tt) / w)
    # first order in dt before: 1.2e-4 m off at this step
    assert np.abs(m["base"] - exact).max() < 1e-7


# --------------------------------------------------------------------------- #
#  7, 8  reading the spectrum in a contract
# --------------------------------------------------------------------------- #
def test_a_contract_reads_a_spectrum_reaction_at_the_nearest_node(tmp_path):
    from vfife.verification import verify
    _write_msh(tmp_path / "portal.msh", [(0, 0, 3), (4, 0, 3), (0, 0, 0), (4, 0, 0)],
               {"base": (0, [2, 3]), "top": (0, [0, 1]), "beam": (1, [(0, 1)]),
                "cols": (1, [(2, 0), (3, 1)]), "brace": (1, [(2, 1)])})
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": "portal.msh"},
         "materials": {"s": {"type": "linear_elastic", "E": 2e11, "nu": 0.3, "rho": 7850}},
         "sections": {"c": {"type": "beam", "material": "s", "shape": "rect", "width": 0.2, "height": 0.2}},
         "assign": [{"region": r, "section": "c", "orientation": [0, 1, 0]} for r in ("beam", "cols", "brace")],
         "nodal_masses": [{"region": "top", "mass": 5000.0}],
         "constraints": [{"region": "base", "dofs": [1] * 6}, {"region": "all", "dofs": [0, 1, 0, 1, 0, 1]}],
         "loads": [], "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 20, "tolerance": 1e-8},
         "output": {"fields": ["displacement"], "every": 10},
         "response_spectrum": {"spectra": [{"direction": [1, 0, 0], "table": [[0, 2.0], [10, 2.0]]}], "modes": 4}}
    _out, h5 = _run(tmp_path, "portal", d)
    r = _spectrum(h5)
    want = float(r["reaction"][2, 0])                   # node 2 is at the origin
    assert want > 1.0 and abs(want - float(r["reaction"][3, 0])) > 1.0
    c = {"version": 1, "case": "portal", "classification": "analytical", "reference": "x", "metrics": [
        {"id": "left", "source": "spectrum", "name": "reaction", "location": {"nearest": [0, 0, 0]},
         "component": "x", "expect": {"target": want, "rtol": 1e-9}},
        {"id": "no_component", "source": "spectrum", "name": "displacement", "location": {"index": 0},
         "expect": {"target": 0.0, "atol": 1.0}},
        {"id": "base_no_component", "source": "spectrum", "name": "base_force",
         "expect": {"target": 0.0, "atol": 1.0}},
        {"id": "period_ok", "source": "spectrum", "name": "period", "location": {"index": 0},
         "expect": {"finite": True}}]}
    cp = tmp_path / "portal_contract.yaml"
    cp.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    checks = {ch["id"]: ch for ch in verify(cp, h5)["checks"]}
    assert checks["left"]["status"] == "PASS"
    for k in ("no_component", "base_no_component"):
        assert checks[k]["status"] == "FAIL" and "name a `component`" in checks[k]["message"]
    assert checks["period_ok"]["status"] == "PASS"


# --------------------------------------------------------------------------- #
#  9  a truss's bar forces
# --------------------------------------------------------------------------- #
def test_a_truss_spectrum_writes_its_bar_axial_forces(tmp_path):
    _out, h5 = _run(tmp_path, "tr", _chain(tmp_path, "tr", RS2, kind="truss"))
    r = _spectrum(h5)
    ref = _closed_form(0.05, "cqc")
    assert r["axial_force"] == pytest.approx([ref["base"], ref["s2"]], rel=1e-5)


# --------------------------------------------------------------------------- #
#  10  a support record and a ground_motion along the same axis
# --------------------------------------------------------------------------- #
def test_a_record_and_a_ground_motion_in_one_direction_are_refused(tmp_path):
    from vfife.api import prepare_case
    a_path, u_path, _final = _pulse(tmp_path, "mix", n=201)
    _write_msh(tmp_path / "mix.msh", [(0, 0, 0), (0, 0, 2)],
               {"base": (0, [0]), "col": (1, [(0, 1)])})

    def case(name, direction):
        d = {"name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
             "mesh": {"type": "gmsh", "path": "mix.msh"},
             "materials": {"s": {"type": "linear_elastic", "E": 2e11, "nu": 0.3, "rho": 7850.0}},
             "sections": {"c": {"type": "beam", "material": "s", "shape": "rect", "width": 0.2, "height": 0.2}},
             "assign": [{"region": "col", "section": "c", "orientation": [1, 0, 0]}],
             "constraints": [{"region": "base", "dofs": [1] * 6},
                             {"region": "base", "dofs": [1, 0, 0, 0, 0, 0], "record": {"path": u_path}}],
             "loads": [{"ground_motion": {"path": a_path, "direction": direction}}],
             "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.01},
             "output": {"path": str(tmp_path / f"{name}.h5"), "fields": ["displacement"]}}
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
        return [i["code"] for i in prepare_case(str(p)).preflight["issues"]]

    # both applied, the response doubled (0.0306 -> 0.0612 m)
    assert "GROUND_MOTION_AND_RECORD" in case("mix_x", [1, 0, 0])
    # across each other (record in x, shaking in y) they are two inputs, not one twice
    assert "GROUND_MOTION_AND_RECORD" not in case("mix_y", [0, 1, 0])


# --------------------------------------------------------------------------- #
#  11  hydrodynamic mass: the true angle
# --------------------------------------------------------------------------- #
def test_the_five_degree_rule_is_the_members_true_angle():
    from vfife.core import added_mass
    spec = {"region": "p", "diameter": 2.0, "water_level": 0.0}
    t = np.radians(4.9)
    d = np.array([np.tan(t), np.tan(t), 1.0])            # 4.9 degrees in each plane, 6.9 in all
    nodes = np.array([[0, 0, -10], (np.array([0, 0, -10]) + 10 * d)], float)
    with pytest.raises(ValueError, match="6.9 degrees off"):
        added_mass.hydrodynamic([spec], lambda r: np.array([0, 1]), nodes, np.array([[0, 1]]))
    t = np.radians(3.0)
    d = np.array([np.tan(t), np.tan(t), 1.0])            # 4.2 in all: accepted
    nodes = np.array([[0, 0, -10], (np.array([0, 0, -10]) + 10 * d)], float)
    m, _ = added_mass.hydrodynamic([spec], lambda r: np.array([0, 1]), nodes, np.array([[0, 1]]))
    assert m[:, 2].sum() == 0.0 and m[:, 0].sum() > 0.0


# --------------------------------------------------------------------------- #
#  12  the spectrum file, before the run
# --------------------------------------------------------------------------- #
def test_a_spectrum_file_is_checked_before_the_run(tmp_path):
    from vfife.core.diagnostics import UserError
    from vfife.spectrum import read_file
    (tmp_path / "one.txt").write_text("0.0\n1.0\n")
    (tmp_path / "neg.txt").write_text("0.0 2.0\n1.0 -3.0\n")
    (tmp_path / "back.txt").write_text("0.5 2.0\n0.1 3.0\n")
    (tmp_path / "comma.txt").write_text("# T, Sa\n0.0, 2.0\n1.0, 3.0\n")
    for name, msg in (("one", "1 value"), ("neg", "negative"), ("back", "increase"), ("nope", "does not exist")):
        with pytest.raises(ValueError, match=msg):
            read_file(tmp_path / f"{name}.txt")
    assert read_file(tmp_path / "comma.txt").tolist() == [[0.0, 2.0], [1.0, 3.0]]
    d = _chain(tmp_path, "late", {"spectra": [{"direction": [1, 0, 0], "path": "missing.txt"}], "modes": 2})
    with pytest.raises(UserError, match="does not exist"):
        _run(tmp_path, "late", d)
    assert not (tmp_path / "late.h5").exists()            # refused before the run, not after it
    d = _chain(tmp_path, "ok", {"spectra": [{"direction": [1, 0, 0], "path": "comma.txt"}], "modes": 1})
    _out, h5 = _run(tmp_path, "ok", d)
    assert np.all(_spectrum(h5)["spectral_acceleration"] > 2.0)


# --------------------------------------------------------------------------- #
#  13  modes and mass_fraction
# --------------------------------------------------------------------------- #
def test_modes_and_a_mass_fraction_are_not_both_given():
    from vfife.io.schema import ResponseSpectrumSpec
    sp = [{"direction": [1, 0, 0], "table": TABLE}]
    with pytest.raises(ValueError, match="not both"):
        ResponseSpectrumSpec(spectra=sp, modes=1, mass_fraction=0.999)
    # a dumped case carries the default beside `modes`: no conflict there
    spec = ResponseSpectrumSpec(spectra=sp, modes=2)
    assert ResponseSpectrumSpec(**spec.model_dump()).modes == 2
    assert ResponseSpectrumSpec(spectra=sp, mass_fraction=0.95).modes is None


# =========================================================================== #
#  1.0 re-review -- the review of the v1.0 fixes
# =========================================================================== #
def test_147_a_fibre_split_keeps_the_rotations_stable_at_the_step(tmp_path):
    """The share a split moves is the member's TRUE rotary inertia, and a
    rotation the setup had floored went under the step on the fragment
    (dt bound 0.90 dt, 'the step is NOT adapted'); it is floored there too."""
    import test_particle_split as PS
    n = 4
    yp = tmp_path / "zsplit.yaml"
    yp_src = PS._case(tmp_path, n_elem=n, name="zsplit")
    y = open(yp_src, encoding="utf-8").read()
    y = y.replace("width: 0.05, height: 0.05,\n        ny: 3, nz: 3", "width: 0.2, height: 0.4,\n        ny: 8, nz: 4")
    y = y.replace("tip: {box: [0.99, 1.01, -0.01, 0.01, -0.01, 0.01]}", "tip: {box: [-0.01, 0.01, -0.01, 0.01, 9.99, 10.01]}")
    y = y.replace("assign: [{region: all, section: rc}]", "assign: [{region: all, section: rc, orientation: [1, 0, 0]}]")
    yp.write_text(y, encoding="utf-8")
    _write_msh(tmp_path / "zsplit.msh", [(0.0, 0.0, 10.0 * i / n) for i in range(n + 1)],
               {"line": (1, [(i, i + 1) for i in range(n)])})
    for e in (0, 1, n - 1):
        s = PS._solver(str(yp))
        assert s.rotary_floored > 0
        before = float(s.physical_mass.reshape(-1, 6)[:, :3].sum())
        PS._fracture(s, e)
        assert s.dt_bound >= s.dt * (1.0 - 1e-9), e
        # the floor is rotary only: the translational mass is unchanged
        assert float(s.physical_mass.reshape(-1, 6)[:, :3].sum()) == pytest.approx(before, rel=1e-14)


def test_147_a_spectrum_file_may_carry_more_columns(tmp_path):
    from vfife import spectrum
    p = tmp_path / "sp.txt"
    p.write_text("# T  Sa  Sd\n0.0 2.0 0.0\n1.0, 3.0, 0.076\n4.0 1.0 0.405 9\n", encoding="utf-8")
    assert spectrum.spectrum_table({"path": str(p)}).tolist() == [[0.0, 2.0], [1.0, 3.0], [4.0, 1.0]]
    p.write_text("0.0 2.0\n1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2: 1 value"):
        spectrum.read_file(p)


def test_147_a_misspelt_units_key_is_refused(tmp_path):
    from vfife import api
    d = _chain(tmp_path, "units")
    d["units"] = {"lenght": "m", "force": "N", "stress": "Pa"}
    p = tmp_path / "units.yaml"
    p.write_text(yaml.safe_dump(dict(d, name="units")), encoding="utf-8")
    rep = api.prepare_case(str(p)).preflight
    assert rep["status"] == "FAIL"
    bad = [i for i in rep["issues"] if i["code"] == "UNITS_UNKNOWN_KEY"]
    assert bad and bad[0]["severity"] == "ERROR" and "units.length" in bad[0]["message"]


def test_147_a_vector_field_at_a_location_needs_its_component(tmp_path):
    """`source: field` with a `location` and no `component` read the node's
    z silently; refused by name, and the shipped contracts that read a field
    at a location all read scalars (axial_stress, connector_force_y, ...)."""
    from vfife.verification import _metric_value
    with h5py.File(tmp_path / "v.h5", "w") as f:
        f["mesh/nodes"] = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        f["steps/1/displacement"] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        f["steps/1/axial_force"] = np.array([7.0])
    with h5py.File(tmp_path / "v.h5", "r") as h5:
        spec = {"source": "field", "name": "displacement", "location": {"index": 1}}
        with pytest.raises(ValueError, match="component"):
            _metric_value(h5, spec)
        assert _metric_value(h5, dict(spec, component="x")) == 4.0
        assert _metric_value(h5, dict(spec, reduction="max")) == 6.0         # over the node's three
        assert _metric_value(h5, {"source": "field", "name": "axial_force",
                                  "location": {"index": 0}}) == 7.0
