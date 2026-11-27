"""The implicit static solver: Newton on a sparse
finite-difference tangent.

Against closed forms where there are some (a linear cantilever; a
cantilever rolled into a quarter circle by an end moment; a settlement's
reactions) and against the library's own dynamic relaxation where the
answer is a nonlinear material's (a steel fibre cantilever pushed past
yield), plus the colouring the tangent's cost rests on and the refusals.
"""
import time

import h5py
import numpy as np
import pytest

from test_member_failure import _run, _write_msh

E, I = 2.0e11, 1.0e-5
STEEL = {"type": "linear_elastic", "E": E, "nu": 0.3, "rho": 7850.0}
BEAM = {"type": "beam", "material": "s", "shape": "general", "area": 1e-3, "iy": I, "iz": I, "j": 2 * I}


def _cantilever(tmp_path, name, loads, solver, n=10, L=2.0, constraints=None, monitors=None):
    _write_msh(tmp_path / f"{name}.msh", [(L * i / n, 0, 0) for i in range(n + 1)],
               {"root": (0, [0]), "tip": (0, [n]), "mid": (0, [n // 2]),
                "b": (1, [(i, i + 1) for i in range(n)])})
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": f"{name}.msh"},
         "materials": {"s": STEEL}, "sections": {"b": BEAM},
         "assign": [{"region": "b", "section": "b", "orientation": [0, 1, 0]}],
         "constraints": constraints or [{"region": "root", "dofs": [1] * 6}],
         "loads": loads, "solver": solver,
         "output": {"fields": ["displacement"], "every": 1,
                    "monitors": monitors or {"w": {"kind": "displacement", "region": "tip", "dof": "z"},
                                             "u": {"kind": "displacement", "region": "tip", "dof": "x"}}}}
    out, h5 = _run(tmp_path, name, d)
    with h5py.File(h5) as f:
        return {k: np.asarray(f["monitors"][k]) for k in f["monitors"]}, out


def test_a_linear_cantilever_in_one_increment(tmp_path):
    P, L = 1.0e3, 2.0
    m, out = _cantilever(tmp_path, "lin", [{"region": "tip", "vector": [0, 0, -P]}],
                         {"mode": "static", "load_steps": 1, "tolerance": 1e-8})
    assert out.status == "CONVERGED"
    assert m["w"][-1] == pytest.approx(-P * L ** 3 / (3 * E * I), rel=1e-5)


def test_an_end_moment_rolls_a_cantilever_into_a_quarter_circle(tmp_path):
    """M = (pi/2) E I / L bends the whole length to a quarter circle of
    radius R = 2L/pi: the tip at (R, R) from the root -- large rotations,
    solved in four increments."""
    L = 2.0
    M = np.pi / 2 * E * I / L
    m, out = _cantilever(tmp_path, "roll", [{"region": "tip", "moment": [0, -M, 0]}],
                         {"mode": "static", "load_steps": 4, "tolerance": 1e-8}, n=20)
    R = L / (np.pi / 2)
    assert m["w"][-1] == pytest.approx(R, rel=5e-4)
    assert m["u"][-1] == pytest.approx(R - L, rel=5e-4)
    assert len(m["w"]) == 4                           # a frame per increment


def test_a_settlement_by_displacement_control_gives_its_reactions(tmp_path):
    L, d = 4.0, 0.01
    cons = [{"region": "root", "dofs": [1] * 6},
            {"region": "tip", "dofs": [1] * 6},
            {"region": "tip", "dofs": [0, 0, 1, 0, 0, 0], "value": -d}]
    mons = {"r0": {"kind": "reaction", "region": "root", "dof": "z"},
            "r1": {"kind": "reaction", "region": "tip", "dof": "z"},
            "mid": {"kind": "displacement", "region": "mid", "dof": "z"}}
    m, out = _cantilever(tmp_path, "settle", [], {"mode": "static", "load_steps": 2, "tolerance": 1e-8},
                         n=8, L=L, constraints=cons, monitors=mons)
    k = 12 * E * I / L ** 3
    assert m["mid"][-1] == pytest.approx(-d / 2, rel=1e-4)
    assert m["r1"][-1] == pytest.approx(-k * d, rel=1e-3)
    assert m["r0"][-1] == pytest.approx(k * d, rel=1e-3)


def _fibre_push(tmp_path, name, solver):
    """A steel fibre cantilever, 1 m, pushed at the tip by 40 mm -- past
    first yield (about 10 mm) into the hardening."""
    n = 4
    _write_msh(tmp_path / f"{name}.msh", [(i / n, 0, 0) for i in range(n + 1)],
               {"root": (0, [0]), "tip": (0, [n]), "b": (1, [(i, i + 1) for i in range(n)])})
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": f"{name}.msh"},
         "materials": {"st": {"type": "steel02", "E": 2.0e11, "Fy": 3.5e8, "b": 0.02, "R0": 20.0, "rho": 7850}},
         "sections": {"f": {"type": "fiber", "material": "st", "shape": "rect", "width": 0.1, "height": 0.1,
                            "ny": 1, "nz": 20, "integration": "lobatto3"}},
         "assign": [{"region": "b", "section": "f", "orientation": [0, 1, 0]}],
         "constraints": [{"region": "root", "dofs": [1] * 6}, {"region": "all", "dofs": [0, 1, 0, 1, 0, 1]}],
         "loads": [],
         "steps": [{"name": "push", "load_steps": 8,
                    "constraints": [{"region": "tip", "dofs": [0, 0, 1, 0, 0, 0], "value": -0.04}]}],
         "solver": solver,
         "output": {"fields": ["displacement"], "every": 1,
                    "monitors": {"R": {"kind": "reaction", "region": "root", "dof": "z"},
                                 "w": {"kind": "displacement", "region": "tip", "dof": "z"}}}}
    t0 = time.perf_counter()
    out, h5 = _run(tmp_path, name, d)
    with h5py.File(h5) as f:
        return {k: np.asarray(f["monitors"][k]) for k in f["monitors"]}, time.perf_counter() - t0


def test_a_fibre_cantilever_past_yield_agrees_with_relaxation(tmp_path):
    st, t_st = _fibre_push(tmp_path, "st", {"backend": "numpy", "mode": "static", "tolerance": 1e-6})
    rx, t_rx = _fibre_push(tmp_path, "rx", {"backend": "numpy", "mode": "relaxation", "tolerance": 1e-6,
                                           "max_steps_per_load": 200000})
    # the same load-displacement curve, to the relaxation's own tolerance
    # (a cut-back increment adds points, so read one curve at the other's)
    pick = [int(np.argmin(np.abs(st["w"] - w))) for w in rx["w"]]
    assert st["w"][pick] == pytest.approx(rx["w"], abs=1e-12)
    assert st["R"][pick] == pytest.approx(rx["R"], rel=2e-3)
    assert abs(st["R"][-1]) > 1.2 * abs(st["R"][0]) * 1.0      # it went past yield and hardened



def test_the_colouring_keeps_every_group_two_apart():
    from vfife.implicit import node_colours
    # a portal: two columns and a beam, 3 x 4 elements
    el = [(i, i + 1) for i in range(4)] + [(4 + i, 5 + i) for i in range(4)] + [(9 + i, 10 + i) for i in range(4)]
    el = np.array(el)
    nn = 14
    groups, nbr = node_colours(el, nn)
    for g in groups:
        for a in g:
            for b in g:
                if a != b:
                    assert not (nbr[a] & nbr[b]), (a, b)   # no shared node, no shared neighbour
    assert sum(len(g) for g in groups) == nn
    chain = np.array([(i, i + 1) for i in range(30)])
    assert len(node_colours(chain, 31)[0]) == 3              # a chain: every third node together


def test_what_the_static_solver_refuses(tmp_path):
    from vfife.io.schema import SolverSpec
    assert SolverSpec(mode="static").static.max_iterations == 30
    # a limit point under load control: a perfectly plastic section pulled
    # past its plastic AXIAL force has no equilibrium to find. (It was a tip
    # load past the plastic moment -- but a cantilever under a dead vertical
    # load always has one, swung down until the arm is short enough, and
    # Newton with the energy line search finds it: v1.0.)
    n = 2
    _write_msh(tmp_path / "lp.msh", [(i / n, 0, 0) for i in range(n + 1)],
               {"root": (0, [0]), "tip": (0, [n]), "b": (1, [(i, i + 1) for i in range(n)])})
    Mp = 3.5e8 * 0.1 * 0.1 ** 2 / 4
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": "lp.msh"},
         "materials": {"st": {"type": "steel02", "E": 2.0e11, "Fy": 3.5e8, "b": 0.0, "R0": 20.0, "rho": 7850}},
         "sections": {"f": {"type": "fiber", "material": "st", "shape": "rect", "width": 0.1, "height": 0.1,
                            "ny": 1, "nz": 20}},
         "assign": [{"region": "b", "section": "f", "orientation": [0, 1, 0]}],
         "constraints": [{"region": "root", "dofs": [1] * 6}, {"region": "all", "dofs": [0, 1, 0, 1, 0, 1]}],
         "loads": [{"region": "tip", "vector": [1.3 * 3.5e8 * 0.1 * 0.1, 0, 0]}],
         "solver": {"backend": "numpy", "mode": "static", "load_steps": 4, "tolerance": 1e-6,
                    "static": {"cutbacks": 3}},
         "output": {"fields": ["displacement"], "monitors": {}}}
    out, _h5 = _run(tmp_path, "lp", d)
    assert out.status == "MAX_STEPS" and "limit point" in (out.message or "") + str(out.diagnostics)


def test_case_21c_meets_case_21s_contract(tmp_path):
    """Vecchio & Emara's frame by Newton: seconds, where relaxation takes
    twenty minutes, and case_21's contract target for target."""
    from pathlib import Path
    from vfife.api import run_case
    from vfife.verification import verify
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "case_21c.h5"
    run_case(str(root / "cases" / "case_21c_vecchio_emara_static.yaml"), output=str(out))
    report = verify(root / "verification" / "case_21c_vecchio_emara_static.yaml", out)
    assert report["status"] == "PASS", [c for c in report["checks"] if c["status"] != "PASS"]


def test_models_it_does_not_take_yet_are_named(tmp_path):
    _write_msh(tmp_path / "t.msh", [(0, 0, 0), (1, 0, 0)], {"a": (0, [0]), "b": (0, [1]), "bar": (1, [(0, 1)])})
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": "t.msh"}, "materials": {"s": STEEL},
         "sections": {"b": {"type": "bar", "material": "s", "shape": "general", "area": 1e-4}},
         "assign": [{"region": "all", "section": "b"}],
         "constraints": [{"region": "a", "dofs": [1, 1, 1]}, {"region": "b", "dofs": [0, 1, 1]}],
         "loads": [{"region": "b", "vector": [1e3, 0, 0]}],
         "solver": {"mode": "static"}, "output": {"fields": ["displacement"], "monitors": {}}}
    # preflight names it now, before the solver is built
    from vfife import api
    rep = api.prepare_case(api.CaseDraft(dict(d, name="truss"), base=tmp_path)).preflight
    assert any(i["code"] == "STATIC_UNSUPPORTED" and "frame and fibre-beam models" in i["message"]
               for i in rep["issues"])
    law = {"stiffness": 1e9, "cracked_stiffness": 1e8}
    f = {"type": "fiber", "material": "s", "shape": "rect", "width": 0.1, "height": 0.1, "ny": 1, "nz": 4,
         "shear": law}
    d2 = {"units": {"length": "m", "force": "N", "stress": "Pa"},
          "mesh": {"type": "gmsh", "path": "t.msh"}, "materials": {"s": STEEL},
          "sections": {"f": f}, "assign": [{"region": "bar", "section": "f", "orientation": [0, 1, 0]}],
          "constraints": [{"region": "a", "dofs": [1] * 6}],
          "loads": [{"region": "b", "vector": [0, 0, -1e3]}],
          "solver": {"backend": "numpy", "mode": "static"}, "output": {"fields": ["displacement"], "monitors": {}}}
    with pytest.raises(Exception, match="cracking shear law"):
        _run(tmp_path, "shear", d2)


# ------------------------------------------------------------ arc length
def _von_mises(tmp_path, name, arc, a=1.0, h=0.1, EA=1.0e6, factor=1.5, steps=None, pre=0.0):
    """Two bars from (-a, 0) and (a, 0) to an apex at height h, loaded down
    at the apex: the snap-through truss, whose exact co-rotational path is
    P(w) = -2 EA (L - L0)/L0 (h - w)/L, L = sqrt(a^2 + (h - w)^2)."""
    _write_msh(tmp_path / f"{name}.msh", [(-a, 0, 0), (0, 0, h), (a, 0, 0)],
               {"sup": (0, [0, 2]), "apex": (0, [1]), "bars": (1, [(0, 1), (1, 2)])})
    L0 = np.hypot(a, h)

    def P_of(w):
        y = h - w
        L = np.hypot(a, y)
        return -2 * EA * (L - L0) / L0 * y / L
    Pmax = float(P_of(np.linspace(0, 2 * h, 20001)).max())
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": f"{name}.msh"},
         "materials": {"s": {"type": "linear_elastic", "E": EA / 1e-3, "nu": 0.3, "rho": 7850}},
         "sections": {"b": {"type": "bar", "material": "s", "shape": "general", "area": 1e-3}},
         "assign": [{"region": "bars", "section": "b"}],
         "constraints": [{"region": "sup", "dofs": [1] * 6}, {"region": "apex", "dofs": [0, 1, 0, 1, 1, 1]}],
         "loads": [{"region": "apex", "vector": [0, 0, -factor * Pmax]}],
         "solver": {"mode": "static", "tolerance": 1e-9, "static": {"arc_length": arc}},
         "output": {"fields": ["displacement"], "every": 1,
                    "monitors": {"w": {"kind": "displacement", "region": "apex", "dof": "z"}}}}
    if steps is not None:
        # a held load `pre` Pmax from the start, then the steps given (each
        # with its own `factor` Pmax on top)
        d["amplitudes"] = {"held": [[0.0, 1.0], [1.0, 1.0]]}
        d["loads"] = [{"region": "apex", "vector": [0, 0, -pre * Pmax], "amplitude": "held"}]
        d["steps"] = []
        for st in steps:
            st = dict(st)
            f = st.pop("factor", None)
            if f is not None:
                st["loads"] = [{"region": "apex", "vector": [0, 0, -f * Pmax]}]
            d["steps"].append(st)
    out, h5 = _run(tmp_path, name, d)
    with h5py.File(h5) as f:
        lam = np.asarray(f["monitors/time"]); w = -np.asarray(f["monitors/w"])
    if steps is not None:
        return lam, w, P_of, Pmax, out
    return lam * factor * Pmax, w, P_of, Pmax, out


def test_arc_length_follows_the_snap_through_truss_exactly(tmp_path):
    P, w, P_of, Pmax, out = _von_mises(tmp_path, "vm", {"initial": 0.02, "max_increments": 400})
    assert out.status == "CONVERGED"
    # every point on the exact path, to the Newton tolerance (1e-9 of a
    # residual scaled by the support reactions, which are larger than P)
    assert np.abs(P - P_of(w)).max() < 1e-8 * Pmax
    # up over the limit point, down through the negative branch, and on to
    # the full load beyond the inverted position
    assert P.max() == pytest.approx(1.5 * Pmax)          # it ends at lambda = 1 ...
    assert P.min() < -0.5 * Pmax                          # ... having been pulled back through zero
    assert w[-1] > 2 * 0.1                                 # past the mirror image of the apex
    k = int(np.argmax(P[:np.argmin(P)]))                   # the limit point before the dip
    assert P[k] > 0.97 * Pmax and P[k] <= Pmax * (1 + 1e-9)


def test_arc_length_stops_where_it_is_told(tmp_path):
    P, w, P_of, Pmax, out = _von_mises(
        tmp_path, "vmstop", {"initial": 0.02, "stop": {"region": "apex", "dof": "z", "value": -0.12}})
    assert w[-1] >= 0.12 and w[-2] < 0.12                  # stopped at the first increment past it
    assert P[-1] < 1.5 * Pmax                              # before the full load


def test_case_16b_traces_lees_frame_through_its_snap_back(tmp_path):
    """NAFEMS NL7 by arc length: both NAFEMS points, and the fold in v_A that
    displacement control (case_16) can only jump across."""
    from pathlib import Path
    from vfife.api import run_case
    from vfife.verification import verify
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "case_16b.h5"
    run_case(str(root / "cases" / "case_16b_lee_frame_arc_length.yaml"), output=str(out))
    report = verify(root / "verification" / "case_16b_lee_frame_arc_length.yaml", out)
    assert report["status"] == "PASS", [c for c in report["checks"] if c["status"] != "PASS"]
    with h5py.File(out) as f:
        v = -np.asarray(f["monitors/v_A"]) / 1.2
    turn = int(np.argmax(v[:len(v) // 2 + 10]))
    assert v[turn] - v[turn:].min() > 0.05                 # v_A/L folds back by more than 0.05


def test_arc_length_on_the_steps_named_and_newton_on_the_others(tmp_path):
    """`arc_length.steps`: a first step by load control, a
    second by arc length. The Hangzhou Bay pushover needs exactly this -- the
    dead load and the prestress settled by Newton, the live load pushed past
    its peak by arc length."""
    t, w, P_of, Pmax, out = _von_mises(
        tmp_path, "vmsteps", {"initial": 0.02, "max_increments": 400, "steps": ["snap"]},
        steps=[{"name": "settle", "load_steps": 3},
               {"name": "snap", "factor": 1.2}], pre=0.3)
    assert out.status == "CONVERGED"
    first = np.arange(len(t)) < 3          # (the arc's k + lambda dips below 1 too)
    # step 1: three load-controlled levels, at the levels asked for, with
    # the held 0.3 Pmax on the exact curve
    assert t[first] == pytest.approx([1 / 3, 2 / 3, 1.0], abs=1e-12)
    assert P_of(w[first][-1]) == pytest.approx(0.3 * Pmax, rel=1e-7)
    # step 2: the arc, from that equilibrium, over the limit point and on
    # to the full 1.5 Pmax -- every point on the exact path
    lam = t[~first] - 1.0
    P = 0.3 * Pmax + lam * 1.2 * Pmax
    assert np.abs(P - P_of(w[~first])).max() < 1e-8 * Pmax
    assert lam[-1] == pytest.approx(1.0) and P.min() < 0.0


def test_the_difference_step_is_scaled_by_the_element_not_the_model(tmp_path):
    """The tangent's step was the model's largest coordinate times 1e-7: on
    the 580 m Hangzhou Bay unit 3e-5 m, enough to straddle the concrete's
    cracking kink in many fibres, and near the peak the arc then shrank
    for ever. It is the median element length now."""
    from vfife.implicit import Newton

    class _S:
        TRIAL_STATE = False
    s = _S()
    import types
    s.mesh = types.SimpleNamespace(nodes=np.array([[0, 0, 0], [5.0, 0, 0], [10.0, 0, 0], [500.0, 0, 0]]),
                                   elements=np.array([[0, 1], [1, 2], [2, 3]]))
    s.dofs = types.SimpleNamespace(stride=6)
    s.nn = 4
    n = Newton(s, {})
    assert n.h_t == pytest.approx(1e-7 * 5.0)


def _fibre_push_backend(tmp_path, name, backend, threads=1):
    return _fibre_push(tmp_path, name, {"backend": backend, "threads": threads, "mode": "static",
                                        "tolerance": 1e-8})[0]


def test_the_fused_kernel_takes_the_newton_trial_path(tmp_path):
    """Under Newton every force evaluation restarts the fibres from the
    converged state. On the kernel that is now two copies of the
    consolidated arrays instead of one copy per fibre group (1.4 ms of the
    Hangzhou Bay unit's 3.4 ms); the answer is the NumPy
    reference's."""
    pytest.importorskip("numba")
    ref = _fibre_push_backend(tmp_path, "np", "numpy")
    ker = _fibre_push_backend(tmp_path, "nb", "numba")
    assert ker["w"] == pytest.approx(ref["w"], abs=1e-12)
    assert ker["R"] == pytest.approx(ref["R"], rel=1e-9)


def test_threads_give_the_serial_kernels_answer_to_the_bit(tmp_path):
    """`solver.threads`: the fibre kernel's elements across
    cores, each writing its own row, summed afterwards in element order --
    so the answer does not depend on the thread count, not even in the last
    bit."""
    numba = pytest.importorskip("numba")
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("one core")
    one = _fibre_push_backend(tmp_path, "t1", "numba", 1)
    two = _fibre_push_backend(tmp_path, "t2", "numba", 2)
    assert np.array_equal(one["R"], two["R"]) and np.array_equal(one["w"], two["w"])


def test_case_24g_pushes_the_hangzhou_bay_unit_past_its_peak(tmp_path):
    """case_24d's 50-minute explicit pushover as two static steps:
    the dead state equal to the relaxation's, the peak, and the
    softening branch after it that a ramp can only jump."""
    from pathlib import Path
    from vfife.api import run_case
    from vfife.verification import verify
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "case_24g.h5"
    run_case(str(root / "cases" / "case_24g_hzb_static_pushover.yaml"), output=str(out))
    report = verify(root / "verification" / "case_24g_hzb_static_pushover.yaml", out)
    assert report["status"] == "PASS", [c for c in report["checks"] if c["status"] != "PASS"]
