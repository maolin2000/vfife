"""Master-slave rigid links (1.21): `rigid_links: [{master, slaves, dofs, arm}]`,
the slaves' tied components following the master as a rigid body in large
rotation (core/constraints). Each check is a closed form or the same
structure built another way."""
import numpy as np
import pytest

from test_connector_laws import _write_msh
from vfife.api import run_case

E, I, L = 2.0e11, 1.0e-4, 4.0
EI = E * I
STEEL = {"type": "linear_elastic", "E": E, "nu": 0.3, "rho": 7850.0}
BEAM = {"type": "beam", "material": "steel", "shape": "general", "area": 0.01, "iy": I, "iz": I, "j": 2 * I}
#: the dynamic relaxation and the static Newton solver (1.21: the constraint
#: layer's map, T^T K T) give the same equilibrium
_SOLVER = {"relaxation": None, "static": {"mode": "static", "backend": "numpy", "tolerance": 1e-8, "load_steps": 1}}


def _case(tmp_path, name, nodes, groups, links, loads, constraints, monitors, sections=None, assign=None,
          solver=None):
    _write_msh(tmp_path / f"{name}.msh", nodes, groups)
    d = {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": str(tmp_path / f"{name}.msh")},
         "materials": {"steel": STEEL},
         "sections": dict({"s": BEAM}, **(sections or {})),
         "assign": assign or [{"region": "beam", "section": "s", "orientation": [0, 1, 0]}],
         "constraints": constraints, "rigid_links": links, "loads": loads,
         "solver": solver or {"mode": "relaxation", "backend": "numpy", "tolerance": 1e-7, "n_steps": 20000,
                              "damping": {"type": "kinetic"}},
         "output": {"path": str(tmp_path / f"{name}.h5"), "monitors": monitors}}
    return d


def _cantilever_with_a_hung_member(n=4, a=1.0):
    """A cantilever A-B of n elements along x, and a member C-D beyond its
    tip, C at the arm `a` from B: C is tied to B by the link, D is free."""
    nodes = [(L * i / n, 0, 0) for i in range(n + 1)] + [(L + a, 0, 0), (L + a + 1.0, 0, 0)]
    groups = {"A": (0, [0]), "B": (0, [n]), "C": (0, [n + 1]),
              "beam": (1, [(i, i + 1) for i in range(n)] + [(n + 1, n + 2)])}
    return nodes, groups


@pytest.mark.parametrize("mode", ["relaxation", "static"])
def test_a_load_on_the_slave_reaches_the_master_with_the_moment_of_its_arm(tmp_path, mode):
    """P at C, a beyond the cantilever's tip: the tip takes P and P a, and C
    stays on the rigid arm, turned with the tip: w_C = w_B + a sin(theta_B)."""
    a, P = 1.0, 1.0e4
    nodes, groups = _cantilever_with_a_hung_member(a=a)
    d = _case(tmp_path, "arm", nodes, groups, [{"master": "B", "slaves": "C"}],
              [{"region": "C", "vector": [0, 0, -P]}], [{"region": "A", "dofs": [1] * 6}],
              {"wB": {"kind": "displacement", "region": "B", "dof": "z"},
               "wC": {"kind": "displacement", "region": "C", "dof": "z"},
               "R": {"kind": "reaction", "region": "A", "dof": "z"}},
              solver=_SOLVER[mode])
    out = run_case(d)
    assert out.converged, out.message
    w = P * L ** 3 / (3 * EI) + P * a * L ** 2 / (2 * EI)
    th = P * L ** 2 / (2 * EI) + P * a * L / EI
    assert float(out.monitors["wB"][-1]) == pytest.approx(-w, rel=1e-4)
    assert float(out.monitors["wC"][-1]) == pytest.approx(-(w + a * np.sin(th)), rel=1e-4)
    assert float(out.monitors["R"][-1]) == pytest.approx(P, rel=1e-6)


@pytest.mark.parametrize("mode", ["relaxation", "static"])
def test_tied_translations_only_are_a_hinge(tmp_path, mode):
    """Two cantilevers meeting at the middle, joined by a link of their
    translations alone (111000): a hinge, each carrying half the load,
    w = (P / 2) L^3 / (3 E I)."""
    P = 1.0e2
    n = 4
    nodes = [(L * i / n, 0, 0) for i in range(n + 1)] + [(L + L * i / n, 0, 0) for i in range(n + 1)]
    groups = {"A": (0, [0]), "B": (0, [n]), "C": (0, [n + 1]), "D": (0, [2 * n + 1]),
              "beam": (1, [(i, i + 1) for i in range(n)] + [(n + 1 + i, n + 2 + i) for i in range(n)])}
    d = _case(tmp_path, "hinge", nodes, groups, [{"master": "B", "slaves": "C", "dofs": [1, 1, 1, 0, 0, 0]}],
              [{"region": "B", "vector": [0, 0, -P]}],
              [{"region": "A", "dofs": [1] * 6}, {"region": "D", "dofs": [1] * 6}],
              {"wB": {"kind": "displacement", "region": "B", "dof": "z"},
               "wC": {"kind": "displacement", "region": "C", "dof": "z"},
               "RA": {"kind": "reaction", "region": "A", "dof": "z"},
               "RD": {"kind": "reaction", "region": "D", "dof": "z"}},
              solver=_SOLVER[mode])
    out = run_case(d)
    assert out.converged, out.message
    w = 0.5 * P * L ** 3 / (3 * EI)          # (the tied ux makes the span fixed-ended: P is small, so no membrane)
    assert float(out.monitors["wB"][-1]) == pytest.approx(-w, rel=1e-4)
    assert float(out.monitors["wC"][-1]) == pytest.approx(float(out.monitors["wB"][-1]), rel=1e-9)
    assert float(out.monitors["RA"][-1]) == pytest.approx(P / 2, rel=1e-4)
    assert float(out.monitors["RD"][-1]) == pytest.approx(P / 2, rel=1e-4)


@pytest.mark.parametrize("mode", ["relaxation", "static"])
def test_a_saddle_slides_along_the_one_direction_it_leaves_free(tmp_path, mode):
    """C tied to the tip B in all but ux (011111), and held along x by a
    spring to the ground: the pull along x goes to the spring alone, the
    vertical load to the cantilever."""
    P, Fx, k = 1.0e4, 2.0e3, 1.0e6
    n = 4
    nodes = [(L * i / n, 0, 0) for i in range(n + 1)] + [(L, 0, 0), (L + 1.0, 0, 0)]
    groups = {"A": (0, [0]), "B": (0, [n]), "C": (0, [n + 1]), "G": (0, [n + 2]),
              "beam": (1, [(i, i + 1) for i in range(n)]), "spring": (1, [(n + 1, n + 2)])}
    d = _case(tmp_path, "saddle", nodes, groups, [{"master": "B", "slaves": "C", "dofs": [0, 1, 1, 1, 1, 1]}],
              [{"region": "C", "vector": [Fx, 0, -P]}],
              [{"region": "A", "dofs": [1] * 6}, {"region": "G", "dofs": [1] * 6}],
              {"wB": {"kind": "displacement", "region": "B", "dof": "z"},
               "Ax": {"kind": "reaction", "region": "A", "dof": "x"},
               "uC": {"kind": "displacement", "region": "C", "dof": "x"},
               "Gx": {"kind": "reaction", "region": "G", "dof": "x"}},
              # in global axes: the tip's rotation, which C shares, must not turn the spring
              sections={"k": {"type": "connector", "stiffness": [k, 0, 0, 0, 0, 0]}},
              assign=[{"region": "beam", "section": "s", "orientation": [0, 1, 0]},
                      {"region": "spring", "section": "k", "coordinate_system": "global"}],
              solver=_SOLVER[mode])
    out = run_case(d)
    assert out.converged, out.message
    # (the spring's mean frame turns with C, which shares the tip's 0.004 rad: 0.1%)
    assert float(out.monitors["uC"][-1]) == pytest.approx(Fx / k, rel=2e-3)
    assert abs(float(out.monitors["Ax"][-1])) < 1e-3 * Fx                  # the girder is not pulled
    assert float(out.monitors["Gx"][-1]) == pytest.approx(-Fx, rel=1e-4)
    # (the global spring's P-delta couple reaches the tip through C's tied rotation: 0.04%)
    assert float(out.monitors["wB"][-1]) == pytest.approx(-P * L ** 3 / (3 * EI), rel=1e-3)


def test_the_arm_turns_with_the_master_and_an_equal_dof_link_has_none(tmp_path):
    """The kinematics alone: the master turned a quarter turn about z; with
    `arm` the slave sits on R r0 (a small-rotation link would put it at
    r0 + theta x r0, 1.86 times as far), without it on the master's
    translation; a tied rotation is the master's."""
    from vfife.api import prepare_case, _with_element
    from vfife.core.structural import make_structural_solver
    from vfife.core.rotations import exp_map
    nodes, groups = _cantilever_with_a_hung_member(a=1.0)
    groups["C2"] = (0, [len(nodes) - 1])
    d = _case(tmp_path, "kin", nodes, groups,
              [{"master": "B", "slaves": "C"}, {"master": "B", "slaves": "C2", "dofs": [1, 1, 1, 0, 0, 0], "arm": False}],
              [], [{"region": "A", "dofs": [1] * 6}], {})
    prep = prepare_case(d)
    s = make_structural_solver(_with_element(prep.case, prep.element, prep.case_path), prep.mesh, prep.material)
    m, c, c2 = (int(np.asarray(s.mesh.regions[r]).ravel()[0]) for r in ("B", "C", "C2"))
    R = exp_map(np.array([[0.0, 0.0, np.pi / 2]]))[0]
    s.orient[m] = R
    qv = s.dofs.view(s.q); qv[m, :3] = [0.1, 0.2, 0.3]
    s.constraint_layer().correct()
    x = s.mesh.nodes + qv[:, :3]
    assert np.allclose(x[c], x[m] + R @ np.array([1.0, 0, 0]), atol=1e-12)
    assert np.allclose(x[c2] - s.mesh.nodes[c2], [0.1, 0.2, 0.3], atol=1e-12)
    assert np.allclose(s.orient[c], R, atol=1e-12)


def _hung(tmp_path, name, backend="numpy", T=0.2):
    """The hung member of the first test, in a free vibration under a step load."""
    nodes, groups = _cantilever_with_a_hung_member(a=1.0)
    return _case(tmp_path, name, nodes, groups, [{"master": "B", "slaves": "C"}],
                 [{"region": "C", "vector": [0, 0, -1.0e4]}], [{"region": "A", "dofs": [1] * 6}],
                 {"wC": {"kind": "displacement", "region": "C", "dof": "z"}},
                 solver={"mode": "dynamic", "backend": backend, "total_time": T})


def test_in_motion_the_loop_under_numba_is_the_numpy_one(tmp_path):
    """Under numba the compiled step declines the links and the loop gives
    the NumPy history; the hung member swings about its static deflection."""
    pytest.importorskip("numba")
    a = run_case(_hung(tmp_path, "np"))
    b = run_case(_hung(tmp_path, "nb", "numba"))
    assert a.converged and b.converged
    wa, wb = np.asarray(a.monitors["wC"]), np.asarray(b.monitors["wC"])
    assert np.allclose(wa, wb, rtol=1e-8, atol=1e-12)
    P, al = 1.0e4, 1.0
    static = P * L ** 3 / (3 * EI) + P * al * L ** 2 / (2 * EI) + al * (P * L ** 2 / (2 * EI) + P * al * L / EI)
    tail = -wa[3 * len(wa) // 4:]
    assert abs(tail.mean() - static) < 0.1 * static           # the load ramps in: it swings about the static state


def test_a_run_stopped_with_links_resumes(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_restart_resumes import _same, _split
    d = _hung(tmp_path, "r", T=0.05)
    d.pop("name"); d["output"].pop("path")
    full, half, rest = _split(tmp_path, d, lambda s, local: local >= 200, progress_every=1)
    _same(full, rest)


def test_what_is_refused(tmp_path):
    nodes, groups = _cantilever_with_a_hung_member()
    groups["D"] = (0, [len(nodes) - 1])
    base = dict(nodes=nodes, groups=groups, loads=[], monitors={})
    cases = [
        ([{"master": "beam", "slaves": "C"}], [{"region": "A", "dofs": [1] * 6}], {}, "a master is one"),
        ([{"master": "B", "slaves": "C"}], [{"region": "A", "dofs": [1] * 6},
                                            {"region": "C", "dofs": [0, 0, 1, 0, 0, 0], "value": [0, 0, 0.01]}],
         {}, "held by a\\s+support"),
        ([{"master": "B", "slaves": "C"}], [{"region": "A", "dofs": [1] * 6}, {"region": "C", "dofs": [0, 0, 1, 0, 0, 0]}],
         {}, "held by a support"),
        ([{"master": "B", "slaves": "C"}, {"master": "C", "slaves": "D"}], [{"region": "A", "dofs": [1] * 6}],
         {}, "chain"),
        ([{"master": "B", "slaves": "C"}], [{"region": "A", "dofs": [1] * 6}],
         {"mode": "relaxation", "backend": "warp", "n_steps": 100}, "warp"),
        ([{"master": "B", "slaves": "C", "dofs": [0, 0, 0, 0, 0, 0]}], [{"region": "A", "dofs": [1] * 6}], {}, "at least one"),
    ]
    for i, (links, cons, solver, msg) in enumerate(cases):
        d = _case(tmp_path, f"bad{i}", base["nodes"], base["groups"], links, [], cons, {},
                  solver=solver or None)
        with pytest.raises(Exception, match=msg):
            out = run_case(d)
            issues = " ".join(i["message"] for i in (out.preflight or {}).get("issues", []))
            raise RuntimeError(f"{out.message} {issues}")


@pytest.mark.parametrize("held", ["translations", "rotation"])
def test_a_master_held_by_a_support_still_carries_its_slaves(tmp_path, held):
    """Found in review: a master with held dofs took a held dof's reaction
    into its free ones through the arm, and the relaxation did not settle.
    The tip B held in its translations (the link's moment P a goes into a
    beam fixed at A and pinned at B: theta_B = P a L / (4 E I)), or in its rotation
    about y (the cantilever takes P with its tip turned not at all:
    w = P L^3 / (12 E I)); C on the arm."""
    a, P = 1.0, 1.0e3
    nodes, groups = _cantilever_with_a_hung_member(a=a)
    dofs = [1, 1, 1, 0, 0, 0] if held == "translations" else [0, 0, 0, 0, 1, 0]
    d = _case(tmp_path, "held_" + held, nodes, groups, [{"master": "B", "slaves": "C"}],
              [{"region": "C", "vector": [0, 0, -P]}],
              [{"region": "A", "dofs": [1] * 6}, {"region": "B", "dofs": dofs}],
              {"wB": {"kind": "displacement", "region": "B", "dof": "z"},
               "wC": {"kind": "displacement", "region": "C", "dof": "z"}})
    out = run_case(d)
    assert out.converged, out.message
    if held == "translations":
        th = P * a * L / (4 * EI)
        assert abs(float(out.monitors["wB"][-1])) < 1e-12
        assert float(out.monitors["wC"][-1]) == pytest.approx(-a * np.sin(th), rel=2e-3)
    else:
        w = P * L ** 3 / (12 * EI)
        assert float(out.monitors["wB"][-1]) == pytest.approx(-w, rel=2e-3)
        assert float(out.monitors["wC"][-1]) == pytest.approx(float(out.monitors["wB"][-1]), rel=1e-9)


def test_a_partial_tie_of_the_rotations_is_the_hinge_too(tmp_path):
    """The hinge of `test_tied_translations_only_are_a_hinge` with rx and rz
    tied as well (only ry, the bending rotation, left free): the same."""
    P = 1.0e2
    n = 4
    nodes = [(L * i / n, 0, 0) for i in range(n + 1)] + [(L + L * i / n, 0, 0) for i in range(n + 1)]
    groups = {"A": (0, [0]), "B": (0, [n]), "C": (0, [n + 1]), "D": (0, [2 * n + 1]),
              "beam": (1, [(i, i + 1) for i in range(n)] + [(n + 1 + i, n + 2 + i) for i in range(n)])}
    d = _case(tmp_path, "hinge_r", nodes, groups, [{"master": "B", "slaves": "C", "dofs": [1, 1, 1, 1, 0, 1]}],
              [{"region": "B", "vector": [0, 0, -P]}],
              [{"region": "A", "dofs": [1] * 6}, {"region": "D", "dofs": [1] * 6}],
              {"wB": {"kind": "displacement", "region": "B", "dof": "z"}})
    out = run_case(d)
    assert out.converged, out.message
    assert float(out.monitors["wB"][-1]) == pytest.approx(-0.5 * P * L ** 3 / (3 * EI), rel=1e-4)


def test_a_master_in_a_rigid_diaphragm_is_refused(tmp_path):
    nodes, groups = _cantilever_with_a_hung_member()
    groups["floor"] = (0, [1, len(nodes) - 3])         # B (the master) and a girder node
    d = _case(tmp_path, "fl", nodes, groups, [{"master": "B", "slaves": "C"}], [],
              [{"region": "A", "dofs": [1] * 6}], {})
    d["diaphragms"] = [{"region": "floor", "normal": "z"}]
    with pytest.raises(Exception, match="rigid diaphragm"):
        out = run_case(d)
        raise RuntimeError(f"{out.message}")


def _solver_of(d):
    from vfife.api import prepare_case, _with_element
    from vfife.core.structural import make_structural_solver
    prep = prepare_case(d)
    return make_structural_solver(_with_element(prep.case, prep.element, prep.case_path), prep.mesh, prep.material)


def test_the_modes_carry_a_rigidly_hung_mass(tmp_path):
    """A mass m at C, on a rigid arm a beyond the tip of a massless
    cantilever: one mode in each bending plane, omega^2 = 1 / (m f_C), f_C
    the flexibility at C through the arm, (L^3 / 3 + a L^2 + a^2 L) / E I."""
    from vfife import modal
    a, m = 1.0, 500.0
    nodes, groups = _cantilever_with_a_hung_member(a=a)
    d = _case(tmp_path, "mass", nodes, groups, [{"master": "B", "slaves": "C"}], [],
              [{"region": "A", "dofs": [1] * 6}], {})
    # (a member mass below the modal analysis' massless floor: condensed)
    d["materials"]["steel"] = dict(STEEL, rho=1e-12)
    d["nodal_masses"] = [{"region": "C", "mass": m}]
    res = modal.analyse(_solver_of(d), n_modes=2)
    f_C = (L ** 3 / 3 + a * L ** 2 + a * a * L) / EI
    assert np.asarray(res.frequencies) == pytest.approx(np.full(2, np.sqrt(1.0 / (m * f_C)) / (2 * np.pi)), rel=1e-6)


def test_a_leaning_column_braced_by_a_link_buckles_at_the_braces_stiffness(tmp_path):
    """A pinned leaning column under P, its head tied in x and y (arm: false)
    to the head of an unloaded cantilever: it sways at P_cr = k H with k =
    3 E I / H^3 the cantilever's, 3 E I / H^2 -- below its own Euler load
    (pi^2 E I / H^2), in both directions."""
    from vfife import buckling
    H, P, n = 4.0, 1.0e5, 4
    nodes = [(0, 0, H * i / n) for i in range(n + 1)] + [(1.0, 0, H * i / n) for i in range(n + 1)]
    groups = {"A": (0, [0]), "B": (0, [n]), "E": (0, [n + 1]), "D": (0, [2 * n + 1]),
              "beam": (1, [(i, i + 1) for i in range(n)] + [(n + 1 + i, n + 2 + i) for i in range(n)])}
    d = _case(tmp_path, "lean", nodes, groups, [{"master": "B", "slaves": "D", "dofs": [1, 1, 0, 0, 0, 0], "arm": False}],
              [{"region": "D", "vector": [0, 0, -P]}],
              [{"region": "A", "dofs": [1] * 6}, {"region": "E", "dofs": [1, 1, 1, 0, 0, 1]}], {})
    res = buckling.analyse(_solver_of(d), n_modes=2)
    assert np.sort(np.asarray(res.factors))[:2] == pytest.approx(np.full(2, 3 * EI / H ** 2 / P), rel=1e-3)


@pytest.mark.parametrize("mode", ["relaxation", "static"])
def test_a_fibre_model_takes_the_link(tmp_path, mode):
    """The eccentric load of the first test on fibre beams (the fibre
    solver is the frame solver's sibling: the constraint layer is the beam
    family's, 1.21): the tip takes P and the moment P a, as the same
    cantilever under P and P a at its tip, and C stays on the arm."""
    a, P = 1.0, 1.0e4
    fibre = {"type": "fiber", "material": "steel", "shape": "tube", "diameter": 0.3, "t": 0.01, "nt": 16}
    tip = {"wB": {"kind": "displacement", "region": "B", "dof": "z"}}
    nodes, groups = _cantilever_with_a_hung_member(a=a)
    d = _case(tmp_path, "fib", nodes, groups, [{"master": "B", "slaves": "C"}],
              [{"region": "C", "vector": [0, 0, -P]}], [{"region": "A", "dofs": [1] * 6}],
              dict(tip, wC={"kind": "displacement", "region": "C", "dof": "z"}),
              sections={"s": fibre}, solver=_SOLVER[mode])
    linked = run_case(d)
    assert linked.converged, linked.message
    n = 4
    plain = [(L * i / n, 0, 0) for i in range(n + 1)]
    d = _case(tmp_path, "fib0", plain, {"A": (0, [0]), "B": (0, [n]), "beam": (1, [(i, i + 1) for i in range(n)])},
              [], [{"region": "B", "vector": [0, 0, -P]}, {"region": "B", "moment": [0, P * a, 0]}],
              [{"region": "A", "dofs": [1] * 6}], tip, sections={"s": fibre}, solver=_SOLVER[mode])
    d.pop("rigid_links")
    box = {}
    alone = run_case(d, on_ready=lambda s, p: box.setdefault("s", s))
    assert alone.converged, alone.message
    s = box["s"]
    assert type(s).__name__ == "FiberBeamSolver"
    th = float(s.dofs.view(s.q)[n, 4])                  # the tip's rotation about y
    wB = float(alone.monitors["wB"][-1])
    # (the arm turns with the tip, so the couple of P about B is P a cos(theta):
    # 1e-5 of it at theta = 0.005; the fixed moment of the plain model does not)
    assert float(linked.monitors["wB"][-1]) == pytest.approx(wB, rel=5e-5)
    assert float(linked.monitors["wC"][-1]) == pytest.approx(wB - a * np.sin(th), rel=5e-5)


def test_a_shell_model_takes_the_link(tmp_path):
    """The clamped square plate (case_10) with a node a tenth of the side
    off the centre tied to the centre by a full link (the shell solvers are
    the beam family's: core/constraints, 1.21): the relaxation converges,
    the slave sits on the master's arm, and the plate is stiffer for it."""
    import pathlib
    import yaml
    root = pathlib.Path(__file__).resolve().parents[1] / "cases"
    d = yaml.safe_load((root / "case_10_clamped_square_shell.yaml").read_text(encoding="utf-8"))
    d["mesh"]["path"] = str((root / d["mesh"]["path"]).resolve())
    d["output"] = {"path": str(tmp_path / "plain.h5"), "monitors": d["output"]["monitors"]}
    box = {}
    plain = run_case(dict(d), on_ready=lambda s, p: box.setdefault("plain", s))
    d["regions"]["near"] = {"nearest": [0.6, 0.5, 0.0]}
    d["rigid_links"] = [{"master": "centre", "slaves": "near"}]
    d["output"] = {"path": str(tmp_path / "linked.h5"), "monitors": d["output"]["monitors"]}
    linked = run_case(d, on_ready=lambda s, p: box.setdefault("linked", s))
    assert plain.converged and linked.converged, linked.message
    s = box["linked"]
    assert type(s).__name__ == "ShellSolver"
    L = s.ties.links.links[0]
    m, n = L.master, int(L.slaves[0])
    x = s.mesh.nodes + s.dofs.view(s.q)[:, :3]
    assert np.allclose(x[n], x[m] + s.orient[m] @ L.Rm0.T @ L.r0[0], atol=1e-12)
    assert abs(float(linked.monitors["centre_w"][-1])) < abs(float(plain.monitors["centre_w"][-1]))


def test_a_buckling_master_with_no_stiffness_of_its_own_still_carries_its_slave(tmp_path):
    """The tip B of a cantilever column, a slave of a master M 1 m above it
    that a negligible member alone reaches: the column buckles at pi^2 E I /
    (4 H^2), as it does without the link. M's dofs carry no stiffness of
    their own, only their slave's through the map; deciding which dofs are
    live before the reduction dropped them, locked B and lost its load (no
    positive factor; found in review, 1.21)."""
    from vfife import buckling
    H, P, n = 4.0, 1.0e5, 4
    nodes = [(0, 0, H * i / n) for i in range(n + 1)] + [(0, 0, H + 1.0), (0, 0, H + 2.0)]
    groups = {"A": (0, [0]), "B": (0, [n]), "M": (0, [n + 1]), "G": (0, [n + 2]),
              "beam": (1, [(i, i + 1) for i in range(n)]), "soft": (1, [(n + 1, n + 2)])}
    soft = dict(BEAM, area=1e-30, iy=1e-40, iz=1e-40, j=1e-40)
    d = _case(tmp_path, "b", nodes, groups, [{"master": "M", "slaves": "B"}], [{"region": "B", "vector": [0, 0, -P]}],
              [{"region": "A", "dofs": [1] * 6}, {"region": "G", "dofs": [1] * 6}], {}, sections={"soft": soft},
              assign=[{"region": "beam", "section": "s", "orientation": [0, 1, 0]},
                      {"region": "soft", "section": "soft", "orientation": [0, 1, 0]}])
    res = buckling.analyse(_solver_of(d), n_modes=2)
    assert np.asarray(res.factors) == pytest.approx(np.full(2, np.pi ** 2 * EI / (4 * H ** 2) / P), rel=1e-3)


# ------------------------------------------------------ links in analysis steps
def _staged(tmp_path, name, n_steps, mode, P=1.0e2, Q=2.0e2):
    """The two cantilevers of the hinge test, B and C meeting: P on B from
    the start; step 1 makes the hinge h (111000) where B and C then stand,
    B already down; step 2 adds Q on B; step 3 releases h."""
    n = 4
    nodes = [(L * i / n, 0, 0) for i in range(n + 1)] + [(L + L * i / n, 0, 0) for i in range(n + 1)]
    groups = {"A": (0, [0]), "B": (0, [n]), "C": (0, [n + 1]), "D": (0, [2 * n + 1]),
              "beam": (1, [(i, i + 1) for i in range(n)] + [(n + 1 + i, n + 2 + i) for i in range(n)])}
    steps = [{}, {"rigid_links": [{"name": "h", "master": "B", "slaves": "C", "dofs": [1, 1, 1, 0, 0, 0]}]},
             {"loads": [{"region": "B", "vector": [0, 0, -Q], "keep": True}]}, {"release": ["h"]}][:n_steps]
    d = _case(tmp_path, name, nodes, groups, [], [{"region": "B", "vector": [0, 0, -P]}],
              [{"region": "A", "dofs": [1] * 6}, {"region": "D", "dofs": [1] * 6}],
              {"wB": {"kind": "displacement", "region": "B", "dof": "z"},
               "wC": {"kind": "displacement", "region": "C", "dof": "z"}},
              solver=_SOLVER[mode] or {"mode": "relaxation", "backend": "numpy", "tolerance": 2e-8,
                                       "n_steps": 20000, "load_steps": 2, "max_steps_per_load": 20000,
                                       "damping": {"type": "kinetic"}})
    d.pop("rigid_links")
    d["steps"] = steps
    return d


@pytest.mark.parametrize("mode", ["static", "relaxation"])
def test_a_link_made_in_a_step_is_born_free_of_force_and_a_released_one_lets_go(tmp_path, mode):
    """Made with B already down under P, the hinge holds C where it is (no
    force at birth); Q added after is shared, each cantilever taking Q / 2;
    released, its force is taken off and C springs back, B carrying P + Q."""
    P, Q = 1.0e2, 2.0e2
    f = L ** 3 / (3 * EI)
    out = {}
    for k in (2, 3, 4):
        o = run_case(_staged(tmp_path, f"st{k}_{mode}", k, mode, P, Q))
        assert o.converged, o.message
        out[k] = (float(o.monitors["wB"][-1]), float(o.monitors["wC"][-1]))
    assert out[2][0] == pytest.approx(-P * f, rel=1e-5)
    assert abs(out[2][1]) < 1e-6 * P * f                       # born free of force: C has not moved
    assert out[3][0] - out[2][0] == pytest.approx(-0.5 * Q * f, rel=1e-3)
    assert out[3][1] - out[2][1] == pytest.approx(-0.5 * Q * f, rel=1e-3)
    assert out[4][0] == pytest.approx(-(P + Q) * f, rel=1e-4)    # released: B alone again
    assert abs(out[4][1]) < 1e-4 * P * f


@pytest.mark.parametrize("at", [2, 3])
def test_a_run_stopped_between_a_links_birth_and_its_release_resumes(tmp_path, at):
    """Stopped in step 2 (the link made, not yet released) or in step 3 (its
    force being taken off): the restart is the uninterrupted run."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_restart_resumes import _same, _split
    d = _staged(tmp_path, "rs", 4, "relaxation")
    d.pop("name"); d["output"].pop("path")
    full, half, rest = _split(tmp_path, d, lambda s, local: s.step_index == at and local >= 1,
                             progress_every=1)
    assert half.step_index == at
    _same(full, rest)


def test_what_a_step_cannot_do_with_a_link(tmp_path):
    base = _staged(tmp_path, "bad", 4, "static")
    link = {"name": "h", "master": "B", "slaves": "C", "dofs": [1, 1, 1, 0, 0, 0]}
    cases = [
        ([{}, {"release": ["h"]}, {"rigid_links": [link]}], "made in a later step"),
        ([{}, {"rigid_links": [link], "release": ["h"]}], "made in this step"),
        ([{}, {"rigid_links": [link]}, {"release": ["h"]}, {"release": ["h"]}], "already freed"),
        ([{}, {"rigid_links": [link, dict(link, slaves="D", master="A")]}], "named 'h'"),
        ([{}, {"rigid_links": [dict(link, name="B")]}, {"release": ["B"]}], "a region and a rigid link"),
    ]
    for i, (steps, msg) in enumerate(cases):
        d = dict(base, steps=steps, name=f"bad{i}")
        d["output"] = dict(base["output"], path=str(tmp_path / f"bad{i}.h5"))
        with pytest.raises(Exception, match=msg):
            o = run_case(d)
            raise RuntimeError(f"{o.message}")


def test_a_steps_link_on_a_held_slave_is_refused(tmp_path):
    """Found in review: the construction check ran over the live links only,
    and a step's link on a held slave reached the relaxation, which did not
    settle. Every link is checked against every support now."""
    d = _staged(tmp_path, "held", 3, "relaxation")
    d["constraints"].append({"region": "C", "dofs": [0, 0, 1, 0, 0, 0]})
    with pytest.raises(Exception, match="held by a\\s+support"):
        o = run_case(d)
        raise RuntimeError(o.message)
