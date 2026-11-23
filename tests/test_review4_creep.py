"""The fourth review's creep, shrinkage, relaxation and ageing findings
(section 三), one test or two per finding.

Each was reproduced before it was fixed; the number in each test's name is
the finding's number in v1.0.
"""
import copy
import math
import pathlib

import numpy as np
import pytest

from vfife.api import _with_element, prepare_case
from vfife.cli import make_solver
from vfife.core import creep as C
from vfife.core import relaxation as R
from vfife.materials import base as mat_base

ROOT = pathlib.Path(__file__).resolve().parents[1]
BAR_MESH = str(ROOT / "benchmarks/assets/creep_bar_8.msh")
# case_29's standard linear solid, phi = 2 (1 - exp(-d / 40 d))
ZENER = {"model": "curve",
         "curve": [[float(d), 2.0 * (1.0 - math.exp(-d / 40.0))]
                   for d in np.r_[0.0, np.exp(np.linspace(math.log(1e-3),
                                                          math.log(1e+5), 200))]]}
SOLVER = {"mode": "relaxation", "backend": "numpy", "n_steps": 200000, "age": 28.0,
          "max_steps_per_load": 20000, "damping": {"type": "kinetic"},
          "tolerance": 1e-6}
CREEP_STEPS = [{"name": "load", "age": 28.0, "load_steps": 1},
               {"name": "creep", "to_age": 1028.0, "load_steps": 3}]


def _write_bars(path, names, nel=2, L=10.0):
    """Gmsh 2.2: parallel bars along x, a line group per bar and a point
    group at each end (`<name>_left`, `<name>_right`)."""
    nodes, lines, points = [], [], []
    for j, nm in enumerate(names):
        base = len(nodes)
        nodes += [(L * i / nel, 2.0 * j, 0.0) for i in range(nel + 1)]
        lines += [(nm, base + i + 1, base + i + 2) for i in range(nel)]
        points += [(nm + "_left", base + 1), (nm + "_right", base + nel + 1)]
    groups = [(1, n) for n in dict.fromkeys(n for n, _, _ in lines)] + \
             [(0, n) for n, _ in points]
    tag = {g: i + 1 for i, g in enumerate(groups)}
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(groups))]
    out += [f'{d} {tag[(d, n)]} "{n}"' for d, n in groups]
    out += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    out += [f"{i + 1} {x} {y} {z}" for i, (x, y, z) in enumerate(nodes)]
    out += ["$EndNodes", "$Elements", str(len(lines) + len(points))]
    k = 1
    for n, nd in points:
        out.append(f"{k} 15 2 {tag[(0, n)]} {tag[(0, n)]} {nd}"); k += 1
    for n, a, b in lines:
        out.append(f"{k} 1 2 {tag[(1, n)]} {tag[(1, n)]} {a} {b}"); k += 1
    out.append("$EndElements")
    pathlib.Path(path).write_text("\n".join(out) + "\n")
    return str(path)


def _bar_case(material, steps=CREEP_STEPS, solver=None, monitors=None, constraints=None,
              section=None):
    """case_29's 10 m bar, 300 kN at the free end."""
    return {
        "name": "review4_creep", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": BAR_MESH},
        "materials": {"concrete": {"type": "linear_elastic", "E": 3.0e+10, "nu": 0.2,
                                   "rho": 2400.0, **material}},
        "sections": {"bar": section or {"type": "beam", "material": "concrete",
                                        "shape": "general", "area": 0.01, "iy": 8.33e-6,
                                        "iz": 8.33e-6, "j": 1.41e-5}},
        "assign": [{"region": "bar", "section": "bar", "orientation": [0, 0, 1]}],
        "constraints": constraints or [{"region": "all", "dofs": [0, 1, 1, 1, 1, 1]},
                                       {"region": "left", "dofs": [1, 1, 1, 1, 1, 1]}],
        "loads": [{"region": "right", "vector": [3.0e+5, 0.0, 0.0]}],
        "steps": copy.deepcopy(steps),
        "solver": dict(SOLVER, **(solver or {})),
        "output": {"every": 1, "fields": ["displacement"],
                   "monitors": monitors or {"u": {"kind": "displacement", "region": "right",
                                                  "dof": "x"}}},
    }


def _solver(case):
    prep = prepare_case(case)
    return make_solver(_with_element(prep.case, prep.element, prep.case_path),
                       prep.mesh, prep.material)


def _run(case):
    s = _solver(case)
    s.run(on_progress=lambda *a, **k: None)
    return s


def _mon(s, name):
    return np.asarray(s.monitors[name].values, float)


# ------------------------------------------------ 1. each member its own law
def _two_bars(tmp_path, first, kinds=("beam", "beam")):
    mesh = _write_bars(tmp_path / "two.msh", ("bar_a", "bar_b"))
    mats = {"concrete": {"type": "linear_elastic", "E": 3e10, "nu": 0.2, "rho": 2400.0,
                         "creep": ZENER},
            "steel": {"type": "linear_elastic", "E": 3e10, "nu": 0.3, "rho": 7850.0}}

    def sec(kind, m):
        if kind == "bar":
            return {"type": "bar", "material": m, "shape": "general", "area": 0.01}
        return {"type": "beam", "material": m, "shape": "general", "area": 0.01,
                "iy": 8.33e-6, "iz": 8.33e-6, "j": 1.41e-5}
    case = _bar_case({}, monitors={
        "u_a": {"kind": "displacement", "region": "bar_a_right", "dof": "x"},
        "u_b": {"kind": "displacement", "region": "bar_b_right", "dof": "x"}},
        constraints=[{"region": "all", "dofs": [0, 1, 1, 1, 1, 1]},
                     {"region": "bar_a_left", "dofs": [1, 1, 1, 1, 1, 1]},
                     {"region": "bar_b_left", "dofs": [1, 1, 1, 1, 1, 1]}])
    case["mesh"]["path"] = mesh
    case["materials"] = {k: mats[k] for k in first}
    case["sections"] = {"sc": sec(kinds[0], "concrete"), "ss": sec(kinds[1], "steel")}
    case["assign"] = [{"region": "bar_a", "section": "sc", "orientation": [0, 0, 1]},
                      {"region": "bar_b", "section": "ss", "orientation": [0, 0, 1]}]
    case["loads"] = [{"region": "bar_a_right", "vector": [3e5, 0, 0]},
                     {"region": "bar_b_right", "vector": [3e5, 0, 0]}]
    return case


@pytest.mark.parametrize("first", [("concrete", "steel"), ("steel", "concrete")])
def test_1_only_the_member_whose_material_creeps_creeps(tmp_path, first):
    """The first material stood for every member: the steel crept with the
    concrete (10 -> 30 mm), or with the steel first nothing crept."""
    s = _run(_two_bars(tmp_path, first))
    assert _mon(s, "u_a")[[0, -1]] == pytest.approx([0.010, 0.030], rel=2e-3)
    assert _mon(s, "u_b")[[0, -1]] == pytest.approx([0.010, 0.010], rel=2e-3)


def test_1_3_a_creeping_bar_keeps_its_creep_beside_a_steel_beam(tmp_path):
    """Finding 3 with finding 1: a `truss2` concrete bar under a held load
    grew to 1.49 phi and crept back to the elastic 10 mm, because the bar's
    force never saw the station's own 1/(1 + phi)."""
    s = _run(_two_bars(tmp_path, ("concrete", "steel"), kinds=("bar", "beam")))
    u = _mon(s, "u_a")
    assert u[[0, -1]] == pytest.approx([0.010, 0.030], rel=2e-3)
    assert np.all(np.diff(u) >= -1e-9)                 # and never comes back
    assert _mon(s, "u_b")[-1] == pytest.approx(0.010, rel=2e-3)


# ------------------------------------------------ 2. mc90 against Ec28
MC90 = {"model": "mc90", "notional_size": 600.0, "rh": 80.0, "fcm": 63.0e+6}


def test_2_mc90_is_referenced_to_the_28_day_modulus():
    """J = 1/Ec(t0) + phi/Ec28: as a multiple of the ELASTIC strain at t0 the
    creep is phi Ec(t0)/Ec28, from E28 or from the ageing law."""
    phi = C.creep_coefficient(MC90, 1030.0, 400.0)
    bound = C.bind_reference(dict(MC90, E28=3.87e10), 4.651e10, None, "t")
    assert C.creep_coefficient(bound, 1030.0, 400.0) == pytest.approx(phi * 4.651 / 3.87)
    aged = C.bind_reference(MC90, 3.0e10, {"s": 0.25, "of": "modulus"}, "t")
    ratio = C.modulus_factor(400.0, 0.25, "modulus")
    assert C.creep_coefficient(aged, 1030.0, 400.0) == pytest.approx(phi * ratio)
    # the vectorised path used by the superposition agrees
    got = C.creep_coefficient(aged, 1030.0, np.array([28.0, 400.0]))
    assert got[1] == pytest.approx(phi * ratio)
    # case_28d's factor, stated in its contract
    assert 1.0 + phi * 4.651 / 3.87 == pytest.approx(1.6406, abs=1e-4)


def test_2_mc90_without_a_28_day_modulus_is_refused_by_name():
    with pytest.raises(ValueError, match="E28"):
        mat_base.create("linear_elastic", E=3e10, nu=0.2, rho=2400.0, creep=dict(MC90))
    with pytest.raises(ValueError, match="only one"):
        mat_base.create("linear_elastic", E=3e10, nu=0.2, rho=2400.0,
                        creep=dict(MC90, E28=3e10), ageing={"s": 0.25})
    # aci209 and a measured curve are already multiples of the strain at t0
    mat_base.create("linear_elastic", E=3e10, nu=0.2, rho=2400.0,
                    creep={"model": "aci209"})


def test_2_a_held_bar_grows_by_one_plus_phi_ec_t0_over_ec28():
    """A determinate bar loaded at 400 days under E = Ec(400) = 1.2 Ec28 and
    held: u(t)/u(t0) = 1 + phi E/E28, exactly (it was 1 + phi)."""
    spec = dict(MC90, E28=2.5e10)
    s = _run(_bar_case({"creep": spec},
                       steps=[{"name": "load", "age": 400.0, "load_steps": 1},
                              {"name": "creep", "to_age": 1030.0, "load_steps": 2}],
                       solver={"age": 400.0}))
    u = _mon(s, "u")
    want = 1.0 + C.creep_coefficient(MC90, 1030.0, 400.0) * 3.0 / 2.5
    assert u[-1] / u[0] == pytest.approx(want, rel=2e-4)


# ------------------------------------------------ 4. a continuous step's clock
def test_4_a_step_without_load_steps_runs_to_its_to_age():
    """It stopped wherever the residual converged: 301 days of 1028."""
    s = _run(_bar_case({"creep": ZENER},
                       steps=[{"name": "load", "age": 28.0, "load_steps": 1},
                              {"name": "creep", "to_age": 1028.0, "n_steps": 20000}]))
    assert s.ages[-1] == pytest.approx(1028.0)
    assert _mon(s, "u")[-1] == pytest.approx(0.030, rel=2e-3)


# ------------------------------------------------ 5. dynamic creep
def test_5_creep_in_a_dynamic_run_is_refused():
    case = _bar_case({"creep": ZENER},
                     steps=[{"name": "load", "age": 28.0}, {"name": "creep", "to_age": 1028.0}],
                     solver={"mode": "dynamic", "total_time": 0.1,
                             "damping": {"type": "viscous", "coeff": 200.0}})
    for k in ("n_steps", "max_steps_per_load", "tolerance"):
        case["solver"].pop(k)
    with pytest.raises(ValueError, match="dynamic run has no time"):
        _solver(case)


# ------------------------------------------------ 6. MC90 swelling
def test_6_mc90_under_water_swells():
    """beta_RH = +0.25 at RH >= 99 (MC90 2.1.6.4.4): the member lengthens."""
    base = {"model": "mc90", "notional_size": 200.0, "fcm": 38.0e6, "drying_start": 7.0}
    eps_s = (160.0 + 10.0 * 5.0 * (9.0 - 3.8)) * 1e-6
    beta_s = math.sqrt(993.0 / (350.0 * 4.0 + 993.0))
    wet = C.shrinkage_strain(dict(base, rh=99.5), 1000.0)
    assert wet == pytest.approx(0.25 * eps_s * beta_s, rel=1e-9)
    assert C.shrinkage_strain(dict(base, rh=98.0), 1000.0) == pytest.approx(
        -1.55 * (1.0 - 0.98 ** 3) * eps_s * beta_s, rel=1e-9)
    for rh in (30.0, 101.0):
        with pytest.raises(ValueError, match="40-100"):
            C.validate_shrinkage(dict(base, rh=rh))


# ------------------------------------------------ 7. aci209 loading age
def test_7_aci209_creeps_less_when_loaded_later():
    """gamma_la = 1.25 t0^-0.118 moist, 1.13 t0^-0.094 steam: phi(t0 + 100,
    t0) was the same at 7 and at 1000 days."""
    s = {"model": "aci209", "ultimate": 2.35}
    hyper = 2.35 * 100 ** 0.6 / (10 + 100 ** 0.6)
    for t0 in (7.0, 28.0, 365.0, 1000.0):
        assert C.creep_coefficient(s, t0 + 100.0, t0) == pytest.approx(
            hyper * 1.25 * t0 ** -0.118)
        assert C.creep_coefficient(dict(s, curing="steam"), t0 + 100.0, t0) == \
            pytest.approx(hyper * 1.13 * t0 ** -0.094)
    assert C.creep_coefficient(s, 107.0, 7.0) == pytest.approx(hyper, rel=1e-2)   # 0.994


# ------------------------------------------------ 8. the clock runs forward
def test_8_the_clock_does_not_run_backwards():
    with pytest.raises(ValueError, match="does not get younger"):
        C.read_step_ages({"name": "back", "age": 60.0}, 1028.0, 28.0, latest=1028.0)
    with pytest.raises(ValueError, match="only runs forward"):
        C.read_step_ages({"name": "c", "age": 1028.0, "to_age": 100.0}, 28.0, 28.0)
    # the first step may start anywhere, before or after solver.age
    assert C.read_step_ages({"age": 7.0}, 28.0, 28.0)[0] == 7.0


def test_8_a_later_step_at_an_earlier_age_is_refused_in_a_run():
    steps = [{"name": "load", "age": 28.0, "load_steps": 1},
             {"name": "back", "age": 20.0, "load_steps": 1}]
    with pytest.raises(ValueError, match="does not get younger"):
        _run(_bar_case({"creep": ZENER}, steps=steps))


# ------------------------------------------------ 9. read by nothing
def test_9_creep_on_a_solid_is_refused():
    case = {"name": "solid", "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "box", "dimensions": [1.0, 0.1, 0.1], "divisions": [2, 1, 1]},
            "regions": {"left": {"where": {"x": 0.0}}},
            "materials": {"c": {"type": "linear_elastic", "E": 3e10, "nu": 0.0,
                                "rho": 2400.0, "creep": ZENER}},
            "sections": {"body": {"type": "solid", "material": "c"}},
            "constraints": [{"region": "left", "dofs": [1, 1, 1]}],
            "steps": copy.deepcopy(CREEP_STEPS), "solver": dict(SOLVER),
            "output": {"every": 1, "fields": ["displacement"]}}
    with pytest.raises(ValueError, match="no concrete clock"):
        _solver(case)


def test_9_shrinkage_on_a_standalone_truss_is_refused():
    case = _bar_case({"shrinkage": {"model": "aci209"}},
                     section={"type": "bar", "material": "concrete", "shape": "general",
                              "area": 0.01})
    case["assign"] = [{"region": "all", "section": "bar"}]
    with pytest.raises(ValueError, match="no concrete clock"):
        _solver(case)


def _prestressed_bar(relax, steps=None):
    case = _bar_case({}, section={"type": "bar", "material": "strand", "area": 1.4e-4},
                     steps=steps or [{"name": "stress", "age": 28.0, "load_steps": 1},
                                     {"name": "relax", "to_age": 444.667, "load_steps": 2}],
                     constraints=[{"region": "all", "dofs": [1, 1, 1, 1, 1, 1]}],
                     monitors={"N": {"kind": "reaction", "region": "left", "dof": "x"}})
    case["materials"] = {"strand": {"type": "linear_elastic", "E": 1.95e11, "nu": 0.3,
                                    "rho": 7850.0, "relaxation": relax}}
    case["assign"] = [{"region": "bar", "section": "bar", "prestress": 1.488e9}]
    case["loads"] = []
    return case


def test_9_relaxation_on_a_prestressed_bar_is_the_loss_of_its_prestrain():
    """Both ends held: the reaction IS the bar force, and it falls by r(t)
    exactly -- it stayed at 208.320 kN, the relaxation read by nothing."""
    spec = {"model": "power", "rho_1000": 0.031, "k": 0.21}
    s = _run(_prestressed_bar(spec))
    N = -_mon(s, "N")
    P0 = 1.488e9 * 1.4e-4
    ages = np.asarray(s.ages)
    want = [P0 * (1.0 - R.relaxation_loss(spec, (a - 28.0) * 24.0)) for a in ages]
    assert N[0] == pytest.approx(P0, rel=1e-6)
    assert N == pytest.approx(want, rel=1e-5)
    assert N[-1] < 0.96 * P0


def test_9_relaxation_on_a_member_with_no_prestrain_is_refused():
    case = _bar_case({"relaxation": {"model": "power", "rho_1000": 0.03, "k": 0.2}})
    with pytest.raises(ValueError, match="PRESTRESSED"):
        _solver(case)


# ------------------------------------------------ 10. keys and the documented ones
def test_10_a_misspelt_key_is_refused_with_its_likely_meaning():
    with pytest.raises(ValueError, match="'RH' \\(did you mean 'rh'\\?\\)"):
        C.validate_creep({"model": "mc90", "notional_size": 200, "RH": 40, "E28": 3e10})
    with pytest.raises(ValueError, match="drying_age"):
        C.validate_shrinkage({"model": "mc90", "notional_size": 200, "drying_age": 3})
    with pytest.raises(ValueError, match="'S'"):
        C.validate_ageing({"S": 0.38})
    with pytest.raises(ValueError, match="strength or modulus"):
        C.validate_ageing({"s": 0.25, "of": "stiffness"})
    with pytest.raises(ValueError, match="curing"):
        C.validate_creep({"model": "curve", "curve": [[1, 1]], "curing": "steam"})
    with pytest.raises(ValueError, match="in Pa"):
        C.validate_creep({"model": "mc90", "notional_size": 200, "fcm": 38, "E28": 3e10})
    # and the case loader reaches it
    case = _bar_case({"creep": {"model": "mc90", "notional_size": 200, "RH": 40,
                                "E28": 3e10}})
    with pytest.raises(ValueError, match="RH"):
        prepare_case(case)


def test_10_mc90_cement_adjusts_the_loading_age():
    """t0,adj = t0 [9 / (2 + t0^1.2) + 1]^alpha, alpha -1 / 0 / 1 for
    cement 4 / 5 / 8: the documented key did nothing to phi."""
    base = {"model": "mc90", "notional_size": 200.0}
    t, t0 = 365.0, 3.0
    normal = C.creep_coefficient(base, t, t0)
    for cement, alpha in ((4, -1), (8, 1)):
        adj = max(t0 * (9.0 / (2.0 + t0 ** 1.2) + 1.0) ** alpha, 0.5)
        want = normal * (0.1 + t0 ** 0.2) / (0.1 + adj ** 0.2)
        assert C.creep_coefficient(dict(base, cement=cement), t, t0) == pytest.approx(want)
    assert C.creep_coefficient(dict(base, cement=4), t, t0) > normal > \
        C.creep_coefficient(dict(base, cement=8), t, t0)
    with pytest.raises(ValueError, match="cement"):
        C.validate_creep(dict(base, cement=6, E28=3e10))


def test_10_aci209_curing_selects_the_loading_age_factor():
    moist = C.creep_coefficient({"model": "aci209"}, 128.0, 28.0)
    steam = C.creep_coefficient({"model": "aci209", "curing": "steam"}, 128.0, 28.0)
    assert steam / moist == pytest.approx((1.13 * 28 ** -0.094) / (1.25 * 28 ** -0.118))


# ------------------------------------------------ 11. tensioned_at
def test_11_a_tendon_stressed_after_the_analysis_starts_is_refused():
    spec = {"model": "power", "rho_1000": 0.031, "k": 0.21, "tensioned_at": 200.0}
    with pytest.raises(ValueError, match="tensioned_at"):
        _solver(_prestressed_bar(spec))
    # stressed before the start: its relaxation clock started then
    ok = dict(spec, tensioned_at=7.0)
    _solver(_prestressed_bar(ok))


@pytest.mark.slow
def test_11_the_fibre_path_refuses_it_too():
    import yaml
    case = yaml.safe_load((ROOT / "cases/case_29f_tendon_relaxation.yaml").read_text())
    case["materials"]["strand"]["relaxation"]["tensioned_at"] = 200.0
    case["mesh"]["path"] = str(ROOT / "benchmarks/assets/creep_bar_8.msh")
    with pytest.raises(ValueError, match="tensioned_at"):
        _solver(case)


# ------------------------------------------------ 12. rho_1000 is a fraction
def test_12_rho_1000_in_percent_is_refused():
    with pytest.raises(ValueError, match="FRACTION"):
        R.validate({"model": "power", "rho_1000": 3.1, "k": 0.21})
    with pytest.raises(ValueError, match="rho_1000"):
        mat_base.create("linear_elastic", E=1.95e11, nu=0.3, rho=7850.0,
                        relaxation={"model": "power", "rho_1000": 3.1, "k": 0.21})
    with pytest.raises(ValueError, match="`k`"):
        R.validate({"model": "power", "rho_1000": 0.031, "k": 1.5})
    with pytest.raises(ValueError, match="FRACTIONS"):
        R.validate({"model": "curve", "curve": [[1, 1.0], [1000, 3.1]]})
    with pytest.raises(ValueError, match="does not read"):
        R.validate({"model": "power", "rho_1000": 0.031, "k": 0.21, "rho1000": 0.03})


# ------------------------------------------------ 13. the curve starts at zero
def test_13_a_creep_curve_starts_from_zero_without_a_zero_row():
    """[[1, 0.5], [1000, 2.0]] read 0.5 at the instant of loading."""
    cv = {"model": "curve", "curve": [[1.0, 0.5], [1000.0, 2.0]]}
    assert C.creep_coefficient(cv, 28.0 + 1e-6, 28.0) == pytest.approx(0.5e-6)
    assert C.creep_coefficient(cv, 28.5, 28.0) == pytest.approx(0.25)
    assert C.creep_coefficient(cv, 29.0, 28.0) == pytest.approx(0.5)
    # a [0, 0] row says the same thing, and a log axis above the first row
    zero = {"model": "curve", "curve": [[0.0, 0.0], [1.0, 0.5], [1000.0, 2.0]]}
    for d in (0.01, 0.5, 1.0, 31.62, 500.0):
        assert C.creep_coefficient(zero, 28.0 + d, 28.0) == pytest.approx(
            C.creep_coefficient(cv, 28.0 + d, 28.0))
    assert C.creep_coefficient(cv, 28.0 + 31.6227766, 28.0) == pytest.approx(1.25)
    with pytest.raises(ValueError, match="duration 0 must be 0"):
        C.validate_creep({"model": "curve", "curve": [[0.0, 0.2], [1.0, 0.5]]})


# ------------------------------------------------ the checkpoint keeps it per member
def test_a_checkpoint_carries_each_members_own_phi(tmp_path):
    from vfife.core.resume import creep_checkpoint, creep_restore
    s = _solver(_two_bars(tmp_path, ("concrete", "steel")))
    s._phi_cur = np.array([0.5, 0.5, 0.0, 0.0])
    ck = creep_checkpoint(s)
    s._phi_cur = np.zeros(4)
    creep_restore(s, ck)
    assert np.asarray(s._phi_cur) == pytest.approx([0.5, 0.5, 0.0, 0.0])


# =========================================================================== #
#  1.0 re-review -- the review of the v1.0 fixes
# =========================================================================== #
def _write_case(tmp_path, d, name):
    import yaml
    d = copy.deepcopy(d)
    d["output"] = dict(d.get("output", {}), path=str(tmp_path / f"{name}.h5"))
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return p


def _held(case):
    case["loads"] = [dict(l, amplitude="held") for l in case["loads"]]
    case["amplitudes"] = {"held": [[0.0, 1.0], [1.0, 1.0]]}
    return case


def test_147_a_dynamic_run_from_a_crept_state_keeps_the_creep(tmp_path):
    """v1.0 refused `creep` in every dynamic run, and dropping it to get
    the run through let the member spring back (0.030 -> 0.010): the history
    an initial_state restored is a frozen imposed strain on a clock that
    stands still. A dynamic step that moves the clock is still refused."""
    from vfife import api
    c = _held(_bar_case({"creep": ZENER}, steps=[
        {"name": "load", "age": 28.0, "load_steps": 1},
        {"name": "creep", "to_age": 1028.0, "load_steps": 3}]))
    first = api.run_case(_write_case(tmp_path, c, "crept"))
    assert first.monitors["u"][-1] == pytest.approx(0.030, rel=2e-3)
    dyn = copy.deepcopy(c)
    dyn["solver"] = {"mode": "dynamic", "backend": "numpy", "total_time": 0.02,
                     "initial_state": str(tmp_path / "crept.h5"),
                     "damping": {"type": "rayleigh", "alpha": 50.0}}
    dyn["steps"] = [{"name": "shake"}]
    u = np.asarray(api.run_case(_write_case(tmp_path, dyn, "dyn")).monitors["u"])
    assert u.min() == pytest.approx(0.030, rel=2e-3) and u.max() == pytest.approx(0.030, rel=2e-3)
    # an `age` that moves the clock from where the state is: refused at the step
    dyn["steps"] = [{"name": "later", "age": 2000.0}]
    with pytest.raises(ValueError, match="moves the concrete clock"):
        api.run_case(_write_case(tmp_path, dyn, "dyn2"))


def test_147_tensioned_at_defaults_to_the_analysis_start_and_a_continuation_keeps_it(tmp_path):
    """A continuation from initial_state whose first step is at 444.667 days
    restarted the relaxation there (the bar got 1.35 kN back): the default is
    solver.age, and the start rides in the checkpoint."""
    from vfife import api
    spec = {"model": "power", "rho_1000": 0.031, "k": 0.21}
    c = _prestressed_bar(spec)
    api.run_case(_write_case(tmp_path, c, "relax1"))
    P0 = 1.488e9 * 1.4e-4
    want = P0 * (1.0 - R.relaxation_loss(spec, (1028.0 - 28.0) * 24.0))
    for age in (28.0, 444.667):          # solver.age kept, and moved to the new start
        c2 = copy.deepcopy(c)
        c2["solver"] = dict(c2["solver"], initial_state=str(tmp_path / "relax1.h5"), age=age)
        c2["steps"] = [{"name": "more", "age": 444.667, "to_age": 1028.0, "load_steps": 2}]
        N = -np.asarray(api.run_case(_write_case(tmp_path, c2, f"relax2_{int(age)}")).monitors["N"])
        assert N[-1] == pytest.approx(want, rel=1e-5), age
    # a first step before solver.age: the default is that step, not refused
    early = _prestressed_bar(spec, steps=[{"name": "s", "age": 20.0, "load_steps": 1}])
    early["solver"]["age"] = 28.0
    assert np.all(_solver(early)._relax_members[0][2] == 20.0)      # each bar its own start since v1.16


def test_147_mc90_fcm_outside_c12_c80_is_refused():
    base = {"model": "mc90", "notional_size": 200.0, "rh": 70.0, "cement": 8}
    # at 115 MPa the base strain had changed sign: the member swelled at 70%
    for fcm in (15.0e6, 100.0e6, 115.0e6, 150.0e6):
        with pytest.raises(ValueError, match="C12-C80"):
            C.validate_shrinkage(dict(base, fcm=fcm))
    for fcm in (20.0e6, 88.0e6):
        C.validate_shrinkage(dict(base, fcm=fcm))
        assert C.shrinkage_strain(dict(base, fcm=fcm), 1000.0) < 0.0


def test_147_one_steel_for_a_prestressed_bar_and_a_beam_relaxes_the_bar(tmp_path):
    """Refused, saying no member was prestressed and naming one region; the
    prestressed bars of the material relax and the beam is untouched."""
    spec = {"model": "power", "rho_1000": 0.031, "k": 0.21}
    c = _two_bars(tmp_path, ("concrete", "steel"), kinds=("bar", "beam"))
    c["materials"] = {"steel": {"type": "linear_elastic", "E": 1.95e11, "nu": 0.3,
                                "rho": 7850.0, "relaxation": spec}}
    c["sections"] = {"stay": {"type": "bar", "material": "steel", "area": 1.4e-4},
                     "girder": {"type": "beam", "material": "steel", "shape": "general",
                                "area": 0.01, "iy": 8.33e-6, "iz": 8.33e-6, "j": 1.41e-5}}
    c["assign"] = [{"region": "bar_a", "section": "stay", "prestress": 1.0e9},
                   {"region": "bar_b", "section": "girder", "orientation": [0, 0, 1]}]
    c["constraints"].append({"region": "bar_a_right", "dofs": [1] * 6})
    c["loads"] = [{"region": "bar_b_right", "vector": [3e5, 0, 0]}]
    c["output"]["monitors"] = {"N": {"kind": "reaction", "region": "bar_a_left", "dof": "x"},
                               "u_b": {"kind": "displacement", "region": "bar_b_right", "dof": "x"}}
    s = _run(c)
    ages = np.asarray(s.ages)
    want = [1.4e5 * (1.0 - R.relaxation_loss(spec, (a - 28.0) * 24.0)) for a in ages]
    assert -_mon(s, "N") == pytest.approx(want, rel=1e-5)
    assert _mon(s, "u_b") == pytest.approx(np.full(len(ages), 3e5 * 10 / (1.95e11 * 0.01)), rel=1e-4)
    # and a material with no prestressed member at all: refused by its name
    c["assign"][0].pop("prestress")
    with pytest.raises(ValueError, match="material of member group\\(s\\) 'bar_a', 'bar_b'.*none of its members"):
        _solver(c)


def test_147_the_static_solver_names_relaxation_too():
    """Refused by name while the static solver had no clock; since
    v1.16 the frame path runs it (tests/test_static_clock.py) and
    only arc-length steps are refused."""
    case = _prestressed_bar({"model": "power", "rho_1000": 0.031, "k": 0.21})
    case["solver"] = {"mode": "static", "backend": "numpy", "static": {"arc_length": {"initial": 0.1}}}
    case["constraints"] = [{"region": "all", "dofs": [0, 1, 1, 1, 1, 1]},
                           {"region": "left", "dofs": [1] * 6}, {"region": "right", "dofs": [1] * 6}]
    from vfife.implicit import check
    with pytest.raises(ValueError, match="relaxation together with arc_length"):
        check(_solver(case))
